"""Daily touchgo bug-hunt: BT(consolidated) vs PROD decision.

Goal: find the EIDO/OSCR/TSDD touchgo false-positive bug. Each weekday next
week this reads the day's live touchgo audit log (trading/touchgo_audit.py),
re-derives the "BT" decision from Alpaca's CONSOLIDATED 1-min bars (the same
input the backtest uses, via the same shared rule), and reports any
divergence — with a ROOT-CAUSE verdict — to Telegram and a log file.

"BT" here = consolidated-bar recompute of the shared touchgo rule. Parity by
construction: the backtest feeds the identical consolidated bars to the same
evaluate_rule_m/d helper, so prod-decision vs consolidated-recompute *is*
prod-vs-BT for touchgo.

Divergence taxonomy (per evaluation):
  MISMATCH  live bb_close_pos differs from consolidated  -> streamed bar OHLC
            != official consolidated bar (close-price gap)
  FLIP      consolidated decision != what prod did       -> the money-loser
  REKEY     breakout_bar_ts != first consolidated bar > range_high -> wrong bar

Window-bounded to next week (2026-06-29 .. 2026-07-03); no-ops otherwise so a
lingering cron auto-stops. Friday emits a week-summary verdict.

Run: python3 scripts/touchgo_daily_debug.py [--day YYYY-MM-DD] [--force]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(str(ROOT / ".env"))
sys.path.insert(0, str(ROOT))

WINDOW_START = "2026-06-29"
# Extended 2026-07-03: the breakout-bar keying fix (range_end_ts on
# OpenPosition + strict session-open anchor + negative-age tripwire) ships
# today. Keep the daily BT-vs-prod verdicts running through the following
# week to CONFIRM the fix live: expect REKEY count = 0 and 'breakout bar
# keyed ... source=pos_range' journal lines on every ORB fill.
WINDOW_END = "2026-07-10"

AUDIT_PATH = Path(os.getenv("ORB_TOUCHGO_AUDIT_PATH", str(ROOT / "logs" / "touchgo_audit.jsonl")))
DEBUG_LOG = ROOT / "logs" / "touchgo_daily_debug.log"
FINDINGS = ROOT / "logs" / "touchgo_debug_findings.jsonl"
EPS = 0.02


def _now_utc():
    return datetime.now(timezone.utc)


def _log(msg: str):
    DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(DEBUG_LOG, "a", encoding="utf-8") as fh:
        fh.write(f"[{_now_utc().isoformat()}] {msg}\n")


def _telegram(msg: str):
    try:
        from notifications.telegram_notifier import TelegramNotifier
        n = TelegramNotifier(os.getenv("TELEGRAM_BOT_TOKEN"),
                             os.getenv("TELEGRAM_CHAT_ID"), enabled=True)
        n.send_message_sync(msg, parse_mode="HTML")
    except Exception as exc:
        _log(f"TELEGRAM SEND FAILED: {exc!r}")


def _orb_client():
    from data_sources.alpaca_client import AlpacaClient
    return AlpacaClient(os.getenv("ALPACA_ORB_API_KEY"), os.getenv("ALPACA_ORB_API_SECRET"),
                        paper=os.getenv("ALPACA_ORB_PAPER", "true").lower() == "true")


def _read_records(day: str):
    if not AUDIT_PATH.exists():
        return []
    recs = []
    for line in AUDIT_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("trade_date") == day:
            recs.append(r)
    return recs


def _bar_at(df, ts_iso):
    if df is None or df.empty or not ts_iso:
        return None
    minute = datetime.fromisoformat(ts_iso).replace(second=0, microsecond=0).astimezone(timezone.utc)
    m = df[df["timestamp"].dt.tz_convert("UTC").dt.floor("min") == minute]
    return m.iloc[0] if len(m) else None


def _range_end_utc(day: str) -> datetime:
    """9:35 ET as UTC for `day` (EDT Mar–Oct → 13:35Z, EST → 14:35Z).

    Month-approximation mirrors trading.orb_engine._et_offset_hours. The
    original checker searched from 13:00Z (9:00 ET), which let PRE-MARKET
    spikes above range_high masquerade as the 'true' breakout bar — that
    false-flagged CAST 6/30 (correctly keyed) as REKEY."""
    d0 = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    hour = 13 if 3 <= d0.month <= 10 else 14
    return d0 + timedelta(hours=hour, minutes=35)


def analyze(day: str):
    """Return (records, flips) where flips is a list of dicts with verdict tags.

    Consolidated decisions are recomputed with the SHARED rule helpers
    (evaluate_rule_m/d) so degenerate-bar guards match live + BT exactly —
    the naive `close_pos < thr` comparison false-flagged SHNY 7/2 (degenerate
    breakout bar: live correctly held, checker said FLIP)."""
    recs = _read_records(day)
    if not recs:
        return recs, []
    from trading.orb_touchgo_filter import (
        evaluate_rule_d, evaluate_rule_m, find_breakout_bar_ts,
        load_touchgo_config,
    )
    tg_cfg = load_touchgo_config({})
    client = _orb_client()
    cache = {}

    def day_bars(sym):
        if sym not in cache:
            d0 = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            df = client.get_historical_1min_bars(sym, d0 + timedelta(hours=13), d0 + timedelta(hours=20))
            if df is not None and "timestamp" not in df.columns:
                df = df.reset_index()
            cache[sym] = df
        return cache[sym]

    rng_end = _range_end_utc(day)
    flips = []
    for r in recs:
        sym, rule = r["symbol"], r["rule"]
        df = day_bars(sym)
        tags = []
        live_v = consol_v = None
        if rule == "M":
            live_v = r.get("bb_close_pos")
            bb = _bar_at(df, r.get("breakout_bar_ts"))
            if bb is not None:
                o, h, l, c = (float(bb["open"]), float(bb["high"]),
                              float(bb["low"]), float(bb["close"]))
                consol_v = (c - l) / (h - l) if h > l else 0.0
                if live_v is not None and abs(live_v - consol_v) > EPS:
                    tags.append("MISMATCH")
                consol_fire, _ = evaluate_rule_m(o, h, l, c, tg_cfg)
                if bool(r.get("fired")) != bool(consol_fire):
                    tags.append("FLIP")
            # REKEY: the true breakout bar via the SAME shared finder,
            # anchored at the real range end (excludes pre-market bars).
            if df is not None and not df.empty:
                true_ts = find_breakout_bar_ts(df, r.get("range_high"), rng_end)
                if true_ts is not None:
                    true_min = true_ts.tz_convert("UTC").floor("min")
                    keyed = datetime.fromisoformat(r["breakout_bar_ts"]).astimezone(timezone.utc).replace(second=0, microsecond=0)
                    if true_min != keyed:
                        tags.append(f"REKEY(true={true_min.strftime('%H:%M')}Z)")
        else:
            live_v = r.get("b1_revert_R")
            b1 = _bar_at(df, r.get("b1_ts"))
            rs = r.get("range_size") or 0
            if b1 is not None and rs > 0:
                b1_low = float(b1["low"])
                consol_v = (r["entry_price"] - b1_low) / rs
                if live_v is not None and abs(live_v - consol_v) > 0.05:
                    tags.append("MISMATCH")
                consol_fire, _ = evaluate_rule_d(r["entry_price"], b1_low, rs, tg_cfg)
                if bool(r.get("fired")) != bool(consol_fire):
                    tags.append("FLIP")
        if tags:
            flips.append({"symbol": sym, "rule": rule, "fired": r.get("fired"),
                          "live": live_v, "consol": consol_v, "tags": tags})
            _log(f"  {sym} {rule} fired={r.get('fired')} live={live_v} consol={consol_v} tags={tags}")
    return recs, flips


def verdict(flips):
    """REKEY is first-class bug evidence: a mis-keyed breakout bar means the
    rule evaluated the wrong bar — whether the decision happened to match is
    luck (KOLD 6/29 fired tag_bb an hour in on a 14:35Z bar and made money;
    TSDD 6/23 did the same and forfeited +$1,392). Root cause localized
    2026-07-03: orb_engine._first_session_open_ts_utc accepts a 14:30Z bar
    as the '9:30 anchor' during EDT (hour.isin([13,14])), so windows missing
    the true 13:30Z bar anchor an hour late -> range_end 14:35Z -> Rule M/D
    keyed to ~14:35Z bars."""
    rekey = [f for f in flips
             if any(str(t).startswith("REKEY") for t in f["tags"])]
    real = [f for f in flips if "FLIP" in f["tags"]]
    if not real and not rekey:
        return "no_divergence", "No FLIP/REKEY — prod touchgo matches BT/consolidated."
    causes = set()
    if rekey:
        causes.add(
            "wrong breakout bar keyed — false 9:30 anchor "
            "(orb_engine._first_session_open_ts_utc hour.isin([13,14]) "
            "accepts 14:30Z=10:30 ET during EDT)"
        )
    for f in real:
        if "MISMATCH" in f["tags"]:
            causes.add("streamed-vs-consolidated close gap (live real-time bar ≠ official bar)")
        elif not any(str(t).startswith("REKEY") for t in f["tags"]):
            causes.add("decision divergence on correctly-keyed bar (guard/threshold)")
    return "bug_found", " / ".join(sorted(causes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default=None)
    ap.add_argument("--force", action="store_true", help="run even outside the window")
    args = ap.parse_args()

    today = args.day or _now_utc().astimezone(timezone.utc).strftime("%Y-%m-%d")
    if not args.force and not (WINDOW_START <= today <= WINDOW_END):
        _log(f"{today} outside window {WINDOW_START}..{WINDOW_END} — no-op.")
        return

    try:
        recs, flips = analyze(today)
    except Exception:
        _log(f"ANALYZE CRASHED:\n{traceback.format_exc()}")
        _telegram(f"<b>[TOUCHGO-DEBUG] {today}</b>\n⚠️ debug script error — see logs/touchgo_daily_debug.log")
        return

    nM = sum(1 for r in recs if r["rule"] == "M")
    nD = sum(1 for r in recs if r["rule"] == "D")
    status, cause = verdict(flips)
    # Everything flagged (FLIP or REKEY) is reportable — a mis-keyed bar is
    # the bug even when the decision coincidentally matched.
    flagged = [f for f in flips
               if "FLIP" in f["tags"]
               or any(str(t).startswith("REKEY") for t in f["tags"])]

    # Persist finding
    FINDINGS.parent.mkdir(parents=True, exist_ok=True)
    with open(FINDINGS, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"day": today, "evals": len(recs), "flips": len(flagged),
                             "status": status, "cause": cause,
                             "detail": flagged}, default=str) + "\n")

    # Build Telegram conclusion
    if not recs:
        msg = (f"<b>[TOUCHGO-DEBUG] {today}</b>\n"
               f"No touchgo evals logged (no ORB fills today, or audit not yet "
               f"deployed). Bug hunt continues.")
    elif status == "no_divergence":
        msg = (f"<b>[TOUCHGO-DEBUG] {today}</b>\n"
               f"Evals: {len(recs)} (M:{nM} D:{nD}) | Divergences: 0\n"
               f"✅ prod touchgo matches BT/consolidated today. No bug surfaced.")
    else:
        lines = [f"<b>[TOUCHGO-DEBUG] {today}</b>",
                 f"Evals: {len(recs)} (M:{nM} D:{nD}) | <b>flagged: {len(flagged)}</b>"]
        for f in flagged:
            lv = f"{f['live']:.3f}" if isinstance(f['live'], (int, float)) else "?"
            cv = f"{f['consol']:.3f}" if isinstance(f['consol'], (int, float)) else "?"
            lines.append(f"⚠️ {f['symbol']} ({f['rule']}) prod_fired={f['fired']} "
                         f"live={lv} vs consol={cv}  [{','.join(str(t) for t in f['tags'])}]")
        lines.append(f"\n<b>🐞 VERDICT: {cause}</b>")
        msg = "\n".join(lines)

    _log(f"VERDICT {today}: status={status} cause={cause} evals={len(recs)} flips={len(real_flips)}")
    _telegram(msg)

    # Friday → week summary
    if today == WINDOW_END:
        _week_summary()


def _week_summary():
    if not FINDINGS.exists():
        return
    days = []
    for line in FINDINGS.read_text().splitlines():
        try:
            d = json.loads(line)
        except Exception:
            continue
        if WINDOW_START <= d.get("day", "") <= WINDOW_END:
            days.append(d)
    total_flips = sum(d["flips"] for d in days)
    causes = sorted({d["cause"] for d in days if d["status"] == "bug_found"})
    if total_flips:
        body = ("🐞 <b>BUG FOUND.</b> Root cause(s):\n  " + "\n  ".join(causes) +
                f"\n\nTotal FLIPs across the week: {total_flips}. "
                f"The fix path is now determined — see logs/touchgo_debug_findings.jsonl.")
    else:
        body = ("No touchgo divergence captured this week (the false-positive "
                "didn't recur in the sample). Instrumentation stays live; it will "
                "catch the next occurrence. Bug not yet localized.")
    _telegram(f"<b>[TOUCHGO-DEBUG] WEEK SUMMARY {WINDOW_START}..{WINDOW_END}</b>\n"
              f"Days run: {len(days)}\n{body}")


if __name__ == "__main__":
    main()
