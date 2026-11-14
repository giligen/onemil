#!/usr/bin/env python3
"""Nightly BT-vs-live(shadow) parity for the gap-fade short (16:35 ET, after gap_short_paper --score).

(1) Re-run the BT rule for TODAY from the BT's inputs: daily open / prev close (Alpaca daily-bar snapshot) for the gap and the
top-10 ranking, 04:00-09:19 ET 1-min SIP bars for L1, the measured last quote at or before 10:30 ET (historical SIP quotes) for the
exit -> BT picks + per-name BT net bp (rules in trading/gap_short_rules.py).  (2) Compare with logs/gap_short_ledger.csv: matched,
bt_only / live_only each with a REASON from REASONS, per-matched-name slippage and net diff.  (3) Runs on 0-live-pick days too (shadow).
(4) ONE Telegram line `[GAPSHORT PARITY]`; ERROR (logger.error) only if unexplained > 0 or an API failure.  Rows -> logs/gap_short_parity.csv
(idempotent per date; Telegram only on the first write of a date).  Live-only names are `universe_diff` by construction: live ranks on the
09:20 pre-market last, the BT on the official open.
"""
from __future__ import annotations

import argparse
import logging
import os
import statistics
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import requests

_REPO = Path(__file__).resolve().parent.parent
for _p in (str(_REPO), str(_REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gap_short_paper as gp  # noqa: E402
import locate_ledger as ll  # noqa: E402
from trading import gap_short_rules as R  # noqa: E402

logger = logging.getLogger("gap_short_parity")
PARITY_CSV = _REPO / "logs" / "gap_short_parity.csv"
COLUMNS = ["date", "symbol", "side", "reason", "bt_open", "bt_ask_1030", "bt_net_bp", "live_net_bp", "entry_slip_bp", "exit_slip_bp", "net_diff_bp"]
REASONS = ("opg_deadline", "rejected_htb", "rejected_ssr", "existing_position", "qty_floor", "pause_flag", "universe_diff", "unexplained")


def classify_bt_only(symbol: str, ledger_row, ctx: dict) -> str:
    """Reason a BT pick was not (successfully) a live pick. ctx: pause_active, universe (set), open_price."""
    if ledger_row is not None:
        status, note = (ledger_row.get("entry_status") or ""), (ledger_row.get("note") or "").lower()
        if status == "not_submitted" or "opg" in note:
            return "opg_deadline"
        if "existing" in note:
            return "existing_position"
        if "threshold" in note:
            return "rejected_htb"
        if status == "rejected":
            if "ssr" in note or "restrict" in note:
                return "rejected_ssr"
            if any(k in note for k in ("borrow", "short", "locate", "htb")):
                return "rejected_htb"
        return "unexplained"
    if ctx.get("pause_active"):
        return "pause_flag"
    if R.position_qty(ctx.get("open_price") or 0) < 1:
        return "qty_floor"
    if symbol not in ctx.get("universe", set()):
        return "universe_diff"
    return "unexplained"


def compare(day: str, bt: list, ledger_rows: list, ctx: dict) -> list:
    """Per-name parity rows. bt: [{symbol, open, ask, net}]; ledger_rows: today's ledger dicts."""
    live = {r["symbol"]: r for r in ledger_rows if r["date"] == day}
    out, bt_syms = [], {b["symbol"] for b in bt}
    f = lambda v: "" if v in ("", None) else float(v)  # noqa: E731
    for b in bt:
        row = {"date": day, "symbol": b["symbol"], "bt_open": b["open"], "bt_ask_1030": b["ask"] or "", "bt_net_bp": b["net"] if b["net"] is not None else ""}
        lr = live.get(b["symbol"])
        filled = lr is not None and lr.get("entry_fill") not in ("", None)
        if filled:
            row.update(side="matched", reason="", live_net_bp=lr.get("net_bp", ""))
            if lr.get("entry_fill") not in ("", None) and b["open"]:
                row["entry_slip_bp"] = round((b["open"] - float(lr["entry_fill"])) / b["open"] * 1e4, 1)
            if lr.get("cover_fill") not in ("", None) and b["ask"]:
                row["exit_slip_bp"] = round((float(lr["cover_fill"]) / b["ask"] - 1) * 1e4, 1)
            if f(row["live_net_bp"]) != "" and row["bt_net_bp"] != "":
                row["net_diff_bp"] = round(float(row["live_net_bp"]) - row["bt_net_bp"], 1)
        elif lr is not None and lr.get("entry_status") in ("accepted", "new", "filled", "") and lr.get("order_id"):
            row.update(side="matched", reason="")          # picked + ordered, fill not yet scored
        else:
            row.update(side="bt_only", reason=classify_bt_only(b["symbol"], lr, dict(ctx, open_price=b["open"])))
        out.append(row)
    for sym, lr in live.items():
        if sym not in bt_syms:
            out.append({"date": day, "symbol": sym, "side": "live_only", "reason": "universe_diff", "live_net_bp": lr.get("net_bp", "")})
    return out


def summary_line(day: str, rows: list, api_error: str = "") -> str:
    """`[GAPSHORT PARITY] date: BT picks n | live n | matched k | unexplained m | BT net x bp vs live y bp | slip entry a / exit b bp`."""
    bt_rows = [r for r in rows if r["side"] in ("matched", "bt_only")]
    matched = [r for r in rows if r["side"] == "matched"]
    live_n = len(matched) + sum(1 for r in rows if r["side"] == "live_only")
    unexpl = sum(1 for r in rows if r.get("reason") == "unexplained")
    mean = lambda k, rs: (f"{statistics.mean(float(r[k]) for r in rs if r.get(k) not in ('', None)):+.0f}"  # noqa: E731
                          if any(r.get(k) not in ("", None) for r in rs) else "n/a")
    line = (f"[GAPSHORT PARITY] {day}: BT picks {len(bt_rows)} | live {live_n} | matched {len(matched)} | unexplained {unexpl} | "
            f"BT net {mean('bt_net_bp', bt_rows)} bp vs live {mean('live_net_bp', matched)} bp | "
            f"slip entry {mean('entry_slip_bp', matched)} / exit {mean('exit_slip_bp', matched)} bp")
    if unexpl or api_error:
        line += f" | ERROR {'api: ' + api_error if api_error else 'unexplained names'}"
    return line


def fetch_ask_1030(headers: dict, symbol: str, day: str):
    """Last SIP ask at or before 10:30:00 ET (BT definition: 10:29:30-10:30:00 window); None when no quote."""
    d = datetime.strptime(day, "%Y-%m-%d")
    t0 = d.replace(hour=10, minute=29, second=30, tzinfo=R.ET).astimezone(R.ZoneInfo("UTC"))
    t1 = d.replace(hour=10, minute=30, tzinfo=R.ET).astimezone(R.ZoneInfo("UTC"))
    r = requests.get(f"https://data.alpaca.markets/v2/stocks/{symbol}/quotes", headers=headers, timeout=30,
                     params=dict(start=t0.strftime("%Y-%m-%dT%H:%M:%SZ"), end=t1.strftime("%Y-%m-%dT%H:%M:%S.000001Z"), limit=1, sort="desc", feed="sip"))
    if r.status_code != 200:
        raise RuntimeError(f"quotes HTTP {r.status_code} for {symbol}")
    q = (r.json().get("quotes") or [])
    return float(q[0]["ap"]) if q and q[0].get("ap") else None


def bt_picks(day: str, client, universe: dict, headers: dict) -> list:
    """BT rule for `day`: gap = daily open / prev close - 1 (snapshots), top-10 by gap, L1 (04:00-09:19 bars) >= cutoff, ask at 10:30."""
    snaps = client.get_snapshots(list(universe))
    if not snaps:
        raise RuntimeError("snapshots empty")
    cands = []
    for sym in universe:
        s = snaps.get(sym) or {}
        if s.get("daily_bar_date") not in (None, day):
            continue
        g = R.gap_fraction(s.get("open"), s.get("prev_close") or universe[sym])
        if g is not None and R.MIN_GAP <= g <= R.MAX_GAP and R.price_ok(s.get("open")) and not R.is_excluded_symbol(sym):
            cands.append({"symbol": sym, "gap": g, "open": s["open"], "l1": None})
    cands.sort(key=lambda c: (-c["gap"], c["symbol"]))
    d = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=R.ET)
    for c in cands[:R.TOP_N]:
        df = client.get_historical_1min_bars(c["symbol"], d.replace(hour=4), d.replace(hour=9, minute=30))
        c["l1"] = R.premarket_dollar_volume({"t": r.timestamp.to_pydatetime(), "c": r.close, "v": r.volume} for r in df.itertuples()) if df is not None and len(df) else None
    out = []
    for c in R.select_picks(cands):
        ask = fetch_ask_1030(headers, c["symbol"], day)
        out.append({"symbol": c["symbol"], "open": c["open"], "ask": ask, "net": round(R.net_bp(c["open"], ask), 1) if ask else None})
    return out


def write_rows(day: str, rows: list, path=None) -> bool:
    """Replace the date's rows in the parity CSV (idempotent); True when the date was new."""
    path = Path(path or PARITY_CSV)
    old = ll.read_csv(path)
    new_date = not any(r["date"] == day for r in old)
    gp.write_ledger([r for r in old if r["date"] != day] + (rows or [{"date": day, "symbol": "-", "side": "none"}]), path, COLUMNS)
    return new_date


def run_parity(dry_run: bool, now=None, client=None, universe=None, headers=None) -> int:
    """Build BT picks, compare to the ledger, write rows, emit one line."""
    now = now or gp.et_now()
    day = now.strftime("%Y-%m-%d")
    api_error, bt, rows = "", [], []
    try:
        client = client or gp.make_client()
        universe = universe if universe is not None else gp.load_universe()
        headers = headers or {"APCA-API-KEY-ID": os.environ[gp.ENV_KEY], "APCA-API-SECRET-KEY": os.environ[gp.ENV_SEC]}
        gp.reconcile(gp.PaperBroker(client), now, dry_run, "parity")
        bt = bt_picks(day, client, universe, headers)
    except Exception as e:  # noqa: BLE001 - reported in the line + logger.error
        api_error = str(e)[:100]
    ctx = {"pause_active": gp.PAUSE_FLAG.exists(), "universe": set(universe or [])}
    rows = compare(day, bt, gp.read_ledger(), ctx)
    line = summary_line(day, rows, api_error)
    if dry_run:
        print(line)
        return 0
    first = write_rows(day, rows)
    if "ERROR" in line:
        logger.error(line)           # Telegram via the attached error handler
    else:
        logger.info(line)
        if first:
            rc = subprocess.run([sys.executable, str(_REPO / "scripts" / "send_telegram_alert.py"), line]).returncode
            if rc != 0:
                logger.error("parity Telegram send failed rc=%s", rc)
    return 0


def main(argv=None) -> int:
    """CLI entry."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from monitoring.cron_guard import attach_telegram_error_handler
    if not a.dry_run:
        attach_telegram_error_handler()
    return run_parity(a.dry_run)


if __name__ == "__main__":
    from monitoring.cron_guard import run_guarded
    sys.exit(run_guarded(main, "gap_short_parity"))
