"""Diagnose touchgo divergence: live-logged decision vs consolidated bars.

Reads logs/touchgo_audit.jsonl (written by trading/touchgo_audit.py at every
live Rule M / Rule D evaluation), then for each record re-pulls Alpaca's
CONSOLIDATED 1-min bars and recomputes the decision. Flags three failure modes:

  MISMATCH    live bb_close_pos differs from consolidated by > --eps
              → live's streamed bar OHLC ≠ the official consolidated bar
  FLIP        consolidated fire-decision differs from what live did
              → the divergence actually changed the exit (real money)
  REKEY       breakout_bar_ts ≠ first consolidated bar with high > range_high
              → live keyed touchgo to the wrong bar

This is the diagnostic that settles EIDO/OSCR/TSDD: was it a streamed-vs-
consolidated close gap, or a re-keying miss?

Usage:
  python3 scripts/audit_touchgo_live_vs_consolidated.py [--since YYYY-MM-DD] [--eps 0.02]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(str(Path(__file__).resolve().parent.parent / ".env"))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_sources.alpaca_client import AlpacaClient

AUDIT_PATH = os.getenv(
    "ORB_TOUCHGO_AUDIT_PATH",
    str(Path(__file__).resolve().parent.parent / "logs" / "touchgo_audit.jsonl"),
)


def _orb_client() -> AlpacaClient:
    return AlpacaClient(
        os.getenv("ALPACA_ORB_API_KEY"), os.getenv("ALPACA_ORB_API_SECRET"),
        paper=os.getenv("ALPACA_ORB_PAPER", "true").lower() == "true",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2000-01-01")
    ap.add_argument("--eps", type=float, default=0.02,
                    help="bb_close_pos mismatch tolerance")
    args = ap.parse_args()

    if not Path(AUDIT_PATH).exists():
        print(f"No audit log at {AUDIT_PATH} — nothing to diagnose yet "
              f"(it populates once the instrumented engine runs live).")
        return

    recs = []
    with open(AUDIT_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("trade_date", "") >= args.since:
                recs.append(r)
    if not recs:
        print(f"No records since {args.since}.")
        return

    client = _orb_client()
    bars_cache: dict = {}

    def day_bars(sym: str, day: str):
        key = (sym, day)
        if key not in bars_cache:
            d0 = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            df = client.get_historical_1min_bars(
                sym, d0 + timedelta(hours=13), d0 + timedelta(hours=20))
            if df is not None and "timestamp" not in df.columns:
                df = df.reset_index()
            bars_cache[key] = df
        return bars_cache[key]

    def bar_at(df, ts_iso):
        if df is None or df.empty or not ts_iso:
            return None
        t = datetime.fromisoformat(ts_iso)
        minute = t.replace(second=0, microsecond=0)
        m = df[df["timestamp"].dt.tz_convert("UTC").dt.floor("min") ==
               minute.astimezone(timezone.utc)]
        return m.iloc[0] if len(m) else None

    print(f"{'date':<11}{'sym':<7}{'rule':<5}{'fired':<6}"
          f"{'live_val':>9}{'consol':>9}{'Δ':>8}  flags")
    print("-" * 78)
    n_mismatch = n_flip = n_rekey = 0
    for r in sorted(recs, key=lambda x: (x.get("trade_date", ""), x.get("symbol", ""))):
        sym, day, rule = r["symbol"], r["trade_date"], r["rule"]
        df = day_bars(sym, day)
        flags = []
        live_val = consol_val = None
        if rule == "M":
            live_val = r.get("bb_close_pos")
            bb = bar_at(df, r.get("breakout_bar_ts"))
            if bb is not None:
                h, l, c = float(bb["high"]), float(bb["low"]), float(bb["close"])
                consol_val = (c - l) / (h - l) if h > l else 0.0
                thr = r.get("rule_m_threshold", 0.5)
                if live_val is not None and abs(live_val - consol_val) > args.eps:
                    flags.append("MISMATCH"); n_mismatch += 1
                live_fire = bool(r.get("fired"))
                consol_fire = consol_val < thr
                if live_fire != consol_fire:
                    flags.append(f"FLIP(consol={'fire' if consol_fire else 'hold'})"); n_flip += 1
            # re-keying: is breakout_bar_ts the first bar > range_high?
            if df is not None and not df.empty:
                rh = r.get("range_high")
                post = df[df["timestamp"] >= df["timestamp"].min()]
                bo = post[post["high"] > rh].head(1)
                if len(bo):
                    true_ts = bo.iloc[0]["timestamp"].tz_convert("UTC").floor("min")
                    keyed = datetime.fromisoformat(r["breakout_bar_ts"]).astimezone(timezone.utc).replace(second=0, microsecond=0)
                    if true_ts != keyed:
                        flags.append(f"REKEY(true={true_ts.strftime('%H:%M')})"); n_rekey += 1
        else:  # D
            live_val = r.get("b1_revert_R")
            b1 = bar_at(df, r.get("b1_ts"))
            rs = r.get("range_size") or 0
            if b1 is not None and rs > 0:
                consol_val = (r["entry_price"] - float(b1["low"])) / rs
                thr = r.get("rule_d_revert_R", 0.75)
                if live_val is not None and abs(live_val - consol_val) > 0.05:
                    flags.append("MISMATCH"); n_mismatch += 1
                if bool(r.get("fired")) != (consol_val >= thr):
                    flags.append("FLIP"); n_flip += 1

        lv = f"{live_val:.3f}" if isinstance(live_val, (int, float)) else "—"
        cv = f"{consol_val:.3f}" if isinstance(consol_val, (int, float)) else "—"
        dd = (f"{live_val - consol_val:+.3f}"
              if isinstance(live_val, (int, float)) and isinstance(consol_val, (int, float))
              else "—")
        print(f"{day:<11}{sym:<7}{rule:<5}{str(r.get('fired')):<6}"
              f"{lv:>9}{cv:>9}{dd:>8}  {' '.join(flags)}")

    print("-" * 78)
    print(f"records: {len(recs)}  |  MISMATCH: {n_mismatch}  FLIP: {n_flip}  REKEY: {n_rekey}")
    if n_flip:
        print("\n⚠️  FLIP rows are the money-losers — live's touchgo decision diverged "
              "from consolidated and changed the exit. Root cause:")
        print("    MISMATCH+FLIP → streamed bar OHLC ≠ consolidated (close-price gap)")
        print("    REKEY+FLIP    → live keyed to the wrong breakout bar")


if __name__ == "__main__":
    main()
