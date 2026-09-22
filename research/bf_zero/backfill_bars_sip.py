"""Backfill the missing (symbol, day) 1-minute bars of the HOD-break population into bars_sip.db.

Why: the failed-break short study (cells 1,357-1,358) found only 7,090 of the 15,656 feature symbol-days
present in `research/bf_zero/bars_sip.db` (45.3 %), below CLAUDE.md's 80 % availability rail, so its result
is VOID rather than negative. This script fetches the missing pairs from Alpaca SIP (04:00-20:00 ET, the same
span the existing rows carry) and APPENDS them to the same table. It never deletes or rewrites a row.

Resume: processed days are recorded in backfill_state.json; re-running continues where it stopped.
Usage: python3 research/bf_zero/backfill_bars_sip.py [--limit-days N]
"""
import argparse
import json
import logging
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DB = ROOT / "research/bf_zero/bars_sip.db"
FEATURES = ROOT / "research/bf_zero/causal_filter/features.csv"
STATE = ROOT / "research/bf_zero/backfill_state.json"
BATCH = 100          # symbols per request
ET_OFFSET_H = 4      # EDT; the span is padded so the exact offset never clips the session

log = logging.getLogger("backfill")


def missing_pairs() -> dict:
    """Map day -> [symbols] present in features.csv but absent from bars_sip.db."""
    log.info("reading features + existing coverage (a full-table distinct scan, ~4 min) ...")
    feat = pd.read_csv(FEATURES, usecols=["day", "symbol"]).drop_duplicates()
    con = sqlite3.connect(DB)
    have = pd.read_sql("select distinct symbol, day from bars", con)
    con.close()
    have_set = set(map(tuple, have[["symbol", "day"]].values))
    out: dict = {}
    for sym, day in feat[["symbol", "day"]].values:
        if (sym, day) not in have_set:
            out.setdefault(str(day), []).append(str(sym))
    log.info("feature pairs %d, covered %d, missing %d across %d days",
             len(feat), len(have_set & set(map(tuple, feat[["symbol", "day"]].values))),
             sum(len(v) for v in out.values()), len(out))
    return out


def fetch_day(client, symbols: list, day: str) -> pd.DataFrame:
    """1-minute SIP bars for `symbols` on `day`, 04:00-20:00 ET, as the bars-table column shape."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    start = datetime.fromisoformat(day).replace(tzinfo=timezone.utc) + timedelta(hours=ET_OFFSET_H + 4)
    end = start + timedelta(hours=16)
    req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Minute,
                           start=start, end=end, feed="sip")
    df = client.get_stock_bars(req).df
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    return pd.DataFrame({
        "symbol": df["symbol"], "day": day,
        "t": pd.to_datetime(df["timestamp"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "o": df["open"], "h": df["high"], "l": df["low"], "c": df["close"], "v": df["volume"]})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit-days", type=int, default=0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    from config import Config
    from alpaca.data.historical import StockHistoricalDataClient
    cfg = Config()
    client = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_api_secret)
    todo = missing_pairs()
    done = set(json.loads(STATE.read_text())["days"]) if STATE.exists() else set()
    days = sorted(d for d in todo if d not in done)
    if a.limit_days:
        days = days[:a.limit_days]
    log.info("backfilling %d days (%d already done)", len(days), len(done))
    con = sqlite3.connect(DB)
    total = 0
    for i, day in enumerate(days, 1):
        syms = sorted(set(todo[day]))
        rows = 0
        for j in range(0, len(syms), BATCH):
            chunk = syms[j:j + BATCH]
            for attempt in range(3):
                try:
                    df = fetch_day(client, chunk, day)
                    if not df.empty:
                        df.to_sql("bars", con, if_exists="append", index=False)
                        rows += len(df)
                    break
                except Exception as e:  # noqa: BLE001
                    log.warning("day %s chunk %d attempt %d failed: %s", day, j // BATCH, attempt + 1, e)
                    time.sleep(2 * (attempt + 1))
            else:
                log.error("day %s chunk %d SKIPPED after 3 attempts (day not marked done)", day, j // BATCH)
                break
        else:
            con.commit()
            done.add(day)
            STATE.write_text(json.dumps({"days": sorted(done)}))
        total += rows
        log.info("[%d/%d] %s: %d symbols, +%d bars (cumulative %d)", i, len(days), day, len(syms), rows, total)
    con.close()
    log.info("DONE — %d bars appended over %d days", total, len(days))
    return 0


if __name__ == "__main__":
    sys.exit(main())
