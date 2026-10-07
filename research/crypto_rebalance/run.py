"""Leveraged-crypto-ETF close-rebalance study (PREREG.md, cell 1).

Usage (always via the research cage):
  bash scripts/research_run.sh -m 2500M python3 research/crypto_rebalance/run.py --fetch [--asset BTC/USD]
  bash scripts/research_run.sh -m 2500M python3 research/crypto_rebalance/run.py --score

Frozen conventions (written before any number was read):
  * Bars are labelled by their START time (Alpaca). "close(HH:MM ET)" = close of the bar stamped HH:MM ET.
  * A required stamp that has no bar (Alpaca omits zero-volume minutes) is forward-filled from the last bar stamped
    within MAX_FFILL_MIN minutes before it; older than that -> the day is LOST (counted, never imputed).
  * Sessions = NYSE trading days (rule-based calendar below; no calendar package on this node).
  * Position = sign(r_day) held from the signal-end stamp close to the exit stamp close, every session.
  * Cost = round-trip bp subtracted from every traded day (0 / 5 / 30).
"""
import argparse
import calendar
import datetime as dt
import logging
import math
import os
import shutil
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
ET = ZoneInfo("America/New_York")
UTC = dt.timezone.utc
FETCH_START = dt.date(2022, 1, 1)
FETCH_END = dt.date(2026, 10, 6)          # inclusive
PRIMARY = (dt.date(2024, 6, 4), dt.date(2026, 10, 6))
PLACEBO_PERIOD = (dt.date(2022, 1, 1), dt.date(2024, 6, 3))
MAX_FFILL_MIN = 5
COSTS_BP = {"gross": 0.0, "5bp": 5.0, "30bp": 30.0}
MIN_FREE_GB = 5.0
MIN_COMPLETENESS = 0.95
log = logging.getLogger("crypto_rebalance")


# ------------------------------------------------------------------ NYSE calendar (rule based)
def _easter_sunday(year):
    """Easter Sunday (Meeus/Jones/Butcher computus)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    return dt.date(year, month, ((h + l - 7 * m + 114) % 31) + 1)


def _nth_weekday(year, month, weekday, n):
    """n-th (1-based) given weekday of a month."""
    d = dt.date(year, month, 1)
    return d + dt.timedelta(days=(weekday - d.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(year, month, weekday):
    """Last given weekday of a month."""
    d = dt.date(year, month, calendar.monthrange(year, month)[1])
    return d - dt.timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d):
    """Saturday holiday -> Friday, Sunday holiday -> Monday."""
    if d.weekday() == 5:
        return d - dt.timedelta(days=1)
    if d.weekday() == 6:
        return d + dt.timedelta(days=1)
    return d


def nyse_holidays(year):
    """Full-day NYSE closures for a year (2022-2026 verified by rule; plus the 2025-01-09 national mourning day)."""
    hols = {
        _observed(dt.date(year, 1, 1)), _nth_weekday(year, 1, 0, 3), _nth_weekday(year, 2, 0, 3),
        _easter_sunday(year) - dt.timedelta(days=2), _last_weekday(year, 5, 0), _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 11, 3, 4), _observed(dt.date(year, 7, 4)), _observed(dt.date(year, 12, 25)),
    }
    if year >= 2022:
        hols.add(_observed(dt.date(year, 6, 19)))
    if year == 2025:
        hols.add(dt.date(2025, 1, 9))
    return hols


def is_trading_day(d):
    """True when NYSE is open for a regular or early-close session."""
    return d.weekday() < 5 and d not in nyse_holidays(d.year)


def is_early_close(d):
    """13:00 ET early closes: day after Thanksgiving, Dec 24 (weekday), Jul 3 (weekday, when Jul 4 is a weekday)."""
    if d == _nth_weekday(d.year, 11, 3, 4) + dt.timedelta(days=1):
        return True
    if d.month == 12 and d.day == 24 and d.weekday() < 5:
        return True
    return d.month == 7 and d.day == 3 and d.weekday() < 4 and is_trading_day(d)


# ------------------------------------------------------------------ fetch
def month_starts(start, end):
    """Yield (first_day, first_day_of_next_month) pairs covering [start, end]."""
    y, m = start.year, start.month
    while dt.date(y, m, 1) <= end:
        nxt = dt.date(y + (m == 12), m % 12 + 1, 1)
        yield dt.date(y, m, 1), nxt
        y, m = nxt.year, nxt.month


def month_path(asset, first):
    """Cache path for one asset-month."""
    return os.path.join(DATA, asset.replace("/", ""), f"{first:%Y-%m}.parquet")


def make_client():
    """Alpaca crypto client; keyless first (public endpoint), keys from .env only if the keyless call is refused."""
    from alpaca.data.historical import CryptoHistoricalDataClient
    return CryptoHistoricalDataClient()


def fetch_month(client, asset, first, nxt):
    """Fetch one month of 1-min bars; atomic write of the month parquet. Returns row count."""
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame
    path = month_path(asset, first)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    start = dt.datetime(first.year, first.month, first.day, tzinfo=UTC)
    end = min(dt.datetime(nxt.year, nxt.month, nxt.day, tzinfo=UTC), dt.datetime(2026, 10, 7, tzinfo=UTC))
    last_err = None
    for attempt in range(5):
        try:
            req = CryptoBarsRequest(symbol_or_symbols=asset, timeframe=TimeFrame.Minute, start=start, end=end)
            df = client.get_crypto_bars(req).df
            break
        except Exception as e:  # network / rate-limit: back off and retry, then raise (never silently drop a month)
            last_err = e
            log.warning("%s %s attempt %d failed: %s", asset, first, attempt + 1, e)
            time.sleep(3 * (attempt + 1))
    else:
        raise RuntimeError(f"fetch failed for {asset} {first}: {last_err}")
    if len(df):
        df = df.reset_index()
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    else:
        df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return len(df)


def do_fetch(assets):
    """Fetch every month for the assets, resume-safe (existing month files are skipped; current partial month is not)."""
    free_gb = shutil.disk_usage("/").free / 1e9
    log.info("free disk %.1f GB", free_gb)
    if free_gb < MIN_FREE_GB:
        raise SystemExit(f"ABORT: only {free_gb:.1f} GB free (< {MIN_FREE_GB})")
    client = make_client()
    for asset in assets:
        months = list(month_starts(FETCH_START, FETCH_END))
        for i, (first, nxt) in enumerate(months, 1):
            if os.path.exists(month_path(asset, first)):
                log.info("%s %s cached (%d/%d)", asset, f"{first:%Y-%m}", i, len(months))
                continue
            n = fetch_month(client, asset, first, nxt)
            log.info("%s %s fetched %d bars (%d/%d)", asset, f"{first:%Y-%m}", n, i, len(months))
            time.sleep(0.4)


# ------------------------------------------------------------------ build per-day table
def load_close_open(asset):
    """Load all month files into a minute-indexed (UTC) frame of open/close."""
    frames = []
    for first, _ in month_starts(FETCH_START, FETCH_END):
        p = month_path(asset, first)
        if os.path.exists(p):
            frames.append(pd.read_parquet(p))
    if not frames:
        return pd.DataFrame(columns=["open", "close"])
    df = pd.concat(frames).drop_duplicates("timestamp").set_index("timestamp").sort_index()
    return df[["open", "close"]]


def stamp_value(df_idx, closes, opens, day, hhmm, field):
    """Value of `field` ('close'/'open') of the bar stamped day@hhmm ET; ffill <= MAX_FFILL_MIN min, else NaN."""
    ts = dt.datetime.combine(day, dt.time(*hhmm), tzinfo=ET).astimezone(UTC)
    pos = df_idx.searchsorted(pd.Timestamp(ts), side="right") - 1
    if pos < 0:
        return np.nan
    gap = (pd.Timestamp(ts) - df_idx[pos]).total_seconds() / 60.0
    if gap > MAX_FFILL_MIN:
        return np.nan
    return (closes if field == "close" else opens)[pos]


def build_days(asset):
    """Per-session table: r_day (09:30->15:30), 15:30->16:00, 16:00->16:30, next-open variant, placebo window."""
    df = load_close_open(asset)
    idx = df.index
    closes, opens = df["close"].to_numpy(), df["open"].to_numpy()
    rows, lost = [], 0
    d = FETCH_START
    sessions = 0
    while d <= FETCH_END:
        if is_trading_day(d):
            sessions += 1
            v = lambda hhmm, f="close": stamp_value(idx, closes, opens, d, hhmm, f)  # noqa: E731
            c0930, c1130, c1200 = v((9, 30)), v((11, 30)), v((12, 0))
            c1530, c1600, c1630 = v((15, 30)), v((16, 0)), v((16, 30))
            o1531 = v((15, 31), "open")
            vals = [c0930, c1130, c1200, c1530, c1600, c1630]
            period = "primary" if PRIMARY[0] <= d <= PRIMARY[1] else ("placebo_period" if d <= PLACEBO_PERIOD[1] else "")
            ok_main = not any(math.isnan(x) for x in (c0930, c1530, c1600, c1630))
            ok_pw = not any(math.isnan(x) for x in (c0930, c1130, c1200))
            if not ok_main:
                lost += 1
            rows.append({
                "date": d.isoformat(), "period": period, "early_close": is_early_close(d),
                "ok_main": ok_main, "ok_placebo_window": ok_pw,
                "r_day": c1530 / c0930 - 1 if ok_main else np.nan,
                "ret_1530_1600": c1600 / c1530 - 1 if ok_main else np.nan,
                "ret_1600_1630": c1630 / c1600 - 1 if ok_main else np.nan,
                "ret_1531open_1600": c1600 / o1531 - 1 if ok_main and not math.isnan(o1531) else np.nan,
                "r_sig_0930_1130": c1130 / c0930 - 1 if ok_pw else np.nan,
                "ret_1130_1200": c1200 / c1130 - 1 if ok_pw else np.nan,
            })
        d += dt.timedelta(days=1)
    out = pd.DataFrame(rows)
    log.info("%s: %d sessions, %d lost (main rule), %d lost (placebo window)", asset, sessions, lost,
             int((~out.ok_placebo_window).sum()))
    return out


# ------------------------------------------------------------------ statistics
def tstat(x):
    """Mean, t-stat (iid == day-clustered here: one observation per day), n."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return (np.nan, np.nan, n)
    sd = x.std(ddof=1)
    return (x.mean(), x.mean() / (sd / math.sqrt(n)) if sd > 0 else np.nan, n)


def ex_top5(x):
    """Mean of the day returns after dropping the best 5 % of days (by the same net return)."""
    x = np.sort(np.asarray(x, dtype=float)[~np.isnan(x)])
    k = int(math.ceil(0.05 * len(x)))
    return x[:len(x) - k].mean() if len(x) > k else np.nan


def mde(x, z=2.5):
    """Minimum per-day mean detectable at t = z given the sample sd and n."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return z * x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 2 else np.nan


def position_returns(sub, sig_col, ret_col, cost_bp):
    """sign(signal) * forward return minus the round-trip cost on every traded day."""
    pos = np.sign(sub[sig_col].to_numpy())
    return pos * sub[ret_col].to_numpy() - np.where(pos != 0, cost_bp / 1e4, 0.0)


def score_block(label, sub, sig_col, ret_col, rows):
    """Append (label, cost, slice) stat rows: all / terciles / halves / ex-top-5 % for one data slice."""
    sub = sub.dropna(subset=[sig_col, ret_col]).sort_values("date").reset_index(drop=True)
    if len(sub) < 6:
        return
    mid = len(sub) // 2
    abs_sig = sub[sig_col].abs()
    q1, q2 = abs_sig.quantile([1 / 3, 2 / 3])
    slices = {
        "all": sub, "tercile_low": sub[abs_sig <= q1], "tercile_mid": sub[(abs_sig > q1) & (abs_sig <= q2)],
        "tercile_high": sub[abs_sig > q2], "half1": sub.iloc[:mid], "half2": sub.iloc[mid:],
    }
    for cname, bp in COSTS_BP.items():
        for sname, s in slices.items():
            m, t, n = tstat(position_returns(s, sig_col, ret_col, bp))
            rows.append({"cell": label, "cost": cname, "slice": sname, "mean_per_day": m, "t": t, "n": n})
        full = position_returns(sub, sig_col, ret_col, bp)
        rows.append({"cell": label, "cost": cname, "slice": "ex_top5pct", "mean_per_day": ex_top5(full),
                     "t": np.nan, "n": len(full)})
        rows.append({"cell": label, "cost": cname, "slice": "MDE_t2.5", "mean_per_day": mde(full), "t": np.nan,
                     "n": len(full)})


def do_score():
    """Build the per-day table, score every pre-declared cell, write results.csv + results_stats.csv."""
    btc = build_days("BTC/USD")
    btc.to_csv(os.path.join(HERE, "results.csv"), index=False)
    rows = []
    prim = btc[btc.period == "primary"]
    plc = btc[btc.period == "placebo_period"]
    score_block("primary_1530_1600", prim, "r_day", "ret_1530_1600", rows)
    score_block("placebo_period_1530_1600", plc, "r_day", "ret_1530_1600", rows)
    pw = prim[prim.ok_placebo_window]
    score_block("placebo_window_1130_1200", pw, "r_sig_0930_1130", "ret_1130_1200", rows)
    score_block("reversal_1600_1630_primary", prim, "r_day", "ret_1600_1630", rows)
    score_block("reversal_1600_1630_placebo_period", plc, "r_day", "ret_1600_1630", rows)
    score_block("exec_next_open_primary", prim, "r_day", "ret_1531open_1600", rows)
    # secondary (declared fetch, same rule): ETH/USD, primary period, not part of the verdict
    eth_path = os.path.join(DATA, "ETHUSD")
    if os.path.isdir(eth_path) and os.listdir(eth_path):
        eth = build_days("ETH/USD")
        eth.to_csv(os.path.join(HERE, "results_eth.csv"), index=False)
        score_block("ETH_primary_1530_1600", eth[eth.period == "primary"], "r_day", "ret_1530_1600", rows)
        score_block("ETH_placebo_period_1530_1600", eth[eth.period == "placebo_period"], "r_day", "ret_1530_1600", rows)
    stats = pd.DataFrame(rows)
    stats.to_csv(os.path.join(HERE, "results_stats.csv"), index=False)
    log.info("wrote results.csv (%d days) and results_stats.csv (%d rows)", len(btc), len(stats))
    comp = {}
    for per, sub in (("primary", prim), ("placebo_period", plc)):
        comp[per] = (int(sub.ok_main.sum()), len(sub))
    comp["placebo_window_primary"] = (int(prim.ok_placebo_window.sum()), len(prim))
    pd.Series({k: f"{a}/{b}={a / b:.4f}" for k, (a, b) in comp.items()}).to_csv(
        os.path.join(HERE, "completeness.csv"), header=["ok/sessions"])
    print(stats.round(6).to_string())
    print(comp)


def main():
    """CLI entry."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--asset", default="", help="fetch only this asset (default BTC/USD then ETH/USD)")
    a = ap.parse_args()
    if a.fetch:
        do_fetch([a.asset] if a.asset else ["BTC/USD", "ETH/USD"])
    if a.score:
        do_score()


if __name__ == "__main__":
    main()
