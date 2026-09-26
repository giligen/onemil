"""
Independent rebuild of PREREG cells 1,445-1,455 (research/hod_entry/PREREG_1445.md).

Written from the PREREG prose ONLY, by an agent that has not read
cell_1445.py / test_cell_1445.py / RESULT_1445.md. Every data source is the
one the PREREG names:
  - base book:        research/hod_entry/causal_arming_causal.csv, status == 'fill'
  - bars:              research/bf_zero/bars_sip.db (table `bars`, columns
                        symbol, day, t, o, h, l, c, v; t is UTC ISO, converted
                        to America/New_York; RTH = 09:30-15:59 ET)
  - float snapshot:    data/cache.db table `universe.float_shares` (current
                        snapshot, disclosed as such in the PREREG)
  - Databento PIT daily bars: data/research/databento/equs_daily_2025_2026.parquet
    joined on instrument_id via data/research/databento/equs_instrument_symbol_map.csv
    (d0 <= day <= d1)
  - NBBO spread:       research/bf_zero/causal_filter/nbbo.csv (spread_mean on
                        (day, symbol))
  - measured stop/EOD slip: research/hod_entry/sip_cache_stopslip/<day>.pkl.gz,
    keyed "SYMBOL|exit_m|why|fill_min"; fallback = cell 1,443's holdout means
    (RESULT_1443.md): stop/stop_bar TRAIN-H2 35.9bps / VAL 34.8bps;
    eod/eod_fallback TRAIN-H2 11.5bps / VAL 9.7bps; target = 0 (limit fill,
    no slippage by construction).

Output: research/hod_entry/cell_1445_rebuild.csv, one row per base-book fill:
  day, symbol, fill_min, split, flag_1445..flag_1455, net_R_corr, net_R_corr_flat30

Does NOT touch cell_1445.py / test_cell_1445.py / RESULT_1445.md. Does NOT
recompute outcomes (R, raw_R, net_R, exit_price, ... are taken as-is from the
base CSV, per PREREG: "Outcomes are NOT recomputed").
"""
import csv
import gzip
import pickle
import sqlite3
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
BASE_CSV = REPO / "research/hod_entry/causal_arming_causal.csv"
BARS_DB = REPO / "research/bf_zero/bars_sip.db"
CACHE_DB = REPO / "data/cache.db"
DAILY_PARQUET = REPO / "data/research/databento/equs_daily_2025_2026.parquet"
SYMBOL_MAP_CSV = REPO / "data/research/databento/equs_instrument_symbol_map.csv"
NBBO_CSV = REPO / "research/bf_zero/causal_filter/nbbo.csv"
STOPSLIP_DIR = REPO / "research/hod_entry/sip_cache_stopslip"
OUT_CSV = REPO / "research/hod_entry/cell_1445_rebuild.csv"

ET = ZoneInfo("America/New_York")
RTH_OPEN_MIN = 9 * 60 + 30   # 570
RTH_CLOSE_MIN = 15 * 60 + 59  # 959

# Cell 1,443 holdout-mean slip fallbacks (RESULT_1443.md "Slip table"), bps.
STOP_FALLBACK_BPS = {"TRAIN": 35.9, "VAL": 34.8}
EOD_FALLBACK_BPS = {"TRAIN": 11.5, "VAL": 9.7}

FLAG_COLS = [f"flag_{n}" for n in range(1445, 1456)]


def log(msg: str) -> None:
    print(f"[cell_1445_rebuild] {msg}", flush=True)


# --------------------------------------------------------------------------
# Reference data: float snapshot, Databento PIT daily bars, NBBO, stop-slip
# --------------------------------------------------------------------------

def load_float_snapshot():
    """symbol -> float_shares (None if unknown), from cache.db `universe`
    (a CURRENT snapshot, not point-in-time -- disclosed in the PREREG)."""
    con = sqlite3.connect(f"file:{CACHE_DB}?mode=ro", uri=True)
    cur = con.cursor()
    cur.execute("SELECT symbol, float_shares FROM universe")
    out = {sym: fs for sym, fs in cur.fetchall()}
    con.close()
    log(f"float snapshot: {len(out)} symbols, {sum(1 for v in out.values() if v)} with float_shares set")
    return out


def load_daily_by_instrument():
    """instrument_id -> sorted list of (bar_date str, open, high, low, close)."""
    df = pd.read_parquet(DAILY_PARQUET, columns=["bar_date", "instrument_id", "open", "high", "low", "close"])
    df = df.sort_values(["instrument_id", "bar_date"])
    out = {}
    for iid, g in df.groupby("instrument_id", sort=False):
        out[int(iid)] = list(zip(g["bar_date"], g["open"], g["high"], g["low"], g["close"]))
    log(f"daily bars indexed for {len(out)} instrument_ids ({len(df)} rows)")
    return out


def load_symbol_map():
    """symbol -> sorted list of (d0, d1, instrument_id)."""
    out = {}
    with open(SYMBOL_MAP_CSV) as f:
        r = csv.DictReader(f)
        for row in r:
            out.setdefault(row["symbol"], []).append((row["d0"], row["d1"], int(row["instrument_id"])))
    for sym in out:
        out[sym].sort()
    return out


def resolve_instrument_id(symbol_map, symbol, day):
    for d0, d1, iid in symbol_map.get(symbol, []):
        if d0 <= day <= d1:
            return iid
    return None


def load_nbbo():
    """(day, symbol) -> spread_mean. If a (day,symbol) has >1 row (multiple
    entry_m), average spread_mean across rows (PREREG names the join as
    (day, symbol) only, with no entry_m disambiguation available for the
    exit side)."""
    sums = {}
    counts = {}
    with open(NBBO_CSV) as f:
        r = csv.DictReader(f)
        for row in r:
            key = (row["day"], row["symbol"])
            try:
                v = float(row["spread_mean"])
            except (ValueError, TypeError):
                continue
            sums[key] = sums.get(key, 0.0) + v
            counts[key] = counts.get(key, 0) + 1
    out = {k: sums[k] / counts[k] for k in sums}
    log(f"nbbo.csv: {len(out)} (day,symbol) keys ({sum(1 for c in counts.values() if c > 1)} averaged over >1 row)")
    return out


_stopslip_cache = {}


def load_stopslip_day(day):
    if day in _stopslip_cache:
        return _stopslip_cache[day]
    fp = STOPSLIP_DIR / f"{day}.pkl.gz"
    if fp.exists():
        with gzip.open(fp, "rb") as f:
            d = pickle.load(f)
    else:
        d = {}
    _stopslip_cache[day] = d
    return d


# --------------------------------------------------------------------------
# Bars: per (symbol, day) RTH series from bars_sip.db
# --------------------------------------------------------------------------

def fetch_rth_bars(con, symbol, day):
    """Return sorted list of (minute_of_day_et, o, h, l, c, v) for RTH bars
    (09:30-15:59 ET) of this symbol on this day. `t` is UTC ISO -> converted
    to ET; day boundary in ET may differ by 1 calendar day from the UTC
    date stored in `day`/`t`, but the base book's `day` is the ET trading
    day, so bars are filtered by ET-converted minute-of-day only, scanning
    both the UTC `day` and the day before/after to catch the ET-shifted
    rows (bars_sip.db `day` is presumed to be the same ET session key used
    by the base CSV; guarded by scanning +/-1 day if empty)."""
    cur = con.cursor()
    candidates = [day]
    rows = []
    for d in candidates:
        cur.execute(
            "SELECT t, o, h, l, c, v FROM bars WHERE symbol = ? AND day = ?",
            (symbol, d),
        )
        rows = cur.fetchall()
        if rows:
            break
    out = []
    for t, o, h, l, c, v in rows:
        ts = pd.Timestamp(t)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        ts_et = ts.tz_convert(ET)
        if ts_et.strftime("%Y-%m-%d") != day:
            continue
        minute = ts_et.hour * 60 + ts_et.minute + ts_et.second / 60.0
        if RTH_OPEN_MIN <= minute <= RTH_CLOSE_MIN:
            out.append((minute, o, h, l, c, v))
    out.sort(key=lambda r: r[0])
    return out


# --------------------------------------------------------------------------
# Flag computation for one fill row
# --------------------------------------------------------------------------

def compute_flags_and_cost(row, bars, float_shares, daily_by_instrument, symbol_map,
                            nbbo, split_key):
    """Returns dict of flag_1445..flag_1455, net_R_corr, net_R_corr_flat30.
    `bars` = RTH bars for (symbol, day) as returned by fetch_rth_bars.
    `split_key` = 'TRAIN' or 'VAL' (for the stop-slip holdout fallback)."""
    symbol = row["symbol"]
    day = row["day"]
    fill_min = float(row["fill_min"])
    R = float(row["R"])
    raw_R = float(row["raw_R"])
    cost_R = float(row["cost_R"])
    net_R = float(row["net_R"])
    exit_price = float(row["exit_price"])
    fill_price = float(row["fill"])
    exit_half_src = row["exit_half_src"]
    why = row["why"]
    level = float(row["level"])

    out = {c: None for c in FLAG_COLS}

    # --- arm bar j: last RTH bar with minute < fill_min ---
    prior_bars = [b for b in bars if b[0] < fill_min]
    if not prior_bars:
        # No causal bar available before the fill -> every bar-derived flag
        # is unknowable point-in-time. Leave as NaN (None); this is reported,
        # not silently defaulted to False.
        close_j = None
        run_high = run_low = None
        dollar_vol = None
        minutes_present = None
    else:
        close_j = prior_bars[-1][4]
        run_high = max(b[2] for b in prior_bars)
        run_low = min(b[3] for b in prior_bars)
        dollar_vol = sum(b[5] * b[4] for b in prior_bars)
        j_minute = prior_bars[-1][0]
        distinct_minutes = len({int(b[0]) for b in prior_bars if b[0] >= RTH_OPEN_MIN})
        total_minutes = int(j_minute) - RTH_OPEN_MIN + 1
        minutes_present = distinct_minutes / total_minutes if total_minutes > 0 else None

    # --- Databento PIT daily: prior session, prior-20-session high, full day ---
    iid = resolve_instrument_id(symbol_map, symbol, day)
    prev_close = prev_high = None
    prior20_high = None
    full_day_high = full_day_low = full_day_close = None
    if iid is not None:
        series = daily_by_instrument.get(iid, [])
        idx_today = None
        for i, (bd, o, h, l, c) in enumerate(series):
            if bd == day:
                idx_today = i
                full_day_high, full_day_low, full_day_close = h, l, c
                break
        if idx_today is not None and idx_today > 0:
            prev_bd, prev_o, prev_h, prev_l, prev_c = series[idx_today - 1]
            prev_close = prev_c
            prev_high = prev_h
            lookback = series[max(0, idx_today - 20):idx_today]
            if lookback:
                prior20_high = max(h for (_, _, h, _, _) in lookback)

    float_val = float_shares.get(symbol)
    float_known = float_val is not None
    float_le50m = float_known and float_val is not None and 0 < float_val <= 50_000_000

    # gap_j / range_j at arm bar j (causal terms)
    gap_j = range_j = None
    if close_j is not None and prev_close and prev_close > 0:
        gap_j = (close_j - prev_close) / prev_close * 100.0
    if run_high is not None and run_low is not None and run_low > 0:
        range_j = (run_high - run_low) / run_low * 100.0

    def mover(threshold):
        if gap_j is None and range_j is None:
            return None
        vals = [v for v in (gap_j, range_j) if v is not None]
        return max(vals) >= threshold if vals else None

    prev_close_band = prev_close is not None and 1 <= prev_close <= 30
    close_j_band = close_j is not None and 1 <= close_j <= 30

    # --- 1,445 PRIMARY ---
    m15 = mover(15.0)
    if None in (prev_close_band, close_j_band, m15) or close_j is None or prev_close is None:
        out["flag_1445"] = None
    else:
        out["flag_1445"] = bool(prev_close_band and float_known and float_le50m and close_j_band and m15)

    # --- 1,446: Q without the two price terms ---
    if m15 is None:
        out["flag_1446"] = None
    else:
        out["flag_1446"] = bool(float_known and float_le50m and m15)

    # --- 1,447: 1,446 at 10% threshold ---
    m10 = mover(10.0)
    if m10 is None:
        out["flag_1447"] = None
    else:
        out["flag_1447"] = bool(float_known and float_le50m and m10)

    # --- 1,448: pure mover term, no float ---
    out["flag_1448"] = m15

    # --- 1,449: float alone ---
    out["flag_1449"] = bool(float_known and float_le50m)

    # --- 1,450: level >= prior session high ---
    out["flag_1450"] = (level >= prev_high) if prev_high is not None else None

    # --- 1,451: level >= max high of prior 20 sessions ---
    out["flag_1451"] = (level >= prior20_high) if prior20_high is not None else None

    # --- 1,452: liquidity (bar-density) >= 0.90 ---
    out["flag_1452"] = (minutes_present >= 0.90) if minutes_present is not None else None

    # --- 1,453: dollar volume through bar j >= $1M ---
    out["flag_1453"] = (dollar_vol >= 1_000_000) if dollar_vol is not None else None

    # --- cost recomputation (needed for 1,454 and net_R_corr) ---
    half_entry = None
    exit_half = None
    if exit_half_src == "fill_instant":
        half_entry = (cost_R * R - 0.0002 * exit_price) / 2.0
        exit_half = half_entry
    elif exit_half_src == "nbbo":
        sm = nbbo.get((day, symbol))
        if sm is not None:
            exit_half = sm / 2.0
            half_entry = cost_R * R - exit_half - 0.0002 * exit_price

    # --- 1,454: quoted spread at fill <= 10bps ---
    if half_entry is not None and fill_price:
        spread_frac = 2.0 * half_entry / fill_price
        out["flag_1454"] = spread_frac <= 0.0010
    else:
        out["flag_1454"] = None

    # --- 1,455 PLACEBO: full-day mover_day_qualifies ---
    if full_day_high is not None and full_day_low is not None and full_day_low > 0 and full_day_close is not None:
        range_ok = (full_day_high - full_day_low) / full_day_low >= 0.10
        gap_ok = prev_close is not None and prev_close > 0 and (full_day_high - prev_close) / prev_close >= 0.10
        price_ok = 1 <= full_day_close <= 30
        float_ok = (not float_known) or float_le50m
        out["flag_1455"] = bool((range_ok or gap_ok) and price_ok and float_ok)
    else:
        out["flag_1455"] = None

    # --- corrected net R ---
    net_R_corr = None
    net_R_corr_flat30 = None
    if half_entry is not None:
        base_corr = net_R + half_entry / R
        slip_bps = 0.0
        slip_bps_flat = 0.0
        if why in ("stop", "stop_bar"):
            fallback = STOP_FALLBACK_BPS[split_key]
            slip_bps_flat = 30.0
        elif why in ("eod", "eod_fallback"):
            fallback = EOD_FALLBACK_BPS[split_key]
            slip_bps_flat = 30.0
        else:  # target
            fallback = 0.0
            slip_bps_flat = 0.0
        if fallback:
            day_cache = load_stopslip_day(day)
            key = f"{symbol}|{row['exit_m']}|{why}|{row['fill_min']}"
            entry = day_cache.get(key)
            if entry is not None and entry.get("measured") and entry.get("slip_bps") is not None:
                slip_bps = entry["slip_bps"]
            else:
                slip_bps = fallback
        slip_R = (slip_bps / 10000.0 * exit_price) / R
        slip_R_flat = (slip_bps_flat / 10000.0 * exit_price) / R
        net_R_corr = base_corr - slip_R
        net_R_corr_flat30 = base_corr - slip_R_flat

    out["net_R_corr"] = net_R_corr
    out["net_R_corr_flat30"] = net_R_corr_flat30
    return out


def main():
    log("loading reference data...")
    float_shares = load_float_snapshot()
    daily_by_instrument = load_daily_by_instrument()
    symbol_map = load_symbol_map()
    nbbo = load_nbbo()

    log("reading base book...")
    base_rows = []
    with open(BASE_CSV) as f:
        r = csv.DictReader(f)
        for row in r:
            if row["status"] == "fill":
                base_rows.append(row)
    log(f"base book: {len(base_rows)} fills")

    con = sqlite3.connect(f"file:{BARS_DB}?mode=ro", uri=True)

    out_rows = []
    n = len(base_rows)
    for i, row in enumerate(base_rows):
        if i % 1000 == 0:
            log(f"progress {i}/{n}")
        symbol, day = row["symbol"], row["day"]
        bars = fetch_rth_bars(con, symbol, day)
        split_key = "TRAIN" if row["split"] == "TRAIN" else "VAL"
        flags = compute_flags_and_cost(row, bars, float_shares, daily_by_instrument, symbol_map, nbbo, split_key)
        out_rows.append({
            "day": day,
            "symbol": symbol,
            "fill_min": row["fill_min"],
            "split": row["split"],
            **{c: flags[c] for c in FLAG_COLS},
            "net_R_corr": flags["net_R_corr"],
            "net_R_corr_flat30": flags["net_R_corr_flat30"],
        })
    con.close()
    log(f"progress {n}/{n} done")

    fieldnames = ["day", "symbol", "fill_min", "split"] + FLAG_COLS + ["net_R_corr", "net_R_corr_flat30"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r_ in out_rows:
            w.writerow(r_)
    log(f"wrote {OUT_CSV} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
