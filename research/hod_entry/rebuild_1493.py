#!/usr/bin/env python3
"""Independent rebuild of PREREG_1493 (the retest bounce, whole exit surface),
built from the prose of research/hod_entry/PREREG_1493.md only -- never reading
cell_1493.py / test_cell_1493.py / cell_1493_surface.csv / cell_1493_fills.csv /
RESULT_1493.md.

Population: the 1,481 retest fills (rebuild_1481_fills.csv, status == 'fill').
Entry = level - $0.01 at the first print strictly below the level within 15 RTH
minutes of the base break fill (already computed upstream as the 'entry' /
'retest_ts' / 'retest_minute' columns).

Exit grid: stop in {0.5,1.0,1.5,2.0,3.0 % below entry, CL = consolidation low}
x exit in {+0.5,+0.75,+1.0,+1.5,+2.0,+3.0 % limit target, NONE (15:55 only),
T30, T60 (flat at the open of the 30th/60th RTH minute after t_r)} = 6x9=54,
plus M = the mirror of the 1,480 short (stop = min(dip-bar low - $0.01,
entry*0.99), target = entry*1.02), long direction (the bounce). 55 cells total.

Path resolution:
  1) Inside the retest minute: the tape (sip_cache_148x pickles) after the
     retest print (retest_ts), first trade to touch stop or target wins
     (time order).
  2) From the next RTH minute: minute bars (bars_fills_1478.db), walk_path
     semantics -- a bar touching both stop and target resolves to stop; a
     gap-through open (o beyond the stop) fills at the open; a target needs
     high > target and always fills AT the target (no gap credit); T30/T60
     exit at the bar's OPEN once its minute index reaches the deadline,
     checked before any stop/target test on that bar.
  3) If nothing triggers by 15:55 ET: exit at the 15:55 bar's close.

Costs: entry and target are passive/limit -> zero cost. Stop exits carry the
stop-limit standard blended bps (SLIP_STOP_BPS in cell_1478.py, already folds
the 12% no-fill tail). Time (T30/T60) and EOD (15:55) exits are marked "at the
bid" per RESULT_1443.md -- modelled as a flat bps haircut off the raw bar
price (no bid-side quotes exist that far from the retest instant in the tape
cache), TRAIN-H2 11.5 bps / VAL 9.7 bps.
"""
import datetime
import os
import sqlite3
import sys
import time
import zoneinfo

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ET = zoneinfo.ZoneInfo("America/New_York")
UTC = datetime.timezone.utc

# ---- cost constants (named in the task, sourced from the programme) --------
SLIP_STOP_BPS = {"TRAIN": 0.88 * 2.9 + 0.12 * 94.0, "VAL": 0.88 * 3.2 + 0.12 * 76.0}
EOD_TIME_BPS = {"TRAIN": 11.5, "VAL": 9.7}
L3_THRESHOLD = 0.3070
RTH_OPEN_MIN = 570   # 09:30 ET
RTH_EOD_MIN = 955    # 15:55 ET
PLACEBO_LO, PLACEBO_HI = 585, 900  # 09:45 .. 15:00 ET
SEED = 1493

FIXED_STOPS = [0.5, 1.0, 1.5, 2.0, 3.0]   # % below entry
FIXED_TARGETS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]  # % above entry
TIME_EXITS = ["NONE", "T30", "T60"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ loading
def load_population(n_smoke=None):
    df = pd.read_csv(os.path.join(HERE, "rebuild_1481_fills.csv"))
    df = df[df["status"] == "fill"].copy()
    assert len(df) == 8973, f"expected 8973 fill rows, got {len(df)}"
    df["retest_ts"] = df["retest_ts"].astype(float)
    if n_smoke:
        df = df.iloc[:n_smoke].copy()
    return df.reset_index(drop=True)


def load_l3(df):
    l3 = pd.read_csv(os.path.join(HERE, "model_1478_L3_predictions.csv"),
                      usecols=["day", "symbol", "fill_min", "hgb_prob_L3", "store_served_1438"])
    l3["l3_top_tercile"] = l3["hgb_prob_L3"] >= L3_THRESHOLD
    merged = df.merge(l3, on=["day", "symbol", "fill_min"], how="left")
    log(f"L3/cache-flag merge: {merged['hgb_prob_L3'].notna().sum()}/{len(merged)} matched")
    return merged


def et_minute_of_day(ts_ns):
    """Convert a unix-ns epoch timestamp to (date_str, minute_of_day_float) in America/New_York."""
    dt = datetime.datetime.fromtimestamp(ts_ns / 1e9, tz=UTC).astimezone(ET)
    return dt.date().isoformat(), dt.hour * 60 + dt.minute + dt.second / 60.0 + dt.microsecond / 6e7


# ------------------------------------------------------------------ tape/bars
class BarStore:
    """Lazy, cached per-(symbol,day) minute bar loader from bars_fills_1478.db."""

    def __init__(self, path):
        self.con = sqlite3.connect(path)
        self.cache = {}

    def get(self, symbol, day):
        key = (symbol, day)
        if key not in self.cache:
            q = "SELECT t,o,h,l,c FROM bars WHERE symbol=? AND day=? ORDER BY t"
            rows = self.con.execute(q, (symbol, day)).fetchall()
            recs = []
            for t, o, h, l, c in rows:
                dt = datetime.datetime.fromisoformat(t).astimezone(ET)
                m = dt.hour * 60 + dt.minute
                recs.append((m, o, h, l, c))
            self.cache[key] = recs
        return self.cache[key]


def load_tape(symbol, day, minute_floor):
    """Return sorted (ts, price) trades for symbol/day, tried against sip_cache_1481
    then sip_cache_1480, keyed by the M= file-name minute bucket."""
    for cache_dir in ("sip_cache_1481", "sip_cache_1480"):
        fp = os.path.join(HERE, cache_dir, f"{symbol}_{day}_{minute_floor}.pkl")
        if os.path.exists(fp):
            import pickle
            with open(fp, "rb") as fh:
                trades, _quotes = pickle.load(fh)
            return trades
    return None


# ------------------------------------------------------------------ walk
def walk_exit(entry, stop_price, target_price, exit_kind, retest_ts, minute_r,
              trades, bars, deadline_minute=None):
    """Resolve one cell's exit. Returns (exit_minute, exit_price_raw, why).
    why in {'stop','target','time','eod'}."""
    m_r_floor = int(minute_r)
    # Phase 1: tape, same ET minute as the retest print, strictly after it.
    if trades is not None and len(trades):
        sel = trades[(trades["ts"] > retest_ts)]
        if len(sel):
            # restrict to the same calendar minute as retest_ts
            mins = sel["ts"].apply(lambda ts: et_minute_of_day(ts)[1])
            sel = sel[mins.astype(int) == m_r_floor]
        for _, row in sel.sort_values("ts").iterrows():
            p = row["price"]
            if target_price is not None and p >= target_price:
                return m_r_floor, target_price, "target"
            if p <= stop_price:
                return m_r_floor, p, "stop"
    # Phase 2: minute bars from m_r_floor+1 onward.
    last_bar = None
    for (m, o, h, l, c) in bars:
        if m <= m_r_floor or m > RTH_EOD_MIN:
            continue
        last_bar = (m, o, h, l, c)
        if exit_kind in ("T30", "T60") and m >= deadline_minute:
            return m, o, "time"
        if target_price is not None:
            if o <= stop_price:
                return m, o, "stop"
            if l <= stop_price and h >= target_price:
                return m, stop_price, "stop"
            if h >= target_price:
                return m, target_price, "target"
            if l <= stop_price:
                return m, stop_price, "stop"
        else:
            if o <= stop_price:
                return m, o, "stop"
            if l <= stop_price:
                return m, stop_price, "stop"
    if last_bar is not None:
        return last_bar[0], last_bar[4], "eod"
    # no bars at all past the retest minute (very late fill) -- fall back to entry
    return m_r_floor, entry, "eod"


def apply_cost(exit_price_raw, why, split):
    if why == "stop":
        return exit_price_raw * (1 - SLIP_STOP_BPS[split] / 10000.0)
    if why in ("time", "eod"):
        return exit_price_raw * (1 - EOD_TIME_BPS[split] / 10000.0)
    return exit_price_raw  # target, no cost


def cell_defs():
    cells = []
    for s in FIXED_STOPS:
        cells.append(("pct", s))
    cells.append(("CL", None))
    stops = cells
    exits = [("tgt", t) for t in FIXED_TARGETS] + [("time", k) for k in TIME_EXITS]
    grid = []
    for stype, sval in stops:
        for etype, eval_ in exits:
            name = (f"s{sval}" if stype == "pct" else "sCL") + "_" + \
                   (f"t{eval_}" if etype == "tgt" else eval_)
            grid.append({"name": name, "stop_type": stype, "stop_val": sval,
                         "exit_type": etype, "exit_val": eval_})
    grid.append({"name": "M_mirror", "stop_type": "M", "stop_val": None,
                 "exit_type": "M", "exit_val": None})
    assert len(grid) == 55, len(grid)
    return grid


def run(df, bar_store, cells, rng_seed=SEED, verbose_every=200):
    fill_rows = []
    n = len(df)
    for i, row in enumerate(df.itertuples(index=False)):
        if i % verbose_every == 0:
            log(f"fill {i}/{n} ({row.symbol} {row.day})")
        symbol, day = row.symbol, row.day
        split = row.split
        entry = float(row.entry)
        retest_ts = float(row.retest_ts)
        _, minute_r_exact = et_minute_of_day(retest_ts)
        m_r_floor = int(minute_r_exact)
        cl_stop = float(row.stop)  # consolidation low of the base fill
        dip_low = float(row.dip_low) if not pd.isna(row.dip_low) else None
        trades = load_tape(symbol, day, m_r_floor)
        bars = bar_store.get(symbol, day)
        for cell in cells:
            if cell["stop_type"] == "M":
                stop_price = min(dip_low - 0.01 if dip_low is not None else entry * 0.99,
                                  entry * 0.99)
                target_price = entry * 1.02
                exit_kind = "tgt"
                deadline = None
            elif cell["stop_type"] == "CL":
                stop_price = cl_stop
                target_price, exit_kind, deadline = _exit_params(cell, entry)
            else:
                stop_price = entry * (1 - cell["stop_val"] / 100.0)
                target_price, exit_kind, deadline = _exit_params(cell, entry)
            if deadline is not None:
                deadline = m_r_floor + deadline
            exit_m, exit_px_raw, why = walk_exit(
                entry, stop_price, target_price, exit_kind, retest_ts, minute_r_exact,
                trades, bars, deadline_minute=deadline)
            exit_px = apply_cost(exit_px_raw, why, split)
            net_pct = (exit_px / entry - 1.0) * 100.0
            r_dollar = entry - stop_price
            net_R = (exit_px - entry) / r_dollar if r_dollar > 0 else np.nan
            fill_rows.append({
                "day": day, "symbol": symbol, "split": split, "wk": row.wk,
                "cell": cell["name"], "minute_r": m_r_floor, "exit_m": exit_m,
                "hold_min": exit_m - m_r_floor, "exit_px": exit_px,
                "why": why, "net_pct": net_pct, "net_R": net_R,
                "store_served_1438": getattr(row, "store_served_1438", np.nan),
            })
    return pd.DataFrame(fill_rows)


def _exit_params(cell, entry):
    if cell["exit_type"] == "tgt":
        return entry * (1 + cell["exit_val"] / 100.0), "tgt", None
    if cell["exit_type"] == "time" and cell["exit_val"] == "NONE":
        return None, "time_none", None
    if cell["exit_type"] == "time" and cell["exit_val"] == "T30":
        return None, "T30", 30
    if cell["exit_type"] == "time" and cell["exit_val"] == "T60":
        return None, "T60", 60
    raise ValueError(cell)


def dayclustered_t(sub, col="net_pct"):
    """Day-clustered mean/t: cluster by day, one mean per day, t-test on day means."""
    g = sub.groupby("day")[col].mean()
    n_days = len(g)
    if n_days < 2:
        return g.mean() if n_days else np.nan, np.nan, n_days
    mean = g.mean()
    se = g.std(ddof=1) / np.sqrt(n_days)
    t = mean / se if se > 0 else np.nan
    return mean, t, n_days


def ex_top5(sub, col="net_pct"):
    if len(sub) < 20:
        return sub[col].mean()
    thresh = sub[col].quantile(0.95)
    kept = sub[sub[col] <= thresh]
    return kept[col].mean()


def summarize(fills, pop):
    """Build the (cell x holdout) surface with n, mean_pct, t, ex_top5_pct, mean_R,
    first-passage probabilities."""
    recs = []
    for split in ("TRAIN", "VAL"):
        for cell, sub in fills[fills["split"] == split].groupby("cell"):
            mean_pct, t, n_days = dayclustered_t(sub, "net_pct")
            n = len(sub)
            n_weeks = pop.loc[pop["split"] == split, "wk"].nunique()
            recs.append({
                "cell": cell, "split": split, "n": n, "n_days": n_days,
                "fills_per_week": n / n_weeks if n_weeks else np.nan,
                "mean_pct": sub["net_pct"].mean(), "t_dayclust": t,
                "ex_top5_pct": ex_top5(sub, "net_pct"),
                "mean_R": sub["net_R"].mean(),
                "p_target": (sub["why"] == "target").mean(),
                "p_stop": (sub["why"] == "stop").mean(),
                "p_time": (sub["why"].isin(["time", "eod"])).mean(),
                "median_hold_min": sub["hold_min"].median(),
            })
    return pd.DataFrame(recs)


def main():
    n_smoke = None
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke":
        n_smoke = int(sys.argv[2]) if len(sys.argv) > 2 else 200

    log("loading population...")
    pop = load_population(n_smoke)
    pop = load_l3(pop)
    log(f"population: {len(pop)} fills, TRAIN={sum(pop.split=='TRAIN')} VAL={sum(pop.split=='VAL')}")

    bar_store = BarStore(os.path.join(HERE, "bars_fills_1478.db"))
    cells = cell_defs()
    log(f"{len(cells)} cells defined")

    t0 = time.time()
    fills = run(pop, bar_store, cells, verbose_every=max(1, len(pop) // 20))
    log(f"walk done in {time.time()-t0:.1f}s, {len(fills)} (fill,cell) rows")

    tag = "smoke" if n_smoke else "full"
    fills_out = os.path.join(HERE, f"rebuild_1493_fills{'_smoke' if n_smoke else ''}.csv")
    fills.to_csv(fills_out, index=False)
    log(f"wrote {fills_out}")

    surface = summarize(fills, pop)
    surf_out = os.path.join(HERE, f"rebuild_1493_surface{'_smoke' if n_smoke else ''}.csv")
    surface.to_csv(surf_out, index=False)
    log(f"wrote {surf_out}")
    log(surface.sort_values(["split", "mean_pct"], ascending=[True, False]).head(10).to_string())


if __name__ == "__main__":
    main()
