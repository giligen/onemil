"""
cell_1610.py -- PREREG cells 1,610-1,616 (research/exec_quality/PREREG_1610.md), FROZEN.

Owner ask (9/28): "you might want to change the Rs on the spread, do an R x 0.75 or whatever got
eaten in each direction -- do a deep job." This is that job, on the same 9,911 causal HOD-break
fills (`causal_arming_causal.csv` status == fill) used throughout cell_1478/1445/1491 etc.

Part A -- the drift map (report-only, the deliverable even if every barrier cell fails): for each
spread quintile (edges fit on TRAIN-H2's causal arm-time spread, `spread_bps_at_arm`, then applied
unchanged to VAL) and for the whole population, the empirical first-passage P(+k before -m) for
the 36 (k, m) pairs in {0.25, 0.5, 1, 1.5, 2, 3} % of price, beside the DRIFTLESS benchmark,
binomial-95%-CI'd, the "neither by 15:55" share, signed excursions at 5/15/30/60/120 min (bps),
the realised entry cost (fill - level, bps) and the exit-cost standard (context, split-level).

*** DRIFTLESS-VALUE CORRECTION (read before trusting any number here) ***
The PREREG's prose literally writes the driftless benchmark as "k/(k+m)". Its own DISCLOSED
worked examples in the same document are P(+1% before -2%) = 0.62 vs driftless "0.67", and
P(+2% before -1%) = 0.27 vs "0.33". k/(k+m) gives 1/3=0.33 and 2/3=0.67 -- the OPPOSITE pairing.
The standard gambler's-ruin identity for a driftless path with barriers at +k and -m (optional
stopping on the martingale X_t: 0 = k*P(+k) - m*(1-P(+k))) gives P(+k before -m) = m/(k+m), which
reproduces BOTH disclosed numbers exactly (k=1,m=2 -> 2/3=0.667; k=2,m=1 -> 1/3=0.333). The prose
formula is therefore an internal typo (k and m swapped). This script uses the mathematically
correct, self-consistent benchmark m/(k+m) throughout and flags this loudly in RESULT_1610.md --
silently propagating a benchmark that contradicts the spec's own disclosed evidence is exactly the
class of error this programme's "read every report's own red flags" rule exists to catch.

Part B -- six barrier cells (1,610-1,615) plus the report-only quintile-optimal cell (1,616),
walked on minute bars with `sip_rebuild.walk_path` semantics (stop checked before target on a bar
that touches both; gap-through at the open on a stop; exact price on a target; EOD force-flat at
15:55 using that bar's open, or the last available close if no >=15:55 bar exists that day), paired
against the base rule's own `outcome_R` (model_1478_L3_predictions.csv) on the same fills.

Inputs (verbatim from the task; see PREREG_1610.md for the full rationale):
  research/hod_entry/causal_arming_causal.csv       (base fills, status == fill, n = 9,911)
  research/hod_entry/bars_fills_1478.db             (RTH minute bars, table bars)
  research/hod_entry/features_1478_A.csv            (half_entry, join day+symbol+fill_min)
  research/hod_entry/features_1478_C.csv            (spread_bps_at_arm, join day+symbol)
  research/hod_entry/cell_1478.py                   (SLIP_STOP_BPS)
  research/hod_entry/model_1478_L3_predictions.csv  (outcome_R, the base rule's standard net R)
  research/hod_entry/cell_1445.py                   (day_clustered_t, ex_top5_mean, weeks_spanned)
  research/hod_entry/sip_rebuild.py                 (walk_path -- semantics reimplemented here in
                                                       a vectorized form and self-tested against
                                                       the real function on a live sample)

Outputs (all under research/exec_quality/, this cell's only writable directory):
  cell_1610_driftmap.csv   Part A, one row per (holdout, quintile-or-ALL, k, m)
  cell_1610_fills.csv      Part B, one row per (fill, cell): net_R, net_pct, exit_reason
  RESULT_1610.md           both parts as tables, pass-bar checklist, caveats

Run: python3 cell_1610.py
"""
import os
import sys
import time
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

HERE = os.path.dirname(os.path.abspath(__file__))
HOD = os.path.join(os.path.dirname(HERE), 'hod_entry')
sys.path.insert(0, HOD)
import cell_1445 as c1445          # day_clustered_t, ex_top5_mean, weeks_spanned
import sip_rebuild as sr           # walk_path (self-test reference), EOD_M

CAUSAL_CSV = os.path.join(HOD, 'causal_arming_causal.csv')
FEATURES_A = os.path.join(HOD, 'features_1478_A.csv')
FEATURES_C = os.path.join(HOD, 'features_1478_C.csv')
MODEL_PRED = os.path.join(HOD, 'model_1478_L3_predictions.csv')
BARS_DB = os.path.join(HOD, 'bars_fills_1478.db')

OUT_DRIFTMAP = os.path.join(HERE, 'cell_1610_driftmap.csv')
OUT_FILLS = os.path.join(HERE, 'cell_1610_fills.csv')
OUT_RESULT = os.path.join(HERE, 'RESULT_1610.md')

ET = ZoneInfo('America/New_York')
EOD_M = sr.EOD_M                       # 955 = 15:55 ET, sip_rebuild.py convention
assert EOD_M == 955
SLIP_STOP_BPS = {'TRAIN': 0.88 * 2.9 + 0.12 * 94.0, 'VAL': 0.88 * 3.2 + 0.12 * 76.0}
SLIP_EOD_BPS = {'TRAIN': 11.5, 'VAL': 9.7}          # RESULT_1443.md eod row, mean column
KM_LEVELS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]         # % of price -- both k (up) and m (down) grids
N_QUINTILES = 5
Z95 = 1.959963984540054                             # two-sided 95% normal quantile
HOLDOUTS = ('TRAIN', 'VAL')
CADENCE_FLOOR_FILLS_WK = 3.0
MEDIAN_R_PCT_FLOOR = 0.5


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# ================================================================================================
# Data loading
# ================================================================================================

def et_minute(iso_ts):
    """UTC ISO string -> ET minutes-since-midnight (float; bar timestamps are :00-aligned)."""
    dt = datetime.fromisoformat(iso_ts).astimezone(ET)
    return dt.hour * 60 + dt.minute + dt.second / 60.0


def load_fills():
    """Base fills + merged costs + base outcome_R + spread quintile. One row per causal fill."""
    causal = pd.read_csv(CAUSAL_CSV, low_memory=False)
    fills = causal[causal.status == 'fill'].reset_index(drop=True)
    log(f'load_fills: {len(fills)} status==fill rows, splits {fills.split.value_counts().to_dict()}')
    assert set(fills.split.unique()) <= {'TRAIN', 'VAL'}, 'unexpected split label -- TEST must stay sealed'

    base_R = fills['fill'] - fills['stop']
    mism = (base_R - fills['R']).abs() > 1e-6
    if mism.any():
        log(f'load_fills: ERROR {int(mism.sum())} rows where fill-stop != R column -- using fill-stop')
    fills['base_R'] = base_R

    a = pd.read_csv(FEATURES_A, usecols=['day', 'symbol', 'fill_min', 'half_entry'])
    fills = fills.merge(a, on=['day', 'symbol', 'fill_min'], how='left')
    n_na = int(fills.half_entry.isna().sum())
    if n_na:
        log(f'load_fills: WARNING {n_na} fills had no half_entry match (features_1478_A) -- dropped')
        fills = fills.dropna(subset=['half_entry']).reset_index(drop=True)

    c = pd.read_csv(FEATURES_C, usecols=['day', 'symbol', 'spread_bps_at_arm'])
    dup = c.duplicated(subset=['day', 'symbol']).sum()
    if dup:
        log(f'load_fills: WARNING {dup} duplicate (day,symbol) in features_1478_C -- keeping first')
        c = c.drop_duplicates(subset=['day', 'symbol'], keep='first')
    fills = fills.merge(c, on=['day', 'symbol'], how='left')
    fills['has_spread'] = fills['spread_bps_at_arm'].notna()
    n_no_spread = int((~fills['has_spread']).sum())
    log(f'load_fills: WARNING {n_no_spread} fills have no causal spread_bps_at_arm (features_1478_C) -- '
        f'excluded from every spread-quintile-conditioned statistic (Part A per-quintile rows, '
        f'cell 1,615 spread-scaled, cell 1,616 quintile-optimal); kept in pooled/ALL and cells 1,610-1,614')

    m = pd.read_csv(MODEL_PRED, usecols=['day', 'symbol', 'fill_min', 'outcome_R'])
    fills = fills.merge(m, on=['day', 'symbol', 'fill_min'], how='left')
    n_na_m = int(fills.outcome_R.isna().sum())
    if n_na_m:
        log(f'load_fills: WARNING {n_na_m} fills had no base outcome_R match (model_1478_L3_predictions) '
            f'-- dropped (cannot pair)')
        fills = fills.dropna(subset=['outcome_R']).reset_index(drop=True)

    fills['entry_cost_bps'] = (fills['fill'] - fills['level']) / fills['level'] * 1e4
    fills['c_in'] = fills['half_entry'] + (fills['fill'] - fills['level'])
    fills['c_out'] = fills['fill'] * fills['split'].map(SLIP_STOP_BPS) / 1e4
    fills['base_outcome_pct'] = fills['outcome_R'] * fills['base_R'] / fills['fill'] * 100.0
    fills['R_pct'] = fills['base_R'] / fills['fill'] * 100.0

    fills = fills.reset_index(drop=True)
    fills['fill_idx'] = fills.index

    # spread quintile edges fit on TRAIN-H2 only, then applied unchanged to both holdouts
    train_spread = fills.loc[(fills.split == 'TRAIN') & fills.has_spread, 'spread_bps_at_arm']
    edges = np.unique(np.quantile(train_spread, [0, .2, .4, .6, .8, 1.0]))
    if len(edges) < 6:
        log(f'load_fills: ERROR duplicate TRAIN-H2 spread quantile edges collapsed {6 - len(edges)} '
            f'bin(s) -- {len(edges) - 1} quintile bins used instead of 5')
    interior = edges[1:-1]
    fills['quintile'] = np.nan
    mask = fills.has_spread
    clipped = fills.loc[mask, 'spread_bps_at_arm'].clip(edges[0], edges[-1])
    fills.loc[mask, 'quintile'] = np.searchsorted(interior, clipped, side='right') + 1
    log(f'load_fills: TRAIN-H2 spread quintile edges (bps) = {np.round(edges, 2).tolist()}')
    log(f'load_fills: final n={len(fills)}, quintile coverage={int(mask.sum())} '
        f'({mask.mean():.1%}), by split {fills.split.value_counts().to_dict()}')
    return fills, edges


# ================================================================================================
# Bar loading + vectorized walk (sip_rebuild.walk_path semantics)
# ================================================================================================

def build_bars_index(symdays):
    """Load only the (symbol,day) bars needed; return dict[(symbol,day)] -> sorted float array
    with columns [m, o, h, l, c]. Table is RTH-only per its own schema note."""
    con = sqlite3.connect(BARS_DB)
    con.execute('CREATE TEMP TABLE need (symbol TEXT, day TEXT)')
    con.executemany('INSERT INTO need VALUES (?,?)', list(symdays))
    q = ('SELECT b.symbol, b.day, b.t, b.o, b.h, b.l, b.c FROM bars b '
         'JOIN need n ON b.symbol = n.symbol AND b.day = n.day')
    df = pd.read_sql_query(q, con)
    con.close()
    log(f'build_bars_index: {len(df)} bar rows for {len(symdays)} (symbol,day) pairs')
    df['m'] = df['t'].map(et_minute)
    df = df.sort_values(['symbol', 'day', 'm'], kind='stable')
    return {k: g[['m', 'o', 'h', 'l', 'c']].to_numpy(dtype=float)
            for k, g in df.groupby(['symbol', 'day'], sort=False)}


def fill_bar_slice(bars_arr, fill_bar_m):
    """Bars from the fill bar on (m >= fill_bar_m) -- sip_rebuild.walk_path's caller contract.
    None if no such bar exists (this fill has 'no bars' and must be excluded, per rebuild_1491's
    identical precedent for the same table)."""
    if bars_arr is None or len(bars_arr) == 0:
        return None
    sub = bars_arr[bars_arr[:, 0] >= fill_bar_m]
    return sub if len(sub) else None


def precompute_path(path):
    """One-time-per-fill setup: the 15:55 split point plus cumulative running extremes on the
    pre-15:55 sub-path, so any number of (stop,target) threshold lookups on this fill are O(log
    n) via searchsorted on a monotonic array instead of a fresh per-pair scan."""
    m_arr, o_arr, h_arr, l_arr, c_arr = path[:, 0], path[:, 1], path[:, 2], path[:, 3], path[:, 4]
    idx_eod = int(np.searchsorted(m_arr, EOD_M, side='left'))
    if idx_eod < len(path):
        eod_exit = (float(m_arr[idx_eod]), float(o_arr[idx_eod]), 'eod')
    else:
        eod_exit = (float(m_arr[-1]), float(c_arr[-1]), 'eod_fallback')
    if idx_eod == 0:
        return dict(idx_eod=0, eod_exit=eod_exit)
    return dict(idx_eod=idx_eod, eod_exit=eod_exit, sub_o=o_arr[:idx_eod], sub_m=m_arr[:idx_eod],
                cummax_h=np.maximum.accumulate(h_arr[:idx_eod]),
                neg_cummin_l=np.maximum.accumulate(-l_arr[:idx_eod]))


def locate(state, stop_price, target_price):
    """First bar (from the fill bar on) touching stop_price or target_price, sip_rebuild.walk_path
    semantics exactly: 15:55 checked first each bar (-> 'eod'/'eod_fallback'); else stop checked
    before target on a bar touching both ('stop' wins ties); gap-through at the open on a stop
    (exit = open if open already <= stop, else the stop price itself); target exits at the exact
    target price (no gap adjustment -- matches walk_path's own asymmetry). Returns (exit_m,
    exit_price, why)."""
    idx_eod = state['idx_eod']
    if idx_eod == 0:
        return state['eod_exit']
    idx_down = int(np.searchsorted(state['neg_cummin_l'], -stop_price, side='left'))
    idx_up = int(np.searchsorted(state['cummax_h'], target_price, side='left'))
    down_hit = idx_down < idx_eod
    up_hit = idx_up < idx_eod
    if down_hit and (not up_hit or idx_down <= idx_up):
        o = state['sub_o'][idx_down]
        px = o if o <= stop_price else stop_price
        return float(state['sub_m'][idx_down]), float(px), 'stop'
    if up_hit:
        return float(state['sub_m'][idx_up]), float(target_price), 'target'
    return state['eod_exit']


def self_test_vs_walk_path(fills, bars_idx, n=40):
    """Correctness gate: locate() must match the REAL sip_rebuild.walk_path exactly on a live
    sample (stop = base consolidation low, target = fill + 2*base_R, i.e. the base rule's own
    geometry). Logs PASS/FAIL; raises on any mismatch -- this is the parity check every cell in
    this programme is required to carry before a number ships."""
    sample = fills[fills.base_R > 0].head(n)
    n_checked, n_fail = 0, 0
    for r in sample.itertuples():
        bars_arr = bars_idx.get((r.symbol, r.day))
        fill_bar_m = float(np.floor(r.fill_min))
        path = fill_bar_slice(bars_arr, fill_bar_m)
        if path is None:
            continue
        target = r.fill + 2.0 * r.base_R
        mine = locate(precompute_path(path), r.stop, target)
        path_df = pd.DataFrame(path, columns=['m', 'o', 'h', 'l', 'c'])
        theirs = sr.walk_path(r.fill, r.stop, target, path_df)
        n_checked += 1
        if (abs(mine[0] - theirs[0]) > 1e-6 or abs(mine[1] - theirs[1]) > 1e-6 or mine[2] != theirs[2]):
            n_fail += 1
            log(f'self_test: MISMATCH {r.day} {r.symbol}: mine={mine} sip_rebuild.walk_path={theirs}')
    if n_fail:
        raise RuntimeError(f'self_test_vs_walk_path: {n_fail}/{n_checked} mismatches vs the real '
                            f'sip_rebuild.walk_path -- locate() is NOT parity-safe, fix before trusting '
                            f'any Part A/B number')
    log(f'self_test_vs_walk_path: PASS, {n_checked}/{len(sample)} sampled fills exact-match the real '
        f'sip_rebuild.walk_path on (exit_m, exit_price, why)')


# ================================================================================================
# Cost model (shared by every barrier cell -- rebuild_1491.build_book's standard, verified against
# cell_1478.build_outcome / substitute_stop_slip: entry half-spread always charged; a stop exit
# additionally charges the SLIP_STOP_BPS expected-value slip; an eod/eod_fallback exit additionally
# charges SLIP_EOD_BPS; a target exit charges nothing beyond entry -- no double charge)
# ================================================================================================

def net_cost(fill, stop_price, exit_price, why, half_entry, split):
    """(R_new, raw_R, net_R, net_pct) for one fill under one (stop,target) barrier. Vectorized:
    all arguments may be pandas Series of equal length."""
    R_new = fill - stop_price
    raw_R = (exit_price - fill) / R_new
    entry_cost_R = half_entry / R_new
    slip_stop = split.map(SLIP_STOP_BPS).astype(float)
    slip_eod = split.map(SLIP_EOD_BPS).astype(float)
    exit_slip_R = np.where(why == 'stop', exit_price * slip_stop / 1e4 / R_new,
                            np.where(np.isin(why, ['eod', 'eod_fallback']),
                                     exit_price * slip_eod / 1e4 / R_new, 0.0))
    net_R = raw_R - entry_cost_R - exit_slip_R
    net_pct = net_R * R_new / fill * 100.0
    return R_new, raw_R, net_R, net_pct


# ================================================================================================
# Walkers
# ================================================================================================

def walk_grid_all(fills, bars_idx):
    """Every fill x every (k,m) in the 36-pair grid, stop=fill*(1-m%), target=fill*(1+k%). Feeds
    both Part A (classification from `why`) and cell 1,616 (net_pct). Fills with no bar at/after
    the fill minute are excluded and counted (not silently dropped)."""
    rows = []
    n_no_bar = 0
    for r in fills.itertuples():
        bars_arr = bars_idx.get((r.symbol, r.day))
        path = fill_bar_slice(bars_arr, float(np.floor(r.fill_min)))
        if path is None:
            n_no_bar += 1
            continue
        state = precompute_path(path)
        for m in KM_LEVELS:
            stop_price = r.fill * (1 - m / 100.0)
            for k in KM_LEVELS:
                target_price = r.fill * (1 + k / 100.0)
                exit_m, exit_price, why = locate(state, stop_price, target_price)
                rows.append((r.fill_idx, r.day, r.symbol, r.split, r.quintile, r.fill,
                             r.half_entry, k, m, exit_m, exit_price, why))
    log(f'walk_grid_all: WARNING {n_no_bar} fills had no bar at/after the fill minute -- excluded '
        f'from Part A and cell 1,616 (of {len(fills)} candidate fills)')
    cols = ['fill_idx', 'day', 'symbol', 'split', 'quintile', 'fill', 'half_entry', 'k', 'm',
            'exit_m', 'exit_price', 'why']
    grid = pd.DataFrame(rows, columns=cols)
    grid['stop_price'] = grid['fill'] * (1 - grid['m'] / 100.0)
    grid['target_price'] = grid['fill'] * (1 + grid['k'] / 100.0)
    _, _, grid['net_R'], grid['net_pct'] = net_cost(grid['fill'], grid['stop_price'],
                                                      grid['exit_price'], grid['why'].to_numpy(),
                                                      grid['half_entry'], grid['split'])
    return grid, n_no_bar


def walk_cell(fills, bars_idx, stop_price, target_price):
    """One bespoke (stop,target) price per fill (cells 1,610-1,615). NaN stop/target -> excluded
    (e.g. cell 1,615 on a no-spread fill). Returns arrays aligned to `fills`' row order plus the
    no-bar count."""
    n = len(fills)
    exit_m = np.full(n, np.nan)
    exit_price = np.full(n, np.nan)
    why = np.full(n, None, dtype=object)
    ok = np.zeros(n, dtype=bool)
    n_no_bar = 0
    for i, r in enumerate(fills.itertuples()):
        sp, tp = stop_price[i], target_price[i]
        if not (np.isfinite(sp) and np.isfinite(tp)) or sp >= r.fill:
            continue
        bars_arr = bars_idx.get((r.symbol, r.day))
        path = fill_bar_slice(bars_arr, float(np.floor(r.fill_min)))
        if path is None:
            n_no_bar += 1
            continue
        em, ep, wy = locate(precompute_path(path), sp, tp)
        exit_m[i], exit_price[i], why[i], ok[i] = em, ep, wy, True
    return exit_m, exit_price, why, ok, n_no_bar


# ================================================================================================
# Part A -- the drift map
# ================================================================================================

def wilson_ci(k_succ, n, z=Z95):
    """Wilson score 95% interval for a binomial proportion (robust near 0/1, unlike the normal
    approximation, which matters at the wide k/m combinations where hit rates are extreme)."""
    if n == 0:
        return np.nan, np.nan
    phat = k_succ / n
    denom = 1 + z ** 2 / n
    center = (phat + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2)) / denom
    return center - half, center + half


def excursion_prices(fills, bars_idx, offsets=(5, 15, 30, 60, 120)):
    """Mean signed excursion (bps) at each offset after the fill: last available close at or
    before min(fill_min+offset, 15:55) (as-of; RTH-only, capped at the system's 15:55 force-flat
    convention so no post-cutoff price ever leaks in), relative to the fill price. Returns a
    per-fill DataFrame plus the realised entry cost (already on `fills` as entry_cost_bps)."""
    out = {f'excursion_{o}m_bps': np.full(len(fills), np.nan) for o in offsets}
    for i, r in enumerate(fills.itertuples()):
        bars_arr = bars_idx.get((r.symbol, r.day))
        path = fill_bar_slice(bars_arr, float(np.floor(r.fill_min)))
        if path is None:
            continue
        m_arr = path[:, 0]
        for o in offsets:
            target_m = min(r.fill_min + o, EOD_M)
            idx = int(np.searchsorted(m_arr, target_m, side='right')) - 1
            if idx < 0:
                continue
            px = path[idx, 4]
            out[f'excursion_{o}m_bps'][i] = (px - r.fill) / r.fill * 1e4
    return pd.DataFrame(out, index=fills.index)


def build_drift_map(fills, bars_idx, grid):
    """Part A: one row per (holdout, quintile-or-ALL, k, m) -- 12 groups x 36 pairs = 432 rows."""
    exc = excursion_prices(fills, bars_idx)
    fills = pd.concat([fills, exc], axis=1)
    exc_cols = [c for c in exc.columns]

    rows = []
    for holdout in HOLDOUTS:
        groups = [('ALL', fills.split == holdout)]
        for q in range(1, N_QUINTILES + 1):
            groups.append((q, (fills.split == holdout) & (fills.quintile == q)))
        for qlabel, mask in groups:
            grp_fills = fills[mask]
            n_group = int(len(grp_fills))
            exc_means = {c: grp_fills[c].mean() for c in exc_cols}
            entry_cost_mean = grp_fills['entry_cost_bps'].mean()
            grid_mask = (grid.split == holdout) & (grid.fill_idx.isin(grp_fills.fill_idx))
            g = grid[grid_mask]
            n_grid = g.fill_idx.nunique()
            for k in KM_LEVELS:
                for m in KM_LEVELS:
                    gg = g[(g.k == k) & (g.m == m)]
                    n_up = int((gg.why == 'target').sum())
                    n_down = int((gg.why == 'stop').sum())
                    n_neither = int(gg.why.isin(['eod', 'eod_fallback']).sum())
                    n_tot = n_up + n_down + n_neither
                    p_hat = n_up / n_tot if n_tot else np.nan
                    driftless = m / (k + m)          # see module docstring: corrected from the
                    ci_lo, ci_hi = wilson_ci(n_up, n_tot)  # PREREG's self-contradictory "k/(k+m)"
                    rows.append(dict(
                        holdout=holdout, quintile=qlabel, n_fills_in_group=n_group,
                        n_fills_walked=n_grid, k_pct=k, m_pct=m, n_up=n_up, n_down=n_down,
                        n_neither=n_neither, n_resolved_or_censored=n_tot,
                        p_up_before_down=p_hat, driftless_p=driftless, diff=p_hat - driftless,
                        ci_lo=ci_lo, ci_hi=ci_hi, neither_share=n_neither / n_tot if n_tot else np.nan,
                        **exc_means, entry_cost_bps_mean=entry_cost_mean,
                        exit_cost_standard_stop_bps=SLIP_STOP_BPS[holdout],
                        exit_cost_standard_eod_bps=SLIP_EOD_BPS[holdout]))
    dmap = pd.DataFrame(rows)
    log(f'build_drift_map: {len(dmap)} rows (2 holdouts x 6 quintile-groups x 36 pairs)')
    return dmap, fills


# ================================================================================================
# Part B -- barrier cells
# ================================================================================================

def score_group(df, base_col='base_outcome_pct', new_col='net_pct', day_col='day'):
    """n, mean net (R and %), day-clustered t, ex-top-5%, paired delta vs base + its own stats,
    median R%. `df` must already be restricted to one (cell, holdout, quintile-or-ALL) group with
    both the new cell's outcome and the base rule's outcome_pct on the SAME fills."""
    n = len(df)
    if n == 0:
        return dict(n=0)
    mean_net_R = df['net_R'].mean()
    mean_net_pct = df[new_col].mean()
    t = c1445.day_clustered_t(df[new_col], df[day_col])
    ex_top5_R = c1445.ex_top5_mean(df['net_R'])
    delta = df[new_col] - df[base_col]
    paired_delta_pct = delta.mean()
    paired_t = c1445.day_clustered_t(delta, df[day_col])
    paired_ex_top5 = c1445.ex_top5_mean(delta)
    why_mix = df['why'].value_counts(normalize=True).round(3).to_dict()
    weeks = c1445.weeks_spanned(df[day_col])
    fills_wk = n / weeks
    median_R_pct = df['R_pct_new'].median()
    return dict(n=n, mean_net_R=mean_net_R, mean_net_pct=mean_net_pct, t=t, ex_top5_R=ex_top5_R,
                paired_delta_pct=paired_delta_pct, paired_t=paired_t, paired_ex_top5=paired_ex_top5,
                exit_mix=why_mix, fills_wk=fills_wk, median_R_pct=median_R_pct)


def passes_bar(val_stats, train_stats):
    """Frozen pass bar (PREREG, VAL, per cell), TRAIN-H2 read as the same-sign/t>=1 cross-check."""
    if val_stats.get('n', 0) == 0 or train_stats.get('n', 0) == 0:
        return False
    same_sign = np.sign(train_stats['paired_delta_pct']) == np.sign(val_stats['paired_delta_pct'])
    return bool(
        val_stats['mean_net_pct'] >= 0.15 and
        val_stats['paired_delta_pct'] >= 0.10 and
        val_stats['paired_t'] >= 2.5 and
        val_stats['paired_ex_top5'] > 0 and
        same_sign and train_stats['paired_t'] >= 1.0 and
        val_stats['fills_wk'] >= CADENCE_FLOOR_FILLS_WK and
        val_stats['median_R_pct'] >= MEDIAN_R_PCT_FLOOR)


def build_cell_result(fills, exit_m, exit_price, why, ok, stop_price, cell_name):
    """Assemble one cell's per-fill result frame (only rows that walked, ok==True)."""
    d = fills.loc[ok, ['fill_idx', 'day', 'symbol', 'split', 'quintile', 'fill', 'half_entry',
                        'base_outcome_pct']].copy()
    d['stop_price'] = stop_price[ok]
    d['exit_m'] = exit_m[ok]
    d['exit_price'] = exit_price[ok]
    d['why'] = why[ok]
    d['R_new'], _, d['net_R'], d['net_pct'] = net_cost(d['fill'], d['stop_price'], d['exit_price'],
                                                         d['why'].to_numpy(), d['half_entry'], d['split'])
    d['R_pct_new'] = d['R_new'] / d['fill'] * 100.0
    d['cell'] = cell_name
    return d


def run_part_b(fills, bars_idx, driftmap):
    """Cells 1,610-1,615 (bespoke stop/target per fill) + 1,616 (quintile-optimal, from the
    36-pair grid already walked for Part A). Returns (fills_csv_df, summary_rows)."""
    fill = fills['fill'].to_numpy()
    base_R = fills['base_R'].to_numpy()
    c_in = fills['c_in'].to_numpy()
    c_out = fills['c_out'].to_numpy()
    spread = fills['spread_bps_at_arm'].to_numpy()
    has_spread = fills['has_spread'].to_numpy()

    cells = {}
    cells['1610'] = (fill - 0.75 * base_R, fill + 1.5 * base_R)
    cells['1611'] = (fill - 0.75 * base_R, fill + 2.0 * base_R)
    cells['1612'] = (fill - base_R, fill + 2.0 * base_R + c_in + c_out)
    cells['1613'] = (fill - base_R - c_out, fill + 2.0 * base_R)
    cells['1614'] = (fill - base_R - c_out, fill + 2.0 * base_R + c_in + c_out)

    R_bps = base_R / fill * 1e4
    s = np.clip(np.where(has_spread, spread / np.where(R_bps > 0, R_bps, np.nan), np.nan), 0, 1)
    stop_1615 = fill - base_R * (1 + s)
    target_1615 = fill + 2.0 * base_R * (1 + s)
    stop_1615 = np.where(has_spread, stop_1615, np.nan)
    target_1615 = np.where(has_spread, target_1615, np.nan)
    cells['1615'] = (stop_1615, target_1615)

    all_dfs = []
    no_bar_counts = {}
    for name, (sp, tp) in cells.items():
        log(f'run_part_b: walking cell {name} ({int(np.isfinite(sp).sum())} fills with a valid '
            f'stop/target)')
        exit_m, exit_price, why, ok, n_no_bar = walk_cell(fills, bars_idx, sp, tp)
        no_bar_counts[name] = n_no_bar
        all_dfs.append(build_cell_result(fills, exit_m, exit_price, why, ok, sp, name))
    log(f'run_part_b: no-bar exclusions per cell = {no_bar_counts}')

    # ---- cell 1,616: quintile-optimal (k,m) selected on TRAIN-H2 net_pct, read on VAL ----
    train_perf = (driftmap[driftmap.holdout == 'TRAIN']
                  .query('quintile != "ALL"'))
    # driftmap doesn't carry net_pct (Part A is classification-only) -- recompute the selection
    # directly from the grid, which run_part_b receives via the caller (see main()).
    return all_dfs, no_bar_counts


def select_1616(grid, fills):
    """Per TRAIN-H2 spread quintile, the (k,m) grid pair with the highest TRAIN-H2 mean net_pct;
    apply that SAME pair to every fill in that quintile on BOTH holdouts (never re-selected on
    VAL, per the frozen pass bar's explicit prohibition)."""
    train_grid = grid[(grid.split == 'TRAIN') & grid.quintile.notna()]
    perf = train_grid.groupby(['quintile', 'k', 'm'])['net_pct'].mean().reset_index()
    best_idx = perf.groupby('quintile')['net_pct'].idxmax()
    best = perf.loc[best_idx].set_index('quintile')[['k', 'm', 'net_pct']]
    log(f'select_1616: TRAIN-H2 chosen (k%,m%) per quintile:\n{best.to_string()}')

    fills = fills.copy()
    fills['k1616'] = fills['quintile'].map(best['k'].to_dict())
    fills['m1616'] = fills['quintile'].map(best['m'].to_dict())
    sel = fills.dropna(subset=['k1616', 'm1616'])[['fill_idx', 'k1616', 'm1616']]
    merged = sel.merge(grid, left_on=['fill_idx', 'k1616', 'm1616'],
                        right_on=['fill_idx', 'k', 'm'], how='inner')
    # NOTE: select the needed `fills` columns BEFORE merging -- the base causal_arming_causal.csv
    # already has its OWN 'exit_m'/'exit_price'/'why'/'net_R' columns (the BASE rule's walk), which
    # would silently suffix (_x/_y) against 1,616's same-named fields on a bare merge.
    base_cols = fills[['fill_idx', 'day', 'symbol', 'split', 'quintile', 'fill', 'half_entry',
                        'base_outcome_pct']]
    d = base_cols.merge(merged[['fill_idx', 'exit_m', 'exit_price', 'why', 'net_R', 'net_pct',
                                 'stop_price']], on='fill_idx', how='inner')
    d['R_new'] = d['fill'] - d['stop_price']
    d['R_pct_new'] = d['R_new'] / d['fill'] * 100.0
    d['cell'] = '1616'
    return d, best


# ================================================================================================
# Reporting
# ================================================================================================

def df_to_md(df, float_fmt='{:.4f}'):
    """Minimal, dependency-free DataFrame -> markdown table."""
    def fmt(v):
        if isinstance(v, float):
            if np.isnan(v):
                return 'nan'
            # iterrows() upcasts a whole mixed-dtype row to float64 -- print true integer counts
            # (n_up, n, n_fills, ...) without a decorative .0000 while keeping real stats at 4dp.
            return str(int(v)) if v == int(v) and abs(v) < 1e15 else float_fmt.format(v)
        return str(v)
    lines = ['| ' + ' | '.join(str(c) for c in df.columns) + ' |',
              '|' + '|'.join(['---'] * len(df.columns)) + '|']
    for _, row in df.iterrows():
        lines.append('| ' + ' | '.join(fmt(v) for v in row) + ' |')
    return '\n'.join(lines)


def build_result_md(dmap, cell_summaries, best_1616, edges, n_no_bar_grid, no_bar_counts, n_no_spread):
    lines = []
    lines.append('# RESULT_1610 -- cost-aware barriers on the HOD break (cells 1,610-1,616)')
    lines.append('')
    lines.append('Built by `research/exec_quality/cell_1610.py` executing the FROZEN '
                  '`research/exec_quality/PREREG_1610.md`. Base population: 9,911 causal fills '
                  '(`causal_arming_causal.csv` status==fill), TRAIN-H2 + VAL, TEST sealed/absent.')
    lines.append('')
    lines.append('## Correction to the PREREG\'s driftless formula (read first)')
    lines.append('The PREREG prose literally writes the driftless benchmark as `k/(k+m)`, but its own '
                  'disclosed worked examples (P(+1% before -2%)=0.62 vs "0.67"; P(+2% before -1%)=0.27 '
                  'vs "0.33") are internally consistent ONLY with `m/(k+m)` -- the standard driftless '
                  'gambler\'s-ruin identity, and this is what this script computes throughout. Using the '
                  'literal `k/(k+m)` would have reproduced neither disclosed number and silently inverted '
                  'every sign in the drift map. Flagged per the "read every report\'s own red flags as an '
                  'adversary" rule; treat `driftless_p` below as `m/(k+m)`.')
    lines.append('')
    lines.append(f'Data-quality notes: TRAIN-H2 spread quintile edges (bps, causal arm-time spread) = '
                  f'{np.round(edges, 2).tolist()}; {n_no_spread} fills have no causal spread and are '
                  f'excluded from every quintile-conditioned statistic; {n_no_bar_grid} fills have no bar '
                  f'at/after the fill minute in `bars_fills_1478.db` and are excluded from Part A and cell '
                  f'1,616; per-cell no-bar exclusions (1,610-1,615): {no_bar_counts}.')
    lines.append('')

    # ---------------- Part A ----------------
    lines.append('## Part A -- the drift map (report-only)')
    lines.append('Per (holdout, quintile) group: n fills walked, mean realised entry cost (fill-level, '
                  'bps), the exit-cost standard (context, split-level constant: stop-limit / EOD-bid), '
                  'mean signed excursion at 5/15/30/60/120 min (bps), then the 36 (k,m) pairs -- empirical '
                  'P(+k before -m), the corrected driftless m/(k+m), the difference, Wilson 95% CI, and '
                  'the "neither by 15:55" share.')
    for holdout in HOLDOUTS:
        for qlabel in ['ALL', 1, 2, 3, 4, 5]:
            g = dmap[(dmap.holdout == holdout) & (dmap.quintile == qlabel)]
            if not len(g):
                continue
            head = g.iloc[0]
            n_up_sign = int((g['diff'] > 0).sum())
            n_down_sign = int((g['diff'] < 0).sum())
            lines.append('')
            lines.append(f'### {holdout} / quintile {qlabel} (n_fills={int(head.n_fills_in_group)}, '
                          f'walked={int(head.n_fills_walked)})')
            lines.append(f'Entry cost mean = {head.entry_cost_bps_mean:.2f} bps; exit-cost standard '
                          f'(stop) = {head.exit_cost_standard_stop_bps:.2f} bps, (eod) = '
                          f'{head.exit_cost_standard_eod_bps:.2f} bps; excursion bps at 5/15/30/60/120 '
                          f'min = {head.excursion_5m_bps:.2f} / {head.excursion_15m_bps:.2f} / '
                          f'{head.excursion_30m_bps:.2f} / {head.excursion_60m_bps:.2f} / '
                          f'{head.excursion_120m_bps:.2f}. Sign summary: {n_up_sign}/36 pairs above '
                          f'driftless, {n_down_sign}/36 below, mean diff = {g["diff"].mean():+.4f}.')
            tbl = g[['k_pct', 'm_pct', 'n_up', 'n_down', 'n_neither', 'p_up_before_down',
                     'driftless_p', 'diff', 'ci_lo', 'ci_hi', 'neither_share']].copy()
            tbl.columns = ['k%', 'm%', 'n_up', 'n_down', 'n_neither', 'P(+k<-m)', 'driftless',
                           'diff', 'ci_lo', 'ci_hi', 'neither%']
            lines.append(df_to_md(tbl))

    # ---------------- Part B ----------------
    lines.append('')
    lines.append('## Part B -- barrier cells (paired vs the base outcome_R on the same fills)')
    lines.append('Cost model (all cells): entry = half_entry always charged; stop exit additionally '
                  'charges the SLIP_STOP_BPS expected-value slip on the exit price; eod/eod_fallback '
                  'additionally charges SLIP_EOD_BPS; target exit charges nothing beyond entry (matches '
                  'cell_1478.build_outcome / rebuild_1491.build_book, no double charge). '
                  'c_in = half_entry + (fill-level); c_out = fill * SLIP_STOP_BPS[split] / 1e4 (the '
                  'stop-limit standard converted to a PRICE offset at construction time, using the fill '
                  'price as the exit-price proxy since the actual exit price is unknown before the walk).')
    lines.append('')
    lines.append('Pass bar (VAL): mean net >= +0.15% of price AND paired delta vs base >= +0.10% of '
                  'price with day-clustered t >= 2.5, ex-top-5% of the delta > 0, TRAIN-H2 same-sign '
                  't >= 1, >= 3 fills/week, median R >= 0.5% of price.')
    lines.append('')
    cell_names_order = ['1610', '1611', '1612', '1613', '1614', '1615', '1616']
    cell_desc = {'1610': 'R x 0.75, target = 2 x new R', '1611': 'R x 0.75, base 2R target',
                 '1612': 'target + eaten (c_in+c_out on the target)',
                 '1613': 'stop + eaten (c_out on the stop)',
                 '1614': 'both + eaten', '1615': 'spread-scaled (s capped at 1)',
                 '1616': 'quintile-optimal (report-only, TRAIN-H2-selected, read on VAL)'}
    for name in cell_names_order:
        lines.append(f'### Cell {name} -- {cell_desc[name]}')
        rows = []
        for holdout in HOLDOUTS:
            for qlabel in ['ALL', 1, 2, 3, 4, 5]:
                st = cell_summaries[name].get((holdout, qlabel))
                if st is None or st.get('n', 0) == 0:
                    continue
                rows.append(dict(holdout=holdout, quintile=qlabel, n=st['n'],
                                  mean_net_R=st['mean_net_R'], mean_net_pct=st['mean_net_pct'],
                                  t=st['t'], ex_top5_R=st['ex_top5_R'],
                                  paired_delta_pct=st['paired_delta_pct'], paired_t=st['paired_t'],
                                  paired_ex_top5=st['paired_ex_top5'], fills_wk=st['fills_wk'],
                                  median_R_pct=st['median_R_pct'],
                                  passes_val_bar=(passes_bar(st, cell_summaries[name].get((
                                      'TRAIN', qlabel), {})) if holdout == 'VAL' else '')))
        if rows:
            lines.append(df_to_md(pd.DataFrame(rows)))
        lines.append('')

    lines.append('### Cell 1,616 TRAIN-H2 selection (per quintile)')
    lines.append(df_to_md(best_1616.reset_index().rename(
        columns={'quintile': 'quintile', 'k': 'k%_opt', 'm': 'm%_opt', 'net_pct': 'TRAIN_net_pct'})))

    # ---------------- caveats ----------------
    lines.append('')
    lines.append('## Caveats (read as an adversary before relaying anything above)')
    lines.append('- **Driftless formula corrected** from the PREREG\'s literal `k/(k+m)` to `m/(k+m)` '
                  '(see the top section) -- re-derive from the disclosed 0.67/0.33 examples if this is '
                  'disputed.')
    lines.append('- `c_out` uses the FILL price as the exit-price proxy when constructing a barrier '
                  '(the true exit price is unknown before the walk); this is a modeling choice, not a '
                  'measured cost -- it is the same order of magnitude as the exit price for a barrier '
                  'this tight but is not identical to it.')
    lines.append('- "Fills/week" is the RAW fill count over ISO weeks spanned, NOT run through the '
                  '12/day-4-concurrent slot simulator (`research/hod_consol/run_consol.simulate_slots` '
                  'is not among this task\'s Inputs). Entries/timing are identical across every cell here '
                  '(only exits differ), so slotting would affect every cell equally and cross-cell '
                  'comparisons are unaffected; but the pass-bar\'s literal "fills/week under the live cap" '
                  'threshold is read against the uncapped number here -- flag before using this as a '
                  'capacity claim.')
    lines.append('- Excursions and first-passage are censored at 15:55 ET (EOD_M=955) with the price used '
                  'at each offset capped at min(fill_min+offset, 955) -- no post-cutoff price is used, '
                  'consistent with the live force-flat rule.')
    lines.append('- "Fill bar" / "no bars" follows the precedent in `rebuild_1491.py`\'s `walk()`: the '
                  'first bar at or after floor(fill_min), not a requirement that a bar sit exactly on '
                  'that minute (bars are sparse/thin-name gapped, per the schema).')
    lines.append('- Part A\'s `P(+k before -m)` is UNCONDITIONAL (neither counts against it, `neither_share` '
                  'reported separately) -- not renormalized over resolved paths only.')
    lines.append('- `locate()` (the vectorized walker) is self-tested against the real '
                  '`sip_rebuild.walk_path` on a live sample every run (see the log); a mismatch raises '
                  'before any number is written.')
    lines.append('- Rebuild protocol per the CLAUDE.md research-claim gate: independent reimplementation, '
                  'causality trace, price-scale check, fill realism, tail dependence and multiplicity are '
                  'NOT independently re-verified by a second agent in this run -- this is the BUILDER pass '
                  'only; per the frozen PREREG\'s own "Independent check" section, a rebuild from this '
                  'prose (Part A within 0.02, Part B >=99% of rows within 0.01 R) is required before any '
                  'PASS/FAIL is acted on.')
    return '\n'.join(lines)


# ================================================================================================
# Main
# ================================================================================================

def main():
    log('cell_1610: loading fills + costs + quintiles')
    fills, edges = load_fills()
    n_no_spread = int((~fills['has_spread']).sum())

    symdays = set(zip(fills.symbol, fills.day))
    log(f'cell_1610: loading bars for {len(symdays)} (symbol,day) pairs')
    bars_idx = build_bars_index(symdays)

    log('cell_1610: self-testing locate() vs the real sip_rebuild.walk_path')
    self_test_vs_walk_path(fills, bars_idx)

    log('cell_1610: walking the 36-pair grid (Part A + cell 1,616 selection) -- all fills')
    grid, n_no_bar_grid = walk_grid_all(fills, bars_idx)

    log('cell_1610: building the Part A drift map')
    dmap, fills = build_drift_map(fills, bars_idx, grid)
    dmap.to_csv(OUT_DRIFTMAP, index=False)
    log(f'cell_1610: wrote {len(dmap)} rows -> {OUT_DRIFTMAP}')

    log('cell_1610: walking barrier cells 1,610-1,615')
    part_b_dfs, no_bar_counts = run_part_b(fills, bars_idx, dmap)

    log('cell_1610: selecting cell 1,616 (TRAIN-H2 quintile-optimal, read on VAL)')
    d1616, best_1616 = select_1616(grid, fills)
    part_b_dfs.append(d1616)

    fills_csv = pd.concat(part_b_dfs, ignore_index=True)
    keep_cols = ['cell', 'fill_idx', 'day', 'symbol', 'split', 'quintile', 'fill', 'stop_price',
                 'exit_m', 'exit_price', 'why', 'net_R', 'net_pct', 'R_pct_new', 'base_outcome_pct']
    fills_csv[keep_cols].to_csv(OUT_FILLS, index=False)
    log(f'cell_1610: wrote {len(fills_csv)} rows ({fills_csv.cell.nunique()} cells) -> {OUT_FILLS}')

    log('cell_1610: scoring every cell x holdout x quintile group')
    cell_summaries = {}
    for name, d in [(df.cell.iloc[0], df) for df in part_b_dfs]:
        cell_summaries[name] = {}
        for holdout in HOLDOUTS:
            for qlabel in ['ALL', 1, 2, 3, 4, 5]:
                if qlabel == 'ALL':
                    sub = d[d.split == holdout]
                else:
                    sub = d[(d.split == holdout) & (d.quintile == qlabel)]
                cell_summaries[name][(holdout, qlabel)] = score_group(sub)

    for name in cell_summaries:
        val = cell_summaries[name].get(('VAL', 'ALL'), {})
        train = cell_summaries[name].get(('TRAIN', 'ALL'), {})
        if val.get('n', 0):
            p = passes_bar(val, train)
            log(f'cell_1610: {name} ALL/VAL n={val["n"]} mean_net%={val["mean_net_pct"]:.3f} '
                f'paired_delta%={val["paired_delta_pct"]:.3f} paired_t={val["paired_t"]:.2f} '
                f'PASSES_BAR={p}')

    log('cell_1610: writing RESULT_1610.md')
    md = build_result_md(dmap, cell_summaries, best_1616, edges, n_no_bar_grid, no_bar_counts,
                          n_no_spread)
    with open(OUT_RESULT, 'w') as f:
        f.write(md)
    log(f'cell_1610: wrote {OUT_RESULT}')

    # ---- machine-readable summary for the calling agent (ALL-quintile pooled rows only) ----
    import json
    summary = []
    for name in cell_names_all():
        for holdout in HOLDOUTS:
            st = cell_summaries[name].get((holdout, 'ALL'), {})
            if not st.get('n', 0):
                continue
            train_st = cell_summaries[name].get(('TRAIN', 'ALL'), {})
            summary.append(dict(
                cell=name, holdout=holdout, n=int(st['n']), mean_net_R=round(float(st['mean_net_R']), 5),
                mean_net_pct=round(float(st['mean_net_pct']), 5), t=round(float(st['t']), 3),
                ex_top5_R=round(float(st['ex_top5_R']), 5),
                paired_delta_pct=round(float(st['paired_delta_pct']), 5),
                paired_t=round(float(st['paired_t']), 3),
                paired_ex_top5=round(float(st['paired_ex_top5']), 5),
                fills_wk=round(float(st['fills_wk']), 3), exit_mix=st['exit_mix'],
                passes_bar=(passes_bar(st, train_st) if holdout == 'VAL' else False)))
    print('SUMMARY_JSON:' + json.dumps(summary), flush=True)
    print('QUINTILE_EDGES_BPS:' + json.dumps(np.round(edges, 2).tolist()), flush=True)
    log('cell_1610: done')


def cell_names_all():
    return ['1610', '1611', '1612', '1613', '1614', '1615', '1616']


if __name__ == '__main__':
    main()
