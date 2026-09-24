#!/usr/bin/env python3
"""Breakout thermometer -- research/thermo/PREREG.md, cells 1,420-1,422 (frozen 2026-09-24).

Does "breakouts have been working lately" predict the next day? Small pure functions (unit-tested in
test_thermo.py), orchestrated here by a CLI. Every definition (window=20, min_n=40, min_hist=60, the
expanding-median cut, the pass bars) is frozen in PREREG.md and must not change after any number
exists (SPEC.md's "Not allowed").

Usage (research/thermo/run_thermo.sh does this after the market-hours DB window closes):
    python3 thermo.py --out research/thermo/REPORT.md --orb-book-2025-26 research/thermo/book_2025_26.csv
Smoke mode (no DB, no 2025-26 book, cell 1420 legs on 2023-24 only -- never write to the real REPORT.md path):
    python3 thermo.py --out /tmp/smoke_REPORT.md
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from trading.orb_csv import read_orb_csv  # noqa: E402 -- ticker-NA-safe CSV reader

logging.basicConfig(level=logging.INFO, format='%(asctime)s [thermo] %(message)s')
log = logging.getLogger('thermo')

ORB_POPULATION_FILES = [
    ROOT / 'research/orb_2023/out_1418/orb_features_20260924_0439.csv',   # 2023-01..2024-06
    ROOT / 'research/orb_2024/out_1415/orb_features_20260923_2052.csv',   # 2024-07..2024-12
    ROOT / 'analysis_results/orb_features_20260923_2049.csv',             # 2025-01..2026-09-23
]
ORB_BOOK_2023 = ROOT / 'research/orb_2023/book_1418.csv'
ORB_BOOK_2024 = ROOT / 'research/orb_2024/book_1415.csv'
HOD_B0_TRADES = ROOT / 'research/hod_exit_lab/b0_trades.csv'
BF_2024H2_CSV = ROOT / 'research/bf_2024/stage2_2024.csv'
BF_2025_26_CSV = ROOT / 'research/bf_frequency/runs/P1.csv'
RUNB_TRUE_2025 = ROOT / 'research/orb_seed_wide/out/runB_true.csv'

STALE_END_LAG = 61  # causal end_lag(1) shifted 60 trading days further into the past (PREREG's stale lens)


# --------------------------------------------------------------------------- data loaders
def load_orb_population(paths=None) -> pd.DataFrame:
    """Concatenate the three ORB population CSVs (PREREG.md 'Data'), via read_orb_csv (ticker-NA-safe).
    Asserts the three files' date sets do not overlap and logs the day/row count per file."""
    paths = paths or ORB_POPULATION_FILES
    frames, date_sets = [], []
    for p in paths:
        df = read_orb_csv(str(p))
        days = set(df['date'].unique())
        log.info(f"load_orb_population: {Path(p).name} rows={len(df)} days={len(days)} "
                 f"range=[{df['date'].min()}, {df['date'].max()}]")
        date_sets.append(days)
        frames.append(df)
    for i in range(len(date_sets)):
        for j in range(i + 1, len(date_sets)):
            overlap = date_sets[i] & date_sets[j]
            assert not overlap, f"date overlap between {paths[i]} and {paths[j]}: {sorted(overlap)[:5]}"
    out = pd.concat(frames, ignore_index=True)
    log.info(f"load_orb_population: TOTAL rows={len(out)} days={out['date'].nunique()}")
    return out


def load_hod_population(path=None) -> pd.DataFrame:
    """Load b0_trades.csv, drop split=='TEST' ON READ (PREREG.md: TEST is dropped on read and never
    touched). Logs TRAIN/VAL/dropped-TEST row counts and the day span."""
    path = path or HOD_B0_TRADES
    df = pd.read_csv(path, keep_default_na=False, na_values=[''])
    n_test = int((df['split'] == 'TEST').sum())
    df = df[df['split'] != 'TEST'].copy()
    log.info(f"load_hod_population: {Path(path).name} TRAIN={int((df.split == 'TRAIN').sum())} "
             f"VAL={int((df.split == 'VAL').sum())} dropped_TEST={n_test} days={df['day'].nunique()} "
             f"range=[{df['day'].min()}, {df['day'].max()}]")
    return df


# --------------------------------------------------------------------------- pure functions
def daily_outcome_series(df: pd.DataFrame, date_col: str, outcome_col: str, calendar_dates=None) -> pd.Series:
    """Group df by date_col into a Series indexed by a SORTED trading-day calendar (calendar_dates if
    given, else df's own unique dates), each value a 1-D float array of that day's outcome_col values
    (an EMPTY array on a calendar day with no rows -- e.g. an ORB scan day with zero entered breakouts).
    This is the input `thermometer()` expects."""
    dates = sorted(pd.unique(df[date_col])) if calendar_dates is None else sorted(pd.unique(pd.Series(list(calendar_dates))))
    grouped = {d: g[outcome_col].to_numpy(dtype=float) for d, g in df.groupby(date_col)}
    empty = np.array([], dtype=float)
    return pd.Series([grouped.get(d, empty) for d in dates], index=pd.Index(dates, name=date_col))


def thermometer(daily_outcomes: pd.Series, window: int = 20, min_n: int = 40, end_lag: int = 1) -> pd.Series:
    """T(d) for every trading day d = daily_outcomes.index[i]: the mean of all outcome values pooled
    over `window` consecutive trading days of the SAME calendar, ending `end_lag` trading days before d
    (calendar positions [i-end_lag-window+1, i-end_lag], clipped at 0). NaN (undefined) unless at
    least `min_n` pooled observations exist.

    end_lag=1 (default, PREREG-frozen): the window is the `window` trading days STRICTLY BEFORE d --
        the only mode used for any shipped or pass-bar number.
    end_lag=0: ORACLE / look-ahead -- the window ends AT d (includes day d's own outcomes). Adversary
        lens only (PREREG.md 'oracle bound'); never used for a pass-bar number.
    end_lag=STALE_END_LAG (61): STALE -- a window ending 60 trading days before the causal end.
    `daily_outcomes` must be indexed by a sorted, unique trading-day calendar (daily_outcome_series())."""
    idx = daily_outcomes.index
    assert idx.is_monotonic_increasing and idx.is_unique, "daily_outcomes index must be sorted, unique trading days"
    assert end_lag >= 0, "end_lag must be >= 0"
    values = daily_outcomes.values
    n = len(idx)
    n_raw = sum(len(v) for v in values)
    n_nan = sum(int(np.isnan(v).sum()) for v in values) if n_raw else 0
    if n_nan:
        log.warning(f"thermometer: {n_nan} of {n_raw} outcome values are NaN (e.g. an undecided exit "
                     f"price) -- dropped from every pooled window, never counted toward min_n")
    T = np.full(n, np.nan)
    for i in range(n):
        hi = i - end_lag + 1  # exclusive end of the window slice
        if hi <= 0:
            continue
        lo = max(0, hi - window)
        window_vals = values[lo:hi]
        if len(window_vals) == 0:
            continue
        pooled = np.concatenate(list(window_vals))
        pooled = pooled[~np.isnan(pooled)]  # undefined outcomes carry no information -- must not poison the mean
        if pooled.size < min_n:
            continue
        T[i] = float(np.mean(pooled))
    return pd.Series(T, index=idx, name='T')


def hot_flags(T: pd.Series, min_hist: int = 60) -> pd.Series:
    """Hot(d) iff T(d) > the expanding median of all STRICTLY EARLIER defined T values, requiring at
    least `min_hist` such earlier defined values (else NaN/undefined). The median at day d NEVER
    includes T(d) itself -- proven in test_thermo.py against an independent reference computation.
    Returns a float Series: 1.0=hot, 0.0=cold, NaN=undefined."""
    idx = T.index
    vals = T.values
    out = np.full(len(idx), np.nan)
    earlier_defined = []
    for i in range(len(idx)):
        if len(earlier_defined) >= min_hist and not np.isnan(vals[i]):
            out[i] = float(vals[i] > np.median(earlier_defined))
        if not np.isnan(vals[i]):
            earlier_defined.append(vals[i])
    return pd.Series(out, index=idx, name='hot')


def split_stats(R: pd.Series, hot: pd.Series, day: pd.Series) -> dict:
    """hot-cold split stats for a set of trades: mean R per cohort, hot-cold difference, OLS
    `R ~ 1 + hot` with day-clustered SE (statsmodels cov_type='cluster', groups=day -- same pattern as
    research/hod_ofi/pipeline.py's _keep_lift_ols) and the iid t beside it, n per cohort. Rows with
    undefined hot/R/day are dropped and logged (WARNING)."""
    import statsmodels.api as sm
    df = pd.DataFrame({'R': pd.Series(R).values, 'hot': pd.Series(hot).values, 'day': pd.Series(day).values})
    n_before = len(df)
    df = df.dropna(subset=['R', 'hot', 'day'])
    if len(df) < n_before:
        log.warning(f"split_stats: dropped {n_before - len(df)} of {n_before} rows with undefined hot/R/day")
    n_hot = int((df['hot'] == 1).sum())
    n_cold = int((df['hot'] == 0).sum())
    if n_hot == 0 or n_cold == 0 or len(df) < 3:
        log.warning(f"split_stats: degenerate cohort (n_hot={n_hot}, n_cold={n_cold}) -- stats undefined")
        return dict(n_hot=n_hot, n_cold=n_cold, mean_hot=float('nan'), mean_cold=float('nan'),
                    hot_minus_cold=float('nan'), t_cluster=float('nan'), t_iid=float('nan'))
    mean_hot = float(df.loc[df.hot == 1, 'R'].mean())
    mean_cold = float(df.loc[df.hot == 0, 'R'].mean())
    X = sm.add_constant(df['hot'].astype(float), has_constant='add')
    y = df['R'].astype(float)
    m_cluster = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': df['day']})
    m_iid = sm.OLS(y, X).fit()
    return dict(n_hot=n_hot, n_cold=n_cold, mean_hot=mean_hot, mean_cold=mean_cold,
                hot_minus_cold=mean_hot - mean_cold,
                t_cluster=float(m_cluster.tvalues['hot']), t_iid=float(m_iid.tvalues['hot']))


def quintile_table(T: pd.Series, R: pd.Series) -> pd.DataFrame:
    """Report-only: mean R by quintile of T (pd.qcut, duplicates='drop')."""
    df = pd.DataFrame({'T': pd.Series(T).values, 'R': pd.Series(R).values}).dropna()
    if len(df) < 5:
        return pd.DataFrame(columns=['q', 'mean', 'count'])
    df['q'] = pd.qcut(df['T'], 5, labels=False, duplicates='drop')
    return df.groupby('q')['R'].agg(['mean', 'count']).reset_index()


def ex_top_pct(R: pd.Series, pct: float = 0.05) -> float:
    """Mean R after dropping the top `pct` share by value (k = max(1, round(n*pct)) -- same convention
    as research/bf_2024/score.py's ex-top-5%)."""
    s = pd.Series(R).dropna().sort_values(ascending=False)
    if len(s) == 0:
        return float('nan')
    k = max(1, int(round(len(s) * pct)))
    return float(s.iloc[k:].mean()) if len(s) > k else float('nan')


def _simulate_slots_fallback(trades: pd.DataFrame, concurrent_cap: int = 4, daily_cap: int = 12) -> pd.Series:
    """Fallback copy of research/hod_consol/run_consol.simulate_slots (first-12/day, 4-concurrent),
    used ONLY if importing the real function fails (logged as an ERROR by the caller)."""
    keep = pd.Series(False, index=trades.index)
    for day, g in trades.groupby('day'):
        g = g.sort_values('entry_m')
        open_exits, daily_count = [], 0
        for row in g.itertuples():
            open_exits = [x for x in open_exits if x > row.entry_m]
            if len(open_exits) < concurrent_cap and daily_count < daily_cap:
                keep.loc[row.Index] = True
                open_exits.append(row.exit_m)
                daily_count += 1
    return keep


# --------------------------------------------------------------------------- cell scorers
def score_orb_cell(population: pd.DataFrame, book: pd.DataFrame, window: int = 20, min_n: int = 40,
                    min_hist: int = 60) -> dict:
    """Cell 1,420 (PREREG.md): ORB live-config fills (entered==1 book rows, R = _sized_pnl/375),
    thermometer T_ORB = win rate of ALL entered==1 population rows over the full scan-day calendar
    (population rows with entered==0 still mark a day as 'scanned' with zero triggers)."""
    full_dates = population['date'].unique()
    pop_entered = population[population['entered'] == 1].copy()
    pop_entered['win'] = (pop_entered['pnl'].astype(float) > 0).astype(float)
    daily_win = daily_outcome_series(pop_entered, 'date', 'win', calendar_dates=full_dates)

    T_causal = thermometer(daily_win, window, min_n, end_lag=1)
    T_oracle = thermometer(daily_win, window, min_n, end_lag=0)
    T_stale = thermometer(daily_win, window, min_n, end_lag=STALE_END_LAG)
    hot_causal = hot_flags(T_causal, min_hist)
    hot_oracle = hot_flags(T_oracle, min_hist)
    hot_stale = hot_flags(T_stale, min_hist)

    fills = book[book['entered'] == 1].copy()
    fills['R'] = fills['_sized_pnl'].astype(float) / 375.0
    fills['hot'] = fills['date'].map(hot_causal)
    fills['hot_oracle'] = fills['date'].map(hot_oracle)
    fills['hot_stale'] = fills['date'].map(hot_stale)

    causal = split_stats(fills['R'], fills['hot'], fills['date'])
    oracle = split_stats(fills['R'], fills['hot_oracle'], fills['date'])
    stale = split_stats(fills['R'], fills['hot_stale'], fills['date'])

    is_early = fills['date'] < '2025-01-01'
    leg_2023_24 = split_stats(fills.loc[is_early, 'R'], fills.loc[is_early, 'hot'], fills.loc[is_early, 'date'])
    leg_2025_26 = split_stats(fills.loc[~is_early, 'R'], fills.loc[~is_early, 'hot'], fills.loc[~is_early, 'date'])

    hot_mask = fills['hot'] == 1
    ex_top5_hot = ex_top_pct(fills.loc[hot_mask, 'R'], 0.05)
    qtable = quintile_table(fills['date'].map(T_causal), fills['R'])

    weeks = pd.to_datetime(fills['date']).dt.to_period('W')
    hot_weeks = fills.loc[hot_mask].groupby(weeks[hot_mask]).size()
    fills_per_week_hot = float(hot_weeks.mean()) if len(hot_weeks) else float('nan')

    hot_defined = hot_causal.dropna()
    share_hot_all = float((hot_defined == 1).mean()) if len(hot_defined) else float('nan')
    early_idx = hot_defined.index < '2025-01-01'
    share_hot_2023_24 = float((hot_defined[early_idx] == 1).mean()) if early_idx.any() else float('nan')
    share_hot_2025_26 = float((hot_defined[~early_idx] == 1).mean()) if (~early_idx).any() else float('nan')

    legs = {
        'pooled_hot_minus_cold_ge_0.15': bool(causal['hot_minus_cold'] >= 0.15),
        'pooled_t_cluster_ge_2': bool(causal['t_cluster'] >= 2),
        '2023_24_hot_minus_cold_gt_0': bool(leg_2023_24['hot_minus_cold'] > 0),
        '2023_24_each_cohort_ge_15': bool(min(leg_2023_24['n_hot'], leg_2023_24['n_cold']) >= 15),  # PREREG: EACH cohort
        '2025_26_hot_minus_cold_gt_0': bool(leg_2025_26['hot_minus_cold'] > 0),
        '2025_26_each_cohort_ge_15': bool(min(leg_2025_26['n_hot'], leg_2025_26['n_cold']) >= 15),  # PREREG: EACH cohort
        'ex_top5_hot_gt_0': bool(ex_top5_hot > 0),
    }
    verdict = 'PASS' if all(legs.values()) else 'FAIL'

    return dict(cell='1420', n_fills=len(fills), causal=causal, oracle=oracle, stale=stale,
                leg_2023_24=leg_2023_24, leg_2025_26=leg_2025_26, ex_top5_hot=ex_top5_hot,
                quintile_table=qtable, fills_per_week_hot=fills_per_week_hot,
                share_hot_all=share_hot_all, share_hot_2023_24=share_hot_2023_24,
                share_hot_2025_26=share_hot_2025_26, verdict=verdict, verdict_legs=legs,
                T_causal=T_causal)


def score_hod_cell(b0_trades: pd.DataFrame, window: int = 20, min_n: int = 40, min_hist: int = 60) -> dict:
    """Cell 1,421 (PREREG.md). b0_trades has split=='TEST' already dropped (load_hod_population()).
    T_HOD = mean net_R of ALL TRAIN+VAL signals; TRAIN-H1 feeds the thermometer's history only (burn-in)
    -- the SCORED sample is TRAIN-H2 + VAL. Hot-cohort fills/week via
    research/hod_consol/run_consol.simulate_slots (first-12/day, 4-concurrent) on VAL."""
    daily_r = daily_outcome_series(b0_trades, 'day', 'net_R')
    T_causal = thermometer(daily_r, window, min_n, end_lag=1)
    T_oracle = thermometer(daily_r, window, min_n, end_lag=0)
    T_stale = thermometer(daily_r, window, min_n, end_lag=STALE_END_LAG)
    hot_causal = hot_flags(T_causal, min_hist)
    hot_oracle = hot_flags(T_oracle, min_hist)
    hot_stale = hot_flags(T_stale, min_hist)

    scored = b0_trades[b0_trades['half'] != 'H1'].copy()  # drop TRAIN-H1 burn-in from the SCORED sample only
    scored['hot'] = scored['day'].map(hot_causal)
    scored['hot_oracle'] = scored['day'].map(hot_oracle)
    scored['hot_stale'] = scored['day'].map(hot_stale)

    train_h2 = scored[scored['split'] == 'TRAIN']
    val = scored[scored['split'] == 'VAL']

    causal = split_stats(scored['net_R'], scored['hot'], scored['day'])
    oracle = split_stats(scored['net_R'], scored['hot_oracle'], scored['day'])
    stale = split_stats(scored['net_R'], scored['hot_stale'], scored['day'])
    leg_train_h2 = split_stats(train_h2['net_R'], train_h2['hot'], train_h2['day'])
    leg_val = split_stats(val['net_R'], val['hot'], val['day'])

    ex_top5_hot_val = ex_top_pct(val.loc[val['hot'] == 1, 'net_R'], 0.05)
    qtable = quintile_table(scored['day'].map(T_causal), scored['net_R'])

    val_hot = val[val['hot'] == 1].copy()
    try:
        sys.path.insert(0, str(ROOT / 'research/hod_consol'))
        from run_consol import simulate_slots  # noqa: E402
    except Exception as e:
        log.error(f"score_hod_cell: import of run_consol.simulate_slots FAILED ({e}) -- "
                  f"using the inline fallback copy of the same first-12/day 4-concurrent rule")
        simulate_slots = _simulate_slots_fallback
    kept = simulate_slots(val_hot) if len(val_hot) else pd.Series(dtype=bool)
    n_weeks_val = val['wk'].nunique() if 'wk' in val.columns else float('nan')
    fills_per_week_hot = float(kept.sum() / n_weeks_val) if n_weeks_val else float('nan')

    legs = {
        'train_h2_hot_minus_cold_ge_0.10': bool(leg_train_h2['hot_minus_cold'] >= 0.10),
        'val_hot_minus_cold_ge_0.10': bool(leg_val['hot_minus_cold'] >= 0.10),
        'val_t_cluster_ge_2': bool(leg_val['t_cluster'] >= 2),
        'val_hot_mean_net_r_ge_0': bool(leg_val['mean_hot'] >= 0),
        'fills_per_week_hot_ge_3': bool(fills_per_week_hot >= 3),
    }
    verdict = 'PASS' if all(legs.values()) else 'FAIL'

    return dict(cell='1421', n_scored=len(scored), causal=causal, oracle=oracle, stale=stale,
                leg_train_h2=leg_train_h2, leg_val=leg_val, ex_top5_hot_val=ex_top5_hot_val,
                quintile_table=qtable, fills_per_week_hot=fills_per_week_hot,
                verdict=verdict, verdict_legs=legs)


def score_bf_cell(bf_trades: pd.DataFrame, T_orb_causal: pd.Series, min_hist: int = 60) -> dict:
    """Cell 1,422 (report-only, no pass bar): bull-flag P1 trades, R = pnl/(shares*|entry-stop|)
    (research/bf_2024/score.py's convention), split by the ORB thermometer's hot/cold (cross-book)."""
    df = bf_trades.copy()
    stop_col = 'stop_price' if 'stop_price' in df.columns else 'stop_loss'
    risk_per_share = (df['entry_price'].astype(float) - df[stop_col].astype(float)).abs()
    denom = (df['shares'].astype(float) * risk_per_share).replace(0, np.nan)
    df['R'] = df['pnl'].astype(float) / denom
    n_before = len(df)
    df = df.dropna(subset=['R'])
    if len(df) < n_before:
        log.warning(f"score_bf_cell: dropped {n_before - len(df)} of {n_before} rows with zero/NaN R denominator")
    hot_series = hot_flags(T_orb_causal, min_hist)
    df['hot'] = df['date'].map(hot_series)
    stats = split_stats(df['R'], df['hot'], df['date'])
    ex_top5 = ex_top_pct(df['R'], 0.05)
    return dict(cell='1422', n=len(df), stats=stats, ex_top5=ex_top5, report_only=True)


# --------------------------------------------------------------------------- report
def _fmt_stats(d: dict) -> str:
    return (f"n_hot={d['n_hot']} n_cold={d['n_cold']} mean_hot={d['mean_hot']:.4f} "
            f"mean_cold={d['mean_cold']:.4f} hot-cold={d['hot_minus_cold']:.4f} "
            f"t_cluster={d['t_cluster']:.2f} t_iid={d['t_iid']:.2f}")


def render_report(cell_1420: dict, cell_1421: dict, cell_1422, population: pd.DataFrame,
                   hod: pd.DataFrame, book: pd.DataFrame, smoke: bool) -> str:
    """Assemble REPORT.md: one table per cell, the frozen PASS/FAIL verdict with each leg shown, the
    adversary lenses, and a Data section with row counts / date ranges (SPEC.md step 6)."""
    L = ["# REPORT -- breakout thermometer (research/thermo/PREREG.md, cells 1,420-1,422)", ""]
    if smoke:
        L += ["**SMOKE RUN** -- no 2025-26 ORB book; cell 1420 legs run on 2023-24 fills only. "
              "Not a scored run; never written to the real REPORT.md path.", ""]
    L += ["## Cell 1,420 -- ORB live-config fills x T_ORB", f"n_fills={cell_1420['n_fills']}",
          f"- Causal: {_fmt_stats(cell_1420['causal'])}",
          f"- Oracle (same-day T, look-ahead): {_fmt_stats(cell_1420['oracle'])}",
          f"- Stale (window ends 60 trading days earlier): {_fmt_stats(cell_1420['stale'])}",
          f"- 2023-24 leg: {_fmt_stats(cell_1420['leg_2023_24'])}",
          f"- 2025-26 leg: {_fmt_stats(cell_1420['leg_2025_26'])}",
          f"- hot-cohort ex-top-5% mean R: {cell_1420['ex_top5_hot']:.4f}",
          f"- fills/week in hot weeks: {cell_1420['fills_per_week_hot']:.3f}",
          f"- share of days hot: all={cell_1420['share_hot_all']:.3f} "
          f"2023-24={cell_1420['share_hot_2023_24']:.3f} 2025-26={cell_1420['share_hot_2025_26']:.3f}",
          f"- quintile table (T bucket -> mean R, count): {cell_1420['quintile_table'].to_dict('records')}",
          f"- **VERDICT: {cell_1420['verdict']}** legs={cell_1420['verdict_legs']}", ""]
    L += ["## Cell 1,421 -- HOD B0 signals x T_HOD (TRAIN-H2 + VAL; TRAIN-H1 = burn-in; TEST dropped on read)",
          f"n_scored={cell_1421['n_scored']}",
          f"- Causal: {_fmt_stats(cell_1421['causal'])}",
          f"- Oracle (same-day T, look-ahead): {_fmt_stats(cell_1421['oracle'])}",
          f"- Stale (window ends 60 trading days earlier): {_fmt_stats(cell_1421['stale'])}",
          f"- TRAIN-H2 leg: {_fmt_stats(cell_1421['leg_train_h2'])}",
          f"- VAL leg: {_fmt_stats(cell_1421['leg_val'])}",
          f"- hot-cohort VAL ex-top-5% mean net_R: {cell_1421['ex_top5_hot_val']:.4f}",
          f"- hot fills/week (VAL, first-12/day 4-concurrent slot sim): {cell_1421['fills_per_week_hot']:.3f}",
          f"- quintile table (T bucket -> mean net_R, count): {cell_1421['quintile_table'].to_dict('records')}",
          f"- **VERDICT: {cell_1421['verdict']}** legs={cell_1421['verdict_legs']}", ""]
    L.append("## Cell 1,422 -- bull-flag P1 trades x T_ORB (cross-book, REPORT-ONLY, no pass bar)")
    if cell_1422 is None:
        L.append("SKIPPED (smoke mode or missing BF CSVs) -- see run_thermo.log.")
    else:
        L += [f"n={cell_1422['n']}", f"- {_fmt_stats(cell_1422['stats'])}",
              f"- ex-top-5% mean R: {cell_1422['ex_top5']:.4f}",
              "- Caveat (research/bf_2024/REPORT.md): the 2024H2 file's raw n includes QBTS.WS, a "
              "warrant kept only by symbol-list match; the live rule excludes it (corrected n=26, "
              "+0.104 R). Not re-filtered here -- report-only, no PREREG basis to change the population."]
    L.append("")
    L += ["## Data",
          f"- ORB population: {len(population)} rows, {population['date'].nunique()} days, "
          f"{population['date'].min()}..{population['date'].max()}",
          f"- ORB book (fills, entered==1): {int((book['entered'] == 1).sum())} of {len(book)} rows",
          f"- HOD (TRAIN+VAL, TEST dropped): {len(hod)} rows, {hod['day'].nunique()} days, "
          f"{hod['day'].min()}..{hod['day'].max()}", ""]
    L += ["## Adversary lenses (PREREG.md)",
          "Oracle (look-ahead) hot-cold must exceed the causal hot-cold, else the causal number is noise. "
          "Stale (window ending 60 trading days earlier) reported beside. Per-period legs rule out a pure "
          "2024->2025 level shift carrying the pooled result alone."]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- consistency check
def check_2025_book_consistency(book_2025_26: pd.DataFrame, tol: float = 0.05) -> tuple:
    """SPEC.md step 3: the rebuilt 2025-26 book's 2025 fills (entered==1) must match
    research/orb_seed_wide/out/runB_true.csv's 2025 fills within `tol` on n and total _sized_pnl.
    Returns (ok: bool, message: str)."""
    ref = read_orb_csv(str(RUNB_TRUE_2025))
    ref_2025 = ref[(ref['date'] < '2026-01-01') & (ref['entered'] == 1)]
    new_2025 = book_2025_26[(book_2025_26['date'] < '2026-01-01') & (book_2025_26['entered'] == 1)]
    ref_n, new_n = len(ref_2025), len(new_2025)
    ref_sum = float(ref_2025['_sized_pnl'].astype(float).sum())
    new_sum = float(new_2025['_sized_pnl'].astype(float).sum())
    n_ok = ref_n > 0 and abs(new_n - ref_n) / ref_n <= tol
    sum_ok = ref_sum != 0 and abs(new_sum - ref_sum) / abs(ref_sum) <= tol
    ok = n_ok and sum_ok
    msg = (f"consistency check: ref(n={ref_n}, sum=${ref_sum:,.2f}) vs new(n={new_n}, sum=${new_sum:,.2f}) "
           f"-- n_ok={n_ok} sum_ok={sum_ok} tol={tol}")
    (log.info if ok else log.error)(f"check_2025_book_consistency: {msg}")
    return ok, msg


# --------------------------------------------------------------------------- CLI
def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default=str(ROOT / 'research/thermo/REPORT.md'))
    parser.add_argument('--orb-book-2025-26', default=None,
                         help="Rebuilt 2025-26 ORB book. Omit for a 2023-24-only smoke run.")
    parser.add_argument('--skip-consistency-check', action='store_true',
                         help="Skip the 2025 vs runB_true.csv consistency check (smoke mode only).")
    args = parser.parse_args()

    log.info("=== thermo scoring run starting ===")
    population = load_orb_population()
    book_parts = [read_orb_csv(str(ORB_BOOK_2023)), read_orb_csv(str(ORB_BOOK_2024))]
    book_2025_26 = None
    if args.orb_book_2025_26:
        book_2025_26 = read_orb_csv(args.orb_book_2025_26)
        if not args.skip_consistency_check:
            ok, msg = check_2025_book_consistency(book_2025_26)
            if not ok:
                Path(args.out).write_text(f"# REPORT -- STOPPED\n\nConsistency check FAILED.\n\n{msg}\n")
                log.error(f"main: STOP -- consistency check failed, wrote discrepancy to {args.out}")
                return
        book_parts.append(book_2025_26)
    else:
        log.warning("main: no --orb-book-2025-26 given -- cell 1420 legs run on 2023-24 books ONLY (smoke mode)")
    book = pd.concat(book_parts, ignore_index=True)

    cell_1420 = score_orb_cell(population, book)

    hod = load_hod_population()
    cell_1421 = score_hod_cell(hod)

    cell_1422 = None
    if BF_2024H2_CSV.exists() and BF_2025_26_CSV.exists():
        bf_2024 = pd.read_csv(BF_2024H2_CSV, keep_default_na=False, na_values=[''])
        bf_2025_26 = pd.read_csv(BF_2025_26_CSV, keep_default_na=False, na_values=[''])
        bf_trades = pd.concat([bf_2024, bf_2025_26], ignore_index=True)
        cell_1422 = score_bf_cell(bf_trades, cell_1420['T_causal'])
    else:
        log.warning(f"main: BF trade CSVs missing ({BF_2024H2_CSV} / {BF_2025_26_CSV}) -- skipping cell 1422")

    report = render_report(cell_1420, cell_1421, cell_1422, population, hod, book, smoke=book_2025_26 is None)
    Path(args.out).write_text(report)
    log.info(f"main: wrote {args.out} (1420={cell_1420['verdict']}, 1421={cell_1421['verdict']})")


if __name__ == '__main__':
    main()
