"""Benchmark 3 runner-capture exit variants vs the shipped static_lock_1R baseline.

Motivation: today 2026-04-24, ATOM ran +8.5R (entry $6.73 → day high $9.45,
range_size $0.32). We captured +1R ($0.28/share = +$1,913 on 6842 shares) via
`static_lock_1R`. The other +7.5R (~$16,700) was left on the table because the
lock caps upside per-trade.

Variants tested (all use the SAME defended pipeline — composite filter +
quintile ranking + family/super-group dedup + adaptive mults + risk-parity
sizing — so only the exit differs):

  V0 (baseline): static_lock_1R — arm at +1.5R, stop fixed at +1R forever.
  V1 trail_after_arm_0.5R — arm at +1.5R, trail 0.5R behind running peak.
  V2 partial_50_at_1R_plus_trail_1R — 50% exits at static +1R, 50% trails
                                       1R behind peak (pocket the R, let
                                       the other half run).
  V3 quintile_aware — Q5/Q4 use V1 trail, Q3/Q2/Q1 use V0 static. Hypothesis:
                      high-composite setups are runner-biased.

Honest comparison caveats:
  - The baseline simulator has an intra-bar sequence ambiguity on arm bars
    (arm fires using bar_high, exit check uses bar_low in the same pass).
    ALL variants use the exact same bar loop convention, so the bias is
    applied uniformly and the comparison is fair.
  - Adaptive mults are refit per-variant on H1 2025 TRAIN since the per-
    quintile mean P&L differs. This is equivalent to re-running the
    pipeline from scratch per variant, which is what you'd do if you
    shipped it.

Output metrics per variant:
  - Total P&L (full timeline Jan'25 → latest)
  - Full-timeline continuous max drawdown (not intra-month)
  - Calmar = P&L / |max DD|
  - Monthly summary (red months, worst/best/median)
  - Capture ratio (realized / theoretical max MFE) — overall + for MFE > 3R runners
"""
from __future__ import annotations

import glob
import os
import sys
from datetime import timedelta
from typing import Callable, Tuple, Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from persistence.database import Database
from study_orb import _bars_to_df
from study_orb_filter import FILTER_FEATURES, fit_z_params, composite_score
from study_orb_sizing import (
    FILTER_THRESHOLD, fit_quintile_cutoffs, assign_quintile,
    ADAPTIVE_MULT_MIN,
)
from study_orb_correlation_filter import symbol_family, symbol_super_group


ACCOUNT = 100_000.0
N = 4
RISK = 3000.0
MIN_STOP_PCT = 1.0
OLD_POS = 50_000.0
Q_CAPS = {'Q1': 3.0, 'Q2': 3.0, 'Q3': 3.0, 'Q4': 3.0, 'Q5': 1.5}
Q_ORDER = {'Q4': 0, 'Q5': 1, 'Q3': 2, 'Q2': 3, 'Q1': 4}
LOCK_TRIGGER_R = 1.5
LOCK_STOP_R = 1.0
EXIT_SLIP_BPS = 10.0

# V1 parameters
V1_TRAIL_R = 0.5

# V2 parameters
V2_PARTIAL_PCT = 0.5      # fraction exited at static +1R lock
V2_RUNNER_TRAIL_R = 1.0   # trail distance for the runner half

# V3 parameters (which quintiles use trail vs static)
V3_TRAIL_QUINTILES = {'Q4', 'Q5'}


def _session_open_timestamp(bars):
    """ET 9:30 timestamp."""
    if bars.empty:
        return None
    bars_et = bars.copy()
    bars_et['et'] = bars_et['timestamp'].dt.tz_convert('America/New_York')
    morning = bars_et[bars_et['et'].dt.time == pd.Timestamp('09:30').time()]
    if len(morning):
        return morning.iloc[0]['timestamp']
    return None


# ---------------------------------------------------------------------------
# Exit simulators. Each returns (exit_price, exit_reason, mfe_r).
# mfe_r = max favorable excursion in R-multiples from entry.
# ---------------------------------------------------------------------------


def simulate_static_lock(bars, entry_price, range_high, range_low, entry_time
                         ) -> Tuple[float, str, float]:
    """V0 baseline: arm at +1.5R, lock stop at +1R forever."""
    range_size = range_high - range_low
    trigger_lvl = entry_price + LOCK_TRIGGER_R * range_size
    lock_stop = entry_price + LOCK_STOP_R * range_size
    stop_price = range_low
    armed = False
    mfe_abs = 0.0
    post = bars[bars['timestamp'] >= entry_time].reset_index(drop=True)
    for _, row in post.iloc[1:].iterrows():
        bar_high = float(row['high']); bar_low = float(row['low'])
        mfe_abs = max(mfe_abs, bar_high - entry_price)
        if not armed and bar_high >= trigger_lvl:
            armed = True
            stop_price = max(stop_price, lock_stop)
        if bar_low <= stop_price:
            px = stop_price * (1 - EXIT_SLIP_BPS/10000)
            return px, 'lock' if armed else 'stop', mfe_abs / range_size
    last = post.iloc[-1]
    px = float(last['close']) * (1 - EXIT_SLIP_BPS/10000)
    return px, 'eod', mfe_abs / range_size


def simulate_trail_after_arm(bars, entry_price, range_high, range_low, entry_time,
                              trail_r: float = V1_TRAIL_R
                              ) -> Tuple[float, str, float]:
    """V1: arm at +1.5R, then trail `trail_r` behind running peak.

    Invariant: post-arm stop is always >= entry+1R (peak >= +1.5R when armed,
    so peak - 0.5R >= +1R). Never exits lower than V0 static_lock_1R, but can
    exit much higher on runners.

    Intra-bar policy: same as V0 — arm check uses bar_high first, then exit
    check uses bar_low. On arm bar, peak is set to bar_high (may be far above
    trigger on a strong breakout bar), stop jumps to peak - trail_r.
    """
    range_size = range_high - range_low
    trigger_lvl = entry_price + LOCK_TRIGGER_R * range_size
    stop_price = range_low
    armed = False
    peak_high = 0.0
    mfe_abs = 0.0
    post = bars[bars['timestamp'] >= entry_time].reset_index(drop=True)
    for _, row in post.iloc[1:].iterrows():
        bar_high = float(row['high']); bar_low = float(row['low'])
        mfe_abs = max(mfe_abs, bar_high - entry_price)
        if bar_high > peak_high:
            peak_high = bar_high
        if armed or peak_high >= trigger_lvl:
            armed = True
            stop_price = max(stop_price, peak_high - trail_r * range_size)
        if bar_low <= stop_price:
            px = stop_price * (1 - EXIT_SLIP_BPS/10000)
            return px, 'trail' if armed else 'stop', mfe_abs / range_size
    last = post.iloc[-1]
    px = float(last['close']) * (1 - EXIT_SLIP_BPS/10000)
    return px, 'eod', mfe_abs / range_size


def simulate_partial_runner(bars, entry_price, range_high, range_low, entry_time,
                             partial_pct: float = V2_PARTIAL_PCT,
                             runner_trail_r: float = V2_RUNNER_TRAIL_R
                             ) -> Tuple[float, str, float]:
    """V2: partial_pct of position exits via static_lock_1R, remainder trails
    `runner_trail_r` behind peak. Blended exit price = weighted avg.

    If the static-half exits via initial stop (range_low) — i.e. never armed
    — both halves exit at the same price (runner half still has range_low
    stop until +1.5R arms). So on a clean stop-out, V2 is identical to V0.
    The divergence is only on trades that ARM.
    """
    range_size = range_high - range_low
    trigger_lvl = entry_price + LOCK_TRIGGER_R * range_size
    lock_stop = entry_price + LOCK_STOP_R * range_size

    # Static half state
    stop_1 = range_low; armed_1 = False
    exit_1: Optional[float] = None; reason_1: Optional[str] = None

    # Runner half state
    stop_2 = range_low; armed_2 = False
    peak_high = 0.0
    exit_2: Optional[float] = None; reason_2: Optional[str] = None

    mfe_abs = 0.0
    post = bars[bars['timestamp'] >= entry_time].reset_index(drop=True)
    for _, row in post.iloc[1:].iterrows():
        bar_high = float(row['high']); bar_low = float(row['low'])
        mfe_abs = max(mfe_abs, bar_high - entry_price)
        if bar_high > peak_high:
            peak_high = bar_high

        # Static half — mirrors V0 exactly
        if exit_1 is None:
            if not armed_1 and bar_high >= trigger_lvl:
                armed_1 = True
                stop_1 = max(stop_1, lock_stop)
            if bar_low <= stop_1:
                exit_1 = stop_1 * (1 - EXIT_SLIP_BPS/10000)
                reason_1 = 'lock' if armed_1 else 'stop'

        # Runner half — trails behind peak after arm
        if exit_2 is None:
            if armed_2 or peak_high >= trigger_lvl:
                armed_2 = True
                stop_2 = max(stop_2, peak_high - runner_trail_r * range_size)
            if bar_low <= stop_2:
                exit_2 = stop_2 * (1 - EXIT_SLIP_BPS/10000)
                reason_2 = 'runner_trail' if armed_2 else 'stop'

        if exit_1 is not None and exit_2 is not None:
            break

    if exit_1 is None:
        last = post.iloc[-1]
        exit_1 = float(last['close']) * (1 - EXIT_SLIP_BPS/10000)
        reason_1 = 'eod'
    if exit_2 is None:
        last = post.iloc[-1]
        exit_2 = float(last['close']) * (1 - EXIT_SLIP_BPS/10000)
        reason_2 = 'eod'

    blended = partial_pct * exit_1 + (1 - partial_pct) * exit_2
    reason = f"{reason_1}|{reason_2}"
    return blended, reason, mfe_abs / range_size


# ---------------------------------------------------------------------------
# Per-variant full-pipeline runner.
# ---------------------------------------------------------------------------


def _simulate_all(df: pd.DataFrame, bars_cache: dict,
                  exit_fn: Callable) -> pd.DataFrame:
    """Run `exit_fn` for every row in df; returns df with pnl/pnl_pct/exit_reason/mfe_r."""
    pnls, pcts, reasons, mfes = [], [], [], []
    for _, row in df.reset_index(drop=True).iterrows():
        key = (row['symbol'], row['date'].strftime('%Y-%m-%d'))
        bars = bars_cache.get(key)
        if bars is None or bars.empty:
            pnls.append(row['pnl']); pcts.append(row['pnl_pct'])
            reasons.append(row['exit_reason']); mfes.append(0.0); continue
        open_ts = _session_open_timestamp(bars)
        if open_ts is None:
            pnls.append(row['pnl']); pcts.append(row['pnl_pct'])
            reasons.append(row['exit_reason']); mfes.append(0.0); continue
        range_end = open_ts + timedelta(minutes=5)
        range_bars = bars[(bars['timestamp'] >= open_ts) &
                          (bars['timestamp'] < range_end)]
        if len(range_bars) < 5:
            pnls.append(row['pnl']); pcts.append(row['pnl_pct'])
            reasons.append(row['exit_reason']); mfes.append(0.0); continue
        rh = float(range_bars['high'].max())
        rl = float(range_bars['low'].min())
        search = bars[(bars['timestamp'] >= range_end) &
                      (bars['timestamp'] < range_end + timedelta(minutes=60))]
        entry_ts = None
        for _, b in search.iterrows():
            if float(b['high']) > rh:
                entry_ts = b['timestamp']; break
        if entry_ts is None:
            pnls.append(row['pnl']); pcts.append(row['pnl_pct'])
            reasons.append(row['exit_reason']); mfes.append(0.0); continue
        entry_p = float(row['entry_price'])
        exit_p, reason, mfe_r = exit_fn(bars, entry_p, rh, rl, entry_ts)
        shares = max(1, int(OLD_POS / entry_p))
        pnls.append((exit_p - entry_p) * shares)
        pcts.append((exit_p - entry_p) / entry_p * 100)
        reasons.append(reason); mfes.append(mfe_r)
    out = df.reset_index(drop=True).copy()
    out['pnl'] = pnls
    out['pnl_pct'] = pcts
    out['exit_reason'] = reasons
    out['mfe_r'] = mfes
    return out


def _run_pipeline(df_with_pnl: pd.DataFrame,
                  label: str,
                  exit_override: Optional[Callable] = None) -> pd.DataFrame:
    """Apply defended pipeline (risk-parity + composite + quintile + adaptive
    mults + top-K + dedup) to df_with_pnl. Returns selected-trade DataFrame
    with _sized_pnl + mfe_r + _quintile columns.

    If `exit_override` is provided, it's a dict {symbol_date: (pnl, pnl_pct,
    reason, mfe_r)} used to override specific trades after quintile assign
    (for V3 quintile-aware, where trade exit depends on quintile).
    """
    df = df_with_pnl.copy()

    per_pos_cap = ACCOUNT / N
    stop = df['range_size_pct'].clip(lower=MIN_STOP_PCT)
    uncap = RISK / (stop / 100.0)
    df['_rp_position'] = uncap.clip(upper=per_pos_cap)
    df['_rp_pnl'] = df['pnl'] * df['_rp_position'] / OLD_POS

    train = df[(df['date'] >= '2025-01-01') & (df['date'] <= '2025-06-30')]
    params = fit_z_params(train, FILTER_FEATURES)
    df['_composite'] = composite_score(df, params)
    train = df[(df['date'] >= '2025-01-01') & (df['date'] <= '2025-06-30')]
    train_k = train[train['_composite'] >= FILTER_THRESHOLD].copy()
    cutoffs = fit_quintile_cutoffs(train_k['_composite'])
    train_k['_quintile'] = assign_quintile(train_k['_composite'], cutoffs)
    avg = float(train_k['_rp_pnl'].mean())
    mults = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        sub = train_k[train_k['_quintile'] == q]
        m = float(sub['_rp_pnl'].mean()) / avg if len(sub) else 1.0
        mults[q] = max(ADAPTIVE_MULT_MIN, min(Q_CAPS[q], m))
    print(f"[{label}] Adaptive mults: { {q: round(v,3) for q,v in mults.items()} }")

    kept = df[df['_composite'] >= FILTER_THRESHOLD].copy()
    kept['_quintile'] = assign_quintile(kept['_composite'], cutoffs)

    sel_rows = []
    for _, dg in kept.groupby('date'):
        d = dg.copy()
        d['_q_rank'] = d['_quintile'].map(Q_ORDER)
        d = d.sort_values(['_q_rank', '_composite'], ascending=[True, False])
        seen_fam = set(); seen_sup = set()
        today = []
        for _, r in d.iterrows():
            fam = symbol_family(r['symbol']); sup = symbol_super_group(r['symbol'])
            if fam and fam in seen_fam: continue
            if sup and sup in seen_sup: continue
            if fam: seen_fam.add(fam)
            if sup: seen_sup.add(sup)
            today.append(r)
            if len(today) >= N: break
        sel_rows.extend(today)
    sel = pd.DataFrame(sel_rows)
    sel['_sized_pnl'] = sel.apply(lambda r: r['_rp_pnl'] * mults[r['_quintile']], axis=1)
    sel['date'] = pd.to_datetime(sel['date'])
    sel['month'] = sel['date'].dt.to_period('M').astype(str)
    return sel


def _metrics(sel: pd.DataFrame, label: str) -> dict:
    """Compute headline metrics from selected-trade df."""
    daily = sel.groupby('date').agg(
        pnl=('_sized_pnl', 'sum'),
        picks=('_sized_pnl', 'count'),
    ).reset_index().sort_values('date').reset_index(drop=True)

    # Full-timeline continuous DD (not intra-month)
    daily['cum'] = daily['pnl'].cumsum()
    peak = -1e18; mdd = 0.0; mdd_date = None; peak_date = None; trough_date = None
    cur_peak_date = None
    for _, r in daily.iterrows():
        if r['cum'] > peak:
            peak = r['cum']; cur_peak_date = r['date']
        dd = r['cum'] - peak
        if dd < mdd:
            mdd = dd; peak_date = cur_peak_date; trough_date = r['date']

    total_pnl = float(daily['pnl'].sum())
    calmar = total_pnl / abs(mdd) if mdd < 0 else float('inf')

    # Monthly
    daily['month'] = daily['date'].dt.to_period('M')
    monthly = daily.groupby('month')['pnl'].sum()
    neg_months = int((monthly < 0).sum())
    worst_month = monthly.min()
    worst_month_label = str(monthly.idxmin())

    # Capture ratios
    total_mfe_pnl = 0.0
    total_realized_pnl = 0.0
    runner_realized = 0.0; runner_mfe = 0.0; runner_count = 0
    for _, r in sel.iterrows():
        # theoretical max R-multiple captured if we exited at MFE peak
        mfe_r = float(r.get('mfe_r', 0.0))
        # realized R-multiple = pnl_pct / (range_size_pct * LOCK_STOP_R)
        # Actually easier: realized_r = (pnl / (shares * 1R_per_share)) but we don't have shares here.
        # Use pnl_pct and range_size_pct: realized_r = pnl_pct / range_size_pct
        rsp = float(r.get('range_size_pct', 0))
        if rsp <= 0:
            continue
        realized_r = float(r.get('pnl_pct', 0)) / rsp
        # Weight by sized pnl contribution? For capture we want per-trade then aggregate R-wise.
        total_mfe_pnl += mfe_r
        total_realized_pnl += realized_r
        if mfe_r >= 3.0:
            runner_count += 1
            runner_mfe += mfe_r
            runner_realized += realized_r

    capture_overall = (total_realized_pnl / total_mfe_pnl * 100) if total_mfe_pnl > 0 else 0.0
    capture_runners = (runner_realized / runner_mfe * 100) if runner_mfe > 0 else 0.0

    return {
        'label': label,
        'trades': len(sel),
        'pnl': total_pnl,
        'max_dd': float(mdd),
        'dd_peak_date': peak_date,
        'dd_trough_date': trough_date,
        'calmar': calmar,
        'neg_months': neg_months,
        'worst_month_pnl': float(worst_month),
        'worst_month': worst_month_label,
        'capture_overall_pct': capture_overall,
        'capture_runners_pct': capture_runners,
        'runner_count': runner_count,
        'avg_mfe_r': float(sel['mfe_r'].mean()) if 'mfe_r' in sel else 0.0,
    }


def _print_variant_header(label: str):
    print(f"\n{'='*78}")
    print(f"  {label}")
    print(f"{'='*78}")


def _print_metrics(m: dict):
    print(f"  Trades:            {m['trades']}")
    print(f"  Total P&L:         ${m['pnl']:+,.0f}")
    print(f"  Max DD:            ${m['max_dd']:+,.0f}   "
          f"(peak {m['dd_peak_date']} → trough {m['dd_trough_date']})")
    print(f"  Calmar:            {m['calmar']:.2f}x")
    print(f"  Neg months:        {m['neg_months']}   "
          f"(worst: {m['worst_month']} ${m['worst_month_pnl']:+,.0f})")
    print(f"  Avg MFE (R):       {m['avg_mfe_r']:.2f}")
    print(f"  Runners (MFE>=3R): {m['runner_count']}")
    print(f"  Capture (overall): {m['capture_overall_pct']:.1f}%")
    print(f"  Capture (runners): {m['capture_runners_pct']:.1f}%")


def main():
    csv = sorted(glob.glob('analysis_results/orb_features_*.csv'))
    csv = [p for p in csv if 'corrmatrix' not in p][-1]
    print(f"Loading features from {csv}")
    df = pd.read_csv(csv)
    needed = [f for f, _ in FILTER_FEATURES]
    df = df.dropna(subset=needed + ['pnl', 'date', 'pnl_pct', 'range_size_pct', 'entry_price'])
    df['date'] = pd.to_datetime(df['date'])

    pairs = list(df[['symbol', 'date']].drop_duplicates().apply(
        lambda r: (r['symbol'], r['date'].strftime('%Y-%m-%d')), axis=1))
    print(f"Loading bars for {len(pairs)} (symbol, date) pairs...")
    db = Database(db_path='data/cache.db')
    raw_bars = db.get_intraday_bars_bulk(pairs)
    db.close()
    bars_cache = {k: _bars_to_df(v) for k, v in raw_bars.items()}

    # ----- Simulate each variant's raw per-trade pnl -----
    print("\nSimulating V0 static_lock_1R (baseline)...")
    df_v0 = _simulate_all(df, bars_cache, simulate_static_lock)

    print("Simulating V1 trail_after_arm_0.5R...")
    df_v1 = _simulate_all(df, bars_cache, simulate_trail_after_arm)

    print("Simulating V2 partial_50_static_1R_plus_trail_1R_runner...")
    df_v2 = _simulate_all(df, bars_cache, simulate_partial_runner)

    # ----- Run pipeline for each -----
    sel_v0 = _run_pipeline(df_v0, 'V0 static_lock_1R')
    sel_v1 = _run_pipeline(df_v1, 'V1 trail_after_arm_0.5R')
    sel_v2 = _run_pipeline(df_v2, 'V2 partial_50_plus_trail_runner_1R')

    # ----- V3: quintile-aware (depends on quintile assignment) -----
    # Strategy: take V0's quintile assignment, then for Q4/Q5 rows swap in V1's pnl.
    # We need to run pipeline once to assign quintile, then rebuild pnl per-row.
    print("\nSimulating V3 quintile_aware (Q4/Q5 trail, else static)...")
    # Use V0's quintile frame as the "authoritative" labeling of quintiles,
    # then map into df_v0/df_v1 row-by-row.
    v0_q_map = dict(zip(
        sel_v0.apply(lambda r: (r['symbol'], r['date'].strftime('%Y-%m-%d')), axis=1),
        sel_v0['_quintile']
    ))
    # For V3 we need pnl on ALL rows (before filtering) because the pipeline
    # refits z-params + cutoffs + mults from the TRAIN slice. So build a
    # df_v3 where each row's pnl = V1's if quintile in Q4/Q5, else V0's.
    # Trick: quintile is a per-composite-score thing — so we can recompute it
    # the same way inside the pipeline, but we need to know each row's
    # quintile BEFORE pnl is fed in. Chicken-and-egg.
    #
    # Solve: quintile depends on composite, not on pnl. So we can derive
    # composite + quintile from a preliminary fit on V0 pnl (since the fit
    # only affects adaptive mults, not quintile assignment itself — cutoffs
    # come from composite distribution on filtered TRAIN).
    #
    # So we do: fit cutoffs once from V0 pipeline, then assign quintile to
    # every row in df, then build df_v3 with row-wise pnl.
    prelim = df_v0.copy()
    prelim_stop = prelim['range_size_pct'].clip(lower=MIN_STOP_PCT)
    prelim['_rp_position'] = (RISK / (prelim_stop / 100.0)).clip(upper=ACCOUNT/N)
    prelim['_rp_pnl'] = prelim['pnl'] * prelim['_rp_position'] / OLD_POS
    train_pre = prelim[(prelim['date'] >= '2025-01-01') & (prelim['date'] <= '2025-06-30')]
    params_pre = fit_z_params(train_pre, FILTER_FEATURES)
    prelim['_composite'] = composite_score(prelim, params_pre)
    train_pre_k = prelim[(prelim['date'] >= '2025-01-01') &
                          (prelim['date'] <= '2025-06-30') &
                          (prelim['_composite'] >= FILTER_THRESHOLD)]
    cutoffs_pre = fit_quintile_cutoffs(train_pre_k['_composite'])
    prelim['_quintile'] = assign_quintile(prelim['_composite'], cutoffs_pre)

    # Now build df_v3: pick V1's pnl for Q4/Q5 rows, V0's for others.
    df_v3 = df_v0.copy()
    v1_idx = df_v1.set_index(['symbol', 'date'])
    v0_idx = df_v0.set_index(['symbol', 'date'])
    new_pnl, new_pct, new_reason, new_mfe = [], [], [], []
    for _, row in df_v3.iterrows():
        q = prelim.loc[row.name, '_quintile'] if row.name in prelim.index else 'Q3'
        key = (row['symbol'], row['date'])
        if q in V3_TRAIL_QUINTILES:
            src = v1_idx.loc[key] if key in v1_idx.index else row
        else:
            src = v0_idx.loc[key] if key in v0_idx.index else row
        # When duplicated index, loc returns DataFrame; take first
        if isinstance(src, pd.DataFrame):
            src = src.iloc[0]
        new_pnl.append(float(src['pnl']))
        new_pct.append(float(src['pnl_pct']))
        new_reason.append(str(src['exit_reason']))
        new_mfe.append(float(src['mfe_r']) if 'mfe_r' in src.index else 0.0)
    df_v3['pnl'] = new_pnl
    df_v3['pnl_pct'] = new_pct
    df_v3['exit_reason'] = new_reason
    df_v3['mfe_r'] = new_mfe

    sel_v3 = _run_pipeline(df_v3, 'V3 quintile_aware')

    # ----- Metrics + comparison -----
    metrics = [
        _metrics(sel_v0, 'V0 static_lock_1R (shipped)'),
        _metrics(sel_v1, 'V1 trail_after_arm_0.5R'),
        _metrics(sel_v2, f'V2 partial_{int(V2_PARTIAL_PCT*100)}_plus_trail_{V2_RUNNER_TRAIL_R}R'),
        _metrics(sel_v3, f'V3 quintile_aware (Q4/Q5 trail, Q1-Q3 static)'),
    ]

    for m in metrics:
        _print_variant_header(m['label'])
        _print_metrics(m)

    # ----- Comparison table -----
    print(f"\n{'='*78}")
    print("  COMPARISON vs baseline")
    print(f"{'='*78}")
    base = metrics[0]
    print(f"{'Variant':<45} {'P&L':>12} {'Δ P&L':>10} {'Calmar':>9} {'Capture':>9}")
    print('-' * 85)
    for m in metrics:
        delta = m['pnl'] - base['pnl']
        print(f"{m['label']:<45} "
              f"${m['pnl']:>+10,.0f}  "
              f"${delta:>+8,.0f}  "
              f"{m['calmar']:>6.2f}x   "
              f"{m['capture_overall_pct']:>5.1f}%")

    # ----- Save trade-level CSVs -----
    out_dir = 'analysis_results'
    for label, sel in [('v0', sel_v0), ('v1', sel_v1), ('v2', sel_v2), ('v3', sel_v3)]:
        path = f'{out_dir}/orb_exit_runner_{label}_trades.csv'
        sel.to_csv(path, index=False)
    print(f"\nSaved trade-level CSVs to {out_dir}/orb_exit_runner_*.csv")

    # ----- Runner-specific deep dive -----
    print(f"\n{'='*78}")
    print("  RUNNER TRADES (MFE >= 3R) — per-variant capture comparison")
    print(f"{'='*78}")
    # Use V0 as the canonical trade-selection set (all variants select same trades pre-exit)
    runners_v0 = sel_v0[sel_v0['mfe_r'] >= 3.0].copy()
    # Map each runner trade's pnl across variants
    sig_v1 = sel_v1.set_index(['symbol', 'date'])
    sig_v2 = sel_v2.set_index(['symbol', 'date'])
    sig_v3 = sel_v3.set_index(['symbol', 'date'])
    rows = []
    for _, r in runners_v0.iterrows():
        k = (r['symbol'], r['date'])
        rows.append({
            'symbol': r['symbol'],
            'date': str(r['date'].date()),
            'mfe_r': r['mfe_r'],
            'v0_pnl': r['_sized_pnl'],
            'v1_pnl': sig_v1.loc[k, '_sized_pnl'] if k in sig_v1.index else None,
            'v2_pnl': sig_v2.loc[k, '_sized_pnl'] if k in sig_v2.index else None,
            'v3_pnl': sig_v3.loc[k, '_sized_pnl'] if k in sig_v3.index else None,
        })
    if rows:
        rdf = pd.DataFrame(rows).sort_values('mfe_r', ascending=False)
        # Flatten if any column has list-like values from duplicate indices
        for c in ('v1_pnl', 'v2_pnl', 'v3_pnl'):
            rdf[c] = rdf[c].apply(lambda v: v.iloc[0] if hasattr(v, 'iloc') else v)
        print(f"Runners found: {len(rdf)}")
        print(f"Top 15 by MFE:")
        pd.set_option('display.width', 200)
        pd.set_option('display.float_format', '{:,.0f}'.format)
        print(rdf.head(15).to_string(index=False))
        # Totals
        print(f"\nTotal runner P&L by variant:")
        print(f"  V0 (static):    ${rdf['v0_pnl'].sum():+,.0f}")
        print(f"  V1 (trail):     ${rdf['v1_pnl'].sum():+,.0f}  "
              f"(Δ ${rdf['v1_pnl'].sum() - rdf['v0_pnl'].sum():+,.0f})")
        print(f"  V2 (partial):   ${rdf['v2_pnl'].sum():+,.0f}  "
              f"(Δ ${rdf['v2_pnl'].sum() - rdf['v0_pnl'].sum():+,.0f})")
        print(f"  V3 (Q-aware):   ${rdf['v3_pnl'].sum():+,.0f}  "
              f"(Δ ${rdf['v3_pnl'].sum() - rdf['v0_pnl'].sum():+,.0f})")


if __name__ == '__main__':
    main()
