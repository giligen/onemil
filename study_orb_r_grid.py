"""ORB lock-arm R-grid sweep with TRAIN/VAL/HOLDOUT walk-forward (2026-05-08).

Question: is +1.5R/+1.0R the right lock-arm/lock-stop pair, or are we
leaving money on the table because most trades never reach +1.5R?

Walk-forward design:
  TRAIN   = 2025-01-01 to 2025-06-30   (fits z-params, cutoffs, adaptive mults)
  VAL     = 2025-07-01 to 2025-12-31   (picks the best variant)
  HOLDOUT = 2026-01-01 to 2026-04-30   (final OOS report — winner only)

Grid:
  trigger_r ∈ {0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5}
  lock_r    ∈ {0.0, 0.25, 0.5, 0.75, 1.0, 1.25}
  Constraint: lock_r < trigger_r

Bug-watch protocol (lessons from earlier today):
1. P&L is computed once per trade, no double counting
2. Sanity check: current 1.5/1.0 reproduces documented HOLDOUT
3. Date splits are disjoint
4. Same trade set across variants (only exits differ; entries identical)
5. Slippage applied consistently (EXIT_SLIP_BPS already in pipeline)
"""
import glob
import os
import sys
from datetime import timedelta
from typing import Dict, List, Tuple

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from persistence.database import Database
from study_orb import _bars_to_df
from study_orb_filter import FILTER_FEATURES, fit_z_params, composite_score
from study_orb_sizing import (
    FILTER_THRESHOLD, fit_quintile_cutoffs, assign_quintile, ADAPTIVE_MULT_MIN,
)
from study_orb_correlation_filter import symbol_family, symbol_super_group


# ---- Constants from study_orb_pipeline_static_lock.py ----
ACCOUNT = 100_000.0
N = 4
RISK = 3000.0
MIN_STOP_PCT = 1.0
OLD_POS = 50_000.0
Q_CAPS = {'Q1': 3.0, 'Q2': 3.0, 'Q3': 3.0, 'Q4': 3.0, 'Q5': 1.5}
Q_ORDER = {'Q4': 0, 'Q5': 1, 'Q3': 2, 'Q2': 3, 'Q1': 4}
EXIT_SLIP_BPS = 10.0

# ---- Splits ----
TRAIN_START, TRAIN_END = '2025-01-01', '2025-06-30'
VAL_START,   VAL_END   = '2025-07-01', '2025-12-31'
HOLD_START,  HOLD_END  = '2026-01-01', '2026-04-30'

# ---- Grid ----
TRIGGER_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
LOCK_GRID    = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]


def _session_open_timestamp(bars):
    if bars.empty:
        return None
    bars_et = bars.copy()
    bars_et['et'] = bars_et['timestamp'].dt.tz_convert('America/New_York')
    morning = bars_et[bars_et['et'].dt.time == pd.Timestamp('09:30').time()]
    if len(morning):
        return morning.iloc[0]['timestamp']
    return None


def simulate_lock(bars, entry_price, range_high, range_low, entry_time,
                  trigger_r: float, lock_r: float):
    """Simulate static-lock exit with configurable (trigger_r, lock_r).

    Mirrors study_orb_pipeline_static_lock.simulate_static_lock — same
    bar-by-bar walk, same slippage. Differences: trigger_r and lock_r are
    parameters not module constants.

    Returns (exit_price, exit_reason).
    """
    range_size = range_high - range_low
    trigger_lvl = entry_price + trigger_r * range_size
    lock_stop = entry_price + lock_r * range_size
    stop_price = range_low
    armed = False
    post = bars[bars['timestamp'] >= entry_time].reset_index(drop=True)
    for _, row in post.iloc[1:].iterrows():
        bar_high = float(row['high']); bar_low = float(row['low'])
        if not armed and bar_high >= trigger_lvl:
            armed = True
            stop_price = max(stop_price, lock_stop)
        if bar_low <= stop_price:
            return stop_price * (1 - EXIT_SLIP_BPS/10000), 'lock' if armed else 'stop'
    last = post.iloc[-1]
    return float(last['close']) * (1 - EXIT_SLIP_BPS/10000), 'eod'


def aggregate(pnls: List[float]) -> Dict:
    if not pnls:
        return {'n': 0, 'pnl': 0.0, 'wr': 0.0, 'mdd': 0.0, 'calmar': 0.0}
    wins = [p for p in pnls if p > 0]
    pnl = sum(pnls)
    cum = 0.0; peak = 0.0; mdd = 0.0
    for p in pnls:
        cum += p
        if cum > peak: peak = cum
        dd = peak - cum
        if dd > mdd: mdd = dd
    return {
        'n': len(pnls), 'pnl': pnl,
        'wr': len(wins) / len(pnls) * 100,
        'mdd': mdd,
        'calmar': pnl / mdd if mdd > 0 else (float('inf') if pnl > 0 else 0.0),
    }


def run_pipeline(df_features: pd.DataFrame,
                 bars_cache: dict,
                 trigger_r: float, lock_r: float):
    """Run the full ORB pipeline for a given (trigger_r, lock_r) pair.

    Returns DataFrame of selected/sized trades with date column for splitting.
    """
    df = df_features.copy()

    # 1. Re-simulate exits with the variant's lock pair
    new_pnls = []; new_pnl_pcts = []; new_reasons = []
    for _, row in df.reset_index(drop=True).iterrows():
        key = (row['symbol'], row['date'].strftime('%Y-%m-%d'))
        bars = bars_cache.get(key)
        if bars is None or bars.empty:
            new_pnls.append(row['pnl']); new_pnl_pcts.append(row['pnl_pct'])
            new_reasons.append(row['exit_reason']); continue
        open_ts = _session_open_timestamp(bars)
        if open_ts is None:
            new_pnls.append(row['pnl']); new_pnl_pcts.append(row['pnl_pct'])
            new_reasons.append(row['exit_reason']); continue
        range_end = open_ts + timedelta(minutes=5)
        range_bars = bars[(bars['timestamp'] >= open_ts) & (bars['timestamp'] < range_end)]
        if len(range_bars) < 5:
            new_pnls.append(row['pnl']); new_pnl_pcts.append(row['pnl_pct'])
            new_reasons.append(row['exit_reason']); continue
        rh = float(range_bars['high'].max()); rl = float(range_bars['low'].min())
        search = bars[(bars['timestamp'] >= range_end) &
                      (bars['timestamp'] < range_end + timedelta(minutes=60))]
        entry_ts = None
        for _, b in search.iterrows():
            if float(b['high']) > rh:
                entry_ts = b['timestamp']; break
        if entry_ts is None:
            new_pnls.append(row['pnl']); new_pnl_pcts.append(row['pnl_pct'])
            new_reasons.append(row['exit_reason']); continue
        entry_p = float(row['entry_price'])
        exit_p, reason = simulate_lock(bars, entry_p, rh, rl, entry_ts,
                                       trigger_r, lock_r)
        shares = max(1, int(OLD_POS / entry_p))
        new_pnls.append((exit_p - entry_p) * shares)
        new_pnl_pcts.append((exit_p - entry_p) / entry_p * 100)
        new_reasons.append(reason)

    df = df.reset_index(drop=True)
    df['pnl'] = new_pnls
    df['pnl_pct'] = new_pnl_pcts
    df['exit_reason'] = new_reasons

    # 2. Risk-parity sizing
    per_pos_cap = ACCOUNT / N
    stop = df['range_size_pct'].clip(lower=MIN_STOP_PCT)
    uncap = RISK / (stop / 100.0)
    df['_rp_position'] = uncap.clip(upper=per_pos_cap)
    df['_rp_pnl'] = df['pnl'] * df['_rp_position'] / OLD_POS

    # 3. Fit z-params/cutoffs/mults on TRAIN ONLY (no leakage into VAL/HOLD)
    train = df[(df['date'] >= TRAIN_START) & (df['date'] <= TRAIN_END)]
    params = fit_z_params(train, FILTER_FEATURES)
    df['_composite'] = composite_score(df, params)
    train = df[(df['date'] >= TRAIN_START) & (df['date'] <= TRAIN_END)]
    train_k = train[train['_composite'] >= FILTER_THRESHOLD].copy()
    cutoffs = fit_quintile_cutoffs(train_k['_composite'])
    train_k['_quintile'] = assign_quintile(train_k['_composite'], cutoffs)
    avg = float(train_k['_rp_pnl'].mean()) if len(train_k) else 0.0
    mults = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        sub = train_k[train_k['_quintile'] == q]
        m = float(sub['_rp_pnl'].mean()) / avg if (len(sub) and avg) else 1.0
        mults[q] = max(ADAPTIVE_MULT_MIN, min(Q_CAPS[q], m))

    # 4. Apply pipeline to ALL data (TRAIN + VAL + HOLD)
    kept = df[df['_composite'] >= FILTER_THRESHOLD].copy()
    kept['_quintile'] = assign_quintile(kept['_composite'], cutoffs)
    # Q1 filter ON (matches production, ORB_SKIP_Q1 default)
    kept = kept[kept['_quintile'] != 'Q1'].copy()

    # Top-K + dedup per day
    sel_rows = []
    for day, dg in kept.groupby('date'):
        d = dg.copy()
        d['_q_rank'] = d['_quintile'].map(Q_ORDER)
        d = d.sort_values(['_q_rank', '_composite'], ascending=[True, False])
        seen_fam = set(); seen_sup = set()
        kept_today = []
        for _, r in d.iterrows():
            fam = symbol_family(r['symbol']); sup = symbol_super_group(r['symbol'])
            if fam and fam in seen_fam: continue
            if sup and sup in seen_sup: continue
            if fam: seen_fam.add(fam)
            if sup: seen_sup.add(sup)
            kept_today.append(r)
            if len(kept_today) >= N: break
        sel_rows.extend(kept_today)
    sel = pd.DataFrame(sel_rows)
    if len(sel) == 0:
        return sel
    sel['_sized_pnl'] = sel.apply(lambda r: r['_rp_pnl'] * mults[r['_quintile']], axis=1)
    sel['date'] = pd.to_datetime(sel['date'])
    return sel


def split_pnls(sel: pd.DataFrame) -> Dict[str, List[float]]:
    """Bucket sized_pnl by TRAIN/VAL/HOLDOUT date splits."""
    if len(sel) == 0:
        return {'TRAIN': [], 'VAL': [], 'HOLDOUT': []}
    sel = sel.copy()
    sel['date'] = pd.to_datetime(sel['date'])
    splits = {
        'TRAIN':   (pd.Timestamp(TRAIN_START), pd.Timestamp(TRAIN_END)),
        'VAL':     (pd.Timestamp(VAL_START),   pd.Timestamp(VAL_END)),
        'HOLDOUT': (pd.Timestamp(HOLD_START),  pd.Timestamp(HOLD_END)),
    }
    out = {}
    for name, (lo, hi) in splits.items():
        s = sel[(sel['date'] >= lo) & (sel['date'] <= hi)]
        out[name] = list(s['_sized_pnl'].values)
    return out


def main():
    csv_path = sorted(
        p for p in glob.glob('analysis_results/orb_features_*.csv')
        if 'corrmatrix' not in p
    )[-1]
    print(f'Reading features cache: {csv_path}')
    df = pd.read_csv(csv_path)
    needed = [f for f, _ in FILTER_FEATURES]
    df = df.dropna(subset=needed + ['pnl', 'date', 'pnl_pct', 'range_size_pct', 'entry_price'])
    df['date'] = pd.to_datetime(df['date'])
    print(f'  {len(df)} candidate trades')

    print('Loading bars from cache...')
    pairs = list(df[['symbol', 'date']].drop_duplicates().apply(
        lambda r: (r['symbol'], r['date'].strftime('%Y-%m-%d')), axis=1))
    db = Database(db_path='data/cache.db')
    raw_bars = db.get_intraday_bars_bulk(pairs)
    db.close()
    bars_cache = {k: _bars_to_df(v) for k, v in raw_bars.items()}
    print(f'  bars loaded for {sum(1 for v in bars_cache.values() if v is not None and not v.empty)} (sym,date)s')

    # Sanity check: current 1.5/1.0 should reproduce documented ~$342K HOLDOUT
    print('\n=== SANITY CHECK: current 1.5R / 1.0R baseline ===')
    sel = run_pipeline(df, bars_cache, trigger_r=1.5, lock_r=1.0)
    bk = split_pnls(sel)
    aT = aggregate(bk['TRAIN']); aV = aggregate(bk['VAL']); aH = aggregate(bk['HOLDOUT'])
    total_n = aT['n'] + aV['n'] + aH['n']
    total_pnl = aT['pnl'] + aV['pnl'] + aH['pnl']
    print(f'  TRAIN  : n={aT["n"]:>4}  P&L=${aT["pnl"]:>+10,.0f}  WR={aT["wr"]:>5.1f}%  Calmar={aT["calmar"]:>6.2f}')
    print(f'  VAL    : n={aV["n"]:>4}  P&L=${aV["pnl"]:>+10,.0f}  WR={aV["wr"]:>5.1f}%  Calmar={aV["calmar"]:>6.2f}')
    print(f'  HOLDOUT: n={aH["n"]:>4}  P&L=${aH["pnl"]:>+10,.0f}  WR={aH["wr"]:>5.1f}%  Calmar={aH["calmar"]:>6.2f}')
    print(f'  TOTAL  : n={total_n:>4}  P&L=${total_pnl:>+10,.0f}')
    print(f'  Documented HOLDOUT: ~+$342K full-timeline; verify HOLDOUT matches "real" Q1+ 2026 number')

    # ----- Grid sweep -----
    print('\n' + '=' * 110)
    print('R-GRID SWEEP — printing each variant as it completes')
    print('=' * 110)
    print(f'{"trig_R":>7} {"lock_R":>7}  '
          f'{"TRAIN_n":>7} {"TRAIN_PnL":>11} {"VAL_n":>5} {"VAL_PnL":>11} {"VAL_Cal":>8}  '
          f'{"HOLD_n":>6} {"HOLD_PnL":>11} {"HOLD_Cal":>8}')
    print('-' * 110)

    results = []
    grid = [(t, l) for t in TRIGGER_GRID for l in LOCK_GRID if l < t]
    for i, (trig, lock) in enumerate(grid, 1):
        sel = run_pipeline(df, bars_cache, trigger_r=trig, lock_r=lock)
        bk = split_pnls(sel)
        aT = aggregate(bk['TRAIN'])
        aV = aggregate(bk['VAL'])
        aH = aggregate(bk['HOLDOUT'])
        marker = ''
        if abs(trig - 1.5) < 0.01 and abs(lock - 1.0) < 0.01:
            marker = ' ← PROD'
        print(f'{trig:>7.2f} {lock:>7.2f}  '
              f'{aT["n"]:>7} ${aT["pnl"]:>+10,.0f} '
              f'{aV["n"]:>5} ${aV["pnl"]:>+10,.0f} {aV["calmar"]:>8.2f}  '
              f'{aH["n"]:>6} ${aH["pnl"]:>+10,.0f} {aH["calmar"]:>8.2f}{marker}',
              flush=True)
        results.append({'trigger_r': trig, 'lock_r': lock,
                        'train': aT, 'val': aV, 'holdout': aH})
        # Progress
        if i % 10 == 0:
            print(f'  ... {i}/{len(grid)} configs done', flush=True)

    # ----- Walk-forward winner selection -----
    # Pick best on VAL P&L (NOT HOLDOUT — that would peek)
    print('\n' + '=' * 110)
    print('WALK-FORWARD WINNER (best on VAL only — HOLDOUT not consulted for selection)')
    print('=' * 110)
    by_val = sorted(results, key=lambda r: r['val']['pnl'], reverse=True)
    print('\nTop 10 by VAL P&L:')
    print(f'{"rank":>4} {"trig":>5} {"lock":>5}  '
          f'{"TRAIN_PnL":>11}  {"VAL_PnL":>11} {"VAL_Cal":>8}  '
          f'{"HOLD_PnL":>11} {"HOLD_Cal":>8}')
    for i, r in enumerate(by_val[:10], 1):
        marker = '★' if abs(r['trigger_r'] - 1.5) < 0.01 and abs(r['lock_r'] - 1.0) < 0.01 else ' '
        print(f'{i:>4} {r["trigger_r"]:>5.2f} {r["lock_r"]:>5.2f}  '
              f'${r["train"]["pnl"]:>+10,.0f}  '
              f'${r["val"]["pnl"]:>+10,.0f} {r["val"]["calmar"]:>8.2f}  '
              f'${r["holdout"]["pnl"]:>+10,.0f} {r["holdout"]["calmar"]:>8.2f}  {marker}')

    by_val_calmar = sorted(results, key=lambda r: r['val']['calmar'], reverse=True)
    print('\nTop 10 by VAL Calmar:')
    print(f'{"rank":>4} {"trig":>5} {"lock":>5}  '
          f'{"VAL_PnL":>11} {"VAL_Cal":>8}  '
          f'{"HOLD_PnL":>11} {"HOLD_Cal":>8}')
    for i, r in enumerate(by_val_calmar[:10], 1):
        print(f'{i:>4} {r["trigger_r"]:>5.2f} {r["lock_r"]:>5.2f}  '
              f'${r["val"]["pnl"]:>+10,.0f} {r["val"]["calmar"]:>8.2f}  '
              f'${r["holdout"]["pnl"]:>+10,.0f} {r["holdout"]["calmar"]:>8.2f}')

    # ----- Final HOLDOUT report — only the VAL winner -----
    winner = by_val[0]
    prod = next(r for r in results if abs(r['trigger_r'] - 1.5) < 0.01 and abs(r['lock_r'] - 1.0) < 0.01)
    print('\n' + '=' * 110)
    print('FINAL HOLDOUT REPORT (VAL winner vs current PROD baseline)')
    print('=' * 110)
    print(f'{"variant":<25} {"VAL_PnL":>11} {"HOLD_PnL":>11} {"HOLD_Cal":>8} {"HOLD_WR":>8}')
    win_label = f'VAL-WINNER ({winner["trigger_r"]:.2f}R / {winner["lock_r"]:.2f}R)'
    print(f'{"PROD (1.5R / 1.0R)":<28} ${prod["val"]["pnl"]:>+10,.0f} ${prod["holdout"]["pnl"]:>+10,.0f} {prod["holdout"]["calmar"]:>8.2f} {prod["holdout"]["wr"]:>7.1f}%')
    print(f'{win_label:<28} ${winner["val"]["pnl"]:>+10,.0f} ${winner["holdout"]["pnl"]:>+10,.0f} {winner["holdout"]["calmar"]:>8.2f} {winner["holdout"]["wr"]:>7.1f}%')
    delta = winner['holdout']['pnl'] - prod['holdout']['pnl']
    pct_str = f'{delta/prod["holdout"]["pnl"]*100:+.1f}%' if prod['holdout']['pnl'] != 0 else 'n/a'
    print(f'\nDelta vs PROD on HOLDOUT: ${delta:>+,.0f}  ({pct_str})')


if __name__ == '__main__':
    main()
