"""ORB two-stage exit BT: BE-lock + static lock (2026-05-08).

Question (user's "money on the table" critique): the BT cohort shows 33%
of trades touch +0.5R but never reach +1.5R. Under PROD they exit at
hard stop or EOD. A two-stage exit would lock breakeven at +0.5R touch
WITHOUT capping the runners (full +1R lock still fires at +1.5R touch).

The simulator must do BOTH layers in order:
  1. when bar.high crosses breakeven_arm_at_r × range_size → stop = entry
  2. when bar.high crosses arm_at_r × range_size → stop = entry + lock_r × R
  3. when bar.low ≤ stop_price → exit

Walk-forward TRAIN/VAL/HOLDOUT, same as study_orb_r_grid.py.

Bug-watch (lessons from this morning):
- Don't double-count partial pnl
- Stop ratchet is one-way (only moves UP — max(prev, candidate))
- BE_arm and arm don't both fire in same bar in conflicting ways
- Slippage applied once at exit only
"""
import glob
import os
import sys
from datetime import timedelta
from typing import Dict, List

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from persistence.database import Database
from study_orb import _bars_to_df
from study_orb_filter import FILTER_FEATURES, fit_z_params, composite_score
from study_orb_sizing import (
    FILTER_THRESHOLD, fit_quintile_cutoffs, assign_quintile, ADAPTIVE_MULT_MIN,
)
from study_orb_correlation_filter import symbol_family, symbol_super_group


ACCOUNT = 100_000.0
N = 4
RISK = 3000.0
MIN_STOP_PCT = 1.0
OLD_POS = 50_000.0
Q_CAPS = {'Q1': 3.0, 'Q2': 3.0, 'Q3': 3.0, 'Q4': 3.0, 'Q5': 1.5}
Q_ORDER = {'Q4': 0, 'Q5': 1, 'Q3': 2, 'Q2': 3, 'Q1': 4}
EXIT_SLIP_BPS = 10.0

TRAIN_START, TRAIN_END = '2025-01-01', '2025-06-30'
VAL_START,   VAL_END   = '2025-07-01', '2025-12-31'
HOLD_START,  HOLD_END  = '2026-01-01', '2026-04-30'


def _session_open_timestamp(bars):
    if bars.empty:
        return None
    bars_et = bars.copy()
    bars_et['et'] = bars_et['timestamp'].dt.tz_convert('America/New_York')
    morning = bars_et[bars_et['et'].dt.time == pd.Timestamp('09:30').time()]
    if len(morning):
        return morning.iloc[0]['timestamp']
    return None


def simulate_two_stage(bars, entry_price, range_high, range_low, entry_time,
                      breakeven_arm_r: float, arm_r: float, lock_r: float):
    """Two-stage lock simulator.

    Stage 1: peak ≥ breakeven_arm_r × R → stop ratchets to entry (BE).
             (Set breakeven_arm_r=0 to disable Stage 1.)
    Stage 2: peak ≥ arm_r × R → stop ratchets to entry + lock_r × R.
    Both ratchets are one-way (stop only moves UP, never down).

    Exit when bar.low ≤ stop_price OR EOD.
    """
    range_size = range_high - range_low
    be_lvl    = entry_price + breakeven_arm_r * range_size if breakeven_arm_r > 0 else None
    arm_lvl   = entry_price + arm_r * range_size
    lock_stop = entry_price + lock_r * range_size
    stop_price = range_low
    be_armed = False
    full_armed = False

    post = bars[bars['timestamp'] >= entry_time].reset_index(drop=True)
    for _, row in post.iloc[1:].iterrows():
        bar_high = float(row['high']); bar_low = float(row['low'])

        # Stage 1: breakeven arm
        if be_lvl is not None and not be_armed and bar_high >= be_lvl:
            be_armed = True
            stop_price = max(stop_price, entry_price)  # move to BE
        # Stage 2: full arm
        if not full_armed and bar_high >= arm_lvl:
            full_armed = True
            stop_price = max(stop_price, lock_stop)

        # Exit check
        if bar_low <= stop_price:
            if full_armed:
                reason = 'full_lock'
            elif be_armed:
                reason = 'be_lock'
            else:
                reason = 'stop'
            return stop_price * (1 - EXIT_SLIP_BPS/10000), reason

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


def run_pipeline(df_features: pd.DataFrame, bars_cache: dict,
                 be_arm_r: float, arm_r: float, lock_r: float):
    df = df_features.copy()
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
        rb = bars[(bars['timestamp'] >= open_ts) & (bars['timestamp'] < range_end)]
        if len(rb) < 5:
            new_pnls.append(row['pnl']); new_pnl_pcts.append(row['pnl_pct'])
            new_reasons.append(row['exit_reason']); continue
        rh = float(rb['high'].max()); rl = float(rb['low'].min())
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
        exit_p, reason = simulate_two_stage(
            bars, entry_p, rh, rl, entry_ts, be_arm_r, arm_r, lock_r)
        shares = max(1, int(OLD_POS / entry_p))
        new_pnls.append((exit_p - entry_p) * shares)
        new_pnl_pcts.append((exit_p - entry_p) / entry_p * 100)
        new_reasons.append(reason)

    df = df.reset_index(drop=True)
    df['pnl'] = new_pnls
    df['pnl_pct'] = new_pnl_pcts
    df['exit_reason'] = new_reasons

    per_pos_cap = ACCOUNT / N
    stop = df['range_size_pct'].clip(lower=MIN_STOP_PCT)
    uncap = RISK / (stop / 100.0)
    df['_rp_position'] = uncap.clip(upper=per_pos_cap)
    df['_rp_pnl'] = df['pnl'] * df['_rp_position'] / OLD_POS

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

    kept = df[df['_composite'] >= FILTER_THRESHOLD].copy()
    kept['_quintile'] = assign_quintile(kept['_composite'], cutoffs)
    kept = kept[kept['_quintile'] != 'Q1'].copy()

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
    print(f'Reading: {csv_path}')
    df = pd.read_csv(csv_path)
    needed = [f for f, _ in FILTER_FEATURES]
    df = df.dropna(subset=needed + ['pnl', 'date', 'pnl_pct', 'range_size_pct', 'entry_price'])
    df['date'] = pd.to_datetime(df['date'])

    pairs = list(df[['symbol', 'date']].drop_duplicates().apply(
        lambda r: (r['symbol'], r['date'].strftime('%Y-%m-%d')), axis=1))
    db = Database(db_path='data/cache.db')
    raw_bars = db.get_intraday_bars_bulk(pairs)
    db.close()
    bars_cache = {k: _bars_to_df(v) for k, v in raw_bars.items()}
    print(f'  {len(df)} candidates, bars cached for {sum(1 for v in bars_cache.values() if v is not None and not v.empty)} (sym,date)s\n')

    # Sanity: PROD (be_arm=0 → no stage-1) should reproduce 1.5R/1.0R baseline
    print('=== Sanity: be_arm=0, arm=1.5, lock=1.0 (PROD with stage-1 disabled) ===')
    sel = run_pipeline(df, bars_cache, be_arm_r=0.0, arm_r=1.5, lock_r=1.0)
    bk = split_pnls(sel)
    aT = aggregate(bk['TRAIN']); aV = aggregate(bk['VAL']); aH = aggregate(bk['HOLDOUT'])
    print(f'  TRAIN: ${aT["pnl"]:>+10,.0f} | VAL: ${aV["pnl"]:>+10,.0f} | HOLDOUT: ${aH["pnl"]:>+10,.0f}  ({aH["wr"]:.1f}% WR)')
    print('  (should match prior PROD: TRAIN $+153,605 / VAL $+92,484 / HOLDOUT $+93,989)\n')

    # ----- Two-stage grid -----
    # Stage 1 (BE arm) thresholds × (arm, lock) combos
    BE_GRID = [0.25, 0.50, 0.75, 1.00, 1.25]  # 0.0 = no stage-1 = PROD baseline
    LOCK_PAIRS = [
        (1.5, 1.0),    # PROD arm/lock
        (1.5, 0.5),    # PROD arm + tighter lock
        (1.75, 0.50),  # prior R-grid winner
        (1.75, 0.75),
        (2.00, 0.50),
    ]

    print('=' * 110)
    print('TWO-STAGE GRID — stage-1 BE arm × (arm, lock)')
    print('=' * 110)
    print(f'{"BE_arm":>7} {"arm":>5} {"lock":>5}  '
          f'{"TRAIN":>10}  {"VAL":>10} {"VAL_Cal":>8}  {"HOLDOUT":>10} {"HOLD_Cal":>8} {"HOLD_WR":>7}')
    print('-' * 110)
    results = []
    for be in BE_GRID:
        for arm, lock in LOCK_PAIRS:
            if be >= arm:
                continue  # BE must come before full arm
            sel = run_pipeline(df, bars_cache, be_arm_r=be, arm_r=arm, lock_r=lock)
            bk = split_pnls(sel)
            aT = aggregate(bk['TRAIN']); aV = aggregate(bk['VAL']); aH = aggregate(bk['HOLDOUT'])
            print(f'{be:>7.2f} {arm:>5.2f} {lock:>5.2f}  '
                  f'${aT["pnl"]:>+9,.0f}  ${aV["pnl"]:>+9,.0f} {aV["calmar"]:>8.2f}  '
                  f'${aH["pnl"]:>+9,.0f} {aH["calmar"]:>8.2f} {aH["wr"]:>6.1f}%',
                  flush=True)
            results.append({'be_arm_r': be, 'arm_r': arm, 'lock_r': lock,
                           'train': aT, 'val': aV, 'holdout': aH})

    # PROD baseline reference (be=0)
    print()
    sel_prod = run_pipeline(df, bars_cache, be_arm_r=0.0, arm_r=1.5, lock_r=1.0)
    bk_prod = split_pnls(sel_prod)
    aT = aggregate(bk_prod['TRAIN']); aV = aggregate(bk_prod['VAL']); aH = aggregate(bk_prod['HOLDOUT'])
    print(f'{"REF":>7} {"1.50":>5} {"1.00":>5}  '
          f'${aT["pnl"]:>+9,.0f}  ${aV["pnl"]:>+9,.0f} {aV["calmar"]:>8.2f}  '
          f'${aH["pnl"]:>+9,.0f} {aH["calmar"]:>8.2f} {aH["wr"]:>6.1f}%  ← PROD (be=off)')

    # Walk-forward winner
    print('\n' + '=' * 110)
    print('WALK-FORWARD WINNER (best by VAL P&L, HOLDOUT not consulted)')
    print('=' * 110)
    by_val = sorted(results, key=lambda r: r['val']['pnl'], reverse=True)
    print(f'{"rank":>4} {"BE":>5} {"arm":>5} {"lock":>5}  '
          f'{"VAL":>10} {"VAL_Cal":>8}  {"HOLDOUT":>10} {"HOLD_Cal":>8}')
    for i, r in enumerate(by_val[:10], 1):
        print(f'{i:>4} {r["be_arm_r"]:>5.2f} {r["arm_r"]:>5.2f} {r["lock_r"]:>5.2f}  '
              f'${r["val"]["pnl"]:>+9,.0f} {r["val"]["calmar"]:>8.2f}  '
              f'${r["holdout"]["pnl"]:>+9,.0f} {r["holdout"]["calmar"]:>8.2f}')

    print('\nWinner vs PROD on HOLDOUT:')
    win = by_val[0]
    pH = aH  # PROD HOLDOUT from above
    delta = win['holdout']['pnl'] - pH['pnl']
    print(f'  PROD          HOLDOUT ${pH["pnl"]:>+10,.0f}  Calmar {pH["calmar"]:>5.2f}')
    print(f'  VAL-winner    HOLDOUT ${win["holdout"]["pnl"]:>+10,.0f}  Calmar {win["holdout"]["calmar"]:>5.2f}')
    print(f'  Delta:        ${delta:>+10,.0f}  ({delta/pH["pnl"]*100:+.1f}%)')


if __name__ == '__main__':
    main()
