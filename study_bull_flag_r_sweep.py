"""Bull flag R-location sweep (2026-05-08).

Systematic grid search over the R-thresholds that govern profit locking:
- activate_at_r: where the trail switches from hard-stop to dynamic
- trail_r:       distance of trail behind high (in R units)
- arm_r/lock_r:  static-lock pair (touch +arm_r → lock at +lock_r forever)

Two families:
  TRAIL family — the production mechanism. Vary activate_at_r and trail_r.
  LOCK family  — ORB-style static lock. Vary arm_r and lock_r (lock_r < arm_r).

Walk-forward splits: TRAIN/VAL/HOLDOUT (same as study_bull_flag_exits.py).

Goal: find the (activation, distance) pair that maximizes HOLDOUT P&L
without overfitting to TRAIN.

Bug-fixed P&L computation: uses trade.pnl directly (already includes any
partial_pnl per backtest.py:739) — earlier study double-counted partials.
"""
import sys
sys.path.insert(0, '.')

from study_bull_flag_exits import (
    load_cache, fetch_1min_bars, find_entry_bar_idx, make_plan,
    aggregate, SPLITS, CACHE_CSV, CACHE_DB,
)
from backtest import TradeSimulator
import sqlite3
from typing import Dict, List, Tuple

# Grid: vary activate_at_r and trail_r.
TRAIL_GRID: List[Tuple[float, float]] = [
    (a, t)
    for a in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    for t in [0.5, 0.75, 1.0, 1.25, 1.5]
    if t <= a  # trail distance must not exceed activation gap
]

# Static lock grid: arm_r ≥ lock_r (else lock is useless).
LOCK_GRID: List[Tuple[float, float]] = [
    (arm, lock)
    for arm in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    for lock in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    if lock < arm  # lock below arm, else no protection
]


def run_one(sim, entries, bar_cache):
    """Run the simulator over a list of cache entries; return list of pnls."""
    pnls = []
    for e in entries:
        bars = bar_cache.get((e.symbol, e.trade_date.isoformat()))
        if bars is None:
            continue
        idx = find_entry_bar_idx(bars, e.entry_time_et)
        if idx is None or idx >= len(bars) - 1:
            continue
        plan = make_plan(e)
        trade = sim.simulate(plan, bars, entry_bar_idx=idx, entry_price_override=e.entry_price)
        pnls.append(trade.pnl if trade.pnl is not None else 0.0)
    return pnls


def main():
    print('Loading cache...')
    entries = load_cache(CACHE_CSV)
    by_split = {s: [] for s in SPLITS}
    for e in entries:
        for split, (lo, hi) in SPLITS.items():
            if lo <= e.trade_date <= hi:
                by_split[split].append(e)
                break
    print(f'  TRAIN: {len(by_split["TRAIN"])}  VAL: {len(by_split["VAL"])}  HOLDOUT: {len(by_split["HOLDOUT"])}')

    print('Pre-fetching bars...')
    conn = sqlite3.connect(CACHE_DB)
    bar_cache = {}
    for e in entries:
        key = (e.symbol, e.trade_date.isoformat())
        if key in bar_cache:
            continue
        bars = fetch_1min_bars(conn, e.symbol, e.trade_date)
        bar_cache[key] = bars if (bars is not None and len(bars) >= 5) else None
    print(f'  cached {sum(1 for v in bar_cache.values() if v is not None)} (sym,date)s\n')

    # ----- TRAIL family sweep -----
    print('=' * 110)
    print(f'TRAIL FAMILY — sweep (activate_at_r, trail_r), {len(TRAIL_GRID)} configs')
    print('=' * 110)
    print(f'{"act_R":>6} {"trail_R":>8}  '
          f'{"TRAIN_PnL":>10} {"TRAIN_Cal":>9}  '
          f'{"VAL_PnL":>10} {"VAL_Cal":>9}  '
          f'{"HOLD_PnL":>10} {"HOLD_Cal":>9} {"HOLD_WR":>7}')
    print('-' * 110)
    trail_results = []
    for activate_r, trail_r in TRAIL_GRID:
        sim = TradeSimulator(
            force_close_time_et=(15, 45),
            exit_slippage_pct=0.001,
            trailing_stop_r=trail_r, trailing_activate_at_r=activate_r,
            vol_confirmed_trail_enabled=True, vol_confirmed_trail_min_ratio=1.0,
            exhaustion_exit_enabled=True,
            exhaustion_min_profit_r=3.0, exhaustion_partial_fraction=0.5,
            exhaustion_tighter_trail_r=0.5,
        )
        agg_t = aggregate(run_one(sim, by_split['TRAIN'], bar_cache))
        agg_v = aggregate(run_one(sim, by_split['VAL'], bar_cache))
        agg_h = aggregate(run_one(sim, by_split['HOLDOUT'], bar_cache))
        marker = ''
        # Mark current production config (1.5/1.0)
        if abs(activate_r - 1.5) < 0.01 and abs(trail_r - 1.0) < 0.01:
            marker = '  ← PROD'
        print(f'{activate_r:>6.2f} {trail_r:>8.2f}  '
              f'${agg_t["pnl"]:>+9.0f} {agg_t["calmar"]:>9.2f}  '
              f'${agg_v["pnl"]:>+9.0f} {agg_v["calmar"]:>9.2f}  '
              f'${agg_h["pnl"]:>+9.0f} {agg_h["calmar"]:>9.2f} {agg_h["wr"]:>6.1f}%{marker}',
              flush=True)
        trail_results.append({
            'activate_r': activate_r, 'trail_r': trail_r,
            'train': agg_t, 'val': agg_v, 'holdout': agg_h,
        })

    # ----- LOCK family sweep -----
    print('\n' + '=' * 110)
    print(f'LOCK FAMILY — sweep (arm_r, lock_r), {len(LOCK_GRID)} configs')
    print('=' * 110)
    print(f'{"arm_R":>6} {"lock_R":>7}  '
          f'{"TRAIN_PnL":>10} {"TRAIN_Cal":>9}  '
          f'{"VAL_PnL":>10} {"VAL_Cal":>9}  '
          f'{"HOLD_PnL":>10} {"HOLD_Cal":>9} {"HOLD_WR":>7}')
    print('-' * 110)
    lock_results = []
    for arm_r, lock_r in LOCK_GRID:
        sim = TradeSimulator(
            force_close_time_et=(15, 45),
            exit_slippage_pct=0.001,
            static_lock_arm_r=arm_r, static_lock_at_r=lock_r,
            trailing_stop_r=0.0,  # no trail when using static lock
        )
        agg_t = aggregate(run_one(sim, by_split['TRAIN'], bar_cache))
        agg_v = aggregate(run_one(sim, by_split['VAL'], bar_cache))
        agg_h = aggregate(run_one(sim, by_split['HOLDOUT'], bar_cache))
        print(f'{arm_r:>6.2f} {lock_r:>7.2f}  '
              f'${agg_t["pnl"]:>+9.0f} {agg_t["calmar"]:>9.2f}  '
              f'${agg_v["pnl"]:>+9.0f} {agg_v["calmar"]:>9.2f}  '
              f'${agg_h["pnl"]:>+9.0f} {agg_h["calmar"]:>9.2f} {agg_h["wr"]:>6.1f}%',
              flush=True)
        lock_results.append({
            'arm_r': arm_r, 'lock_r': lock_r,
            'train': agg_t, 'val': agg_v, 'holdout': agg_h,
        })

    # ----- Summaries -----
    def top_n(results, key, n=10, reverse=True):
        return sorted(results, key=lambda r: r['holdout'][key], reverse=reverse)[:n]

    print('\n' + '=' * 110)
    print('TOP 10 TRAIL configs by HOLDOUT P&L')
    print('=' * 110)
    print(f'{"rank":>4} {"act_R":>6} {"trail_R":>8}  '
          f'{"TRAIN_PnL":>10} {"VAL_PnL":>10} {"HOLD_PnL":>10} {"HOLD_Cal":>9} {"HOLD_WR":>7}')
    for i, r in enumerate(top_n(trail_results, 'pnl'), 1):
        print(f'{i:>4} {r["activate_r"]:>6.2f} {r["trail_r"]:>8.2f}  '
              f'${r["train"]["pnl"]:>+9.0f} ${r["val"]["pnl"]:>+9.0f} ${r["holdout"]["pnl"]:>+9.0f} '
              f'{r["holdout"]["calmar"]:>9.2f} {r["holdout"]["wr"]:>6.1f}%')

    print('\nTOP 10 TRAIL configs by HOLDOUT Calmar')
    print('-' * 110)
    for i, r in enumerate(top_n(trail_results, 'calmar'), 1):
        print(f'{i:>4} {r["activate_r"]:>6.2f} {r["trail_r"]:>8.2f}  '
              f'${r["train"]["pnl"]:>+9.0f} ${r["val"]["pnl"]:>+9.0f} ${r["holdout"]["pnl"]:>+9.0f} '
              f'{r["holdout"]["calmar"]:>9.2f} {r["holdout"]["wr"]:>6.1f}%')

    print('\n' + '=' * 110)
    print('TOP 10 LOCK configs by HOLDOUT P&L')
    print('=' * 110)
    print(f'{"rank":>4} {"arm_R":>6} {"lock_R":>7}  '
          f'{"TRAIN_PnL":>10} {"VAL_PnL":>10} {"HOLD_PnL":>10} {"HOLD_Cal":>9} {"HOLD_WR":>7}')
    for i, r in enumerate(top_n(lock_results, 'pnl'), 1):
        print(f'{i:>4} {r["arm_r"]:>6.2f} {r["lock_r"]:>7.2f}  '
              f'${r["train"]["pnl"]:>+9.0f} ${r["val"]["pnl"]:>+9.0f} ${r["holdout"]["pnl"]:>+9.0f} '
              f'{r["holdout"]["calmar"]:>9.2f} {r["holdout"]["wr"]:>6.1f}%')

    print('\nTOP 10 LOCK configs by HOLDOUT Calmar')
    print('-' * 110)
    for i, r in enumerate(top_n(lock_results, 'calmar'), 1):
        print(f'{i:>4} {r["arm_r"]:>6.2f} {r["lock_r"]:>7.2f}  '
              f'${r["train"]["pnl"]:>+9.0f} ${r["val"]["pnl"]:>+9.0f} ${r["holdout"]["pnl"]:>+9.0f} '
              f'{r["holdout"]["calmar"]:>9.2f} {r["holdout"]["wr"]:>6.1f}%')

    # Walk-forward consistency: best-on-TRAIN, what does it do OOS?
    print('\n' + '=' * 110)
    print('WALK-FORWARD CHECK — best-on-TRAIN, applied to VAL + HOLDOUT')
    print('=' * 110)
    best_t_trail = max(trail_results, key=lambda r: r['train']['pnl'])
    print(f'TRAIL best-on-TRAIN: act_R={best_t_trail["activate_r"]}, trail_R={best_t_trail["trail_r"]}')
    print(f'  TRAIN  P&L=${best_t_trail["train"]["pnl"]:>+9.0f}  Calmar={best_t_trail["train"]["calmar"]:>5.2f}')
    print(f'  VAL    P&L=${best_t_trail["val"]["pnl"]:>+9.0f}  Calmar={best_t_trail["val"]["calmar"]:>5.2f}')
    print(f'  HOLDOUT P&L=${best_t_trail["holdout"]["pnl"]:>+9.0f}  Calmar={best_t_trail["holdout"]["calmar"]:>5.2f}')

    best_v_trail = max(trail_results, key=lambda r: r['val']['pnl'])
    print(f'\nTRAIL best-on-VAL: act_R={best_v_trail["activate_r"]}, trail_R={best_v_trail["trail_r"]}')
    print(f'  HOLDOUT P&L=${best_v_trail["holdout"]["pnl"]:>+9.0f}  Calmar={best_v_trail["holdout"]["calmar"]:>5.2f}')

    best_t_lock = max(lock_results, key=lambda r: r['train']['pnl'])
    print(f'\nLOCK best-on-TRAIN: arm_R={best_t_lock["arm_r"]}, lock_R={best_t_lock["lock_r"]}')
    print(f'  HOLDOUT P&L=${best_t_lock["holdout"]["pnl"]:>+9.0f}  Calmar={best_t_lock["holdout"]["calmar"]:>5.2f}')

    best_v_lock = max(lock_results, key=lambda r: r['val']['pnl'])
    print(f'\nLOCK best-on-VAL: arm_R={best_v_lock["arm_r"]}, lock_R={best_v_lock["lock_r"]}')
    print(f'  HOLDOUT P&L=${best_v_lock["holdout"]["pnl"]:>+9.0f}  Calmar={best_v_lock["holdout"]["calmar"]:>5.2f}')


if __name__ == '__main__':
    main()
