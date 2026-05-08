"""Plan-R vs fill-R BT with realistic LIVE slippage (2026-05-08).

The critical test: for each cached entry, simulate what would happen if the
buy-stop fill slipped above the planned breakout level by a LIVE-realistic
amount (TTGT slipped 1.4%, IREZ slipped 1.6% today). Then compare:

- fill-R:    R = (real_fill - planned_stop)        ← slippage INFLATES R,
                                                     pushing activation
                                                     gate further from entry
- planned-R: R = (planned_breakout - planned_stop) ← R locked to setup,
                                                     activation gate stays
                                                     at the level the
                                                     pattern intended

For each slippage level (0.5%, 1.0%, 1.5%, 2.0%), the ProdBaseline trail
config is run twice — once with use_planned_r=False, once with True — and
their HOLDOUT P&L is compared.

The cache's `entry_price` is treated as the planned breakout level (BT
cache built with 0.1% slippage ≈ negligible inflation; the cache entry is
basically the planned price). Entry override = cache_entry × (1 + slip%)
simulates the LIVE buy-stop slipping above breakout.
"""
import sys
sys.path.insert(0, '.')

from study_bull_flag_exits import (
    load_cache, fetch_1min_bars, find_entry_bar_idx, make_plan,
    aggregate, SPLITS, CACHE_CSV, CACHE_DB,
)
from backtest import TradeSimulator
import sqlite3

# Slippage levels to test (% of planned entry that the fill slips above)
SLIPPAGE_LEVELS = [0.001, 0.005, 0.010, 0.015, 0.020, 0.025]


def run_one(sim, entries, bar_cache, slip_pct):
    """Run simulator with realistic-slippage entry override.

    For each entry, the simulator's `actual_entry` is `cache_entry × (1+slip_pct)`
    — simulating LIVE buy-stop slippage. The plan's `entry_price` stays at
    cache_entry (the planned breakout level). The simulator's planned-R logic
    uses plan.entry_price as the R baseline; fill-R uses actual_entry. That's
    where the difference shows up.
    """
    pnls = []
    for e in entries:
        bars = bar_cache.get((e.symbol, e.trade_date.isoformat()))
        if bars is None:
            continue
        idx = find_entry_bar_idx(bars, e.entry_time_et)
        if idx is None or idx >= len(bars) - 1:
            continue
        plan = make_plan(e)  # plan.entry_price = cache_entry (treated as planned)
        real_fill = e.entry_price * (1.0 + slip_pct)
        # Sanity: if the simulated real-fill is below stop, this is degenerate.
        if real_fill <= plan.stop_loss_price:
            continue
        trade = sim.simulate(plan, bars, entry_bar_idx=idx, entry_price_override=real_fill)
        pnls.append(trade.pnl if trade.pnl is not None else 0.0)
    return pnls


def make_simulator(use_planned_r: bool, trail_r=1.0, activate_at_r=1.5):
    """ProdBaseline-style simulator with toggleable use_planned_r."""
    return TradeSimulator(
        force_close_time_et=(15, 45),
        exit_slippage_pct=0.001,
        trailing_stop_r=trail_r, trailing_activate_at_r=activate_at_r,
        vol_confirmed_trail_enabled=True, vol_confirmed_trail_min_ratio=1.0,
        exhaustion_exit_enabled=True,
        exhaustion_min_profit_r=3.0, exhaustion_partial_fraction=0.5,
        exhaustion_tighter_trail_r=0.5,
        use_planned_r=use_planned_r,
    )


def main():
    print('Loading cache and bars...')
    entries = load_cache(CACHE_CSV)
    by_split = {s: [] for s in SPLITS}
    for e in entries:
        for split, (lo, hi) in SPLITS.items():
            if lo <= e.trade_date <= hi:
                by_split[split].append(e)
                break

    conn = sqlite3.connect(CACHE_DB)
    bar_cache = {}
    for e in entries:
        key = (e.symbol, e.trade_date.isoformat())
        if key in bar_cache:
            continue
        bars = fetch_1min_bars(conn, e.symbol, e.trade_date)
        bar_cache[key] = bars if (bars is not None and len(bars) >= 5) else None
    print(f'  TRAIN: {len(by_split["TRAIN"])}  VAL: {len(by_split["VAL"])}  HOLDOUT: {len(by_split["HOLDOUT"])}')
    print(f'  bars cached for {sum(1 for v in bar_cache.values() if v is not None)} (sym,date)s\n')

    # ----- Slippage sweep, ProdBaseline trail (1.5R activation, 1.0R trail) -----
    print('=' * 110)
    print('PROD TRAIL (activate_at_r=1.5, trail_r=1.0) — slippage sweep')
    print('=' * 110)
    print(f'{"slip%":>6}  | {"R-mode":>8}  '
          f'{"TRAIN_PnL":>11} {"TRAIN_Cal":>9}  '
          f'{"VAL_PnL":>11} {"VAL_Cal":>9}  '
          f'{"HOLD_PnL":>11} {"HOLD_Cal":>9} {"HOLD_WR":>7}')
    print('-' * 110)
    results = []
    for slip in SLIPPAGE_LEVELS:
        for use_planned_r in [False, True]:
            sim = make_simulator(use_planned_r=use_planned_r)
            agg_t = aggregate(run_one(sim, by_split['TRAIN'], bar_cache, slip))
            agg_v = aggregate(run_one(sim, by_split['VAL'], bar_cache, slip))
            agg_h = aggregate(run_one(sim, by_split['HOLDOUT'], bar_cache, slip))
            mode = 'plan-R' if use_planned_r else 'fill-R'
            print(f'{slip*100:>5.1f}%  | {mode:>8}  '
                  f'${agg_t["pnl"]:>+10,.0f} {agg_t["calmar"]:>9.2f}  '
                  f'${agg_v["pnl"]:>+10,.0f} {agg_v["calmar"]:>9.2f}  '
                  f'${agg_h["pnl"]:>+10,.0f} {agg_h["calmar"]:>9.2f} {agg_h["wr"]:>6.1f}%',
                  flush=True)
            results.append({
                'slip': slip, 'mode': mode, 'use_planned_r': use_planned_r,
                'train': agg_t, 'val': agg_v, 'holdout': agg_h,
            })
        print()

    # ----- Per-slippage delta table -----
    print('\n' + '=' * 110)
    print('plan-R MINUS fill-R per slippage level (positive = plan-R wins)')
    print('=' * 110)
    print(f'{"slip%":>6}  {"TRAIN_Δ":>12} {"VAL_Δ":>12} {"HOLDOUT_Δ":>12}  {"HOLD_winner"}')
    print('-' * 110)
    for slip in SLIPPAGE_LEVELS:
        fill = next(r for r in results if r['slip'] == slip and not r['use_planned_r'])
        plan = next(r for r in results if r['slip'] == slip and r['use_planned_r'])
        d_t = plan['train']['pnl'] - fill['train']['pnl']
        d_v = plan['val']['pnl'] - fill['val']['pnl']
        d_h = plan['holdout']['pnl'] - fill['holdout']['pnl']
        winner = 'plan-R' if d_h > 0 else ('fill-R' if d_h < 0 else 'tie')
        print(f'{slip*100:>5.1f}%  ${d_t:>+10,.0f}  ${d_v:>+10,.0f}  ${d_h:>+10,.0f}    {winner}')

    # ----- Best overall config across the grid -----
    print('\n' + '=' * 110)
    print('TOP HOLDOUT P&L CONFIGS')
    print('=' * 110)
    print(f'{"slip%":>6}  {"R-mode":>8}  {"HOLD_PnL":>12} {"HOLD_Cal":>9}')
    for r in sorted(results, key=lambda x: x['holdout']['pnl'], reverse=True)[:10]:
        print(f'{r["slip"]*100:>5.1f}%  {r["mode"]:>8}  ${r["holdout"]["pnl"]:>+11,.0f} {r["holdout"]["calmar"]:>9.2f}')


if __name__ == '__main__':
    main()
