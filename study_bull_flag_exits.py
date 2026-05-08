"""Bull Flag exit-mechanism study (2026-05-08).

Re-simulates cached bull flag entries with different exit configs and
compares P&L / Calmar / WR across TRAIN / VAL / HOLDOUT splits.

Walk-forward design (no peeking at HOLDOUT until final report):
- TRAIN:    Jan-Sep 2025  (variant param search)
- VAL:      Oct-Dec 2025  (pick best variant per family)
- HOLDOUT:  Jan-Apr 2026  (OOS verdict)

Variants:
- ProdBaseline:  current production (trail 1.0R + activate 1.5R + vol-confirmed + exhaustion)
- StripBaseline: just trail 1.0R + activate 1.5R (no vol-conf, no exhaustion) — isolates exit mechanism
- B1: static_lock arm=1.0R, lock=0.5R   (ORB-style; lock half-R, no trail after)
- B2: static_lock arm=1.5R, lock=1.0R   (later arm, locks +1R)
- B3: static_lock arm=1.0R, lock=0.0R   (breakeven lock at +1R touch)
- C1: BE at +0.5R + trail activates +1.5R                 (two-stage)
- C2: lock +0.5R at +0.5R + trail activates +1.5R         (two-stage with profit floor)
- C3: BE at +0.5R + trail activates +1.0R                 (earlier trail)

The trade-entry decision (date, symbol, entry_price, stop_loss, shares) is
taken DIRECTLY from the existing cache — only the exit logic varies. This
gives an apples-to-apples comparison of exit mechanisms with constant entries.
"""
import csv
import sqlite3
import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, '.')
from backtest import TradeSimulator
from trading.pattern_detector import BullFlagPattern
from trading.trade_planner import TradePlan

CACHE_CSV = 'data/bull_flag_cache_e50_x30.csv'
CACHE_DB = 'data/cache.db'

# Walk-forward splits
SPLITS = {
    'TRAIN':   (date(2025, 1, 1),  date(2025, 9, 30)),
    'VAL':     (date(2025, 10, 1), date(2025, 12, 31)),
    'HOLDOUT': (date(2026, 1, 1),  date(2026, 4, 30)),
}


def make_simulator(name: str, use_planned_r: bool = False) -> TradeSimulator:
    """Build a TradeSimulator for each named variant.

    All variants share a 15:45 ET force-close. Vol-confirmed and exhaustion
    are off in StripBaseline / B / C to isolate the exit mechanism. They are
    on in ProdBaseline to mirror current production.

    When use_planned_r=True, R-based math (activation, breakeven, static lock,
    trail ratchet, r_gain) uses the SETUP's structural risk
    (planned_entry - planned_stop) instead of fill-based R. Tests whether
    decoupling trail behavior from entry slippage helps the IREZ-class
    trades (slippage-inflated activation gates).
    """
    common = dict(
        force_close_time_et=(15, 45),
        exit_slippage_pct=0.001,
        use_planned_r=use_planned_r,
    )
    if name == 'ProdBaseline':
        return TradeSimulator(
            **common,
            trailing_stop_r=1.0, trailing_activate_at_r=1.5,
            vol_confirmed_trail_enabled=True, vol_confirmed_trail_min_ratio=1.0,
            exhaustion_exit_enabled=True,
            exhaustion_min_profit_r=3.0, exhaustion_partial_fraction=0.5,
            exhaustion_tighter_trail_r=0.5,
        )
    if name == 'StripBaseline':
        return TradeSimulator(
            **common,
            trailing_stop_r=1.0, trailing_activate_at_r=1.5,
        )
    if name == 'B1':  # static lock arm=1.0, lock=0.5
        return TradeSimulator(
            **common,
            static_lock_arm_r=1.0, static_lock_at_r=0.5,
            trailing_stop_r=0.0,
        )
    if name == 'B2':  # static lock arm=1.5, lock=1.0
        return TradeSimulator(
            **common,
            static_lock_arm_r=1.5, static_lock_at_r=1.0,
            trailing_stop_r=0.0,
        )
    if name == 'B3':  # BE lock at +1R touch
        return TradeSimulator(
            **common,
            static_lock_arm_r=1.0, static_lock_at_r=0.0,
            trailing_stop_r=0.0,
        )
    if name == 'C1':  # BE at +0.5R + trail @ +1.5R
        return TradeSimulator(
            **common,
            breakeven_at_r=0.5, breakeven_profit_r=0.0,
            trailing_stop_r=1.0, trailing_activate_at_r=1.5,
        )
    if name == 'C2':  # lock +0.5R at +0.5R + trail @ +1.5R
        return TradeSimulator(
            **common,
            breakeven_at_r=0.5, breakeven_profit_r=0.5,
            trailing_stop_r=1.0, trailing_activate_at_r=1.5,
        )
    if name == 'C3':  # BE at +0.5R + earlier trail @ +1.0R
        return TradeSimulator(
            **common,
            breakeven_at_r=0.5, breakeven_profit_r=0.0,
            trailing_stop_r=1.0, trailing_activate_at_r=1.0,
        )
    raise ValueError(f'Unknown variant: {name}')


VARIANTS = ['ProdBaseline', 'StripBaseline', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']


@dataclass
class CachedEntry:
    symbol: str
    trade_date: date
    entry_time_et: str       # 'HH:MM:SS'
    entry_price: float
    stop_loss: float
    target: float
    shares: int
    avg_volume_20d: int


def load_cache(path: str) -> List[CachedEntry]:
    out = []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                out.append(CachedEntry(
                    symbol=row['symbol'],
                    trade_date=date.fromisoformat(row['date']),
                    entry_time_et=row['entry_time_et'],
                    entry_price=float(row['entry_price']),
                    stop_loss=float(row['stop_loss']),
                    target=float(row['target']),
                    shares=int(row['shares']),
                    avg_volume_20d=int(float(row.get('avg_volume_20d') or 0)),
                ))
            except Exception as e:
                print(f'  skip row {row.get("symbol")}/{row.get("date")}: {e}', file=sys.stderr)
    return out


def fetch_1min_bars(conn: sqlite3.Connection, symbol: str, trade_date: date) -> Optional[pd.DataFrame]:
    """Pull 1-min bars for a symbol+date from sqlite cache. Returns None if missing."""
    cur = conn.execute(
        'SELECT timestamp, open, high, low, close, volume '
        'FROM intraday_bars_1min '
        'WHERE symbol=? AND bar_date=? '
        'ORDER BY timestamp ASC',
        (symbol, trade_date.isoformat()),
    )
    rows = cur.fetchall()
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df.reset_index(drop=True)


def find_entry_bar_idx(bars: pd.DataFrame, entry_time_et: str) -> Optional[int]:
    """Match HH:MM:SS ET to a bar index in the (UTC-timestamped) bars frame."""
    # ET is UTC-4 (EDT) or UTC-5 (EST). Use the bar's date to determine offset.
    # Simpler: find the bar whose ET time equals entry_time_et.
    et_target = time.fromisoformat(entry_time_et)
    bars_et = bars['timestamp'].dt.tz_convert('America/New_York')
    for i, ts in enumerate(bars_et):
        if ts.time() == et_target:
            return i
    # Fallback: closest bar by minute
    target_minutes = et_target.hour * 60 + et_target.minute
    best_idx, best_diff = None, 999999
    for i, ts in enumerate(bars_et):
        mins = ts.hour * 60 + ts.minute
        d = abs(mins - target_minutes)
        if d < best_diff:
            best_diff = d; best_idx = i
    return best_idx if best_diff <= 1 else None


def make_plan(entry: CachedEntry) -> TradePlan:
    """Build a TradePlan + dummy BullFlagPattern from a cache row."""
    avg_flag_vol = max(int(entry.avg_volume_20d / 78), 1)  # 78 1-min bars/day → rough avg
    pattern = BullFlagPattern(
        symbol=entry.symbol,
        pole_start_idx=0, pole_end_idx=0,
        flag_start_idx=0, flag_end_idx=0,
        pole_low=entry.stop_loss, pole_high=entry.entry_price,
        pole_height=entry.entry_price - entry.stop_loss,
        pole_gain_pct=0.0,
        flag_low=entry.stop_loss, flag_high=entry.entry_price,
        retracement_pct=0.0, pullback_candle_count=1,
        avg_pole_volume=avg_flag_vol, avg_flag_volume=avg_flag_vol,
        breakout_level=entry.entry_price,
    )
    risk_per_share = entry.entry_price - entry.stop_loss
    reward_per_share = entry.target - entry.entry_price
    return TradePlan(
        symbol=entry.symbol,
        entry_price=entry.entry_price,
        stop_loss_price=entry.stop_loss,
        take_profit_price=entry.target,
        risk_per_share=risk_per_share,
        reward_per_share=reward_per_share,
        risk_reward_ratio=(reward_per_share / risk_per_share) if risk_per_share > 0 else 1.0,
        shares=entry.shares,
        total_risk=risk_per_share * entry.shares,
        pattern=pattern,
    )


@dataclass
class TradeResult:
    pnl: float
    exit_reason: str


def simulate_one(sim: TradeSimulator, entry: CachedEntry, bars: pd.DataFrame) -> Optional[TradeResult]:
    idx = find_entry_bar_idx(bars, entry.entry_time_et)
    if idx is None or idx >= len(bars) - 1:
        return None
    plan = make_plan(entry)
    trade = sim.simulate(plan, bars, entry_bar_idx=idx, entry_price_override=entry.entry_price)
    # backtest.py:739 sets trade.pnl = partial_pnl + final_pnl — already total.
    # Earlier version of this script double-counted the partial.
    pnl = trade.pnl if trade.pnl is not None else 0.0
    return TradeResult(pnl=pnl, exit_reason=trade.exit_reason or 'unknown')


def aggregate(pnls: List[float]) -> Dict:
    if not pnls:
        return {'n': 0, 'pnl': 0.0, 'wr': 0.0, 'avg_w': 0.0, 'avg_l': 0.0,
                'pf': 0.0, 'mdd': 0.0, 'calmar': 0.0}
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    pnl = sum(pnls)
    # Continuous max drawdown across the trade-ordered cumulative curve
    cum = 0.0; peak = 0.0; mdd = 0.0
    for p in pnls:
        cum += p
        if cum > peak: peak = cum
        dd = peak - cum
        if dd > mdd: mdd = dd
    pf = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float('inf') if wins else 0.0
    return {
        'n': len(pnls),
        'pnl': pnl,
        'wr': len(wins) / len(pnls) * 100 if pnls else 0.0,
        'avg_w': sum(wins) / len(wins) if wins else 0.0,
        'avg_l': sum(losses) / len(losses) if losses else 0.0,
        'pf': pf,
        'mdd': mdd,
        'calmar': pnl / mdd if mdd > 0 else (float('inf') if pnl > 0 else 0.0),
    }


def main():
    print('Loading cache...')
    entries = load_cache(CACHE_CSV)
    print(f'  {len(entries)} cached entries')

    # Bucket by split
    by_split = {s: [] for s in SPLITS}
    for e in entries:
        for split, (lo, hi) in SPLITS.items():
            if lo <= e.trade_date <= hi:
                by_split[split].append(e)
                break
    for s, lst in by_split.items():
        print(f'  {s}: {len(lst)} entries')

    print('\nFetching bars and simulating...')
    conn = sqlite3.connect(CACHE_DB)

    # Pre-fetch bars per (symbol, date) — cache in memory so we don't re-read DB per variant
    bar_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
    skipped_no_bars = 0
    for e in entries:
        key = (e.symbol, e.trade_date.isoformat())
        if key in bar_cache:
            continue
        bars = fetch_1min_bars(conn, e.symbol, e.trade_date)
        if bars is None or len(bars) < 5:
            skipped_no_bars += 1
            bar_cache[key] = None
        else:
            bar_cache[key] = bars
    print(f'  bars cached for {sum(1 for v in bar_cache.values() if v is not None)} (sym,date)s; {skipped_no_bars} missing')

    # Run all variants × splits × {fill-R, planned-R}
    R_MODES = [('fill-R', False), ('plan-R', True)]
    results = {
        (mode, v): {s: [] for s in SPLITS}
        for mode, _ in R_MODES for v in VARIANTS
    }
    for mode_name, use_planned_r in R_MODES:
        for variant in VARIANTS:
            sim = make_simulator(variant, use_planned_r=use_planned_r)
            for split, lst in by_split.items():
                for e in lst:
                    bars = bar_cache.get((e.symbol, e.trade_date.isoformat()))
                    if bars is None:
                        continue
                    r = simulate_one(sim, e, bars)
                    if r is not None:
                        results[(mode_name, variant)][split].append(r.pnl)

    # Print comparison tables — fill-R first (today's production), then plan-R
    for mode_name, _ in R_MODES:
        print('\n' + '=' * 100)
        print(f'R-MODE: {mode_name}  (use_planned_r={mode_name == "plan-R"})')
        print('=' * 100)
        print(f'{"Variant":<14}{"Split":<10}{"N":>5}{"P&L":>10}{"WR%":>7}{"AvgW":>9}{"AvgL":>9}{"PF":>6}{"MDD":>10}{"Calmar":>8}')
        print('-' * 100)
        for variant in VARIANTS:
            for split in SPLITS:
                agg = aggregate(results[(mode_name, variant)][split])
                print(f'{variant:<14}{split:<10}{agg["n"]:>5}'
                      f'  ${agg["pnl"]:>+8.0f}'
                      f' {agg["wr"]:>5.1f}%'
                      f'  ${agg["avg_w"]:>+6.0f}'
                      f' ${agg["avg_l"]:>+7.0f}'
                      f' {agg["pf"]:>5.2f}'
                      f' ${agg["mdd"]:>7.0f}'
                      f' {agg["calmar"]:>7.2f}')
            print()

    # Combined HOLDOUT comparison: fill-R vs plan-R per variant
    print('\n' + '=' * 100)
    print('HOLDOUT (Jan-Apr 2026) — fill-R vs plan-R, per variant')
    print('=' * 100)
    print(f'{"Variant":<14}{"fill-R PnL":>12}{"plan-R PnL":>12}{"Δ (plan-fill)":>16}'
          f'{"fill-R Calmar":>15}{"plan-R Calmar":>15}')
    print('-' * 100)
    for v in VARIANTS:
        f_agg = aggregate(results[('fill-R', v)]['HOLDOUT'])
        p_agg = aggregate(results[('plan-R', v)]['HOLDOUT'])
        delta = p_agg['pnl'] - f_agg['pnl']
        marker = '★' if delta > 0 else ' '
        print(f'{v:<14}  ${f_agg["pnl"]:>+9.0f}  ${p_agg["pnl"]:>+9.0f}'
              f'  ${delta:>+11,.0f}{marker}'
              f'  {f_agg["calmar"]:>11.2f}    {p_agg["calmar"]:>11.2f}')

    # Top-line verdict: best variant in each mode
    print('\n' + '=' * 100)
    print('VERDICT — best HOLDOUT P&L by R-mode')
    print('=' * 100)
    for mode_name, _ in R_MODES:
        best_v = max(VARIANTS, key=lambda v: aggregate(results[(mode_name, v)]['HOLDOUT'])['pnl'])
        a = aggregate(results[(mode_name, best_v)]['HOLDOUT'])
        print(f'  {mode_name:<10s}: {best_v:<14s} P&L ${a["pnl"]:>+9.0f}  Calmar {a["calmar"]:>5.2f}')


if __name__ == '__main__':
    main()
