#!/usr/bin/env python3
"""5-min Opening Range Breakout (ORB) walk-forward study.

Standalone research — no prod code touched. Determines whether an ORB strategy
has real, orthogonal edge on our existing universe of gap-up movers. If yes,
motivates a Strategy-3 engine in a follow-up PR; if no, we drop ORB as a
non-starter.

See /home/ec2-user/.claude/plans/mellow-sniffing-abelson.md for context,
ship criteria, and known limitations.

Usage:
    python3 study_orb.py

Outputs (timestamped):
    analysis_results/orb_study_{YYYYMMDD_HHMM}.md
    analysis_results/orb_trades_{YYYYMMDD_HHMM}.csv
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from persistence.database import Database

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CACHE_DB = 'data/cache.db'
BULL_FLAG_CACHE = 'data/bull_flag_cache_e50_x30.csv'
OUT_DIR = 'analysis_results'

# Market open in UTC (9:30 ET = 13:30 UTC with daylight saving; code tolerates
# either 13:30 or 14:30 UTC depending on DST — we slice by bar timestamps directly).
ET_MARKET_OPEN_HOUR_UTC_EDT = 13   # during EDT (most of year)
ET_MARKET_OPEN_HOUR_UTC_EST = 14   # during EST (Nov-Mar)

# Slippage — matches macd_wave.yaml after 2026-04-17 recalibration
ENTRY_SLIP_BPS_DEFAULT = 30.0
EXIT_SLIP_BPS_DEFAULT = 10.0

# Fixed position sizing — $50K / entry_price (matches macd_wave default)
POSITION_SIZE_USD = 50_000

# Walk-forward splits — same as study_bull_flag_v2_clean.py for consistency
SPLITS: List[Tuple[str, str, str, str, str]] = [
    ('A: H1\'25 -> H2\'25-Apr\'26',  '2025-01-01', '2025-06-30', '2025-07-01', '2026-04-30'),
    ('B: Y2025 -> Q1+Apr\'26',       '2025-01-01', '2025-12-31', '2026-01-01', '2026-04-30'),
    ('C: Jan-Sep\'25 -> Oct\'25-Apr\'26', '2025-01-01', '2025-09-30', '2025-10-01', '2026-04-30'),
]

# Variant grid — locked BEFORE running (no iterative tuning on test data).
# (name, range_minutes, entry_mode, stop_mode, target_mult, time_stop_min, vol_confirm, max_spread_bps)
VARIANTS: List[Tuple] = [
    ('ORB_5_vanilla',         5,  'touch',       'range_low', 2.0, 60,  False, None),
    ('ORB_5_close',           5,  'close_above', 'range_low', 2.0, 60,  False, None),
    ('ORB_5_mid_stop',        5,  'touch',       'midpoint',  2.0, 60,  False, None),
    ('ORB_5_target_3x',       5,  'touch',       'range_low', 3.0, 60,  False, None),
    ('ORB_5_time_30',         5,  'touch',       'range_low', 2.0, 30,  False, None),
    ('ORB_5_vol_conf',        5,  'touch',       'range_low', 2.0, 60,  True,  None),
    ('ORB_5_spread_gate_100', 5,  'touch',       'range_low', 2.0, 60,  False, 100.0),
    ('ORB_15_vanilla',        15, 'touch',       'range_low', 2.0, 60,  False, None),
    ('ORB_15_close',          15, 'close_above', 'range_low', 2.0, 60,  False, None),
    ('ORB_30_vanilla',        30, 'touch',       'range_low', 2.0, 60,  False, None),
]

# Ship criteria — defined BEFORE running results.
SHIP_CRITERIA = {
    'min_total_test_pnl': 30_000,   # >= +$30K over 15.5mo
    'min_split_test_pnl': 0,        # min across splits > 0 (ROBUST)
    'max_dd_pct_of_peak': 0.20,     # max DD < 20% of peak equity
    'max_corr_with_bullflag': 0.4,  # daily P&L correlation ceiling
    'min_unique_winning_days_pct': 0.30,  # > 30% of winning days unique to ORB
}


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_universe_by_date(cache_path: str = BULL_FLAG_CACHE) -> Dict[str, List[str]]:
    """Return {date_str: [symbol, symbol, ...]} from the bull flag cache.

    Known bias: this is the set of (symbol, date) pairs that generated a bull
    flag pattern — NOT the full gap-up movers universe. Broader universe is
    Phase-B work if Phase-A results justify it.
    """
    df = pd.read_csv(cache_path)
    grouped: Dict[str, List[str]] = {}
    for _, row in df[['symbol', 'date']].drop_duplicates().iterrows():
        grouped.setdefault(row['date'], []).append(row['symbol'])
    return grouped


def bulk_load_bars(
    pairs: List[Tuple[str, str]], db: Database,
) -> Dict[Tuple[str, str], List[dict]]:
    """Batch-fetch 1-min bars for a list of (symbol, date_str) pairs."""
    return db.get_intraday_bars_bulk(pairs)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

@dataclass
class OrbTrade:
    variant: str
    symbol: str
    date: str
    range_high: float
    range_low: float
    range_size: float
    entry_time: Optional[datetime] = None
    entry_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: str = ''
    shares: int = 0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    entered: bool = False


def _bars_to_df(bars: List[dict]) -> pd.DataFrame:
    """Convert list-of-dicts (from db.get_intraday_bars_bulk) to DataFrame."""
    if not bars:
        return pd.DataFrame()
    df = pd.DataFrame(bars)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df.sort_values('timestamp').reset_index(drop=True)


def _session_open_timestamp(bars_df: pd.DataFrame) -> Optional[pd.Timestamp]:
    """Find the bar timestamp for 9:30 ET (the first regular-session bar).

    Alpaca timestamps are UTC and bar-START. 9:30 ET = 13:30 or 14:30 UTC
    (EDT vs EST). The bars we cache include premarket, so we pick the FIRST
    bar at minute 30 past the hour that corresponds to market open.
    """
    if bars_df.empty:
        return None
    # Market open ET = 9:30. In UTC:
    #   EDT (Mar-Nov): 13:30
    #   EST (Nov-Mar): 14:30
    # Pick the first bar where hour+minute matches either.
    mask = (bars_df['timestamp'].dt.minute == 30) & (
        bars_df['timestamp'].dt.hour.isin([13, 14])
    )
    if not mask.any():
        return None
    return bars_df.loc[mask, 'timestamp'].iloc[0]


def simulate_orb_trade(
    bars_df: pd.DataFrame,
    symbol: str,
    date_str: str,
    variant_name: str,
    range_minutes: int = 5,
    entry_mode: str = 'touch',
    stop_mode: str = 'range_low',
    target_mult: float = 2.0,
    time_stop_minutes: int = 60,
    require_vol_confirm: bool = False,
    max_spread_bps: Optional[float] = None,
    entry_slip_bps: float = ENTRY_SLIP_BPS_DEFAULT,
    exit_slip_bps: float = EXIT_SLIP_BPS_DEFAULT,
    position_size_usd: float = POSITION_SIZE_USD,
) -> OrbTrade:
    """Simulate one ORB trade. Returns OrbTrade (entered=False if no trigger)."""
    trade = OrbTrade(variant=variant_name, symbol=symbol, date=date_str,
                     range_high=0, range_low=0, range_size=0)

    if bars_df.empty:
        return trade

    open_ts = _session_open_timestamp(bars_df)
    if open_ts is None:
        return trade

    range_end_ts = open_ts + timedelta(minutes=range_minutes)
    search_end_ts = range_end_ts + timedelta(minutes=time_stop_minutes)

    # Range window: bars with timestamp in [open_ts, range_end_ts) -- i.e.,
    # bars 0..(range_minutes-1) inclusive. (Alpaca bars are bar-START.)
    range_mask = (bars_df['timestamp'] >= open_ts) & (bars_df['timestamp'] < range_end_ts)
    range_bars = bars_df.loc[range_mask]
    if len(range_bars) < range_minutes:
        # Missing bars during range formation — skip
        return trade

    range_high = float(range_bars['high'].max())
    range_low = float(range_bars['low'].min())
    range_size = range_high - range_low
    range_avg_vol = float(range_bars['volume'].mean()) if len(range_bars) else 0.0
    trade.range_high = range_high
    trade.range_low = range_low
    trade.range_size = range_size

    if range_size <= 0:
        return trade

    # Search window for entry trigger
    search_mask = (bars_df['timestamp'] >= range_end_ts) & (bars_df['timestamp'] < search_end_ts)
    search_bars = bars_df.loc[search_mask].reset_index(drop=True)
    if search_bars.empty:
        return trade

    # Find entry trigger bar
    entry_bar_idx = None
    for i, row in search_bars.iterrows():
        if entry_mode == 'touch':
            if row['high'] > range_high:
                entry_bar_idx = i
                break
        elif entry_mode == 'close_above':
            if row['close'] > range_high:
                entry_bar_idx = i
                break
        else:
            raise ValueError(f"unknown entry_mode: {entry_mode}")

    if entry_bar_idx is None:
        return trade  # No trigger within time window

    entry_bar = search_bars.iloc[entry_bar_idx]

    # Volume confirmation
    if require_vol_confirm and range_avg_vol > 0:
        if entry_bar['volume'] < 1.5 * range_avg_vol:
            return trade

    # Spread gate proxy — we don't have bid/ask in bars, use (high - low) / close
    # on the entry bar as a rough spread proxy. Not as good as real bid/ask but
    # the best we can do from OHLCV data.
    if max_spread_bps is not None:
        close_p = float(entry_bar['close'])
        if close_p > 0:
            bar_range_bps = (float(entry_bar['high']) - float(entry_bar['low'])) / close_p * 10000
            if bar_range_bps > max_spread_bps:
                return trade

    # Entry price: 'touch' mode enters at range_high + slip (market-style);
    # 'close_above' mode enters at bar close + slip
    if entry_mode == 'touch':
        raw_entry = range_high
    else:
        raw_entry = float(entry_bar['close'])
    entry_price = raw_entry * (1 + entry_slip_bps / 10000)

    shares = max(1, int(position_size_usd / entry_price))

    # Determine stop
    if stop_mode == 'range_low':
        stop_price = range_low
    elif stop_mode == 'midpoint':
        stop_price = range_low + range_size * 0.5
    elif stop_mode == 'atr1':
        # Use range_size as a 1-ATR proxy, stop = entry - range_size
        stop_price = entry_price - range_size
    else:
        raise ValueError(f"unknown stop_mode: {stop_mode}")

    # Target = range_high + target_mult * range_size
    target_price = range_high + target_mult * range_size

    # Simulate from entry bar onward
    trade.entered = True
    trade.entry_time = entry_bar['timestamp'].to_pydatetime()
    trade.entry_price = entry_price
    trade.shares = shares

    # Iterate bars from entry bar (inclusive) to EOD
    post_entry_mask = bars_df['timestamp'] >= entry_bar['timestamp']
    sim_bars = bars_df.loc[post_entry_mask].reset_index(drop=True)

    exit_price = None
    exit_reason = ''
    exit_time = None
    # Skip the ENTRY bar's own high/low check for exits — we entered on this
    # bar's high touching range_high, can't fairly evaluate stop hit on same bar.
    # Start stop/target checks from NEXT bar.
    for _, row in sim_bars.iloc[1:].iterrows():
        bar_high = float(row['high'])
        bar_low = float(row['low'])
        bar_close = float(row['close'])
        ts = row['timestamp']

        # Stop first (conservative — if both touched in same bar, assume stop hit
        # since a 1-min bar cannot be decomposed into intra-bar order).
        if bar_low <= stop_price:
            raw_exit = stop_price
            exit_price = raw_exit * (1 - exit_slip_bps / 10000)
            exit_reason = 'stop'
            exit_time = ts.to_pydatetime()
            break
        if bar_high >= target_price:
            raw_exit = target_price
            exit_price = raw_exit * (1 - exit_slip_bps / 10000)
            exit_reason = 'target'
            exit_time = ts.to_pydatetime()
            break

    # No stop/target hit — exit at EOD (last bar close)
    if exit_price is None:
        last_bar = sim_bars.iloc[-1]
        raw_exit = float(last_bar['close'])
        exit_price = raw_exit * (1 - exit_slip_bps / 10000)
        exit_reason = 'eod'
        exit_time = last_bar['timestamp'].to_pydatetime()

    trade.exit_time = exit_time
    trade.exit_price = exit_price
    trade.exit_reason = exit_reason
    trade.pnl = (exit_price - entry_price) * shares
    trade.pnl_pct = (exit_price - entry_price) / entry_price * 100
    return trade


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def run_variant(
    variant: Tuple,
    universe: Dict[str, List[str]],
    bars_cache: Dict[Tuple[str, str], pd.DataFrame],
) -> List[OrbTrade]:
    """Run one variant across the full universe. Returns list of OrbTrade."""
    name, range_min, entry_mode, stop_mode, target_mult, time_stop, vol_conf, spread_gate = variant
    trades: List[OrbTrade] = []
    for date_str in sorted(universe.keys()):
        for symbol in universe[date_str]:
            bars_df = bars_cache.get((symbol, date_str))
            if bars_df is None or bars_df.empty:
                continue
            t = simulate_orb_trade(
                bars_df, symbol, date_str, name,
                range_minutes=range_min, entry_mode=entry_mode,
                stop_mode=stop_mode, target_mult=target_mult,
                time_stop_minutes=time_stop, require_vol_confirm=vol_conf,
                max_spread_bps=spread_gate,
            )
            if t.entered:
                trades.append(t)
    return trades


def summarize_trades(trades: List[OrbTrade]) -> Dict[str, float]:
    """Aggregate stats for a list of trades."""
    if not trades:
        return {'n': 0, 'wins': 0, 'wr': 0.0, 'pnl': 0.0, 'avg_pnl': 0.0,
                'max_dd': 0.0, 'peak': 0.0}
    pnls = [t.pnl for t in trades]
    n = len(trades)
    wins = sum(1 for p in pnls if p > 0)
    total = sum(pnls)
    avg = total / n
    # Equity curve for DD
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        cum += p
        if cum > peak:
            peak = cum
        dd = cum - peak
        if dd < max_dd:
            max_dd = dd
    return {
        'n': n, 'wins': wins, 'wr': wins / n * 100,
        'pnl': total, 'avg_pnl': avg,
        'max_dd': max_dd, 'peak': peak,
    }


def filter_trades_by_date(trades: List[OrbTrade], start: str, end: str) -> List[OrbTrade]:
    return [t for t in trades if start <= t.date <= end]


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------

def walk_forward_stats(trades: List[OrbTrade]) -> Dict[str, Dict[str, float]]:
    """Per-split train/test summary for ONE variant."""
    out: Dict[str, Dict[str, float]] = {}
    for split_name, train_s, train_e, test_s, test_e in SPLITS:
        train = filter_trades_by_date(trades, train_s, train_e)
        test = filter_trades_by_date(trades, test_s, test_e)
        out[split_name] = {
            'train': summarize_trades(train),
            'test': summarize_trades(test),
        }
    return out


# ---------------------------------------------------------------------------
# Orthogonality
# ---------------------------------------------------------------------------

def bull_flag_daily_pnl() -> pd.Series:
    """Daily P&L from the bull flag cache (sum of pnl per trade_date)."""
    df = pd.read_csv(BULL_FLAG_CACHE)
    df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce').fillna(0)
    daily = df.groupby('date')['pnl'].sum()
    daily.index = pd.to_datetime(daily.index)
    return daily


def orthogonality(trades: List[OrbTrade]) -> Dict[str, float]:
    """Compute daily P&L correlation + unique-winning-days vs bull flag."""
    if not trades:
        return {'corr': float('nan'), 'unique_win_days_pct': 0.0,
                'combined_max_dd': 0.0}
    df = pd.DataFrame([{'date': t.date, 'pnl': t.pnl} for t in trades])
    orb_daily = df.groupby('date')['pnl'].sum()
    orb_daily.index = pd.to_datetime(orb_daily.index)

    bf_daily = bull_flag_daily_pnl()

    # Align dates (outer join, NaN=0 for strategy-inactive days)
    all_dates = sorted(set(orb_daily.index) | set(bf_daily.index))
    orb_aligned = orb_daily.reindex(all_dates, fill_value=0)
    bf_aligned = bf_daily.reindex(all_dates, fill_value=0)

    # Pearson on daily P&L
    corr = float(orb_aligned.corr(bf_aligned))

    # Unique winning days: ORB won AND bull flag lost/flat
    orb_won = orb_aligned > 0
    bf_won = bf_aligned > 0
    total_orb_wins = int(orb_won.sum())
    unique_orb_wins = int(((orb_aligned > 0) & (bf_aligned <= 0)).sum())
    pct = unique_orb_wins / total_orb_wins if total_orb_wins > 0 else 0.0

    # Combined-book max DD (simultaneous P&L)
    combined = orb_aligned + bf_aligned
    cum = combined.cumsum()
    peak = cum.cummax()
    dd = (cum - peak).min()

    return {
        'corr': corr,
        'unique_win_days_pct': pct,
        'unique_win_days': unique_orb_wins,
        'total_win_days': total_orb_wins,
        'combined_max_dd': float(dd),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _verdict(test_min_d: float, test_mean_d: float) -> str:
    if test_min_d > 0:
        return 'ROBUST'
    if test_mean_d > 0:
        return 'mixed'
    return 'losing'


def main() -> None:
    t0 = datetime.now()
    print(f"[{t0.isoformat(timespec='seconds')}] Loading universe + bars...")

    universe = load_universe_by_date()
    n_dates = len(universe)
    n_pairs = sum(len(v) for v in universe.values())
    print(f"  Universe: {n_dates} dates, {n_pairs} (symbol, date) pairs")

    db = Database(db_path=CACHE_DB)
    # Flatten to pair list for bulk fetch
    pair_list: List[Tuple[str, str]] = []
    for d, syms in universe.items():
        for s in syms:
            pair_list.append((s, d))
    print(f"  Fetching {len(pair_list)} bar sets...")
    raw_bars = db.get_intraday_bars_bulk(pair_list)
    print(f"  Got bars for {len(raw_bars)} pairs")
    db.close()

    # Convert to DataFrames once
    bars_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
    for k, v in raw_bars.items():
        bars_cache[k] = _bars_to_df(v)

    # Run sweep
    print(f"\nRunning {len(VARIANTS)} variants...")
    all_trades_by_variant: Dict[str, List[OrbTrade]] = {}
    for variant in VARIANTS:
        name = variant[0]
        trades = run_variant(variant, universe, bars_cache)
        all_trades_by_variant[name] = trades
        s = summarize_trades(trades)
        print(f"  {name:<28} n={s['n']:>3}  WR={s['wr']:>4.1f}%  "
              f"P&L=${s['pnl']:>+9,.0f}  avg=${s['avg_pnl']:>+6,.0f}  "
              f"DD=${s['max_dd']:>+8,.0f}")

    # Walk-forward
    print("\nWalk-forward per variant (test ΔP&L vs ORB_5_vanilla baseline):")
    vanilla_wf = walk_forward_stats(all_trades_by_variant['ORB_5_vanilla'])
    variant_wf: Dict[str, Dict[str, Dict]] = {}
    print(f"{'Variant':<28} {'A_test':>12} {'B_test':>12} {'C_test':>12} "
          f"{'mean Δ':>12} {'min Δ':>12}  Verdict")
    print('-' * 110)
    for name, trades in all_trades_by_variant.items():
        wf = walk_forward_stats(trades)
        variant_wf[name] = wf
        if name == 'ORB_5_vanilla':
            # Print its absolute numbers on first row
            cells = []
            for split_name, *_ in SPLITS:
                cells.append(f"${wf[split_name]['test']['pnl']:>+10,.0f}")
            print(f"{'ORB_5_vanilla (baseline)':<28} {' '.join(cells)}  {'— (baseline)':<12}")
            continue
        deltas = []
        cells = []
        for split_name, *_ in SPLITS:
            test_pnl = wf[split_name]['test']['pnl']
            base_pnl = vanilla_wf[split_name]['test']['pnl']
            delta = test_pnl - base_pnl
            deltas.append(delta)
            cells.append(f"${delta:>+10,.0f}")
        mean_d = sum(deltas) / len(deltas)
        min_d = min(deltas)
        v = _verdict(min_d, mean_d)
        print(f"{name:<28} {' '.join(cells)} ${mean_d:>+10,.0f} ${min_d:>+10,.0f}  {v}")

    # Pick winner: highest mean test P&L across all 3 splits AND min > 0
    print("\nSelecting winner by ship-criteria rank...")
    candidates = []
    for name, wf in variant_wf.items():
        test_pnls = [wf[s[0]]['test']['pnl'] for s in SPLITS]
        test_mean = sum(test_pnls) / len(test_pnls)
        test_min = min(test_pnls)
        candidates.append((name, test_mean, test_min, test_pnls))
    candidates.sort(key=lambda c: (c[2] > 0, c[1]), reverse=True)
    winner_name, w_mean, w_min, w_tpnls = candidates[0]
    print(f"  Winner (rank-1 by test mean P&L with min > 0 preferred): {winner_name}")
    print(f"    Test mean: ${w_mean:+,.0f}  min: ${w_min:+,.0f}  per-split: {w_tpnls}")

    # Orthogonality on winner
    print("\nOrthogonality analysis on winner...")
    orth = orthogonality(all_trades_by_variant[winner_name])
    print(f"  Daily P&L corr with bull flag: {orth['corr']:+.3f}")
    print(f"  Unique winning days (won while bull flag didn't): "
          f"{orth['unique_win_days']}/{orth['total_win_days']} "
          f"({orth['unique_win_days_pct']*100:.1f}%)")
    print(f"  Combined-book (ORB+BF) max DD: ${orth['combined_max_dd']:+,.0f}")

    # Ship criteria gate
    print("\n=== SHIP CRITERIA GATE ===")
    winner_total = sum(t.pnl for t in all_trades_by_variant[winner_name])
    winner_summary = summarize_trades(all_trades_by_variant[winner_name])
    crit_results = {}

    v1 = winner_total >= SHIP_CRITERIA['min_total_test_pnl']
    crit_results['total_15.5mo_pnl'] = (
        v1, f"${winner_total:+,.0f} vs >=${SHIP_CRITERIA['min_total_test_pnl']:+,.0f}"
    )
    print(f"  [{'PASS' if v1 else 'FAIL'}] Total 15.5mo P&L: ${winner_total:+,.0f} "
          f"(need >= ${SHIP_CRITERIA['min_total_test_pnl']:+,.0f})")

    v2 = w_min > SHIP_CRITERIA['min_split_test_pnl']
    crit_results['min_split_test_pnl'] = (v2, f"${w_min:+,.0f} vs > 0")
    print(f"  [{'PASS' if v2 else 'FAIL'}] Min split test P&L: ${w_min:+,.0f} "
          f"(need > 0)")

    peak = winner_summary['peak'] or 1.0
    dd_pct = abs(winner_summary['max_dd']) / peak if peak > 0 else 0.0
    v3 = dd_pct < SHIP_CRITERIA['max_dd_pct_of_peak']
    crit_results['max_dd_pct'] = (v3, f"{dd_pct*100:.1f}% vs < {SHIP_CRITERIA['max_dd_pct_of_peak']*100:.0f}%")
    print(f"  [{'PASS' if v3 else 'FAIL'}] Max DD / peak: {dd_pct*100:.1f}% "
          f"(need < {SHIP_CRITERIA['max_dd_pct_of_peak']*100:.0f}%)")

    corr = orth['corr']
    v4 = corr < SHIP_CRITERIA['max_corr_with_bullflag']
    crit_results['daily_pnl_corr'] = (v4, f"{corr:+.3f} vs < {SHIP_CRITERIA['max_corr_with_bullflag']}")
    print(f"  [{'PASS' if v4 else 'FAIL'}] Daily P&L corr with bull flag: {corr:+.3f} "
          f"(need < {SHIP_CRITERIA['max_corr_with_bullflag']})")

    v5 = orth['unique_win_days_pct'] > SHIP_CRITERIA['min_unique_winning_days_pct']
    crit_results['unique_win_days_pct'] = (
        v5, f"{orth['unique_win_days_pct']*100:.1f}% vs > {SHIP_CRITERIA['min_unique_winning_days_pct']*100:.0f}%"
    )
    print(f"  [{'PASS' if v5 else 'FAIL'}] Unique winning days: "
          f"{orth['unique_win_days_pct']*100:.1f}% "
          f"(need > {SHIP_CRITERIA['min_unique_winning_days_pct']*100:.0f}%)")

    all_pass = all(r[0] for r in crit_results.values())
    print(f"\n{'='*60}")
    print(f"FINAL VERDICT: {'✓ ALL CRITERIA PASS — candidate for Strategy 3' if all_pass else '✗ REJECTED'}")
    print(f"{'='*60}")

    # Write output files
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    md_path = f"{OUT_DIR}/orb_study_{ts}.md"
    csv_path = f"{OUT_DIR}/orb_trades_{ts}.csv"

    # Per-trade CSV
    all_trades_flat: List[dict] = []
    for name, trades in all_trades_by_variant.items():
        for t in trades:
            all_trades_flat.append(asdict(t))
    trade_df = pd.DataFrame(all_trades_flat)
    trade_df.to_csv(csv_path, index=False)
    print(f"\nPer-trade CSV: {csv_path} ({len(trade_df)} rows)")

    # Markdown report
    with open(md_path, 'w') as f:
        f.write(f"# ORB Walk-Forward Study\n\n")
        f.write(f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}_\n\n")
        f.write(f"**Universe**: {n_pairs} (symbol, date) pairs across {n_dates} trading days "
                f"(Jan'25–Apr'26 from bull flag cache).\n\n")
        f.write(f"**Slippage model**: {ENTRY_SLIP_BPS_DEFAULT:.0f}bps entry / "
                f"{EXIT_SLIP_BPS_DEFAULT:.0f}bps exit (matches macd_wave calibrated).\n\n")
        f.write(f"**Position sizing**: ${POSITION_SIZE_USD:,} per trade (flat).\n\n")

        f.write("## Per-variant summary (all 15.5mo)\n\n")
        f.write("| Variant | n | WR | Total P&L | Avg | Max DD |\n|---|---:|---:|---:|---:|---:|\n")
        for name, trades in all_trades_by_variant.items():
            s = summarize_trades(trades)
            f.write(f"| {name} | {s['n']} | {s['wr']:.0f}% | "
                    f"${s['pnl']:+,.0f} | ${s['avg_pnl']:+,.0f} | "
                    f"${s['max_dd']:+,.0f} |\n")

        f.write("\n## Walk-forward (ΔP&L vs ORB_5_vanilla baseline, test only)\n\n")
        f.write("| Variant | A test Δ | B test Δ | C test Δ | Mean Δ | Min Δ | Verdict |\n")
        f.write("|---|---:|---:|---:|---:|---:|---|\n")
        for name, wf in variant_wf.items():
            if name == 'ORB_5_vanilla':
                continue
            deltas = []
            cells = []
            for split_name, *_ in SPLITS:
                d = wf[split_name]['test']['pnl'] - vanilla_wf[split_name]['test']['pnl']
                deltas.append(d)
                cells.append(f"${d:+,.0f}")
            mean_d = sum(deltas) / len(deltas)
            min_d = min(deltas)
            f.write(f"| {name} | {' | '.join(cells)} | ${mean_d:+,.0f} | "
                    f"${min_d:+,.0f} | {_verdict(min_d, mean_d)} |\n")

        f.write(f"\n## Winner: `{winner_name}`\n\n")
        f.write(f"- Test mean P&L: ${w_mean:+,.0f}\n")
        f.write(f"- Test min across splits: ${w_min:+,.0f}\n")
        f.write(f"- Per-split test P&L: {w_tpnls}\n\n")

        f.write("## Orthogonality (winner vs bull flag)\n\n")
        f.write(f"- Daily P&L correlation: **{corr:+.3f}**\n")
        f.write(f"- Unique winning days: **{orth['unique_win_days']}/{orth['total_win_days']} "
                f"({orth['unique_win_days_pct']*100:.1f}%)**\n")
        f.write(f"- Combined-book max DD: **${orth['combined_max_dd']:+,.0f}**\n\n")

        f.write("## Ship criteria\n\n")
        f.write("| Criterion | Result | Threshold | Pass? |\n|---|---|---|:-:|\n")
        for k, (passed, detail) in crit_results.items():
            f.write(f"| {k} | {detail} | — | {'✓' if passed else '✗'} |\n")
        f.write(f"\n**Final verdict: {'✓ ALL PASS — candidate for Strategy 3' if all_pass else '✗ REJECTED'}**\n\n")

        f.write("## Known limitations\n\n")
        f.write("- Universe biased to bull-flag-qualified stocks only. Phase B: broader gap-up universe.\n")
        f.write("- Flat 30/10 bps slippage; does not model per-stock spread or wider 9:30 ET spread regime.\n")
        f.write("- No halt simulation; real fills on halted stocks are worse.\n")
        f.write("- max_concurrent not simulated; combined live book may take fewer trades.\n")
        f.write("- Single-year data (one macro regime).\n")

    print(f"Report: {md_path}")
    print(f"\nElapsed: {(datetime.now() - t0).total_seconds():.1f}s")


if __name__ == '__main__':
    main()
