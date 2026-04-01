#!/usr/bin/env python3
"""
Oil/Energy War Volatility Backtest

Tests MACD wave and bull flag strategies on curated oil/energy symbols.
Outputs per-symbol P&L summary for quick comparison.

Usage:
    python backtest_oil.py --strategy macd --period march
    python backtest_oil.py --strategy flag --threshold 5 --period march
    python backtest_oil.py --strategy both --period q1
"""

import argparse
import logging
import os
import sys
import time as time_mod
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Tuple

import pandas as pd
import pytz
import yaml

from dotenv import load_dotenv
load_dotenv()

from data_sources.alpaca_client import AlpacaClient
from persistence.database import get_database

logger = logging.getLogger(__name__)
ET = pytz.timezone('US/Eastern')

# ── Symbol Lists ──────────────────────────────────────────────────────────

MACD_SYMBOLS = [
    'OXY', 'HAL', 'SLB', 'DVN', 'MRO',
    'APA', 'FANG', 'RIG', 'TELL', 'AR',
]

FLAG_SYMBOLS = [
    'UCO', 'USO', 'XLE', 'OIH', 'GUSH',
    'ERX', 'XOP', 'CRAK', 'OXY', 'MRO',
]

# ── Helpers ───────────────────────────────────────────────────────────────

def get_trading_days(start: date, end: date, client: AlpacaClient) -> List[date]:
    """Get trading days in range using daily bars for SPY."""
    import sqlite3
    conn = sqlite3.connect('data/onemil.db')
    rows = conn.execute(
        "SELECT DISTINCT bar_date FROM daily_bars WHERE symbol='SPY' "
        "AND bar_date >= ? AND bar_date <= ? ORDER BY bar_date",
        (str(start), str(end))
    ).fetchall()
    if rows:
        return [date.fromisoformat(r[0]) for r in rows]
    # Fallback: fetch from API
    bars = client.get_daily_bars_range(['SPY'], str(start), str(end))
    if bars is not None and not bars.empty:
        dates = sorted(set(str(bars.iloc[i]['timestamp'])[:10] for i in range(len(bars))))
        return [date.fromisoformat(d) for d in dates]
    return []


def fetch_1min_bars(symbol: str, trade_date: date, client: AlpacaClient) -> pd.DataFrame:
    """Fetch 1-min bars for a symbol on a date, with caching."""
    db = get_database()
    cached = db.get_intraday_bars_bulk([(symbol, str(trade_date))])
    key = (symbol, str(trade_date))
    if key in cached and cached[key]:
        df = pd.DataFrame(cached[key])
        if len(df) >= 30:
            for col in ['timestamp', 't']:
                if col in df.columns:
                    df['ts'] = pd.to_datetime(df[col])
                    break
            return df.sort_values('ts').reset_index(drop=True)

    # Fetch from API
    mo = ET.localize(datetime(trade_date.year, trade_date.month, trade_date.day, 9, 30))
    mc = ET.localize(datetime(trade_date.year, trade_date.month, trade_date.day, 16, 0))
    mo_utc = mo.astimezone(timezone.utc)
    mc_utc = mc.astimezone(timezone.utc)
    bars = client.get_historical_1min_bars(symbol, mo_utc, mc_utc)
    if bars is not None and not bars.empty:
        recs = bars.to_dict('records')
        db.save_intraday_bars(symbol, str(trade_date), recs)
        bars['ts'] = pd.to_datetime(bars['timestamp'])
        return bars.sort_values('ts').reset_index(drop=True)
    return pd.DataFrame()


# ── MACD Wave Backtest ────────────────────────────────────────────────────

def run_macd_single(symbol: str, trade_date: date, bars: pd.DataFrame,
                    cfg: dict, trail_pct: float = 0.003) -> List[dict]:
    """Run MACD wave signal generation on a single symbol/date."""
    from macd_wave_backtest import generate_signals

    entry_cfg = cfg.get('entry', {})
    wave_cfg = cfg.get('waves', {})
    sizing_cfg = cfg.get('sizing', {})
    slip_cfg = cfg.get('slippage', {})
    entry_filters = {
        **entry_cfg,
        'max_waves': wave_cfg.get('max_waves', 1),
        'w1_scout': False,
        'w1_min_pct': 0.0,
        'position_size': sizing_cfg.get('position_size', 50000),
        'entry_pct': slip_cfg.get('entry_pct', 0.001),
        'exit_pct': slip_cfg.get('exit_pct', 0.001),
        'trail_stop_pct': trail_pct,
        # Override: no price cap for oil stocks (FANG is $140+)
        'max_price_at_entry': 0,
        # Override: no volume filter (we hand-picked these)
        'max_vol_at_cross': 0,
        'min_vol_at_cross': 0,
    }

    # Generate signals
    signals = generate_signals(bars, cfg, entry_filters)
    for sig in signals:
        sig['symbol'] = symbol
        sig['date'] = str(trade_date)
    return signals


def run_macd_backtest(symbols: List[str], start: date, end: date,
                      cfg: dict, client: AlpacaClient) -> List[dict]:
    """Run MACD wave on all symbols across date range."""
    all_signals = []
    trading_days = get_trading_days(start, end, client)

    for sym in symbols:
        sym_signals = 0
        for td in trading_days:
            bars = fetch_1min_bars(sym, td, client)
            if bars.empty or len(bars) < 35:
                continue
            # Check if stock moved enough (>5% intraday range)
            if 'high' in bars.columns and 'low' in bars.columns and 'open' in bars.columns:
                op = bars.iloc[0]['open']
                hi = bars['high'].max()
                lo = bars['low'].min()
                if op > 0 and (hi - lo) / op < 0.03:
                    continue  # Less than 3% range, skip
            sigs = run_macd_single(sym, td, bars, cfg)
            all_signals.extend(sigs)
            sym_signals += len(sigs)
        if sym_signals > 0:
            logger.info(f"  {sym}: {sym_signals} signals")

    return all_signals


# ── Bull Flag Backtest ────────────────────────────────────────────────────

def run_flag_single(symbol: str, trade_date: str, bars: pd.DataFrame,
                    threshold_pct: float) -> dict:
    """Run bull flag backtest on a single symbol/date."""
    from backtest import BacktestRunner

    runner = BacktestRunner(min_price=0.0, skip_midday=True)

    # Override the qualification threshold
    # BacktestRunner uses config.yaml internally, but we can set
    # a custom threshold by modifying the _run_realistic method's check
    # We pass prev_day_bars=None (no MACD warmup needed for this test)
    result = runner.run(symbol, bars, trade_date, prev_day_bars=None)
    return result


def run_flag_backtest(symbols: List[str], start: date, end: date,
                      threshold_pct: float, client: AlpacaClient) -> List[dict]:
    """Run bull flag on all symbols across date range."""
    all_trades = []
    trading_days = get_trading_days(start, end, client)

    for sym in symbols:
        for td in trading_days:
            bars = fetch_1min_bars(sym, td, client)
            if bars.empty or len(bars) < 30:
                continue
            # Check intraday range vs threshold
            if 'high' in bars.columns and 'low' in bars.columns and 'open' in bars.columns:
                op = bars.iloc[0]['open']
                hi = bars['high'].max()
                lo = bars['low'].min()
                if op > 0 and (hi - lo) / op < threshold_pct / 100:
                    continue

            result = run_flag_single(sym, str(td), bars, threshold_pct)
            if result and result.trades_simulated:
                for t in result.trades_simulated:
                    all_trades.append({
                        'symbol': sym,
                        'date': str(td),
                        'entry_price': t.entry_price,
                        'exit_price': t.exit_price or 0,
                        'pnl': t.pnl,
                        'pnl_pct': t.pnl_pct,
                        'exit_reason': t.exit_reason or '',
                        'shares': t.shares,
                    })
    return all_trades


# ── Output ────────────────────────────────────────────────────────────────

def print_macd_results(signals: List[dict], label: str):
    """Print MACD wave results summary."""
    from macd_wave_backtest import simulate_positions
    if not signals:
        print(f"\n{label}: 0 signals\n")
        return

    results, stats = simulate_positions(signals, max_concurrent=5, daily_loss_limit=-5000)

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    wins = sum(1 for r in results if r.pnl_dollar > 0)
    losses = sum(1 for r in results if r.pnl_dollar <= 0)
    total_pnl = sum(r.pnl_dollar for r in results)
    wr = wins / len(results) * 100 if results else 0

    print(f"  Trades: {len(results)} ({wins}W {losses}L)")
    print(f"  WR: {wr:.0f}%  |  P&L: ${total_pnl:+,.0f}")
    if wins > 0:
        avg_win = sum(r.pnl_dollar for r in results if r.pnl_dollar > 0) / wins
        print(f"  Avg Win: ${avg_win:+,.0f}")
    if losses > 0:
        avg_loss = sum(r.pnl_dollar for r in results if r.pnl_dollar <= 0) / losses
        print(f"  Avg Loss: ${avg_loss:+,.0f}")

    # Per-symbol breakdown
    sym_pnl = {}
    for r in results:
        sym = r.symbol
        if sym not in sym_pnl:
            sym_pnl[sym] = {'trades': 0, 'wins': 0, 'pnl': 0}
        sym_pnl[sym]['trades'] += 1
        sym_pnl[sym]['pnl'] += r.pnl_dollar
        if r.pnl_dollar > 0:
            sym_pnl[sym]['wins'] += 1

    print(f"\n  {'Symbol':<8} {'Trades':>6} {'Wins':>5} {'WR':>5} {'P&L':>10}")
    print(f"  {'-'*40}")
    for sym in sorted(sym_pnl, key=lambda s: sym_pnl[s]['pnl'], reverse=True):
        d = sym_pnl[sym]
        wr_s = d['wins'] / d['trades'] * 100 if d['trades'] else 0
        print(f"  {sym:<8} {d['trades']:>6} {d['wins']:>5} {wr_s:>4.0f}% ${d['pnl']:>+9,.0f}")


def print_flag_results(trades: List[dict], label: str):
    """Print bull flag results summary."""
    if not trades:
        print(f"\n{label}: 0 trades\n")
        return

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    wins = sum(1 for t in trades if t['pnl'] > 0)
    losses = sum(1 for t in trades if t['pnl'] <= 0)
    total_pnl = sum(t['pnl'] for t in trades)
    wr = wins / len(trades) * 100 if trades else 0

    print(f"  Trades: {len(trades)} ({wins}W {losses}L)")
    print(f"  WR: {wr:.0f}%  |  P&L: ${total_pnl:+,.0f}")
    if wins > 0:
        avg_win = sum(t['pnl'] for t in trades if t['pnl'] > 0) / wins
        print(f"  Avg Win: ${avg_win:+,.0f}")
    if losses > 0:
        avg_loss = sum(t['pnl'] for t in trades if t['pnl'] <= 0) / losses
        print(f"  Avg Loss: ${avg_loss:+,.0f}")

    # Per-symbol breakdown
    sym_pnl = {}
    for t in trades:
        sym = t['symbol']
        if sym not in sym_pnl:
            sym_pnl[sym] = {'trades': 0, 'wins': 0, 'pnl': 0}
        sym_pnl[sym]['trades'] += 1
        sym_pnl[sym]['pnl'] += t['pnl']
        if t['pnl'] > 0:
            sym_pnl[sym]['wins'] += 1

    print(f"\n  {'Symbol':<8} {'Trades':>6} {'Wins':>5} {'WR':>5} {'P&L':>10}")
    print(f"  {'-'*40}")
    for sym in sorted(sym_pnl, key=lambda s: sym_pnl[s]['pnl'], reverse=True):
        d = sym_pnl[sym]
        wr_s = d['wins'] / d['trades'] * 100 if d['trades'] else 0
        print(f"  {sym:<8} {d['trades']:>6} {d['wins']:>5} {wr_s:>4.0f}% ${d['pnl']:>+9,.0f}")

    # Per-trade detail
    print(f"\n  {'Date':<11} {'Symbol':<7} {'Entry':>7} {'Exit':>7} {'P&L':>9} {'Reason'}")
    print(f"  {'-'*55}")
    for t in sorted(trades, key=lambda x: x['date']):
        print(f"  {t['date']:<11} {t['symbol']:<7} ${t['entry_price']:>6.2f} ${t['exit_price']:>6.2f} ${t['pnl']:>+8,.0f} {t['exit_reason']}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Oil/Energy War Volatility Backtest")
    parser.add_argument('--strategy', choices=['macd', 'flag', 'both'], default='both')
    parser.add_argument('--period', choices=['march', 'q1'], default='march')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Bull flag intraday threshold %% (default: test 3,5,10)')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s [%(levelname)s] %(message)s',
                        stream=sys.stdout)

    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_API_SECRET')
    client = AlpacaClient(api_key, api_secret)

    if args.period == 'march':
        start, end = date(2026, 3, 1), date(2026, 3, 31)
    else:
        start, end = date(2026, 1, 1), date(2026, 3, 31)

    cfg = yaml.safe_load(open('macd_wave.yaml'))
    # Override for oil/energy: lower threshold, raise price cap
    cfg.setdefault('universe', {})['min_intraday_pct'] = 3.0  # 3% not 10%
    cfg.setdefault('universe', {})['max_price'] = 200.0  # Allow FANG, USO, etc.

    t0 = time_mod.time()

    # ── MACD Wave ──
    if args.strategy in ('macd', 'both'):
        logger.info(f"MACD Wave on {len(MACD_SYMBOLS)} energy stocks ({start} to {end})")
        signals = run_macd_backtest(MACD_SYMBOLS, start, end, cfg, client)
        print_macd_results(signals, f"MACD Wave — Oil/Energy ({start} to {end})")

    # ── Bull Flag ──
    if args.strategy in ('flag', 'both'):
        thresholds = [args.threshold] if args.threshold else [3.0, 5.0, 10.0]
        for thr in thresholds:
            logger.info(f"Bull Flag ({thr}%% threshold) on {len(FLAG_SYMBOLS)} symbols ({start} to {end})")
            trades = run_flag_backtest(FLAG_SYMBOLS, start, end, thr, client)
            print_flag_results(trades, f"Bull Flag {thr}%% threshold — Oil/Energy ({start} to {end})")

    elapsed = time_mod.time() - t0
    print(f"\nCompleted in {elapsed:.0f}s")


if __name__ == '__main__':
    main()
