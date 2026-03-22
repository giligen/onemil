"""
Run 6 backtest configurations and produce comparison report.

Configs:
1. BASELINE: current config (no new filters)
2. MIN_DIST: min_stop_distance = $0.09
3. SMA_SLOPE: regime filter with sma_slope_filter = True
4. VOL_DEAD: vol dead zone 2-5x filter
5. NO_POP: 5-bar no-pop exit
6. COMBINED: all 4 together
"""

import logging
import time
import sys
from datetime import date, timedelta
from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

# Suppress noisy loggers
for name in ['trading.pattern_detector', 'backtest', 'trading.trade_planner',
             'trading.indicators', 'persistence.database']:
    logging.getLogger(name).setLevel(logging.ERROR)


def run_config(label, runner, market_regime, movers_by_date, filtered_dates,
               all_bars, all_prev_bars, db):
    """Run a single backtest configuration and return results."""
    from backtest import BacktestResult
    from batch_backtest import _get_previous_trading_date

    results = []
    t0 = time.time()

    for trade_date in filtered_dates:
        # Regime filter
        if market_regime and not market_regime.is_regime_ok(trade_date):
            continue

        # Thin liquidity
        if market_regime and market_regime.is_thin_liquidity(trade_date):
            runner._min_breakout_vol_override = market_regime.get_min_breakout_volume_ratio(
                trade_date, default=0
            )
        else:
            runner._min_breakout_vol_override = 0

        for sym, d, prev_close in movers_by_date[trade_date]:
            date_str = d.isoformat()
            bars = all_bars.get((sym, date_str))
            if bars is None or bars.empty:
                continue

            prev_day_bars = None
            if all_prev_bars:
                prev_date = _get_previous_trading_date(trade_date)
                if prev_date:
                    prev_day_bars = all_prev_bars.get((sym, prev_date.isoformat()))

            try:
                result = runner.run(
                    sym, bars, date_str,
                    prev_close=prev_close if prev_close > 0 else None,
                    prev_day_bars=prev_day_bars,
                )
                results.append(result)
            except Exception:
                pass

    elapsed = time.time() - t0
    n_trades = sum(len(r.trades_simulated) for r in results)
    logger.info(f"  {label}: {n_trades} trades in {elapsed:.0f}s")
    return results


def results_to_df(results):
    """Convert BacktestResult list to trade-level DataFrame."""
    from batch_backtest import utc_to_et_str
    rows = []
    for r in results:
        for t in r.trades_simulated:
            rows.append({
                'symbol': t.symbol,
                'date': r.trade_date,
                'entry_time_et': utc_to_et_str(t.entry_time),
                'entry_price': t.entry_price,
                'stop_loss': t.stop_loss,
                'target': t.take_profit,
                'shares': t.shares,
                'exit_time_et': utc_to_et_str(t.exit_time),
                'exit_price': t.exit_price,
                'exit_reason': t.exit_reason,
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct,
                'partial_taken': t.partial_exit_taken,
                'partial_price': t.partial_exit_price,
                'partial_shares': t.partial_shares,
                'partial_pnl': t.partial_pnl,
            })
    return pd.DataFrame(rows)


def evaluate(label, df):
    """Compute metrics for a trade DataFrame."""
    if len(df) == 0:
        return {}
    w = df[df['pnl'] > 0]
    l = df[df['pnl'] <= 0]
    monthly = df.groupby(pd.to_datetime(df['date']).dt.to_period('M'))['pnl'].sum()
    sharpe = monthly.mean() / monthly.std() * np.sqrt(12) if monthly.std() > 0 else 0
    cum = df.sort_values('date')['pnl'].cumsum()
    dd = (cum - cum.cummax()).min()
    return {
        'label': label,
        'trades': len(df),
        'wr': len(w) / len(df) * 100,
        'pnl': df['pnl'].sum(),
        'avg_win': w['pnl'].mean() if len(w) > 0 else 0,
        'avg_loss': l['pnl'].mean() if len(l) > 0 else 0,
        'pf': w['pnl'].sum() / abs(l['pnl'].sum()) if len(l) > 0 and l['pnl'].sum() != 0 else 0,
        'sharpe': sharpe,
        'dd': dd,
        'lm': (monthly < 0).sum(),
    }


def main():
    from backtest import BacktestRunner
    from trading.market_regime import MarketRegimeFilter
    from data_sources.alpaca_client import AlpacaClient
    from persistence.database import Database
    from batch_backtest import (
        fetch_daily_bars_cached, find_big_movers,
        _get_previous_trading_date, _market_hours_utc,
    )

    db = Database('data/onemil.db')
    start_date = date(2025, 1, 1)
    end_date = date(2026, 3, 21)

    logger.info("Loading data...")

    from config import Config
    cfg = Config._load_yaml_only()
    import os
    from dotenv import load_dotenv
    load_dotenv()
    client = AlpacaClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET"))

    # Get symbols and daily bars
    universe = db.get_active_universe()
    symbols = [s['symbol'] for s in universe]
    universe_dict = {s['symbol']: s for s in universe}
    daily_bars = fetch_daily_bars_cached(symbols, start_date - timedelta(days=7), end_date, client, db)

    scanner_cfg = cfg.get("scanner", {})
    movers = find_big_movers(
        daily_bars, universe_dict=universe_dict,
        price_min=float(scanner_cfg.get("price_min", 2.0)),
        price_max=float(scanner_cfg.get("price_max", 30.0)),
        float_max=int(scanner_cfg.get("float_max", 10_000_000)),
        start_date=start_date, end_date=end_date,
    )
    logger.info(f"Movers: {len(movers)}")

    # Group by date
    movers_by_date = defaultdict(list)
    for m in movers:
        sym, d = m[0], m[1]
        prev_close = m[2] if len(m) > 2 else 0.0
        movers_by_date[d].append((sym, d, prev_close))

    # Pre-filter dates (skip fridays per config)
    skip_fridays = bool(cfg.get("trading", {}).get("skip_fridays", False))
    filtered_dates = sorted(d for d in movers_by_date.keys()
                           if not (skip_fridays and d.weekday() == 4))

    # Pre-load bars
    logger.info("Pre-loading bars...")
    symbol_dates = []
    for td in filtered_dates:
        for sym, d, _ in movers_by_date[td]:
            symbol_dates.append((sym, d.isoformat()))
    bulk = db.get_intraday_bars_bulk(symbol_dates)
    all_bars = {k: pd.DataFrame(v) for k, v in bulk.items()}

    # Prev day bars for MACD warm-up
    prev_symbol_dates = set()
    for td in filtered_dates:
        prev = _get_previous_trading_date(td)
        if prev:
            for sym, d, _ in movers_by_date[td]:
                prev_symbol_dates.add((sym, prev.isoformat()))
    prev_bulk = db.get_intraday_bars_bulk(list(prev_symbol_dates))
    all_prev_bars = {k: pd.DataFrame(v) for k, v in prev_bulk.items()}
    logger.info(f"Loaded {len(all_bars)} bar sets + {len(all_prev_bars)} prev-day sets")

    # Create market regime filters
    regime_cfg = cfg.get("trading", {}).get("market_regime", {})
    regime_base = MarketRegimeFilter(
        enabled=bool(regime_cfg.get("enabled", False)),
        vol_threshold=float(regime_cfg.get("vol_threshold", 1.5)),
        sma_period=int(regime_cfg.get("sma_period", 50)),
        min_spy_volume_ratio=float(regime_cfg.get("min_spy_volume_ratio", 0.70)),
        thin_liquidity_breakout_vol_ratio=float(regime_cfg.get("thin_liquidity_breakout_vol_ratio", 2.0)),
        sma_slope_filter=False,
    )
    regime_slope = MarketRegimeFilter(
        enabled=bool(regime_cfg.get("enabled", False)),
        vol_threshold=float(regime_cfg.get("vol_threshold", 1.5)),
        sma_period=int(regime_cfg.get("sma_period", 50)),
        min_spy_volume_ratio=float(regime_cfg.get("min_spy_volume_ratio", 0.70)),
        thin_liquidity_breakout_vol_ratio=float(regime_cfg.get("thin_liquidity_breakout_vol_ratio", 2.0)),
        sma_slope_filter=True,
    )

    # Load SPY bars for regime
    sma_period = int(regime_cfg.get("sma_period", 50))
    spy_lookback = int(sma_period * 1.5) + 14
    spy_bars_raw = fetch_daily_bars_cached(['SPY'], start_date - timedelta(days=spy_lookback), end_date, client, db)
    spy_bars = spy_bars_raw.get('SPY', [])
    regime_base.load_spy_bars(spy_bars)
    regime_slope.load_spy_bars(spy_bars)

    # =========================================================================
    # 6 CONFIGS
    # =========================================================================
    configs = {
        'BASELINE': {
            'runner_kwargs': {},
            'regime': regime_base,
        },
        'MIN_DIST': {
            'runner_kwargs': {'min_stop_distance': 0.09},
            'regime': regime_base,
        },
        'SMA_SLOPE': {
            'runner_kwargs': {},
            'regime': regime_slope,
        },
        'VOL_DEAD': {
            'runner_kwargs': {'vol_dead_zone_enabled': True},
            'regime': regime_base,
        },
        'NO_POP': {
            'runner_kwargs': {'no_pop_exit_bars': 5, 'no_pop_exit_min_pct': 0.005},
            'regime': regime_base,
        },
        'COMBINED': {
            'runner_kwargs': {
                'min_stop_distance': 0.09,
                'vol_dead_zone_enabled': True,
                'no_pop_exit_bars': 5,
                'no_pop_exit_min_pct': 0.005,
            },
            'regime': regime_slope,
        },
    }

    all_metrics = []
    all_dfs = {}

    for label, conf in configs.items():
        logger.info(f"Running {label}...")
        runner = BacktestRunner(**conf['runner_kwargs'])
        results = run_config(
            label, runner, conf['regime'],
            movers_by_date, filtered_dates,
            all_bars, all_prev_bars, db,
        )
        df = results_to_df(results)
        csv_path = f"backtest_{label.lower()}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"  Saved {csv_path} ({len(df)} trades)")

        metrics = evaluate(label, df)
        all_metrics.append(metrics)
        all_dfs[label] = df

    # =========================================================================
    # COMPARISON TABLE
    # =========================================================================
    print()
    print("=" * 120)
    print("FILTER COMPARISON — 15 Months (Jan 2025 — Mar 2026)")
    print("=" * 120)
    print(f"{'Config':<12} {'Trades':>6} {'WR':>6} {'P&L':>12} {'AvgW':>8} {'AvgL':>8} {'PF':>6} {'Sharpe':>7} {'MaxDD':>10} {'LM':>3}")
    print("-" * 120)
    for m in all_metrics:
        print(f"{m['label']:<12} {m['trades']:>6} {m['wr']:>5.1f}% ${m['pnl']:>+10,.0f} ${m['avg_win']:>+6,.0f} ${m['avg_loss']:>+6,.0f} {m['pf']:>5.2f} {m['sharpe']:>6.2f} ${m['dd']:>+8,.0f} {m['lm']:>3}")
    print()

    # =========================================================================
    # MONTH-BY-MONTH COMPARISON
    # =========================================================================
    print("=" * 120)
    print("MONTH-BY-MONTH P&L")
    print("=" * 120)

    # Get all months
    all_months = set()
    for label, df in all_dfs.items():
        if len(df) > 0:
            months = pd.to_datetime(df['date']).dt.to_period('M').unique()
            all_months.update(months)
    all_months = sorted(all_months)

    header = f"{'Month':<10}"
    for label in configs.keys():
        header += f" {label:>12}"
    print(header)
    print("-" * 120)

    for mo in all_months:
        row = f"{str(mo):<10}"
        for label in configs.keys():
            df = all_dfs[label]
            if len(df) == 0:
                row += f" {'$0':>12}"
                continue
            m_pnl = df[df['date'].apply(lambda x: pd.Period(x, 'M')) == mo]['pnl'].sum()
            row += f" ${m_pnl:>+10,.0f}"
        print(row)

    # Totals
    row = f"{'TOTAL':<10}"
    for label in configs.keys():
        df = all_dfs[label]
        row += f" ${df['pnl'].sum():>+10,.0f}"
    print("-" * 120)
    print(row)
    print()

    # =========================================================================
    # COMBINED MONTH-BY-MONTH DETAIL
    # =========================================================================
    print("=" * 120)
    print("COMBINED — MONTH-BY-MONTH DETAIL")
    print("=" * 120)
    df_c = all_dfs.get('COMBINED', pd.DataFrame())
    if len(df_c) > 0:
        monthly = df_c.groupby(pd.to_datetime(df_c['date']).dt.to_period('M'))['pnl'].sum()
        cum = 0
        for mo in sorted(monthly.index):
            mt = df_c[df_c['date'].apply(lambda x: pd.Period(x, 'M')) == mo]
            cum += monthly[mo]
            w = (mt['pnl'] > 0).sum()
            mc = mt.sort_values('date')['pnl'].cumsum()
            dd = (mc - mc.cummax()).min()
            flag = ' <<< LOSS' if monthly[mo] < 0 else ''
            print(f"  {mo}  {len(mt):>3}t  {w:>2}W/{len(mt)-w:>2}L  WR {w/len(mt)*100:>5.1f}%  "
                  f"P&L ${monthly[mo]:>+10,.0f}  Cum ${cum:>+10,.0f}  DD ${dd:>+8,.0f}{flag}")


if __name__ == "__main__":
    main()
