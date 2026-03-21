"""
Batch backtester for momentum day trading strategy.

Scans the universe for stocks with 10%+ intraday moves in a date range,
runs backtests on each qualifying (symbol, date) pair, and produces a
CSV report for TradingView validation.

Usage:
    python batch_backtest.py
    python batch_backtest.py --start 2026-03-01 --end 2026-03-13
    python batch_backtest.py --verbose
"""

import argparse
import csv
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone, date, timedelta
from typing import List, Tuple, Dict, Optional

import pandas as pd
import pytz
from dotenv import load_dotenv

from backtest import BacktestRunner, BacktestResult
from data_sources.alpaca_client import AlpacaClient, AlpacaAPIError
from persistence.database import Database, get_database

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INTRADAY_MOVE_THRESHOLD = 0.10  # 10% (high - low) / low
ET = pytz.timezone('US/Eastern')


def _market_hours_utc(trade_date: date) -> tuple:
    """
    Convert 09:30-16:00 ET market hours to UTC for a given date.

    Handles EDT/EST automatically via pytz — no hardcoded UTC offsets.

    Args:
        trade_date: The trading date

    Returns:
        Tuple of (market_open_utc, market_close_utc) as datetime objects
    """
    open_et = ET.localize(datetime(trade_date.year, trade_date.month, trade_date.day, 9, 30))
    close_et = ET.localize(datetime(trade_date.year, trade_date.month, trade_date.day, 16, 0))
    return open_et.astimezone(timezone.utc), close_et.astimezone(timezone.utc)
DEFAULT_START = "2026-03-01"
DEFAULT_END = "2026-03-13"
CSV_OUTPUT = "backtest_results_march_2026.csv"

CSV_HEADERS = [
    "symbol", "date", "entry_time_et", "entry_price", "stop_loss",
    "target", "shares", "exit_time_et", "exit_price", "exit_reason",
    "pnl", "pnl_pct",
    "partial_taken", "partial_price", "partial_shares", "partial_pnl",
]


# ---------------------------------------------------------------------------
# Step 1: Find 10%+ intraday movers
# ---------------------------------------------------------------------------


def find_big_movers(
    daily_bars: Dict[str, List[Dict]],
    threshold: float = INTRADAY_MOVE_THRESHOLD,
    universe_dict: Optional[Dict] = None,
    price_min: float = 0.0,
    price_max: float = 0.0,
    float_max: int = 0,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> List[Tuple[str, date]]:
    """
    Filter daily bars for (symbol, date) pairs matching scanner criteria.

    Applies price and float filters at the daily level. Relative volume
    is NOT filtered here — it's checked at entry time inside BacktestRunner
    using cumulative volume (matching how the live scanner works).

    Args:
        daily_bars: Dict mapping symbol -> list of daily bar dicts
        threshold: Minimum (high-low)/low ratio (default 0.10 = 10%)
        universe_dict: Dict mapping symbol -> universe record (for float)
        price_min: Minimum price filter (0 = disabled)
        price_max: Maximum price filter (0 = disabled)
        float_max: Maximum float shares (0 = disabled)

    Returns:
        Sorted list of (symbol, date) tuples qualifying for backtest
    """
    universe_dict = universe_dict or {}
    movers = []
    skipped_price = 0
    skipped_float = 0

    skipped_direction = 0

    for symbol, bars in daily_bars.items():
        uni = universe_dict.get(symbol, {})
        sym_float = uni.get('float_shares')

        # Float filter (applied once per symbol)
        if float_max > 0 and sym_float and sym_float > float_max:
            skipped_float += len([b for b in bars if b['low'] > 0 and (b['high'] - b['low']) / b['low'] >= threshold])
            continue

        # Need prev close for direction check — build lookup from consecutive bars
        prev_close_map = {}
        sorted_bars = sorted(bars, key=lambda b: str(b['date']))
        for j in range(1, len(sorted_bars)):
            prev_close_map[str(sorted_bars[j]['date'])] = sorted_bars[j - 1]['close']

        for bar in bars:
            low = bar['low']
            high = bar['high']
            if low <= 0:
                continue
            move = (high - low) / low
            if move < threshold:
                continue

            # Direction filter: high must be 10%+ above prev close (gap UP).
            # Matches live scanner: (current_price - prev_close) / prev_close >= 10%.
            # Without this, crashing stocks with wide ranges qualify as "movers".
            prev_close = prev_close_map.get(str(bar['date']))
            if prev_close and prev_close > 0:
                upside = (high - prev_close) / prev_close
                if upside < threshold:
                    skipped_direction += 1
                    continue

            # Price filter: use closing price as proxy for tradeable range
            bar_close = bar.get('close', 0)
            if price_min > 0 and bar_close < price_min:
                skipped_price += 1
                continue
            if price_max > 0 and bar_close > price_max:
                skipped_price += 1
                continue

            bar_date = bar['date'] if isinstance(bar['date'], date) else date.fromisoformat(str(bar['date']))
            # Only include movers within the requested date range
            # (daily_bars may include lookback days for prev_close computation)
            if start_date and bar_date < start_date:
                continue
            if end_date and bar_date > end_date:
                continue
            movers.append((symbol, bar_date, prev_close or 0.0))
            logger.debug(
                f"  {symbol} {bar_date}: move {move:.1%} "
                f"(low=${low:.2f}, high=${high:.2f})"
            )

    movers.sort(key=lambda x: (x[1], x[0]))
    logger.info(
        f"Found {len(movers)} symbol/date pairs with {threshold:.0%}+ upside move "
        f"(filtered out {skipped_price + skipped_float + skipped_direction}: "
        f"{skipped_price} price, {skipped_float} float, {skipped_direction} direction)"
    )
    return movers


# ---------------------------------------------------------------------------
# Step 1.5: Cached daily bars fetching
# ---------------------------------------------------------------------------


def fetch_daily_bars_cached(
    symbols: List[str],
    start_date: date,
    end_date: date,
    client: AlpacaClient,
    db: Database,
) -> Dict[str, List[Dict]]:
    """
    Fetch daily bars with DB caching — only hits API for uncached symbols.

    Args:
        symbols: List of stock symbols
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        client: AlpacaClient for API fetches
        db: Database for caching

    Returns:
        Dict mapping symbol -> list of daily bar dicts
    """
    start_str = start_date.isoformat()
    end_str = end_date.isoformat()

    # Check which symbols already have cached data
    cached_symbols = db.get_cached_daily_bar_symbols(start_str, end_str)
    uncached = [s for s in symbols if s not in cached_symbols]

    logger.info(
        f"Daily bars: {len(cached_symbols)} cached, "
        f"{len(uncached)} need API fetch"
    )

    # Fetch uncached from API and store
    if uncached:
        logger.info(f"Fetching daily bars from API for {len(uncached)} symbols...")
        api_bars = client.get_daily_bars_range(uncached, start_date, end_date)

        # Flatten for DB storage
        flat_bars = []
        for symbol, bars in api_bars.items():
            for bar in bars:
                flat_bars.append({
                    'symbol': symbol,
                    'date': bar['date'].isoformat() if isinstance(bar['date'], date) else bar['date'],
                    'open': bar['open'],
                    'high': bar['high'],
                    'low': bar['low'],
                    'close': bar['close'],
                    'volume': bar['volume'],
                })
        db.save_daily_bars(flat_bars)
        logger.info(f"Cached {len(flat_bars)} daily bars to DB")

    # Return all from cache (now includes freshly fetched)
    all_bars = db.get_daily_bars_cached(symbols, start_str, end_str)
    logger.info(f"Total: {len(all_bars)} symbols with daily bar data")
    return all_bars


# ---------------------------------------------------------------------------
# Step 2: Backtest each qualifying day (with 1-min bar caching)
# ---------------------------------------------------------------------------


def get_1min_bars_cached(
    symbol: str,
    trade_date: date,
    client: AlpacaClient,
    db: Database,
) -> pd.DataFrame:
    """
    Get 1-min bars for a symbol/date, using DB cache first.

    Args:
        symbol: Stock symbol
        trade_date: Trade date
        client: AlpacaClient for API fetch if not cached
        db: Database for cache

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    date_str = trade_date.isoformat()

    # Check cache
    cached = db.get_intraday_bars_cached(symbol, date_str)
    if cached:
        logger.debug(f"Cache hit: {len(cached)} 1-min bars for {symbol} on {date_str}")
        return pd.DataFrame(cached)

    # Fetch from API — use DST-safe ET→UTC conversion
    market_open, market_close = _market_hours_utc(trade_date)

    bars = client.get_historical_1min_bars(symbol, market_open, market_close)

    # Cache the results
    if not bars.empty:
        bar_records = bars.to_dict('records')
        db.save_intraday_bars(symbol, date_str, bar_records)
        logger.debug(f"Cached {len(bar_records)} 1-min bars for {symbol} on {date_str}")

    return bars


def run_batch_backtest(
    movers: List[Tuple[str, date]],
    client: AlpacaClient,
    runner: BacktestRunner,
    db: Optional[Database] = None,
    universe_dict: Optional[Dict] = None,
    volume_profiles: Optional[Dict[str, Dict[str, int]]] = None,
    market_regime: Optional['MarketRegimeFilter'] = None,
    max_consecutive_losses: int = 0,
    max_trades_per_day: int = 0,
) -> List[BacktestResult]:
    """
    Run backtests on all qualifying (symbol, date) pairs.

    Fetches 1-min bars (cached) for each pair, runs the backtest, collects results.
    API errors are logged and skipped (never abort the batch).

    When market_regime is provided, entire dates are skipped if SPY regime is bearish.
    When circuit_breaker_dd > 0, tracks intraday drawdown per date and skips
    trades after drawdown threshold is hit.

    Args:
        movers: List of (symbol, date) pairs to backtest
        client: AlpacaClient for fetching historical bars
        runner: BacktestRunner instance
        db: Database for 1-min bar caching (optional, fetches without caching if None)
        universe_dict: Dict mapping symbol -> universe record
        volume_profiles: Dict mapping symbol -> {bucket: avg_volume} for bucket rvol
        market_regime: Optional MarketRegimeFilter for SPY regime check
        circuit_breaker_dd: Drawdown threshold in dollars (0 = disabled)
        circuit_breaker_pause: Number of trades to skip when CB triggers

    Returns:
        List of BacktestResult objects (one per successful run)
    """
    from trading.market_regime import MarketRegimeFilter  # avoid circular at module level

    results = []
    total = len(movers)

    # Group movers by date for regime check and circuit breaker tracking
    # Movers are (symbol, date, prev_close) tuples
    movers_by_date: Dict[date, List[tuple]] = defaultdict(list)
    for mover in movers:
        sym, d = mover[0], mover[1]
        prev_close = mover[2] if len(mover) > 2 else 0.0
        movers_by_date[d].append((sym, d, prev_close))

    idx = 0
    regime_skipped = 0
    cb_skipped = 0
    max_trades_skipped = 0
    friday_skipped = 0

    # Load skip_fridays from config
    from config import Config
    _cfg = Config._load_yaml_only()
    skip_fridays = bool(_cfg.get("trading", {}).get("skip_fridays", False))

    # Resolve max_trades_per_day: explicit param > regime attr > 0 (disabled)
    effective_max_trades = max_trades_per_day
    if effective_max_trades <= 0 and market_regime:
        effective_max_trades = getattr(market_regime, 'max_trades_per_day', 0)


    for trade_date in sorted(movers_by_date.keys()):
        # --- Friday filter ---
        if skip_fridays and trade_date.weekday() == 4:
            n_skip = len(movers_by_date[trade_date])
            friday_skipped += n_skip
            idx += n_skip
            logger.info(f"FRIDAY SKIP {trade_date}: skipping {n_skip} symbols")
            continue

        # --- Market regime filter ---
        if market_regime and not market_regime.is_regime_ok(trade_date):
            info = market_regime.get_regime_info(trade_date)
            vol_str = f"{info['vol_5d']:.2f}%" if info['vol_5d'] is not None else "N/A"
            svr_str = f"{info['spy_volume_ratio']:.2f}" if info.get('spy_volume_ratio') is not None else "N/A"
            n_skip = len(movers_by_date[trade_date])
            regime_skipped += n_skip
            idx += n_skip
            logger.warning(
                f"REGIME SKIP {trade_date}: vol_5d={vol_str}, SPY_vol_ratio={svr_str} "
                f"— skipping {n_skip} symbols"
            )
            continue

        # --- Thin liquidity: tighten breakout volume requirement (H5 OR filter) ---
        if market_regime and market_regime.is_thin_liquidity(trade_date):
            runner._min_breakout_vol_override = market_regime.get_min_breakout_volume_ratio(
                trade_date, default=0
            )
        else:
            runner._min_breakout_vol_override = 0  # disabled on normal days

        # --- Consecutive loss tracking (reset per date) ---
        consec_losses = 0
        stopped_for_day = False
        date_trade_count = 0

        for mover_tuple in movers_by_date[trade_date]:
            symbol = mover_tuple[0]
            prev_close = mover_tuple[2] if len(mover_tuple) > 2 else 0.0
            idx += 1
            date_str = trade_date.isoformat()

            # Consecutive loss limit
            if stopped_for_day:
                cb_skipped += 1
                continue

            # Max trades per day cap
            if effective_max_trades > 0 and date_trade_count >= effective_max_trades:
                max_trades_skipped += 1
                logger.info(f"[{idx}/{total}] {symbol} {date_str} — max trades/day ({effective_max_trades}) reached, skipping")
                continue

            try:
                if db:
                    bars = get_1min_bars_cached(symbol, trade_date, client, db)
                else:
                    market_open, market_close = _market_hours_utc(trade_date)
                    bars = client.get_historical_1min_bars(symbol, market_open, market_close)

                if bars.empty:
                    logger.warning(f"[{idx}/{total}] {symbol} {date_str} — no bars, skipping")
                    continue

                avg_vol = None
                if universe_dict:
                    uni = universe_dict.get(symbol, {})
                    avg_vol = uni.get('avg_daily_volume') or uni.get('avg_volume_daily')

                vol_profile = volume_profiles.get(symbol) if volume_profiles else None
                result = runner.run(symbol, bars, date_str,
                                    avg_daily_volume=avg_vol,
                                    volume_profile=vol_profile,
                                    prev_close=prev_close if prev_close > 0 else None)
                results.append(result)

                # Track trades for max trades per day cap
                date_trade_count += len(result.trades_simulated)

                # Verbose progress line
                n_patterns = result.patterns_detected
                n_trades = len(result.trades_simulated)
                pnl = result.summary_pnl
                logger.info(
                    f"[{idx}/{total}] {symbol} {date_str} — "
                    f"{n_patterns} patterns, {n_trades} trade(s), "
                    f"P&L ${pnl:+.2f}"
                )

                # --- Consecutive loss tracking ---
                if max_consecutive_losses > 0:
                    for trade in result.trades_simulated:
                        if trade.pnl > 0:
                            consec_losses = 0
                        else:
                            consec_losses += 1
                            if consec_losses >= max_consecutive_losses:
                                stopped_for_day = True
                                logger.warning(
                                    f"CONSECUTIVE LOSS LIMIT on {date_str}: "
                                    f"{consec_losses} losses in a row — done for day"
                                )

            except AlpacaAPIError as e:
                logger.error(f"[{idx}/{total}] {symbol} {date_str} — API error: {e}, skipping")
            except Exception as e:
                logger.error(f"[{idx}/{total}] {symbol} {date_str} — unexpected error: {e}, skipping")

    if friday_skipped > 0:
        logger.info(f"Friday filter skipped {friday_skipped} symbol/date pairs")
    if regime_skipped > 0:
        logger.info(f"Regime filter skipped {regime_skipped} symbol/date pairs")
    if cb_skipped > 0:
        logger.info(f"Circuit breaker skipped {cb_skipped} symbol/date pairs")
    if max_trades_skipped > 0:
        logger.info(f"Max trades/day skipped {max_trades_skipped} symbol/date pairs")
    logger.info(f"Batch backtest complete: {len(results)}/{total} runs succeeded")
    return results


# ---------------------------------------------------------------------------
# Step 2b: Parallel batch backtest (multiprocessing for cached re-runs)
# ---------------------------------------------------------------------------


def _backtest_worker(args: Tuple) -> Optional[dict]:
    """
    Worker function for parallel backtest processing.

    Each worker creates its own Database connection (processes don't share state).
    Returns a serializable dict instead of BacktestResult (for pickling).

    Args:
        args: Tuple of (symbol, trade_date_iso, db_path) or
              (symbol, trade_date_iso, db_path, prev_close, min_breakout_vol_override)

    Returns:
        Serializable dict with backtest results, or None on error
    """
    # Unpack args — supports both legacy 3-tuple and extended 5-tuple
    if len(args) == 5:
        symbol, trade_date_iso, db_path, prev_close, min_bv_override = args
    else:
        symbol, trade_date_iso, db_path = args
        prev_close = 0.0
        min_bv_override = 0

    try:
        from persistence.database import Database
        from backtest import BacktestRunner

        # Suppress verbose logging in workers — major speedup
        logging.getLogger('trading.pattern_detector').setLevel(logging.WARNING)
        logging.getLogger('backtest').setLevel(logging.WARNING)

        db = Database(db_path=db_path)
        try:
            cached = db.get_intraday_bars_cached(symbol, trade_date_iso)
            if not cached:
                return None

            bars = pd.DataFrame(cached)
            if bars.empty:
                return None

            # Look up avg_daily_volume for cumulative rvol check
            uni = db.get_universe_stock(symbol)
            avg_vol = uni.get('avg_volume_daily') if uni else None

            # Look up volume profile for bucket rvol check
            vol_profile = db.get_volume_profile(symbol)

            runner = BacktestRunner()  # uses from_config() for all settings
            runner._min_breakout_vol_override = min_bv_override
            result = runner.run(symbol, bars, trade_date_iso,
                                avg_daily_volume=avg_vol,
                                volume_profile=vol_profile,
                                prev_close=prev_close if prev_close > 0 else None)

            return _serialize_result(result)
        finally:
            db.close()

    except Exception as e:
        # Log but don't crash the worker pool
        return None


def _serialize_result(result: 'BacktestResult') -> dict:
    """
    Serialize a BacktestResult to a picklable dict.

    Args:
        result: BacktestResult to serialize

    Returns:
        Dict suitable for cross-process transfer
    """
    trades = []
    for t in result.trades_simulated:
        trade_dict = {
            'symbol': t.symbol,
            'entry_time': t.entry_time,
            'entry_price': t.entry_price,
            'stop_loss': t.stop_loss,
            'take_profit': t.take_profit,
            'shares': t.shares,
            'exit_time': t.exit_time,
            'exit_price': t.exit_price,
            'exit_reason': t.exit_reason,
            'pnl': t.pnl,
            'pnl_pct': t.pnl_pct,
            'bars_held': t.bars_held,
            'entry_bar_open': t.entry_bar_open,
            'entry_bar_high': t.entry_bar_high,
            'entry_bar_low': t.entry_bar_low,
            'entry_bar_close': t.entry_bar_close,
            'entry_bar_volume': t.entry_bar_volume,
            'partial_exit_taken': t.partial_exit_taken,
            'partial_exit_price': t.partial_exit_price,
            'partial_shares': t.partial_shares,
            'partial_pnl': t.partial_pnl,
        }
        if t.plan:
            trade_dict['plan'] = {
                'risk_per_share': t.plan.risk_per_share,
                'reward_per_share': t.plan.reward_per_share,
                'risk_reward_ratio': t.plan.risk_reward_ratio,
                'total_risk': t.plan.total_risk,
            }
            if t.plan.pattern:
                p = t.plan.pattern
                trade_dict['pattern'] = {
                    'pole_gain_pct': p.pole_gain_pct,
                    'retracement_pct': p.retracement_pct,
                    'pullback_candle_count': p.pullback_candle_count,
                    'avg_pole_volume': p.avg_pole_volume,
                    'avg_flag_volume': p.avg_flag_volume,
                    'pole_height': p.pole_height,
                    'flag_low': p.flag_low,
                    'flag_high': p.flag_high,
                    'breakout_level': p.breakout_level,
                    'pole_start_idx': p.pole_start_idx,
                    'pole_end_idx': p.pole_end_idx,
                    'flag_start_idx': p.flag_start_idx,
                    'flag_end_idx': p.flag_end_idx,
                    'pole_low': p.pole_low,
                    'pole_high': p.pole_high,
                }
        trades.append(trade_dict)

    return {
        'symbol': result.symbol,
        'trade_date': result.trade_date,
        'total_bars': result.total_bars,
        'patterns_detected': result.patterns_detected,
        'trades': trades,
    }


def _reconstruct_result(result_dict: dict) -> BacktestResult:
    """
    Reconstruct a BacktestResult from a serialized dict.

    Args:
        result_dict: Dict from _backtest_worker

    Returns:
        BacktestResult with full object graph
    """
    from backtest import SimulatedTrade, BacktestResult
    from trading.pattern_detector import BullFlagPattern
    from trading.trade_planner import TradePlan

    trades = []
    for td in result_dict['trades']:
        pattern = None
        plan = None

        if 'pattern' in td:
            pd_data = td['pattern']
            pattern = BullFlagPattern(
                symbol=td['symbol'],
                **pd_data,
            )

        if 'plan' in td:
            plan = TradePlan(
                symbol=td['symbol'],
                entry_price=td['entry_price'],
                stop_loss_price=td['stop_loss'],
                take_profit_price=td['take_profit'],
                shares=td['shares'],
                pattern=pattern,
                **td['plan'],
            )

        trade = SimulatedTrade(
            symbol=td['symbol'],
            entry_time=td['entry_time'],
            entry_price=td['entry_price'],
            stop_loss=td['stop_loss'],
            take_profit=td['take_profit'],
            shares=td['shares'],
            exit_time=td['exit_time'],
            exit_price=td['exit_price'],
            exit_reason=td['exit_reason'],
            pnl=td['pnl'],
            pnl_pct=td['pnl_pct'],
            bars_held=td['bars_held'],
            plan=plan,
            entry_bar_open=td.get('entry_bar_open'),
            entry_bar_high=td.get('entry_bar_high'),
            entry_bar_low=td.get('entry_bar_low'),
            entry_bar_close=td.get('entry_bar_close'),
            entry_bar_volume=td.get('entry_bar_volume'),
        )
        trades.append(trade)

    result = BacktestResult(
        symbol=result_dict['symbol'],
        trade_date=result_dict['trade_date'],
        total_bars=result_dict['total_bars'],
        patterns_detected=result_dict['patterns_detected'],
        trades_simulated=trades,
    )
    return result


def run_batch_backtest_parallel(
    movers: List[Tuple[str, date]],
    db_path: str = "data/onemil.db",
    max_workers: int = 4,
    market_regime: Optional['MarketRegimeFilter'] = None,
    max_consecutive_losses: int = 0,
) -> List[BacktestResult]:
    """
    Run backtests in parallel using multiprocessing.

    Pre-computes regime filter decisions, then distributes all work to a
    ProcessPoolExecutor. ~4x faster than sequential on 4 cores, plus
    suppressed debug logging in workers for additional ~2-3x speedup.

    Args:
        movers: List of (symbol, date, prev_close) tuples to backtest
        db_path: Path to SQLite database
        max_workers: Number of parallel processes (default 4)
        market_regime: Optional MarketRegimeFilter for SPY regime check
        max_consecutive_losses: Stop trading after N consecutive losses per day
            (0 = disabled). NOTE: only approximated in parallel mode —
            results may differ slightly from sequential when enabled.

    Returns:
        List of BacktestResult objects
    """
    from concurrent.futures import ProcessPoolExecutor
    from collections import defaultdict

    total = len(movers)

    # Load skip_fridays from config
    from config import Config
    _cfg = Config._load_yaml_only()
    skip_fridays = bool(_cfg.get("trading", {}).get("skip_fridays", False))

    # Group movers by date for regime/friday pre-filtering
    movers_by_date: Dict[date, List[tuple]] = defaultdict(list)
    for mover in movers:
        sym, d = mover[0], mover[1]
        prev_close = mover[2] if len(mover) > 2 else 0.0
        movers_by_date[d].append((sym, d, prev_close))

    # Pre-compute regime decisions and thin liquidity overrides
    filtered_items = []
    regime_skipped = 0
    friday_skipped = 0

    for trade_date in sorted(movers_by_date.keys()):
        date_movers = movers_by_date[trade_date]

        # Friday filter
        if skip_fridays and trade_date.weekday() == 4:
            friday_skipped += len(date_movers)
            continue

        # Regime filter
        if market_regime and not market_regime.is_regime_ok(trade_date):
            regime_skipped += len(date_movers)
            continue

        # Thin liquidity: compute breakout volume override for this date
        min_bv_override = 0
        if market_regime and market_regime.is_thin_liquidity(trade_date):
            min_bv_override = market_regime.get_min_breakout_volume_ratio(
                trade_date, default=0
            )

        for sym, d, prev_close in date_movers:
            filtered_items.append(
                (sym, d.isoformat(), db_path, prev_close, min_bv_override)
            )

    if friday_skipped:
        logger.info(f"Friday filter skipped {friday_skipped} symbol/date pairs")
    if regime_skipped:
        logger.info(f"Regime filter skipped {regime_skipped} symbol/date pairs")

    filtered_total = len(filtered_items)
    logger.info(
        f"Parallel batch: {filtered_total} movers to process "
        f"(of {total} total), {max_workers} workers"
    )

    results = []
    completed = 0
    import time
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for result_dict in executor.map(
            _backtest_worker, filtered_items, chunksize=100
        ):
            completed += 1
            if result_dict is None:
                continue

            result = _reconstruct_result(result_dict)
            results.append(result)

            if completed % 1000 == 0 or completed == filtered_total:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                n_trades = sum(len(r.trades_simulated) for r in results)
                logger.info(
                    f"[{completed}/{filtered_total}] {n_trades} trades, "
                    f"{rate:.0f} symbols/sec, "
                    f"ETA {(filtered_total - completed) / rate:.0f}s"
                    if rate > 0 else
                    f"[{completed}/{filtered_total}] {n_trades} trades"
                )

    elapsed = time.time() - t0
    logger.info(
        f"Parallel batch complete: {len(results)}/{filtered_total} runs, "
        f"{elapsed:.1f}s ({filtered_total / elapsed:.0f} symbols/sec)"
    )

    # Post-hoc consecutive loss filter (approximate — applied per-date after all results)
    if max_consecutive_losses > 0:
        results = _apply_consecutive_loss_filter(results, max_consecutive_losses)

    return results


def run_batch_backtest_fast(
    movers: List[Tuple[str, date]],
    db: 'Database',
    market_regime: Optional['MarketRegimeFilter'] = None,
    max_consecutive_losses: int = 0,
    max_workers: int = 0,
) -> List[BacktestResult]:
    """
    Fastest batch backtest: single-process, pre-loaded bars, minimal logging.

    Pre-loads all 1-min bars from SQLite into memory, then runs all backtests
    in a single process with debug/info logging suppressed. Avoids per-item
    DB overhead and multiprocessing serialization costs.

    ~90+ symbols/sec on cached data (vs 35/sec parallel, 24/sec sequential).

    Args:
        movers: List of (symbol, date, prev_close) tuples
        db: Database instance (shared, single-process)
        market_regime: Optional MarketRegimeFilter
        max_consecutive_losses: Stop after N consecutive losses per day (0=disabled)

    Returns:
        List of BacktestResult objects
    """
    import time
    from collections import defaultdict

    total = len(movers)

    # Load config
    from config import Config
    _cfg = Config._load_yaml_only()
    skip_fridays = bool(_cfg.get("trading", {}).get("skip_fridays", False))

    # Suppress verbose logging for speed — saves ~2-3x
    suppressed_loggers = [
        'trading.pattern_detector', 'backtest', 'trading.trade_planner',
        'trading.indicators', 'persistence.database',
    ]
    original_levels = {}
    for name in suppressed_loggers:
        lg = logging.getLogger(name)
        original_levels[name] = lg.level
        lg.setLevel(logging.ERROR)

    try:
        # Group by date
        movers_by_date: Dict[date, List[tuple]] = defaultdict(list)
        for mover in movers:
            sym, d = mover[0], mover[1]
            prev_close = mover[2] if len(mover) > 2 else 0.0
            movers_by_date[d].append((sym, d, prev_close))

        # Pre-filter dates (regime + friday)
        filtered_dates = []
        regime_skipped = 0
        friday_skipped = 0
        for trade_date in sorted(movers_by_date.keys()):
            if skip_fridays and trade_date.weekday() == 4:
                friday_skipped += len(movers_by_date[trade_date])
                continue
            if market_regime and not market_regime.is_regime_ok(trade_date):
                regime_skipped += len(movers_by_date[trade_date])
                continue
            filtered_dates.append(trade_date)

        # Count filtered items
        filtered_total = sum(len(movers_by_date[d]) for d in filtered_dates)
        if friday_skipped:
            logger.info(f"Friday filter skipped {friday_skipped} symbol/date pairs")
        if regime_skipped:
            logger.info(f"Regime filter skipped {regime_skipped} symbol/date pairs")
        logger.info(f"Fast batch: {filtered_total} items to process (of {total} total)")

        # Pre-load ALL bars into memory via bulk query — eliminates per-item DB reads
        logger.info("Pre-loading all 1-min bars from cache (bulk)...")
        t_preload = time.time()
        symbol_dates = []
        for trade_date in filtered_dates:
            for sym, d, prev_close in movers_by_date[trade_date]:
                symbol_dates.append((sym, d.isoformat()))

        bulk_data = db.get_intraday_bars_bulk(symbol_dates)
        all_bars = {
            key: pd.DataFrame(bars) for key, bars in bulk_data.items()
        }
        preload_elapsed = time.time() - t_preload
        logger.info(
            f"Pre-loaded {len(all_bars)} bar sets in {preload_elapsed:.1f}s"
        )

        # Create single runner instance — reused across all items
        runner = BacktestRunner()
        results = []
        completed = 0
        t0 = time.time()

        for trade_date in filtered_dates:
            # Thin liquidity override
            if market_regime and market_regime.is_thin_liquidity(trade_date):
                runner._min_breakout_vol_override = market_regime.get_min_breakout_volume_ratio(
                    trade_date, default=0
                )
            else:
                runner._min_breakout_vol_override = 0

            consec_losses = 0
            stopped_for_day = False

            for sym, d, prev_close in movers_by_date[trade_date]:
                completed += 1
                date_str = d.isoformat()

                if stopped_for_day:
                    continue

                bars = all_bars.get((sym, date_str))
                if bars is None or bars.empty:
                    continue

                try:
                    result = runner.run(
                        sym, bars, date_str,
                        prev_close=prev_close if prev_close > 0 else None,
                    )
                    results.append(result)

                    # Consecutive loss tracking
                    if max_consecutive_losses > 0:
                        for trade in result.trades_simulated:
                            if trade.pnl > 0:
                                consec_losses = 0
                            else:
                                consec_losses += 1
                                if consec_losses >= max_consecutive_losses:
                                    stopped_for_day = True
                except Exception:
                    pass

                if completed % 2000 == 0:
                    elapsed = time.time() - t0
                    rate = completed / elapsed if elapsed > 0 else 0
                    n_trades = sum(len(r.trades_simulated) for r in results)
                    eta = (filtered_total - completed) / rate if rate > 0 else 0
                    logger.info(
                        f"[{completed}/{filtered_total}] {n_trades} trades, "
                        f"{rate:.0f}/sec, ETA {eta:.0f}s"
                    )

        elapsed = time.time() - t0
        n_trades = sum(len(r.trades_simulated) for r in results)
        logger.info(
            f"Fast batch complete: {n_trades} trades from "
            f"{filtered_total} items in {elapsed:.1f}s "
            f"({filtered_total / elapsed:.0f}/sec)"
        )
        return results

    finally:
        # Restore logging levels
        for name, level in original_levels.items():
            logging.getLogger(name).setLevel(level)


def _fast_worker(args: tuple) -> Optional[dict]:
    """
    Worker for fast parallel backtest — receives pre-loaded bars, no DB needed.

    Args:
        args: (symbol, date_str, bars_df, prev_close, min_bv_override)

    Returns:
        Serializable dict with backtest results, or None on error
    """
    symbol, date_str, bars, prev_close, min_bv_override = args
    try:
        # Suppress logging in workers
        logging.getLogger('trading.pattern_detector').setLevel(logging.ERROR)
        logging.getLogger('backtest').setLevel(logging.ERROR)
        logging.getLogger('trading.trade_planner').setLevel(logging.ERROR)
        logging.getLogger('trading.indicators').setLevel(logging.ERROR)

        from backtest import BacktestRunner
        runner = BacktestRunner()
        runner._min_breakout_vol_override = min_bv_override

        result = runner.run(
            symbol, bars, date_str,
            prev_close=prev_close if prev_close > 0 else None,
        )
        return _serialize_result(result)
    except Exception:
        return None


def _apply_consecutive_loss_filter(
    results: List['BacktestResult'],
    max_consecutive_losses: int,
) -> List['BacktestResult']:
    """
    Post-hoc consecutive loss filter: removes trades that would have been
    skipped in sequential mode after N consecutive losses on the same day.

    This is an approximation — in sequential mode, the order of symbols
    within a day affects which trades are skipped. Here we sort by entry time.

    Args:
        results: All backtest results (unfiltered)
        max_consecutive_losses: Threshold for stopping

    Returns:
        Filtered results with excess trades removed
    """
    from collections import defaultdict

    # Group trades by date
    trades_by_date: Dict[str, List] = defaultdict(list)
    for r in results:
        for t in r.trades_simulated:
            trades_by_date[r.trade_date].append((r, t))

    removed = 0
    for trade_date, trade_pairs in trades_by_date.items():
        # Sort by entry time within the day
        trade_pairs.sort(key=lambda x: x[1].entry_time or datetime.min)

        consec = 0
        for r, t in trade_pairs:
            if consec >= max_consecutive_losses:
                r.trades_simulated.remove(t)
                removed += 1
            elif t.pnl > 0:
                consec = 0
            else:
                consec += 1

    if removed:
        logger.info(f"Consecutive loss filter removed {removed} trades")
    return results


# ---------------------------------------------------------------------------
# Step 3: CSV report + console summary
# ---------------------------------------------------------------------------


def utc_to_et_str(ts: datetime) -> str:
    """Convert UTC datetime to ET string (UTC-4 for EDT March 2026)."""
    if ts is None:
        return ""
    et = ts.replace(tzinfo=None) - __import__('datetime').timedelta(hours=4)
    return et.strftime("%H:%M:%S")


def write_csv_report(results: List[BacktestResult], output_path: str) -> int:
    """
    Write trade-level CSV report from backtest results.

    One row per trade for TradingView validation.

    Args:
        results: List of BacktestResult objects
        output_path: Path for the CSV file

    Returns:
        Number of trade rows written
    """
    trade_count = 0
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADERS)

        for result in results:
            for trade in result.trades_simulated:
                writer.writerow([
                    trade.symbol,
                    result.trade_date,
                    utc_to_et_str(trade.entry_time),
                    f"{trade.entry_price:.2f}",
                    f"{trade.stop_loss:.2f}",
                    f"{trade.take_profit:.2f}",
                    trade.shares,
                    utc_to_et_str(trade.exit_time),
                    f"{trade.exit_price:.2f}" if trade.exit_price else "",
                    trade.exit_reason or "",
                    f"{trade.pnl:.2f}",
                    f"{trade.pnl_pct:.2f}",
                    trade.partial_exit_taken,
                    f"{trade.partial_exit_price:.2f}" if trade.partial_exit_price else "",
                    trade.partial_shares,
                    f"{trade.partial_pnl:.2f}",
                ])
                trade_count += 1

    logger.info(f"CSV report written to {output_path} ({trade_count} trades)")
    return trade_count


def print_summary(
    total_universe: int,
    movers: List[Tuple[str, date]],
    results: List[BacktestResult],
) -> None:
    """
    Print batch backtest summary to console.

    Args:
        total_universe: Number of symbols in active universe
        movers: List of qualifying (symbol, date) pairs
        results: List of BacktestResult objects
    """
    all_trades = [t for r in results for t in r.trades_simulated]
    wins = [t for t in all_trades if t.pnl > 0]
    losses = [t for t in all_trades if t.pnl <= 0]
    total_pnl = sum(t.pnl for t in all_trades)
    win_rate = len(wins) / len(all_trades) * 100 if all_trades else 0.0

    print("\n" + "=" * 70)
    print("  BATCH BACKTEST SUMMARY — March 2026")
    print("=" * 70)
    print(f"  Universe size:           {total_universe} symbols")
    print(f"  Symbol/date combos:      {total_universe * 9} (approx trading days)")
    print(f"  10%+ intraday movers:    {len(movers)}")
    print(f"  Backtests completed:     {len(results)}")
    print("-" * 70)
    print(f"  Total trades taken:      {len(all_trades)}")
    print(f"  Winning trades:          {len(wins)}")
    print(f"  Losing trades:           {len(losses)}")
    print(f"  Win rate:                {win_rate:.1f}%")
    print(f"  Total P&L:              ${total_pnl:+.2f}")
    if wins:
        print(f"  Avg win:                ${sum(t.pnl for t in wins) / len(wins):+.2f}")
    if losses:
        print(f"  Avg loss:               ${sum(t.pnl for t in losses) / len(losses):+.2f}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for batch backtesting."""
    parser = argparse.ArgumentParser(
        description="Batch backtest: scan universe for 10%+ movers and run strategy"
    )
    parser.add_argument(
        "--start", type=str, default=DEFAULT_START,
        help=f"Start date YYYY-MM-DD (default: {DEFAULT_START})"
    )
    parser.add_argument(
        "--end", type=str, default=DEFAULT_END,
        help=f"End date YYYY-MM-DD (default: {DEFAULT_END})"
    )
    parser.add_argument(
        "--output", type=str, default=CSV_OUTPUT,
        help=f"CSV output path (default: {CSV_OUTPUT})"
    )
    parser.add_argument(
        "--monthly", action="store_true",
        help="Run in monthly-chunked mode with parallel processing and rich CSV output"
    )
    parser.add_argument(
        "--workers", type=int, default=2,
        help="Number of parallel month workers (default: 2)"
    )
    parser.add_argument(
        "--scan-workers", type=int, default=0,
        help="Number of parallel worker processes. 0 = auto (cpu_count). "
             "Uses multiprocessing — scales with CPU cores."
    )
    parser.add_argument(
        "--no-parallel", action="store_true",
        help="Disable fast mode (use sequential processing with full logging)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose/debug logging (implies --no-parallel)"
    )
    parser.add_argument(
        "--trailing-stop-r", type=float, default=0.0,
        help="Replace fixed TP with trailing stop N×R below high (0 = disabled)"
    )
    parser.add_argument(
        "--trailing-activate-r", type=float, default=0.0,
        help="Activate trailing stop after +NR from entry (e.g., 2.0)"
    )
    args = parser.parse_args()

    # Auto-detect worker count from CPU cores
    if args.scan_workers <= 0:
        import multiprocessing
        args.scan_workers = multiprocessing.cpu_count()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Load environment
    load_dotenv()
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        logger.error("Missing ALPACA_API_KEY or ALPACA_API_SECRET in environment")
        sys.exit(1)

    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date()

    logger.info(f"Batch backtest: {start_date} to {end_date}")

    # Monthly mode: use MonthlyBacktestRunner for parallel, chunked processing
    if args.monthly:
        from batch.monthly_runner import MonthlyBacktestRunner

        runner = MonthlyBacktestRunner(
            max_workers=args.workers,
            scan_workers=args.scan_workers,
            verbose=args.verbose,
        )
        master_csv = runner.run_all(start_date, end_date, output_dir="backtest_results")
        logger.info(f"Monthly backtest complete: {master_csv}")
        return

    # Standard (non-monthly) mode
    # Step 1: Load universe
    db = get_database()
    universe = db.get_active_universe()
    symbols = [s['symbol'] for s in universe]
    logger.info(f"Loaded {len(symbols)} symbols from active universe")

    if not symbols:
        logger.error("No active symbols in universe — run universe builder first")
        sys.exit(1)

    # Step 2: Fetch daily bars (cached) and find 10%+ movers with scanner filters
    # Fetch from 7 days before start_date so the direction filter has prev_close
    # (need trading day before start, not calendar day — 7 days covers weekends/holidays)
    client = AlpacaClient(api_key=api_key, api_secret=api_secret)
    logger.info("Fetching daily bars for date range (cache-first)...")
    daily_bars = fetch_daily_bars_cached(symbols, start_date - timedelta(days=7), end_date, client, db)
    universe_dict = {s['symbol']: s for s in universe}

    from config import Config
    from trading.market_regime import MarketRegimeFilter

    cfg = Config._load_yaml_only()
    scanner_cfg = cfg.get("scanner", {})
    movers = find_big_movers(
        daily_bars,
        universe_dict=universe_dict,
        price_min=float(scanner_cfg.get("price_min", 2.0)),
        price_max=float(scanner_cfg.get("price_max", 20.0)),
        float_max=int(scanner_cfg.get("float_max", 10_000_000)),
        start_date=start_date,
        end_date=end_date,
    )

    if not movers:
        logger.warning("No symbols with 10%+ intraday move found — nothing to backtest")
        print_summary(len(symbols), movers, [])
        return

    # Step 3: Build market regime filter
    trading_cfg = cfg.get("trading", {})
    regime_cfg = trading_cfg.get("market_regime", {})
    sma_period = int(regime_cfg.get("sma_period", 50))
    spy_lookback_days = int(sma_period * 1.5) + 14
    spy_start = start_date - timedelta(days=spy_lookback_days)
    spy_bars_raw = fetch_daily_bars_cached(['SPY'], spy_start, end_date, client, db)
    spy_bars = spy_bars_raw.get('SPY', [])
    max_trades_per_day = int(trading_cfg.get("max_trades_per_day", 5))
    market_regime = MarketRegimeFilter(
        enabled=bool(regime_cfg.get("enabled", True)),
        vol_threshold=float(regime_cfg.get("vol_threshold", 1.5)),
        sma_period=sma_period,
        max_trades_per_day=max_trades_per_day,
        min_spy_volume_ratio=float(regime_cfg.get("min_spy_volume_ratio", 0.70)),
        thin_liquidity_breakout_vol_ratio=float(regime_cfg.get("thin_liquidity_breakout_vol_ratio", 2.0)),
    )
    market_regime.load_spy_bars(spy_bars)
    max_consec = int(trading_cfg.get("max_consecutive_losses", 2))
    logger.info(
        f"Regime filter: enabled={market_regime.enabled}, "
        f"vol_threshold={market_regime.vol_threshold}%, sma_period={sma_period}, "
        f"min_spy_vol_ratio={market_regime.min_spy_volume_ratio}, "
        f"SPY bars={len(spy_bars)}, max_trades/day={max_trades_per_day}, "
        f"max_consec_losses={max_consec}"
    )

    # Step 4: Run backtests (1-min bars also cached)
    # Fast mode is default; --verbose or --no-parallel disables it
    use_fast = not args.no_parallel and not args.verbose
    if use_fast:
        results = run_batch_backtest_fast(
            movers, db=db,
            market_regime=market_regime,
            max_consecutive_losses=max_consec,
            max_workers=args.scan_workers,
        )
    else:
        runner = BacktestRunner()  # uses from_config() for all settings
        results = run_batch_backtest(
            movers, client, runner, db=db, universe_dict=universe_dict,
            market_regime=market_regime,
            max_consecutive_losses=max_consec,
        )

    # Step 5: Write CSV + print summary
    write_csv_report(results, args.output)
    print_summary(len(symbols), movers, results)


if __name__ == "__main__":
    main()
