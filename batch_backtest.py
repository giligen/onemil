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
from typing import List, Tuple, Dict, Optional, Set

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
    "daily_range_pct",
]

def _trade_to_cache_row(trade) -> Optional[Dict]:
    """Convert a SimulatedTrade to a cache CSV row dict."""
    try:
        return {
            'symbol': trade.symbol,
            'date': str(trade.entry_time.date()) if hasattr(trade.entry_time, 'date') else '',
            'entry_time_et': utc_to_et_str(trade.entry_time),
            'entry_price': f"{trade.entry_price:.2f}",
            'stop_loss': f"{trade.stop_loss:.2f}",
            'target': f"{trade.take_profit:.2f}",
            'shares': trade.shares,
            'exit_time_et': utc_to_et_str(trade.exit_time),
            'exit_price': f"{trade.exit_price:.2f}" if trade.exit_price else "",
            'exit_reason': trade.exit_reason or "",
            'pnl': f"{trade.pnl:.2f}",
            'pnl_pct': f"{trade.pnl_pct:.2f}",
            'partial_taken': trade.partial_exit_taken,
            'partial_price': f"{trade.partial_exit_price:.2f}" if trade.partial_exit_price else "",
            'partial_shares': trade.partial_shares,
            'partial_pnl': f"{trade.partial_pnl:.2f}",
            'daily_range_pct': f"{getattr(trade, '_daily_range_pct', 0):.1f}",
        }
    except Exception as e:
        logger.warning(f"Failed to convert trade to cache row: {e}")
        return None


def _get_bull_flag_cache_path(entry_slip: float, exit_slip: float) -> str:
    """Cache path includes slippage params since they affect trade simulation."""
    e = int(entry_slip * 10000)  # 0.005 → 50
    x = int(exit_slip * 10000)   # 0.003 → 30
    return f"data/bull_flag_cache_e{e}_x{x}.csv"

# Default cache path (for backward compat)
BULL_FLAG_CACHE_PATH = "data/bull_flag_signal_cache.csv"


def load_bull_flag_cache(cache_path: str, start_date: date, end_date: date,
                         **kwargs) -> List[Dict]:
    """Load cached bull flag trades, filter by date range.

    Slippage is baked into cached prices (filename includes slippage params).
    Change slippage → rebuild cache with --build-cache.
    """
    trades = []
    with open(cache_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = row['date']
            if d < str(start_date) or d > str(end_date):
                continue
            row['pnl'] = float(row['pnl'])
            row['pnl_pct'] = float(row['pnl_pct'])
            row['shares'] = int(row['shares'])
            row['entry_price'] = float(row['entry_price'])
            row['exit_price'] = float(row['exit_price']) if row['exit_price'] else 0
            row['daily_range_pct'] = float(row.get('daily_range_pct', 100))
            trades.append(row)
    logger.info(f"Loaded {len(trades)} cached trades from {cache_path} ({start_date} to {end_date})")
    return trades


def filter_bull_flag_trades(
    trades: List[Dict],
    market_regime=None,
    max_trades_per_day: int = 0,
    max_consecutive_losses: int = 0,
    min_daily_range_pct: float = 0,
    universe_symbols: Optional[Set[str]] = None,
    max_concurrent: int = 0,
    daily_loss_limit: float = 0,
    universe_vol_map: Optional[Dict[str, int]] = None,
) -> List[Dict]:
    """Apply regime, max trades/day, consecutive loss, threshold, concurrent, and loss limit filters."""
    from collections import defaultdict
    from datetime import datetime as _dt
    from data_sources.alpaca_client import AlpacaClient

    # Pre-filter: remove leveraged/inverse ETFs (synthetic, not real stocks)
    before_lev = len(trades)
    trades = [t for t in trades if t['symbol'] not in AlpacaClient._LEVERAGED_ETF_SYMBOLS]
    lev_removed = before_lev - len(trades)
    if lev_removed:
        logger.info(f"Leveraged ETF filter: {before_lev} → {len(trades)} trades ({lev_removed} removed)")

    # Pre-filter by universe
    if universe_symbols is not None:
        before = len(trades)
        trades = [t for t in trades if t['symbol'] in universe_symbols]
        logger.info(f"Universe filter: {before} → {len(trades)} trades")

    # Pre-filter by minimum daily volume (avoid illiquid stocks)
    from config import Config as _Cfg
    _cfg_vol = _Cfg._load_yaml_only()
    min_daily_vol = int(_cfg_vol.get("scanner", {}).get("min_daily_volume", 0))
    if min_daily_vol > 0 and universe_vol_map:
        before_vol = len(trades)
        trades = [t for t in trades if universe_vol_map.get(t['symbol'], 0) >= min_daily_vol]
        vol_removed = before_vol - len(trades)
        if vol_removed:
            logger.info(f"Volume filter (>={min_daily_vol:,}): {before_vol} → {len(trades)} trades ({vol_removed} removed)")

    # Apply risk tier scaling to PnL
    tier_cfg = _cfg_vol.get("trading", {}).get("risk_tiers", {})
    if bool(tier_cfg.get("enabled", False)) and universe_vol_map:
        tiers = []
        for prefix in ['tier1', 'tier2', 'tier3']:
            mult = float(tier_cfg.get(f"{prefix}_multiplier", 0))
            if mult > 0:
                tiers.append({
                    'min_price': float(tier_cfg.get(f"{prefix}_min_price", 0)),
                    'max_price': float(tier_cfg.get(f"{prefix}_max_price", 999)),
                    'min_volume': int(tier_cfg.get(f"{prefix}_min_volume", 0)),
                    'max_volume': int(tier_cfg.get(f"{prefix}_max_volume", 999999999)),
                    'multiplier': mult,
                })
        scaled = 0
        for t in trades:
            ep = t['entry_price']
            vol = universe_vol_map.get(t['symbol'], 0)
            for tier in tiers:
                if (tier['min_price'] <= ep < tier['max_price'] and
                        tier['min_volume'] <= vol <= tier['max_volume']):
                    t['pnl'] *= tier['multiplier']
                    t['shares'] = int(t['shares'] * tier['multiplier'])
                    scaled += 1
                    break
        if scaled:
            logger.info(f"Risk tiers: {scaled} trades scaled (of {len(trades)})")

    # Pre-filter by daily range threshold
    if min_daily_range_pct > 0:
        trades = [t for t in trades if float(t.get('daily_range_pct', 100)) >= min_daily_range_pct]

    # Group by date
    by_date = defaultdict(list)
    for t in trades:
        by_date[t['date']].append(t)

    filtered = []
    concurrent_skipped = 0
    loss_limit_skipped = 0

    for d in sorted(by_date):
        td = date.fromisoformat(d)

        # Regime filter
        if market_regime and not market_regime.is_regime_ok(td):
            continue

        day_trades = by_date[d]
        day_count = 0
        consec_losses = 0
        daily_pnl = 0.0
        active_positions = []  # list of exit times

        for t in day_trades:
            if max_trades_per_day > 0 and day_count >= max_trades_per_day:
                break
            if max_consecutive_losses > 0 and consec_losses >= max_consecutive_losses:
                break

            # Daily loss limit
            if daily_loss_limit < 0 and daily_pnl <= daily_loss_limit:
                loss_limit_skipped += 1
                continue

            # Concurrent position limit
            if max_concurrent > 0:
                entry_str = t.get('entry_time_et', '')
                exit_str = t.get('exit_time_et', '')
                if entry_str and exit_str:
                    try:
                        entry_t = _dt.strptime(f"{d} {entry_str}", '%Y-%m-%d %H:%M:%S')
                        # Remove expired positions
                        active_positions = [
                            ex for ex in active_positions if ex > entry_t
                        ]
                        if len(active_positions) >= max_concurrent:
                            concurrent_skipped += 1
                            continue
                        exit_t = _dt.strptime(f"{d} {exit_str}", '%Y-%m-%d %H:%M:%S')
                        active_positions.append(exit_t)
                    except (ValueError, TypeError):
                        pass  # Can't parse times, allow trade

            filtered.append(t)
            day_count += 1
            daily_pnl += t['pnl']

            if t['pnl'] > 0:
                consec_losses = 0
            else:
                consec_losses += 1

    logger.info(f"Filter: {len(trades)} → {len(filtered)} trades "
                f"(regime={'on' if market_regime and market_regime.enabled else 'off'}, "
                f"max_trades={max_trades_per_day}, max_concurrent={max_concurrent}, "
                f"daily_loss_limit=${daily_loss_limit:,.0f}, "
                f"min_range={min_daily_range_pct}%)"
                + (f", {concurrent_skipped} concurrent-skipped" if concurrent_skipped else "")
                + (f", {loss_limit_skipped} loss-limit-skipped" if loss_limit_skipped else ""))
    return filtered


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
    min_dollar_volume: float = 0,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> List[Tuple[str, date]]:
    """
    Filter daily bars for (symbol, date) pairs matching scanner criteria.

    Applies price, float, and dollar volume filters at the daily level.

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

        # Float filter: only exclude stocks with KNOWN float > max.
        # NULL float = unknown = include (many BATS-listed stocks like BEX, CORD).
        if float_max > 0 and sym_float is not None and sym_float > float_max:
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

            # Direction filter: qualify gap-ups OR V-reversals.
            # Gap-up: high >= prev_close * (1 + threshold) — stock UP from yesterday
            # V-reversal: range >= threshold AND close > open — big intraday range, closed green
            # Without either, crashing stocks with wide ranges would qualify.
            prev_close = prev_close_map.get(str(bar['date']))
            bar_close = bar.get('close', 0)
            bar_open = bar.get('open', 0)
            if prev_close and prev_close > 0:
                upside = (high - prev_close) / prev_close
                is_v_reversal = move >= threshold and bar_close > bar_open  # big range + green close
                if upside < threshold and not is_v_reversal:
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

            # Dollar volume filter: close * volume
            if min_dollar_volume > 0:
                bar_vol = bar.get('volume', 0)
                dollar_vol = bar_close * bar_vol
                if dollar_vol < min_dollar_volume:
                    continue

            bar_date = bar['date'] if isinstance(bar['date'], date) else date.fromisoformat(str(bar['date']))
            # Only include movers within the requested date range
            # (daily_bars may include lookback days for prev_close computation)
            if start_date and bar_date < start_date:
                continue
            if end_date and bar_date > end_date:
                continue
            movers.append((symbol, bar_date, prev_close or 0.0, upside if prev_close else move))
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


def _get_previous_trading_date(trade_date: date, movers_by_date: dict = None) -> Optional[date]:
    """
    Get the previous trading date (skipping weekends).

    Args:
        trade_date: Current trading date
        movers_by_date: Optional dict of known trading dates (for fast lookup)

    Returns:
        Previous trading date, or None if unknown
    """
    # Simple approach: go back 1-3 calendar days to skip weekends
    for delta in range(1, 4):
        prev = trade_date - timedelta(days=delta)
        if prev.weekday() < 5:  # Mon-Fri
            return prev
    return None


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

    # UD risk scaling config
    _ud_cfg = _cfg.get("trading", {}).get("ud_risk_scaling", {})
    ud_scaling_enabled = bool(_ud_cfg.get("enabled", False))
    ud_threshold = float(_ud_cfg.get("ud_threshold", 1.2))
    ud_scale_factor = float(_ud_cfg.get("scale_factor", 0.5))

    # Resolve max_trades_per_day: explicit param > regime attr > 0 (disabled)
    effective_max_trades = max_trades_per_day
    if effective_max_trades <= 0 and market_regime:
        effective_max_trades = getattr(market_regime, 'max_trades_per_day', 0)

    # SPY MACD afternoon cutoff (same logic as fast path)
    spy_cutoff_cfg = _cfg.get("trading", {}).get("spy_macd_cutoff", {})
    spy_cutoff_enabled = bool(spy_cutoff_cfg.get("enabled", False))
    spy_cutoff_time = None
    if spy_cutoff_enabled:
        cutoff_str = spy_cutoff_cfg.get("cutoff_time", "11:30")
        _ch, _cm = cutoff_str.split(':')
        spy_cutoff_time = (int(_ch), int(_cm))
        logger.info(
            f"SPY MACD cutoff enabled: block after "
            f"{spy_cutoff_time[0]:02d}:{spy_cutoff_time[1]:02d} ET when SPY MACD > 0"
        )

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

        # --- SPY MACD afternoon cutoff ---
        if spy_cutoff_enabled and db:
            try:
                from trading.market_regime import SpyMacdCutoff, compute_spy_macd_for_day
                spy_bars = get_1min_bars_cached('SPY', trade_date, client, db)
                spy_prev_date = _get_previous_trading_date(trade_date)
                spy_prev_bars = get_1min_bars_cached('SPY', spy_prev_date, client, db) if spy_prev_date else None
                if spy_bars is not None and not spy_bars.empty:
                    macd_by_time = compute_spy_macd_for_day(spy_bars, spy_prev_bars)
                    spy_cutoff = SpyMacdCutoff(enabled=True, cutoff_time=spy_cutoff_time)
                    spy_cutoff.load_spy_macd(macd_by_time)
                    runner.set_spy_macd_cutoff(spy_cutoff)
                else:
                    runner.set_spy_macd_cutoff(None)
            except Exception as e:
                logger.debug(f"SPY MACD cutoff: failed for {trade_date}: {e}")
                runner.set_spy_macd_cutoff(None)
        elif spy_cutoff_enabled:
            runner.set_spy_macd_cutoff(None)

        # --- Thin liquidity: tighten breakout volume requirement (H5 OR filter) ---
        if market_regime and market_regime.is_thin_liquidity(trade_date):
            runner._min_breakout_vol_override = market_regime.get_min_breakout_volume_ratio(
                trade_date, default=0
            )
        else:
            runner._min_breakout_vol_override = 0  # disabled on normal days

        # --- UD risk scaling ---
        if ud_scaling_enabled and market_regime:
            ud = market_regime.get_spy_ud_volume_ratio(trade_date)
            if ud is not None and isinstance(ud, (int, float)) and ud > ud_threshold:
                runner._ud_risk_scale = ud_scale_factor
            else:
                runner._ud_risk_scale = 0.0
        else:
            runner._ud_risk_scale = 0.0

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

                # Fetch previous day bars for MACD warm-up (MACD filter or zone filter)
                prev_day_bars = None
                if (runner.detector.require_macd_positive or runner.macd_zones_enabled) and db:
                    prev_date = _get_previous_trading_date(trade_date, movers_by_date)
                    if prev_date:
                        prev_day_bars = get_1min_bars_cached(symbol, prev_date, client, db)

                # Set per-symbol avg_daily_volume for relative vol rate gate
                runner.avg_daily_volume = avg_vol or 0

                # Pass prev_close when volume gates are active (needed for qualification loop)
                # prev_close from mover tuple can be 0.0 — fetch from daily bars if needed
                _prev_close = None
                if (runner.min_cum_dollar_vol > 0
                        or runner.min_cum_shares > 0
                        or runner.min_relative_vol_rate > 0):
                    _prev_close = prev_close if prev_close and prev_close > 0 else None
                    if not _prev_close and db:
                        _pc_row = db._cache_conn.execute(
                            'SELECT close FROM daily_bars WHERE symbol = ? AND bar_date < ? ORDER BY bar_date DESC LIMIT 1',
                            (symbol, date_str)).fetchone()
                        _prev_close = float(_pc_row[0]) if _pc_row else None
                result = runner.run(symbol, bars, date_str,
                                    avg_daily_volume=avg_vol,
                                    volume_profile=vol_profile,
                                    prev_close=_prev_close,
                                    prev_day_bars=prev_day_bars)
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

            # Fetch prev day bars for MACD warm-up (only when enabled)
            prev_day_bars = None
            if runner.detector.require_macd_positive:
                from datetime import date as date_cls
                td = date_cls.fromisoformat(trade_date_iso)
                prev_date = _get_previous_trading_date(td)
                if prev_date:
                    prev_cached = db.get_intraday_bars_cached(symbol, prev_date.isoformat())
                    if prev_cached:
                        prev_day_bars = pd.DataFrame(prev_cached)

            result = runner.run(symbol, bars, trade_date_iso,
                                avg_daily_volume=avg_vol,
                                volume_profile=vol_profile,
                                prev_close=None,  # Daily bar pre-filter already ensures 20%+ range
                                prev_day_bars=prev_day_bars)

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
    db_path: str = None,
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
    next_pct_milestone = 10

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for result_dict in executor.map(
            _backtest_worker, filtered_items, chunksize=100
        ):
            completed += 1
            if result_dict is None:
                continue

            result = _reconstruct_result(result_dict)
            results.append(result)

            pct_done = (completed / filtered_total * 100) if filtered_total > 0 else 100
            if pct_done >= next_pct_milestone or completed == filtered_total:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                n_trades = sum(len(r.trades_simulated) for r in results)
                if rate > 0:
                    eta = (filtered_total - completed) / rate
                    eta_str = f"{eta:.0f}s" if eta < 120 else f"{eta / 60:.1f}m"
                    logger.info(
                        f"⏳ {int(pct_done)}% [{completed}/{filtered_total}] "
                        f"{n_trades} trades, {rate:.0f}/sec, ETA {eta_str}"
                    )
                else:
                    logger.info(
                        f"⏳ {int(pct_done)}% [{completed}/{filtered_total}] "
                        f"{n_trades} trades"
                    )
                next_pct_milestone += 10

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
    build_cache: bool = False,
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

    # UD risk scaling config
    ud_cfg = _cfg.get("trading", {}).get("ud_risk_scaling", {})
    ud_scaling_enabled = bool(ud_cfg.get("enabled", False))
    ud_threshold = float(ud_cfg.get("ud_threshold", 1.2))
    ud_scale_factor = float(ud_cfg.get("scale_factor", 0.5))

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
        # Group by date, sorted by move size descending (strongest movers first).
        # This matches live scanner behavior where biggest movers surface first.
        movers_by_date: Dict[date, List[tuple]] = defaultdict(list)
        for mover in movers:
            sym, d = mover[0], mover[1]
            prev_close = mover[2] if len(mover) > 2 else 0.0
            move_size = mover[3] if len(mover) > 3 else 0.0
            movers_by_date[d].append((sym, d, prev_close, move_size))
        # Sort by move size descending — biggest movers first.
        # Live scanner naturally surfaces biggest gappers first (premarket qualification).
        for d in movers_by_date:
            movers_by_date[d].sort(key=lambda x: x[3], reverse=True)

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
            for sym, d, prev_close, *_ in movers_by_date[trade_date]:
                symbol_dates.append((sym, d.isoformat()))

        bulk_data = db.get_intraday_bars_bulk(symbol_dates)
        all_bars = {
            key: pd.DataFrame(bars) for key, bars in bulk_data.items()
        }

        # Fetch and cache any missing 1-min bars from API (one-time cost per symbol/date)
        missing = [sd for sd in symbol_dates if sd not in all_bars]
        if missing:
            import os as _os
            from dotenv import load_dotenv as _load_env
            _load_env()
            _key = _os.getenv("ALPACA_API_KEY")
            _secret = _os.getenv("ALPACA_API_SECRET")
            if _key and _secret:
                _fetch_client = AlpacaClient(_key, _secret)
                logger.info(f"Fetching {len(missing)} uncached 1-min bar sets from API (one-time)...")
                fetched = 0
                for sym, date_str in missing:
                    try:
                        td = date.fromisoformat(date_str)
                        market_open, market_close = _market_hours_utc(td)
                        bars_fetched = _fetch_client.get_historical_1min_bars(
                            sym, market_open, market_close
                        )
                        if bars_fetched is not None and not bars_fetched.empty:
                            bar_records = bars_fetched.to_dict('records')
                            db.save_intraday_bars(sym, date_str, bar_records)
                            all_bars[(sym, date_str)] = bars_fetched
                        fetched += 1
                        if fetched % 200 == 0:
                            logger.info(f"  Fetched {fetched}/{len(missing)} bar sets...")
                    except Exception:
                        pass
                logger.info(f"  Cached {fetched}/{len(missing)} bar sets for future runs")

        # Create runner early to check if MACD warm-up is needed
        runner = BacktestRunner()

        # Pre-load previous day bars for MACD warm-up (needed for MACD filter OR zone filter)
        # First pass: identify needed prev-day pairs and fetch any uncached from API
        all_prev_bars: Dict[tuple, pd.DataFrame] = {}
        need_prev_bars = runner.detector.require_macd_positive or runner.macd_zones_enabled
        if need_prev_bars:
            prev_symbol_dates = set()
            for trade_date in filtered_dates:
                prev_date = _get_previous_trading_date(trade_date)
                if prev_date:
                    for sym, d, *_ in movers_by_date[trade_date]:
                        prev_symbol_dates.add((sym, prev_date.isoformat()))

            if prev_symbol_dates:
                # Check which prev-day bars are already cached
                prev_bulk = db.get_intraday_bars_bulk(list(prev_symbol_dates))
                cached_keys = set(prev_bulk.keys())
                uncached = [sd for sd in prev_symbol_dates if sd not in cached_keys]

                if uncached:
                    # Need AlpacaClient to fetch missing bars
                    import os
                    from dotenv import load_dotenv
                    load_dotenv()
                    api_key = os.getenv("ALPACA_API_KEY")
                    api_secret = os.getenv("ALPACA_API_SECRET")
                    if api_key and api_secret:
                        fetch_client = AlpacaClient(api_key, api_secret)
                        logger.info(
                            f"Fetching {len(uncached)} uncached prev-day bar sets "
                            f"for MACD warm-up (one-time)..."
                        )
                        fetched = 0
                        for sym, date_str in uncached:
                            try:
                                td = date.fromisoformat(date_str)
                                market_open, market_close = _market_hours_utc(td)
                                bars_fetched = fetch_client.get_historical_1min_bars(
                                    sym, market_open, market_close
                                )
                                if not bars_fetched.empty:
                                    bar_records = bars_fetched.to_dict('records')
                                    db.save_intraday_bars(sym, date_str, bar_records)
                                    prev_bulk[(sym, date_str)] = bar_records
                                fetched += 1
                                if fetched % 100 == 0:
                                    logger.info(f"  Fetched {fetched}/{len(uncached)} prev-day bar sets...")
                            except Exception as e:
                                logger.debug(f"  {sym} {date_str}: fetch failed: {e}")
                        logger.info(f"  Cached {fetched} prev-day bar sets for future runs")

                all_prev_bars = {
                    key: pd.DataFrame(bars) for key, bars in prev_bulk.items()
                }
                logger.info(f"MACD warm-up: {len(all_prev_bars)} prev-day bar sets available")

        # Pre-load SPY 1-min bars for SPY MACD afternoon cutoff
        spy_cutoff_cfg = _cfg.get("trading", {}).get("spy_macd_cutoff", {})
        spy_cutoff_enabled = bool(spy_cutoff_cfg.get("enabled", False))
        all_spy_1min: Dict[date, pd.DataFrame] = {}
        all_spy_prev_1min: Dict[date, pd.DataFrame] = {}

        if spy_cutoff_enabled:
            cutoff_str = spy_cutoff_cfg.get("cutoff_time", "11:30")
            _ch, _cm = cutoff_str.split(':')
            spy_cutoff_time = (int(_ch), int(_cm))

            # Collect SPY bar keys
            spy_date_keys = [('SPY', d.isoformat()) for d in filtered_dates]
            spy_prev_keys = []
            for d in filtered_dates:
                prev_d = _get_previous_trading_date(d)
                if prev_d:
                    spy_prev_keys.append(('SPY', prev_d.isoformat()))

            # Bulk load from cache
            spy_bulk = db.get_intraday_bars_bulk(spy_date_keys)
            spy_prev_bulk = db.get_intraday_bars_bulk(spy_prev_keys) if spy_prev_keys else {}

            # Fetch uncached SPY bars from API
            uncached_spy = [k for k in spy_date_keys if k not in spy_bulk]
            uncached_spy_prev = [k for k in spy_prev_keys if k not in spy_prev_bulk]
            all_uncached = uncached_spy + uncached_spy_prev
            if all_uncached:
                import os as _os
                from dotenv import load_dotenv as _load_env
                _load_env()
                _key = _os.getenv("ALPACA_API_KEY")
                _secret = _os.getenv("ALPACA_API_SECRET")
                if _key and _secret:
                    _client = AlpacaClient(_key, _secret)
                    logger.info(f"Fetching {len(all_uncached)} SPY bar sets for MACD cutoff...")
                    for sym, ds in all_uncached:
                        try:
                            td = date.fromisoformat(ds)
                            mo, mc = _market_hours_utc(td)
                            bars_f = _client.get_historical_1min_bars(sym, mo, mc)
                            if not bars_f.empty:
                                recs = bars_f.to_dict('records')
                                db.save_intraday_bars(sym, ds, recs)
                                if (sym, ds) in set(spy_date_keys):
                                    spy_bulk[(sym, ds)] = recs
                                else:
                                    spy_prev_bulk[(sym, ds)] = recs
                        except Exception:
                            pass

            for d in filtered_dates:
                key = ('SPY', d.isoformat())
                data = spy_bulk.get(key)
                if data:
                    all_spy_1min[d] = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
                prev_d = _get_previous_trading_date(d)
                if prev_d:
                    pkey = ('SPY', prev_d.isoformat())
                    pdata = spy_prev_bulk.get(pkey)
                    if pdata:
                        all_spy_prev_1min[d] = pd.DataFrame(pdata) if not isinstance(pdata, pd.DataFrame) else pdata

            logger.info(f"SPY MACD cutoff: {len(all_spy_1min)} trade-day + {len(all_spy_prev_1min)} prev-day bar sets")

        preload_elapsed = time.time() - t_preload
        logger.info(
            f"📦 Data loading done — {len(all_bars)} bar sets in {preload_elapsed:.1f}s"
            + (f" + {len(all_prev_bars)} prev-day sets" if all_prev_bars else "")
        )
        results = []
        completed = 0
        t0 = time.time()
        next_pct_milestone = 10  # Report progress at 10%, 20%, ... 100%

        for trade_date in filtered_dates:
            # Thin liquidity override
            if market_regime and market_regime.is_thin_liquidity(trade_date):
                runner._min_breakout_vol_override = market_regime.get_min_breakout_volume_ratio(
                    trade_date, default=0
                )
            else:
                runner._min_breakout_vol_override = 0

            # UD risk scaling: reduce size on euphoric SPY days
            if ud_scaling_enabled and market_regime:
                ud = market_regime.get_spy_ud_volume_ratio(trade_date)
                if ud is not None and isinstance(ud, (int, float)) and ud > ud_threshold:
                    runner._ud_risk_scale = ud_scale_factor
                else:
                    runner._ud_risk_scale = 0.0
            else:
                runner._ud_risk_scale = 0.0

            # SPY MACD afternoon cutoff for this date
            if spy_cutoff_enabled and trade_date in all_spy_1min:
                from trading.market_regime import SpyMacdCutoff, compute_spy_macd_for_day
                spy_bars_day = all_spy_1min[trade_date]
                spy_prev_day = all_spy_prev_1min.get(trade_date)
                macd_by_time = compute_spy_macd_for_day(spy_bars_day, spy_prev_day)
                spy_cutoff = SpyMacdCutoff(enabled=True, cutoff_time=spy_cutoff_time)
                spy_cutoff.load_spy_macd(macd_by_time)
                runner.set_spy_macd_cutoff(spy_cutoff)
            else:
                runner.set_spy_macd_cutoff(None)

            consec_losses = 0
            stopped_for_day = False
            day_trade_count = 0
            # Disable max_trades during cache builds — cache ALL trades, filter at query time
            max_trades = 0 if build_cache else int(_cfg.get("trading", {}).get("max_trades_per_day", 0))

            for sym, d, prev_close, *_ in movers_by_date[trade_date]:
                completed += 1
                date_str = d.isoformat()

                if stopped_for_day:
                    continue

                if max_trades > 0 and day_trade_count >= max_trades:
                    continue

                bars = all_bars.get((sym, date_str))
                if bars is None or bars.empty:
                    continue

                try:
                    # Look up prev day bars for MACD warm-up
                    prev_day_bars = None
                    if all_prev_bars:
                        prev_date = _get_previous_trading_date(trade_date)
                        if prev_date:
                            prev_day_bars = all_prev_bars.get((sym, prev_date.isoformat()))

                    result = runner.run(
                        sym, bars, date_str,
                        prev_close=None,  # Daily bar pre-filter already ensures 20%+ range
                        prev_day_bars=prev_day_bars,
                    )
                    results.append(result)

                    # Track trades per day
                    day_trade_count += len(result.trades_simulated)

                    # Consecutive loss tracking
                    if max_consecutive_losses > 0:
                        for trade in result.trades_simulated:
                            if trade.pnl > 0:
                                consec_losses = 0
                            else:
                                consec_losses += 1
                                if consec_losses >= max_consecutive_losses:
                                    stopped_for_day = True
                except Exception as e:
                    logger.warning(f"{sym} {date_str}: backtest error: {e}")

                pct_done = (completed / filtered_total * 100) if filtered_total > 0 else 100
                if pct_done >= next_pct_milestone:
                    elapsed = time.time() - t0
                    rate = completed / elapsed if elapsed > 0 else 0
                    n_trades = sum(len(r.trades_simulated) for r in results)
                    eta = (filtered_total - completed) / rate if rate > 0 else 0
                    eta_str = f"{eta:.0f}s" if eta < 120 else f"{eta / 60:.1f}m"
                    logger.info(
                        f"⏳ {int(pct_done)}% [{completed}/{filtered_total}] "
                        f"{n_trades} trades, {rate:.0f}/sec, ETA {eta_str}"
                    )
                    next_pct_milestone += 10

        # Post-process: enforce max_positions across symbols per day.
        # The live engine can only hold N positions simultaneously, so later
        # setups are skipped when all slots are full. Without this, the backtest
        # trades every setup independently and over-counts.
        max_positions = int(_cfg.get("trading", {}).get("max_positions", 0))
        if max_positions > 0:
            from collections import defaultdict as _defaultdict

            # Group results by date
            results_by_date: Dict[str, list] = _defaultdict(list)
            for r in results:
                if r.trades_simulated:
                    results_by_date[r.trade_date].append(r)

            filtered_results = []
            total_dropped = 0

            for trade_date_str in sorted(results_by_date.keys()):
                day_results = results_by_date[trade_date_str]

                # Collect all trades for this day with entry/exit times
                day_trades = []
                for r in day_results:
                    for t in r.trades_simulated:
                        day_trades.append((t.entry_time, t.exit_time, t, r))

                # Sort by entry time (first-come-first-served, like live engine)
                day_trades.sort(key=lambda x: x[0] if x[0] else '')

                # Simulate position limit
                open_positions = []  # list of exit_times
                kept_trades = set()  # trade ids we keep

                for entry_time, exit_time, trade, result in day_trades:
                    # Close expired positions
                    open_positions = [et for et in open_positions if et and et > entry_time]

                    if len(open_positions) < max_positions:
                        open_positions.append(exit_time)
                        kept_trades.add(id(trade))
                    else:
                        total_dropped += 1

                # Rebuild results keeping only allowed trades
                for r in day_results:
                    kept = [t for t in r.trades_simulated if id(t) in kept_trades]
                    r.trades_simulated = kept
                    filtered_results.append(r)

            # Also add results with no trades (pattern-only)
            no_trade_results = [r for r in results if not any(
                r.trade_date == rd for rd in results_by_date
            )]
            filtered_results.extend(no_trade_results)

            if total_dropped > 0:
                logger.info(f"Position limit (max {max_positions}): dropped {total_dropped} trades")
            results = filtered_results

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
            prev_close=None,  # Daily bar pre-filter already ensures 20%+ range
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
    """Convert UTC datetime to ET string, handling EST/EDT correctly."""
    if ts is None:
        return ""
    import pytz as _pytz
    _et_tz = _pytz.timezone('US/Eastern')
    if hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
        et = ts.astimezone(_et_tz)
    else:
        from datetime import timezone as _tz
        et = ts.replace(tzinfo=_tz.utc).astimezone(_et_tz)
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
        "--monthly", action="store_true", default=True,
        help="Monthly-chunked mode (default). Use --no-monthly to disable."
    )
    parser.add_argument(
        "--no-monthly", action="store_true",
        help="Disable monthly chunking — single-process mode (may OOM on large ranges)"
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
    parser.add_argument(
        "--regime-on", action="store_true",
        help="Force regime filter ON (overrides yaml)"
    )
    parser.add_argument(
        "--regime-off", action="store_true",
        help="Force regime filter OFF (overrides yaml)"
    )
    parser.add_argument(
        "--capital", type=float, default=0,
        help="Override capital (e.g., 50000). 0 = use yaml."
    )
    parser.add_argument(
        "--risk", type=float, default=0,
        help="Override risk_per_trade (e.g., 2000). 0 = use yaml."
    )
    parser.add_argument(
        "--max-shares", type=int, default=0,
        help="Override max_shares (e.g., 10000). 0 = use yaml."
    )
    parser.add_argument(
        "--build-cache", action="store_true",
        help="Generate ALL trades (no regime/max_trades) and save to cache CSV"
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Force regeneration even if cache exists"
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Override intraday_change_pct_min (e.g., 20 for 20%%)"
    )
    parser.add_argument(
        "--full-market", action="store_true",
        help="Include all stocks (default: universe-only for production realism)"
    )
    parser.add_argument(
        "--float-max", type=int, default=None,
        help="Override float_max (e.g., 50000000 for 50M). 0 = disabled."
    )
    parser.add_argument(
        "--min-dollar-vol", type=float, default=None,
        help="Min daily dollar volume (e.g., 3000000 for $3M). 0 = disabled."
    )
    parser.add_argument(
        "--min-cum-dollar-vol", type=float, default=0,
        help="Volume gate: min cumulative dollar volume at qualification (live alignment)"
    )
    parser.add_argument(
        "--min-cum-shares", type=int, default=0,
        help="Volume gate: min cumulative shares at qualification"
    )
    parser.add_argument(
        "--min-relative-vol-rate", type=float, default=0,
        help="Volume gate: min relative volume rate (vol_rate / expected_rate)"
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
    if args.monthly and not args.no_monthly:
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
    from config import Config
    from trading.market_regime import MarketRegimeFilter

    cfg = Config._load_yaml_only()
    db_cfg = cfg.get("database", {})
    db = get_database(
        db_path=db_cfg.get("path"),
        cache_path=db_cfg.get("cache_path"),
        trades_path=db_cfg.get("trades_path"),
    )
    client = None  # Lazy-init Alpaca client (only when API calls needed)
    symbols = []   # Lazy-init universe symbols

    # Check if we can use cache (skip heavy data loading)
    cache_available = os.path.exists(BULL_FLAG_CACHE_PATH) and not args.build_cache and not args.no_cache

    if not cache_available:
        # Step 1: Load active universe (matches live scanner)
        universe = db.get_active_universe()
        symbols = [s['symbol'] for s in universe]
        logger.info(f"Loaded {len(symbols)} symbols from active universe")

        # Also include ALL symbols from daily_bars cache
        import sqlite3
        conn = sqlite3.connect(db.db_path)
        all_cached = [r[0] for r in conn.execute(
            "SELECT DISTINCT symbol FROM daily_bars WHERE bar_date >= ? AND bar_date <= ?",
            (str(start_date), str(end_date))
        ).fetchall()]
        symbols = sorted(set(symbols + all_cached))
        logger.info(f"Expanded to {len(symbols)} symbols (universe + daily_bars cache)")

        if not symbols:
            logger.error("No symbols found")
            sys.exit(1)

        # Step 2: Fetch daily bars and find movers
        client = AlpacaClient(api_key=api_key, api_secret=api_secret)
        logger.info("Fetching daily bars for date range (cache-first)...")
        daily_bars = fetch_daily_bars_cached(symbols, start_date - timedelta(days=7), end_date, client, db)
        universe_dict = {s['symbol']: s for s in db.get_active_universe()}

        scanner_cfg = cfg.get("scanner", {})
        # Use 10% for cache builds (broadest), config value otherwise
        if args.build_cache:
            intraday_threshold = 0.10  # Cache at 10% — filter at query time
        elif args.threshold is not None:
            intraday_threshold = args.threshold / 100.0
        else:
            intraday_threshold = float(scanner_cfg.get("intraday_change_pct_min", 20.0)) / 100.0
        _min_dv = args.min_dollar_vol if args.min_dollar_vol is not None else float(scanner_cfg.get("min_dollar_volume", 0))
        movers = find_big_movers(
            daily_bars,
            threshold=intraday_threshold,
            universe_dict=universe_dict,
            price_min=float(scanner_cfg.get("price_min", 2.0)),
            price_max=float(scanner_cfg.get("price_max", 20.0)),
            float_max=args.float_max if args.float_max is not None else int(scanner_cfg.get("float_max", 10_000_000)),
            min_dollar_volume=_min_dv,
            start_date=start_date,
            end_date=end_date,
        )
    else:
        movers = []
        symbols = []

    if not movers and not cache_available:
        logger.warning("No symbols with 10%+ intraday move found — nothing to backtest")
        print_summary(len(symbols), movers, [])
        return

    # CLI overrides for sizing (apply before runner creation)
    if args.capital > 0:
        cfg.setdefault("trading", {})["capital"] = args.capital
        cfg["trading"]["position_size_dollars"] = args.capital
        logger.info(f"CLI override: capital=${args.capital:,.0f}")
    if args.risk > 0:
        cfg.setdefault("trading", {})["risk_per_trade"] = args.risk
        logger.info(f"CLI override: risk_per_trade=${args.risk:,.0f}")
    if args.max_shares > 0:
        cfg.setdefault("trading", {})["max_shares"] = args.max_shares
        logger.info(f"CLI override: max_shares={args.max_shares}")

    # Write overrides to config.yaml so BacktestRunner.from_config() picks them up
    if args.capital > 0 or args.risk > 0 or args.max_shares > 0:
        import yaml
        with open("config.yaml", "r") as f:
            live_cfg = yaml.safe_load(f)
        if args.capital > 0:
            live_cfg["trading"]["capital"] = args.capital
            live_cfg["trading"]["position_size_dollars"] = args.capital
        if args.risk > 0:
            live_cfg["trading"]["risk_per_trade"] = args.risk
        if args.max_shares > 0:
            live_cfg["trading"]["max_shares"] = args.max_shares
        with open("config.yaml", "w") as f:
            yaml.dump(live_cfg, f, default_flow_style=False)
        _sizing_overridden = True
    else:
        _sizing_overridden = False

    # Step 3: Build market regime filter
    trading_cfg = cfg.get("trading", {})
    regime_cfg = trading_cfg.get("market_regime", {})
    sma_period = int(regime_cfg.get("sma_period", 50))
    spy_lookback_days = int(sma_period * 1.5) + 14
    spy_start = start_date - timedelta(days=spy_lookback_days)
    if cache_available:
        # Load SPY from DB only (no API client needed)
        import sqlite3
        _conn = sqlite3.connect(db.db_path)
        spy_bars = [{'date': r[0], 'open': r[1], 'high': r[2], 'low': r[3], 'close': r[4], 'volume': r[5]}
                    for r in _conn.execute(
                        "SELECT bar_date, open, high, low, close, volume FROM daily_bars "
                        "WHERE symbol='SPY' AND bar_date >= ? AND bar_date <= ? ORDER BY bar_date",
                        (str(spy_start), str(end_date))
                    ).fetchall()]
    else:
        spy_bars_raw = fetch_daily_bars_cached(['SPY'], spy_start, end_date, client, db)
        spy_bars = spy_bars_raw.get('SPY', [])
    max_trades_per_day = int(trading_cfg.get("max_trades_per_day", 5))
    # CLI override for regime filter
    regime_enabled = bool(regime_cfg.get("enabled", True))
    if args.regime_on:
        regime_enabled = True
    elif args.regime_off:
        regime_enabled = False
    market_regime = MarketRegimeFilter(
        enabled=regime_enabled,
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

    # Step 4: Run backtests — use cache if available
    _entry_slip = float(trading_cfg.get("entry_slippage_pct", 0.005))
    _exit_slip = float(trading_cfg.get("exit_slippage_pct", 0.003))
    _cache_path = _get_bull_flag_cache_path(_entry_slip, _exit_slip)
    # Also check legacy path for backward compat
    if os.path.exists(_cache_path):
        _active_cache = _cache_path
    elif os.path.exists(BULL_FLAG_CACHE_PATH):
        _active_cache = BULL_FLAG_CACHE_PATH
    else:
        _active_cache = None
    use_cache = _active_cache is not None and not args.build_cache and not args.no_cache

    if use_cache:
        # Fast path: load cached trades, apply filters in memory
        _pos_size = float(trading_cfg.get("position_size_dollars", 50000))
        cached_trades = load_bull_flag_cache(
            _active_cache, start_date, end_date,
            entry_slippage_pct=_entry_slip, exit_slippage_pct=_exit_slip,
            position_size=_pos_size,
        )

        # Auto-build missing dates: check which trading days in range are NOT cached
        cached_dates = set(t['date'] for t in cached_trades)
        # Get all trading days in range from Alpaca calendar
        try:
            if not client:
                client = AlpacaClient(api_key=api_key, api_secret=api_secret)
            _cal = client.get_market_calendar(start_date, end_date)
            all_trading_days = set(str(d['date']) for d in _cal)
        except Exception:
            # Fallback: generate weekdays
            all_trading_days = set()
            d = start_date
            while d <= end_date:
                if d.weekday() < 5:
                    all_trading_days.add(str(d))
                d += timedelta(days=1)

        # Also include dates from the full cache (beyond requested range) to avoid
        # re-processing dates that exist in cache but outside the current query
        _all_cached = set()
        try:
            import csv as _csv
            with open(_active_cache) as _f:
                for _row in _csv.DictReader(_f):
                    _all_cached.add(_row['date'])
        except Exception:
            pass

        missing_days = sorted(all_trading_days - _all_cached)
        if missing_days:
            logger.info(f"Auto-building {len(missing_days)} missing dates: {missing_days[0]} to {missing_days[-1]}")
            # Fetch daily bars for missing dates
            if not client:
                client = AlpacaClient(api_key=api_key, api_secret=api_secret)
            if not symbols:
                symbols = [s['symbol'] for s in db.get_active_universe()]
            _miss_start = date.fromisoformat(missing_days[0])
            _miss_end = date.fromisoformat(missing_days[-1])
            daily_bars = fetch_daily_bars_cached(symbols, _miss_start - timedelta(days=7), _miss_end, client, db)
            universe_dict = {s['symbol']: s for s in db.get_active_universe()}

            _miss_movers = find_big_movers(
                daily_bars, threshold=0.10,  # Cache at 10%
                universe_dict=universe_dict,
                price_min=float(cfg.get("scanner", {}).get("price_min", 2.0)),
                price_max=float(cfg.get("scanner", {}).get("price_max", 20.0)),
                float_max=int(cfg.get("scanner", {}).get("float_max", 10_000_000)),
                start_date=_miss_start, end_date=_miss_end,
            )
            if _miss_movers:
                logger.info(f"Found {len(_miss_movers)} movers on missing dates, running backtests...")
                # Run backtests on missing movers (reuse existing runner setup)
                from backtest import BacktestRunner
                runner = BacktestRunner(realistic=True)
                _new_trades = []
                volume_profiles = db.get_all_volume_profiles()
                for mover_tuple in _miss_movers:
                    sym, trade_date = mover_tuple[0], mover_tuple[1]
                    td_str = str(trade_date)
                    import pandas as pd
                    bars_raw = db.get_intraday_bars_cached(sym, td_str)
                    if not bars_raw:
                        bars_df = client.get_1min_bars(sym, trade_date=td_str)
                        if bars_df is not None and len(bars_df) > 0:
                            # Save raw bars to DB for future cache hits
                            bars_list = bars_df.to_dict('records') if hasattr(bars_df, 'to_dict') else bars_df
                            db.save_intraday_bars(sym, td_str, bars_list)
                        bars = bars_df
                    else:
                        bars = pd.DataFrame(bars_raw)
                    if bars is None or len(bars) < runner.MIN_BARS_FOR_SETUP:
                        continue
                    uni = universe_dict.get(sym, {})
                    avg_vol = uni.get('avg_daily_volume') or uni.get('avg_volume_daily')
                    vol_profile = volume_profiles.get(sym)
                    # Compute daily range for cache
                    _day_bars = daily_bars.get(sym, [])
                    _day_bar = next((b for b in _day_bars if str(b.get('date', '')) == td_str), None)
                    _range_pct = 0
                    if _day_bar and _day_bar.get('low', 0) > 0:
                        _range_pct = (_day_bar['high'] - _day_bar['low']) / _day_bar['low'] * 100
                    result = runner.run(sym, bars, td_str,
                                        avg_daily_volume=avg_vol,
                                        volume_profile=vol_profile,
                                        prev_close=None,
                                        prev_day_bars=None)
                    for trade in result.trades_simulated:
                        trade._daily_range_pct = _range_pct  # attach for cache row
                        _new_trades.append(trade)

                if _new_trades:
                    # Append to cache CSV
                    _append_count = 0
                    with open(_active_cache, 'a', newline='') as _f:
                        import csv as _csv
                        writer = _csv.DictWriter(_f, fieldnames=CSV_HEADERS)
                        for trade in _new_trades:
                            row = _trade_to_cache_row(trade)
                            if row:
                                writer.writerow(row)
                                _append_count += 1
                    logger.info(f"Appended {_append_count} trades to cache for {len(missing_days)} new dates")

                    # Reload cache with new trades included
                    cached_trades = load_bull_flag_cache(
                        _active_cache, start_date, end_date,
                        entry_slippage_pct=_entry_slip, exit_slippage_pct=_exit_slip,
                        position_size=_pos_size,
                    )
            else:
                logger.info(f"No movers found on {len(missing_days)} missing dates")
        # Determine threshold for filtering
        if args.threshold is not None:
            _min_range = args.threshold
        else:
            _min_range = float(cfg.get("scanner", {}).get("intraday_change_pct_min", 20.0))
        # Universe filter: default = universe-only, --full-market = all
        _uni_syms = None
        if not args.full_market:
            _active = db.get_active_universe()
            _uni_syms = set(s['symbol'] for s in _active)
            logger.info(f"Active universe: {len(_uni_syms)} stocks")

        _max_pos = int(trading_cfg.get("max_positions", 3))
        _daily_loss = float(trading_cfg.get("daily_loss_limit", -5000))
        # Build volume map for min_daily_volume filter
        _vol_map = {s['symbol']: s.get('avg_volume_daily', 0) or 0 for s in db.get_active_universe()}
        filtered_trades = filter_bull_flag_trades(
            cached_trades,
            market_regime=market_regime if market_regime.enabled else None,
            max_trades_per_day=max_trades_per_day,
            max_consecutive_losses=max_consec,
            min_daily_range_pct=_min_range,
            universe_symbols=_uni_syms,
            max_concurrent=_max_pos,
            daily_loss_limit=_daily_loss,
            universe_vol_map=_vol_map,
        )
        # Write output CSV
        trade_count = 0
        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
            for t in filtered_trades:
                writer.writerow([
                    t['symbol'], t['date'], t.get('entry_time_et', ''),
                    t['entry_price'], t.get('stop_loss', ''), t.get('target', ''),
                    t['shares'], t.get('exit_time_et', ''), t['exit_price'],
                    t.get('exit_reason', ''), f"{t['pnl']:.2f}", f"{t['pnl_pct']:.2f}",
                    t.get('partial_taken', ''), t.get('partial_price', ''),
                    t.get('partial_shares', ''), t.get('partial_pnl', ''),
                ])
                trade_count += 1
        logger.info(f"CSV report written to {args.output} ({trade_count} trades)")

        # Print summary
        n = len(filtered_trades)
        wins = sum(1 for t in filtered_trades if t['pnl'] > 0)
        losses = n - wins
        total_pnl = sum(t['pnl'] for t in filtered_trades)
        wr = wins / n * 100 if n else 0
        avg_win = sum(t['pnl'] for t in filtered_trades if t['pnl'] > 0) / wins if wins else 0
        avg_loss = sum(t['pnl'] for t in filtered_trades if t['pnl'] <= 0) / losses if losses else 0
        import time as _t; elapsed = 0  # Cache path is instant
        print(f"\n{'='*70}")
        print(f"  BATCH BACKTEST (from cache) — {start_date} to {end_date}")
        print(f"{'='*70}")
        print(f"  Total trades taken:      {n}")
        print(f"  Winning trades:          {wins}")
        print(f"  Losing trades:           {losses}")
        print(f"  Win rate:                {wr:.1f}%")
        print(f"  Total P&L:              ${total_pnl:+,.2f}")
        print(f"  Avg win:                ${avg_win:+,.2f}")
        print(f"  Avg loss:               ${avg_loss:+,.2f}")
        print(f"  Elapsed:                {elapsed:.1f}s")
        print(f"{'='*70}")
        return

    # Slow path: generate from scratch
    use_fast = not args.no_parallel and not args.verbose

    if args.build_cache:
        # Disable regime + max_trades for cache generation
        # Slippage from config is baked in — cache filename includes slippage params
        cache_regime = MarketRegimeFilter(enabled=False, max_trades_per_day=0)
        cache_regime.load_spy_bars(spy_bars)
        logger.info(f"Building bull flag cache (regime OFF, no max_trades, entry_slip={_entry_slip:.1%}, exit_slip={_exit_slip:.1%})...")

        # Chunk movers by month to cap memory (~500MB/chunk instead of 5GB+ all-at-once)
        from itertools import groupby as _groupby
        monthly_movers = []
        for _, grp in _groupby(sorted(movers, key=lambda m: m[1]), key=lambda m: (m[1].year, m[1].month)):
            monthly_movers.append(list(grp))
        logger.info(f"Processing {len(monthly_movers)} monthly chunks")

        results = []
        for chunk_idx, chunk in enumerate(monthly_movers):
            chunk_label = f"{chunk[0][1].strftime('%Y-%m')}"
            logger.info(f"📦 Chunk {chunk_idx+1}/{len(monthly_movers)} ({chunk_label}): {len(chunk)} movers")
            if use_fast:
                chunk_results = run_batch_backtest_fast(
                    chunk, db=db,
                    market_regime=cache_regime,
                    max_consecutive_losses=0,
                    max_workers=args.scan_workers,
                    build_cache=True,
                )
            else:
                runner = BacktestRunner(
                    min_cum_dollar_vol=args.min_cum_dollar_vol,
                    min_cum_shares=args.min_cum_shares,
                    min_relative_vol_rate=args.min_relative_vol_rate,
                )
                chunk_results = run_batch_backtest(
                    chunk, client, runner, db=db, universe_dict=universe_dict,
                    market_regime=cache_regime,
                    max_consecutive_losses=0,
                    max_trades_per_day=0,
                )
            results.extend(chunk_results)
            n_chunk_trades = sum(len(r.trades_simulated) for r in chunk_results)
            logger.info(f"  {chunk_label}: {n_chunk_trades} trades found")
            import gc; gc.collect()
        # Build daily range lookup: (sym, date) -> (high-low)/low * 100
        import sqlite3 as _sql
        _conn = _sql.connect(db.db_path)
        _range_map = {}
        for r in _conn.execute(
            "SELECT symbol, bar_date, high, low FROM daily_bars "
            "WHERE bar_date >= ? AND bar_date <= ?",
            (str(start_date), str(end_date))
        ).fetchall():
            if r[3] > 0:
                _range_map[(r[0], r[1])] = (r[2] - r[3]) / r[3] * 100

        # Save ALL trades to cache (slippage baked in from config)
        _save_path = _get_bull_flag_cache_path(_entry_slip, _exit_slip)
        trade_count = 0
        with open(_save_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
            for result in results:
                for trade in result.trades_simulated:
                    dr = _range_map.get((trade.symbol, result.trade_date), 0)
                    writer.writerow([
                        trade.symbol, result.trade_date,
                        utc_to_et_str(trade.entry_time),
                        f"{trade.entry_price:.2f}", f"{trade.stop_loss:.2f}",
                        f"{trade.take_profit:.2f}", trade.shares,
                        utc_to_et_str(trade.exit_time),
                        f"{trade.exit_price:.2f}" if trade.exit_price else "",
                        trade.exit_reason or "", f"{trade.pnl:.2f}",
                        f"{trade.pnl_pct:.2f}", trade.partial_exit_taken,
                        f"{trade.partial_exit_price:.2f}" if trade.partial_exit_price else "",
                        trade.partial_shares, f"{trade.partial_pnl:.2f}",
                        f"{dr:.1f}",
                    ])
                    trade_count += 1
        logger.info(f"Bull flag cache saved: {_save_path} ({trade_count} trades, entry_slip={_entry_slip:.1%}, exit_slip={_exit_slip:.1%})")
    else:
        if use_fast:
            results = run_batch_backtest_fast(
                movers, db=db,
                market_regime=market_regime,
                max_consecutive_losses=max_consec,
                max_workers=args.scan_workers,
            )
        else:
            from backtest import BacktestRunner as _BtRunner
            runner = _BtRunner(
                min_cum_dollar_vol=args.min_cum_dollar_vol,
                min_cum_shares=args.min_cum_shares,
                min_relative_vol_rate=args.min_relative_vol_rate,
            )
            results = run_batch_backtest(
                movers, client, runner, db=db, universe_dict=universe_dict,
                market_regime=market_regime,
                max_consecutive_losses=max_consec,
            )

    # Restore config.yaml if we overrode sizing
    if _sizing_overridden:
        import shutil
        shutil.copy("config.yaml.template", "config.yaml")
        logger.info("Restored config.yaml from template (sizing overrides removed)")

    # Step 5: Write CSV + print summary
    write_csv_report(results, args.output)
    print_summary(len(symbols), movers, results)


if __name__ == "__main__":
    main()
