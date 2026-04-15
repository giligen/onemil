"""
MACD Wave Strategy — Proper Sequential Backtest.

Finds medium/large cap stocks with +10% intraday moves, enters on MACD
histogram positive confirmation, exits on histogram flip. Sequential
position simulation with configurable filters, slippage, and risk limits.

Usage:
    python macd_wave_backtest.py                              # March 2026 defaults
    python macd_wave_backtest.py --start 2026-01-01 --end 2026-03-27
    python macd_wave_backtest.py --cross-time 10 --macd-min 0.3 --max-price 25
    python macd_wave_backtest.py --w1-scout --w1-min 5 --max-waves 3
    python macd_wave_backtest.py --no-slippage
"""

import argparse
import csv
import logging
import os
import sys
import time as time_mod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz
import yaml
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

for name in ['data_sources.alpaca_client', 'persistence.database',
             'anthropic', 'httpcore', 'httpx']:
    logging.getLogger(name).setLevel(logging.WARNING)

ET = pytz.timezone('US/Eastern')
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'macd_wave.yaml')


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str = CONFIG_PATH) -> dict:
    """Load MACD wave config from YAML."""
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Conviction scoring — shared with PROD via trading/macd_conviction.py
# ---------------------------------------------------------------------------
# BT-PROD parity is enforced by a single implementation in
# trading/macd_conviction.py. See that module for the V4 formula, research
# provenance (step 1 bucket analysis + step 2 walk-forward study), and
# thresholds. Both backtest and `trading/macd_wave_engine.py` import from it.

from trading.macd_conviction import compute_conviction_score  # noqa: E402


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class TradeEvent:
    """An entry or exit event for position simulation."""
    time: str           # ISO timestamp for sorting
    event_type: str     # 'entry' or 'exit'
    symbol: str
    date_str: str
    price: float
    wave: int
    # Metadata (for entry events)
    shares: int = 0
    hard_stop: float = 0.0
    cross_time_min: int = 0
    vol_at_cross: int = 0
    macd_hist_pct: float = 0.0
    w1_pnl: float = 0.0  # For W2+ trades, the W1 result


@dataclass
class Trade:
    """Completed trade record."""
    symbol: str
    date_str: str
    wave: int
    entry_price: float
    exit_price: float
    shares: int
    pnl_pct: float
    pnl_dollar: float
    entry_time: str
    exit_time: str
    exit_reason: str      # 'macd_flip', 'hard_stop', 'eod_close'
    cross_time_min: int = 0
    vol_at_cross: int = 0
    macd_hist_pct: float = 0.0
    w1_pnl: float = 0.0
    conv_mult: float = 1.0


# ---------------------------------------------------------------------------
# Universe scanning
# ---------------------------------------------------------------------------

def find_movers(
    start_date: date,
    end_date: date,
    min_price: float,
    max_price: float,
    min_volume: int,
    min_intraday_pct: float,
) -> List[Tuple[str, str, float, float, int]]:
    """
    Find all stocks with +min_intraday_pct% intraday range in date range.

    Returns list of (symbol, date_str, intraday_pct, close, volume).
    Uses cached daily bars from DB first, fetches missing from Alpaca and caches.
    """
    import sqlite3
    from data_sources.alpaca_client import AlpacaClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.trading.requests import GetAssetsRequest
    from alpaca.trading.enums import AssetClass, AssetStatus

    client = AlpacaClient(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_API_SECRET'))
    conn = sqlite3.connect('data/cache.db')

    # Check which dates have FULL market daily bars cached (not just bull flag universe).
    # A date is "fully cached" if it has bars for 5000+ symbols (full market scan).
    # Dates with fewer bars only have our 859-stock bull flag universe.
    cur = conn.execute(
        "SELECT bar_date, COUNT(DISTINCT symbol) as cnt FROM daily_bars "
        "WHERE bar_date >= ? AND bar_date <= ? GROUP BY bar_date",
        (str(start_date), str(end_date))
    )
    date_counts = {r[0]: r[1] for r in cur.fetchall()}
    # Consider "fully cached" if 5000+ symbols (full market has ~12K, but many are low-price/no-vol)
    cached_dates = set(d for d, cnt in date_counts.items() if cnt >= 5000)

    # Generate all trading dates in range (approximate — weekdays)
    all_dates = set()
    d = start_date
    while d <= end_date:
        if d.weekday() < 5:  # Mon-Fri
            all_dates.add(str(d))
        d += timedelta(days=1)

    missing_dates = all_dates - cached_dates
    partially_cached = set(date_counts.keys()) - cached_dates
    logger.info(
        f"Daily bars: {len(cached_dates)} fully cached, "
        f"{len(partially_cached)} partial (bull flag only), "
        f"{len(missing_dates)} need full market fetch"
    )

    # Fetch missing dates from Alpaca
    if missing_dates:
        assets = client.trading_client.get_all_assets(
            GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)
        )
        all_symbols = [
            a.symbol for a in assets
            if a.tradable and a.exchange in ('NYSE', 'NASDAQ', 'AMEX', 'ARCA', 'BATS')
        ]

        # Find contiguous date ranges to minimize API calls
        sorted_missing = sorted(missing_dates)
        if sorted_missing:
            fetch_start = date.fromisoformat(sorted_missing[0])
            fetch_end = date.fromisoformat(sorted_missing[-1])
            logger.info(f"Fetching daily bars for {len(all_symbols)} symbols, {fetch_start} to {fetch_end}...")

            chunk_size = 500
            chunks_done = 0
            total_chunks = (len(all_symbols) + chunk_size - 1) // chunk_size
            bars_cached = 0

            for i in range(0, len(all_symbols), chunk_size):
                chunk = all_symbols[i:i + chunk_size]
                chunks_done += 1
                try:
                    req = StockBarsRequest(
                        symbol_or_symbols=chunk,
                        timeframe=TimeFrame.Day,
                        start=datetime(fetch_start.year, fetch_start.month, fetch_start.day),
                        end=datetime(fetch_end.year, fetch_end.month, fetch_end.day) + timedelta(days=1),
                    )
                    bars = client.data_client.get_stock_bars(req)
                    for sym, bar_list in bars.data.items():
                        for b in bar_list:
                            d_str = str(b.timestamp)[:10]
                            if d_str not in missing_dates:
                                continue
                            conn.execute(
                                "INSERT OR IGNORE INTO daily_bars (symbol, bar_date, open, high, low, close, volume, fetched_at) "
                                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                (sym, d_str, float(b.open), float(b.high), float(b.low),
                                 float(b.close), int(b.volume), datetime.now(timezone.utc).isoformat())
                            )
                            bars_cached += 1
                except Exception as e:
                    logger.debug(f"Chunk {chunks_done} failed: {e}")

                if chunks_done % 5 == 0:
                    conn.commit()
                    logger.info(f"  Scanned {chunks_done}/{total_chunks} chunks, {bars_cached:,} bars cached...")

            conn.commit()
            logger.info(f"Cached {bars_cached:,} daily bars to DB")

    # Build previous-day volume lookup (matches production: universe filtered by prev day vol)
    # For each (symbol, date), find the most recent trading day's volume BEFORE that date
    prev_vol_map = {}  # (symbol, date_str) -> prev_day_volume
    if min_volume > 0:
        prev_cur = conn.execute(
            "SELECT a.symbol, a.bar_date, "
            "  (SELECT b.volume FROM daily_bars b "
            "   WHERE b.symbol = a.symbol AND b.bar_date < a.bar_date "
            "   ORDER BY b.bar_date DESC LIMIT 1) as prev_volume "
            "FROM daily_bars a "
            "WHERE a.bar_date >= ? AND a.bar_date <= ?",
            (str(start_date), str(end_date))
        )
        for sym, d, prev_vol in prev_cur:
            if prev_vol is not None:
                prev_vol_map[(sym, d)] = int(prev_vol)

    # Now read ALL movers from DB cache (fast)
    movers = []
    vol_filtered = 0
    cur = conn.execute(
        "SELECT symbol, bar_date, open, high, low, close, volume FROM daily_bars "
        "WHERE bar_date >= ? AND bar_date <= ?",
        (str(start_date), str(end_date))
    )
    for sym, d, opn, high, low, close, vol in cur:
        if low <= 0:
            continue
        pct = (high - low) / low * 100
        if pct < min_intraday_pct:
            continue
        if close < min_price:
            continue
        if max_price > 0 and close > max_price:
            continue
        # Use PREVIOUS day's volume (matches production — no look-ahead bias)
        if min_volume > 0:
            prev_vol = prev_vol_map.get((sym, d), 0)
            if prev_vol < min_volume:
                vol_filtered += 1
                continue
        movers.append((sym, d, pct, close, vol))

    movers.sort(key=lambda x: (x[1], x[0]))
    logger.info(
        f"Found {len(movers)} movers (price>=${min_price}, prev_day_vol>={min_volume:,}, "
        f"range>={min_intraday_pct}%, {vol_filtered} filtered by prev-day volume)"
    )
    return movers


# ---------------------------------------------------------------------------
# 1-min bar loading
# ---------------------------------------------------------------------------

def load_intraday_bars(
    movers: List[Tuple],
    db,
) -> Dict[Tuple[str, str], pd.DataFrame]:
    """Load 1-min bars for all movers, fetch missing from API."""
    from data_sources.alpaca_client import AlpacaClient

    keys = [(m[0], m[1]) for m in movers]
    bar_cache_raw = db.get_intraday_bars_bulk(keys)
    missing = [k for k in keys if k not in bar_cache_raw]

    if missing:
        logger.info(f"Fetching {len(missing)} missing 1-min bar sets...")
        client = AlpacaClient(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_API_SECRET'))
        fetched = 0
        for sym, ds in missing:
            try:
                td = date.fromisoformat(ds)
                mo = ET.localize(datetime(td.year, td.month, td.day, 9, 30)).astimezone(timezone.utc)
                mc = ET.localize(datetime(td.year, td.month, td.day, 16, 0)).astimezone(timezone.utc)
                bf = client.get_historical_1min_bars(sym, mo, mc)
                if bf is not None and not bf.empty:
                    recs = bf.to_dict('records')
                    db.save_intraday_bars(sym, ds, recs)
                    bar_cache_raw[(sym, ds)] = recs
                    fetched += 1
                    if fetched % 100 == 0:
                        logger.info(f"  Fetched {fetched}/{len(missing)}...")
            except Exception:
                pass
        logger.info(f"Fetched {fetched} bar sets, total: {len(bar_cache_raw)}")

    # Convert to DataFrames
    bar_cache = {}
    for key, raw in bar_cache_raw.items():
        df = pd.DataFrame(raw)
        if len(df) < 30:
            continue
        for col in ['timestamp', 't']:
            if col in df.columns:
                df['ts'] = pd.to_datetime(df[col])
                break
        else:
            continue
        df = df.sort_values('ts').reset_index(drop=True)
        bar_cache[key] = df

    logger.info(f"Loaded {len(bar_cache)} bar sets with 30+ bars")
    return bar_cache


# ---------------------------------------------------------------------------
# MACD wave signal generation (per stock per day)
# ---------------------------------------------------------------------------

def generate_signals(
    bars: pd.DataFrame,
    cfg: dict,
    entry_filters: dict,
) -> List[dict]:
    """
    Generate MACD wave entry/exit signals for a single stock/day.

    Returns list of signal dicts with entry/exit times, prices, wave number,
    and metadata for filtering.
    """
    intraday_pct = cfg.get('universe', {}).get('min_intraday_pct', 10.0)
    macd_fast = cfg.get('macd', {}).get('fast_period', 12)
    macd_slow = cfg.get('macd', {}).get('slow_period', 26)
    macd_signal = cfg.get('macd', {}).get('signal_period', 9)
    confirm_bars = cfg.get('macd', {}).get('confirm_bars', 3)
    # Phase A: CLI override for confirm_bars
    if entry_filters.get('confirm_bars_override') is not None:
        confirm_bars = entry_filters['confirm_bars_override']
    max_waves = entry_filters.get('max_waves', 1)
    w1_scout = entry_filters.get('w1_scout', False)
    w1_min_pct = entry_filters.get('w1_min_pct', 0.0)
    hard_stop_pct = cfg.get('risk', {}).get('hard_stop_pct', 0.02)
    # Phase B: CLI override for hard stop
    if entry_filters.get('hard_stop_override') is not None:
        hard_stop_pct = entry_filters['hard_stop_override']
    entry_slippage = entry_filters.get('entry_pct', 0.005)
    exit_slippage = entry_filters.get('exit_pct', 0.002)
    # Conviction-based sizing (opt-in; default off to preserve legacy behavior)
    conviction_sizing_enabled = bool(entry_filters.get('conviction_sizing', False))

    # Entry filter values
    cross_time_max = entry_filters.get('cross_time_max_min', 0)
    min_vol_cross = entry_filters.get('min_vol_at_cross', 0)
    max_vol_cross = entry_filters.get('max_vol_at_cross', 0)
    min_macd_pct = entry_filters.get('min_macd_hist_pct', 0.0)
    max_price_entry = entry_filters.get('max_price_at_entry', 0)
    position_size = entry_filters.get('position_size', 40000)
    # New indicator filters
    max_rsi_val = entry_filters.get('max_rsi', 0)
    require_hist_accel = entry_filters.get('require_hist_accel', False)
    require_obv_rising = entry_filters.get('require_obv_rising', False)
    require_ema_above = entry_filters.get('require_ema_above', False)
    # Phase A
    require_vwap_above = entry_filters.get('require_vwap_above', False)
    require_vol_surge = entry_filters.get('require_vol_surge', False)
    require_strong_close = entry_filters.get('require_strong_close', False)
    require_higher_high = entry_filters.get('require_higher_high', False)
    # Phase B
    trail_activate_pct = entry_filters.get('trail_activate_pct', 0)
    # Phase C
    partial_pct = entry_filters.get('partial_pct', 0)
    partial_fraction = entry_filters.get('partial_fraction', 0.5)
    # Phase D: vol_at_cross exclusion zone
    exclude_vol_zone = entry_filters.get('exclude_vol_zone')
    excl_vol_low = excl_vol_high = None
    if exclude_vol_zone:
        try:
            lo, hi = exclude_vol_zone.split('-')
            excl_vol_low, excl_vol_high = int(lo), int(hi)
        except Exception:
            excl_vol_low = excl_vol_high = None

    if not isinstance(bars, pd.DataFrame):
        bars = pd.DataFrame(bars)
    if bars.empty or 'open' not in bars.columns:
        return []

    # Clip to regular session (09:30 ET onward). The intraday bar cache
    # sometimes includes pre-market bars (08:30 ET+) for certain symbols.
    # Without this clip, `op = bars.iloc[0]['open']` would use the pre-market
    # open as the day anchor, allowing pre-market crosses + pre-market entries
    # that PROD would never take (PROD scanner only runs after market open).
    # Fix introduced 2026-04-15 after Q1 2026 ToD analysis showed ~58% of
    # baseline P&L coming from pre-market BT artifacts.
    if 'ts' in bars.columns and len(bars) > 0:
        bars_ts = pd.to_datetime(bars['ts'], utc=True)
        et_dt = bars_ts.dt.tz_convert(ET)
        # Regular session: 09:30 ET onward (any minute >= 09:30)
        regular = (et_dt.dt.hour > 9) | (
            (et_dt.dt.hour == 9) & (et_dt.dt.minute >= 30)
        )
        bars = bars[regular].reset_index(drop=True)
        if bars.empty:
            return []

    op = bars.iloc[0]['open']
    if op <= 0:
        return []

    # Find +threshold% cross
    threshold = op * (1 + intraday_pct / 100)
    si = None
    for i in range(len(bars)):
        if bars.iloc[i]['high'] >= threshold:
            si = i
            break
    if si is None or si + 10 >= len(bars):
        return []

    cross_time_min = si + 1
    vol_at_cross = int(bars.iloc[:si + 1]['volume'].sum())

    # NOTE: cross_time and vol_at_cross filters are applied at ENTRY time
    # (inside the MACD loop below), NOT here. Early rejection here was a bug
    # that skipped stocks before MACD was even evaluated — lost 50 trades
    # and $73K of P&L vs the original validated results.

    # Compute MACD
    close = bars['close']
    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=macd_signal, adjust=False).mean()
    histogram = macd_line - signal_line

    # Generate wave signals
    signals = []
    in_trade = False
    entry_price = 0.0
    entry_time = ''
    entry_idx = 0
    pos_count = 0
    wave = 0
    w1_result = 0.0
    entry_hist_pct = 0.0

    effective_max_waves = max_waves
    if w1_scout:
        effective_max_waves = max(max_waves, 1)  # Need at least W1 for scouting

    for i in range(max(si, 1), len(bars)):
        h = histogram.iloc[i]
        bar_ts = str(bars.iloc[i]['ts'])

        if wave >= effective_max_waves:
            break

        if not in_trade:
            if h > 0:
                pos_count += 1
            else:
                pos_count = 0

            if pos_count >= confirm_bars:
                raw_price = bars.iloc[i]['close']
                hist_pct = h / raw_price * 100 if raw_price > 0 else 0

                # Apply ALL entry filters at MACD confirmation point
                # (not earlier — stock must be evaluated by MACD first)
                if cross_time_max > 0 and cross_time_min > cross_time_max:
                    pos_count = 0
                    continue
                if min_vol_cross > 0 and vol_at_cross < min_vol_cross:
                    pos_count = 0
                    continue
                if max_vol_cross > 0 and vol_at_cross > max_vol_cross:
                    pos_count = 0
                    continue
                # Phase D: vol_at_cross exclusion zone (data-driven dead zone)
                if excl_vol_low is not None and excl_vol_low <= vol_at_cross < excl_vol_high:
                    pos_count = 0
                    continue
                if min_macd_pct > 0 and hist_pct < min_macd_pct:
                    pos_count = 0
                    continue
                if max_price_entry > 0 and raw_price > max_price_entry:
                    pos_count = 0
                    continue

                # --- New indicator filters ---
                closes_so_far = bars['close'].iloc[:i + 1]

                # RSI: skip overbought entries
                if max_rsi_val > 0 and len(closes_so_far) >= 15:
                    from trading.indicators import rsi as _rsi
                    rsi_val = _rsi(closes_so_far).iloc[-1]
                    if not pd.isna(rsi_val) and rsi_val > max_rsi_val:
                        pos_count = 0
                        continue

                # MACD histogram acceleration: current bar > previous bar
                if require_hist_accel and i >= 1:
                    if histogram.iloc[i] <= histogram.iloc[i - 1]:
                        pos_count = 0
                        continue

                # OBV rising: OBV at entry > OBV 3 bars ago
                if require_obv_rising and len(closes_so_far) >= 4:
                    from trading.indicators import obv as _obv
                    vols_so_far = bars['volume'].iloc[:i + 1]
                    obv_series = _obv(closes_so_far, vols_so_far)
                    if obv_series.iloc[-1] <= obv_series.iloc[-4]:
                        pos_count = 0
                        continue

                # EMA alignment: price above EMA(21)
                if require_ema_above and len(closes_so_far) >= 22:
                    from trading.indicators import ema as _ema
                    ema21 = _ema(closes_so_far, 21).iloc[-1]
                    if raw_price < ema21:
                        pos_count = 0
                        continue

                # --- Phase A: earlier-entry secondary confirmations ---
                # VWAP: price above intraday VWAP at entry bar
                if require_vwap_above and i >= 1:
                    from trading.indicators import vwap as _vwap
                    bars_so_far = bars.iloc[:i + 1]
                    v = _vwap(bars_so_far['high'], bars_so_far['low'],
                              bars_so_far['close'], bars_so_far['volume']).iloc[-1]
                    if raw_price < v:
                        pos_count = 0
                        continue

                # Volume surge: entry-bar volume > prev-bar volume * 1.5
                if require_vol_surge and i >= 1:
                    cur_vol = bars.iloc[i]['volume']
                    prev_vol = bars.iloc[i - 1]['volume']
                    if prev_vol <= 0 or cur_vol < prev_vol * 1.5:
                        pos_count = 0
                        continue

                # Strong close: entry-bar close in upper third of bar range
                if require_strong_close:
                    bar = bars.iloc[i]
                    bar_range = bar['high'] - bar['low']
                    if bar_range <= 0:
                        pos_count = 0
                        continue
                    upper_third = bar['low'] + bar_range * (2.0 / 3.0)
                    if bar['close'] < upper_third:
                        pos_count = 0
                        continue

                # Higher high: entry-bar high > prev-bar high
                if require_higher_high and i >= 1:
                    if bars.iloc[i]['high'] <= bars.iloc[i - 1]['high']:
                        pos_count = 0
                        continue
                # --- End new indicator filters ---

                entry_price = raw_price * (1 + entry_slippage)
                entry_time = bar_ts
                entry_idx = i
                entry_hist_pct = hist_pct
                in_trade = True
                highest_since_entry = entry_price
                pos_count = 0
                # Phase C: partial profit tracking
                partial_taken = False
                partial_pnl_dollar = 0.0
                partial_shares_sold = 0
                # Conviction-based sizing (OOS-validated in step 1 research)
                conv_mult, _ = compute_conviction_score(cross_time_min, vol_at_cross)
                if conviction_sizing_enabled:
                    effective_position = position_size * conv_mult
                else:
                    effective_position = position_size
                entry_shares = int(effective_position / entry_price) if entry_price > 0 else 0
        else:
            # Force close at 15:45 ET (matches production)
            # Bar index 0 = 9:30 ET, so index 375 = 15:45 ET
            if i >= 375:
                raw_exit = bars.iloc[i]['close']
                exit_price = raw_exit * (1 - exit_slippage)
                wave += 1
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                shares = entry_shares
                # Phase C: blend partial profit with remaining shares
                remaining_shares = shares - partial_shares_sold
                pnl_dollar = (exit_price - entry_price) * remaining_shares + partial_pnl_dollar
                is_paper = w1_scout and wave == 1
                if not is_paper:
                    signals.append({
                        'wave': wave, 'entry_price': entry_price, 'exit_price': exit_price,
                        'shares': shares, 'pnl_pct': pnl_dollar / (entry_price * shares) * 100 if shares > 0 else pnl_pct,
                        'pnl_dollar': pnl_dollar,
                        'entry_time': entry_time, 'exit_time': bar_ts,
                        'exit_reason': 'force_close', 'paper': False,
                        'cross_time_min': cross_time_min, 'vol_at_cross': vol_at_cross,
                        'macd_hist_pct': entry_hist_pct, 'w1_pnl': w1_result,
                        'conv_mult': conv_mult,
                        'entry_idx': entry_idx,
                        'partial_taken': partial_taken, 'partial_pnl': partial_pnl_dollar,
                    })
                in_trade = False
                continue

            # Track highest high for trailing stop
            bar_high = bars.iloc[i]['high']
            if bar_high > highest_since_entry:
                highest_since_entry = bar_high

            # Phase C: take partial profit at +X% (uses HIGH so target gets hit if bar reached it)
            if partial_pct > 0 and not partial_taken:
                partial_target = entry_price * (1 + partial_pct / 100.0)
                if bar_high >= partial_target:
                    total_shares = entry_shares
                    partial_shares_sold = int(total_shares * partial_fraction)
                    partial_exit_price = partial_target * (1 - exit_slippage)
                    partial_pnl_dollar = (partial_exit_price - entry_price) * partial_shares_sold
                    partial_taken = True

            # Check hard stop
            bar_low = bars.iloc[i]['low']
            hard_stop_price = entry_price * (1 - hard_stop_pct)
            if bar_low <= hard_stop_price:
                exit_price = hard_stop_price * (1 - exit_slippage)
                wave += 1
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                shares = entry_shares

                if wave == 1:
                    w1_result = pnl_pct

                is_paper = w1_scout and wave == 1
                if not is_paper:
                    # Check W1 scout qualification for W2+
                    if w1_scout and wave >= 2 and w1_result < w1_min_pct:
                        break  # W1 didn't qualify, skip remaining waves

                # Phase C: blend partial profit with remaining shares
                remaining_shares = shares - partial_shares_sold
                pnl_dollar = (exit_price - entry_price) * remaining_shares + partial_pnl_dollar
                signals.append({
                    'wave': wave, 'entry_price': entry_price, 'exit_price': exit_price,
                    'shares': shares,
                    'pnl_pct': pnl_dollar / (entry_price * shares) * 100 if shares > 0 else pnl_pct,
                    'pnl_dollar': pnl_dollar,
                    'entry_time': entry_time, 'exit_time': bar_ts,
                    'exit_reason': 'hard_stop', 'paper': is_paper,
                    'cross_time_min': cross_time_min, 'vol_at_cross': vol_at_cross,
                    'macd_hist_pct': entry_hist_pct, 'w1_pnl': w1_result,
                    'conv_mult': conv_mult,
                    'partial_taken': partial_taken, 'partial_pnl': partial_pnl_dollar,
                })
                in_trade = False
                pos_count = 0
                continue

            # Check trailing stop
            trail_stop_pct = entry_filters.get('trail_stop_pct', 0)
            # Phase B: trail activation — only check trail after profit > threshold
            current_profit_pct = (highest_since_entry - entry_price) / entry_price * 100
            trail_active = (trail_activate_pct <= 0) or (current_profit_pct >= trail_activate_pct)
            if trail_stop_pct > 0 and trail_active:
                trail_stop_price = highest_since_entry * (1 - trail_stop_pct)
                if bar_low <= trail_stop_price:
                    exit_price = trail_stop_price * (1 - exit_slippage)
                    wave += 1
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                    shares = entry_shares

                    if wave == 1:
                        w1_result = pnl_pct

                    is_paper = w1_scout and wave == 1
                    if not is_paper:
                        if w1_scout and wave >= 2 and w1_result < w1_min_pct:
                            break

                    # Phase C: blend partial profit with remaining shares
                    remaining_shares = shares - partial_shares_sold
                    pnl_dollar = (exit_price - entry_price) * remaining_shares + partial_pnl_dollar
                    signals.append({
                        'wave': wave, 'entry_price': entry_price, 'exit_price': exit_price,
                        'shares': shares,
                        'pnl_pct': pnl_dollar / (entry_price * shares) * 100 if shares > 0 else pnl_pct,
                        'pnl_dollar': pnl_dollar,
                        'entry_time': entry_time, 'exit_time': bar_ts,
                        'exit_reason': 'trail_stop', 'paper': is_paper,
                        'cross_time_min': cross_time_min, 'vol_at_cross': vol_at_cross,
                        'macd_hist_pct': entry_hist_pct, 'w1_pnl': w1_result,
                        'conv_mult': conv_mult,
                        'partial_taken': partial_taken, 'partial_pnl': partial_pnl_dollar,
                    })
                    in_trade = False
                    pos_count = 0
                    continue

            # Check MACD flip
            if h <= 0:
                raw_exit = bars.iloc[i]['close']
                exit_price = raw_exit * (1 - exit_slippage)
                wave += 1
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                shares = entry_shares

                if wave == 1:
                    w1_result = pnl_pct

                is_paper = w1_scout and wave == 1
                if not is_paper:
                    if w1_scout and wave >= 2 and w1_result < w1_min_pct:
                        break

                # Phase C: blend partial profit with remaining shares
                remaining_shares = shares - partial_shares_sold
                pnl_dollar = (exit_price - entry_price) * remaining_shares + partial_pnl_dollar
                signals.append({
                    'wave': wave, 'entry_price': entry_price, 'exit_price': exit_price,
                    'shares': shares,
                    'pnl_pct': pnl_dollar / (entry_price * shares) * 100 if shares > 0 else pnl_pct,
                    'pnl_dollar': pnl_dollar,
                    'entry_time': entry_time, 'exit_time': bar_ts,
                    'exit_reason': 'macd_flip', 'paper': is_paper,
                    'cross_time_min': cross_time_min, 'vol_at_cross': vol_at_cross,
                    'macd_hist_pct': entry_hist_pct, 'w1_pnl': w1_result,
                    'conv_mult': conv_mult,
                    'partial_taken': partial_taken, 'partial_pnl': partial_pnl_dollar,
                })
                in_trade = False
                pos_count = 0

    # Close open trade at EOD
    if in_trade:
        raw_exit = bars.iloc[-1]['close']
        exit_price = raw_exit * (1 - exit_slippage)
        wave += 1
        pnl_pct = (exit_price - entry_price) / entry_price * 100
        shares = entry_shares
        if wave == 1:
            w1_result = pnl_pct
        is_paper = w1_scout and wave == 1
        if not is_paper:
            if w1_scout and wave >= 2 and w1_result < w1_min_pct:
                pass  # Don't append
            else:
                signals.append({
                    'wave': wave, 'entry_price': entry_price, 'exit_price': exit_price,
                    'shares': shares, 'pnl_pct': pnl_pct,
                    'pnl_dollar': (exit_price - entry_price) * shares,
                    'entry_time': entry_time, 'exit_time': str(bars.iloc[-1]['ts']),
                    'exit_reason': 'eod_close', 'paper': is_paper,
                    'cross_time_min': cross_time_min, 'vol_at_cross': vol_at_cross,
                    'macd_hist_pct': entry_hist_pct, 'w1_pnl': w1_result,
                    'conv_mult': conv_mult,
                })
        elif is_paper:
            # Still record W1 for stats even though it's paper
            signals.append({
                'wave': wave, 'entry_price': entry_price, 'exit_price': exit_price,
                'shares': shares, 'pnl_pct': pnl_pct,
                'pnl_dollar': (exit_price - entry_price) * shares,
                'entry_time': entry_time, 'exit_time': str(bars.iloc[-1]['ts']),
                'exit_reason': 'eod_close', 'paper': True,
                'cross_time_min': cross_time_min, 'vol_at_cross': vol_at_cross,
                'macd_hist_pct': entry_hist_pct, 'w1_pnl': w1_result,
                'conv_mult': conv_mult,
            })

    return signals


# ---------------------------------------------------------------------------
# Sequential position simulation
# ---------------------------------------------------------------------------

def simulate_positions(
    all_signals: List[dict],
    max_concurrent: int,
    daily_loss_limit: float,
) -> Tuple[List[Trade], dict]:
    """
    Simulate trades sequentially with position limits.

    Returns (trades, stats) where stats includes skipped counts.
    """
    # Build entry/exit events sorted by time
    events = []
    for sig in all_signals:
        if sig.get('paper', False):
            continue
        events.append(('entry', sig['entry_time'], sig))
        events.append(('exit', sig['exit_time'], sig))

    events.sort(key=lambda e: e[1])

    open_positions = {}  # symbol_date_wave -> signal
    trades = []
    skipped_capacity = 0
    skipped_loss_limit = 0
    daily_pnl = defaultdict(float)  # date -> cumulative P&L

    for event_type, event_time, sig in events:
        trade_date = sig.get('entry_time', '')[:10]

        if event_type == 'entry':
            sym_key = f"{sig['symbol']}_{sig['date']}_{sig['wave']}"

            # Check daily loss limit
            if daily_loss_limit < 0 and daily_pnl[trade_date] <= daily_loss_limit:
                skipped_loss_limit += 1
                continue

            # Check capacity
            if len(open_positions) >= max_concurrent:
                skipped_capacity += 1
                continue

            open_positions[sym_key] = sig

        elif event_type == 'exit':
            sym_key = f"{sig['symbol']}_{sig['date']}_{sig['wave']}"
            if sym_key not in open_positions:
                continue  # Was skipped on entry

            pos = open_positions.pop(sym_key)
            trade = Trade(
                symbol=sig['symbol'],
                date_str=sig['date'],
                wave=sig['wave'],
                entry_price=sig['entry_price'],
                exit_price=sig['exit_price'],
                shares=sig['shares'],
                pnl_pct=sig['pnl_pct'],
                pnl_dollar=sig['pnl_dollar'],
                entry_time=sig['entry_time'],
                exit_time=sig['exit_time'],
                exit_reason=sig['exit_reason'],
                cross_time_min=sig['cross_time_min'],
                vol_at_cross=sig['vol_at_cross'],
                macd_hist_pct=sig['macd_hist_pct'],
                w1_pnl=sig.get('w1_pnl', 0.0),
                conv_mult=sig.get('conv_mult', 1.0),
            )
            trades.append(trade)
            daily_pnl[trade_date] += trade.pnl_dollar

    stats = {
        'skipped_capacity': skipped_capacity,
        'skipped_loss_limit': skipped_loss_limit,
        'total_signals': len([s for s in all_signals if not s.get('paper', False)]),
        'paper_signals': len([s for s in all_signals if s.get('paper', False)]),
    }
    return trades, stats


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_results(trades: List[Trade], stats: dict, cfg: dict, label: str = "") -> None:
    """Print formatted backtest results."""
    if not trades:
        print(f"\n{'='*65}")
        print(f"  {label}: NO TRADES")
        print(f"  Signals: {stats.get('total_signals', 0)} (skipped: {stats.get('skipped_capacity', 0)} capacity, {stats.get('skipped_loss_limit', 0)} loss limit)")
        print(f"{'='*65}")
        return

    wins = [t for t in trades if t.pnl_dollar > 0]
    losses = [t for t in trades if t.pnl_dollar <= 0]
    total_pnl = sum(t.pnl_dollar for t in trades)
    total_pnl_pct = sum(t.pnl_pct for t in trades)
    wr = len(wins) / len(trades) * 100
    avg_win = np.mean([t.pnl_dollar for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl_dollar for t in losses]) if losses else 0
    avg_win_pct = np.mean([t.pnl_pct for t in wins]) if wins else 0
    avg_loss_pct = np.mean([t.pnl_pct for t in losses]) if losses else 0
    pf = abs(sum(t.pnl_dollar for t in wins)) / abs(sum(t.pnl_dollar for t in losses)) if losses else float('inf')

    # Equity curve + max drawdown
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in sorted(trades, key=lambda x: x.entry_time):
        equity += t.pnl_dollar
        peak = max(peak, equity)
        dd = equity - peak
        max_dd = min(max_dd, dd)

    # Daily stats
    by_date = defaultdict(list)
    for t in trades:
        by_date[t.date_str].append(t)
    green_days = len([d for d, tt in by_date.items() if sum(t.pnl_dollar for t in tt) > 0])
    red_days = len([d for d, tt in by_date.items() if sum(t.pnl_dollar for t in tt) <= 0])

    # Sharpe
    daily_returns = [sum(t.pnl_dollar for t in tt) for tt in by_date.values()]
    sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * (252 ** 0.5) if np.std(daily_returns) > 0 else 0

    pos_size = cfg.get('sizing', {}).get('position_size', 40000)

    print(f"\n{'='*65}")
    print(f"  MACD Wave Backtest — {label}")
    print(f"  Position size: ${pos_size:,.0f}")
    print(f"{'='*65}")
    print(f"  Trades: {len(trades)} ({len(wins)}W {len(losses)}L)")
    print(f"  Signals: {stats.get('total_signals', 0)} total, {stats.get('skipped_capacity', 0)} skipped (capacity), {stats.get('skipped_loss_limit', 0)} skipped (loss limit)")
    if stats.get('paper_signals', 0):
        print(f"  Paper (W1 scout): {stats['paper_signals']}")
    print(f"  WR: {wr:.1f}%  |  PF: {pf:.2f}  |  Sharpe: {sharpe:.2f}")
    print(f"  Avg Win: ${avg_win:+,.0f} ({avg_win_pct:+.2f}%)  |  Avg Loss: ${avg_loss:+,.0f} ({avg_loss_pct:+.2f}%)")
    print(f"  Total P&L: ${total_pnl:+,.0f} ({total_pnl_pct:+.1f}%)")
    print(f"  Max Drawdown: ${max_dd:,.0f}")
    print(f"  Days: {green_days} green, {red_days} red")

    # Exit reason breakdown
    reasons = defaultdict(int)
    for t in trades:
        reasons[t.exit_reason] += 1
    print(f"  Exits: {dict(reasons)}")

    # Per-day breakdown
    print(f"\n  Per-day:")
    print(f"  {'Date':<12s} {'Trades':>6} {'Wins':>5} {'P&L $':>10} {'P&L %':>7}")
    print(f"  {'-'*45}")
    for d in sorted(by_date.keys()):
        tt = by_date[d]
        w = len([t for t in tt if t.pnl_dollar > 0])
        pnl = sum(t.pnl_dollar for t in tt)
        pnl_pct = sum(t.pnl_pct for t in tt)
        syms = ', '.join(sorted(set(t.symbol for t in tt)))[:40]
        print(f"  {d:<12s} {len(tt):>6} {w:>5} ${pnl:>+9,.0f} {pnl_pct:>+6.1f}%  {syms}")
    print()


def write_csv(trades: List[Trade], path: str) -> None:
    """Write trade-level CSV."""
    if not trades:
        return
    fieldnames = [
        'symbol', 'date', 'wave', 'entry_time', 'exit_time',
        'entry_price', 'exit_price', 'shares', 'pnl_pct', 'pnl_dollar',
        'exit_reason', 'cross_time_min', 'vol_at_cross', 'macd_hist_pct', 'w1_pnl',
        'conv_mult',
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in sorted(trades, key=lambda x: x.entry_time):
            writer.writerow({
                'symbol': t.symbol, 'date': t.date_str, 'wave': t.wave,
                'entry_time': t.entry_time, 'exit_time': t.exit_time,
                'entry_price': round(t.entry_price, 4),
                'exit_price': round(t.exit_price, 4),
                'shares': t.shares, 'pnl_pct': round(t.pnl_pct, 2),
                'pnl_dollar': round(t.pnl_dollar, 2),
                'exit_reason': t.exit_reason,
                'cross_time_min': t.cross_time_min,
                'vol_at_cross': t.vol_at_cross,
                'macd_hist_pct': round(t.macd_hist_pct, 4),
                'w1_pnl': round(t.w1_pnl, 2),
                'conv_mult': round(t.conv_mult, 3),
            })
    logger.info(f"CSV written: {path} ({len(trades)} trades)")


# ---------------------------------------------------------------------------
# Signal cache — generate once, filter instantly
# ---------------------------------------------------------------------------

CACHE_DIR = "data"


def get_cache_path(trail_pct: float, slip_entry: float, slip_exit: float) -> str:
    """Build cache filename from params that affect exit prices."""
    trail_key = int(trail_pct * 10000)  # 0.003 → 30
    slip_key = int((slip_entry + slip_exit) * 10000)
    return os.path.join(CACHE_DIR, f"macd_signal_cache_t{trail_key}_s{slip_key}.csv")


def save_signal_cache(signals: List[dict], cache_path: str) -> None:
    """Save all unfiltered signals to CSV cache."""
    import csv as csv_mod
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    fields = ['symbol', 'date', 'wave', 'entry_price', 'exit_price', 'shares',
              'pnl_pct', 'pnl_dollar', 'entry_time', 'exit_time', 'exit_reason',
              'cross_time_min', 'vol_at_cross', 'macd_hist_pct', 'w1_pnl', 'paper']
    with open(cache_path, 'w', newline='') as f:
        w = csv_mod.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(signals)
    logger.info(f"Signal cache saved: {cache_path} ({len(signals)} signals)")


def load_signal_cache(cache_path: str, start_date: date, end_date: date) -> List[dict]:
    """Load cached signals, filter by date range.

    Also computes conv_mult for each signal from cross_time_min + vol_at_cross
    so downstream can apply conviction-based sizing without regenerating signals.
    """
    import csv as csv_mod
    signals = []
    with open(cache_path) as f:
        for row in csv_mod.DictReader(f):
            d = row['date']
            if d < str(start_date) or d > str(end_date):
                continue
            row['wave'] = int(row['wave'])
            row['entry_price'] = float(row['entry_price'])
            row['exit_price'] = float(row['exit_price'])
            row['shares'] = int(row['shares'])
            row['pnl_pct'] = float(row['pnl_pct'])
            row['pnl_dollar'] = float(row['pnl_dollar'])
            row['cross_time_min'] = int(row['cross_time_min'])
            row['vol_at_cross'] = int(row['vol_at_cross'])
            row['macd_hist_pct'] = float(row['macd_hist_pct'])
            row['w1_pnl'] = float(row['w1_pnl'])
            row['paper'] = row.get('paper', 'False') == 'True'
            # Compute conviction multiplier from raw signal features
            conv, _ = compute_conviction_score(row['cross_time_min'], row['vol_at_cross'])
            row['conv_mult'] = conv
            signals.append(row)
    logger.info(f"Loaded {len(signals)} cached signals from {cache_path} ({start_date} to {end_date})")
    return signals


def filter_signals(signals: List[dict], entry_filters: dict) -> List[dict]:
    """Apply entry filters to cached signals in memory.

    If conviction_sizing is enabled, scale each signal's shares and pnl_dollar
    by conv_mult. This lets us apply conviction sizing to cached signals without
    regenerating (since shares and pnl_dollar both scale linearly with the
    position-size multiplier).
    """
    cross_time_max = entry_filters.get('cross_time_max_min', 0)
    min_vol_cross = entry_filters.get('min_vol_at_cross', 0)
    max_vol_cross = entry_filters.get('max_vol_at_cross', 0)
    min_macd_pct = entry_filters.get('min_macd_hist_pct', 0.0)
    max_price_entry = entry_filters.get('max_price_at_entry', 0)
    conviction_sizing = bool(entry_filters.get('conviction_sizing', False))

    filtered = []
    for sig in signals:
        if cross_time_max > 0 and sig['cross_time_min'] > cross_time_max:
            continue
        if min_vol_cross > 0 and sig['vol_at_cross'] < min_vol_cross:
            continue
        if max_vol_cross > 0 and sig['vol_at_cross'] > max_vol_cross:
            continue
        if min_macd_pct > 0 and sig['macd_hist_pct'] < min_macd_pct:
            continue
        if max_price_entry > 0 and sig['entry_price'] > max_price_entry:
            continue
        # Conviction sizing: shares and pnl_dollar scale linearly with position size.
        # pnl_pct is a ratio and stays the same.
        if conviction_sizing:
            cm = sig.get('conv_mult', 1.0)
            if cm != 1.0:
                sig = dict(sig)  # avoid mutating cached dict
                sig['shares'] = int(sig['shares'] * cm)
                sig['pnl_dollar'] = sig['pnl_dollar'] * cm
        filtered.append(sig)

    logger.info(f"Filter: {len(signals)} → {len(filtered)} signals "
                f"(cross<{cross_time_max}m, vol {min_vol_cross}-{max_vol_cross}, "
                f"macd>={min_macd_pct}%, price<={max_price_entry})"
                + (" [conviction sizing ON]" if conviction_sizing else ""))
    return filtered


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="MACD Wave Strategy Backtest")
    parser.add_argument("--start", type=str, default="2026-03-01")
    parser.add_argument("--end", type=str, default="2026-03-27")
    parser.add_argument("--config", type=str, default=CONFIG_PATH)
    parser.add_argument("--output", type=str, default="macd_wave_results.csv")

    # Filter overrides
    parser.add_argument("--cross-time", type=int, default=None, help="Max minutes to +10% cross")
    parser.add_argument("--macd-min", type=float, default=None, help="Min MACD hist pct at entry")
    parser.add_argument("--max-price", type=float, default=None, help="Max price at entry")
    parser.add_argument("--min-vol", type=int, default=None, help="Min volume at cross")
    parser.add_argument("--max-vol", type=int, default=None, help="Max volume at cross")
    parser.add_argument("--max-waves", type=int, default=None, help="Max waves per stock/day")
    parser.add_argument("--w1-scout", action="store_true", help="W1 is paper, trade W2+")
    parser.add_argument("--w1-min", type=float, default=None, help="Min W1 %% to qualify W2+")
    parser.add_argument("--position-size", type=float, default=None)
    parser.add_argument("--max-concurrent", type=int, default=None)
    parser.add_argument("--trail", type=float, default=None, help="Trailing stop %% below highest (e.g., 0.5 = 0.5%%)")
    parser.add_argument("--conviction-sizing", action="store_true",
                        help="Scale position size by conviction_mult (2-rule score, OOS-validated)")
    parser.add_argument("--no-slippage", action="store_true")
    parser.add_argument("--entry-slip", type=float, default=None, help="Entry slippage pct (e.g. 0.005 = 0.5%%)")
    parser.add_argument("--exit-slip", type=float, default=None, help="Exit slippage pct")
    # New indicator filters
    parser.add_argument("--max-rsi", type=float, default=0, help="Max RSI at entry (e.g. 70 = skip overbought)")
    parser.add_argument("--require-hist-accel", action="store_true", help="Only enter when MACD histogram accelerating")
    parser.add_argument("--require-obv-rising", action="store_true", help="Only enter when OBV confirms move")
    parser.add_argument("--require-ema-above", action="store_true", help="Only enter when price > EMA21")
    # Phase A: earlier-entry hypotheses
    parser.add_argument("--confirm-bars", type=int, default=None, help="Override MACD confirmation bars (default 3)")
    parser.add_argument("--require-vwap-above", action="store_true", help="Require price > VWAP at entry")
    parser.add_argument("--require-vol-surge", action="store_true", help="Require entry-bar volume > prev-bar volume * 1.5")
    parser.add_argument("--require-strong-close", action="store_true", help="Require entry-bar close in upper third of bar range")
    parser.add_argument("--require-higher-high", action="store_true", help="Require entry-bar high > prev-bar high")
    # Phase B: exit optimization
    parser.add_argument("--hard-stop", type=float, default=None, help="Hard stop pct (e.g. 0.015 = 1.5)")
    parser.add_argument("--trail-activate-pct", type=float, default=0, help="Trail activates only after profit > X pct (e.g. 0.5)")
    # Phase C: partial profit
    parser.add_argument("--partial-pct", type=float, default=0, help="Take partial profit at +X pct (e.g. 1.0)")
    parser.add_argument("--partial-fraction", type=float, default=0.5, help="Fraction to sell on partial (default 0.5)")
    # Phase D: data-driven bucket exclusion
    parser.add_argument("--exclude-vol-zone", type=str, default=None, help="Skip vol_at_cross in this zone, format LOW-HIGH (e.g. 100000-150000)")
    parser.add_argument("--build-cache", action="store_true",
                        help="Generate ALL signals (no entry filters) and save to cache CSV")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force regeneration even if cache exists")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config
    cfg = load_config(args.config)

    # Apply CLI overrides
    entry_cfg = cfg.setdefault('entry', {})
    wave_cfg = cfg.setdefault('waves', {})
    sizing_cfg = cfg.setdefault('sizing', {})
    slip_cfg = cfg.setdefault('slippage', {})
    risk_cfg = cfg.setdefault('risk', {})

    if args.cross_time is not None:
        entry_cfg['cross_time_max_min'] = args.cross_time
    if args.macd_min is not None:
        entry_cfg['min_macd_hist_pct'] = args.macd_min
    if args.max_price is not None:
        entry_cfg['max_price_at_entry'] = args.max_price
    if args.min_vol is not None:
        entry_cfg['min_vol_at_cross'] = args.min_vol
    if args.max_vol is not None:
        entry_cfg['max_vol_at_cross'] = args.max_vol
    if args.max_waves is not None:
        wave_cfg['max_waves'] = args.max_waves
    if args.w1_scout:
        wave_cfg['w1_scout'] = True
    if args.w1_min is not None:
        wave_cfg['w1_min_pct'] = args.w1_min
    if args.position_size is not None:
        sizing_cfg['position_size'] = args.position_size
    if args.max_concurrent is not None:
        sizing_cfg['max_concurrent'] = args.max_concurrent
    if args.no_slippage:
        slip_cfg['entry_pct'] = 0.0
        slip_cfg['exit_pct'] = 0.0

    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)
    uni_cfg = cfg.get('universe', {})

    # Build label
    filters = []
    if entry_cfg.get('cross_time_max_min', 0) > 0:
        filters.append(f"cross<{entry_cfg['cross_time_max_min']}m")
    if entry_cfg.get('min_macd_hist_pct', 0) > 0:
        filters.append(f"MACD>={entry_cfg['min_macd_hist_pct']}%")
    if entry_cfg.get('max_price_at_entry', 0) > 0:
        filters.append(f"price<=${entry_cfg['max_price_at_entry']}")
    if entry_cfg.get('min_vol_at_cross', 0) > 0:
        filters.append(f"vol>={entry_cfg['min_vol_at_cross']:,}")
    if wave_cfg.get('w1_scout', False):
        filters.append(f"W1scout>={wave_cfg.get('w1_min_pct', 0)}%")
    label = f"{args.start} to {args.end}"
    if filters:
        label += f" | {' + '.join(filters)}"
    if args.trail is not None:
        filters.append(f"trail={args.trail}%")
    if args.no_slippage:
        label += " | NO SLIPPAGE"

    t0 = time_mod.time()

    # Step 3: Generate signals (use cache if available)
    trail_pct = args.trail / 100 if args.trail is not None else risk_cfg.get('trail_stop_pct', 0.003)
    slip_entry = args.entry_slip if args.entry_slip is not None else slip_cfg.get('entry_pct', 0.003)
    slip_exit = args.exit_slip if args.exit_slip is not None else slip_cfg.get('exit_pct', 0.003)

    entry_filters = {
        **entry_cfg,
        'max_waves': wave_cfg.get('max_waves', 1),
        'w1_scout': wave_cfg.get('w1_scout', False),
        'w1_min_pct': wave_cfg.get('w1_min_pct', 0.0),
        'position_size': sizing_cfg.get('position_size', 40000),
        'entry_pct': slip_entry,
        'exit_pct': slip_exit,
        'trail_stop_pct': trail_pct,
        # New indicator filters
        'max_rsi': args.max_rsi,
        'require_hist_accel': args.require_hist_accel,
        'require_obv_rising': args.require_obv_rising,
        'require_ema_above': args.require_ema_above,
        # Phase A
        'confirm_bars_override': args.confirm_bars,
        'require_vwap_above': args.require_vwap_above,
        'require_vol_surge': args.require_vol_surge,
        'require_strong_close': args.require_strong_close,
        'require_higher_high': args.require_higher_high,
        # Phase B
        'hard_stop_override': args.hard_stop,
        'trail_activate_pct': args.trail_activate_pct,
        # Phase C
        'partial_pct': args.partial_pct,
        'partial_fraction': args.partial_fraction,
        # Phase D
        'exclude_vol_zone': args.exclude_vol_zone,
        # Conviction-based sizing (opt-in)
        'conviction_sizing': bool(args.conviction_sizing),
    }

    cache_path = get_cache_path(trail_pct, slip_entry, slip_exit)
    use_cache = os.path.exists(cache_path) and not args.build_cache and not args.no_cache

    if use_cache:
        # Fast path: skip find_movers + load_bars entirely
        all_signals = load_signal_cache(cache_path, start_date, end_date)
        all_signals = filter_signals(all_signals, entry_filters)
        movers = []  # Not needed for cached path
    else:
        # Slow path: find movers, load bars, generate signals
        movers = find_movers(
            start_date, end_date,
            min_price=uni_cfg.get('min_price', 10.0),
            max_price=uni_cfg.get('max_price', 0),
            min_volume=int(uni_cfg.get('min_daily_volume', 1_000_000)),
            min_intraday_pct=uni_cfg.get('min_intraday_pct', 10.0),
        )
        if not movers:
            print("No movers found.")
            return

        from persistence.database import get_database
        from config import Config
        _cfg = Config._load_yaml_only()
        _db_cfg = _cfg.get("database", {})
        db = get_database(
            db_path=_db_cfg.get("path"),
            cache_path=_db_cfg.get("cache_path"),
            trades_path=_db_cfg.get("trades_path"),
        )
        bar_cache = load_intraday_bars(movers, db)

        # For --build-cache: disable all entry filters to capture everything
        gen_filters = dict(entry_filters)
        if args.build_cache:
            gen_filters['cross_time_max_min'] = 0
            gen_filters['min_vol_at_cross'] = 0
            gen_filters['max_vol_at_cross'] = 0
            gen_filters['min_macd_hist_pct'] = 0
            gen_filters['max_price_at_entry'] = 0
            logger.info("Building signal cache (all entry filters disabled)...")

        all_signals = []
        candidates = 0
        filtered_out = 0

        for sym, d, pct, close, vol in movers:
            bars = bar_cache.get((sym, d))
            if bars is None:
                continue
            candidates += 1
            sigs = generate_signals(bars, cfg, gen_filters)
            for sig in sigs:
                sig['symbol'] = sym
                sig['date'] = d
            if sigs:
                all_signals.extend(sigs)
            else:
                filtered_out += 1

        logger.info(f"Generated {len(all_signals)} signals from {candidates} candidates ({filtered_out} filtered out)")

        if args.build_cache:
            save_signal_cache(all_signals, cache_path)
            # Apply entry filters for this run's output
            all_signals = filter_signals(all_signals, entry_filters)

    # Step 4: Sequential position simulation
    trades, sim_stats = simulate_positions(
        all_signals,
        max_concurrent=sizing_cfg.get('max_concurrent', 5),
        daily_loss_limit=risk_cfg.get('daily_loss_limit', -5000),
    )

    # Step 5: Output
    elapsed = time_mod.time() - t0
    print_results(trades, sim_stats, cfg, label)
    write_csv(trades, args.output)
    logger.info(f"Completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
