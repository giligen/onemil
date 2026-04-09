#!/usr/bin/env python3
"""
Item 14: Winner/Loser Differentiation — Feature Enrichment & Analysis

Phase 0: Enrich each trade with 22 features from DB (1-min bars, daily bars, SPY data)
Phase 1: Statistical analysis on Q1 2026 (find what separates W from L)
Phase 2: Hypothesis generation & testing on Q1
Phase 3: Out-of-sample validation on 2025

All features must be KNOWN AT ENTRY TIME — no look-ahead bias.
"""
import csv
import os
import sys
import sqlite3
import logging
import statistics
import numpy as np
from collections import defaultdict
from datetime import datetime, date, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = "data/cache.db"
CACHE_PATH = "data/bull_flag_cache_e50_x30_t20.csv"
OUTPUT_DIR = "analysis_results"
ENRICHED_CSV = f"{OUTPUT_DIR}/enriched_trades_t20.csv"


def get_db():
    """Get SQLite connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def load_cache_trades(path):
    """Load all trades from cache CSV."""
    trades = []
    with open(path) as f:
        for row in csv.DictReader(f):
            trades.append(dict(row))
    logger.info(f"Loaded {len(trades)} trades from {path}")
    return trades


def get_prev_close(conn, symbol, trade_date):
    """Get previous day's close price."""
    row = conn.execute(
        'SELECT close FROM daily_bars WHERE symbol=? AND bar_date<? ORDER BY bar_date DESC LIMIT 1',
        (symbol, trade_date)
    ).fetchone()
    return float(row['close']) if row else None


def get_1min_bars(conn, symbol, trade_date):
    """Get 1-min bars for a symbol on a date."""
    rows = conn.execute(
        'SELECT timestamp, open, high, low, close, volume FROM intraday_bars_1min '
        'WHERE symbol=? AND DATE(timestamp)=? ORDER BY timestamp',
        (symbol, trade_date)
    ).fetchall()
    if not rows:
        return None
    return [dict(r) for r in rows]


def get_spy_bars(conn, trade_date):
    """Get SPY 1-min bars for a date."""
    return get_1min_bars(conn, 'SPY', trade_date)


def get_spy_daily(conn, trade_date, lookback=5):
    """Get SPY daily bars for trend calculation."""
    rows = conn.execute(
        'SELECT bar_date, close FROM daily_bars WHERE symbol=? AND bar_date<=? ORDER BY bar_date DESC LIMIT ?',
        ('SPY', trade_date, lookback + 1)
    ).fetchall()
    return [dict(r) for r in rows]


def get_recent_range(conn, symbol, trade_date, lookback=5):
    """Get stock's recent average daily range."""
    rows = conn.execute(
        'SELECT high, low FROM daily_bars WHERE symbol=? AND bar_date<? ORDER BY bar_date DESC LIMIT ?',
        (symbol, trade_date, lookback)
    ).fetchall()
    if not rows:
        return None
    ranges = [(float(r['high']) - float(r['low'])) / float(r['low']) * 100
              for r in rows if float(r['low']) > 0]
    return statistics.mean(ranges) if ranges else None


def compute_vwap(bars, up_to_idx):
    """Compute VWAP up to a specific bar index."""
    cum_vol = 0
    cum_tp_vol = 0
    for i in range(up_to_idx + 1):
        b = bars[i]
        tp = (float(b['high']) + float(b['low']) + float(b['close'])) / 3
        vol = float(b['volume'])
        cum_vol += vol
        cum_tp_vol += tp * vol
    return cum_tp_vol / cum_vol if cum_vol > 0 else None


def compute_macd(closes, fast=12, slow=26, signal=9):
    """Compute MACD histogram from close prices."""
    if len(closes) < slow + signal:
        return None

    def ema(data, period):
        result = [data[0]]
        mult = 2.0 / (period + 1)
        for i in range(1, len(data)):
            result.append(data[i] * mult + result[-1] * (1 - mult))
        return result

    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
    signal_line = ema(macd_line[slow-1:], signal)  # signal of MACD from slow start

    if not signal_line:
        return None

    histogram = macd_line[-1] - signal_line[-1]
    # Express as % of price
    last_price = closes[-1]
    return (histogram / last_price * 100) if last_price > 0 else None


def find_entry_bar_idx(bars, entry_time_str, trade_date):
    """Find the index of the entry bar in 1-min bars."""
    # entry_time_str is like "09:45:00"
    # bars have timestamps like "2026-01-05 14:45:00+00:00" (UTC)
    # Convert ET entry time to UTC (+5h)
    h, m, s = entry_time_str.split(':')
    et_h, et_m = int(h), int(m)
    utc_h = et_h + 5  # ET to UTC (simplified, ignoring DST edge cases)
    target_utc = f"{utc_h:02d}:{et_m:02d}"

    for i, b in enumerate(bars):
        ts = b['timestamp']
        # Extract HH:MM from timestamp
        if target_utc in ts:
            return i
    # Fallback: find closest bar
    for i, b in enumerate(bars):
        ts = b['timestamp']
        bar_h = int(ts[11:13])
        bar_m = int(ts[14:16])
        if bar_h == utc_h and bar_m == et_m:
            return i
    return None


def estimate_pole_and_flag(bars, entry_idx, stop_loss, entry_price):
    """Estimate pole and flag characteristics from bars before entry.

    The pattern is: pole up → flag (pullback) → breakout at entry.
    Stop loss is typically at the flag low.
    Entry price is at the breakout level.
    """
    if entry_idx is None or entry_idx < 5:
        return {}

    stop = float(stop_loss)
    entry = float(entry_price)

    # Work backwards from entry to find the flag and pole
    # The flag low ≈ stop_loss level
    # The flag starts where the pullback begins (local high before entry)

    # Find the local high before entry (pole top / flag start)
    pole_top = 0
    pole_top_idx = entry_idx
    for i in range(max(0, entry_idx - 30), entry_idx):
        h = float(bars[i]['high'])
        if h > pole_top:
            pole_top = h
            pole_top_idx = i

    # Find the pole start (local low before the pole top)
    pole_bottom = float('inf')
    pole_bottom_idx = pole_top_idx
    for i in range(max(0, pole_top_idx - 20), pole_top_idx):
        l = float(bars[i]['low'])
        if l < pole_bottom:
            pole_bottom = l
            pole_bottom_idx = i

    if pole_bottom <= 0 or pole_top <= pole_bottom:
        return {}

    pole_gain_pct = (pole_top - pole_bottom) / pole_bottom * 100
    pole_bars = pole_top_idx - pole_bottom_idx
    retracement_pct = (pole_top - stop) / (pole_top - pole_bottom) * 100 if (pole_top - pole_bottom) > 0 else 0

    # Flag tightness: (flag_high - flag_low) / pole_height
    flag_bars = list(range(pole_top_idx, entry_idx + 1))
    if flag_bars:
        flag_high = max(float(bars[i]['high']) for i in flag_bars if i < len(bars))
        flag_low = min(float(bars[i]['low']) for i in flag_bars if i < len(bars))
        flag_tightness = (flag_high - flag_low) / (pole_top - pole_bottom) * 100 if (pole_top - pole_bottom) > 0 else 0
    else:
        flag_tightness = 0

    # Green bar ratio in pole
    pole_range = range(pole_bottom_idx, pole_top_idx + 1)
    green_bars = sum(1 for i in pole_range if i < len(bars) and float(bars[i]['close']) > float(bars[i]['open']))
    green_ratio = green_bars / max(1, len(list(pole_range))) * 100

    # Volume analysis
    pole_vols = [float(bars[i]['volume']) for i in pole_range if i < len(bars)]
    flag_vols = [float(bars[i]['volume']) for i in flag_bars if i < len(bars)]
    avg_pole_vol = statistics.mean(pole_vols) if pole_vols else 0
    avg_flag_vol = statistics.mean(flag_vols) if flag_vols else 0
    vol_ratio = avg_pole_vol / avg_flag_vol if avg_flag_vol > 0 else 0

    # Pullback volume slope (positive = increasing = bearish)
    pullback_vol_slope = 0
    if len(flag_vols) >= 3:
        x = list(range(len(flag_vols)))
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(flag_vols)
        num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, flag_vols))
        den = sum((xi - x_mean) ** 2 for xi in x)
        pullback_vol_slope = num / den if den > 0 else 0

    return {
        'pole_gain_pct': round(pole_gain_pct, 2),
        'pole_bars': pole_bars,
        'retracement_pct': round(retracement_pct, 2),
        'flag_tightness': round(flag_tightness, 2),
        'green_bar_ratio': round(green_ratio, 1),
        'vol_ratio_pole_vs_pullback': round(vol_ratio, 2),
        'pullback_vol_slope': round(pullback_vol_slope, 1),
    }


def enrich_trade(trade, conn, spy_cache, recent_trades_by_date):
    """Enrich a single trade with all 22 features."""
    symbol = trade['symbol']
    trade_date = trade['date']
    entry_time = trade['entry_time_et']
    entry_price = float(trade['entry_price'])
    stop_loss = float(trade['stop_loss'])

    features = {}

    # 1. Gap % from prev close
    prev_close = get_prev_close(conn, symbol, trade_date)
    if prev_close and prev_close > 0:
        features['gap_pct'] = round((entry_price - prev_close) / prev_close * 100, 2)
        # Actually, gap is open vs prev_close. Let's use first bar's open if available
    else:
        features['gap_pct'] = None

    # Get 1-min bars
    bars = get_1min_bars(conn, symbol, trade_date)
    entry_idx = None
    if bars:
        entry_idx = find_entry_bar_idx(bars, entry_time, trade_date)

        # Fix gap_pct to use actual open price
        if prev_close and prev_close > 0:
            open_price = float(bars[0]['open'])
            features['gap_pct'] = round((open_price - prev_close) / prev_close * 100, 2)

    # 2-4. Volume features at entry
    if bars and entry_idx is not None and entry_idx > 0:
        entry_bar = bars[entry_idx]
        features['breakout_bar_volume'] = float(entry_bar['volume'])

        # Avg bar volume up to entry (first 30 min or up to entry, whichever is less)
        first_30_end = min(30, entry_idx)
        if first_30_end > 0:
            avg_vol = statistics.mean([float(bars[i]['volume']) for i in range(first_30_end)])
            features['avg_bar_volume_30m'] = round(avg_vol, 0)
            features['relative_volume_at_entry'] = round(
                features['breakout_bar_volume'] / avg_vol, 2
            ) if avg_vol > 0 else None
        else:
            features['avg_bar_volume_30m'] = None
            features['relative_volume_at_entry'] = None

        # Cumulative dollar volume at entry
        cum_dollar_vol = sum(
            float(bars[i]['close']) * float(bars[i]['volume'])
            for i in range(entry_idx + 1)
        )
        features['cum_dollar_vol_at_entry'] = round(cum_dollar_vol, 0)
    else:
        features['breakout_bar_volume'] = None
        features['avg_bar_volume_30m'] = None
        features['relative_volume_at_entry'] = None
        features['cum_dollar_vol_at_entry'] = None

    # 5. VWAP position at entry
    if bars and entry_idx is not None:
        vwap = compute_vwap(bars, entry_idx)
        if vwap:
            features['entry_vs_vwap_pct'] = round((entry_price - vwap) / vwap * 100, 2)
        else:
            features['entry_vs_vwap_pct'] = None
    else:
        features['entry_vs_vwap_pct'] = None

    # 6. MACD histogram at entry
    if bars and entry_idx is not None and entry_idx >= 35:
        closes = [float(bars[i]['close']) for i in range(entry_idx + 1)]
        # Include prev day bars for warm-up if available
        macd_val = compute_macd(closes)
        features['macd_histogram_pct'] = round(macd_val, 4) if macd_val is not None else None
    else:
        features['macd_histogram_pct'] = None

    # 7-8. Pole/flag analysis
    if bars and entry_idx is not None:
        pattern_features = estimate_pole_and_flag(bars, entry_idx, stop_loss, entry_price)
        features.update(pattern_features)
    else:
        features['pole_gain_pct'] = None
        features['pole_bars'] = None
        features['retracement_pct'] = None
        features['flag_tightness'] = None
        features['green_bar_ratio'] = None
        features['vol_ratio_pole_vs_pullback'] = None
        features['pullback_vol_slope'] = None

    # 9. SPY return at entry time
    spy_bars = spy_cache.get(trade_date)
    if spy_bars and entry_idx is not None:
        spy_open = float(spy_bars[0]['open']) if spy_bars else None
        # Find SPY bar at entry time
        spy_entry_idx = min(entry_idx, len(spy_bars) - 1) if spy_bars else None
        if spy_open and spy_entry_idx is not None and spy_open > 0:
            spy_at_entry = float(spy_bars[spy_entry_idx]['close'])
            features['spy_return_at_entry'] = round((spy_at_entry - spy_open) / spy_open * 100, 3)
        else:
            features['spy_return_at_entry'] = None
    else:
        features['spy_return_at_entry'] = None

    # 10. SPY 5-day trend
    spy_daily = get_spy_daily(conn, trade_date, 5)
    if len(spy_daily) >= 2:
        spy_now = float(spy_daily[0]['close'])
        spy_5d = float(spy_daily[-1]['close'])
        features['spy_5day_return'] = round((spy_now - spy_5d) / spy_5d * 100, 2) if spy_5d > 0 else None
    else:
        features['spy_5day_return'] = None

    # 11. Setup number (how many trades on same symbol same day before this one)
    same_day_trades = recent_trades_by_date.get(trade_date, [])
    setup_num = 1
    for t in same_day_trades:
        if t['symbol'] == symbol and t['entry_time_et'] < entry_time:
            setup_num += 1
    features['setup_number'] = setup_num

    # 12. Bars since qualification (approximate: entry_idx gives bars from open)
    features['bars_since_open'] = entry_idx if entry_idx is not None else None

    # 13. News catalyst type — from trades DB
    # Trades DB is separate from cache DB
    try:
        trades_conn = sqlite3.connect("data/trades.db")
        trades_conn.row_factory = sqlite3.Row
        news_row = trades_conn.execute(
            'SELECT news_headline, news_catalyst FROM trades WHERE symbol=? AND trade_date=? LIMIT 1',
            (symbol, trade_date)
        ).fetchone()
        trades_conn.close()
    except Exception:
        news_row = None
    if news_row:
        features['news_headline'] = news_row['news_headline'] if news_row['news_headline'] else ''
        features['news_catalyst'] = news_row['news_catalyst'] if news_row['news_catalyst'] else 'unknown'
    else:
        features['news_headline'] = ''
        features['news_catalyst'] = 'unknown'

    # 17. Relative move magnitude
    avg_range = get_recent_range(conn, symbol, trade_date, 5)
    daily_range = float(trade.get('daily_range_pct', 0) or 0)
    if avg_range and avg_range > 0:
        features['relative_move_magnitude'] = round(daily_range / avg_range, 2)
    else:
        features['relative_move_magnitude'] = None

    # 20. Spread proxy at entry
    if bars and entry_idx is not None:
        entry_bar = bars[entry_idx]
        bar_range = float(entry_bar['high']) - float(entry_bar['low'])
        bar_close = float(entry_bar['close'])
        features['spread_proxy'] = round(bar_range / bar_close * 100, 3) if bar_close > 0 else None
    else:
        features['spread_proxy'] = None

    # 21. Sector
    sector_row = conn.execute(
        'SELECT sector FROM universe WHERE symbol=? LIMIT 1', (symbol,)
    ).fetchone()
    features['sector'] = sector_row['sector'] if sector_row and sector_row['sector'] else 'unknown'

    # 22. Repeat mover
    # Check if symbol appeared in cache within last 5 dates
    trade_dt = datetime.strptime(trade_date, '%Y-%m-%d').date()
    lookback_date = str(trade_dt - timedelta(days=7))  # 7 calendar days ≈ 5 trading days
    repeat_row = conn.execute(
        'SELECT COUNT(*) as cnt FROM daily_bars WHERE symbol=? AND bar_date>=? AND bar_date<? '
        'AND (high-low)/NULLIF(low,0) > 0.10',
        (symbol, lookback_date, trade_date)
    ).fetchone()
    features['repeat_mover'] = int(repeat_row['cnt']) > 0 if repeat_row else False

    # 19. Concurrent qualified count (trades on same date)
    features['concurrent_trades'] = len([t for t in same_day_trades
                                          if abs(int(t['entry_time_et'][:2]) * 60 + int(t['entry_time_et'][3:5]) -
                                                 (int(entry_time[:2]) * 60 + int(entry_time[3:5]))) <= 30])

    # Derived features from CSV
    features['stop_dist_pct'] = round((entry_price - stop_loss) / entry_price * 100, 3)
    features['price_tier'] = (
        '$0-5' if entry_price < 5 else
        '$5-10' if entry_price < 10 else
        '$10-15' if entry_price < 15 else
        '$15-20' if entry_price < 20 else '$20+'
    )

    # 18. Gap direction + intraday behavior
    if features.get('gap_pct') is not None:
        gap = features['gap_pct']
        if gap >= 15 and entry_price > (prev_close * 1.15 if prev_close else 0):
            features['gap_behavior'] = 'gap_and_go'
        elif gap >= 15:
            features['gap_behavior'] = 'gap_fading'
        elif gap <= -5:
            features['gap_behavior'] = 'v_reversal'
        elif gap >= 5:
            features['gap_behavior'] = 'moderate_gap_up'
        else:
            features['gap_behavior'] = 'small_gap'
    else:
        features['gap_behavior'] = 'unknown'

    return features


def phase0_enrich(trades):
    """Phase 0: Enrich all trades with features from DB."""
    logger.info("Phase 0: Enriching trades with DB features...")
    conn = get_db()

    # Pre-cache SPY bars by date
    spy_cache = {}
    all_dates = set(t['date'] for t in trades)
    logger.info(f"Loading SPY bars for {len(all_dates)} dates...")
    for d in sorted(all_dates):
        spy_bars = get_1min_bars(conn, 'SPY', d)
        if spy_bars:
            spy_cache[d] = spy_bars

    logger.info(f"SPY bars loaded for {len(spy_cache)} dates")

    # Index trades by date for concurrent count
    trades_by_date = defaultdict(list)
    for t in trades:
        trades_by_date[t['date']].append(t)

    enriched = []
    for i, trade in enumerate(trades):
        if (i + 1) % 50 == 0:
            logger.info(f"  Enriching trade {i+1}/{len(trades)}...")
        features = enrich_trade(trade, conn, spy_cache, trades_by_date)
        enriched.append({**trade, **features})

    conn.close()
    logger.info(f"Enrichment complete: {len(enriched)} trades with features")
    return enriched


def save_enriched(enriched, path):
    """Save enriched trades to CSV."""
    if not enriched:
        return
    fieldnames = list(enriched[0].keys())
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in enriched:
            writer.writerow(row)
    logger.info(f"Saved enriched trades to {path}")


def phase1_analyze(enriched, period_label="ALL", min_date=None, max_date=None):
    """Phase 1: Statistical analysis — find what separates W from L."""
    # Filter by period
    if min_date:
        enriched = [t for t in enriched if t['date'] >= min_date]
    if max_date:
        enriched = [t for t in enriched if t['date'] <= max_date]

    winners = [t for t in enriched if float(t['pnl']) > 0]
    losers = [t for t in enriched if float(t['pnl']) <= 0]

    n = len(enriched)
    nw = len(winners)
    nl = len(losers)

    print(f"\n{'='*80}")
    print(f"  PHASE 1: WINNER vs LOSER ANALYSIS — {period_label}")
    print(f"  {n} trades: {nw} winners ({nw/n*100:.1f}%), {nl} losers ({nl/n*100:.1f}%)")
    print(f"{'='*80}")

    # Compare each numeric feature
    numeric_features = [
        ('gap_pct', 'Gap % from prev close'),
        ('breakout_bar_volume', 'Breakout bar volume'),
        ('relative_volume_at_entry', 'Relative volume at entry'),
        ('entry_vs_vwap_pct', 'Entry vs VWAP %'),
        ('macd_histogram_pct', 'MACD histogram %'),
        ('pole_gain_pct', 'Pole gain %'),
        ('pole_bars', 'Pole length (bars)'),
        ('retracement_pct', 'Retracement depth %'),
        ('flag_tightness', 'Flag tightness %'),
        ('green_bar_ratio', 'Green bar ratio in pole %'),
        ('vol_ratio_pole_vs_pullback', 'Vol ratio pole/pullback'),
        ('pullback_vol_slope', 'Pullback vol slope'),
        ('spy_return_at_entry', 'SPY return at entry %'),
        ('spy_5day_return', 'SPY 5-day return %'),
        ('stop_dist_pct', 'Stop distance %'),
        ('relative_move_magnitude', 'Relative move magnitude'),
        ('spread_proxy', 'Spread proxy %'),
        ('concurrent_trades', 'Concurrent trades'),
        ('cum_dollar_vol_at_entry', 'Cum $ vol at entry'),
        ('bars_since_open', 'Bars since open'),
    ]

    print(f"\n  {'Feature':<30} {'W mean':>10} {'L mean':>10} {'Δ%':>8} {'Signal':>8}")
    print(f"  {'-'*68}")

    signals = []
    for feat_key, feat_label in numeric_features:
        w_vals = [float(t[feat_key]) for t in winners
                  if t.get(feat_key) is not None and t[feat_key] != '' and t[feat_key] != 'None']
        l_vals = [float(t[feat_key]) for t in losers
                  if t.get(feat_key) is not None and t[feat_key] != '' and t[feat_key] != 'None']

        if len(w_vals) < 10 or len(l_vals) < 10:
            continue

        w_mean = statistics.mean(w_vals)
        l_mean = statistics.mean(l_vals)
        diff_pct = (w_mean - l_mean) / abs(l_mean) * 100 if l_mean != 0 else 0

        # Cohen's d for effect size
        pooled_std = ((statistics.stdev(w_vals) ** 2 + statistics.stdev(l_vals) ** 2) / 2) ** 0.5
        cohens_d = (w_mean - l_mean) / pooled_std if pooled_std > 0 else 0

        signal = "***" if abs(cohens_d) > 0.5 else "**" if abs(cohens_d) > 0.3 else "*" if abs(cohens_d) > 0.2 else ""
        print(f"  {feat_label:<30} {w_mean:>10.2f} {l_mean:>10.2f} {diff_pct:>+7.1f}% {signal:>8}")
        signals.append((feat_key, feat_label, cohens_d, diff_pct, w_mean, l_mean))

    # Sort by absolute Cohen's d
    signals.sort(key=lambda x: abs(x[2]), reverse=True)
    print(f"\n  TOP FEATURES BY SEPARATION POWER (Cohen's d):")
    print(f"  {'Feature':<30} {'Cohen d':>10} {'Direction':>15}")
    print(f"  {'-'*57}")
    for feat_key, feat_label, d, diff, wm, lm in signals[:10]:
        direction = "W higher" if d > 0 else "L higher"
        print(f"  {feat_label:<30} {d:>10.3f} {direction:>15}")

    # Categorical features
    print(f"\n  CATEGORICAL FEATURES:")

    # Gap behavior
    gap_buckets = defaultdict(lambda: {'w': 0, 'l': 0, 'pnl': 0})
    for t in enriched:
        gb = t.get('gap_behavior', 'unknown')
        if float(t['pnl']) > 0:
            gap_buckets[gb]['w'] += 1
        else:
            gap_buckets[gb]['l'] += 1
        gap_buckets[gb]['pnl'] += float(t['pnl'])

    print(f"\n  Gap Behavior:")
    print(f"  {'Type':<20} {'N':>5} {'WR%':>6} {'P&L':>12}")
    print(f"  {'-'*45}")
    for gb in sorted(gap_buckets):
        d = gap_buckets[gb]
        total = d['w'] + d['l']
        wr = d['w'] / total * 100 if total > 0 else 0
        print(f"  {gb:<20} {total:>5} {wr:>5.1f}% {d['pnl']:>+12,.0f}")

    # Repeat mover
    repeat_buckets = defaultdict(lambda: {'w': 0, 'l': 0, 'pnl': 0})
    for t in enriched:
        rm = 'repeat' if t.get('repeat_mover') in [True, 'True', 1, '1'] else 'fresh'
        if float(t['pnl']) > 0:
            repeat_buckets[rm]['w'] += 1
        else:
            repeat_buckets[rm]['l'] += 1
        repeat_buckets[rm]['pnl'] += float(t['pnl'])

    print(f"\n  Repeat Mover:")
    print(f"  {'Type':<20} {'N':>5} {'WR%':>6} {'P&L':>12}")
    print(f"  {'-'*45}")
    for rm in sorted(repeat_buckets):
        d = repeat_buckets[rm]
        total = d['w'] + d['l']
        wr = d['w'] / total * 100 if total > 0 else 0
        print(f"  {rm:<20} {total:>5} {wr:>5.1f}% {d['pnl']:>+12,.0f}")

    # Setup number
    setup_buckets = defaultdict(lambda: {'w': 0, 'l': 0, 'pnl': 0})
    for t in enriched:
        sn = int(t.get('setup_number', 1))
        label = f"setup_{sn}" if sn <= 3 else "setup_4+"
        if float(t['pnl']) > 0:
            setup_buckets[label]['w'] += 1
        else:
            setup_buckets[label]['l'] += 1
        setup_buckets[label]['pnl'] += float(t['pnl'])

    print(f"\n  Setup Number:")
    print(f"  {'Type':<20} {'N':>5} {'WR%':>6} {'P&L':>12}")
    print(f"  {'-'*45}")
    for sn in sorted(setup_buckets):
        d = setup_buckets[sn]
        total = d['w'] + d['l']
        wr = d['w'] / total * 100 if total > 0 else 0
        print(f"  {sn:<20} {total:>5} {wr:>5.1f}% {d['pnl']:>+12,.0f}")

    return signals


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load trades
    trades = load_cache_trades(CACHE_PATH)

    # Phase 0: Enrich
    if os.path.exists(ENRICHED_CSV):
        logger.info(f"Loading existing enriched data from {ENRICHED_CSV}")
        enriched = load_cache_trades(ENRICHED_CSV)
    else:
        enriched = phase0_enrich(trades)
        save_enriched(enriched, ENRICHED_CSV)

    # Phase 1: Analyze Q1 2026
    phase1_analyze(enriched, "Q1 2026", min_date="2026-01-01", max_date="2026-03-31")

    # Phase 1: Analyze full 2025 (for comparison)
    phase1_analyze(enriched, "2025 (out-of-sample)", min_date="2025-01-01", max_date="2025-12-31")

    # Phase 1: Analyze all
    signals = phase1_analyze(enriched, "ALL 15 MONTHS")

    logger.info("Phase 1 complete. Review output, then run Phase 2 (hypothesis testing).")


if __name__ == "__main__":
    main()
