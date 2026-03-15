"""
Diagnostic script: Investigate sparse morning bars and MACD blocking.

Two questions:
1. Why do most "movers" have sparse/gappy 1-min bars in the morning?
2. How does MACD's 35-bar minimum block morning setups, and what alternatives exist?
"""

import logging
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

from persistence.database import Database
from trading.pattern_detector import BullFlagDetector

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

load_dotenv()


def get_movers(db, start_date, end_date):
    """Get all big movers from cached daily bars."""
    daily = db.get_daily_bars_cached([], start_date, end_date)
    # get_daily_bars_cached needs symbols — get all cached symbols instead
    symbols = db.get_cached_daily_bar_symbols(start_date, end_date)
    daily = db.get_daily_bars_cached(list(symbols), start_date, end_date)

    movers = []
    for sym, bars in daily.items():
        for bar in bars:
            h, l = bar['high'], bar['low']
            if l > 0 and (h - l) / l >= 0.10:
                movers.append((sym, bar['date']))
    return movers


def analyze_bar_density(db, movers, sample_size=500):
    """Analyze 1-min bar density in the first 30 and 60 minutes of trading."""
    print("\n" + "=" * 70)
    print("  INVESTIGATION 1: Morning Bar Density")
    print("=" * 70)

    stats = {
        'total': 0,
        'has_bars': 0,
        'first_30min_counts': [],
        'first_60min_counts': [],
        'total_bar_counts': [],
        'first_bar_times_utc': [],
        'gap_examples': [],  # Store examples of gappy bars
        'price_at_open': [],
    }

    sampled = movers[:sample_size]
    for sym, date_str in sampled:
        cached = db.get_intraday_bars_cached(sym, date_str)
        if not cached:
            continue

        df = pd.DataFrame(cached)
        if df.empty:
            continue

        stats['total'] += 1
        stats['has_bars'] += 1
        stats['total_bar_counts'].append(len(df))

        # Parse timestamps
        if isinstance(df['timestamp'].iloc[0], str):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Market open is 13:30 UTC (9:30 ET)
        market_open_utc = 13.5  # 13:30
        first_30min_end = 14.0  # 14:00 UTC = 10:00 ET
        first_60min_end = 14.5  # 14:30 UTC = 10:30 ET

        # Calculate bar time in fractional UTC hours
        df['hour_utc'] = df['timestamp'].apply(
            lambda t: t.hour + t.minute / 60.0 if hasattr(t, 'hour') else 0
        )

        first_30 = df[(df['hour_utc'] >= market_open_utc) & (df['hour_utc'] < first_30min_end)]
        first_60 = df[(df['hour_utc'] >= market_open_utc) & (df['hour_utc'] < first_60min_end)]

        stats['first_30min_counts'].append(len(first_30))
        stats['first_60min_counts'].append(len(first_60))

        if len(df) > 0:
            stats['first_bar_times_utc'].append(df['hour_utc'].iloc[0])
            stats['price_at_open'].append(df['open'].iloc[0])

        # Collect examples of sparse morning bars (>$2 stocks with few bars)
        if len(first_30) <= 5 and len(df) > 0 and df['open'].iloc[0] >= 2.0:
            if len(stats['gap_examples']) < 10:
                stats['gap_examples'].append({
                    'symbol': sym,
                    'date': date_str,
                    'open_price': df['open'].iloc[0],
                    'total_bars': len(df),
                    'first_30_bars': len(first_30),
                    'first_60_bars': len(first_60),
                    'first_bar_time': df['timestamp'].iloc[0],
                    'bar_times_first_60': list(first_60['timestamp']) if len(first_60) > 0 else [],
                })

    print(f"\n  Analyzed {stats['total']} mover/dates with cached bars (of {len(sampled)} sampled)")

    # First 30 min distribution
    c30 = Counter()
    for n in stats['first_30min_counts']:
        if n == 0:
            c30['0 bars'] += 1
        elif n <= 5:
            c30['1-5 bars'] += 1
        elif n <= 15:
            c30['6-15 bars'] += 1
        elif n <= 25:
            c30['16-25 bars'] += 1
        else:
            c30['26-30 bars'] += 1

    print(f"\n  First 30 min (9:30-10:00 ET) bar count distribution:")
    for bucket in ['0 bars', '1-5 bars', '6-15 bars', '16-25 bars', '26-30 bars']:
        cnt = c30[bucket]
        pct = cnt / stats['total'] * 100 if stats['total'] > 0 else 0
        bar = '#' * int(pct / 2)
        print(f"    {bucket:>12s}: {cnt:4d} ({pct:5.1f}%) {bar}")

    # First 60 min distribution
    c60 = Counter()
    for n in stats['first_60min_counts']:
        if n == 0:
            c60['0 bars'] += 1
        elif n <= 10:
            c60['1-10 bars'] += 1
        elif n <= 30:
            c60['11-30 bars'] += 1
        elif n <= 50:
            c60['31-50 bars'] += 1
        else:
            c60['51-60 bars'] += 1

    print(f"\n  First 60 min (9:30-10:30 ET) bar count distribution:")
    for bucket in ['0 bars', '1-10 bars', '11-30 bars', '31-50 bars', '51-60 bars']:
        cnt = c60[bucket]
        pct = cnt / stats['total'] * 100 if stats['total'] > 0 else 0
        bar = '#' * int(pct / 2)
        print(f"    {bucket:>12s}: {cnt:4d} ({pct:5.1f}%) {bar}")

    # First bar time distribution — when does trading actually START?
    print(f"\n  First bar time distribution (when does first 1-min bar appear?):")
    time_buckets = Counter()
    for t in stats['first_bar_times_utc']:
        if t < 13.5:
            time_buckets['Before 9:30 ET'] += 1
        elif t < 14.0:
            time_buckets['9:30-10:00 ET'] += 1
        elif t < 14.5:
            time_buckets['10:00-10:30 ET'] += 1
        elif t < 15.5:
            time_buckets['10:30-11:30 ET'] += 1
        elif t < 18.0:
            time_buckets['11:30-14:00 ET'] += 1
        else:
            time_buckets['14:00+ ET'] += 1

    for bucket in ['Before 9:30 ET', '9:30-10:00 ET', '10:00-10:30 ET',
                    '10:30-11:30 ET', '11:30-14:00 ET', '14:00+ ET']:
        cnt = time_buckets[bucket]
        pct = cnt / stats['total'] * 100 if stats['total'] > 0 else 0
        bar = '#' * int(pct / 2)
        print(f"    {bucket:>18s}: {cnt:4d} ({pct:5.1f}%) {bar}")

    # Price distribution for sparse-morning stocks
    print(f"\n  Price distribution of ALL movers:")
    price_buckets = Counter()
    for p in stats['price_at_open']:
        if p < 1.0:
            price_buckets['<$1'] += 1
        elif p < 2.0:
            price_buckets['$1-$2'] += 1
        elif p < 5.0:
            price_buckets['$2-$5'] += 1
        elif p < 10.0:
            price_buckets['$5-$10'] += 1
        elif p < 20.0:
            price_buckets['$10-$20'] += 1
        else:
            price_buckets['$20+'] += 1

    for bucket in ['<$1', '$1-$2', '$2-$5', '$5-$10', '$10-$20', '$20+']:
        cnt = price_buckets[bucket]
        pct = cnt / stats['total'] * 100 if stats['total'] > 0 else 0
        bar = '#' * int(pct / 2)
        print(f"    {bucket:>8s}: {cnt:4d} ({pct:5.1f}%) {bar}")

    # Sparse bar examples
    if stats['gap_examples']:
        print(f"\n  Examples of sparse morning bars (>$2 stocks, <=5 bars in first 30 min):")
        for ex in stats['gap_examples'][:5]:
            print(f"\n    {ex['symbol']} on {ex['date']} — ${ex['open_price']:.2f}")
            print(f"      Total bars: {ex['total_bars']}, First 30min: {ex['first_30_bars']}, First 60min: {ex['first_60_bars']}")
            print(f"      First bar: {ex['first_bar_time']}")
            if ex['bar_times_first_60']:
                print(f"      First 60min bar times:")
                for bt in ex['bar_times_first_60'][:10]:
                    print(f"        {bt}")

    return stats


def analyze_bar_gaps(db, movers, sample_size=100):
    """Investigate WHY bars are gappy — is it low volume (no trades) or data issue?"""
    print("\n" + "=" * 70)
    print("  INVESTIGATION 1b: WHY Are Bars Gappy?")
    print("=" * 70)
    print("  (Alpaca only returns bars for minutes WITH trades — no-volume minutes = no bar)")

    gap_analysis = {
        'morning_gap_minutes': [],  # Minutes with no bar in first 60 min
        'morning_volume_per_bar': [],  # Avg volume per bar in first 60 min
        'morning_max_volume': [],
        'price_vs_density': [],  # (price, bar_count_first_60)
    }

    sampled = movers[:sample_size]
    for sym, date_str in sampled:
        cached = db.get_intraday_bars_cached(sym, date_str)
        if not cached:
            continue
        df = pd.DataFrame(cached)
        if df.empty:
            continue

        if isinstance(df['timestamp'].iloc[0], str):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        df['hour_utc'] = df['timestamp'].apply(
            lambda t: t.hour + t.minute / 60.0 if hasattr(t, 'hour') else 0
        )

        first_60 = df[(df['hour_utc'] >= 13.5) & (df['hour_utc'] < 14.5)]
        open_price = df['open'].iloc[0]

        bars_in_60 = len(first_60)
        gap_minutes = 60 - bars_in_60  # Expected 60 bars, missing = gap minutes

        gap_analysis['morning_gap_minutes'].append(gap_minutes)
        gap_analysis['price_vs_density'].append((open_price, bars_in_60))

        if bars_in_60 > 0:
            gap_analysis['morning_volume_per_bar'].append(first_60['volume'].mean())
            gap_analysis['morning_max_volume'].append(first_60['volume'].max())

    # Price vs bar density
    print(f"\n  Price bracket → Avg bars in first 60 min (expected: 60):")
    price_brackets = {
        '<$1': [], '$1-$2': [], '$2-$5': [], '$5-$10': [], '$10-$20': [], '$20+': []
    }
    for price, count in gap_analysis['price_vs_density']:
        if price < 1.0:
            price_brackets['<$1'].append(count)
        elif price < 2.0:
            price_brackets['$1-$2'].append(count)
        elif price < 5.0:
            price_brackets['$2-$5'].append(count)
        elif price < 10.0:
            price_brackets['$5-$10'].append(count)
        elif price < 20.0:
            price_brackets['$10-$20'].append(count)
        else:
            price_brackets['$20+'].append(count)

    for bracket, counts in price_brackets.items():
        if counts:
            avg = sum(counts) / len(counts)
            median = sorted(counts)[len(counts) // 2]
            dense = sum(1 for c in counts if c >= 30)
            pct_dense = dense / len(counts) * 100
            print(f"    {bracket:>8s}: avg {avg:5.1f} bars, median {median:3d}, "
                  f"{dense}/{len(counts)} ({pct_dense:.0f}%) have 30+ bars")

    # Volume analysis
    if gap_analysis['morning_volume_per_bar']:
        avg_vol = sum(gap_analysis['morning_volume_per_bar']) / len(gap_analysis['morning_volume_per_bar'])
        print(f"\n  Avg volume per 1-min bar in first 60 min: {avg_vol:,.0f}")
        low_vol = sum(1 for v in gap_analysis['morning_volume_per_bar'] if v < 1000)
        print(f"  Bars with avg volume < 1000: {low_vol}/{len(gap_analysis['morning_volume_per_bar'])} "
              f"({low_vol/len(gap_analysis['morning_volume_per_bar'])*100:.0f}%)")


def analyze_macd_impact(db, movers, sample_size=200):
    """Test pattern detection with and without MACD on morning bars."""
    print("\n" + "=" * 70)
    print("  INVESTIGATION 2: MACD Impact on Morning Setups")
    print("=" * 70)

    # Detector WITHOUT MACD
    detector_no_macd = BullFlagDetector(require_macd_positive=False)
    # Detector WITH standard MACD (needs 35 bars)
    detector_macd = BullFlagDetector(require_macd_positive=True)
    # Detector WITH fast MACD (5-13-4, needs 17 bars)
    detector_fast_macd = BullFlagDetector(
        require_macd_positive=True,
        macd_fast=5,
        macd_slow=13,
        macd_signal=4,
    )

    results = {
        'no_macd_setups': 0,
        'macd_setups': 0,
        'fast_macd_setups': 0,
        'no_macd_morning': 0,  # Setups detected before 10:30 ET (14:30 UTC)
        'macd_morning': 0,
        'fast_macd_morning': 0,
        'total_checked': 0,
        'stocks_with_30plus_bars': 0,
        'morning_setup_times_no_macd': [],
        'morning_setup_times_fast_macd': [],
        'morning_setup_details': [],
    }

    sampled = movers[:sample_size]
    for sym, date_str in sampled:
        cached = db.get_intraday_bars_cached(sym, date_str)
        if not cached:
            continue
        df = pd.DataFrame(cached)
        if df.empty or len(df) < 7:
            continue

        if isinstance(df['timestamp'].iloc[0], str):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        df['hour_utc'] = df['timestamp'].apply(
            lambda t: t.hour + t.minute / 60.0 if hasattr(t, 'hour') else 0
        )

        # Only check stocks with enough morning bars to form a pattern
        morning = df[df['hour_utc'] < 14.5]  # Before 10:30 ET
        if len(morning) < 7:
            continue

        results['total_checked'] += 1
        if len(morning) >= 30:
            results['stocks_with_30plus_bars'] += 1

        # Slide through morning bars, check for setups at each point
        found_no_macd = False
        found_macd = False
        found_fast_macd = False

        for i in range(7, len(morning)):
            # No MACD
            if not found_no_macd:
                setup = detector_no_macd.detect_setup(sym, morning, end_idx=i)
                if setup:
                    results['no_macd_setups'] += 1
                    results['no_macd_morning'] += 1
                    found_no_macd = True
                    bar_time = morning.iloc[i-1]['hour_utc']
                    results['morning_setup_times_no_macd'].append(bar_time)
                    if len(results['morning_setup_details']) < 10:
                        results['morning_setup_details'].append({
                            'symbol': sym,
                            'date': date_str,
                            'bar_idx': i,
                            'time': morning.iloc[i-1]['timestamp'],
                            'pole_gain': setup.pole_gain_pct,
                            'retracement': setup.retracement_pct,
                            'breakout_level': setup.breakout_level,
                            'bars_available': len(morning.iloc[:i]),
                        })

            # Standard MACD
            if not found_macd:
                setup = detector_macd.detect_setup(sym, morning, end_idx=i)
                if setup:
                    results['macd_setups'] += 1
                    results['macd_morning'] += 1
                    found_macd = True

            # Fast MACD
            if not found_fast_macd:
                setup = detector_fast_macd.detect_setup(sym, morning, end_idx=i)
                if setup:
                    results['fast_macd_setups'] += 1
                    results['fast_macd_morning'] += 1
                    found_fast_macd = True
                    bar_time = morning.iloc[i-1]['hour_utc']
                    results['morning_setup_times_fast_macd'].append(bar_time)

            if found_no_macd and found_macd and found_fast_macd:
                break

    print(f"\n  Checked {results['total_checked']} movers with 7+ morning bars (of {len(sampled)} sampled)")
    print(f"  {results['stocks_with_30plus_bars']} had 30+ morning bars (enough for std MACD)")

    print(f"\n  Morning setups detected (before 10:30 ET):")
    print(f"    No MACD filter:     {results['no_macd_morning']:4d} setups")
    print(f"    Standard MACD:      {results['macd_morning']:4d} setups (needs 35 bars)")
    print(f"    Fast MACD (5-13-4): {results['fast_macd_morning']:4d} setups (needs 17 bars)")

    if results['no_macd_morning'] > 0:
        blocked = results['no_macd_morning'] - results['macd_morning']
        pct = blocked / results['no_macd_morning'] * 100
        print(f"\n  Standard MACD BLOCKS {blocked}/{results['no_macd_morning']} ({pct:.0f}%) of morning setups")

        recovered = results['fast_macd_morning'] - results['macd_morning']
        if results['no_macd_morning'] > 0:
            pct_recovered = recovered / results['no_macd_morning'] * 100
            print(f"  Fast MACD RECOVERS {recovered} additional setups ({pct_recovered:.0f}% of total)")

    # Timing of no-MACD morning setups
    if results['morning_setup_times_no_macd']:
        print(f"\n  Morning setup timing (no MACD) — when do they appear?")
        time_buckets = Counter()
        for t in results['morning_setup_times_no_macd']:
            et_hour = t - 4  # UTC to ET approximate
            if et_hour < 9.67:  # 9:40
                time_buckets['9:30-9:40 ET'] += 1
            elif et_hour < 10.0:
                time_buckets['9:40-10:00 ET'] += 1
            elif et_hour < 10.5:
                time_buckets['10:00-10:30 ET'] += 1
            else:
                time_buckets['10:30+ ET'] += 1

        for bucket in ['9:30-9:40 ET', '9:40-10:00 ET', '10:00-10:30 ET', '10:30+ ET']:
            cnt = time_buckets.get(bucket, 0)
            pct = cnt / len(results['morning_setup_times_no_macd']) * 100
            print(f"    {bucket:>15s}: {cnt:3d} ({pct:.0f}%)")

    # Example setups
    if results['morning_setup_details']:
        print(f"\n  Example morning setups (no MACD):")
        for ex in results['morning_setup_details'][:5]:
            print(f"    {ex['symbol']} {ex['date']} @ {ex['time']} "
                  f"(bar {ex['bar_idx']}, {ex['bars_available']} bars avail): "
                  f"pole +{ex['pole_gain']:.1f}%, retrace {ex['retracement']:.0f}%, "
                  f"breakout ${ex['breakout_level']:.2f}")


def analyze_volume_filter_alternative(db, movers, sample_size=200):
    """Test if breakout volume ratio could replace MACD as quality filter."""
    print("\n" + "=" * 70)
    print("  INVESTIGATION 3: Volume-Based Alternatives to MACD")
    print("=" * 70)

    # Detector with different breakout volume thresholds
    configs = [
        ('Vol 1.5x (current)', BullFlagDetector(min_breakout_volume_ratio=1.5)),
        ('Vol 2.0x', BullFlagDetector(min_breakout_volume_ratio=2.0)),
        ('Vol 3.0x', BullFlagDetector(min_breakout_volume_ratio=3.0)),
        ('Vol 1.0x (relaxed)', BullFlagDetector(min_breakout_volume_ratio=1.0)),
    ]

    results = defaultdict(lambda: {'total': 0, 'morning': 0})

    sampled = movers[:sample_size]
    for sym, date_str in sampled:
        cached = db.get_intraday_bars_cached(sym, date_str)
        if not cached:
            continue
        df = pd.DataFrame(cached)
        if df.empty or len(df) < 7:
            continue

        if isinstance(df['timestamp'].iloc[0], str):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        df['hour_utc'] = df['timestamp'].apply(
            lambda t: t.hour + t.minute / 60.0 if hasattr(t, 'hour') else 0
        )

        for name, detector in configs:
            found = False
            # Scan through ALL bars for this stock, track morning vs not
            for i in range(7, len(df)):
                if found:
                    break
                # detect() requires breakout bar — use it for volume check
                pattern = detector.detect(sym, df, end_idx=i)
                if pattern:
                    results[name]['total'] += 1
                    bar_time = df.iloc[i-1]['hour_utc']
                    if bar_time < 14.5:  # Before 10:30 ET
                        results[name]['morning'] += 1
                    found = True

    print(f"\n  Breakout volume ratio impact on pattern detection:")
    print(f"  {'Config':>22s}  {'Total':>7s}  {'Morning':>7s}  {'Morning %':>9s}")
    print(f"  {'-'*22}  {'-'*7}  {'-'*7}  {'-'*9}")
    for name, _ in configs:
        r = results[name]
        morning_pct = r['morning'] / r['total'] * 100 if r['total'] > 0 else 0
        print(f"  {name:>22s}  {r['total']:7d}  {r['morning']:7d}  {morning_pct:8.1f}%")


def main():
    """Run all diagnostic investigations."""
    db = Database()

    # Get movers from the batch backtest date range
    movers = get_movers(db, '2026-02-01', '2026-03-13')
    print(f"\nTotal movers found: {len(movers)}")

    # Investigation 1: Bar density
    analyze_bar_density(db, movers, sample_size=len(movers))

    # Investigation 1b: Why are bars gappy?
    analyze_bar_gaps(db, movers, sample_size=500)

    # Investigation 2: MACD impact
    analyze_macd_impact(db, movers, sample_size=500)

    # Investigation 3: Volume alternatives
    # (This is slower — uses detect() with sliding window)
    analyze_volume_filter_alternative(db, movers, sample_size=200)

    print("\n" + "=" * 70)
    print("  DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
