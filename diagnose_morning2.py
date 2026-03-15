"""
Diagnostic part 2: Focus on REAL morning momentum stocks.

The first diagnostic showed that 53% of our "movers" have ZERO bars in
the first 30 min. These aren't Ross Cameron stocks. Let's find the ones
that ARE and see how the pattern detector performs on them.
"""

import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

from persistence.database import Database
from trading.pattern_detector import BullFlagDetector
from trading.trade_planner import TradePlanner
from backtest import BacktestRunner, TradeSimulator

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

load_dotenv()


def get_movers(db):
    """Get all big movers."""
    symbols = db.get_cached_daily_bar_symbols('2026-02-01', '2026-03-13')
    daily = db.get_daily_bars_cached(list(symbols), '2026-02-01', '2026-03-13')
    movers = []
    for sym, bars in daily.items():
        for bar in bars:
            h, l = bar['high'], bar['low']
            if l > 0 and (h - l) / l >= 0.10:
                movers.append((sym, bar['date']))
    return movers


def classify_movers(db, movers):
    """Classify each mover as morning-momentum vs not."""
    morning_movers = []
    other_movers = []

    for sym, date_str in movers:
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

        open_price = df['open'].iloc[0]
        # Skip sub-$2 stocks
        if open_price < 2.0:
            continue

        first_30 = df[(df['hour_utc'] >= 13.5) & (df['hour_utc'] < 14.0)]
        first_60 = df[(df['hour_utc'] >= 13.5) & (df['hour_utc'] < 14.5)]

        # Morning momentum criteria:
        # 1. At least 15 bars in first 30 min (active trading)
        # 2. Price > $2
        if len(first_30) >= 15:
            # Calculate morning move: (high of first 30 min - open) / open
            if len(first_30) > 0:
                morning_high = first_30['high'].max()
                morning_move = (morning_high - open_price) / open_price * 100

                # Check volume concentration: morning volume vs total volume
                total_vol = df['volume'].sum()
                morning_vol = first_30['volume'].sum()
                vol_concentration = morning_vol / total_vol * 100 if total_vol > 0 else 0

                morning_movers.append({
                    'symbol': sym,
                    'date': date_str,
                    'open_price': open_price,
                    'morning_bars_30': len(first_30),
                    'morning_bars_60': len(first_60),
                    'total_bars': len(df),
                    'morning_move_pct': morning_move,
                    'morning_vol_pct': vol_concentration,
                    'df': df,
                })
        else:
            other_movers.append((sym, date_str, len(first_30), open_price))

    return morning_movers, other_movers


def backtest_morning_movers(morning_movers):
    """Run pattern detection on morning movers with different configs."""
    print("\n" + "=" * 70)
    print("  MORNING MOMENTUM STOCKS — Pattern Detection Analysis")
    print("=" * 70)

    print(f"\n  Found {len(morning_movers)} morning momentum stocks (>=$2, 15+ bars in first 30 min)")

    # Price distribution
    print(f"\n  Price distribution:")
    for bracket, lo, hi in [('$2-$5', 2, 5), ('$5-$10', 5, 10), ('$10-$20', 10, 20), ('$20+', 20, 999)]:
        cnt = sum(1 for m in morning_movers if lo <= m['open_price'] < hi)
        print(f"    {bracket:>8s}: {cnt}")

    # Morning move distribution
    print(f"\n  Morning move (first 30 min high vs open):")
    for bracket, lo, hi in [('0-3%', 0, 3), ('3-5%', 3, 5), ('5-10%', 5, 10),
                             ('10-20%', 10, 20), ('20%+', 20, 999)]:
        cnt = sum(1 for m in morning_movers if lo <= m['morning_move_pct'] < hi)
        print(f"    {bracket:>8s}: {cnt}")

    # Now run DIFFERENT detector configs on these stocks
    configs = [
        ('No MACD, Vol 1.5x', BullFlagDetector(min_breakout_volume_ratio=1.5)),
        ('No MACD, Vol 1.0x', BullFlagDetector(min_breakout_volume_ratio=1.0)),
        ('No MACD, Vol 2.0x', BullFlagDetector(min_breakout_volume_ratio=2.0)),
        ('Std MACD, Vol 1.5x', BullFlagDetector(require_macd_positive=True, min_breakout_volume_ratio=1.5)),
        ('Fast MACD (5-13-4)', BullFlagDetector(require_macd_positive=True,
                                                  macd_fast=5, macd_slow=13, macd_signal=4,
                                                  min_breakout_volume_ratio=1.5)),
        ('No MACD, Vol 1.5x, max_pullback=3', BullFlagDetector(
            min_breakout_volume_ratio=1.5, max_pullback_candles=3)),
    ]

    for config_name, detector in configs:
        setup_count = 0
        morning_setup_count = 0  # Before 10:30 ET
        early_setup_count = 0   # Before 10:00 ET
        setup_times = []

        for m in morning_movers:
            df = m['df']
            sym = m['symbol']
            found = False

            # Only scan morning bars (before 10:30 ET = 14:30 UTC)
            morning = df[df['hour_utc'] < 14.5]

            for i in range(7, len(morning)):
                if found:
                    break
                # Use detect_setup (pre-breakout)
                setup = detector.detect_setup(sym, morning, end_idx=i)
                if setup:
                    setup_count += 1
                    bar_time = morning.iloc[i-1]['hour_utc']
                    morning_setup_count += 1
                    setup_times.append(bar_time)
                    if bar_time < 14.0:  # Before 10:00 ET
                        early_setup_count += 1
                    found = True

        print(f"\n  {config_name}:")
        print(f"    Setups found (morning): {morning_setup_count}/{len(morning_movers)} "
              f"({morning_setup_count/len(morning_movers)*100:.0f}%)")
        print(f"    Early setups (<10:00 ET): {early_setup_count}")

        if setup_times:
            # Average time of detection
            avg_time = sum(setup_times) / len(setup_times)
            et_avg = avg_time - 4
            hours = int(et_avg)
            minutes = int((et_avg - hours) * 60)
            print(f"    Avg setup time: {hours}:{minutes:02d} ET")


def backtest_morning_with_trades(db, morning_movers):
    """Actually simulate trades on morning movers — the key test."""
    print("\n" + "=" * 70)
    print("  MORNING MOMENTUM — Full Backtest (Trades + P&L)")
    print("=" * 70)

    # H10a config without MACD, realistic mode, restricted to morning only
    from trading.trade_planner import TradePlanner

    # H10a params (our current best)
    planner = TradePlanner(
        position_size_dollars=50000,
        sizing_mode='fixed_risk',
        risk_per_trade=2000,
        min_risk_per_share=0.05,
        max_risk_per_share=0.20,
        min_risk_pct=0.01,
        max_risk_pct=0.05,
        min_risk_reward=2.5,
        max_shares=10000,
    )

    # Test WITHOUT MACD
    detector_no_macd = BullFlagDetector(
        min_breakout_volume_ratio=1.5,
        require_macd_positive=False,
    )

    # Test WITH MACD
    detector_with_macd = BullFlagDetector(
        min_breakout_volume_ratio=1.5,
        require_macd_positive=True,
    )

    # Test with fast MACD
    detector_fast_macd = BullFlagDetector(
        min_breakout_volume_ratio=1.5,
        require_macd_positive=True,
        macd_fast=5,
        macd_slow=13,
        macd_signal=4,
    )

    configs = [
        ('No MACD', detector_no_macd),
        ('Std MACD', detector_with_macd),
        ('Fast MACD (5-13-4)', detector_fast_macd),
    ]

    for config_name, detector in configs:
        runner = BacktestRunner(
            detector=detector,
            planner=planner,
            realistic=True,
            min_price=2.0,
            skip_midday=False,  # We're ONLY running morning bars
            last_entry_time_et=(10, 30),  # Only enter before 10:30 ET
            force_close_time_et=(15, 45),
        )

        all_trades = []
        for m in morning_movers:
            result = runner.run(m['symbol'], m['df'], m['date'])
            # Only count trades that entered before 14:30 UTC (10:30 ET)
            for t in result.trades_simulated:
                entry_hour = t.entry_time.hour + t.entry_time.minute / 60.0 if hasattr(t.entry_time, 'hour') else 0
                if entry_hour < 14.5:
                    all_trades.append(t)

        wins = [t for t in all_trades if t.pnl > 0]
        losses = [t for t in all_trades if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in all_trades)
        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0
        wr = len(wins) / len(all_trades) * 100 if all_trades else 0
        rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        print(f"\n  {config_name}:")
        print(f"    Trades: {len(all_trades)}")
        print(f"    Win Rate: {wr:.1f}%")
        print(f"    Avg Win: ${avg_win:,.0f}")
        print(f"    Avg Loss: ${avg_loss:,.0f}")
        print(f"    R:R: {rr:.2f}")
        print(f"    Total P&L: ${total_pnl:,.0f}")

        if all_trades:
            print(f"\n    Trade details:")
            for t in sorted(all_trades, key=lambda x: x.pnl, reverse=True)[:5]:
                print(f"      {t.symbol} {t.entry_time.strftime('%Y-%m-%d %H:%M')} "
                      f"entry ${t.entry_price:.2f} → ${t.exit_price:.2f} "
                      f"({t.exit_reason}) P&L ${t.pnl:,.0f}")
            if len(all_trades) > 5:
                print(f"      ... ({len(all_trades)-5} more)")
                # Show worst too
                for t in sorted(all_trades, key=lambda x: x.pnl)[:3]:
                    print(f"      {t.symbol} {t.entry_time.strftime('%Y-%m-%d %H:%M')} "
                          f"entry ${t.entry_price:.2f} → ${t.exit_price:.2f} "
                          f"({t.exit_reason}) P&L ${t.pnl:,.0f}")


def analyze_all_movers_backtest(db, movers):
    """Run the FULL universe through no-MACD vs MACD to see total impact."""
    print("\n" + "=" * 70)
    print("  FULL UNIVERSE BACKTEST — No MACD vs Std MACD vs Fast MACD")
    print("=" * 70)

    planner = TradePlanner(
        position_size_dollars=50000,
        sizing_mode='fixed_risk',
        risk_per_trade=2000,
        min_risk_per_share=0.05,
        max_risk_per_share=0.20,
        min_risk_pct=0.01,
        max_risk_pct=0.05,
        min_risk_reward=2.5,
        max_shares=10000,
    )

    configs = [
        ('H10a (std MACD)', BullFlagDetector(require_macd_positive=True, min_breakout_volume_ratio=1.5)),
        ('No MACD', BullFlagDetector(require_macd_positive=False, min_breakout_volume_ratio=1.5)),
        ('Fast MACD', BullFlagDetector(require_macd_positive=True, macd_fast=5, macd_slow=13, macd_signal=4, min_breakout_volume_ratio=1.5)),
        ('No MACD + Vol 2.0x', BullFlagDetector(require_macd_positive=False, min_breakout_volume_ratio=2.0)),
    ]

    for config_name, detector in configs:
        runner = BacktestRunner(
            detector=detector,
            planner=planner,
            realistic=True,
            min_price=2.0,
            skip_midday=True,
            force_close_time_et=(15, 45),
        )

        all_trades = []
        morning_trades = []
        processed = 0

        for sym, date_str in movers:
            cached = db.get_intraday_bars_cached(sym, date_str)
            if not cached:
                continue
            df = pd.DataFrame(cached)
            if df.empty or len(df) < 7:
                continue

            if isinstance(df['timestamp'].iloc[0], str):
                df['timestamp'] = pd.to_datetime(df['timestamp'])

            result = runner.run(sym, df, date_str)
            for t in result.trades_simulated:
                all_trades.append(t)
                entry_hour = t.entry_time.hour + t.entry_time.minute / 60.0
                if entry_hour < 14.5:  # Before 10:30 ET
                    morning_trades.append(t)

            processed += 1
            if processed % 500 == 0:
                print(f"    [{config_name}] Processed {processed} movers, {len(all_trades)} trades so far...")

        wins = [t for t in all_trades if t.pnl > 0]
        losses = [t for t in all_trades if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in all_trades)
        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0
        wr = len(wins) / len(all_trades) * 100 if all_trades else 0
        rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        morning_wins = sum(1 for t in morning_trades if t.pnl > 0)
        morning_wr = morning_wins / len(morning_trades) * 100 if morning_trades else 0

        print(f"\n  {config_name}:")
        print(f"    Total trades: {len(all_trades)}, Morning trades (<10:30 ET): {len(morning_trades)}")
        print(f"    Win Rate: {wr:.1f}% (morning: {morning_wr:.1f}%)")
        print(f"    Avg Win: ${avg_win:,.0f}, Avg Loss: ${avg_loss:,.0f}")
        print(f"    R:R: {rr:.2f}")
        print(f"    Total P&L: ${total_pnl:,.0f}")


def main():
    db = Database()
    movers = get_movers(db)
    print(f"Total movers: {len(movers)}")

    morning_movers, other = classify_movers(db, movers)

    print(f"\nMorning momentum stocks (>=$2, 15+ bars in first 30 min): {len(morning_movers)}")
    print(f"Other movers: {len(other)}")

    # Pattern detection analysis
    backtest_morning_movers(morning_movers)

    # Full trade simulation on morning movers
    backtest_morning_with_trades(db, morning_movers)

    # Full universe comparison: MACD vs no-MACD
    analyze_all_movers_backtest(db, movers)


if __name__ == '__main__':
    main()
