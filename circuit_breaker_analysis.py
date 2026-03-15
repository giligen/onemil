"""
Circuit breaker analysis: simulate drawdown management rules on H12 trades.

Tests multiple rules to find which ones cut MaxDD without sacrificing too much P&L.
"""

import csv
import sys
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Tuple


def load_trades(path: str) -> List[Dict]:
    """Load trades from CSV, sorted by entry time."""
    trades = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['pnl_float'] = float(row['pnl'])
            row['entry_dt'] = datetime.fromisoformat(row['entry_time'])
            row['entry_date'] = row['entry_dt'].date()
            trades.append(row)
    trades.sort(key=lambda t: t['entry_dt'])
    return trades


def compute_stats(trades: List[Dict], label: str, filtered_mask: List[bool]) -> Dict:
    """Compute stats for trades where filtered_mask is True (trade taken)."""
    taken = [t for t, m in zip(trades, filtered_mask) if m]
    if not taken:
        return {'label': label, 'trades': 0}

    wins = [t for t in taken if t['pnl_float'] > 0]
    losses = [t for t in taken if t['pnl_float'] <= 0]
    total_pnl = sum(t['pnl_float'] for t in taken)
    avg_win = sum(t['pnl_float'] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t['pnl_float'] for t in losses) / len(losses) if losses else 0
    wr = len(wins) / len(taken) * 100

    # Max drawdown
    cum = 0
    peak = 0
    max_dd = 0
    for t in taken:
        cum += t['pnl_float']
        if cum > peak:
            peak = cum
        dd = cum - peak
        if dd < max_dd:
            max_dd = dd

    rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    return {
        'label': label,
        'trades': len(taken),
        'skipped': len(trades) - len(taken),
        'wr': wr,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'rr': rr,
        'total_pnl': total_pnl,
        'max_dd': max_dd,
        'pnl_per_trade': total_pnl / len(taken),
    }


def print_stats(stats: Dict):
    """Print formatted stats line."""
    if stats['trades'] == 0:
        print(f"  {stats['label']:>40s}:  NO TRADES")
        return
    print(
        f"  {stats['label']:>40s}: "
        f"{stats['trades']:3d} trades ({stats['skipped']:2d} skipped) "
        f"WR {stats['wr']:5.1f}% "
        f"R:R {stats['rr']:.2f} "
        f"P&L ${stats['total_pnl']:>+9,.0f} "
        f"MaxDD ${stats['max_dd']:>+8,.0f} "
        f"$/trade ${stats['pnl_per_trade']:>+7,.0f}"
    )


def rule_baseline(trades):
    """No filter — take all trades."""
    return [True] * len(trades)


def rule_daily_loss_cap(trades, max_daily_loss):
    """Stop trading for the day after cumulative daily loss exceeds cap."""
    mask = []
    daily_pnl = defaultdict(float)
    daily_stopped = set()

    for t in trades:
        date = t['entry_date']
        if date in daily_stopped:
            mask.append(False)
            continue

        mask.append(True)
        daily_pnl[date] += t['pnl_float']

        if daily_pnl[date] <= -max_daily_loss:
            daily_stopped.add(date)

    return mask


def rule_consecutive_loss_pause(trades, max_consecutive, pause_trades):
    """After N consecutive losses, skip the next M trades."""
    mask = []
    consecutive_losses = 0
    skip_remaining = 0

    for t in trades:
        if skip_remaining > 0:
            mask.append(False)
            skip_remaining -= 1
            continue

        mask.append(True)

        if t['pnl_float'] <= 0:
            consecutive_losses += 1
            if consecutive_losses >= max_consecutive:
                skip_remaining = pause_trades
                consecutive_losses = 0
        else:
            consecutive_losses = 0

    return mask


def rule_consecutive_loss_stop_day(trades, max_consecutive):
    """After N consecutive losses, stop for rest of the day."""
    mask = []
    consecutive_losses = 0
    stopped_date = None

    for t in trades:
        date = t['entry_date']

        # Reset on new day
        if stopped_date is not None and date != stopped_date:
            stopped_date = None
            consecutive_losses = 0

        if stopped_date == date:
            mask.append(False)
            continue

        mask.append(True)

        if t['pnl_float'] <= 0:
            consecutive_losses += 1
            if consecutive_losses >= max_consecutive:
                stopped_date = date
        else:
            consecutive_losses = 0

    return mask


def rule_trailing_dd_pause(trades, max_dd_threshold, pause_trades):
    """When drawdown from equity peak exceeds threshold, pause N trades."""
    mask = []
    cum = 0
    peak = 0
    skip_remaining = 0

    for t in trades:
        if skip_remaining > 0:
            mask.append(False)
            skip_remaining -= 1
            continue

        mask.append(True)
        cum += t['pnl_float']

        if cum > peak:
            peak = cum

        dd = cum - peak
        if dd <= -max_dd_threshold:
            skip_remaining = pause_trades

    return mask


def rule_rolling_wr(trades, window, min_wr_pct):
    """Pause when rolling WR over last N trades drops below threshold."""
    mask = []
    recent_results = []  # True=win, False=loss

    for t in trades:
        if len(recent_results) >= window:
            wr = sum(recent_results[-window:]) / window * 100
            if wr < min_wr_pct:
                mask.append(False)
                # Still track this trade's result for recovery detection
                recent_results.append(t['pnl_float'] > 0)
                continue

        mask.append(True)
        recent_results.append(t['pnl_float'] > 0)

    return mask


def rule_max_daily_trades(trades, max_trades_per_day):
    """Cap the number of trades per day."""
    mask = []
    daily_count = defaultdict(int)

    for t in trades:
        date = t['entry_date']
        if daily_count[date] >= max_trades_per_day:
            mask.append(False)
        else:
            mask.append(True)
            daily_count[date] += 1

    return mask


def rule_combined(trades, daily_loss_cap, max_consecutive, max_daily_trades):
    """Combine daily loss cap + consecutive loss stop + max daily trades."""
    mask = []
    daily_pnl = defaultdict(float)
    daily_stopped = set()
    daily_count = defaultdict(int)
    consecutive_losses = 0
    last_date = None

    for t in trades:
        date = t['entry_date']

        # Reset consecutive on new day
        if date != last_date:
            consecutive_losses = 0
            last_date = date

        # Check all stop conditions
        if date in daily_stopped:
            mask.append(False)
            continue
        if daily_count[date] >= max_daily_trades:
            mask.append(False)
            continue

        mask.append(True)
        daily_count[date] += 1
        daily_pnl[date] += t['pnl_float']

        if t['pnl_float'] <= 0:
            consecutive_losses += 1
        else:
            consecutive_losses = 0

        # Trigger stops
        if daily_pnl[date] <= -daily_loss_cap:
            daily_stopped.add(date)
        if consecutive_losses >= max_consecutive:
            daily_stopped.add(date)

    return mask


def main():
    trades = load_trades('results/H12_trades.csv')
    print(f"Loaded {len(trades)} H12 trades")

    # Count trades per day
    daily_counts = defaultdict(int)
    for t in trades:
        daily_counts[t['entry_date']] += 1
    max_per_day = max(daily_counts.values())
    avg_per_day = sum(daily_counts.values()) / len(daily_counts)
    print(f"Trading days: {len(daily_counts)}, avg {avg_per_day:.1f} trades/day, max {max_per_day}/day")

    # Daily P&L distribution
    daily_pnl = defaultdict(float)
    for t in trades:
        daily_pnl[t['entry_date']] += t['pnl_float']
    worst_days = sorted(daily_pnl.items(), key=lambda x: x[1])
    print(f"\nWorst 5 days:")
    for date, pnl in worst_days[:5]:
        n = daily_counts[date]
        print(f"  {date}: ${pnl:>+8,.0f} ({n} trades)")

    print(f"\nBest 5 days:")
    for date, pnl in worst_days[-5:]:
        n = daily_counts[date]
        print(f"  {date}: ${pnl:>+8,.0f} ({n} trades)")

    print("\n" + "=" * 120)
    print("  CIRCUIT BREAKER ANALYSIS")
    print("=" * 120)

    # Baseline
    print("\n--- BASELINE ---")
    print_stats(compute_stats(trades, "No filter (baseline)", rule_baseline(trades)))

    # Rule 1: Daily loss cap
    print("\n--- DAILY LOSS CAP ---")
    for cap in [2000, 3000, 4000, 5000]:
        mask = rule_daily_loss_cap(trades, cap)
        print_stats(compute_stats(trades, f"Daily loss cap ${cap:,}", mask))

    # Rule 2: Consecutive losses — stop for day
    print("\n--- CONSECUTIVE LOSS → STOP FOR DAY ---")
    for n in [2, 3, 4]:
        mask = rule_consecutive_loss_stop_day(trades, n)
        print_stats(compute_stats(trades, f"Stop day after {n} consecutive losses", mask))

    # Rule 3: Consecutive losses — skip N trades
    print("\n--- CONSECUTIVE LOSS → SKIP N TRADES ---")
    for n, skip in [(3, 1), (3, 2), (3, 3), (2, 1), (2, 2)]:
        mask = rule_consecutive_loss_pause(trades, n, skip)
        print_stats(compute_stats(trades, f"{n} losses → skip {skip} trade(s)", mask))

    # Rule 4: Trailing drawdown pause
    print("\n--- TRAILING DRAWDOWN PAUSE ---")
    for dd, pause in [(4000, 3), (5000, 3), (6000, 5), (3000, 2), (4000, 5)]:
        mask = rule_trailing_dd_pause(trades, dd, pause)
        print_stats(compute_stats(trades, f"DD > ${dd:,} → skip {pause} trades", mask))

    # Rule 5: Rolling WR
    print("\n--- ROLLING WIN RATE FILTER ---")
    for window, min_wr in [(10, 25), (10, 30), (8, 25), (15, 30)]:
        mask = rule_rolling_wr(trades, window, min_wr)
        print_stats(compute_stats(trades, f"Last {window} trades WR < {min_wr}% → skip", mask))

    # Rule 6: Max trades per day
    print("\n--- MAX TRADES PER DAY ---")
    for max_t in [1, 2, 3]:
        mask = rule_max_daily_trades(trades, max_t)
        print_stats(compute_stats(trades, f"Max {max_t} trade(s)/day", mask))

    # Rule 7: Combined rules
    print("\n--- COMBINED RULES ---")
    combos = [
        (3000, 3, 3, "DailyLoss $3K + 3 consec loss + 3/day max"),
        (4000, 3, 3, "DailyLoss $4K + 3 consec loss + 3/day max"),
        (3000, 2, 2, "DailyLoss $3K + 2 consec loss + 2/day max"),
        (4000, 2, 3, "DailyLoss $4K + 2 consec loss + 3/day max"),
        (2000, 2, 2, "DailyLoss $2K + 2 consec loss + 2/day max"),
        (5000, 3, 5, "DailyLoss $5K + 3 consec loss + 5/day max"),
    ]
    for daily_cap, consec, max_trades, label in combos:
        mask = rule_combined(trades, daily_cap, consec, max_trades)
        print_stats(compute_stats(trades, label, mask))

    # Find the best rule (highest P&L with MaxDD < 6000)
    print("\n" + "=" * 120)
    print("  BEST RULES (MaxDD < $6,000)")
    print("=" * 120)

    all_rules = []

    # Collect all rules
    for cap in [2000, 3000, 4000, 5000]:
        mask = rule_daily_loss_cap(trades, cap)
        all_rules.append(compute_stats(trades, f"Daily loss cap ${cap:,}", mask))

    for n in [2, 3, 4]:
        mask = rule_consecutive_loss_stop_day(trades, n)
        all_rules.append(compute_stats(trades, f"Stop day after {n} losses", mask))

    for n, skip in [(3, 1), (3, 2), (3, 3), (2, 1), (2, 2)]:
        mask = rule_consecutive_loss_pause(trades, n, skip)
        all_rules.append(compute_stats(trades, f"{n} losses → skip {skip}", mask))

    for dd, pause in [(4000, 3), (5000, 3), (6000, 5), (3000, 2), (4000, 5)]:
        mask = rule_trailing_dd_pause(trades, dd, pause)
        all_rules.append(compute_stats(trades, f"DD>${dd:,} skip {pause}", mask))

    for window, min_wr in [(10, 25), (10, 30), (8, 25), (15, 30)]:
        mask = rule_rolling_wr(trades, window, min_wr)
        all_rules.append(compute_stats(trades, f"WR<{min_wr}% last {window}", mask))

    for max_t in [1, 2, 3]:
        mask = rule_max_daily_trades(trades, max_t)
        all_rules.append(compute_stats(trades, f"Max {max_t}/day", mask))

    for daily_cap, consec, max_trades, label in combos:
        mask = rule_combined(trades, daily_cap, consec, max_trades)
        all_rules.append(compute_stats(trades, label, mask))

    # Filter for MaxDD < 6000 and sort by P&L
    good_rules = [r for r in all_rules if r.get('max_dd', 0) > -6000 and r['trades'] > 0]
    good_rules.sort(key=lambda r: r['total_pnl'], reverse=True)

    for r in good_rules[:10]:
        print_stats(r)


if __name__ == '__main__':
    main()
