"""
Exhaustion Exit Signal Analysis.

For each trade in a backtest CSV, re-walks the 1-min bars and checks
4 exhaustion signals. When a signal fires (while trade is profitable),
simulates a partial exit (50% at bar close with 0 slippage) and
tighter trailing stop on the remaining 50%.

Tests each signal independently to identify which ones help vs hurt.

Usage:
    python3 analyze_exhaustion_exits.py backtest_15month_honest_slip.csv
"""

import argparse
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pytz

from persistence.database import Database

logger = logging.getLogger(__name__)
ET = pytz.timezone('US/Eastern')


# ---------------------------------------------------------------------------
# Exhaustion signal detectors
# ---------------------------------------------------------------------------

def check_volume_divergence(bars: pd.DataFrame, idx: int, lookback: int = 3) -> bool:
    """
    Volume declining over lookback bars while price makes higher highs.

    Requires: each bar's volume < previous bar's volume for lookback bars,
    AND each bar's high >= previous bar's high (price still rising).
    """
    if idx < lookback:
        return False

    for j in range(1, lookback + 1):
        curr = idx - lookback + j
        prev = curr - 1
        if prev < 0:
            return False
        # Volume must be declining
        if bars.iloc[curr]['volume'] >= bars.iloc[prev]['volume']:
            return False
        # Price must be making higher highs (or equal)
        if bars.iloc[curr]['high'] < bars.iloc[prev]['high'] - 0.01:
            return False

    return True


def check_climax_candle(bars: pd.DataFrame, idx: int, lookback: int = 5,
                         body_mult: float = 2.0, vol_mult: float = 2.0) -> bool:
    """
    Climax candle: body AND volume both > mult × average of previous lookback bars.

    This is the blow-off top — everyone piling in at once.
    """
    if idx < lookback:
        return False

    curr_body = abs(bars.iloc[idx]['close'] - bars.iloc[idx]['open'])
    curr_vol = bars.iloc[idx]['volume']

    avg_body = sum(
        abs(bars.iloc[idx - j]['close'] - bars.iloc[idx - j]['open'])
        for j in range(1, lookback + 1)
    ) / lookback

    avg_vol = sum(
        bars.iloc[idx - j]['volume']
        for j in range(1, lookback + 1)
    ) / lookback

    if avg_body <= 0 or avg_vol <= 0:
        return False

    return (curr_body >= avg_body * body_mult and
            curr_vol >= avg_vol * vol_mult)


def check_shrinking_bodies(bars: pd.DataFrame, idx: int, lookback: int = 3,
                            shrink_ratio: float = 0.5) -> bool:
    """
    Current candle body < shrink_ratio × body from lookback bars ago,
    while price is still near highs (close >= close from lookback bars ago).
    """
    if idx < lookback:
        return False

    curr_body = abs(bars.iloc[idx]['close'] - bars.iloc[idx]['open'])
    prev_body = abs(bars.iloc[idx - lookback]['close'] - bars.iloc[idx - lookback]['open'])

    if prev_body <= 0:
        return False

    # Bodies shrinking
    if curr_body >= prev_body * shrink_ratio:
        return False

    # But price still near highs (not already falling)
    if bars.iloc[idx]['close'] < bars.iloc[idx - lookback]['close']:
        return False

    return True


def check_shooting_star(bars: pd.DataFrame, idx: int, wick_ratio: float = 2.0) -> bool:
    """
    Long upper wick (> wick_ratio × body) with close near the low.
    Shows rejection at highs.
    """
    bar = bars.iloc[idx]
    body = abs(bar['close'] - bar['open'])
    upper_wick = bar['high'] - max(bar['open'], bar['close'])
    lower_wick = min(bar['open'], bar['close']) - bar['low']

    if body <= 0.001:  # doji — skip
        return False

    # Upper wick must be significant
    if upper_wick < body * wick_ratio:
        return False

    # Close should be in lower half of bar (bearish rejection)
    bar_range = bar['high'] - bar['low']
    if bar_range <= 0:
        return False
    close_position = (bar['close'] - bar['low']) / bar_range
    if close_position > 0.4:  # close should be in bottom 40% of bar
        return False

    return True


# ---------------------------------------------------------------------------
# Signal names and functions
# ---------------------------------------------------------------------------

SIGNALS = {
    'vol_divergence': check_volume_divergence,
    'climax_candle': check_climax_candle,
    'shrinking_bodies': check_shrinking_bodies,
    'shooting_star': check_shooting_star,
}


# ---------------------------------------------------------------------------
# Trade re-simulation with exhaustion exit overlay
# ---------------------------------------------------------------------------

@dataclass
class ExhaustionResult:
    """Result of testing one exhaustion signal on one trade."""

    symbol: str
    trade_date: str
    baseline_pnl: float
    baseline_exit_price: float
    signal_fired: bool
    signal_bar_idx: Optional[int] = None
    signal_price: Optional[float] = None  # bar close when signal fired
    signal_r_gain: Optional[float] = None  # R-multiple when signal fired
    # Exhaustion exit: 50% at signal price (0 slippage), 50% at tighter trail
    exhaustion_pnl: Optional[float] = None
    partial_price: Optional[float] = None
    remainder_exit_price: Optional[float] = None
    remainder_exit_reason: Optional[str] = None


def simulate_trade_with_signal(
    bars: pd.DataFrame,
    entry_bar_idx: int,
    entry_price: float,
    stop_loss: float,
    shares: int,
    signal_fn,
    trail_r: float = 1.0,
    activate_at_r: float = 2.0,
    exit_slippage_pct: float = 0.003,
    tighter_trail_r: float = 0.5,  # tighter trail on remainder after partial
    min_r_for_signal: float = 1.0,  # only check signals when +NR profitable
    force_close_et: Tuple[int, int] = (15, 45),
) -> ExhaustionResult:
    """
    Re-simulate a trade checking for exhaustion signal.

    Args:
        bars: Full day's 1-min bars
        entry_bar_idx: Index of entry bar
        entry_price: Actual entry price
        stop_loss: Original stop loss price
        shares: Position size
        signal_fn: Exhaustion signal function to test
        trail_r: Original trail distance in R
        activate_at_r: Activate trail after +NR
        exit_slippage_pct: Slippage on stop exits
        tighter_trail_r: Trail distance for remainder after partial
        min_r_for_signal: Minimum R-gain before checking signals
        force_close_et: Force close time (hour, minute) ET

    Returns:
        ExhaustionResult with baseline and signal-modified P&L
    """
    risk = entry_price - stop_loss
    if risk <= 0:
        risk = 0.01

    # --- Baseline simulation (current trailing stop) ---
    highest = entry_price
    trailing_active = False
    current_stop = stop_loss
    baseline_exit_price = None
    baseline_exit_reason = None

    for i in range(entry_bar_idx + 1, len(bars)):
        bar = bars.iloc[i]

        # Force close
        bar_et = _get_bar_time_et(bar['timestamp'])
        if bar_et >= force_close_et:
            baseline_exit_price = bar['open']
            baseline_exit_reason = 'force_close'
            break

        if bar['high'] > highest:
            highest = bar['high']

        # Activate trail
        if not trailing_active:
            r_gain = (highest - entry_price) / risk
            if r_gain >= activate_at_r:
                trailing_active = True

        # Ratchet stop
        if trailing_active:
            new_stop = highest - risk * trail_r
            if new_stop > current_stop:
                current_stop = new_stop

        # Check stop
        if bar['low'] <= current_stop:
            slip = current_stop * exit_slippage_pct
            baseline_exit_price = current_stop - slip
            baseline_exit_reason = 'trail_stop' if trailing_active else 'stop'
            break

    if baseline_exit_price is None:
        baseline_exit_price = bars.iloc[-1]['close']
        baseline_exit_reason = 'eod'

    baseline_pnl = (baseline_exit_price - entry_price) * shares

    # --- Signal simulation ---
    signal_fired = False
    signal_bar_idx = None
    signal_price = None
    signal_r_gain_val = None

    # Re-walk for signal detection
    highest2 = entry_price
    trailing_active2 = False
    current_stop2 = stop_loss

    for i in range(entry_bar_idx + 1, len(bars)):
        bar = bars.iloc[i]

        bar_et = _get_bar_time_et(bar['timestamp'])
        if bar_et >= force_close_et:
            break

        if bar['high'] > highest2:
            highest2 = bar['high']

        # Check if profitable enough for signal
        r_gain = (highest2 - entry_price) / risk
        current_profit_r = (bar['close'] - entry_price) / risk

        # Only check signals when trade is at least +min_r_for_signal
        if current_profit_r >= min_r_for_signal:
            if signal_fn(bars, i):
                signal_fired = True
                signal_bar_idx = i
                signal_price = bar['close']  # sell into strength at close
                signal_r_gain_val = current_profit_r
                break

        # Activate trail (same logic)
        if not trailing_active2:
            if r_gain >= activate_at_r:
                trailing_active2 = True

        if trailing_active2:
            new_stop = highest2 - risk * trail_r
            if new_stop > current_stop2:
                current_stop2 = new_stop

        # Check stop — if stop triggers before signal, signal is useless
        if bar['low'] <= current_stop2:
            break

    if not signal_fired:
        return ExhaustionResult(
            symbol="", trade_date="",
            baseline_pnl=baseline_pnl,
            baseline_exit_price=baseline_exit_price,
            signal_fired=False,
        )

    # --- Compute exhaustion exit P&L ---
    partial_shares = shares // 2
    remainder_shares = shares - partial_shares

    # Partial exit at signal price (0 slippage — selling into strength)
    partial_pnl = (signal_price - entry_price) * partial_shares

    # Remainder: continue with TIGHTER trailing stop (0.5R instead of 1R)
    highest3 = highest2
    trailing_active3 = True  # already activated (we're past +1R)
    current_stop3 = max(current_stop2, highest3 - risk * tighter_trail_r)
    remainder_exit_price = None
    remainder_exit_reason = None

    for i in range(signal_bar_idx + 1, len(bars)):
        bar = bars.iloc[i]

        bar_et = _get_bar_time_et(bar['timestamp'])
        if bar_et >= force_close_et:
            remainder_exit_price = bar['open']
            remainder_exit_reason = 'force_close'
            break

        if bar['high'] > highest3:
            highest3 = bar['high']

        # Ratchet with tighter trail
        new_stop = highest3 - risk * tighter_trail_r
        if new_stop > current_stop3:
            current_stop3 = new_stop

        if bar['low'] <= current_stop3:
            slip = current_stop3 * exit_slippage_pct
            remainder_exit_price = current_stop3 - slip
            remainder_exit_reason = 'tighter_trail'
            break

    if remainder_exit_price is None:
        remainder_exit_price = bars.iloc[-1]['close']
        remainder_exit_reason = 'eod'

    remainder_pnl = (remainder_exit_price - entry_price) * remainder_shares
    exhaustion_pnl = partial_pnl + remainder_pnl

    return ExhaustionResult(
        symbol="", trade_date="",
        baseline_pnl=baseline_pnl,
        baseline_exit_price=baseline_exit_price,
        signal_fired=True,
        signal_bar_idx=signal_bar_idx,
        signal_price=signal_price,
        signal_r_gain=signal_r_gain_val,
        exhaustion_pnl=exhaustion_pnl,
        partial_price=signal_price,
        remainder_exit_price=remainder_exit_price,
        remainder_exit_reason=remainder_exit_reason,
    )


def _get_bar_time_et(ts) -> Tuple[int, int]:
    """Convert bar timestamp to ET (hour, minute)."""
    if hasattr(ts, 'astimezone'):
        bar_et = ts.astimezone(ET)
    elif isinstance(ts, str):
        bar_et = pd.Timestamp(ts).tz_localize('UTC').tz_convert(ET)
    else:
        return (0, 0)
    return (bar_et.hour, bar_et.minute)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run exhaustion exit analysis."""
    parser = argparse.ArgumentParser(description="Exhaustion exit signal analysis")
    parser.add_argument("csv", help="Backtest CSV file path")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(message)s")
    logging.getLogger('persistence.database').setLevel(logging.ERROR)

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} trades from {args.csv}")

    # Only analyze trades that were stopped out (not EOD/force close)
    # Actually test ALL trades — exhaustion exit might turn an EOD into a profit-take
    print(f"Testing {len(df)} trades across {len(SIGNALS)} signals\n")

    db = Database(db_path="data/onemil.db")

    # Run analysis per signal
    results_by_signal: Dict[str, List[ExhaustionResult]] = {
        name: [] for name in SIGNALS
    }

    for idx, row in df.iterrows():
        symbol = row['symbol']
        trade_date = row['date']
        entry_price = row['entry_price']
        stop_loss = row['stop_loss']
        shares = int(row['shares'])
        entry_time = row['entry_time_et']
        entry_h = int(entry_time.split(':')[0])
        entry_m = int(entry_time.split(':')[1])

        cached = db.get_intraday_bars_cached(symbol, trade_date)
        if not cached:
            continue
        bars = pd.DataFrame(cached)
        if len(bars) < 10:
            continue

        # Find entry bar
        entry_bar_idx = None
        for i, r in bars.iterrows():
            ts = r['timestamp']
            bar_et = _get_bar_time_et(ts)
            if bar_et >= (entry_h, entry_m):
                entry_bar_idx = i
                break
        if entry_bar_idx is None:
            continue

        # Test each signal
        for sig_name, sig_fn in SIGNALS.items():
            result = simulate_trade_with_signal(
                bars, entry_bar_idx, entry_price, stop_loss, shares, sig_fn,
            )
            result.symbol = symbol
            result.trade_date = trade_date
            results_by_signal[sig_name].append(result)

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(df)} trades...")

    db.close()

    # Print results
    print(f"\n{'=' * 95}")
    print(f"  EXHAUSTION EXIT SIGNAL ANALYSIS")
    print(f"  Partial exit: 50% at signal (0 slip), 50% tighter trail (0.5R, 0.3% slip)")
    print(f"{'=' * 95}\n")

    baseline_total = None

    for sig_name in SIGNALS:
        results = results_by_signal[sig_name]
        if not results:
            continue

        fired = [r for r in results if r.signal_fired]
        not_fired = [r for r in results if not r.signal_fired]

        total_baseline = sum(r.baseline_pnl for r in results)
        if baseline_total is None:
            baseline_total = total_baseline

        # Signal P&L: fired trades use exhaustion_pnl, unfired use baseline
        total_signal = sum(
            r.exhaustion_pnl if r.signal_fired else r.baseline_pnl
            for r in results
        )

        delta = total_signal - total_baseline

        # How many fired trades improved vs worsened
        improved = [r for r in fired if r.exhaustion_pnl > r.baseline_pnl]
        worsened = [r for r in fired if r.exhaustion_pnl < r.baseline_pnl]
        neutral = [r for r in fired if abs(r.exhaustion_pnl - r.baseline_pnl) < 1]

        # Average R-gain when signal fired
        avg_r = sum(r.signal_r_gain for r in fired) / len(fired) if fired else 0

        # Average improvement per fired trade
        avg_delta = sum(r.exhaustion_pnl - r.baseline_pnl for r in fired) / len(fired) if fired else 0

        sig_label = {
            'vol_divergence': 'Volume Divergence',
            'climax_candle': 'Climax Candle',
            'shrinking_bodies': 'Shrinking Bodies',
            'shooting_star': 'Shooting Star',
        }[sig_name]

        print(f"  {sig_label}")
        print(f"  {'-' * 70}")
        print(f"  Fired on:         {len(fired)}/{len(results)} trades ({len(fired)/len(results)*100:.0f}%)")
        print(f"  Avg R when fired: +{avg_r:.1f}R")
        print(f"  Improved:         {len(improved)} trades  |  Worsened: {len(worsened)} trades")
        print(f"  Avg delta/trade:  ${avg_delta:+,.0f}")
        print(f"  Baseline P&L:     ${total_baseline:+,.0f}")
        print(f"  Signal P&L:       ${total_signal:+,.0f}")
        print(f"  DELTA:            ${delta:+,.0f} ({'+' if delta > 0 else ''}{delta/abs(total_baseline)*100:.1f}%)")

        # Breakdown: winners that got better vs worse
        fired_winners = [r for r in fired if r.baseline_pnl > 0]
        fired_losers = [r for r in fired if r.baseline_pnl <= 0]
        if fired_winners:
            w_better = sum(1 for r in fired_winners if r.exhaustion_pnl > r.baseline_pnl)
            w_delta = sum(r.exhaustion_pnl - r.baseline_pnl for r in fired_winners)
            print(f"    On winning trades: fired {len(fired_winners)}x, {w_better} improved, delta ${w_delta:+,.0f}")
        if fired_losers:
            l_better = sum(1 for r in fired_losers if r.exhaustion_pnl > r.baseline_pnl)
            l_delta = sum(r.exhaustion_pnl - r.baseline_pnl for r in fired_losers)
            print(f"    On losing trades:  fired {len(fired_losers)}x, {l_better} improved, delta ${l_delta:+,.0f}")
        print()

    # Combined signal analysis: fire on ANY signal (first one wins)
    print(f"  {'=' * 70}")
    print(f"  COMBINED: Fire on FIRST signal (any of the above)")
    print(f"  {'-' * 70}")

    combined_total = 0
    combined_fired = 0
    combined_improved = 0
    combined_worsened = 0

    for idx_r in range(len(results_by_signal['vol_divergence'])):
        baseline_pnl = results_by_signal['vol_divergence'][idx_r].baseline_pnl
        best_signal_pnl = baseline_pnl
        any_fired = False

        for sig_name in SIGNALS:
            r = results_by_signal[sig_name][idx_r]
            if r.signal_fired:
                any_fired = True
                # Use the first signal that fires (best exhaustion pnl)
                if r.exhaustion_pnl is not None:
                    if not any_fired or r.signal_bar_idx is not None:
                        best_signal_pnl = r.exhaustion_pnl
                        break

        combined_total += best_signal_pnl
        if any_fired:
            combined_fired += 1
            if best_signal_pnl > baseline_pnl:
                combined_improved += 1
            elif best_signal_pnl < baseline_pnl:
                combined_worsened += 1

    print(f"  Fired on:    {combined_fired} trades")
    print(f"  Improved:    {combined_improved}  |  Worsened: {combined_worsened}")
    print(f"  Baseline:    ${baseline_total:+,.0f}")
    print(f"  Combined:    ${combined_total:+,.0f}")
    print(f"  DELTA:       ${combined_total - baseline_total:+,.0f}")
    print(f"  {'=' * 70}")


if __name__ == "__main__":
    main()
