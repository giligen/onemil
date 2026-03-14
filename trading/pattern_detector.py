"""
Bull flag pattern detector for 1-minute bars.

Implements Ross Cameron's bull flag criteria:
- Pole: 3+ green candles with 3%+ gain and above-average volume
- Pullback: 2-3 red candles retracing <= 50% of pole on declining volume
- Breakout: First candle breaking above flag high on 1.5x+ volume expansion

The detector operates on COMPLETED bars only — the current (in-progress)
minute bar is always dropped to prevent false signals from partial data.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BullFlagPattern:
    """Detected bull flag pattern with all measurements."""

    symbol: str
    pole_start_idx: int
    pole_end_idx: int
    flag_start_idx: int
    flag_end_idx: int
    pole_low: float
    pole_high: float
    pole_height: float
    pole_gain_pct: float
    flag_low: float
    flag_high: float
    retracement_pct: float
    pullback_candle_count: int
    avg_pole_volume: float
    avg_flag_volume: float
    breakout_level: float
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# BullFlagSetup is the same structure as BullFlagPattern — detected BEFORE
# breakout so we can pre-place a buy-stop order at breakout_level.
BullFlagSetup = BullFlagPattern


class BullFlagDetector:
    """
    Detects bull flag patterns on 1-minute bar data.

    Algorithm:
    1. Drop last bar (in-progress, partial data)
    2. Need minimum 6 completed bars (3 pole + 2 pullback + 1 breakout)
    3. Scan backwards from most recent completed bar looking for pullback
    4. Before pullback, find pole (3+ green candles with 3%+ gain)
    5. Validate: retracement <= 50%, declining volume in flag, expanding on breakout
    6. Return BullFlagPattern or None
    """

    def __init__(
        self,
        min_pole_candles: int = 3,
        min_pole_gain_pct: float = 3.0,
        max_retracement_pct: float = 50.0,
        max_pullback_candles: int = 5,
        min_breakout_volume_ratio: float = 1.5,
    ):
        """
        Initialize BullFlagDetector with configurable thresholds.

        Args:
            min_pole_candles: Minimum consecutive green candles in pole
            min_pole_gain_pct: Minimum pole gain percentage
            max_retracement_pct: Maximum pullback retracement percentage
            max_pullback_candles: Maximum pullback candles (reject if more)
            min_breakout_volume_ratio: Min breakout vol / avg pullback vol
        """
        self.min_pole_candles = min_pole_candles
        self.min_pole_gain_pct = min_pole_gain_pct
        self.max_retracement_pct = max_retracement_pct
        self.max_pullback_candles = max_pullback_candles
        self.min_breakout_volume_ratio = min_breakout_volume_ratio

    def _scan_pole_and_flag(
        self, symbol: str, completed: pd.DataFrame, scan_from_idx: int
    ) -> Optional[dict]:
        """
        Scan for pole + flag structure ending at scan_from_idx.

        Shared logic between detect() (where scan_from_idx = breakout_idx - 1)
        and detect_setup() (where scan_from_idx = last completed bar index).

        Args:
            symbol: Stock ticker symbol (for logging)
            completed: Completed bars DataFrame
            scan_from_idx: Index to start pullback scan backwards from

        Returns:
            Dict with pole/flag indices and metrics, or None if invalid
        """
        if scan_from_idx < 0:
            logger.debug(f"{symbol}: No room for pullback")
            return None

        pullback_indices = self._find_pullback(completed, scan_from_idx)

        if pullback_indices is None:
            logger.debug(f"{symbol}: No valid pullback found")
            return None

        flag_start_idx, flag_end_idx = pullback_indices
        pullback_count = flag_end_idx - flag_start_idx + 1

        if pullback_count < 2:
            logger.debug(f"{symbol}: Pullback too short ({pullback_count} candles, need 2+)")
            return None

        if pullback_count > self.max_pullback_candles:
            logger.debug(f"{symbol}: Pullback too long ({pullback_count} candles, max {self.max_pullback_candles})")
            return None

        # Find the pole (green candles before the pullback)
        pole_end_candidate = flag_start_idx - 1
        if pole_end_candidate < 0:
            logger.debug(f"{symbol}: No room for pole before pullback")
            return None

        pole_indices = self._find_pole(completed, pole_end_candidate)

        if pole_indices is None:
            logger.debug(f"{symbol}: No valid pole found")
            return None

        pole_start_idx, pole_end_idx = pole_indices
        pole_candle_count = pole_end_idx - pole_start_idx + 1

        if pole_candle_count < self.min_pole_candles:
            logger.debug(f"{symbol}: Pole too short ({pole_candle_count} candles, need {self.min_pole_candles}+)")
            return None

        # Calculate pole metrics using vectorized pandas operations
        pole_slice = completed.iloc[pole_start_idx:pole_end_idx + 1]
        pole_low = pole_slice['low'].min()
        pole_high = pole_slice['high'].max()

        pole_height = pole_high - pole_low
        if pole_low <= 0:
            logger.debug(f"{symbol}: Invalid pole_low ({pole_low})")
            return None

        pole_gain_pct = (pole_height / pole_low) * 100

        if pole_gain_pct < self.min_pole_gain_pct:
            logger.debug(f"{symbol}: Pole gain {pole_gain_pct:.1f}% < {self.min_pole_gain_pct}% minimum")
            return None

        # Calculate pullback metrics using vectorized operations
        flag_slice = completed.iloc[flag_start_idx:flag_end_idx + 1]
        flag_low = flag_slice['low'].min()
        flag_high = completed.iloc[flag_end_idx]['high']

        if pole_height <= 0:
            logger.debug(f"{symbol}: Zero pole height")
            return None

        retracement = pole_high - flag_low
        retracement_pct = (retracement / pole_height) * 100

        if retracement_pct > self.max_retracement_pct:
            logger.debug(f"{symbol}: Retracement {retracement_pct:.1f}% > {self.max_retracement_pct}% max")
            return None

        # Volume analysis using vectorized operations
        avg_pole_volume = pole_slice['volume'].mean()
        avg_flag_volume = flag_slice['volume'].mean()

        # Volume should decrease during pullback
        if avg_flag_volume > 0 and avg_pole_volume > 0:
            if avg_flag_volume >= avg_pole_volume:
                logger.debug(f"{symbol}: Flag volume ({avg_flag_volume:.0f}) >= pole volume ({avg_pole_volume:.0f})")
                return None

        return {
            'pole_start_idx': pole_start_idx,
            'pole_end_idx': pole_end_idx,
            'flag_start_idx': flag_start_idx,
            'flag_end_idx': flag_end_idx,
            'pole_low': pole_low,
            'pole_high': pole_high,
            'pole_height': pole_height,
            'pole_gain_pct': pole_gain_pct,
            'flag_low': flag_low,
            'flag_high': flag_high,
            'retracement_pct': retracement_pct,
            'pullback_count': pullback_count,
            'avg_pole_volume': avg_pole_volume,
            'avg_flag_volume': avg_flag_volume,
            'pole_candle_count': pole_candle_count,
        }

    def detect(
        self, symbol: str, bars: pd.DataFrame, end_idx: Optional[int] = None
    ) -> Optional[BullFlagPattern]:
        """
        Detect a bull flag pattern in 1-minute bar data.

        This fires AFTER the breakout candle has completed. The last completed
        bar is treated as the breakout bar and must close above flag_high with
        volume expansion.

        Args:
            symbol: Stock ticker symbol
            bars: DataFrame with columns: open, high, low, close, volume
                  Must be sorted chronologically (oldest first).
            end_idx: If provided, use bars.iloc[:end_idx] as completed bars
                     instead of dropping the last bar. This avoids expensive
                     DataFrame copies in the backtest sliding window.

        Returns:
            BullFlagPattern if detected, None otherwise
        """
        if bars is None or bars.empty:
            logger.debug(f"{symbol}: No bars provided")
            return None

        if len(bars) < 2:
            logger.debug(f"{symbol}: Only {len(bars)} bar(s), need at least 7 (6 completed + 1 dropped)")
            return None

        # When end_idx is provided (backtest mode), use it as the slice boundary.
        # Otherwise, drop the last bar (live mode — it's the current minute, still forming).
        if end_idx is not None:
            completed = bars.iloc[:end_idx]
        else:
            completed = bars.iloc[:-1]

        min_required = self.min_pole_candles + 2 + 1  # pole + min pullback + breakout
        if len(completed) < min_required:
            logger.debug(f"{symbol}: Only {len(completed)} completed bars, need >= {min_required}")
            return None

        # The breakout candle is the last completed bar
        breakout_idx = len(completed) - 1
        breakout_bar = completed.iloc[breakout_idx]

        # Scan for pole + flag ending one bar before breakout
        scan = self._scan_pole_and_flag(symbol, completed, breakout_idx - 1)
        if scan is None:
            return None

        # Breakout validation: price must break above flag high
        breakout_level = scan['flag_high']
        if breakout_bar['close'] <= breakout_level:
            logger.debug(f"{symbol}: No breakout — close {breakout_bar['close']:.2f} <= flag high {breakout_level:.2f}")
            return None

        # Breakout volume expansion
        breakout_volume = breakout_bar['volume']
        if scan['avg_flag_volume'] > 0:
            volume_ratio = breakout_volume / scan['avg_flag_volume']
            if volume_ratio < self.min_breakout_volume_ratio:
                logger.debug(
                    f"{symbol}: Breakout volume ratio {volume_ratio:.1f}x "
                    f"< {self.min_breakout_volume_ratio}x minimum"
                )
                return None

        pattern = BullFlagPattern(
            symbol=symbol,
            pole_start_idx=scan['pole_start_idx'],
            pole_end_idx=scan['pole_end_idx'],
            flag_start_idx=scan['flag_start_idx'],
            flag_end_idx=scan['flag_end_idx'],
            pole_low=scan['pole_low'],
            pole_high=scan['pole_high'],
            pole_height=scan['pole_height'],
            pole_gain_pct=scan['pole_gain_pct'],
            flag_low=scan['flag_low'],
            flag_high=scan['flag_high'],
            retracement_pct=scan['retracement_pct'],
            pullback_candle_count=scan['pullback_count'],
            avg_pole_volume=scan['avg_pole_volume'],
            avg_flag_volume=scan['avg_flag_volume'],
            breakout_level=breakout_level,
        )

        logger.info(
            f"{symbol}: BULL FLAG DETECTED — "
            f"pole {scan['pole_candle_count']} candles (+{scan['pole_gain_pct']:.1f}%), "
            f"pullback {scan['pullback_count']} candles ({scan['retracement_pct']:.0f}% retrace), "
            f"breakout @ {breakout_level:.2f}"
        )
        return pattern

    def detect_setup(
        self, symbol: str, bars: pd.DataFrame, end_idx: Optional[int] = None
    ) -> Optional[BullFlagSetup]:
        """
        Detect a bull flag setup BEFORE breakout — for pre-placing buy-stop orders.

        Unlike detect(), this does NOT require a breakout candle. The last
        completed bar is treated as the last flag bar, and the setup's
        breakout_level = flag_high. A buy-stop order can then be placed at
        breakout_level to get filled when the breakout actually occurs.

        Args:
            symbol: Stock ticker symbol
            bars: DataFrame with columns: open, high, low, close, volume
            end_idx: If provided, use bars.iloc[:end_idx] as completed bars

        Returns:
            BullFlagSetup if a valid pole+flag is found, None otherwise
        """
        if bars is None or bars.empty:
            logger.debug(f"{symbol}: No bars provided")
            return None

        if len(bars) < 2:
            logger.debug(f"{symbol}: Only {len(bars)} bar(s), insufficient")
            return None

        if end_idx is not None:
            completed = bars.iloc[:end_idx]
        else:
            completed = bars.iloc[:-1]

        # Setup needs pole + pullback only (no breakout bar)
        min_required = self.min_pole_candles + 2  # pole + min pullback
        if len(completed) < min_required:
            logger.debug(f"{symbol}: Only {len(completed)} completed bars, need >= {min_required}")
            return None

        # Scan for pole + flag ending at the last completed bar
        last_idx = len(completed) - 1
        scan = self._scan_pole_and_flag(symbol, completed, last_idx)
        if scan is None:
            return None

        breakout_level = scan['flag_high']

        setup = BullFlagSetup(
            symbol=symbol,
            pole_start_idx=scan['pole_start_idx'],
            pole_end_idx=scan['pole_end_idx'],
            flag_start_idx=scan['flag_start_idx'],
            flag_end_idx=scan['flag_end_idx'],
            pole_low=scan['pole_low'],
            pole_high=scan['pole_high'],
            pole_height=scan['pole_height'],
            pole_gain_pct=scan['pole_gain_pct'],
            flag_low=scan['flag_low'],
            flag_high=scan['flag_high'],
            retracement_pct=scan['retracement_pct'],
            pullback_candle_count=scan['pullback_count'],
            avg_pole_volume=scan['avg_pole_volume'],
            avg_flag_volume=scan['avg_flag_volume'],
            breakout_level=breakout_level,
        )

        logger.info(
            f"{symbol}: BULL FLAG SETUP — "
            f"pole {scan['pole_candle_count']} candles (+{scan['pole_gain_pct']:.1f}%), "
            f"pullback {scan['pullback_count']} candles ({scan['retracement_pct']:.0f}% retrace), "
            f"buy-stop @ {breakout_level:.2f}"
        )
        return setup

    def _find_pullback(
        self, bars: pd.DataFrame, end_idx: int
    ) -> Optional[Tuple[int, int]]:
        """
        Find pullback (red candles) scanning backwards from end_idx.

        A red candle is one where close < open.
        Doji candles (close == open) are treated as neutral and included in pullback.

        Uses numpy arrays for fast comparison instead of per-row iloc access.

        Args:
            bars: Completed bars DataFrame
            end_idx: Index to start scanning backwards from

        Returns:
            Tuple of (start_idx, end_idx) for pullback, or None
        """
        opens = bars['open'].values
        closes = bars['close'].values

        flag_end_idx = None
        flag_start_idx = None

        i = end_idx
        while i >= 0:
            is_red = closes[i] < opens[i]
            is_doji = closes[i] == opens[i]

            if is_red or is_doji:
                if flag_end_idx is None:
                    flag_end_idx = i
                flag_start_idx = i
            else:
                # Green candle — end of pullback
                if flag_end_idx is not None:
                    break
                else:
                    # No red candle found yet, this isn't a pullback
                    return None
            i -= 1

        if flag_end_idx is None or flag_start_idx is None:
            return None

        return (flag_start_idx, flag_end_idx)

    def _find_pole(
        self, bars: pd.DataFrame, end_idx: int
    ) -> Optional[Tuple[int, int]]:
        """
        Find pole (green candles) scanning backwards from end_idx.

        A green candle is one where close > open.

        Uses numpy arrays for fast comparison instead of per-row iloc access.

        Args:
            bars: Completed bars DataFrame
            end_idx: Index to start scanning backwards from

        Returns:
            Tuple of (start_idx, end_idx) for pole, or None
        """
        opens = bars['open'].values
        closes = bars['close'].values

        pole_end_idx = None
        pole_start_idx = None

        i = end_idx
        while i >= 0:
            is_green = closes[i] > opens[i]

            if is_green:
                if pole_end_idx is None:
                    pole_end_idx = i
                pole_start_idx = i
            else:
                # Non-green candle — end of pole
                if pole_end_idx is not None:
                    break
                else:
                    # No green candle found at end_idx
                    return None
            i -= 1

        if pole_end_idx is None or pole_start_idx is None:
            return None

        return (pole_start_idx, pole_end_idx)
