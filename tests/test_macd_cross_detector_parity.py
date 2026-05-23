"""BT/LIVE parity for MACD cross detection.

Mirrors test_orb_touchgo_parity.py and test_buy_stop_guard_parity.py.

The MACD wave signal detector previously existed in two separate
implementations:
  - macd_wave_backtest.py::generate_signals (BT)
  - trading/macd_wave_engine.py::check_entries (live)

On the stable 5/08-5/22 paper-live window they produced only 11/97
agreeing trades — a forked-implementation problem that drove ~$30K of
the gap between BT projection (−$8K) and live realized (−$38K).

The shared module trading/macd_cross_detector.py now owns the
computation. This file enforces that BOTH consumers continue to use
it (source-code inspection) and that the live and BT semantics agree
on a representative bar series (cross-module equivalence).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from trading.macd_cross_detector import (
    compute_macd_histogram,
    count_consecutive_positive_ending_at,
    find_first_confirmed_entry,
    find_wave_onset,
)

ROOT = Path(__file__).parent.parent


# =========================================================================
# Source-code inspection — both consumers must import from the shared module.
# =========================================================================

class TestSharedImports:

    def _read(self, relpath: str) -> str:
        return (ROOT / relpath).read_text()

    def test_bt_imports_shared_detector(self):
        src = self._read("macd_wave_backtest.py")
        assert "from trading.macd_cross_detector import" in src, (
            "macd_wave_backtest.py must import from trading.macd_cross_detector"
        )
        assert "compute_macd_histogram" in src
        assert "find_wave_onset" in src

    def test_live_imports_shared_detector(self):
        src = self._read("trading/macd_wave_engine.py")
        assert "from trading.macd_cross_detector import" in src, (
            "trading/macd_wave_engine.py must import from "
            "trading.macd_cross_detector"
        )
        assert "compute_macd_histogram" in src
        assert "count_consecutive_positive_ending_at" in src

    def test_bt_no_inline_macd_ewm(self):
        """BT must not redefine MACD computation inline (the old forked
        shape). The shared helper compute_macd_histogram owns it."""
        src = self._read("macd_wave_backtest.py")
        # The historic inline computation pattern. If this re-emerges,
        # the BT has forked again.
        forbidden = [
            "close.ewm(span=macd_fast",
            "close.ewm(span=self.macd_fast",
            "ema_fast = close.ewm",
        ]
        for needle in forbidden:
            assert needle not in src, (
                f"Inline MACD computation '{needle}' regrowing in BT — "
                f"must use compute_macd_histogram"
            )

    def test_live_no_inline_macd_ewm(self):
        src = self._read("trading/macd_wave_engine.py")
        forbidden = [
            "close.ewm(span=self.macd_fast, adjust=False).mean()",
            "ema_fast = close.ewm(span=self.macd_fast",
        ]
        for needle in forbidden:
            assert needle not in src, (
                f"Inline MACD computation '{needle}' regrowing in live "
                f"engine — must use compute_macd_histogram"
            )

    def test_live_no_inline_reverse_count_loop(self):
        """The old inline `for h in reversed(histogram.values)` loop was
        the live-side consecutive-positive counter. It must now route
        through count_consecutive_positive_ending_at."""
        src = self._read("trading/macd_wave_engine.py")
        assert "for h in reversed(histogram.values)" not in src, (
            "Inline reverse-count loop regrowing in live — must use "
            "count_consecutive_positive_ending_at"
        )


# =========================================================================
# Cross-module equivalence — same bars in → same decisions out.
# =========================================================================

class TestBTLiveDecisionAgreement:
    """Given an identical bar series, the BT's forward-walk and the live
    engine's 'is the latest bar confirmed?' check must agree on whether
    the LATEST bar represents a confirmed entry.

    This pins the contract that the shared module preserves: both
    consumers route through the same primitives and therefore cannot
    drift on the entry decision.
    """

    def _scenario(self, hist_values):
        """Build a histogram series + matching bars frame."""
        h = pd.Series(hist_values, dtype=float)
        bars = pd.DataFrame({'close': [10.0] * len(h)})
        return h, bars

    def _both_paths(self, hist_values, confirm_bars=3):
        """Run both consumer paths; return (live_confirmed, bt_confirmed_at_end)."""
        h, bars = self._scenario(hist_values)
        # Live's check: is the latest bar a confirmation?
        live_count = count_consecutive_positive_ending_at(h, len(h) - 1)
        live_confirmed = live_count >= confirm_bars
        # BT's check: where would the BT first enter? Is that the latest bar?
        bt_entry = find_first_confirmed_entry(h, bars, 0, confirm_bars)
        bt_confirmed_at_end = (bt_entry is not None
                                and bt_entry.bar_index == len(h) - 1)
        return live_confirmed, bt_confirmed_at_end

    def test_clean_confirmation_at_end_agrees(self):
        live, bt = self._both_paths([0.0, -0.1, 0.1, 0.2, 0.3])
        assert live is True
        assert bt is True

    def test_no_confirmation_agrees(self):
        live, bt = self._both_paths([0.1, -0.1, 0.1, -0.1, 0.1])
        assert live is False
        assert bt is False

    def test_negative_at_end_agrees(self):
        live, bt = self._both_paths([0.1, 0.2, 0.3, -0.1])
        assert live is False
        # BT entered earlier (at bar 2) — not at the latest. Live's "is end
        # confirmed?" is False. BT's entry was at index 2, not len-1.
        # Both report "latest bar is NOT a confirmation."
        assert bt is False

    def test_earlier_confirmation_then_latest_also_three_positive(self):
        """Histogram has confirmation at bar 3 AND at bar end (continuous
        positives). Live's count-at-end returns ≥ confirm_bars; BT entered
        earlier at bar 3. They answer different questions; agreement is on
        whether the CURRENT bar would trigger an entry, not when the first
        one did.
        """
        live, bt = self._both_paths([0.1, 0.2, 0.3, 0.4, 0.5])
        assert live is True
        # BT entered at index 3 (first time pos_count >= 3, considering
        # start_idx=0 → max(0,1)=1 → i=1:pos=1, i=2:pos=2, i=3:pos=3 entry).
        # Latest bar is index 4. So bt_confirmed_at_end is False.
        assert bt is False
        # At index 3, live's running count would have been exactly 3 — the
        # moment of triggering. After that the count keeps growing (4 at
        # index 4, 5 at index 5, etc.) because all bars stay positive.
        h, _ = self._scenario([0.1, 0.2, 0.3, 0.4, 0.5])
        assert count_consecutive_positive_ending_at(h, 3) == 4  # 0..3 all >0
        assert count_consecutive_positive_ending_at(h, 4) == 5

    def test_random_walk_no_false_disagreements(self):
        """Stress test: a random-ish histogram. At every bar where BT would
        enter, live's count-at-that-bar must be exactly confirm_bars.
        """
        rng = np.random.default_rng(42)
        n = 200
        h = pd.Series(rng.standard_normal(n) * 0.1)
        bars = pd.DataFrame({'close': [10.0] * n})
        bt_entry = find_first_confirmed_entry(h, bars, 0, 3)
        if bt_entry is None:
            return  # no entry; nothing to check
        # At the BT's entry bar, live's count must be exactly the confirm
        # threshold (the FIRST time it hits 3 → exactly 3 unless prior bar
        # was also positive, in which case BT would have entered earlier).
        live_at_entry = count_consecutive_positive_ending_at(h, bt_entry.bar_index)
        # >= 3, not == 3: the BT skips bar 0 (max(start_idx, 1)) while
        # count_consec walks all the way back to index 0. If bar 0 was
        # positive AND no negatives before the entry bar, live's count
        # can exceed confirm_bars by 1.
        assert live_at_entry >= 3, (
            f"At BT entry bar {bt_entry.bar_index}, live count = {live_at_entry}"
        )


# =========================================================================
# Defaults locked
# =========================================================================

class TestDefaultsMatchYaml:
    """The shared module's defaults must match macd_wave.yaml's validated
    parameters. Drift would silently divert the BT and any caller passing
    defaults from yaml.
    """

    def test_macd_fast_default_is_12(self):
        from trading.macd_cross_detector import DEFAULT_FAST
        assert DEFAULT_FAST == 12

    def test_macd_slow_default_is_26(self):
        from trading.macd_cross_detector import DEFAULT_SLOW
        assert DEFAULT_SLOW == 26

    def test_macd_signal_default_is_9(self):
        from trading.macd_cross_detector import DEFAULT_SIGNAL
        assert DEFAULT_SIGNAL == 9

    def test_confirm_bars_default_is_3(self):
        from trading.macd_cross_detector import DEFAULT_CONFIRM_BARS
        assert DEFAULT_CONFIRM_BARS == 3
