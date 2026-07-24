"""Shared Ignition rules (2026-07-24 parity refactor) — every gate value
pinned against the audited research book (1,331 trades):
open floor 2.0 / gap<5 / chase max 1.05x level / R>=5.04 min /
pos>=2,022 min / max entry-vs-level 1.0030."""
from __future__ import annotations

import pandas as pd
import pytest

from trading import ignition_rules as R


class TestUniverseGates:
    def test_price_floor_on_open(self):
        assert R.universe_reject(1.99, 1.80) == 'skip_price_floor'
        assert R.universe_reject(2.00, 1.95) is None

    def test_open_gap_orb_disjoint(self):
        # 5%+ true open gap belongs to ORB
        assert R.universe_reject(10.5, 10.0) == 'skip_gap_orb_territory'
        assert R.universe_reject(10.49, 10.0) is None

    def test_missing_prev_close_fails_open(self):
        assert R.universe_reject(10.5, None) is None


class TestTriggerAndEntry:
    def test_level(self):
        assert R.level(10.0) == pytest.approx(11.0)

    def test_level_crossed_on_high_not_current(self):
        # pullback after the cross still counts (7/21 BIYA class)
        assert R.level_crossed([10.5, 11.2, 10.8], 10.0) is True
        assert R.level_crossed([10.5, 10.9], 10.0) is False

    def test_chase_guard_book_bound(self):
        # book max entry/level = 1.0030 — well within; guard at 1.05
        assert R.chase_reject(11.0 * 1.049, 10.0) is False
        assert R.chase_reject(11.0 * 1.051, 10.0) is True


class TestStopSizing:
    def test_stop_min_of_pre_low_and_99pct(self):
        assert R.stop_from_pre_lows(9.5, 10.0) == pytest.approx(9.5)
        assert R.stop_from_pre_lows(9.99, 10.0) == pytest.approx(9.9)

    def test_position_risk_parity_and_caps(self):
        # R=10% -> 3000/0.10 = 30K -> account cap 25K
        assert R.position_usd(10.0, 1e9) == pytest.approx(25000)
        # participation binds: 15% of $20K bar = $3K
        assert R.position_usd(10.0, 20000) == pytest.approx(3000)

    def test_position_floor(self):
        assert R.position_reject(1999) is True
        assert R.position_reject(2000) is False


class TestCatalyst:
    def test_news_confirms(self):
        assert R.catalyst_confirmed(True, None, 0) is True

    def test_trigger_cohort_confirms(self):
        assert R.catalyst_confirmed(False, 'NBIS', 2) is True
        assert R.catalyst_confirmed(False, 'NBIS', 1) is False

    def test_news_unknown_needs_complex(self):
        # fetch failure: only complex can confirm (7/24 semantics)
        assert R.catalyst_confirmed(None, 'X', 2) is True
        assert R.catalyst_confirmed(None, 'X', 1) is False
        assert R.catalyst_confirmed(None, None, 5) is False


class TestResimExit:
    def _bars(self, rows):
        return pd.DataFrame([{'m': m, 'open': o, 'high': h, 'low': lo,
                              'close': c} for m, o, h, lo, c in rows])

    def test_arm_lock_stop_eod(self):
        e, s = 10.0, 9.0
        b = self._bars([(601, 10.5, 11.8, 10.4, 11.7),
                        (602, 11.0, 11.1, 10.4, 10.5)])
        rr, why = R.resim_exit(b, e, s, 600)
        assert why == 'lock' and rr == pytest.approx((10.5 * .999 - e))

    def test_gap_down_fills_at_open(self):
        b = self._bars([(601, 8.0, 8.2, 7.9, 8.1)])
        rr, why = R.resim_exit(b, 10.0, 9.0, 600)
        assert why == 'stop' and rr == pytest.approx(8.0 * .999 - 10.0)
