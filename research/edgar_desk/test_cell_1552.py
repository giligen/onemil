"""Unit tests for research/edgar_desk/cell_1552.py -- entry-auction timing, SSR exclusion,
E5 window, direction sign, and class assignment. All synthetic (no dependency on events_raw.csv
or the live fetch), per CLAUDE.md's bug protocol (every logic path gets a unit test)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cell_1552 import (  # noqa: E402
    classify_from_row,
    entry_session,
    is_test_ticker,
    score_event,
)


def _session_index(dates):
    return np.sort(np.array([np.datetime64(d) for d in dates]))


class TestEntrySessionTiming:
    def test_before_0900_et_same_day(self):
        sessions = _session_index(["2026-01-05", "2026-01-06", "2026-01-07"])
        result = entry_session(pd.Timestamp("2026-01-06 07:15:00"), sessions)
        assert result == pd.Timestamp("2026-01-06")

    def test_0931_acceptance_goes_to_next_session(self):
        """The refuter named in the PREREG: a 09:31 acceptance must NEVER get a same-day open."""
        sessions = _session_index(["2026-01-05", "2026-01-06", "2026-01-07"])
        result = entry_session(pd.Timestamp("2026-01-06 09:31:00"), sessions)
        assert result == pd.Timestamp("2026-01-07")

    def test_exactly_0900_is_not_before_0900(self):
        sessions = _session_index(["2026-01-05", "2026-01-06", "2026-01-07"])
        result = entry_session(pd.Timestamp("2026-01-06 09:00:00"), sessions)
        assert result == pd.Timestamp("2026-01-07")

    def test_weekend_acceptance_rolls_to_next_session(self):
        # Saturday 2026-01-10 at 03:00 ET, before 09:00 -> "same day" candidate is a non-session
        # date, so it must roll forward to the next actual session, not disappear.
        sessions = _session_index(["2026-01-09", "2026-01-12", "2026-01-13"])
        result = entry_session(pd.Timestamp("2026-01-10 03:00:00"), sessions)
        assert result == pd.Timestamp("2026-01-12")

    def test_none_when_beyond_loaded_bars(self):
        sessions = _session_index(["2026-01-05", "2026-01-06"])
        result = entry_session(pd.Timestamp("2026-06-01 08:00:00"), sessions)
        assert result is None

    def test_nan_acceptance_returns_none(self):
        sessions = _session_index(["2026-01-05", "2026-01-06"])
        assert entry_session(pd.NaT, sessions) is None
        assert entry_session(None, sessions) is None


class TestClassAssignment:
    def test_424b_forms_are_offering(self):
        assert classify_from_row("424B3", "") == ["OFFERING"]

    def test_item_302_is_offering_even_off_form(self):
        assert "OFFERING" in classify_from_row("8-K", "3.02")

    def test_shelf_forms(self):
        for form in ("S-3", "S-3ASR", "S-1"):
            assert classify_from_row(form, "") == ["SHELF"]

    def test_reverse_split_item_503(self):
        assert classify_from_row("8-K", "5.03") == ["REVERSE_SPLIT"]

    def test_auditor_item_401(self):
        assert classify_from_row("8-K", "4.01") == ["AUDITOR"]

    def test_non_reliance_item_402(self):
        assert classify_from_row("8-K", "4.02") == ["NON_RELIANCE"]

    def test_late_filing_forms(self):
        assert classify_from_row("NT 10-K", "") == ["LATE_FILING"]
        assert classify_from_row("NT 10-Q", "") == ["LATE_FILING"]

    def test_officer_exit_item_502(self):
        assert classify_from_row("8-K", "5.02") == ["OFFICER_EXIT"]

    def test_contract_item_101_alone(self):
        assert classify_from_row("8-K", "1.01") == ["CONTRACT"]

    def test_contract_gated_off_by_302(self):
        """1.01 co-filed with 3.02 (the PREREG's own contradiction flag) must NOT also fire
        CONTRACT -- OFFERING only, per the 'WITHOUT items 3.02/2.03' clause in cell 1,559."""
        classes = classify_from_row("8-K", "1.01,3.02")
        assert "OFFERING" in classes
        assert "CONTRACT" not in classes

    def test_contract_gated_off_by_203(self):
        classes = classify_from_row("8-K", "1.01,2.03")
        assert "CONTRACT" not in classes

    def test_activist_sc13d(self):
        assert classify_from_row("SC 13D", "") == ["ACTIVIST"]

    def test_report_only_forms_get_no_class(self):
        assert classify_from_row("10-Q", "") == []
        assert classify_from_row("8-K", "2.02") == []
        assert classify_from_row("8-K", "7.01") == []
        assert classify_from_row("8-K", "8.01") == []

    def test_multi_item_filing_can_carry_several_classes(self):
        classes = classify_from_row("8-K", "4.01,5.02")
        assert set(classes) == {"AUDITOR", "OFFICER_EXIT"}


class TestTestTickerExclusion:
    def test_zzzt_pattern_excluded(self):
        assert is_test_ticker("ZVZZT")
        assert is_test_ticker("ZBZZT")

    def test_zz_prefix_excluded(self):
        assert is_test_ticker("ZZTEST")

    def test_ordinary_symbol_kept(self):
        assert not is_test_ticker("AAPL")
        assert not is_test_ticker("Z")


class TestScoreEventDirectionAndWindows:
    def _bars(self):
        # 10 sessions, flat $10 open->close except a scripted move on session index 3 (entry)
        dates = pd.date_range("2026-02-02", periods=10, freq="B")
        closes = [10.0, 10.0, 10.0, 9.0, 9.0, 9.0, 9.0, 9.0, 8.0, 8.0]  # entry day drops to 9
        opens = [10.0] * 3 + [10.0] + [9.0] * 5 + [8.0]
        df = pd.DataFrame({"bar_date": dates, "open": opens, "close": closes,
                            "high": closes, "low": closes, "volume": [1e6] * 10,
                            "dvol20": [2e6] * 10})
        df = df.set_index("bar_date")
        df["prior_close"] = df["close"].shift(1)
        df["prior_dvol20"] = df["dvol20"].shift(1)
        return df

    def test_short_direction_flips_sign_on_a_down_move(self):
        """Entry day 3: open 10.0 -> close 9.0 is a -10% raw move; a SHORT cell should show a
        POSITIVE net return in the trade direction (net of the 10 bps round-trip cost)."""
        bars = self._bars()
        entry_date = bars.index[3]
        res = score_event(bars, entry_date, direction=-1)
        assert res is not None
        assert res["ret_e1_net"] > 0
        # gross = +0.10 (short profits from the drop); cost = 10 bps + 1 day borrow
        expected_gross = 0.10
        assert abs(res["ret_e1_net"] - (expected_gross - 0.0010 - (0.03 / 365))) < 1e-9

    def test_long_direction_same_move_is_negative(self):
        bars = self._bars()
        entry_date = bars.index[3]
        res = score_event(bars, entry_date, direction=1)
        assert res["ret_e1_net"] < 0

    def test_ssr_flag_on_prior_session_minus_10pct(self):
        bars = self._bars()
        # session index 3's prior session (index 2, close 10.0) vs the session before that
        # (index 1, close 10.0) is flat -- no SSR. Force a >=10% prior-day drop instead.
        bars = bars.copy()
        bars.iloc[2, bars.columns.get_loc("close")] = 8.5  # index1->index2 drop of 15%
        bars["prior_close"] = bars["close"].shift(1)
        entry_date = bars.index[3]
        res = score_event(bars, entry_date, direction=-1)
        assert res["ssr"] is True

    def test_no_ssr_when_prior_move_is_small(self):
        bars = self._bars()
        entry_date = bars.index[1]  # prior session flat vs the one before
        res = score_event(bars, entry_date, direction=-1)
        assert res["ssr"] is False

    def test_e5_window_is_exactly_five_sessions_after_entry(self):
        bars = self._bars()
        entry_date = bars.index[2]
        res = score_event(bars, entry_date, direction=1)
        exit_close = bars.iloc[2 + 5]["close"]
        entry_open = bars.iloc[2]["open"]
        expected_gross = (exit_close / entry_open) - 1.0
        cost = 0.0010
        assert abs(res["ret_e5_net"] - (expected_gross - cost)) < 1e-9

    def test_e5_missing_when_fewer_than_5_sessions_remain(self):
        bars = self._bars()
        entry_date = bars.index[8]  # only 1 session left after it
        res = score_event(bars, entry_date, direction=1)
        assert np.isnan(res["ret_e5_net"])

    def test_ineligible_when_prior_dvol20_below_floor(self):
        bars = self._bars().copy()
        bars["prior_dvol20"] = 100.0  # below MIN_DVOL20
        entry_date = bars.index[3]
        res = score_event(bars, entry_date, direction=-1)
        assert bool(res["eligible"]) is False

    def test_price_floor_exclusion_flagged_for_short(self):
        bars = self._bars().copy()
        bars.iloc[2, bars.columns.get_loc("close")] = 4.0  # prior_close for entry index 3
        bars["prior_close"] = bars["close"].shift(1)
        entry_date = bars.index[3]
        res = score_event(bars, entry_date, direction=-1)
        assert bool(res["price_floor_excl"]) is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
