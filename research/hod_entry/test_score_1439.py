"""Synthetic unit tests for research/hod_entry/score_1439.py (cell 1,439 scorer).

Covers: SSR-today flag from a bar low at 0.9x prior close, SSR-carried flag, coverage/gap
arithmetic, and ex-top-5% mean. Run: nice -n 19 python3 -m pytest research/hod_entry/test_score_1439.py -q
"""
import pandas as pd
import pytest

from score_1439 import coverage_and_gap, ex_top5, ssr_carried_hit, ssr_today_hit


def _bars(rows):
    """rows: list of (m, l) -> DataFrame with columns m, l."""
    return pd.DataFrame(rows, columns=['m', 'l'])


class TestSsrTodayHit:
    def test_hit_at_exactly_point9x(self):
        bars = _bars([(570, 10.0), (571, 9.0)])  # 9.0 == 0.9 x 10.0
        assert ssr_today_hit(bars, fill_min=571, prior_close=10.0) is True

    def test_no_hit_above_threshold(self):
        bars = _bars([(570, 10.0), (571, 9.5)])
        assert ssr_today_hit(bars, fill_min=571, prior_close=10.0) is False

    def test_hit_ignores_bars_after_fill_min(self):
        # low breaches SSR only AFTER the fill minute -> not counted (causal)
        bars = _bars([(570, 10.0), (571, 9.6), (572, 8.0)])
        assert ssr_today_hit(bars, fill_min=571, prior_close=10.0) is False

    def test_none_prior_close_is_false(self):
        bars = _bars([(570, 5.0)])
        assert ssr_today_hit(bars, fill_min=570, prior_close=None) is False

    def test_empty_bars_is_false(self):
        assert ssr_today_hit(_bars([]), fill_min=571, prior_close=10.0) is False
        assert ssr_today_hit(None, fill_min=571, prior_close=10.0) is False


class TestSsrCarriedHit:
    def test_hit(self):
        prev_bars = _bars([(570, 20.0), (571, 17.9)])  # 17.9 < 0.9*20=18.0
        assert ssr_carried_hit(prev_bars, close_prev2=20.0) is True

    def test_no_hit(self):
        prev_bars = _bars([(570, 20.0), (571, 18.5)])
        assert ssr_carried_hit(prev_bars, close_prev2=20.0) is False

    def test_missing_close_is_false(self):
        prev_bars = _bars([(570, 20.0)])
        assert ssr_carried_hit(prev_bars, close_prev2=None) is False


class TestCoverageAndGap:
    def test_full_coverage_no_gap(self):
        df = pd.DataFrame(dict(
            status=['fill', 'fill', 'nofill', 'nofill'],
            b0_net_R=[1.0, -1.0, 1.0, -1.0],
        ))
        cov, gap = coverage_and_gap(df)
        assert cov == pytest.approx(1.0)
        assert gap == pytest.approx(0.0)

    def test_partial_coverage_and_gap(self):
        # winners: 2 rows, 1 no_tape (50%); losers: 2 rows, 0 no_tape (0%) -> gap 50pp
        df = pd.DataFrame(dict(
            status=['no_tape', 'fill', 'fill', 'nofill'],
            b0_net_R=[1.0, 1.0, -1.0, -1.0],
        ))
        cov, gap = coverage_and_gap(df)
        assert cov == pytest.approx(1 - 1 / 4)
        assert gap == pytest.approx(50.0)

    def test_not_armed_excluded_from_crossed(self):
        df = pd.DataFrame(dict(
            status=['not_armed', 'not_armed', 'fill'],
            b0_net_R=[float('nan'), float('nan'), 1.0],
        ))
        cov, gap = coverage_and_gap(df)
        assert cov == pytest.approx(1.0)

    def test_empty_returns_nan(self):
        df = pd.DataFrame(dict(status=[], b0_net_R=[]))
        cov, gap = coverage_and_gap(df)
        assert pd.isna(cov) and pd.isna(gap)


class TestExTop5:
    def test_drops_top_values(self):
        # 20 values 1..20, top 5% (1 value) dropped -> mean of 1..19
        x = list(range(1, 21))
        result = ex_top5(x)
        assert result == pytest.approx(sum(range(1, 20)) / 19)

    def test_small_n_still_drops_at_least_one(self):
        x = [1, 2, 3, 100]
        result = ex_top5(x)
        assert result < 100  # the outlier must be excluded

    def test_empty_is_nan(self):
        assert pd.isna(ex_top5([]))
