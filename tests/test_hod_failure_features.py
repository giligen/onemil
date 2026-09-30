"""Known-answer tests for trading/hod_failure_features.py (cell 1,675, PREREG_1675.md).

Every expected value below is hand-computed from the bar table in `_bars()` (see the module
docstring's worked example) -- an independent arithmetic check, not a re-run of the function under
test. TA-Lib pattern outputs are excluded from the arithmetic assertions (61 patterns on a 5-bar
synthetic day is not a meaningful "known answer"); a separate smoke test only checks the talib_*
keys appear with the right shape when TA-Lib is available.
"""
import math

import numpy as np
import pytest

from trading import hod_failure_features as hf


def _bars():
    """5 RTH minute bars, all volume=1000 (so VWAP-through-idx = plain mean of typical price)."""
    return dict(
        minarr=np.array([570, 571, 572, 573, 574]),
        o=np.array([10.00, 10.00, 10.10, 10.10, 10.10]),
        h=np.array([10.00, 10.20, 10.20, 10.15, 10.12]),
        l=np.array([10.00, 10.00, 10.02, 10.05, 10.00]),
        c=np.array([10.00, 10.10, 10.10, 10.10, 10.05]),
        v=np.array([1000, 1000, 1000, 1000, 1000], dtype=float),
    )


class TestFindFillIndex:
    def test_fractional_minute_uses_last_bar_at_or_before(self):
        assert hf.find_fill_index(_bars(), 571.7) == 1

    def test_exact_minute_match(self):
        assert hf.find_fill_index(_bars(), 572.0) == 2

    def test_before_first_bar_returns_none(self):
        assert hf.find_fill_index(_bars(), 569.0) is None


class TestFindBreakBar:
    def test_first_matching_high_within_tolerance(self):
        # idx0 high=10.00 (no match), idx1 high=10.20 (matches level=10.20) -> first match is idx1
        assert hf.find_break_bar(_bars(), 572.0, level=10.20) == 1

    def test_no_level_returns_none(self):
        assert hf.find_break_bar(_bars(), 572.0, level=None) is None


class TestWalkK:
    def test_no_preempt_when_neither_stop_nor_target_touched(self):
        w = hf.walk_k(_bars(), i0=1, k=1, stop=10.00, target=10.45)
        assert w['preempt'] == ''
        assert w['close_k'] == pytest.approx(10.10)
        assert w['high_max'] == pytest.approx(10.20)
        assert w['low_min'] == pytest.approx(10.02)
        assert w['next_open'] == pytest.approx(10.10)

    def test_stop_preempt(self):
        # fill+1 low (10.02) at/below a stop of 10.05 -> preempt='stop'
        w = hf.walk_k(_bars(), i0=1, k=1, stop=10.05, target=10.45)
        assert w['preempt'] == 'stop'

    def test_none_when_horizon_bar_missing(self):
        assert hf.walk_k(_bars(), i0=4, k=1, stop=9.0, target=11.0) is None


class TestBar0Features:
    def test_known_values(self):
        f = hf.bar0_features(_bars(), i0=1, entry=10.15, R=0.15, atr14=0.20, break_bar_v=1000.0)
        assert f['close0'] == pytest.approx(10.10)
        assert f['clv0'] == pytest.approx(0.5)
        assert f['body0'] == pytest.approx(0.5)
        assert f['rangeATR0'] == pytest.approx(1.0)
        assert f['volratio0'] == pytest.approx(1.0)
        assert f['closeR0'] == pytest.approx(-1.0 / 3.0)

    def test_zero_range_bar_is_nan_not_crash(self):
        f = hf.bar0_features(_bars(), i0=0, entry=10.0, R=0.1, atr14=0.2, break_bar_v=1000.0)
        assert math.isnan(f['clv0']) and math.isnan(f['body0'])


class TestK1Features:
    ARM_CTX = dict(r_pct=1.5, atr14_pct=2.0, F11=1, F12=0, F13=1, F14=0, F15=1, minutes_since_open=1.0)

    def test_computable_known_answer(self, monkeypatch):
        monkeypatch.setattr(hf, 'TALIB_AVAILABLE', False)  # isolate the hand-checked arithmetic
        feats, computable = hf.k1_features(
            _bars(), i0=1, entry=10.15, stop=10.00, target=10.45, level=10.20, atr14=0.20,
            break_bar_v=1000.0, arm_ctx=self.ARM_CTX, spy_bars=None, fill_min=571.0)
        assert computable is True
        # arm-time passthrough
        assert feats['r_pct'] == pytest.approx(1.5)
        assert feats['minutes_since_open'] == pytest.approx(1.0)
        # bar0
        assert feats['clv0'] == pytest.approx(0.5)
        assert feats['closeR0'] == pytest.approx(-1.0 / 3.0)
        # k=1 continuous (hand-computed, see module docstring)
        assert feats['cS1_dist_level_1'] == pytest.approx(-2.0 / 3.0)
        assert feats['cS2_mfe_1'] == pytest.approx(1.0 / 3.0)
        assert feats['cS3_ret_1'] == pytest.approx(-1.0 / 3.0)
        assert feats['cS4_volratio_1'] == pytest.approx(1.0)
        assert math.isnan(feats['cS5_spyret_1'])  # no spy_bars passed
        assert feats['cS6_dist_vwap_1'] == pytest.approx(0.2074074, abs=1e-5)
        assert feats['cS7_mae_1'] == pytest.approx(13.0 / 15.0)
        assert feats['cA1_progvol_1'] == pytest.approx(-1.0 / 3.0)
        # k=1 shape
        assert feats['clv_last_1'] == pytest.approx(4.0 / 9.0)
        assert feats['wick_last_1'] == pytest.approx(4.0 / 9.0)
        assert feats['body_last_1'] == pytest.approx(0.0)
        assert feats['clv_mean_1'] == pytest.approx(4.0 / 9.0)
        assert feats['red_share_1'] == pytest.approx(0.0)
        assert feats['close_1'] == pytest.approx(10.10)
        assert feats['next_open_1'] == pytest.approx(10.10)
        # no talib_*/cdl_* keys leaked in with TALIB_AVAILABLE patched off
        assert not any(k.startswith('talib_') or k.startswith('cdl_') for k in feats)

    def test_not_computable_when_stop_hit_at_fill_plus_1(self, monkeypatch):
        monkeypatch.setattr(hf, 'TALIB_AVAILABLE', False)
        feats, computable = hf.k1_features(
            _bars(), i0=1, entry=10.15, stop=10.05, target=10.45, level=10.20, atr14=0.20,
            break_bar_v=1000.0, arm_ctx=self.ARM_CTX, spy_bars=None, fill_min=571.0)
        assert computable is False
        # arm/bar0 features are still returned (known at decision time); k=1 block is not
        assert 'clv0' in feats
        assert 'cS1_dist_level_1' not in feats

    @pytest.mark.skipif(not hf.TALIB_AVAILABLE, reason='talib not installed in this environment')
    def test_talib_keys_present_when_available(self):
        feats, computable = hf.k1_features(
            _bars(), i0=1, entry=10.15, stop=10.00, target=10.45, level=10.20, atr14=0.20,
            break_bar_v=1000.0, arm_ctx=self.ARM_CTX, spy_bars=None, fill_min=571.0)
        assert computable is True
        assert 'talib_nbull_0' in feats and 'talib_nbear_0' in feats
        assert 'talib_nbull_1' in feats and 'talib_nbear_1' in feats
        assert isinstance(feats['talib_nbull_1'], int)


class TestOrderedFeatureList:
    def test_restricts_to_available_columns_and_preserves_group_order(self):
        available = ['r_pct', 'clv0', 'cS1_dist_level_1', 'clv_last_1', 'talib_nbull_1', 'unrelated_col']
        cols = hf.ordered_feature_list(available)
        assert cols == ['r_pct', 'clv0', 'cS1_dist_level_1', 'clv_last_1', 'talib_nbull_1']
