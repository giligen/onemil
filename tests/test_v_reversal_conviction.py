"""Unit tests for V-reversal conviction bonus (Rule 9) + marginal scaling.

Both features tested at the conviction-function level plus BT↔PROD parity.
"""
from __future__ import annotations

import pytest

from backtest import BacktestRunner
from trading.pattern_detector import BullFlagSetup
from trading.trading_engine import TradingEngine


def _setup(pole_gain_pct=6.0):
    """Minimal BullFlagSetup stub for conviction scoring (ignores bar indices)."""
    return BullFlagSetup(
        symbol="TEST",
        pole_start_idx=0, pole_end_idx=3,
        flag_start_idx=4, flag_end_idx=6,
        pole_low=4.70, pole_high=5.10,
        pole_height=0.40,
        pole_gain_pct=pole_gain_pct,
        flag_low=4.85, flag_high=4.95,
        retracement_pct=30.0,
        pullback_candle_count=2,
        avg_pole_volume=10000,
        avg_flag_volume=5000,
        breakout_level=4.95,
    )


def _call(scorer, setup, *, v_enabled=False, v_bonus=0.4,
          v_gap_max=0.0, v_range_min=20.0, v_pole_min=5.0,
          gap_pct=0.0, intraday_range_pct=0.0):
    """Call conviction score and return the breakdown dict."""
    _, brk = scorer(
        setup, 1.0,
        vwap_dist_pct=0.0, gap_fading=False,
        gap_pct=gap_pct, intraday_range_pct=intraday_range_pct,
        v_reversal_enabled=v_enabled,
        v_reversal_bonus=v_bonus,
        v_reversal_gap_pct_max=v_gap_max,
        v_reversal_intraday_range_min=v_range_min,
        v_reversal_pole_gain_min=v_pole_min,
        return_breakdown=True,
    )
    return brk


class TestVReversalRule:
    """Rule 9: adds V-reversal bonus when all triggers fire + flag on."""

    @pytest.fixture
    def bt_scorer(self):
        return BacktestRunner()._compute_conviction_score_setup

    def test_disabled_never_fires(self, bt_scorer):
        """enabled=False → no V-reversal bonus, even with triggers satisfied."""
        brk = _call(bt_scorer, _setup(pole_gain_pct=6.0),
                    v_enabled=False,
                    gap_pct=-2.0, intraday_range_pct=25.0)
        assert brk['v_reversal'] == 0.0

    def test_enabled_all_triggers_fires(self, bt_scorer):
        """gap<0 AND range>=20 AND pole>=5 → +bonus."""
        brk = _call(bt_scorer, _setup(pole_gain_pct=6.0),
                    v_enabled=True, v_bonus=0.4,
                    gap_pct=-2.0, intraday_range_pct=25.0)
        assert brk['v_reversal'] == pytest.approx(0.4)

    def test_gap_positive_does_not_fire(self, bt_scorer):
        """Gap-up (gap >= 0) → rule does not apply, even with other triggers."""
        brk = _call(bt_scorer, _setup(pole_gain_pct=6.0),
                    v_enabled=True,
                    gap_pct=+2.0, intraday_range_pct=25.0)
        assert brk['v_reversal'] == 0.0

    def test_gap_zero_does_not_fire(self, bt_scorer):
        """gap_pct == gap_pct_max (strict less-than required)."""
        brk = _call(bt_scorer, _setup(pole_gain_pct=6.0),
                    v_enabled=True,
                    gap_pct=0.0, intraday_range_pct=25.0)
        assert brk['v_reversal'] == 0.0

    def test_range_too_low_does_not_fire(self, bt_scorer):
        """Intraday range < threshold → no bonus."""
        brk = _call(bt_scorer, _setup(pole_gain_pct=6.0),
                    v_enabled=True,
                    gap_pct=-2.0, intraday_range_pct=15.0)
        assert brk['v_reversal'] == 0.0

    def test_pole_too_small_does_not_fire(self, bt_scorer):
        """Pole gain < threshold → no bonus."""
        brk = _call(bt_scorer, _setup(pole_gain_pct=3.0),
                    v_enabled=True,
                    gap_pct=-2.0, intraday_range_pct=25.0)
        assert brk['v_reversal'] == 0.0

    def test_custom_thresholds_respected(self, bt_scorer):
        """Caller-supplied thresholds are used, not hardcoded."""
        # Set tight thresholds: must be gap<-1, range>=30, pole>=7
        brk = _call(bt_scorer, _setup(pole_gain_pct=6.0),
                    v_enabled=True,
                    v_gap_max=-1.0, v_range_min=30.0, v_pole_min=7.0,
                    gap_pct=-2.0, intraday_range_pct=25.0)
        assert brk['v_reversal'] == 0.0  # range 25 < threshold 30

    def test_custom_bonus_size_used(self, bt_scorer):
        """bonus kwarg controls the contribution magnitude."""
        brk = _call(bt_scorer, _setup(pole_gain_pct=6.0),
                    v_enabled=True, v_bonus=0.6,
                    gap_pct=-2.0, intraday_range_pct=25.0)
        assert brk['v_reversal'] == pytest.approx(0.6)

    def test_final_score_clamped_to_3(self, bt_scorer):
        """Max raw score + V-rev bonus still clamped to 3.0."""
        # Setup that maxes many other rules:
        # pole_gain sweet spot 6% (+0.3), tight flag (+0.3), vol ratio (need setup mod)
        # Use max positive rules by tweaking setup + inputs.
        s = _setup(pole_gain_pct=6.0)
        s.avg_pole_volume = 20000  # vol_ratio 4 > 1.7 → +0.3
        _, brk = BacktestRunner()._compute_conviction_score_setup(
            s, spy_3d_range=1.5,  # +0.3 SPY regime
            vwap_dist_pct=3.0,    # +0.2 vwap
            gap_fading=False,
            gap_pct=-2.0, intraday_range_pct=25.0,
            v_reversal_enabled=True, v_reversal_bonus=0.6,  # big bonus
            return_breakdown=True,
        )
        # Raw score: 1.0 + 0.3 + 0.3 + 0.3 + 0.3 + 0.2 + 0.2 + 0 + 0.6 = 3.2
        assert brk['raw_score'] > 3.0
        assert brk['final_score'] == 3.0  # clamped


class TestBTPRODParity:
    """BT + PROD conviction functions must produce identical output."""

    def test_identical_inputs_identical_output(self):
        bt_score = BacktestRunner()._compute_conviction_score_setup
        # PROD engine method is pure — can call unbound on None.
        prod_score = lambda *a, **kw: TradingEngine._compute_conviction_score_setup(
            None, *a, **kw)

        s = _setup(pole_gain_pct=6.0)
        kwargs = dict(
            vwap_dist_pct=3.0, gap_fading=False,
            gap_pct=-2.0, intraday_range_pct=25.0,
            v_reversal_enabled=True, v_reversal_bonus=0.4,
            return_breakdown=True,
        )
        bt_r, bt_brk = bt_score(s, 1.2, **kwargs)
        prod_r, prod_brk = prod_score(s, 1.2, **kwargs)
        assert bt_r == prod_r, f"BT={bt_r} PROD={prod_r}"
        assert bt_brk == prod_brk, f"BT={bt_brk} PROD={prod_brk}"

    def test_parity_v_disabled(self):
        """Flag OFF: both return identical scores without V-rev contribution."""
        bt_score = BacktestRunner()._compute_conviction_score_setup
        prod_score = lambda *a, **kw: TradingEngine._compute_conviction_score_setup(
            None, *a, **kw)

        s = _setup(pole_gain_pct=6.0)
        kwargs = dict(
            vwap_dist_pct=0.0, gap_fading=False,
            gap_pct=-2.0, intraday_range_pct=25.0,
            v_reversal_enabled=False,
            return_breakdown=True,
        )
        _, bt_brk = bt_score(s, 1.0, **kwargs)
        _, prod_brk = prod_score(s, 1.0, **kwargs)
        assert bt_brk['v_reversal'] == 0.0
        assert prod_brk['v_reversal'] == 0.0
        assert bt_brk == prod_brk


class TestMarginalScalingLoadsFromConfig:
    """BT and PROD both surface marginal-scaling state from config."""

    def test_disabled_by_default_bt(self):
        """Default config has marginal_scaling disabled → factor == 1.0."""
        runner = BacktestRunner()
        # scale_factor exposed as 1.0 when disabled (no-op downstream)
        assert runner.conviction_marginal_scale_factor == 1.0

    def test_disabled_by_default_config(self):
        """Config accessor returns effective factor=1.0 when disabled."""
        from config import Config
        cfg = Config().conviction_marginal_scaling_cfg
        # Even if YAML has scale_factor=0.5, enabled=false forces 1.0.
        assert cfg['scale_factor'] == 1.0


class TestConvictionConfigAccessors:
    """Config properties surface both features cleanly."""

    def test_v_reversal_cfg_has_all_keys(self):
        from config import Config
        cfg = Config().v_reversal_bonus_cfg
        for key in ('enabled', 'bonus', 'gap_pct_max',
                    'intraday_range_min', 'pole_gain_min'):
            assert key in cfg, f"missing {key}"

    def test_marginal_scaling_cfg_has_all_keys(self):
        from config import Config
        cfg = Config().conviction_marginal_scaling_cfg
        for key in ('enabled', 'scale_factor', 'upper_bound'):
            assert key in cfg, f"missing {key}"
