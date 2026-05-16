"""Unit tests for trading.orb_touchgo_filter.

Mirrors test_two_tier_filter.py / test_regime_helpers.py patterns.
"""
import os
import pytest

from trading.orb_touchgo_filter import (
    TouchgoConfig,
    evaluate_rule_m,
    evaluate_rule_d,
    load_touchgo_config,
)


# Default cfg fixture
@pytest.fixture
def cfg() -> TouchgoConfig:
    return TouchgoConfig()


@pytest.fixture
def cfg_disabled() -> TouchgoConfig:
    return TouchgoConfig(master_enabled=False)


@pytest.fixture
def cfg_m_off() -> TouchgoConfig:
    return TouchgoConfig(rule_m_enabled=False)


@pytest.fixture
def cfg_d_off() -> TouchgoConfig:
    return TouchgoConfig(rule_d_enabled=False)


# ============================================================
# Rule M tests
# ============================================================
class TestEvaluateRuleM:
    def test_fires_when_close_in_bottom_third(self, cfg):
        # Bar: O=10, H=10.5, L=9.5, C=9.7 -> close_pos = (9.7-9.5)/1.0 = 0.2
        fire, exit_p = evaluate_rule_m(10.0, 10.5, 9.5, 9.7, cfg)
        assert fire is True
        assert exit_p == pytest.approx(9.7)

    def test_fires_at_threshold_boundary(self, cfg):
        # close_pos = 0.499 (just under threshold 0.5) -> fires (strict <)
        # H=10.0, L=9.0, range=1.0. close = 9.499 -> pos = 0.499
        fire, exit_p = evaluate_rule_m(9.5, 10.0, 9.0, 9.499, cfg)
        assert fire is True
        assert exit_p == pytest.approx(9.499)

    def test_no_fire_at_exact_threshold(self, cfg):
        # close_pos = 0.5 exactly -> does NOT fire (strict <)
        fire, exit_p = evaluate_rule_m(9.5, 10.0, 9.0, 9.5, cfg)
        assert fire is False
        assert exit_p is None

    def test_no_fire_when_close_in_top_half(self, cfg):
        # close_pos = 0.6
        fire, exit_p = evaluate_rule_m(9.5, 10.0, 9.0, 9.6, cfg)
        assert fire is False
        assert exit_p is None

    def test_no_fire_when_rule_m_disabled(self, cfg_m_off):
        # Even with strong trigger, disabled -> no fire
        fire, exit_p = evaluate_rule_m(10.0, 10.5, 9.5, 9.7, cfg_m_off)
        assert fire is False
        assert exit_p is None

    def test_no_fire_when_master_disabled(self, cfg_disabled):
        fire, exit_p = evaluate_rule_m(10.0, 10.5, 9.5, 9.7, cfg_disabled)
        assert fire is False
        assert exit_p is None

    def test_safe_default_on_degenerate_bar(self, cfg):
        # high == low (single tick) -> safe default
        fire, exit_p = evaluate_rule_m(10.0, 10.0, 10.0, 10.0, cfg)
        assert fire is False
        assert exit_p is None

    def test_safe_default_on_inverted_bar(self, cfg):
        # high < low (corrupt data) -> safe default
        fire, exit_p = evaluate_rule_m(10.0, 9.5, 10.5, 10.0, cfg)
        assert fire is False
        assert exit_p is None

    def test_custom_threshold_lower(self):
        # Threshold 0.3 -> close_pos=0.4 should NOT fire (0.4 not < 0.3)
        cfg_tight = TouchgoConfig(rule_m_threshold=0.3)
        fire, _ = evaluate_rule_m(9.5, 10.0, 9.0, 9.4, cfg_tight)
        assert fire is False

    def test_custom_threshold_higher(self):
        cfg_loose = TouchgoConfig(rule_m_threshold=0.7)
        # close_pos = 0.6 -> with threshold 0.7, fires
        fire, exit_p = evaluate_rule_m(9.5, 10.0, 9.0, 9.6, cfg_loose)
        assert fire is True
        assert exit_p == pytest.approx(9.6)

    def test_exit_price_equals_bb_close(self, cfg):
        # Confirms exit_price is exactly bb_close (caller adds slippage)
        fire, exit_p = evaluate_rule_m(10.0, 10.5, 9.5, 9.55, cfg)
        assert fire is True
        assert exit_p == pytest.approx(9.55)


# ============================================================
# Rule D tests
# ============================================================
class TestEvaluateRuleD:
    def test_fires_when_revert_above_threshold(self, cfg):
        # entry=10, b1_low=9.2, range=1.0 -> revert = 0.8R >= 0.75
        fire, exit_p = evaluate_rule_d(10.0, 9.2, 1.0, cfg)
        assert fire is True
        # exit = entry + (-0.5) * range = 10.0 - 0.5 = 9.5
        assert exit_p == pytest.approx(9.5)

    def test_fires_at_threshold_boundary(self, cfg):
        # revert = 0.75R exactly -> fires (>=)
        fire, exit_p = evaluate_rule_d(10.0, 9.25, 1.0, cfg)
        assert fire is True
        assert exit_p == pytest.approx(9.5)

    def test_no_fire_when_revert_below_threshold(self, cfg):
        # revert = 0.5R
        fire, exit_p = evaluate_rule_d(10.0, 9.5, 1.0, cfg)
        assert fire is False
        assert exit_p is None

    def test_no_fire_when_above_entry(self, cfg):
        # b1_low above entry -> revert is negative -> no fire
        fire, exit_p = evaluate_rule_d(10.0, 10.5, 1.0, cfg)
        assert fire is False
        assert exit_p is None

    def test_no_fire_when_rule_d_disabled(self, cfg_d_off):
        fire, exit_p = evaluate_rule_d(10.0, 9.2, 1.0, cfg_d_off)
        assert fire is False
        assert exit_p is None

    def test_no_fire_when_master_disabled(self, cfg_disabled):
        fire, exit_p = evaluate_rule_d(10.0, 9.2, 1.0, cfg_disabled)
        assert fire is False
        assert exit_p is None

    def test_safe_default_on_zero_range(self, cfg):
        fire, exit_p = evaluate_rule_d(10.0, 9.2, 0.0, cfg)
        assert fire is False
        assert exit_p is None

    def test_safe_default_on_negative_range(self, cfg):
        fire, exit_p = evaluate_rule_d(10.0, 9.2, -1.0, cfg)
        assert fire is False
        assert exit_p is None

    def test_exit_price_at_negative_half_R(self, cfg):
        # default rule_d_exit_R = -0.5
        fire, exit_p = evaluate_rule_d(20.0, 18.5, 2.0, cfg)
        assert fire is True
        # exit = 20.0 + (-0.5) * 2.0 = 19.0
        assert exit_p == pytest.approx(19.0)

    def test_custom_exit_R_applied(self):
        # rule_d_exit_R = -0.3 (less aggressive exit, smaller loss)
        cfg_tight = TouchgoConfig(rule_d_exit_R=-0.3)
        fire, exit_p = evaluate_rule_d(10.0, 9.2, 1.0, cfg_tight)
        assert fire is True
        assert exit_p == pytest.approx(9.7)

    def test_custom_revert_R_threshold(self):
        # Lower threshold catches more trades
        cfg_aggressive = TouchgoConfig(rule_d_revert_R=0.5)
        # revert = 0.6R, doesn't fire with default 0.75 but fires with 0.5
        fire_default, _ = evaluate_rule_d(10.0, 9.4, 1.0, TouchgoConfig())
        fire_aggr, _ = evaluate_rule_d(10.0, 9.4, 1.0, cfg_aggressive)
        assert fire_default is False
        assert fire_aggr is True


# ============================================================
# load_touchgo_config tests
# ============================================================
class TestLoadTouchgoConfig:
    def setup_method(self):
        # Clean env vars before each test
        for k in [
            'ORB_TOUCHGO_ENABLED',
            'ORB_TOUCHGO_RULE_M_ENABLED',
            'ORB_TOUCHGO_RULE_M_THRESH',
            'ORB_TOUCHGO_RULE_D_ENABLED',
            'ORB_TOUCHGO_RULE_D_R',
            'ORB_TOUCHGO_RULE_D_EXIT_R',
        ]:
            os.environ.pop(k, None)

    def teardown_method(self):
        self.setup_method()

    def test_empty_dict_uses_defaults(self):
        cfg = load_touchgo_config({})
        assert cfg.master_enabled is True
        assert cfg.rule_m_enabled is True
        assert cfg.rule_m_threshold == 0.5
        assert cfg.rule_d_enabled is True
        assert cfg.rule_d_revert_R == 0.75
        assert cfg.rule_d_exit_R == -0.5

    def test_none_uses_defaults(self):
        cfg = load_touchgo_config(None)
        assert cfg.master_enabled is True
        assert cfg.rule_m_threshold == 0.5

    def test_full_yaml_dict_parsed(self):
        yaml_dict = {
            'enabled': True,
            'rule_m': {'enabled': True, 'threshold': 0.4},
            'rule_d': {'enabled': False, 'revert_R': 0.5, 'exit_R': -0.3},
        }
        cfg = load_touchgo_config(yaml_dict)
        assert cfg.master_enabled is True
        assert cfg.rule_m_threshold == 0.4
        assert cfg.rule_d_enabled is False
        assert cfg.rule_d_revert_R == 0.5
        assert cfg.rule_d_exit_R == -0.3

    def test_master_disable_from_yaml(self):
        cfg = load_touchgo_config({'enabled': False})
        assert cfg.master_enabled is False

    def test_partial_overrides_apply(self):
        # Only one rule's threshold specified; others default
        cfg = load_touchgo_config({'rule_m': {'threshold': 0.3}})
        assert cfg.rule_m_threshold == 0.3
        assert cfg.rule_d_revert_R == 0.75  # default
        assert cfg.rule_d_exit_R == -0.5  # default

    def test_env_var_master_disable(self):
        os.environ['ORB_TOUCHGO_ENABLED'] = '0'
        cfg = load_touchgo_config({})
        assert cfg.master_enabled is False

    def test_env_var_master_disable_false(self):
        os.environ['ORB_TOUCHGO_ENABLED'] = 'false'
        cfg = load_touchgo_config({})
        assert cfg.master_enabled is False

    def test_env_var_threshold_override(self):
        os.environ['ORB_TOUCHGO_RULE_M_THRESH'] = '0.4'
        cfg = load_touchgo_config({})
        assert cfg.rule_m_threshold == 0.4

    def test_env_var_d_revert_override(self):
        os.environ['ORB_TOUCHGO_RULE_D_R'] = '0.6'
        cfg = load_touchgo_config({})
        assert cfg.rule_d_revert_R == 0.6

    def test_env_var_overrides_yaml(self):
        os.environ['ORB_TOUCHGO_RULE_M_THRESH'] = '0.4'
        cfg = load_touchgo_config({'rule_m': {'threshold': 0.6}})
        # Env wins
        assert cfg.rule_m_threshold == 0.4

    def test_invalid_env_var_falls_back(self):
        os.environ['ORB_TOUCHGO_RULE_M_THRESH'] = 'not_a_float'
        cfg = load_touchgo_config({})
        # Should fall back to default rather than raise
        assert cfg.rule_m_threshold == 0.5

    def test_malformed_dict_uses_defaults(self):
        cfg = load_touchgo_config({'rule_m': 'not_a_dict'})
        assert cfg.rule_m_threshold == 0.5


# ============================================================
# Default thresholds match validated values
# ============================================================
class TestDefaultsMatchValidated:
    """These are the values the walk-forward analysis validated.

    DO NOT change without re-running walk-forward.
    """
    def test_rule_m_threshold_is_50pct(self):
        cfg = TouchgoConfig()
        assert cfg.rule_m_threshold == 0.5

    def test_rule_d_revert_is_075R(self):
        cfg = TouchgoConfig()
        assert cfg.rule_d_revert_R == 0.75

    def test_rule_d_exit_is_negative_half_R(self):
        cfg = TouchgoConfig()
        assert cfg.rule_d_exit_R == -0.5

    def test_all_enabled_by_default(self):
        cfg = TouchgoConfig()
        assert cfg.master_enabled is True
        assert cfg.rule_m_enabled is True
        assert cfg.rule_d_enabled is True
