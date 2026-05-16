"""BT/LIVE parity for ORB touchgo filter.

Enforces that both backtest (study_orb_pipeline_static_lock.py) and live
(trading/orb_engine.py) import the rule functions from the same shared
module — preventing accidental forks. Mirrors test_bt_prod_parity.py +
test_regime_sizing_parity.py patterns.
"""
from pathlib import Path

import pytest

from trading.orb_touchgo_filter import (
    TouchgoConfig, evaluate_rule_m, evaluate_rule_d, load_touchgo_config,
)


REPO = Path(__file__).parent.parent


# =========================================================================
# Source-code parity: both BT and LIVE must import from the shared module.
# =========================================================================

def test_bt_imports_shared_helper():
    """study_orb_pipeline_static_lock.py must import from trading.orb_touchgo_filter."""
    src = (REPO / 'study_orb_pipeline_static_lock.py').read_text()
    assert 'from trading.orb_touchgo_filter import' in src, (
        "BT (study_orb_pipeline_static_lock.py) must import "
        "trading.orb_touchgo_filter for parity"
    )
    # Must use the helper functions, not its own implementation
    assert 'evaluate_rule_m' in src
    assert 'evaluate_rule_d' in src
    assert 'load_touchgo_config' in src


def test_live_imports_shared_helper():
    """trading/orb_engine.py must import from trading.orb_touchgo_filter."""
    src = (REPO / 'trading' / 'orb_engine.py').read_text()
    assert 'from trading.orb_touchgo_filter import' in src, (
        "LIVE (trading/orb_engine.py) must import "
        "trading.orb_touchgo_filter for parity"
    )
    assert 'evaluate_rule_m' in src
    assert 'evaluate_rule_d' in src
    assert 'load_touchgo_config' in src


def test_no_fork_in_bt():
    """Confirm BT doesn't redefine its own rule functions."""
    src = (REPO / 'study_orb_pipeline_static_lock.py').read_text()
    # The shared module owns these. BT should NOT define them.
    assert 'def evaluate_rule_m(' not in src
    assert 'def evaluate_rule_d(' not in src


def test_no_fork_in_live():
    """Confirm LIVE doesn't redefine its own rule functions."""
    src = (REPO / 'trading' / 'orb_engine.py').read_text()
    assert 'def evaluate_rule_m(' not in src
    assert 'def evaluate_rule_d(' not in src


# =========================================================================
# Scalar equivalence — same inputs produce identical outputs.
# (Trivially true since both call the same function; this is a smoke test
# of the wiring path.)
# =========================================================================

class TestScalarEquivalence:
    @pytest.fixture
    def cfg(self):
        return load_touchgo_config({})

    def test_rule_m_identical_across_callers(self, cfg):
        """Build the exact same call args used by BT and LIVE and confirm result."""
        # BT call shape (from simulate_static_lock):
        #   evaluate_rule_m(float(bar.open), float(bar.high), float(bar.low),
        #                   float(bar.close), TOUCHGO_CFG)
        # LIVE call shape (from _evaluate_touchgo):
        #   evaluate_rule_m(float(bb_bar['open']), float(bb_bar['high']),
        #                   float(bb_bar['low']), float(bb_bar['close']),
        #                   self.touchgo_cfg)
        # Both pass through to the same function with the same dtype.
        bt_result = evaluate_rule_m(9.95, 10.10, 9.60, 9.70, cfg)
        live_result = evaluate_rule_m(9.95, 10.10, 9.60, 9.70, cfg)
        assert bt_result == live_result
        assert bt_result == (True, pytest.approx(9.70))

    def test_rule_d_identical_across_callers(self, cfg):
        # BT: evaluate_rule_d(entry_price, float(b1.low), range_size, cfg)
        # LIVE: evaluate_rule_d(pos.entry_price, float(b1_bar['low']),
        #                       range_size, self.touchgo_cfg)
        bt_result = evaluate_rule_d(10.03, 9.20, 1.0, cfg)
        live_result = evaluate_rule_d(10.03, 9.20, 1.0, cfg)
        assert bt_result == live_result


# =========================================================================
# Defaults sanity (catches accidental threshold drift)
# =========================================================================

class TestDefaultsLocked:
    """Walk-forward validated values. Changing without re-running the
    walk-forward analysis WILL break the BT-validated lift.
    """

    def test_rule_m_threshold_locked_at_0_5(self):
        cfg = load_touchgo_config({})
        assert cfg.rule_m_threshold == 0.5, (
            "Rule M threshold defaults to 0.5 per Jan'25-May'26 walk-forward. "
            "Re-validate before changing."
        )

    def test_rule_d_revert_R_locked_at_0_75(self):
        cfg = load_touchgo_config({})
        assert cfg.rule_d_revert_R == 0.75

    def test_rule_d_exit_R_locked_at_neg_0_5(self):
        cfg = load_touchgo_config({})
        assert cfg.rule_d_exit_R == -0.5

    def test_master_enabled_by_default(self):
        cfg = load_touchgo_config({})
        assert cfg.master_enabled is True

    def test_both_rules_enabled_by_default(self):
        cfg = load_touchgo_config({})
        assert cfg.rule_m_enabled is True
        assert cfg.rule_d_enabled is True


# =========================================================================
# Exit-reason whitelist (StopMonitor.force_exit accepts tag_bb + tag_b1)
# =========================================================================

def test_stop_monitor_whitelists_touchgo_reasons():
    """StopMonitor.force_exit must accept 'tag_bb' and 'tag_b1' as valid
    exit reasons for the engine-routed exit path.
    """
    from trading.stop_monitor import StopMonitor
    assert 'tag_bb' in StopMonitor._FORCE_EXIT_REASON_WHITELIST
    assert 'tag_b1' in StopMonitor._FORCE_EXIT_REASON_WHITELIST
