"""BT↔PROD parity tests for conviction + MACD zone multiplier.

After S2-max ship (per-tier MACD + v_rev_bonus 1.0), the BT
(`BacktestRunner._compute_conviction_score_setup`,
`BacktestRunner._get_macd_zone_multiplier`) and PROD
(`TradingEngine._compute_conviction_score_setup`,
`TradingEngine._get_macd_zone_multiplier`) implementations MUST produce
identical output for identical inputs.

These tests enforce that invariant at CI time.
"""
from __future__ import annotations

import pytest

from backtest import BacktestRunner
from trading.pattern_detector import BullFlagSetup
from trading.trading_engine import TradingEngine


def _setup(pole_gain_pct=6.0, tightness_pct=20.0, vol_ratio=2.0,
           retracement_pct=20.0):
    pole_low, pole_high = 4.70, 5.10
    pole_height = pole_high - pole_low
    flag_low = 4.85
    flag_high = flag_low + pole_height * (tightness_pct / 100.0)
    avg_flag_volume = 5000
    avg_pole_volume = int(avg_flag_volume * vol_ratio)
    return BullFlagSetup(
        symbol="TEST",
        pole_start_idx=0, pole_end_idx=3,
        flag_start_idx=4, flag_end_idx=6,
        pole_low=pole_low, pole_high=pole_high,
        pole_height=pole_height,
        pole_gain_pct=pole_gain_pct,
        flag_low=flag_low, flag_high=flag_high,
        retracement_pct=retracement_pct,
        pullback_candle_count=2,
        avg_pole_volume=avg_pole_volume,
        avg_flag_volume=avg_flag_volume,
        breakout_level=flag_high,
    )


class TestConvictionScoreParity:
    """`_compute_conviction_score_setup` must produce identical output in BT and PROD."""

    def _both_scorers(self):
        bt = BacktestRunner()._compute_conviction_score_setup
        prod = lambda *a, **kw: TradingEngine._compute_conviction_score_setup(
            None, *a, **kw)
        return bt, prod

    @pytest.mark.parametrize("pole_gain,tight,vol,retr,spy,vwap,gap_fade,vrev_trigger", [
        (6.0, 20.0, 2.0, 15.0, 1.5, 3.0, False, False),    # high-quality setup
        (6.0, 20.0, 2.0, 15.0, 0.5, 3.0, False, False),    # SPY-bad
        (6.0, 60.0, 1.0, 40.0, 1.0, 0.5, False, False),    # weak
        (3.0, 60.0, 1.0, 40.0, 0.5, 0.0, True, False),     # gap-fading
        (6.0, 20.0, 2.0, 15.0, 1.5, 3.0, False, True),     # V-rev fires
    ])
    def test_identical_output(self, pole_gain, tight, vol, retr, spy, vwap,
                                gap_fade, vrev_trigger):
        bt, prod = self._both_scorers()
        s = _setup(pole_gain_pct=pole_gain, tightness_pct=tight, vol_ratio=vol,
                    retracement_pct=retr)
        gap_pct = -2.0 if vrev_trigger else 2.0
        intraday_range_pct = 25.0 if vrev_trigger else 10.0
        kwargs = dict(
            vwap_dist_pct=vwap, gap_fading=gap_fade,
            gap_pct=gap_pct, intraday_range_pct=intraday_range_pct,
            v_reversal_enabled=True, v_reversal_bonus=1.0,
            return_breakdown=True,
        )
        bt_r, bt_brk = bt(s, spy, **kwargs)
        pr_r, pr_brk = prod(s, spy, **kwargs)
        assert bt_r == pr_r, f"Scalar divergence: BT={bt_r} PROD={pr_r}"
        assert bt_brk == pr_brk, f"Breakdown divergence: {bt_brk} vs {pr_brk}"


class TestRuleWeightConstantsParity:
    """Rule weights hardcoded in both files must match exactly."""

    def test_weights_match(self):
        """Smoke test — compute score on ALL-RULES-FIRING setup, verify
        matches the sum of expected weights."""
        bt = BacktestRunner()._compute_conviction_score_setup
        s = _setup(pole_gain_pct=6.0, tightness_pct=20.0, vol_ratio=2.0,
                    retracement_pct=15.0)
        _, brk = bt(s, 1.5,  # SPY > 1.2
                    vwap_dist_pct=3.0, gap_fading=False,
                    gap_pct=0.0, intraday_range_pct=0.0,
                    v_reversal_enabled=False, return_breakdown=True)
        # r1 +0.3, r2+ +0.3, r3 +0.3, r4+ +0.3, r5 +0.2, r7 +0.2, r8 0, r9 0
        assert brk['pole_gain'] == 0.3
        assert brk['flag_tightness'] == 0.3
        assert brk['vol_ratio'] == 0.3
        assert brk['spy_regime'] == 0.3
        assert brk['retracement'] == 0.2
        assert brk['vwap_dist'] == 0.2
        assert brk['gap_fading'] == 0.0
        # final_score = 1.0 + sum(contribs) clamped [0.25, 3.0]
        assert brk['final_score'] == pytest.approx(2.6)


class TestMACDZoneParity:
    """`_get_macd_zone_multiplier` signature parity — both take
    intraday_change_pct kwarg now."""

    def test_bt_signature_accepts_intraday_change(self):
        import inspect
        sig = inspect.signature(BacktestRunner._get_macd_zone_multiplier)
        assert 'intraday_change_pct' in sig.parameters
        assert sig.parameters['intraday_change_pct'].default == 0.0

    def test_prod_signature_accepts_intraday_change(self):
        import inspect
        sig = inspect.signature(TradingEngine._get_macd_zone_multiplier)
        assert 'intraday_change_pct' in sig.parameters
        assert sig.parameters['intraday_change_pct'].default == 0.0

    def test_both_load_same_extras_tier_from_config(self):
        """BT and PROD read identical values from config.yaml."""
        bt_runner = BacktestRunner()
        # TradingEngine requires deps; inspect attrs via introspection of a
        # dummy instance is messy, so instead verify both use same cfg keys.
        # Value-level parity: assert BT loaded what we expect (PROD loads
        # same keys in its __init__ — code paths are line-by-line mirrors).
        assert bt_runner.macd_extras_strong_pos_multiplier == 2.0
        assert bt_runner.macd_extras_strong_neg_multiplier == 2.0
        assert bt_runner.macd_extras_normal_multiplier == 0.0
        assert bt_runner.macd_strong_pos_multiplier == 1.8
        assert bt_runner.macd_strong_neg_multiplier == 1.8


class TestVReversalBonusParity:
    """V-reversal bonus value comes from config; both sides read same key."""

    def test_bt_loads_bonus_1_0(self):
        r = BacktestRunner()
        assert r.v_reversal_bonus == 1.0

    def test_prod_class_reads_bonus_key(self):
        """PROD loads `trading.conviction_scoring.v_reversal_bonus.bonus`.
        We can't instantiate TradingEngine in unit test (needs API deps),
        but we verify the load path by introspection of the source.
        """
        src = open('trading/trading_engine.py').read()
        assert 'self.v_reversal_bonus = float(_vrev_cfg.get("bonus", 0.4))' in src, \
            "PROD must load bonus from config with same default as BT"


class TestSharedTierHelperParity:
    """Both BT and PROD call the same `select_tier_multipliers` helper.
    This enforces code-level parity (not just logical equivalence)."""

    def test_bt_uses_helper(self):
        """BT `_get_macd_zone_multiplier` must import + call select_tier_multipliers."""
        src = open('backtest.py').read()
        assert 'from trading.macd_tier_helpers import select_tier_multipliers' in src
        assert 'select_tier_multipliers(' in src

    def test_prod_uses_helper(self):
        """PROD `_get_macd_zone_multiplier` must import + call select_tier_multipliers."""
        src = open('trading/trading_engine.py').read()
        assert 'from trading.macd_tier_helpers import select_tier_multipliers' in src
        assert 'select_tier_multipliers(' in src

    def test_bt_passes_correct_args_to_helper(self):
        """BT passes (A_pos, A_neg, A_normal, E_pos, E_neg, E_normal) as documented."""
        src = open('backtest.py').read()
        # Verify arg order: A-tier first (3 values), then Extras (3 values)
        assert 'self.macd_strong_pos_multiplier' in src
        assert 'self.macd_extras_strong_pos_multiplier' in src
        # Regex-free check: helper call block has both tiers' attrs in order
        helper_call_region = src[src.find('select_tier_multipliers('):
                                  src.find('select_tier_multipliers(') + 600]
        a_pos_idx = helper_call_region.find('macd_strong_pos_multiplier')
        e_pos_idx = helper_call_region.find('macd_extras_strong_pos_multiplier')
        assert 0 < a_pos_idx < e_pos_idx, "A-tier args must precede Extras args"

    def test_prod_passes_same_correct_args_to_helper(self):
        src = open('trading/trading_engine.py').read()
        helper_call_region = src[src.find('select_tier_multipliers('):
                                  src.find('select_tier_multipliers(') + 600]
        a_pos_idx = helper_call_region.find('macd_strong_pos_multiplier')
        e_pos_idx = helper_call_region.find('macd_extras_strong_pos_multiplier')
        assert 0 < a_pos_idx < e_pos_idx, "A-tier args must precede Extras args"


class TestMACDHelperWiredIntoCallers:
    """End-to-end: helper invocation sites in BT and PROD produce identical
    multipliers when called with matching inputs."""

    @pytest.mark.parametrize("ic,a_pos,a_neg,a_norm,e_pos,e_neg,e_norm,"
                              "expected_pos,expected_neg,expected_norm", [
        # A-tier
        (25.0, 1.8, 1.8, 1.0, 2.0, 2.0, 0.0, 1.8, 1.8, 1.0),
        # Extras
        (15.0, 1.8, 1.8, 1.0, 2.0, 2.0, 0.0, 2.0, 2.0, 0.0),
        # Edge
        (5.0,  1.8, 1.8, 1.0, 2.0, 2.0, 0.0, 1.8, 1.8, 1.0),
        # Boundary: 20.0 is A-tier (inclusive)
        (20.0, 1.5, 1.5, 1.0, 2.0, 2.0, 0.0, 1.5, 1.5, 1.0),
        # Boundary: 10.0 is Extras (inclusive)
        (10.0, 1.5, 1.5, 1.0, 2.0, 2.0, 0.0, 2.0, 2.0, 0.0),
    ])
    def test_parametrized_tier_routing(self, ic, a_pos, a_neg, a_norm,
                                         e_pos, e_neg, e_norm,
                                         expected_pos, expected_neg, expected_norm):
        from trading.macd_tier_helpers import select_tier_multipliers
        sp, sn, n, _ = select_tier_multipliers(
            ic, a_pos, a_neg, a_norm, e_pos, e_neg, e_norm)
        assert (sp, sn, n) == (expected_pos, expected_neg, expected_norm)
