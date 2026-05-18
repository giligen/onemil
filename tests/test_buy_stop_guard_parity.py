"""
Parity tests: bull flag and ORB use the SAME buy-stop guard decision.

Mirrors the pattern in tests/test_orb_touchgo_parity.py and
tests/test_regime_sizing_parity.py — source-code inspection asserting both
strategies import the shared helper from `trading.buy_stop_guard`, plus
behavioral assertions that the decision tree fires identically.

Why parity matters here: the rejection-avoidance logic determines whether an
order is submitted at all and on which path. Drift between the two strategies
would re-introduce the very bug we're closing (BF had marketable-limit
fallback; ORB didn't → ORB BTCZ/YSS/BMNZ rejected on 2026-05-18).
"""

from pathlib import Path

from trading.buy_stop_guard import (
    BuyStopAction,
    evaluate_buy_stop,
)


ROOT = Path(__file__).resolve().parent.parent


class TestSourceImportsSharedHelper:
    """Both callers must import the decision function from the shared module."""

    def _read(self, relpath: str) -> str:
        return (ROOT / relpath).read_text()

    def test_order_executor_imports_evaluate_buy_stop(self):
        src = self._read("trading/order_executor.py")
        assert "from trading.buy_stop_guard import" in src, (
            "order_executor.py must import from trading.buy_stop_guard"
        )
        assert "evaluate_buy_stop" in src, (
            "order_executor.py must call evaluate_buy_stop"
        )
        assert "BuyStopAction" in src, (
            "order_executor.py must dispatch on BuyStopAction"
        )

    def test_orb_engine_imports_evaluate_buy_stop(self):
        src = self._read("trading/orb_engine.py")
        assert "from trading.buy_stop_guard import" in src, (
            "orb_engine.py must import from trading.buy_stop_guard"
        )
        assert "evaluate_buy_stop" in src, (
            "orb_engine.py must call evaluate_buy_stop"
        )
        assert "BuyStopAction" in src, (
            "orb_engine.py must dispatch on BuyStopAction"
        )

    def test_no_duplicate_inline_logic_in_order_executor(self):
        """The 3-branch inline logic was the old shape — make sure it didn't
        regrow here. If a future change reintroduces inline bid/ask checks
        outside the helper, this test will fail and force a refactor."""
        src = self._read("trading/order_executor.py")
        # Tolerate the import + dispatch lines. The OLD inline shape had
        # comparisons like `_bid >= stop_price` and `_ask >= stop_price`
        # inside submit_buy_stop_order. Those are now inside the helper.
        # We don't want to see those exact comparisons re-emerge in the
        # caller code.
        bad = [
            "_bid >= stop_price",
            "_ask >= stop_price",
            "_new_stop > limit_price",
        ]
        for needle in bad:
            assert needle not in src, (
                f"Inline guard logic '{needle}' has regrown in "
                f"order_executor.py — should live only in buy_stop_guard.py"
            )


class TestDecisionParityBetweenStrategies:
    """Same (bid, ask, stop, limit) inputs must produce the same decision
    regardless of which strategy is the caller.

    This is trivially true since both call the same pure function — these
    tests pin that contract so any future refactor that diverges (e.g.,
    adds a strategy-specific override) is caught.
    """

    SCENARIOS = [
        # (label, bid, ask, stop, limit, expected_action)
        ("kpti_straddle",   9.56, 9.79,  9.71, 9.90, BuyStopAction.REBUMP_STOP),
        ("trt_straddle",   13.09, 13.30, 13.27, 13.54, BuyStopAction.REBUMP_STOP),
        ("btcz_marketable", 4.11,  4.12,  4.09, 4.10, BuyStopAction.MARKETABLE_LIMIT),
        ("btcz_skip",       4.08,  4.15,  4.09, 4.10, BuyStopAction.SKIP),
        ("yss_skip",       26.35, 26.55, 26.44, 26.52, BuyStopAction.SKIP),
        ("bmnz_marketable",18.52, 18.53, 18.48, 18.54, BuyStopAction.MARKETABLE_LIMIT),
        ("frmm_normal",     3.75,  3.76,  4.07, 4.08, BuyStopAction.SUBMIT_AS_IS),
        ("normal_below",    4.30,  4.38,  4.40, 4.49, BuyStopAction.SUBMIT_AS_IS),
        ("marketable",      4.45,  4.46,  4.40, 4.49, BuyStopAction.MARKETABLE_LIMIT),
        ("rebump",          4.35,  4.45,  4.40, 4.49, BuyStopAction.REBUMP_STOP),
        ("extended",        4.35,  4.48,  4.40, 4.49, BuyStopAction.SKIP),
    ]

    def test_each_scenario_resolves_to_expected_action(self):
        """Scenario coverage: every (bid, ask, stop, limit) tuple maps to
        the documented action. Drift in either caller would not affect this
        — only changes to evaluate_buy_stop itself would."""
        for label, bid, ask, stop, limit, expected in self.SCENARIOS:
            d = evaluate_buy_stop(bid=bid, ask=ask, stop_price=stop,
                                  limit_price=limit)
            assert d.action == expected, (
                f"{label}: bid={bid} ask={ask} stop={stop} limit={limit} → "
                f"expected {expected}, got {d.action}"
            )

    def test_rebump_produces_consistent_new_stop(self):
        """The REBUMP_STOP action's new_stop must equal round(ask + buffer, 2)
        and stay within (ask, limit_price]."""
        d = evaluate_buy_stop(bid=4.35, ask=4.45, stop_price=4.40, limit_price=4.49)
        assert d.action == BuyStopAction.REBUMP_STOP
        assert d.new_stop_price == 4.47
        assert d.new_stop_price > 4.45        # strictly above ask
        assert d.new_stop_price <= 4.49       # at most limit


class TestConfigShared:
    """Both callers must read from the same config property."""

    def test_both_callers_reference_marketable_limit_fallback_cfg(self):
        oe_src = (ROOT / "trading/order_executor.py").read_text()
        orb_src = (ROOT / "trading/orb_engine.py").read_text()
        assert "marketable_limit_fallback_cfg" in oe_src
        assert "marketable_limit_fallback_cfg" in orb_src
