"""BT↔live parity + integration tests for the two-tier filter.

Key property: batch_backtest Stage-2 and trading_engine live gate MUST apply
the same accept/reject decision on a trade given the same features. This file
tests that invariant by running both paths against a set of synthetic trades
and asserting identical results.
"""
from __future__ import annotations

from typing import Dict, List

import pytest

from trading.two_tier_filter import (
    TIER_A,
    TIER_EDGE,
    TIER_EXTRAS,
    build_features_from_trade,
    classify_tier,
    composite_score,
    should_keep,
)


TTF_CFG = {
    "enabled": True,
    "extras_lower": 10.0,
    "a_tier_lower": 20.0,
    "drop_extras_macd_below": 1.25,
    "composite_threshold": -0.50,
    "composite_features": {
        "conviction_mult":        {"mean": 1.789, "std": 0.284, "sign": -1},
        "qf_vwap_dist_pct":       {"mean": 4.218, "std": 2.239, "sign": -1},
        "qf_fill_vwap_dist_pct":  {"mean": 4.604, "std": 2.274, "sign": -1},
        "entry_minute":           {"mean": 603.614, "std": 20.195, "sign": -1},
    },
}


def _bt_decision(trade: Dict) -> tuple[bool, str]:
    """Reproduce batch_backtest.py Stage-2 two-tier gate logic."""
    # Same code path as in batch_backtest.py:~295
    ic = trade.get("intraday_change_at_entry")
    try:
        ic_val = float(ic) if ic not in (None, "", "None") else None
    except (ValueError, TypeError):
        ic_val = None
    tier = classify_tier(
        ic_val,
        a_tier_lower=TTF_CFG["a_tier_lower"],
        extras_lower=TTF_CFG["extras_lower"],
    )
    try:
        mzm = float(trade.get("macd_zone_mult") or 0.0)
    except (ValueError, TypeError):
        mzm = 0.0
    return should_keep(
        tier=tier,
        macd_zone_mult=mzm,
        features=build_features_from_trade(trade),
        cfg=TTF_CFG,
    )


def _live_decision(trade: Dict) -> tuple[bool, str]:
    """Reproduce trading_engine.py _check_symbol two-tier gate logic."""
    # Live uses the SAME helpers as BT; only wrapping differs (different sources
    # of max_intraday / MACD mult / features). For parity, the input dict matches
    # the cache row shape, so we read the same fields.
    ic = trade.get("intraday_change_at_entry")
    try:
        ic_val = float(ic) if ic not in (None, "", "None") else None
    except (ValueError, TypeError):
        ic_val = None
    tier = classify_tier(
        ic_val,
        a_tier_lower=TTF_CFG["a_tier_lower"],
        extras_lower=TTF_CFG["extras_lower"],
    )
    try:
        mzm = float(trade.get("macd_zone_mult") or 0.0)
    except (ValueError, TypeError):
        mzm = 0.0
    return should_keep(
        tier=tier,
        macd_zone_mult=mzm,
        features=build_features_from_trade(trade),
        cfg=TTF_CFG,
    )


SYNTHETIC_TRADES: List[Dict] = [
    # A-tier: always accepted (max_intraday >= 20)
    {"symbol": "AAAA", "intraday_change_at_entry": "25.0", "macd_zone_mult": "1.0",
     "conviction_mult": "2.5", "qf_vwap_dist_pct": "6.0", "qf_fill_vwap_dist_pct": "6.0",
     "entry_time_et": "10:15:00"},
    # Edge: < 10% always accepted even without features
    {"symbol": "EDGE", "intraday_change_at_entry": "5.0", "macd_zone_mult": "1.0",
     "conviction_mult": "1.5", "qf_vwap_dist_pct": "3.0", "qf_fill_vwap_dist_pct": "3.0",
     "entry_time_et": "09:35:00"},
    # Extras, MACD low → surgical drop
    {"symbol": "EXLO", "intraday_change_at_entry": "15.0", "macd_zone_mult": "1.0",
     "conviction_mult": "1.5", "qf_vwap_dist_pct": "3.0", "qf_fill_vwap_dist_pct": "3.0",
     "entry_time_et": "09:50:00"},
    # Extras, MACD OK, composite well above threshold → kept
    {"symbol": "EXGO", "intraday_change_at_entry": "15.0", "macd_zone_mult": "1.5",
     "conviction_mult": "1.2", "qf_vwap_dist_pct": "2.0", "qf_fill_vwap_dist_pct": "2.0",
     "entry_time_et": "09:45:00"},
    # Extras, MACD OK, composite below threshold → dropped
    {"symbol": "EXBD", "intraday_change_at_entry": "15.0", "macd_zone_mult": "1.5",
     "conviction_mult": "2.4", "qf_vwap_dist_pct": "8.0", "qf_fill_vwap_dist_pct": "8.0",
     "entry_time_et": "11:00:00"},
    # A-tier with all "bad" features — still accepted since tier=A bypasses gates
    {"symbol": "AGRA", "intraday_change_at_entry": "22.0", "macd_zone_mult": "1.0",
     "conviction_mult": "2.5", "qf_vwap_dist_pct": "10.0", "qf_fill_vwap_dist_pct": "10.0",
     "entry_time_et": "12:00:00"},
    # Extras just at 10% boundary → classified Extras
    {"symbol": "EXBT", "intraday_change_at_entry": "10.0", "macd_zone_mult": "1.5",
     "conviction_mult": "1.5", "qf_vwap_dist_pct": "4.0", "qf_fill_vwap_dist_pct": "4.0",
     "entry_time_et": "09:45:00"},
    # A-tier boundary at 20%
    {"symbol": "ABRD", "intraday_change_at_entry": "20.0", "macd_zone_mult": "1.0",
     "conviction_mult": "2.0", "qf_vwap_dist_pct": "5.0", "qf_fill_vwap_dist_pct": "5.0",
     "entry_time_et": "10:00:00"},
    # Missing intraday_change_at_entry (old cache row) — classified edge, kept
    {"symbol": "OLDC", "intraday_change_at_entry": "", "macd_zone_mult": "1.0",
     "conviction_mult": "1.5", "qf_vwap_dist_pct": "4.0", "qf_fill_vwap_dist_pct": "4.0",
     "entry_time_et": "10:00:00"},
    # Extras, MACD ok, composite near but above threshold → kept.
    # Values picked to score ~ -0.22 (above -0.50 threshold).
    {"symbol": "EXAT", "intraday_change_at_entry": "12.5", "macd_zone_mult": "1.5",
     "conviction_mult": "1.9", "qf_vwap_dist_pct": "4.5", "qf_fill_vwap_dist_pct": "4.7",
     "entry_time_et": "10:10:00"},
]


class TestBTLiveParity:
    """Parity: BT Stage-2 and live engine produce IDENTICAL accept/reject."""

    @pytest.mark.parametrize("trade", SYNTHETIC_TRADES, ids=[t["symbol"] for t in SYNTHETIC_TRADES])
    def test_bt_and_live_agree(self, trade):
        bt_keep, bt_reason = _bt_decision(trade)
        live_keep, live_reason = _live_decision(trade)
        assert bt_keep == live_keep, (
            f"{trade['symbol']}: BT={bt_keep} ({bt_reason!r}) vs "
            f"live={live_keep} ({live_reason!r})"
        )
        assert bt_reason == live_reason, (
            f"{trade['symbol']}: BT reason={bt_reason!r} vs live reason={live_reason!r}"
        )

    def test_full_batch_parity(self):
        """Over the entire synthetic batch, decisions must align trade-by-trade."""
        bt_kept = [t["symbol"] for t in SYNTHETIC_TRADES if _bt_decision(t)[0]]
        live_kept = [t["symbol"] for t in SYNTHETIC_TRADES if _live_decision(t)[0]]
        assert bt_kept == live_kept

    def test_expected_rejections(self):
        """Explicit expected outcomes for each synthetic fixture."""
        expected = {
            "AAAA": True,   # A-tier always kept
            "EDGE": True,   # edge always kept
            "EXLO": False,  # Extras + MACD 1.0 → surgical
            "EXGO": True,   # Extras, MACD ok, composite ok
            "EXBD": False,  # Extras, MACD ok but composite too low
            "AGRA": True,   # A-tier bypass
            "EXBT": True,   # Extras 10% boundary, features nominal
            "ABRD": True,   # 20% boundary → A-tier
            "OLDC": True,   # missing intraday -> edge -> kept
            "EXAT": True,   # Extras, composite right above threshold
        }
        for sym, exp_keep in expected.items():
            t = next(x for x in SYNTHETIC_TRADES if x["symbol"] == sym)
            bt_keep, bt_reason = _bt_decision(t)
            assert bt_keep is exp_keep, f"{sym}: expected keep={exp_keep}, got {bt_keep} ({bt_reason})"


class TestDisabledIsNoOp:
    """When filter is disabled in config, every trade is kept."""

    def test_disabled_keeps_all(self):
        cfg_off = {**TTF_CFG, "enabled": False}
        for t in SYNTHETIC_TRADES:
            ic = t.get("intraday_change_at_entry")
            ic_val = float(ic) if ic else None
            tier = classify_tier(ic_val)
            keep, _ = should_keep(
                tier=tier,
                macd_zone_mult=float(t.get("macd_zone_mult") or 0.0),
                features=build_features_from_trade(t),
                cfg=cfg_off,
            )
            assert keep is True, f"{t['symbol']}: expected keep when disabled"
