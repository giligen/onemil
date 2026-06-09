"""ORB engine must extract tp_leg_id / sl_leg_id from the
submit_bracket_order response and populate them on the OpenPosition.

Without this, stop_monitor.py's BRANCH_SL_LEG_RACE recovery path is
dead code for ORB — exactly what bit FABC on 2026-06-09.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent.parent


def _read(rel: str) -> str:
    return (ROOT / rel).read_text()


class TestOrbExtractsLegs:

    def test_orb_engine_extracts_leg_ids_from_result(self):
        """Source inspection: ORB's _submit_entry path reads
        result['legs'] and picks TP / SL leg IDs."""
        src = _read("trading/orb_engine.py")
        # The extraction block from Fix 3
        assert "result.get('legs')" in src, (
            "ORB no longer reads bracket legs from the submit response"
        )
        # Pin the discriminator: TP by limit_price, SL by stop_price
        assert "'limit_price'" in src and "is not None" in src
        assert "'stop_price'" in src

    def test_open_position_gets_leg_ids(self):
        """OpenPosition construction in _submit_entry must pass the
        extracted tp_leg_id / sl_leg_id (not the default empty strings)."""
        src = _read("trading/orb_engine.py")
        # The OpenPosition kwarg form. The leg-id extraction creates
        # local variables tp_leg_id and sl_leg_id that get passed in.
        assert "tp_leg_id=tp_leg_id" in src
        assert "sl_leg_id=sl_leg_id" in src


class TestAlpacaClientLegsContract:

    def test_alpaca_client_submit_bracket_returns_legs(self):
        """The shared bracket-submit helper must return a 'legs' list
        — confirmed by source inspection mirroring the get_order pattern."""
        src = _read("data_sources/alpaca_client.py")
        # The Fix 3 addition
        assert "'legs':" in src
        # Must extract from the SDK Order's `legs` attribute
        assert "getattr(order, 'legs'" in src or "order.legs" in src
