"""Unit tests: per-pool exit-override resolution (owner 2026-10-01, P1
half-out at +1R). Covers trading.orb_addon_gates.resolve_pool_exit_params
and the OpenPosition pool_id/pool_exit field defaults -- a pool that sets
none of the exit_* keys must resolve to exactly production behaviour
(CLAUDE.md "ONE spec" / no-accidental-behaviour rule).
"""
from datetime import datetime, timezone

from trading.orb_addon_gates import (
    EXIT_OVERRIDE_KEYS,
    KNOWN_POOL_KEYS,
    resolve_pool_exit_params,
)
from trading.orb_engine import OpenPosition


def _pos(**overrides):
    base = dict(
        symbol='TEST', entry_price=10.0, stop_price=9.5, shares=100,
        trade_id=1, order_id='', entry_time=datetime.now(timezone.utc),
        range_high=10.2, range_low=10.0, lock_arm_at_r=1.75, lock_stop_r=0.5,
        composite_score=0.5, quintile='Q3',
    )
    base.update(overrides)
    return OpenPosition(**base)


class TestResolvePoolExitParamsAbsent:
    def test_empty_pool_cfg_returns_empty(self):
        assert resolve_pool_exit_params({}) == {}

    def test_pool_with_only_gate_keys_returns_empty(self):
        cfg = {'name': 'addon_p30', 'min_gap_pct': 3.0, 'max_gap_pct': 5.0}
        assert resolve_pool_exit_params(cfg) == {}

    def test_exit_override_keys_are_known_pool_keys(self):
        # Otherwise warn_unknown_pool_keys would flag every P1-style pool
        # dict as having unrecognized keys.
        assert EXIT_OVERRIDE_KEYS <= KNOWN_POOL_KEYS


class TestResolvePoolExitParamsPresent:
    def test_p1_half_out_keys_translate(self):
        cfg = {'name': 'addon_gap35_range5', 'pool_id': 'P1',
               'exit_scale_out_pct': 0.5, 'exit_scale_out_at_r': 1.0}
        out = resolve_pool_exit_params(cfg)
        assert out == {'scale_out_pct': 0.5, 'scale_out_at_r': 1.0}

    def test_all_five_keys_translate(self):
        cfg = {
            'name': 'p1', 'exit_scale_out_pct': 0.4, 'exit_scale_out_at_r': 1.5,
            'exit_lock_arm_at_r': 2.0, 'exit_lock_stop_r': 0.75,
            'exit_target_r': 3.0,
        }
        out = resolve_pool_exit_params(cfg)
        assert out == {
            'scale_out_pct': 0.4, 'scale_out_at_r': 1.5,
            'lock_arm_at_r': 2.0, 'lock_stop_r': 0.75, 'target_r': 3.0,
        }

    def test_partial_override_only_sets_given_keys(self):
        cfg = {'name': 'p1', 'exit_scale_out_pct': 0.5}
        out = resolve_pool_exit_params(cfg)
        assert out == {'scale_out_pct': 0.5}
        assert 'scale_out_at_r' not in out


class TestResolvePoolExitParamsInvalid:
    """A present-but-nonsense value reverts to production for THAT key
    (dropped, WARNING logged) rather than crashing or arming garbage."""

    def test_pct_above_one_dropped(self):
        assert resolve_pool_exit_params({'name': 'p1', 'exit_scale_out_pct': 1.5}) == {}

    def test_pct_zero_dropped(self):
        assert resolve_pool_exit_params({'name': 'p1', 'exit_scale_out_pct': 0.0}) == {}

    def test_negative_r_dropped(self):
        assert resolve_pool_exit_params({'name': 'p1', 'exit_scale_out_at_r': -1.0}) == {}

    def test_non_numeric_dropped(self):
        assert resolve_pool_exit_params({'name': 'p1', 'exit_scale_out_pct': 'half'}) == {}

    def test_none_value_is_absent_not_invalid(self):
        assert resolve_pool_exit_params({'name': 'p1', 'exit_scale_out_pct': None}) == {}


class TestOpenPositionPoolExitDefaults:
    """absent = production behaviour, byte-identical."""

    def test_defaults_are_production(self):
        pos = _pos()
        assert pos.pool_id == 'production'
        assert pos.pool_exit == {}

    def test_pool_fields_settable(self):
        pos = _pos(pool_id='P1', pool_exit={'scale_out_pct': 0.5, 'scale_out_at_r': 1.0})
        assert pos.pool_id == 'P1'
        assert pos.pool_exit['scale_out_pct'] == 0.5

    def test_absent_pool_exit_falls_back_to_production_default(self):
        # The exact dict.get(key, production_default) pattern every
        # consumer (_maybe_arm_scale, _handle_scale_fill_event) uses.
        pos = _pos()
        production_frac = 0.4
        resolved = float((pos.pool_exit or {}).get('scale_out_pct', production_frac))
        assert resolved == production_frac
