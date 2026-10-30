"""docs/hod_arm_gate_parity_spec_20261030.md (cell 1,729): the research twin's arm gate goes live.

An arm is skipped when `trigger < min_price` or `(trigger - stop) / trigger x 100 < min_r_pct` (strict '<').
ONE predicate (`trading.hod_break.arm_gate_reject`) is used by the live engine (`HodBreakEngine._apply_arm_gate`) and
by the twin (`research/hod_entry/causal_arming.arm_state`). Unit / parity / integration levels below.
"""
import dataclasses
import logging
import os
import sys

import numpy as np
import pytest

from trading.hod_break import HodBreakParams, arm_gate_reject, arm_state, profile_fraction
from tests.test_hod_resting_entry import (
    tape, ADV20, mock_alpaca, mock_db, mock_sm, trades_db, resting_engine, read_ledger,
)
from tests.test_hod_break_engine import admit, bars_df

MIN_PRICE = 20.0
MIN_R = 1.5


def gated_engine(mock_alpaca, mock_db, mock_sm, tmp_path, min_price=MIN_PRICE, min_r_pct=MIN_R):
    """Resting-mode dry engine carrying the LIVE gate values (config.yaml: min_price 20, params.min_r_pct 1.5)."""
    e = resting_engine(mock_alpaca, mock_db, mock_sm, tmp_path, min_price=min_price)
    e.params = dataclasses.replace(e.params, min_r_pct=min_r_pct)
    return e


def arm_with(trigger, stop):
    """An arm dict as arm_state returns it (level = trigger - 1c)."""
    return dict(level=trigger - 0.01, trigger=trigger, limit=trigger * 1.0015, stop=stop)


# ------------------------------------------------------------------------------------------------ unit
class TestArmGatePredicate:
    def test_tight_r_1p2pct_is_rejected(self):
        assert arm_gate_reject(50.0, 50.0 * (1 - 0.012), MIN_PRICE, MIN_R) == 'r_pct'

    def test_wide_r_1p6pct_arms(self):
        assert arm_gate_reject(50.0, 50.0 * (1 - 0.016), MIN_PRICE, MIN_R) is None

    def test_trigger_below_min_price_is_rejected(self):
        assert arm_gate_reject(19.99, 19.0, MIN_PRICE, MIN_R) == 'price'

    def test_trigger_exactly_at_min_price_arms(self):
        assert arm_gate_reject(20.0, 19.0, MIN_PRICE, MIN_R) is None

    def test_boundary_exactly_1p5pct_arms(self):
        trigger, stop = 100.0, 98.5                                 # (100 - 98.5) / 100 x 100 == 1.5 exactly
        assert (trigger - stop) / trigger * 100.0 == MIN_R
        assert arm_gate_reject(trigger, stop, MIN_PRICE, MIN_R) is None


class TestEngineGate:
    def test_tight_arm_skipped_with_log_line_and_counter(self, mock_alpaca, mock_db, mock_sm, tmp_path, caplog):
        e = gated_engine(mock_alpaca, mock_db, mock_sm, tmp_path)
        with caplog.at_level(logging.INFO):
            assert e._apply_arm_gate('ABC', arm_with(50.0, 50.0 * (1 - 0.012)), e.params) is None
            e._apply_arm_gate('ABC', arm_with(50.0, 50.0 * (1 - 0.012)), e.params)   # same arm re-walked on the next bar
        lines = [r.message for r in caplog.records if 'ARM SKIP ABC' in r.message]
        assert len(lines) == 1 and 'r_pct=1.20% < 1.5%' in lines[0] and lines[0].startswith('[HOD]')
        assert e.arm_gate_skips == {'price': 0, 'r_pct': 1}

    def test_wide_arm_passes_untouched(self, mock_alpaca, mock_db, mock_sm, tmp_path):
        e = gated_engine(mock_alpaca, mock_db, mock_sm, tmp_path)
        a = arm_with(50.0, 50.0 * (1 - 0.016))
        assert e._apply_arm_gate('ABC', a, e.params) is a
        assert e.arm_gate_skips == {'price': 0, 'r_pct': 0}

    def test_price_skip_logged_and_counted(self, mock_alpaca, mock_db, mock_sm, tmp_path, caplog):
        e = gated_engine(mock_alpaca, mock_db, mock_sm, tmp_path)
        with caplog.at_level(logging.INFO):
            assert e._apply_arm_gate('ABC', arm_with(19.5, 18.0), e.params) is None
        assert any('ARM SKIP ABC: trigger 19.50 < min_price 20.00' in r.message for r in caplog.records)
        assert e.arm_gate_skips['price'] == 1

    def test_boundary_1p5pct_arms_in_engine(self, mock_alpaca, mock_db, mock_sm, tmp_path):
        e = gated_engine(mock_alpaca, mock_db, mock_sm, tmp_path)
        a = arm_with(100.0, 98.5)
        assert e._apply_arm_gate('ABC', a, e.params) is a

    def test_counters_reset_on_new_session(self, mock_alpaca, mock_db, mock_sm, tmp_path):
        e = gated_engine(mock_alpaca, mock_db, mock_sm, tmp_path)
        e._apply_arm_gate('ABC', arm_with(50.0, 49.9), e.params)
        e.session_date = '1999-01-01'; e._roll_session()
        assert e.arm_gate_skips == {'price': 0, 'r_pct': 0} and not e._arm_skip_logged


# ------------------------------------------------------------------------------------------------ parity
def _twin():
    """research/hod_entry/causal_arming (the cell 1,438/1,729 research build); skip if not importable here."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'research', 'hod_entry'))
        import causal_arming as ca
        return ca
    except Exception as e:      # pragma: no cover - environment dependent
        pytest.skip(f"research/hod_entry/causal_arming.py not importable here: {e}")


def synthetic_day(level, stop):
    """Minute bars arming at j=7: HOD `level` set at bar 2, five consolidation bars whose lowest low is `stop`.
    Returns (o, h, l, v, m, adv20) with relative volume 2.0 at j=7 (inside the [1, 5) band)."""
    o = [level * 0.9] * 8; h = [level * 0.95, level * 0.97, level] + [level * 0.995] * 5
    l = [level * 0.9, level * 0.95, level * 0.99] + [stop + 0.5 * (level - stop)] * 5
    l[5] = stop                                                                  # lowest of the last 5 lows (bars 3..7)
    v = [10_000.0] * 8; m = list(range(570, 578))
    adv20 = float(np.sum(v)) / (2.0 * profile_fraction(int(m[7])))
    return [np.asarray(x, dtype=float) for x in (o, h, l)] + [np.asarray(v), np.asarray(m), adv20]


class TestParityWithTwin:
    def test_200_trigger_stop_pairs_identical_decisions(self, mock_alpaca, mock_db, mock_sm, tmp_path):
        ca = _twin()
        e = gated_engine(mock_alpaca, mock_db, mock_sm, tmp_path)
        rng = np.random.default_rng(20261030)
        agree = {'armed': 0, 'skipped': 0}
        for i in range(200):
            level = float(np.round(rng.uniform(15.0, 60.0), 2))
            r_pct = float(rng.uniform(0.2, 3.9))                                 # within consol_pct (4 %) so the base exists
            stop = float(np.round(level * (1.0 - r_pct / 100.0), 2))
            if i % 25 == 0:                                                      # force exact-boundary cases
                stop = float(np.round(level * (1.0 - MIN_R / 100.0), 2))
            o, h, l, v, m, adv20 = synthetic_day(level, stop)
            twin = ca.arm_state(o, h, l, v, m, 7, adv20, e.params, MIN_PRICE)
            raw = arm_state(o, h, l, v, m, 7, adv20, e.params)
            assert raw is not None, f"pair {i}: base did not form (level {level} stop {stop})"
            live = e._apply_arm_gate('ABC', raw, e.params)
            assert (twin is None) == (live is None), f"pair {i}: twin={twin} live={live}"
            if live is not None:
                assert live['trigger'] == pytest.approx(twin['trigger']) and live['stop'] == pytest.approx(twin['stop'])
            agree['armed' if live is not None else 'skipped'] += 1
        assert agree['armed'] > 20 and agree['skipped'] > 20, agree             # both branches really exercised

    def test_live_config_keys_equal_the_twins(self, mock_alpaca, mock_db, mock_sm, tmp_path):
        """The engine reads config `hod_break.min_price` and `hod_break.params.min_r_pct` -- the twin's `live_params()`."""
        ca = _twin()
        from config import Config
        cfg = Config().hod_break_cfg
        p, floor, _ = ca.live_params()
        assert float(cfg['min_price']) == floor == MIN_PRICE
        assert HodBreakParams(**cfg['params']).min_r_pct == p.min_r_pct == MIN_R
        e = resting_engine(mock_alpaca, mock_db, mock_sm, tmp_path, min_price=cfg['min_price'])
        assert e.min_price == floor


# ------------------------------------------------------------------------------------------------ integration
def tight_tape():
    """tape() shape, but the consolidation lows sit 0.4 % under the 11.00 HOD (R ~ 0.5 % << 1.5 %); the break bar just crosses the 11.01 trigger."""
    t = [(10.0, 10.3, 9.9, 10.3, 5000), (10.3, 10.7, 10.2, 10.7, 6000), (10.7, 11.0, 10.96, 10.97, 8000)]
    t += [(10.96, 10.99, 10.96, 10.97, 3000)] * 5
    t += [(10.97, 11.01, 10.96, 11.0, 9000), (11.0, 11.1, 10.99, 11.05, 4000)]
    return t


class TestEngineIntegration:
    def test_tight_base_places_no_order_and_writes_no_ledger_row(self, mock_alpaca, mock_db, mock_sm, tmp_path, caplog):
        e = gated_engine(mock_alpaca, mock_db, mock_sm, tmp_path, min_price=1.0)   # price floor out of the way: R decides
        with caplog.at_level(logging.INFO):
            admit(e); e._ingest_bars('ABC', bars_df(tight_tape()))
        assert not os.path.exists(e.dry_ledger_path) or read_ledger(e.dry_ledger_path) == []   # no ledger row though the tape crossed
        assert e.candidates['ABC'].resting_arm is None and not e.candidates['ABC'].resting_filled
        assert e.candidates['ABC'].live_order is None
        assert e.arm_gate_skips['r_pct'] >= 1
        assert any('ARM SKIP ABC: r_pct=' in r.message for r in caplog.records)

    def test_same_tape_arms_when_the_gate_is_off(self, mock_alpaca, mock_db, mock_sm, tmp_path):
        """Control: with min_r_pct 0.1 the identical tape crosses and writes its ledger row -- the gate is the only difference."""
        e = gated_engine(mock_alpaca, mock_db, mock_sm, tmp_path, min_price=1.0, min_r_pct=0.1)
        admit(e); e._ingest_bars('ABC', bars_df(tight_tape()))
        assert len(read_ledger(e.dry_ledger_path)) == 1 and e.arm_gate_skips['r_pct'] == 0

    def test_wide_base_still_arms_under_the_live_gate(self, mock_alpaca, mock_db, mock_sm, tmp_path):
        e = gated_engine(mock_alpaca, mock_db, mock_sm, tmp_path, min_price=1.0)   # tape(): R = 3.7 % > 1.5 %
        admit(e); e._ingest_bars('ABC', bars_df(tape()))
        assert len(read_ledger(e.dry_ledger_path)) == 1
