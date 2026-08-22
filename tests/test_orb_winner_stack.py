"""Unit tests for trading/orb_winner_stack.py — the shared ATR-floor +
scale-out physics (frozen semantics; docs/orb_winner_stack_design_aug2026.md).

Includes the harness-formula golden: atr14_t1 must equal the validated
phaseB_regime_atr.atr14_lookup value for the same day (shift(1) equivalence),
and the 14/15-bar boundary golden (P0-6.1 frozen rule).
"""
import numpy as np
import pandas as pd
import pytest

from trading.orb_winner_stack import (
    DEFAULT_ATR_K, DEFAULT_SCALE_FRAC, DEFAULT_SCALE_LEVEL_R,
    DEGENERATE_STOP_EPS, FLOOR_BOUND, FLOOR_DEGENERATE, FLOOR_NO_ATR,
    FLOOR_UNBOUND, atr14_t1, floored_stop, scale_params,
)


def _daily(n, base=10.0, tr=1.0, seed=7):
    """n daily bars with pseudo-random but deterministic OHLC."""
    rng = np.random.RandomState(seed)
    close = base + np.cumsum(rng.uniform(-0.3, 0.3, n))
    high = close + rng.uniform(0.2, tr, n)
    low = close - rng.uniform(0.2, tr, n)
    return pd.DataFrame({'high': high, 'low': low, 'close': close})


def _harness_atr_for_day_T(full_df):
    """The EXACT phaseB_regime_atr.atr14_lookup formula: rolling(14).mean()
    of TR, shift(1) — evaluated for a hypothetical day T that follows the
    last row of full_df (i.e. the shifted value one past the end)."""
    d = full_df.reset_index(drop=True)
    pc = d['close'].shift(1)
    tr = np.maximum(d['high'] - d['low'],
                    np.maximum((d['high'] - pc).abs(), (d['low'] - pc).abs()))
    atr = tr.rolling(14).mean()
    # shift(1) indexed at T == unshifted value at T-1 == last row here.
    return atr.iloc[-1]


class TestAtr14T1:
    def test_matches_harness_formula(self):
        """Golden: identical to the validated harness shift(1) value."""
        df = _daily(30)
        expected = _harness_atr_for_day_T(df)
        got = atr14_t1(df)
        assert got == pytest.approx(float(expected), abs=1e-12)

    def test_boundary_15_bars_available(self):
        """Exactly 15 bars strictly before T -> ATR available (14 valid TRs)."""
        df = _daily(15)
        got = atr14_t1(df)
        assert got is not None
        assert got == pytest.approx(float(_harness_atr_for_day_T(df)),
                                    abs=1e-12)

    def test_boundary_14_bars_unavailable(self):
        """P0-6.1 frozen boundary: 14 bars -> the rolling window still holds
        the NaN first-TR -> None (fail-open). This deliberately REJECTS the
        phaseB_frontier 13-TR variant."""
        assert atr14_t1(_daily(14)) is None

    def test_short_history_none(self):
        assert atr14_t1(_daily(3)) is None

    def test_empty_and_none_inputs(self):
        assert atr14_t1(None) is None
        assert atr14_t1(pd.DataFrame()) is None
        assert atr14_t1([]) is None

    def test_accepts_list_of_dicts(self):
        df = _daily(20)
        rows = df.to_dict('records')
        assert atr14_t1(rows) == pytest.approx(atr14_t1(df), abs=1e-12)

    def test_missing_columns_none(self):
        assert atr14_t1(pd.DataFrame({'close': [1.0] * 20})) is None

    def test_accepts_objects_with_attrs(self):
        from types import SimpleNamespace
        df = _daily(20)
        objs = [SimpleNamespace(high=r.high, low=r.low, close=r.close)
                for r in df.itertuples()]
        assert atr14_t1(objs) == pytest.approx(atr14_t1(df), abs=1e-12)

    def test_computation_error_fails_open_with_warning(self, caplog):
        import logging
        bad = pd.DataFrame({'high': ['x'] * 20, 'low': ['y'] * 20,
                            'close': ['z'] * 20})
        with caplog.at_level(logging.WARNING):
            assert atr14_t1(bad) is None
        assert any('floor will not apply' in r.message
                   for r in caplog.records)

    def test_extra_history_irrelevant(self):
        """Only the trailing 15 bars matter — extra history changes nothing."""
        df = _daily(60)
        assert atr14_t1(df) == pytest.approx(atr14_t1(df.iloc[-20:]),
                                             abs=1e-12)

    def test_gap_tr_uses_prev_close(self):
        """TR includes |H-prevC| / |L-prevC| — a gap day inflates ATR."""
        flat = pd.DataFrame({'high': [10.5] * 16, 'low': [10.0] * 16,
                             'close': [10.2] * 16})
        gapped = flat.copy()
        gapped.loc[10, ['high', 'low', 'close']] = [20.5, 20.0, 20.2]
        assert atr14_t1(gapped) > atr14_t1(flat)


class TestFlooredStop:
    def test_bound(self):
        stop, status = floored_stop(9.0, 10.0, 1.0, 0.25)
        assert stop == pytest.approx(9.75)
        assert status == FLOOR_BOUND

    def test_unbound_when_range_low_tighter(self):
        stop, status = floored_stop(9.9, 10.0, 2.0, 0.25)
        # entry - 0.25*2.0 = 9.5 < 9.9 -> range_low wins
        assert stop == pytest.approx(9.9)
        assert status == FLOOR_UNBOUND

    def test_no_atr_fail_open(self):
        stop, status = floored_stop(9.0, 10.0, None, 0.25)
        assert (stop, status) == (9.0, FLOOR_NO_ATR)

    def test_nan_atr_fail_open(self):
        stop, status = floored_stop(9.0, 10.0, float('nan'), 0.25)
        assert (stop, status) == (9.0, FLOOR_NO_ATR)

    def test_degenerate_clamp_zero_atr(self):
        """P1-3: ATR=0 -> floor == entry -> clamp rejects, falls to range_low."""
        stop, status = floored_stop(9.0, 10.0, 0.0, 0.25)
        assert (stop, status) == (9.0, FLOOR_DEGENERATE)

    def test_degenerate_clamp_boundary(self):
        """Floor exactly at entry*(1-eps) is allowed; above it is rejected."""
        entry = 100.0
        # floor = entry - k*atr; choose atr so floor == entry*(1-eps) exactly
        atr_ok = entry * DEGENERATE_STOP_EPS / 0.25
        stop, status = floored_stop(1.0, entry, atr_ok, 0.25)
        assert status == FLOOR_BOUND
        assert stop == pytest.approx(entry * (1 - DEGENERATE_STOP_EPS))
        # a hair tighter -> degenerate
        stop2, status2 = floored_stop(1.0, entry, atr_ok * 0.999, 0.25)
        assert status2 == FLOOR_DEGENERATE
        assert stop2 == 1.0

    def test_default_k_frozen(self):
        assert DEFAULT_ATR_K == 0.25


class TestScaleParams:
    def test_level_and_qty(self):
        px, qty = scale_params(10.0, 0.5, 0.40, 3.0, 100)
        assert px == pytest.approx(11.5)   # 10 + 3*0.5
        assert qty == 40

    def test_floor_to_int(self):
        _, qty = scale_params(10.0, 0.5, 0.40, 3.0, 101)
        assert qty == 40   # floor(40.4)

    def test_tiny_qty_no_scale(self):
        _, qty = scale_params(10.0, 0.5, 0.40, 3.0, 2)
        assert qty == 0    # floor(0.8) < 1 -> all-runner

    def test_one_share_boundary(self):
        _, qty = scale_params(10.0, 0.5, 0.40, 3.0, 3)
        assert qty == 1    # floor(1.2)

    def test_zero_shares(self):
        _, qty = scale_params(10.0, 0.5, 0.40, 3.0, 0)
        assert qty == 0

    def test_frozen_defaults(self):
        assert DEFAULT_SCALE_FRAC == 0.40
        assert DEFAULT_SCALE_LEVEL_R == 3.0


class TestSharedModuleParity:
    """Both BT and LIVE must import the shared module (parity by
    construction — mirrors test_orb_touchgo_parity)."""

    def test_bt_imports_shared_module(self):
        from pathlib import Path
        src = (Path(__file__).parent.parent
               / 'study_orb_pipeline_static_lock.py').read_text()
        assert 'from trading.orb_winner_stack import' in src
        assert 'atr14_t1' in src and 'floored_stop' in src \
            and 'scale_params' in src

    def test_live_engine_imports_shared_module(self):
        from pathlib import Path
        src = (Path(__file__).parent.parent / 'trading'
               / 'orb_engine.py').read_text()
        assert 'from trading.orb_winner_stack import' in src
        assert 'atr14_t1' in src and 'floored_stop' in src \
            and 'scale_params' in src
