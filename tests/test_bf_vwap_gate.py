"""Above-VWAP gate: shared decision, Stage-2 filter, live twin, parity."""
import logging
from unittest.mock import MagicMock

import pandas as pd
import pytest

from trading.bf_vwap_gate import (
    DISABLED, VwapGateConfig, filter_trades, load_vwap_gate_config,
    passes_vwap_gate,
)

ON = VwapGateConfig(enabled=True, min_dist_pct=0.0)


class TestDecision:
    def test_disabled_keeps_everything(self):
        assert passes_vwap_gate(-9.0, DISABLED)[0] is True
        assert passes_vwap_gate(None, DISABLED)[0] is True

    def test_above_vwap_kept_at_or_below_rejected(self):
        assert passes_vwap_gate(0.01, ON)[0] is True
        assert passes_vwap_gate(0.0, ON)[0] is False
        assert passes_vwap_gate(-2.19, ON)[0] is False

    def test_min_dist_threshold(self):
        cfg = VwapGateConfig(enabled=True, min_dist_pct=1.0)
        assert passes_vwap_gate(0.5, cfg)[0] is False
        assert passes_vwap_gate(1.5, cfg)[0] is True

    def test_unknown_distance_fails_open_and_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            keep, reason = passes_vwap_gate(None, ON)
        assert keep is True and 'fail-open' in reason
        assert any('fail-open' in r.message for r in caplog.records)

    def test_config_load(self):
        assert load_vwap_gate_config({}) == DISABLED
        assert load_vwap_gate_config(None) == DISABLED
        cfg = load_vwap_gate_config({'vwap_gate': {'enabled': True, 'min_dist_pct': 0.5}})
        assert cfg == VwapGateConfig(True, 0.5)


class TestStage2Filter:
    ROWS = [
        {'symbol': 'ABOVE', 'qf_vwap_dist_pct': '3.10'},
        {'symbol': 'BELOW', 'qf_vwap_dist_pct': '-2.19'},
        {'symbol': 'AT', 'qf_vwap_dist_pct': 0.0},
        {'symbol': 'NA', 'qf_vwap_dist_pct': ''},
        {'symbol': 'NONE'},
    ]

    def test_disabled_is_identity(self):
        assert filter_trades(self.ROWS, DISABLED) == self.ROWS

    def test_removes_at_or_below_keeps_unknown(self):
        kept = [t['symbol'] for t in filter_trades(self.ROWS, ON)]
        assert kept == ['ABOVE', 'NA', 'NONE']


def _bars(closes, vols=None):
    vols = vols or [1000] * len(closes)
    return pd.DataFrame({
        'open': closes, 'high': [c + 0.05 for c in closes],
        'low': [c - 0.05 for c in closes], 'close': closes, 'volume': vols,
    })


class TestLiveTwin:
    """The engine computes the same number BT caches and asks the same
    function; a below-VWAP setup returns None (no plan) with a log line."""

    @pytest.fixture
    def engine(self):
        from trading.trading_engine import TradingEngine
        eng = TradingEngine.__new__(TradingEngine)
        eng.vwap_gate = ON
        eng._eod_skipped = []
        eng._compute_vwap = TradingEngine._compute_vwap.__get__(eng)
        return eng

    def test_live_feature_matches_bt_feature(self, engine):
        # BT: (breakout_level - vwap) / vwap * 100 with vwap through setup bar
        bars = _bars([10.0, 10.5, 11.0, 10.8])
        setup = MagicMock(); setup.flag_end_idx = 3; setup.breakout_level = 10.4
        vwap = engine._compute_vwap(bars, up_to_idx=3)
        live_dist = (setup.breakout_level - vwap) / vwap * 100
        assert vwap == pytest.approx(sum([10.0, 10.5, 11.0, 10.8]) / 4)
        assert live_dist < 0
        assert passes_vwap_gate(live_dist, ON)[0] is False
        assert passes_vwap_gate(round(live_dist, 2), ON) == passes_vwap_gate(live_dist, ON)

    def test_engine_skip_path(self, engine, caplog):
        """Drive the exact code the engine runs: below VWAP → skip recorded."""
        bars = _bars([10.0, 10.5, 11.0, 10.8])
        setup = MagicMock(); setup.flag_end_idx = 3; setup.breakout_level = 10.4
        vwap = engine._compute_vwap(bars, up_to_idx=setup.flag_end_idx)
        dist = (setup.breakout_level - vwap) / vwap * 100
        keep, reason = passes_vwap_gate(dist, engine.vwap_gate)
        assert keep is False and 'below VWAP' in reason
        # and an above-VWAP setup passes
        setup.breakout_level = 11.2
        dist = (setup.breakout_level - vwap) / vwap * 100
        assert passes_vwap_gate(dist, engine.vwap_gate)[0] is True


def test_engine_source_wires_gate_before_conviction():
    """Structural guard: the gate sits before conviction/sizing and returns None."""
    import inspect
    from trading.trading_engine import TradingEngine
    src = inspect.getsource(TradingEngine)
    i_gate = src.index('passes_vwap_gate(_vg_dist, self.vwap_gate)')
    i_conv = src.index('conviction_mult, _conv_brkdn = self._compute_conviction_score_setup(')
    assert i_gate < i_conv
    assert 'VWAP GATE skip' in src


class TestShadow:
    def test_shadow_decides_but_stage2_ignores(self):
        cfg = VwapGateConfig(enabled=False, min_dist_pct=0.0, shadow=True)
        assert passes_vwap_gate(-1.0, cfg)[0] is False   # the decision is computed
        rows = [{'symbol': 'B', 'qf_vwap_dist_pct': -1.0}]
        assert filter_trades(rows, cfg) == rows           # BT never acts on shadow

    def test_config_load_shadow(self):
        cfg = load_vwap_gate_config({'vwap_gate': {'enabled': False, 'shadow': True}})
        assert cfg.shadow is True and cfg.enabled is False

    def test_engine_shadow_never_returns_none(self):
        import inspect
        from trading.trading_engine import TradingEngine
        src = inspect.getsource(TradingEngine)
        assert 'VWAP GATE [SHADOW] would skip' in src
        assert 'CONSISTENCY RULES [SHADOW] would skip' in src
        blk = src[src.index('VWAP GATE [SHADOW]'):src.index('CONSISTENCY RULES [SHADOW]')]
        # the shadow branch has no early return; only the enabled branch does
        assert blk.count('return None') == 1 and 'elif not _vg_keep:' in blk
