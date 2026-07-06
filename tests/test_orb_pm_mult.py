"""Premarket dollar-volume sizing mult (2026-07-04 ship).

Pins the shared math (upsize-only, fail-open), engine wiring (once-per-day
batch fetch, mult reaches the planner), and config/env knobs.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
import yaml

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.orb_engine import ORBEngine
from trading.orb_pm_mult import (
    DEFAULT_HIGH_CUT_USD, DEFAULT_HIGH_MULT,
    compute_pm_dollar_vol, pm_size_multiplier,
)
from trading.stop_monitor import StopMonitor


class TestPmSizeMultiplier:
    def test_above_cut_boosts(self):
        assert pm_size_multiplier(6_000_000) == DEFAULT_HIGH_MULT

    def test_below_cut_neutral(self):
        assert pm_size_multiplier(1_000_000) == 1.0

    def test_at_cut_neutral(self):
        """Boundary is strictly > (matches the study's tercile mask)."""
        assert pm_size_multiplier(DEFAULT_HIGH_CUT_USD) == 1.0

    def test_none_fails_open(self):
        assert pm_size_multiplier(None) == 1.0

    def test_garbage_fails_open(self):
        assert pm_size_multiplier('x') == 1.0

    def test_never_downsizes(self):
        """Upsize-only invariant: no input may produce mult < 1.0."""
        for v in (None, 0, 1, 1e12, -5, float('nan')):
            assert pm_size_multiplier(v) >= 1.0


class TestComputePmDollarVol:
    def _bars(self, times_et, vols, closes, vwaps=None):
        rows = []
        for i, t in enumerate(times_et):
            rows.append({
                'timestamp': pd.Timestamp(f'2026-07-06 {t}',
                                          tz='America/New_York')
                .tz_convert('UTC'),
                'open': closes[i], 'high': closes[i], 'low': closes[i],
                'close': closes[i], 'volume': vols[i],
                **({'vwap': vwaps[i]} if vwaps else {}),
            })
        return pd.DataFrame(rows)

    def test_sums_only_premarket_rows(self):
        bars = self._bars(['09:00', '09:29', '09:30', '09:35'],
                          [1000, 2000, 99999, 99999],
                          [5.0, 5.0, 5.0, 5.0])
        assert compute_pm_dollar_vol(bars) == pytest.approx(3000 * 5.0)

    def test_prefers_vwap_when_present(self):
        bars = self._bars(['09:00'], [1000], [5.0], vwaps=[4.0])
        assert compute_pm_dollar_vol(bars) == pytest.approx(4000.0)

    def test_no_premarket_rows_returns_none(self):
        bars = self._bars(['09:31', '09:40'], [10, 10], [5.0, 5.0])
        assert compute_pm_dollar_vol(bars) is None

    def test_empty_or_none_returns_none(self):
        assert compute_pm_dollar_vol(None) is None
        assert compute_pm_dollar_vol(pd.DataFrame()) is None


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.delenv('ORB_PM_MULT', raising=False)
    with open(Path(__file__).parent.parent / 'orb.yaml') as f:
        cfg = yaml.safe_load(f)
    cfg['strategy']['enabled'] = True
    return ORBEngine(alpaca_client=MagicMock(spec=AlpacaClient),
                     db=MagicMock(spec=Database),
                     stop_monitor=MagicMock(spec=StopMonitor), config=cfg)


class TestEngineWiring:
    def test_config_loaded(self, engine):
        assert engine.pm_mult_enabled is True
        assert engine.pm_mult_high_cut == 5816688
        assert engine.pm_mult_high == 1.5

    def test_env_disable(self, monkeypatch):
        monkeypatch.setenv('ORB_PM_MULT', '0')
        with open(Path(__file__).parent.parent / 'orb.yaml') as f:
            cfg = yaml.safe_load(f)
        cfg['strategy']['enabled'] = True
        eng = ORBEngine(alpaca_client=MagicMock(spec=AlpacaClient),
                        db=MagicMock(spec=Database),
                        stop_monitor=MagicMock(spec=StopMonitor), config=cfg)
        assert eng.pm_mult_enabled is False
        assert eng._get_pm_mult('ANY') == 1.0

    def test_batch_fetch_once_per_day(self, engine):
        engine.build_universe(source_loader=lambda: ['AAA', 'BBB'])
        pm_bars = pd.DataFrame([{
            'timestamp': pd.Timestamp('2026-07-06 09:00',
                                      tz='America/New_York').tz_convert('UTC'),
            'open': 5.0, 'high': 5.0, 'low': 5.0, 'close': 5.0,
            'volume': 2_000_000,
        }])
        engine.alpaca.get_premarket_1min_bars_multi = MagicMock(
            return_value={'AAA': pm_bars, 'BBB': pm_bars.iloc[:0]})
        m1 = engine._get_pm_mult('AAA')   # 2M sh x $5 = $10M > cut -> 1.5
        m2 = engine._get_pm_mult('BBB')   # empty bars -> None -> 1.0
        m3 = engine._get_pm_mult('AAA')
        assert m1 == 1.5 and m2 == 1.0 and m3 == 1.5
        assert engine.alpaca.get_premarket_1min_bars_multi.call_count == 1

    def test_fetch_failure_fails_open(self, engine):
        engine.build_universe(source_loader=lambda: ['AAA'])
        engine.alpaca.get_premarket_1min_bars_multi = MagicMock(side_effect=RuntimeError('api down'))
        assert engine._get_pm_mult('AAA') == 1.0

    def test_uses_premarket_method_not_clamped_one(self, engine):
        """2026-07-06 incident: get_1min_bars_multi clamps to 9:30 open by
        design, silently starving PM mult. The engine must use the
        premarket-specific method."""
        engine.build_universe(source_loader=lambda: ['AAA'])
        engine.alpaca.get_premarket_1min_bars_multi = MagicMock(return_value={})
        engine.alpaca.get_1min_bars_multi = MagicMock(return_value={})
        engine._get_pm_mult('AAA')
        engine.alpaca.get_premarket_1min_bars_multi.assert_called_once()
        engine.alpaca.get_1min_bars_multi.assert_not_called()

    def test_prefetch_gated_by_et_time(self, engine, monkeypatch):
        """_maybe_prefetch_pm fires at >=9:31 ET, not before — so the 9:35
        burst finds the cache warm without an early-morning fetch storm."""
        from datetime import datetime, timezone
        from unittest.mock import patch
        engine.build_universe(source_loader=lambda: ['AAA'])
        engine.alpaca.get_premarket_1min_bars_multi = MagicMock(return_value={})
        with patch('trading.orb_engine.datetime') as mdt:
            mdt.now.return_value = datetime(2026, 7, 7, 13, 25, 0, tzinfo=timezone.utc)  # 9:25 ET
            engine._maybe_prefetch_pm()
            engine.alpaca.get_premarket_1min_bars_multi.assert_not_called()
            mdt.now.return_value = datetime(2026, 7, 7, 13, 32, 0, tzinfo=timezone.utc)  # 9:32 ET
            engine._maybe_prefetch_pm()
            engine.alpaca.get_premarket_1min_bars_multi.assert_called_once()

    def test_planner_receives_pm_mult(self, engine):
        """Source-level pin: the submit loop passes pm_mult to planner.build
        and the planner stacks it into position_dollars."""
        import inspect
        from trading import orb_engine as em, orb_planner as pm
        src = inspect.getsource(em.ORBEngine.check_entries)
        assert 'pm_mult=self._get_pm_mult(sym)' in src
        psrc = inspect.getsource(pm.OrbTradePlanner.build)
        assert 'adaptive_mult * pm_mult' in psrc
