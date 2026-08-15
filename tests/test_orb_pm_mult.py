"""Premarket dollar-volume sizing mult (2026-07-04 ship) + news gate
(2026-07-10 A2 ship, research/orb_news_catalyst_jul2026.md).

Pins the shared math (upsize-only, fail-open on BOTH channels), engine
wiring (once-per-day batch fetches for PM$ AND news, mult reaches the
planner, pattern_data attribution), and config/env knobs.
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
    DEFAULT_HIGH_CUT_USD, DEFAULT_HIGH_MULT, DEFAULT_HIGH_MULT_NEWS,
    LEGACY_HIGH_MULT, compute_pm_dollar_vol, pm_size_multiplier,
)
from trading.stop_monitor import StopMonitor


class TestPmSizeMultiplierNewsGate:
    """Default (news_gate=True) semantics — the shipped A2 variant."""

    def test_above_cut_with_news_boosts_news_mult(self):
        assert pm_size_multiplier(6_000_000, has_news=True) \
            == DEFAULT_HIGH_MULT_NEWS

    def test_above_cut_without_news_gets_high_mult(self):
        """The flat bucket: de-boosted default 1.0."""
        assert pm_size_multiplier(6_000_000, has_news=False) \
            == DEFAULT_HIGH_MULT == 1.0

    def test_above_cut_news_unknown_fails_open_no_boost(self):
        """News fetch failed -> never boost blind."""
        assert pm_size_multiplier(6_000_000, has_news=None) \
            == DEFAULT_HIGH_MULT

    def test_below_cut_news_never_boosts(self):
        """News alone is a TRAP (negative all 3 eras) — no boost without
        the PM$ leg."""
        assert pm_size_multiplier(1_000_000, has_news=True) == 1.0

    def test_at_cut_neutral(self):
        """Boundary is strictly > (matches the study's tercile mask)."""
        assert pm_size_multiplier(DEFAULT_HIGH_CUT_USD, has_news=True) == 1.0

    def test_none_pm_fails_open(self):
        assert pm_size_multiplier(None, has_news=True) == 1.0

    def test_garbage_fails_open(self):
        assert pm_size_multiplier('x', has_news=True) == 1.0

    def test_never_downsizes(self):
        """Upsize-only invariant: no input may produce mult < 1.0."""
        for v in (None, 0, 1, 1e12, -5, float('nan')):
            for hn in (True, False, None):
                assert pm_size_multiplier(v, has_news=hn) >= 1.0


class TestPmSizeMultiplierLegacy:
    """news_gate=False + LEGACY_HIGH_MULT reproduces the pre-2026-07-10
    ungated x1.5 byte-identically (BT rollback path ORB_PM_NEWS_GATE=0)."""

    def test_above_cut_boosts_legacy(self):
        assert pm_size_multiplier(6_000_000, high_mult=LEGACY_HIGH_MULT,
                                  news_gate=False) == 1.5

    def test_news_ignored_when_gate_off(self):
        for hn in (True, False, None):
            assert pm_size_multiplier(6_000_000, high_mult=LEGACY_HIGH_MULT,
                                      has_news=hn, news_gate=False) == 1.5

    def test_below_cut_neutral(self):
        assert pm_size_multiplier(1_000_000, high_mult=LEGACY_HIGH_MULT,
                                  news_gate=False) == 1.0


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


PM_BARS_10M = pd.DataFrame([{
    'timestamp': pd.Timestamp('2026-07-06 09:00',
                              tz='America/New_York').tz_convert('UTC'),
    'open': 5.0, 'high': 5.0, 'low': 5.0, 'close': 5.0,
    'volume': 2_000_000,   # 2M sh x $5 = $10M > cut
}])


def _make_engine(monkeypatch=None, **cfg_overrides):
    with open(Path(__file__).parent.parent / 'orb.yaml') as f:
        cfg = yaml.safe_load(f)
    cfg['strategy']['enabled'] = True
    # This file tests the PM/news mult behavior, which B+ 2026-08-15 ships
    # DISABLED by default. Force it ON here (overrides still win) so the
    # feature-under-test is exercised; test_env_disable etc. re-disable it.
    cfg['sizing']['pm_dollar_vol_mult']['enabled'] = True
    for k, v in cfg_overrides.items():
        cfg['sizing']['pm_dollar_vol_mult'][k] = v
    return ORBEngine(alpaca_client=MagicMock(spec=AlpacaClient),
                     db=MagicMock(spec=Database),
                     stop_monitor=MagicMock(spec=StopMonitor), config=cfg)


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.delenv('ORB_PM_MULT', raising=False)
    monkeypatch.delenv('ORB_PM_NEWS_GATE', raising=False)
    return _make_engine()


def _mock_feeds(engine, pm_map=None, news_map=None,
                asset_name='Test Industries Inc Common Stock'):
    engine.alpaca.get_premarket_1min_bars_multi = MagicMock(
        return_value=pm_map or {})
    engine.alpaca.get_premarket_news_multi = MagicMock(
        return_value=news_map or {})
    # class rule: newsy symbols get classified; default mock = plain stock.
    # Blank the offline map so tests are hermetic from the shipped 33K CSV
    # (mock symbols like AAA collide with real tickers in it).
    engine._class_map = {}
    engine.alpaca.get_asset_name = MagicMock(return_value=asset_name)


class TestEngineWiring:
    def test_config_loaded(self, engine):
        assert engine.pm_mult_enabled is True
        assert engine.pm_mult_high_cut == 5816688
        assert engine.pm_mult_high == 1.0
        assert engine.pm_mult_high_news == 2.0
        assert engine.pm_news_gate is True

    def test_env_disable(self, monkeypatch):
        monkeypatch.setenv('ORB_PM_MULT', '0')
        eng = _make_engine()
        assert eng.pm_mult_enabled is False
        assert eng._get_pm_mult('ANY') == 1.0

    def test_env_news_gate_disable(self, monkeypatch):
        monkeypatch.delenv('ORB_PM_MULT', raising=False)
        monkeypatch.setenv('ORB_PM_NEWS_GATE', '0')
        eng = _make_engine()
        assert eng.pm_news_gate is False

    def test_batch_fetch_once_per_day_both_feeds(self, engine):
        engine.build_universe(source_loader=lambda: ['AAA', 'BBB'])
        _mock_feeds(engine,
                    pm_map={'AAA': PM_BARS_10M, 'BBB': PM_BARS_10M.iloc[:0]},
                    news_map={'AAA': {'n_articles': 2, 'headline': 'AAA pops'},
                              'BBB': {'n_articles': 0, 'headline': ''}})
        m1 = engine._get_pm_mult('AAA')   # $10M + news -> 2.0
        m2 = engine._get_pm_mult('BBB')   # no PM bars -> None -> 1.0
        m3 = engine._get_pm_mult('AAA')
        assert m1 == 2.0 and m2 == 1.0 and m3 == 2.0
        assert engine.alpaca.get_premarket_1min_bars_multi.call_count == 1
        assert engine.alpaca.get_premarket_news_multi.call_count == 1

    def test_above_cut_without_news_not_boosted(self, engine):
        engine.build_universe(source_loader=lambda: ['AAA'])
        _mock_feeds(engine, pm_map={'AAA': PM_BARS_10M},
                    news_map={'AAA': {'n_articles': 0, 'headline': ''}})
        assert engine._get_pm_mult('AAA') == 1.0

    def test_news_fetch_failure_fails_open_no_boost(self, engine, caplog):
        """PM$ above cut but news API down -> no boost, loud WARNING."""
        import logging
        engine.build_universe(source_loader=lambda: ['AAA'])
        engine.alpaca.get_premarket_1min_bars_multi = MagicMock(
            return_value={'AAA': PM_BARS_10M})
        engine.alpaca.get_premarket_news_multi = MagicMock(
            side_effect=RuntimeError('news api down'))
        with caplog.at_level(logging.WARNING):
            assert engine._get_pm_mult('AAA') == 1.0
        assert any('news' in r.message.lower() and 'fail-open' in r.message
                   for r in caplog.records)

    def test_pm_fetch_failure_fails_open(self, engine):
        engine.build_universe(source_loader=lambda: ['AAA'])
        engine.alpaca.get_premarket_1min_bars_multi = MagicMock(
            side_effect=RuntimeError('api down'))
        engine.alpaca.get_premarket_news_multi = MagicMock(
            return_value={'AAA': {'n_articles': 3, 'headline': 'x'}})
        assert engine._get_pm_mult('AAA') == 1.0

    def test_news_gate_off_uses_high_mult_ignores_news(self, monkeypatch):
        """Gate off: no news fetch at all, above-cut gets high_mult."""
        monkeypatch.delenv('ORB_PM_MULT', raising=False)
        monkeypatch.delenv('ORB_PM_NEWS_GATE', raising=False)
        eng = _make_engine(news_gate=False, high_mult=1.5)
        eng.build_universe(source_loader=lambda: ['AAA'])
        _mock_feeds(eng, pm_map={'AAA': PM_BARS_10M})
        assert eng._get_pm_mult('AAA') == 1.5
        eng.alpaca.get_premarket_news_multi.assert_not_called()

    def test_late_batch_news_merge(self, engine):
        """Universe arrives in batches — the news fetch must cover the
        late batch like the PM fetch does (2026-07-07 BEZ/IREZ lesson)."""
        engine.build_universe(source_loader=lambda: ['AAA'])
        _mock_feeds(engine, pm_map={'AAA': PM_BARS_10M},
                    news_map={'AAA': {'n_articles': 1, 'headline': 'x'}})
        assert engine._get_pm_mult('AAA') == 2.0
        engine.build_universe(source_loader=lambda: ['AAA', 'CCC'])
        engine.alpaca.get_premarket_1min_bars_multi = MagicMock(
            return_value={'CCC': PM_BARS_10M})
        engine.alpaca.get_premarket_news_multi = MagicMock(
            return_value={'CCC': {'n_articles': 1, 'headline': 'y'}})
        # late batches are covered by the >=9:31 prefetch tick calling the
        # fetchers again (the lazy _get_pm_mult path only re-fetches on
        # day change — by design); drive the merge logic directly.
        engine._fetch_pm_dollar_vols()
        engine._fetch_news_flags()
        assert engine._get_pm_mult('CCC') == 2.0
        # only the MISSING symbol was fetched in round 2
        assert engine.alpaca.get_premarket_news_multi.call_args.args[0] \
            == ['CCC']

    def test_uses_premarket_method_not_clamped_one(self, engine):
        """2026-07-06 incident: get_1min_bars_multi clamps to 9:30 open by
        design, silently starving PM mult. The engine must use the
        premarket-specific method."""
        engine.build_universe(source_loader=lambda: ['AAA'])
        _mock_feeds(engine)
        engine.alpaca.get_1min_bars_multi = MagicMock(return_value={})
        engine._get_pm_mult('AAA')
        engine.alpaca.get_premarket_1min_bars_multi.assert_called_once()
        engine.alpaca.get_1min_bars_multi.assert_not_called()

    def test_prefetch_gated_by_et_time(self, engine):
        """_maybe_prefetch_pm fires at >=9:31 ET, not before — and warms
        BOTH the PM$ and news caches."""
        from datetime import datetime, timezone
        from unittest.mock import patch
        engine.build_universe(source_loader=lambda: ['AAA'])
        _mock_feeds(engine)
        with patch('trading.orb_engine.datetime') as mdt:
            mdt.now.return_value = datetime(2026, 7, 7, 13, 25, 0,
                                            tzinfo=timezone.utc)  # 9:25 ET
            engine._maybe_prefetch_pm()
            engine.alpaca.get_premarket_1min_bars_multi.assert_not_called()
            engine.alpaca.get_premarket_news_multi.assert_not_called()
            mdt.now.return_value = datetime(2026, 7, 7, 13, 32, 0,
                                            tzinfo=timezone.utc)  # 9:32 ET
            engine._maybe_prefetch_pm()
            engine.alpaca.get_premarket_1min_bars_multi.assert_called_once()
            engine.alpaca.get_premarket_news_multi.assert_called_once()

    def test_newsy_wrapper_not_boosted_deliberate_rule(self, engine):
        """2026-07-11 class rule: a wrapper with articles tagged to its
        own ticker must NOT get the news boost — wrappers' newsy days are
        the crowding cell (negative all 3 eras)."""
        engine.build_universe(source_loader=lambda: ['WRPX'])
        _mock_feeds(engine, pm_map={'WRPX': PM_BARS_10M},
                    news_map={'WRPX': {'n_articles': 3, 'headline': 'x'}},
                    asset_name='Tradr 2X Long WRAP Daily ETF')
        assert engine._get_pm_mult('WRPX') == 1.0
        assert engine._asset_class['WRPX'] == 'wrapper'

    def test_newsy_unknown_class_not_boosted(self, engine):
        """Name fetch fails -> unknown -> never boost blind."""
        engine.build_universe(source_loader=lambda: ['NEWCO'])
        _mock_feeds(engine, pm_map={'NEWCO': PM_BARS_10M},
                    news_map={'NEWCO': {'n_articles': 1, 'headline': 'x'}})
        engine.alpaca.get_asset_name = MagicMock(return_value=None)
        assert engine._get_pm_mult('NEWCO') == 1.0
        assert engine._asset_class['NEWCO'] == 'unknown'

    def test_lev_family_wrapper_needs_no_api(self, engine):
        """Known leveraged-family symbols classify without a name fetch."""
        from trading.orb_correlation import LEVERAGED_SHORT_ALL
        sym = LEVERAGED_SHORT_ALL[0]
        engine.build_universe(source_loader=lambda: [sym])
        _mock_feeds(engine, pm_map={sym: PM_BARS_10M},
                    news_map={sym: {'n_articles': 2, 'headline': 'x'}})
        engine.alpaca.get_asset_name = MagicMock(return_value=None)
        assert engine._get_pm_mult(sym) == 1.0
        engine.alpaca.get_asset_name.assert_not_called()

    def test_class_map_hit_needs_no_api(self, engine):
        """Symbols in the shipped 33K map classify offline (CRCA)."""
        engine.build_universe(source_loader=lambda: ['CRCA'])
        _mock_feeds(engine, pm_map={'CRCA': PM_BARS_10M},
                    news_map={'CRCA': {'n_articles': 2, 'headline': 'x'}})
        engine._class_map = None   # use the REAL shipped map (that's the test)
        engine.alpaca.get_asset_name = MagicMock(return_value=None)
        assert engine._get_pm_mult('CRCA') == 1.0        # wrapper via map
        engine.alpaca.get_asset_name.assert_not_called()

    def test_933_lag_refresh_upgrades_no_news_only(self, engine):
        """Benzinga indexing-lag pass: at >=9:33, no-news symbols are
        re-fetched ONCE; upgrades are applied, downgrades impossible."""
        from datetime import datetime, timezone
        from unittest.mock import patch
        engine.build_universe(source_loader=lambda: ['AAA', 'BBB'])
        _mock_feeds(engine,
                    pm_map={'AAA': PM_BARS_10M, 'BBB': PM_BARS_10M},
                    news_map={'AAA': {'n_articles': 0, 'headline': ''},
                              'BBB': {'n_articles': 1, 'headline': 'x'}})
        with patch('trading.orb_engine.datetime') as mdt:
            mdt.now.return_value = datetime(2026, 7, 7, 13, 32, 0,
                                            tzinfo=timezone.utc)  # 9:32 ET
            engine._maybe_prefetch_pm()
            assert engine._get_has_news('AAA') is False
            # 9:33: AAA's article has now indexed
            engine.alpaca.get_premarket_news_multi = MagicMock(
                return_value={'AAA': {'n_articles': 1, 'headline': 'late'}})
            mdt.now.return_value = datetime(2026, 7, 7, 13, 33, 30,
                                            tzinfo=timezone.utc)  # 9:33 ET
            engine._maybe_prefetch_pm()
            assert engine._get_has_news('AAA') is True
            assert engine._get_has_news('BBB') is True   # untouched
            # only stale symbols re-fetched, and only once per day
            engine.alpaca.get_premarket_news_multi.assert_called_once_with(
                ['AAA'])
            engine._maybe_prefetch_pm()
            engine.alpaca.get_premarket_news_multi.assert_called_once()

    def test_933_pass_dumps_snapshot_for_eod_audit(self, engine, tmp_path,
                                                   monkeypatch):
        """After the lag pass the live news view must be on disk — it is
        the EoD lag audit's ground truth."""
        import json
        from datetime import datetime, timezone
        from unittest.mock import patch
        engine._news_snapshot_dir = str(tmp_path / 'logs')
        (tmp_path / 'logs').mkdir()
        engine.build_universe(source_loader=lambda: ['AAA'])
        _mock_feeds(engine, pm_map={'AAA': PM_BARS_10M},
                    news_map={'AAA': {'n_articles': 0, 'headline': ''}})
        with patch('trading.orb_engine.datetime') as mdt:
            mdt.now.return_value = datetime(2026, 7, 7, 13, 34, 0,
                                            tzinfo=timezone.utc)  # 9:34 ET
            engine._maybe_prefetch_pm()
        snap = json.loads(
            (tmp_path / 'logs' / 'orb_news_flags_2026-07-07.json').read_text())
        assert snap['flags'] == {'AAA': 0}
        assert snap['pm_dollar_vols']['AAA'] == pytest.approx(10_000_000)

    def test_933_refresh_failure_keeps_931_flags(self, engine):
        from datetime import datetime, timezone
        from unittest.mock import patch
        engine.build_universe(source_loader=lambda: ['AAA'])
        _mock_feeds(engine, pm_map={'AAA': PM_BARS_10M},
                    news_map={'AAA': {'n_articles': 0, 'headline': ''}})
        with patch('trading.orb_engine.datetime') as mdt:
            mdt.now.return_value = datetime(2026, 7, 7, 13, 32, 0,
                                            tzinfo=timezone.utc)
            engine._maybe_prefetch_pm()
            engine.alpaca.get_premarket_news_multi = MagicMock(
                side_effect=RuntimeError('down'))
            mdt.now.return_value = datetime(2026, 7, 7, 13, 34, 0,
                                            tzinfo=timezone.utc)
            engine._maybe_prefetch_pm()   # must not raise
            assert engine._get_has_news('AAA') is False

    def test_planner_receives_pm_mult(self, engine):
        """Source-level pin: the submit loop passes pm_mult to planner.build
        and the planner stacks it into position_dollars."""
        import inspect
        from trading import orb_engine as em, orb_planner as pm
        src = inspect.getsource(em.ORBEngine.check_entries)
        assert 'pm_mult=self._get_pm_mult(sym)' in src
        psrc = inspect.getsource(pm.OrbTradePlanner.build)
        assert 'adaptive_mult * pm_mult' in psrc


class TestBtLiveParity:
    """BT (study_orb_pipeline_static_lock.py) and live (orb_engine) must
    call the SAME shared helper with the SAME gate semantics."""

    def test_bt_imports_shared_helper_with_news_kwargs(self):
        src = Path(Path(__file__).parent.parent
                   / 'study_orb_pipeline_static_lock.py').read_text()
        assert 'pm_size_multiplier' in src
        assert 'has_news=_news_map.get(k)' in src
        assert 'ORB_PM_NEWS_GATE' in src

    def test_live_passes_tristate_news(self):
        import inspect
        from trading import orb_engine as em
        src = inspect.getsource(em.ORBEngine._get_pm_mult)
        assert 'has_news=has_news' in src
        assert 'news_gate=self.pm_news_gate' in src

    def test_unknown_news_maps_to_none_both_sides(self):
        """BT unknown symbol-day -> dict.get -> None; live failed fetch ->
        None. The helper treats None identically (no boost)."""
        assert pm_size_multiplier(9e6, has_news=None) \
            == pm_size_multiplier(9e6, has_news=False)
