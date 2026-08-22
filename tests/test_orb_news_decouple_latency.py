"""News-fetch decoupling + latency tripwire — B+ RESTART 2026-08-15.

P0 interaction bug (coordinator): B+ turns the PM/news mult OFF but keeps the
catalyst veto ON. The news fetch must NOT be gated on the PM path alone, or
`_get_has_news` returns None for everything and the veto silently fail-opens.
`_news_fetch_needed()` = (pm_news_gate consumer) OR catalyst_veto — the single
source of truth.

Latency (item H): the PM-mult short-circuit (return 1.0 before any news fetch)
means NO blocking news call happens between the 9:35 signal and submit; the
off-critical-path prefetch (>=9:31) is still allowed.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.orb_engine import ORBEngine
from trading.stop_monitor import StopMonitor


@pytest.fixture
def base_cfg():
    cfg = yaml.safe_load(open(Path(__file__).parent.parent / 'orb.yaml'))
    cfg['strategy']['enabled'] = True
    return cfg


def _engine(cfg):
    return ORBEngine(alpaca_client=MagicMock(spec=AlpacaClient),
                     db=MagicMock(spec=Database),
                     stop_monitor=MagicMock(spec=StopMonitor), config=cfg)


class TestNewsFetchNeeded:
    def test_bplus_pm_off_catalyst_on_needs_news(self, base_cfg):
        """The shipping B+ combo: pm OFF, catalyst ON -> MUST fetch news."""
        e = _engine(base_cfg)
        assert e.pm_mult_enabled is False       # B+ default
        assert e.catalyst_veto_enabled is True
        assert e._news_fetch_needed() is True

    def test_both_off_no_news(self, base_cfg):
        base_cfg['filter']['catalyst_veto']['enabled'] = False
        e = _engine(base_cfg)
        assert e.pm_mult_enabled is False
        assert e.catalyst_veto_enabled is False
        assert e._news_fetch_needed() is False

    def test_pm_on_newsgate_on_needs_news(self, base_cfg):
        base_cfg['sizing']['pm_dollar_vol_mult']['enabled'] = True
        base_cfg['filter']['catalyst_veto']['enabled'] = False
        e = _engine(base_cfg)
        assert e._news_fetch_needed() is True

    def test_pm_on_newsgate_off_catalyst_off_no_news(self, base_cfg):
        base_cfg['sizing']['pm_dollar_vol_mult']['enabled'] = True
        base_cfg['sizing']['pm_dollar_vol_mult']['news_gate'] = False
        base_cfg['filter']['catalyst_veto']['enabled'] = False
        e = _engine(base_cfg)
        assert e._news_fetch_needed() is False


class TestPrefetchTriggersNewsForVeto:
    def _force_et(self, monkeypatch, hh=14, mm=0):
        """Freeze the module clock at a UTC time that maps to >= 9:31 ET."""
        import trading.orb_engine as mod
        fixed = datetime(2026, 8, 17, hh, mm, tzinfo=timezone.utc)  # ~10:00 ET

        class _DT(datetime):
            @classmethod
            def now(cls, tz=None):
                return fixed.astimezone(tz) if tz else fixed
        monkeypatch.setattr(mod, 'datetime', _DT)

    def test_pm_off_catalyst_on_fetches_news_once(self, base_cfg, monkeypatch):
        e = _engine(base_cfg)
        e.candidates = {'AAA': MagicMock(), 'BBB': MagicMock()}
        self._force_et(monkeypatch)
        calls = {'n': 0}

        import trading.orb_engine as mod

        def _fake_fetch():
            calls['n'] += 1
            e._news_flags = {s: {'n_articles': 1, 'headline': 'x'}
                             for s in e.candidates}
            # stamp with the engine's (frozen) notion of "today"
            e._news_fetch_done_day = mod.datetime.now(timezone.utc).date()
            e._news_refresh_done_day = e._news_fetch_done_day
        monkeypatch.setattr(e, '_fetch_news_flags', _fake_fetch)
        monkeypatch.setattr(e, '_fetch_pm_dollar_vols', lambda: None)
        e._maybe_prefetch_pm()
        e._maybe_prefetch_pm()   # second call must NOT refetch (done-stamped)
        assert calls['n'] == 1
        # the veto now sees REAL has_news (not None)
        assert e._get_has_news('AAA') is True

    def test_both_off_no_news_fetch(self, base_cfg, monkeypatch):
        base_cfg['filter']['catalyst_veto']['enabled'] = False
        e = _engine(base_cfg)
        e.candidates = {'AAA': MagicMock()}
        self._force_et(monkeypatch)
        fetch = MagicMock()
        monkeypatch.setattr(e, '_fetch_news_flags', fetch)
        monkeypatch.setattr(e, '_fetch_pm_dollar_vols', MagicMock())
        e._maybe_prefetch_pm()
        fetch.assert_not_called()


def _now_date():
    return datetime.now(timezone.utc).date()


class TestEntryPathNoBlockingNews:
    """Latency (item H): during the 9:35 signal->submit window, the PM-mult
    short-circuit means NO news API call is made. Prefetch (>=9:31) is the
    only place news is fetched, and it is off the critical path."""

    def test_get_pm_mult_short_circuits_before_news(self, base_cfg):
        e = _engine(base_cfg)   # pm OFF in B+
        e.alpaca.get_premarket_news_multi = MagicMock()
        assert e._get_pm_mult('AAA') == 1.0
        e.alpaca.get_premarket_news_multi.assert_not_called()

    def test_catalyst_veto_uses_cached_news_no_fetch(self, base_cfg):
        e = _engine(base_cfg)
        e.alpaca.get_premarket_news_multi = MagicMock()
        from trading.orb_engine import CandidateState
        e.candidates = {'AAA': CandidateState(symbol='AAA')}
        # pre-warmed cache state (as prefetch would leave it)
        e._news_flags = {'AAA': {'n_articles': 0, 'headline': ''}}
        e._anchor_cache = {'AAA': None}
        cand = e.candidates['AAA']
        # veto path must not hit the news API (cache lookups only)
        e._catalyst_veto_reject(cand)
        e.alpaca.get_premarket_news_multi.assert_not_called()


class TestLatencyTripwire:
    def test_config_default(self, base_cfg):
        e = _engine(base_cfg)
        assert e.latency_warn_secs == 10.0

    def test_warns_when_late(self, base_cfg, monkeypatch):
        e = _engine(base_cfg)
        e._notify = MagicMock()
        # freeze ET well past 9:35 (>10s)
        import trading.orb_engine as mod
        late = datetime(2026, 8, 17, 13, 36, 30, tzinfo=timezone.utc)  # 9:36:30 ET

        def _etnow(self):
            from zoneinfo import ZoneInfo
            return late.astimezone(ZoneInfo('America/New_York'))
        monkeypatch.setattr(mod.ORBEngine, '_et_now', _etnow)
        e._check_first_submit_latency()
        assert e._notify.called
        assert e._first_submit_latency_logged is True

    def test_no_warn_when_prompt(self, base_cfg, monkeypatch):
        e = _engine(base_cfg)
        e._notify = MagicMock()
        import trading.orb_engine as mod
        prompt = datetime(2026, 8, 17, 13, 35, 3, tzinfo=timezone.utc)  # 9:35:03 ET

        def _etnow(self):
            from zoneinfo import ZoneInfo
            return prompt.astimezone(ZoneInfo('America/New_York'))
        monkeypatch.setattr(mod.ORBEngine, '_et_now', _etnow)
        e._check_first_submit_latency()
        e._notify.assert_not_called()

    def test_fires_once(self, base_cfg, monkeypatch):
        e = _engine(base_cfg)
        e._notify = MagicMock()
        import trading.orb_engine as mod
        late = datetime(2026, 8, 17, 13, 40, 0, tzinfo=timezone.utc)

        def _etnow(self):
            from zoneinfo import ZoneInfo
            return late.astimezone(ZoneInfo('America/New_York'))
        monkeypatch.setattr(mod.ORBEngine, '_et_now', _etnow)
        e._check_first_submit_latency()
        e._check_first_submit_latency()
        assert e._notify.call_count == 1
