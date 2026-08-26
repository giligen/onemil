"""Approach-band intake (2026-08-26, shadow-day-2 fix).

Shadow day 2 finding: every ignition trigger's prestage candidate was
discovered AT or AFTER its own cross (src=trigger_bar, late 2-49 min) —
nothing was ever stageable, because the scanner only fed the shadow at
the >=10% mover bar and the shadow dropped anything below trigger_pct.

The fix feeds sub-trigger sightings (approach band, default >=6%) into
the shadow ONLY when a prestage consumer is wired, resolves news for
level-parked pre-cross candidates, and re-fires prestage intake so the
scheduler can stage before the cross. These tests pin:
  - default OFF = byte-identical trigger-only gate
  - pre-cross candidate reaches prestage WITH resolved news, is parked,
    and never fires the engine trigger
  - the approach eval budget is separate (never starves trigger evals)
  - the scanner feeds the band only when the shadow exposes feed_min_pct
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from data_sources.alpaca_client import AlpacaClient
from trading.ignition_shadow import IgnitionShadow


def _shadow(tmp_path, **cfg):
    a = MagicMock(spec=AlpacaClient)
    a.get_latest_quote.return_value = {'bid_price': 10.0,
                                      'ask_price': 10.05,
                                      'bid_size': 5, 'ask_size': 7}
    ts = pd.date_range('2026-07-20 13:30', '2026-07-20 14:10',
                       freq='1min', tz='UTC')[:41]
    bars = pd.DataFrame({'timestamp': ts, 'open': [9.0] + [10.0] * 40,
                         'high': [9.1] + [10.4] * 40,
                         'low': [8.9] + [9.4] * 40,
                         'close': [9.05] + [10.2] * 40,
                         'volume': [10000] * 41})
    a.get_1min_bars.return_value = bars
    a.get_premarket_news_multi.return_value = {}
    s = IgnitionShadow(a, {'ignition_shadow': {'enabled': True, **cfg}},
                       log_dir=str(tmp_path))
    return s, a


def _flat_bars():
    """Bars where the +10% level (9.0*1.10=9.90) is NOT crossed."""
    return pd.DataFrame({'timestamp': pd.date_range(
        '2026-07-20 13:30', periods=10, freq='1min', tz='UTC'),
        'open': [9.0] * 10, 'high': [9.5] * 10, 'low': [8.9] * 10,
        'close': [9.4] * 10, 'volume': [10000] * 10})


def _fire(s, sym='IGNI', chg=7.0, gap=5.0, news=None, minute=(9, 50)):
    with patch('trading.ignition_shadow.datetime') as md:
        md.now.return_value = datetime(2026, 7, 20, minute[0] + 4,
                                       minute[1], tzinfo=timezone.utc)
        s.on_mover(sym, intraday_change_pct=chg, gap_pct=gap, price=9.55,
                   has_news=news, price_ts_utc=None)
    assert s.drain(10.0), 'shadow worker failed to drain queue'


def _recs(tmp_path):
    f = list(tmp_path.glob('ignition_shadow_*.jsonl'))
    if not f:
        return []
    return [json.loads(l) for l in f[0].read_text().splitlines()]


class TestApproachGate:
    def test_default_off_drops_below_trigger(self, tmp_path):
        """approach_min_pct=None (default): sub-trigger sighting is
        dropped at the gate even with a consumer — byte-identical."""
        s, a = _shadow(tmp_path)
        s.on_candidate = MagicMock()
        _fire(s, chg=7.0)
        assert _recs(tmp_path) == []
        s.on_candidate.assert_not_called()
        a.get_1min_bars.assert_not_called()

    def test_no_consumer_drops_even_with_threshold(self, tmp_path):
        """Threshold set but no prestage consumer: pre-cross sightings
        have no use — dropped, zero API cost."""
        s, a = _shadow(tmp_path)
        s.approach_min_pct = 6.0
        _fire(s, chg=7.0)
        assert _recs(tmp_path) == []
        a.get_1min_bars.assert_not_called()

    def test_below_band_still_dropped(self, tmp_path):
        s, a = _shadow(tmp_path)
        s.approach_min_pct = 6.0
        s.on_candidate = MagicMock()
        _fire(s, chg=4.0)
        assert _recs(tmp_path) == []
        s.on_candidate.assert_not_called()

    def test_open_flood_deferred_not_consumed(self, tmp_path):
        """Day-3 recalibration: approach sightings before 9:37 (<6 RTH
        bars -> guaranteed no_bars) are deferred WITHOUT marking seen or
        consuming budget; the re-sighting at 9:37+ evaluates normally."""
        s, a = _shadow(tmp_path)
        s.approach_min_pct = 6.0
        s.on_candidate = MagicMock()
        a.get_1min_bars.return_value = _flat_bars()
        _fire(s, chg=7.0, minute=(9, 35))
        assert _recs(tmp_path) == []
        assert 'IGNI' not in s._seen_today
        assert s._approach_evals_today == 0
        a.get_1min_bars.assert_not_called()
        _fire(s, chg=7.0, minute=(9, 40))      # re-sighting, bars exist
        r = _recs(tmp_path)
        assert r[-1]['verdict'] == 'skip_level_not_crossed'
        assert s._approach_evals_today == 1

    def test_default_approach_budget_is_400(self, tmp_path):
        s, _ = _shadow(tmp_path)
        assert s.max_approach_evals == 400

    def test_feed_min_pct_property(self, tmp_path):
        s, _ = _shadow(tmp_path)
        assert s.feed_min_pct is None            # no threshold
        s.approach_min_pct = 6.0
        assert s.feed_min_pct is None            # no consumer
        s.on_candidate = lambda rec: None
        assert s.feed_min_pct == 6.0
        s.approach_min_pct = 15.0                # never above trigger
        assert s.feed_min_pct == s.trigger_pct


class TestApproachIntake:
    def _armed(self, tmp_path, **cfg):
        s, a = _shadow(tmp_path, **cfg)
        s.approach_min_pct = 6.0
        s.on_candidate = MagicMock()
        s.on_trigger = MagicMock()
        return s, a

    def test_precross_candidate_fed_with_news_and_parked(self, tmp_path):
        s, a = self._armed(tmp_path)
        a.get_1min_bars.return_value = _flat_bars()
        a.get_premarket_news_multi.return_value = {
            'IGNI': {'n_articles': 2, 'headline': 'FDA nod'}}
        _fire(s, chg=7.0, news=None)
        r = _recs(tmp_path)
        assert r[-1]['verdict'] == 'skip_level_not_crossed'
        assert r[-1].get('approach_intake') is True
        # intake fired twice: pre-level-check hook + the news refire
        assert s.on_candidate.call_count == 2
        refire = s.on_candidate.call_args[0][0]
        assert refire['has_news'] is True
        assert refire['day_open'] == 9.0         # level computable
        assert 'IGNI' in s._await_level
        s.on_trigger.assert_not_called()         # pre-cross NEVER trades

    def test_newsless_refire_carries_false_not_none(self, tmp_path):
        """Resolved-to-no-news must reach prestage as False (no_news,
        working as designed) — not None (news_unknown, the day-2 bug)."""
        s, a = self._armed(tmp_path)
        a.get_1min_bars.return_value = _flat_bars()
        _fire(s, chg=7.0, news=None)
        refire = s.on_candidate.call_args[0][0]
        assert refire['has_news'] is False

    def test_parked_approach_then_cross_triggers(self, tmp_path):
        """The full arc: approach sighting parks pre-cross, the cross
        happens, a re-sighting re-finalizes into a real trigger."""
        s, a = self._armed(tmp_path)
        a.get_1min_bars.return_value = _flat_bars()
        a.get_premarket_news_multi.return_value = {
            'IGNI': {'n_articles': 1, 'headline': 'h'}}
        _fire(s, chg=7.0, news=None, minute=(9, 40))
        assert _recs(tmp_path)[-1]['verdict'] == 'skip_level_not_crossed'
        ts = pd.date_range('2026-07-20 13:30', periods=41, freq='1min',
                           tz='UTC')
        crossed = pd.DataFrame({'timestamp': ts,
                                'open': [9.0] + [10.0] * 40,
                                'high': [9.1] + [10.4] * 40,
                                'low': [8.9] + [9.4] * 40,
                                'close': [9.05] + [10.2] * 40,
                                'volume': [10000] * 41})
        a.get_1min_bars.return_value = crossed
        with patch('trading.ignition_shadow.datetime') as md:
            md.now.return_value = datetime(2026, 7, 20, 13, 55,
                                           tzinfo=timezone.utc)
            s.on_mover('IGNI', intraday_change_pct=12.0, gap_pct=5.0,
                       price=10.35, has_news=None, price_ts_utc=None)
        assert s.drain(10.0)
        assert _recs(tmp_path)[-1]['verdict'] == 'SHADOW_TRIGGER'

    def test_approach_discovered_cross_still_triggers(self, tmp_path):
        """USDE class: scanner approximation says 7% but the bars show
        the cross already happened — full trigger path fires (earlier
        discovery of a BT-taken trigger = parity improvement)."""
        s, a = self._armed(tmp_path)   # default fixture bars ARE crossed
        a.get_premarket_news_multi.return_value = {
            'IGNI': {'n_articles': 1, 'headline': 'h'}}
        _fire(s, chg=7.0, news=None)
        r = _recs(tmp_path)
        assert r[-1]['verdict'] == 'SHADOW_TRIGGER'
        assert r[-1].get('approach_intake') is True
        s.on_trigger.assert_called_once()

    def test_approach_budget_separate_from_trigger_budget(self, tmp_path):
        s, a = self._armed(tmp_path, max_approach_evals_per_day=1)
        a.get_1min_bars.return_value = _flat_bars()
        _fire(s, sym='AAA', chg=7.0)
        _fire(s, sym='BBB', chg=7.0)
        r = _recs(tmp_path)
        assert r[-1]['symbol'] == 'BBB'
        assert r[-1]['verdict'] == 'skip_approach_eval_cap'
        assert s._approach_evals_today == 1
        assert s._evals_today == 0               # trigger budget intact
        # trigger evals still work after the approach cap is exhausted
        ts = pd.date_range('2026-07-20 13:30', periods=41, freq='1min',
                           tz='UTC')
        a.get_1min_bars.return_value = pd.DataFrame({
            'timestamp': ts, 'open': [9.0] + [10.0] * 40,
            'high': [9.1] + [10.4] * 40, 'low': [8.9] + [9.4] * 40,
            'close': [9.05] + [10.2] * 40, 'volume': [10000] * 41})
        _fire(s, sym='CCC', chg=15.0, gap=2.0, news=True)
        assert _recs(tmp_path)[-1]['verdict'] == 'SHADOW_TRIGGER'

    def test_day_rollover_resets_approach_budget(self, tmp_path):
        s, a = self._armed(tmp_path)
        s._approach_evals_today = 99
        s._roll_day('2026-07-21')
        assert s._approach_evals_today == 0


class TestScannerApproachFeed:
    def _scanner(self):
        from data_sources.news_provider import NewsProvider
        from persistence.database import Database
        from scanner.criteria import ScannerCriteria
        from scanner.realtime_scanner import RealtimeScanner
        alpaca = MagicMock(spec=AlpacaClient)
        news = MagicMock(spec=NewsProvider)
        news.has_interesting_news.return_value = (False, '')
        news.classify_news.return_value = {'has_news': False,
                                           'catalyst': None,
                                           'headline': '', 'reason': ''}
        db = MagicMock(spec=Database)
        sc = RealtimeScanner(alpaca_client=alpaca, news_provider=news,
                             db=db, criteria=ScannerCriteria(),
                             verbose=False)
        return sc, alpaca

    def _wire(self, sc, alpaca, price):
        import pytz
        from datetime import datetime as rdt
        sc._universe = [{'symbol': 'APPR', 'price_close': 4.0,
                         'company_name': 'Appr Co',
                         'float_shares': 2_000_000}]
        alpaca.get_latest_trades.return_value = {
            'APPR': {'price': price,
                     'timestamp': '2026-07-20T13:49:45+00:00'}}
        now_et = rdt.now(pytz.timezone('US/Eastern'))
        bar_ts = now_et.replace(minute=(now_et.minute // 15) * 15,
                                second=0, microsecond=0)
        alpaca.get_current_bars.return_value = {
            'APPR': {'volume': 100_000, 'timestamp': bar_ts,
                     'high': price, 'low': price * 0.97}}
        bucket = f"{now_et.hour:02d}:{(now_et.minute // 15) * 15:02d}"
        sc._volume_profiles = {'APPR': {bucket: 10_000}}

    def test_approach_band_reaches_shadow(self, tmp_path):
        sc, alpaca = self._scanner()
        self._wire(sc, alpaca, price=4.30)        # +7.5% — sub-mover
        sc.ignition_shadow = MagicMock(spec=IgnitionShadow)
        sc.ignition_shadow.feed_min_pct = 6.0
        sc._run_intraday_cycle()
        sc.ignition_shadow.on_mover.assert_called_once()
        _, kw = sc.ignition_shadow.on_mover.call_args
        assert kw['intraday_change_pct'] == pytest.approx(7.5)

    def test_no_feed_min_keeps_legacy_threshold(self, tmp_path):
        sc, alpaca = self._scanner()
        self._wire(sc, alpaca, price=4.30)        # +7.5% — sub-mover
        sc.ignition_shadow = MagicMock(spec=IgnitionShadow)
        sc.ignition_shadow.feed_min_pct = None    # prestage not wired
        sc._run_intraday_cycle()
        sc.ignition_shadow.on_mover.assert_not_called()

    def test_mover_still_feeds_regardless_of_band(self, tmp_path):
        sc, alpaca = self._scanner()
        self._wire(sc, alpaca, price=5.0)         # +25% — mover
        sc.ignition_shadow = MagicMock(spec=IgnitionShadow)
        sc.ignition_shadow.feed_min_pct = None
        sc._run_intraday_cycle()
        sc.ignition_shadow.on_mover.assert_called_once()
