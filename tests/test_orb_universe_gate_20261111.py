"""ORB universe gate parity (cell 1,733 class U, spec docs/orb_universe_gate_parity_spec_20261111.md).

Unit: trading.orb_universe_gate.admit on the four 1,733 shapes.
Integration: the live engine's 09:30 filter + scoring-time re-check, and the BT builder's per-symbol path.
Parity: the same inputs through the engine path and the builder path give the same Admission.
"""
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
import yaml

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from study_orb_features import builder_admission, diagnose_extract_failure
from trading import orb_universe_gate as G
from trading.orb_engine import CandidateState, ORBEngine, RangeData
from trading.stop_monitor import StopMonitor

CFG = {'min_gap_pct': 5.0, 'min_prev_volume': 500_000, 'min_price': 3.0, 'max_price': 30.0}


def _engine():
    cfg = yaml.safe_load((Path(__file__).parent.parent / 'orb.yaml').read_text())
    cfg['strategy']['enabled'] = True
    db = MagicMock(spec=Database)
    return ORBEngine(alpaca_client=MagicMock(spec=AlpacaClient), db=db,
                     stop_monitor=MagicMock(spec=StopMonitor), config=cfg)


class TestAdmitUnit:
    def test_astx_official_gap_1_95_is_rejected_even_with_phantom_snapshot_open(self):
        # ASTX 9/30: prev close 9.76, official open 9.95 (+1.95 %), snapshot read ~10.32 (+5.7 %)
        assert G.admit(9.76, 10.32, 1e6, 10.32, CFG, 'ASTX').ok            # the phantom input passes
        a = G.admit(9.76, 9.95, 1e6, 9.95, CFG, 'ASTX')                    # the official open does not
        assert (a.ok, a.reason) == (False, G.GAP_BELOW_MIN) and a.gap_pct == pytest.approx(1.9467, abs=1e-3)

    def test_aeye_3_07_is_rejected(self):
        a = G.admit(10.0, 10.307, 2e6, 10.307, CFG, 'AEYE')
        assert (a.ok, a.reason) == (False, G.GAP_BELOW_MIN)

    def test_exact_five_percent_edge_is_admitted_and_just_below_is_not(self):
        assert G.admit(20.0, 21.0, 1e6, 21.0, CFG).ok                      # exactly +5.0 %
        assert not G.admit(20.0, 20.99, 1e6, 20.99, CFG).ok

    def test_prev_volume_floor_price_band_test_ticker_and_bad_inputs(self):
        assert G.admit(10, 11, 499_999, 11, CFG).reason == G.PREV_VOLUME_BELOW_MIN
        assert G.admit(10, 11, 500_000, 11, CFG).ok
        assert G.admit(2.5, 2.9, 1e6, 2.9, CFG).reason == G.PRICE_OUT_OF_BAND
        assert G.admit(10, 11, 1e6, 11, CFG, 'ZVZZT').reason == G.TEST_TICKER
        assert G.admit(0, 11, 1e6, 11, CFG).reason == G.BAD_PREV_CLOSE
        assert G.admit(10, 0, 1e6, 0, CFG).reason == G.BAD_OPEN

    def test_none_arguments_are_not_checked(self):
        assert G.admit(10, 10.6, None, None, {'min_gap_pct': 5.0}).ok
        assert G.admit(10, 10.6, None, None, {'min_gap_pct': 5.0, 'max_gap_pct': 5.5}).reason == G.GAP_ABOVE_MAX


def _range(sym, open_px, prev_close_unused=None):
    ts = pd.Timestamp('2026-11-10T14:30:00Z')
    return RangeData(symbol=sym, range_high=open_px * 1.02, range_low=open_px * 0.99, range_volume=200_000,
                     range_avg_bar_range_pct=1.0, range_close=open_px * 1.01, range_start_ts=ts,
                     range_open=open_px)


def _providers(prev_close):
    return {'prev_day_bar': {'open': prev_close, 'high': prev_close * 1.02, 'low': prev_close * 0.98,
                             'close': prev_close, 'volume': 1_000_000},
            'daily_stats_20d': {'high_20d': prev_close * 1.5, 'range_pct_20d': 5.0, 'volume_20d': 1e6,
                                'return_volatility_20d': 0.0}}


class TestEngineIntegration:
    def test_0930_filter_uses_shared_gate(self):
        eng = _engine()
        assert eng._production_universe_cfg_admit('X', 10.0, 10.6, 1e6).ok
        a = eng._production_universe_cfg_admit('X', 10.0, 10.3, 1e6)
        assert (a.ok, a.reason) == (False, G.GAP_BELOW_MIN)

    def test_recheck_rejects_when_bar1_open_differs_from_snapshot_open(self, caplog):
        """Snapshot open 10.60 (+6 %) admitted the name at 09:30; bar1.open 10.20 (+2 %) -> rejected at scoring,
        reason logged at INFO, never scored."""
        eng = _engine()
        cand = CandidateState(symbol='PHAN')
        cand.range_data = _range('PHAN', 10.20)
        eng.candidates['PHAN'] = cand
        eng._symbol_pool['PHAN'] = 'production'
        with caplog.at_level(logging.INFO, logger='trading.orb_engine'):
            out = eng._run_pool_selection('production', ['PHAN'], set(), {'PHAN': _providers(10.0)},
                                          True, 0.0)
        assert out == []
        assert cand.rejected_reason == 'phantom_gap' and cand.composite is None
        assert any('PHAN phantom-gap reject (production)' in r.message and 'real gap 2.00%' in r.message
                   for r in caplog.records)

    def test_recheck_missing_prev_day_bar_warns_instead_of_failing_open_silently(self, caplog):
        eng = _engine()
        cand = CandidateState(symbol='NOPB')
        cand.range_data = _range('NOPB', 10.8)
        eng.candidates['NOPB'] = cand
        eng._symbol_pool['NOPB'] = 'production'
        with caplog.at_level(logging.WARNING, logger='trading.orb_engine'):
            eng._run_pool_selection('production', ['NOPB'], set(), {'NOPB': {'prev_day_bar': None,
                                    'daily_stats_20d': None}}, True, 0.0)
        assert any('NOPB phantom-gap re-check SKIPPED' in r.message for r in caplog.records)

    def test_final_ranges_logged_once_per_day(self, caplog):
        eng = _engine()
        c = CandidateState(symbol='AAA')
        c.range_data = _range('AAA', 10.0)
        eng.candidates['AAA'] = c
        with caplog.at_level(logging.INFO, logger='trading.orb_engine'):
            eng._log_final_ranges()
            eng._log_final_ranges()
        lines = [r.message for r in caplog.records if r.message.startswith('[ORB] final ranges:')]
        assert lines == ['[ORB] final ranges: AAA 10.2000/9.9000']


def _daily(rows):
    """rows: (symbol, date, open, close, volume) -> {symbol: frame} like study_orb_features.daily_by_sym."""
    df = pd.DataFrame(rows, columns=['symbol', 'bar_date', 'open', 'close', 'volume'])
    df['bar_date'] = pd.to_datetime(df['bar_date'])
    df['high'] = df['close'] * 1.01
    df['low'] = df['close'] * 0.99
    return {s: g.reset_index(drop=True) for s, g in df.groupby('symbol')}


DAILY = _daily([
    ('GOOD', '2026-11-09', 10.0, 10.0, 1_000_000), ('GOOD', '2026-11-10', 10.8, 10.9, 2_000_000),   # +8 %
    ('ASTX', '2026-11-09', 9.76, 9.76, 1_000_000), ('ASTX', '2026-11-10', 9.95, 9.6, 1_000_000),    # +1.95 %
    ('AEYE', '2026-11-09', 10.0, 10.0, 1_000_000), ('AEYE', '2026-11-10', 10.6, 10.7, 1_000_000),   # daily +6 %
])


class TestBuilderIntegration:
    def test_three_symbol_fixture_one_admitted_two_dropped_with_named_reasons(self):
        # feats['gap_pct'] is the bar1.open gap; AEYE's bar1 open is 3.07 % although the daily open is +6 %.
        feats = {'GOOD': {'gap_pct': 8.0}, 'ASTX': {'gap_pct': 1.95}, 'AEYE': {'gap_pct': 3.07}}
        res = {s: builder_admission(s, '2026-11-10', DAILY, feats[s]) for s in feats}
        assert all(a.ok for a in res['GOOD'])
        assert res['ASTX'][0].reason == 'gap_below_min'            # universe gate on the official open
        assert res['AEYE'][0].ok and not res['AEYE'][1].ok         # passes the daily open, fails the re-check
        assert res['AEYE'][1].reason == 'gap_below_min'

    def test_missing_daily_bar_is_a_named_drop(self):
        uni, rechk = builder_admission('NOPE', '2026-11-10', DAILY, {'gap_pct': 9.0})
        assert uni.reason == 'no_daily_bar_for_gate' and rechk.reason == 'no_daily_bar_for_gate'

    def test_extract_failure_reasons_are_named(self):
        assert diagnose_extract_failure(pd.DataFrame(), 'GOOD', '2026-11-10', DAILY) == 'no_intraday_bars'
        t0 = pd.Timestamp('2026-11-10T14:30:00Z')            # 09:30 ET
        bars = pd.DataFrame({'timestamp': [t0 + pd.Timedelta(minutes=i) for i in range(3)],
                             'open': 1.0, 'high': 1.0, 'low': 1.0, 'close': 1.0, 'volume': 1})
        assert diagnose_extract_failure(bars, 'GOOD', '2026-11-10', DAILY).startswith('range_incomplete_3_of_')


class TestParity:
    @pytest.mark.parametrize('prev_close,open_px,prev_vol', [
        (9.76, 9.95, 1e6), (9.76, 10.32, 1e6), (10.0, 10.5, 1e6), (10.0, 10.499, 1e6),
        (10.0, 11.0, 4e5), (2.0, 2.5, 1e6), (10.0, 10.6, 1e6),
    ])
    def test_engine_and_builder_paths_give_the_same_admission(self, prev_close, open_px, prev_vol):
        eng = _engine()
        cfg = {'min_gap_pct': eng.universe_min_gap_pct, 'min_prev_volume': eng.universe_min_prev_volume,
               'min_price': eng.universe_min_price, 'max_price': eng.universe_max_price}
        daily = _daily([('PAR', '2026-11-09', prev_close, prev_close, prev_vol),
                        ('PAR', '2026-11-10', open_px, open_px, prev_vol)])
        feats = {'gap_pct': (open_px - prev_close) / prev_close * 100.0}
        built_uni, built_rechk = builder_admission('PAR', '2026-11-10', daily, feats, cfg=cfg)
        live = eng._production_universe_cfg_admit('PAR', prev_close, open_px, prev_vol)
        assert built_uni == live
        cand = CandidateState(symbol='PAR')
        cand.range_data = _range('PAR', open_px)
        assert built_rechk.ok == eng._recheck_gap_admission(feats, cand, 5.0).ok


class TestRecheckLegacyParity20261111:
    """Regression (14 preplace tests broke at 2a55c82): the re-check re-derived the gap from
    providers['prev_day_bar'] / range_open, rejecting fixtures whose _compute_features is stubbed (feats gap 12 %,
    no prev_day_bar, range_open 0). Admission must equal the legacy ``gap_pct < floor`` on feats['gap_pct']."""

    @pytest.mark.parametrize('gap', [-3.0, 0.0, 4.999, 5.0, 5.001, 12.0, 80.0])
    def test_unit_gap_value_matches_legacy_comparison(self, gap):
        a = G.admit_gap_value(gap, {'min_gap_pct': 5.0}, symbol='X')
        assert a.ok == (not gap < 5.0)

    def test_unit_missing_gap_is_not_admitted_by_the_helper(self):
        assert not G.admit_gap_value(None, {'min_gap_pct': 5.0}).ok
        assert not G.admit_gap_value(float('nan'), {'min_gap_pct': 5.0}).ok

    @pytest.mark.parametrize('gap', [4.0, 5.0, 12.0])
    @pytest.mark.parametrize('providers', [
        {'prev_day_bar': None, 'daily_stats_20d': None}, {}, _providers(10.0)])
    def test_integration_engine_recheck_old_equals_new_whatever_the_providers(self, gap, providers):
        eng = _engine()
        cand = CandidateState(symbol='FIX')
        cand.range_data = _range('FIX', 0.0)  # range_open 0, as in the preplace fixtures
        new = eng._recheck_gap_admission({'gap_pct': gap}, cand, 5.0).ok
        assert new == (not gap < 5.0)

    def test_integration_scoring_keeps_candidate_with_stubbed_features_and_no_prev_day_bar(self, monkeypatch):
        eng = _engine()
        cand = CandidateState(symbol='DFDV')
        cand.range_data = _range('DFDV', 0.0)
        eng.candidates['DFDV'] = cand
        eng._symbol_pool['DFDV'] = 'production'
        monkeypatch.setattr(eng, '_compute_features', lambda c, prev_day_bar=None, daily_stats_20d=None:
                            {'gap_pct': 12.0})
        eng._run_pool_selection('production', ['DFDV'], set(),
                                {'DFDV': {'prev_day_bar': None, 'daily_stats_20d': None}}, True, 0.0)
        assert cand.rejected_reason != 'phantom_gap'
