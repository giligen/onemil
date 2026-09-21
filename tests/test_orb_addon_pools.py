"""Unit/integration/parity tests for ORB add-on pools (PREREG_LIVE_UNION.md,
owner GO 2026-09-21): production universe ∪ two exploration-tier pools the
production gap/price thresholds exclude, each run through the UNCHANGED
selection chain SEPARATELY and unioned subject to the shared slot cap.

Style reference: tests/test_orb_engine.py fixtures (`engine`, `mock_alpaca`,
`mock_db`, `orb_cfg`) and tests/test_orb_touchgo_parity.py.
"""
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from pathlib import Path

import pandas as pd
import pytest
import yaml

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.orb_engine import ORBEngine, RangeData, CandidateState
from trading.stop_monitor import StopMonitor


ADDON_CFG = {
    'enabled': True,
    'dry_run': True,
    'pools': [
        {'name': 'addon_gap4', 'min_gap_pct': 4.0, 'max_gap_pct': 5.0,
         'min_price': 3.0, 'max_price': 30.0},
        {'name': 'addon_p30', 'min_gap_pct': 3.0, 'max_gap_pct': 5.0,
         'min_price': 30.0, 'max_price': 50.0},
    ],
}


def _base_cfg():
    with open(Path(__file__).parent.parent / 'orb.yaml') as f:
        cfg = yaml.safe_load(f)
    cfg['strategy']['enabled'] = True
    return cfg


def _mock_alpaca():
    client = MagicMock(spec=AlpacaClient)
    client.get_open_positions.return_value = []
    client.get_account_info.return_value = {'buying_power': 500_000.0}
    client.get_latest_quote.return_value = {'bid_price': 9.95, 'ask_price': 10.00}
    client.submit_stop_bracket_order.return_value = {'id': 'order-1', 'status': 'accepted'}
    client.cancel_order.return_value = True
    return client


def _snap(open_price, prev_close, prev_volume=2_000_000):
    return {'open': open_price, 'prev_close': prev_close,
            'prev_volume': prev_volume, 'latest_price': open_price}


def _range(symbol, range_open=10.0, range_high=10.5, range_low=9.9):
    return RangeData(
        symbol=symbol, range_high=range_high, range_low=range_low,
        range_volume=500_000, range_avg_bar_range_pct=1.0,
        range_close=range_high - 0.02, range_start_ts=pd.Timestamp.utcnow(),
        range_open=range_open,
    )


def _disable_gates(eng):
    """Bypass time/PDT/kill-rail gates that are irrelevant to pool routing —
    mirrors tests/test_orb_engine.py::test_fcfs_skip_if_other_strategy_open."""
    return patch.multiple(
        eng,
        _past_last_entry_time=MagicMock(return_value=False),
        _kill_rails_blocked=MagicMock(return_value=False),
        _pdt_would_block=MagicMock(return_value=False),
        _daily_loss_limit_hit=MagicMock(return_value=False),
    )


class TestAddonPoolUniverse:
    """build_orb_universe_from_snapshots — universe extension only."""

    def test_disabled_is_byte_identical_to_no_addon_config(self):
        cfg_off = _base_cfg()
        cfg_with_addon_off = _base_cfg()
        cfg_with_addon_off['universe']['addon_pools'] = {**ADDON_CFG, 'enabled': False}
        snaps = {
            'PROD': _snap(10.0, 9.0),          # gap 11.1% -> production
            'ADDON4': _snap(10.0, 9.6),        # gap 4.17% -> would match addon_gap4
            'ADDON30': _snap(35.0, 34.0),       # gap 2.9% -> would match addon_p30
        }
        results = []
        for cfg in (cfg_off, cfg_with_addon_off):
            a = _mock_alpaca()
            a.get_snapshots.return_value = snaps
            eng = ORBEngine(alpaca_client=a, db=MagicMock(spec=Database),
                             stop_monitor=MagicMock(spec=StopMonitor), config=cfg)
            keep = eng.build_orb_universe_from_snapshots(list(snaps.keys()))
            results.append(sorted(keep))
        assert results[0] == results[1] == ['PROD']

    def test_addon_pool_membership_by_gap_and_price(self):
        cfg = _base_cfg()
        cfg['universe']['addon_pools'] = ADDON_CFG
        a = _mock_alpaca()
        a.get_snapshots.return_value = {
            'PROD': _snap(10.0, 9.0),      # gap 11.1%, $10 -> production
            'G4': _snap(10.0, 9.6),        # gap 4.17%, $10 -> addon_gap4
            'P30': _snap(35.0, 33.9),      # gap 3.24%, $35 -> addon_p30
            'NONE': _snap(10.0, 9.99),     # gap 0.1% -> matches nothing
        }
        eng = ORBEngine(alpaca_client=a, db=MagicMock(spec=Database),
                         stop_monitor=MagicMock(spec=StopMonitor), config=cfg)
        keep = set(eng.build_orb_universe_from_snapshots(['PROD', 'G4', 'P30', 'NONE']))
        assert keep == {'PROD', 'G4', 'P30'}
        assert eng._symbol_pool == {
            'PROD': 'production', 'G4': 'addon_gap4', 'P30': 'addon_p30'}

    def test_production_match_always_wins_tag(self):
        """A symbol satisfying BOTH production and (hypothetically) an
        overlapping add-on range is tagged 'production' — never the pool."""
        cfg = _base_cfg()
        overlap_cfg = {'enabled': True, 'dry_run': True,
                        'pools': [{'name': 'overlap', 'min_gap_pct': 0.0,
                                   'max_gap_pct': 100.0, 'min_price': 0.0,
                                   'max_price': 100.0}]}
        cfg['universe']['addon_pools'] = overlap_cfg
        a = _mock_alpaca()
        a.get_snapshots.return_value = {'PROD': _snap(10.0, 9.0)}
        eng = ORBEngine(alpaca_client=a, db=MagicMock(spec=Database),
                         stop_monitor=MagicMock(spec=StopMonitor), config=cfg)
        eng.build_orb_universe_from_snapshots(['PROD'])
        assert eng._symbol_pool['PROD'] == 'production'


class TestAddonPoolSelectionChain:
    """check_entries — per-pool chain isolation, dry-run, union, slot cap."""

    def _engine(self, addon_dry_run=True, max_concurrent=None):
        cfg = _base_cfg()
        addon = {**ADDON_CFG, 'dry_run': addon_dry_run}
        cfg['universe']['addon_pools'] = addon
        if max_concurrent is not None:
            cfg['sizing']['max_concurrent'] = max_concurrent
        db = MagicMock(spec=Database)
        db.get_open_trades.return_value = []
        db.get_trades_by_date.return_value = []
        a = _mock_alpaca()
        eng = ORBEngine(alpaca_client=a, db=db,
                         stop_monitor=MagicMock(spec=StopMonitor), config=cfg)
        # Vetoes are orthogonal to pool routing — the study behind
        # PREREG_LIVE_UNION.md evaluated the chain BEFORE them; disable so
        # these tests isolate union/routing mechanics, not veto math.
        eng.pdr_veto_enabled = False
        eng.g1_veto_enabled = False
        eng.range_size_veto_enabled = False
        eng.catalyst_veto_enabled = False
        eng.skip_q1 = False
        eng.filter_threshold = -999.0
        return eng, a, db

    def _seed(self, eng, symbol, pool):
        eng.universe.add(symbol)
        eng.candidates[symbol] = CandidateState(symbol=symbol)
        eng.candidates[symbol].range_data = _range(symbol)
        eng._symbol_pool[symbol] = pool

    def test_pool_only_symbol_never_changes_production_pick(self):
        """An add-on-only candidate must never enter production's scored
        list or displace a production pick (isolation, PREREG mechanism)."""
        eng, a, db = self._engine(addon_dry_run=True)
        self._seed(eng, 'PRODSYM', 'production')
        self._seed(eng, 'ADDONSYM', 'addon_gap4')
        fp = {s: {'prev_day_bar': {}, 'daily_stats_20d': {}}
              for s in ('PRODSYM', 'ADDONSYM')}
        with _disable_gates(eng), \
                patch('trading.orb_engine.composite_score', return_value=1.0), \
                patch('trading.orb_engine.assign_quintile', return_value='Q5'):
            submitted = eng.check_entries(feature_providers=fp)
        assert 'PRODSYM' in submitted
        assert eng.candidates['PRODSYM'].rejected_reason is None
        # Addon pick is dry-run -> not in the submitted (ordered) list.
        assert 'ADDONSYM' not in submitted

    def test_dry_run_submits_nothing_and_logs_would_buy(self, caplog):
        import logging
        eng, a, db = self._engine(addon_dry_run=True)
        self._seed(eng, 'ADDONSYM', 'addon_gap4')
        fp = {'ADDONSYM': {'prev_day_bar': {}, 'daily_stats_20d': {}}}
        with _disable_gates(eng), \
                patch('trading.orb_engine.composite_score', return_value=1.0), \
                patch('trading.orb_engine.assign_quintile', return_value='Q5'), \
                caplog.at_level(logging.INFO):
            submitted = eng.check_entries(feature_providers=fp)
        assert submitted == []
        a.submit_stop_bracket_order.assert_not_called()
        db.save_trade.assert_not_called()
        assert 'ADDONSYM' not in eng.open_positions
        assert '[ORB+ DRY] WOULD BUY ADDONSYM' in caplog.text
        assert eng.candidates['ADDONSYM'].rejected_reason == 'addon_dry_run'

    def test_shared_slot_cap_production_first(self):
        """Production consumes slots first; add-ons only get what's left —
        PREREG 'Slot arithmetic: production picks submitted first'."""
        eng, a, db = self._engine(addon_dry_run=False, max_concurrent=1)
        self._seed(eng, 'PRODSYM', 'production')
        self._seed(eng, 'ADDONSYM', 'addon_gap4')
        fp = {s: {'prev_day_bar': {}, 'daily_stats_20d': {}}
              for s in ('PRODSYM', 'ADDONSYM')}
        with _disable_gates(eng), \
                patch('trading.orb_engine.composite_score', return_value=1.0), \
                patch('trading.orb_engine.assign_quintile', return_value='Q5'):
            submitted = eng.check_entries(feature_providers=fp)
        assert submitted == ['PRODSYM']
        assert 'ADDONSYM' not in eng.open_positions


class TestAddonPoolIntegration:
    """Two pools through real check_entries + a real temp Database:
    union of picks, slot cap, pattern_data.pool persisted."""

    def test_union_persists_pool_via_real_db(self, tmp_path):
        cfg = _base_cfg()
        cfg['sizing']['max_concurrent'] = 3
        cfg['universe']['addon_pools'] = {**ADDON_CFG, 'dry_run': False}
        db = Database(db_path=str(tmp_path / 'test.db'))
        a = _mock_alpaca()
        eng = ORBEngine(alpaca_client=a, db=db,
                         stop_monitor=MagicMock(spec=StopMonitor), config=cfg)
        eng.pdr_veto_enabled = False
        eng.g1_veto_enabled = False
        eng.range_size_veto_enabled = False
        eng.catalyst_veto_enabled = False
        eng.skip_q1 = False
        eng.filter_threshold = -999.0

        self_seed = {
            'PRODSYM': 'production', 'G4SYM': 'addon_gap4', 'P30SYM': 'addon_p30',
        }
        for sym, pool in self_seed.items():
            eng.universe.add(sym)
            eng.candidates[sym] = CandidateState(symbol=sym)
            eng.candidates[sym].range_data = _range(sym)
            eng._symbol_pool[sym] = pool
        fp = {s: {'prev_day_bar': {}, 'daily_stats_20d': {}} for s in self_seed}

        with _disable_gates(eng), \
                patch('trading.orb_engine.composite_score', return_value=1.0), \
                patch('trading.orb_engine.assign_quintile', return_value='Q5'):
            submitted = eng.check_entries(feature_providers=fp)

        assert set(submitted) == {'PRODSYM', 'G4SYM', 'P30SYM'}
        today = datetime.now(timezone.utc).date().isoformat()
        rows = {r['symbol']: r for r in db.get_trades_by_date(today)}
        assert rows['PRODSYM']['pattern_data'] and '"pool": "production"' in rows['PRODSYM']['pattern_data']
        assert '"pool": "addon_gap4"' in rows['G4SYM']['pattern_data']
        assert '"pool": "addon_p30"' in rows['P30SYM']['pattern_data']


class TestUnionParity:
    """Independent BT-side reimplementation of the union rule must match
    the engine's rule on the same synthetic candidate table (PREREG:
    'production book + add-on book, production first, slot cap 8')."""

    @staticmethod
    def _bt_union(production_ranked, addon_ranked_by_pool, slot_cap):
        """Standalone reimplementation — production first, then add-on
        pools in config order, composite DESC within each, shared cap."""
        picks = list(production_ranked)
        for pool_ranked in addon_ranked_by_pool:
            for sym in pool_ranked:
                if len(picks) >= slot_cap:
                    break
                if sym not in picks:
                    picks.append(sym)
        return picks[:slot_cap]

    def test_bt_union_matches_engine_union(self):
        slot_cap = 8
        production = ['P1', 'P2', 'P3']
        addon_gap4 = ['G1', 'G2', 'G3']
        addon_p30 = ['R1', 'R2']
        bt_picks = self._bt_union(production, [addon_gap4, addon_p30], slot_cap)
        assert bt_picks == ['P1', 'P2', 'P3', 'G1', 'G2', 'G3', 'R1', 'R2']

        cfg = _base_cfg()
        cfg['sizing']['max_concurrent'] = slot_cap
        cfg['universe']['addon_pools'] = {**ADDON_CFG, 'dry_run': False}
        db = MagicMock(spec=Database)
        db.get_open_trades.return_value = []
        db.get_trades_by_date.return_value = []
        a = _mock_alpaca()
        eng = ORBEngine(alpaca_client=a, db=db,
                         stop_monitor=MagicMock(spec=StopMonitor), config=cfg)
        eng.pdr_veto_enabled = False
        eng.g1_veto_enabled = False
        eng.range_size_veto_enabled = False
        eng.catalyst_veto_enabled = False
        eng.skip_q1 = False
        eng.filter_threshold = -999.0

        pool_map = {}
        for sym in production:
            pool_map[sym] = 'production'
        for sym in addon_gap4:
            pool_map[sym] = 'addon_gap4'
        for sym in addon_p30:
            pool_map[sym] = 'addon_p30'
        # Composite DESC within a pool == the reimplementation's input order
        # (P1>P2>P3, G1>G2>G3, R1>R2) — fake distinct scores that preserve it.
        scores = {}
        for grp in (production, addon_gap4, addon_p30):
            for i, sym in enumerate(grp):
                scores[sym] = 10.0 - i
        for sym, pool in pool_map.items():
            eng.universe.add(sym)
            eng.candidates[sym] = CandidateState(symbol=sym)
            eng.candidates[sym].range_data = _range(sym)
            eng._symbol_pool[sym] = pool
        fp = {s: {'prev_day_bar': {}, 'daily_stats_20d': {}} for s in pool_map}

        with _disable_gates(eng), \
                patch('trading.orb_engine.composite_score',
                      side_effect=lambda feats, z, _s=scores: _s.get(feats.get('_sym'))), \
                patch('trading.orb_engine.assign_quintile', return_value='Q5'):
            # composite_score is called with (feats, z_params) and has no
            # symbol arg in the real signature — patch _compute_features
            # instead to tag the symbol onto feats so the fake scorer can
            # look it up deterministically.
            orig_compute = eng._compute_features
            def _tagged(cand, **kw):
                f = orig_compute(cand, **kw)
                f['_sym'] = cand.symbol
                return f
            with patch.object(eng, '_compute_features', side_effect=_tagged):
                submitted = eng.check_entries(feature_providers=fp)

        assert submitted == bt_picks
