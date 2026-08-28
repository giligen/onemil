"""Promote/demote hysteresis — validated against the PRODUCTION config.

2026-08-28 live-day-1 incident: fade-demotion (4.5) was set below the
promote gate (20), inverting the hysteresis — every order staged at
distance 4.5-20 was demoted on the next scheduler pass. 18/18 live
stages churned place->cancel in ~60s (fail-safe, $0 lost, but staging
was neutered). Fix: promote_distance_pct 20 -> 3.6.

These tests load the REAL values from config.yaml (skipped in checkouts
without it) and drive the real PrestageManager through:
  A. this morning's churn class (distance in the old dead zone) — must
     never stage at all now;
  B. the healthy arc — stage, SURVIVE passes, demote only on real fade;
  C. the hysteresis band (3.6 < d < 4.5) — staged orders must hold.
"""
from __future__ import annotations
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_ignition_prestage import (   # reuse the manager fixture
    _mgr, _cand, _feed_and_tick, _et, DAY, STATE_STAGED,
    STATE_CANCEL_CONFIRMED)

CONFIG_YAML = Path(__file__).resolve().parent.parent / 'config.yaml'
pytestmark = pytest.mark.skipif(
    not CONFIG_YAML.exists(), reason='config.yaml (instance config) absent')


def _prod_prestage_values():
    from config import get_config
    p = get_config().ignition_live_cfg['prestage']
    return {'promote_distance_pct': p['promote_distance_pct'],
            'demote_distance_pct': p['demote_distance_pct'],
            'promote_consecutive': p.get('promote_consecutive', 2)}


def _prod_mgr(tmp_path, monkeypatch):
    vals = _prod_prestage_values()
    m, a, *rest = _mgr(tmp_path, monkeypatch, **vals)
    return m, a, vals


class TestProdConfigHysteresis:
    def test_geometry_is_sane(self):
        """The invariant the 8/28 incident violated: promote strictly
        tighter than demote, both meaningful for the 6-10%% band."""
        v = _prod_prestage_values()
        assert v['promote_distance_pct'] < v['demote_distance_pct'], (
            'inverted hysteresis: promote gate must be TIGHTER than '
            'demote or every stage churns place->cancel')
        # band sanity: promote reachable by a +6.2%-from-open stock,
        # demote fires by ~+5% — the owner's fade rule
        assert 3.0 <= v['promote_distance_pct'] <= 4.0
        assert 4.0 <= v['demote_distance_pct'] <= 5.5

    def test_morning_churn_class_never_stages(self, tmp_path, monkeypatch):
        """+3%%-from-open candidate (distance ~6.4, the old 4.5-20 dead
        zone): under the fixed config it must NOT stage at all."""
        m, a, vals = _prod_mgr(tmp_path, monkeypatch)
        # day_open 10 -> level 11.0; price 10.30 -> distance 6.36
        c = _cand(day_open=10.0, price=10.30)
        _feed_and_tick(m, [c], _et(DAY, 9, 40),
                       ticks=vals['promote_consecutive'] + 2)
        assert 'PSTG' not in m._stages
        a.submit_stop_limit_order.assert_not_called()

    def test_healthy_arc_stage_survive_demote(self, tmp_path, monkeypatch):
        """+6.5%% stages, HOLDS across passes (the churn regression),
        demotes only when price fades below ~+5%%."""
        m, a, vals = _prod_mgr(tmp_path, monkeypatch)
        # price 10.65 -> distance (11-10.65)/11 = 3.18 < promote 3.6
        c = _cand(day_open=10.0, price=10.65)
        _feed_and_tick(m, [c], _et(DAY, 9, 40),
                       ticks=vals['promote_consecutive'])
        assert m._stages['PSTG']['state'] == STATE_STAGED
        # 5 more passes at the same price: MUST still be staged
        for i in range(5):
            m.process_tick(now_et=_et(DAY, 9, 41 + i))
        assert m._stages['PSTG']['state'] == STATE_STAGED, (
            '8/28 churn regression: stage canceled without a fade')
        # real fade: 10.44 -> distance 5.09 > demote 4.5
        m.on_price('PSTG', 10.44)
        m.process_tick(now_et=_et(DAY, 9, 47))
        assert m._stages['PSTG']['state'] == STATE_CANCEL_CONFIRMED

    def test_hysteresis_band_holds_stage(self, tmp_path, monkeypatch):
        """Price drifting into the band between promote (3.6) and demote
        (4.5) must NOT cancel — that is the whole point of hysteresis."""
        m, a, vals = _prod_mgr(tmp_path, monkeypatch)
        c = _cand(day_open=10.0, price=10.65)          # d 3.18: stages
        _feed_and_tick(m, [c], _et(DAY, 9, 40),
                       ticks=vals['promote_consecutive'])
        assert m._stages['PSTG']['state'] == STATE_STAGED
        # drift to 10.56 -> distance 4.0: inside the band
        m.on_price('PSTG', 10.56)
        for i in range(3):
            m.process_tick(now_et=_et(DAY, 9, 41 + i))
        assert m._stages['PSTG']['state'] == STATE_STAGED
