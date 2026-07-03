"""Breakout-bar keying fix (2026-07-03) — KOLD/PLTU/TSDD false-anchor incident.

Bug chain being locked down:
  1. Alpaca WS delivers bars from subscribe-time forward, so live streamed
     windows usually MISS the 9:30 ET bar (universe build subscribes ~9:31).
  2. `_first_session_open_ts_utc` accepted a bar at minute :30 in hour 13 OR
     14 UTC. During EDT, 14:30Z = 10:30 ET — a mid-session bar. Windows
     missing 13:30Z anchored there instead → range_end 14:35Z → breakout bar
     keyed ~1 hour late → Rule M fired tag_bb at 14:36:2xZ on random bars
     (KOLD 6/29, PLTU 7/1, TSDD 6/23 — audited via logs/touchgo_audit.jsonl).
  3. The late-fill guard couldn't catch it: bb_age was NEGATIVE (breakout bar
     AFTER the fill) and the guard only checked age > +15min.

Fix layers under test:
  A. OpenPosition.range_end_ts (stored at entry submit from the candidate's
     range_data) is used directly as the keying anchor — no window-derived
     session open in the hot path.
  B. `_first_session_open_ts_utc` derives the single date-correct hour
     (13 EDT / 14 EST) via `_et_offset_hours` — 14:30Z is rejected in summer.
  C. Negative-age tripwire: bb_ts postdating the fill by >5min declines
     touchgo with a WARNING instead of evaluating a random bar.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

from trading.orb_engine import (
    ORBEngine, OpenPosition, RangeData,
    _first_session_open_ts_utc,
)
from trading.orb_touchgo_filter import load_touchgo_config

EDT_DATE = '2026-07-06'   # Monday, EDT: 9:30 ET = 13:30Z
EST_DATE = '2026-01-15'   # EST: 9:30 ET = 14:30Z


def _bars(rows, date=EDT_DATE):
    return pd.DataFrame([
        {'timestamp': pd.Timestamp(f'{date} {h:02d}:{m:02d}:00', tz='UTC'),
         'open': o, 'high': hi, 'low': lo, 'close': c, 'volume': v}
        for (h, m, o, hi, lo, c, v) in rows
    ])


def _engine_skel():
    """ORBEngine with only the attrs the keying/touchgo paths touch."""
    eng = ORBEngine.__new__(ORBEngine)
    eng.open_positions = {}
    eng.touchgo_cfg = load_touchgo_config({})   # market mode, age cap 15min
    eng.stop_monitor = MagicMock()
    eng.notifier = None
    return eng


def _pos(sym='KOLD', rh=23.13, rl=23.00, entry=23.15, range_end_ts=None,
         order_id='', entry_time=None):
    return OpenPosition(
        symbol=sym, entry_price=entry, stop_price=rl, shares=595,
        trade_id=1, order_id=order_id,
        entry_time=entry_time or pd.Timestamp(f'{EDT_DATE} 13:41:55', tz='UTC'),
        range_high=rh, range_low=rl,
        lock_arm_at_r=1.75, lock_stop_r=0.5,
        composite_score=0.4, quintile='Q4',
        range_end_ts=range_end_ts,
    )


# =========================================================================
# Layer B — session-open anchor rejects the false 14:30Z bar
# =========================================================================

class TestSessionOpenAnchor:
    def test_edt_accepts_1330z(self):
        bars = _bars([(13, 30, 10, 10.1, 9.9, 10, 1000),
                      (13, 31, 10, 10.2, 9.9, 10.1, 1000)])
        ts = _first_session_open_ts_utc(bars)
        assert ts is not None and ts.hour == 13 and ts.minute == 30

    def test_edt_rejects_1430z_incident_regression(self):
        """THE incident shape: window missing 13:30Z, containing 14:30Z.
        Old code anchored at 14:30Z (10:30 ET!). Must now return None."""
        bars = _bars([(13, 31, 10, 10.2, 9.9, 10.1, 1000),
                      (14, 30, 11, 11.2, 10.9, 11.1, 1000),
                      (14, 35, 11, 11.3, 10.9, 11.2, 1000)])
        assert _first_session_open_ts_utc(bars) is None

    def test_est_accepts_1430z(self):
        bars = _bars([(14, 30, 10, 10.1, 9.9, 10, 1000)], date=EST_DATE)
        ts = _first_session_open_ts_utc(bars)
        assert ts is not None and ts.hour == 14 and ts.minute == 30

    def test_est_rejects_1530z(self):
        """Winter mirror of the incident: 15:30Z = 10:30 ET during EST."""
        bars = _bars([(14, 31, 10, 10.2, 9.9, 10.1, 1000),
                      (15, 30, 11, 11.2, 10.9, 11.1, 1000)], date=EST_DATE)
        assert _first_session_open_ts_utc(bars) is None

    def test_est_rejects_1330z(self):
        """13:30Z during EST is 8:30 ET pre-market — must not anchor."""
        bars = _bars([(13, 30, 10, 10.1, 9.9, 10, 1000)], date=EST_DATE)
        assert _first_session_open_ts_utc(bars) is None


# =========================================================================
# Layer A — keying uses the position's own range_end_ts
# =========================================================================

class TestRangeEndKeying:
    def test_kold_incident_replay_keys_true_breakout(self):
        """KOLD 6/29 window shape: stream starts 13:31 (NO 13:30 bar), price
        chops below range_high 23.13 until the true breakout at 13:42.
        With range_end_ts on the position the capture must key 13:42Z.
        (Pre-fix: anchor None -> capture declined -> later false-anchored
        to 14:35Z.)"""
        eng = _engine_skel()
        pos = _pos(range_end_ts=pd.Timestamp(f'{EDT_DATE} 13:35:00', tz='UTC'),
                   order_id='pending-1')
        eng.open_positions['KOLD'] = pos
        bars = _bars([
            (13, 31, 23.05, 23.10, 23.02, 23.06, 900),
            (13, 35, 23.06, 23.12, 23.03, 23.08, 800),   # touches, no break
            (13, 40, 23.08, 23.13, 23.05, 23.10, 700),   # == rh, strict > fails
            (13, 42, 23.11, 23.20, 23.09, 23.18, 2500),  # TRUE breakout
            (13, 43, 23.18, 23.25, 23.15, 23.22, 2000),
        ])
        eng._ensure_breakout_bar_ts('KOLD', bars)
        assert pos.breakout_bar_ts == pd.Timestamp(f'{EDT_DATE} 13:42:00', tz='UTC')

    def test_fallback_without_range_end_declines_on_false_anchor_window(self):
        """No range_end_ts (rehydrated position) + window missing 13:30Z but
        containing 14:30Z/14:35Z bars above range_high: capture must DECLINE
        (breakout_bar_ts stays None). Old code keyed 14:35Z here."""
        eng = _engine_skel()
        pos = _pos(range_end_ts=None, order_id='pending-1')
        eng.open_positions['KOLD'] = pos
        bars = _bars([
            (13, 31, 23.05, 23.10, 23.02, 23.06, 900),
            (14, 30, 23.40, 23.46, 23.38, 23.44, 1200),  # false-anchor bait
            (14, 35, 23.45, 23.46, 23.44, 23.4421, 900), # the KOLD keyed bar
        ])
        eng._ensure_breakout_bar_ts('KOLD', bars)
        assert pos.breakout_bar_ts is None

    def test_late_breakout_cast_shape_keys_correctly(self):
        """CAST 6/30: stop-limit sat 32 min; true breakout at 14:07. With
        range_end_ts the search spans [13:35, ...] and keys 14:07Z."""
        eng = _engine_skel()
        pos = _pos(sym='CAST', rh=6.31, rl=6.1006, entry=6.33,
                   range_end_ts=pd.Timestamp(f'{EDT_DATE} 13:35:00', tz='UTC'),
                   order_id='pending-1',
                   entry_time=pd.Timestamp(f'{EDT_DATE} 14:07:32', tz='UTC'))
        eng.open_positions['CAST'] = pos
        rows = [(13, 36 + i, 6.20, 6.28, 6.15, 6.22, 500) for i in range(4)]
        rows.append((14, 7, 6.12, 6.43, 6.12, 6.4001, 3000))  # breakout
        eng._ensure_breakout_bar_ts('CAST', _bars(rows))
        assert pos.breakout_bar_ts == pd.Timestamp(f'{EDT_DATE} 14:07:00', tz='UTC')

    def test_pre_range_spike_not_keyed(self):
        """A bar above range_high BEFORE range end (e.g. inside 9:30-9:35 on
        a partial window) must not be keyed — range_end_ts bounds the search."""
        eng = _engine_skel()
        pos = _pos(range_end_ts=pd.Timestamp(f'{EDT_DATE} 13:35:00', tz='UTC'),
                   order_id='pending-1')
        eng.open_positions['KOLD'] = pos
        bars = _bars([
            (13, 33, 23.10, 23.30, 23.05, 23.12, 900),   # pre-range-end spike > rh
            (13, 36, 23.12, 23.19, 23.08, 23.16, 1500),  # true post-range breakout
        ])
        eng._ensure_breakout_bar_ts('KOLD', bars)
        assert pos.breakout_bar_ts == pd.Timestamp(f'{EDT_DATE} 13:36:00', tz='UTC')


# =========================================================================
# Layer C — negative-age tripwire
# =========================================================================

class TestNegativeAgeTripwire:
    def test_bb_postdating_fill_declines_touchgo(self, caplog):
        """Insane keying (bb 1h after fill — the incident signature) must
        decline touchgo loudly, never evaluate the bar."""
        import logging as _logging
        caplog.set_level(_logging.WARNING, logger='trading.orb_engine')
        eng = _engine_skel()
        pos = _pos(order_id='',  # filled
                   entry_time=pd.Timestamp(f'{EDT_DATE} 13:41:55', tz='UTC'))
        pos.breakout_bar_ts = pd.Timestamp(f'{EDT_DATE} 14:35:00', tz='UTC')
        eng.open_positions['KOLD'] = pos
        # Weak 14:35 bar — would fire Rule M if (wrongly) evaluated.
        bars = _bars([(14, 35, 23.45, 23.46, 23.44, 23.4421, 900)])
        eng._evaluate_touchgo('KOLD', bars)
        assert not eng.stop_monitor.force_exit.called
        assert pos.rule_m_evaluated and pos.rule_d_evaluated
        assert any('touchgo DECLINED' in r.getMessage() for r in caplog.records)

    def test_small_negative_age_tolerated(self):
        """bb within a couple of minutes after fill (touch-without-exceed
        edge) is legitimate — rule must still evaluate."""
        eng = _engine_skel()
        pos = _pos(order_id='',
                   entry_time=pd.Timestamp(f'{EDT_DATE} 13:40:00', tz='UTC'))
        pos.breakout_bar_ts = pd.Timestamp(f'{EDT_DATE} 13:42:00', tz='UTC')  # -2min
        eng.open_positions['KOLD'] = pos
        # Weak breakout bar -> Rule M fires (proves it was evaluated).
        bars = _bars([(13, 42, 23.20, 23.30, 23.14, 23.15, 2500)])
        eng._evaluate_touchgo('KOLD', bars)
        assert eng.stop_monitor.force_exit.called


# =========================================================================
# Plumbing — range_end_ts set at entry submit
# =========================================================================

class TestRangeEndPlumbedAtSubmit:
    @pytest.fixture
    def engine(self):
        import yaml
        from pathlib import Path
        from data_sources.alpaca_client import AlpacaClient
        from persistence.database import Database
        from trading.stop_monitor import StopMonitor
        with open(Path(__file__).parent.parent / 'orb.yaml') as f:
            cfg = yaml.safe_load(f)
        cfg['strategy']['enabled'] = True
        alpaca = MagicMock(spec=AlpacaClient)
        alpaca.get_latest_quote.return_value = {'bid_price': 23.10, 'ask_price': 23.12}
        alpaca.submit_stop_bracket_order.return_value = {'id': 'o-1', 'status': 'accepted'}
        db = MagicMock(spec=Database)
        db.save_trade.return_value = 42
        sm = MagicMock(spec=StopMonitor)
        return ORBEngine(alpaca_client=alpaca, db=db, stop_monitor=sm, config=cfg)

    def test_submit_entry_sets_range_end_from_candidate(self, engine):
        from trading.orb_planner import OrbTradePlan
        engine.build_universe(source_loader=lambda: ['KOLD'])
        engine.candidates['KOLD'].range_data = RangeData(
            symbol='KOLD', range_high=23.13, range_low=23.00,
            range_volume=10_000, range_avg_bar_range_pct=0.4,
            range_close=23.10,
            range_start_ts=pd.Timestamp(f'{EDT_DATE} 13:30:00', tz='UTC'),
            range_open=23.02,
        )
        plan = OrbTradePlan(
            symbol='KOLD', range_high=23.13, range_low=23.00, range_size=0.13,
            entry_price=23.15, stop_price=23.13, shares=595,
            position_dollars=13_774, lock_arm_at_r=1.75, lock_stop_r=0.5,
            risk_per_share=0.15, total_risk=89.25,
            composite_score=0.4, quintile='Q4', adaptive_mult=1.0,
        )
        engine._submit_entry(plan)
        pos = engine.open_positions['KOLD']
        assert pos.range_end_ts == pd.Timestamp(f'{EDT_DATE} 13:35:00', tz='UTC')
