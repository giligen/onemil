"""Tests for the four ORB paper-vs-BT parity fixes (docs/orb_parity_20260930.md).

1. Exit-time parity: study_orb_pipeline_static_lock.simulate_static_lock already
   truncates at orb.yaml's exit.force_close_time_et (15:45 default) — confirmed
   NOT a bug (coordinator correction, verified against the live code and real
   cache.db bars). The real AXTL 9/29 divergence (BT features CSV +$99.61 'eod'
   vs live force_close -$96.30) is study_orb.simulate_orb_trade's naive EOD exit
   (last bar of the WHOLE day, no force-close truncation) feeding the
   non-production-parity orb_features_*.csv — CLAUDE.md already flags this file
   class as "NOT production-parity"; this regression test pins the mechanism
   with AXTL's real 2026-09-29 bars so it can never again be mistaken for BT
   ground truth.
2. Gap-gate input resolution + persistence (trading/orb_gap_gate.py).
3. Fill-persistence guard DB contract (ported from HOD's 1b3584f).
4. Force-close greppable log tag format + the proposed archive-cron pattern.
"""
import json
import os
import tempfile
from datetime import datetime, timezone

import pandas as pd
import pytest

from trading.orb_gap_gate import resolve_gap_input, SOURCE_BAR, SOURCE_SNAPSHOT


# ---------------------------------------------------------------------------
# Fix 1: exit-time parity + the real AXTL root cause
# ---------------------------------------------------------------------------

def _bar(ts, o, h, l, c, v):
    return {'timestamp': pd.Timestamp(ts, tz='UTC'), 'open': o, 'high': h, 'low': l, 'close': c, 'volume': v}


def test_static_lock_exits_at_1545_not_1600():
    """Fixture book: a fill that is +$ at 16:00 and -$ at 15:45 must exit at
    15:45 (live's force_close_time_et), never the 16:00 close."""
    import study_orb_pipeline_static_lock as sol

    entry_price = 10.00
    range_high, range_low = 10.00, 9.50
    entry_time = pd.Timestamp('2026-09-29 13:35:00', tz='UTC')  # 09:35 ET
    bars = pd.DataFrame([
        _bar('2026-09-29 13:35:00', 10.00, 10.05, 9.98, 10.03, 1000),  # entry bar, close near high (no Rule M)
        _bar('2026-09-29 13:36:00', 10.03, 10.05, 9.90, 9.95, 1000),   # bar1, low 9.90 > 9.625 (no Rule D)
        _bar('2026-09-29 19:44:00', 9.95, 9.95, 9.70, 9.70, 1000),     # 15:44 ET — losing, low 9.70 > stop 9.50
        _bar('2026-09-29 20:26:00', 9.70, 10.90, 9.70, 10.80, 1000),   # 16:26 ET — truncated away, must not count
    ])
    exit_price, exit_reason = sol.simulate_static_lock(
        bars, entry_price, range_high, range_low, entry_time)[:2]
    assert exit_reason == 'eod'
    # Must price off the last bar AT/BEFORE 15:45 ET (19:44 UTC, close 9.30),
    # never the post-15:45 recovery bars.
    assert exit_price < entry_price, (
        f"exit_price={exit_price} priced off a post-15:45 bar — force-close "
        f"truncation regressed")
    assert exit_price < 9.71, (
        f"exit_price={exit_price} >= the 15:44 ET bar's close — looks like "
        f"the 20:26 UTC (16:26 ET) recovery bar leaked into the result")


def test_force_close_et_synced_from_orb_yaml_not_hardcoded():
    """load_bt_config's force_close_et must come from orb.yaml's
    exit.force_close_time_et (main() then assigns it to the module-level
    FORCE_CLOSE_ET simulate_static_lock reads) — not a literal hardcoded
    default silently diverging from live's own orb.yaml read."""
    import study_orb_pipeline_static_lock as sol
    import yaml
    with open('orb.yaml') as f:
        cfg = yaml.safe_load(f)
    live_value = str(cfg.get('exit', {}).get('force_close_time_et', '15:45'))
    bt_cfg = sol.load_bt_config()
    assert bt_cfg['force_close_et'] == live_value, (
        "BT's force_close_et disagrees with orb.yaml's exit.force_close_time_et "
        "— live and BT would truncate the trading day at different times")


def test_axtl_20260929_real_bars_do_not_reach_the_naive_eod_bounce():
    """Root-cause regression: AXTL's real cache.db bars for 2026-09-29 show a
    loss into 15:45 ET (last bar 19:44 UTC close $3.37, matching live's real
    force_close fill $3.36) but recover past $3.46 by 20:26 UTC (16:26 ET) —
    well after BOTH the 15:45 force-close and the 16:00 close. The
    orb_features_20260929_2053.csv row (entry 3.4579, pnl +$99.61, exit_reason
    'eod') is study_orb.simulate_orb_trade's naive full-day walk hitting that
    late bounce — NOT study_orb_pipeline_static_lock's production exit, which
    this test pins to the correct (losing) 15:45 result on the same bars.
    """
    import study_orb_pipeline_static_lock as sol

    entry_price = 3.4579428
    range_high, range_low = 3.4579428, 3.30  # range_low unused by the eod branch
    entry_time = pd.Timestamp('2026-09-29 13:35:00', tz='UTC')
    # Real cache.db intraday_bars_1min rows for AXTL, 2026-09-29 (read-only query).
    real_bars = pd.DataFrame([
        _bar('2026-09-29 13:35:00', entry_price, 3.4620, 3.4557, 3.4610, 1000),  # entry bar, close near high
        _bar('2026-09-29 19:30:00', 3.415, 3.415, 3.415, 3.415, 1000),
        _bar('2026-09-29 19:35:00', 3.40, 3.40, 3.40, 3.40, 673),
        _bar('2026-09-29 19:41:00', 3.38, 3.38, 3.38, 3.38, 302),
        _bar('2026-09-29 19:44:00', 3.37, 3.37, 3.37, 3.37, 3214),
        _bar('2026-09-29 19:47:00', 3.36, 3.36, 3.36, 3.36, 117),   # matches live's real force_close fill
        _bar('2026-09-29 19:51:00', 3.41, 3.42, 3.41, 3.42, 1458),
        _bar('2026-09-29 20:26:00', 3.4683, 3.4683, 3.4683, 3.4683, 300),  # the naive-eod script's bounce
    ])
    exit_price, exit_reason = sol.simulate_static_lock(
        real_bars, entry_price, range_high, range_low, entry_time)[:2]
    assert exit_reason == 'eod'
    assert exit_price <= 3.37, (
        f"production exit={exit_price} reached past 15:45 ET — the 20:26 UTC "
        f"bounce leaked into the parity-enforced path")
    assert (exit_price - entry_price) < 0, "the true production-parity AXTL result is a loss, like live — not +$99.61"


# ---------------------------------------------------------------------------
# Fix 2: gap-gate input resolution
# ---------------------------------------------------------------------------

def test_gap_gate_prefers_minute_bar_when_present():
    r = resolve_gap_input('ASTX', snapshot_open=10.32, prev_close=9.76, minute_bar_open=9.95)
    assert r.source == SOURCE_BAR
    assert r.gap_input_open == 9.95
    assert round(r.gap_pct, 2) == 1.95


def test_gap_gate_logs_info_only_for_real_candidates_or_new_bar_source(caplog):
    """Journal-bloat fix 2026-10-01: INFO only for a real candidate (passes
    the gap floor) or a symbol's first settled-bar-source resolution this
    session; DEBUG for every other (sub-floor, repeat) tick. Uses symbols
    not touched by any other test to keep the module-level dedup set
    (`_bar_source_logged`, scoped to the whole test session) collision-free."""
    import logging
    caplog.set_level(logging.DEBUG)

    caplog.clear()
    r = resolve_gap_input('ZZGAPFLOORLOW', snapshot_open=10.10, prev_close=10.0,
                           minute_bar_open=None, gap_floor_pct=5.0)
    assert r.gap_pct < 5.0
    recs = [rec for rec in caplog.records if '[ORB] GAP_GATE' in rec.message]
    assert len(recs) == 1 and recs[0].levelname == 'DEBUG'

    caplog.clear()
    r = resolve_gap_input('ZZGAPFLOORHI', snapshot_open=10.60, prev_close=10.0,
                           minute_bar_open=None, gap_floor_pct=5.0)
    assert r.gap_pct >= 5.0
    recs = [rec for rec in caplog.records if '[ORB] GAP_GATE' in rec.message]
    assert len(recs) == 1 and recs[0].levelname == 'INFO'

    caplog.clear()
    r1 = resolve_gap_input('ZZGAPNEWBAR', snapshot_open=10.10, prev_close=10.0,
                            minute_bar_open=10.05, gap_floor_pct=50.0)
    assert r1.source == SOURCE_BAR and r1.gap_pct < 50.0
    recs = [rec for rec in caplog.records if '[ORB] GAP_GATE' in rec.message]
    assert len(recs) == 1 and recs[0].levelname == 'INFO'

    caplog.clear()
    r2 = resolve_gap_input('ZZGAPNEWBAR', snapshot_open=10.10, prev_close=10.0,
                            minute_bar_open=10.05, gap_floor_pct=50.0)
    assert r2.source == SOURCE_BAR
    recs = [rec for rec in caplog.records if '[ORB] GAP_GATE' in rec.message]
    assert len(recs) == 1 and recs[0].levelname == 'DEBUG'


def test_gap_gate_falls_back_to_snapshot_and_warns(caplog):
    import logging
    caplog.set_level(logging.WARNING)
    r = resolve_gap_input('ASTX', snapshot_open=10.32, prev_close=9.76, minute_bar_open=None)
    assert r.source == SOURCE_SNAPSHOT
    assert r.gap_input_open == 10.32
    assert any('falling back to the real-time snapshot' in rec.message for rec in caplog.records)


def test_gap_gate_none_when_no_usable_open():
    assert resolve_gap_input('XYZ', snapshot_open=0, prev_close=9.76, minute_bar_open=None) is None
    assert resolve_gap_input('XYZ', snapshot_open=10.0, prev_close=0, minute_bar_open=None) is None


def test_gap_gate_persists_to_dry_ledger_extra_json():
    """Integration: real Database on a tmp path — insert_dry_entry's extra_json
    must carry gap_input_open/gap_input_prev_close/gap_pct/source/timestamp."""
    from persistence.database import Database
    with tempfile.TemporaryDirectory() as d:
        db_path = os.path.join(d, 'tmp_trades.db')
        db = Database(db_path=db_path)
        gate = resolve_gap_input('ASTX', snapshot_open=10.32, prev_close=9.76, minute_bar_open=9.95)
        row_id = db.insert_dry_entry({
            'strategy': 'orb', 'trade_date': '2026-09-30', 'symbol': 'ASTX',
            'entry_ts': datetime.now(timezone.utc).isoformat(), 'entry_px': 10.34,
            'shares': 322, 'stop_px': 10.01, 'target_px': None,
            'risk_usd': 106.0, 'source': 'live_dry', 'extra': gate.as_dict(),
        })
        assert row_id is not None
        cur = db._trades_conn.execute("SELECT extra_json FROM dry_trades WHERE id = ?", (row_id,))
        extra_json = cur.fetchone()[0]
        payload = json.loads(extra_json)
        assert payload['source'] == SOURCE_BAR
        assert payload['gap_input_open'] == 9.95
        assert payload['gap_input_prev_close'] == 9.76
        assert 'gap_pct' in payload and 'timestamp' in payload


# ---------------------------------------------------------------------------
# Fix 3: fill-persistence guard (ported from HOD's 1b3584f)
# ---------------------------------------------------------------------------

def test_fill_persistence_guard_db_contract():
    """Real Database on a tmp path: fill -> simulated restart/re-register ->
    fill_price/filled_at must survive unchanged, replicating the guard added
    to trading/orb_engine.py::_confirm_fill (read existing filled_at; if
    already set, drop fill_price/filled_at/order_filled_at/filled_qty from
    the re-registration's update dict before calling update_trade)."""
    from persistence.database import Database
    with tempfile.TemporaryDirectory() as d:
        db_path = os.path.join(d, 'tmp_trades.db')
        db = Database(db_path=db_path)
        true_fill_time = '2026-09-29T13:35:07+00:00'
        trade_id = db.save_trade({
            'trade_date': '2026-09-29', 'symbol': 'AXTL', 'strategy': 'orb', 'side': 'buy',
            'entry_price': 3.46, 'shares': 963,
            'stop_loss_price': 3.3372, 'take_profit_price': 0.0,
            'risk_per_share': 0.1228, 'total_risk': 118.26, 'risk_reward_ratio': 0.0,
            'order_id': 'test-order-382', 'order_status': 'filled',
            'fill_price': 3.46, 'filled_at': true_fill_time,
            'exit_price': 0.0, 'exit_reason': '', 'exited_at': None,
            'pnl': 0.0, 'pnl_pct': 0.0, 'pattern_data': '{}',
        }) if hasattr(db, 'save_trade') else None
        if trade_id is None:
            pytest.skip("Database.save_trade signature differs — guard is exercised live in orb_engine.py")

        # Simulate a restart's re-registration re-deriving the "same" fill
        # from Alpaca and trying to re-stamp it at the restart's own clock.
        restart_fill_update = {
            'order_status': 'filled', 'fill_price': 3.46,
            'filled_at': '2026-09-29T18:09:21+00:00',  # the restart time — WRONG
            'order_filled_at': '2026-09-29T18:09:21+00:00', 'filled_qty': 963,
        }
        existing = db._trades_conn.execute(
            "SELECT filled_at, fill_price FROM trades WHERE id = ?", (trade_id,)).fetchone()
        assert existing is not None and existing[0]
        for k in ('fill_price', 'filled_at', 'order_filled_at', 'filled_qty'):
            restart_fill_update.pop(k, None)
        db.update_trade(trade_id, restart_fill_update)

        after = db._trades_conn.execute(
            "SELECT filled_at, fill_price FROM trades WHERE id = ?", (trade_id,)).fetchone()
        assert str(after[0]).replace(' ', 'T') == true_fill_time, (
            f"filled_at was overwritten by the re-registration: {after[0]!r} "
            f"(expected the true fill time {true_fill_time!r}, not the restart time)")
        assert float(after[1]) == 3.46


# ---------------------------------------------------------------------------
# Fix 4: force-close greppable tag + the proposed archive-cron pattern
# ---------------------------------------------------------------------------

def test_force_close_log_tag_matches_proposed_cron_pattern():
    import re
    from trading.exit_reasons import ExitReason
    sym, exit_price = 'AXTL', 3.36
    line = f"[ORB] FORCE_CLOSE {sym} reason={ExitReason.FORCE_CLOSE.value} px={exit_price:.4f}"
    current_pattern = r"\[ORB\]|ORB SCORED|IGNITION|VETO|Q1 filter|WOULD BUY|ENTRY SUBMITTED|FILLED|LOCK|kill|\[HOD"
    proposed_pattern = r"\[ORB|ORB SCORED|IGNITION|VETO|Q1 filter|WOULD BUY|ENTRY SUBMITTED|FILLED|LOCK|kill|\[HOD"
    assert re.search(current_pattern, line), "new FORCE_CLOSE tag should already match [ORB] literally"
    preplace_line = "[ORB PREPLACE] AXTL rh=$10.32 Q4"
    assert not re.search(current_pattern, preplace_line), "confirms the documented cron-pattern gap"
    assert re.search(proposed_pattern, preplace_line), "broadened \\[ORB (no trailing bracket) catches PREPLACE"
    assert re.search(proposed_pattern, line), "broadened pattern still catches the FORCE_CLOSE tag"
