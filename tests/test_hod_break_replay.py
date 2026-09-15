"""HOD-break REPLAY integration test (2026-09-15, after three input-parity defects in two dry days).

Feeds the ENGINE the same world the SPEC sees — a multi-symbol day of 1-min bars — through the live seams:
  * a simulated per-minute broad scan calling `on_mover` when the latest close is >= the admission threshold
    above the open (the scanner hook), then `process_tick`
  * a simulated StopMonitor bar stream delivering, each minute, the DataFrame of bars SINCE SUBSCRIPTION
    (exactly what `_on_bar` hands out) for every subscribed symbol
  * a REST backfill mock returning bars from 09:30 to the current minute
and asserts the engine's submitted orders equal the spec's `simulate` signals (symbol, level, stop) — minute
by minute, no look-ahead. Covers: late admission (the CRWL class), stream-only bars (the DBI class), R on the
ask, and the documented residual miss (a stock that never prints the admission threshold before its break).
"""
import sqlite3
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.stop_monitor import StopMonitor
from trading.hod_break import HodBreakParams, simulate
from trading.hod_break_engine import HodBreakEngine
from tests.test_hod_break_engine import cfg

BASE = datetime(2026, 9, 15, 13, 30, tzinfo=timezone.utc)          # 09:30 ET


def tape(open_px, path):
    """path: list of (high_pct_from_open, low_pct, close_pct, volume) per minute → bars (o,h,l,c,v)."""
    bars = []; prev_close = open_px
    for hp, lp, cp, vol in path:
        o = prev_close; h = open_px * (1 + hp / 100); l = open_px * (1 + lp / 100); c = open_px * (1 + cp / 100)
        bars.append((round(o, 2), round(max(h, o, c), 2), round(min(l, o, c), 2), round(c, 2), vol)); prev_close = c
    return bars


def drive_consolidate_break(open_px, drive_to, consol_lo, consol_hi, break_to, pre=(), volume=20000):
    """generic day: `pre` minutes of quiet, a drive to +drive_to%, 5 minutes between consol_lo/hi %, then a break to +break_to%."""
    path = [(0.5, -0.5, 0.2, volume)] * len(pre) if pre else []
    path += [(drive_to * 0.5, 0, drive_to * 0.5, volume * 3), (drive_to, drive_to * 0.4, drive_to * 0.95, volume * 4)]
    path += [(consol_hi, consol_lo, (consol_lo + consol_hi) / 2, volume)] * 5
    path += [(break_to, consol_hi - 0.3, break_to - 0.3, volume * 3)]
    path += [(break_to + 0.2, break_to - 0.5, break_to, volume)] * 3
    return tape(open_px, path)


DAY = {
    # A: drives to +8% (admitted early at >= 3.5%), consolidates, breaks → must be traded
    'ZZTA': drive_consolidate_break(30.0, 8.0, 6.6, 7.9, 8.6),
    # B (the CRWL class): only +4.5% before the break bar; the break bar itself reaches +5.3%. Admitted at 3.5% → traded.
    'ZZTB': drive_consolidate_break(40.0, 4.5, 4.0, 4.5, 5.3),
    # C: the HOD (+5.5%) is a WICK — every close before the break bar stays under the +3.5% admission threshold, so a
    # lagged scan admits it only after the break bar closed (+5.8%) → the documented residual miss (streaming catches it)
    'ZZTC': tape(25.0, [(5.5, -0.2, 3.2, 80000)] + [(3.4, 2.0, 3.0, 20000)] * 5 + [(6.0, 2.9, 5.8, 60000)] + [(6.2, 5.3, 5.8, 20000)] * 3),
    # D: below the $20 floor → never admitted
    'ZZTD': drive_consolidate_break(12.0, 8.0, 6.6, 7.9, 8.6),
}
ADV = {s: 8_000_000 for s in DAY}


def arrays(bars):
    o = np.array([b[0] for b in bars]); h = np.array([b[1] for b in bars]); l = np.array([b[2] for b in bars]); c = np.array([b[3] for b in bars]); v = np.array([b[4] for b in bars], float)
    m = np.arange(570, 570 + len(bars)); return o, h, l, c, v, m


def df_of(bars, start_idx, end_idx):
    return pd.DataFrame([{'timestamp': BASE + timedelta(minutes=i), 'open': b[0], 'high': b[1], 'low': b[2], 'close': b[3], 'volume': b[4]} for i, b in enumerate(bars) if start_idx <= i < end_idx])


@pytest.fixture
def world(tmp_path):
    trades = tmp_path / 't.db'; con = sqlite3.connect(trades); con.execute("create table trades (id integer primary key, strategy text, trade_date text, pnl real, symbol text, order_status text)"); con.commit(); con.close()
    alp = MagicMock(spec=AlpacaClient); db = MagicMock(spec=Database); sm = MagicMock(spec=StopMonitor); sm.polling_mode = False
    db._trades_path = str(trades); db.get_active_universe.return_value = [{'symbol': s, 'avg_volume_daily': ADV[s]} for s in DAY]; db.save_trade.return_value = 1; db.get_open_trades.return_value = []
    alp.submit_bracket_order.side_effect = lambda **kw: {'id': f"o-{kw['symbol']}", 'status': 'accepted', 'legs': [{'id': 'tp', 'limit_price': kw['tp_price'], 'stop_price': None}, {'id': 'sl', 'limit_price': None, 'stop_price': kw['sl_price']}]}
    alp.get_order.return_value = {'status': 'accepted', 'filled_qty': 0}; alp.get_open_positions.return_value = []; alp.cancel_order.return_value = True
    state = {'now': 0}                                                    # minutes since 09:30, the clock of the simulated day
    alp.get_1min_bars_multi.side_effect = lambda syms, lookback_minutes=30: {s: df_of(DAY[s], 0, state['now']) for s in syms}   # backfill: 09:30 .. last CLOSED bar
    alp.get_latest_quote.side_effect = lambda s: {'bid_price': round(DAY[s][state['now'] - 1][3] * 0.999, 2), 'ask_price': round(DAY[s][state['now'] - 1][3] * 1.001, 2)}
    e = HodBreakEngine(alp, db, sm, cfg=cfg(min_price=20.0, admit_above_open_pct=3.5, stream_universe=False))
    return e, alp, sm, state


@pytest.fixture
def streamed_world(world, tmp_path):
    """the core fix: every universe name streams from 09:30 — no admission step"""
    e, alp, sm, state = world
    e.stream_universe = True; e.universe_min_prev_close = 17.0; e.stream_list_dir = str(tmp_path)
    e._last_close = {s: DAY[s][0][0] for s in DAY}
    return e, alp, sm, state


def run_day(e, alp, sm, state, use_scan=True, scan_lag=1):
    """minute loop: at minute t the bars [0..t) are closed; the broad scan sees the close of bar t-1-scan_lag (the live
    scan cycle is ~60s+, so an admission lands on average one bar late); stream delivers since-subscription bars."""
    subscribed = {}; sm.subscribe_bars.side_effect = lambda s: subscribed.setdefault(s, state['now'])
    sm.subscribe_bars_many.side_effect = lambda syms: [subscribed.setdefault(s, state['now']) for s in syms] and len(syms)
    n = max(len(b) for b in DAY.values())
    with patch.object(HodBreakEngine, '_minute_of_day', side_effect=lambda: 570 + state['now']):
        e._roll_session()
        for t in range(1, n + 1):
            state['now'] = t
            for s, bars in (DAY.items() if use_scan else []):            # the broad scan: latest trade vs the 09:30 open
                k = t - 1 - scan_lag
                if 0 <= k < len(bars):
                    above = (bars[k][3] / bars[0][0] - 1) * 100
                    if above >= e.admit_above_open_pct: e.on_mover(s, price=bars[k][3], day_open=bars[0][0], cum_volume=sum(b[4] for b in bars[:k + 1]), above_open_pct=above)
            e.process_tick()                                             # admits + backfills + evaluates
            for s, since in list(subscribed.items()):                    # the bar stream: since subscription only
                if since < t <= len(DAY[s]): e._on_bar_close(s, df_of(DAY[s], since, t))
            e.drain_bar_events()
    return {c.kwargs['symbol']: c.kwargs for c in alp.submit_bracket_order.call_args_list}


def test_engine_matches_the_spec_through_the_live_seams(world):
    e, alp, sm, state = world
    orders = run_day(e, alp, sm, state)
    spec = {s: simulate(*arrays(DAY[s]), ADV[s], HodBreakParams()) for s in DAY}
    assert spec['ZZTA'] is not None and spec['ZZTB'] is not None and spec['ZZTC'] is not None and spec['ZZTD'] is not None
    assert 'ZZTA' in orders and 'ZZTB' in orders, 'admitted-before-break symbols must be traded'
    for s in ('ZZTA', 'ZZTB'):
        assert orders[s]['sl_price'] == pytest.approx(spec[s].stop, abs=0.011), s
        assert orders[s]['limit_price'] == pytest.approx(round(spec[s].entry / (1 + 0.0) , 2), rel=0.01), s   # limit at level x 1.006 ≈ spec entry (next open) within 1%
    assert 'ZZTD' not in orders, 'below the price floor'
    assert 'ZZTC' not in orders, 'documented residual: a stock that never prints the admission threshold before its break is admitted one scan cycle after the break — missed'


def test_scan_admission_with_zero_latency_would_catch_the_break_bar(world):
    """the residual is LATENCY, not logic: a scan that runs exactly at the break bar's close admits + backfills + trades ZZTC at the spec's minute"""
    e, alp, sm, state = world
    orders = run_day(e, alp, sm, state, scan_lag=0)
    assert 'ZZTC' in orders


def test_stream_only_bars_never_evaluated(world):
    """the DBI class: without the backfill the engine must not act on a truncated day"""
    e, alp, sm, state = world
    alp.get_1min_bars_multi.side_effect = lambda syms, lookback_minutes=30: {}   # backfill silently fails
    orders = run_day(e, alp, sm, state)
    assert orders == {}, 'no order may be placed from stream-only bars'


def test_streamed_universe_catches_every_spec_signal(streamed_world):
    """with the whole universe streaming from 09:30 there is no admission gap: CCC (never +3.5% before its break) is traded too"""
    e, alp, sm, state = streamed_world
    orders = run_day(e, alp, sm, state, use_scan=False)
    spec = {s: simulate(*arrays(DAY[s]), ADV[s], HodBreakParams()) for s in DAY}
    for s in ('ZZTA', 'ZZTB', 'ZZTC'):
        assert s in orders, f'{s} must be traded from the streamed bars'
        assert orders[s]['sl_price'] == pytest.approx(spec[s].stop, abs=0.011), s
    assert 'ZZTD' not in orders, 'below the price floor'
    assert sm.subscribe_bars_many.called


def test_a_missed_break_never_becomes_a_later_break(world):
    """the spec trades a symbol's FIRST break only: a candidate admitted after its break is done for the day (stale_break)"""
    e, alp, sm, state = world
    run_day(e, alp, sm, state, scan_lag=1)
    assert e.candidates['ZZTC'].rejected_reason == 'stale_break'


def test_stream_outage_after_the_open_re_backfills_every_candidate(streamed_world):
    """a WebSocket reconnect after 09:30 invalidates the streamed day; candidates are re-backfilled from REST before any evaluation"""
    e, alp, sm, state = streamed_world
    sm.ws_generation = 1
    with patch.object(HodBreakEngine, '_minute_of_day', side_effect=lambda: 569):
        e._roll_session(); e.process_tick()                    # pre-open roll: the stream will carry the day; first sight of the generation
        assert all(c.backfill_ok and not c.needs_refill for c in e.candidates.values())
    with patch.object(HodBreakEngine, '_minute_of_day', side_effect=lambda: 570 + state['now']):
        state['now'] = 12; sm.ws_generation = 2; e.process_tick()
        assert alp.get_1min_bars_multi.called, 'the re-backfill must hit REST'
        assert all(not c.needs_refill for c in e.candidates.values() if c.rejected_reason is None), 'complete again after the backfill'


def test_stream_outage_before_the_open_loses_nothing(streamed_world):
    e, alp, sm, state = streamed_world
    sm.ws_generation = 1
    with patch.object(HodBreakEngine, '_minute_of_day', side_effect=lambda: 560):
        e._roll_session(); e.process_tick(); sm.ws_generation = 2; e.process_tick()
    assert not alp.get_1min_bars_multi.called and all(c.backfill_ok for c in e.candidates.values())


def test_target_is_re_anchored_to_the_actual_fill(world):
    """the spec's target = fill + 2R on the ACTUAL fill: the take-profit leg moves when the fill differs from the ask estimate"""
    e, alp, sm, state = world
    e.dry_run = False
    orders = run_day(e, alp, sm, state)
    pos = e.positions['ZZTA']; est_target = pos.target
    alp.replace_order_limit_price.return_value = {'id': 'tp2', 'status': 'new'}
    alp.get_order.side_effect = lambda oid: {'status': 'filled', 'filled_qty': pos.shares, 'filled_avg_price': pos.limit_price} if oid == pos.order_id else {'status': 'new', 'filled_qty': 0}   # filled at the cap, above the ask estimate
    e._process_pending_fills()
    alp.replace_order_limit_price.assert_called_once()
    leg, new_target = alp.replace_order_limit_price.call_args.args
    assert leg == 'tp' and new_target == pytest.approx(round(pos.limit_price + 2 * (pos.limit_price - pos.stop), 2)) and new_target > est_target
    assert e.positions['ZZTA'].target == new_target and e.positions['ZZTA'].tp_leg_id == 'tp2' and e.positions['ZZTA'].pattern_data['tp_leg_id'] == 'tp2'


def test_drain_thread_evaluates_without_the_scan_cycle(streamed_world):
    """bars are evaluated the moment they close: the drain thread alone (no process_tick) must produce the ZZTA order"""
    import time
    e, alp, sm, state = streamed_world
    with patch.object(HodBreakEngine, '_minute_of_day', side_effect=lambda: 570 + state['now']):
        e._roll_session(); e.start_drain_thread()
        try:
            for t in range(1, len(DAY['ZZTA']) + 1):
                state['now'] = t; e._on_bar_close('ZZTA', df_of(DAY['ZZTA'], 0, t)); time.sleep(0.5)   # one bar per 'minute': the drain batches a minute's bars for 0.4 s
            for _ in range(100):
                if alp.submit_bracket_order.called: break
                time.sleep(0.05)
        finally:
            e.shutdown_requested = True
    assert alp.submit_bracket_order.called and alp.submit_bracket_order.call_args.kwargs['symbol'] == 'ZZTA'


def test_outage_seen_by_the_drain_path_blocks_evaluation_until_the_refill(streamed_world):
    """9/15 reviews C/G: a reconnect after the open must stop evaluation at bar arrival (not at the next scanner tick), and
    a failed REST refill must keep the day blocked — the 09:30-bar shortcut must not declare a gapped day complete"""
    e, alp, sm, state = streamed_world
    sm.ws_generation = 1
    with patch.object(HodBreakEngine, '_minute_of_day', side_effect=lambda: 570 + state['now']):
        e._roll_session(); e.process_tick()
        for t in range(1, 6):
            state['now'] = t; e._on_bar_close('ZZTA', df_of(DAY['ZZTA'], t - 1, t)); e.drain_bar_events()
        sm.ws_generation = 2                                                     # the stream reconnected; bars 6..8 were never delivered
        alp.get_1min_bars_multi.side_effect = lambda syms, lookback_minutes=30: {}   # and the REST refill fails
        for t in range(9, len(DAY['ZZTA']) + 1):
            state['now'] = t; e._on_bar_close('ZZTA', df_of(DAY['ZZTA'], t - 1, t)); e.drain_bar_events(); e.process_tick()
        assert not alp.submit_bracket_order.called and e.candidates['ZZTA'].needs_refill, 'a gapped day must never be judged'
        alp.get_1min_bars_multi.side_effect = lambda syms, lookback_minutes=30: {s: df_of(DAY[s], 0, state['now']) for s in syms}
        e.process_tick()                                                        # the refill lands: the whole day is re-scanned from bar 0
        assert not e.candidates['ZZTA'].needs_refill and e.candidates['ZZTA'].rejected_reason == 'stale_break'


def test_a_late_earlier_bar_forces_a_rescan(streamed_world):
    """9/15 review C: a bar that lands before already-scanned bars changes HOD/volume — detect must rescan from it"""
    e, alp, sm, state = streamed_world
    with patch.object(HodBreakEngine, '_minute_of_day', side_effect=lambda: 570 + state['now']):
        e._roll_session()
        bars = DAY['ZZTA']; n = len(bars)
        order = [i for i in range(n) if i != 2] + [2]                           # bar index 2 (the drive bar) arrives last
        for k, i in enumerate(order):
            state['now'] = max(state['now'], i + 1); e._on_bar_close('ZZTA', df_of(bars, i, i + 1)); e.drain_bar_events()
        cand = e.candidates['ZZTA']
        assert cand.next_idx <= 2 or cand.rejected_reason is not None
        spec = simulate(*arrays(bars), ADV['ZZTA'], HodBreakParams())
        assert cand.rejected_reason in ('stale_break', 'ordered') and spec is not None
