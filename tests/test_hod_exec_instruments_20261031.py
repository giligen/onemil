"""docs/hod_paper_exec_instruments_spec_20261031.md — the HOD paper book's two execution instruments:

Change 1: the NBBO snapshot at the moment the engine sees the cross of a resting entry lands on the booked trade row
          (entry_quote_bid/ask/bid_size/ask_size/spread, ORB convention spread = ask - bid in $); missing -> NULL + one
          WARNING per symbol-day.
Change 2: the passive 14:00 squeeze — the take-profit bracket leg of every open position below `squeeze_max_r` is
          replaced to the CURRENT ASK once; exit reason 'squeeze', pricing_method 'squeeze_ask'. OFF by default.
All collaborators are MagicMock(spec=...) fixtures from tests/conftest.py; the engine is the real HodBreakEngine.
"""
import logging
import time
from datetime import datetime, timezone

import pytest

from trading.hod_break_engine import HodBreakEngine, Position
from tests.test_hod_break_engine import admit
from tests.test_hod_live_resting import BIG_VOL_ARM, live_engine


def _engine(alpaca, db, sm, stream, tmp_path, minute=840, **over):
    """A live-path engine whose wall clock sits at `minute` ET (14:00 = 840) and whose broker mock replaces to ids tp-N-r."""
    e = live_engine(alpaca, db, sm, stream, tmp_path, **over)
    e._minute_of_day = lambda: minute
    alpaca.replace_order_limit_price.side_effect = lambda oid, px: {'id': f'{oid}-r', 'status': 'accepted'}
    return e


def _position(sym='ABC', fill=10.0, stop=9.0, target=12.0, tp='tp-ABC', sl='sl-ABC', shares=100, trade_id=11):
    """An OPEN bracket position: R = fill - stop = 1.0, target = fill + 2R."""
    now = datetime.now(timezone.utc)
    return Position(symbol=sym, trade_id=trade_id, order_id=f'o-{sym}', shares=shares, limit_price=fill, stop=stop, target=target,
                    level=fill, submitted_at=now, tp_leg_id=tp, sl_leg_id=sl, fill_price=fill, filled_at=now, status='open',
                    client_order_id=f'hod-{sym}', pattern_data={'tp_leg_id': tp, 'sl_leg_id': sl})


def _nbbo(sm, bid, ask, bs=300, asz=200):
    """Quote-watch snapshot the squeeze (and the cross sizes) read."""
    sm.get_quote_watch_snapshot.return_value = {'latest_bid': bid, 'latest_ask': ask, 'latest_bid_size': bs, 'latest_ask_size': asz}


def _update_calls(db, trade_id):
    return [c.args[1] for c in db.update_trade.call_args_list if c.args[0] == trade_id]


# ------------------------------------------------------------------------------------------- Change 1: entry quote capture
def _arm_and_cross(e, sm, quote_ts, bid=10.98, ask=11.02):
    """Arm a live order on ABC, then feed the cross print with the print-watch quote (bid, ask, quote_ts) cached."""
    admit(e)
    cand = e.candidates['ABC']
    arm = dict(BIG_VOL_ARM)
    e._arm_live_order(cand, arm)
    cand.resting_arm = arm
    sm.get_print_quote.return_value = (bid, ask, quote_ts)
    e._on_trade_print('ABC', arm['trigger'] + 0.01, 100, time.time())
    return cand


def _fill(e, stream, cand, qty=150, px=11.02):
    coid = cand.live_order['coid']
    stream.snapshot_by_client_prefix.return_value = {coid: {'status': 'filled', 'filled_qty': qty, 'filled_avg_price': px, 'client_order_id': coid}}
    e._poll_live_fills()


class TestEntryQuoteCapture:
    def test_snapshot_at_the_cross_lands_on_the_booked_trade_row_all_five_columns(
            self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path):
        e = live_engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path)
        _nbbo(hod_live_sm, 10.98, 11.02, bs=300, asz=200)
        cand = _arm_and_cross(e, hod_live_sm, time.time())
        lo = cand.live_order
        assert (lo['cross_bid'], lo['cross_ask'], lo['cross_bid_size'], lo['cross_ask_size']) == (10.98, 11.02, 300, 200) and lo['cross_ts']
        _fill(e, hod_live_stream, cand)
        eq = [u for u in _update_calls(hod_live_db, 7) if 'entry_quote_bid' in u]
        assert len(eq) == 1
        assert eq[0]['entry_quote_bid'] == 10.98 and eq[0]['entry_quote_ask'] == 11.02
        assert eq[0]['entry_quote_bid_size'] == 300 and eq[0]['entry_quote_ask_size'] == 200
        assert eq[0]['entry_quote_spread'] == pytest.approx(0.04)          # ask - bid in dollars, ORB's convention

    def test_partial_top_up_does_not_rewrite_the_quote(self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path):
        e = live_engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path)
        cand = _arm_and_cross(e, hod_live_sm, time.time())
        coid = cand.live_order['coid']
        hod_live_stream.snapshot_by_client_prefix.return_value = {coid: {'status': 'partially_filled', 'filled_qty': 50, 'filled_avg_price': 11.02, 'client_order_id': coid}}
        e._poll_live_fills()
        _fill(e, hod_live_stream, cand)
        assert len([u for u in _update_calls(hod_live_db, 7) if 'entry_quote_bid' in u]) == 1

    def test_missing_snapshot_leaves_nulls_and_warns_once(self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path, caplog):
        e = live_engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path)
        admit(e)
        cand = e.candidates['ABC']
        e._arm_live_order(cand, dict(BIG_VOL_ARM))                         # no cross print ever seen -> no snapshot
        with caplog.at_level(logging.WARNING):
            coid = cand.live_order['coid']
            hod_live_stream.snapshot_by_client_prefix.return_value = {coid: {'status': 'partially_filled', 'filled_qty': 50, 'filled_avg_price': 11.02, 'client_order_id': coid}}
            e._poll_live_fills()
            _fill(e, hod_live_stream, cand)                                   # second booking of the same symbol-day
        assert not [u for u in _update_calls(hod_live_db, 7) if 'entry_quote_bid' in u]
        msgs = [r.message for r in caplog.records if 'entry quote missing ABC' in r.message]
        assert len(msgs) == 1 and msgs[0].startswith('[HOD] entry quote missing ABC: ')

    def test_stale_quote_is_not_stored_and_the_warning_says_why(self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path, caplog):
        e = live_engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path)
        cand = _arm_and_cross(e, hod_live_sm, time.time() - 60.0)           # quote 60 s old > max_quote_age_s (5)
        assert 'cross_bid' not in cand.live_order
        with caplog.at_level(logging.WARNING):
            _fill(e, hod_live_stream, cand)
        assert not [u for u in _update_calls(hod_live_db, 7) if 'entry_quote_bid' in u]
        assert any('entry quote missing ABC' in r.message and 'old at the cross' in r.message for r in caplog.records)


# ------------------------------------------------------------------------------------------- Change 2: the 14:00 squeeze
class TestSqueezeRule:
    def _run(self, e, pos):
        e.positions[pos.symbol] = pos
        e._run_squeeze()

    def test_unrealised_0p3R_gets_one_replace_to_the_ask(self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path, caplog):
        e = _engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path, squeeze_enabled=True)
        _nbbo(hod_live_sm, 10.29, 10.31)                                      # mid 10.30 -> +0.30 R
        pos = _position()
        with caplog.at_level(logging.INFO):
            self._run(e, pos)
        hod_live_alpaca.replace_order_limit_price.assert_called_once_with('tp-ABC', 10.31)
        assert pos.tp_leg_id == 'tp-ABC-r' and pos.pattern_data['squeeze_done'] is True
        assert (e.squeeze_placed, e.squeeze_filled) == (1, 0)
        assert any(r.message == '[HOD] SQUEEZE ABC: unreal=+0.30R tp 12.00 -> ask 10.31' for r in caplog.records)

    def test_exactly_the_threshold_is_untouched_strict_less_than(self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path):
        e = _engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path, squeeze_enabled=True)
        _nbbo(hod_live_sm, 10.49, 10.51)                                      # mid 10.50 -> exactly +0.50 R
        self._run(e, _position())
        hod_live_alpaca.replace_order_limit_price.assert_not_called()

    def test_runner_at_0p8R_is_untouched(self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path):
        e = _engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path, squeeze_enabled=True)
        _nbbo(hod_live_sm, 10.79, 10.81)
        pos = _position()
        self._run(e, pos)
        hod_live_alpaca.replace_order_limit_price.assert_not_called()
        assert pos.tp_leg_id == 'tp-ABC' and e.squeeze_placed == 0

    def test_second_tick_places_no_second_replace(self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path):
        e = _engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path, squeeze_enabled=True)
        _nbbo(hod_live_sm, 10.29, 10.31)
        pos = _position()
        self._run(e, pos)
        _nbbo(hod_live_sm, 10.19, 10.21)                                      # the market moved: never re-chased
        e._run_squeeze(); e._run_squeeze()
        assert hod_live_alpaca.replace_order_limit_price.call_count == 1

    def test_before_the_squeeze_minute_and_after_the_flat_nothing_happens(self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path):
        _nbbo(hod_live_sm, 10.29, 10.31)
        for minute in (839, 955):
            e = _engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path, minute=minute, squeeze_enabled=True)
            self._run(e, _position())
        hod_live_alpaca.replace_order_limit_price.assert_not_called()

    def test_ask_not_above_bid_is_skipped_with_a_warning(self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path, caplog):
        e = _engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path, squeeze_enabled=True)
        _nbbo(hod_live_sm, 10.32, 10.28)                                      # crossed book, mid 10.30 -> +0.3 R
        pos = _position()
        with caplog.at_level(logging.WARNING):
            self._run(e, pos); e._run_squeeze()
        hod_live_alpaca.replace_order_limit_price.assert_not_called()
        assert pos.tp_leg_id == 'tp-ABC'
        assert len([r for r in caplog.records if 'SQUEEZE skip ABC' in r.message and 'ask 10.28 <= bid 10.32' in r.message]) == 1

    def test_squeeze_enabled_absent_from_config_does_nothing(self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path):
        e = _engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path)         # no squeeze keys at all
        assert (e.squeeze_enabled, e.squeeze_minute, e.squeeze_max_r) == (False, 840, 0.5)
        _nbbo(hod_live_sm, 10.29, 10.31)
        self._run(e, _position())
        hod_live_alpaca.replace_order_limit_price.assert_not_called()

    def test_position_without_a_tp_leg_is_skipped_with_one_warning_and_no_order(self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path, caplog):
        e = _engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path, squeeze_enabled=True)
        _nbbo(hod_live_sm, 10.29, 10.31)
        with caplog.at_level(logging.WARNING):
            self._run(e, _position(tp=None)); e._run_squeeze()
        hod_live_alpaca.replace_order_limit_price.assert_not_called()
        hod_live_alpaca.submit_limit_sell_order.assert_not_called()
        assert len([r for r in caplog.records if 'SQUEEZE skip ABC: no tp_leg_id' in r.message]) == 1

    def test_no_quote_falls_back_to_the_last_bar_close_with_a_warning_and_places_nothing_without_an_ask(
            self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path, caplog):
        e = _engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path, squeeze_enabled=True)
        hod_live_sm.get_quote_watch_snapshot.return_value = None
        hod_live_alpaca.get_latest_quote.side_effect = Exception('no quote')
        admit(e)
        cand = e.candidates['ABC']; cand.have[5] = True; cand.ohlcv[5] = [10.3, 10.4, 10.2, 10.3, 1000.0]
        with caplog.at_level(logging.WARNING):
            self._run(e, _position())
        hod_live_alpaca.replace_order_limit_price.assert_not_called()
        assert any('SQUEEZE skip ABC: no quote' in r.message and '10.30' in r.message for r in caplog.records)

    def test_replace_failure_is_an_error_and_leaves_the_leg(self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path, caplog):
        e = _engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path, squeeze_enabled=True)
        hod_live_alpaca.replace_order_limit_price.side_effect = Exception('boom')
        _nbbo(hod_live_sm, 10.29, 10.31)
        pos = _position()
        with caplog.at_level(logging.ERROR):
            self._run(e, pos)
        assert pos.tp_leg_id == 'tp-ABC' and e.squeeze_placed == 0 and not pos.pattern_data.get('squeeze_done')
        assert any('take-profit replace to 10.31 FAILED' in r.message for r in caplog.records)


class TestSqueezeLegFillBooking:
    def test_leg_fill_after_the_replace_books_reason_squeeze_and_pricing_method_squeeze_ask(
            self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path):
        e = _engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path, squeeze_enabled=True)
        _nbbo(hod_live_sm, 10.29, 10.31)
        pos = _position()
        e.positions['ABC'] = pos
        e._run_squeeze()
        st = {'status': 'filled', 'filled_qty': 100, 'filled_avg_price': 10.31}
        e._book_leg_fill(pos, pos.tp_leg_id, st, e._tp_reason(pos))
        assert 'ABC' not in e.positions and e.squeeze_filled == 1
        upd = _update_calls(hod_live_db, 11)[-1]
        assert upd['exit_reason'] == 'squeeze' and upd['exit_pricing_method'] == 'squeeze_ask'
        assert upd['exit_price'] == 10.31 and upd['exit_quote_bid'] == 10.29 and upd['exit_quote_ask'] == 10.31

    def test_an_unsqueezed_target_fill_is_still_target(self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path):
        e = _engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path, squeeze_enabled=True)
        pos = _position(); e.positions['ABC'] = pos
        e._book_leg_fill(pos, pos.tp_leg_id, {'status': 'filled', 'filled_qty': 100, 'filled_avg_price': 12.0}, e._tp_reason(pos))
        upd = _update_calls(hod_live_db, 11)[-1]
        assert upd['exit_reason'] == 'target' and upd['exit_pricing_method'] == 'bracket_target' and e.squeeze_filled == 0

    def test_replaced_leg_is_followed_to_its_successor_under_the_squeeze_reason(self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path):
        e = _engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path, squeeze_enabled=True)
        pos = _position(); pos.pattern_data['squeeze_done'] = True
        hod_live_stream.get_status.side_effect = lambda oid: {'status': 'replaced', 'replaced_by': 'tp-new'} if oid == 'tp-ABC' else {'status': 'accepted', 'filled_qty': 0}
        leg, st = e._leg_status(pos, 'tp-ABC', 'squeeze')
        assert leg == 'tp-new' and pos.tp_leg_id == 'tp-new'


# ------------------------------------------------------------------------------------------- integration
class TestSqueezeIntegration:
    def test_two_positions_one_replace_then_fill_counters_and_the_flat_sees_one_position(
            self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path, caplog):
        e = _engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path, squeeze_enabled=True)
        a = _position('ABC', fill=10.0, stop=9.0, target=12.0, tp='tp-A', sl='sl-A', trade_id=21)
        b = _position('XYZ', fill=20.0, stop=19.0, target=22.0, tp='tp-X', sl='sl-X', trade_id=22)
        e.positions.update({'ABC': a, 'XYZ': b})
        snaps = {'ABC': {'latest_bid': 10.29, 'latest_ask': 10.31, 'latest_bid_size': 1, 'latest_ask_size': 1},    # +0.30 R
                 'XYZ': {'latest_bid': 20.79, 'latest_ask': 20.81, 'latest_bid_size': 1, 'latest_ask_size': 1}}    # +0.80 R
        hod_live_sm.get_quote_watch_snapshot.side_effect = lambda s: snaps.get(s)
        statuses = {}
        hod_live_stream.get_status.side_effect = lambda oid: statuses.get(oid, {'status': 'accepted', 'filled_qty': 0})
        hod_live_stream.snapshot_by_client_prefix.return_value = {}

        with caplog.at_level(logging.INFO):
            e.process_tick()                                                    # 14:00 tick
        calls = hod_live_alpaca.replace_order_limit_price.call_args_list
        assert [(c.args[0], c.args[1]) for c in calls] == [('tp-A', 10.31)]     # exactly one replace, right leg, right price
        assert (e.squeeze_placed, e.squeeze_filled) == (1, 0)
        assert a.tp_leg_id == 'tp-A-r' and b.tp_leg_id == 'tp-X'

        e.process_tick()                                                         # later tick: no second replace
        assert hod_live_alpaca.replace_order_limit_price.call_count == 1

        statuses['tp-A-r'] = {'status': 'filled', 'filled_qty': 100, 'filled_avg_price': 10.31}
        e.process_tick()                                                         # the replaced leg fills
        assert (e.squeeze_placed, e.squeeze_filled) == (1, 1)
        assert list(e.positions) == ['XYZ']
        upd = _update_calls(hod_live_db, 21)[-1]
        assert upd['exit_reason'] == 'squeeze' and upd['exit_pricing_method'] == 'squeeze_ask' and upd['order_status'] == 'closed'

        e._minute_of_day = lambda: 955                                            # 15:55 flat sees ONE position only
        hod_live_alpaca.get_open_positions.return_value = [{'symbol': 'XYZ', 'qty': 100, 'side': 'long'}]
        with caplog.at_level(logging.INFO):
            e.force_close_all()
        assert [p.symbol for p in e.positions.values() if p.status == 'open'] == ['XYZ']
        assert not any('ABC' in r.message and 'FORCE CLOSE' in r.message for r in caplog.records)
        assert any('squeeze=1/1' in r.message for r in caplog.records)
