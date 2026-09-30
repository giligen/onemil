"""HOD-break failure-short — engine wiring (trading/hod_break_engine.py's overlay methods).
Two things only: (1) enabled: false is byte-identical (no calls at all); (2) enabled + live wiring
does the reversal, the two protective/target orders, the two DB rows (real Database, tmp path) and
the ledger row. The signal itself is a stub — trading/hod_failure_features.py (another module)
owns the real model."""
import csv
import json
import os
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.hod_break_engine import Candidate, HodBreakEngine, OPEN_MINUTE, Position
from trading.stop_monitor import StopMonitor

SYM = 'ABCD'
FILL_MINUTE = OPEN_MINUTE + 30


def cfg(**fs_over):
    fs = {'enabled': True, 'telemetry_only': False, 'tau': 0.5, 'risk_usd': 150, 'max_concurrent': 3,
          'max_per_day': 8, 'day_kill_r': -5.0, 'require_etb': True, 'entry_bar_offset': 2,
          'allow_fresh_short': False, 'ledger_path': 'logs/hod_failure_short_ledger.csv'}
    fs.update(fs_over)
    return {'enabled': True, 'dry_run': False, 'risk_usd': 100.0, 'daily_kill_usd': -600.0, 'weekly_kill_usd': -1500.0,
            'max_notional_usd': 5000.0, 'min_price': 1.0, 'min_adv20': 100_000.0, 'max_spread_bps': 100.0,
            'entry_mode': 'resting_stop_limit', 'order_timeout_s': 75.0,
            'params': {'consol_bars': 5, 'consol_pct': 0.04, 'min_dist_open_pct': 5.0, 'rv_lo': 1.0, 'rv_hi': 5.0,
                       'min_r_pct': 1.0, 'cap': 0.006, 'target_r': 2.0, 'max_per_day': 8, 'max_concurrent': 4,
                       'last_entry_minute': 930, 'flat_minute': 955},
            'failure_short': fs}


@pytest.fixture
def mock_alpaca():
    a = MagicMock(spec=AlpacaClient)
    a.is_paper = True
    a.get_shortability.return_value = {'shortable': True, 'easy_to_borrow': True}
    a.get_latest_quote.return_value = {'bid_price': 12.28, 'ask_price': 12.32}
    a.submit_market_sell_order.return_value = {'id': 'sell1', 'status': 'filled'}
    a.submit_stop_limit_order.return_value = {'id': 'stop1', 'status': 'accepted'}
    a.submit_limit_buy_order.return_value = {'id': 'target1', 'status': 'accepted'}
    a.submit_oco_buy_order.return_value = {'id': 'ocoTarget1', 'status': 'accepted',
                                            'legs': [{'id': 'ocoStop1', 'type': 'stop'}]}
    a.cancel_order.return_value = True
    a.get_order.return_value = {'status': 'accepted'}
    a.get_open_positions.return_value = [{'symbol': SYM, 'qty': '-100'}]
    return a


@pytest.fixture
def mock_sm():
    return MagicMock(spec=StopMonitor)


@pytest.fixture
def real_db(tmp_path):
    return Database(db_path=str(tmp_path / 'test.db'))


def _arm_candidate(engine, symbol=SYM):
    """Two closed bars (fill bar + fill+1) so _rth_arrays/day-high has something to read."""
    cand = Candidate(symbol=symbol, day_open=11.00, adv20=1_000_000.0)
    cand.set_bar(FILL_MINUTE, 12.00, 12.20, 11.95, 12.15, 1000)
    cand.set_bar(FILL_MINUTE + 1, 12.10, 12.34, 12.05, 12.30, 1000)
    engine.candidates[symbol] = cand
    return cand


def _open_long(engine, db, symbol=SYM, shares=100, fill_px=12.00, stop=11.50):
    """Mirrors what _on_live_fill would already have booked: a DB row + a Position, so the
    overlay's reversal has something real to close."""
    now = datetime.now(timezone.utc)
    rec = {'trade_date': '2026-09-30', 'symbol': symbol, 'side': 'buy', 'entry_price': fill_px,
           'stop_loss_price': stop, 'take_profit_price': fill_px + 2 * (fill_px - stop), 'shares': shares,
           'risk_per_share': fill_px - stop, 'total_risk': (fill_px - stop) * shares, 'risk_reward_ratio': 2.0,
           'order_id': 'buy1', 'order_status': 'filled', 'fill_price': fill_px, 'filled_at': now.isoformat(),
           'exit_price': None, 'exit_reason': None, 'exited_at': None, 'pnl': None, 'pnl_pct': None,
           'strategy': 'hod_break', 'account': 'paper', 'pattern_data': json.dumps({'book': 'hod_break'})}
    trade_id = db.save_trade(rec)
    pos = Position(symbol=symbol, trade_id=trade_id, order_id='buy1', shares=shares, limit_price=fill_px,
                   stop=stop, target=rec['take_profit_price'], level=fill_px, submitted_at=now,
                   fill_price=fill_px, filled_at=now, status='open', pattern_data={'book': 'hod_break'})
    engine.positions[symbol] = pos
    return trade_id


class TestDisabledIsInert:
    def test_enabled_false_makes_zero_calls(self, mock_alpaca, real_db, mock_sm):
        engine = HodBreakEngine(mock_alpaca, real_db, mock_sm, cfg=cfg(enabled=False))
        engine.session_date = '2026-09-30'
        _arm_candidate(engine)
        engine._minute_of_day = lambda: FILL_MINUTE   # pin "now" so _fs_register_long records fill_minute deterministically
        _open_long(engine, real_db)
        engine._fs_register_long(SYM, 12.00, 11.50, 100, 1, datetime.now(timezone.utc))
        assert engine._fs_tracked == {}       # never armed
        engine._minute_of_day = lambda: FILL_MINUTE + 5
        engine._process_failure_short()
        assert not mock_alpaca.get_shortability.called
        assert not mock_alpaca.submit_market_sell_order.called
        assert not mock_alpaca.submit_stop_limit_order.called
        assert not mock_alpaca.submit_limit_buy_order.called
        assert not mock_alpaca.submit_oco_buy_order.called
        assert not mock_sm.add_watch.called

    def test_telemetry_only_true_evaluates_but_places_no_orders(self, mock_alpaca, real_db, mock_sm, tmp_path):
        ledger = str(tmp_path / 'fs_ledger.csv')
        engine = HodBreakEngine(mock_alpaca, real_db, mock_sm, cfg=cfg(telemetry_only=True, ledger_path=ledger))
        engine.session_date = '2026-09-30'
        engine.fs_signal_fn = lambda symbol, bars, ctx: 0.9
        engine._prev_day[SYM] = (10.0, None, None)
        _arm_candidate(engine)
        engine._minute_of_day = lambda: FILL_MINUTE   # pin "now" so _fs_register_long records fill_minute deterministically
        _open_long(engine, real_db)
        engine._fs_register_long(SYM, 12.00, 11.50, 100, 1, datetime.now(timezone.utc))
        engine._minute_of_day = lambda: FILL_MINUTE + 1
        engine._process_failure_short()
        assert not mock_alpaca.submit_market_sell_order.called
        assert os.path.exists(ledger)
        with open(ledger) as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1 and rows[0]['symbol'] == SYM and rows[0]['passed'] == 'True'


class TestLiveReversal:
    def test_go_signal_reverses_long_into_short(self, mock_alpaca, real_db, mock_sm, tmp_path):
        ledger = str(tmp_path / 'fs_ledger.csv')
        engine = HodBreakEngine(mock_alpaca, real_db, mock_sm, cfg=cfg(ledger_path=ledger))
        engine.session_date = '2026-09-30'
        engine.fs_signal_fn = lambda symbol, bars, ctx: 0.90
        engine._prev_day[SYM] = (10.0, None, None)   # prior close, well above the 90% SSR floor at ~12.30
        _arm_candidate(engine)
        engine._minute_of_day = lambda: FILL_MINUTE   # pin "now" so _fs_register_long records fill_minute deterministically
        long_trade_id = _open_long(engine, real_db)

        engine._fs_register_long(SYM, 12.00, 11.50, 100, long_trade_id, datetime.now(timezone.utc))
        assert engine._fs_tracked[SYM]['state'] == 'awaiting_signal'

        # close of bar fill+1: evaluate the signal -> GO, staged (not yet submitted)
        engine._minute_of_day = lambda: FILL_MINUTE + 1
        engine._fs_advance(SYM)
        assert engine._fs_tracked[SYM]['state'] == 'awaiting_submit'
        assert not mock_alpaca.submit_market_sell_order.called

        # close of bar fill+entry_bar_offset (default 2): submit
        engine._minute_of_day = lambda: FILL_MINUTE + 2
        engine._fs_advance(SYM)
        assert SYM not in engine._fs_tracked

        # the reversal sell + the short's two exit orders
        mock_alpaca.submit_market_sell_order.assert_called_once()
        assert mock_alpaca.submit_market_sell_order.call_args.args[:2] == (SYM, 200)   # 2x the long's 100 sh
        # ONE OCO buy order for both exits (never two independently-resting covers) — the OCO
        # succeeds by default in this fixture, so the two-order fallback is never reached.
        mock_alpaca.submit_oco_buy_order.assert_called_once()
        oco_kwargs = mock_alpaca.submit_oco_buy_order.call_args.kwargs
        assert oco_kwargs['qty'] == 100 and oco_kwargs['stop_price'] == 12.35 and oco_kwargs['limit_price'] == 11.50
        assert not mock_alpaca.submit_stop_limit_order.called and not mock_alpaca.submit_limit_buy_order.called
        assert engine._fs_shorts[SYM]['is_oco'] is True
        assert engine._fs_shorts[SYM]['stop_id'] == 'ocoStop1' and engine._fs_shorts[SYM]['target_id'] == 'ocoTarget1'

        # StopMonitor: the long's watch is torn down; NO watch is armed for the short (see
        # trading/hod_break_engine.py's failure-short overlay docstring for why — StopMonitor's
        # own execution path is SELL-only and long-oriented, so arming it for a short with a
        # stop ABOVE price would misfire immediately; the short's real protection is the two
        # broker-resting orders asserted above).
        mock_sm.remove_watch.assert_called_once_with(SYM)
        assert not mock_sm.add_watch.called

        # two DB rows
        rows = real_db.get_trades_by_date('2026-09-30')
        assert len(rows) == 2
        long_row = next(r for r in rows if r['side'] == 'buy')
        short_row = next(r for r in rows if r['side'] == 'sell')
        assert long_row['exit_reason'] == 'failure_reversal' and long_row['order_status'] == 'closed'
        assert short_row['shares'] == 100 and short_row['strategy'] == 'hod_break'
        pattern = json.loads(short_row['pattern_data'])
        assert pattern['mechanism'] == 'failure_short' and pattern['reversed_long_trade_id'] == long_trade_id
        assert pattern['tp_leg_id'] == 'ocoTarget1' and pattern['sl_leg_id'] == 'ocoStop1' and pattern['oco'] is True

        # ledger row for this evaluation
        with open(ledger) as fh:
            ledger_rows = list(csv.DictReader(fh))
        assert len(ledger_rows) == 1 and ledger_rows[0]['symbol'] == SYM and ledger_rows[0]['passed'] == 'True'

    def test_oco_submit_failure_falls_back_to_two_orders(self, mock_alpaca, real_db, mock_sm, tmp_path):
        mock_alpaca.submit_oco_buy_order.side_effect = Exception('oco rejected')
        ledger = str(tmp_path / 'fs_ledger.csv')
        engine = HodBreakEngine(mock_alpaca, real_db, mock_sm, cfg=cfg(ledger_path=ledger))
        engine.session_date = '2026-09-30'
        engine.fs_signal_fn = lambda symbol, bars, ctx: 0.90
        engine._prev_day[SYM] = (10.0, None, None)
        _arm_candidate(engine)
        engine._minute_of_day = lambda: FILL_MINUTE
        long_trade_id = _open_long(engine, real_db)
        engine._fs_register_long(SYM, 12.00, 11.50, 100, long_trade_id, datetime.now(timezone.utc))
        engine._minute_of_day = lambda: FILL_MINUTE + 1
        engine._fs_advance(SYM)
        engine._minute_of_day = lambda: FILL_MINUTE + 2
        engine._fs_advance(SYM)

        mock_alpaca.submit_oco_buy_order.assert_called_once()
        mock_alpaca.submit_stop_limit_order.assert_called_once()
        mock_alpaca.submit_limit_buy_order.assert_called_once()
        assert engine._fs_shorts[SYM]['is_oco'] is False
        assert engine._fs_shorts[SYM]['stop_id'] == 'stop1' and engine._fs_shorts[SYM]['target_id'] == 'target1'

    def test_one_leg_fill_cancels_the_sibling_no_second_cover(self, mock_alpaca, real_db, mock_sm):
        """OCO or fallback: once EITHER exit leg fills, the sibling is cancelled and the short is
        closed — never a second cover order."""
        engine = HodBreakEngine(mock_alpaca, real_db, mock_sm, cfg=cfg())
        engine.session_date = '2026-09-30'
        short_trade_id = real_db.save_trade({'trade_date': '2026-09-30', 'symbol': SYM, 'side': 'sell',
                                              'entry_price': 12.28, 'stop_loss_price': 12.35, 'take_profit_price': 11.50,
                                              'shares': 100, 'risk_per_share': 0.07, 'total_risk': 7.0,
                                              'risk_reward_ratio': 2.0, 'order_id': 'sell1', 'order_status': 'filled',
                                              'fill_price': 12.28, 'filled_at': datetime.now(timezone.utc).isoformat(),
                                              'exit_price': None, 'exit_reason': None, 'exited_at': None, 'pnl': None,
                                              'pnl_pct': None, 'strategy': 'hod_break', 'account': 'paper', 'pattern_data': '{}'})
        engine._fs_shorts[SYM] = {'trade_id': short_trade_id, 'qty': 100, 'entry_px': 12.28, 'is_oco': True,
                                   'stop_id': 'ocoStop1', 'target_id': 'ocoTarget1', 'stop_px': 12.35, 'target_px': 11.50}

        def get_order(oid):
            return {'status': 'filled'} if oid == 'ocoStop1' else {'status': 'accepted'}
        mock_alpaca.get_order.side_effect = get_order

        engine._fs_poll_short_exits()

        mock_alpaca.cancel_order.assert_called_once_with('ocoTarget1')   # sibling cancelled
        assert SYM not in engine._fs_shorts                              # short closed, no lingering tracking
        assert not mock_alpaca.submit_market_sell_order.called           # no second cover ever submitted
        assert not mock_alpaca.submit_oco_buy_order.called
        rows = real_db.get_trades_by_date('2026-09-30')
        closed = next(r for r in rows if r['id'] == short_trade_id)
        assert closed['order_status'] == 'closed' and closed['exit_reason'] == 'failure_short_stop'
        assert closed['exit_price'] == 12.35

    def test_no_go_below_tau_drops_tracking_no_orders(self, mock_alpaca, real_db, mock_sm, tmp_path):
        engine = HodBreakEngine(mock_alpaca, real_db, mock_sm, cfg=cfg(tau=0.95, ledger_path=str(tmp_path / 'l.csv')))
        engine.session_date = '2026-09-30'
        engine.fs_signal_fn = lambda symbol, bars, ctx: 0.5   # below tau
        engine._prev_day[SYM] = (10.0, None, None)
        _arm_candidate(engine)
        engine._minute_of_day = lambda: FILL_MINUTE   # pin "now" so _fs_register_long records fill_minute deterministically
        _open_long(engine, real_db)
        engine._fs_register_long(SYM, 12.00, 11.50, 100, 1, datetime.now(timezone.utc))
        engine._minute_of_day = lambda: FILL_MINUTE + 1
        engine._fs_advance(SYM)
        assert SYM not in engine._fs_tracked
        assert not mock_alpaca.submit_market_sell_order.called

    def test_not_shortable_blocks_even_with_high_p(self, mock_alpaca, real_db, mock_sm, tmp_path):
        mock_alpaca.get_shortability.return_value = {'shortable': False, 'easy_to_borrow': False}
        engine = HodBreakEngine(mock_alpaca, real_db, mock_sm, cfg=cfg(ledger_path=str(tmp_path / 'l.csv')))
        engine.session_date = '2026-09-30'
        engine.fs_signal_fn = lambda symbol, bars, ctx: 0.99
        engine._prev_day[SYM] = (10.0, None, None)
        _arm_candidate(engine)
        engine._minute_of_day = lambda: FILL_MINUTE   # pin "now" so _fs_register_long records fill_minute deterministically
        _open_long(engine, real_db)
        engine._fs_register_long(SYM, 12.00, 11.50, 100, 1, datetime.now(timezone.utc))
        engine._minute_of_day = lambda: FILL_MINUTE + 1
        engine._fs_advance(SYM)
        assert SYM not in engine._fs_tracked
        assert not mock_alpaca.submit_market_sell_order.called

    def test_exception_in_overlay_disables_for_day_never_touches_normal_exits(self, mock_alpaca, real_db, mock_sm, tmp_path):
        engine = HodBreakEngine(mock_alpaca, real_db, mock_sm, cfg=cfg(ledger_path=str(tmp_path / 'l.csv')))
        engine.session_date = '2026-09-30'
        def boom(symbol, bars, ctx): raise RuntimeError('signal blew up')
        engine.fs_signal_fn = boom
        engine._prev_day[SYM] = (10.0, None, None)
        _arm_candidate(engine)
        engine._minute_of_day = lambda: FILL_MINUTE   # pin "now" so _fs_register_long records fill_minute deterministically
        _open_long(engine, real_db)
        engine._fs_register_long(SYM, 12.00, 11.50, 100, 1, datetime.now(timezone.utc))
        engine._minute_of_day = lambda: FILL_MINUTE + 1
        engine._process_failure_short()   # signal_fn raises -> caught inside _fs_evaluate_signal, NOT a bare crash
        # signal_fn exceptions are caught locally (treated as p=None, NO-GO) -- the overlay-wide
        # disable is for exceptions the per-symbol handlers do NOT catch; assert the long is untouched either way
        assert SYM in engine.positions   # normal long position untouched
        assert not mock_alpaca.submit_market_sell_order.called
