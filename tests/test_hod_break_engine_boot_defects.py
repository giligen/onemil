"""Regression tests for three 2026-09-29 boot-time defects in trading/hod_break_engine.py, found on
the CONI/TTAN paper-session incident (docs/hod_live_incident_20260929.md, Part 2):

1. Instance methods used the MODULE constant STRATEGY_NAME instead of self.STRATEGY_NAME — the
   red_to_green instance could read and mutate hod_break's own DB rows and drain hod_break's own
   StopMonitor exit events.
2. `_adopt_unregistered_positions_on_boot` saved the adopted position as order_status 'pending_new',
   which `sync_positions` never restores (only filled/partially_filled/exit_pending_verification are
   "open") — so the same broker position got adopted a second time on the next boot, and a later fill
   of the resting order inserted a THIRD row instead of updating the adopted one.
3. The exit_pending_verification reconciler declared a row UNRECONCILED (ERROR + Telegram) without
   ever checking whether the broker still held the shares.

Real `persistence.database.Database` on a tmp path (unmocked save_trade/get_open_trades/update_trade)
+ real `HodBreakEngine` instances + a `MagicMock(spec=AlpacaClient)` broker — no service restarted, no
order submitted/cancelled, no production DB/config touched.
"""
from datetime import datetime
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.stop_monitor import StopMonitor
from trading.hod_break_engine import HodBreakEngine, Candidate
from tests.test_hod_break_engine import cfg as hod_cfg

ET = ZoneInfo('America/New_York')
TODAY = datetime.now(ET).strftime('%Y-%m-%d')   # matches HodBreakEngine._et_now()-based session_date


def r2g_cfg(**over):
    base = {'enabled': True, 'dry_run': False, 'risk_usd': 100.0, 'daily_kill_usd': -600.0, 'weekly_kill_usd': -1500.0,
            'max_notional_usd': 5000.0, 'min_price': 1.0, 'min_adv20': 100_000.0, 'max_spread_bps': 100.0,
            'order_timeout_s': 75.0, 'book': 'red_to_green', 'params': {}}
    base.update(over); return base


@pytest.fixture
def real_db(tmp_path):
    """Real Database (full schema) on a temp file — save_trade/get_open_trades/update_trade run unmocked,
    never touching data/*.db."""
    db = Database(db_path=str(tmp_path / "hod_defects.db"))
    yield db
    db.close()


@pytest.fixture
def mock_alpaca():
    a = MagicMock(spec=AlpacaClient)
    a.get_open_positions.return_value = []
    a.get_open_orders.return_value = []
    a.get_account_info.return_value = {'account_number': 'PA_TEST_1'}
    a.is_paper = True
    return a


@pytest.fixture
def mock_sm():
    s = MagicMock(spec=StopMonitor)
    s.polling_mode = False
    return s


def _seed_row(db, symbol, strategy, shares, fill_price, order_status, stop=1.0, pattern_data='{}'):
    """Insert one trade row directly via the real Database, bypassing the engine (mirrors a row a
    prior process already wrote)."""
    rec = {
        'trade_date': TODAY, 'symbol': symbol, 'side': 'buy', 'entry_price': fill_price,
        'stop_loss_price': stop, 'take_profit_price': fill_price + 1.0, 'shares': shares,
        'risk_per_share': fill_price - stop, 'total_risk': (fill_price - stop) * shares,
        'risk_reward_ratio': 2.0, 'order_id': f'seed-{symbol}', 'order_status': order_status,
        'fill_price': fill_price, 'filled_at': datetime.now(ET).isoformat(),
        'exit_price': None, 'exit_reason': None, 'exited_at': None, 'pnl': None, 'pnl_pct': None,
        'strategy': strategy, 'account': 'paper', 'pattern_data': pattern_data,
    }
    return db.save_trade(rec)


class TestDefect1InstanceStrategyName:
    """Instance code must key off self.STRATEGY_NAME, never the module constant."""

    def test_r2g_sync_positions_ignores_and_does_not_mutate_hod_break_rows(self, real_db, mock_alpaca, mock_sm):
        _seed_row(real_db, 'CONI', 'hod_break', shares=25, fill_price=10.0, order_status='filled', stop=9.5)
        mock_alpaca.get_open_positions.return_value = []   # broker holds none of it under R2G's own (empty) book
        r2g = HodBreakEngine(mock_alpaca, real_db, mock_sm, cfg=r2g_cfg())
        r2g._roll_session()
        r2g.sync_positions()
        assert 'CONI' not in r2g.positions, "R2G must not adopt hod_break's own open row"
        row = real_db.get_open_trades(TODAY, strategy='hod_break')[0]
        assert row['order_status'] == 'filled', \
            "R2G's sync_positions marked HOD's own row exit_pending_verification — the 2026-09-29 CONI incident"

    def test_drain_exit_events_uses_the_instance_strategy(self, real_db, mock_alpaca, mock_sm):
        r2g = HodBreakEngine(mock_alpaca, real_db, mock_sm, cfg=r2g_cfg())
        r2g._roll_session()
        mock_sm.drain_exit_events.return_value = []
        r2g._drain_stop_monitor_exits()
        mock_sm.drain_exit_events.assert_called_once_with(strategy='red_to_green')

    def test_hod_instance_still_drains_its_own_strategy(self, real_db, mock_alpaca, mock_sm):
        hod = HodBreakEngine(mock_alpaca, real_db, mock_sm, cfg=hod_cfg())
        hod._roll_session()
        mock_sm.drain_exit_events.return_value = []
        hod._drain_stop_monitor_exits()
        mock_sm.drain_exit_events.assert_called_once_with(strategy='hod_break')


class TestDefect2AdoptAsFilledAndDBDedup:
    """Boot adoption must insert a FILLED row (restorable by sync_positions) and merge against an
    existing open DB row instead of ever inserting a second one for the same symbol."""

    def test_adopts_as_filled_and_links_trade_id_into_live_order(self, real_db, mock_alpaca, mock_sm):
        mock_alpaca.get_open_positions.return_value = [{'symbol': 'PRIM', 'qty': 87, 'avg_entry_price': 5.00}]
        eng = HodBreakEngine(mock_alpaca, real_db, mock_sm, cfg=hod_cfg())
        eng._roll_session()
        cand = Candidate(symbol='PRIM', day_open=0.0, adv20=0.0)
        cand.live_order = {'stop': 4.50}
        eng.candidates['PRIM'] = cand
        eng._adopt_unregistered_positions_on_boot()
        rows = real_db.get_open_trades(TODAY, strategy='hod_break')
        assert len(rows) == 1
        row = rows[0]
        assert row['order_status'] == 'filled' and row['shares'] == 87 and row['fill_price'] == 5.00
        assert cand.live_order['trade_id'] == row['id'], \
            "a later fill of the resting order must dedup onto this trade_id, not insert a third row"

    def test_second_boot_restores_via_sync_positions_and_does_not_readopt(self, real_db, mock_alpaca, mock_sm):
        mock_alpaca.get_open_positions.return_value = [{'symbol': 'WRBY', 'qty': 40, 'avg_entry_price': 8.00}]
        boot1 = HodBreakEngine(mock_alpaca, real_db, mock_sm, cfg=hod_cfg())
        boot1._roll_session()
        boot1.entered_today.add('WRBY')   # this book's own resting order fired earlier this session (record of "ours")
        boot1._adopt_unregistered_positions_on_boot()
        assert len(real_db.get_open_trades(TODAY, strategy='hod_break')) == 1

        boot2 = HodBreakEngine(mock_alpaca, real_db, mock_sm, cfg=hod_cfg())
        boot2._roll_session()
        boot2.sync_positions()
        assert 'WRBY' in boot2.positions and boot2.positions['WRBY'].shares == 40
        boot2._adopt_unregistered_positions_on_boot()
        rows = real_db.get_open_trades(TODAY, strategy='hod_break')
        assert len(rows) == 1, f"WRBY was re-adopted as a second row: {rows}"

    def test_broker_ahead_of_open_db_row_merges_with_weighted_average_price(self, real_db, mock_alpaca, mock_sm):
        trade_id = _seed_row(real_db, 'CONI', 'hod_break', shares=25, fill_price=22.50, order_status='filled', stop=22.00)
        mock_alpaca.get_open_positions.return_value = [{'symbol': 'CONI', 'qty': 112, 'avg_entry_price': 23.00}]
        eng = HodBreakEngine(mock_alpaca, real_db, mock_sm, cfg=hod_cfg())
        eng._roll_session()
        eng._adopt_unregistered_positions_on_boot()
        rows = real_db.get_open_trades(TODAY, strategy='hod_break')
        assert len(rows) == 1, f"a second row was inserted instead of merging into the open one: {rows}"
        row = rows[0]
        assert row['id'] == trade_id and row['shares'] == 112
        expected_price = round((22.50 * 25 + 23.00 * 87) / 112, 4)
        assert row['fill_price'] == pytest.approx(expected_price)

    def test_non_positive_difference_adopts_nothing(self, real_db, mock_alpaca, mock_sm):
        _seed_row(real_db, 'ASTN', 'hod_break', shares=100, fill_price=6.00, order_status='filled', stop=5.80)
        mock_alpaca.get_open_positions.return_value = [{'symbol': 'ASTN', 'qty': 100, 'avg_entry_price': 6.00}]
        eng = HodBreakEngine(mock_alpaca, real_db, mock_sm, cfg=hod_cfg())
        eng._roll_session()
        eng._adopt_unregistered_positions_on_boot()
        rows = real_db.get_open_trades(TODAY, strategy='hod_break')
        assert len(rows) == 1 and rows[0]['shares'] == 100


class TestDefect3ExitPendingVerificationBrokerCheck:
    """A row must not be declared UNRECONCILED when the broker still holds the shares."""

    def test_broker_still_holds_shares_restores_to_open_no_error_no_telegram(self, real_db, mock_alpaca, mock_sm):
        _seed_row(real_db, 'CONI', 'hod_break', shares=112, fill_price=22.882231,
                  order_status='exit_pending_verification', stop=22.63)
        mock_alpaca.get_open_positions.return_value = [{'symbol': 'CONI', 'qty': 112, 'avg_entry_price': 22.882231}]
        notifier = MagicMock()
        eng = HodBreakEngine(mock_alpaca, real_db, mock_sm, notifier=notifier, cfg=hod_cfg())
        eng._roll_session()
        n = eng.reconcile_pending_exits()
        assert n == 1
        row = real_db.get_open_trades(TODAY, strategy='hod_break')[0]
        assert row['order_status'] == 'filled'
        assert not any('UNRECONCILED' in str(c) for c in notifier.send_message.call_args_list)

    def test_broker_holds_fewer_shares_and_legs_dont_cover_rest_is_still_an_error(self, real_db, mock_alpaca, mock_sm):
        _seed_row(real_db, 'TTAN', 'hod_break', shares=50, fill_price=12.00,
                  order_status='exit_pending_verification', stop=11.50)
        mock_alpaca.get_open_positions.return_value = []   # broker holds none
        notifier = MagicMock()
        eng = HodBreakEngine(mock_alpaca, real_db, mock_sm, notifier=notifier, cfg=hod_cfg())
        eng._roll_session()
        n = eng.reconcile_pending_exits()
        assert n == 0
        row = real_db.get_open_trades(TODAY, strategy='hod_break')[0]
        assert row['order_status'] == 'exit_pending_verification'
        assert any('UNRECONCILED' in str(c) for c in notifier.send_message.call_args_list)


class TestSyncPositionsWarningNamesAccount:
    def test_broker_holds_fewer_warning_looks_up_the_account(self, real_db, mock_alpaca, mock_sm):
        _seed_row(real_db, 'VECO', 'hod_break', shares=60, fill_price=9.00, order_status='filled', stop=8.50)
        mock_alpaca.get_open_positions.return_value = [{'symbol': 'VECO', 'qty': 0}]
        eng = HodBreakEngine(mock_alpaca, real_db, mock_sm, cfg=hod_cfg())
        eng._roll_session()
        eng.sync_positions()
        assert mock_alpaca.get_account_info.called, \
            "sync_positions' 'broker holds N' WARNING must look up the account it compared against"


class TestDefect4SyncRestoresWatchWithoutLegs:
    """A row restored to 'open' by sync_positions with no broker-side exit legs (an adopted position,
    or a fill whose OCO placement failed) must get a StopMonitor watch registered right there — not
    only at boot-adoption time — else a later restart silently drops the stop. Concrete case: CONI
    112 sh filled, stop 22.63, pattern_data.closed_qty 25 (no tp_leg_id/sl_leg_id), broker holds 87."""

    def test_coni_restored_without_legs_gets_a_watch_for_open_qty(self, real_db, mock_alpaca, mock_sm):
        entry, stop = 22.882231, 22.63
        trade_id = _seed_row(real_db, 'CONI', 'hod_break', shares=112, fill_price=entry, order_status='filled',
                              stop=stop, pattern_data='{"closed_qty": 25}')
        mock_alpaca.get_open_positions.return_value = [{'symbol': 'CONI', 'qty': 87}]
        eng = HodBreakEngine(mock_alpaca, real_db, mock_sm, cfg=hod_cfg())
        eng._roll_session()
        n = eng.sync_positions()
        assert n == 1
        assert 'CONI' in eng.positions and eng.positions['CONI'].status == 'open'
        mock_sm.add_watch.assert_called_once_with(symbol='CONI', stop_price=stop, shares=87, tp_leg_id='', sl_leg_id='',
                                                    trade_db_id=trade_id, entry_price=entry, risk_per_share=entry - stop,
                                                    strategy='hod_break')

    def test_row_with_legs_gets_no_watch(self, real_db, mock_alpaca, mock_sm):
        """Existing behaviour for rows that DO have broker exit legs must be unchanged: no watch call."""
        _seed_row(real_db, 'ABC', 'hod_break', shares=50, fill_price=10.0, order_status='filled', stop=9.5,
                  pattern_data='{"tp_leg_id": "tp-1", "sl_leg_id": "sl-1"}')
        mock_alpaca.get_open_positions.return_value = [{'symbol': 'ABC', 'qty': 50}]
        eng = HodBreakEngine(mock_alpaca, real_db, mock_sm, cfg=hod_cfg())
        eng._roll_session()
        eng.sync_positions()
        assert 'ABC' in eng.positions
        mock_sm.add_watch.assert_not_called()

    def test_no_stop_monitor_and_no_stop_price_logs_unmanaged_and_notifies(self, real_db, mock_alpaca):
        """No StopMonitor instance at all: the row is restored but flagged UNMANAGED via ERROR + Telegram,
        never a silent gap."""
        _seed_row(real_db, 'XYZ', 'hod_break', shares=30, fill_price=5.0, order_status='filled', stop=0.0,
                  pattern_data='{"closed_qty": 0}')
        mock_alpaca.get_open_positions.return_value = [{'symbol': 'XYZ', 'qty': 30}]
        notifier = MagicMock()
        eng = HodBreakEngine(mock_alpaca, real_db, None, notifier=notifier, cfg=hod_cfg())
        eng._roll_session()
        eng.sync_positions()
        assert 'XYZ' in eng.positions
        assert any('UNMANAGED' in str(c) for c in notifier.send_message.call_args_list)
