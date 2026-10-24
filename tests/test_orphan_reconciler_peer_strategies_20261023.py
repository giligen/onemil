"""2026-10-23: the live-account orphan scan must only count trade rows of strategies on the SAME broker account
(TXG: a hod_break paper-account row named the owner's live position 'suspect over-exit' at every boot)."""
from datetime import date
from unittest.mock import MagicMock

from data_sources.alpaca_client import AlpacaClient
from trading.orphan_reconciler import (
    peer_strategies_for, reconcile_strategy_orphans, reset_state_for_tests, _recent_any_strategy_trade,
)

TODAY = date(2026, 10, 23)


def _scan(peer_strategies):
    a = MagicMock(spec=AlpacaClient)
    a.get_open_positions.return_value = [{'symbol': 'TXG', 'qty': 7, 'avg_entry_price': 69.675}]
    db = MagicMock()   # Database lacks get_trades_in_window (prod uses the SQL path, covered below)
    db.get_strategy_trades_in_window.return_value = []
    db.get_trades_in_window.return_value = [{'strategy': 'hod_break', 'symbol': 'TXG', 'trade_date': str(TODAY)}]
    reset_state_for_tests()
    return reconcile_strategy_orphans(
        strategy='bull_flag', alpaca=a, db=db, notifier=MagicMock(), tracked_symbols=set(), today_et=TODAY,
        broker_positions=a.get_open_positions.return_value, peer_strategies=peer_strategies)


def test_foreign_when_only_bull_flag_shares_the_account(caplog):
    actions = _scan({'bull_flag'})
    assert actions[0].classification == 'foreign'
    assert not [r for r in caplog.records if r.levelname in ('ERROR', 'CRITICAL')]


def test_legacy_none_keeps_any_strategy_lookup():
    assert _scan(None)[0].classification == 'suspect_over_exit'


def test_suspect_when_hod_break_shares_the_account():
    assert _scan({'bull_flag', 'hod_break'})[0].classification == 'suspect_over_exit'


def test_sql_path_filters_strategies_in_clause():
    db = MagicMock()
    del db.get_trades_in_window                       # legacy path
    cur = MagicMock(); cur.fetchall.return_value = []
    db._trades_conn.execute.return_value = cur
    assert _recent_any_strategy_trade(db, 'TXG', '2026-10-16', {'bull_flag', 'orb'}) is None
    sql, params = db._trades_conn.execute.call_args.args
    assert 'strategy IN (?,?)' in sql and params == ('TXG', '2026-10-16', 'bull_flag', 'orb')


def _client(acct):
    c = MagicMock(spec=AlpacaClient); c.account_number = acct; return c


def test_peer_set_from_account_numbers():
    m = {'bull_flag': _client('LIVE1'), 'ignition': _client('LIVE1'), 'hod_break': _client('PA39'),
         'orb': _client('PAORB'), 'macd_wave': _client(None)}
    assert peer_strategies_for('bull_flag', m) == {'bull_flag', 'ignition'}
    assert peer_strategies_for('hod_break', m) == {'hod_break'}


def test_unknown_own_account_and_unknown_peer_fail_toward_foreign():
    m = {'bull_flag': _client(None), 'hod_break': _client(None)}
    assert peer_strategies_for('bull_flag', m) == {'bull_flag'}
    m = {'bull_flag': _client('L'), 'hod_break': _client(None)}
    assert peer_strategies_for('bull_flag', m) == {'bull_flag'}
    assert peer_strategies_for('bull_flag', None) is None
