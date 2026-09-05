"""trading/bf_universe_filter — the BF backtest applies live's name-based
asset rule (2026-09-05, after wrappers entered daily_bars for ORB)."""
import logging

from trading.bf_universe_filter import filter_trades, is_bf_eligible, load_names

NAMES = {
    'AAPL': 'Apple Inc. Common Stock',
    'AXTX': 'Tradr 2X Long AXTI Daily ETF',
    'SNDG': 'Leverage Shares 2X Long SNDK Daily ETF',
    'USARW': 'USA Rare Earth Warrant',
    'TQQQ': 'ProShares UltraPro QQQ',
}


def test_common_stock_eligible():
    assert is_bf_eligible('AAPL', NAMES)


def test_wrappers_and_warrants_excluded_by_name():
    for s in ('AXTX', 'SNDG', 'USARW', 'TQQQ'):
        assert not is_bf_eligible(s, NAMES), s


def test_unknown_name_falls_back_to_symbol_list():
    assert is_bf_eligible('ZZZZ', NAMES)          # unknown, not on the list → kept
    assert not is_bf_eligible('MSTU', NAMES)      # unknown name but on the legacy list


def test_filter_trades_logs_and_keeps_order(caplog):
    trades = [{'symbol': s} for s in ('AAPL', 'AXTX', 'ZZZZ', 'SNDG', 'MSTU')]
    with caplog.at_level(logging.INFO):
        kept = filter_trades(trades, NAMES)
    assert [t['symbol'] for t in kept] == ['AAPL', 'ZZZZ']
    assert any('3 removed' in r.message for r in caplog.records)
    assert any('no known name' in r.message and 'ZZZZ' in r.message for r in caplog.records)


def test_offline_name_sources_cover_the_new_wrappers():
    """The real dumps must know the post-April wrappers (the whole point)."""
    names = load_names()
    assert len(names) > 20_000
    for s in ('AXTX', 'LITZ', 'SNDQ', 'CBRG'):
        assert s in names and not is_bf_eligible(s, names), s
    assert is_bf_eligible('AAPL', names)
