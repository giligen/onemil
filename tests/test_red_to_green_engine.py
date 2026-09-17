"""The red-to-green book through the SAME engine (HodBreakEngine, book='red_to_green'): gates, universe screen, the
signal path to a dry-run WOULD BUY with the R2G tag, its own strategy tag / handler id / client-order prefix, and the
HOD book untouched by the switch."""
import numpy as np
import pytest
from unittest.mock import MagicMock

from data_sources.alpaca_client import AlpacaClient
from trading.hod_break_engine import HodBreakEngine, Candidate
from trading.stop_monitor import StopMonitor
from trading.red_to_green import RedToGreenParams
from persistence.database import Database


def r2g_cfg(**kw):
    base = {'book': 'red_to_green', 'enabled': True, 'dry_run': True, 'risk_usd': 100.0, 'min_price': 5.0, 'min_adv20': 100_000.0,
            'max_spread_bps': 300.0, 'stream_universe': False, 'universe_min_prev_close': 5.0, 'stream_list_dir': '/tmp',
            'params': {'pdr_min_pct': 8.0, 'range_floor_pct': 5.0, 'level_buffer': 0.003, 'cap': 0.006, 'min_r_pct': 1.0,
                       'target_r': 2.0, 'max_per_day': 12, 'max_concurrent': 4, 'last_entry_minute': 840, 'flat_minute': 955}}
    base.update(kw); return base


@pytest.fixture
def alpaca():
    a = MagicMock(spec=AlpacaClient)
    a.get_latest_quote.return_value = {'bid_price': 10.04, 'ask_price': 10.05, 'bid_size': 100, 'ask_size': 100}
    a.get_open_positions.return_value = []; a.get_1min_bars_multi.return_value = {}
    return a


@pytest.fixture
def db(tmp_path):
    import sqlite3
    p = tmp_path / 'trades.db'
    con = sqlite3.connect(p); con.execute("create table trades (id integer primary key, strategy text, trade_date text, pnl real, symbol text, order_status text)"); con.commit(); con.close()
    d = MagicMock(spec=Database); d._trades_path = str(p)
    d.get_active_universe.return_value = [{'symbol': 'ABC', 'avg_volume_daily': 1_000_000}]
    d.get_open_trades.return_value = []; d.save_trade.return_value = 1
    return d


@pytest.fixture
def sm():
    s = MagicMock(spec=StopMonitor); s.polling_mode = False; return s


def test_book_switch_sets_tags_params_and_strategy(alpaca, db, sm):
    e = HodBreakEngine(alpaca, db, sm, cfg=r2g_cfg())
    assert e.book == 'red_to_green' and isinstance(e.params, RedToGreenParams)
    assert e.STRATEGY_NAME == 'red_to_green' and e.tag == '[R2G]' and e.dry_tag == '[R2G DRY]' and e.coid_prefix == 'r2g'
    e.register_on_stop_monitor()
    sm.register_bar_handler.assert_called_once()
    assert sm.register_bar_handler.call_args[0][0] == 'red_to_green'


def test_hod_book_is_unchanged_by_the_switch(alpaca, db, sm):
    from trading.hod_break import HodBreakParams
    e = HodBreakEngine(alpaca, db, sm, cfg={'enabled': True, 'dry_run': True, 'stream_universe': False, 'stream_list_dir': '/tmp'})
    assert e.book == 'hod_break' and isinstance(e.params, HodBreakParams) and e.STRATEGY_NAME == 'hod_break' and e.tag == '[HOD]'


def test_unknown_book_is_refused(alpaca, db, sm):
    with pytest.raises(ValueError):
        HodBreakEngine(alpaca, db, sm, cfg=r2g_cfg(book='nope'))


def test_universe_screen_keeps_only_prior_day_range_names(alpaca, db, sm):
    e = HodBreakEngine(alpaca, db, sm, cfg=r2g_cfg(stream_universe=True))
    e._adv_map = {'BIG': 1e6, 'QUIET': 1e6, 'NOPD': 1e6}
    e._last_close = {'BIG': 10.0, 'QUIET': 10.0, 'NOPD': 10.0}
    e._prev_day = {'BIG': (10.0, 11.0, 9.5), 'QUIET': (10.0, 10.3, 10.0)}      # BIG ranged 15.8%, QUIET 3%, NOPD unknown
    e.session_date = '2026-09-17'
    e._stream_the_universe()
    assert set(e.candidates) == {'BIG'}
    assert e.candidates['BIG'].prior_close == 10.0 and e.candidates['BIG'].pdr_pct == pytest.approx((11.0 - 9.5) / 9.5 * 100)


def _feed(e, cand, bars):
    """bars: list of (minute, o, h, l, c, v) — closed bars, in order."""
    for m, o, h, l, c, v in bars:
        cand.set_bar(m, o, h, l, c, v)


def test_signal_path_reaches_a_dry_would_buy_with_the_r2g_tag(alpaca, db, sm, caplog):
    e = HodBreakEngine(alpaca, db, sm, cfg=r2g_cfg())
    e._roll_session(); e.calendar_ok = True
    cand = Candidate(symbol='ABC', day_open=0.0, adv20=1e6, subscribed=True, backfill_ok=True, prior_close=10.0, pdr_pct=15.0)
    e.candidates['ABC'] = cand
    # 09:30 opens below the prior close (9.50); bar 1 ranges 9.40-10.00 so bars before bar 2 span 6.4% (floor met);
    # bar 2's high reaches the level 10.03 -> signal; the ask (10.05) is under the cap 10.09 -> WOULD BUY
    _feed(e, cand, [(570, 9.50, 9.60, 9.50, 9.55, 1000), (571, 9.55, 10.00, 9.40, 9.95, 1000), (572, 9.95, 10.06, 9.90, 10.02, 1000)])
    import logging
    with caplog.at_level(logging.INFO):
        e._evaluate(cand)
    assert any('[R2G DRY] WOULD BUY ABC' in r.message and 'prior close 10.00' in r.message for r in caplog.records), [r.message for r in caplog.records]
    assert cand.rejected_reason == 'dry_run'
    alpaca.submit_bracket_order.assert_not_called()


def test_no_signal_when_the_open_is_above_the_prior_close(alpaca, db, sm):
    e = HodBreakEngine(alpaca, db, sm, cfg=r2g_cfg())
    e._roll_session(); e.calendar_ok = True
    cand = Candidate(symbol='ABC', day_open=0.0, adv20=1e6, subscribed=True, backfill_ok=True, prior_close=9.0, pdr_pct=15.0)
    e.candidates['ABC'] = cand
    _feed(e, cand, [(570, 9.50, 9.60, 9.50, 9.55, 1000), (571, 9.55, 10.00, 9.40, 9.95, 1000), (572, 9.95, 10.06, 9.90, 10.02, 1000)])
    e._evaluate(cand)
    assert cand.rejected_reason is None and not cand.dry_logged
