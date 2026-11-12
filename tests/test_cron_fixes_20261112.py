"""Cron code fixes 11/12: universe build survives an invalid symbol, cron guard on every entry point,
BF parity on a paused book is not a breach."""
from __future__ import annotations

import argparse
import logging
import runpy
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))

from data_sources.alpaca_client import AlpacaClient, invalid_symbols_in_error  # noqa: E402
from monitoring import cron_guard  # noqa: E402
from trading.ramp_freeze import FreezeState  # noqa: E402
import bf_decision_parity as bp  # noqa: E402
import main as main_mod  # noqa: E402


# ---------------------------------------------------------------- §1 universe build
@pytest.fixture
def client():
    """AlpacaClient with every SDK constructor mocked (external SDK objects, no spec)."""
    with patch("data_sources.alpaca_client.StockHistoricalDataClient") as d, \
            patch("data_sources.alpaca_client.TradingClient"), patch("data_sources.alpaca_client.NewsClient"):
        d.return_value = MagicMock()
        c = AlpacaClient(api_key="k", api_secret="s")
        c._api_timeout = 5
        yield c


def _bar(close=5.0, volume=1000):
    b = MagicMock()
    b.close, b.volume, b.timestamp = close, volume, "2026-11-11"
    return b


def test_invalid_symbol_parser():
    """The 11/10 message names the symbol; anything else names nothing."""
    assert invalid_symbols_in_error(Exception('{"message":"invalid symbol: 037ESC019"}')) == ['037ESC019']
    assert invalid_symbols_in_error(Exception("invalid symbol: AAA,BBB")) == ['AAA', 'BBB']
    assert invalid_symbols_in_error(Exception("400 bad request: missing start")) == []


def test_invalid_symbol_dropped_chunk_retried_once_and_counted(client, caplog):
    """3-symbol chunk, one 400 'invalid symbol: BAD' -> BAD dropped, retried once, LOST=[BAD], WARNING logged."""
    calls = []

    def get_stock_bars(req):
        calls.append(list(req.symbol_or_symbols))
        if 'BAD' in req.symbol_or_symbols:
            raise Exception('{"message":"invalid symbol: BAD"}')
        return {s: [_bar()] for s in req.symbol_or_symbols}

    client.data_client.get_stock_bars.side_effect = get_stock_bars
    with caplog.at_level(logging.WARNING):
        out = client.get_daily_bars(['AAA', 'BAD', 'CCC'])
    assert set(out) == {'AAA', 'CCC'}
    assert calls == [['AAA', 'BAD', 'CCC'], ['AAA', 'CCC']]
    assert client.lost_daily_bar_symbols == ['BAD'] and client.daily_bars_requested == 3
    assert any('BAD' in r.getMessage() and r.levelname == 'WARNING' for r in caplog.records)


def test_other_400_still_raises(client):
    """A 400 that does not name an invalid symbol is not swallowed."""
    from data_sources.alpaca_client import AlpacaAPIError
    client.data_client.get_stock_bars.side_effect = Exception('400: start must be before end')
    with pytest.raises(AlpacaAPIError):
        client.get_daily_bars(['AAA', 'BBB'])
    assert client.lost_daily_bar_symbols == []


def test_lost_symbols_verdict_gate(caplog):
    """LOST > 2 % of the request = ERROR; below it only the suffix."""
    with caplog.at_level(logging.ERROR):
        assert main_mod.lost_symbols_verdict(['X'], 1000) == " lost=1 [X]"
        assert not caplog.records
        main_mod.lost_symbols_verdict(['A'] * 30, 1000)
    assert len([r for r in caplog.records if r.levelname == 'ERROR']) == 1


def test_run_batch_crash_logs_error_traceback_and_exits_nonzero(caplog, monkeypatch):
    """Integration: a crashing build -> ERROR with traceback, SystemExit(1), no silent pass."""
    monkeypatch.setattr(cron_guard, '_wait_for_telegram_sends', lambda t: None)
    alpaca = MagicMock(spec=AlpacaClient)
    alpaca.test_connection.return_value = True
    builder = MagicMock()
    builder.build.side_effect = RuntimeError("HTTPError 400 invalid symbol")
    cfg = MagicMock()
    with patch.object(main_mod, 'AlpacaClient', return_value=alpaca), patch.object(main_mod, 'FloatProvider'), \
            patch.object(main_mod, 'get_database'), patch.object(main_mod, 'UniverseBuilder', return_value=builder), \
            caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as ei:
            main_mod.run_batch(cfg)
    assert ei.value.code == 1
    errs = [r for r in caplog.records if r.levelname == 'ERROR']
    assert len(errs) == 1 and errs[0].exc_info is not None and 'CRASHED' in errs[0].getMessage()


def test_run_batch_complete_line_carries_lost(caplog):
    """Success path: 'Batch complete: ... lost=N [names]' with the client's LOST list."""
    alpaca = MagicMock(spec=AlpacaClient)
    alpaca.test_connection.return_value = True
    alpaca.lost_daily_bar_symbols = ['037ESC019']
    alpaca.daily_bars_requested = 10000
    builder = MagicMock()
    builder.build.return_value = {'n': 1}
    with patch.object(main_mod, 'AlpacaClient', return_value=alpaca), patch.object(main_mod, 'FloatProvider'), \
            patch.object(main_mod, 'get_database'), patch.object(main_mod, 'UniverseBuilder', return_value=builder), \
            caplog.at_level(logging.INFO):
        main_mod.run_batch(MagicMock())
    assert any('Batch complete' in r.getMessage() and 'lost=1 [037ESC019]' in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------- §2 cron guard on every entry point
def _boom(*a, **k):
    raise RuntimeError('boom 11/12')


def _patch_failure(name, monkeypatch):
    """Make `main()` of script `name` raise at its first step."""
    if name == 'trader_watchdog':
        import subprocess
        monkeypatch.setattr(subprocess, 'run', _boom)
    elif name == 'hod_break_deadman_flat':
        import book_spec
        monkeypatch.setattr(book_spec, 'book_from_argv', _boom)
    else:
        monkeypatch.setattr(argparse.ArgumentParser, 'parse_args', _boom)


@pytest.mark.parametrize('name', ['trader_watchdog', 'bf_decision_parity', 'hourly_summary', 'guardrail',
                                  'hod_break_deadman_flat', 'eod_report', 'orb_pick_parity', 'daily_green_check'])
def test_script_main_crash_is_one_error_naming_script_and_reraises(name, monkeypatch, caplog):
    """`main` raising under `__main__` -> exactly one ERROR '<name> crashed', exception propagates (exit 1)."""
    monkeypatch.setattr(cron_guard, 'attach_telegram_error_handler', lambda: True)
    monkeypatch.setattr(cron_guard, '_wait_for_telegram_sends', lambda t: None)
    monkeypatch.setattr(sys, 'argv', [name + '.py'])
    _patch_failure(name, monkeypatch)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match='boom 11/12'):
            runpy.run_path(str(ROOT / 'scripts' / f'{name}.py'), run_name='__main__')
    errs = [r for r in caplog.records if r.levelname == 'ERROR' and 'crashed' in r.getMessage()]
    assert len(errs) == 1 and f'{name} crashed (RuntimeError)' in errs[0].getMessage()


# ---------------------------------------------------------------- §3 paused BF book
def _patch_build(monkeypatch, enabled):
    monkeypatch.setattr(bp, 'production_cache_path', lambda: Path('/x/cache.csv'))
    monkeypatch.setattr(bp, 'scan_cache', lambda p, d: ('2026-11-11', set()))
    monkeypatch.setattr(bp, 'coverage_covers', lambda d: False)
    monkeypatch.setattr(bp, 'run_stage2_backtest', lambda d, o: [{'symbol': 'ABC', 'pnl': 5.0, 'exit_reason': 'x'}])
    monkeypatch.setattr(bp, 'live_trading_config', lambda: {
        'capital': 50000.0, 'risk': 150.0, 'max_shares': 10000, 'enabled': enabled,
        'min_daily_volume': None, 'conviction_min_threshold': None})
    loaded = []
    monkeypatch.setattr(bp, 'load_live_bf_rows', lambda d, db_path=None: loaded.append(d) or [])
    return loaded


def test_paused_book_report_is_bt_only_no_breach(monkeypatch):
    """enabled False: status PAUSED (BT-only), BT trades kept, live rows never loaded, no divergences."""
    loaded = _patch_build(monkeypatch, enabled=False)
    rep = bp.build_report('2026-11-11')
    assert rep['status'] == bp.STATUS_PAUSED and rep['paused'] is True
    assert rep['n_bt_trades'] == 1 and rep['n_live_rows'] == 0 and rep['divergences'] == []
    assert loaded == [] and bp.hard_parity_breaches(rep) == []


def test_enabled_book_unchanged(monkeypatch):
    """enabled True: live rows are loaded and a BT-only trade is a divergence as before."""
    loaded = _patch_build(monkeypatch, enabled=True)
    rep = bp.build_report('2026-11-11')
    assert loaded == ['2026-11-11'] and rep['paused'] is False
    assert rep['status'].startswith('DIVERGE') and bp.hard_parity_breaches(rep)


def _run_main(monkeypatch, tmp_path, enabled, caplog, state):
    _patch_build(monkeypatch, enabled)
    monkeypatch.setattr(bp, 'PARITY_DIR', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['bf_decision_parity.py', '--date', '2026-11-11', '--no-telegram'])
    rf = MagicMock()
    rf.get.return_value = state
    monkeypatch.setattr(bp, 'ramp_freeze', rf)
    with caplog.at_level(logging.INFO):
        assert bp.main() == 0
    return rf


def test_paused_main_no_freeze_no_error_json_written(monkeypatch, tmp_path, caplog):
    """Paused: no set_freeze, no ERROR record, BT side written to the JSON."""
    rf = _run_main(monkeypatch, tmp_path, False, caplog, FreezeState(book='bf'))
    rf.set_freeze.assert_not_called()
    rf.clear_freeze.assert_not_called()
    assert not [r for r in caplog.records if r.levelname == 'ERROR']
    assert (tmp_path / 'bf_parity_2026-11-11.json').exists()


def test_paused_main_clears_false_path_freeze_with_reason(monkeypatch, tmp_path, caplog):
    """A freeze left by the false path (hard-parity marker reason) is cleared with the paused-book reason."""
    st = FreezeState(book='bf', frozen=True, since='2026-11-06', reason="BT_ONLY — ABC")
    rf = _run_main(monkeypatch, tmp_path, False, caplog, st)
    rf.clear_freeze.assert_called_once()
    assert rf.clear_freeze.call_args.args[:2] == ('bf', 'paused book, parity not applicable')
    rf.set_freeze.assert_not_called()


def test_paused_main_leaves_manual_freeze(monkeypatch, tmp_path, caplog):
    """A freeze whose reason is not a parity marker keeps its owner."""
    st = FreezeState(book='bf', frozen=True, since='2026-11-06', reason="owner hold")
    rf = _run_main(monkeypatch, tmp_path, False, caplog, st)
    rf.clear_freeze.assert_not_called()


def test_enabled_main_still_freezes_on_breach(monkeypatch, tmp_path, caplog):
    """Enabled book with a BT-only trade: freeze path unchanged (set_freeze called)."""
    rf = _run_main(monkeypatch, tmp_path, True, caplog, FreezeState(book='bf'))
    rf.set_freeze.assert_called_once()
    rf.clear_freeze.assert_not_called()


def test_eod_parity_section_prints_paused_status(tmp_path, monkeypatch):
    """eod_report.parity_section shows the PAUSED status verbatim."""
    import json
    import eod_report
    (tmp_path / 'logs/bf_parity').mkdir(parents=True)
    (tmp_path / 'logs/bf_parity/bf_parity_2026-11-11.json').write_text(json.dumps(
        {'status': bp.STATUS_PAUSED, 'paused': True, 'n_bt_trades': 2, 'n_live_rows': 0, 'bt_stale': False}))
    monkeypatch.setattr(eod_report, 'ROOT', tmp_path)
    monkeypatch.setattr(eod_report, 'freeze_summary', lambda: 'freeze: none')
    out = eod_report.parity_section('2026-11-11')
    assert 'BF parity PAUSED (BT-only)' in out
