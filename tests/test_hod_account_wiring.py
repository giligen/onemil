"""Wiring tests for the HOD-break paper account (owner decision 2026-09-28: run
HOD-break on its OWN Alpaca paper account, same isolation pattern as ORB/BF).

Mirrors tests/test_bf_account_wiring.py: Config properties + the
_strategy_uses_separate_account routing decision (empty/same/different key),
source-inspection of the main.py wiring block, the injected-client contract on
HodBreakEngine (orders/positions/quotes must go through the client main.py
hands it, never a hidden default), and the `account` tag written to the trades
row / read back in the EOD split.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from config import Config
from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.hod_break_engine import HodBreakEngine
from trading.stop_monitor import StopMonitor


@pytest.fixture
def empty_env_file(tmp_path, monkeypatch):
    """Empty .env + required main keys only — scoped to HOD wiring decisions,
    not main credential presence (same fixture shape as test_bf_account_wiring.py)."""
    p = tmp_path / "empty.env"
    p.touch()
    monkeypatch.setenv("ALPACA_API_KEY", "test-main-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test-main-secret")
    return str(p)


class TestConfigHODAccessors:
    def test_empty_hod_key_returns_empty_string(self, monkeypatch, empty_env_file):
        monkeypatch.delenv("ALPACA_HOD_API_KEY", raising=False)
        c = Config(env_path=empty_env_file)
        assert c.alpaca_hod_api_key == ""

    def test_empty_hod_secret_returns_empty_string(self, monkeypatch, empty_env_file):
        monkeypatch.delenv("ALPACA_HOD_API_SECRET", raising=False)
        c = Config(env_path=empty_env_file)
        assert c.alpaca_hod_api_secret == ""

    def test_hod_paper_defaults_true(self, monkeypatch, empty_env_file):
        monkeypatch.delenv("ALPACA_HOD_PAPER", raising=False)
        c = Config(env_path=empty_env_file)
        assert c.alpaca_hod_paper is True

    def test_hod_paper_explicit_false(self, monkeypatch, empty_env_file):
        monkeypatch.setenv("ALPACA_HOD_PAPER", "false")
        c = Config(env_path=empty_env_file)
        assert c.alpaca_hod_paper is False

    def test_hod_keys_independent_from_orb_and_bf(self, monkeypatch, empty_env_file):
        monkeypatch.setenv("ALPACA_HOD_API_KEY", "hod-key")
        monkeypatch.setenv("ALPACA_HOD_API_SECRET", "hod-secret")
        monkeypatch.setenv("ALPACA_ORB_API_KEY", "orb-key")
        monkeypatch.setenv("ALPACA_BF_API_KEY", "bf-key")
        c = Config(env_path=empty_env_file)
        assert c.alpaca_hod_api_key == "hod-key"
        assert c.alpaca_hod_api_key != c.alpaca_orb_api_key
        assert c.alpaca_hod_api_key != c.alpaca_bf_api_key


class TestHODRoutingCases:
    """The three routing cases main.py's hod_alpaca block decides between,
    driven through the SAME _strategy_uses_separate_account helper BF/ORB use."""

    def test_empty_hod_key_is_same_account(self):
        from main import _strategy_uses_separate_account
        assert _strategy_uses_separate_account("", "PKMAIN") is False

    def test_hod_key_equal_main_is_same_account(self, monkeypatch, empty_env_file):
        from main import _strategy_uses_separate_account
        monkeypatch.setenv("ALPACA_API_KEY", "PKMAIN")
        monkeypatch.setenv("ALPACA_API_SECRET", "main-secret")
        monkeypatch.setenv("ALPACA_HOD_API_KEY", "PKMAIN")
        monkeypatch.setenv("ALPACA_HOD_API_SECRET", "main-secret")
        c = Config(env_path=empty_env_file)
        assert _strategy_uses_separate_account(c.alpaca_hod_api_key, c.alpaca_api_key) is False

    def test_hod_key_different_from_main_is_separate_account(self, monkeypatch, empty_env_file):
        from main import _strategy_uses_separate_account
        monkeypatch.setenv("ALPACA_API_KEY", "PKMAIN")
        monkeypatch.setenv("ALPACA_API_SECRET", "main-secret")
        monkeypatch.setenv("ALPACA_HOD_API_KEY", "PKHOD")
        monkeypatch.setenv("ALPACA_HOD_API_SECRET", "hod-secret")
        c = Config(env_path=empty_env_file)
        assert _strategy_uses_separate_account(c.alpaca_hod_api_key, c.alpaca_api_key) is True


class TestMainPyHODWiring:
    """Source-inspection (same tier as test_bf_account_wiring.py's
    test_main_py_wiring_uses_helper): catches a refactor that silently drops
    the HOD routing without a test failure."""

    def test_hod_block_uses_the_shared_helper(self):
        import re
        src = open("/home/ec2-user/onemil/main.py").read()
        assert re.search(
            r"_strategy_uses_separate_account\(\s*config\.alpaca_hod_api_key,\s*config\.alpaca_api_key\s*\)",
            src,
        ), "HOD must route its same-account decision through _strategy_uses_separate_account"

    def test_hod_same_account_log_line_present(self):
        src = open("/home/ec2-user/onemil/main.py").read()
        assert "HOD Alpaca client: keys match main account" in src

    def test_hod_added_to_stop_monitor_strategy_clients(self):
        """CRITICAL routing: the HOD engine registers exit watches with
        strategy='hod_break' (trading/hod_break_engine.py STRATEGY_NAME) — the
        StopMonitor submits exit orders via `_client_for(watch.strategy)`, so a
        missing 'hod_break' entry here would fill entries on the HOD paper
        account but submit stop/target EXITS on the main account."""
        src = open("/home/ec2-user/onemil/main.py").read()
        assert "strategy_clients['hod_break'] = hod_alpaca" in src

    def test_hod_engine_constructed_with_hod_client_not_bare_main(self):
        src = open("/home/ec2-user/onemil/main.py").read()
        assert "hod_client = hod_alpaca if hod_alpaca is not None else alpaca" in src
        assert "alpaca_client=hod_client" in src

    def test_hod_dedicated_order_stream_on_separate_account(self):
        src = open("/home/ec2-user/onemil/main.py").read()
        assert "HOD OrderStreamWatcher STARTED — separate account" in src
        assert "falling back to REST polling on HOD fills" in src


class TestHODEngineInjectedClient:
    """The engine must never fall back to a hidden default client — every
    order/position/quote call goes through whatever `alpaca_client` main.py
    handed it at construction (paper OR the shared main client)."""

    def _cfg(self, **over):
        base = {'enabled': True, 'dry_run': False, 'risk_usd': 100.0, 'daily_kill_usd': -600.0,
                'weekly_kill_usd': -1500.0, 'max_notional_usd': 5000.0, 'min_price': 1.0,
                'min_adv20': 100_000.0, 'max_spread_bps': 100.0, 'order_timeout_s': 75.0,
                'params': {'consol_bars': 5, 'consol_pct': 0.04, 'min_dist_open_pct': 5.0, 'rv_lo': 1.0,
                           'rv_hi': 5.0, 'min_r_pct': 1.0, 'cap': 0.006, 'target_r': 2.0, 'max_per_day': 8,
                           'max_concurrent': 4, 'last_entry_minute': 930, 'flat_minute': 955}}
        base.update(over)
        return base

    def _hod_paper_client(self, is_paper=True):
        a = MagicMock(spec=AlpacaClient)
        a.is_paper = is_paper
        a.get_open_positions.return_value = []
        a.get_open_orders.return_value = []
        return a

    def test_engine_stores_the_injected_client(self):
        """`self.alpaca is client` — proves the constructor doesn't silently
        swap in a different (e.g. main-account) client."""
        client = self._hod_paper_client()
        db = MagicMock(spec=Database)
        db.get_active_universe.return_value = []
        db.get_open_trades.return_value = []
        sm = MagicMock(spec=StopMonitor); sm.polling_mode = False
        e = HodBreakEngine(client, db, sm, notifier=None, cfg=self._cfg())
        assert e.alpaca is client

    def test_sync_positions_reads_the_injected_client_only(self):
        """sync_positions must see ONLY the HOD account's positions — i.e. it
        must call get_open_positions on the client it was given, not some
        other client."""
        client = self._hod_paper_client()
        client.get_open_positions.return_value = [{'symbol': 'ABC', 'qty': 100}]
        db = MagicMock(spec=Database)
        db.get_active_universe.return_value = []
        db.get_open_trades.return_value = []
        sm = MagicMock(spec=StopMonitor); sm.polling_mode = False
        e = HodBreakEngine(client, db, sm, notifier=None, cfg=self._cfg())
        e.sync_positions()
        assert client.get_open_positions.called

    def test_save_pending_trade_tags_account_from_client_is_paper(self):
        """_save_pending_trade must derive the trades-row `account` tag from
        THIS engine's own client.is_paper — never a hardcoded 'paper'."""
        client = self._hod_paper_client(is_paper=True)
        db = MagicMock(spec=Database)
        db.get_active_universe.return_value = []
        db.get_open_trades.return_value = []
        saved = {}
        db.save_trade.side_effect = lambda rec: saved.update(rec) or 1
        sm = MagicMock(spec=StopMonitor); sm.polling_mode = False
        e = HodBreakEngine(client, db, sm, notifier=None, cfg=self._cfg())
        e._save_pending_trade('ABC', 10, 11.0, 10.5, 12.0, 'order-1', {})
        assert saved['account'] == 'paper'

    def test_save_pending_trade_tags_live_when_client_is_not_paper(self):
        client = self._hod_paper_client(is_paper=False)
        db = MagicMock(spec=Database)
        db.get_active_universe.return_value = []
        db.get_open_trades.return_value = []
        saved = {}
        db.save_trade.side_effect = lambda rec: saved.update(rec) or 1
        sm = MagicMock(spec=StopMonitor); sm.polling_mode = False
        e = HodBreakEngine(client, db, sm, notifier=None, cfg=self._cfg())
        e._save_pending_trade('ABC', 10, 11.0, 10.5, 12.0, 'order-1', {})
        assert saved['account'] == 'live'


class TestTradesAccountColumnMigration:
    """persistence/database.py Migration 17 (account column) + save_trade default."""

    def test_account_column_exists_after_migration(self, tmp_path):
        db = Database(trades_path=str(tmp_path / "trades.db"), cache_path=str(tmp_path / "cache.db"))
        try:
            cols = [r[1] for r in db._trades_conn.execute("PRAGMA table_info(trades)").fetchall()]
            assert 'account' in cols
        finally:
            db.close()

    def test_save_trade_defaults_account_to_paper_and_warns(self, tmp_path, caplog):
        import logging
        db = Database(trades_path=str(tmp_path / "trades.db"), cache_path=str(tmp_path / "cache.db"))
        try:
            with caplog.at_level(logging.WARNING):
                trade_id = db.save_trade({
                    'trade_date': '2026-09-28', 'symbol': 'ABC', 'side': 'buy', 'entry_price': 10.0,
                    'stop_loss_price': 9.5, 'take_profit_price': 11.0, 'shares': 10,
                    'risk_per_share': 0.5, 'total_risk': 5.0, 'risk_reward_ratio': 2.0,
                    'order_id': 'o1', 'order_status': 'pending_new', 'fill_price': None, 'filled_at': None,
                    'exit_price': None, 'exit_reason': None, 'exited_at': None, 'pnl': None, 'pnl_pct': None,
                    'pattern_data': None, 'strategy': 'hod_break',
                })
            row = db.get_trade_by_order_id('o1')
            assert row['account'] == 'paper'
            assert "without explicit account" in caplog.text
        finally:
            db.close()

    def test_save_trade_persists_explicit_account(self, tmp_path):
        db = Database(trades_path=str(tmp_path / "trades.db"), cache_path=str(tmp_path / "cache.db"))
        try:
            db.save_trade({
                'trade_date': '2026-09-28', 'symbol': 'XYZ', 'side': 'buy', 'entry_price': 10.0,
                'stop_loss_price': 9.5, 'take_profit_price': 11.0, 'shares': 10,
                'risk_per_share': 0.5, 'total_risk': 5.0, 'risk_reward_ratio': 2.0,
                'order_id': 'o2', 'order_status': 'pending_new', 'fill_price': None, 'filled_at': None,
                'exit_price': None, 'exit_reason': None, 'exited_at': None, 'pnl': None, 'pnl_pct': None,
                'pattern_data': None, 'strategy': 'hod_break', 'account': 'live',
            })
            row = db.get_trade_by_order_id('o2')
            assert row['account'] == 'live'
        finally:
            db.close()
