#!/usr/bin/env python3
"""ORB pre-flight check — run BEFORE enabling the service for trading.

Validates:
  1. orb.yaml loads cleanly + has all required fields
  2. Env vars: ALPACA_ORB_API_KEY / SECRET present (for Phase 1 paper)
  3. Alpaca paper account connects + has reasonable buying power
  4. Can create AlpacaClient + fetch snapshots (universe query)
  5. Database opens + trades table has strategy column
  6. All ORB Python modules import cleanly
  7. z_params / quintile_cutoffs / adaptive_mults parse correctly
  8. Telegram notifier wired (sends test message if --send-telegram)

Usage:
    python3 preflight_orb.py
    python3 preflight_orb.py --send-telegram   # also send a test ping
    python3 preflight_orb.py --skip-alpaca     # skip network-dependent checks
"""
from __future__ import annotations

import argparse
import sys
import traceback
from datetime import date, datetime, timezone


def _ok(msg): print(f"  \033[32m✓\033[0m {msg}")
def _fail(msg): print(f"  \033[31m✗\033[0m {msg}")
def _warn(msg): print(f"  \033[33m⚠\033[0m {msg}")
def _section(msg): print(f"\n=== {msg} ===")


def check_config_yaml() -> bool:
    _section("1. orb.yaml loads + schema valid")
    try:
        import yaml
        with open('orb.yaml') as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        _fail(f"orb.yaml load failed: {e}")
        return False
    _ok("orb.yaml parsed")
    required = {
        'strategy.enabled': cfg.get('strategy', {}).get('enabled'),
        'universe.source': cfg.get('universe', {}).get('source'),
        'universe.min_gap_pct': cfg.get('universe', {}).get('min_gap_pct'),
        'universe.min_prev_volume': cfg.get('universe', {}).get('min_prev_volume'),
        'filter.threshold': cfg.get('filter', {}).get('threshold'),
        'filter.features': cfg.get('filter', {}).get('features'),
        'quintile_cutoffs': cfg.get('quintile_cutoffs'),
        'adaptive_mults': cfg.get('adaptive_mults'),
        'sizing.account_budget_usd': cfg.get('sizing', {}).get('account_budget_usd'),
        'sizing.max_concurrent': cfg.get('sizing', {}).get('max_concurrent'),
        'sizing.risk_per_trade_usd': cfg.get('sizing', {}).get('risk_per_trade_usd'),
        'exit.lock_arm_at_r': cfg.get('exit', {}).get('lock_arm_at_r'),
        'exit.lock_stop_r': cfg.get('exit', {}).get('lock_stop_r'),
        'exit.force_close_time_et': cfg.get('exit', {}).get('force_close_time_et'),
        'risk.daily_loss_limit_usd': cfg.get('risk', {}).get('daily_loss_limit_usd'),
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        _fail(f"Missing keys: {missing}")
        return False
    _ok(f"All {len(required)} required keys present")

    # Check filter.features has 7 entries
    feats = cfg.get('filter', {}).get('features', {})
    if len(feats) != 7:
        _fail(f"filter.features has {len(feats)} entries (expected 7)")
        return False
    _ok(f"filter.features has 7 entries: {', '.join(feats.keys())}")

    # Check quintile_cutoffs length
    cutoffs = cfg.get('quintile_cutoffs', [])
    if len(cutoffs) != 4:
        _fail(f"quintile_cutoffs has {len(cutoffs)} values (expected 4)")
        return False
    _ok(f"quintile_cutoffs: {cutoffs}")

    # Check adaptive_mults has Q1-Q5
    mults = cfg.get('adaptive_mults', {})
    if set(mults.keys()) != {'Q1', 'Q2', 'Q3', 'Q4', 'Q5'}:
        _fail(f"adaptive_mults keys: {set(mults.keys())}")
        return False
    if mults.get('Q5', 999) > 1.5:
        _fail(f"Q5 mult {mults['Q5']} exceeds 1.5 cap (anti-overfit violated)")
        return False
    _ok(f"adaptive_mults: Q5={mults['Q5']} (at/below 1.5 cap) ✓")

    # Master kill switch
    enabled = cfg.get('strategy', {}).get('enabled', False)
    if enabled:
        _warn(f"strategy.enabled=TRUE — service WILL trade when started")
    else:
        _ok("strategy.enabled=FALSE (safe default, code dormant in prod)")
    return True


def check_env_vars(skip_alpaca: bool) -> bool:
    _section("2. Environment variables")
    import os
    from pathlib import Path
    from dotenv import load_dotenv
    load_dotenv(Path('.env'))

    orb_key = os.getenv('ALPACA_ORB_API_KEY', '')
    orb_secret = os.getenv('ALPACA_ORB_API_SECRET', '')
    orb_paper = os.getenv('ALPACA_ORB_PAPER', 'true').lower() in ('true', '1', 'yes')

    if not orb_key or not orb_secret:
        if skip_alpaca:
            _warn("ALPACA_ORB_API_KEY/SECRET empty (skip_alpaca=on, OK for offline check)")
            return True
        _fail(f"ALPACA_ORB_API_KEY set={bool(orb_key)} / SECRET set={bool(orb_secret)}")
        _fail("Add to .env before enabling ORB")
        return False
    _ok(f"ALPACA_ORB_API_KEY set ({orb_key[:6]}...{orb_key[-4:] if len(orb_key) > 10 else ''})")
    _ok(f"ALPACA_ORB_API_SECRET set ({len(orb_secret)} chars)")
    _ok(f"ALPACA_ORB_PAPER={'paper' if orb_paper else 'LIVE'}")
    if not orb_paper:
        _warn("NOT paper mode — real money at risk!")
    return True


def check_alpaca_connection(skip: bool) -> bool:
    _section("3. Alpaca paper account connection")
    if skip:
        _warn("Skipped (skip_alpaca flag)")
        return True
    try:
        from config import Config
        from data_sources.alpaca_client import AlpacaClient
        cfg = Config()
        if not cfg.alpaca_orb_api_key:
            _warn("ORB API key not set — skipping Alpaca connection check")
            return True
        client = AlpacaClient(cfg.alpaca_orb_api_key, cfg.alpaca_orb_api_secret,
                              paper=cfg.alpaca_orb_paper)
        if not client.test_connection():
            _fail("AlpacaClient.test_connection() returned False")
            return False
        info = client.get_account_info()
        bp = float(info.get('buying_power', 0))
        mode = "paper" if client.is_paper else "LIVE"
        _ok(f"Connected ({mode} mode)")
        _ok(f"Buying power: ${bp:,.0f}")
        if bp < 100_000:
            _warn(f"Buying power < $100K — ORB's budget is $100K, may hit limits")
        # Try a snapshot query (universe builder would use this)
        try:
            snaps = client.get_snapshots(['AAPL']) if hasattr(client, 'get_snapshots') else None
            if snaps:
                _ok(f"get_snapshots() works (fetched {len(snaps)} symbol)")
            else:
                _warn("get_snapshots returned empty — universe builder may not work")
        except Exception as e:
            _warn(f"get_snapshots failed: {e}")
        return True
    except Exception as e:
        _fail(f"Alpaca connection error: {e}")
        traceback.print_exc()
        return False


def check_database() -> bool:
    _section("4. Database opens + trades.strategy column")
    try:
        from persistence.database import Database, get_database
        db = get_database()
        # Check trades table has strategy column via a filtered query
        today = date.today()
        orb_trades = db.get_open_trades(today, strategy='orb')
        _ok(f"trades.strategy column queryable (ORB open trades today: {len(orb_trades)})")
        bf_trades = db.get_open_trades(today, strategy='bull_flag')
        mw_trades = db.get_open_trades(today, strategy='macd_wave')
        _ok(f"  bull_flag open today: {len(bf_trades)}, macd_wave: {len(mw_trades)}")
        return True
    except Exception as e:
        _fail(f"DB check failed: {e}")
        return False


def check_module_imports() -> bool:
    _section("5. ORB Python modules import")
    modules = [
        'trading.orb_engine',
        'trading.orb_filter',
        'trading.orb_correlation',
        'trading.orb_conviction',
        'trading.orb_planner',
        'trading.stop_monitor',
        'scanner.realtime_scanner',
    ]
    ok = True
    for m in modules:
        try:
            __import__(m)
            _ok(m)
        except Exception as e:
            _fail(f"{m}: {e}")
            ok = False
    return ok


def check_orb_params_loadable() -> bool:
    _section("6. ORB params load through loaders")
    try:
        import yaml
        from trading.orb_filter import load_feature_params
        from trading.orb_conviction import load_adaptive_mults

        with open('orb.yaml') as f:
            cfg = yaml.safe_load(f)
        params = load_feature_params(cfg['filter'])
        _ok(f"z-score params loaded ({len(params)} features)")
        for name, p in params.items():
            if p.std <= 0:
                _fail(f"Feature '{name}' has std={p.std}")
                return False
        _ok("All stds > 0")

        mults = load_adaptive_mults(cfg['adaptive_mults'])
        _ok(f"adaptive mults loaded: "
            + " ".join(f"{q}={mults[q]:.2f}" for q in ['Q1','Q2','Q3','Q4','Q5']))
        if mults['Q5'] > 1.5 + 1e-9:
            _fail(f"Q5 cap violated: {mults['Q5']}")
            return False
        _ok("Q5 cap enforced (<= 1.5x)")
        return True
    except Exception as e:
        _fail(f"Loader error: {e}")
        traceback.print_exc()
        return False


def check_engine_can_instantiate(skip_alpaca: bool) -> bool:
    _section("7. ORBEngine instantiates cleanly")
    try:
        from unittest.mock import MagicMock
        from data_sources.alpaca_client import AlpacaClient
        from persistence.database import Database
        from trading.stop_monitor import StopMonitor
        from trading.orb_engine import ORBEngine
        import yaml

        with open('orb.yaml') as f:
            cfg = yaml.safe_load(f)
        cfg['strategy']['enabled'] = True

        mock_alpaca = MagicMock(spec=AlpacaClient)
        mock_alpaca.get_open_positions.return_value = []
        mock_alpaca.get_account_info.return_value = {'buying_power': 100_000.0}
        mock_alpaca.trading_client = MagicMock()
        mock_alpaca.trading_client.get_orders.return_value = []
        mock_db = MagicMock(spec=Database)
        mock_db.save_trade.return_value = 1
        mock_db.get_open_trades.return_value = []
        mock_sm = MagicMock(spec=StopMonitor)
        mock_sm.polling_mode = False

        engine = ORBEngine(
            alpaca_client=mock_alpaca, db=mock_db,
            stop_monitor=mock_sm, config=cfg,
        )
        _ok(f"ORBEngine instantiated — enabled={engine.enabled}, "
            f"N={engine.max_concurrent}, risk=${engine.planner.risk_per_trade_usd:,.0f}")
        # Try sync_positions (should succeed with empty state)
        engine.sync_positions()
        _ok("sync_positions() runs without error")
        # Try build_universe
        n = engine.build_universe(source_loader=lambda: ['AAPL', 'TSLA'])
        _ok(f"build_universe() returned {n} candidates")
        # reset_daily
        engine.reset_daily()
        _ok("reset_daily() runs without error")
        return True
    except Exception as e:
        _fail(f"Engine instantiation failed: {e}")
        traceback.print_exc()
        return False


def check_tests_green() -> bool:
    _section("8. Unit + integration tests pass")
    import subprocess
    try:
        result = subprocess.run(
            ['python3', '-m', 'pytest',
             'tests/test_orb_filter.py',
             'tests/test_orb_correlation.py',
             'tests/test_orb_conviction.py',
             'tests/test_orb_planner.py',
             'tests/test_orb_engine.py',
             'tests/test_orb_stop_monitor_routing.py',
             'tests/test_orb_integration.py',
             'tests/test_orb_fixes.py',
             'tests/test_orb_fixes2.py',
             'tests/test_orb_fixes3.py',
             '-q', '--no-header'],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            # Extract pass count from summary line
            last = (result.stdout.strip().splitlines() or [''])[-1]
            _ok(last)
            return True
        else:
            _fail("pytest failed:")
            print(result.stdout[-2000:])
            print(result.stderr[-500:])
            return False
    except Exception as e:
        _fail(f"Test runner error: {e}")
        return False


def send_telegram_ping() -> bool:
    _section("9. Telegram test message")
    try:
        from config import Config
        cfg = Config()
        if not cfg.telegram_bot_token or not cfg.telegram_chat_id:
            _warn("Telegram not configured (TELEGRAM_BOT_TOKEN/CHAT_ID empty)")
            return True
        from notifications.telegram_notifier import TelegramNotifier
        notifier = TelegramNotifier(
            bot_token=cfg.telegram_bot_token,
            chat_id=cfg.telegram_chat_id,
            enabled=True,
        )
        import asyncio
        async def send():
            await notifier.send_message(
                f"[ORB PREFLIGHT] {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
                f"— pre-flight check from preflight_orb.py"
            )
        asyncio.run(send())
        _ok("Telegram message sent (check your chat)")
        return True
    except Exception as e:
        _fail(f"Telegram ping failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-alpaca', action='store_true',
                        help='Skip Alpaca-network-dependent checks (offline mode)')
    parser.add_argument('--send-telegram', action='store_true',
                        help='Also send a test Telegram message')
    parser.add_argument('--skip-tests', action='store_true',
                        help='Skip pytest run (save ~30s)')
    args = parser.parse_args()

    print(f"\nORB PRE-FLIGHT CHECK — {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"Mode: {'OFFLINE' if args.skip_alpaca else 'FULL'}")

    checks = [
        ('orb.yaml', check_config_yaml()),
        ('env vars', check_env_vars(args.skip_alpaca)),
        ('Alpaca paper', check_alpaca_connection(args.skip_alpaca)),
        ('database', check_database()),
        ('module imports', check_module_imports()),
        ('params loadable', check_orb_params_loadable()),
        ('engine instantiates', check_engine_can_instantiate(args.skip_alpaca)),
    ]
    if not args.skip_tests:
        checks.append(('tests green', check_tests_green()))
    if args.send_telegram:
        checks.append(('telegram ping', send_telegram_ping()))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)
    for name, ok in checks:
        mark = '\033[32m✓\033[0m' if ok else '\033[31m✗\033[0m'
        print(f"  {mark} {name}")
    print(f"\n  {passed}/{total} checks passed")
    if passed == total:
        print("\n  \033[32mALL CHECKS GREEN — ready for paper validation\033[0m")
        sys.exit(0)
    else:
        print("\n  \033[31mSOME CHECKS FAILED — fix before enabling ORB\033[0m")
        sys.exit(1)


if __name__ == '__main__':
    main()
