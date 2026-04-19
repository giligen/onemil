"""ORB startup smoke test — boots the full ORB stack against the paper account,
verifies wiring, then shuts down cleanly. No orders submitted.

Runs the SAME wiring main.py uses:
  * Load orb.yaml + env
  * Construct paper AlpacaClient
  * Construct shared StopMonitor with alpaca_clients_by_strategy={'orb': paper}
  * Construct ORB OrderStreamWatcher
  * Construct ORBEngine
  * Register bar handler
  * sync_positions() against paper account
  * Tear down (stop stream, stop monitor, close DB)

Exit code 0 = all green. Any exception = non-zero + traceback.

Usage:
    python3 smoke_orb_startup.py
    python3 smoke_orb_startup.py --verbose   # log every step
    python3 smoke_orb_startup.py --universe  # also build snapshot universe (slower)
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from typing import Optional

import yaml


def _run(verbose: bool = False, build_universe: bool = False) -> int:
    """Boot the ORB stack, verify it's healthy, tear down. Return 0 on success."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )
    logger = logging.getLogger('smoke_orb')

    from config import Config
    config = Config()

    if not config.alpaca_orb_api_key or not config.alpaca_orb_api_secret:
        logger.error("ALPACA_ORB_API_KEY/SECRET missing in .env — aborting smoke test")
        return 1

    orb_cfg = yaml.safe_load(open('orb.yaml'))
    if not orb_cfg.get('strategy', {}).get('enabled'):
        logger.warning("orb.yaml strategy.enabled=false — engine will construct but not trade")

    from data_sources.alpaca_client import AlpacaClient
    from persistence.database import Database
    from trading.stop_monitor import StopMonitor
    from trading.order_stream import OrderStreamWatcher
    from trading.orb_engine import ORBEngine

    orb_alpaca: Optional[AlpacaClient] = None
    main_alpaca: Optional[AlpacaClient] = None
    stop_monitor: Optional[StopMonitor] = None
    orb_stream: Optional[OrderStreamWatcher] = None
    db: Optional[Database] = None
    engine: Optional[ORBEngine] = None

    try:
        # 1) Main-account Alpaca (for StopMonitor's market-data WS)
        logger.info("[1/7] Connecting main Alpaca (market-data feed)...")
        main_alpaca = AlpacaClient(config.alpaca_api_key, config.alpaca_api_secret,
                                   paper=True)
        assert main_alpaca.test_connection(), "main Alpaca connection failed"
        logger.info("      main Alpaca OK")

        # 2) ORB paper Alpaca (for order execution)
        logger.info("[2/7] Connecting ORB paper Alpaca (order routing)...")
        orb_alpaca = AlpacaClient(config.alpaca_orb_api_key, config.alpaca_orb_api_secret,
                                  paper=config.alpaca_orb_paper)
        assert orb_alpaca.test_connection(), "ORB Alpaca connection failed"
        acct = orb_alpaca.get_account_info()
        logger.info(f"      ORB paper OK — buying power ${float(acct.get('buying_power', 0)):,.0f}")

        # 3) Database
        logger.info("[3/7] Opening database...")
        db = Database()
        from datetime import datetime
        today = datetime.now().strftime('%Y-%m-%d')
        try:
            orb_today = db.get_open_trades(today, strategy='orb') if hasattr(db, 'get_open_trades') else []
        except TypeError:
            orb_today = db.get_open_trades(today) if hasattr(db, 'get_open_trades') else []
        logger.info(f"      DB OK (currently {len(orb_today)} open ORB trades today)")

        # 4) StopMonitor with strategy-routing
        logger.info("[4/7] Starting StopMonitor (with ORB routing)...")
        stop_monitor = StopMonitor(
            api_key=config.alpaca_api_key,
            api_secret=config.alpaca_api_secret,
            alpaca_client=main_alpaca,
            alpaca_clients_by_strategy={'orb': orb_alpaca},
        )
        stop_monitor.start()
        # Wait briefly for WebSocket to connect (best effort; smoke test doesn't require live data)
        for _ in range(30):
            if stop_monitor._ws_connected:
                break
            time.sleep(0.2)
        ws_state = "connected" if stop_monitor._ws_connected else "not yet connected (OK — smoke check)"
        logger.info(f"      StopMonitor OK — WS {ws_state}")

        # 5) ORB OrderStreamWatcher (paper account)
        logger.info("[5/7] Starting ORB OrderStreamWatcher...")
        orb_stream = OrderStreamWatcher(
            api_key=config.alpaca_orb_api_key,
            api_secret=config.alpaca_orb_api_secret,
            paper=orb_alpaca.is_paper,
            alpaca_client=orb_alpaca,
        )
        orb_stream.start()
        logger.info("      OrderStreamWatcher OK")

        # 6) ORBEngine
        logger.info("[6/7] Instantiating ORBEngine...")
        engine = ORBEngine(
            alpaca_client=orb_alpaca,
            db=db,
            stop_monitor=stop_monitor,
            order_stream=orb_stream,
            config=orb_cfg,
        )
        engine.register_on_stop_monitor()
        logger.info(
            f"      ORBEngine OK — enabled={engine.enabled}, "
            f"N={engine.max_concurrent}, risk=${engine.planner.risk_per_trade_usd:,.0f}"
        )

        # 7) sync_positions + optional universe build
        logger.info("[7/7] sync_positions() against paper account...")
        engine.sync_positions()
        logger.info(f"      sync_positions OK — rehydrated {len(engine.open_positions)} positions")

        if build_universe:
            logger.info("      Building snapshot universe (seed candidates)...")
            probe_syms = ['AAPL', 'TSLA', 'NVDA', 'AMD', 'MSFT', 'META', 'GOOGL', 'AMZN']
            kept = engine.build_orb_universe_from_snapshots(probe_syms)
            logger.info(f"      Snapshot universe probe: {len(kept)}/{len(probe_syms)} pass ORB criteria")

        logger.info("SMOKE TEST GREEN — full ORB stack boots cleanly")
        return 0

    except Exception:
        logger.error("SMOKE TEST FAILED:\n" + traceback.format_exc())
        return 1
    finally:
        # Always tear down in reverse order
        for name, obj, method in [
            ("ORB OrderStream", orb_stream, 'stop'),
            ("StopMonitor", stop_monitor, 'stop'),
        ]:
            if obj is None:
                continue
            try:
                getattr(obj, method)()
            except Exception as e:
                logger.warning(f"  teardown: {name}.{method}() raised {e}")
        try:
            if db is not None:
                db.close() if hasattr(db, 'close') else None
        except Exception:
            pass


def main() -> int:
    p = argparse.ArgumentParser(description="ORB startup smoke test")
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--universe', action='store_true',
                   help="also probe Alpaca snapshots for a small universe")
    args = p.parse_args()
    return _run(verbose=args.verbose, build_universe=args.universe)


if __name__ == '__main__':
    sys.exit(main())
