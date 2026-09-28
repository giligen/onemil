#!/usr/bin/env python3
"""Read-only real-API probe for the HOD-break / ORB paper accounts (owner 2026-09-28: run each
book on its OWN Alpaca paper account, real paper orders). NEVER submits an order, cancels
anything, or touches a position — GET calls only.

For each of ALPACA_HOD_API_KEY / ALPACA_ORB_API_KEY that is SET, connects with
`paper=config.alpaca_<strategy>_paper` and prints account id/number, the paper flag, equity,
buying power and open-position count. A strategy with neither key nor secret set is SKIPPED
(not configured yet — not an error). A strategy with only ONE of key/secret set is a partial
credential and fails the probe.

The dangerous case this exists to catch: pasting the LIVE (main) account's key into
ALPACA_HOD_API_KEY / ALPACA_ORB_API_KEY while the *_PAPER flag stays true — main.py would then
silently route "paper" orders onto the owner's real live account (the one he trades manually
on; CLAUDE.md: his positions are NEVER touched). Detected two ways: (1) the probed account's
own `account_number` equals the main account's `account_number` while the *_PAPER flag is
true, and (2) the client's own `is_paper` (set once, at construction, from the same flag)
disagrees with the flag we asked for — cheap, but catches a future refactor that hardcodes the
wrong value.

Usage:
    python3 scripts/check_paper_accounts.py

Exit code: 0 if every configured strategy connected AND matched its expected paper/live state;
non-zero (count of problems) otherwise. Run this AFTER the owner adds ALPACA_HOD_API_KEY/SECRET
or ALPACA_ORB_API_KEY/SECRET to .env, before flipping the strategy live.
"""
from __future__ import annotations

import logging
import sys

import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # run from anywhere
from config import Config
from data_sources.alpaca_client import AlpacaClient, AlpacaAPIError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("check_paper_accounts")

STRATEGIES = ("hod", "orb")  # ALPACA_HOD_* / ALPACA_ORB_* — the two books getting their own account (9/28)


def probe_strategy(name: str, key: str, secret: str, expected_paper: bool,
                    main_account_number: str) -> int:
    """Probe one strategy's Alpaca account. Returns the number of problems found (0 = clean)."""
    label = name.upper()
    if not key and not secret:
        logger.info(f"{label}: ALPACA_{label}_API_KEY/SECRET not set — skipping (not configured yet)")
        return 0
    if not key or not secret:
        logger.error(f"{label}: PARTIAL credentials (only one of API_KEY/API_SECRET set) — "
                      f"probe cannot connect, treat as a missing key")
        return 1

    problems = 0
    try:
        client = AlpacaClient(key, secret, paper=expected_paper)
    except AlpacaAPIError as e:
        logger.error(f"{label}: AlpacaClient init failed ({e}) — missing/invalid key")
        return 1

    if not client.test_connection():
        logger.error(f"{label}: connection test FAILED — key/secret invalid for paper={expected_paper}")
        return 1

    try:
        info = client.get_account_info()
        acct_number = str(info.get("account_number") or "")
        positions = client.get_open_positions() or []
    except Exception as e:
        logger.error(f"{label}: account/positions fetch failed ({e})")
        return 1

    mode = "paper" if client.is_paper else "LIVE"
    logger.info(
        f"{label}: account={acct_number} paper={client.is_paper} ({mode}) "
        f"equity=${info['equity']:,.2f} buying_power=${info['buying_power']:,.2f} "
        f"open_positions={len(positions)}"
    )

    # Defensive: is_paper is set once, from the exact `expected_paper` we passed in — this
    # can only disagree if a future refactor hardcodes the wrong value.
    if client.is_paper != expected_paper:
        logger.error(f"{label}: client.is_paper={client.is_paper} but *_PAPER flag says "
                      f"{expected_paper} — wiring bug, treat as UNSAFE")
        problems += 1

    if expected_paper and main_account_number and acct_number == main_account_number:
        logger.error(
            f"{label}: account_number {acct_number} MATCHES the main (LIVE) account while "
            f"ALPACA_{label}_PAPER=true — this strategy would route 'paper' orders onto the "
            f"owner's real live account. DO NOT enable this book."
        )
        problems += 1

    if not expected_paper:
        logger.warning(f"{label}: ALPACA_{label}_PAPER=false — this strategy is configured "
                        f"to trade its account LIVE (real money). Confirm this is intentional.")

    return problems


def main() -> int:
    config = Config()
    total_problems = 0

    main_account_number = ""
    try:
        main_client = AlpacaClient(config.alpaca_api_key, config.alpaca_api_secret,
                                    paper=config.alpaca_paper)
        if main_client.test_connection():
            main_account_number = str(main_client.get_account_info().get("account_number") or "")
            logger.info(f"MAIN: account={main_account_number} paper={main_client.is_paper} "
                        f"(reference account for the cross-check below)")
        else:
            logger.warning("MAIN: connection test failed — the live-account cross-check "
                            "will be skipped for every strategy")
    except Exception as e:
        logger.warning(f"MAIN: account probe failed ({e}) — the live-account cross-check "
                        f"will be skipped for every strategy")

    for name in STRATEGIES:
        key = getattr(config, f"alpaca_{name}_api_key")
        secret = getattr(config, f"alpaca_{name}_api_secret")
        expected_paper = getattr(config, f"alpaca_{name}_paper")
        total_problems += probe_strategy(name, key, secret, expected_paper, main_account_number)

    if total_problems:
        logger.error(f"check_paper_accounts: {total_problems} problem(s) found — see errors above")
    else:
        logger.info("check_paper_accounts: clean")
    return total_problems


if __name__ == "__main__":
    sys.exit(main())
