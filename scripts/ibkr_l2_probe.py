"""IBKR L2 data subscription probe.

Connects to a running TWS or IB Gateway instance, subscribes to L2 depth
on a small sample of test symbols, and reports:
  - Whether L2 data flows at all
  - Which exchanges are available (depends on user's data subscriptions)
  - Update rate and depth visible
  - Gaps / errors

Use this BEFORE committing to L2 microstructure work. If no L2 data flows,
the user needs to activate Nasdaq TotalView / ArcaBook subscriptions in IBKR
account management first.

Prerequisites (IF NOT ALREADY DONE):
  1. Install TWS or IB Gateway on this host (or accessible host)
  2. Login with IBKR credentials, enable API access in Configure → API
     - Default port 7497 (paper) or 7496 (live)
     - Enable "Allow connections from localhost"
  3. Subscribe to L2 feeds at https://www.interactivebrokers.com/en/index.php?f=14193
     Recommended for ORB universe:
       - Nasdaq TotalView ($25/mo) — covers Nasdaq-listed small caps
       - ArcaBook ($25-35/mo) — covers NYSE Arca / NYSE listings
     Single-user, non-pro. Total ~$50-60/mo.
  4. pip install ib_insync

Usage:
  python3 scripts/ibkr_l2_probe.py [--port 7497] [--symbols SPY AAPL ATOM]

Run for ~30 seconds during market hours. Output should show book updates
streaming. If you see "no permission" or empty output, your subscriptions
need activation.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from typing import Dict, List


# Soft-import — let user know they need ib_insync if they don't have it
try:
    from ib_insync import IB, Stock, Ticker
except ImportError:
    print("ERROR: ib_insync not installed. Install with:")
    print("  pip install ib_insync")
    print("\nAlternatively, use the official ibapi (more verbose).")
    sys.exit(1)


def probe(port: int, symbols: List[str], duration_s: int) -> Dict:
    """Connect to TWS/Gateway, subscribe to L2 for symbols, collect stats.

    Returns dict per symbol with: bids/asks observed, exchanges seen, errors.
    """
    ib = IB()
    print(f"Connecting to TWS/Gateway on 127.0.0.1:{port}...")
    try:
        ib.connect('127.0.0.1', port, clientId=42, timeout=10)
    except Exception as e:
        print(f"\nERROR: Could not connect to TWS/Gateway:")
        print(f"  {e}")
        print(f"\nIs TWS or IB Gateway running on port {port}?")
        print(f"  Paper: 7497 (default)")
        print(f"  Live:  7496 (default)")
        print(f"Make sure 'Enable API' is checked in TWS Configure → API.")
        sys.exit(1)

    print(f"Connected: server version={ib.client.serverVersion()}")
    print(f"Account: {ib.managedAccounts()}")

    # Subscribe to L2 (mkt depth) on each symbol
    contracts = {sym: Stock(sym, 'SMART', 'USD') for sym in symbols}
    tickers: Dict[str, Ticker] = {}
    print(f"\nSubscribing to L2 depth on {len(symbols)} symbols...")
    for sym, contract in contracts.items():
        try:
            ib.qualifyContracts(contract)
            # numRows=10 = 10 levels deep on each side. Adjust per subscription.
            ticker = ib.reqMktDepth(contract, numRows=10, isSmartDepth=True)
            tickers[sym] = ticker
            print(f"  ✓ {sym}: subscription request sent")
        except Exception as e:
            print(f"  ✗ {sym}: error: {e}")

    # Wait for data to flow
    print(f"\nCollecting for {duration_s}s. Watching for L2 book updates...")
    stats: Dict[str, dict] = {sym: {
        'bid_updates': 0, 'ask_updates': 0,
        'exchanges_bid': set(), 'exchanges_ask': set(),
        'max_bid_depth': 0, 'max_ask_depth': 0,
        'errors': [],
    } for sym in symbols}

    start = time.time()
    while time.time() - start < duration_s:
        ib.sleep(0.5)
        for sym, ticker in tickers.items():
            try:
                if ticker.domBids:
                    stats[sym]['bid_updates'] += 1
                    stats[sym]['max_bid_depth'] = max(
                        stats[sym]['max_bid_depth'],
                        sum(b.size for b in ticker.domBids)
                    )
                    for b in ticker.domBids:
                        if hasattr(b, 'marketMaker') and b.marketMaker:
                            stats[sym]['exchanges_bid'].add(b.marketMaker)
                if ticker.domAsks:
                    stats[sym]['ask_updates'] += 1
                    stats[sym]['max_ask_depth'] = max(
                        stats[sym]['max_ask_depth'],
                        sum(a.size for a in ticker.domAsks)
                    )
                    for a in ticker.domAsks:
                        if hasattr(a, 'marketMaker') and a.marketMaker:
                            stats[sym]['exchanges_ask'].add(a.marketMaker)
            except Exception as e:
                stats[sym]['errors'].append(str(e))

    ib.disconnect()
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=7497,
                        help='TWS/Gateway port (7497=paper default, 7496=live)')
    parser.add_argument('--duration', type=int, default=30,
                        help='Probe duration in seconds (default 30)')
    parser.add_argument('--symbols', nargs='+',
                        default=['SPY', 'AAPL', 'TSLA'],
                        help='Symbols to subscribe (try a small-cap gapper too)')
    args = parser.parse_args()

    stats = probe(args.port, args.symbols, args.duration)

    print(f"\n{'='*72}")
    print(f"  L2 PROBE RESULTS")
    print(f"{'='*72}")
    has_data = False
    for sym, s in stats.items():
        bu = s['bid_updates']; au = s['ask_updates']
        if bu == 0 and au == 0:
            print(f"\n  {sym}: NO DATA RECEIVED")
            print(f"    Likely cause: L2 subscription not active for this venue")
            print(f"    Check: IBKR account → Settings → Market Data Subscriptions")
            if s['errors']:
                print(f"    Errors: {s['errors'][:3]}")
        else:
            has_data = True
            print(f"\n  {sym}:")
            print(f"    Bid updates: {bu}  (depth max: {s['max_bid_depth']} shares)")
            print(f"    Ask updates: {au}  (depth max: {s['max_ask_depth']} shares)")
            if s['exchanges_bid'] | s['exchanges_ask']:
                print(f"    Exchanges seen: {sorted(s['exchanges_bid'] | s['exchanges_ask'])}")
            if s['errors']:
                print(f"    Errors: {s['errors'][:3]}")

    print(f"\n{'='*72}")
    if has_data:
        print("  ✓ L2 data flowing — IBKR setup is functional")
        print("  Next step: backfill historical L2 from Databento for validation")
        print("  See docs/orb_research_apr_2026.md for the L2 plan.")
    else:
        print("  ✗ No L2 data received from any symbol")
        print("  Required actions:")
        print("    1. Verify TWS/Gateway is running with API enabled")
        print("    2. Activate Nasdaq TotalView + ArcaBook subscriptions in")
        print("       IBKR Client Portal → Settings → Market Data Subscriptions")
        print("    3. Re-run this probe after subscriptions become active")


if __name__ == '__main__':
    main()
