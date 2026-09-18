#!/usr/bin/env python3
"""S1-PASSIVE step 3 — Alpaca's `shortable` / `easy_to_borrow` flags, read-only.

PREREG.md §5: an event is `tradeable` iff BOTH flags are true. This is TODAY's
snapshot (the assets endpoint carries no history), so it is a weak proxy for the
borrow state on the halt date — the survivorship caveat is stated in the REPORT.

Read-only: `TradingClient.get_all_assets`. No order, no config, nothing written
outside this directory.
"""
from __future__ import annotations

import os
import sys

import pandas as pd
from dotenv import load_dotenv

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
load_dotenv(f'{ROOT}/.env')

from alpaca.trading.client import TradingClient          # noqa: E402
from alpaca.trading.requests import GetAssetsRequest     # noqa: E402
from alpaca.trading.enums import AssetClass, AssetStatus  # noqa: E402

P = f'{ROOT}/research/fuckup_audit/O_halt/PASSIVE'


def main() -> int:
    cl = TradingClient(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_API_SECRET'),
                       paper=False)
    assets = cl.get_all_assets(GetAssetsRequest(asset_class=AssetClass.US_EQUITY,
                                                status=AssetStatus.ACTIVE))
    rows = [dict(symbol=a.symbol, tradable=bool(a.tradable),
                 shortable=bool(getattr(a, 'shortable', False)),
                 easy_to_borrow=bool(getattr(a, 'easy_to_borrow', False)),
                 exchange=str(a.exchange)) for a in assets]
    d = pd.DataFrame(rows)
    d.to_csv(f'{P}/borrow_flags.csv', index=False)
    print(f'assets {len(d):,}  shortable {d.shortable.mean():.3f}  etb {d.easy_to_borrow.mean():.3f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
