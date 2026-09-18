#!/usr/bin/env python3
"""R_daily step 2 — the REAL corporate-action source (CLAUDE.md price-scale rule).

Preference order stated in the task: (a) Databento definition/corporate-action fields — the bought
`pit_definition` files carry instrument definitions only (no split ratios) and no corporate-actions
schema is entitled on this account, so (b) **Alpaca's corporate-actions endpoint** is the source.  It
is free, covers 2016 on, and returns `ex_date` + `old_rate`/`new_rate` per event.

Split-like events fetched: forward_splits, reverse_splits, unit_splits, stock_dividends.  A
stock dividend is a price-scale event too (`new_rate` extra shares per `old_rate` held).

Writes R_daily/corporate_actions.csv  (symbol, ex_date, kind, old_rate, new_rate, ratio) where
`ratio = new_rate / old_rate` is the SHARE multiplier: prices BEFORE ex_date must be divided by
`ratio` and volumes multiplied by it to sit on the post-event scale.
"""
from __future__ import annotations

import os
import sys
from datetime import date, datetime, timezone

import pandas as pd
from dotenv import load_dotenv

os.chdir('/home/ec2-user/onemil')
load_dotenv('/home/ec2-user/onemil/.env')
R = 'research/fuckup_audit/R_daily'
OUT = f'{R}/corporate_actions.csv'
TYPES = ['forward_split', 'reverse_split', 'unit_split', 'stock_dividend']


def log(m):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {m}', flush=True)


def main() -> int:
    from alpaca.data.historical.corporate_actions import CorporateActionsClient
    from alpaca.data.requests import CorporateActionsRequest
    key, sec = os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_API_SECRET')
    if not key or not sec:
        raise SystemExit('ALPACA_API_KEY / ALPACA_API_SECRET missing from .env')
    cli = CorporateActionsClient(api_key=key, secret_key=sec)

    frames = []
    for y in range(2018, 2027):
        for q, (a, b) in enumerate(((f'{y}-01-01', f'{y}-03-31'), (f'{y}-04-01', f'{y}-06-30'),
                                    (f'{y}-07-01', f'{y}-09-30'), (f'{y}-10-01', f'{y}-12-31'))):
            if a > '2026-09-05':
                continue
            b = min(b, '2026-09-05')
            req = CorporateActionsRequest(types=TYPES, start=date.fromisoformat(a),
                                          end=date.fromisoformat(b), limit=10000)
            d = cli.get_corporate_actions(req).df
            if len(d):
                frames.append(d.reset_index())
            log(f'{a}..{b}: {len(d):,} events')
    if not frames:
        raise SystemExit('no corporate actions returned')
    d = pd.concat(frames, ignore_index=True)
    d = d[['corporate_action_type', 'symbol', 'ex_date', 'process_date', 'old_rate', 'new_rate']]
    d['ex_date'] = d.ex_date.fillna(d.process_date).astype(str).str[:10]
    d = d.drop(columns=['process_date'])
    d['old_rate'] = pd.to_numeric(d.old_rate, errors='coerce')
    d['new_rate'] = pd.to_numeric(d.new_rate, errors='coerce')
    d = d.rename(columns={'corporate_action_type': 'kind'})
    # stock dividend: `new_rate` NEW shares per `old_rate` held -> share multiplier 1 + new/old
    sd = d.kind.astype(str).str.startswith('stock_div')
    d['ratio'] = d.new_rate / d.old_rate
    d.loc[sd, 'ratio'] = 1.0 + d.new_rate[sd] / d.old_rate[sd]
    d = d[d.symbol.notna() & d.ratio.notna() & (d.ratio > 0)]
    d = d.drop_duplicates(subset=['symbol', 'ex_date', 'kind', 'ratio'])
    d = d.sort_values(['symbol', 'ex_date'])
    d.to_csv(OUT, index=False, float_format='%.10g')
    log(f'wrote {OUT}: {len(d):,} events, {d.symbol.nunique():,} symbols, '
        f'{d.ex_date.min()}..{d.ex_date.max()}')
    print(d.kind.value_counts().to_string(), flush=True)
    print(f'ratio != 1 events: {(d.ratio != 1.0).sum():,}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
