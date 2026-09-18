#!/usr/bin/env python3
"""Multi-day DATA stage — print every count DATA.md quotes, from the artifacts.

One place to regenerate the report's numbers, so DATA.md can never drift from the
panel it describes. Reads only; prints a block per section.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def hdr(t: str) -> None:
    print(f'\n===== {t} =====', flush=True)


def main() -> None:
    uni = pd.read_parquet(HERE / 'universe.parquet')
    hdr('1 UNIVERSE')
    print(f'symbols: {len(uni)}')
    print(uni['exchange'].value_counts().to_string())
    print(uni['kind'].value_counts().to_string())
    print(uni[['marginable', 'shortable', 'easy_to_borrow', 'fractionable']].mean()
          .round(4).to_string())
    if 'sic2' in uni.columns:
        hdr('6 SIC')
        print(f'sic2 present: {uni["sic2"].notna().sum()}/{len(uni)} '
              f'({100 * uni["sic2"].notna().mean():.1f}%)')
        for kind, g in uni.groupby('kind'):
            print(f'  {kind}: {100 * g["sic2"].notna().mean():.1f}% of {len(g)}')
        print('top 2-digit SIC:')
        print(uni['sic2'].value_counts().head(8).to_string())

    hdr('2 PRICES')
    for adj in ('raw', 'all'):
        files = sorted((HERE / 'prices_by_year' / adj).glob('year=*.parquet'))
        tot, syms = 0, set()
        for f in files:
            d = pd.read_parquet(f, columns=['symbol'])
            tot += len(d)
            syms |= set(d['symbol'].astype(str).unique())
            print(f'  [{adj}] {f.stem}: {len(d):,}')
        print(f'  [{adj}] TOTAL rows {tot:,} over {len(syms):,} symbols, '
              f'{len(files)} year files')

    scale = HERE / 'price_scale_check.csv'
    if scale.exists():
        s = pd.read_csv(scale)
        hdr('PRICE-SCALE CHECK')
        print(f'keys tested {len(s)}; fail@0.01% '
              f'{100 * (s["rel_err"] > 1e-4).mean():.2f}%; '
              f'median {s["rel_err"].median():.2e}; p99 {s["rel_err"].quantile(0.99):.2e}')

    pit = HERE / 'survivorship_pit.csv'
    if pit.exists():
        hdr('3a SURVIVORSHIP — Databento PIT definitions')
        t = pd.read_csv(pit)
        print(t[t.scope == 'common_stock'].to_string(index=False))
    xn = HERE / 'survivorship_xnas.csv'
    if xn.exists():
        hdr('3b SURVIVORSHIP — XNAS.ITCH daily')
        print(pd.read_csv(xn).to_string(index=False))
    dl = HERE / 'delisted_names.parquet'
    if dl.exists():
        d = pd.read_parquet(dl)
        print(f'\ndelisted_names.parquet: {len(d)} names; by source:')
        print(d['source'].value_counts().to_string())

    ev_path = HERE / 'earnings_events.parquet'
    if ev_path.exists():
        ev = pd.read_parquet(ev_path)
        hdr('4 EARNINGS EVENTS')
        print(f'events {len(ev):,}; symbols {ev["symbol"].nunique():,}; '
              f'CIKs {ev["cik"].nunique():,}')
        print(f'share of universe with >=1 event: '
              f'{100 * ev["symbol"].nunique() / len(uni):.1f}% of {len(uni)}; '
              f'of common stocks: '
              f'{100 * uni[uni.kind == "common"]["symbol"].isin(ev["symbol"]).mean():.1f}%')
        print('events per year:')
        print(ev.groupby(pd.to_datetime(ev['event_session']).dt.year).size().to_string())
        print('acceptance bucket:')
        print((100 * ev['acceptance_bucket'].value_counts(normalize=True)).round(2).to_string())
        if 'n_prior_qfacts' in ev.columns:
            hdr('5 EPS AVAILABILITY')
            c = ev['n_prior_qfacts'].fillna(0)
            print(f'events with >=1 prior EPS fact: {100 * (c >= 1).mean():.1f}%')
            print(f'events with >=9 prior EPS facts (SUE-ready): {100 * (c >= 9).mean():.1f}%')
            print(f'symbols with >=9 on their last event: '
                  f'{100 * (ev.groupby("symbol")["n_prior_qfacts"].max() >= 9).mean():.1f}%')
            by_year = ev.assign(y=pd.to_datetime(ev['event_session']).dt.year).groupby('y')
            print('SUE-ready share per year:')
            print((100 * by_year['n_prior_qfacts'].apply(lambda s: (s >= 9).mean())).round(1)
                  .to_string())

    fp = HERE / 'eps_facts.parquet'
    if fp.exists():
        f = pd.read_parquet(fp)
        hdr('5 EPS FACTS')
        print(f'quarterly facts {len(f):,} on {f["cik"].nunique():,} CIKs; '
              f'derived Q4 {int((f["source"] == "derived_q4").sum()):,}')
        print(f['tag'].value_counts().to_string())


if __name__ == '__main__':
    main()
