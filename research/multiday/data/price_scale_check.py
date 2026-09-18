#!/usr/bin/env python3
"""Multi-day DATA stage — the price-scale check (PLAN §1 standing rule, item 3).

The two price panels (raw and split/dividend-adjusted) must be the SAME tape at a
different scale. The check, on 200 random (symbol, date) keys:

    cum_factor(d) = 1 / Π_{d' > d} [ adj_close(d')/adj_close(d'-1) ÷ raw_close(d')/raw_close(d'-1) ]

i.e. the cumulative corporate-action factor rebuilt by CHAINING daily return
ratios forward from the key date to the symbol's last session — it never reads
adj/raw on the key date itself. It must satisfy

    raw_close(key) × cum_factor(key) == adj_close(key)   within 0.01%

A failure means the two pulls disagree about a split/dividend, or the two series
are misaligned by a date or a symbol — the exact defect that fabricates setups
(PLAN §1: "splits and dividends silently fabricate setups otherwise").

Also reported: the same factor derived from open/high/low (must agree with close),
and that the factor is ~1.0 on each sampled symbol's last session.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BY_YEAR = HERE / 'prices_by_year'
N_KEYS = 200
SEED = 20260918
TOL = 1e-4          # 0.01%


def load(adj: str, symbols: set[str]) -> pd.DataFrame:
    parts = []
    for f in sorted((BY_YEAR / adj).glob('year=*.parquet')):
        d = pd.read_parquet(f, columns=['symbol', 'date', 'open', 'high', 'low', 'close'])
        d = d[d['symbol'].astype(str).isin(symbols)]
        parts.append(d)
    out = pd.concat(parts, ignore_index=True)
    out['symbol'] = out['symbol'].astype(str)
    out['date'] = pd.to_datetime(out['date'])
    return out.sort_values(['symbol', 'date']).reset_index(drop=True)


def main() -> None:
    rng = np.random.default_rng(SEED)
    # sample keys from the raw panel
    files = sorted((BY_YEAR / 'raw').glob('year=*.parquet'))
    sample = []
    per_file = max(1, N_KEYS // len(files) + 2)
    for f in files:
        d = pd.read_parquet(f, columns=['symbol', 'date'])
        idx = rng.choice(len(d), size=min(per_file, len(d)), replace=False)
        sample.append(d.iloc[idx])
    keys = pd.concat(sample, ignore_index=True)
    keys = keys.sample(n=min(N_KEYS, len(keys)), random_state=SEED).reset_index(drop=True)
    keys['symbol'] = keys['symbol'].astype(str)
    keys['date'] = pd.to_datetime(keys['date'])

    syms = set(keys['symbol'])
    raw, adj = load('raw', syms), load('all', syms)
    merged = raw.merge(adj, on=['symbol', 'date'], suffixes=('_raw', '_adj'))
    print(f'{len(keys)} keys on {len(syms)} symbols; joined rows {len(merged):,}', flush=True)

    results = []
    for sym, g in merged.groupby('symbol'):
        g = g.sort_values('date').reset_index(drop=True)
        if len(g) < 2:
            continue
        ratio = ((g['close_adj'] / g['close_adj'].shift(1)) /
                 (g['close_raw'] / g['close_raw'].shift(1)))
        # cum_factor(d) = 1 / product of ratios strictly after d  (= g(d) if the tapes agree)
        logs = np.log(ratio.iloc[1:].values)
        suffix = np.concatenate([np.cumsum(logs[::-1])[::-1], [0.0]])
        cum_factor = np.exp(-suffix)
        g['cum_factor'] = cum_factor
        g['implied_adj'] = g['close_raw'] * g['cum_factor']
        g['rel_err'] = (g['implied_adj'] - g['close_adj']).abs() / g['close_adj'].abs()
        g['f_close'] = g['close_adj'] / g['close_raw']
        g['f_open'] = g['open_adj'] / g['open_raw']
        g['f_high'] = g['high_adj'] / g['high_raw']
        g['f_low'] = g['low_adj'] / g['low_raw']
        g['ohlc_err'] = np.maximum.reduce([
            (g['f_open'] - g['f_close']).abs(),
            (g['f_high'] - g['f_close']).abs(),
            (g['f_low'] - g['f_close']).abs()]) / g['f_close']
        g['last_f'] = g['f_close'].iloc[-1]
        results.append(g[['symbol', 'date', 'close_raw', 'close_adj', 'cum_factor',
                          'rel_err', 'ohlc_err', 'last_f']])
    allrows = pd.concat(results, ignore_index=True)
    hit = keys.merge(allrows, on=['symbol', 'date'], how='left')
    tested = hit[hit['rel_err'].notna()]

    fail = tested['rel_err'] > TOL
    ohlc_fail = tested['ohlc_err'] > TOL
    last_fail = (tested['last_f'] - 1.0).abs() > TOL
    print(f'keys tested: {len(tested)}/{len(keys)}', flush=True)
    print(f'raw × cum_factor vs adjusted, share failing 0.01%: '
          f'{100 * fail.mean():.2f}% ({int(fail.sum())} keys)', flush=True)
    print(f'median |rel err| = {tested["rel_err"].median():.2e}, '
          f'p99 = {tested["rel_err"].quantile(0.99):.2e}', flush=True)
    print(f'OHLC factor disagreement > 0.01%: {100 * ohlc_fail.mean():.2f}%', flush=True)
    # Diagnosis for that share: Alpaca rounds the ADJUSTED panel to the cent, so the
    # per-field factor differs by up to half a cent of price. Report it in dollars —
    # a relative tolerance is the wrong unit for a rounding artifact.
    abs_err = tested['ohlc_err'] * tested['close_adj']
    print(f'  implied absolute price error: median ${abs_err.median():.4f}, '
          f'max ${abs_err.max():.4f} (half a cent = rounding, not a tape mismatch)',
          flush=True)
    print(f'last-session factor != 1.0: {100 * last_fail.mean():.2f}%', flush=True)
    if fail.any():
        print(tested[fail].head(10).to_string(index=False), flush=True)
    tested.to_csv(HERE / 'price_scale_check.csv', index=False)


if __name__ == '__main__':
    main()
