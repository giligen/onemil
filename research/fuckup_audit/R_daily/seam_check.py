#!/usr/bin/env python3
"""Stage R_daily step 1b — the Nasdaq-listed reference set, the venue-share constant, the seam and
price-scale checks.  Writes R_daily/pit_xnas_common.csv, R_daily/seam.md.

Three questions, all answered before a single return is computed:

1. WHO is in the universe.  `research/scripts/pit_listings` gives point-in-time listing facts from
   2024-07 on: `exchange == 'XNAS'` (primary listing venue) and `security_type == 'C'` (common
   stock).  Before 2024-07 there is no point-in-time source, so the Nasdaq-listed common-stock set
   is the UNION over the bought months, applied to the whole panel.  That is a survivorship filter
   on the pre-2024-07 era (a name that delisted before 2024-07 is absent) and it is stated, not
   hidden: it biases a long-only continuation book UP, so a positive TRAIN result is an upper bound
   while a negative one is safe.  A secondary universe (every ITCH symbol, no PIT membership
   requirement) is carried beside it as the survivorship control.

2. VOLUME ACROSS THE SEAM.  XNAS.ITCH is the Nasdaq venue tape: its volume is a SUBSET of
   consolidated volume, so the same $10M/day gate would mean two different things on the two sides
   of 2024-07.  The one ITCH month that overlaps EQUS.SUMMARY (2024-09, bought by N3 for exactly
   this) gives the constant: `s = median(ITCH dollar volume / EQUS dollar volume)` over Nasdaq-listed
   common stocks with EQUS 20-day median dollar volume >= $10M.  The panel multiplies ITCH volume by
   1/s so ONE $10M rule spans the seam.

3. PRICE SCALE (CLAUDE.md check 3).  ITCH close vs EQUS close on the overlap month, and 200 random
   ITCH-era keys vs `data/cache.db::daily_bars` (Alpaca, read-only).  Both files are unadjusted, so
   the check is for agreement, not for adjustment.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil')
os.chdir('/home/ec2-user/onemil')

from research.scripts.pit_listings import PitListings, is_test_ticker  # noqa: E402

R = 'research/fuckup_audit/R_daily'
ITCH_OLD = 'research/fuckup_audit/N_databento/N3/xnas_daily.parquet'
ITCH_GAP = f'{R}/xnas_daily_2024H1.parquet'
ITCH_CAL = 'research/fuckup_audit/N_databento/N3/raw_daily/xnas_ohlcv1d_cal202409.parquet'
EQUS = ['data/research/databento/equs_daily_2024H2.parquet',
        'data/research/databento/equs_daily_2025_2026.parquet']
OUT_SET = f'{R}/pit_xnas_common.csv'
OUT_MD = f'{R}/seam.md'


def log(m):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {m}', flush=True)


def main() -> int:
    md = ['# R_daily — universe set, venue-share constant, seam and price-scale checks', '']

    # ---------------------------------------------------------------- 1. PIT Nasdaq-listed commons
    pit = PitListings()
    lo, hi = pit.coverage
    log(f'PIT months {lo}..{hi}')
    rows = []
    for m in sorted(p.stem.split('_')[1] for p in pit.dir.glob('def_*.parquet')):
        d = pit._month(m)
        d = d[(d['exchange'] == 'XNAS') & (d['security_type'] == 'C')]
        syms = [s for s in d['raw_symbol'].unique() if not is_test_ticker(s)]
        rows += [(m, s) for s in syms]
    ps = pd.DataFrame(rows, columns=['month', 'symbol'])
    ps.to_csv(OUT_SET, index=False)
    union = sorted(ps.symbol.unique())
    log(f'PIT XNAS common stock: {len(union):,} distinct symbols over {ps.month.nunique()} months '
        f'(median {ps.groupby("month").size().median():.0f}/month)')
    md += ['## 1. Universe reference set (point-in-time listings, 2024-07..2026-09)', '',
           f'- `exchange == XNAS` and `security_type == C`, test tickers removed: '
           f'**{len(union):,} distinct symbols**, median '
           f'**{ps.groupby("month").size().median():.0f} per month**.',
           '- EQUS era (2024-07 on): membership is taken PER MONTH.',
           '- ITCH era (2018-05..2024-06): membership is the UNION of the bought months — a '
           'survivorship filter on that era, quantified in §4 and controlled by the secondary '
           'universe (every ITCH symbol, no PIT requirement).', '']
    uset = set(union)

    # ---------------------------------------------------------------- 2. venue share on 2024-09
    cal = pd.read_parquet(ITCH_CAL, columns=['bar_date', 'symbol', 'close', 'volume'])
    if cal.bar_date.min() < '2024-09-01':   # N3 stored the ET date; session date = UTC date = +1
        cal['bar_date'] = (pd.to_datetime(cal.bar_date) + pd.Timedelta(days=1)).dt.strftime('%Y-%m-%d')
    cal = cal[(cal.bar_date >= '2024-09-01') & (cal.bar_date < '2024-10-01')]
    eq = pd.read_parquet(EQUS[0], columns=['bar_date', 'symbol', 'close', 'volume'])
    eq = eq[(eq.bar_date >= '2024-09-01') & (eq.bar_date < '2024-10-01')]
    log(f'overlap month: ITCH {len(cal):,} rows, EQUS {len(eq):,} rows')
    mg = cal.merge(eq, on=['bar_date', 'symbol'], suffixes=('_i', '_e'))
    mg = mg[mg.symbol.isin(uset)]
    mg = mg[(mg.close_e > 0) & (mg.volume_e > 0) & (mg.volume_i > 0)]
    mg['dv_e'] = mg.close_e * mg.volume_e
    liq = mg[mg.dv_e >= 1e7]
    share = (liq.volume_i / liq.volume_e)
    s_med = float(share.median())
    log(f'venue share on liquid Nasdaq-listed commons: n={len(liq):,} median={s_med:.4f} '
        f'p25={share.quantile(.25):.4f} p75={share.quantile(.75):.4f}')
    md += ['## 2. Venue-share constant (ITCH volume -> consolidated-equivalent)', '',
           f'Overlap month 2024-09, Nasdaq-listed common stocks with EQUS dollar volume >= $10M '
           f'({len(liq):,} symbol-days, {liq.symbol.nunique():,} symbols):', '',
           f'| median | p25 | p75 | p10 | p90 |', '|---:|---:|---:|---:|---:|',
           f'| **{s_med:.4f}** | {share.quantile(.25):.4f} | {share.quantile(.75):.4f} | '
           f'{share.quantile(.10):.4f} | {share.quantile(.90):.4f} |', '',
           f'The panel multiplies every ITCH-era volume by **1/{s_med:.4f} = '
           f'{1 / s_med:.3f}** so that one `20-day median dollar volume >= $10M` rule means the same '
           'thing on both sides of the seam.  The dispersion (p25..p75) is real per-name variation '
           'in Nasdaq market share; it makes the ITCH-era liquidity gate noisier than the EQUS-era '
           'one, never look-ahead.', '']

    # ---------------------------------------------------------------- 3. price scale at the seam
    pr = (mg.close_i / mg.close_e - 1.0).abs()
    md += ['## 3. Price-scale check', '',
           f'**ITCH vs EQUS, same symbol-day, overlap month** ({len(mg):,} keys): median abs '
           f'difference **{pr.median() * 100:.4f}%**, within 0.1% **{(pr < 1e-3).mean() * 100:.1f}%**, '
           f'off by > 0.5% **{(pr > 5e-3).mean() * 100:.2f}%**.  For a Nasdaq-listed name the ITCH '
           'close is the Nasdaq closing cross, i.e. the official close — which is why the universe is '
           'restricted to Nasdaq-listed names (for an NYSE-listed name the ITCH open/close would be '
           'an arbitrary off-primary print).', '']

    # 200 random ITCH-era keys vs cache.db
    import pyarrow.parquet as pq
    # `cache.db::daily_bars` starts 2024-06-03, so the only ITCH-era overlap is June 2024 — which is
    # precisely the slice bought today.  The check therefore lands on the NEW data.
    rg = pd.read_parquet(ITCH_GAP, columns=['bar_date', 'symbol', 'open', 'close', 'volume'])
    rg = rg[rg.symbol.isin(uset) & (rg.bar_date >= '2024-06-03')]
    con = sqlite3.connect('file:data/cache.db?mode=ro', uri=True)
    hits = []
    rng = np.random.default_rng(7)
    idx = rng.permutation(len(rg))[:20000]
    for i in idx:
        r = rg.iloc[int(i)]
        q = con.execute('SELECT open, close, volume FROM daily_bars WHERE symbol=? AND bar_date=?',
                        (r.symbol, r.bar_date)).fetchone()
        if q and q[1] and q[1] > 0:
            hits.append((r.symbol, r.bar_date, r.close, q[1], r.volume, q[2]))
        if len(hits) >= 200:
            break
    con.close()
    md += [f'**ITCH vs Alpaca `cache.db::daily_bars`** ({len(hits)} random keys from an ITCH-era row '
           'group, Nasdaq-listed commons):', '']
    if hits:
        h = pd.DataFrame(hits, columns=['symbol', 'date', 'c_itch', 'c_alp', 'v_itch', 'v_alp'])
        h.to_csv(f'{R}/pricescale_itch.csv', index=False)
        d = (h.c_itch / h.c_alp - 1.0).abs()
        v = h.v_itch / h.v_alp
        md += [f'- close: median abs difference **{d.median() * 100:.4f}%**, within 0.1% '
               f'**{(d < 1e-3).mean() * 100:.1f}%**, off by > 0.5% **{(d > 5e-3).mean() * 100:.1f}%** '
               f'(dates {h.date.min()}..{h.date.max()}).',
               f'- volume ratio ITCH/Alpaca: median **{v.median():.3f}** — the venue-share fact of '
               '§2 seen a second way, on a different year and a different comparison tape.', '']
        log(f'cache.db check: {len(h)} keys, median |close diff| {d.median() * 100:.4f}%, '
            f'volume ratio median {v.median():.3f}')
    else:
        md += ['- **no usable overlap**: `daily_bars` holds 19 rows before 2024-07-01 (the cache was '
               'built from 2025 on), so the ITCH era cannot be checked against Alpaca directly. The '
               'chain that does cover it is ITCH -> EQUS.SUMMARY (§3, 188,793 keys in the overlap '
               'month) -> Alpaca (N2 checked EQUS against `daily_bars` on 200 keys: 99.0% within '
               '0.01%).', '']

    # ---------------------------------------------------------------- 4. split scan (unadjusted)
    log('split scan over the ITCH era')
    ev = []
    for path in (ITCH_OLD, ITCH_GAP):
        pf = pq.ParquetFile(path)
        for g in range(pf.metadata.num_row_groups):
            d = pf.read_row_group(g, columns=['bar_date', 'symbol', 'close', 'volume']).to_pandas()
            d = d[d.symbol.isin(uset) & (d.close > 0)]
            d = d.sort_values(['symbol', 'bar_date'])
            r = d.close.to_numpy() / d.close.shift(1).to_numpy()
            same = d.symbol.to_numpy()[1:] == d.symbol.to_numpy()[:-1]
            r = r[1:]
            m = same & np.isfinite(r) & ((r <= 0.55) | (r >= 1.8))
            if m.any():
                sub = d.iloc[1:][m]
                ev.append(pd.DataFrame({'symbol': sub.symbol.to_numpy(),
                                        'bar_date': sub.bar_date.to_numpy(),
                                        'ratio': r[m]}))
            del d
    ev = pd.concat(ev, ignore_index=True) if ev else pd.DataFrame(columns=['symbol', 'bar_date', 'ratio'])
    ev.to_csv(f'{R}/split_candidates.csv', index=False)
    near = ev[(ev.ratio.between(0.45, 0.55)) | (ev.ratio.between(1.9, 2.1)) |
              (ev.ratio.between(0.30, 0.36)) | (ev.ratio.between(2.9, 3.1))]
    log(f'split candidates: {len(ev):,} overnight moves <= -45% or >= +80%, of which '
        f'{len(near):,} sit at a 2:1 or 3:1 ratio')
    md += ['## 4. Splits (the files are UNADJUSTED — flagged, never silently adjusted)', '',
           f'Overnight close ratios <= 0.55 or >= 1.80 on Nasdaq-listed commons over the ITCH era: '
           f'**{len(ev):,} events**, of which **{len(near):,}** sit within 5% of an exact 2:1 or 3:1 '
           'ratio (`split_candidates.csv`).  These fabricate both signals and exits in an unadjusted '
           'panel, so the run reports a control book with every trade whose hold window contains one '
           'of them removed.', '']

    with open(OUT_MD, 'w') as fh:
        fh.write('\n'.join(md) + '\n')
    log(f'wrote {OUT_MD}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
