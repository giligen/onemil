#!/usr/bin/env python3
"""Stage E, STEP 0 — verification + the AVAILABILITY AUDIT (PLAN.md §1 standing rule, D1 2026-09-17).

Runs BEFORE any cell is scored.  Writes research/fuckup_audit/E/score_e_availability.md.

What it checks
  1. candidates_causal.csv row count vs the builder log's running total, day count vs the state file,
     header vs B4.COLS + E's extra columns.
  2. survivorship residual: coverage_causal_missing.csv joined to bars_causal_index.csv
     (Alpaca served nothing / ticker rejected  vs  fewer than 10 usable RTH bars).
  3. AVAILABILITY of every column with < 100% coverage on the SCOREABLE population
     (pm_dollar_vol, has_news, prev_day_range_pct, adv20, gap_pct): missingness per split and per
     time band, and whether missingness correlates with the OUTCOME (mean net R of missing vs
     present, per split).  A D1-style structure (missing bucket strongly negative / present bucket
     strongly positive, missing rate varying by band) disqualifies the feature.
  4. pm_dollar_vol's PROVENANCE: whether the key's bars came from the E parquet store (full
     04:00-15:59 session, so pm is a real measurement or a real zero) or from bars_sip.db (whose
     window decides whether "missing pm" means "no premarket trades" or "never fetched").

Read-only everywhere except E/score_e_*.
"""
import os, sys, json, sqlite3, time
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)

E = f'{ROOT}/research/fuckup_audit/E'
SRC = f'{E}/candidates_causal.csv'
RD = dict(keep_default_na=False, na_values=[''])
PM_CUT = 5_816_688.0
OUT = f'{E}/score_e_availability.md'

BANDS = [(570, 575, '09:30-09:35'), (575, 600, '09:35-10:00'), (600, 660, '10:00-11:00'),
         (660, 780, '11:00-13:00'), (780, 842, '13:00-14:01')]


def band_of(m):
    o = pd.Series('other', index=m.index)
    for a, b, nm in BANDS:
        o[(m >= a) & (m < b)] = nm
    return o


def log(s):
    print(f'{time.strftime("%H:%M:%S")} {s}', flush=True)


# ------------------------------------------------------------------ news (union D + E, 100% of E keys)
def load_news():
    fr = []
    for p in (f'{E}/news_presence_e.csv', f'{ROOT}/research/fuckup_audit/D/news_presence.csv'):
        fr.append(pd.read_csv(p, usecols=['day', 'symbol', 'n_prev15_to_0930'],
                              dtype={'day': str, 'symbol': str}, **RD))
    n = pd.concat(fr, ignore_index=True).drop_duplicates(['day', 'symbol'])
    n['has_news'] = (n.n_prev15_to_0930.fillna(0) > 0).astype(int)
    return n[['day', 'symbol', 'has_news']]


# ------------------------------------------------------------------ the scoreable population
BASE = ['day', 'symbol', 'fam', 'cfg', 'sig_m', 'range_so_far_pct', 'spread_cc_bps',
        'pm_dollar_vol', 'prev_day_range_pct', 'adv20', 'gap_pct', 'u1', 'u2', 'split']
FCOLS = ['entry', 'entry_m', 'r_pct', 'rr_2r', 'why_2r', 'exit_m_2r',
         'rr_hold', 'why_hold', 'exit_m_hold']
USE = BASE + [f'{t}_{c}' for t in ('next', 'rest') for c in FCOLS]
ENTRY_MULT = {'next': 0.25, 'rest': 1.00}
EXIT_RATIO = {'stop': 0.875, 'lock': 0.875, 'eod': 0.412, 'target': 0.875, 'none': 0.875}


def load_pop():
    """One chunked pass -> {fill: frame} of rows that can enter SOME Stage-E cell."""
    keep = {'next': [], 'rest': []}
    n_in = 0
    for ch in pd.read_csv(SRC, usecols=USE, dtype={'day': str, 'symbol': str, 'fam': str, 'cfg': str},
                          chunksize=200_000, low_memory=True, **RD):
        n_in += len(ch)
        for t in ('next', 'rest'):
            e, em, rp = ch[f'{t}_entry'], ch[f'{t}_entry_m'], ch[f'{t}_r_pct']
            ok = e.notna() & (e >= 5) & (em <= 841) & (em >= 570) & (rp >= 1.0)
            if not ok.any():
                continue
            x = ch.loc[ok, BASE + [f'{t}_{c}' for c in FCOLS]].copy()
            x.columns = BASE + FCOLS
            keep[t].append(x)
    out = {}
    for t in ('next', 'rest'):
        d = pd.concat(keep[t], ignore_index=True)
        d['key'] = d.fam + ' ' + d.cfg
        d['band'] = band_of(d.entry_m)
        out[t] = d
    return n_in, out


def net_r(d, rr, why, fill):
    half = 0.5 * (d.spread_cc_bps / 100.0) / d.r_pct.clip(lower=0.05)
    return d[rr] - ENTRY_MULT[fill] * half - half * d[why].map(EXIT_RATIO).fillna(0.875)


def main():
    L = ['# Stage E — STEP 0: verification and the availability audit', '',
         f'_generated {time.strftime("%Y-%m-%d %H:%M:%S")} — nothing below this line is a trading result._', '']

    # ---- 1. build verification
    st = json.load(open(f'{E}/build_causal_state.json'))
    ndays = len(st.get('done', st.get('days', [])))
    nrows = sum(1 for _ in open(SRC)) - 1
    hdr = open(SRC).readline().strip().split(',')
    L += ['## 1. Build verification', '',
          f'| item | value |', '|---|---|',
          f'| days in `build_causal_state.json` | {ndays} |',
          f'| data rows in `candidates_causal.csv` | {nrows:,} |',
          f'| columns in header | {len(hdr)} |', '']
    tail = [l for l in open(f'{E}/build_causal2.log').read().split('\n') if ' total ' in l]
    if tail:
        L.append(f'last builder log line: `{tail[-1].strip()}`')
    L.append('')

    # ---- 2. survivorship residual
    idx = pd.read_csv(f'{E}/bars_causal_index.csv', dtype={'day': str, 'symbol': str}, **RD)
    miss = pd.read_csv(f'{E}/coverage_causal_missing.csv', dtype={'bar_date': str, 'symbol': str}, **RD)
    miss = miss.rename(columns={'bar_date': 'day'})
    m = miss.merge(idx, on=['day', 'symbol'], how='left', suffixes=('', '_idx'))
    srccol = 'src' if 'src' in m.columns else ('src_idx' if 'src_idx' in m.columns else None)
    L += ['## 2. Survivorship residual (keys with no usable tape)', '',
          f'`coverage_causal_missing.csv` rows: **{len(miss):,}**  |  `bars_causal_index.csv` rows: {len(idx):,}', '']
    if srccol:
        t = m[srccol].fillna('not-in-index (served from bars_sip.db)').value_counts()
        L += ['| index `src` of the missing key | keys |', '|---|---:|']
        L += [f'| `{k}` | {v:,} |' for k, v in t.items()]
        L.append('')
    if 'n_bars' in idx.columns:
        L.append(f'index keys with `n_bars == 0` (Alpaca served nothing): **{int((idx.n_bars == 0).sum()):,}** '
                 f'of {len(idx):,} = {100*(idx.n_bars == 0).mean():.2f}%')
        L.append('')

    # ---- 3+4. availability on the scoreable population
    n_in, pop = load_pop()
    news = load_news()
    L += ['## 3. The scoreable population', '',
          f'signal rows read: **{n_in:,}**  |  scoreable (entry >= $5, 570 <= entry_m <= 841, r_pct >= 1%): '
          f"next **{len(pop['next']):,}**, rest **{len(pop['rest']):,}**", '']
    for t in ('next', 'rest'):
        d = pop[t]
        L.append(f'### fill `{t}` — rows per family x split')
        L.append(pd.crosstab(d.key, d.split).to_string())
        L.append('')

    d = pop['next'].merge(news, on=['day', 'symbol'], how='left')
    d['news_missing'] = d.has_news.isna().astype(int)
    d['net_hold'] = net_r(d, 'rr_hold', 'why_hold', 'next')

    L += ['## 4. AVAILABILITY AUDIT (PLAN §1 standing rule) — fill `next`, exit hold, per-trade net R', '',
          'Coverage of every column a Stage-E cell reads, on the scoreable population:', '',
          '| column | non-null | coverage |', '|---|---:|---:|']
    for c in ['pm_dollar_vol', 'has_news', 'prev_day_range_pct', 'adv20', 'gap_pct', 'spread_cc_bps',
              'range_so_far_pct']:
        nn = int(d[c].notna().sum())
        L.append(f'| `{c}` | {nn:,} | {100*nn/len(d):.2f}% |')
    L.append('')

    for c in ['pm_dollar_vol', 'has_news', 'prev_day_range_pct', 'adv20', 'gap_pct']:
        miss_rate = d[c].isna().mean()
        if miss_rate <= 0:
            L.append(f'`{c}`: 100% coverage — no missingness table needed.')
            L.append('')
            continue
        d['_m'] = d[c].isna().astype(int)
        L += [f'### `{c}` — missing on {100*miss_rate:.2f}% of scoreable rows', '',
              '**missing rate per split x time band**', '',
              (d.pivot_table(index='band', columns='split', values='_m', aggfunc='mean') * 100
               ).round(1).to_string(), '',
              '**mean net R (hold), missing vs present, per split**', '',
              d.pivot_table(index='_m', columns='split', values='net_hold', aggfunc=['mean', 'size']
                            ).round(4).to_string(), '',
              '**mean net R (hold), missing vs present, per time band (all splits)**', '',
              d.pivot_table(index='band', columns='_m', values='net_hold', aggfunc=['mean', 'size']
                            ).round(4).to_string(), '']

    # pm provenance: was the key in bars_sip.db (not fetched with premarket) or in the E store?
    inE = set(zip(idx.day, idx.symbol))
    d['pm_src_store'] = [1 if k in inE else 0 for k in zip(d.day, d.symbol)]
    d['pm_missing'] = d.pm_dollar_vol.isna().astype(int)
    d['pm_zero'] = ((d.pm_dollar_vol.fillna(-1) == 0)).astype(int)
    L += ['## 5. `pm_dollar_vol` provenance — the D1 question', '',
          'A key whose bars came from the Stage-E parquet store was fetched 04:00-15:59, so a missing/zero '
          'premarket value is a real "no premarket trades". A key served from `bars_sip.db` carries whatever '
          'window that store holds; if that store is RTH-only, "pm missing" would mean "this key was already '
          'in the >=5%-range fetch" — a D1-style availability leak.', '',
          '**pm missing rate by provenance**', '',
          d.pivot_table(index='pm_src_store', columns='split', values='pm_missing', aggfunc=['mean', 'size']
                        ).round(4).to_string(), '',
          '**mean net R (hold) by provenance x pm-missing**', '',
          d.pivot_table(index=['pm_src_store', 'pm_missing'], columns='split', values='net_hold',
                        aggfunc=['mean', 'size']).round(4).to_string(), '']

    # the buckets the 60 cells use
    d['b_news'] = d.has_news.fillna(0).astype(int)
    d['b_pm'] = (d.pm_dollar_vol.fillna(0) > PM_CUT).astype(int)
    d['bucket'] = np.where(d.b_news & d.b_pm, 'combo',
                  np.where(d.b_pm, 'pm_only', np.where(d.b_news, 'news_only', 'neither')))
    L += ['## 6. The bucket shares the 60 cells will restrict to', '',
          '(`pm_dollar_vol` NaN is treated as BELOW the cut — the pre-registered rule needs the leg to be '
          'positively established, exactly as the live ORB gate does.)', '',
          pd.crosstab(d.bucket, d.split, normalize='columns').mul(100).round(2).to_string(), '',
          pd.crosstab(d.bucket, d.split).to_string(), '',
          '**mean net R (hold) per bucket per split, ALL families pooled, per trade (not booked)**', '',
          d.pivot_table(index='bucket', columns='split', values='net_hold', aggfunc=['mean', 'size']
                        ).round(4).to_string(), '']

    open(OUT, 'w').write('\n'.join(L))
    log(f'wrote {OUT}')
    print('\n'.join(L[:40]))


if __name__ == '__main__':
    main()
