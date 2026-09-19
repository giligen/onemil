#!/usr/bin/env python3
"""BF frequency frontier — scorer.

Reads each Stage-2 output CSV in runs/, joins R back to the CACHE row (so R is
invariant to risk tiers / regime mult / BP clamp), and emits per-split metrics.

TEST (2026-06-01..2026-08-31) is SEALED: nothing about it is computed or printed
without --reveal-test (see FREEZE.md).

Usage: python3 research/bf_frequency/score.py [--reveal-test CELL_ID ...]
"""
import os
import sys
import json
import math

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
OUT = f'{ROOT}/research/bf_frequency'
RUNS = f'{OUT}/runs'
CACHE = f'{ROOT}/data/bull_flag_cache_causal_full_20260905.csv'

SPLITS = {
    'TRAIN': ('2025-01-01', '2025-12-31', 12),
    'VAL':   ('2026-01-01', '2026-05-31', 5),
    'TEST':  ('2026-06-01', '2026-08-31', 3),
}

argv = sys.argv[1:]
reveal = []
if '--reveal-test' in argv:
    i = argv.index('--reveal-test')
    reveal = argv[i + 1:]
    argv = argv[:i]


def load_cache_R():
    """R per detection, from the cache row. R = pnl / (shares x (entry-stop))."""
    c = pd.read_csv(CACHE, keep_default_na=False, na_values=[''],
                    dtype={'symbol': str})
    for col in ('entry_price', 'stop_loss', 'pnl', 'shares'):
        c[col] = pd.to_numeric(c[col], errors='coerce')
    c['R'] = c['pnl'] / ((c['entry_price'] - c['stop_loss']) * c['shares'])
    c = c[np.isfinite(c['R'])]
    return {(s, d, t): r for s, d, t, r in
            zip(c['symbol'], c['date'], c['entry_time_et'], c['R'])}


RMAP = load_cache_R()


def mdd(series_pnl):
    """Max drawdown of the cumulative equity curve (returned positive-as-loss)."""
    if len(series_pnl) == 0:
        return 0.0
    eq = np.cumsum(series_pnl)
    peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:]
    return float((eq - peak).min())


def extop(r, pct):
    """mean R with the top `pct`% of trades by R removed (at least 1 trade)."""
    if len(r) < 3:
        return float('nan')
    k = max(1, int(round(len(r) * pct / 100.0)))
    return float(np.sort(r)[:-k].mean())


def ex_pnl(p, k):
    """total $ with the k largest trades removed (the consistency headline)."""
    if len(p) <= k:
        return float('nan')
    return float(np.sort(p)[:-k].sum()) if k > 0 else float(p.sum())


def topshare(p, k):
    """share of total net $ contributed by the k largest trades."""
    tot = p.sum()
    if len(p) < k or tot == 0:
        return float('nan')
    return float(np.sort(p)[-k:].sum() / tot * 100)


def market_weeks(lo, hi):
    """Every ISO week that contains at least one trading day in [lo, hi].

    Source: SPY daily bars in cache.db (read-only). A week with NO trade counts
    as a FLAT week — the owner's week-level objective (2026-09-19) is measured
    over the calendar the book could have traded, not over the weeks it did.
    """
    import sqlite3
    con = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True)
    d = pd.read_sql_query(
        "select bar_date from daily_bars where symbol='SPY' and bar_date>=? "
        "and bar_date<=? order by bar_date", con, params=(lo, hi))
    con.close()
    return sorted(set(pd.to_datetime(d['bar_date']).dt.strftime('%G-W%V')))


def longest_loss_streak(p):
    best = cur = 0
    for v in p:
        cur = cur + 1 if v <= 0 else 0
        best = max(best, cur)
    return best


def score(df, split):
    lo, hi, months = SPLITS[split]
    s = df[(df['date'] >= lo) & (df['date'] <= hi)].copy()
    n = len(s)
    if n == 0:
        return dict(wk_total=0, wk_green_pct=float('nan'), wk_flat_pct=float('nan'),
                    wk_red_pct=float('nan'), wk_traded_pct=float('nan'),
                    worst_wk=0.0, red_wk_streak=0,
                    pnl_ex1pct=float('nan'), pnl_ex5pct=float('nan'),
                    pnl_ex5tr=float('nan'), top1_share=float('nan'),
                    top5_share=float('nan'), top10_share=float('nan'),
                    medR=float('nan'), iqrR=float('nan'), p25R=float('nan'),
                    p75R=float('nan'), mo_green_pct=float('nan'),
                    streak_tr=0, streak_wk=0,
                    split=split, n=0, tr_per_mo=0.0, Rpick=float('nan'),
                    totR=0.0, pnl=0.0, wr=float('nan'), mdd=0.0, mdd_R=0.0,
                    worst_mo=0.0, worst_mo_R=0.0, red_mo=0, months_traded=0,
                    ex1=float('nan'), ex5=float('nan'), t=float('nan'),
                    mde80=float('nan'), sd=float('nan'))
    s = s.sort_values(['date', 'exit_time_et'], kind='mergesort')
    r = s['R'].values
    pnl = s['pnl'].values
    s['month'] = s['date'].str[:7]
    mo_pnl = s.groupby('month')['pnl'].sum()
    mo_R = s.groupby('month')['R'].sum()
    sd = float(r.std(ddof=1)) if n > 1 else float('nan')
    se = sd / math.sqrt(n) if n > 1 else float('nan')
    # --- consistency block (owner 2026-09-19: rank on these, not on total $) --
    k1 = max(1, int(round(n * 0.01)))
    k5 = max(1, int(round(n * 0.05)))
    wk = pd.to_datetime(s['date']).dt.strftime('%G-W%V')
    # --- week-level book over EVERY market week (no-trade week = FLAT) -------
    allw = market_weeks(lo, hi)
    wpnl_full = s.groupby(wk)['pnl'].sum().reindex(allw).fillna(0.0)
    nw_all = len(allw)
    wpnl = wpnl_full
    cons = dict(
        wk_total=nw_all,
        wk_green_pct=float((wpnl_full > 0).mean() * 100),
        wk_flat_pct=float((wpnl_full == 0).mean() * 100),
        wk_red_pct=float((wpnl_full < 0).mean() * 100),
        wk_traded_pct=float((s.groupby(wk)['pnl'].size().reindex(allw).fillna(0) > 0).mean() * 100),
        worst_wk=float(wpnl_full.min()),
        red_wk_streak=longest_loss_streak((wpnl_full < 0).astype(float).map({1.0: -1.0, 0.0: 1.0}).values),
        pnl_ex1pct=ex_pnl(pnl, k1), pnl_ex5pct=ex_pnl(pnl, k5),
        pnl_ex5tr=ex_pnl(pnl, 5),
        top1_share=topshare(pnl, 1), top5_share=topshare(pnl, 5),
        top10_share=topshare(pnl, 10),
        medR=float(np.median(r)),
        iqrR=float(np.percentile(r, 75) - np.percentile(r, 25)),
        p25R=float(np.percentile(r, 25)), p75R=float(np.percentile(r, 75)),
        mo_green_pct=float((mo_pnl > 0).mean() * 100),
        streak_tr=longest_loss_streak(pnl),
        streak_wk=longest_loss_streak(wpnl.values),
    )
    return dict(**cons,
        split=split, n=n, tr_per_mo=n / months,
        Rpick=float(r.mean()), totR=float(r.sum()), pnl=float(pnl.sum()),
        wr=float((pnl > 0).mean() * 100),
        mdd=mdd(pnl), mdd_R=mdd(r),
        worst_mo=float(mo_pnl.min()), worst_mo_R=float(mo_R.min()),
        red_mo=int((mo_pnl < 0).sum()), months_traded=int(mo_pnl.size),
        ex1=extop(r, 1), ex5=extop(r, 5),
        t=float(r.mean() / se) if n > 1 and se else float('nan'),
        # MDE80 for a one-sample mean vs 0 at 5%/80%
        mde80=float((1.96 + 0.84) * sd / math.sqrt(n)) if n > 1 else float('nan'),
        sd=sd)


def weeks_green(df, split):
    lo, hi, _ = SPLITS[split]
    s = df[(df['date'] >= lo) & (df['date'] <= hi)].copy()
    if len(s) == 0:
        return float('nan'), 0
    wk = pd.to_datetime(s['date']).dt.strftime('%G-W%V')
    g = s.groupby(wk)['pnl'].sum()
    return float((g > 0).mean() * 100), int(g.size)


def load_run(path):
    d = pd.read_csv(path, keep_default_na=False, na_values=[''],
                    dtype={'symbol': str})
    for col in ('pnl', 'entry_price', 'stop_loss', 'shares'):
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors='coerce')
    d['R'] = [RMAP.get((s, dt, t), float('nan'))
              for s, dt, t in zip(d['symbol'], d['date'], d['entry_time_et'])]
    miss = int(d['R'].isna().sum())
    if miss:
        print(f'  WARNING: {miss}/{len(d)} rows could not be joined to the cache '
              f'for R — they are dropped from R statistics')
        d = d[np.isfinite(d['R'])]
    return d


def main():
    meta = json.load(open(f'{RUNS}/meta.json'))
    rows = []
    for m in meta:
        if not m['csv'] or not os.path.exists(m['csv']):
            continue
        d = load_run(m['csv'])
        for split in ('TRAIN', 'VAL', 'TEST'):
            if split == 'TEST' and m['id'] not in reveal:
                continue                     # SEALED — see FREEZE.md
            sc = score(d, split)
            wg, nw = weeks_green(d, split)
            sc.update(id=m['id'], desc=m['desc'], block=m['block'],
                      full_market=m['full_market'], wk_green=wg, n_weeks=nw,
                      **{f'k_{k}': v for k, v in m['knobs'].items()})
            rows.append(sc)
    g = pd.DataFrame(rows)
    cols = ['id', 'block', 'split', 'n', 'tr_per_mo', 'wk_total', 'wk_green_pct',
            'wk_flat_pct', 'wk_red_pct', 'red_wk_streak', 'worst_wk', 'pnl_ex5tr', 'pnl_ex5pct',
            'pnl_ex1pct', 'mo_green_pct', 'worst_mo', 'mdd', 'pnl', 'top1_share',
            'top5_share', 'top10_share', 'wr', 'medR', 'p25R', 'p75R', 'iqrR',
            'streak_tr', 'streak_wk', 'Rpick', 'totR', 'ex1', 'ex5', 'mdd_R',
            'worst_mo_R', 'red_mo', 'months_traded', 't', 'mde80', 'wk_green',
            'n_weeks', 'full_market', 'desc']
    g = g[cols + [c for c in g.columns if c not in cols]]
    g.to_csv(f'{OUT}/grid.csv', index=False)
    pd.set_option('display.width', 250)
    for split in ('TRAIN', 'VAL', 'TEST'):
        sub = g[g.split == split]
        if len(sub) == 0:
            continue
        print(f'\n===== {split} =====')
        print(sub[['id', 'n', 'tr_per_mo', 'wk_green_pct', 'wk_flat_pct', 'wk_red_pct',
                   'red_wk_streak', 'worst_wk', 'mo_green_pct', 'mdd', 'pnl',
                   'top1_share', 'top5_share', 'top10_share', 'wr', 'pnl_ex5tr',
                   'ex5', 'Rpick', 'wk_total']]
              .to_string(index=False, float_format=lambda x: f'{x:9.2f}'))
    print(f'\nwrote {OUT}/grid.csv')


if __name__ == '__main__':
    main()
