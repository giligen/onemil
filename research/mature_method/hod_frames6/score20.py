#!/usr/bin/env python3
"""hod_frames6 / F20 — THE PLACEBO.  Cells exactly as declared in PREREG.md.

Reads the walk artifacts and produces, per split:
  * the break book's own mean gross R
  * the 200-draw control distribution of the mean gross R for arms a, b, c
  * the break's PERCENTILE in each
  * the PAIRED difference with a day-clustered SE and t
  * the exit-reason / win-rate mix of every arm (F20-e)
  * the universe bound (F20-d)
and applies the PRE-COMMITTED reading rule to arm a.
"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common6 import D6, SPLITS                                              # noqa: E402

DRAWS = 200
SEED = 20260920
KEY = ['day', 'symbol', 'entry_m']
RD = dict(dtype={'day': str, 'symbol': str, 'ctrl': str, 'why': str},
          keep_default_na=False, na_values=[''])


def clustered_paired_t(d, col):
    """Cluster-robust t of the mean of `col`, clusters = trading DAYS."""
    if len(d) < 3:
        return np.nan
    x = d[col].values.astype(float); mu = x.mean()
    g = pd.Series(x - mu).groupby(d.day.values).sum().values
    se = np.sqrt((g ** 2).sum()) / len(x)
    return float(mu / se) if se > 0 else np.nan


def draw_dist(g, bk, rng, draws=DRAWS):
    """`g` = long frame keyed to booked trades, many controls per trade.  Each draw takes ONE
    control per booked trade (uniformly) and returns that control book's mean gross R."""
    order = g.sort_values(KEY, kind='mergesort')
    kidx = order.groupby(KEY, sort=True).ngroup().values
    rr = order.rr.values.astype(float)
    starts = np.searchsorted(kidx, np.arange(kidx.max() + 1))
    counts = np.bincount(kidx)
    out = np.empty(draws)
    for i in range(draws):
        pick = starts + (rng.random(len(counts)) * counts).astype(int)
        out[i] = rr[pick].mean()
    return out


def boot_dist(g, rng, draws=DRAWS):
    """Day-clustered bootstrap of a DETERMINISTIC control book's mean gross R (arm c)."""
    days = g.day.unique()
    by = {d: v.rr.values.astype(float) for d, v in g.groupby('day')}
    out = np.empty(draws)
    for i in range(draws):
        pick = rng.choice(days, size=len(days), replace=True)
        out[i] = np.concatenate([by[d] for d in pick]).mean()
    return out


def pct_of(obs, dist):
    return float((dist < obs).mean() * 100)


def main():
    bk = pd.read_csv(f'{D6}/book6.csv', **RD)
    par = pd.read_csv(f'{D6}/parity6.csv', **RD)
    par['d'] = (par.rr_booked - par.rr_walk).abs()
    print('== F20 · 0. R3 independent-rebuild rail ==', flush=True)
    print(f'  placebo walker re-priced {len(par)} of {len(bk)} booked trades from their own bar and '
          f'own stop; max |Δrr| = {par.d.max():.3e}', flush=True)
    assert par.d.max() < 1e-9, 'R3 FAIL — the placebo walker does not reproduce the booked trade'
    print('  R3: ASSERTED (the run aborts otherwise)\n', flush=True)

    sp_of = dict(zip(zip(bk.day, bk.symbol, bk.entry_m), bk.split))
    arms = {}
    for k, f in (('a', 'pa6.csv'), ('b', 'pb6.csv'), ('c', 'pc6.csv'), ('d', 'pd6.csv')):
        g = pd.read_csv(f'{D6}/{f}', **RD)
        g['split'] = [sp_of.get(t, '') for t in zip(g.day, g.symbol, g.entry_m)]
        arms[k] = g[g.split.isin(SPLITS)]

    # ---------------------------------------------------------------- coverage (availability rail)
    print('== F20 · 1. coverage (availability rail) ==', flush=True)
    print('| arm | split | booked | matched | coverage | controls/trade (median) |', flush=True)
    print('|---|---|---|---|---|---|', flush=True)
    cov = {}
    for k in ('a', 'b', 'c', 'd'):
        for sp in SPLITS:
            nb = int((bk.split == sp).sum())
            g = arms[k][arms[k].split == sp]
            nm = g.drop_duplicates(KEY).shape[0]
            per = g.groupby(KEY).size().median() if len(g) else 0
            cov[(k, sp)] = nm / nb if nb else np.nan
            print(f'| {k} | {sp} | {nb} | {nm} | {nm / nb * 100:.1f} % | {per:.0f} |', flush=True)
    print('', flush=True)

    rows = []
    rng = np.random.default_rng(SEED)
    print('== F20 · 2. the placebo cells ==', flush=True)
    print('| cell | split | break grossR | control mean | ctrl p5 | ctrl p95 | break pctile | '
          'paired Δ | iid t | day-clust t | n paired |', flush=True)
    print('|---|---|---|---|---|---|---|---|---|---|---|', flush=True)
    for k, name in (('a', 'F20-a same symbol-day, non-break minute'),
                    ('b', 'F20-b matched non-signal symbol, same minute'),
                    ('c', 'F20-c same symbol, 15 min earlier')):
        for sp in SPLITS:
            g = arms[k][arms[k].split == sp]
            bsel = bk[bk.split == sp]
            have = set(map(tuple, g[KEY].values.tolist()))
            bm = bsel[[tuple(t) in have for t in bsel[KEY].values.tolist()]]
            obs = float(bm.rr.mean())
            dist = boot_dist(g, rng) if k == 'c' else draw_dist(g, bm, rng)
            # paired difference: control mean per booked trade vs the break's own rr
            cm = g.groupby(KEY, sort=True).rr.mean().rename('ctrl').reset_index()
            pr = bm.merge(cm, on=KEY, how='inner')
            pr['diff'] = pr.rr - pr.ctrl
            t_iid = float(pr['diff'].mean() / (pr['diff'].std(ddof=1) / np.sqrt(len(pr))))
            t_cl = clustered_paired_t(pr, 'diff')
            p = pct_of(obs, dist)
            print(f'| {name} | {sp} | {obs:+.4f} | {dist.mean():+.4f} | '
                  f'{np.percentile(dist, 5):+.4f} | {np.percentile(dist, 95):+.4f} | {p:.1f} | '
                  f'{pr["diff"].mean():+.4f} | {t_iid:+.2f} | {t_cl:+.2f} | {len(pr)} |', flush=True)
            rows.append(dict(cell=f'F20-{k}', split=sp, n=len(pr), break_gross=obs,
                             ctrl_mean=float(dist.mean()), ctrl_p5=float(np.percentile(dist, 5)),
                             ctrl_p95=float(np.percentile(dist, 95)), pctile=p,
                             paired=float(pr['diff'].mean()), t_iid=t_iid, t_clust=t_cl,
                             coverage=cov[(k, sp)]))
    print('', flush=True)

    # ------------------------------------------------------------------ TRAIN halves on arm a
    print('== F20 · 3. both TRAIN halves (arm a, the reading arm) ==', flush=True)
    print('| half | break grossR | control mean | break pctile | paired Δ | day-clust t | n |',
          flush=True)
    print('|---|---|---|---|---|---|---|', flush=True)
    for lab, msk in (('H1-2025', lambda d: d.day < '2025-07-01'),
                     ('H2-2025', lambda d: (d.day >= '2025-07-01') & (d.day < '2026-01-01')),
                     ('VAL', lambda d: d.day >= '2026-01-01')):
        g = arms['a'][msk(arms['a'])]
        bm = bk[msk(bk)]
        have = set(map(tuple, g[KEY].values.tolist()))
        bm = bm[[tuple(t) in have for t in bm[KEY].values.tolist()]]
        if not len(bm):
            continue
        dist = draw_dist(g, bm, rng)
        cm = g.groupby(KEY, sort=True).rr.mean().rename('ctrl').reset_index()
        pr = bm.merge(cm, on=KEY, how='inner'); pr['diff'] = pr.rr - pr.ctrl
        print(f'| {lab} | {bm.rr.mean():+.4f} | {dist.mean():+.4f} | {pct_of(bm.rr.mean(), dist):.1f} '
              f'| {pr["diff"].mean():+.4f} | {clustered_paired_t(pr, "diff"):+.2f} | {len(pr)} |',
              flush=True)
        rows.append(dict(cell='F20-a-half', split=lab, n=len(pr), break_gross=float(bm.rr.mean()),
                         ctrl_mean=float(dist.mean()), pctile=pct_of(bm.rr.mean(), dist),
                         paired=float(pr['diff'].mean()), t_clust=clustered_paired_t(pr, 'diff')))
    print('', flush=True)

    # ------------------------------------------------------------------ F20-e mix, F20-d bound
    print('== F20 · 4. F20-e  exit mix and win rate ==', flush=True)
    print('| arm | split | n | mean grossR | WR (R>0) | stop % | target % | eod % |', flush=True)
    print('|---|---|---|---|---|---|---|---|', flush=True)
    for sp in SPLITS:
        d = bk[bk.split == sp]
        wm = d.why.value_counts(normalize=True)
        print(f'| BREAK | {sp} | {len(d)} | {d.rr.mean():+.4f} | {(d.rr > 0).mean() * 100:.1f} % | '
              f'{wm.get("stop", 0) * 100:.1f} | {wm.get("target", 0) * 100:.1f} | '
              f'{wm.get("eod", 0) * 100:.1f} |', flush=True)
        for k in ('a', 'b', 'c', 'd'):
            g = arms[k][arms[k].split == sp]
            wm = g.why.value_counts(normalize=True)
            print(f'| {k} | {sp} | {len(g)} | {g.rr.mean():+.4f} | {(g.rr > 0).mean() * 100:.1f} % | '
                  f'{wm.get("stop", 0) * 100:.1f} | {wm.get("target", 0) * 100:.1f} | '
                  f'{wm.get("eod", 0) * 100:.1f} |', flush=True)
    print('', flush=True)
    print('== F20 · 5. F20-d  the universe bound (no break in symbol OR minute) ==', flush=True)
    for sp in SPLITS:
        g = arms['d'][arms['d'].split == sp]
        se = g.rr.std(ddof=1) / np.sqrt(g.day.nunique())
        print(f'  {sp}: n {len(g)} over {g.day.nunique()} sessions, mean gross R '
              f'{g.rr.mean():+.4f} (day-clustered SE {se:.4f})', flush=True)
    print('', flush=True)

    # ------------------------------------------------------------------ the reading rule
    ra = {r['split']: r for r in rows if r['cell'] == 'F20-a'}
    rb = {r['split']: r for r in rows if r['cell'] == 'F20-b'}
    rc = {r['split']: r for r in rows if r['cell'] == 'F20-c'}
    better = all(ra[s]['pctile'] >= 95 for s in SPLITS) and any(ra[s]['t_clust'] > 2 for s in SPLITS)
    worse = all(ra[s]['pctile'] <= 5 for s in SPLITS) and any(ra[s]['t_clust'] < -2 for s in SPLITS)
    read = 'BETTER' if better else ('WORSE' if worse else 'NO DIFFERENT')
    agree = []
    for lab, r in (('b', rb), ('c', rc)):
        sgn = [np.sign(r[s]['paired']) for s in SPLITS]
        agree.append(f'{lab}: paired Δ {r["TRAIN"]["paired"]:+.4f} / {r["VAL"]["paired"]:+.4f} '
                     f'(pctile {r["TRAIN"]["pctile"]:.0f} / {r["VAL"]["pctile"]:.0f})')
    print('== F20 · 6. THE PRE-COMMITTED READING ==', flush=True)
    print(f'  arm a percentiles: TRAIN {ra["TRAIN"]["pctile"]:.1f} · VAL {ra["VAL"]["pctile"]:.1f}',
          flush=True)
    print(f'  arm a day-clustered paired t: TRAIN {ra["TRAIN"]["t_clust"]:+.2f} · '
          f'VAL {ra["VAL"]["t_clust"]:+.2f}', flush=True)
    for a in agree:
        print(f'  corroboration {a}', flush=True)
    print(f'\n  >>> READING: **{read}** <<<\n', flush=True)
    pd.DataFrame(rows).to_csv(f'{D6}/cells20.csv', index=False)
    print(f'cells20.csv written ({len(rows)} rows)', flush=True)


if __name__ == '__main__':
    main()
