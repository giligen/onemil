#!/usr/bin/env python3
"""ORB short mirror — Stage A + Stage B scoring. PREREG §3, §5-§8."""
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from trading.orb_csv import read_orb_csv           # noqa: E402
from trading.orb_asset_class import classify_asset, load_class_map, WRAPPER  # noqa: E402
CMAP = load_class_map()

D = f'{ROOT}/research/orb_short'
F7 = ['gap_pct', 'range_total_volume', 'range_avg_bar_range_pct', 'range_size_pct',
      'price_vs_20d_high_pct', 'prev_day_close_position', 'range_close_position']
RISK, N_SLOT = 375.0, 8
# Cell C (PREREG "Cell C"): chase-tolerant stop entry, fill = next bar's open capped at
# range_low*(1-100bps).  --cell c reads sigC.csv (built by buildC.py); everything else identical.
CELL = 'c' if '--cell' in sys.argv and sys.argv[sys.argv.index('--cell') + 1] == 'c' else 'ab'
SIGF = 'sigC.csv' if CELL == 'c' else 'sig.csv'
TAGN = ('1,273', '1,273') if CELL == 'c' else ('1,271', '1,272')


def clus(x, days):
    """(mean, iid_se, day-clustered se)."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return (x.mean() if n else np.nan), np.nan, np.nan
    mu = x.mean()
    iid = x.std(ddof=1) / np.sqrt(n)
    s = pd.Series(x).groupby(np.asarray(days)).agg(['sum', 'size'])
    v = ((s['sum'] - s['size'] * mu) ** 2).sum() / n ** 2
    return mu, iid, np.sqrt(v)


def cost_table(nb):
    """Minute-of-day median half-spread in bps of mid, from the measured sample."""
    g = nb[(nb.n_q > 0) & nb.mid_med.gt(0)].copy()
    g['hs_bps'] = g.sp_mean / 2.0 / g.mid_med * 1e4
    t = g.groupby('m').hs_bps.median()
    return t, float(g.hs_bps.median())


def attach_cost(df, nb, tbl, glob, mcol, pxcol):
    k = nb.set_index(['day', 'symbol', 'm'])
    idx = list(zip(df.day, df.symbol, df[mcol].astype(int)))
    sp = k.sp_mean.reindex(idx).to_numpy()
    mid = k.mid_med.reindex(idx).to_numpy()
    hs = np.where(np.isfinite(sp) & np.isfinite(mid) & (mid > 0), sp / 2.0, np.nan)
    imp = ~np.isfinite(hs)
    fb = df[mcol].astype(int).map(tbl).fillna(glob).to_numpy() / 1e4 * df[pxcol].to_numpy()
    hs = np.where(imp, fb, hs)
    return hs, imp


def book(df):
    """Net R per trade + $ at $10K/$50K stage sizing."""
    out = df.copy()
    out['netR'] = (out.entry - out['exit'] - out.cost) / out.R
    stop_pct = np.maximum((out.R / out.entry) * 100.0, 1.0)
    for tag, acct, risk in (('10k', 10000.0, RISK), ('50k', 50000.0, RISK * 5)):
        notion = np.minimum(risk / (stop_pct / 100.0), acct / N_SLOT)
        sh = np.floor(notion / out.entry)
        out[f'pnl_{tag}'] = (out.entry - out['exit'] - out.cost) * sh
    return out


def line(tag, d, col='netR'):
    mu, iid, cl_ = clus(d[col], d.day)
    if len(d) == 0:
        return f'{tag:28s} n=0'
    srt = np.sort(d[col].to_numpy())
    k1 = max(1, int(round(0.01 * len(srt))))
    k5 = max(1, int(round(0.05 * len(srt))))
    ex1 = srt[:-k1].mean() if len(srt) > k1 else np.nan
    ex5 = srt[:-k5].mean() if len(srt) > k5 else np.nan
    cap = np.minimum(d[col], 3.0).mean()
    t = mu / cl_ if cl_ and cl_ > 0 else np.nan
    return (f'{tag:28s} n={len(d):4d} netR={mu:+.3f} iid_se={iid:.3f} cl_se={cl_:.3f} '
            f't={t:+.2f} MDE={2.8*cl_:.3f} ex1%={ex1:+.3f} ex5%={ex5:+.3f} cap3R={cap:+.3f} '
            f'$10k={d.pnl_10k.sum():+,.0f} $50k={d.pnl_50k.sum():+,.0f}')


def weeks(d):
    w = d.assign(wk=pd.to_datetime(d.day).dt.to_period('W')).groupby('wk').netR.sum()
    return len(w), float((w > 0).mean())


def green_null(d, reps=500, seed=7):
    """Count-matched null: recentre to zero, bootstrap within the same week counts."""
    rng = np.random.default_rng(seed)
    x = d.netR.to_numpy() - d.netR.mean()
    cnt = d.assign(wk=pd.to_datetime(d.day).dt.to_period('W')).groupby('wk').size().to_numpy()
    out = []
    for _ in range(reps):
        s = rng.choice(x, size=len(x), replace=True)
        pos = 0
        i = 0
        for c in cnt:
            pos += s[i:i + c].sum() > 0
            i += c
        out.append(pos / len(cnt))
    return float(np.mean(out))


def main():
    s = read_orb_csv(f'{D}/{SIGF}')
    print(f'[cell] {CELL} · signals from {SIGF}', flush=True)
    c = read_orb_csv(f'{D}/ctl.csv')
    nb = pd.read_csv(f'{D}/nbbo_short.csv', dtype={'day': str, 'symbol': str},
                     keep_default_na=False, na_values=['']).drop_duplicates(['day', 'symbol', 'm'])
    tbl, glob = cost_table(nb)
    print(f'[cost] measured legs {int((nb.n_q>0).sum()):,}/{len(nb):,} · '
          f'global median half-spread {glob:.1f} bps', flush=True)

    for d in (s, c):
        d['split'] = np.where(d.day < '2026-01-01', 'TRAIN', 'VAL')
        d['half'] = np.where(d.day < '2025-07-01', 'TRAIN_H1',
                             np.where(d.day < '2026-01-01', 'TRAIN_H2', 'VAL'))
        d['wrapper'] = [classify_asset(x, CMAP.get(x)) == WRAPPER for x in d.symbol]

    f = s[s.filled == 1].copy()
    he, ie = attach_cost(f, nb, tbl, glob, 'entry_m', 'entry')
    hx, ix = attach_cost(f, nb, tbl, glob, 'exit_m', 'exit')
    f['cost'] = he + hx
    f['imp'] = (ie | ix)
    f = book(f)
    ce, _ = attach_cost(c, nb, tbl, glob, 'entry_m', 'entry')
    cx, _ = attach_cost(c, nb, tbl, glob, 'exit_m', 'exit')
    c['cost'] = ce + cx
    c = book(c)

    print(f'\n=== STAGE A (cell {TAGN[0]}) — raw short breakdown vs matched control ===')
    for sp in ('TRAIN', 'VAL'):
        print(line(f'A raw {sp}', f[f.split == sp]))
        print(line(f'A control {sp}', c[c.split == sp]))
    for h in ('TRAIN_H1', 'TRAIN_H2'):
        print(line(f'A raw {h}', f[f.half == h]))
    print(line('A raw VAL no-SSR', f[(f.split == 'VAL') & (f.ssr == 0)]))
    print(line('A raw VAL no-wrapper', f[(f.split == 'VAL') & (~f.wrapper)]))
    print(f'\nfill rate: {len(f)}/{len(s)} = {len(f)/len(s)*100:.1f}%  '
          f'| gap-through no-fills {int((s.nofill=="gap_through").sum())}  '
          f'| SSR share (signals) {s.ssr.mean()*100:.1f}%  '
          f'| SSR share (fills) {f.ssr.mean()*100:.1f}%  '
          f'| wrapper share (fills) {f.wrapper.mean()*100:.1f}%  '
          f'| imputed-cost share {f.imp.mean()*100:.1f}%')

    # ---- Stage B: composite, refit on TRAIN ----
    tr = s[s.split == 'TRAIN'].copy()
    trf = f[f.split == 'TRAIN']
    signs, mus, sds = {}, {}, {}
    for col in F7:
        mus[col] = float(tr[col].mean()); sds[col] = float(tr[col].std(ddof=0)) or 1.0
        q = trf[col].quantile([1/3, 2/3]).to_numpy()
        lo = trf[trf[col] <= q[0]].netR.mean()
        hi = trf[trf[col] >= q[1]].netR.mean()
        signs[col] = 1.0 if hi > lo else -1.0
    print('\n[stage B] TRAIN signs:', {k: int(v) for k, v in signs.items()})

    def comp(d):
        z = np.zeros(len(d))
        for col in F7:
            z += signs[col] * ((d[col].to_numpy(float) - mus[col]) / sds[col])
        return z / len(F7)

    s['comp'] = comp(s)
    cut = np.quantile(s.loc[s.split == 'TRAIN', 'comp'], [.2, .4, .6, .8])
    s['q'] = np.digitize(s.comp, cut) + 1
    print(f'[stage B] TRAIN quintile cutoffs: {np.round(cut,6).tolist()}')

    sel_rows = []
    for day, g in s.groupby('day'):
        g = g[g.q > 1]                                    # Q1 filter
        g = g.assign(_k=np.where(g.q == 4, 0, 1)).sort_values(['_k', 'comp'],
                                                              ascending=[True, False])
        sel_rows.append(g.head(N_SLOT))
    sel = pd.concat(sel_rows) if sel_rows else s.head(0)
    v_pdr = sel.prev_day_range_pct <= 11.0
    v_rs = sel.range_size_pct <= 2.221
    rv = sel.return_volatility_20d
    v_g1 = ~((rv >= 7.106) & (sel.prev_day_range_pct >= 9.226)) & rv.notna() & (rv != 0.0)
    sel['veto'] = v_pdr | v_rs | v_g1
    print(f'[stage B] picks {len(sel)} · vetoed {int(sel.veto.sum())} '
          f'(pdr {int(v_pdr.sum())}, range-size {int(v_rs.sum())}, g1 {int(v_g1.sum())}) '
          f'· no-fill picks {int((sel.filled==0).sum())}')

    bsel = sel[(~sel.veto) & (sel.filled == 1)].copy()
    bsel = bsel.merge(f[['day', 'symbol', 'cost', 'imp']], on=['day', 'symbol'], how='left')
    bsel = book(bsel)
    print(f'\n=== STAGE B (cell {TAGN[1]}) — selection ===')
    for sp in ('TRAIN', 'VAL'):
        print(line(f'B {sp}', bsel[bsel.split == sp]))
    for h in ('TRAIN_H1', 'TRAIN_H2'):
        print(line(f'B {h}', bsel[bsel.half == h]))
    bv = bsel[bsel.split == 'VAL']
    print(line('B VAL no-SSR', bv[bv.ssr == 0]))
    print(line('B VAL no-wrapper', bv[~bv.wrapper]))
    nw, gw = weeks(bv)
    print(f'\nVAL: fills {len(bv)} over {nw} weeks = {len(bv)/max(nw,1):.2f}/wk · '
          f'green-week {gw*100:.1f}% vs count-matched null {green_null(bv)*100:.1f}% · '
          f'$10k/wk {bv.pnl_10k.sum()/max(nw,1):+,.0f} · $50k/wk {bv.pnl_50k.sum()/max(nw,1):+,.0f} · '
          f'wrapper {bv.wrapper.mean()*100:.1f}% · SSR {bv.ssr.mean()*100:.1f}% · '
          f'imputed-cost {bv.imp.mean()*100:.1f}%')
    ctlv = c[c.split == 'VAL']
    print(f'B VAL netR − A control VAL netR = '
          f'{bv.netR.mean() - ctlv.netR.mean():+.3f} R')
    sfx = '_cellC' if CELL == 'c' else ''
    bsel.to_csv(f'{D}/book_stage_b{sfx}.csv', index=False)
    f.to_csv(f'{D}/book_stage_a{sfx}.csv', index=False)


if __name__ == '__main__':
    main()
