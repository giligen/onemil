#!/usr/bin/env python3
"""frames18 F55 — score the 12-cell (k, w) grid of the passive mirror short.

PREREG frames18/PREREG.md (frozen; FREEZE.md carries the git hash).  Entry charged ZERO, exit
charged the measured NBBO half-spread.  Reg SHO 201 rail applied per PREREG §3a.  TEST sealed.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/frames15')
from common15 import clust_t1, null_green, week_shape                 # noqa: E402
from scoreB_intra import book                                         # noqa: E402

D = f'{ROOT}/research/mature_method/frames18'
D16 = f'{ROOT}/research/mature_method/frames16'
D17 = f'{ROOT}/research/mature_method/frames17'
TEST_FROM = '2026-06-01'
KS = (0.004, 0.006, 0.008, 0.010)
WS = (3, 5, 10)


def load_quotes():
    parts = []
    for p in (f'{D16}/nbbo16.csv', f'{D17}/nbbo17.csv', f'{D}/nbbo18.csv'):
        q = pd.read_csv(p, dtype={'day': str, 'symbol': str})
        parts.append(q[['day', 'symbol', 'm', 'sp_med', 'mid_med', 'n_q']])
    q = pd.concat(parts, ignore_index=True)
    q = q[(q.n_q > 0) & np.isfinite(q.sp_med)].drop_duplicates(['day', 'symbol', 'm'])
    sp = {(a, b, int(c)): float(v) for a, b, c, v in zip(q.day, q.symbol, q.m, q.sp_med)}
    # pre-committed NBB proxy (PREREG 3a): mid_med - sp_med/2
    nbb = {(a, b, int(c)): float(mm) - float(s) / 2.0
           for a, b, c, mm, s in zip(q.day, q.symbol, q.m, q.mid_med, q.sp_med)
           if np.isfinite(mm)}
    return sp, nbb


def iid_t(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan, np.nan
    se = x.std(ddof=1) / np.sqrt(len(x))
    return se, (x.mean() / se if se > 0 else np.nan)


def main():
    d = pd.read_csv(f'{D}/grid18.csv', dtype={'day': str, 'symbol': str})
    assert d.day.max() < TEST_FROM, f'TEST leaked: max day {d.day.max()}'
    print(f'[freeze] max day scored {d.day.max()} < TEST_FROM {TEST_FROM}', flush=True)
    sp, nbb = load_quotes()

    d['sp_exit'] = [sp.get((a, b, int(c)), np.nan) if c >= 0 else np.nan
                    for a, b, c in zip(d.day, d.symbol, d.exit_m)]
    d['nbb_fill'] = [nbb.get((a, b, int(c)), np.nan) if c >= 0 else np.nan
                     for a, b, c in zip(d.day, d.symbol, d.fill_m)]
    # --- reacting counterfactual, net of ITS OWN measured entry+exit half-spreads (score17 contract)
    d['sp_re'] = [sp.get((a, b, int(c)), np.nan) for a, b, c in zip(d.day, d.symbol, d.react_entry_m)]
    d['sp_rx'] = [sp.get((a, b, int(c)), np.nan) for a, b, c in zip(d.day, d.symbol, d.react_exit_m)]
    d['react_net'] = d.rr_reacting - (0.5 * d.sp_re / d.react_entry
                                      + 0.5 * d.sp_rx / d.react_entry) / d.react_rpct

    # --- Reg SHO 201 rail (PREREG 3a): an SSR-active fill is valid only if limit > NBB at that
    # minute; NBB unavailable -> VOID.  SSR undetermined (no prior daily bar) -> VOID (conservative;
    # recorded in the REPORT as a mid-run specification of an item the PREREG left as "counted").
    touched = d.touch_off >= 0
    d['ssr_void'] = False
    m = touched & d.ssr_active.astype(bool)
    d.loc[m & ~np.isfinite(d.nbb_fill), 'ssr_void'] = True
    d.loc[m & np.isfinite(d.nbb_fill) & (d.limit <= d.nbb_fill), 'ssr_void'] = True
    d.loc[touched & d.ssr_undet.astype(bool), 'ssr_void'] = True

    rows = []
    for k in KS:
        for w in WS:
            z = d[np.isclose(d.k, k)].copy()
            z['filled'] = (z.touch_off >= 0) & (z.touch_off < w) & ~z.ssr_void
            n_all = len(z)
            f = z[z.filled].copy()
            u = z[~z.filled].copy()
            cov = float(np.isfinite(f.sp_exit).mean()) if len(f) else np.nan
            cost = (0.5 * f.sp_exit / f.entry) / f.rpct
            f['net'] = f.rr - cost.fillna(cost.mean())
            raw_touch = int(((z.touch_off >= 0) & (z.touch_off < w)).sum())
            voided = int(((z.touch_off >= 0) & (z.touch_off < w) & z.ssr_void).sum())
            ssr_act = int(((z.touch_off >= 0) & (z.touch_off < w) & z.ssr_active).sum())
            st = dict(k=k, w=w, n_signal=n_all, n_touch=raw_touch, n_filled=len(f),
                      fill_rate=len(f) / n_all, exit_cov=cov,
                      ssr_active_n=ssr_act, ssr_void_n=voided,
                      ssr_void_share=voided / raw_touch if raw_touch else np.nan,
                      ssr_undet_n=int((z.touch_off >= 0).mul(z.ssr_undet.astype(bool)).sum()))
            for spl in ('TRAIN', 'VAL'):
                fx, ux = f[f.split == spl], u[u.split == spl]
                if len(fx) < 10:
                    st[f'{spl}_n'] = len(fx)
                    continue
                mg, tg = clust_t1(fx.rr.values, fx.day.values)
                mn, tn = clust_t1(fx.net.values, fx.day.values)
                se_i, t_i = iid_t(fx.net.values)
                un, ut = clust_t1(ux.react_net.dropna().values,
                                  ux.day.values[np.isfinite(ux.react_net.values)])
                bk = book(fx.assign(entry_m=fx.fill_m.astype(int),
                                    exit_m=fx.exit_m.astype(int)), 'exit_m')
                wk = week_shape(bk.assign(rr=bk.net), spl)
                nl = null_green(bk.assign(rr=bk.net), spl)
                q95 = fx.rr.quantile(0.95)
                ex5 = float(fx.net[fx.rr <= q95].mean())
                se_c = mn / tn if tn not in (0, np.nan) and np.isfinite(tn) and tn != 0 else np.nan
                st.update({f'{spl}_n': len(fx), f'{spl}_gross': mg, f'{spl}_gross_t': tg,
                           f'{spl}_net': mn, f'{spl}_net_t': tn, f'{spl}_net_se_clust': se_c,
                           f'{spl}_net_se_iid': se_i, f'{spl}_net_t_iid': t_i,
                           f'{spl}_ex5': ex5, f'{spl}_unf_n': len(ux), f'{spl}_unf_net': un,
                           f'{spl}_margin': mn - un, f'{spl}_green': wk['green'],
                           f'{spl}_null50': nl[1], f'{spl}_wk': wk['wk_mean']})
            tr = f[f.split == 'TRAIN']
            st['h1'] = float(tr[tr.day < '2025-07-01'].net.mean()) if len(tr) else np.nan
            st['h2'] = float(tr[tr.day >= '2025-07-01'].net.mean()) if len(tr) else np.nan
            # pre-committed pass bar
            ok = all(np.isfinite(st.get(f'{s}_net', np.nan)) for s in ('TRAIN', 'VAL'))
            st['pass'] = bool(ok and st['TRAIN_net'] >= 0.10 and st['VAL_net'] >= 0.10
                              and st['TRAIN_net_t'] >= 2.0 and st['VAL_net_t'] >= 2.0
                              and st['TRAIN_ex5'] >= 0 and st['VAL_ex5'] >= 0
                              and st['TRAIN_margin'] >= 0.02 and st['VAL_margin'] >= 0.02
                              and np.sign(st['h1']) == np.sign(st['h2'])
                              and st['TRAIN_green'] > st['TRAIN_null50']
                              and st['VAL_green'] > st['VAL_null50'])
            rows.append(st)
            print(f"k={k*100:.1f}% w={w:2d} | fill {st['fill_rate']*100:5.1f}% "
                  f"n={st['n_filled']:4d} ssrvoid {voided:3d} | "
                  f"TR net {st.get('TRAIN_net', float('nan')):+.3f} "
                  f"(tc {st.get('TRAIN_net_t', float('nan')):+.2f}, ti "
                  f"{st.get('TRAIN_net_t_iid', float('nan')):+.2f}) ex5 "
                  f"{st.get('TRAIN_ex5', float('nan')):+.3f} marg "
                  f"{st.get('TRAIN_margin', float('nan')):+.3f} | VA net "
                  f"{st.get('VAL_net', float('nan')):+.3f} "
                  f"(tc {st.get('VAL_net_t', float('nan')):+.2f}, ti "
                  f"{st.get('VAL_net_t_iid', float('nan')):+.2f}) ex5 "
                  f"{st.get('VAL_ex5', float('nan')):+.3f} marg "
                  f"{st.get('VAL_margin', float('nan')):+.3f} | "
                  f"{'PASS' if st['pass'] else 'fail'}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(f'{D}/cells18.csv', index=False)

    # --- neighbourhood rule (PREREG 5)
    sign = {(r.k, r.w): (np.sign(r.TRAIN_net), np.sign(r.VAL_net)) for r in out.itertuples()}
    print('\n== sign map (TRAIN/VAL net) ==', flush=True)
    for k in KS:
        print('  k=%.1f%% ' % (k * 100) + '  '.join(
            f'w{w}:{"+" if sign[(k, w)][0] > 0 else "-"}{"+" if sign[(k, w)][1] > 0 else "-"}'
            for w in WS), flush=True)
    for r in out[out['pass']].itertuples():
        nb = []
        ki = KS.index(r.k)
        wi = WS.index(r.w)
        for dk, dw in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            a, b = ki + dk, wi + dw
            if 0 <= a < len(KS) and 0 <= b < len(WS):
                nb.append(((KS[a], WS[b]), sign[(KS[a], WS[b])]))
        same = all(s == (1.0, 1.0) for _, s in nb)
        print(f"\n[neighbourhood] PASS cell k={r.k*100:.1f}% w={r.w}: neighbours "
              f"{[(f'{a*100:.1f}/{b}', ('+' if s[0] > 0 else '-') + ('+' if s[1] > 0 else '-')) for (a, b), s in nb]}"
              f" -> {'SMOOTH' if same else 'ISOLATED CELL'}", flush=True)
    if not out['pass'].any():
        print('\n[neighbourhood] no cell passes the pre-committed bar — rule moot; sign map is the '
              'deliverable.', flush=True)
    print(f'\n[score18] wrote cells18.csv ({len(out)} cells)', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
