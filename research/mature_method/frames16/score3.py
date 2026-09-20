#!/usr/bin/env python3
"""frames16 ARM 3 (F49) — score the discovered-mirror SHORT.

PREREG §5.6/§5.7. Every rail is applied here so it can be switched without re-walking:
  borrow  — Alpaca `shortable AND easy_to_borrow`; a symbol ABSENT from the list is NOT shortable
  SSR     — Reg SHO 201: session low <= prior close x 0.90 (same day, up to the entry bar) OR the
            prior session closed <= -10 %; under SSR the sell limit fills only on an UPTICK
            (entry bar's open > the previous bar's close), else NO FILL / 0 P&L
  cap     — the sell limit at ref x (1 - 0.6 %); below it, no fill, and the unfilled counterfactual
            is measured (the halt-resume adverse-selection check)
  cost    — the frames14 F45 MEASURED minute-of-day NBBO median, half at each leg, in % of price,
            converted to R at the cell's OWN r_pct (the short's R is smaller than the long's, so the
            same spread costs more per R — F2's finding, applied)
  universe— price >= $5, leveraged wrappers out, test tickers out

Reproduction gate on `frames15` B12 runs first and RAISES.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/frames15')
from common15 import (D as D15, RISK, attach_instrument, clust_t1, mde_pct,   # noqa: E402
                      null_green, week_shape)
from scoreB_intra import book, cost_at, cost_table                            # noqa: E402

D = f'{ROOT}/research/mature_method/frames16'
BORROW = f'{ROOT}/research/fuckup_audit/O_halt/PASSIVE/borrow_flags.csv'
TEST_FROM = '2026-06-01'

CELLS = [('S1', 'MIR2  |hour ret|>2%   stop 2%      bare',    'f_mir2', 'a', 'bare'),
         ('S2', 'MIR2  |hour ret|>2%   stop 2%      +2R',     'f_mir2', 'a', 'tgt'),
         ('S3', 'UP2   hour ret>+2%    stop 2%      bare',    'f_up2',  'a', 'bare'),
         ('S4', 'UP2   hour ret>+2%    stop 2%      +2R',     'f_up2',  'a', 'tgt'),
         ('S5', 'MIR2  |hour ret|>2%   stop hr-high bare',    'f_mir2', 'b', 'bare'),
         ('S6', 'MIR2  |hour ret|>2%   stop hr-high +2R',     'f_mir2', 'b', 'tgt'),
         ('P1', 'PLACEBO volume, no price (|ret|<=1%) bare',  'f_abs',  'a', 'bare'),
         ('P2', 'PLACEBO random other hour, same name-days',  'f_rand', 'a', 'bare')]


# ------------------------------------------------------------------ the reproduction gate
def repro():
    fs = sorted(glob.glob(f'{D15}/intra_*.csv'))
    d = pd.concat([pd.read_csv(f, dtype={'symbol': str, 'day': str}) for f in fs],
                  ignore_index=True)
    d = d[d.day < TEST_FROM]
    assert len(d) == 42224, f'frames15 intraday population is {len(d)}, expected 42,224'
    g5 = float(d.gate5.mean())
    assert abs(g5 - 0.2693) < 5e-4, f'gate5 share {g5:.4f}, expected 0.2693'
    g = d[d.gate5.astype(bool) & d.f_mir.astype(bool)]
    out = {}
    for sp, lo, hi, n_e, r_e, t_e in (('TRAIN', '2025-01-01', '2026-01-01', 2238, -0.398, -13.33),
                                      ('VAL', '2026-01-01', TEST_FROM, 1363, -0.323, -10.51)):
        z = g[(g.day >= lo) & (g.day < hi)]
        mu, t = clust_t1(z.rr_bare.values, z.day.values)
        assert len(z) == n_e, f'B12 {sp} n={len(z)} expected {n_e}'
        assert abs(mu - r_e) < 5e-4 and abs(t - t_e) < 0.02, f'B12 {sp} {mu:.4f}/{t:.2f}'
        out[sp] = (len(z), mu, t)
        print(f'[REPRO] frames15 B12 {sp}: n={len(z):,} gross {mu:+.3f} R (t {t:+.2f})  MATCH',
              flush=True)
    return out


# ------------------------------------------------------------------ the rails
def load():
    fs = sorted(glob.glob(f'{D}/sw_*.csv'))
    d = pd.concat([pd.read_csv(f, dtype={'symbol': str, 'day': str},
                               keep_default_na=False, na_values=['']) for f in fs],
                  ignore_index=True)
    d = d[d.day < TEST_FROM].copy()
    d['split'] = np.where(d.day < '2026-01-01', 'TRAIN', 'VAL')
    return d


def add_borrow(d):
    b = pd.read_csv(BORROW, dtype={'symbol': str}, keep_default_na=False, na_values=[''])
    b['etb'] = b.shortable.astype(str).str.lower().eq('true') & \
        b.easy_to_borrow.astype(str).str.lower().eq('true')
    m = dict(zip(b.symbol, b.etb))
    d['etb'] = d.symbol.map(m).fillna(False).astype(bool)   # ABSENT => NOT shortable
    d['in_list'] = d.symbol.isin(set(b.symbol))
    return d


def add_ssr(d):
    import pyarrow.parquet as pq
    syms = list(dict.fromkeys(d.symbol.astype(str)))
    parts = []
    for y in ('2024', '2025', '2026'):
        p = f'{D15}/daily15_{y}.parquet'
        if not os.path.exists(p):
            continue
        t = pq.read_table(p, columns=['symbol', 'date', 'p_close', 'ret1'],
                          filters=[('symbol', 'in', syms)]).to_pandas()
        parts.append(t)
    dd = pd.concat(parts, ignore_index=True)
    dd['symbol'] = dd.symbol.astype(str); dd['date'] = dd.date.astype(str)
    dd = dd.sort_values(['symbol', 'date'], kind='mergesort')
    dd['prev_ret1'] = dd.groupby('symbol').ret1.shift(1)
    d = d.merge(dd[['symbol', 'date', 'p_close', 'prev_ret1']].rename(columns={'date': 'day'}),
                on=['symbol', 'day'], how='left')
    d['ssr_today'] = np.where(d.p_close.notna(),
                              d.low_to_entry <= d.p_close * 0.90, False)
    d['ssr_prev'] = np.where(d.prev_ret1.notna(), d.prev_ret1 <= -0.10, False)
    d['ssr'] = d.ssr_today | d.ssr_prev
    d['ssr_prev7'] = np.where(d.prev_ret1.notna(), d.prev_ret1 <= -0.07, False)   # the upper bound
    d['ssr_wide'] = d.ssr_today | d.ssr_prev7
    d['fill_ssr'] = (~d.ssr) | d.uptick.astype(bool)
    d['fill'] = d.filled_cap.astype(bool) & d.fill_ssr
    return d


def universe(d):
    n0 = len(d)
    d = d[d.gate5.astype(bool)]
    d = d[d.price >= 5.0]
    d = d[d.asset_class != 'wrapper']
    print(f'[universe] {n0:,} walked -> gate5 -> price>=$5 -> ex-wrapper = {len(d):,}', flush=True)
    return d


# ------------------------------------------------------------------ scoring
def score(tag, name, z, spec, mode, ct, rows, etb_only=True):
    z = z.copy()
    z['rr'] = z[f'rr_{spec}_{mode}']
    z['rpct'] = z[f'rpct_{spec}']
    z['exit_m'] = z[f'exitm_{spec}_{mode}']
    n_all = len(z)
    sh_unshort = float(1.0 - z.etb.mean())
    sh_ssr = float(z.ssr.mean())
    sh_ssrblock = float((z.ssr & ~z.uptick.astype(bool)).mean())
    sh_capskip = float(1.0 - z.filled_cap.astype(bool).mean())
    cf = float(z.loc[~z.filled_cap.astype(bool), 'rr'].mean()) if (~z.filled_cap.astype(bool)).any() else np.nan
    cf_f = float(z.loc[z.filled_cap.astype(bool), 'rr'].mean())
    if etb_only:
        z = z[z.etb]
    z = z[z.fill]
    cpct = 0.5 * cost_at(ct, z.entry_m.values) + 0.5 * cost_at(ct, z.exit_m.values)
    z['cost_R'] = cpct / z.rpct.values
    z['net'] = z.rr - z.cost_R
    st = dict(cell=tag, name=name, n_signal=n_all, n=len(z), unshortable=sh_unshort,
              ssr=sh_ssr, ssr_blocked=sh_ssrblock, cap_skipped=sh_capskip,
              cf_unfilled=cf, cf_filled=cf_f)
    for sp in ('TRAIN', 'VAL'):
        x = z[z.split == sp]
        if len(x) < 10:
            continue
        mu, tc = clust_t1(x.rr.values, x.day.values)
        ti = float(mu / (x.rr.std(ddof=1) / np.sqrt(len(x))))
        mun, tcn = clust_t1(x.net.values, x.day.values)
        tin = float(mun / (x.net.std(ddof=1) / np.sqrt(len(x))))
        bk = book(x.assign(entry_m=x.entry_m, exit_m=x.exit_m), 'exit_m')
        w = week_shape(bk.assign(rr=bk.net), sp)
        nl = null_green(bk.assign(rr=bk.net), sp)
        q95 = x.rr.quantile(0.95)
        st.update({f'{sp}_n': len(x), f'{sp}_g': mu, f'{sp}_gpct': float((x.rr * x.rpct).mean() * 100),
                   f'{sp}_ti': ti, f'{sp}_tc': tc, f'{sp}_cost': float(x.cost_R.mean()),
                   f'{sp}_net': mun, f'{sp}_tin': tin, f'{sp}_tcn': tcn,
                   f'{sp}_mde': mde_pct(x.rr.values, x.day.values),
                   f'{sp}_ex5': float(x.rr[x.rr <= q95].mean()),
                   f'{sp}_ex5net': float(x.net[x.rr <= q95].mean()),
                   f'{sp}_wk': w['per_wk'], f'{sp}_bk': w['n'], f'{sp}_green': w['green'],
                   f'{sp}_null95': nl[2], f'{sp}_null50': nl[1], f'{sp}_total': w['total'],
                   f'{sp}_wkmean': w['wk_mean'], f'{sp}_worst': w['worst'],
                   f'{sp}_streak': w['redstreak'],
                   # how many times the F45 measured spread this cell can absorb before it is flat
                   f'{sp}_bekx': float(mu / x.cost_R.mean()) if x.cost_R.mean() > 0 else np.nan})
        print(f'  {tag} {sp:5s} n={len(x):5,} gross {mu:+.3f} R ({(x.rr*x.rpct).mean()*100:+.3f}% '
              f'of price) t_iid {ti:+.2f} t_clu {tc:+.2f} | cost {x.cost_R.mean():.3f} '
              f'net {mun:+.3f} (t_clu {tcn:+.2f}) | ex5 {x.rr[x.rr<=q95].mean():+.3f} | '
              f'book {w["n"]:4d} ({w["per_wk"]:4.1f}/wk) green {w["green"]:5.1f}% '
              f'(null p50 {nl[1]:.1f} p95 {nl[2]:.1f}) ${w["total"]:+,.0f} '
              f'wk ${w["wk_mean"]:+,.0f} worst ${w["worst"]:+,.0f} streak {w["redstreak"]}',
              flush=True)
    tr = z[z.split == 'TRAIN']
    st['h1'] = float(tr[tr.day < '2025-07-01'].rr.mean()) if len(tr) else np.nan
    st['h2'] = float(tr[tr.day >= '2025-07-01'].rr.mean()) if len(tr) else np.nan
    st['h1net'] = float(tr[tr.day < '2025-07-01'].net.mean()) if len(tr) else np.nan
    st['h2net'] = float(tr[tr.day >= '2025-07-01'].net.mean()) if len(tr) else np.nan
    print(f'        halves gross {st["h1"]:+.3f} / {st["h2"]:+.3f}  net {st["h1net"]:+.3f} / '
          f'{st["h2net"]:+.3f}  | unshortable {sh_unshort*100:.1f}% · SSR {sh_ssr*100:.1f}% '
          f'(blocked {sh_ssrblock*100:.1f}%) · cap-skipped {sh_capskip*100:.1f}% '
          f'(counterfactual: unfilled {cf:+.3f} R vs filled {cf_f:+.3f} R)', flush=True)
    rows.append(st)
    return st


def main():
    repro()
    ct = cost_table()
    d = load()
    d = attach_instrument(d)
    d = add_borrow(d)
    d = add_ssr(d)
    print(f'[avail] borrow list coverage {float(d.in_list.mean())*100:.1f}% · '
          f'ETB {float(d.etb.mean())*100:.1f}% · p_close coverage '
          f'{float(d.p_close.notna().mean())*100:.1f}%', flush=True)
    d = universe(d)

    # ---- availability audit (PREREG §1.3): missingness by outcome on the primary cell
    m = d[d.f_mir2.astype(bool)].copy()
    m['win'] = m.rr_a_bare > 0
    for f, nm in ((~m.in_list, 'borrow flag absent'), (m.p_close.isna(), 'prior close missing')):
        gw, gl = float(f[m.win].mean()), float(f[~m.win].mean())
        print(f'[avail] {nm}: coverage {100*(1-float(f.mean())):.1f}% · miss winners '
              f'{gw*100:.1f}% vs losers {gl*100:.1f}% (gap {abs(gw-gl)*100:.1f} pp)', flush=True)

    rows = []
    print('\n== ARM 3 cells (gate5 · price>=$5 · ex-wrapper · ETB only · SSR-aware fill) ==',
          flush=True)
    for tag, name, col, spec, mode in CELLS:
        z = d[d[col].astype(bool)]
        score(tag, name, z, spec, mode, ct, rows)

    print('\n== diagnostics: the rails switched OFF one at a time (S1) ==', flush=True)
    z = d[d.f_mir2.astype(bool)]
    score('S1-noETB', 'S1 without the borrow rail', z, 'a', 'bare', ct, rows, etb_only=False)
    z2 = z.copy(); z2['fill'] = z2.filled_cap.astype(bool)
    score('S1-noSSR', 'S1 ignoring SSR', z2, 'a', 'bare', ct, rows)
    z3 = z.copy(); z3['ssr'] = z3.ssr_wide
    z3['fill'] = z3.filled_cap.astype(bool) & ((~z3.ssr_wide) | z3.uptick.astype(bool))
    score('S1-SSRwide', 'S1 with the conservative SSR upper bound', z3, 'a', 'bare', ct, rows)

    print('\n== cap sensitivity on S1 (gross R on the FILLED set, ETB, SSR-aware) ==', flush=True)
    cs = []
    for cap, lbl in ((0.003, '0.3%'), (0.006, '0.6% (shipped)'), (0.012, '1.2%'), (9.9, 'none')):
        y = z[z.etb & z.fill_ssr & (z.entry >= z.ref * (1 - cap))]
        r = dict(cap=lbl, n=len(y))
        for sp in ('TRAIN', 'VAL'):
            x = y[y.split == sp]
            mu, tc = clust_t1(x.rr_a_bare.values, x.day.values) if len(x) > 10 else (np.nan, np.nan)
            r[f'{sp}_n'] = len(x); r[f'{sp}_g'] = mu; r[f'{sp}_t'] = tc
        cs.append(r)
        print(f'  cap {lbl:14s} n={r["n"]:5,}  TRAIN {r.get("TRAIN_g", float("nan")):+.3f} '
              f'(t {r.get("TRAIN_t", float("nan")):+.2f}, n={r.get("TRAIN_n",0):,})  '
              f'VAL {r.get("VAL_g", float("nan")):+.3f} (t {r.get("VAL_t", float("nan")):+.2f}, '
              f'n={r.get("VAL_n",0):,})', flush=True)
    pd.DataFrame(cs).to_csv(f'{D}/cap_sens.csv', index=False)
    pd.DataFrame(rows).to_csv(f'{D}/cells3.csv', index=False)
    print(f'\n[arm3] wrote cells3.csv ({len(rows)} rows)', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
