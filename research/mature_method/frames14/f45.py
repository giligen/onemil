#!/usr/bin/env python3
"""F45 — THE COST MODEL AT THE CLOCKS THE BOOKS ACTUALLY TRADE.  The audit and the six cells.

  python3 f45.py            # the minute-of-day table + E1..E6

Everything is declared in `frames14/PREREG.md` §1 BEFORE any cell was read.  Stores READ-ONLY.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)

D14 = f'{ROOT}/research/mature_method/frames14'
PC = f'{ROOT}/research/fuckup_audit/P_cost'
NBBO = f'{ROOT}/research/bf_zero/causal_filter/nbbo.csv'
BREAKS = f'{ROOT}/research/mature_method/hod_break/breaks.csv'

PB_EDGES = [0, 5, 10, 20, 30, 50, 100, 1e9]
PB_LAB = ['<$5', '$5-10', '$10-20', '$20-30', '$30-50', '$50-100', '$100+']
HB_EDGES = [569, 585, 600, 660, 780, 960]
HB_LAB = ['0930-0945', '0945-1000', '1000-1100', '1100-1300', '1300+']
CLOCKS = [570, 571, 575, 577, 580, 585, 600, 660, 720, 780, 840, 900, 945, 955]
CLK_LAB = {570: '09:30', 571: '09:31', 575: '09:35*ORB submit', 577: '09:37', 580: '09:40',
           585: '09:45', 600: '10:00', 660: '11:00', 720: '12:00', 780: '13:00', 840: '14:00',
           900: '15:00', 945: '15:45*ORB flat', 955: '15:55*HOD/BF flat'}
RATIO_EOD = 0.412          # the score4 contract's exit-leg ratio for an eod/flat fill


def hb_of(mn):
    i = int(np.searchsorted(np.array(HB_EDGES[1:-1]), mn, 'right'))
    return HB_LAB[min(i, len(HB_LAB) - 1)]


def pb_of(px):
    return PB_LAB[int(np.searchsorted(np.array(PB_EDGES[1:-1]), px, 'right'))]


# ------------------------------------------------------------------ the table under audit
def build_impute():
    """Rebuild `hod_break/score.py::build_impute` exactly: median spread-% by (pb x hb) of the
    MEASURED sample.  The printed global median is the reproduction gate (score.py prints 0.345 %)."""
    bk = pd.read_csv(BREAKS, usecols=['day', 'symbol', 'entry_m', 'next_open'],
                     dtype={'day': str, 'symbol': str})
    nb = pd.read_csv(NBBO, dtype={'day': str, 'symbol': str}).drop_duplicates(
        ['day', 'symbol', 'entry_m'])
    m = bk.merge(nb[['day', 'symbol', 'entry_m', 'spread_mean']], on=['day', 'symbol', 'entry_m'],
                 how='inner')
    m = m[m.spread_mean.notna() & (m.next_open > 0)].copy()
    m['sp_pct'] = m.spread_mean / m.next_open * 100
    m['pb'] = pd.cut(m.next_open, PB_EDGES, labels=PB_LAB)
    m['hb'] = pd.cut(m.entry_m, HB_EDGES, labels=HB_LAB)
    imp = m.groupby(['pb', 'hb'], observed=True).sp_pct.median().to_dict()
    gl = float(m.sp_pct.median())
    print(f'\nGATE build_impute: measured rows {len(m):,}  cells {len(imp)}  '
          f'global median {gl:.3f} %   (score.py prints 0.345 %)', flush=True)
    assert abs(gl - 0.345) < 0.002, f'imputation table does not reproduce (global {gl})'
    print(f'  the FIT WINDOW of that table: entry_m {int(m.entry_m.min())} '
          f'({int(m.entry_m.min())//60:02d}:{int(m.entry_m.min())%60:02d}) -> '
          f'{int(m.entry_m.max())} ({int(m.entry_m.max())//60:02d}:{int(m.entry_m.max())%60:02d})',
          flush=True)
    return imp, gl


# ------------------------------------------------------------------ the minute-of-day table
def minute_table(imp, gl):
    d = pd.read_csv(f'{D14}/f45_minutes.csv', dtype={'day': str, 'symbol': str})
    nsd = d.drop_duplicates(['day', 'symbol']).shape[0]
    d['sp_pct'] = d.sp_mean / d.price * 100
    ok = d[d.sp_pct.notna() & (d.n_q > 0)]
    print(f'\n### THE MINUTE-OF-DAY NBBO TABLE — {nsd} symbol-days of the imputation table\'s OWN '
          f'population, {len(CLOCKS)} declared clocks, Alpaca SIP\n', flush=True)
    print(f'{"clock":>18} {"n":>5} {"cov":>6} {"med %":>8} {"mean %":>8} {"med bps":>8} '
          f'{"IMPUTED %":>10} {"meas/imp":>9}', flush=True)
    rows = []
    for cm in CLOCKS:
        x = ok[ok.clock_m == cm]
        if not len(x):
            continue
        med = float(x.sp_pct.median()); mean = float(x.sp_pct.mean())
        # the table's own prediction for these same names at this clock
        pred = np.array([imp.get((pb_of(p), hb_of(cm)), gl) for p in x.price])
        ip = float(np.median(pred))
        cov = len(x) / nsd
        rows.append(dict(clock_m=cm, label=CLK_LAB[cm], n=len(x), cov=cov, med=med, mean=mean,
                         imputed=ip, ratio=med / ip if ip else np.nan))
        flag = '' if cov >= 0.80 else '  <-- UNDER-COVERED'
        print(f'{CLK_LAB[cm]:>18} {len(x):5d} {cov:6.1%} {med:8.3f} {mean:8.3f} {med*100:8.1f} '
              f'{ip:10.3f} {med/ip if ip else np.nan:9.2f}x{flag}', flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(f'{D14}/f45_minute_table.csv', index=False)
    # by price band, the three clocks that matter most
    print('\n  by price band (median % of price):', flush=True)
    hdr = ' '.join(f'{CLK_LAB[c].split("*")[0]:>7}' for c in CLOCKS)
    print(f'{"band":>10} {hdr}', flush=True)
    for pb in PB_LAB:
        x = ok[ok.pb == pb]
        if not len(x):
            continue
        cells = ' '.join(f'{x[x.clock_m == c].sp_pct.median():7.3f}' if len(x[x.clock_m == c])
                         else f'{"-":>7}' for c in CLOCKS)
        print(f'{pb:>10} {cells}', flush=True)
    return t


def independent_confirmations(t):
    """The two pulls already on disk, reported beside the new one (nothing refetched)."""
    print('\n  independent confirmations already on disk (not refetched):', flush=True)
    f = f'{ROOT}/research/mature_method/frames13/openspread.csv'
    if os.path.exists(f):
        o = pd.read_csv(f)
        o = o[o.err.isna() | (o.err == '')]
        for k, lab in (('sp_0930', '09:30'), ('sp_0931', '09:31'), ('sp_1555', '15:55')):
            v = (o[k] / o.close * 100).dropna()
            mine = t[t.clock_m == {'09:30': 570, '09:31': 571, '15:55': 955}[lab]]
            m2 = float(mine.med.iloc[0]) if len(mine) else np.nan
            print(f'    frames13 openspread {lab}: median {v.median():.3f} % (n={len(v)}) '
                  f'| frames14 {m2:.3f} %', flush=True)


# ------------------------------------------------------------------ E1 / E2 — ORB
def orb_cells(t):
    sr = pd.read_csv(f'{PC}/spread_rows.csv', dtype={'symbol': str, 'date': str})
    n0 = len(sr)
    e = sr[sr.entry_spread.notna() & (sr.entry_price > 0)].copy()
    e['t'] = pd.to_datetime(e.entry_q_ts, utc=True, errors='coerce').dt.tz_convert('America/New_York')
    e['m'] = e.t.dt.hour * 60 + e.t.dt.minute
    e['sp_pct'] = e.entry_spread / e.entry_price * 100
    x = sr[sr.exit_spread.notna() & (sr.exit_price > 0)].copy()
    x['t'] = pd.to_datetime(x.exit_q_ts, utc=True, errors='coerce').dt.tz_convert('America/New_York')
    x['m'] = x.t.dt.hour * 60 + x.t.dt.minute
    x['sp_pct'] = x.exit_spread / x.exit_price * 100
    print(f'\n### E1 — ORB at 09:35.  Stage P rows {n0:,}; entry quotes {len(e):,} '
          f'({len(e)/n0:.1%}), exit quotes {len(x):,} ({len(x)/n0:.1%})', flush=True)
    print('  PROVENANCE (code-read, not assumed): P_cost/fetch_spreads.py pulls the SIP NBBO at the '
          'FILL INSTANT\n  (last quote at or before the first trade above range_high) and at the '
          'EXIT INSTANT — per trade,\n  never a table constant.  Stage Q then walked the whole order '
          'life.  So ORB is NOT priced by S.IMPUTE.', flush=True)
    e35 = e[e.m == 575]
    print(f'  ORB entry instants landing in minute 09:35: n={len(e35):,} '
          f'({len(e35)/len(e):.1%} of fills) median {e35.sp_pct.median():.3f} % '
          f'({e35.sp_pct.median()*100:.1f} bps), mean {e35.sp_pct.mean():.3f} %', flush=True)
    fresh = t[t.clock_m == 575]
    fm = float(fresh.med.iloc[0]) if len(fresh) else np.nan
    print(f'  fresh 09:35 measurement on the HOD universe: {fm:.3f} % '
          f'| residual vs ORB\'s own = {e35.sp_pct.median() - fm:+.3f} pp '
          f'(different universes; ORB screens 500K prev volume)', flush=True)
    print(f'\n### E2 — ORB at the 15:45 flat.  ORB charges a flat EXIT_SLIP_BPS = 10 bps.', flush=True)
    for lab, mm in (('15:45 (the flat)', 945), ('15:40-15:45', None)):
        xx = x[x.m == mm] if mm else x[(x.m >= 940) & (x.m <= 945)]
        if not len(xx):
            continue
        half = xx.sp_pct.median() / 2
        print(f'  ORB exit instants at {lab}: n={len(xx):,} median full spread '
              f'{xx.sp_pct.median():.3f} % = {xx.sp_pct.median()*100:.1f} bps, '
              f'half {half*100:.1f} bps vs charged 10.0 bps -> residual {half*100 - 10:+.1f} bps',
              flush=True)
    eod = x[x.exit_instant_rule.astype(str).str.contains('close', na=False)]
    print(f'  (closed-bar exits, all reasons: n={len(eod):,})', flush=True)
    med_r = None
    try:
        b = pd.read_csv(f'{PC}/book_measured_n8.csv')
        if 'r_pct' in b.columns:
            med_r = float(b.r_pct.median())
    except Exception as ex:
        print(f'    WARNING book_measured_n8 not read for r_pct ({ex})', flush=True)
    if med_r is None:
        med_r = 4.03                               # P_cost §5.2's measured median, stated there
    xx = x[(x.m >= 940) & (x.m <= 945)]
    resid_pp = (xx.sp_pct.median() / 2) - 0.10
    print(f'  ORB median r_pct = {med_r:.2f} % of price -> the 15:45 residual is '
          f'{resid_pp/med_r:+.4f} R per force-closed trade', flush=True)
    return e, x


# ------------------------------------------------------------------ E3 / E4 — BULL FLAG
def bf_cells():
    f = f'{D14}/f45_bf.csv'
    if not os.path.exists(f):
        print('\n### E3/E4 — BF pull not on disk yet; skipped', flush=True)
        return None
    q = pd.read_csv(f, dtype={'day': str, 'symbol': str})
    q = q[q.sp_mean.notna() & (q.px > 0)].copy()
    q['sp_pct'] = q.sp_mean / q.px * 100
    q['sp_pct_med'] = q.sp_med / q.px * 100
    raw = pd.read_csv(f'{ROOT}/data/bull_flag_cache_causal_full_20260905.csv',
                      dtype={'symbol': str, 'date': str})
    p1 = pd.read_csv(f'{ROOT}/research/bf_frequency/runs/P1.csv', dtype={'symbol': str, 'date': str})
    qe_raw = q[(q['pop'] == 'raw') & (q.leg == 'entry')]
    qe_p1 = q[(q['pop'] == 'p1') & (q.leg == 'entry')]
    qx_p1 = q[(q['pop'] == 'p1') & (q.leg == 'exit')]
    print(f'\n### E3 — BF entry NBBO at its OWN detection minutes (the shipped model is a FLAT '
          f'50 bps, spread-blind)', flush=True)
    for lab, n_pop, z in (('896 raw regen-7 detections', len(raw), qe_raw),
                          ('56 P1 trades', len(p1), qe_p1)):
        cov = len(z) / n_pop
        print(f'  {lab}: measured {len(z)}/{n_pop} = {cov:.1%}'
              f'{"  <-- BELOW THE 80 % RAIL" if cov < 0.80 else ""}', flush=True)
        print(f'     full spread  median {z.sp_pct.median()*100:6.1f} bps  mean '
              f'{z.sp_pct.mean()*100:6.1f}  p25 {z.sp_pct.quantile(.25)*100:6.1f}  p75 '
              f'{z.sp_pct.quantile(.75)*100:6.1f}  p90 {z.sp_pct.quantile(.90)*100:6.1f}',
              flush=True)
        print(f'     HALF spread  median-of-MEAN {z.sp_pct.median()*50:6.1f} bps | '
              f'median-of-MEDIAN (= the fill instant, P_cost §3.2) '
              f'{z.sp_pct_med.median()*50:6.1f} bps  vs the charged 50.0 bps '
              f'-> {z.sp_pct.median()*50/50:.2f}x / {z.sp_pct_med.median()*50/50:.2f}x',
              flush=True)
    d = qe_raw.copy()
    d['hb'] = [hb_of(m) for m in d.clock_m]
    print('\n  BF\'s detection clock (the 896 raw detections):', flush=True)
    for hb in HB_LAB:
        z = d[d.hb == hb]
        if len(z):
            print(f'    {hb:>10}  n={len(z):4d} ({len(z)/len(d):5.1%})  median half-spread '
                  f'{z.sp_pct.median()*50:6.1f} bps', flush=True)
    print(f'  exit minute (56 P1): measured {len(qx_p1)}/{len(p1)}  full spread MEAN '
          f'{qx_p1.sp_pct.median()*100:.1f} bps / MEDIAN {qx_p1.sp_pct_med.median()*100:.1f} bps '
          f'-> half {qx_p1.sp_pct_med.median()*50:.1f} bps vs the shipped 30 bps exit slip '
          f'(stop-type exits only)', flush=True)
    return q, raw, p1


def bf_book(q, p1, col='sp_med', arm=''):
    """E4 — the honest BF P1 book under MEASURED cost, in R and in % of price, both halves + VAL."""
    def _m(t):
        hh, mm = str(t).split(':')[:2]
        return int(hh) * 60 + int(mm)

    def _dict(leg, col):
        z = q[(q['pop'] == 'p1') & (q.leg == leg)]
        return {(d, s, int(c)): float(v) for d, s, c, v
                in zip(z.day, z.symbol, z.clock_m, z[col])}

    e, x = _dict('entry', col), _dict('exit', col)
    b = p1.copy()
    b['ke'] = [(d, s, _m(t)) for d, s, t in zip(b.date, b.symbol, b.entry_time_et)]
    b['kx'] = [(d, s, _m(t)) for d, s, t in zip(b.date, b.symbol, b.exit_time_et)]
    b['sp_e'] = [e.get(k, np.nan) for k in b.ke]
    b['sp_x'] = [x.get(k, np.nan) for k in b.kx]
    b['shares'] = b.shares.astype(float)
    b['ex_shares'] = b.shares - b.partial_shares.fillna(0).astype(float)
    raw_fill = b.entry_price / 1.005                      # the shipped 50 bps is IN entry_price
    b['R_dollar'] = b.shares * (b.entry_price - b.stop_loss)
    # the entry leg: pay HALF the measured spread above the level instead of the flat 50 bps
    d_entry = b.shares * ((raw_fill + 0.5 * b.sp_e) - b.entry_price)
    # the exit leg: stop-type exits already carry a 30 bps slip; every other exit carries nothing
    stopish = b.exit_reason.astype(str).str.contains('stop', na=False)
    charged_x = np.where(stopish, 0.003 * b.exit_price, 0.0)
    d_exit = b.ex_shares * (0.5 * b.sp_x - charged_x)
    # the partial leg is marketable and is charged NOTHING today; charge it the ENTRY-minute
    # half-spread (the wider of the two on this book -> conservative). Declared approximation.
    d_part = b.partial_shares.fillna(0).astype(float) * 0.5 * b.sp_e
    b['d_entry'] = -d_entry.fillna(0)
    b['d_exit'] = -d_exit.fillna(0)
    b['d_part'] = -d_part.fillna(0)
    b['pnl_meas'] = b.pnl + b.d_entry + b.d_exit + b.d_part
    b['pnl_entry_only'] = b.pnl + b.d_entry
    b['covd'] = b.sp_e.notna() & b.sp_x.notna()
    b['R_booked'] = b.pnl / b.R_dollar
    b['R_meas'] = b.pnl_meas / b.R_dollar
    b['pct_booked'] = b.pnl / (b.shares * b.entry_price) * 100
    b['pct_meas'] = b.pnl_meas / (b.shares * b.entry_price) * 100
    b['half'] = np.where(b.date < '2025-07-01', 'H1-25',
                         np.where(b.date < '2026-01-01', 'H2-25', 'VAL'))
    print(f'\n### E4{arm} — the honest BF P1 book under MEASURED cost [{col}] '
          f'(coverage {b.covd.mean():.1%}; uncovered rows keep their booked price)', flush=True)
    print(f'{"split":>8} {"n":>4} {"booked $":>12} {"measured $":>12} {"d$":>10} {"d%":>7} '
          f'{"R book":>8} {"R meas":>8} {"% px book":>10} {"% px meas":>10}', flush=True)
    for sp in ('H1-25', 'H2-25', 'VAL', 'ALL'):
        z = b if sp == 'ALL' else b[b.half == sp]
        if not len(z):
            continue
        bk, ms = float(z.pnl.sum()), float(z.pnl_meas.sum())
        print(f'{sp:>8} {len(z):4d} {bk:12,.0f} {ms:12,.0f} {ms-bk:10,.0f} '
              f'{(ms-bk)/abs(bk)*100 if bk else np.nan:6.1f}% {z.R_booked.mean():+8.3f} '
              f'{z.R_meas.mean():+8.3f} {z.pct_booked.mean():+10.3f} {z.pct_meas.mean():+10.3f}',
              flush=True)
    b.to_csv(f'{D14}/f45_bf_book_{col}.csv', index=False)
    print(f'  LEG DECOMPOSITION ($): entry {b.d_entry.sum():+,.0f} | exit {b.d_exit.sum():+,.0f} '
          f'| partial {b.d_part.sum():+,.0f}', flush=True)
    print(f'  ENTRY LEG ALONE (exit left exactly as shipped — the unambiguous half; a marketable '
          f'buy at the\n  breakout level pays the ask, and nothing in the book already charges '
          f'it beyond the flat 50 bps):', flush=True)
    for sp in ('H1-25', 'H2-25', 'VAL', 'ALL'):
        z = b if sp == 'ALL' else b[b.half == sp]
        if not len(z):
            continue
        bk2, ms2 = float(z.pnl.sum()), float(z.pnl_entry_only.sum())
        print(f'    {sp:>6} {len(z):3d}  booked {bk2:>10,.0f} -> entry-measured {ms2:>10,.0f} '
              f'({(ms2-bk2)/abs(bk2)*100 if bk2 else np.nan:+.1f} %)  R '
              f'{(z.pnl_entry_only/z.R_dollar).mean():+.3f}', flush=True)
    tot_b, tot_m = float(b.pnl.sum()), float(b.pnl_meas.sum())
    mv = abs(tot_m - tot_b) / abs(tot_b) * 100
    print(f'\n  >>> BF honest number moves {mv:.1f} % '
          f'({"MORE" if mv > 10 else "LESS"} than the 10 % reporting threshold)', flush=True)
    return b


# ------------------------------------------------------------------ E5 / E6 — HOD-break
def hod_cells(t, imp, gl):
    print('\n### E5 — HOD dry: is the imputation right at the clocks it was FIT on (09:37-14:01)?',
          flush=True)
    bad = []
    for cm in (577, 580, 585, 600, 660, 720, 780, 840):
        r = t[t.clock_m == cm]
        if not len(r):
            continue
        ratio = float(r.ratio.iloc[0])
        ok = 0.75 <= ratio <= 1.25
        if not ok:
            bad.append((CLK_LAB[cm], ratio))
        print(f'  {CLK_LAB[cm]:>7}  measured {float(r.med.iloc[0]):.3f} %  imputed '
              f'{float(r.imputed.iloc[0]):.3f} %  ratio {ratio:.2f}x  '
              f'{"OK" if ok else "OUTSIDE +/-25 %"}', flush=True)
    print(f'  verdict: {"CONFIRMED" if not bad else "NOT confirmed at " + str(bad)}', flush=True)
    print('\n### E6 — HOD at the 15:55 flat.  The contract charges the EOD leg as '
          f'{RATIO_EOD} x the SIGNAL-minute half-spread.', flush=True)
    r55 = t[t.clock_m == 955]
    sig = t[t.clock_m.isin([577, 580, 585, 600, 660, 720, 780, 840])]
    if len(r55) and len(sig):
        m55 = float(r55.med.iloc[0]); msig = float(sig.med.median())
        charged = RATIO_EOD * 0.5 * msig
        actual = 0.5 * m55
        print(f'  measured 15:55 full spread {m55:.3f} % -> half {actual*100:.1f} bps', flush=True)
        print(f'  the contract charges {RATIO_EOD} x half of the typical SIGNAL minute '
              f'({msig:.3f} %) = {charged*100:.1f} bps', flush=True)
        print(f'  residual {actual*100 - charged*100:+.1f} bps per force-closed trade '
              f'({(actual - charged):+.4f} pp of price)', flush=True)
        print(f'  HOD force-closes 27.9 % (TRAIN) / 33.0 % (VAL) of its trades (frames13 §3.2) -> '
              f'{(actual-charged)*0.279*100:+.2f} / {(actual-charged)*0.330*100:+.2f} bps per trade',
              flush=True)


def main():
    imp, gl = build_impute()
    t = minute_table(imp, gl)
    independent_confirmations(t)
    orb_cells(t)
    r = bf_cells()
    if r is not None:
        q, raw, p1 = r
        bf_book(q, p1, col='sp_med', arm=' (PRIMARY: the minute MEDIAN, which P_cost §3.2 '
                 'measured to equal the fill instant, ratio 1.000)')
        bf_book(q, p1, col='sp_mean', arm=' (CONSERVATIVE: the minute MEAN, ~1.9x the median on '
                 'this distribution)')
    hod_cells(t, imp, gl)
    return 0


if __name__ == '__main__':
    sys.exit(main())
