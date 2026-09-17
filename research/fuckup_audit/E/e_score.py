#!/usr/bin/env python3
"""Stage E scorer — the 60 pre-registered cells on the CAUSAL universe (E/REPORT.md §0).

The cost/book/gate code is Stage C's contract (c) as implemented in research/fuckup_audit/B/score5.py,
re-stated here in code (the scorer is re-pointed, not re-designed):

  POPULATION  entry (the FILL being scored) >= $5, 570 <= entry_m <= 841, r_pct >= 1.0.
              NO range_so_far_pct floor — membership in U1/U2 is causal at 09:30 by construction.
  COST        half = 0.5 * (spread_cc_bps/100) / max(r_pct, 0.05)   [R units]
              entry 0.25 x half for `next` (an already-printed ask-side price), 1.00 x half for `rest`
              (an arrival execution); exit stop/lock 0.875, eod 0.412, target 0.875, none 0.875.
  BOOK        trading.hod_break.run_book(rows, 12, 4).
  GATES       G1 TRAIN mean net R > 0, t >= 2.0, >= 5 trades/week.
              G2 VAL mean net R > 0, t >= 1.0, >= 55% weeks green, weekly R >= (G1 passes // 10) x SE(wkR).
              G3 TEST only with E_READ_TEST=1, after the selection is frozen in writing.

  CELLS (pre-registered, E/REPORT.md §0): 5 families x 2 fills x 2 exits x 3 populations = 60,
  of which 6 (F13 x rest) are structurally empty and are reported as such.
  Declared sensitivities, counted separately: U1-only and U2-only re-scores; the resting fill
  restricted to rest_queue_ok == 1; the gross (zero-cost) column.
  Pre-registered extra (PLAN §1, 2026-09-17): the `news_only` bucket per family and per time band,
  and the D0b-style disjoint 2x2 {combo, pm_only, news_only, neither} per family.

Usage: ulimit -v 1800000; nice -n 10 python3 research/fuckup_audit/E/e_score.py
"""
import os, sys, time
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from trading.hod_break import run_book

E = f'{ROOT}/research/fuckup_audit/E'
SRC = f'{E}/candidates_causal.csv'
RD = dict(keep_default_na=False, na_values=[''])
PM_CUT = 5_816_688.0
READ_TEST = os.environ.get('E_READ_TEST') == '1'
NPERM = int(os.environ.get('E_PERM', '0'))

FAM_KEYS = ['F8 {"N": 5}', 'F8 {"N": 15}', 'F8 {"N": 30}', 'F6 {}', 'F13 {"K": 5, "X": 0.04}']
CLOSE_TRIGGERED = {'F13'}
FILLS = ('next', 'rest')
EXITS = {'hold': ('rr_hold', 'why_hold', 'exit_m_hold'), '2r': ('rr_2r', 'why_2r', 'exit_m_2r')}
POPS = ('all', 'combo', 'pm_only')
ENTRY_MULT = {'next': 0.25, 'rest': 1.00}
EXIT_RATIO = {'stop': 0.875, 'lock': 0.875, 'eod': 0.412, 'target': 0.875, 'none': 0.875}
MIN_TPW, G1_T, G2_T, G2_GREEN = 5.0, 2.0, 1.0, 0.55

BANDS = [(570, 575, '09:30-09:35'), (575, 600, '09:35-10:00'), (600, 660, '10:00-11:00'),
         (660, 780, '11:00-13:00'), (780, 842, '13:00-14:01')]

BASE = ['day', 'symbol', 'fam', 'cfg', 'spread_cc_bps', 'pm_dollar_vol', 'u1', 'u2', 'rest_queue_ok', 'price',
        'range_so_far_pct']
FCOLS = ['entry', 'entry_m', 'r_pct', 'rr_2r', 'why_2r', 'exit_m_2r', 'rr_hold', 'why_hold', 'exit_m_hold']
USE = BASE + [f'{t}_{c}' for t in FILLS for c in FCOLS]


def log(s):
    print(f'{time.strftime("%H:%M:%S")} {s}', flush=True)


def band_of(m):
    o = pd.Series('other', index=m.index)
    for a, b, nm in BANDS:
        o[(m >= a) & (m < b)] = nm
    return o


def split_of(day):
    return np.where(day < '2026-01-01', 'TRAIN', np.where(day < '2026-06-01', 'VAL', 'TEST'))


def load_news():
    fr = [pd.read_csv(p, usecols=['day', 'symbol', 'n_prev15_to_0930'], dtype={'day': str, 'symbol': str}, **RD)
          for p in (f'{E}/news_presence_e.csv', f'{ROOT}/research/fuckup_audit/D/news_presence.csv')]
    n = pd.concat(fr, ignore_index=True).drop_duplicates(['day', 'symbol'])
    n['has_news'] = (n.n_prev15_to_0930.fillna(0) > 0).astype(int)
    return n[['day', 'symbol', 'has_news']]


def load():
    keep = {t: [] for t in FILLS}
    n_in = 0
    t0 = time.time()
    for ch in pd.read_csv(SRC, usecols=USE, dtype={'day': str, 'symbol': str, 'fam': str, 'cfg': str},
                          chunksize=200_000, low_memory=True, **RD):
        n_in += len(ch)
        for t in FILLS:
            e, em, rp = ch[f'{t}_entry'], ch[f'{t}_entry_m'], ch[f'{t}_r_pct']
            ok = e.notna() & (e >= 5) & (em <= 841) & (em >= 570) & (rp >= 1.0)
            if not ok.any():
                continue
            x = ch.loc[ok, BASE + [f'{t}_{c}' for c in FCOLS]].copy()
            x.columns = BASE + FCOLS
            keep[t].append(x)
    news = load_news()
    out = {}
    for t in FILLS:
        d = pd.concat(keep[t], ignore_index=True)
        d['key'] = d.fam + ' ' + d.cfg
        d['split'] = split_of(d.day.values)
        d['wk'] = pd.to_datetime(d.day).dt.to_period('W-FRI').astype(str)
        d['band'] = band_of(d.entry_m)
        d = d.merge(news, on=['day', 'symbol'], how='left')
        d['b_news'] = d.has_news.fillna(0).astype(int)
        d['b_pm'] = (d.pm_dollar_vol.fillna(0) > PM_CUT).astype(int)
        d['bucket'] = np.where(d.b_news.astype(bool) & d.b_pm.astype(bool), 'combo',
                      np.where(d.b_pm.astype(bool), 'pm_only',
                      np.where(d.b_news.astype(bool), 'news_only', 'neither')))
        d['half'] = 0.5 * (d.spread_cc_bps / 100.0) / d.r_pct.clip(lower=0.05)
        for ex, (rr, why, xm) in EXITS.items():
            d[f'net_{ex}'] = d[rr] - ENTRY_MULT[t] * d.half - d.half * d[why].map(EXIT_RATIO).fillna(0.875)
        out[t] = d
    log(f'read {n_in:,} rows in {(time.time()-t0)/60:.1f} min | scoreable next {len(out["next"]):,} '
        f'rest {len(out["rest"]):,}')
    return n_in, out


def book_stats(x, split, weeks, ex):
    """run_book(12,4) on one split; returns (stats dict, booked trades)."""
    xm = EXITS[ex][2]
    y = x[(x.split == split) & x[f'net_{ex}'].notna() & x[xm].notna()]
    if len(y) < 40:
        return None, None
    rows = [(r.day, int(r.entry_m), int(getattr(r, xm)), r.symbol, float(getattr(r, f'net_{ex}')), r.wk,
             float(getattr(r, EXITS[ex][0])), getattr(r, EXITS[ex][1]))
            for r in y.itertuples()]
    bk = run_book([r[:6] for r in rows], 12, 4)
    t = pd.DataFrame(bk, columns=['day', 'em', 'xm', 'symbol', 'net', 'wk'])
    if len(t) < 20:
        return None, None
    # re-attach gross / why by (day, symbol, entry minute)
    gm = {(r[0], r[3], r[1]): (r[6], r[7]) for r in rows}
    gw = [gm.get((a, b, c), (np.nan, '')) for a, b, c in zip(t.day, t.symbol, t.em)]
    t['gross'] = [g for g, _ in gw]
    t['why'] = [w for _, w in gw]
    w = t.groupby('wk').net.sum().reindex(weeks[split]).fillna(0.0)
    sd = t.net.std(ddof=1)
    mix = t.why.value_counts(normalize=True)
    return dict(n=len(t), tpw=round(len(t) / max(len(weeks[split]), 1), 1), meanR=round(t.net.mean(), 4),
                grossR=round(float(t.gross.mean()), 4),
                t=round(t.net.mean() / (sd / np.sqrt(len(t))), 2) if sd > 0 else 0.0,
                SE=round(float(sd / np.sqrt(len(t))), 4),
                WR=round(float((t.net > 0).mean() * 100), 1), wkR=round(float(w.mean()), 2),
                wkSE=round(float(w.std(ddof=1) / np.sqrt(len(w))), 2),
                green=round(float((w > 0).mean()), 2), worst=round(float(w.min()), 1),
                stopP=round(float(mix.get('stop', 0.0) * 100), 1),
                tgtP=round(float(mix.get('target', 0.0) * 100), 1),
                eodP=round(float(mix.get('eod', 0.0) * 100), 1)), t


def tail_tests(t):
    n, v = len(t), t.net.values
    s = np.sort(v)
    return dict(ex1=round(float(s[:max(n - max(int(n * 0.01), 1), 1)].mean()), 4),
                ex5=round(float(s[:max(n - max(int(n * 0.05), 1), 1)].mean()), 4),
                cap3=round(float(np.minimum(v, 3.0).mean()), 4))


def pop_mask(d, pop):
    if pop == 'all':
        return pd.Series(True, index=d.index)
    if pop == 'combo':
        return (d.b_news == 1) & (d.b_pm == 1)
    return d.b_pm == 1                       # pm_only = the PM$ leg alone (superset of combo)


def score_grid(data, weeks, uni='U1uU2', queue_ok=False, tag=''):
    """The 60 pre-registered cells (or a declared sensitivity re-score of them)."""
    res, cells, cellcount, empty = [], {}, 0, 0
    for fill in FILLS:
        d0 = data[fill]
        if uni == 'U1':
            d0 = d0[d0.u1 == 1]
        elif uni == 'U2':
            d0 = d0[d0.u2 == 1]
        if queue_ok and fill == 'rest':
            d0 = d0[d0.rest_queue_ok == 1]
        for key in FAM_KEYS:
            fam = key.split(' ')[0]
            dk = d0[d0.key == key]
            for ex in EXITS:
                for pop in POPS:
                    cellcount += 1
                    if fill == 'rest' and fam in CLOSE_TRIGGERED:
                        empty += 1
                        continue
                    x = dk[pop_mask(dk, pop)]
                    if not len(x):
                        empty += 1
                        continue
                    st, tr = book_stats(x, 'TRAIN', weeks, ex)
                    if st is None:
                        empty += 1
                        continue
                    sv, tv = book_stats(x, 'VAL', weeks, ex)
                    row = dict(uni=uni, key=key, fill=fill, exit=ex, pop=pop,
                               **{f'TR_{k}': v for k, v in st.items()},
                               **{f'VA_{k}': (sv or {}).get(k) for k in
                                  ('n', 'tpw', 'meanR', 'grossR', 't', 'wkR', 'wkSE', 'green', 'WR')})
                    row.update(tail_tests(tr))
                    g = tr.groupby('day').net
                    cells[(uni, key, fill, ex, pop)] = dict(t=st['t'], trades=tr.net.values,
                                                            bydayN=g.size().values)
                    res.append(row)
    T = pd.DataFrame(res)
    if len(T):
        T = T.sort_values('TR_meanR', ascending=False)
        for c in [c for c in T.columns if c.startswith('VA_')]:
            T[c] = pd.to_numeric(T[c], errors='coerce')
    return T, cells, cellcount, empty


def gates(T):
    g1 = T[(T.TR_meanR > 0) & (T.TR_t >= G1_T) & (T.TR_tpw >= MIN_TPW)]
    bar = len(g1) // 10
    g2 = g1[(g1.VA_meanR > 0) & (g1.VA_t >= G2_T) & (g1.VA_green >= G2_GREEN)
            & (g1.VA_wkR >= bar * g1.VA_wkSE.fillna(0))]
    return g1, g2, bar


def welch(a, b):
    if len(a) < 5 or len(b) < 5:
        return np.nan, np.nan
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    s = np.sqrt(va + vb)
    return float(a.mean() - b.mean()), (float((a.mean() - b.mean()) / s) if s > 0 else np.nan)


def bucket_tables(data):
    """The pre-registered news_only cell + the D0b 2x2, per family and per time band.
    Per-trade (unbooked) — the cleanest comparison of a bucket against the rest of the same family."""
    L, rows = [], []
    d = data['next']
    for key in FAM_KEYS:
        dk = d[d.key == key]
        for scope, dd in [('all-day', dk)] + [(nm, dk[dk.band == nm]) for _, _, nm in BANDS]:
            for sp in ('TRAIN', 'VAL', 'TEST'):
                s = dd[dd.split == sp]
                if len(s) < 30:
                    continue
                for bk in ('combo', 'pm_only', 'news_only', 'neither'):
                    a = s[s.bucket == bk].net_hold.dropna()
                    b = s[s.bucket != bk].net_hold.dropna()
                    dm, tt = welch(a.values, b.values) if len(a) >= 5 else (np.nan, np.nan)
                    rows.append(dict(key=key, scope=scope, split=sp, bucket=bk, n=len(a),
                                     meanR=round(float(a.mean()), 4) if len(a) else np.nan,
                                     rest_meanR=round(float(b.mean()), 4) if len(b) else np.nan,
                                     diff=round(dm, 4) if dm == dm else np.nan,
                                     t=round(tt, 2) if tt == tt else np.nan))
    return pd.DataFrame(rows)


def perm_pvalue(cells, nperm, seed=11):
    rng = np.random.default_rng(seed)
    obs = max(c['t'] for c in cells.values())
    mx = []
    for _ in range(nperm):
        best = -np.inf
        for c in cells.values():
            flip = rng.integers(0, 2, len(c['bydayN'])) * 2 - 1
            v = np.repeat(flip, c['bydayN']) * c['trades']
            sd = v.std(ddof=1)
            if sd > 0:
                best = max(best, v.mean() / (sd / np.sqrt(len(v))))
        mx.append(best)
    mx = np.array(mx)
    return float((mx >= obs).mean()), obs, float(np.percentile(mx, 95))


def main():
    n_in, data = load()
    allr = pd.concat([data['next'], data['rest']], ignore_index=True)
    weeks = {s: sorted(allr[allr.split == s].wk.unique()) for s in ('TRAIN', 'VAL', 'TEST')}
    log(f'weeks TRAIN {len(weeks["TRAIN"])} VAL {len(weeks["VAL"])} TEST {len(weeks["TEST"])}')

    T, cells, cc, empty = score_grid(data, weeks, 'U1uU2')
    g1, g2, bar = gates(T)
    log(f'PRIMARY U1uU2: {cc} cells declared, {len(T)} scoreable, G1 {len(g1)}, G2 {len(g2)}')
    T.to_csv(f'{E}/score_e_results.csv', index=False)

    sens = {}
    for uni in ('U1', 'U2'):
        Tu, _, ccu, _ = score_grid(data, weeks, uni)
        sens[uni] = (Tu, ccu)
        Tu.to_csv(f'{E}/score_e_results_{uni}.csv', index=False)
        log(f'sensitivity {uni}: {len(Tu)} scoreable cells')
    Tq, _, ccq, _ = score_grid(data, weeks, 'U1uU2', queue_ok=True)
    Tq = Tq[Tq.fill == 'rest']
    Tq.to_csv(f'{E}/score_e_results_queueok.csv', index=False)

    B = bucket_tables(data)
    B.to_csv(f'{E}/score_e_buckets.csv', index=False)

    L = ['# Stage E — SCORING OUTPUT (machine tables; the prose lives in E/REPORT.md)', '',
         f'source `{SRC}` | contract (c) | run_book(12,4) | all-day primary | generated '
         f'{time.strftime("%Y-%m-%d %H:%M:%S")}', '',
         f'**cells declared {cc}** (5 families x 2 fills x 2 exits x 3 populations), scoreable {len(T)}, '
         f'structurally empty/too-small {empty}', '',
         '## Every primary cell, ranked by TRAIN mean net R', '',
         T.to_string(index=False), '',
         f'## G1 (TRAIN meanR>0, t>={G1_T}, >={MIN_TPW} tr/wk): {len(g1)} of {len(T)}', '',
         (g1.to_string(index=False) if len(g1) else '(none)'), '',
         f'## G2 (VAL meanR>0, t>={G2_T}, >={G2_GREEN:.0%} green, wkR >= {bar} x SE): {len(g2)}', '',
         (g2.to_string(index=False) if len(g2) else '(none)'), '']
    for uni in ('U1', 'U2'):
        Tu, ccu = sens[uni]
        gu1, gu2, _ = gates(Tu)
        L += [f'## SENSITIVITY — universe {uni} only ({ccu} cells declared, {len(Tu)} scoreable; '
              f'G1 {len(gu1)}, G2 {len(gu2)})', '', Tu.head(20).to_string(index=False), '']
    L += ['## SENSITIVITY — resting fill restricted to rest_queue_ok == 1', '',
          (Tq.to_string(index=False) if len(Tq) else '(none)'), '']
    L += ['## Pre-registered bucket cells (news_only + the D0b 2x2), per-trade, fill next, exit hold', '',
          B.to_string(index=False), '']
    if NPERM and len(cells):
        p, obs, q95 = perm_pvalue(cells, NPERM)
        L += [f'## Permutation null over all {len(cells)} primary cells ({NPERM} day-label sign-flip draws)', '',
              f'observed max TRAIN t = {obs:.2f} | null 95th pct = {q95:.2f} | p = {p:.3f}', '']
    if len(g2) and READ_TEST:
        L += ['## TEST — read once (E_READ_TEST=1)', '']
        for r in g2.itertuples():
            d0 = data[r.fill]
            x = d0[d0.key == r.key]
            x = x[pop_mask(x, r.pop)]
            st, tt = book_stats(x, 'TEST', weeks, getattr(r, 'exit'))
            L.append(f'{r.key} | {r.fill} | {getattr(r, "exit")} | {r.pop} | TEST {st}')
            if tt is not None:
                L.append(tt.assign(mon=tt.day.str[:7]).groupby('mon').net.agg(['size', 'sum', 'mean'])
                         .round(3).to_string())
                L.append('')
    elif len(g2):
        L += ['## TEST — NOT read (E_READ_TEST=1 only after the selection is frozen in writing)', '']
    open(f'{E}/score_e_tables.md', 'w').write('\n'.join(L))
    log(f'wrote {E}/score_e_tables.md  G1 {len(g1)} G2 {len(g2)}')


if __name__ == '__main__':
    main()
