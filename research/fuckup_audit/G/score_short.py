#!/usr/bin/env python3
"""Stage G scorer — the 24 pre-registered SHORT cells plus the declared sensitivities.

The cost / book / gate code is `research/fuckup_audit/B/score5.py`'s, reused with the exit-reason map
extended by `cover` (a timed market cover = the same 0.412 x half spread an `eod` cover pays) and a
borrow term added. If this file and `G/build_candidates_short.py`'s header ever disagree, the
builder's header is the specification.

  POPULATION   entry >= $10 (the FILL), entry_m <= 841 (14:01), r_pct >= 1.0.
               UA = in_u12 == 1 (S1-S4) / attn_grp == 'attention' (S5);  UB = UA + range_so_far_pct >= 5.
               SSR days (ssr == 1) EXCLUDED under the primary contract.
  COST         half = 0.5 * (spread_cc_bps / 100) / max(r_pct, 0.05)
               net  = rr - 0.25*half - half*{stop:0.875, target:0.875, eod:0.412, cover:0.412}[why]
                        - borrow           (borrow = 0 primary; (5.274/100)/r_pct in the sensitivity)
  BOOK         trading.hod_break.run_book(rows, 12, 4)
  GATE         PLAN §1 G1/G2/G3; TEST only with SCORE_READ_TEST=1, for G2 survivors only.

Usage: ulimit -v 1800000; nice -n 10 python3 research/fuckup_audit/G/score_short.py [--perm N]
"""
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from trading.hod_break import run_book

G = f'{ROOT}/research/fuckup_audit/G'
ARGS = sys.argv[1:]
SRC = ARGS[ARGS.index('--src') + 1] if '--src' in ARGS else f'{G}/candidates_short.csv'
NPERM = int(ARGS[ARGS.index('--perm') + 1]) if '--perm' in ARGS else 0
READ_TEST = os.environ.get('SCORE_READ_TEST') == '1'
OUT_MD = f'{G}/score_short_tables.md'
OUT_CSV = f'{G}/score_short_results.csv'

ENTRY_MULT = 0.25
EXIT_RATIO = {'stop': 0.875, 'target': 0.875, 'eod': 0.412, 'cover': 0.412, 'none': 0.875}
BORROW_BPS = 5.274          # 5 bps locate + 1% annualised / 365 for the one day held
MIN_TPW, G1_T, G2_T, G2_GREEN = 5.0, 2.0, 1.0, 0.55
PRICE_MIN, MAX_ENTRY_M, R_MIN = 10.0, 841, 1.0

FAM_KEYS = ['S1 {}', 'S2 {"N": 15}', 'S2 {"N": 30}', 'S3 {}', 'S4 {"K": 5, "X": 0.04}', 'S5 {}']
S5 = 'S5 {}'
# family -> [(exit label, rr col, why col, exit-minute col)]
EXITS_PATTERN = [('hold', 'rr_hold', 'why_hold', 'exit_m_hold'),
                 ('2R', 'rr_2r', 'why_2r', 'exit_m_2r')]
EXITS_S5 = [('1030', 'rr_1030', 'why_1030', 'exit_m_1030'),
            ('eod', 'rr_hold', 'why_hold', 'exit_m_hold')]
BANDS = [('09:30-10:00', 570, 600), ('10:00-11:00', 600, 660), ('11:00+', 660, 842)]

USE = ['day', 'symbol', 'fam', 'cfg', 'split', 'sig_m', 'range_so_far_pct', 'spread_cc_bps',
       'in_u12', 'attn_grp', 'ssr', 'entry', 'entry_m', 'r_pct', 'fside_entry', 'queue_ok',
       'rr_hold', 'why_hold', 'exit_m_hold', 'fside_hold',
       'rr_2r', 'why_2r', 'exit_m_2r', 'fside_2r',
       'rr_1030', 'why_1030', 'exit_m_1030', 'fside_1030',
       'ret_1030_bps', 'ret_eod_bps', 'mae_pct', 'mfe_r']
L = []


def log(m):
    print(m, flush=True)
    L.append(m)


def load():
    keep, n_in, n_sig = [], 0, 0
    t0 = time.time()
    for ch in pd.read_csv(SRC, usecols=USE, dtype={'symbol': str, 'day': str, 'fam': str,
                                                   'cfg': str, 'split': str, 'attn_grp': str,
                                                   'why_hold': str, 'why_2r': str, 'why_1030': str},
                          keep_default_na=False, na_values=[''], chunksize=250_000,
                          low_memory=True):
        n_in += len(ch)
        n_sig += int(ch.entry.notna().sum())
        ok = (ch.entry.notna() & (ch.entry >= PRICE_MIN) & (ch.entry_m <= MAX_ENTRY_M)
              & (ch.r_pct >= R_MIN))
        if ok.any():
            keep.append(ch.loc[ok].copy())
    d = pd.concat(keep, ignore_index=True)
    d['key'] = d.fam + ' ' + d.cfg
    d['wk'] = pd.to_datetime(d.day).dt.to_period('W-FRI').astype(str)
    log(f'- read {n_in:,} signal rows ({n_sig:,} filled) in {(time.time()-t0)/60:.1f} min; '
        f'scoreable after entry >= ${PRICE_MIN:.0f} / entry_m <= {MAX_ENTRY_M} / '
        f'r_pct >= {R_MIN}: **{len(d):,}**')
    return d


def net_r(x, rr, why, borrow):
    half = 0.5 * (x.spread_cc_bps / 100.0) / x.r_pct.clip(lower=0.05)
    n = x[rr] - ENTRY_MULT * half - half * x[why].map(EXIT_RATIO).fillna(0.875)
    if borrow:
        n = n - (BORROW_BPS / 100.0) / x.r_pct.clip(lower=0.05)
    return n


def book_stats(x, split, weeks):
    y = x[(x.split == split) & x.net.notna() & x.xm.notna()]
    if len(y) < 40:
        return None, None
    rows = [(r.day, int(r.entry_m), int(r.xm), r.symbol, float(r.net), r.wk) for r in y.itertuples()]
    t = pd.DataFrame(run_book(rows, 12, 4), columns=['day', 'em', 'xm', 'symbol', 'net', 'wk'])
    if len(t) < 20:
        return None, None
    t = t.merge(y[['day', 'symbol', 'entry_m', 'gross', 'why']].rename(columns={'entry_m': 'em'}),
                on=['day', 'symbol', 'em'], how='left')
    w = t.groupby('wk').net.sum().reindex(weeks[split]).fillna(0.0)
    sd = t.net.std(ddof=1)
    se = sd / np.sqrt(len(t)) if sd > 0 else np.nan
    return dict(n=len(t), tpw=round(len(t) / max(len(weeks[split]), 1), 1),
                meanR=round(t.net.mean(), 4), grossR=round(float(t.gross.mean()), 4),
                t=round(t.net.mean() / se, 2) if se and se > 0 else 0.0,
                MDE=round(2.80 * se, 4) if se and se > 0 else np.nan,
                WR=round(float((t.net > 0).mean() * 100), 1),
                stopP=round(float((t.why == 'stop').mean() * 100), 1),
                wkR=round(float(w.mean()), 2),
                wkSE=round(float(w.std(ddof=1) / np.sqrt(len(w))), 2),
                green=round(float((w > 0).mean()), 2), worst=round(float(w.min()), 1)), t


def tail_tests(t):
    n = len(t)
    v = t.net.values
    s = np.sort(v)
    return dict(ex1=round(float(s[:max(n - max(int(n * 0.01), 1), 1)].mean()), 4),
                ex5=round(float(s[:max(n - max(int(n * 0.05), 1), 1)].mean()), 4),
                cap3=round(float(np.minimum(v, 3.0).mean()), 4))


def universe_mask(d, key, uni, grp='attention', strict=False):
    if key == S5:
        m = d.attn_grp == grp
        if strict:
            m = m & (d.in_u12 == 1)
    else:
        m = d.in_u12 == 1
    if uni == 'UB':
        m = m & (d.range_so_far_pct >= 5)
    return m


def grid(d, weeks, variant, borrow=False, ssr_include=False, s5_grp='attention', s5_strict=False,
         cells=None):
    res = []
    for key in FAM_KEYS:
        exits = EXITS_S5 if key == S5 else EXITS_PATTERN
        for uni in ('UA', 'UB'):
            base = d[(d.key == key) & universe_mask(d, key, uni, s5_grp, s5_strict)]
            if not ssr_include:
                base = base[base.ssr == 0]
            for lab, rr, why, xm in exits:
                x = base.copy()
                if not len(x):
                    continue
                x['net'] = net_r(x, rr, why, borrow)
                x['gross'] = x[rr]
                x['why'] = x[why]
                x['xm'] = x[xm]
                st, tr = book_stats(x, 'TRAIN', weeks)
                if st is None:
                    continue
                sv, tv = book_stats(x, 'VAL', weeks)
                row = dict(variant=variant, key=key, uni=uni, exit=lab,
                           **{f'TRAIN_{k}': v for k, v in st.items()},
                           **{f'VAL_{k}': (sv or {}).get(k)
                              for k in ('n', 'tpw', 'meanR', 'grossR', 't', 'wkR', 'wkSE', 'green')})
                row.update(tail_tests(tr))
                res.append(row)
                if cells is not None:
                    g = tr.groupby('day').net
                    cells[(variant, key, uni, lab)] = dict(t=st['t'], trades=tr.net.values,
                                                           bydayN=g.size().values)
    return res


def perm_pvalue(cells, nperm, rng):
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


def fmt(T, cols=None):
    t = T if cols is None else T[cols]
    return t.to_string(index=False)


def main():
    d = load()
    weeks = {s: sorted(d[d.split == s].wk.unique()) for s in ('TRAIN', 'VAL', 'TEST')}
    log(f'- weeks TRAIN {len(weeks["TRAIN"])} VAL {len(weeks["VAL"])} TEST {len(weeks["TEST"])}')

    # ---- availability / obtainability audit (PLAN §1 standing rule), BEFORE any result
    log('')
    log('## Obtainability and availability audit (before any result)')
    log('')
    rowsA = []
    for key in FAM_KEYS:
        x = d[d.key == key]
        if not len(x):
            continue
        r = dict(key=key, n=len(x), entry_inside=f'{(x.fside_entry == 0).mean()*100:.2f}%')
        for lab, _, _, _ in (EXITS_S5 if key == S5 else EXITS_PATTERN):
            fs = x[f'fside_{"hold" if lab == "eod" else lab.replace("2R", "2r")}']
            r[f'{lab}_inside'] = f'{(fs == 0).mean()*100:.1f}%'
            r[f'{lab}_worse'] = f'{(fs == 1).mean()*100:.1f}%'
            r[f'{lab}_FAVOURABLE'] = f'{(fs == -1).mean()*100:.2f}%'
        r['queue_ok'] = f'{x.queue_ok.mean()*100:.1f}%'
        r['spread_known'] = f'{x.spread_cc_bps.notna().mean()*100:.2f}%'
        rowsA.append(r)
    log(pd.DataFrame(rowsA).to_string(index=False))
    log('')
    log('`*_inside` = the fill is inside the bar that fills it; `*_worse` = outside on the side that '
        'COSTS the short (a cover above the bar\'s high — conservative, the 0.1% stop slip and the '
        'at-target cover can both print there); `*_FAVOURABLE` = outside on the side that would PAY '
        'the short, the only defect — reported per family.')

    cells = {}
    res = grid(d, weeks, 'primary', cells=cells)
    res += grid(d, weeks, 'ssr_included', ssr_include=True)
    res += grid(d, weeks, 'borrow_5.27bps', borrow=True)
    res += [r for r in grid(d, weeks, 's5_control', s5_grp='control') if r['key'] == S5]
    res += [r for r in grid(d, weeks, 's5_strict_u12', s5_strict=True) if r['key'] == S5]
    T = pd.DataFrame(res)
    T.to_csv(OUT_CSV, index=False)
    P = T[T.variant == 'primary'].sort_values('TRAIN_meanR', ascending=False)
    for c in [c for c in T.columns if c.startswith('VAL_')]:
        P[c] = pd.to_numeric(P[c], errors='coerce')
    g1 = P[(P.TRAIN_meanR > 0) & (P.TRAIN_t >= G1_T) & (P.TRAIN_tpw >= MIN_TPW)]
    bar = len(g1) // 10
    g2 = g1[(g1.VAL_meanR > 0) & (g1.VAL_t >= G2_T) & (g1.VAL_green >= G2_GREEN)
            & (g1.VAL_wkR >= bar * g1.VAL_wkSE.fillna(0))]

    log('')
    log('## The 24 pre-registered cells (PRIMARY contract: borrow 0, SSR days excluded)')
    log('')
    log(fmt(P, ['key', 'uni', 'exit', 'TRAIN_n', 'TRAIN_tpw', 'TRAIN_meanR', 'TRAIN_grossR',
                'TRAIN_t', 'TRAIN_MDE', 'TRAIN_WR', 'TRAIN_stopP', 'TRAIN_wkR', 'TRAIN_green',
                'VAL_n', 'VAL_meanR', 'VAL_grossR', 'VAL_t', 'VAL_wkR', 'VAL_green',
                'ex1', 'ex5', 'cap3']))
    log('')
    log(f'**G1 (TRAIN mean net R > 0, t >= {G1_T}, >= {MIN_TPW} tr/wk): {len(g1)} of {len(P)}**')
    log('')
    log(fmt(g1) if len(g1) else '(none)')
    log('')
    log(f'**G2 (VAL mean net R > 0, t >= {G2_T}, >= {G2_GREEN:.0%} weeks green, '
        f'weekly R >= {bar} x SE): {len(g2)}**')
    log('')
    log(fmt(g2) if len(g2) else '(none)')

    # ---- sensitivities
    for v, title in (('ssr_included', 'SSR days INCLUDED (the uptick rule un-modelled)'),
                     ('borrow_5.27bps', 'borrow charged 5.274 bps of notional'),
                     ('s5_control', 'S5 on the CONTROL group (attention ranks 101-120)'),
                     ('s5_strict_u12', 'S5 restricted to U1uU2 (the strict-universe variant)')):
        S = T[T.variant == v]
        if not len(S):
            continue
        log('')
        log(f'## Sensitivity — {title}')
        log('')
        log(fmt(S.sort_values('TRAIN_meanR', ascending=False),
                ['key', 'uni', 'exit', 'TRAIN_n', 'TRAIN_meanR', 'TRAIN_t', 'TRAIN_wkR',
                 'VAL_n', 'VAL_meanR', 'VAL_t']))

    # ---- time bands on the primary cells
    log('')
    log('## Time bands (reported, not gated) — mean net R and n of the BOOKED trades, primary contract')
    log('')
    rowsB = []
    for key in FAM_KEYS:
        exits = EXITS_S5 if key == S5 else EXITS_PATTERN
        for uni in ('UA', 'UB'):
            base = d[(d.key == key) & universe_mask(d, key, uni) & (d.ssr == 0)]
            for lab, rr, why, xm in exits:
                if not len(base):
                    continue
                x = base.copy()
                x['net'] = net_r(x, rr, why, False)
                x['gross'] = x[rr]
                x['why'] = x[why]
                x['xm'] = x[xm]
                row = dict(key=key, uni=uni, exit=lab)
                for bl, lo, hi in BANDS:
                    y = x[(x.entry_m >= lo) & (x.entry_m < hi)]
                    st, tr = book_stats(y, 'TRAIN', weeks)
                    sv, _ = book_stats(y, 'VAL', weeks)
                    row[f'{bl} TRAIN'] = (f"{st['meanR']:+.3f} (n{st['n']})" if st else '-')
                    row[f'{bl} VAL'] = (f"{sv['meanR']:+.3f} (n{sv['n']})" if sv else '-')
                rowsB.append(row)
    log(pd.DataFrame(rowsB).to_string(index=False))

    # ---- the pure M18 spec, no stop, in bps (S5 only)
    log('')
    log('## Sensitivity — the pure M18 spec for S5: NO stop, cover at 10:30 / at the close, in bps')
    log('')
    rowsC = []
    x5 = d[(d.key == S5) & (d.ssr == 0)]
    for grp in ('attention', 'control'):
        for hz, col in (('10:30', 'ret_1030_bps'), ('close', 'ret_eod_bps')):
            y = x5[x5.attn_grp == grp]
            if not len(y):
                continue
            cost = (ENTRY_MULT + EXIT_RATIO['cover']) * y.spread_cc_bps / 2.0
            for sp in ('TRAIN', 'VAL', 'TEST'):
                z = y[y.split == sp]
                if not len(z) or (sp == 'TEST' and not READ_TEST):
                    continue
                c = cost[z.index]
                gross, net = z[col], z[col] - c
                nb = net - BORROW_BPS
                rowsC.append(dict(group=grp, horizon=hz, split=sp, n=len(z),
                                  gross_bps=round(float(gross.mean()), 1),
                                  net_bps=round(float(net.mean()), 1),
                                  t=round(float(net.mean() / (net.std(ddof=1) / np.sqrt(len(net)))), 2),
                                  net_borrow_bps=round(float(nb.mean()), 1)))
    log(pd.DataFrame(rowsC).to_string(index=False))
    log('')
    log('(this is the whole cross-section, NOT a 12/4 book: every name is taken, so it is a drift '
        'measurement like M18, not a tradable P&L)')

    if NPERM and cells:
        rng = np.random.default_rng(11)
        p, obs, q95 = perm_pvalue(cells, NPERM, rng)
        log('')
        log(f'## Permutation, search-adjusted over all {len(cells)} primary cells '
            f'({NPERM} day-level sign-flip draws)')
        log('')
        log(f'observed max TRAIN t = {obs:.2f} | null 95th pct = {q95:.2f} | **p = {p:.3f}**')

    if len(g2) and READ_TEST:
        log('')
        log('## TEST — read once, for the G2 survivors only')
        log('')
        for r in g2.itertuples():
            exits = EXITS_S5 if r.key == S5 else EXITS_PATTERN
            rr, why, xm = next((a, b, c) for lab, a, b, c in exits if lab == r.exit)
            base = d[(d.key == r.key) & universe_mask(d, r.key, r.uni) & (d.ssr == 0)].copy()
            base['net'] = net_r(base, rr, why, False)
            base['gross'] = base[rr]
            base['why'] = base[why]
            base['xm'] = base[xm]
            st, tt = book_stats(base, 'TEST', weeks)
            log(f'{r.key} | {r.uni} | {r.exit} | TEST {st}')
            if tt is not None:
                mm = tt.assign(mon=tt.day.str[:7]).groupby('mon').net.agg(['size', 'sum', 'mean'])
                log(mm.round(3).to_string())
                log(f'tail: {tail_tests(tt)}')
                tt.to_csv(f'{G}/g2_survivor_trades.csv', index=False)
    elif len(g2):
        log('')
        log('## TEST — NOT read (set SCORE_READ_TEST=1 only after freezing the selection in writing)')
    else:
        log('')
        log('## TEST — NOT read: 0 cells cleared G2, so there is nothing to read it for.')

    open(OUT_MD, 'w').write('\n'.join(L) + '\n')
    print(f'\n-> {OUT_MD}  |  G1 {len(g1)}  G2 {len(g2)}')


if __name__ == '__main__':
    main()
