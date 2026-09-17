#!/usr/bin/env python3
"""Stage L step 3 — the declared transfer-filter grid on the five run_book books (B1-B4, B6).

One cell = one filter applied to one book's POPULATION, then the book's own slot rule re-run
(`trading.hod_break.run_book(12, 4)`) so freed slots refill.  T4 is the exception declared in
PREREG §2: it is a cap of one trade a day, so it is applied to the booked sequence (there is nothing
for a freed slot to refill into).  T6 and T7 are exit rules: they rewrite `exit_m` and `net` on the
POPULATION and the book is then re-run, so an earlier exit really does free its slot earlier.

Writes L/cells.csv, L/avail.md, L/perm.csv.  TEST is NOT read here (see l4_test.py).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lcore as C                                                             # noqa: E402

L = C.L
BOOKS5 = ('B1', 'B2', 'B3', 'B4', 'B6')
FORCED_COST_PCT_R = 0.003        # B1: 0.3% of R for the crossing (orb_timestop_validation convention)
DEGENERATE = {('B2', 'T1'): 'universe is prev_day_range_pct >= 8',
              ('B2', 'T2'): 'scan applies the range_so_far_pct >= 5 floor',
              ('B3', 'T2'): 'Stage-C contract applies the floor',
              ('B4', 'T2'): 'Stage-C contract applies the floor',
              ('B6', 'T2'): 'universe UB is range_so_far_pct >= 5'}


# --------------------------------------------------------------------------- population assembly
def load(bid, spy):
    p = pd.read_csv(f'{L}/pop_{bid}.csv', keep_default_na=False, na_values=[''],
                    dtype={'symbol': str, 'day': str, 'split': str, 'wk': str, 'mon': str})
    b = pd.read_csv(f'{L}/bars_{bid}.csv', keep_default_na=False, na_values=[''], index_col='idx')
    p = p.join(b, how='left')
    for col in ('v_fill', 'v_prev5', 'o_next', 'h_next', 'l_next', 'm_next',
                'o_p10', 'h_p10', 'l_p10', 'm_p10', 'rsf_tape'):
        if col not in p.columns:
            p[col] = np.nan
    if bid == 'B1' and 'rsf_tape' in p.columns:
        p['rsf'] = p.rsf.fillna(p.rsf_tape)
    p['spy_at_entry'] = [spy.get((d, int(m) - 1), np.nan) if m == m else np.nan
                         for d, m in zip(p.day, p.entry_m)]
    p['vol_ft'] = p.v_fill / p.v_prev5.replace(0, np.nan)
    return p


# --------------------------------------------------------------------------- the eight filters
def veto_mask(p, tid):
    """True = KEEP.  A missing feature always keeps the trade (PREREG §1 fail-open)."""
    if tid == 'T1':
        return ~(p.pdr < 8.0)
    if tid == 'T2':
        return ~(p.rsf < 5.0)
    if tid == 'T3':
        return ~((p.side * p.spy_at_entry) <= 0)
    if tid == 'T5':
        return ~(p.entry_m >= 600)
    if tid == 'T8':
        return ~(p.r_pct < 3.0)
    raise KeyError(tid)


def apply_exit_rules(p, tids, bid):
    """T6 / T7 rewrite the exit.  Chronological precedence: T7 (fill+1) is checked before T6 (fill+10)."""
    d = p.copy()
    gross = d.gross.astype(float).values
    net = d.net.astype(float).values
    xm = d.exit_m.astype(float).values
    side = d.side.values
    entry = d.entry.values
    rps = d.R_ps.values
    half = d.half.values
    fired = np.zeros(len(d), dtype=bool)
    n_bad_fill = 0

    def forced(mask, o, m_new, lo, hi):
        nonlocal gross, net, xm, fired, n_bad_fill
        ok = mask & np.isfinite(o) & np.isfinite(m_new) & (m_new < xm) & (rps > 0)
        bad = ok & ~((o >= lo - 1e-9) & (o <= hi + 1e-9))
        n_bad_fill += int(np.nansum(bad))
        ok = ok & ~bad
        g = side * (o - entry) / np.where(rps > 0, rps, np.nan)
        if bid == 'B1':
            nn = g - FORCED_COST_PCT_R
        else:
            nn = g - 0.25 * half - 0.875 * half
        gross = np.where(ok, g, gross)
        net = np.where(ok, nn, net)
        xm = np.where(ok, m_new, xm)
        fired = fired | ok

    if 'T7' in tids:
        vf = d.vol_ft.values
        forced(np.isfinite(vf) & (vf < 1.5) & ~fired,
               d.o_next.values, d.m_next.values, d.l_next.values, d.h_next.values)
    if 'T6' in tids:
        o10 = d.o_p10.values
        prog = side * (o10 - entry) / np.where(rps > 0, rps, np.nan)
        forced(np.isfinite(prog) & (prog < 0.25) & ~fired,
               o10, d.m_p10.values, d.l_p10.values, d.h_p10.values)
    d['gross'], d['net'], d['exit_m'] = gross, net, xm
    return d, int(fired.sum()), n_bad_fill


def cell_book(p, tids, bid):
    """Population -> filtered population -> re-booked trades, for one set of filter ids."""
    q = p
    for t in tids:
        if t in ('T1', 'T2', 'T3', 'T5', 'T8'):
            q = q[veto_mask(q, t)]
    n_fire = n_bad = 0
    if 'T6' in tids or 'T7' in tids:
        q, n_fire, n_bad = apply_exit_rules(q, tids, bid)
    bk = C.book(q)
    if 'T4' in tids:
        bk = bk[bk['ord'] == 1]
    return bk, len(q), n_fire, n_bad


# --------------------------------------------------------------------------- driver
def main():
    spy = C.spy_minute_ret()
    cells, perm_rows, avail = [], {}, []
    singles = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8']
    for bid in BOOKS5:
        p = load(bid, spy)
        base = C.book(p)
        wk = {sp: C.weeks_of(p, sp) for sp, _l, _h in C.SPLITS}
        bstat = {sp: C.stats(base[base.split == sp], wk[sp], C.months_in(sp)) for sp in ('TRAIN', 'VAL')}
        # availability, per split, of every feature a filter reads
        for col, tid in (('pdr', 'T1'), ('rsf', 'T2'), ('spy_at_entry', 'T3'), ('entry_m', 'T5'),
                         ('o_p10', 'T6'), ('vol_ft', 'T7'), ('r_pct', 'T8')):
            for sp in ('TRAIN', 'VAL', 'TEST'):
                s = p[p.split == sp]
                avail.append(dict(book=bid, filter=tid, col=col, split=sp, n=len(s),
                                  cov=float(s[col].notna().mean()) if len(s) and col in s else np.nan))
        keep_frac = {}
        for tid in singles:
            if (bid, tid) in DEGENERATE:
                cells.append(dict(book=bid, cell=tid, status='degenerate', note=DEGENERATE[(bid, tid)]))
                continue
            bk, npop, n_fire, n_bad = cell_book(p, [tid], bid)
            row = dict(book=bid, cell=tid, status='run', pop_n=npop, fired=n_fire, bad_fill=n_bad)
            for sp in ('TRAIN', 'VAL'):
                st = C.stats(bk[bk.split == sp], wk[sp], C.months_in(sp))
                b = bstat[sp]
                row[f'{sp}_n'] = st['n']
                row[f'{sp}_base'] = round(b['meanR'], 4)
                row[f'{sp}_mean'] = round(st['meanR'], 4) if st['n'] else np.nan
                row[f'{sp}_imp'] = round(st['meanR'] - b['meanR'], 4) if st['n'] else np.nan
                row[f'{sp}_tpw'] = round(st['tpw'], 1) if st['n'] else 0.0
                row[f'{sp}_permo'] = round(st['permo'], 0)
                row[f'{sp}_ex5'] = round(st['ex5'], 4) if st['n'] else np.nan
                row[f'{sp}_cap3'] = round(st['cap3'], 4) if st['n'] else np.nan
                row[f'{sp}_green'] = round(st['green'], 2) if st['n'] else np.nan
            row['pass'] = int(row['TRAIN_imp'] == row['TRAIN_imp'] and row['TRAIN_imp'] >= 0.03
                              and row['VAL_imp'] == row['VAL_imp'] and row['VAL_imp'] >= 0
                              and row['VAL_mean'] > 0)
            cells.append(row)
            keep_frac[tid] = row
            perm_rows[(bid, tid)] = (base, bk)
        # the declared best-two stack
        cand = [t for t, r in keep_frac.items()
                if r['VAL_imp'] == r['VAL_imp'] and r['VAL_imp'] >= 0 and r['TRAIN_imp'] == r['TRAIN_imp']]
        cand.sort(key=lambda t: -keep_frac[t]['TRAIN_imp'])
        if len(cand) >= 2:
            pair = cand[:2]
            bk, npop, n_fire, n_bad = cell_book(p, pair, bid)
            row = dict(book=bid, cell='STACK ' + '+'.join(pair), status='run', pop_n=npop,
                       fired=n_fire, bad_fill=n_bad)
            for sp in ('TRAIN', 'VAL'):
                st = C.stats(bk[bk.split == sp], wk[sp], C.months_in(sp))
                b = bstat[sp]
                row[f'{sp}_n'] = st['n']
                row[f'{sp}_base'] = round(b['meanR'], 4)
                row[f'{sp}_mean'] = round(st['meanR'], 4) if st['n'] else np.nan
                row[f'{sp}_imp'] = round(st['meanR'] - b['meanR'], 4) if st['n'] else np.nan
                row[f'{sp}_tpw'] = round(st['tpw'], 1) if st['n'] else 0.0
                row[f'{sp}_permo'] = round(st['permo'], 0)
                row[f'{sp}_ex5'] = round(st['ex5'], 4) if st['n'] else np.nan
                row[f'{sp}_cap3'] = round(st['cap3'], 4) if st['n'] else np.nan
                row[f'{sp}_green'] = round(st['green'], 2) if st['n'] else np.nan
            row['pass'] = int(row['TRAIN_imp'] == row['TRAIN_imp'] and row['TRAIN_imp'] >= 0.03
                              and row['VAL_imp'] == row['VAL_imp'] and row['VAL_imp'] >= 0
                              and row['VAL_mean'] > 0)
            cells.append(row)
            perm_rows[(bid, row['cell'])] = (base, bk)
        else:
            cells.append(dict(book=bid, cell='STACK', status='not run',
                              note=f'only {len(cand)} filter(s) with VAL improvement >= 0'))
        C.log(f'{bid} done — {len([c for c in cells if c["book"] == bid])} cells')
    pd.DataFrame(cells).to_csv(f'{L}/cells.csv', index=False)
    pd.DataFrame(avail).to_csv(f'{L}/avail.csv', index=False)

    # ------------------------------------------------- 200-draw day-level sign-flip null over the grid
    rng = np.random.default_rng(20260917)
    obs, draws = [], []
    prepped = []
    for k, (base, bk) in perm_rows.items():
        a = base[base.split == 'TRAIN']
        f = bk[bk.split == 'TRAIN']
        if len(a) < 20 or len(f) < 20:
            continue
        prepped.append((k, a.day.values, a.net.values.astype(float),
                        f.day.values, f.net.values.astype(float)))
        obs.append(abs(f.net.mean() - a.net.mean()))
    days = sorted({d for _k, ad, _an, _fd, _fn in prepped for d in ad})
    dix = {d: i for i, d in enumerate(days)}
    idx = [(k, np.array([dix[d] for d in ad]), an, np.array([dix[d] for d in fd]), fn)
           for k, ad, an, fd, fn in prepped]
    for _ in range(200):
        s = rng.choice([-1.0, 1.0], size=len(days))
        draws.append(max(abs((fn * s[fi]).mean() - (an * s[ai]).mean()) for _k, ai, an, fi, fn in idx))
    draws = np.array(draws)
    p95 = float(np.percentile(draws, 95))
    pval = float((draws >= max(obs)).mean()) if obs else np.nan
    pd.DataFrame(dict(draw=draws)).to_csv(f'{L}/perm.csv', index=False)
    C.log(f'PERM: cells {len(idx)}  observed max |TRAIN improvement| {max(obs):.4f}  '
          f'null 95th pct {p95:.4f}  p = {pval:.3f}')
    open(f'{L}/perm_summary.txt', 'w').write(
        f'cells {len(idx)}\nobserved_max_abs_train_improvement {max(obs):.4f}\n'
        f'null_p95 {p95:.4f}\nnull_mean {draws.mean():.4f}\np {pval:.3f}\n')


if __name__ == '__main__':
    main()
