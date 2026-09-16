#!/usr/bin/env python3
"""STAGE C scorer — family x fill x stop-variant x exit on candidates4.csv, under the corrected cost contract and
PLAN.md §1's statistical gate.  The contract is stated in build_candidates4.py's header and repeated here in code;
if the two ever disagree, the builder's header is the specification.

  POPULATION   entry >= $5 (the FILL of the model being scored), entry_m <= 841 (14:01), r_pct >= 1.0 OF THE STOP
               VARIANT being scored, and range_so_far_pct >= 5 for every family except F1-F4 (the causal membership
               guarantee for an end-of-day-range universe).  Early-close days are already absent from the universe.
  COST         half_cc = 0.5 * (spread_cc_bps/100) / max(r_pct_variant, 0.05)      [R units]
               entry: 0.25 x half_cc for the next-open fill (an already-printed, ask-side price), 1.00 x half_cc for
               the resting fill (an arrival execution).  exit: stop/lock 0.875, eod 0.412, target 0.875 by default
               (ONLY an engine that rests a take-profit leg pays 0 — reported as a sensitivity, never as the headline).
               `--legacy-spread` re-scores with score4's 1.90/1.20/... band table for the before/after comparison.
  BOOK         trading.hod_break.run_book(rows, 12, 4).
  GATES        G1 TRAIN mean net R > 0, t >= 2.0, >= 5 trades/week.
               G2 VAL mean net R > 0, t >= 1.0, >= 55% weeks green, and weekly R >= (cells_passing_G1 // 10) x SE(wkR).
               G3 TEST is computed ONLY with SCORE5_READ_TEST=1 — read once, after the selection is frozen in writing.
  EXTRAS       cell count, tail test (top 1% / 5% removed, winners capped at +3R), per-month table and a day-level
               sign-flip permutation p (search-adjusted: max |t| over ALL cells) for every G2 survivor.

Usage:  ulimit -v 1500000; nice -n 15 python3 research/fuckup_audit/B/score5.py [--src FILE] [--perm N]
        [--legacy-spread] [--free-target]
"""
import os, sys, time
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from trading.hod_break import run_book

D = 'research/fuckup_audit/B'
ARGS = sys.argv[1:]
SRC = ARGS[ARGS.index('--src') + 1] if '--src' in ARGS else f'{D}/candidates4.csv'
NPERM = int(ARGS[ARGS.index('--perm') + 1]) if '--perm' in ARGS else 0
LEGACY_SPREAD = '--legacy-spread' in ARGS
FREE_TARGET = '--free-target' in ARGS
# Stage A's recommendation was "entries >= 10:00 for F6/F8"; that is a SECOND, separately counted pass, never a
# silent default. --min-entry-m 600 runs it and writes its own tables so the cell count stays honest.
MIN_ENTRY_M = int(ARGS[ARGS.index('--min-entry-m') + 1]) if '--min-entry-m' in ARGS else 0
OUT_MD = (f'{D}/score5_tables.md' + ('.legacy' if LEGACY_SPREAD else '') + ('.freetgt' if FREE_TARGET else '')
          + (f'.m{MIN_ENTRY_M}' if MIN_ENTRY_M else ''))
OUT_CSV = OUT_MD.replace('_tables.md', '_results.csv')
READ_TEST = os.environ.get('SCORE5_READ_TEST') == '1'

ENTRY_MULT = {'next': 0.25, 'rest': 1.00}
EXIT_RATIO = {'stop': 0.875, 'lock': 0.875, 'eod': 0.412, 'target': 0.0 if FREE_TARGET else 0.875, 'none': 0.875}
MIN_TPW, G1_T, G2_T, G2_GREEN = 5.0, 2.0, 1.0, 0.55
EXEMPT = ('F1', 'F2', 'F3', 'F4')                 # families whose signal does not need the causal range floor
CLOSE_TRIGGERED = {'F11', 'F12', 'F13'}           # no resting fill by construction
# outcome -> (rr col, why col, exit-minute col, which r_pct the R floor and the cost denominator use)
OUTCOMES = {
    '2R close-fill': ('rr_2r', 'why_2r', 'exit_m_2r', 'r_pct'),
    'hold-to-close': ('rr_hold', 'why_hold', 'exit_m_hold', 'r_pct'),
    'lock 1.75/0.5': ('rr_lock', 'why_lock', 'exit_m_lock', 'r_pct'),
    '2R close-stop': ('rr_2r_closestop', 'why_2r_closestop', 'exit_m_2r_closestop', 'r_pct'),
    '2R stop-1%': ('rr_2r_stopm1', 'why_2r_stopm1', 'exit_m_2r_stopm1', 'r_pct_m1'),
}
# The pre-registered cell list — the SAME 14 family-configs build_candidates4.py emits, written out so the cell
# count is a pre-registration, not a count of whatever happened to appear in the data.
FAM_KEYS = ['F1 {"P": 0.12}', 'F5 {"K": 5, "X": 0.04}', 'F6 {}', 'F8 {"N": 5}', 'F8 {"N": 15}', 'F8 {"N": 30}',
            'F9 {"G": 0.05}', 'F10 {}', 'F11 {"N": 15, "base": "F8"}', 'F11 {"base": "F6"}',
            'F12 {"N": 15, "base": "F8"}', 'F12 {"base": "F6"}', 'F13 {"K": 5, "X": 0.04}', 'F14 {"N": 15}']
FILL_COLS = ['entry', 'entry_m', 'r_pct', 'r_pct_m1'] + sorted({c for v in OUTCOMES.values() for c in v[:3]})
BASE_COLS = ['day', 'symbol', 'fam', 'cfg', 'range_so_far_pct', 'spread_cc_bps', 'spread_pct']
USE = BASE_COLS + [f'{t}_{c}' for t in ('next', 'rest') for c in FILL_COLS]


def split_of(day):
    return np.where(day < '2026-01-01', 'TRAIN', np.where(day < '2026-06-01', 'VAL', 'TEST'))


def load():
    """One chunked pass: keep only rows that can enter SOME cell, in a compact per-fill frame."""
    keep = {'next': [], 'rest': []}
    n_in = 0
    t0 = time.time()
    for ch in pd.read_csv(SRC, usecols=USE, dtype={'symbol': str, 'day': str, 'fam': str, 'cfg': str},
                          keep_default_na=False, na_values=[''], chunksize=250_000, low_memory=True):
        n_in += len(ch)
        causal = ch.fam.isin(EXEMPT) | (ch.range_so_far_pct >= 5)
        for t in ('next', 'rest'):
            e, em = ch[f'{t}_entry'], ch[f'{t}_entry_m']
            ok = (causal & e.notna() & (e >= 5) & (em <= 841) & (em >= MIN_ENTRY_M)
                  & ((ch[f'{t}_r_pct'] >= 1.0) | (ch[f'{t}_r_pct_m1'] >= 1.0)))
            if not ok.any():
                continue
            x = ch.loc[ok, BASE_COLS + [f'{t}_{c}' for c in FILL_COLS]].copy()
            x.columns = BASE_COLS + FILL_COLS
            keep[t].append(x)
    out = {}
    for t in ('next', 'rest'):
        d = pd.concat(keep[t], ignore_index=True) if keep[t] else pd.DataFrame(columns=BASE_COLS + FILL_COLS)
        d['key'] = d.fam + ' ' + d.cfg
        d['split'] = split_of(d.day.values)
        d['wk'] = pd.to_datetime(d.day).dt.to_period('W-FRI').astype(str)
        out[t] = d
    print(f'read {n_in:,} signal rows in {(time.time()-t0)/60:.1f} min | scoreable: '
          f"next {len(out['next']):,}  rest {len(out['rest']):,}", flush=True)
    return out


def net_r(d, rr, why, rp, fill):
    """The corrected cost contract, in R units of the variant being scored."""
    spread = d.spread_pct if LEGACY_SPREAD else d.spread_cc_bps / 100.0
    half = 0.5 * spread / d[rp].clip(lower=0.05)
    return d[rr] - ENTRY_MULT[fill] * half - half * d[why].map(EXIT_RATIO).fillna(0.875)


def book_stats(x, split, weeks):
    """run_book(12,4) on one split, then the weekly/per-trade statistics the gate reads."""
    y = x[(x.split == split) & x.net.notna() & x.xm.notna()]
    if len(y) < 40:
        return None, None
    rows = [(r.day, int(r.entry_m), int(r.xm), r.symbol, float(r.net), r.wk) for r in y.itertuples()]
    t = pd.DataFrame(run_book(rows, 12, 4), columns=['day', 'em', 'xm', 'symbol', 'net', 'wk'])
    if len(t) < 20:
        return None, None
    w = t.groupby('wk').net.sum().reindex(weeks[split]).fillna(0.0)
    sd = t.net.std(ddof=1)
    return dict(n=len(t), tpw=round(len(t) / max(len(weeks[split]), 1), 1), meanR=round(t.net.mean(), 4),
                t=round(t.net.mean() / (sd / np.sqrt(len(t))), 2) if sd > 0 else 0.0,
                WR=round(float((t.net > 0).mean() * 100), 1), wkR=round(float(w.mean()), 2),
                wkSE=round(float(w.std(ddof=1) / np.sqrt(len(w))), 2),
                green=round(float((w > 0).mean()), 2), worst=round(float(w.min()), 1)), t


def tail_tests(t):
    n = len(t)
    v = t.net.values
    s = np.sort(v)
    return dict(ex1=round(float(s[:max(n - max(int(n * 0.01), 1), 1)].mean()), 4),
                ex5=round(float(s[:max(n - max(int(n * 0.05), 1), 1)].mean()), 4),
                cap3=round(float(np.minimum(v, 3.0).mean()), 4))


def perm_pvalue(cells, nperm, rng):
    """Search-adjusted permutation p: flip the sign of every trade of a randomly chosen half of the DAYS (day-level
    blocks keep the within-day correlation), recompute each cell's TRAIN t, and take the MAX over all cells. p = the
    share of permutations whose max exceeds the observed max. Returns (p, observed max t, null 95th pct)."""
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
    data = load()
    allrows = pd.concat([data['next'], data['rest']], ignore_index=True)
    weeks = {s: sorted(allrows[allrows.split == s].wk.unique()) for s in ('TRAIN', 'VAL', 'TEST')}
    print(f"weeks TRAIN {len(weeks['TRAIN'])} VAL {len(weeks['VAL'])} TEST {len(weeks['TEST'])}", flush=True)
    seen = set(allrows.key.unique())
    unknown = seen - set(FAM_KEYS)
    if unknown:
        print(f'WARNING: keys in the file that are not pre-registered: {sorted(unknown)}', flush=True)
    absent = [k for k in FAM_KEYS if k not in seen]
    if absent:
        print(f'WARNING: pre-registered keys with no scoreable row: {absent}', flush=True)
    keys = FAM_KEYS
    res, cells, cellcount, skipped = [], {}, 0, 0
    for fill in ('next', 'rest'):
        d0 = data[fill]
        for key in keys:
            fam = key.split(' ')[0]
            if fill == 'rest' and fam in CLOSE_TRIGGERED:
                skipped += len(OUTCOMES)          # the whole column of outcomes is impossible for that cell
                continue
            dk = d0[d0.key == key]
            for oname, (rr, why, xm, rp) in OUTCOMES.items():
                cellcount += 1
                x = dk[dk[rp] >= 1.0].copy()
                if not len(x):
                    continue
                x['net'] = net_r(x, rr, why, rp, fill)
                x['xm'] = x[xm]
                st, tr = book_stats(x, 'TRAIN', weeks)
                if st is None:
                    continue
                sv, tv = book_stats(x, 'VAL', weeks)
                row = dict(key=key, fill=fill, outcome=oname, **{f'TRAIN_{k}': v for k, v in st.items()},
                           **{f'VAL_{k}': (sv or {}).get(k) for k in ('n', 'tpw', 'meanR', 't', 'wkR', 'wkSE', 'green')})
                g = tr.groupby('day').net
                # run_book returns trades in day order, so np.repeat(day flips, day counts) aligns with tr.net
                cells[(key, fill, oname)] = dict(t=st['t'], trades=tr.net.values, bydayN=g.size().values)
                row.update(tail_tests(tr))
                res.append(row)
    T = pd.DataFrame(res).sort_values('TRAIN_meanR', ascending=False)
    for col in [c for c in T.columns if c.startswith('VAL_')]:
        T[col] = pd.to_numeric(T[col], errors='coerce')   # cells with no VAL book carry None -> NaN -> gate False
    g1 = T[(T.TRAIN_meanR > 0) & (T.TRAIN_t >= G1_T) & (T.TRAIN_tpw >= MIN_TPW)]
    bar = (len(g1) // 10)
    g2 = g1[(g1.VAL_meanR > 0) & (g1.VAL_t >= G2_T) & (g1.VAL_green >= G2_GREEN)
            & (g1.VAL_wkR >= bar * g1.VAL_wkSE.fillna(0))]
    L = ['# score5 — Stage B/C: family x fill x stop-variant x exit, corrected cost contract, PLAN §1 gate', '',
         f'source: `{SRC}`  |  spread: {"score4 LEGACY band table" if LEGACY_SPREAD else "cost_curve band x time-of-day"}'
         f'  |  target exit charged: {EXIT_RATIO["target"]}x half spread'
         f'  |  entry window: {MIN_ENTRY_M or 570}-841', '',
         f'**cells looked at: {cellcount}** ({len(FAM_KEYS)} family-configs x 2 fills x {len(OUTCOMES)} outcomes = '
         f'{len(FAM_KEYS)*2*len(OUTCOMES)}, minus {skipped} impossible resting cells for the close-triggered '
         f'families F11/F12/F13); cells with a TRAIN book: {len(T)}',
         '', '## every cell, ranked by TRAIN mean net R', T.head(40).to_string(index=False), '',
         f'## G1 — TRAIN mean net R > 0, t >= {G1_T}, >= {MIN_TPW} trades/week: {len(g1)} of {len(T)}',
         g1.to_string(index=False) if len(g1) else '(none)', '',
         f'## G2 — VAL mean net R > 0, t >= {G2_T}, >= {G2_GREEN:.0%} weeks green, weekly R >= {bar} x SE: {len(g2)}',
         g2.to_string(index=False) if len(g2) else '(none)', '']
    if len(g2) and NPERM:
        rng = np.random.default_rng(11)
        p, obs, q95 = perm_pvalue(cells, NPERM, rng)
        L += [f'## permutation, search-adjusted over all {len(cells)} cells ({NPERM} day-level sign-flip draws)',
              f'observed max TRAIN t = {obs:.2f} | null 95th pct = {q95:.2f} | p = {p:.3f}', '']
    if len(g2) and READ_TEST:
        L += ['## TEST — read once', '(run with SCORE5_READ_TEST=1 after the selection above was frozen in writing)']
        for r in g2.itertuples():
            d0 = data[r.fill]
            rr, why, xm, rp = OUTCOMES[r.outcome]
            x = d0[(d0.key == r.key) & (d0[rp] >= 1.0)].copy()
            x['net'] = net_r(x, rr, why, rp, r.fill)
            x['xm'] = x[xm]
            st, tt = book_stats(x, 'TEST', weeks)
            L.append(f'{r.key} | {r.fill} | {r.outcome} | TEST {st}')
            if tt is not None:
                mm = tt.assign(mon=tt.day.str[:7]).groupby('mon').net.agg(['size', 'sum', 'mean']).round(3)
                L.append(mm.to_string())
    elif len(g2):
        L += ['## TEST — NOT read (set SCORE5_READ_TEST=1 only after freezing the selection in writing)']
    open(OUT_MD, 'w').write('\n'.join(L))
    T.to_csv(OUT_CSV, index=False)
    print('\n'.join(L[:10]))
    print(f'\nG1 {len(g1)}  G2 {len(g2)}  -> {OUT_MD}', flush=True)


if __name__ == '__main__':
    main()
