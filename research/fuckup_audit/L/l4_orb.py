#!/usr/bin/env python3
"""Stage L — the B5 (ORB B+) cells.

ORB's slot rule is not `run_book`; it is the pipeline's own rank -> family/super-group dedup -> top N,
followed by the shipped post-selection vetoes.  So a Stage-L filter is applied to the CANDIDATE
UNIVERSE before ranking (both the features CSV and the resim dump, whose key sets the pipeline
requires to be identical), and the pipeline's own selection then refills the freed slots by
construction.  Nothing in orb.yaml, no threshold, no z-param and no mult is touched: every run is
`study_orb_pipeline_static_lock.py` with ORB_BT_RESIM_CACHE, N=3 and the $10K-stage sizing.

  T1  skipped — the PDR veto already ships (prev_day_range_pct >= 11 in B+)
  T2  keep range_size_pct >= 5            (the 09:30-09:34 range IS this book's range-so-far)
  T3  keep SPY 09:30-open -> 09:34-close return > 0
  T4  first pick of the day only: ORB_BT_N=1 with ORB_BT_ACCOUNT=3,333.33 so per-position sizing is unchanged
  T5  skipped — ORB enters at 09:35 only
  T6  10-minute time stop: pnl/pnl_pct rewritten on the fills it fires on, universe unchanged
  T7  fill-minute volume rule: same
  T8  keep range_size_pct >= 3

Writes L/orb_bars.csv, L/orb/<cell>_{feat,dump,book,monthly}.csv, L/orb_cells.csv.
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lcore as C                                                             # noqa: E402
from l2_bars import Tape                                                      # noqa: E402

L = C.L
OD = f'{L}/orb'
FEAT = 'analysis_results/orb_features_20260916_2053.csv'
DUMP = f'{C.FA}/D1_orb/candidates_dump.csv'
OLD_POS = 50_000.0
TRIG = 1.003          # the pre-placed stop-limit's trigger, range_high x (1 + 30 bps)
FORCED = 0.003        # 0.3% of R for the forced crossing (orb_timestop_validation convention)


def rd(p):
    return pd.read_csv(p, keep_default_na=False, na_values=[''], dtype={'symbol': str, 'date': str})


# --------------------------------------------------------------------------- the tape walk
def orb_bars():
    out = f'{L}/orb_bars.csv'
    if os.path.exists(out):
        return rd(out)
    F = rd(FEAT)
    f = F[F.entered == 1].copy().sort_values(['symbol', 'date'])
    tape = Tape()
    rows = []
    for i, r in enumerate(f.itertuples()):
        if i % 2000 == 0:
            C.log(f'  orb bars {i}/{len(f)}')
        b = tape.get(r.symbol, r.date)
        o = dict(symbol=r.symbol, date=r.date)
        if b:
            rng = [m for m in b if 570 <= m < 575]
            if len(rng) >= 3:
                hi = max(b[m][1] for m in rng)
                lo = min(b[m][2] for m in rng)
                R = hi - lo
                after = sorted(m for m in b if m >= 575 and b[m][1] >= hi * TRIG)
                if R > 0 and after:
                    fm = after[0]
                    entry = float(r.entry_price) if r.entry_price == r.entry_price and r.entry_price > 0 \
                        else hi * TRIG
                    o['R_range'] = R
                    o['fm'] = fm
                    o['entry'] = entry
                    pre = [b[m][4] for m in sorted(m for m in b if m < fm)][-5:]
                    o['vol_ft'] = (b[fm][4] / np.mean(pre)) if len(pre) == 5 and np.mean(pre) > 0 else np.nan
                    nx = [m for m in b if m > fm]
                    if nx:
                        m1 = min(nx)
                        o['m_next'], o['o_next'] = m1, b[m1][0]
                    t10 = [m for m in b if m >= fm + 10]
                    if t10:
                        m10 = min(t10)
                        o['m_p10'], o['o_p10'] = m10, b[m10][0]
        rows.append(o)
    tape.close()
    d = pd.DataFrame(rows)
    d.to_csv(out, index=False)
    C.log(f'orb bars: {len(d)} fills, R+fill minute on {d.R_range.notna().sum()}, '
          f'vol ratio on {d.vol_ft.notna().sum()}, fill+10 bar on {d.o_p10.notna().sum()}')
    return d


def rewrite_exit(F, D, W, rule):
    """T6 / T7: recompute pnl_pct and pnl on the fills the rule fires on. Universe unchanged."""
    w = W.dropna(subset=['R_range', 'entry']).set_index(['symbol', 'date'])
    keyF = list(zip(F.symbol, F.date))
    newpct = F.pnl_pct.astype(float).copy().values
    fired = 0
    for i, k in enumerate(keyF):
        if k not in w.index or F.entered.values[i] != 1:
            continue
        row = w.loc[k]
        R, entry = float(row.R_range), float(row.entry)
        if rule == 'T6':
            o, m = row.o_p10, row.m_p10
            if o != o:
                continue
            prog = (float(o) - entry) / R
            if prog >= 0.25:
                continue
        else:                                   # T7
            vf, o = row.vol_ft, row.o_next
            if vf != vf or o != o or vf >= 1.5:
                continue
            prog = (float(o) - entry) / R
        newpct[i] = (prog - FORCED) * R / entry * 100.0
        fired += 1
    out = []
    for X in (F, D):
        Y = X.copy()
        m = dict(zip(keyF, newpct))
        Y['pnl_pct'] = [m[(s, d)] for s, d in zip(Y.symbol, Y.date)]
        sh = np.maximum(1, (OLD_POS / Y.entry_price.replace(0, np.nan)).fillna(1).astype(int))
        Y['pnl'] = np.where(Y.entered == 1, sh * Y.entry_price * Y.pnl_pct / 100.0, 0.0)
        out.append(Y)
    return out[0], out[1], fired


def run(tag, F, D, n=3, account=10000.0):
    os.makedirs(OD, exist_ok=True)
    fp, dp = f'{OD}/{tag}_feat.csv', f'{OD}/{tag}_dump.csv'
    F.to_csv(fp, index=False)
    D.to_csv(dp, index=False)
    env = dict(os.environ, ORB_BT_FEATURES_CSV=fp, ORB_BT_RESIM_CACHE=dp, ORB_BT_RISK='375',
               ORB_BT_N=str(n), ORB_BT_ACCOUNT=str(account),
               ORB_BT_BOOK_OUT=f'{OD}/{tag}_book.csv', ORB_BT_MONTHLY_OUT=f'{OD}/{tag}_monthly.csv')
    r = subprocess.run([sys.executable, '-u', 'study_orb_pipeline_static_lock.py'],
                       env=env, capture_output=True, text=True)
    open(f'{OD}/{tag}.log', 'w').write(r.stdout + '\n---STDERR---\n' + r.stderr)
    if r.returncode != 0:
        C.log(f'  {tag}: FAILED rc={r.returncode}\n{r.stderr[-800:]}')
        return None
    b = rd(f'{OD}/{tag}_book.csv')
    C.log(f'  {tag}: {len(b)} picks, {int((b.entered == 1).sum())} fills, ${b._sized_pnl.sum():,.2f}')
    return b


def stat(b, sp):
    x = b[C.split_of(b.date) == sp]
    if not len(x):
        return dict(n=0)
    r = (x.pnl_pct / x.range_size_pct).astype(float)
    v = np.sort(r.values)
    n = len(v)
    return dict(n=n, meanR=float(r.mean()), usd=float(x._sized_pnl.sum()),
                permo=float(x._sized_pnl.sum() / C.months_in(sp)),
                fills=int((x.entered == 1).sum()),
                ex5=float(v[:max(n - max(int(n * 0.05), 1), 1)].mean()),
                cap3=float(np.minimum(v, 3.0).mean()),
                se=float(r.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan)


def main():
    W = orb_bars()
    F0, D0 = rd(FEAT), rd(DUMP)
    assert set(zip(F0.symbol, F0.date)) == set(zip(D0.symbol, D0.date)), 'key sets differ'
    etf = pd.read_csv(C.ETF_RET, keep_default_na=False, na_values=[''], dtype={'day': str, 'symbol': str})
    spy574 = etf[(etf.symbol == 'SPY') & (etf.m == 574)].set_index('day').ret
    C.log(f'T3 SPY coverage on ORB dates: {F0.date.map(spy574).notna().mean()*100:.2f}%')

    cells = {}
    cells['base'] = (F0, D0, 3, 10000.0)
    def univ(mask):
        """A universe filter: the same rows removed from the features CSV and from the resim dump."""
        k = F0.loc[mask, ['symbol', 'date']]
        return F0[mask.values if hasattr(mask, 'values') else mask], D0.merge(k, on=['symbol', 'date'])

    m2 = (F0.range_size_pct >= 5)
    cells['T2'] = (*univ(m2), 3, 10000.0)
    s = F0.date.map(spy574)
    m3 = ~(s <= 0)                                   # fail-open on a missing index return
    cells['T3'] = (*univ(m3), 3, 10000.0)
    cells['T4'] = (F0, D0, 1, 3333.333333333333)
    exitF = {}
    for t in ('T6', 'T7'):
        Fx, Dx, fired = rewrite_exit(F0, D0, W, t)
        C.log(f'{t}: rewrote {fired} of {int((F0.entered == 1).sum())} fills')
        exitF[t] = (Fx, Dx)
        cells[t] = (Fx, Dx, 3, 10000.0)
    m8 = (F0.range_size_pct >= 3)
    cells['T8'] = (*univ(m8), 3, 10000.0)

    books = {}
    for tag, (F, D, n, acct) in cells.items():
        C.log(f'RUN {tag}  universe {len(F)}')
        books[tag] = run(tag, F, D, n, acct)

    def row_of(tag, b, base):
        row = dict(book='B5', cell=tag, picks=len(b), fills=int((b.entered == 1).sum()))
        for sp in ('TRAIN', 'VAL'):
            st, bs = stat(b, sp), stat(base, sp)
            row[f'{sp}_n'] = st['n']
            row[f'{sp}_base'] = round(bs.get('meanR', np.nan), 4)
            row[f'{sp}_mean'] = round(st.get('meanR', np.nan), 4)
            row[f'{sp}_imp'] = round(st.get('meanR', np.nan) - bs.get('meanR', np.nan), 4)
            row[f'{sp}_permo'] = round(st.get('permo', 0), 0)
            row[f'{sp}_ex5'] = round(st.get('ex5', np.nan), 4)
            row[f'{sp}_cap3'] = round(st.get('cap3', np.nan), 4)
        row['pass'] = int(row['TRAIN_imp'] == row['TRAIN_imp'] and row['TRAIN_imp'] >= 0.03
                          and row['VAL_imp'] == row['VAL_imp'] and row['VAL_imp'] >= 0
                          and row['VAL_mean'] > 0)
        return row

    base = books['base']
    singles = {t: row_of(t, b, base) for t, b in books.items() if b is not None and t != 'base'}
    cand = [t for t, r in singles.items() if r['VAL_imp'] == r['VAL_imp'] and r['VAL_imp'] >= 0]
    cand.sort(key=lambda t: -singles[t]['TRAIN_imp'])
    if len(cand) >= 2:
        a, bq = cand[:2]
        # build the pair: universe filters compose on F0/D0; exit rewrites compose on the filtered frames
        Fp, Dp, n, acct = F0, D0, 3, 10000.0
        masks = {'T2': m2, 'T3': m3, 'T8': m8}
        um = None
        for t in (a, bq):
            if t in masks:
                um = masks[t] if um is None else (um & masks[t])
        if um is not None:
            Fp, Dp = univ(um)
        for t in ('T7', 'T6'):                      # chronological: fill+1 before fill+10
            if t in (a, bq):
                Fp, Dp, fired = rewrite_exit(Fp, Dp, W, t)
        if 'T4' in (a, bq):
            n, acct = 1, 3333.333333333333
        tag = 'STACK ' + '+'.join(sorted([a, bq]))
        books[tag] = run(tag, Fp, Dp, n, acct)
    else:
        C.log(f'B5 stack NOT run — only {len(cand)} filter(s) with VAL improvement >= 0')

    rows = []
    for tag, b in books.items():
        if b is None:
            continue
        rows.append(row_of(tag, b, base))
    d = pd.DataFrame(rows)
    d.to_csv(f'{L}/orb_cells.csv', index=False)
    C.log(d.to_string(index=False))


if __name__ == '__main__':
    main()
