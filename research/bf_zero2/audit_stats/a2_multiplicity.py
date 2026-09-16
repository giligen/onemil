#!/usr/bin/env python3
"""a2 — Q1: multiplicity. How likely is it that SOME cell of the search shows t > 2.8 on all three
splits when no cell has an edge?

Method (no formula — the actual data):
  1. Rebuild all 108 score3 cells (27 family-configs x 2 exits x 2 book sizes) exactly as score3.py did.
  2. Reduce each cell to per-DAY sufficient statistics (n, sum R, sum R^2) so a day-block bootstrap
     can recompute the exact per-split t of every cell in one matrix product.
  3. Impose H0 by centering each cell's trade R within each split (zero mean, real variance,
     real cross-cell correlation, real autocorrelation).
  4. Circular block-bootstrap DAYS (block = 5 trading days = one week), the SAME day draws applied to
     every cell simultaneously, so the joint distribution keeps the cells' correlation.
  5. Report P(at least one cell has t > 2.8 on all three splits) and the null distribution of
     max_cells min_splits t. Also the effective number of independent cells from the eigenvalues of
     the daily-R correlation matrix.
"""
import os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
from trading.hod_break import run_book
A = 'research/bf_zero2/audit_stats'
rng = np.random.default_rng(917)
OUT = []
def P(*a):
    s = ' '.join(str(x) for x in a); OUT.append(s); print(s, flush=True)

c = pd.read_parquet(f'{A}/pool.parquet')
c['day'] = c.day.astype(str); c['symbol'] = c.symbol.astype(str)
c['key'] = c.key.astype(str)
c['exit_m_e1c'] = c.exit_m_e1c.fillna(955).astype(int)
c['entry_m'] = c.entry_m.astype(int)
SPL = ('TRAIN', 'VAL', 'TEST')
days = {s: np.array(sorted(c.day[c.split == s].unique())) for s in SPL}
didx = {s: {d: i for i, d in enumerate(days[s])} for s in SPL}
P(f'pool rows {len(c):,} | keys {c.key.nunique()} | days ' + str({s: len(days[s]) for s in SPL}))

# ---------------- 1-2. rebuild every cell, reduce to per-day (n, sum, sumsq) ----------------
cells = []
stat = {s: [] for s in SPL}          # list of (n, s, ss) arrays, one row per cell
obs = []
for key, dk in c.groupby('key', observed=True):
    for tag in ('e1c', 'e4'):
        col, xmc = f'net_{tag}', f'exit_m_{tag}'
        dkk = dk[dk[col].notna()]
        for N in (4, 20):
            per = {}
            ok = True
            for s in SPL:
                x = dkk[dkk.split == s]
                if len(x) < 50: ok = False; break
                rows = [(r.day, int(r.entry_m), int(getattr(r, xmc)), r.symbol, getattr(r, col)) for r in x.itertuples()]
                t = pd.DataFrame(run_book(rows, N, N), columns=['day', 'em', 'xm', 'sym', 'net'])
                if len(t) < 50: ok = False; break
                g = t.groupby('day').net.agg(['size', 'sum', lambda v: (v ** 2).sum()])
                g.columns = ['n', 'sm', 'ss']
                arr = np.zeros((3, len(days[s])))
                ii = np.array([didx[s][d] for d in g.index])
                arr[0, ii] = g.n.values; arr[1, ii] = g.sm.values; arr[2, ii] = g.ss.values
                per[s] = arr
            if not ok: continue
            cells.append((key, tag, N))
            for s in SPL: stat[s].append(per[s])
            o = {}
            for s in SPL:
                n, sm, ss = per[s].sum(axis=1)
                m = sm / n; v = (ss - sm ** 2 / n) / (n - 1)
                o[s] = (n, m, m / np.sqrt(v / n))
            obs.append(o)
P(f'cells rebuilt: {len(cells)}')

STAT = {s: np.stack(stat[s]) for s in SPL}   # (ncell, 3, nday)
K = len(cells)
O = pd.DataFrame([{'key': k, 'exit': t, 'slots': N,
                   **{f'{s}_n': obs[i][s][0] for s in SPL},
                   **{f'{s}_meanR': obs[i][s][1] for s in SPL},
                   **{f'{s}_t': obs[i][s][2] for s in SPL}} for i, (k, t, N) in enumerate(cells)])
O['min_t'] = O[[f'{s}_t' for s in SPL]].min(axis=1)
O['all_pos'] = (O[[f'{s}_meanR' for s in SPL]] > 0).all(axis=1)
O = O.sort_values('min_t', ascending=False)
O.to_csv(f'{A}/cells_observed.csv', index=False)
P('\n## observed — the 12 cells with the highest min-across-splits t')
P(O.head(12).round(3).to_string(index=False))
P(f'\ncells positive on all three splits: {int(O.all_pos.sum())} of {K}')
P(f'cells with t > 2.8 on all three: {int((O.min_t > 2.8).sum())}   '
  f'| t > 2.5: {int((O.min_t > 2.5).sum())} | t > 2.0: {int((O.min_t > 2.0).sum())}')
P(f'the winner: {O.iloc[0].key} {O.iloc[0].exit} slots {int(O.iloc[0].slots)} min_t {O.iloc[0].min_t:.2f}')

# ---------------- effective number of independent cells ----------------
P('\n## effective number of independent tests (eigenvalues of the daily-R correlation matrix)')
for s in SPL + ('ALL',):
    if s == 'ALL':
        M = np.concatenate([STAT[q][:, 1, :] for q in SPL], axis=1)
    else:
        M = STAT[s][:, 1, :]
    Msd = M.std(axis=1); keep = Msd > 0
    C = np.corrcoef(M[keep])
    ev = np.linalg.eigvalsh(C); ev = np.clip(ev, 0, None)
    meff_cn = ev.sum() ** 2 / (ev ** 2).sum()                     # Cheverud/Nyholt
    meff_li = sum(min(e, 1) + (e >= 1) * 0 for e in ev)           # Li-Ji lower piece
    meff_lj = sum((e >= 1) + (e - np.floor(e)) for e in ev if e > 0)
    P(f'  {s:6} cells {keep.sum():3d}  mean |corr| {np.abs(C[np.triu_indices_from(C,1)]).mean():.3f}  '
      f'M_eff(Cheverud) {meff_cn:5.1f}  M_eff(Li-Ji) {meff_lj:5.1f}  '
      f'eigenvalues >1: {(ev>1).sum()}, top-5 share {np.sort(ev)[::-1][:5].sum()/ev.sum():.1%}')

# ---------------- 3-4. block bootstrap under H0 ----------------
NB, BLOCK = 10_000, 5
cent = {}
for s in SPL:
    n = STAT[s][:, 0, :]; sm = STAT[s][:, 1, :]; ss = STAT[s][:, 2, :]
    mu = (sm.sum(axis=1) / n.sum(axis=1))[:, None]
    cent[s] = np.stack([n, sm - n * mu, ss - 2 * mu * sm + n * mu ** 2], axis=1)

def draw_counts(nday, nb, block):
    nblk = int(np.ceil(nday / block))
    st = rng.integers(0, nday, size=(nb, nblk))
    idx = (st[:, :, None] + np.arange(block)[None, None, :]).reshape(nb, -1)[:, :nday] % nday
    C = np.zeros((nb, nday), dtype=np.float32)
    np.add.at(C, (np.repeat(np.arange(nb), nday), idx.ravel()), 1.0)
    return C

T = {}
for s in SPL:
    nday = len(days[s])
    Cnt = draw_counts(nday, NB, BLOCK)          # (NB, nday)
    n = cent[s][:, 0, :] @ Cnt.T                # (K, NB)
    sm = cent[s][:, 1, :] @ Cnt.T
    ss = cent[s][:, 2, :] @ Cnt.T
    with np.errstate(divide='ignore', invalid='ignore'):
        m = sm / n
        v = (ss - sm ** 2 / n) / (n - 1)
        T[s] = m / np.sqrt(v / n)
    T[s] = np.nan_to_num(T[s], nan=0.0)
    P(f'  bootstrapped {s}: t matrix {T[s].shape}, median n per cell {np.median(n):.0f}')

MIN = np.minimum(np.minimum(T['TRAIN'], T['VAL']), T['TEST'])    # (K, NB)
best = MIN.max(axis=0)                                            # per replicate

P('\n## Q1 — null distribution of the SEARCH (10,000 day-block resamples, every cell centered to zero mean)')
for thr in (2.0, 2.5, 2.78, 2.8, 3.0, 3.5):
    P(f'  P(at least one of the {K} cells has t > {thr:4.2f} on ALL THREE splits) = {(best > thr).mean():.4f}')
P(f'  P(a SINGLE pre-specified cell clears t > 2.8 on all three)          = {(MIN > 2.8).mean():.6f}')
P(f'  null quantiles of max-over-cells min-over-splits t: '
  f'50% {np.percentile(best,50):.2f} | 90% {np.percentile(best,90):.2f} | 95% {np.percentile(best,95):.2f} | 99% {np.percentile(best,99):.2f}')
P(f'  observed max-over-cells min-over-splits t = {O.min_t.iloc[0]:.2f}  ->  search-adjusted p = {(best >= O.min_t.iloc[0]).mean():.4f}')

# how many cells positive on all three under the null (calibration check against the observed 24)
pos = ((T['TRAIN'] > 0) & (T['VAL'] > 0) & (T['TEST'] > 0)).sum(axis=0)
P(f'  null #cells positive on all three: mean {pos.mean():.1f}, 95th pct {np.percentile(pos,95):.0f} '
  f'(observed {int(O.all_pos.sum())}) — the cells are strongly correlated, so this is NOT ~K/8')

# single-cell p, and the census extrapolation
p1 = (MIN > 2.8).mean()
meff = None
M = np.concatenate([STAT[q][:, 1, :] for q in SPL], axis=1)
keep = M.std(axis=1) > 0
ev = np.clip(np.linalg.eigvalsh(np.corrcoef(M[keep])), 0, None)
meff = ev.sum() ** 2 / (ev ** 2).sum()
P(f'\n## the wider census (score3 is only the last stage)')
for name, ncell in (('score3 grid (this bootstrap)', K), ('score2, 275 cells', 275),
                    ('bf_zero 8fam x 21cfg x 4 exits', 672), ('lit-review cells (RESULTS.md census)', 90 + 63),
                    ('all of the above', K + 275 + 672 + 153)):
    # scale the effective count by the same independence ratio measured inside score3
    ratio = meff / K
    me = ncell * ratio
    P(f'  {name:38} cells {ncell:5d}  M_eff ~ {me:6.1f}  P(some cell t>2.8 on all 3) ~ {1-(1-p1)**me:.4f}')

np.save(f'{A}/null_best_mint.npy', best)
open(f'{A}/q1_multiplicity.md', 'w').write('\n'.join(OUT))
print('DONE', flush=True)
