#!/usr/bin/env python3
"""Assemble A/REPORT.md = one-page summary (A4) + the pre-registration written before the runs + the results."""
import os
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
A = 'research/fuckup_audit/A'
pd.set_option('display.width', 400); pd.set_option('display.max_columns', 80); pd.set_option('display.max_rows', 300)
prereg = open(f'{A}/PREREG.md').read()   # frozen copy, written before any run.split('(results appended below after the runs)')[0]
T = pd.read_csv(f'{A}/a0_cells.csv', keep_default_na=False, na_values=[''])
A2 = pd.read_csv(f'{A}/a2_cells.csv', keep_default_na=False, na_values=[''])
A3 = pd.read_csv(f'{A}/a3_cells.csv', keep_default_na=False, na_values=[''])
COLS = ['key', 'exit', 'corr_n', 'corr_tpw', 'gross_meanR', 's4_meanR', 'corr_meanR', 'corrp_meanR', 'gate_meanR',
        'gatep_meanR', 'corr_se', 'corr_t', 'corr_WR', 'corr_wkR', 'corr_green', 'corr_worst', 'gate_n', 'gate_tpw',
        'gate_wkR', 'mix_stop', 'mix_target', 'mix_eod']
def tab(sp, n=None):
    x = T[T.split == sp][COLS].sort_values('corr_meanR', ascending=False)
    return (x if n is None else x.head(n)).to_string(index=False)
a1 = open(f'{A}/a1_gate.md').read()
a4 = open(f'{A}/a4_decomp.md').read()
tr = T[T.split == 'TRAIN']
S = []
S.append("""## A4 — one page: what Stage A found

**The corrected cost contract is worth +0.412 R per trade on average across the 52 cells, and it does not produce a
book.** Under the measured signal-minute NBBO spread of this population (22-76 bps by price band x time of day, from
`lit_review_2026/cost_curve.csv`) instead of score4's banded 190/120/80/60/50 bps, plus the 0.25 entry coefficient the
live fill data supports, **9 of the 52 family-config x exit cells turn positive on TRAIN** (score4: 0 of 52). None of
the nine clears G1, because none has t >= 2.0:

| TRAIN, contract (c) | mean net R | SE | MDE (2.8 SE) | t | trades/wk | wkR | weeks green |
|---|---|---|---|---|---|---|---|
| F2 {"P":0.12} hold | +0.427 | 0.298 | 0.836 | 1.43 | 15.7 | +6.7 | 0.47 |
| F1 {"P":0.12} hold | +0.266 | 0.170 | 0.476 | 1.56 | 15.7 | +4.2 | 0.43 |
| F6 {} hold | +0.051 | 0.038 | 0.106 | 1.34 | 23.4 | +1.2 | 0.57 |
| F6 {} 2r | +0.028 | 0.026 | 0.073 | 1.06 | 25.3 | +0.7 | 0.57 |
| F8 {"N":30} hold | +0.012 | 0.024 | 0.066 | 0.49 | 21.2 | +0.3 | 0.57 |

The two big means (F1/F2 at P=0.12, hold-to-close) are lottery tickets by construction: WR 14-19%, 80-85% of exits are
stops, and the whole mean sits in a handful of runners — they are reported, not proposed. The two cells with a
believable shape (F6 red-to-green, F8 30-min opening-range, WR 44-49%, worst week -10 to -12R) are +0.01 to +0.05 R,
i.e. **inside their own MDE**: at these n the test could not have seen an effect smaller than 0.07-0.11 R, and the
point estimates are smaller than that.

**Time bands (A2) produce the stage's only G1 passes, and they do not replicate.** Restricting F6 to entries >= 10:30
gives TRAIN +0.048 R, t 2.08, 16.9 trades/week (hold) and +0.049 R, t 2.25, 17.1/week (2r) — both clear G1 — and
`F1 {"P":0.12}` hold >= 10:30 gives +0.838 R, t 2.01 at exactly 5.0 trades/week. On VAL the same three cells are
+0.016 (t 0.49), +0.017 (t 0.53) and -0.104 (t -0.38): **0 of 3 clear G2**, so TEST was not read for selection.

**Intraday market state (A3) does not carry.** Over 104 buckets (entries >= 10:00, IWM/SPY open->entry-minute return in
TRAIN-cut terciles and by sign, plus an approximate breadth-so-far), TRAIN->VAL sign agreement is **0.51** — a coin
flip (brd 0.54, iwm 0.52, spy 0.48). The single direction that does repeat is the one `probe_days.md` already found:
for F6, SPY-positive-so-far buckets are +0.082 R (t 2.15) TRAIN and +0.200 R (t 2.25) VAL while SPY-negative buckets are
-0.017 and -0.027. It fails the pre-registered adoption rule by a hair on TRAIN (kept bucket improves +0.046 R vs the
+0.05 R required) and it halves the trade count.

**Where the correction's money comes from** (mean over the 52 TRAIN cells): the spread TABLE +0.357 R, the entry
coefficient 1.0 -> 0.25 +0.066 R, charging the +2R target exit 0.875 x half -0.010 R. In other words H7 was right about
the size of the defect and right about which part of it mattered: **it was the 3.8x-too-wide band table, not the
double-charged entry crossing.**

**Power / phrasing.** In THIS universe (the point-in-time >= 5%-range day population with the causal floor), at THIS
1-minute horizon, over 2025-01-02 -> 2026-09-11, under THIS book (12/day, 4 concurrent) and THIS corrected cost
contract, **no family-config x exit and no time band or intraday-state bucket showed an edge that survives TRAIN
significance and VAL replication**. The smallest per-trade effect the headline tests could have seen at 80% power is
**0.066 R (F8 N=30), 0.106 R (F6), 0.476 R (F1 P=0.12)**; at the book level that is roughly 1.4 / 2.5 / 7.5 R per week.
Effects below those sizes are invisible here and are not excluded. The largest positive t over 52 cells is 1.56, BELOW
the ~2.7 that 52 independent nulls would be expected to produce, so no permutation test was needed (and none was run:
nothing cleared G2, which is what the pre-registration conditioned it on).

**What goes to Stage C/D.** (1) The corrected contract (c) replaces score4's for every later stage — it is the more
accurate model whether or not it flips a cell; (c') when the engine's take-profit leg genuinely rests. (2) Base
families: **F6 {} and F8 {"N":30}** (positive gross AND positive corrected net on TRAIN, WR 44-49%, the smallest worst
weeks in the grid), with `F8 {"N":15}` as the third. F5 stays dropped; F1/F2 P=0.12 are carried only as a tail-risk
reference. (3) Time window: **entries >= 10:00** for F6/F8 (the >= 10:30 cut is where F6's TRAIN t goes above 2 and it
does not hold on VAL — carry the window, do not treat it as validated). (4) The intraday index-return sign at the entry
minute goes to Stage D as a FEATURE, never as a standalone day filter, per PLAN §3 H3.

**Caveat that limits every number above**: the corrected spread is a per-cell MEDIAN (25 cells, 50-120 sampled quotes
each) applied to every candidate in the cell, so within-cell dispersion is uncharged, and the "live liquidity gate" (d)
is therefore not really a liquidity filter — with spread constant inside a cell, `spread/R <= 0.15` reduces to a
minimum-R floor of 1.50% ($5-10 after 13:00) to 5.07% ($200+ 10-11am). Contract (d)'s gains are the H7 stop-width
mechanism a second time, not a liquidity effect, and they are labelled as such.
""")
S.append('\n---\n\n## A0 — every family-config x exit under the six contracts\n')
S.append(f'population 446,012 rows (parity with score4: identical) | 26 keys | weeks TRAIN 53 / VAL 22 / TEST 14 |'
         f' exit mix of the population (2r): stop 46.0%, eod 34.2%, target 19.9%\n')
S.append('**Parity anchor (CLAUDE.md independent-check rule, coding-error half):** re-deriving contract (b) reproduces '
         '`bf_zero2/score4_results.csv` cell for cell — 52 of 52 merged, max |delta n| = 0, max |delta mean R| = 0.000, '
         'max |delta weekly R| = 0.05 (score4 rounds weekly R to 1 dp, this run to 2). `A/a0_parity.csv`. This catches '
         'coding errors only; the specification caveats are in §Caveats.\n')
S.append('corrected spread table, median signal-minute NBBO in bps of price (rows = price band, cols = time band):\n')
cc = pd.read_csv('research/lit_review_2026/cost_curve.csv', keep_default_na=False, na_values=[''])
cc = cc[cc.n_q > 0]; cc['sp_bps'] = cc.spread / cc.price * 1e4
S.append('```\n' + cc.pivot_table(index='pb', columns='hb', values='sp_bps', aggfunc='median').round(1).to_string() +
         '\n\nn quotes per cell:\n' + cc.pivot_table(index='pb', columns='hb', values='sp_bps', aggfunc='size').to_string() + '\n```\n')
S.append('score4 charged, for the same names: 190 / 120 / 80 / 60 / 50 bps at $5-10 / $10-20 / $20-50 / $50-100 / $100+,'
         ' at every hour.\n')
for sp in ('TRAIN', 'VAL', 'TEST'):
    note = '' if sp != 'TEST' else ('\n*(TEST is shown because A0 selects nothing and `score4_tables.md` already '
                                    'published TEST for all 52 of these cells on 9/16. It is not used to choose '
                                    'anything in Stage A.)*\n')
    S.append(f'### {sp}{note}\n```\n{tab(sp)}\n```\n')
S.append('\n---\n\n## A1 — the re-gate\n\n' + a1.split('\n', 2)[2])
S.append('\n---\n\n## A2 — time bands (contract (c), book re-run inside each window)\n')
S.append('The >= 10:30 window for the two keys involved in Stage A\'s only G1 passes. G1 passes: `F6 {}` hold '
         '(t 2.08), `F6 {}` 2r (t 2.25) and `F1 {"P":0.12}` hold (t 2.01 at exactly 5.0 trades/week). `F1` 2r is '
         'shown for contrast. Every one of them fails G2 on VAL:\n')
sel = A2[(A2.window == '>=10:30') & (A2.key.isin(['F6 {}', 'F1 {"P": 0.12}'])) & (A2.exit.isin(['hold', '2r']))]
S.append('```\n' + sel[['key', 'exit', 'window', 'split', 'corr_n', 'corr_tpw', 'corr_meanR', 'corr_se', 'corr_t',
                        'corr_WR', 'corr_wkR', 'corr_green', 'corr_worst']].to_string(index=False) + '\n```\n')
S.append('Full A2 grid (6 keys x 2 exits x 4 windows x 2 splits):\n')
for sp in ('TRAIN', 'VAL'):
    x = A2[A2.split == sp].sort_values(['key', 'exit', 'window'])
    S.append(f'### {sp}\n```\n' + x[['key', 'exit', 'window', 'corr_n', 'corr_tpw', 'gross_meanR', 'corr_meanR',
                                     'corr_se', 'corr_t', 'corr_WR', 'corr_wkR', 'corr_green', 'corr_worst',
                                     'gate_meanR', 'mix_stop', 'mix_target', 'mix_eod']].to_string(index=False) + '\n```\n')
S.append('\n---\n\n## A3 — intraday market state at the entry minute (entries >= 10:00)\n')
piv = A3.pivot_table(index=['key', 'exit', 'feat', 'bucket'], columns='split', values=['n', 'meanR', 't'])
piv.columns = [f'{a}_{b}' for a, b in piv.columns]; piv = piv.reset_index()
piv['sign_agree'] = np.sign(piv.meanR_TRAIN) == np.sign(piv.meanR_VAL)
S.append(f'104 buckets, TRAIN->VAL sign agreement **{piv.sign_agree.mean():.2f}**. By feature:\n\n```\n' +
         piv.groupby('feat').sign_agree.agg(['mean', 'size']).round(2).to_string() + '\n```\n')
S.append('```\n' + piv[['key', 'exit', 'feat', 'bucket', 'n_TRAIN', 'meanR_TRAIN', 't_TRAIN', 'n_VAL', 'meanR_VAL',
                        't_VAL', 'sign_agree']].to_string(index=False) + '\n```\n')
S.append('\n---\n\n## A4 support — decomposition of the correction\n\n' + a4.split('\n', 2)[2])
S.append("""
---

## Caveats — what these numbers cannot say

1. **The corrected spread is a cell median, not a per-trade quote.** 25 (price band x time band) cells, 50-120 sampled
   signal-minute NBBO medians each, from `build_cost_curve.py`'s stratified sample of THIS candidate pool (its own
   filter was `price >= 5, r_pct >= 0.5`, without the F5-F10 range floor — a slightly wider pool than score4's).
   Within-cell dispersion is uncharged. PLAN §3 H7 Stage B (per-trade NBBO for every booked trade of a surviving cell)
   is the fix and is unchanged by this stage.
2. **The 0.25 entry coefficient is an inference, not a measurement on this population.** It comes from 191 live entries
   in `trades.db` (median 0.0 bps vs the fill-moment quote, 46% at or below the mid) in names that passed live liquidity
   gates the research universe never applied. If the untraded remainder of this universe crosses a full half spread,
   contract (b)'s entry charge is right for those names and every positive cell above disappears. Contract (c) is
   therefore an UPPER bound on what the account keeps, and (b) a lower bound; both are reported side by side for that
   reason.
3. **Everything here inherits the fill convention under audit.** `candidates3.csv` only contains signals whose next-bar
   open came back under level x 1.006, i.e. the population is conditioned on cheap fills (probe_stops §3). H2's resting
   fill (+30% trades) cannot be tested on this file; Stage B rebuilds pass 1 with both fills.
4. **No stop/exit variation.** Stage A re-scores the two exits that exist in `candidates3.csv` (hold, +2R on a close).
   H1's stop x exit cross needs the re-walk in Stage B.
5. **Early-close days are not excluded** (score4 did not exclude them, and this stage reproduces score4's population
   exactly so the difference is attributable to the cost contract alone).
6. **Breadth-so-far is an approximation** and is described as one in the pre-registration: it is breadth inside the
   study's own candidate pool, deduped on (day, symbol, entry_m) across all 26 family-configs, and each candidate's
   `dist_open_pct` is its value at its own entry minute.
7. **TEST.** A0 reprints TEST because `score4_tables.md` already published it for these 52 cells; nothing in Stage A
   selects on it. No cell reached G2, so the pre-registered TEST read (G3), the tail tests, the per-month tables and the
   permutation p were not triggered.

## Cell count

| block | cells looked at |
|---|---|
| A0 | 52 family-config x exit cells x 3 splits x 6 contracts = 936 numbers; **52 decision-relevant** (contract (c), TRAIN) |
| A1 | the same 52 under G1 for contract (c), and 52 again for the secondary contract (d) = 104 gate evaluations; 1 VAL read (the single (d) G1 survivor) |
| A2 | 6 keys x 2 exits x 4 windows x 2 splits = 96 (of which 12 duplicate A0's ALL-window cells, so 36 new window cells per split) |
| A3 | 104 buckets x 2 splits = 208 |
| Stage A total | **~1,240 numbers, 192 pre-declared decision cells** |
| program to date | 52 (score4) + ~100 (probe_stops) + 156 (probe_days) + 192 (Stage A) = **~500 cells** |

## Scripts (all under `research/fuckup_audit/A/`, each run `nice -n 10` with `ulimit -v 1300000`)

| script | what | runtime |
|---|---|---|
| `PREREG.md` | the pre-registration block, frozen before the first run and pasted verbatim into §0 | - |
| `extract_pop_a.py` | chunked scan of the 678 MB `candidates3.csv` -> `pop_a.csv` (446,012 rows = score4's population exactly) | 20 s |
| `acore.py` | ONE definition of the population, the six cost contracts and the book (`run_book(rows, 12, 4)`) | - |
| `a0_score.py` | A0: 26 keys x 2 exits x 3 splits under all six contracts + the score4 parity anchor -> `a0_cells.csv`, `a0_parity.csv`, `a0_tables.md` | 3 min |
| `a1_gate.py` | A1: G1/G2 under (c) and (d), closest miss per family with its MDE -> `a1_gate.md` | 5 s |
| `a2_bands.py` | A2: time bands -> `a2_cells.csv`, `a2_tables.md` | 1 min |
| `a3_state.py` | A3: IWM/SPY open->entry-minute return and breadth-so-far buckets -> `a3_cells.csv`, `a3_tables.md` | 2 min |
| `a4_decomp.py` | A4: decomposition of the score4 -> corrected move -> `a4_decomp.csv/.md` | 1 min |
| `assemble_report.py` | builds this REPORT.md from the pre-registration + the outputs | 5 s |
""")
body = '\n'.join(S)
head, rest = prereg.split('## 0. PRE-REGISTRATION', 1)
summary, results = body.split('\n---\n\n## A0 —', 1)
open(f'{A}/REPORT.md', 'w').write(head + summary + '\n---\n\n## 0. PRE-REGISTRATION' + rest +
                                  '\n---\n\n## A0 —' + results + '\n')
print('written', os.path.getsize(f'{A}/REPORT.md'))
