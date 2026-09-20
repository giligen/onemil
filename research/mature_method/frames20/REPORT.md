# frames20 — REPORT (F57)

**Cell F57 — the POWERED placebo test of the passive mirror short (5 matched controls per signal).**
PREREG frozen at `4ff9e674092092e2b134e067bf719bb77a3271c5` before any scoring (`FREEZE.md`).
TEST (`day >= 2026-06-01`) never opened; walker and scorer both assert it.
Programme cell count **1,268 -> 1,270** (`w=5`, `w=10`, both at `k=1.0 %`).

## 0. One correction to the brief, recorded in the PREREG before scoring

The brief called frames19's control "ONE matched non-signal name per signal"; `frames19/walk19.py` used
**`NMATCH = 3`** (its own PREREG SS1: "up to **3** symbols are drawn"). F57's increase is **3 -> 5**, not
1 -> 5 — power gain ~1.29x, not ~2.2x. Nothing else changed: same match rule, same entry/exit/cost/rail
stack, `swalk` verbatim, seed 19 -> 57. Control rows **868 -> 1,365** (+57 %); filled VAL controls at
`w=5` **13 -> 27**. Bookkeeping: 2,242/2,294 S rows (97.7 %) matched; 10,514 draws -> 1,365 rows past the
`gate5`/`price >= $5` rails. **Shortfall**: 305 S rows (13.6 % of matched) had a bucket < 5, 696 draws
missing; bucket size p10/p50/p90 = 4/13/27, mean 14.7.

## 1. PRIMARY — S vs P1x5, per window (filled books, net of measured cost)

| w | split | pop | n / rows | fill | gross | net | t(day) | t(iid) | ex-top-5 % |
|---|---|---|---|---|---|---|---|---|---|
| 5 | TRAIN | S | 219 / 1,366 | 16.0 % | +0.302 | **+0.211** | +2.35 | +2.43 | +0.051 |
| 5 | TRAIN | P1x5 | 99 / 879 | 11.3 % | +0.183 | **+0.121** | +1.27 | +1.55 | +0.028 |
| 5 | VAL | S | 133 / 928 | 14.3 % | +0.471 | **+0.372** | +2.36 | +2.17 | +0.081 |
| 5 | VAL | P1x5 | 27 / 486 | 5.6 % | +0.451 | **+0.390** | +1.52 | +1.67 | +0.232 |
| 10 | TRAIN | S | 343 / 1,366 | 25.1 % | +0.325 | **+0.246** | +3.47 | +3.73 | +0.090 |
| 10 | TRAIN | P1x5 | 145 / 879 | 16.5 % | +0.179 | **+0.118** | +1.74 | +1.69 | +0.004 |
| 10 | VAL | S | 225 / 928 | 24.2 % | +0.427 | **+0.329** | +3.04 | +2.84 | +0.078 |
| 10 | VAL | P1x5 | 50 / 486 | 10.3 % | +0.213 | **+0.151** | +0.86 | +1.04 | +0.020 |

## 2. The difference — S - P1x5 (the cell)

| w | split | S - P1x5 | t(day) | t(iid) | **MDE80** | ex-top-5 % of the diff (t) | touch-cond. (t) | fill S vs P1 | pass? |
|---|---|---|---|---|---|---|---|---|---|
| 5 | TRAIN | **+0.089** | +0.70 | +0.77 | **0.357** | +0.022 (+0.18) | +0.020 (+1.15) | 16.0 % vs 11.3 % ok | **fail** |
| 5 | VAL | **-0.018** | -0.05 | -0.06 | **0.932** | -0.150 (-0.49) | +0.032 (+1.06) | 14.3 % vs 5.6 % ok | **fail** |
| 10 | TRAIN | **+0.128** | +1.33 | +1.33 | **0.270** | +0.086 (+0.84) | +0.042 (+2.03) | 25.1 % vs 16.5 % ok | **fail** |
| 10 | VAL | **+0.177** | +0.83 | +0.95 | **0.597** | +0.058 (+0.28) | +0.064 (+2.03) | 24.2 % vs 10.3 % **VOID** | **fail** |

**Pre-committed verdict (PREREG SS4): FAIL at both windows. 0 of 4 required cells clear** — no split
reaches `t >= 2` on the difference, `w=5` VAL is *negative* (and negative ex-top-5 %), and `w=10` VAL
is additionally VOID on the 10 pp fill-rate rail. The one rail that *does* pass is the one that was
never the question: the touch-conditioned per-row difference is `+0.042` / `+0.064 R` with `t 2.03` at
`w=10`, i.e. the signal reliably picks name-hours that **touch** the limit more often (25 % vs 16 %),
which is fill frequency, not fill quality.

**Powering it moved the sign, not the conclusion.** frames19 (P1x3) read S - P1 `+0.088 / +0.382` at
`w=5`; frames20 (P1x5) reads `+0.089 / -0.018`. TRAIN is stable to the third decimal; **VAL swung by
0.40 R because the control's VAL cell went from 13 filled rows (net -0.011) to 27 (net +0.390)**. A
statistic that moves 0.4 R when 14 rows are added was never measuring anything.

## 3. SECONDARY (report-only diagnostic, PREREG SS5) — is "fade a 1 % pop" a book by itself?

| w | split | n | net | t(day) | ex-top-5 % | TRAIN halves | green wk | null50 | fills/wk | wk mean | worst wk |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | TRAIN | 99 | +0.121 | +1.27 | +0.028 | **+0.236 / -0.256** | 21 % | 30 % | 1.87 | +$23 | -$309 |
| 5 | VAL | 27 | +0.390 | +1.52 | +0.232 | — | 43 % | 48 % | 1.17 | +$46 | -$229 |
| 10 | TRAIN | 145 | +0.118 | +1.74 | +0.004 | **+0.163 / +0.022** | 36 % | 40 % | 2.74 | +$32 | -$223 |
| 10 | VAL | 50 | +0.151 | +0.86 | +0.020 | — | 35 % | 43 % | 2.17 | +$33 | -$214 |

Read (no ship language — this is a control population): a matched non-signal name that popped 1 % is
**positive on average in all four cells and is not a book**: `t < 2` everywhere; ex-top-5 % is +0.004
to +0.03 R on TRAIN (the mean is tail); TRAIN halves flip sign at `w=5` (+0.236 -> -0.256); the
**green-week share is BELOW the count-matched permutation null in all four cells** (21 vs 30, 43 vs 48,
36 vs 40, 35 vs 43 %); it fires 1.2-2.7 times a week for ~$30/wk at $100 risk. The F55 mechanism (enter
1 % above `ref` against a 2 %-of-entry stop = a mechanical head start) reproduces on names with no
signal at all, and does not deliver weekly there either.

## 4. Caveats, read as an adversary

1. **THE one that alone explains the headline: power, again — this test still cannot see the effect
   it is looking for.** MDE80 is **0.27 / 0.36 / 0.60 / 0.93 R** against point estimates of
   0.09-0.18 R. The best-powered cell (`w=10` TRAIN, MDE 0.270) needs **twice** the measured effect
   before this design can call it; 3 -> 5 controls barely moved VAL (0.87 -> 0.93/0.60 R — VAL *signal*
   rows, not control rows, bind). Honest sentence: **"no contribution of the MIR2 flag beyond the
   touch-and-fade mechanics was detectable in THIS universe, at THIS cost, with MDE 0.27-0.93 R"**.
2. **The cost model is asymmetric and it favours the control.** `meas` coverage is 99-100 % for S and
   **0 % for P1x5** — the pooled NBBO table was built on signal names, so every control row is charged
   the price-decile fallback (mean cost 0.061 R) against S's measured 0.079-0.099 R. The pre-committed
   common-constant sensitivity removes it: the differences become the **gross** gaps
   `+0.119 / +0.020 / +0.146 / +0.214 R` — every one still under `t 2`, and `w=5` VAL still ~zero.
   The asymmetry understates S - P1 by ~0.03 R and moves no verdict.
3. **The control's fill rate is roughly half S's** (11.3 vs 16.0, 10.3 vs 24.2 %) — one comparison is
   formally VOID, the others sit near the rail; the populations are not conditioned on the same event,
   which is why the touch-conditioned row is carried. It is the only passing row and it measures
   frequency. `w=5` VAL rests on 27 control rows; its ex-top-5 % (-0.150) is one trimmed tail wide.
4. **Nothing here rescues or refutes F55's named mechanism.** The stop width is still un-swept and this
   frame deliberately did not touch it. **Multiplicity**: 1,270 programme cells; the two here were fixed
   before scoring, the secondary table is a control diagnostic and proposes nothing.

## 5. Mid-run changes to anything pre-committed

**None.** The only deviation from the brief (`NMATCH` 3 -> 5 rather than 1 -> 5) was found, written into
the PREREG and frozen **before** `walk20.py` ran. No threshold, definition, window, seed or rail was
touched after scoring began.

## 6. Verdict

**F57 FAILS at both windows.** The MIR2 signal's contribution over a matched non-signal name that popped
the same 1 % is **not established** at 5 controls per signal, and the `w=5` VAL estimate is now
*negative*. The passive mirror short's positive book remains attributable to the touch-and-fade mechanics
it shares with the control. **No dry run, nothing to propose.** The line is not closed: the binding
constraint is VAL **signal** fills, and the question the programme has still never asked is the **stop
width** — to be pre-registered against this same P1x5 control, not rescued from these tables.
