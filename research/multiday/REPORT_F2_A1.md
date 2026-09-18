# REPORT — F2 (announcement-return drift, 6 cells) + A1 (announcement premium, 2 cells)

Run 2026-09-18 on the panel in `DATA.md`. Pre-registration: `PREREG_F2_A1.md`, written **before any
return was computed**. Code: `build_panel.py` (18.2M-row parquet -> a 3,807 x 2,693 matrix panel) and
`run_f2_a1.py`. Artifacts: `out_f2a1/` (primary) + `out_f2a1_{adv10m,flat5,noskip}/` (arms).

**Verdict in one line:** *no edge was detectable in the US-domestic-8-K-filer common-stock universe, at
20/40/60-session and 4/5-session horizons, at a $66K / 20-slot book, over 2016-01 -> 2023-12, at
close-to-close auction cost — and the smallest monthly effect this test could have seen is
**$670–$3,034/month at book size**, against a Chen–Velikov prior of **$25–65/month**. The test is
10–120x under-powered for the effect size the literature predicts. Nothing here licenses "these
families do not work"; it licenses only "this panel cannot see them."*

**TEST (2024-01 -> 2026-09) was NOT opened. No `FREEZE.md` was written** — 0 of 8 cells cleared G2, and
under the independent rebuild's sealed-split convention (§8) **0 of 8 clear even G1**.

---

## 1. What was run, and the one look-ahead I found in my own code

Execution is **close-to-close, `cls` both legs** (AMENDMENT 2(a); Goyal–Jegadeesh–Wu JFQA 2026:
*"Opening auctions are illiquid"*). No open is touched anywhere; "entry at the opening auction" was
**not tested** and is carried as refuted-by-literature, not as a hypothesis. No quoted spread on an
auction cross. Costs: impact `10 bps x (order$ / 1% of ADV20$)` both sides + 0.4 bps SEC/TAF on the
sell. No short leg was ever executed (see §3), so the 0.3%/yr ETB borrow charge never binds — the
`easy_to_borrow` look-ahead of DATA.md §7 gap 5 is therefore **not load-bearing in this report**.

**Self-caught defect, fixed before any number was recorded.** The first A1 builder looped
`for k in range(len(S) - 4)` and required `len(S) >= 6` — both condition on FUTURE events, silently
deleting every symbol that *stopped* announcing (acquired, delisted, went dark). That is a look-ahead
*and* a second survivorship layer on top of the panel's. Fixed to a causal "at least 5 announcements
already filed at prediction time". **The fix cut A1-b's TRAIN t from 2.21 to 1.76 and was the difference
between this report opening TEST and not.** Recorded here because it is the whole point of the rail.

**Availability audit** (`out_f2a1/availability.csv`): the `event_session` rule (closing auction strictly
after `acceptanceDateTime`) has **0 violations in 130,891 events**. 4.3% of events have no price at S.
The missingness is *not* random and is reported per split x acceptance bucket: TRAIN 6.96% / VAL 3.21% /
TEST 1.34%, and intraday-acceptance events are worst (TRAIN 13.46%). That gradient **is** the
survivorship shape — today's names backfilled — and it means TRAIN is the thinnest split, not the fattest.

---

## 2. The 8-cell table (+1 declared robustness), per split

Primary statistic is the **benchmark-adjusted monthly portfolio series**, because that is the correctly
clustered object: per-trade means are inflated by the cross-sectional correlation of ~500 announcers in
the same earnings week. Benchmarks: F2 long-only vs the **all-decile equal-weighted book of the same
events, same hold, same costs** (strips beta *and* the announcer-population effect); F2 L-S is
self-benchmarked; A1 vs the **same stock's mid-quarter window 31 sessions earlier** (the published
Frazzini–Lamont announcement-vs-non-announcement contrast).

`bps` = monthly, `t` on the monthly series, `MDE` = 2*SE of the monthly mean (bps/month).
`LLS` = long-leg share of the L-S spread. `be x` = break-even cost / the honest auction cost.
`tr/wk` and `$/mo` are the **executable $66K / 20-slot book** (alpha basis).

| cell | split | n | raw bps | **excess bps** | **t** | %mo+ | ex-Jan | **MDE** | **LLS** | be x | **tr/wk** | **$/mo** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F2-LS-20 | TRAIN | 4,466 | -39.9 | -39.9 | -0.91 | 45.1 | -36.4 | 87 | n/a(1) | 11.2 | 3.9 | -143 |
| F2-LS-20 | VAL | 1,880 | +137.0 | +137.0 | 1.42 | 66.7 | +160.7 | 192 | **72%** | -18.0 | 4.0 | -320 |
| F2-LO-20 | TRAIN | 4,466 | +139.7 | -10.4 | -0.34 | 56.3 | -22.3 | 60 | n/a(1) | 11.2 | 3.9 | -143 |
| F2-LO-20 | VAL | 1,880 | +94.7 | +99.2 | 1.23 | 62.5 | +103.3 | 161 | **72%** | -18.0 | 4.0 | -320 |
| F2-LS-40 | TRAIN | 4,466 | +47.9 | +47.9 | 1.25 | 62.0 | +77.2 | 77 | **87%** | 23.3 | 2.2 | +297 |
| F2-LS-40 | VAL | 1,880 | -44.2 | -44.2 | -0.91 | 45.8 | -24.3 | 97 | n/a(1) | -122 | 2.2 | -206 |
| F2-LO-40 | TRAIN | 4,466 | +201.4 | +41.8 | 1.97 | 54.9 | +42.6 | 42 | **87%** | 23.3 | 2.2 | +297 |
| F2-LO-40 | VAL | 1,880 | -42.4 | -33.9 | -0.77 | 37.5 | -27.8 | 88 | n/a(1) | -122 | 2.2 | -206 |
| F2-LS-60 | TRAIN | 4,466 | +46.9 | +46.9 | 1.72 | 60.6 | +65.2 | 54 | **98%** | 48.6 | 1.5 | +151 |
| F2-LS-60 | VAL | 1,880 | -43.7 | -43.7 | -1.03 | 37.5 | -13.1 | 85 | n/a(1) | -145 | 1.5 | -1,029 |
| **F2-LO-60** | TRAIN | 4,466 | +209.0 | **+45.9** | **2.15** | 59.2 | +45.0 | 43 | **98%** | 48.6 | 1.5 | +151 |
| **F2-LO-60** | VAL | 1,880 | -54.8 | **-56.4** | -1.28 | 37.5 | -38.6 | 88 | n/a(1) | -145 | 1.5 | -1,029 |
| A1-a | TRAIN | 19,790 | +212.9 | +121.0 | 1.71 | 64.3 | +131.8 | 142 | 100%(2) | 47.9 | **12.8** | +237 |
| A1-a | VAL | 11,938 | +3.5 | +0.6 | 0.01 | 50.0 | +30.2 | 190 | 100%(2) | 60.4 | **19.5** | +953 |
| A1-b | TRAIN | 19,790 | +266.8 | +137.2 | 1.76 | 62.5 | +146.8 | 156 | 100%(2) | 65.1 | **11.1** | -52 |
| A1-b | VAL | 11,938 | +42.8 | +6.3 | 0.07 | 58.3 | +46.7 | 183 | 100%(2) | 78.7 | **16.8** | +373 |
| *A1-rob*(3) | TRAIN | 19,790 | +191.7 | +45.9 | 0.67 | 60.7 | +52.6 | 138 | 100%(2) | 72.0 | 9.5 | -275 |
| *A1-rob*(3) | VAL | 11,938 | +3.9 | -44.5 | -0.46 | 62.5 | -14.8 | 192 | 100%(2) | 82.2 | 14.2 | +288 |

(1) LLS is undefined where the D10-D1 spread is <= 0 — a "share of a negative spread" is not a number.
(2) A1's **published** form is long-only; there is no short leg to miss. Reported as 100% by construction.
(3) `A1-rob` (sell close(Ehat+1), i.e. hold *through* the event) is a **declared robustness, not a scored
cell**. It is the trade Johnson–So predicts gives the premium back, and it does: TRAIN excess falls
137.2 -> 45.9 bps and VAL flips negative. **That is the only prediction in this report that replicated.**

### The four mandatory columns, read out
1. **Long-leg share.** Where F2's spread is positive at all, **87–98% of it is in the LONG leg**
   (D1 is approximately the all-decile book; the short leg adds ~nothing). Structurally this is the
   *good* answer for a long-only, no-margin account — better than F3's published ~50%
   (Israel–Moskowitz 2013). It is also moot, because the spread does not replicate.
2. **Break-even cost.** The honest auction round trip on this book is **1.14–1.45 bps** (impact at
   $3,300 orders against >= $1M ADV is near-nil; the whole charge is 0.4 bps of SEC/TAF plus pennies).
   Break-even is 11–65x that on TRAIN and **negative on VAL** for every F2 cell — i.e. on VAL no cost
   reduction whatsoever makes F2 positive. Annualised against the published benchmark (9–21 bps/yr L/S,
   our halved ~5–12 bps/yr long-only analogue): **F2-LO-60 turns 4.2x/yr ~= 6 bps/yr — inside the band;
   A1 turns ~50x/yr ~= 60 bps/yr — 5–12x the band.** A1 is a fast book wearing a slow book's cost model.
3. **Ex-January.** Nothing here is a January effect: ex-Jan tracks the all-months number within ~15 bps
   everywhere, and where the cell is positive ex-Jan is usually *larger*. No cell is rescued or killed by it.
4. **Trades/week at $50–66K.** F2 is structurally slow — **1.5/wk at H=60, 2.2 at H=40, 3.9 at H=20** —
   because 20 slots divided by a 60-session hold is one entry every 3 days, whatever the candidate count.
   **A1 is the fast one: 11–20/wk**, and it is the only cell in this stage that clears the owner's
   >= 10/week column.

---

## 3. The executable $66K book — the money object, and its power

The academic overlapping portfolio is the *return* estimate; the 20-slot book is the *money* estimate.
Alpha basis (benchmark-differenced), costs charged:

| cell | split | tr/wk | $/month | t | % months + | **MDE $/month** | worst month | book MDD |
|---|---|---|---|---|---|---|---|---|
| F2-LO-20 | TRAIN | 3.9 | -92 | -0.24 | 43.1 | 780 | -8,750 | -27,063 |
| F2-LO-20 | VAL | 4.0 | -187 | -0.32 | 41.7 | 1,172 | -7,444 | -15,354 |
| F2-LO-40 | TRAIN | 2.2 | +392 | 1.11 | 47.2 | 705 | -7,289 | -25,433 |
| F2-LO-40 | VAL | 2.2 | -475 | -0.74 | 37.5 | 1,287 | -9,188 | -14,997 |
| F2-LO-60 | TRAIN | 1.5 | +231 | 0.69 | 20.8 | 671 | -8,641 | -25,149 |
| F2-LO-60 | VAL | 1.5 | -975 | -1.51 | 16.7 | 1,294 | -11,147 | -23,412 |
| A1-a | TRAIN | 12.8 | +212 | 0.31 | 37.5 | 1,345 | -15,601 | -24,935 |
| A1-a | VAL | 19.5 | +1,179 | 0.96 | 50.0 | 2,450 | -9,407 | -14,939 |
| A1-b | TRAIN | 11.1 | -65 | -0.10 | 38.9 | 1,337 | -14,987 | -27,963 |
| A1-b | VAL | 16.8 | +571 | 0.38 | 50.0 | 3,034 | -14,284 | -19,336 |

**This table is the report.** Every |t| < 1.6. Every sign flips or collapses between TRAIN and VAL. And
the MDE column says the book could not have detected anything smaller than **$670–$3,034/month** — while
the pre-committed calibration prior (Chen & Velikov: 204 published anomalies net ~4 bps/month, strongest
~10 before impact) puts the *expected* effect at **$25–65/month on $66K**. The experiment is
**10–120x too blunt for the thing it is looking for.** Per the phrasing rule, that is the finding.

A second, separable fact: **the executable book's sign is set by the tiebreak, not by the effect.**
A1-b's academic excess is +137 bps/month on TRAIN, but the 20-slot book that takes the *most liquid*
announcers first earns **-$65/month** there. Concentrating $66K into 20 of ~500 weekly candidates by an
arbitrary rule throws away the estimate. Any future A1 work must choose that selection rule on TRAIN
and score it as a new cell — it is not a free parameter.

---

## 4. Gates, tails, multiplicity, survivorship

**G1 (TRAIN t >= 2 on the benchmark-adjusted monthly series):** 1 of 8 — `F2-LO-60`, t = 2.15 — and
**0 of 8** under the independent rebuild's sealed-split convention, where the same cell reads t = 1.83
(§8b). G1 for this stage is therefore borderline-at-best and implementation-dependent.
**G2 (VAL same sign AND >= 55% of months positive):** **0 of 8.** F2-LO-60's VAL excess is -56.4 bps
(sign flips) with 37.5% of months positive. **TEST stays sealed.**

**Robustness arms — the verdict is identical in all four** (`out_f2a1_*`):

| arm | what changes | F2-LO-60 TRAIN t | F2-LO-60 VAL | A1-b TRAIN t | A1-b VAL |
|---|---|---|---|---|---|
| primary ($1M ADV, auction cost, skip 1 session) | — | 2.15 | -56 bps | 1.76 | +6 bps |
| ADV20$ >= $10M | liquidity floor x10 | 1.61 | -89 bps | 0.85 | +109 bps |
| flat 5 bps/side | the secondary cost arm | 2.15 | -56 bps | 1.77 | +6 bps |
| no-skip (entry at close(S)) | **NOT OBTAINABLE**(4) | 2.30 | -41 bps | 1.76 | +6 bps |

(4) The no-skip arm is the academically faithful F2 (form the portfolio the instant the announcement
return is known) and it is **not executable**: a `cls` order must rest *before* the close it fills, so a
signal computed from close(S) cannot be filled at close(S). It is reported only to show that the
one-session skip is not what kills F2 — the unobtainable version fails VAL too.

**Tail dependence (PLAN §1 item 5).** Net, benchmark-adjusted, per trade:
**removing the top 5% of trades turns EVERY one of the 8 cells negative** (F2-LO-60: +68.7 -> -313.2 bps;
A1-b: +78.3 -> -70.2 bps). Capping winners at 3x the cell's median win: F2 goes negative at every hold
(-80 to -359 bps); A1-b survives but shrinks to **+6.3 bps TRAIN / +50.6 VAL**. So even on the side that
looked alive, essentially all of the point estimate is in the right tail. This owner has already rejected
one lottery ticket; this is another.

**Multiplicity.** 8 scored cells this stage. Block-bootstrap p on the TRAIN excess (5,000 resamples):
best cell `F2-LO-60` p = 0.03, `A1-b` p = 0.08. **Sidak-adjusted across the 8 cells: 0.21 and 0.48.**
Total distinct looks in this stage = 9 series x 4 execution/liquidity arms = **36**.
Cumulative multi-day scored-cell count: K 20 + N2 4 + R_daily 20 + **8 here = 52**.

**Survivorship (DATA.md §7 gap 1: longs biased UP 5–8.75%/yr ~= 42–73 bps/month).** The PIT Nasdaq
re-run was **not triggered — there are no G2 survivors**. Two things must be said anyway:
- Every *scored* number is a **difference against a benchmark drawn from the same surviving population**
  (F2: the all-decile book of the same events; A1: the same stock, 31 sessions earlier). The 5–8.75%/yr
  long bias very largely cancels in that difference. **This is what makes the null credible rather than
  survivorship-driven** — the bias would have inflated the cells, not deflated them.
- The **raw** column is a different story: at 42–73 bps/month of survivorship lift, the raw monthly
  figures (+94 to +267 bps) are *substantially* an artifact of the panel being today's names backfilled.
  **The raw column must not be read as P&L.** It is printed only to show how much of the long-only
  headline is beta + survivorship rather than signal.

Price-scale (item 3) was cleared at the data stage: 0 of 200 keys fail at 0.01%, median relative error
1.0e-06. Fill realism (item 4): every fill is an official consolidated auction print, decided one
session in advance on both legs; no gap-through, no intrabar stop, no double-counted slip. Causality
(item 2): the `event_session` rule has 0 violations, the F2 decile uses a strictly trailing 250-day
reference set with a 200-observation warm-up, and A1's `Ehat` is set 364 calendar days ahead with a
"3 quarters have landed" check that reads nothing after the entry session.

---

## 5. Additivity vs the live ORB book

**Resource overlap: none on ORB's binding constraint, per cell.** ORB's binding resources are (a) the
4 concurrent intraday slots, (b) 09:35 buying power, (c) the 09:35 attention window; every ORB position
is flat by 15:45. Both F2 and A1 open in the **16:00 closing auction** and hold overnight for 4–60
sessions. They never contend for an ORB slot, never need capital at 09:35 on the day they enter, and
never touch the ORB engine, StopMonitor, or the intraday tape. **Structurally additive on all three.**

**The one real interaction, and it is not zero:** a sleeve holding $66K of equity overnight *reduces the
next morning's day-trading buying power* by that notional in a margin account. ORB's current stage
budget is $10K, so the two fit in the live account today — but this is an account-level BP check that
must be run before any multi-day sleeve is armed, not assumed. Stage I's "stacking hurts" result is
**not** imported (that was same-day intraday books fighting for the same slots); the interaction here is
a balance-sheet one and it is the only one.

**Return correlation with ORB: NOT COMPUTED, and the reason is the seal.** The live ORB book
(`research/fuckup_audit/D1_orb/book_n8_q1on.csv`) runs **2025-01-07 -> 2026-09-15** — entirely inside the
sealed TEST window. Computing the correlation means computing this stage's monthly returns on 2024–2026,
which is opening TEST. No cell earned that, so the correlation is deferred to the FREEZE step of whatever
family first clears G2. (The overlap on TRAIN/VAL is zero months, so there is no partial answer.)

---

## 6. The two bars

**Claim bar (G1 + G2): FAILED, 0 of 8.** Nothing from this stage may be stated as a finding, and TEST
stays sealed.

**Live-exploration bar** (positive point estimate + mechanism + bounded downside + resolution inside a
quarter at its own trade frequency):

| cell | point est. | mechanism | bounded downside | resolves in a quarter | verdict |
|---|---|---|---|---|---|
| F2-LS/LO-20 | TRAIN negative | published (CJL 1996) | yes | no (3.9/wk, MDE $780/mo) | **FAIL** |
| F2-LS/LO-40 | VAL negative | published | yes | no (2.2/wk) | **FAIL** |
| F2-LS/LO-60 | VAL negative, sign flip | published | yes | no (1.5/wk, 20.8% months +) | **FAIL** |
| A1-a | + both splits (+121 / +0.6 bps) | published & specific | yes | **frequency yes (12.8–19.5/wk)** | **FAIL on noise** |
| A1-b | + both splits (+137 / +6.3 bps) | published & specific | yes | **frequency yes (11.1–16.8/wk)** | **FAIL on noise** |

A1 passes conditions 1–3 and passes the *frequency* half of condition 4 — it is the only thing in this
stage that could resolve inside a quarter. It fails on the half that matters: at book size the point
estimate (-$65 to +$1,179/month) sits inside a **$1,337–$3,034/month** noise band, so a quarter of live
trading would resolve nothing. Frequency without a measurable estimate is the HOD-break failure mode again.

**Bounded downside, stated because it is the one thing that *is* well measured:** a $66K long-only
auction sleeve at 20 x $3,300, no leverage, no short, no stop — its worst month across these cells is
**-$7.3K to -$15.6K** and its book drawdown **-$15K to -$28K**, i.e. **23–42% of the sleeve**. That is
the honest risk of holding an equal-weighted basket of small announcers through 2016–2023, and it is
what any "it's only a small sleeve" framing has to answer to.

---

## 7. What I would do next, and what I would not

- **Do not re-propose F2.** Its TRAIN drift is 87–98% long-leg (the good structural answer), is
  ex-January-robust, and dies completely on VAL in all four arms and on the unobtainable-fill arm too.
  Its one live number — 1.5–3.9 trades/week — also puts it in the "cannot resolve" corner by construction.
- **A1 is the only survivor worth a second cell**, and only as a *selection* question, not a *signal*
  question: the effect is visible in the 500-name cross-section and destroyed by taking the 20 most
  liquid. The pre-registered next cell would rank A1 candidates by the F2-style trailing percentile of
  a causal pre-announcement feature, chosen on TRAIN only, scored as a NEW cell in the denominator.
- **The binding problem is power, not signal.** A $66K, 20-slot book has an MDE 10–120x the effect the
  literature predicts. No amount of further searching on this panel fixes that; only more slots
  (more capital) or a genuinely larger effect would. Every subsequent multi-day cell should state its
  MDE in $/month *before* it is run, and cells whose MDE exceeds ~$100/month should be understood as
  unable to confirm anything, only to refute something large.
- **Newest verified evidence age** (AMENDMENT 2(e)): F2 — CJL 1996 seminal, nothing verified post-2015
  on this measure; the nearest 2023–2026 item (JLST, RFS 2025) is indirect. A1 — Johnson–So 2018 JAR
  plus PBFJ 2023 (the premium survives post-crisis, framed as compensation for *expected* volatility
  risk, not mispricing). A1's prior is the younger and better-supported of the two, and it is the one
  that did not die.

---

## 8. Independent rebuild (CLAUDE.md "No research claim ships without an independent check")

A second implementation was written from a **prose specification only**, by an agent forbidden to read
`run_f2_a1.py`, `build_panel.py` or any output of it (`independent_check.py`, `out_indep/`). It rebuilt
the two strongest cells — `F2-LO-60` and `A1-b` — from the raw parquets, including its own session
calendar, its own ADV20, its own causal decile, and its own portfolio aggregation.

**(a) The trade sets are identical.** Merged on (symbol, entry date) across TRAIN+VAL:
**0 trades only-mine, 0 trades only-theirs** in either cell; max |gross-return difference| **1.3e-06**
(F2-LO-60, 14 trades above 1e-06, all float32 rounding), **2.8e-07** (A1-b); max cost difference
4.5e-04 bps. **A coding error is ruled out for the entry rule, the event join, the decile, the
availability gates, the hold, and the return.** Per the standing rail, this cannot rule out a
*specification* error, and it did not: the two below are specification findings, not code findings.

**(b) It found a seal leak I had not.** My VAL cells hold 40 and 60 sessions from entries in
Oct–Dec 2023, so their exits land in **2024 — inside the sealed TEST window**. The independent run
reports both conventions. Adopting the sealed one (drop any trade whose exit is after the split end):

| cell | split | mine ("complete") | independent, **sealed** | sign | gate outcome |
|---|---|---|---|---|---|
| F2-LO-60 | TRAIN | +45.9 bps, t 2.15 | +42.6 bps, **t 1.83** | same | **G1 now FAILS too** |
| F2-LO-60 | VAL | -56.4 bps, t -1.28 | -38.9 bps, t -0.71 | same | G2 fails |
| A1-b | TRAIN | +137.2 bps, t 1.76 | +129.2 bps, t 1.68 | same | G1 fails |
| A1-b | VAL | +6.3 bps, t 0.07 | +96.1 bps, t 0.78 | same | G2 fails |

**The conclusion does not move — it gets stronger.** Under the independent implementation's sealed
convention, **0 of 8 cells clear even G1**, where my run had 1. The sealed convention is adopted as the
standing rule for every future multi-day cell: *a trade whose exit falls outside the split is not in
that split.* (F2's 60-session hold makes this a 3-month bleed across every boundary, which is a general
property of long-hold families and belongs in the next family's pre-registration.)

**(c) The discrepancy that is itself a finding.** The two implementations agree trade-for-trade and
agree on every sign and every gate outcome, but the *monthly-series convention* alone (which months
count, how empty months are handled) moves A1-b's VAL point estimate from **+6.3 to +96.1 bps/month** —
a factor of 15 — and F2-LO-60's TRAIN t across the 2.0 line. **A real effect does not move by 15x on a
bookkeeping convention.** This is independent confirmation of §3's conclusion by a different route: the
estimates are noise-dominated at this book size, and the MDE column is the honest summary of the stage.
