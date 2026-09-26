# RESULT — cells 1,493–1,547: the retest bounce, the whole exit surface (FROZEN spec, FAIL)

Builder for `PREREG_1493.md`. Population: `rebuild_1481_fills.csv` status==fill, 8,973 retest fills
(TRAIN-H2 3,957 / VAL 5,016). Code: `cell_1493.py` (+ `test_cell_1493.py`, 18/18 passing). Outputs:
`cell_1493_fills.csv` (493,515 rows), `cell_1493_surface.csv` (220 rows, all/L3 x TRAIN/VAL),
`cell_1493_placebo.csv`, `cell_1493_1482_replication.csv`.

## Headline

**Every one of the 54 (stop × exit) cells, on both TRAIN-H2 and VAL, is net-negative in % of price.
No cell clears the TRAIN-H2 selection gate (stop ≥ 1.0 %, ≥ 3 fills/wk, day-clustered t ≥ 2) — the
gate needs a positive, significant cell and none exists. The mirror cell M (long, sized like the
1,480 short) is also negative (TRAIN −0.174 %, t −5.3; VAL −0.194 %, t −5.8) — the hoped-for bounce
does not appear even in its own best-case framing.** Worse: on every fixed-% stop the retest-timed
entry loses to its OWN placebo (same symbol-day, a random RTH minute, same exit) by −0.56 to −0.73
percentage points, t between −5.8 and −12.6 (VAL) — the retest timing is not neutral, it is adverse
selection. This is the PREREG's own FAIL branch: **the retest is closed for every exit measured.**
Combined with 1,487 (confirmation) and 1,491 (shallow stops), this closes the long side of the
1,438 population at every entry and every exit measured to date.

## TRAIN-H2 surface (mean net % of price; the whole 6×9 grid, unstratified)

| stop | tgt0.5 | tgt0.75 | tgt1.0 | tgt1.5 | tgt2.0 | tgt3.0 | NONE | T30 | T60 |
|---|---|---|---|---|---|---|---|---|---|
| 0.5% | −0.315 | −0.337 | −0.350 | −0.365 | −0.360 | −0.372 | −0.353 | −0.388 | −0.384 |
| 1.0% | −0.206 | −0.214 | −0.224 | −0.249 | −0.244 | −0.248 | −0.220 | −0.288 | −0.283 |
| 1.5% | −0.156 | −0.156 | −0.159 | −0.181 | −0.186 | −0.188 | −0.146 | −0.243 | −0.216 |
| 2.0% | −0.153 | −0.156 | −0.160 | −0.170 | −0.177 | −0.176 | −0.140 | −0.245 | −0.202 |
| 3.0% | −0.130 | −0.135 | −0.123 | −0.142 | −0.145 | −0.142 | **−0.116** | −0.229 | −0.189 |
| CL | −0.160 | −0.168 | −0.167 | −0.192 | −0.193 | −0.196 | −0.167 | −0.260 | −0.224 |

Best TRAIN-H2 cell = **3.0%|NONE** (n 3,957, mean −0.116 %, t −0.77, fails t≥2 like every other
cell). 0.5% stop rows are report-only per the PREREG (fail the R-vs-spread rail) and are shown for
completeness only. No cell — including the 0.5% row — is positive.

## VAL surface — every cell UNSELECTED (no cell passed TRAIN-H2)

| stop | tgt0.5 | tgt0.75 | tgt1.0 | tgt1.5 | tgt2.0 | tgt3.0 | NONE | T30 | T60 |
|---|---|---|---|---|---|---|---|---|---|
| 0.5% | −0.360 | −0.388 | −0.403 | −0.417 | −0.423 | −0.433 | −0.442 | −0.440 | −0.432 |
| 1.0% | −0.225 | −0.249 | −0.264 | −0.287 | −0.291 | −0.298 | −0.304 | −0.300 | −0.278 |
| 1.5% | −0.169 | −0.187 | −0.202 | −0.213 | −0.220 | −0.228 | −0.230 | −0.238 | −0.209 |
| 2.0% | −0.146 | −0.161 | −0.167 | −0.180 | −0.179 | −0.188 | −0.172 | −0.201 | −0.166 |
| 3.0% | −0.125 | −0.126 | −0.135 | −0.147 | −0.134 | −0.117 | **−0.078** | −0.137 | −0.083 |
| CL | −0.152 | −0.156 | −0.170 | −0.190 | −0.177 | −0.176 | −0.123 | −0.176 | −0.122 |

Best VAL cell (unselected — TRAIN-H2 never nominated it): **3.0%|NONE**, mean −0.078 %, t −0.53.
Same cell is best on both holdouts (rank-stable), still negative on both.

## Pass-bar checklist (illustrated on 3.0%|NONE, the best-by-mean cell — NOT a passing cell)

| criterion | requirement | result | pass? |
|---|---|---|---|
| TRAIN-H2 gate | stop≥1.0%, ≥3 fills/wk, t≥2 | **no cell clears t≥2** (best t=−0.77) | **FAIL** |
| VAL mean | ≥ +0.15 % | −0.078 % | FAIL |
| VAL t | ≥ 2.5 | −0.53 | FAIL |
| ex-top-5 % | > 0 | −0.640 % (VAL) | FAIL |
| winner-capped (+3%) | still positive | still −0.078 % (cell's own winners never exceed the cap much; capping barely moves it) | FAIL |
| count-matched null | ≥ 99th pctile | **100th pctile** (beats the null — see caveat) | pass (alone; see below) |
| placebo margin | ≥ +0.10 pp, t≥2 | **−0.608 pp, t = −5.8** | **FAIL (wrong sign)** |
| cache-only share | within 5pp of 19.5% | 18.2% (population, unfiltered by cell) | pass |
| neighbour stability | all 4 grid-neighbours same-signed on VAL | 2.0%\|NONE −0.172, 3.0%\|T30 −0.137, 3.0%\|tgt3.0 −0.117, CL\|NONE −0.123 — **all negative, same-signed** | pass (trivially — everything is negative) |

**passes_bar = FALSE.** Failing: TRAIN-H2 selection gate (no cell), VAL mean, VAL t, ex-top-5%,
winner-capped positivity, placebo margin.

Null-percentile caveat: the "100th pctile" pass is not a vote of confidence — the base-fill (break
entry) population that seeds the null is itself deeply negative (disclosed base book −0.16 to
−0.21R), so a −0.078% retest cell can out-rank it while still losing money outright. A null test
answers "is this different from the base population," never "is this profitable." Read together
with the placebo failure (which asks the right question — "is this timing informative at all") the
correct reading is: the retest-then-exit combination is worse than doing nothing special with the
same names on the same days.

## Placebo table (same symbol-day, random RTH minute in [09:45,15:00], same exit; paired, day-clustered t)

| cell | TRAIN real | TRAIN placebo | VAL real | VAL placebo | VAL margin | VAL t |
|---|---|---|---|---|---|---|
| 0.5%\|NONE | −0.336 | +0.274 | −0.423 | +0.180 | −0.603 | −12.61 |
| 1.0%\|NONE | −0.191 | +0.419 | −0.261 | +0.302 | −0.563 | −8.69 |
| 1.5%\|NONE | −0.102 | +0.524 | −0.163 | +0.402 | −0.566 | −7.59 |
| 2.0%\|NONE | −0.091 | +0.608 | −0.107 | +0.484 | −0.591 | −6.44 |
| 3.0%\|NONE | −0.066 | +0.669 | −0.022 | +0.586 | −0.608 | −5.82 |
| CL\|NONE | −0.129 | −0.062 | −0.055 | −0.110 | +0.055 | +0.62 |

Every fixed-% stop loses badly to its own placebo (t −5.8 to −12.6): buying the SAME name at a
random hour of the SAME day beats buying it at the retest of the break level, by roughly half a
point of price. Only the CL (consolidation-low) stop is placebo-neutral, and it is still net
negative on both holdouts. This separates cleanly from "no edge" into "the retest instant is a
worse-than-average entry time" — consistent with the disclosed mechanism (a break-then-dip is where
the crowd that chased the break is now underwater and selling into the bounce).

## First-passage matrix (selected pairs, VAL)

| cell | P(target first) | P(stop first) | P(time/EOD first) |
|---|---|---|---|
| 2.0% stop / tgt1.0 (up 1% before down 2%) | 0.617 | 0.349 | 0.034 |
| 1.0% stop / tgt2.0 (up 2% before down 1%) | 0.272 | 0.700 | 0.028 |
| M (mirror) | 0.313 | 0.658 | 0.030 |

`first_passage_up1_before_down2_val` = **0.617**, `first_passage_up2_before_down1_val` = **0.272**.
The asymmetry runs the WRONG way for a "bounce": reaching +1% before −2% happens 62% of the time
(consistent with chop, not drift), but reaching +2% before −1% happens only 27% of the time — a real
2%-sized bounce is rare, matching the deduction's own worry ("a 2R target is rarely reached") but
NOT rescued by any of the tighter targets once the standard costs are charged.

## Mirror cell M vs. the 1,480 short

| | stop | target | VAL mean | VAL t | mechanism |
|---|---|---|---|---|---|
| 1,480 short (disclosed) | max(break-bar high+$0.01, entry×1.01) | 2R (≈ −3.2%) | **−0.53 to −0.68 R** | −8.6 to −12.6 | short loses big → priced a bounce |
| M, this cell (long) | min(dip-bar low−$0.01, entry×0.99) | entry×1.02 | **−0.194 %** (net_R −0.146) | −5.8 | going long into the SAME dip does not recover the short's loss |

The short's catastrophic loss does NOT mirror into a long profit at the same geometry: p_target=31%,
p_stop=66% for M on VAL — the position is stopped out roughly twice as often as it reaches +2%, so
the "bounce" the short's numbers implied is smaller and less reliable than a naive mirror predicts,
exactly the caveat the PREREG flagged in advance ("part of the short's loss is its own cost").

## L3 stratum (report-only, hgb_prob_L3 ≥ 0.3070) — 3.0%|NONE

| holdout | n | mean_pct | t |
|---|---|---|---|
| TRAIN | 1,265 | +0.623 | 1.94 |
| VAL | 1,862 | +0.047 | 0.20 |

The only positive-looking slice in the whole programme (TRAIN, t<2, not selectable) evaporates on
VAL (t 0.20) — consistent with the prior finding that L3's top tercile is −0.09R under both the
break and the retest entry; not a rescuable filter.

## 1,482 replication (report-only, deeper retest, builder-only — 8,519 fills, 3.0%|NONE)

| holdout | n | mean_pct | t | mean_R |
|---|---|---|---|---|
| TRAIN-H2 | 3,750 | −0.091 | −0.60 | −0.030 |
| VAL | 4,769 | −0.036 | −0.24 | −0.012 |

Same sign, same order of magnitude as the 1,481 population on the same cell — the deeper retest
does not change the conclusion.

## Reused

`cell_1445.day_clustered_t` and `ex_top5_mean` (verbatim import). `sip_rebuild.py`'s EOD_M=955
constant and its bar-walk convention (stop-first on a bar touching both, gap-through at the open,
target needs a strict high>target) — reimplemented vectorized in numpy for a ~90x speedup (55
cells × 8,973 fills in ~20s vs. an estimated ~75 min row-by-row), same semantics, confirmed against
`sip_rebuild.walk_path`'s own logic by inspection and unit-tested independently.

## Every caveat

1. **Fills/wk is UNCAPPED.** `fills_wk` in the surface CSV is raw count/weeks-spanned (~147-228/wk),
   NOT run through the live 12/day-4-concurrent slot simulator (`consol.simulate_slots`, as
   `cell_1445.fills_per_week` does) — every cell in this population shares the SAME 8,973 fills, so
   the raw count trivially clears the ≥3/wk gate regardless; the slot-capped number would be lower
   but the selection failure is on t≥2 and mean>0, not frequency, so this does not change the
   verdict. Flagged, not fixed, given the whole surface is negative before frequency matters.
2. **Count-matched null draws with replacement, day-matched.** Not literally specified in the
   PREREG's prose beyond "random base fills on the same days" — implemented as: for each day in the
   cell's VAL fills, draw that day's fill-count (with replacement) from the SAME day's base-fill
   population (pooled fallback + WARNING if a day has none), 1,000 draws, seed 1493. An independent
   rebuild should confirm this exact draw scheme before trusting the 100th-percentile number, and
   should read it per the null caveat above (it answers a different question than the placebo).
3. **Tape-phase fill convention.** For a tape print that trades AT OR THROUGH the stop, the exit
   price is `min(print, stop)` (worse of the two) — a disclosed, not-PREREG-specified choice
   modeling a resting stop-limit order at the tape (tick) level, consistent with the bar-phase's own
   gap-through convention. Target fills are always AT the target (a resting limit), never at the
   better print.
4. **Placebo entry has no sub-minute tape.** The placebo (random-hour) leg enters at that minute's
   bar OPEN and walks bars only (no tape phase) — there is no tape pickle for an arbitrary minute.
   This slightly favors the placebo (no intra-minute stop check in its own entry minute) but the
   margin against it is large enough (5.8–12.6 t) that this bias cannot explain the result.
5. **Neighbour-adjacency convention** (STOP_ORDER / EXIT_ORDER list adjacency) is this builder's own
   disclosed convention, not written out in the PREREG's prose — moot here since the whole surface
   is same-signed.
6. **Cache-only share is population-level, not cell-level** — every cell shares the exact same 8,973
   fills (the grid only changes the EXIT, never which fills are in it), so a per-cell cache-only
   share is definitionally identical to the population's 18.2% for every one of the 55 cells.
7. **No TEST read.** Per the PREREG ("TEST: one read of the selected cell if VAL passes") and the
   frozen not-allowed list, TEST is not touched anywhere in this build — VAL failed outright.
8. Independent reimplementation from this prose has NOT yet been done (this is the builder only,
   per the assigned task) — required before this result is relayed to the owner as a conclusion,
   per the project's independent-check rule.

## Judge (main session, 2026-09-26 ~19:00 UTC) — FAIL, the retest is closed for every exit; two builder claims withdrawn

* Both implementations (builder `cell_1493.py`; rebuild `rebuild_1493.py` from the prose) find 0 of 55 cells positive
  on either holdout and the same gate outcome (no cell reaches t ≥ 2 on TRAIN-H2). Best cell 3.0 %|NONE: VAL −0.078 %
  (builder) / −0.064 % (rebuild); the mirror of the 1,480 short −0.194 % / −0.195 % (t −5.8 / −7.4). Row agreement
  80 % within 0.01 % of price (bar 99 %): the two sides handle the retest minute differently (fractional vs rounded;
  for 3,819 fills the print sits in minute retest_minute + 1 while the builder's bar walk starts at retest_minute —
  a defect that favours the book). With the best cell's ZERO-COST gross at +0.03 % of price (t 0.1), no convention or
  cost fix can reach +0.15 %; the verdict does not depend on the 20 % of rows that differ.
* WITHDRAWN: "the retest instant is adverse selection (loses to its placebo by 0.6 pp, t −6 to −13)". The placebo drew
  random minutes from 09:45 and 21 % fell BEFORE the break; those symbol-days are in the population because they later
  break their high, so a pre-break entry rides +3 % of look-ahead (t 15–23). Restricted to minutes after the retest
  window the margin is +0.03 pp (t 0.25) on VAL, +0.04 (t 0.26) on TRAIN-H2: the retest timing is neutral, and the
  in-play name does NOT drift after the break (post-window placebo −0.01 % / −0.18 % with the 3 % stop to the close).
  Recorded in [[placebo-window-selection]] (memory) — a same-name-day placebo may only draw minutes after the signal
  window when the population is selected on the signal.
* WITHDRAWN: the surface as published is inflated by the sparse-cache cohort: cache-only rows +0.89 / +0.66 % on
  3.0 %|NONE (43 of 55 cells positive on TRAIN); real-SIP rows −0.36 / −0.23 %, 0 cells positive, best real-SIP cell
  −0.18 / −0.15 %. The real-SIP surface is the honest one.
* The bounce deduction of this PREREG is falsified: the long mirror loses as much as the short; both carry a 1 % stop
  inside a ~3 % intraday noise band and the noise hits both stops (first passage: P(+1 % before −2 %) = 0.62 ≈ the
  driftless 0.67, P(+2 % before −1 %) = 0.27 ≈ 0.33). A losing tight-stop trade does not imply a winning mirror.
Consequence per PREREG: with 1,487 and 1,491, the long side of the 1,438 population is closed at every entry (break,
retest, deeper retest, confirmation) and every exit measured (55 here, 35 in the exit lab, the stop cells). Programme
count 1,547. The HOD dry run stays on as a free forward instrument only.
