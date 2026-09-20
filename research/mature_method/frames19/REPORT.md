# frames19 — REPORT (F56)

**Cell F56 — placebo decomposition of the passive mirror short.** PREREG `frames19/PREREG.md`,
frozen at git `b86b2870` before any scoring (`FREEZE.md`). Programme cell count **1,266 -> 1,268**
(two cells: `k=1.0 %`, `w in {5, 10}`; P3 and P1 are controls, not cells). TEST never opened — walker
and scorer both assert `max(day) < 2026-06-01`.

**Question**: does the MIR2 volume signal add anything beyond the touch-and-fade mechanics, or is
F55's headline a generic "fade a 1 % intraday pop in a small cap"?

## 1. Verdict — FAIL on both windows. The signal's contribution is NOT established.

| w | split | S − P3 | t | S − P1 | t | bar |
|---|---|---|---|---|---|---|
| 5 | TRAIN | **+0.213 R** | +1.43 | **+0.088 R** | +0.52 | fail (both) |
| 5 | VAL | **+0.565 R** | +3.18 | **+0.382 R** | +1.23 | S−P3 ok, S−P1 fail |
| 10 | TRAIN | **+0.195 R** | +1.83 | **+0.103 R** | +0.81 | fail (both) |
| 10 | VAL | **+0.447 R** | +3.71 | **+0.389 R** | +1.72 | fail (both, t<2) |

Pre-committed bar (PREREG §4) required **both** differences ≥ +0.10 R with day-clustered **t ≥ 2 on
both splits** and both comparisons non-VOID. **Only 1 of 8 required cells clears** (VAL S−P3 at w=5),
and its window's TRAIN twin is +1.43. Per the PREREG, anything less than both is a refutation of the
signal's contribution, not a partial pass.

## 2. The three populations alone (net R, measured-cost primary)

| w | split | pop | n rows | n filled | fill % | gross | **net** | t(clust) | t(iid) | **ex-top-5 %** | mean cost | meas cov |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | TRAIN | **S** | 1,366 | 219 | 16.0 | +0.302 | **+0.211** | +2.35 | +2.43 | **+0.051** | 0.091 | 100 % |
| 5 | TRAIN | P3 | 2,824 | 139 | 4.9 | +0.065 | −0.002 | −0.02 | −0.02 | −0.126 | 0.067 | 56 % |
| 5 | TRAIN | P1 | 571 | 61 | 10.7 | +0.184 | **+0.122** | +0.83 | +1.28 | **+0.056** | 0.062 | 0 % |
| 5 | VAL | **S** | 928 | 133 | 14.3 | +0.471 | **+0.372** | +2.36 | +2.17 | **+0.081** | 0.099 | 100 % |
| 5 | VAL | P3 | 1,797 | 83 | 4.6 | −0.111 | −0.193 | −1.47 | −1.53 | −0.380 | 0.082 | 42 % |
| 5 | VAL | P1 | 297 | 13 | 4.4 | +0.052 | −0.011 | −0.04 | −0.03 | −0.138 | 0.063 | 0 % |
| 10 | TRAIN | **S** | 1,366 | 343 | 25.1 | +0.325 | **+0.246** | +3.47 | +3.73 | **+0.090** | 0.079 | 99 % |
| 10 | TRAIN | P3 | 2,824 | 281 | 10.0 | +0.118 | +0.051 | +0.61 | +0.81 | −0.074 | 0.066 | 62 % |
| 10 | TRAIN | P1 | 571 | 86 | 15.1 | +0.205 | **+0.143** | +1.40 | +1.75 | **+0.066** | 0.062 | 0 % |
| 10 | VAL | **S** | 928 | 225 | 24.2 | +0.427 | **+0.329** | +3.04 | +2.84 | **+0.078** | 0.098 | 100 % |
| 10 | VAL | P3 | 1,797 | 171 | 9.5 | −0.045 | −0.118 | −1.35 | −1.54 | −0.250 | 0.073 | 47 % |
| 10 | VAL | P1 | 297 | 22 | 7.4 | +0.003 | −0.060 | −0.31 | −0.29 | −0.213 | 0.063 | 0 % |

S with the SSR rail OFF is within 0.005 R of S everywhere (the rail removes 5 of 357 touches at w=5).
P1 match attrition: 2,242 of 2,294 S rows (97.7 %) found ≥ 1 price/ADV match; 6,543 draws → **868 rows
survive the `gate5` + `price ≥ $5` rails (13 %)** — a matched non-signal name is rarely also up 5 %.
P3 hour-stratified to S's hour mix: **+0.241 / −0.309 (w=5), +0.359 / −0.304 (w=10)** TRAIN/VAL — the
hour re-weighting *raises* P3 on TRAIN, i.e. the raw P3 mean flatters S there.

## 3. The comparability gate, the touch-conditioned read, and the sensitivities

**Fill rates are not comparable.** S fills 14–25 %; P3 fills 4.6–10.0 %, P1 4.4–15.1 %. Six of the
eight comparisons are **VOID (> 10 pp)** under PREREG §4. As pre-committed, the touch-conditioned
version (every row scored, unfilled = 0) is reported: S−P3 **+0.034 (t 2.07) / +0.062 (t 2.82)** at
w=5 and **+0.057 (t 2.88) / +0.091 (t 3.55)** at w=10; S−P1 **+0.021 (t 0.99) / +0.054 (t 2.09)** and
**+0.040 (t 1.63) / +0.084 (t 2.69)**. Per-*row* these are 2–9 bp of R — they say the signal picks
name-hours that TOUCH more often, which is what a volume-elevation flag is supposed to do; they do not
say the filled trade is better.

**Sensitivity — one common constant cost** (difference = gross): S−P3 +0.238 (t 1.59) / +0.582
(t 3.26) at w=5, +0.207 (t 1.93) / +0.472 (t 3.90) at w=10; S−P1 +0.118 (t 0.69) / +0.419 (t 1.34) and
+0.119 (t 0.95) / +0.424 (t 1.87). No verdict moves.

**Tail test on the difference** (both sides trimmed at their own 95th percentile of `rr`) — the most
damaging cut in this frame: **S−P1 on TRAIN is −0.005 R (t −0.03) at w=5 and +0.025 R (t +0.20) at
w=10.** Once the top 5 % is removed, a *matched non-signal name that also popped 1 %* is
indistinguishable from the signal name on TRAIN. S−P3 ex-top-5 %: +0.177 (t 1.25) / +0.461 (t 3.30) and
+0.165 (t 1.65) / +0.328 (t 3.41).

**Power.** 80 %-power MDE on the differences (day-clustered): S−P3 0.42 / 0.50 (w=5), 0.30 / 0.34
(w=10); S−P1 0.48 / 0.87 (w=5), 0.35 / 0.63 (w=10). The S−P1 VAL test could only have seen ~0.6–0.9 R
— at n=13 and n=22 filled it is badly underpowered. The correct statement is **"no contribution beyond
touch-and-fade was detectable in THIS universe, at THIS horizon, over THIS window, at THIS cost, with
an MDE of 0.35–0.87 R"** — not "the signal contributes nothing."

## 4. Caveats read as an adversary

1. **THE ONE THAT ALONE EXPLAINS THE HEADLINE.** On TRAIN — the split that chose everything — a
   matched non-signal name that popped 1 % nets **+0.122 / +0.143 R**, versus S's **+0.211 / +0.246 R**,
   and **ex-top-5 % the gap is zero (−0.005 / +0.025 R)**. F55's warning stands: the TRAIN book is the
   touch-and-fade mechanics (enter 1 % higher against a 2 % stop = a mechanical head start, F55's own
   adverse-selection finding), not the volume signal. VAL disagrees — but VAL's P1 is 13 and 22 filled
   rows, and a split that disagrees with TRAIN at t<2 on 13 rows is noise, not confirmation.
2. **P1 is a selected population, not a clean control.** Only 13 % of matched draws survive `gate5`;
   what survives is a non-signal name that nonetheless rose 5 % — arguably the *hardest* control, which
   makes the null result stronger, but its small n is exactly what destroys the power.
3. **Cost asymmetry.** P1's exit legs are 0 % measured (decile-median fallback); P3's are 42–62 %. The
   fallback charges placebos LESS than S (0.062–0.082 vs 0.091–0.099), which *shrinks* S−P differences —
   conservative for the verdict reached, but it means the placebo nets are model, not measurement.
4. **Declared asymmetry (PREREG §3):** the Reg SHO 201 rail is applied to S only (no measured NBB at
   placebo fill minutes). Bounded at ~1.5 % of touches; S is reported both ways and moves < 0.005 R.
5. **Borrow**: the ETB flag is today's snapshot, not the trade-date state, and the borrow fee is
   assumed 0 for an intraday position. Unchanged assumption from frames16/17/18, repeated regardless.
6. **Multiplicity**: 1,268 cells cumulative on this programme. Two of them (F55's) passed a full bar;
   this frame shows the passing pair does not survive its own placebo.

## 5. Mid-run changes to anything pre-committed

* **One display bug fixed after the first scoring run, no statistic changed**: the per-population fill
  % in the table printed a both-splits denominator; the VOID gate had always used the per-split rate.
  Corrected and re-run; all differences, t's and verdicts are byte-identical.
* `attach_instrument` needs a `day` column; the pool classification is called with a constant
  `day='2025-01-02'` (wrapper/asset-class membership is symbol-level, not day-level). No effect on any
  population membership rule stated in the PREREG.
* Nothing else in PREREG.md changed.

## 6. What this frame does NOT authorise

No TEST read. No dry run. Nothing under `trading/`, `config.yaml`, `orb.yaml`, systemd or cron was
touched. The honest next question — pre-registered, not rescued post-hoc — is whether a **stop-width
sweep** (F55's named mechanism) has an edge at all, measured against this same P1 control, since the
placebo says the 1 %-above-ref entry against a 2 % stop is doing the work.
