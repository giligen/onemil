# RESULT — cells 1,562-1,563: ORB retest bid at range_high minus a tick (PREREG_1562, Amendment 1)

Decision number pre-committed in PREREG_1562.md. **VERDICT: FAIL on both cells — ORB keeps the chase
entry.** Read the caveats below before trusting any number in this report; this file is its own
adversary.

## Population
410 signals (301 `status==filled` chase entries + 109 `status==skipped_guard` chase-guard skips,
Amendment 1's frozen count, asserted in code) from `research/orb_latency_bt/results.csv` delay_s==0
joined to `population.csv`. 228 signals (207 `unfilled_no_tstar` + 21 `missing_tick_data`) excluded
per Amendment 1, not scored. Splits by entry date: TRAIN = 2023-01-12..2025-06-30, VAL =
2025-07-01..2026-09-23 (Amendment 1).

## Headline table (per cell, per split; POOLED = TRAIN+VAL together)

| cell | split | n_signals | n_excl | n_scored | n_filled | fill_share | withdrawal_15m | med_dip_bps | med_min_to_retest | mean_net_R' | own_t | ex_top5_R | winner_cap_R | fills/wk | med_R'_%price | book_incl_zero_R | paired_ΔR | paired_t | paired_ex_top5 | never_retest_n | never_retest_base_R | base_mean_R |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1562 | TRAIN | 164 | 108 | 56 | 44 | 0.786 | 1.000 | 72.8 bps | 0.46 min | +0.674 | 1.69 | +0.174 | +0.279 | 1.91 | 4.04% | +0.530 | +0.355 | 1.32 | -0.040 | 12 | +0.542 | +0.175 |
| 1562 | VAL | 246 | 0 | 246 | 210 | 0.854 | 1.000 | 91.6 bps | 0.52 min | +0.244 | 1.20 | **-0.133** | -0.007 | 3.68 | 3.95% | +0.209 | +0.146 | 0.97 | -0.161 | 36 | +0.234 | +0.062 |
| 1562 | POOLED | 410 | 108 | 302 | 254 | 0.841 | 1.000 | 86.4 bps | 0.51 min | +0.319 | 1.75 | -0.102 | +0.042 | 3.18 | 4.01% | +0.268 | +0.185 | **1.40** | -0.139 | 48 | +0.311 | +0.083 |
| 1563 | TRAIN | 164 | 108 | 56 | 48 | 0.857 | 0.929 | 79.4 bps | 0.47 min | +0.724 | 1.94 | +0.269 | +0.352 | 2.09 | 3.93% | +0.620 | +0.445 | 1.64 | +0.055 | 8 | +0.630 | +0.175 |
| 1563 | VAL | 246 | 0 | 246 | 216 | 0.878 | 0.959 | 95.4 bps | 0.69 min | +0.207 | 1.04 | **-0.190** | -0.043 | 3.79 | 3.88% | +0.182 | +0.120 | 0.78 | -0.192 | 30 | +0.319 | +0.062 |
| 1563 | POOLED | 410 | 108 | 302 | 264 | 0.874 | 0.954 | 90.1 bps | 0.69 min | +0.301 | 1.70 | -0.106 | +0.029 | 3.30 | 3.90% | +0.263 | +0.180 | **1.33** | -0.147 | 38 | +0.385 | +0.083 |

Exit mix (filled retest signals): 1562 POOLED stop 52% / eod 33% / lock 15%; 1563 POOLED stop 52% /
eod 31% / lock 17%.

**The better of the two on VAL is cell 1,562** (higher own-mean R', higher own-t, higher paired ΔR) —
still a FAIL (below).

## Pass bar (frozen, PREREG_1562.md) — checked line by line

| criterion | 1562 result | 1563 result | pass? |
|---|---|---|---|
| paired ΔR ≥ +0.10 R on BOTH splits | TRAIN +0.355, VAL +0.146 | TRAIN +0.445, VAL +0.120 | **yes**, both cells, both splits |
| pooled day-clustered t ≥ 2.5, same sign each split | t=1.40, signs (+,+) | t=1.33, signs (+,+) | **NO** — t well under 2.5 |
| VAL own mean R' ≥ +0.15 with t ≥ 2 | mean +0.244 (size OK), t=1.20 | mean +0.207 (size OK), t=1.04 | **NO** — t under 2 on both |
| ex-top-5% of BOTH legs > 0 | TRAIN +0.174, VAL **-0.133** | TRAIN +0.269, VAL **-0.190** | **NO** — VAL is tail-carried on both cells |
| winner-capped positive | TRAIN +0.279, VAL -0.007 | TRAIN +0.352, VAL -0.043 | **NO** on VAL |
| ≥ 3 fills/week | TRAIN 1.91, VAL 3.68 | TRAIN 2.09, VAL 3.79 | **NO** on TRAIN (see caveat 2) |
| never-retest disclosed, book-incl-zero ≥ base mean | 0.530≥0.175 (TRAIN), 0.209≥0.062 (VAL) | 0.620≥0.175, 0.182≥0.062 | yes, both cells |
| median R' ≥ 0.5% of price | 3.9-4.0% | 3.9-3.9% | yes, both cells |

**Both cells fail on three independent, frozen criteria**: the pooled t-stat (needs ≥2.5, gets
1.3-1.4), the VAL ex-top-5% tail check (negative on both cells — the VAL headline number is carried
entirely by the top 5% of fills, exactly the failure mode flagged in
`feedback_paired_lift_tail_check`), and VAL's own t≥2 requirement. **FAIL is unambiguous even before
weighing caveats 1-3 below** — per PREREG: "FAIL → ORB keeps the chase entry; the withdrawal share and
the paired number stay on record."

## What the mechanism check itself shows (disclosed, separate from the pass-bar verdict)
The HOD-learning DOES generalise on the "does the market offer the retest" question: withdrawal share
within 15 minutes is 93-100% here (vs HOD's 87%), fill share is 79-88%, and paired ΔR is positive with
the right sign on every split of both cells — the retest bid recovers real immediacy cost, same
direction as HOD. It fails on STATISTICAL POWER and TAIL DEPENDENCE, not on sign: n_filled is only
44-48 signals in TRAIN and 210-216 in VAL, and the positive VAL point estimate reverses to negative
once the top 5% of fills are excluded. This is "no edge proven", not "edge disproven" — see caveats.

## Caveats (read as an adversary before repeating any number above)

1. **Base leg is NOT re-costed to the SLIP_STOP_BPS standard the PREREG asks for.** `population.csv`'s
   per-share `pnl` column does not resolve to a literal per-share price (entry_price + pnl gives
   impossible negative prices for `exit_reason=='stop'` rows, e.g. VTAK entry $3.70, pnl -467.95 →
   "exit" -$464) — recovering the raw pre-slip exit price to reapply reason-specific costs needs the
   position-sizing formula in `trading/orb_planner.py`, out of this task's step budget. `base_R` uses
   `pnl_replay/375` as-is (the ORB program's own established convention, `replay.py:82`,
   `REPORT.md`'s own `mean_R`) — its original embedded 10bps uniform exit slip, not the PREREG's
   blended stop-limit standard. Comparing it to `retest_R` (built from real tape/bar prices) assumes
   both legs used fixed-fractional $375 risk sizing off their own entry-to-stop distance — a stated,
   unverified assumption (see the code docstring's "Comparability" paragraph). **This is the single
   biggest thing to independently re-derive before this report moves any capital decision.**

2. **TRAIN is data-starved by construction, not by this study's choice.** The tape this task's prompt
   pointed at (`research/hod_ofi/raw/*.parquet`) is the HOD-OFI study's own tape (different symbols,
   2025-01+ only) and does not cover this population at all; the correct tape is what `replay.py`
   itself reads (`research/orb_latency_bt/raw/*.parquet`, verified to span 2023-2026, used here).
   That tape's fetch window is 09:34:50-09:40:00 ET only — built for the cell-1,426 trigger print, not
   a 15/30-minute retest search. `data/cache.db` (read-only, verified `MIN/MAX(bar_date)` query) has
   1-minute bars only from 2025-01-02; `research/orb_2023/bars.db` and `research/orb_2024`'s equivalent
   (referenced by `study_orb_pipeline_static_lock.py`'s own docstring) are not present on disk. Result:
   108 of 164 nominal-TRAIN signals (2023-01-12..2024-12-31) are excluded (`no_data_beyond_tape` or
   `no_bars_for_exit_walk`, both logged as WARNING and counted) — **the scored TRAIN book is actually
   2025-01-02..2025-06-30 (n=56), not the full 2023-2024 window.** TRAIN's 1.9-2.1 fills/week
   (nominally failing the ≥3/wk bar) is an artifact of this shrunken, unrepresentative TRAIN, not
   necessarily the true ORB retest frequency. A real verdict on TRAIN needs the 2023-2024 minute bars
   fetched, out of scope here.

3. **Exit rule reproduces the static-lock core only** (stop=range_low, arm at fill+1.75R', ratchet to
   fill+0.5R', 15:55 ET time exit per the PREREG text) — NOT the Rule-M/Rule-D "touchgo" early exits or
   the SZ1 ATR stop floor (`study_orb_pipeline_static_lock.py:313-389,392-524`). The PREREG's own
   language ("the stop, the target as a function of R, partials if any, the time exit") does not
   clearly require touchgo; disclosed as a simplification, not a hidden one.

4. **`range_low` is reconstructed, not sourced directly.** `population.csv` has no `range_low`/
   `range_size` (dollar) column; `range_low = trigger * (1 - range_size_pct/100)` follows
   `study_orb_features.py:262-263`'s naming convention but was not independently checked against the
   sizing code within this budget. If this reconstruction is off, every R' and stop/lock level here
   shifts with it.

5. **Independent reimplementation (project rule #1) has NOT happened.** This report is the single
   builder's implementation from the PREREG's prose; per `feedback_independent_check_before_claims`,
   a second agent that has not read `cell_1562.py` must rebuild it and compare fills row-by-row
   (fill-set Jaccard ≥ 0.99, net R' within 0.01R on ≥99%) before this FAIL (or any future PASS) is
   relayed to the owner as a claim, not just a research note.

6. **Bar-fallback fills use the limit price, not a sub-limit tape print**, for the ~206 (of 302
   scored, 68%) signal-cell rows whose retest fill (or continued search) fell outside the tape's
   09:40:00 ET cutoff — an obtainability-conservative choice (never better than the resting order's
   own price), but it means the tape's "strictly below" fill mechanic (the PREREG's literal rule) is
   tape-verified for only the minority of fills; the rest are a coarser bar-level proxy.

## Consequence (per PREREG)
FAIL on both cells at the pass bar as specified. ORB keeps the chase entry; the withdrawal share
(93-100%, generalises the HOD pattern) and the paired-ΔR point estimates (+0.12 to +0.45 R, positive
every split) stay on record as a mechanism that is directionally real but not yet statistically
resolved — a candidate for re-test once (a) the 2023-2024 minute bars exist locally and (b) the base
leg's re-costing (caveat 1) is fixed, not a closed question. Per
`feedback_negative_result_is_not_a_stopping_point`: this is a claim about the test's power and data
coverage first, not a mechanism disproof.

## Files
- `research/orb_retest/cell_1562.py` — builder (docstrings, verbose progress, WARNING on every
  fallback/exclusion)
- `research/orb_retest/test_cell_1562.py` — 15 unit tests, all passing
- `research/orb_retest/cell_1562_fills.csv` — 820 rows (410 signals x 2 cells), one row per
  signal-per-cell with every intermediate field (trigger/limit/fill/exit/R'/excluded reason)

## Judge (main session, 2026-09-27 07:00 UTC) — FAIL both cells; the ORB break does NOT reward waiting for the pullback

* Frozen bar: FAIL on power and tails on both sides (builder pooled paired t 1.4, VAL ex-top-5 % −0.13 / −0.19;
  rebuild pooled t 1.9). Row agreement 68 % within 0.01 R (same-bar stop/lock tie-breaks; 143 fills the builder could
  not walk for lack of 2023–24 bars). TRAIN is effectively 2025H1 only (n 56): the 2023–24 half has no minute bars on
  disk, and neither tape covers the retest windows (the ORB tape ends at 09:40), so the retest search ran on 1-minute
  bars — every fill and exit here is a bar approximation.
* WITHDRAWN (both refuters): the builder's "paired ΔR positive on every split, the HOD learning generalises". The base
  leg was pnl_replay / 375 on a fixed $3,333 notional — about 0.41 × a true R — while the retest leg was a true R′.
  In consistent units (the replay chase fill walked through the SAME exit code) the paired ΔR is NEGATIVE on every
  split of both cells: −0.01 to −0.12, pooled −0.07 (t −0.8) / −0.06 (t −0.6). Three more defects all favoured the
  retest: tape fills booked at the print below the limit (+0.10 R′ unobtainable), fill-bar look-ahead in the lock
  arming, gap-through stops filled at the stop.
* The informative part: the ORB signals that never retest are the winners — +1.66 / +2.10 R on the never-retest cohort
  (n 48 / 38), so the retest book INCLUDING its missed signals (+0.20 / +0.21 R) sits below the chase base (+0.27 R).
  On HOD the retest recovered an immediacy cost on a population with zero gross; on ORB the immediacy cost is the
  price of catching the fills that run without looking back. Withdrawal within 15 min is 93–100 % on bars (the
  pattern exists), but the pullback buyers get the losers. Programme count 1,563.
Consequence per PREREG: ORB keeps the chase entry. What transfers from the HOD line is the exit side: the stop-limit
exit (measured slip 35 → ≈ 13 bps) goes into the ORB engine with the execution repair. No retest mode.
