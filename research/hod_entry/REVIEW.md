# REVIEW — cell 1,427 (HOD-break resting buy-stop-limit, "E1") — adversarial judge, 2026-09-25

Inputs: five lens files in `research/hod_entry/review/` (A look-ahead, B fill obtainability, C cost, D consistency /
tail / multiplicity, E capacity / operations), plus the judge's own re-score of the claim's CSVs
(`review/judge_stress.py`, `review/judge_stress2.py`, outputs `review/judge_stress.csv`, `review/slot_*.csv`), plus
`scripts/cadence_bar.py` on the slotted live-config series. Nothing committed.

## The claim

A buy-stop-limit resting at the HOD-break level + $0.01, limit = level × 1.0015, filled at the SIP NBBO ask at the
first consolidated print ≥ trigger inside the break bar, B0 stop / 2 R target / 15:55 exit, measured half-spread entry
cost, earns net per fill **+0.285 R (2025H2, n 1,165, t 3.45), +0.238 R (2026 Jan–May, n 1,443, t 3.53), +0.330 R
(sealed TEST 2026-06-01..09-04, n 972, t 2.12, ex-top-5 % +0.243)**; fill rate 28–34 %; ~34 fills/wk after 12/day,
4-concurrent slots; the unfilled signals average −0.57 R.

## Verdict: REFUTED as stated. The mechanism survives as an exploration-tier candidate at $100 risk.

The headline numbers are not what a live order would earn. Three things stand between them and a live book:
1. **The population is not the live population.** The rv admission gate uses the break bar's full volume, which a
   resting order cannot know when it fills (lens A, confirmed in code). The causal rebuild, cell 1,438, has
   **2,513 causal fills through 2025-12-18, against 1,165 for all of 2025H2 in 1,427**
   (`causal_arming.log`, day 120/230). A live order fills about twice as many symbol-days, and nobody has scored those
   extra fills yet. Neither VAL nor TEST has been read for 1,438.
2. **Under realistic fills and costs, TEST no longer clears t ≥ 2** (judge re-score below). The fill mean stays
   ≥ +0.10 R in every split unless stop slippage is severe. That meets the pre-committed default-refute rule.
3. **The live order path does not exist.** `entry_mode: resting_stop_limit` is hard-coded dry, has no submit,
   cancel or replace lifecycle, and resolves crosses at bar close with a later quote. Its own WARNING says "not
   tape-accurate" (lenses C, E).

## Judge re-score: the live config (12/day, 4 concurrent) under realistic haircuts

The haircuts come from the lens measurements; none is fitted here. **Stop slip** is charged only on touch-stops
(1,450 of 1,636 stop exits; gap-throughs already fill at the open in B0), **net of the exit leg the model already
charges** (mean 16.4 bps: half-spread plus 2 bp). This avoids a double charge; lens C's stress charged the spread
twice. Plausible slip = 0.488 × 47 bps (live BF/ORB stop exits, lens C). Severe slip = 79 bps on every touch-stop
(p10). **Size** haircut = −0.024 R at $100 risk (−0.048 R severe, the $375 figure, lens B). **Latency** removes 32.4 %
of fills at random and then re-runs the slots; the table shows the median of 100 draws and, in brackets, the share of
draws with t ≥ 2. Lens B found that latency-dropped fills were worse than the survivors, so random removal is
conservative for the mean. t is the claim's own day-clustered t (`hod_exit_lab/score_cells.day_clustered_t`).

| mean net R / day-clustered t | TRAIN-H2 | VAL | TEST | TEST ex best week | pooled |
|---|---|---|---|---|---|
| as reported (unslotted) | +0.285 / 3.45 | +0.238 / 3.53 | +0.330 / 2.12 | +0.241 / 1.56 | +0.278 / 5.29 |
| **live slots, model costs** | +0.237 / 3.85 | +0.193 / 3.39 | +0.261 / 2.64 | +0.253 / 2.35 | +0.225 / 5.77 |
| slots + plausible stop slip + size | +0.186 / 3.02 | +0.147 / 2.60 | **+0.218 / 2.09** | +0.210 / 1.83 | +0.178 / 4.50 |
| **same + 32 % latency attrition** | +0.188 / 2.43 (0.80) | +0.174 / 2.54 (0.81) | **+0.206 / 1.46 (0.22)** | +0.172 / 1.13 (0.11) | +0.190 / 3.71 (1.00) |
| slots + severe stop slip + size | +0.033 / 0.65 | +0.004 / 0.25 | +0.086 / 0.41 | +0.078 / 0.26 | +0.033 / 0.79 |

Reading:
* At the live config, TEST sits on the t = 2 line once a moderate stop slip and the size haircut are applied. With
  latency attrition, the median draw falls to t 1.46.
* **Stop-exit slippage decides the verdict.** At the plausible slip the book is +0.15 to +0.22 R. At the severe slip
  it is ≈ 0 in every split. It has never been measured on these names, which are $20+ with a median of $34, but it can
  be measured on SIP tape now.
* Unslotted, the day-clustered t is much weaker than the fill mean suggests. With severe slip, TEST's fill mean is
  +0.159 but its day-mean t is −0.29. The profit sits on the high-fill days, and one of them is the 2026-08-01/07 week:
  +121.1 R of TEST's +320.7 R, 38 %. The live 12/day cap spreads that concentration out, which is why the slotted t
  is higher.
* **Cadence bar on the slotted live config (VAL; `scripts/cadence_bar.py`)**. C1, C2, C4 (80 % green against a null
  of 50 %), C5 (37.5 fills/wk) and C7 pass. **C3 (shallow reds) fails**: weekly P10 −5.99 R, worst week −10.17 R.
  With the plausible haircut: −8.09 R and −11.58 R. At $100 risk that is a −$600 to −$800 P10 week and a worst week
  of about −$1,200. The TRAIN cadence row is not usable: the scorer's TRAIN window is all of 2025, but this study
  covers H2 only, so 26 empty weeks distort C3's under-water count, C5 and C7.

## Per-lens verdicts (with the judge's corrections)

| lens | lens threat | judge | what stands / what was corrected |
|---|---|---|---|
| A look-ahead | major | **major, upheld** | The rv gate on `cumv[i]` is a real look-ahead. On re-score, 10.8 % of fills were mis-armed; they still averaged +0.154 R, and purging them lifts the rest to +0.334 R. Correction: the re-score cannot see population identity, and the 1,438 log shows the causal rule fills about 2.2× more symbol-days. So "purging helps" says nothing about what the live order earns. Only 1,438 does. |
| B fill | major | **major, upheld** | There is no size check anywhere (`as`/`bs` are never stored). Of the modeled fills, 32 % do not survive 250 ms, and 23 % ($100 risk) to 59 % ($375) exceed the displayed ask size. Per-fill R moves only −0.02 to −0.05 R; the damage is to frequency and t. Addition: the model assumes the order dies after a missed first print. If the live order keeps resting, it fills on pullbacks, which is adverse selection (the halt-resume lesson). The lifecycle must be pinned, and the backtest must match it. |
| C cost | minor | **major (upgraded)** | The flat stop = target = eod exit cost is real (0.174 / 0.173 / 0.119 R), and the dry run has zero rows for this mechanism. Two corrections: (i) the stress charged the full live slip on top of the modeled exit leg, which double-charges the spread; (ii) "t ≥ 2 is unaffected" is **wrong**, because day-clustered t falls much faster than the fill mean (table). Net of the double charge, plausible slip leaves TEST knife-edge and severe slip removes the edge. |
| D consistency | major | **major, upheld** | 15 of 15 months are positive, and the count-matched null ranks 200 of 200; both are real. TEST is 38 % one week. C3 fails at the live config (judge's slotted run; it is no longer unknown). Corrections: (i) the −1.5 to −2.1 R stop-outs are **modeled** gap-through fills, not evidence of unmodeled slippage; (ii) Bonferroni at N = 1,438 against a sealed single-read TEST uses the wrong family, because the seal is the multiplicity control. The programme count still stands: 1,438 cells. At the live config the pooled base t of 5.77 clears a 1,438-cell Bonferroni bar (≈ 4.3). Under realistic costs, pooled t is 3.7 to 4.5, which does not clear it cleanly. |
| E capacity / ops | major | **major, upheld** | The binding constraint is the entry minute's volume, not ADV. $1,000 risk breaks buying power (p90 $337K against $260K). The live order path does not exist, and the dry resolver is bar-close with a stale quote. Correction: downgrading oversized fills to −0.571 R overstates the damage, because an unfilled limit order earns 0, not the B0 counterfactual. The right model is a partial fill with adverse selection. The conclusion stands: **$100 risk maximum**. |

## Surviving statement

In three disjoint periods, HOD-break signals whose buy-stop-limit (level + $0.01, limit level × 1.0015) could fill at
the SIP ask inside the break bar beat the unfilled signals by ≈ 0.8–0.9 R. The fills are net positive at the live
12/4 config:
* **+0.237 / +0.193 / +0.261 R (t 3.85 / 3.39 / 2.64)** at model costs;
* **+0.186 / +0.147 / +0.218 R (t 3.0 / 2.6 / 2.09)** after a moderate incremental stop slip and the $100-risk size
  haircut;
* with 250 ms latency attrition, the median TEST draw is **+0.206 R at t 1.46**, and pooled is **+0.190 R, t 3.71**.

A 79 bps stop slip takes every split to ≈ 0. This is a positive point estimate with a mechanism: frequency is high
(≈ 25–37 fills/wk) and the downside is bounded at $100 risk. It qualifies for the **live exploration tier at $100 risk
once the live rule (cell 1,438) passes VAL and the fixes below are in**. It is **not** a locked +0.33 R. The live
causal order's own number is cell 1,438, which is running.

## Must be fixed or checked before any real order (in order)

1. **Cell 1,438 must pass VAL; then read TEST, disclosed as the second TEST read.** That rule is the live rule, and it
   fills about 2× as many symbol-days. Report it at the **12/4 slot config**, not unslotted. If it fails, the book is
   closed, as its PREREG pre-commits.
2. **Measure stop-exit slippage on SIP tape** for 1,427's 1,450 touch-stops and for 1,438's fills. Fill at the bid
   250 ms after the first print ≤ stop, and report it against the modeled exit leg (mean 16.4 bps). This single
   number moves the book between +0.2 R and 0. Use the measured per-trade figure as the exit cost, never a band.
   Do not go live if the mean incremental slip is > 40 bps.
3. **Put latency and depth into the shared entry helper** (`trading/hod_break.resting_entry_fill` and the research
   copy, parity-tested). Fill at the ask ≥ 250 ms after the trigger print, and only if it is ≤ the limit. Store
   `as`/`bs` in the SIP fetcher, fill up to the displayed size, and handle the remainder with the live lifecycle's rule.
   **Pin that lifecycle**: cancel at the end of the break bar or keep resting. Measure the pullback-fill cohort if it
   keeps resting. Re-score 1,438 VAL (and TEST, if read) under this helper. Pass bar: slotted mean ≥ +0.10 R and
   t ≥ 2.
4. **Check the trigger semantics on the real API (paper account).** Find out whether Alpaca's stop-limit triggers on
   odd-lot, 1-share and sub-penny prints. B found 14 % of triggers are 1-share prints and 6 % are sub-penny. If the
   broker ignores them, the fill population changes. Also pin whether Alpaca holds the stop or routes it, and
   whether the tape shows extended-condition prints.
5. **Build the live order lifecycle** (it does not exist):
   * submit at arm (bar j close) and cancel/replace on re-arm;
   * cancel at the lifecycle rule and at `last_entry_minute`;
   * on a fill (including partial), attach the consolidation-low stop and the 2 R target through StopMonitor at the
     filled quantity, and exit at 15:55;
   * use a `client_order_id` prefix so **owner orders are never touched**;
   * write a crash-restart reconciliation for resting orders.
   Unit, integration (paper) and system tests are all required.
6. **Size and buying-power guards.**
   * $100 risk per trade (exploration tier).
   * `shares ≤ min(risk / R, 5 % of the prior bar's volume)`, with a WARNING log when the cap binds.
   * A 4-concurrent buying-power guard that accounts for the owner's positions.
7. **C3 shallow-reds fails at the live config.** Either the owner accepts it explicitly (worst week ≈ −$1,200 at
   $100 risk), or a weekly loss tripwire is set. Proposed: pause the book for the week at −8 R realized.
8. **Kill and pause rules, written before launch.**
   * Pause at the BT P90 gap + 2 weeks without a strong week (cadence tripwire).
   * Kill if the live fill mean is < −0.15 R after 40 fills.
   * Never advance a size stage while realized P&L is < 0.

## What the 10-session dry run must show

Run it on the **paper account with the real lifecycle** (item 5), and run a **nightly SIP tape replay** of every arm
through the shared helper (item 3). The engine's bar-close resolution is not tape-accurate, so the ledger alone proves
nothing. Honest power: 10 sessions ≈ 50–70 fills, with an SE of ≈ 0.16 R per fill. The dry run can confirm
**obtainability and parity**. It cannot confirm a +0.2 R edge; that comes from the exploration tier at 40 live fills.

| gate | band |
|---|---|
| logging | ≥ 95 % of arms logged (symbol, arm time, level, trigger, limit, stop); zero orphan orders at 15:56; zero orders on non-book symbols |
| parity: paper fills against the tape replay | fill/no-fill agreement ≥ 90 %; fill price within 1 tick on ≥ 90 % |
| frequency (after slots, 250 ms latency) | ≥ 30 fills in 10 sessions (≥ 3 per session). Expected ≈ 46–68 on the 1,427 population, up to 120 under causal arming (12/day cap) |
| fill rate (tape replay, latency-adjusted) | within ±10 pp of 1,438's latency-adjusted VAL fill rate. 1,427 reference: TEST 27.8 % raw, ≈ 19–23 % after latency |
| mechanism signature | the armed-but-unfilled cohort's B0-counterfactual mean ≤ fills' mean − 0.30 R (1,427 gap: ≈ 0.9 R) |
| stop-exit slip (measured) | mean incremental over the modeled exit leg ≤ 20 bps to proceed; > 40 bps means stop, the book is at breakeven |
| fill mean net R (tape replay, measured costs) | **confirm ≥ +0.10 R**; **continue** −0.15 to +0.10 R (exploration tier at $100 only if the signature holds); **stop** < −0.15 R on ≥ 50 fills (≈ 2 SE below the stressed +0.19 R) |

## Provenance

The claim's numbers were reconciled exactly by lenses A and D from `sip_rebuild_val.csv` and `sip_rebuild_test.csv`.
The judge table comes from `review/judge_stress.py` (as reported, live slots, gross haircuts, with the double charge)
and `review/judge_stress2.py` (haircuts net of the modeled exit leg), both on those two CSVs, using the report's own
`run_consol.simulate_slots` and `score_cells.day_clustered_t`. The cadence run is `scripts/cadence_bar.py --trades
review/slot_base.csv|slot_plaus.csv --split VAL --slots 4 --r-dollars 100`. The 1,438 interim count comes from
`causal_arming.log` ([run] day 120/230, causal variant fills only, `causal_arming.py:323`). Programme count: 1,438 cells.
