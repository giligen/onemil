# ORB in-flight family: EARLY CUTS (first 2-15 min) — clean-data walk-forward (2026-08-14)

Parallel loser-drag program, family `early_cut` (H1 touchgo re-tune, H2
stagnation time-stop, H3 range re-entry, H4 below-entry dwell, H5 VWAP loss).
Owner goal: cut the clean book's -$101.7K drag toward -$50K while keeping
>=85-90% of top-40 winner P&L.

**Headline: 4 of 5 hypotheses are dead — H3/H4/H5 are structural
winner-killers (retention 56-84% — monsters routinely chop below range_high /
entry / VWAP in their first 15 minutes before running), and H1's TRAIN-optimal
re-tune fails validation. The single survivor is H2: a stagnation time-stop
(no +0.25R MFE within 20 min of fill -> exit at market). It passes the full
pre-registered bar — drag falls AND winner retention is 100% in all three
windows on BOTH configs, zero top-40 casualties, ANNA/BNAI untouched — but its
magnitude is modest: OOS-2026 drag reduction +$1.5K on the $100K book (~4% of
OOS drag), +$6.8K full-timeline. It is a real, cheap improvement, not a cure.**

## Method

- Data: `analysis_results/orb_features_20260814_1741.csv` (DST-clean, 6,918
  rows). Per-trade physics rebuilt from cache.db 1-min bars by an extended
  copy of `research/scripts/orb_clean_harness.py` (scratchpad `ec_harness.py`;
  adds per-bar session VWAP from 9:30, typical-price x volume). 6,918/6,918
  pairs reconstruct, 0 trigger mismatches. Exit-sim parity vs the original
  harness: **$0.00 difference** on both books.
- Configs: (a) shipped-clean $100K (yaml fit, N4, $3K risk, PM mult, pdr8,
  cvY): 219 trades, +$96,454, matches the official clean book; (b) B+ $10K
  (TRAIN-refit, q40, pdr11, N3, cvY, $375 risk, uniform, no PM): 105 trades,
  +$9,931.
- Structural note: exit rules don't touch the selection stack, so the selected
  trade set is identical baseline-vs-variant — paired per-trade comparison is
  pipeline-integrated by construction (slots/dedup/vetoes unchanged).
- Walk-forward: grids fixed a priori from the task spec; winning parameter
  chosen on TRAIN=2025H1 ONLY (max TRAIN drag reduction s.t. TRAIN winner
  retention >= 85%), frozen, then reported on VAL=2025H2 and OOS=2026.
- Cohorts (config a): winners = top-40 baseline trades of the full clean book
  (TRAIN 9 / VAL 14 / OOS 17 members; +$22.1K / +$47.2K / +$127.8K); drag =
  ranks 41+ (-$30.3K / -$34.0K / -$36.4K). Config b: top-15 of its 105.
- All rules are live-implementable: 1-min bar closes, fill-anchored timers,
  running MFE, session VWAP from streamed bars. No lookahead.

## Baseline (for reference)

| config | total | tWR | dWR | MDD | months+ | worst m | top5 | ex-top5 |
|---|---|---|---|---|---|---|---|---|
| a $100K | +$96,454 | 39.7% | 41.1% | -$20,832 | 9/20 | -$13,187 | 105.4% | -$5,246 |
| b B+ $10K | +$9,931 | 39.0% | 40.4% | -$1,210 | 12/20 | -$413 | 88.1% | +$1,185 |

## H1 — Touchgo threshold re-tune (Rule M 0.3-0.7 x Rule D 0.5-1.0) — **FAIL**

36-cell grid on config (a). Facts:
- The Rule D axis is nearly inert on clean data (fires ~1-2x/book); every
  clean-data reading of d!=0.75 is noise-level. Rule D at 0.75 keeps its small
  positive contribution; leave it alone.
- The Rule M axis is monotone: drag reduction rises with the threshold
  (m=0.5 -> 0.6 -> 0.7 gives TRAIN dragΔ +0 -> +3.3K -> +6.0K) — until it
  starts eating winners.
- TRAIN-chosen (per pre-registered criterion): **m=0.7, d off** (TRAIN dragΔ
  +5,953, ret 100%). Frozen -> **VAL retention 83.2% — FAILS the 85% bar.**
  Casualties: DGNX 2025-11-17 (-$2,573), CRCA 2025-12-16 (-$1,866), CCUP
  2025-12-16 (-$1,761), BTQ 2025-10-16 (-$1,742), all tag_bb false-positives
  on eventual winners. On B+ VAL retention is 69.6%. Dead per protocol.
- Honest footnote (NOT a walk-forward result — visible only after unblinding
  VAL/OOS): m=0.6 passes everywhere on config (a) (ret 96.3/100/100%, dragΔ
  +3,277/+2,194/+7,263 = +$12.7K) but on B+ VAL its drag *worsens* (-$183),
  and it sits one grid step from the 0.7 retention cliff. If any axis in this
  family deserves a fresh, pre-registered forward test on new live data, it is
  Rule M 0.5->0.6. Do not ship it off this study.

## H2 — Stagnation time-stop — **PASS (chosen: MFE < +0.25R within 20 min of fill -> exit at market)**

Grid: mfe {0.25, 0.5} x window {10, 15, 20} min. TRAIN eliminated mfe=0.5
(winner retention 73-85% — too blunt) and windows <20 (TRAIN ret 82-84.5%:
slow-starting winners get clipped; several eventual multi-$K trades sit below
+0.25R MFE at 10-15 min). TRAIN-chosen: **(0.25R, 20 min)**, frozen.

| window | config a dragΔ | a retention | config b dragΔ | b retention |
|---|---|---|---|---|
| TRAIN 2025H1 | **+$4,783** (-30.3K -> -25.5K) | 100.0% | +$243 | 100.0% |
| VAL 2025H2 | **+$503** (-34.0K -> -33.4K) | 100.0% | +$67 | 100.0% |
| OOS 2026 | **+$1,522** (-36.4K -> -34.8K) | 100.0% | +$153 | 100.0% |

- **Zero top-40 casualties** (both configs). ANNA 2026-03-20 (+$41.0K) and
  BNAI 2026-01-23 (+$27.2K) untouched (both reached +0.25R inside 20 min).
- Standalone book, config a: total +$96,454 -> **+$103,262** (+$6,808), MDD
  -$20,832 -> -$20,328, ex-top-5 **-$5,246 -> +$1,562** (the drag book goes
  positive), top-5 share 105.4% -> 98.5%. Config b: +$9,931 -> +$10,393, MDD
  -$1,210 -> -$1,152, worst month -$413 -> -$369, ex-top-5 +$1,185 -> +$1,647.
- Fire anatomy (config a): 20/219 trades fire (9%); base exits of fired = 10
  stop / 10 losing-eod; 15 helped (+$10.5K, e.g. PDYN -$3,393 -> -$1,216), 5
  hurt (-$3.7K, worst WULF 2026-03-16 +$891 -> -$682). Not
  single-trade-carried: leave-one-best-fire-out stays positive in every
  window on both configs (TRAIN +2,606 / VAL +113 / OOS +437 on a).
- Monthly deltas (config a, nonzero): 2025-01 +2,433, 2025-02 +759, 2025-03
  +704, 2025-06 +888, 2025-12 +503, 2026-01 +1,493, 2026-02 +718, 2026-03
  -200, 2026-04 +390, **2026-06 -1,561 (month flips +1,501 -> -60; CWVX
  06-15 and TSDD 06-23 both cut then recovered)**, 2026-07 +682. 17 of 20
  months >= baseline; disclosed cost: one green month flips slightly red.
- Sensitivity (+/-20%): window 20 -> 24 is stable (dragΔ +3,789/+874/+2,169,
  ret 100% everywhere — passes outright); window -> 16 trips the TRAIN
  retention bar at 84.8% (one slow winner); mfe -> 0.2 loses the OOS effect
  (dragΔ -533); mfe -> 0.3 keeps drag gains but starts touching winners
  (ret 94.8-96.2% OOS). The passing plateau is (0.25R, 20-24 min) — real but
  not wide. Prefer the plateau's interior: if shipped, 20-22 min with 0.25R.
- Mechanism (why it should generalize): ORB monetizes immediate continuation;
  a trade that hasn't printed even +0.25R MFE in 20 minutes has demonstrated
  the morning momentum it was selected for is absent, and its expected value
  is the stop/losing-eod distribution (base exits of fired trades were 100%
  stops/losing-eods except the 2 recoverers). This is the same "momentum that
  works, works fast" logic as touchgo, extended from 2 min to 20.
- Live wiring sketch (NOT done here): fill-anchored 20-min timer in
  orb_engine `_ingest_bars`; track running MFE from fill; at the first bar
  close >= 20 min post-fill with MFE < entry + 0.25 x (range_high-range_low),
  `stop_monitor.force_exit(symbol, reason='stag')`. BT keys the timer to the
  market breakout bar; live should key to actual fill and reuse the touchgo
  late-fill guard pattern. Parity test mandatory before any enable.

## H3 — Range re-entry (K consecutive closes < range_high, first 15 min) — **NOGO**

K=2/3/5 all fail TRAIN retention (76.7/76.2/75.7%) and never recover (VAL
66-84%, OOS 66-89%). Entry is 30bps above range_high, so nearly every trade
— including the monsters — trades "back inside the range" early; K=2 costs
-$31K of OOS winner P&L on config a. The drag reductions (+$7-12K/window)
are real but bought with winner amputation. No parameter in the grid is
salvageable; structurally dead, not tuning-dead.

## H4 — Below-entry dwell (cumulative closes < entry > D in first 15 min) — **NOGO**

D=5/8/10: TRAIN retention 76.1-82.3% — no TRAIN-eligible parameter (bar is
85%). The gentlest setting (D=10) reads 82.3/87.1/93.1% with dragΔ
+6,394/+3,020/+6,816 — the best texture of the three dead rules, but it fails
the pre-registered bar in TRAIN and VAL and damages 2025 winners (winners
commonly spend 10+ of their first 15 minutes below the 30bps-padded entry).
Dead as specified. (If a future program revisits, the direction would be
longer windows + larger D, converging on... H2, which already does this job
without touching winners.)

## H5 — VWAP loss (close < session VWAP within first 10/15 min) — **NOGO**

Worst of the family: retention 66.8/77.1/70.8% (w=10) and 55.7/41.3/70.8%
(w=15) on config a; B+ VAL retention 30%. Gappers open far above session
VWAP but routinely tag it in the first minutes; the monsters are no
exception. Kills BNAI-class trades. Dead on arrival — do not revisit at
minute-scale windows.

## Family verdict

- Best rule: **H2 stagnation time-stop, exit at market if MFE < +0.25R
  within 20 min of fill** (2 free parameters, plateau at 0.25R x 20-24 min).
  Passed the full pre-registered walk-forward bar on both configs with 100%
  winner retention and zero top-40 casualties.
- Magnitude honesty: OOS-2026 drag reduction is +$1,522 on the $100K book
  (drag -$36.4K -> -$34.8K) and +$153 on B+. Full-timeline +$6,808 (a) /
  +$462 (b), leave-one-out robust, TRAIN-heavy. This rule alone moves total
  drag from -$100.6K to -$93.8K — nowhere near the -$50K target. The
  first-15-minutes axis beyond touchgo is largely mined out: winners and
  losers are near-indistinguishable at minute-scale price action (H3/H4/H5),
  and the one separable signal is *absence of any progress by minute 20*.
- Interaction caveat: H2 was validated ON TOP of shipped touchgo (0.5/0.75)
  with all other knobs at config defaults. If other families ship rules that
  change early-trade composition, re-run the H2 windows before stacking.
- Artifacts: scratchpad `ec_harness.py` (physics v2 + parity), `ec_sweep.py`
  (grids + TRAIN-only selection), `ec_final.py` (survivor deep-dive),
  `ec_physics_v2.pkl`, `ec_selections.pkl`, `ec_sweep_results.pkl`. Nothing
  in trading/ or orb.yaml touched; nothing committed.
