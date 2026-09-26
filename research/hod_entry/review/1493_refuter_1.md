# Refuter 1 — cell 1,493–1,547 (retest bounce exit surface): obtainability, look-ahead, cost

Lens: obtainability / look-ahead / cost. Read `cell_1493.py`, `RESULT_1493.md`, `PREREG_1493.md`; recomputed
independently (scratch scripts r1–r4 in the session scratchpad; placebo recompute saved to
`review/1493_refuter1_placebo.csv`).

## Verdict under this lens
**The FAIL verdict stands (refuted = false).** Every defect found biases the book UP or is too small to matter, and
the zero-cost GROSS surface is already flat: max raw mean over all 55 cells = +0.009 % (TRAIN-H2) / +0.028 % (VAL),
best cell 3.0%|NONE gross t ≈ 0.0 / 0.1 (day-clustered). No cost convention, however generous, reaches the
+0.15 % / t ≥ 2.5 bar.

**But one headline interpretation IS refuted: "the retest instant is adverse selection, t −5.8 to −12.6 vs the
placebo".** That is a look-ahead artifact of the placebo window (below). It must be withdrawn before relaying.

## (d) Placebo — LOOK-AHEAD, headline claim withdrawn
The PREREG draws the placebo minute in [09:45, 15:00] excluding only the retest window, so ~21–23 % of draws fall
BEFORE the break. The symbol-day is in the population because it later breaks its HOD level, so a pre-break placebo
entry (median 2.8 % below the level) rides a run-up that is guaranteed by selection. Recompute (own RNG seed 1493,
3.0%|NONE, bar-open entry, same costs), paired with the builder's real cell:

| holdout | subset | n | real | placebo | margin | day-cl. t |
|---|---|---|---|---|---|---|
| TRAIN | all | 3,364 | −0.039 | +0.645 | −0.684 | −5.1 |
| TRAIN | post-window only | 2,599 | −0.035 | −0.074 | +0.039 | +0.26 |
| TRAIN | pre-break only | 765 | −0.055 | +3.087 | −3.14 | −25.4 |
| VAL | all | 4,307 | −0.035 | +0.620 | −0.655 | −5.9 |
| VAL | post-window only | 3,410 | +0.020 | −0.013 | +0.033 | +0.25 |
| VAL | pre-break only | 897 | −0.243 | +3.027 | −3.27 | −22.9 |

The "all" row reproduces the builder (−0.608, t −5.8). Restricted to causal minutes (after the retest window), the
retest timing is placebo-NEUTRAL (+0.03 pp, t 0.25): not adverse, not favourable. The placebo still fails the
+0.10 pp bar, so the verdict does not move. Bar-open placebo entry is itself not a real fill (touch of the open,
no queue); it favours the placebo slightly, irrelevant next to the selection effect. Also: `placebo_minute` seeds from
Python `hash()` of strings, randomised per process (PYTHONHASHSEED) — the placebo draw is not reproducible.

## (a) Target obtainability
Builder uses the through-print rule correctly: tape `price > target`, bars `h > target`, fill AT the target.
Entry is a through fill too (0.2 % of 1,500 sampled entry prints at the limit exactly). Queue risk on the
tape-phase targets (1.0%|tgt0.5 VAL: 38.6 % of target exits resolve inside the retest tape): of 325 sampled,
volume printed above the target in the tape phase median 1,678 sh, but 16 % ≤ 100 sh and 32 % ≤ 500 sh —
those are at risk of being lost to queue priority. Direction: optimistic for the book.
Defect: for 43 % of fills (3,819 / 8,973) the retest print lies in minute retest_minute + 1, yet the bar phase starts
at retest_minute + 1, so the entry minute's full bar (pre-entry prints ≥ level) is walked. 0.39 % of all
(fill, cell) rows are target exits at that bar with tsoff = 1 and in a 29-row sample NONE was confirmed by the
post-entry tape — spurious targets from pre-entry highs, concentrated on tgt0.5 cells (~180 rows each). Optimistic,
small.

## (b) Stop-limit tail / (f) cost units
Stops fill at min(print, stop) / the bar open on gap-through AND are charged the 1,463 stop-limit standard
(2.9/3.2 bps + 12 % tail at 94/76 bps), which is itself measured vs the stop price — a mild double charge:
mean extra gap 5.4 bps per stop on 3.0%|NONE (p99 87), 12.1 bps on 1.0%|tgt0.5. ≤ 2–5 bps per fill; cannot flip.
The tail was measured on the CL (~1.6 %) stop population; wider fixed stops on fast drops are unmeasured (could be
worse). Units are consistent: all costs in % of price on the exit leg; passive entry and limit target carry none,
EOD/time at the bid (9.7/11.5 bps) — consistent with the base standard (half-spread once on the base's aggressive
entry; a passive entry pays none). Removing every cost: best cell +0.028 % VAL, t 0.11.

## (c) Halts
84 `eod_fallback` rows (0.02 %) — bar paths ending before 15:55. No halt-specific handling; immaterial.

## (e) L3 stratum
The builder reads `model_1478_L3_predictions.csv` (14:35) — the ORIGINAL leaky run whose arm-bar features include the
fill bar (withdrawn per commit 3bf03e9); `model_1478_L3_v2_predictions.csv` (Amendment 3, arm bar = last closed bar)
exists and was not used. The report-only L3 TRAIN +0.62 % (t 1.94) is therefore not causal as stated; VAL +0.05 %
t 0.2 anyway. Report-only; no verdict effect. Should be re-cut on v2 before quoting.

## What must change in the relay
1. Drop "the retest instant is adverse selection / loses to its placebo by 0.6 pp, t −5.8 to −12.6"; say "placebo-
   neutral on causal minutes (+0.03 pp, t 0.25); the PREREG placebo window admitted pre-break minutes".
2. The closure is "gross ≈ 0 across the whole exit surface (max +0.03 %), cost makes it negative" — not adverse timing.
3. L3 stratum numbers come from the leaky v1 predictions.
