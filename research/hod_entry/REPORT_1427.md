# REPORT — cell 1,427: independent SIP rebuild of E1 (PREREG_1427.md)

Generated 2026-09-25 15:44 UTC by research/hod_entry/sip_rebuild.py (written from the PREREG prose only; entry_replay.py / REPORT.md / hod_ofi/pipeline.py not read).

## Step 2 — agreement check vs cell 1,423 VAL (frozen bands)

* VAL E1 mean net R (fills): **+0.240** vs +0.384 → diff -0.144 (band ±0.05) → OUT
* VAL fill rate (fills / usable tape): **32.7 %** vs 29.9 % → diff +2.8 pp (band ±10 pp) → IN
* AGREEMENT VERDICT: FAIL — TEST NOT OPENED

## TRAIN-H2 / VAL tables (SIP consolidated tape)

| cohort | signals | usable (coverage) | lost | missing win / lose | fills (rate of usable) | mean net R | day-clust t | ex-top-5 % | stopped in break bar | fills/wk after slots (mean R) | no-fill cohort B0 net R |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 15 bps (frozen) — TRAIN-H2 | 3503 | 3208 (91.6 %) | 0 | 4.0 / 8.9 (gap 5.0 pp) | 1148 (35.8 %) | +0.293 | 3.72 (120 d) | +0.204 | 8 | 30.0 (+0.239) | -0.464 (n 2060) vs fills +0.276 |
| E1 15 bps (frozen) — VAL | 4745 | 4357 (91.8 %) | 0 | 3.8 / 9.4 (gap 5.6 pp) | 1424 (32.7 %) | +0.240 | 3.63 (102 d) | +0.148 | 4 | 37.4 (+0.189) | -0.489 (n 2933) vs fills +0.237 |
| limit 5 bps — TRAIN-H2 | 3503 | 3208 (91.6 %) | 0 | 4.0 / 8.9 (gap 5.0 pp) | 607 (18.9 %) | +0.360 | 3.55 (116 d) | +0.274 | 3 | 19.5 (+0.312) | -0.317 (n 2601) vs fills +0.307 |
| limit 5 bps — VAL | 4745 | 4357 (91.8 %) | 0 | 3.8 / 9.4 (gap 5.6 pp) | 772 (17.7 %) | +0.319 | 4.00 (101 d) | +0.231 | 3 | 26.7 (+0.322) | -0.374 (n 3585) vs fills +0.316 |
| limit 30 bps — TRAIN-H2 | 3503 | 3208 (91.6 %) | 0 | 4.0 / 8.9 (gap 5.0 pp) | 1467 (45.7 %) | +0.263 | 3.76 (123 d) | +0.173 | 9 | 33.7 (+0.202) | -0.585 (n 1741) vs fills +0.259 |
| limit 30 bps — VAL | 4745 | 4357 (91.8 %) | 0 | 3.8 / 9.4 (gap 5.6 pp) | 1847 (42.4 %) | +0.218 | 3.15 (102 d) | +0.126 | 8 | 40.0 (+0.169) | -0.593 (n 2510) vs fills +0.212 |
| 15 bps, fill ask + 1 tick — TRAIN-H2 | 3503 | 3208 (91.6 %) | 0 | 4.0 / 8.9 (gap 5.0 pp) | 1148 (35.8 %) | +0.283 | 3.54 (120 d) | +0.194 | 8 | 29.8 (+0.225) | -0.464 (n 2060) vs fills +0.276 |
| 15 bps, fill ask + 1 tick — VAL | 4745 | 4357 (91.8 %) | 0 | 3.8 / 9.4 (gap 5.6 pp) | 1424 (32.7 %) | +0.228 | 3.43 (102 d) | +0.136 | 4 | 37.0 (+0.179) | -0.489 (n 2933) vs fills +0.237 |

Cost: entry = half NBBO spread at the fill instant; exit = B0 per-signal half-spread + 2 bp. Fill rate = fills / usable-tape signals. Slots = run_consol.simulate_slots (first 12/day, 4 concurrent) ordered by fill minute.

## Step 2b — discrepancy investigation (TEST NOT opened)

The VAL mean is OUT of band (+0.240 vs +0.384, −0.144 R); the fill rate is IN (32.7 % vs 29.9 %). Trade-by-trade
comparison on (day, symbol, entry_m) against cell 1,423's per-signal output `research/hod_entry/replay_signals.csv`
(A = cell 1,423 single-venue XNAS replay, B = this SIP rebuild; level identical on 99.96 % of signals):

| split | A fills (mean net R) | B fills (mean net R) | filled in both | B-only fills, A tape usable (B mean) | B-only fills, A tape VOID (B mean) | A-only fills (A mean; B reason) | B fills on A's usable set (B mean) |
|---|---|---|---|---|---|---|---|
| TRAIN-H2 | 895 (+0.262) | 1148 (+0.293) | 798: A +0.292 / B +0.281, exit agree 98.2 %, median dR +0.001 | 148 (+0.342) | 202 (+0.305) | 97 (+0.015; ask>limit 83, no SIP print>=trigger 0, no_tape 14) | 946 (+0.291) |
| VAL | 1059 (+0.384) | 1424 (+0.240) | 923: A +0.370 / B +0.341, exit agree 98.8 %, median dR +0.002 | 183 (+0.139) | 318 (+0.005) | 136 (+0.482; ask>limit 120, no SIP print>=trigger 0, no_tape 16) | 1106 (+0.308) |

Reading: where both implementations fill the same signal they agree (exit reason ~98–99 %, median ΔR ≈ 0), so the
coding and the cost convention reproduce. The VAL gap comes from the NON-overlapping fills, i.e. the tape (XNAS vs
SIP NBBO), not the code: on VAL the SIP-only fills earn +0.005 (A VOID) / +0.139 (A ask above limit) and the A-only
fills +0.482. **But the sign flips on TRAIN-H2**: there the SIP-only fills earn +0.31 / +0.34 and the A-only fills
+0.015, and the SIP book is HIGHER than A (+0.293 vs +0.262). So the single-venue tape does not systematically pick a
better subset; the VAL +0.384 was partly a favourable draw of ~140 non-overlapping fills. On the consolidated NBBO the
E1 book is +0.293 (TRAIN-H2) / +0.240 (VAL), day-clustered t 3.7 / 3.6, ex-top-5 % +0.20 / +0.15, coverage 92 %,
winner/loser missingness gap 5.0 / 5.6 pp (at/over the 5 pp rail). Per the PREREG the agreement band is missed →
TEST stays sealed; how to proceed is the PREREG owner's decision.

## Step 2c — re-run with the prevailing-quote history (quotes back to S-900 s, else 09:30)

Only the quote fetch changed (`extend_quotes`: a signal whose cached window has no valid NBBO at or before S-60 s gets the last valid quote in [S-900 s, S-65 s), else in [09:30, S-900 s)). No rule constant or fill logic changed. TEST not opened.

| cohort | signals | usable (coverage) | lost | missing win / lose | fills (rate of usable) | mean net R | day-clust t | ex-top-5 % | stopped in break bar | fills/wk after slots (mean R) | no-fill cohort B0 net R |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 15 bps, quote history — TRAIN-H2 (all signals) | 3503 | 3481 (99.4 %) | 0 | 0.5 / 0.5 (gap 0.1 pp) | 1165 (33.5 %) | +0.285 | 3.45 (120 d) | +0.196 | 8 | 30.4 (+0.237) | -0.517 (n 2316) vs fills +0.274 |
| same, restricted to the 3208 previously-usable — TRAIN-H2 | 3208 | 3208 (100.0 %) | 0 | 0.0 / 0.0 (gap 0.0 pp) | 1148 (35.8 %) | +0.293 | 3.72 (120 d) | +0.204 | 8 | 30.0 (+0.239) | -0.464 (n 2060) vs fills +0.276 |
| S-65 s run, same 3208 signals — TRAIN-H2 | 3208 | 3208 (100.0 %) | 0 | 0.0 / 0.0 (gap 0.0 pp) | 1148 (35.8 %) | +0.293 | 3.72 (120 d) | +0.204 | 8 | 30.0 (+0.239) | -0.464 (n 2060) vs fills +0.276 |
| E1 15 bps, quote history — VAL (all signals) | 4745 | 4717 (99.4 %) | 0 | 0.2 / 0.7 (gap 0.5 pp) | 1443 (30.6 %) | +0.238 | 3.53 (102 d) | +0.146 | 4 | 37.5 (+0.193) | -0.538 (n 3274) vs fills +0.231 |
| same, restricted to the 4357 previously-usable — VAL | 4357 | 4357 (100.0 %) | 0 | 0.0 / 0.0 (gap 0.0 pp) | 1424 (32.7 %) | +0.240 | 3.63 (102 d) | +0.148 | 4 | 37.4 (+0.189) | -0.489 (n 2933) vs fills +0.237 |
| S-65 s run, same 4357 signals — VAL | 4357 | 4357 (100.0 %) | 0 | 0.0 / 0.0 (gap 0.0 pp) | 1424 (32.7 %) | +0.240 | 3.63 (102 d) | +0.148 | 4 | 37.4 (+0.189) | -0.489 (n 2933) vs fills +0.237 |

* TRAIN-H2: previously-usable signals identical per signal (status and net R): YES
* VAL: previously-usable signals identical per signal (status and net R): YES
* VAL vs cell 1,423 bands (informational): mean +0.238 vs +0.384 (diff -0.146), fill rate 30.6 % vs 29.9 % (diff +0.7 pp)

