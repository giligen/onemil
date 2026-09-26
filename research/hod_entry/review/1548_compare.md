# Compare 1,548 — builder vs rebuild

Builder = `cell_1548_fills.csv` (from `cell_1548.py`, `RESULT_1548.md`).
Rebuild = `rebuild_1548_fills.csv` (from `rebuild_1548.py`, prose-only, `REBUILD_1548.md`).
Keyed by (day, symbol); both files carry cells 1548 and 1549 together (9,911 rows/cell each).

## 1. Parameters (d, s, W) — DO NOT MATCH

| | builder (RESULT_1548.md, given) | rebuild (REBUILD_1548.md) | relative diff |
|---|---|---|---|
| d (limit below level) | 0.9441% | 1.021% | +8.1% |
| s (stop below level)  | 1.9713% | 2.137%  | +8.4% |
| W (resting window)    | 120.00 min | 120.0 min | 0% (match) |

**Root cause, traced exactly.** Part-A anatomy counts (TRAIN-H2, all-rows scope):
- extender n: builder 865 vs rebuild 948 (**+83**, +9.6%)
- non_extender n: builder 518 vs rebuild 518 (**exact match**)

Same pattern on every split/scope: VAL all-rows extender 833 vs 913 (+80); TRAIN real-SIP
extender 584 vs 639 (+55); VAL real-SIP extender 589 vs 643 (+54). Non-extender counts match
exactly in all four. Rebuild's extender+non-extender totals equal the Part-B population size
exactly (948+518=1466=TRAIN-H2 kept n; 913+1135=2048=VAL kept n); builder's totals fall short by
precisely the extender gap (865+518=1383=1466−83; 833+1135=1968=2048−80). Builder's own caveat
("missing-bar / no-touch-found / invalid-R rows are excluded ... never imputed") accounts for
this: builder drops ~83/80 such rows from the anatomy population; rebuild instead classifies
them as extenders. That inflates rebuild's extender n by ~9.6% and pulls its d/s quantiles up by
~8% — outside the 5% agreement bar. **Part A quantiles do NOT agree within 5% relative.**

## 2. Jaccard of the 1,548 filled sets

Builder filled n = 7,090; rebuild filled n = 6,866; intersection = 6,866; union = 7,090.
**Jaccard = 0.9684.** Rebuild's filled set is a strict subset of builder's (224 rows filled-only-
in-builder, 0 filled-only-in-rebuild) — consistent with rebuild's wider d/s shifting the limit
price so a small number of builder fills don't fill in rebuild.
Split: TRAIN Jaccard 0.9706 (3,131 vs 3,039 filled), VAL Jaccard 0.9667 (3,959 vs 3,827 filled).

Cell 1549 (unconditional base entry, no d/s dependency): fill sets are identical both files
(fill_share = 1.0 everywhere) → **Jaccard = 1.0**.

## 3. Share of common-filled rows within 0.01 R

| cell | n common-filled | share within 0.01R |
|---|---|---|
| 1548 overall | 6,866 | **30.7%** |
| 1548 TRAIN | 3,039 | 2.8% |
| 1548 VAL | 3,827 | 52.8% |
| 1549 (all, no filter needed) | 9,911 | **27.5%** |

Low agreement even on 1549, which has no d/s dependency at all — this points to a **second,
independent discrepancy**: a systematic small offset (0.01–0.1 R) present even when both sides
classify the exit the same way. Sampled same-`why` (`stop`) rows on 1549 show builder
consistently a few hundredths more negative than rebuild (e.g. NGNE 2025-07-02: builder −1.455
vs rebuild −1.497; MRAL 2025-07-01: −1.136 vs −1.126) — small entry-price / cost-model deltas,
not a why-classification difference. Bucket counts on 1549: 2,721 rows ≤0.01R, 4,402 rows in
0.01–0.1R, 2,614 in 0.1–1R, 174 >1R.

## 4. VAL means side by side (from RESULT_1548.md / REBUILD_1548.md, kept population)

| cell | builder VAL mean R (t, n, fill share) | rebuild VAL mean R (t, n, fill share) |
|---|---|---|
| 1548 SWEEP | −0.0249 (t −0.29, n=1,641, 0.801) | −0.006 (t −0.07, n=1,596, 0.779) |
| 1549 WIDE  | −0.1134 (t −2.14, n=2,048, 1.00) | −0.088 (t −1.67, n=2,048, 1.00) |

Same sign both cells, both books FAIL the pass bar either way (VAL mean R needs ≥ +0.15, t ≥
2.5). Magnitudes differ (1548: −0.025 vs −0.006, both near zero; 1549: −0.113 vs −0.088, ~22%
relative gap) but the qualitative conclusion — both cells fail on VAL — is unaffected by the
discrepancy.

## 5. Dominant cause of the 20 largest per-fill differences (cell 1548, common-filled)

All 20 largest |Δ net_R| (magnitude ~6.5–6.9 R) are **stop-vs-target exit-classification
flips**: builder says `stop` where rebuild says `target`, or the reverse, for the identical
(day, symbol) fill — e.g. 2025-08-04 RGTX: builder stop −1.21R, rebuild target +5.40R;
2026-05-28 HOOW: builder −1.21R stop, rebuild +5.40R target. Because a SWEEP target/stop payoff
is a fixed function of (d, s) alone (independent of price), every "hit target" row nets the same
constant R (+5.3955) and every "hit stop" row nets ~−1.13 to −1.20 (small per-trade cost
variance) — so these swings are binary regime flips, not scaling noise. Cause: the wider
rebuild stop (s=2.137% vs builder 1.9713%) changes which barrier price touches first within the
120-min window for a minority of names — a direct downstream effect of the §1 parameter gap.
Across the full common-filled population the same flip type shows up as the 2nd-most-common
why-mismatch (`stop`→`target` or reverse: 29/3,039 TRAIN, 43/3,827 VAL), behind the
more numerous but lower-magnitude `stop`↔`eod` flips (82 TRAIN, 117 VAL) which are also
attributable to the shifted stop level (a marginally-wider stop lets some trades ride to EOD
instead of stopping).

## Bottom line

- **Params do not match** (d, s off by ~8%, W matches); traced to a builder-side exclusion of
  ~83/80 missing-bar/no-touch rows from the anatomy population that rebuild instead counts as
  extenders.
- **Fill sets agree well** (Jaccard 0.968 on 1548, 1.0 on 1549 as expected).
- **Per-fill R agreement is weak** (31%/27% within 0.01R) from two distinct causes: (a) the
  d/s-driven stop-vs-target/eod classification flips (dominant on the 20 largest diffs), and
  (b) a smaller, pervasive cost/entry-price offset present even on 1549's unconditional entry,
  not yet isolated — worth a follow-up diff on the cost/entry-price computation between
  `cell_1548.py` and `rebuild_1548.py` before either result is used as reportable.
- **Directional conclusion is robust to the discrepancy**: both builder and rebuild show both
  1548 SWEEP and 1549 WIDE **failing the VAL pass bar**, matching the frozen verdict in
  `4c4331a`/`8ae61e9` that this exit surface is negative out of sample. The owner-relayed framing
  ("AUC 0.715 ... kept set still loses on VAL ... 2025H2 only, not 2026H1 — regime, not edge")
  is corroborated by the rebuild, not contradicted by it.
