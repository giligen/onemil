# RESULT 1,690 — can ORB add-on pool P1 be improved? Verdict: **no improvement**

PREREG: `PREREG_1690.md` (FROZEN). Script: `1690_variants.py` (0 ERROR lines in `1690_variants.log`).
Full reads: `1690_reads.csv` (30 rows). P1 base book n=394 (2025-01-02..2026-09-18); bar-walk coverage for
variants (b)/(c) was **394/394 (100%)** from `data/cache.db` (not bars_sip.db — see PREREG).

## Reference: plain P1 (idea1), all three windows
| Window | n | fills/wk | mean R | day-clust t | worst wk | $ |
|---|---|---|---|---|---|---|
| TRAIN (2025) | 178 | 3.49 | +0.041 | 0.18 | −1.38R | +$2,753 |
| VAL (2026 H1) | 152 | 5.63 | +0.076 | 1.69 | −2.04R | +$4,335 |
| HELD-OUT (Q3, thru 09-18) | 64 | 5.33 | **−0.0077** | −0.19 | −1.26R | **−$185** |

HELD-OUT matches `WEEKLY_P1_vs_PROD_2026Q3.md`'s already-known Q3 read (n=64, −0.008R) almost exactly —
confirms this script's reference book is the same population, not a new one.

## Selection rule (PREREG, fixed): candidate iff mean R≥+0.05 AND day-clustered t≥1.5 on BOTH TRAIN and VAL
| Variant | TRAIN n | TRAIN mean R (t) | VAL n | VAL mean R (t) | Candidate? |
|---|---|---|---|---|---|
| a_F1 (rel-vol≥3×) | 18 | +0.009 (0.31) | 12 | +0.198 (0.86) | No — both legs fail |
| a_F3 (above-VWAP+top-half) | 130 | +0.009 (−1.13) | 118 | +0.093 (2.07) | No — TRAIN fails (barely a filter: 330/394 of P1 matches F3) |
| a_F4 (within 5% of 52wk-hi) | 5 | −0.134 (−1.19) | 1 | −0.146 (n/a) | No — n too small, both negative |
| a_F5 (prev-day range≥1.5×ATR14) | 29 | −0.012 (−0.05) | 30 | +0.096 (0.92) | No — both legs fail |
| a_F6 (day-2 of ≥10% gapper) | 17 | −0.067 (−0.83) | 15 | +0.011 (−0.13) | No — both legs fail |
| b_scale50_1R (50%@1R + live) | 178 | +0.091 (0.15) | 152 | **+0.374 (2.68)** | No — TRAIN dc_t fails despite a strong VAL |
| b_noexit2R_half3R_trail1R | 178 | +0.066 (−0.06) | 152 | +0.162 (0.45) | No — both legs fail |
| c_cost_sizing (range<0.75%→cap cost@0.10R) | 178 | +0.041 (0.18) | 152 | +0.076 (1.69) | No — **0/394 fills qualify** (identical to plain P1: this population's ranges are never that tight; the mechanism doesn't apply, not "tested and failed") |
| b_live_rule | — | — | — | — | Excluded by design (sanity check, not a candidate) |

**No variant clears both legs → verdict is "no improvement" exactly as the PREREG's fallback specifies.**
HELD-OUT was computed for every variant (in `1690_reads.csv`, for audit) but per the PREREG gate is not
reported as a finding for any of them — only the reference above is a HELD-OUT result.

## F7 (pre-market-high break): VOID
bars_sip.db coverage of P1's population = 32.5% (13/40 sample), below the 80% rail — stated in PREREG,
confirmed by the 100% cache.db coverage achieved for the SAME population once the right store was used.

## Disclosed limitation (read as an adversary, not hidden)
The live_rule sanity check — re-walking P1's own fills on cache.db bars with the exact live-lock rule,
which should ≈ reproduce the base book's R (1,679's analogous check was ≈0) — came back **median
|diff|=0.75R, mean 0.93R**, far from "small." P1's book was built from a separate wide-seed minute
extraction (not cache.db); the two sources evidently disagree enough on intraday OHLC (and/or range_low)
that variant (b)/(c)'s reconstructed R is NOT the same measuring stick as the base book's R. This does not
change the bottom-line verdict here — the one near-miss (b_scale50_1R) already fails on TRAIN by a wide
margin (dc_t 0.15, nowhere near 1.5), not on a close call — but it means variant (b)'s numbers above are
directional/exploratory only, not production-grade, and a real proposal to change ORB's exit would need an
independent rebuild on a bar source verified to match P1's own book before going further.

## Files
`PREREG_1690.md`, `RESULT_1690.md` (this file), `1690_reads.csv`, `1690_variants.py`, `1690_variants.log`,
`1690_verdict.txt`.
