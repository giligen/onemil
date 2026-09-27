# Refuter 2 — PREREG_1564 result, lens: STATISTICS AND CONTROLS (2026-09-27)

Recomputed from `cell_1564_events.csv` only (script: scratchpad `ref2.py`). Signal rows = entered & event==FAILURE (1564/1565, 1565 also excluding report_only) or SUCCESS (1566).

**Verdict: the FAIL on all three cells STANDS (refuted=false).** No defect found here changes a verdict. Several builder statistics are wrong or mislabelled, and each correction makes the result more negative, not less.

## 1. Headline, tails, concentration (per split, VAL second)
| cell | n | ev/wk (calendar) | builder ev/wk | mean R | t(day) | gross raw R | ex-top-5 % | ex-top-1 % | cap +3R | drop best 2 days | months > 0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1564 TRAIN | 389 | 7.48 | 12.63 | −0.165 | −2.49 | −0.058 | −0.277 | −0.187 | −0.165 | −0.223 | 2/12 |
| 1564 VAL | 246 | 6.44 | 12.18 | −0.325 | −4.30 | −0.163 | −0.438 | −0.352 | −0.325 | −0.373 | 1/9 |
| 1565 TRAIN | 43 | 0.83 | 5.66 | −0.286 | −1.12 | −0.006 | −0.617 | −0.472 | −0.391 | −0.566 | 3/11 |
| 1565 VAL | 23 | 0.60 | 6.05 | −0.648 | −2.98 | −0.233 | −0.841 | −0.746 | −0.648 | −0.862 | 3/8 |
| 1566 TRAIN | 1312 | 25.2 | 27.9 | −0.086 | −1.53 | −0.028 | −0.193 | −0.108 | −0.086 | −0.115 | 3/12 |
| 1566 VAL | 1581 | 41.4 | 44.2 | −0.050 | −0.90 | +0.020 | −0.152 | −0.071 | −0.050 | −0.107 | 3/9 |

* **Defect A: events/week.** `stats()` divides n by (distinct EVENT days / 5), not by calendar weeks, so the rate is inflated 1.1–9x (1565: 6.05 → 0.60 per week). The verdict does not change (1565 already fails on mean R and now also fails the ≥ 3/week rail).
* Mean R is negative in both splits, both before and after cost. For 1564 the gross (zero-cost) R is negative (−0.06 / −0.16), so no cost, borrow or half-spread assumption can rescue it. The TRAIN real-NBBO subset gives −0.055 R (n 213). VAL has 0 real quotes, but gross is already negative.
* There is no tail story in either direction. Ex-top and capped values are more negative, dropping the best 2 days makes it worse, 1–3 months are positive per split, and no single day dominates.

## 2. The control — does the failure add anything? (defect B, the key one)
The builder reports 1564 VAL null percentile 99.1 (control −0.48 R), which reads as "failure beats non-failure". **That is an R-denominator artifact.** The control pool includes SUCCESS names shorted at 10:30, and their stop (the day high) sits right above the price: median R is 1.15–1.5 % of price against 4–4.5 % for FAILURE. Tiny R inflates their negative R-multiples (SUCCESS short −0.84 R VAL).
| 1564 null variant (1000 draws, seed 1564, same-day count-matched) | TRAIN pctile | VAL pctile |
|---|---|---|
| net R, all non-failure (builder's) | 91.3 | 99.1 |
| net R, pool = NO_BREAK + INDETERMINATE only | 84.2 | 85.8 |
| net R, matched within day × R%-of-price quintile | **4.2** | 95.2 |
| net return in % of price | **11.8** | **19.8** |
| gross return in % of price | **11.0** | **19.8** |

Measured in % of price, the failed-break short loses MORE than shorting the non-failure candidates of the same day: −0.73 % vs −0.52 % TRAIN and −0.91 % vs −0.77 % VAL, with gross −0.41 % vs −0.20 % and −0.55 % vs −0.40 %. **A failed opening-range break at 10:30 is followed by a relative BOUNCE, not a continuation lower.** The mechanism ("trapped buyers, later breaks fail") has the wrong sign on the 13,316-candidate population. The universe placebo is also negative in both splits (−0.17 / −0.44 R), because gap-day candidates drift up after 10:30 when shorted to 15:55.

## 3. 1566 held-break long = ORB entered late
* **Every one of the 2,893 SUCCESS events is an ORB-entered day** in `orb_features_20260925_2054.csv`. The 1566 entry sits a median **+2.18 % above** the ORB entry on the same name-day.
* On those days the ORB file books +3.2 % / +3.4 % (conditioned on survival to 10:30, so it is not a fair comparison). 1566 books −0.32 % / −0.21 % net. The 10:30 "confirmation" pays away the move that already happened.
* Its separation from the control is real in % of price: null pctile 99.3–100 gross and net, and gross +0.01 % / +0.14 %. But it is a gross drift of ~0.1–0.3 % of price that cost turns negative.
* The builder's null 100 in R is additionally inflated by tiny-R controls. FAILURE longs average −2.7 / −3.7 R at 1.3 % of price, because the stop at range_low sits just under the entry.

## 4. HOD calibration reproduction
| HOD base fills (model_1478) on … | TRAIN n / mean / t | VAL n / mean / t |
|---|---|---|
| ORB loser, all fills (the diagnostic) | 30 / **−0.903** / −7.1 | 38 / **−0.537** / −2.8 (reproduced exactly) |
| ORB loser, fill ≥ 10:30 | 17 / −0.73 / −3.4 | 23 / −0.36 / −1.2 |
| ORB loser, fill < 10:30 | 13 / −1.13 | 15 / −0.82 |
| FAILURE declared by 10:30, fill ≥ 10:30 (causal) | 4 / −0.95 | 10 / **+0.38** / 0.8 |
| ORB loser NOT failed by 10:30, fill ≥ 10:30 | 13 / −0.66 / −2.4 | 13 / −0.92 / −4.1 |
| SUCCESS by 10:30, fill ≥ 10:30 | 36 / +0.07 | 52 / −0.34 / −2.2 |

The diagnostic's −0.90 / −0.54 does NOT come from a failure knowable at 10:30. It comes from ORB losers whose stop is hit AFTER 10:30 (or after the HOD fill), which is same-day outcome co-movement: a name that fails its HOD break also falls through its ORB range low. It is not a declarable signal. The causal subset is n 4 / 10 with split signs. A later declaration hour cannot fix this without selecting among hours, which the PREREG does not allow.

## 5. Short capacity
* 896 of 2,014 FAILURE candidates (44 %) are excluded as not shortable, from a STATIC, non-point-in-time `borrow_flags.csv`. A missing name counts as not shortable, so delisted names are excluded.
* Realistic capacity is at most about 6.4 entries per week in VAL, before locate failures on gappers.
* Hard-to-borrow rates far above 3 %/yr only lower an already negative gross.

## 6. Minor
The builder's "universe placebo" includes the signal rows themselves. It is not a disjoint control. This does not matter to the verdict.

## Conclusion
All three cells fail the frozen pass bar under every tail cut, every control and every cost assumption. For 1564 the sign is wrong gross, and wrong relative to the same-day control in % of price. The builder's two "positive" readings are measurement artifacts: 1564 "beats the control" (the R denominator) and events/week (the denominator). This is a closure on this population (ORB candidates 2025-01..2026-09, entry at the 10:30 open, exit by 15:55), not a statement about other horizons. The FAILURE flag stays usable as a veto feature for longs; that is its only defensible use.
