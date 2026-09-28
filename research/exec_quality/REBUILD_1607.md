# REBUILD_1607 — independent rebuild of cell 1,607 (HOD-SPREAD)

Built from `research/exec_quality/PREREG_1607.md` (FROZEN 2026-09-28 16:20 UTC) ONLY. `cell_1607.py`
and `RESULT_1607.md` were never opened. Script: `research/exec_quality/rebuild_1607.py`.

Owner's question (9/28): *"maybe the ones with the bigger spread are the winners? if they are the
winners, maybe the spread is a filter?"*

## 0. Inputs and joins
9,911 base fills (`causal_arming_causal.csv`, `status=='fill'`) joined on `(day, symbol, fill_min)` to
`model_1478_L3_predictions.csv` (`outcome_R`, standard-cost net R) and `features_1478_A.csv`
(`half_entry`, `spread_frac_at_fill`) — all three joins are exact, 9,911/9,911, zero split mismatches.
TRAIN-H2 = 4,398 rows (`causal_arming_causal.csv`'s TRAIN is entirely `half=='H2'`, verified), VAL =
5,513. Sanity check: population mean `outcome_R` = **−0.167R (TRAIN-H2, t −4.2) / −0.171R (VAL, t
−4.4)**, in line with the book's documented net (~−0.21R live, −0.22/−0.30R exit lab) — no wiring error.

## 1. Causality finding (the refuter's first lens, per the PREREG's own instruction)
The PREREG names two set-A candidates and asks to pick "the causal one": `spread_frac_at_fill × 1e4`
and `2 × half_entry / fill`.

* **These are the same number.** `max|diff| = 1e-12 bps` across all 9,911 rows — `half_entry` is
  defined as `spread_frac_at_fill × fill / 2` by construction inside `cell_1445.corrected_cost()`.
  There is no "state which one" choice to make; the PREREG's phrasing assumed two independent
  candidates and there is only one.
* **Neither is an arm-time quote.** `corrected_cost()` recovers `half_entry` as a *residual of the
  realized round-trip cost*: `half_entry = cost_R·R − exit_half − fee`, and `cost_R = raw_R − net_R`
  of the SAME fill — both only knowable once the trade has been **exited** (`exit_price`, `exit_m`,
  `why`). This is not "the quote before the fill print" — it is the fill's own post-hoc realized cost,
  exactly the failure mode the PREREG names ("never the fill's own quote"). **Set-A fails the PREREG's
  own causality bar** and cannot be operationalized as a live pre-trade gate no matter what its VAL
  numbers show, because it is not known at (or even shortly after) the arming decision.
* **A genuinely causal arm-time quote exists**, in `features_1478_C.csv`: `spread_bps_at_arm` — the
  prevailing NBBO quote at the first print of the breakout bar, strictly before the trigger print that
  produced the fill (`FEATURES_C.md` item 4; per-row timestamp proof, 0 lookahead violations found in
  9,737/9,911 rows with a resolved cache key, 98.2% coverage, confirmed against `FEATURES_C.md`'s own
  disclosed coverage). This was not named in PREREG_1607's cell text (it names only `features_1478_A.csv`)
  but is exactly the "beside" comparison the PREREG's own inputs section anticipates for this ambiguity,
  and it correlates 0.83 with the set-A proxy — related, not identical.

**This rebuild reports set-A exactly as the PREREG specifies it (for reproducibility), and treats set-C
as the authoritative feature for the pass-bar verdict**, since a filter that cannot be known before the
decision cannot pass "causality of the spread field" regardless of its point estimate.

## 2. Quintile edges (TRAIN-H2 only, pre-declared before touching VAL)
| Quintile | Set-A (bps, non-causal) | Set-C (bps, causal arm-time) |
|---|---|---|
| Q1 | ≤ 8.58 | ≤ 10.91 |
| Q2 | 8.58–17.85 | 10.91–21.10 |
| Q3 | 17.85–28.28 | 21.10–34.33 |
| Q4 | 28.28–48.00 | 34.33–58.44 |
| Q5 | > 48.00 | > 58.44 |

(Set-C drops 174 rows with no resolved tape key before quintiling, per its disclosed coverage.)

## 3. Per-quintile, per-holdout (n, mean net R, day-clustered t, ex-top-5%, fills/wk at 12/day 4-concurrent)

**Set-A** (fill-derived, non-causal):
| holdout | Q | n | mean R | t | ex-top5 | fills/wk |
|---|---|---|---|---|---|---|
| TRAIN-H2 | Q1 (tightest) | 880 | −0.001 | −0.02 | −0.106 | 26.0 |
| TRAIN-H2 | Q2 | 879 | −0.145 | −2.81 | −0.257 | 25.6 |
| TRAIN-H2 | Q3 | 880 | −0.209 | −3.78 | −0.323 | 26.3 |
| TRAIN-H2 | Q4 | 879 | −0.118 | −1.89 | −0.227 | 25.6 |
| TRAIN-H2 | Q5 (widest) | 880 | **−0.360** | **−6.75** | −0.478 | 26.0 |
| VAL | Q1 (tightest) | 1000 | −0.037 | −0.70 | −0.143 | 31.9 |
| VAL | Q2 | 1154 | −0.102 | −1.90 | −0.211 | 35.2 |
| VAL | Q3 | 994 | −0.099 | −1.57 | −0.208 | 33.7 |
| VAL | Q4 | 1084 | −0.184 | −4.08 | −0.295 | 34.2 |
| VAL | Q5 (widest) | 1281 | **−0.381** | **−7.72** | −0.500 | 38.1 |

**Set-C** (tape-derived, causal):
| holdout | Q | n | mean R | t | ex-top5 | fills/wk |
|---|---|---|---|---|---|---|
| TRAIN-H2 | Q1 (tightest) | 864 | −0.008 | −0.14 | −0.113 | 25.7 |
| TRAIN-H2 | Q2 | 865 | −0.155 | −2.78 | −0.267 | 25.1 |
| TRAIN-H2 | Q3 | 863 | −0.141 | −2.17 | −0.251 | 25.0 |
| TRAIN-H2 | Q4 | 864 | −0.164 | −2.75 | −0.275 | 26.2 |
| TRAIN-H2 | Q5 (widest) | 864 | **−0.335** | **−6.69** | −0.453 | 25.6 |
| VAL | Q1 (tightest) | 923 | −0.008 | −0.13 | −0.112 | 30.6 |
| VAL | Q2 | 996 | −0.108 | −1.78 | −0.218 | 33.0 |
| VAL | Q3 | 1088 | −0.159 | −2.52 | −0.270 | 34.8 |
| VAL | Q4 | 1176 | −0.146 | −2.98 | −0.257 | 35.9 |
| VAL | Q5 (widest) | 1234 | **−0.367** | **−7.16** | −0.487 | 36.8 |

Both features tell the same story: mean net R is roughly flat-to-zero in Q1, and monotonically worse
into Q5 (widest quintile is 2–4× worse than the book average, t around −7 on VAL in both features).

## 4. Pre-declared filters, read on VAL (pass bar: kept VAL mean ≥ +0.15R, t ≥ 2.5, ex-top5 > 0,
≥ 3 fills/wk, TRAIN-H2 same-sign t ≥ 1, dropped < kept on both holdouts)

| Filter | Feature | VAL kept mean | VAL t | VAL ex-top5 | VAL fills/wk | VAL dropped mean | TRAIN-H2 kept mean (t) | PASS? |
|---|---|---|---|---|---|---|---|---|
| KEEP-WIDE (top 2 Q) | Set-A | −0.290R | −7.20 | −0.407 | 45.4 | −0.081 | −0.239 (−4.97) | **FAIL** |
| KEEP-TIGHT (bottom 2 Q) | Set-A | −0.072R | −1.59 | −0.180 | 43.8 | −0.234 | −0.073 (−1.53) | **FAIL** |
| KEEP-WIDE (top 2 Q) | Set-C (causal) | −0.259R | −6.39 | −0.374 | 45.7 | −0.096 | −0.249 (−5.39) | **FAIL** |
| KEEP-TIGHT (bottom 2 Q) | Set-C (causal) | −0.060R | −1.25 | −0.167 | 42.0 | −0.228 | −0.082 (−1.69) | **FAIL** |

Both filters fail on both features. **KEEP-WIDE fails in the wrong direction, not just short of the
bar**: VAL mean is deeply negative (−0.26 to −0.29R, t ≈ −6 to −7) and its *dropped* subset
(−0.08 to −0.10R) is far better than its *kept* subset — the opposite of "dropped < kept" required
for a pass. KEEP-TIGHT is directionally the right idea (kept beats dropped, correctly signed) but its
kept mean is still net-negative and neither the +0.15R magnitude bar nor the t ≥ 2.5 significance bar
is met.

## 5. Bottom line
**The owner's hypothesis is refuted, and refuted in the opposite direction.** The wide-quoted-spread
fills are not the winners — on this population (9,911 HOD-break fills, standard-cost net R), they are
the *worst* trades: the top spread quintile loses roughly 2–3× the book's already-negative average
(−0.33 to −0.38R vs a −0.17R population mean), with day-clustered t around −6.7 to −7.7 on both the
literal PREREG feature and the causally-correct arm-time quote. The tightest-spread quintile is close
to flat (−0.001 to −0.04R) but not positive, and neither pre-declared KEEP-WIDE nor KEEP-TIGHT filter
clears the frozen pass bar. **Per the PREREG's own consequence clause: cell 1,607 is a FAIL — the
spread is closed as a filter on this book** (no `min_spread_bps`/`max_spread_bps` gate for the dry run).
Separately and worth carrying forward: the set-A feature the PREREG named is not causal and should not
be reused as a "spread" feature in any future cell without the set-C substitution made here.

## Caveats
* This rebuild covers **cell 1,607 only** (HOD), per this task's explicit scope — 1,608 (ORB) and 1,609
  (drift-as-signal) were not touched.
* Quintile edges used `pd.qcut` (5 equal-count bins) with the outer edges widened to ±∞ so VAL rows
  outside the TRAIN-H2 range still land in an extreme quintile; this is a reasonable but not the only
  possible edge convention — if the original cell used a different tie-breaking or edge convention,
  edges could differ by more than the ±1 bps reproducibility bar near the Q2/Q3/Q4 boundaries where
  bin counts aren't perfectly even (e.g., TRAIN-H2 n=880/879/880/879/880 — a handful of tied values sit
  on a boundary).
* "Standard cost" `outcome_R` was taken as given from `model_1478_L3_predictions.csv` without
  re-deriving the standard-cost model itself (out of this cell's scope; that model was independently
  built and reviewed under cell 1,478's own PREREG).
* `net_R_corr_flat30` was aliased to `outcome_R` purely to satisfy `cell_1445.score_one()`'s column
  read (that field is not reported anywhere in this rebuild).
