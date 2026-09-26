# RESULT -- cells 1,487-1,488: confirmation entry + no-withdrawal pyramid

## Cell 1,487 -- confirmation entry (R'' = entry - stop)
| cell | holdout | book | n | eligible_share | r_pct_median | runners_lost_share | mean_net_R | t | ex_top5 | fills_wk | null_pctile | paired_base_mean | calibration_base_mean_on_cohort | kept_cacheonly_share | passes_bar |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1487 | TRAIN-H2 | primary | 376 | 0.1021 | 1.4354 | 0.0480 | -0.1287 | -1.5380 | -0.2420 | 12.5185 | 27.5000 | 0.7705 | 0.6896 | 0.3112 | False |
| 1487 | TRAIN-H2 | all_eligible | 402 | 0.1021 | 1.3685 | 0.0480 | -0.1892 | -2.2109 | -0.3038 | 13.4444 | 7.0000 | 0.7234 | 0.6896 | 0.3060 | False |
| 1487 | VAL | primary | 421 | 0.0985 | 1.5797 | 0.0455 | -0.0450 | -0.5214 | -0.1524 | 16.7727 | 89.3000 | 0.8194 | 0.6451 | 0.2637 | False |
| 1487 | VAL | all_eligible | 477 | 0.0985 | 1.4573 | 0.0455 | -0.1416 | -1.7025 | -0.2550 | 18.8182 | 33.8000 | 0.7381 | 0.6451 | 0.2558 | False |

## Cell 1,488 -- pyramid keyed on no-withdrawal (original-R units, paired vs base)
| cell | holdout | n | share_added | mean_net_R | t | paired_delta | paired_delta_t | paired_base_mean | passes_bar |
|---|---|---|---|---|---|---|---|---|---|
| 1488 | TRAIN-H2 | 376 | 0.0855 | 0.2246 | 3.5269 | -0.5460 | -10.2365 | 0.7705 | False |
| 1488 | VAL | 421 | 0.0764 | 0.2521 | 4.7136 | -0.5673 | -12.1144 | 0.8194 | False |

## Caveats
* half_entry source: PREREG names cell_1445_features.csv, whose write_features_csv() (cell_1445.py) drops half_entry -- it is recovered here via the IDENTICAL formula (cell_1445.corrected_cost via cell_1478.build_outcome), not read from that CSV. Flag for the independent rebuild.
* data loss: no_bars=0, no_entry_bar=106 (halt/gap at minute fill_min+16), nonpositive_R2=7 -- excluded from eligible/entered counts, not silently dropped (see cell_1487_fills.csv `reason`).
* 1,488's 'same fills' = the 1,487 PRIMARY (post R''-floor) cohort, not the pre-floor eligible set -- a reading choice, since the PREREG does not separately floor-gate 1,488.
* EOD_BPS (11.5/9.7 bps) and SLIP_STOP_BPS are holdout-level EXPECTED VALUES (1,443/1,478), not this new leg's own measured tape -- same disclosed-proxy status as half_entry.
* fills_wk is slot-capped (research/hod_consol.simulate_slots, 4 concurrent/12 daily default caps), not a raw count/week ratio.
* runtime 144s.

## Judge (main session, 2026-09-26 17:15 UTC) — FAIL, null adequate

Both cells fail the frozen VAL bar; the independent rebuild from the prose agrees on all 9,911 base fills (eligibility
reasons identical, primary book Jaccard 1.0, 100 % of rows within 0.01 R, VAL means −0.045 / +0.252 on both sides).
* 1,487 (buy at the ask after 15 minutes without a dip, stop level − $0.01, 2 R″): eligible ≈ 10 % of base fills; VAL
  n 421, −0.045 R (t −0.5), all-eligible −0.14 R; ex-top-5 % negative in every cut. The calibration line reproduces:
  the eligible cohort's BASE outcome is +0.65 / +0.69 R (the disclosed +0.67), i.e. the cohort is real, and the late
  entry gives its edge back: the run-up already gone by minute 16 makes R″ ≈ 1.5 % of price with the target far away.
* Critic's decomposition: on real SIP bars the book is −0.20 / −0.22 R (t −2.0 / −2.3, n 259 / 310); the only positive
  sits in the cache-only cohort (+0.43 R, n 111, 26–31 % of the kept set vs the 19.5 % baseline) — the sparse-cache
  artifact of cell 1,427. The null is therefore not power-starved.
* 1,488 (1/3 at the break, 2/3 at minute 16, stop to level − $0.01): own mean +0.25 R (t 4.7) but −0.57 R paired against
  simply holding the base position on the same fills (t −12 on both holdouts). Adding size at the run-up price and
  tightening the stop destroys value.
Consequence per PREREG: the long side of this population is closed at the break (−0.21 R), at the confirmation (−0.05 R),
and at the retest (−0.09 R on the rebuild, cell 1,481, builder agreement pending). Programme count 1,488.
