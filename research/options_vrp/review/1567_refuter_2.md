# PREREG_1567 — refuter 2 (statistics and risk lens)

Recomputed from cell_1567_cycles.csv (script `review/refuter2_stats.py`, outputs `refuter2_cellstats.csv`,
`refuter2_exposure.csv`). B = $6,500.

## Verdict: the build's FAIL stands (and fails on more criteria than reported). Not refuted.

1. **Monthly convention defect (inflates gated cells).** `monthly_series` keeps only months with an exit; skipped
   months drop out instead of counting as $0. Cell 1574 TRAIN has 10 exit months of 19 calendar months. With
   zero-filled calendar months: TRAIN Sharpe 7.60 -> 3.01, green 100 % -> 53 % (below the 55 % eligibility bar).
   Correct selection = **1573** (d .15, W10, hold, no gate; TRAIN Sharpe 1.59, green 84 %). 1573 VAL: 1.96 %/mo
   zero-filled (2.26 % builder), Sharpe 2.37, worst -$427 -> still FAIL on the 4 % bar. The verdict does not flip.
2. **1574 VAL under the correct convention**: mean 0.70 %/mo ($46), Sharpe 0.77, green 47 % -> fails 3 criteria,
   not 1. SPY buy-and-hold per unit of drawdown, VAL window: +24.3 % / 9.1 % DD = 2.66; 1574 = $686 / $565 = 1.21
   -> **fails the BH-per-drawdown criterion too** (RESULT never scored it).
3. **Over-selection.** TRAIN-vs-VAL Sharpe rank correlation across 24 cells = -0.07 (builder) / -0.02 (zero-filled):
   TRAIN selection carries no information. 1574 ranks 14th/24 on VAL. VAL-best = 1589 (d .30, W10, hold, no gate)
   5.2-5.6 %/mo, which ranked 16th on TRAIN (TRAIN: -$2,432 in 2025-03, -$1,500 in 2025-04). Only 2-3/24 cells clear
   4 % on VAL, none selectable from TRAIN.
4. **Neighbours** of 1574 on VAL (1582 d .20; 1570 W5) same-signed positive: pass (weak, +$1,393 / +$422).
5. **Month concentration.** 1574 VAL: 92 % of net P&L from the top 2 months (Apr/May 26); one month (-$565, 2026-03,
   one cycle) wipes 82 % of the total. Top-5 % cycle share 16 %. TRAIN $0 DD = the IV gate skipped the Jan/Feb 2025
   low-IV entries that cost the ungated twin 1573 -$848 in 2025-03 - one event, not a property.
6. **Spike months.** 2024-08: 1574 entered 8/5 at 455 (+$98). 2025-04: +$158; the d .15 W10 cells are VOID on
   3/24, 3/31, 4/07 (no entry print) - missing exactly the crash weeks; under hold-to-expiry those strikes would have
   expired OTM, so no sign change, but crash-week VOIDs are a tail-missingness hazard for management A.
7. **Budget.** Worst month >= -B in every cell/split (min -$2,566). Concurrent worst-case exposure max $6,498 (1569,
   7 open) <= B in every cell, but only by floor rounding: sizing is B/6 per ladder while 38-52 DTE allows 7-8
   concurrent spreads -> the construction can exceed B (8 x $1,083). Live engine must assert the open sum, not B/6.
8. **Naked comparison.** 1574 VAL spread $686 vs naked (same 1 contract) $6,893: the cap gives up ~90 %; naked is
   not on the same risk basis, report-only.
9. **Sharpe with few months.** 1574 VAL Sharpe 1.05 on 8 months: 95 % CI [-1.4, 3.5]; zero-filled 0.77 on 15: [-1.0,
   2.5]. TRAIN 7.6 on 10 months = artefact of item 1 (per-cycle sd $10.6, no losing cycle).
10. **Honest $ at B = $6,500.** Selected 1574: $46/mo (range -$565..+$361). Correct selection 1573: $127/mo; if the
   ~44 % data VOIDs were filled at the same per-cycle mean (MCAR, 59 vs 33 VAL cycles): ~$227/mo = 3.5 % -> still
   under $260. Post-hoc best 1589: ~$337/mo with -$2.4K / -$1.5K spike months in TRAIN.

## Unresolved defect outside this lens (blocks reporting either way)
Build vs rebuild cycle-set Jaccard = 0.31-0.44 per cell (PREREG bar >= 0.99); P&L within $5 on 14-82 % of matched
cycles. The rebuild's "pass-looking" 1586 VAL (9.98 %/mo) contains a 9-contract cycle with $3.90 credit on a $5-wide
d .30 spread (+$3,509, a bad print) and a negative-credit entry (2026-02-23, -$48 credit) - defective. The FAIL is the
build's number; the independent check has not agreed with it yet.
