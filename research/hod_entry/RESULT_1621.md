# RESULT -- cells 1,621-1,622: short-window confirmation entry (Frame C, PREREG_1617.md)

Same mechanism as cell 1,487 (buy after the break holds without a $0.01 dip, stop=level-0.01, target=entry+2R''), window W swapped for 15: 1,621 W=3 min, 1,622 W=5 min.

| cell | W | holdout | book | n | eligible_share | r_pct_median | runners_lost_share | mean_net_R | t | ex_top5 | fills_wk | null_pctile | paired_base_mean | paired_delta | paired_delta_t | paired_delta_ex_top5 | calibration_base_mean_on_cohort | passes_bar |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1621 | 3 | TRAIN-H2 | primary | 752 | 0.2615 | 1.0301 | 0.0041 | -0.1604 | -2.5288 | -0.2754 | 25.4074 | 37.7000 | 0.4569 | -0.6172 | -11.7403 | -0.7644 | 0.2385 | False |
| 1621 | 3 | TRAIN-H2 | all_eligible | 1011 | 0.2615 | 0.8195 | 0.0041 | -0.3978 | -4.6388 | -0.5251 | 31.9259 | 0.0000 | 0.3244 | -0.7222 | -9.5805 | -0.9126 | 0.2385 | False |
| 1621 | 3 | VAL | primary | 902 | 0.2525 | 1.1042 | 0.0044 | -0.2536 | -4.2835 | -0.3719 | 35.0455 | 1.4000 | 0.4281 | -0.6817 | -12.4902 | -0.8165 | 0.2397 | False |
| 1621 | 3 | VAL | all_eligible | 1197 | 0.2525 | 0.8933 | 0.0044 | -2.1274 | -1.8369 | -2.3452 | 43.0000 | 0.0000 | 0.3206 | -2.4480 | -2.1033 | -2.7152 | 0.2397 | False |
| 1622 | 5 | TRAIN-H2 | primary | 638 | 0.2008 | 1.1977 | 0.0084 | -0.1909 | -2.7806 | -0.3066 | 21.3333 | 10.0000 | 0.6115 | -0.8025 | -13.7343 | -0.9192 | 0.4214 | False |
| 1622 | 5 | TRAIN-H2 | all_eligible | 774 | 0.2008 | 1.0124 | 0.0084 | -0.3586 | -3.8389 | -0.4837 | 25.7037 | 0.0000 | 0.5255 | -0.8841 | -11.0003 | -1.0536 | 0.4214 | False |
| 1622 | 5 | VAL | primary | 767 | 0.1928 | 1.2383 | 0.0087 | -0.2079 | -2.7041 | -0.3230 | 30.1364 | 6.7000 | 0.5317 | -0.7396 | -11.9665 | -0.8615 | 0.3465 | False |
| 1622 | 5 | VAL | all_eligible | 928 | 0.1928 | 1.0421 | 0.0087 | -0.3407 | -4.7374 | -0.4628 | 34.7727 | 0.0000 | 0.4285 | -0.7692 | -12.3905 | -0.9385 | 0.3465 | False |

## Caveats
* half_entry source: same recovered-not-read status as cell 1,487 -- it is NOT a column of causal_arming_causal.csv; c1487.load_base() attaches it via cell_1445.corrected_cost (called inside cell_1478.build_outcome -> cell_1457.build_base_cost). Flag for the independent rebuild.
* data loss cell 1621 (W=3): no_bars=0, no_entry_bar=292 (halt/gap at minute fill_min+4), nonpositive_R2=42.
* data loss cell 1622 (W=5): no_bars=0, no_entry_bar=212 (halt/gap at minute fill_min+6), nonpositive_R2=32.
* Excluded rows are counted, never imputed -- see cell_1621_fills.csv `why` for every non-entered row's reason (no_bars / dip_in_window / base_exited_by_W / no_entry_bar / nonpositive_R2).
* EOD_BPS (11.5/9.7 bps) and SLIP_STOP_BPS are holdout-level EXPECTED VALUES (cells 1,443/1,478), not this leg's own measured tape -- same disclosed-proxy status cell 1,487 carries.
* fills_wk is slot-capped (research/hod_consol.simulate_slots, 4 concurrent/12 daily default caps), not a raw count/week ratio.
* count-matched null uses seed=1621 for BOTH cells (this PREREG's explicit pin names only one seed for Frame C) -- a reading choice, disclosed here for the rebuild.
* paired_delta/_t/_ex_top5 (net_R2 - base_outcome_R, day-clustered) are ADDED beyond the PREREG's literal report list, to catch tail-driven paired lift per the paired-lift-tail-check convention -- not part of the frozen pass bar.
* R''-as-%-of-price rail: r_pct_median reported per book; PRIMARY excludes R'' < 0.5% of price (RFLOOR_PCT), same floor cell 1,487 applies, never moved.
* runtime 130s.
