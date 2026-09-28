# RESULT -- cell 1,624: break breadth (the crowd)

`PREREG_1623.md` section "Cell 1,624". Base = the 9,911 fills of cell 1,438 (status=='fill', VAL union TRAIN-H2, TEST sealed/never read). B30/B60 = count of ARM events (any status, all names, cell 1,438's own armed_crossing_bars replayed over the full causal superset -- see module docstring for why a replay was required) in the 30/60 minutes strictly before the fill's own fill_min, same day. Terciles of B30 frozen on TRAIN-H2 WITHIN each hour-of-day bucket, applied unchanged to both holdouts. Gate = top tercile of B30 for the fill's own hour.

**Arm-event replay verification**: 100.00% exact agreement with causal_arming_causal.csv's own n_cross column (0 disagreements) across the full population. arm_m cross-check (features_1478_A.csv vs this replay's own m_hi on the winning candidate of every fill): 98.21% exact match.

## Main table -- top tercile (kept) vs dropped, per holdout

| holdout | n kept | mean net R | t | ex-top5 | fills/wk (12/4) | n dropped | dropped mean | delta (kept-dropped) | kept cache-only % | rows excluded (no TRAIN-H2 cut for their hour) |
|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN-H2 | 1460 | -0.1644 | -2.44 | -0.2767 | 16.42 | 2938 | -0.1680 | 0.0035 | 21.4 | 0 |
| VAL | 2940 | -0.1726 | -3.08 | -0.2853 | 31.50 | 2573 | -0.1682 | -0.0043 | 18.2 | 0 |

## Beside -- bottom tercile of B30 and top tercile of B60 (report-only)

| holdout | bottom-B30 n | bottom-B30 mean | bottom-B30 t | B60-top n | B60-top mean | B60-top t |
|---|---|---|---|---|---|---|
| TRAIN-H2 | 1524 | -0.2189 | -4.87 | 1454 | -0.1652 | -2.58 |
| VAL | 896 | -0.1784 | -3.77 | 3079 | -0.1814 | -3.34 |

## Shuffle placebo (B30 permuted WITHIN each hour bucket, seed 1624)

| holdout | true kept mean | placebo kept n | placebo kept mean | margin (true - placebo) | placebo t |
|---|---|---|---|---|---|
| TRAIN-H2 | -0.1644 | 1460 | -0.1754 | 0.0110 | -3.86 |
| VAL | -0.1726 | 2940 | -0.1890 | 0.0164 | -4.33 |

## Pass bar (frozen, PREREG_1623.md, scored on VAL)

| criterion | pass? | value |
|---|---|---|
| kept mean net R >= +0.15 (VAL) | FAIL | -0.1726 |
| day-clustered t >= 2.5 (VAL) | FAIL | -3.08 |
| ex-top-5% > 0 (VAL) | FAIL | -0.2853 |
| >= 3 fills/wk at 12/4 (VAL) | PASS | 31.50 |
| dropped < kept, VAL | FAIL | -0.1682 < -0.1726 |
| dropped < kept, TRAIN-H2 | PASS | -0.1680 < -0.1644 |
| TRAIN-H2 same sign, t >= 1 | PASS | mean=-0.1644 t=-2.44 |
| placebo margin >= +0.10 R, t >= 2 (VAL) | FAIL | margin=0.0164 t=-4.33 |
| kept cache-only share within 5pp of 19.5% (VAL) | PASS | 18.2 |

**4/9 criteria met. Overall: FAIL.**

## Caveats (read as an adversary)

- B30/B60 count EVERY prior arm event on EVERY name including f's own symbol and f's own triggering event (it lands strictly before fill_min by construction whenever the SIP print is not at second-0 of its minute) -- this adds a near-universal +1 floor to B30/B60 shared by every fill equally; it is not excluded because the PREREG states no exclusion and it is a genuine, causal, chronologically-prior event.

- The arm-event replay uses TODAY's live config params (ca.live_params()) applied retroactively across the whole 2025-07..2026-05 window, the SAME convention cell 1,438 itself already uses for the shipped population (not a new look-ahead this cell introduces).

- Symbol-days with no usable minute bars contribute 0 arm events, identically to how cell 1,438 itself represents them (no row at all) -- see the WARNING count in the run log.

- Hour buckets with a thin TRAIN-H2 sample (<9 fills) give an unstable tercile cut; check the run log for which hours triggered that warning before trusting a rare early/late hour.

- TEST is sealed and not touched by this script (absent from every input file).

