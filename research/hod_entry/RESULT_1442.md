# RESULT — cell 1,442: tape-triggered override

`research/hod_entry/cell_1442.py`, frozen per `PREREG_WEEKEND.md`. Population = E1 fill + nofill
signals of cell 1,427 (`sip_rebuild_val.csv`, TRAIN-H2 D=3,481 / VAL D=4,717); R-scored population
restricted to E1 fills only (stop is only recoverable there); 36/2,608 E1 fills dropped for a
tape/CSV fill-price mismatch (WARNING, tape/CSV drift).

| holdout | (a) BROKER n / mean R (slip) | (b) OVERRIDE n / mean R (slip) | fill rate base/a/b | (b)−(a) paired n | ΔR | VAL t |
|---|---|---|---|---|---|---|
| TRAIN-H2 | 994 / +0.236 | 813 / +0.278 | 0.335 / 0.292 / 0.240 | 723 | -0.016 | -5.77 |
| VAL | 1233 / +0.183 | 985 / +0.226 | 0.306 / 0.270 / 0.215 | 879 | -0.016 | -5.13 |

**Verdict: FAIL, decisively.** (b)−(a) is NEGATIVE on both holdouts (-0.016/-0.016 vs the +0.03
bar), t < -5 both splits, and OVERRIDE's own fill rate (0.24/0.22) is well below both BROKER's
(0.29/0.27) and E1's own baseline (0.34/0.31) — waiting 300ms lets the ask run away more often than
it improves the print. Do not ship. Size-class cross (report-only): round-lot-led VAL fills score
far worse under both rules (mean_R_slip_a -0.017, _b -0.022) than odd-lot-led (+0.327/+0.359) —
consistent with cell 1,429's odd-lot finding, opposite direction from what the override hoped to
buy.
