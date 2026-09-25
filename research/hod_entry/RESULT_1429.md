# RESULT — cell 1,429: fill-quality sizing

`research/hod_entry/cell_1429.py`, frozen per `PREREG_WEEKEND.md`. Base = E1 fills of cell 1,427
(`sip_rebuild_val.csv`, TRAIN-H2 n=1,165 / VAL n=1,443). Mult 1.5x when the pre-trigger ask was
≤5 bps above the level, else 1x (all fills already ≤15 bps, sip_rebuild's own entry-limit gate —
verified, zero WARNINGs). No missing tapes.

| holdout | charge | n | weighted R/risk | flat R | ΔR | VAL t (diff) | worst wk (weighted/flat) |
|---|---|---|---|---|---|---|---|
| TRAIN-H2 | no stop-slip | 1165 | +0.299 | +0.285 | +0.013 | 4.13 | -1.11 / -1.11 |
| TRAIN-H2 | +30bps slip | 1165 | +0.220 | +0.207 | +0.013 | 3.01 | -1.19 / -1.19 |
| VAL | no stop-slip | 1443 | +0.255 | +0.238 | +0.017 | 4.42 | -0.22 / -0.24 |
| VAL | +30bps slip | 1443 | +0.177 | +0.161 | +0.017 | 3.19 | -0.32 / -0.34 |

**Verdict: FAIL.** ΔR is positive and day-clustered t is strong (3–4.4), but ΔR (+0.013/+0.017)
falls well short of the PREREG's +0.05 bar on both holdouts — real but too small to size on.
Worst week is not worse (weighted ≥ flat both splits). Report-only trigger-print size class: VAL
odd-lot triggers (n=917) mean net_R +0.332 vs round-lot triggers (n=526) +0.074 — a large gap,
unexplored by this cell.
