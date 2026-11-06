# RESULT 1,723–1,725: ORB loser anatomy, stop cost and early-exit triggers (2026-11-06)

**Verdict: no rule passes.** The best VAL read is the cluster rule T5 (cell 1,725): +0.20 R, t_day 2.23, ex-top-5 % +0.155,
Δ$ +$2.9K. It misses the t ≥ 2.5 bar, its TRAIN ex-top-5 % is negative, and 31 of its 39 VAL fills are L2 paper.
R4 (stop-limit band) is positive on both splits and never clips a winner, but it is worth only +0.02 R.

Population: 154 fills (L1 123, L2 31), 1 R-floor fail. TRAIN = L1 to 08-15 (114 R-reads). VAL = L1 08-16..09-23 (8) ∪ L2 (31) = 39.
Bars: bars_sip.db for 123 fills, plus 31 L2 symbol-days fetched from Alpaca SIP into the scratchpad (LOST 0).
Code: `anatomy_1723.py`. Files: `per_fill_1723.csv`, `reads_1723.csv`.

## FREEZE (written from TRAIN only, before the VAL rule reads were printed)
- 1,724: R3 was the only rule with ΔR ≥ +0.05 on TRAIN (+0.065, t 3.36, 0 winners clipped), so it was the candidate.
- 1,725: cluster C5a "pop-and-fade" carries 52 % of TRAIN stop $ (bar ≥ 40 %). Its trigger is **T5**: after any bar
  high ≥ +0.25 R, the first 1-min close below the fill → exit at the next open − 35 bps.
- T1–T4 were mapped to C1–C4 before any read. They are reported here but none is a candidate (each cluster < 40 %).

## Stop cost (64 stopped fills)
| split | n | slip bps mean / med / p90 / max | stop $ | slip $ (share) | median min to stop | stop then back above fill within 30 min |
|---|---|---|---|---|---|---|
| TRAIN | 43 | 8 / 0 / 30 / 245 | −13,383 | −630 (4.7 %) | 35 | 7 |
| VAL | 21 | 19 / 22 / 70 / 94 | −2,739 | −175 (6.4 %) | 23 | 3 |

About 95 % of the stop dollars are the 1 R itself. Slip is only about 5 %, so even a perfect stop fill saves ≤ 0.02 R per fill.
The stop rate is high because of where the stop sits, not because of how it fills.

## Loser clusters (rule order: C2 → C1 → C3 → C4 → C5a → C5b)
| cluster (rule) | TRAIN n / $ / R / stop$ share | VAL n / $ / R / stop$ share |
|---|---|---|
| C1 immediate (MAE ≤ −0.5 R in 5 min, MFE < 0.25) | 8 / −2,025 / −0.94 / 14 % | 5 / −513 / −1.07 / 19 % |
| C2 give-back (MFE ≥ +1 R, then lost) | 10 / −2,287 / −0.82 / 13 % | 3 / −534 / −1.08 / 20 % |
| C3 slow bleed (MFE < 0.25, ≥ 20 closes below the fill) | 4 / −1,312 / −0.72 / 8 % | 0 |
| C4 late (exit after 12:00 ET) | 8 / −1,596 / −0.64 / 9 % | 4 / −202 / −0.54 / 0 % |
| **C5a pop-and-fade (0.25 ≤ MFE < 1, then stopped, median 27–55 min)** | **23 / −7,007 / −0.93 / 52 %** | **12 / −1,336 / −0.91 / 48 %** |
| C5b tag scratch (tag_bb exit within 5 min) | 16 / −969 / −0.18 / 0 % | 0 |
| C5c other | 2 / −502 / −1.02 / 4 % | 4 / −394 / −0.85 / 14 % |

Winners by causal shape (TRAIN / VAL): hit +1 R 25 / 9 (+$8.6K / +$2.4K), stagnant at 20 min 7 / 2, immediate dip 4 / 0, other 8 / 0.

## Rules: paired ΔR vs the actual exit (R-floored; Δ$ at $375)
| rule | TRAIN ΔR (t_day) | VAL ΔR / t_day / t_iid / ex-top-5 % / Δ$ / MDE | L2-only ΔR (t) | winners clipped VAL (n, $) | saved / forgone R (VAL) |
|---|---|---|---|---|---|
| R1 stagnation | +0.002 (0.08) | −0.010 / −0.99 / −1.00 / −0.011 / −148 / 0.02 | −0.013 (−0.98) | 1, −148 | 0.0 / −0.4 |
| R2 bar-1 fail | −0.050 (−0.50) | +0.132 / 1.76 / 1.85 / +0.078 / +1,925 / 0.15 | +0.161 (2.87) | 1, −609 | 6.8 / −1.7 |
| R3 VWAP loss | **+0.065 (3.36)** | **−0.089 / −0.89** / −0.82 / −0.118 / −1,306 / 0.20 | −0.112 (−0.90) | 1, −1,545 | 0.9 / −4.4 |
| R4 stop-limit −20 bps | +0.007 (2.02) | +0.021 / 2.74 / 3.05 / +0.015 / +310 / 0.015 | +0.021 (2.45) | 0 | 0.8 / 0 |
| R5 close-confirm stop | −0.062 (−2.06) | +0.054 / 0.53 / 0.35 / −0.129 / +783 / 0.20 | +0.037 (0.31) | 0 | 7.8 / −5.7 |
| **T5 pop-fade (1,725)** | +0.036 (0.53), ex-top-5 % −0.017 | **+0.201 / 2.23 / 2.25 / +0.155 / +2,943 / 0.18** | +0.201 (1.81) | 3, −1,092 | 10.8 / −2.9 |
| T1 / T2 / T3 / T4 (diagnostic) | −0.01 / +0.03 / 0.00 / +0.01 | +0.02 / −0.06 / −0.01 / −0.04 (all |t| < 1.4) | — | — | — |

TRAIN numbers for T5: 50 fills affected, 22.0 R saved vs 17.9 R forgone, 14 winners clipped (−$6.7K).

## Why each rule fails the bar
- **R3**: the TRAIN pick reverses on VAL (−0.089 R), with 1 VAL winner clipped for −$1.5K. Its TRAIN cohort of 16 fills
  had zero winners, which was luck.
- **T5**: close on the edge but fails t (2.23 < 2.5). On TRAIN, the R it saved almost equals the continuation it gave up.
  Its VAL is mostly the 31 L2 paper fills. Most of what this test can say: the pop-and-fade cluster is the one place a trigger reaches.
- **R4**: a real but tiny cost lever (+0.02 R, both splits, 0 clips). It cannot reach the size bar because slip is only about 5 % of stop $.
- **R2**: the sign flips between TRAIN and VAL. **R1, T3**: affect 1 VAL fill. **R5**: deeper stops cost more than the stops they avoid on TRAIN.
- MDE on VAL ≈ 0.15–0.20 R for the selective rules: n = 39 cannot detect anything smaller.

## Anomalies and caveats
- **CRDU 2026-10-27 (L2)**: the ledger pnl is +$949, but price × shares gives +$389 (R_pnl 6.6 vs R 2.7).
  Shares/pnl are inconsistent in trades.db. The reads here use the price-based R.
- The PREREG's R4 text has the inequalities inverted. A sell limit fails only when price trades THROUGH it.
  - R4 = the mechanical model: an in-band fill keeps the actual market fill; a fill through the band rests at L, fills
    at L on the first later bar with high ≥ L, and falls back to the 15:45 open. None of the 19 through-band stops was carried to the force close.
  - "R4lit" applies stop − 13 bps to every in-band stop, which is worse than the actual median slip of 0 bps (TRAIN −0.022 R).
- L1 VAL has only 8 fills, so VAL is effectively the L2 paper ledger.
- MFE/MAE are computed from 1-min bars from the first full bar after the fill through the exit bar, so the exit minute's prints after the exit are included.
- Cells: 1,723–1,725 (3 cells; 11 rule reads × 4 sets inside them). Report-only. No orb.yaml change.
