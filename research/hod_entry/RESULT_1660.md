# RESULT — cells 1,660–1,661 (PREREG_1660.md, frozen 2026-09-29 17:25 UTC)

Scripts: `1660_rescore.py`, `1661_rescore.py`. Full grids: `1660_full_grid.csv` (360 rows: 30 cells x
2 splits x 3 floors x 2 cost lines), `1661_limit_rescore.csv`, `1661_filled_detail.csv`. TEST split
never touched. Reuses `hod_exit_lab/score_cells.py`'s `day_clustered_t`/`ex_top5` unchanged.

## 1,660 — exit lab re-read at measured cost

Cost: entry 7bps of entry price + exit leg (stop 6bps / target 0bps / eod 11bps-bid or 1bp-MOC) of
exit price, both /R$ (=entry-stop). 30 scoreable cells (X1-X12,X5b,X7b · D1-D5 · W1-W2 · T1-T2 ·
M1-M2 · E1 · O1; O2 stays VOID, no data). "35" in the PREREG rounds the lab's actual count.

### Re-read base (B0), both halves, both cost lines, with/without the stop-% floors
| split | line | floor | n | mean net R | t |
|---|---|---|---|---|---|
| TRAIN | bid | none | 7390 | -0.014 | -4.4 |
| VAL | bid | none | 4745 | -0.061 | -3.3 |
| TRAIN | moc | none | 7390 | -0.005 | -4.1 |
| VAL | moc | none | 4745 | -0.053 | -3.1 |
| TRAIN | bid | >=1.5% | 4914 | +0.007 | -2.8 |
| VAL | bid | >=1.5% | 3013 | -0.028 | -2.3 |
| TRAIN | moc | >=1.5% | 4914 | +0.017 | -2.6 |
| VAL | moc | >=1.5% | 3013 | -0.019 | -2.1 |
| TRAIN | bid | >=3% | 1326 | -0.042 | -1.9 |
| VAL | bid | >=3% | 578 | -0.025 | -1.8 |

Old cost (tape-replay) read -0.221/-0.304 R. Measured cost pulls it to -0.014/-0.061 (bid) or
-0.005/-0.053 (MOC) — most of the original loss WAS the cost model, as the owner expected — but VAL
stays negative in every line x floor combination; only TRAIN+MOC+1.5% floor crosses zero (+0.017),
VAL does not (-0.019 same row). Day-clustered t stays <= -1.8 everywhere: the negative is not noise.

### 30 cells, paired dR vs the re-read B0 (bid line, floor=none), ranked by min(TRAIN,VAL) dR
| cell | train dR | t | val dR | t | ex5 dR tr | ex5 dR val | val fills/wk |
|---|---|---|---|---|---|---|---|
| M2 | +0.093 | -1.4 | +0.184 | +1.7 | -0.068 | +0.022 | 33 |
| M1 | +0.020 | -0.8 | +0.010 | -1.3 | -0.144 | -0.156 | 84 |
| D1 | +0.016 | -4.1 | -0.005 | -2.6 | -0.148 | -0.171 | 148 |
| T1 | +0.023 | -3.0 | -0.010 | -2.2 | -0.140 | -0.178 | 136 |
| X3 | -0.001 | -0.9 | -0.009 | -1.3 | -0.217 | -0.226 | 211 |
| X11 | -0.011 | -1.8 | +0.011 | -1.2 | -0.177 | -0.155 | 210 |
| X9 | -0.011 | -1.1 | -0.011 | -1.7 | -0.175 | -0.176 | 212 |
| O1 | -0.028 | -1.2 | +0.023 | -0.9 | -0.199 | -0.149 | 213 |
| E1 | +0.030 | -1.2 | -0.028 | -1.0 | -0.265 | -0.289 | 167 |
| W1 | +0.026 | -0.1 | -0.050 | -0.8 | -0.139 | -0.228 | 150 |
| (18 more, all worse; full detail in 1660_full_grid.csv) | | | | | | | |

Every remaining cell (X1,X2,X4-X8,X10,X12,X5b,X7b,D2-D5,W2,S1,S2,T2) has min(TRAIN,VAL) dR < 0.
**No cell clears the reading rule** (dR>=+0.05 R AND t>=2 on BOTH halves AND ex-top5 dR>0 both AND
>=3 fills/wk): the best point estimate, M2 ("prior 5-session signals net positive"), misses on t
(TRAIN t -1.4) and on ex-top5 (TRAIN ex5 dR -0.068, tail-negative) despite a promising VAL mean;
D1/T1 have the flip sign (some of the larger |t| are t<-2 evidence FOR B0, not against). **Verdict:
no candidate goes to paper.**

Assumptions (declared per the "no accidental behaviour" rule, not hidden in the code): (1) 5 exit
labels outside {stop,target,eod} — `eod_partial`,`stop_runner` (X8), `vwap_exit` (X9), `time_stop`
(X7/X7b), `clock_1200` (X10) — billed at the EOD-bucket rate (closest in character: a crossing exit,
not a resting order); none of these 5 cells are near the reading bar, so the choice is not
outcome-relevant here. (2) O1's 530 held (overnight) rows are billed at the EOD bucket too (next-day
open is a marketable cross, not a resting fill) — PREREG's cost table has no "overnight" leg.

## 1,661 — entry limit width, dry cross records since 9/26

**`logs/hod_dry_entry_ledger.csv` stopped being written after 2026-09-25** (174 rows, all dated
9/25; rows from line 176 also gain a stray 14th field — a logging regression, not a data choice).
Zero 9/26+ arms are in that CSV. Rebuilt from the journal instead: `journalctl` ARMED+CROSS lines
(3,051 ARMED / 129 CROSS, 0 orphaned when paired same-symbol/same-day) give level/trigger/live-limit
/stop and the print/ask/FILL outcome; reproduces the live 0.15% rule's own FILL/NO-FILL 128/129
(1 rounding edge). **This logging gap is a live defect worth fixing** (same class as the 9/24
completeness-gate and 9/19 crontab misses in memory) — flagged, not fixed here (read-only task).

| limit | n arms | n filled | fill rate | mean slip vs trigger | n w/ outcome | net R (filled) |
|---|---|---|---|---|---|---|
| +0.05% | 129 | 23 | 17.8% | +0.002% | 14 | -0.205 |
| +0.10% | 129 | 46 | 35.7% | +0.022% | 31 | -0.350 |
| +0.15% (today) | 129 | 65 | 50.4% | +0.042% | 47 | -0.397 |
| +0.25% | 129 | 81 | 62.8% | +0.062% | 54 | -0.496 |

Net R uses each filled arm's own outcome from `logs/hod_dry_counterfactuals.csv` (exit_px,
actual_stop), joined on (date,symbol). **Coverage gap**: cf-watch logging only starts 9/28 (9/26 has
zero outcome rows); 9/29 is 14/32 resolved (day still open) — n_with_outcome is thin (14-54) and
concentrated in ~1.5 sessions.

Reading rule (narrower beats +0.15% only if netR >= +0.03 R better AND fill count >= 70% of today's):
+0.05% improves net R by +0.192 R but fills only 23/65 = 35% of today's count — fails the count
floor. **+0.10% technically clears both** (net R +0.047 R better; 46 fills = 70.8% of 65, a
one-fill margin) **but rests on n=31 filled-with-outcome trades across effectively 1.5 sessions** —
too thin to promote given the coverage gap above; flagged as "watch," not shipped to paper. No wider
limit ever qualifies by the rule's own text (worse fill AND worse net R here too).

## Files
`research/hod_entry/1660_rescore.py`, `1660_full_grid.csv` (aggregates), `1660_per_fill.csv`
(265,851 per-fill rows: cell, day, symbol, entry_m, split, entry, stop, r_pct, exit_price, why,
net_R_bid, net_R_moc — floor is a filter on r_pct, applied by the reader), `1661_rescore.py`,
`1661_limit_rescore.csv`, `1661_filled_detail.csv`, this file.
