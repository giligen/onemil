# RESULT — cells 1,607–1,609: the quoted spread as a filter

Executed research/exec_quality/PREREG_1607.md (FROZEN 2026-09-28 16:20 UTC) exactly. Owner question: "maybe the ones with the bigger spread are the winners? if they are the winners, maybe the spread is a filter?"

## CAUSALITY FINDING — read this before the numbers below
`spread_frac_at_fill` and `2*half_entry/fill` are **the same column** (max abs diff = 1.00e-12; `build_features_1478_A.py:370` defines `spread_frac_at_fill = 2.0 * half_entry / fill`). Neither is the causal arm-time quote the PREREG asked for: both come from `cell_1445.corrected_cost()`, which is solved from the REALIZED fill's `cost_R`/`exit_price`/`exit_half_src` — i.e. from the trade's own outcome, not a quote sitting at the close of arm bar j (FEATURES_A.md's own caution says the same). **`features_1478_A.csv` has no causal, pre-fill spread field.** Cell 1,607 below is therefore an EXECUTION-QUALITY lens on the realized fills ("is the expensive-to-execute trade the winning trade"), not a forward pre-trade signal — a PASS does not by itself license a live `min_spread_bps`/`max_spread_bps` arm-time gate; that would need a genuinely causal quote capture (e.g. the NBBO at the close of arm bar j, independent of whether/how the order later filled).

## 1,607 HOD-SPREAD
Population: `causal_arming_causal.csv` status==fill (9,911) joined to `model_1478_L3_predictions.csv` (outcome_R = standard-cost net R) and `features_1478_A.csv` (spread_frac_at_fill) on (day,symbol,fill_min); merged n = 9911. TRAIN-H2 n=4398, VAL n=5513. Spread field used: `spread_frac_at_fill x 1e4` (bps); identical to `2*half_entry/fill x 1e4`, reported as one column per the identity above.

Quintile edges set on TRAIN-H2 only (bps): [-0.0, 8.58, 17.85, 28.28, 48.0, 528.98]

| Sample | Quintile | spread range (bps) | n | mean net R | day-clust t | win rate | ex-top-5% | fills/wk |
|---|---|---|---|---|---|---|---|---|
| TRAIN-H2 | Q1 | -0–9 | 880 | -0.0014 | -0.02 | 39.2% | -0.1058 | 26.00 |
| TRAIN-H2 | Q2 | 9–18 | 879 | -0.1451 | -2.81 | 35.0% | -0.2574 | 25.56 |
| TRAIN-H2 | Q3 | 18–28 | 880 | -0.2089 | -3.78 | 34.4% | -0.3225 | 26.26 |
| TRAIN-H2 | Q4 | 28–48 | 879 | -0.1181 | -1.89 | 38.1% | -0.2266 | 25.63 |
| TRAIN-H2 | Q5 | 48–529 | 880 | -0.3604 | -6.75 | 34.3% | -0.4782 | 26.00 |
| VAL | Q1 | -0–9 | 1000 | -0.0373 | -0.70 | 38.7% | -0.1435 | 31.91 |
| VAL | Q2 | 9–18 | 1154 | -0.1016 | -1.90 | 37.4% | -0.2112 | 35.23 |
| VAL | Q3 | 18–28 | 994 | -0.0994 | -1.57 | 38.0% | -0.2082 | 33.68 |
| VAL | Q4 | 28–48 | 1084 | -0.1837 | -4.08 | 36.2% | -0.2950 | 34.18 |
| VAL | Q5 | 48–610 | 1281 | -0.3806 | -7.72 | 33.3% | -0.5004 | 38.14 |

### The two pre-declared filters
| Filter | Holdout | n kept | kept mean R | t (kept) | ex-top-5% | fills/wk | n dropped | dropped mean R |
|---|---|---|---|---|---|---|---|---|
| KEEP-WIDE | TRAIN-H2 | 1759 | -0.2393 | -4.97 | -0.3534 | 39.41 | 2639 | -0.1185 |
| KEEP-WIDE | VAL | 2365 | -0.2904 | -7.20 | -0.4069 | 45.41 | 3148 | -0.0805 |
| KEEP-TIGHT | TRAIN-H2 | 1759 | -0.0732 | -1.53 | -0.1817 | 38.15 | 2639 | -0.2292 |
| KEEP-TIGHT | VAL | 2154 | -0.0718 | -1.59 | -0.1799 | 43.82 | 3359 | -0.2339 |

**Pass bar (frozen)**: VAL kept mean >= +0.15 R, t >= 2.5, ex-top-5% > 0, fills/wk >= 3, TRAIN-H2 same sign with t >= 1, dropped < kept on both holdouts.
* **KEEP-WIDE: FAIL**
* **KEEP-TIGHT: FAIL**

## 1,608 ORB-SPREAD
Live sample: `data/trades.db` strategy=orb, closed fills, dates 2026-05-19..2026-09-23, n=123. R = pnl / risk, risk = (entry_price - stop_loss_price) * filled_qty. spread_bps = entry_quote_spread / entry_price * 1e4.
Backtest sample: `analysis_results/orb_features_20260925_2054.csv` (via `read_orb_csv`) carries no spread/quote/bid/ask column -> fallback to `research/orb_latency_bt/results.csv` delay_s==0 & status==filled (n=301 before NBBO resolution, n=301 after), dates 2023-02-08..2026-09-22, NBBO at t* from the XNAS tick tape (`research/orb_latency_bt/raw`, falling back to `research/hod_ofi/raw`). R = pnl_replay / 375 (`research/orb_latency_bt/replay.py`'s own convention, reused unchanged — the same book already vetted at cell 1,426).

| Sample | Bucket | n | mean R | day-clust t | win rate | P&L share |
|---|---|---|---|---|---|---|
| LIVE | <=50 | 81 | -0.0453 | -0.33 | 38.3% | 27.6% |
| LIVE | 50-100 | 31 | -0.2103 | -1.49 | 32.3% | 13.5% |
| LIVE | 100-150 | 9 | -0.1191 | -0.41 | 55.6% | 48.5% |
| LIVE | 150-300 | 2 | -0.7511 | -3.02 | 0.0% | 10.5% |
| LIVE | >300 | 0 | nan | n/a | nan% | n/a |
| LIVE | ALL | 123 | -0.1038 | -0.98 | 37.4% | 100.0% |
| BACKTEST | <=50 | 188 | 0.0910 | 1.68 | 30.3% | 69.3% |
| BACKTEST | 50-100 | 59 | 0.0040 | 0.04 | 28.8% | 1.0% |
| BACKTEST | 100-150 | 21 | 0.1737 | 1.10 | 47.6% | 14.8% |
| BACKTEST | 150-300 | 29 | 0.1202 | 0.93 | 34.5% | 14.1% |
| BACKTEST | >300 | 4 | 0.0477 | 0.18 | 50.0% | 0.8% |
| BACKTEST | ALL | 301 | 0.0820 | 1.98 | 31.9% | 100.0% |

### Bucket-rule test vs the existing 300 bps gate baseline
Pass bar (frozen): dropped bucket mean R < 0 on BOTH samples with n>=20 each, kept-book mean R rises by >= +0.03 R vs the full sample.
| Rule | Live n dropped | Live dropped mean R | Live kept delta | BT n dropped | BT dropped mean R | BT kept delta | Verdict |
|---|---|---|---|---|---|---|---|
| skip_le_50 | 81 | -0.0453 | -0.1127 | 188 | 0.0910 | -0.0150 | FAIL |
| skip_gt_300 | 0 | nan | 0.0000 | 4 | 0.0477 | 0.0005 | FAIL |

## 1,609 ORB-DRIFT-AS-SIGNAL (report-only)
drift_ask_to_fill_bps vs R on the live ORB fills, n=123. Spearman rank correlation = 0.093. Report-only: drift is realized (post-decision), not knowable at order-submit time.

| Drift bucket (bps) | n | mean R | mean drift (bps) | day-clust t | win rate |
|---|---|---|---|---|---|
| (-214.287, 10.241] | 25 | -0.3085 | -37.3 | -2.19 | 36.0% |
| (10.241, 31.499] | 24 | 0.1718 | 22.7 | 0.53 | 41.7% |
| (31.499, 64.096] | 25 | -0.3576 | 47.7 | -2.57 | 36.0% |
| (64.096, 139.409] | 24 | -0.0520 | 91.0 | -0.17 | 25.0% |
| (139.409, 744.681] | 25 | 0.0404 | 234.5 | 0.23 | 48.0% |

## Caveats
* **1,607 is not a causal pre-trade field** (see the causality finding above) — a PASS is evidence for "expensive fills co-vary with outcome on this population", not a ready `min/max_spread_bps` arm-time gate.
* **Cost double-count**: `outcome_R` is already the standard-cost NET R (half-spread charged once at exit per cell 1445/1457's corrected-cost convention) — the spread quintiles here are read on a cost-inclusive outcome, per the PREREG's own warning not to judge this on gross.
* **1,608 backtest NBBO join**: resolved per-fill from the raw XNAS mbp-1 tape at the replay t* used to fill the BT order (same causal rule as replay.py's ask_at — last two-sided quote strictly before t*); fills with no parquet file or no prior two-sided quote are excluded and counted in the log, not silently dropped.
* **1,608 live sample size**: n=123 total closed ORB fills split across 5 spread buckets — day-clustered t on a single bucket can be on a handful of trading days; read the bucket table beside the n column, not the t column alone.
* **1,609** drift is a realized, not causal, quantity — explicitly report-only per the PREREG; no cap or gate should be inferred from it without a forward (order-submit-time) proxy.
* Independent rebuild still required before this ships anywhere (per CLAUDE.md's "no research claim ships" protocol) — this is the BUILDER pass only.

Generated by `research/exec_quality/cell_1607.py`.

## Judge (main session, 2026-09-28 16:05 UTC) — FAIL both HOD filters; the spread is a cost, not a signal; ORB inconclusive

* HOD (9,911 fills): the spread quintiles run monotonically from Q1 ≈ 0 R to Q5 −0.38 R on BOTH halves (net); the
  refuter recomputed on the causal arm-time field (features C spread_bps_at_arm — the builder's spread_frac_at_fill is
  back-solved from the realized cost and is not causal) with the same FAIL, and showed GROSS R is FLAT across quintiles:
  the wide-spread fills are not the winners before cost, and lose more after cost because the spread IS the cost
  (≈ 2/3 of the net gradient). KEEP-WIDE VAL −0.29 R (t −7.2); KEEP-TIGHT VAL −0.07 R (t −1.6) — the least bad cut, still
  negative, still no book. The owner's hypothesis ("the wide-spread ones are the winners") is refuted on this
  population; the R-must-exceed-the-spread rule is the right reading of it.
* ORB (1,608): live 123 fills all buckets negative (the net-negative window) with no pattern; backtest 301 replay fills:
  ≤ 50 bps +0.09 R (n 188, 69 % of the P&L), 100–150 +0.17 (n 21), 150–300 +0.12 (n 29), > 300 n 4 — no bucket rule
  passes the frozen test; the 300 bps gate is untestable at n 4 and stays. (1,609) entry drift vs outcome: Spearman
  0.09, no monotone pattern — the fastest, most expensive fills are not the winners, and a latency/drift cap is not
  costing edge either. Programme count 1,609.
Consequence: no spread gate for HOD; the ORB gate unchanged; latency stays the lever (the drift is cost, not signal).
