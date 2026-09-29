# RESULT — cells 1,655-1,657: the first five seconds after 09:35:00 ET

Report-only + one pre-committed decision (1,656 ships to paper only if the frozen pass bar passes). PREREG_1655.md frozen 2026-09-29 08:20 UTC. t* = AT-or-through (price >= trigger, research/orb_latency_bt/replay.py::find_t_star) unless labeled THROUGH-only. delay-0/delay-5 status and pnl are 1,426's own results.csv rows, not recomputed.

## Coverage
- fills with ticks / total: 617/638 (96.7%)
- uncovered fills' BT outcome (population.csv, not replayed):

| group | n | win_pct | mean_bt_pnl |
|---|---|---|---|
| all | 21 | 0.429 | -50.665 |
| out_of_sample | 17 | 0.353 | -509.787 |
| 2025H1 | 4 | 0.750 | 1900.604 |

## 1,655 — bucket table at delay 0 (t* AT-or-through)

### out_of_sample

| bucket | n | mean_R | day_t | win_pct | ex_top5pct_R | total_usd |
|---|---|---|---|---|---|---|
| 0-5 | 44 | -0.057 | -1.778 | 0.136 | -0.102 | -941.048 |
| 5-15 | 63 | 0.009 | 0.130 | 0.190 | -0.104 | 209.806 |
| 15-30 | 42 | 0.220 | 2.098 | 0.357 | 0.111 | 3468.978 |
| 30-60 | 53 | 0.016 | 0.207 | 0.189 | -0.085 | 325.533 |
| 60-300 | 152 | 0.044 | 0.830 | 0.217 | -0.073 | 2514.745 |
| no_t* | 183 | 0.000 | nan | 0.000 | 0.000 | 0.000 |

### 2025H1

| bucket | n | mean_R | day_t | win_pct | ex_top5pct_R | total_usd |
|---|---|---|---|---|---|---|
| 0-5 | 9 | -0.014 | -0.091 | 0.222 | -0.141 | -46.625 |
| 5-15 | 8 | 0.455 | 1.952 | 0.500 | 0.292 | 1365.673 |
| 15-30 | 10 | -0.028 | -0.316 | 0.500 | -0.067 | -105.003 |
| 30-60 | 6 | 0.325 | 0.901 | 0.333 | -0.052 | 731.351 |
| 60-300 | 23 | 0.200 | 1.511 | 0.304 | 0.040 | 1729.115 |
| no_t* | 24 | 0.000 | nan | 0.000 | 0.000 | 0.000 |

### May-Sep_2026

| bucket | n | mean_R | day_t | win_pct | ex_top5pct_R | total_usd |
|---|---|---|---|---|---|---|
| 0-5 | 16 | -0.015 | -0.233 | 0.125 | -0.066 | -91.455 |
| 5-15 | 18 | -0.043 | -1.153 | 0.167 | -0.064 | -291.913 |
| 15-30 | 14 | 0.273 | 1.412 | 0.429 | 0.201 | 1435.630 |
| 30-60 | 15 | -0.032 | -0.284 | 0.200 | -0.122 | -178.232 |
| 60-300 | 43 | -0.009 | -0.153 | 0.140 | -0.081 | -144.761 |
| no_t* | 47 | 0.000 | nan | 0.000 | 0.000 | 0.000 |

## 1,655 — THROUGH-only t* variant beside it (0-5 / 5-15 buckets, same pnl)

| period | n_0_5_AT | meanR_0_5_AT | n_0_5_THROUGH | meanR_0_5_THROUGH | n_5_15_AT | n_5_15_THROUGH |
|---|---|---|---|---|---|---|
| out_of_sample | 44 | -0.057 | 33 | -0.047 | 63 | 51 |
| 2025H1 | 9 | -0.014 | 7 | 0.104 | 8 | 7 |
| May-Sep_2026 | 16 | -0.015 | 15 | -0.011 | 18 | 15 |

## Reconciliation with 1,426's published delay-0 totals (pass bar: same n, total $ within $1)

| period | n_mine | n_1426 | total_mine | total_1426 | delta_usd | match |
|---|---|---|---|---|---|---|
| out_of_sample | 537 | 537 | 5578.015 | 5578.015 | -0.000 | True |
| 2025H1 | 80 | 80 | 3674.511 | 3674.511 | -0.000 | True |
| May-Sep_2026 | 153 | 153 | 729.269 | 729.269 | 0.000 | True |

**Reconciliation: PASS -- all three periods match.**

## 1,656 — DELAY-5 (buy-stops armed at 09:35:05.000)

| period | n | n_filled | n_skipped_guard | n_skipped_guard_d0 | total_usd | mean_R | baseline_mean_R | day_t | mean_delta_R | ex_top5pct_delta_R |
|---|---|---|---|---|---|---|---|---|---|---|
| out_of_sample | 537 | 235 | 119 | 95 | 5914.022 | 0.029 | 0.028 | 1.378 | 0.002 | -0.004 |
| 2025H1 | 80 | 35 | 21 | 14 | 3459.205 | 0.115 | 0.122 | 1.955 | -0.007 | -0.018 |
| May-Sep_2026 | 153 | 63 | 43 | 32 | 759.725 | 0.013 | 0.013 | 0.395 | 0.001 | -0.008 |

## 1,657 — SKIP-INSTANT (t* < 5s excluded, no refill; report-only)

| period | n_total | n_excluded | n_remaining | mean_R_remaining | day_t_remaining | total_usd_remaining | total_usd_baseline | usd_given_up |
|---|---|---|---|---|---|---|---|---|
| out_of_sample | 537 | 44 | 493 | 0.035 | 1.525 | 6519.063 | 5578.015 | -941.048 |
| 2025H1 | 80 | 9 | 71 | 0.140 | 2.137 | 3721.136 | 3674.511 | -46.625 |
| May-Sep_2026 | 153 | 16 | 137 | 0.016 | 0.428 | 820.725 | 729.269 | -91.455 |

## Verdict — frozen pass bar for `preplace_submit_delay_s: 5`

- Condition A (0-5s bucket negative, out-of-sample mean_R=-0.0570 t=-1.78 n=44; 2025H1 mean_R=-0.0138 n=9): **PASS**
- Condition B (1,656 mean_R=0.0294 vs baseline-0.005=0.0227, ex-top5% ΔR=-0.0042 vs -0.01 floor): **PASS**

**VERDICT: `preplace_submit_delay_s: 5` SHIPS to the paper session** (both conditions of the frozen pass bar pass on the out-of-sample book).

## Judge note (main session, 2026-09-29 09:05 UTC)
Both frozen conditions met on the 1,426 population (617/638 fills with ticks; the delay-0 totals reconcile to the cent
with the published 1,426 tables): the 0–5 s bucket is negative out of sample (n 44, −0.057 R, day-clustered t −1.78)
with the same sign on 2025H1 (n 9 — fragile; flips under the through-only t* on n 7), and arming the buy-stops at
09:35:05 costs nothing (+0.002 R/fill vs delay 0, ex-top-5 % of ΔR −0.004 R). Reading: the evidence that instant breaks
lose is weak-to-moderate; the evidence that a 5-s delay is free is solid. A free hedge against a plausible loser ships:
`entry.preplace_submit_delay_s: 5` goes to the paper session today (engine parameter with tests, default 0). The
paper-session bar for ORB live moves to "first order ≤ 1.5 s after 09:35:05". The live histogram of trigger times keeps
being recorded; the bucket is re-read after 40 paper/live fills.
