# WEEKEND_RESULTS — cells 1,429 / 1,430 / 1,431 / 1,439 / 1,440 / 1,441 / 1,442

Frozen per `research/hod_entry/PREREG_WEEKEND.md` (2026-09-25 18:35 UTC). One table appended per
cell as it completes; full detail in each cell's own RESULT_*.md.

## 1,430 — exits on the winning fills

| variant | n (TRAIN/VAL) | ΔR TRAIN-H2 | ΔR VAL | VAL t | ex-top-5 % (VAL) | worst week (var / B0, R) | verdict |
|---|---|---|---|---|---|---|---|
| (a) 90-min time stop | 1165/1443 | -0.119 (-0.051) | -0.132 (-0.065) | -6.56 | -0.220 | -12.10 / -17.17 | FAIL |
| (b) breakeven lock (+1R→BE) | 1163/1442 | -0.153 (-0.054) | -0.170 (-0.071) | -8.64 | -0.229 | -26.14 / -17.17 | FAIL |
| (c) ORB lock (+1.5R→+0.5R) | 1164/1440 | -0.108 (-0.015) | -0.135 (-0.039) | -8.14 | -0.206 | -17.68 / -17.17 | FAIL |
| (d) 50% scale-out at +2R | 1157/1433 | -0.008 (+0.083) | -0.040 (+0.050) | -1.49 | -0.168 | -20.35 / -20.49 | FAIL |
| (e) VWAP-close after +0.5R | 1163/1442 | -0.091 (-0.021) | -0.096 (-0.029) | -5.92 | -0.152 | -20.93 / -17.17 | FAIL |
| (f) close at 14:30 | 1165/1443 | -0.076 (-0.002) | -0.076 (-0.001) | -6.98 | -0.130 | -24.51 / -17.17 | FAIL |

Parenthetical ΔR = without the new 30 bps stop-slip charge. All six FAIL; (d) closest (see
`RESULT_1430.md`). Full report: `research/hod_entry/RESULT_1430.md`.

## 1,429 — fill-quality sizing

| holdout | charge | n | weighted R/risk | flat R | ΔR | VAL t (diff) | worst wk (weighted/flat) |
|---|---|---|---|---|---|---|---|
| TRAIN-H2 | no stop-slip | 1165 | +0.299 | +0.285 | +0.013 | 4.13 | -1.11 / -1.11 |
| TRAIN-H2 | +30bps slip | 1165 | +0.220 | +0.207 | +0.013 | 3.01 | -1.19 / -1.19 |
| VAL | no stop-slip | 1443 | +0.255 | +0.238 | +0.017 | 4.42 | -0.22 / -0.24 |
| VAL | +30bps slip | 1443 | +0.177 | +0.161 | +0.017 | 3.19 | -0.32 / -0.34 |

FAIL: ΔR positive with strong day-clustered t (3.0–4.4) but under the +0.05 pass bar on both
holdouts. Report-only: VAL odd-lot trigger prints (n=917) mean net_R +0.332 vs round-lot (n=526)
+0.074. Full report: `research/hod_entry/RESULT_1429.md`.

## 1,431 — no-fill cohort short

| holdout | D (post break-bar-close filter) | scored n | mean net R (no-slip) | mean net R (30bps slip) | VAL t | share shortable | fills/wk |
|---|---|---|---|---|---|---|---|
| TRAIN-H2 | 237 | 64 | -1.375 | -1.866 | -9.42 | 0.375 | 2.67 |
| VAL | 324 | 93 | -1.058 | -1.438 | -7.90 | 0.376 | 4.43 |

FAIL, decisively: mean net R deeply negative both holdouts (bar +0.10), VAL t strongly negative
(bar +2), shortable share 0.375/0.376 misses the 60% bar too. R had to be floored at 0.5% of price
(unfloored PREREG text let R collapse to 1-2 ticks for signals whose next-bar open passed the
break-bar high, exploding R-multiples to ±10^13 — 501 rows dropped, WARNING-logged). No shortable
column existed in the PREREG's named asset file; used `borrow_flags.csv` instead (documented in
`RESULT_1431.md`). Full report: `research/hod_entry/RESULT_1431.md`.

## 1,442 — tape-triggered override

| holdout | (a) BROKER n / mean R (slip) | (b) OVERRIDE n / mean R (slip) | fill rate base/a/b | (b)−(a) paired n | ΔR | VAL t |
|---|---|---|---|---|---|---|
| TRAIN-H2 | 994 / +0.236 | 813 / +0.278 | 0.335 / 0.292 / 0.240 | 723 | -0.016 | -5.77 |
| VAL | 1233 / +0.183 | 985 / +0.226 | 0.306 / 0.270 / 0.215 | 879 | -0.016 | -5.13 |

FAIL, decisively: (b)-(a) is negative both holdouts (bar was +0.03), t < -5, and OVERRIDE's own
fill rate (0.24/0.22) is below both BROKER's and E1's baseline — 300ms of waiting loses more fills
than it improves prices. Full report: `research/hod_entry/RESULT_1442.md`.

## 1,439 — low-of-day mirror short — FIRST FULL RUN VOID (population look-ahead: `low <= floor − 1c` kept only days that fell below $20 AFTER the entry; +0.61/+0.75 R is not a finding); filter fixed, rerun 20:22 UTC, scored below when done

<details><summary>void run (record only)</summary>

## 1,439 — low-of-day mirror short (corrected population)

| holdout / book | n | fill rate | mean net R | t | ex-top-5 % | fills/wk | SSR share | shortable |
|---|---|---|---|---|---|---|---|---|
| TRAIN-H2 primary (non-SSR, shortable) | 2,398 | 0.43 | −0.150 | −2.25 | −0.262 | 36 | 0 % | 100 % |
| VAL primary | 3,344 | 0.48 | −0.289 | −4.80 | −0.407 | 43 | 0 % | 100 % |
| VAL all fills | 5,877 | 0.84 | −0.279 | −5.82 | −0.396 | 50 | 25 % | 67 % |

FAIL decisively (bar ≥ +0.10 R): the short mirror loses on both holdouts, worse ex-top-5 %; coverage 100 %, gap 0.
The FIRST full run (+0.61/+0.75 R) was VOID: its population filter kept only days that fell below the $20 floor AFTER
the entry (look-ahead) — fixed, regression-tested, rerun. Full: `RESULT_1439.md`.


## 1,428 — gapper universe

# RESULT — cell 1,428: gapper-universe causal arming (LADDER.md row 1,428)

Population: 7487 symbol-days requested, 0 LOST (0.0 %), 3 unsimulable (< K+2 minute bars).

| holdout | fills (rate) | mean net R | stop-slip R | day-clust t | ex-top-5 % | fills/wk | coverage / gap | median spread bps | R % of price | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN-H2 | 343 (66.1 %) | -0.067 | -0.166 | -1.11 | -0.180 | 11.4 | 100.0 % / 0.0 pp | 15 | 2.02 % | report-only (TRAIN-H2) |
| VAL | 444 (73.8 %) | +0.109 | +0.017 | 0.49 | +0.008 | 16.8 | 100.0 % / 0.0 pp | 17 | 1.88 % | FAIL: mean >= +0.15, t >= 2 |
* VAL verdict: FAIL: mean >= +0.15, t >= 2.
* Cost: measured half-spread at fill + B0 exit leg (causal_arming); nbbo.csv is the $20+ book's spread table and rarely covers this pool, so exit cost falls back to the fill-instant half-spread for most fills.
* Spread bps is derived from cost_R * R (approx, assumes entry and exit half-spreads are close) — not an independent NBBO measurement for this pool; read as indicative.
* Small caps: R must exceed the spread — see the R %-of-price and spread-bps columns above per CLAUDE.md.
* Population: point-in-time Databento EQUS.SUMMARY daily bars, gap >= 5 %, open $3-30, prior-day volume >= 500K, 2025-07-01..2026-05-31, test tickers excluded; TEST not read.

## 1,441 — prior-day-high level

| split | fills (rate) | mean net R | day-clust t | ex-top-5 % | fills/wk | coverage / gap | verdict |
|---|---|---|---|---|---|---|---|
| TRAIN-H2 | 145 (61.2 %) | +0.016 | -0.44 | -0.098 | 5.3 | 100.0 % / 0.0 pp | report-only |
| VAL | 239 (70.9 %) | -0.168 | -3.37 | -0.281 | 10.1 | 100.0 % / 0.0 pp | FAIL: mean ≥ +0.10, t ≥ 2, ex-top-5 % > 0 |

* Same order/arming as cell 1,438 (causal_arming.py) with the level SUBSTITUTED: PDH = prior session's RTH high from `bars_sip.db` (not the running HOD); one entry/day at the first cross since a break permanently disarms the running-high guard.
* 30 bps stop-slip: mean net R TRAIN-H2 -0.072, VAL -0.270 (worse in both — not a tail effect).
* Population: base superset (universe.csv, date range, adv20 >= min_adv20, test tickers excluded) = 350,694 symbol-days; 259,238 (74 %) had no prior-session `bars_sip.db` data -> `no_pdh`, excluded (never backfilled from cache.db/daily_bars per spec); of the 91,456 with a known PDH, 2,096 symbol-days clear day-high >= PDH+1c, PDH >= open x 1.05, PDH >= floor+1c.
* Coverage 100 %, gap 0 pp on both holdouts — the population, not tape availability, is what's small; ex-top-5 % more negative than the raw mean means the loss is NOT winner-capped, it is broad.
* Verdict: FAIL, both holdouts negative-signed and VAL strongly so (t -3.37) — the PDH level (fixed at the prior day's high) does not reproduce cell 1,438's running-HOD edge; a level fixed a day stale is a worse entry, not an equivalent one.

## 1,443 — stop slippage on the tape

Base = 1,438's fills (`causal_arming_causal.csv`); 1,427 fills are VOID; TEST not read (never run for 1,438). Coverage 4879/7096 tape windows (68.8%; gaps: no_valid_quote 1154, no_print_le_stop 999, no_tape 64, zero fetch_error).

| holdout | kind | n meas/req | mean bps | median | p75 | p90 | >30bps | >100bps | mean slip R |
|---|---|---|---|---|---|---|---|---|---|
| TRAIN-H2 | stop | 1831/2565 | 35.9 | 23.7 | 46.4 | 82.6 | 41.7% | 6.3% | 0.209 |
| TRAIN-H2 | eod | 332/567 | 11.5 | 10.3 | 25.4 | 39.5 | — | — | — |
| VAL | stop | 2291/3210 | 34.8 | 22.4 | 47.2 | 81.9 | 40.3% | 6.5% | 0.199 |
| VAL | eod | 425/754 | 9.7 | 6.5 | 22.0 | 46.7 | — | — | — |

| cell | holdout | n stops | slip bps | net R before | net R after |
|---|---|---|---|---|---|
| 1438 | TRAIN-H2 | 2565 (1831 meas) | 35.9 | -0.208 | -0.295 |
| 1438 | VAL | 3210 (2291 meas) | 34.8 | -0.224 | -0.307 |
| 1439 | TRAIN-H2 | 2702 | 35.9 | -0.129 | -0.263 |
| 1439 | VAL | 3618 | 34.8 | -0.279 | -0.414 |
| 1428 | TRAIN-H2 | 209 | 35.9 | -0.067 | -0.185 |
| 1428 | VAL | 244 | 34.8 | +0.109 | +0.002 |
| 1441 | TRAIN-H2 | 74 | 35.9 | +0.016 | -0.089 |
| 1441 | VAL | 139 | 34.8 | -0.168 | -0.286 |

Live 9/25 had no stop exit (VECO hit target, CDNA hit EOD); the only live-tape number is ENTRY slip: VECO +14.3 bps, CDNA -2.0 bps (n=2). VAL mean measured stop slip 34.8 bps < the 40 bps size gate: PASS, no block, but every restated cell's net R got worse under the measured tape than under the flat 30 bps assumption. Full detail: `research/hod_entry/RESULT_1443.md`.
