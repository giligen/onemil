# Stage H — book **F14, second break** (`fam F14`, `cfg {"N": 15}`)

Method `H/METHOD.md`; freeze `FREEZE_F14.md` (written before `--val` ran). **TEST was not read.**

## One page
| question | answer |
|---|---|
| profit on TRAIN? | **Yes.** +0.0564 -> **+0.1398 R/trade, t 2.41** (no-refill; +0.1260, t 2.44 refill), 9.8 tr/wk, halves +0.160/+0.123, MDD −31.3 -> −19.8 R, 55% weeks green. |
| profit on VAL? | **No.** The frozen stack takes VAL from **+0.0503 to +0.0096** (t 0.17). **8 of 9** vetoed buckets flip from −0.03..−0.28 on BOTH TRAIN halves to **+0.11..+0.34** on VAL; the 9th has no VAL instance. |
| filters + mechanisms | V4 `dist_open_pct > 0.8` (a continuation trade needs a day that has left its open) · V2 `vwap_dist_pct >= 0` (below VWAP the average share bought today is held above the break — the shipped BF `vwap_gate`) · V1 `adv20` exists (>= 20 prior daily bars — the shipped ORB G1 short-history veto). |
| verdict | **FAIL at METHOD step 4.** The losers are not separable by causal features at this power. |
| smallest visible effect (VAL) | **0.156 R/trade** = 1.9 R/week at 4 slots. |

## 0. Parity anchor
`h0_parity.csv`: TRAIN n 768 / +0.0564 / t 1.32 and VAL n 363 / +0.0503 / t 0.91 vs `C/score5_results.csv` —
**18 of 18 cells exact, max|dn| 0, max|d meanR| 0.0**. F14's signals are all post-10:00 by construction, so the
all-day and `>=10:00` windows are the same book.

## 1. Step 1 — anatomy (TRAIN; `h1_anatomy_F14.md`, `h1_buckets_F14.csv`)
### 1.0 Availability audit — two features REJECTED before use
| feature | TRAIN coverage | missing mean net R (TR/VAL/TEST) | present | verdict |
|---|---:|---|---|---|
| `pm_dollar_vol` | 0.4665 | **+0.380** / +0.116 / +0.022 | −0.096 / −0.111 / −0.136 | **REJECTED** — D1's signature: a 0.48 R availability indicator. `D/pm_bars.db` was backfilled over D0's key set (F6/F8 signals >=10:00 in `candidates3`); "premarket known" here means "this symbol-day also produced a D0 signal". |
| `news_pre` (D ∪ E) | 0.7768 | −0.010 / −0.003 / −0.338 | +0.206 / +0.016 / −0.000 | **REJECTED** — 22% missing, 0.22 R present-minus-missing on TRAIN; coverage is another stage's key set. |
| `spy_at_entry`, `spread_cc_bps` | 1.000 | — | — | admitted |
| `prev_day_range_pct` 0.994 · `adv20` 0.9725 | | | | admitted; `adv20`'s missingness IS the V1 veto (a real category) |

### 1.1 Concentration — the book is its tail
231 booked days, 110 green / 121 red (48%), total **+43.3 R**. Losing days −168.6 · winning +212.0 ·
worst 5% of days (11) **−41.5** (25% of all day-losses) · worst 10% −73.4 (44%) · best 5% **+88.3** ·
**both tails removed: −3.4 R**. Weeks 45% green, worst −8.1, best +27.4. April 2025 alone is +33.7 R of +74.9.
The 20 worst days are mildly down (SPY c-o −0.27%, IWM −0.48% vs +0.03/+0.04 overall) but so is every red day
(−0.09/−0.13); the 5 worst include 2025-04-07 with SPY **+3.1%**. Booked trades by the day's SPY close-open:
−0.013 / +0.002 / +0.066 / **+0.170** — a 0.18 R spread that is NOT causal; `spy_at_entry` (100% coverage, the
SPY return to the entry minute) splits the book only +0.05 / −0.03 / +0.14 / +0.04.

### 1.2 Winners vs losers
The two biggest raw gaps are Simpson artefacts: `gap_pct` losers 6.39 vs winners 0.34, yet every gap bucket mean
is **positive** (+0.11..+0.24); `rv_adv` losers 1.00 vs winners 0.64, yet every rv bucket is positive (+0.12..+0.21).

### 1.3 Path anatomy
eod 605 / stop 163 (21.2%); 100 min to the stop (median 95). Of the stops, **25.2% had +0.5R on the table first,
8.6% +1R, 3.7% +1.5R** — a breakeven rule is the right SHAPE for a quarter of them; it is not a veto and was
outside the 3-filter budget. MAE: winners 1.26% of price, losers 3.78%. Holding time: 0-15 min −1.157 (100% stops)
· 15-60 −1.076 · 60-150 −0.312 · 150+ **+0.285** (mechanical). MFE buckets: <0.5R −0.507 · 0.5-1 −0.035 ·
1-2 +0.609 · 2+ +2.241. Wick stops (touch stop fires, the bar closes back above): the close-stop proxy marks
**100% of F14's stops** — F14's stop is the lowest low from the first break's stop bar through the signal bar,
a level the tape has already visited.

### 1.4 Era consistency
138 cells; **16 negative in both halves**, 9 with >=150 pop rows and >=20 booked trades. Removing the two
availability-rejected features (4 `pm_dollar_vol` cells) and the two non-causal ones (`spy_co`, `iwm_co` = the
whole day's close-to-open) leaves exactly the filter set below.

## 2-3. Filters and the stack on TRAIN (9 cells; `h_eval_F14.md`)
| filter (KEEP) | vetoed bucket pop H1/H2 | vetoed booked H1/H2 |
|---|---|---|
| V4 `dist_open_pct > 0.8` | −0.048 / −0.021 | −0.215 / −0.010 |
| V2 `vwap_dist_pct >= 0` | −0.031 / −0.130 | −0.058 / −0.018 |
| V1 `adv20` exists | −0.363 / −0.118 | −0.397 / −0.108 |

| cell (no-refill) | n | tr/wk | net R | t | WR% | green | MDD | ex5 | cap3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 768 | 14.5 | +0.0564 | 1.32 | 45.2 | 0.45 | −31.3 | −0.1189 | +0.0125 |
| + V4 | 576 | 10.9 | +0.1116 | 2.08 | 45.8 | 0.51 | −20.3 | −0.0844 | +0.0530 |
| + V2 | 539 | 10.2 | +0.1230 | 2.18 | 45.5 | 0.55 | −19.9 | −0.0780 | +0.0604 |
| **+ V1 = STACK** | 521 | 9.8 | **+0.1398** | **2.41** | 46.3 | 0.55 | −19.8 | −0.0676 | +0.0750 |
| STACK H1 | 237 | 4.5 | +0.1602 | 1.69 | 45.6 | | | −0.0579 | +0.0774 |
| STACK H2 | 284 | 5.4 | +0.1229 | 1.72 | 46.8 | | | −0.0661 | +0.0731 |
| STACK refill | 598 | 11.3 | +0.1260 | 2.44 | 46.3 | 0.51 | −20.4 | −0.0671 | +0.0695 |

Grid also covered `vwap>=1`, `range<10/12/15`, `dist_open>0.5/1.5`; `dist_open>1.5` is worse than `>0.8` in both
halves, so the chosen cut is not at a grid edge. The stack clears G1 on TRAIN, dies ex-top-5% (−0.068), survives
the +3R cap (+0.075).

## 4. VAL — read once, stack frozen
| cell | n | tr/wk | net R | t | green | ex5 | cap3 | MDE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 363 | 16.5 | **+0.0503** | 0.91 | 0.45 | −0.1048 | +0.0303 | 0.154 |
| STACK (no-refill) | 269 | 12.2 | **+0.0096** | 0.17 | 0.55 | −0.1118 | +0.0079 | 0.156 |
| STACK (refill) | 303 | 13.8 | −0.0053 | −0.10 | 0.55 | −0.1252 | −0.0068 | 0.144 |

**FAIL on leg 1** (no improvement). Per-filter vetoed bucket, TRAIN booked -> VAL booked (`valdiag_F14.csv`):
V4 −0.109 -> **+0.210** · V2 −0.036 -> **+0.328** · V3 `range<12` −0.177 -> **+0.269** · V3b −0.129 -> +0.114 ·
V3c −0.277 -> +0.337 · V4b −0.140 -> +0.239 · V4c +0.025 -> +0.140 · V2b −0.075 -> +0.149 · V1 −0.337 -> (n=0).

## 5. TEST — **not read** (VAL failed)

## Declared sensitivities
`2R stop-1%` on the same stack: TRAIN +0.0428 (t 1.35) vs its baseline +0.0190; VAL **+0.0313 vs +0.0432** — still
no VAL improvement. No-floor twin: see the combined REPORT §5 (it is a look-ahead on this universe).
Refill vs no-refill: printed in every table; the two agree for this book.

## Phrasing (PLAN §1)
In THIS universe (point-in-time >=5%-range days with the causal `range_so_far_pct >= 5` floor), at THIS horizon
(1-min bars, entries 09:30-14:01, hold to 15:55 on the touch stop), at THIS book size (12/day, 4 concurrent), over
THIS window (TRAIN 2025, VAL 2026-01..05), at THIS cost (contract (c)), the F14 losers are **not separable** by any
of the 9 causal cuts tested; the smallest per-trade effect the VAL test could have seen is **0.156 R**
(~1.9 R/week at 4 slots, ~$190/week at $100 risk). Effects below that are not excluded.
