# Stage H / F5 — the LIVE HOD-break book: loser anatomy → filters → VAL (2026-09-17)

Written by the coordinator from the agent's result (the agent's harness blocked its own REPORT.md write); all scripts,
`FREEZE.md` (written before VAL), per-trade books and tables are on disk in this directory. TEST was NOT read —
neither frozen stack passed VAL. No engine change is proposed; the book stays `enabled: true, dry_run: true`.

## 0. One page
Parity exact vs `C/score5_results.csv` (TRAIN 2R −0.1721 net / −0.0831 gross / WR 35.0 / stop 59.0 / n 2,688; hold
−0.0629 / +0.0187). Three declared populations before any filter (run_book 12/4, contract (c), next-open fill):

| population | exit | TRAIN net / gross | VAL net / gross | tr/wk | TRAIN total $ @$100 |
|---|---|---|---|---|---|
| DECL (Stage C: range floor ≥5, price ≥5) | 2R | −0.172 / −0.083 | −0.079 / +0.016 | 50.7 | −$46,261 |
| | hold | −0.063 / +0.019 | −0.048 / +0.039 | 40.4 | −$13,446 |
| LIVE-C (lvl ≥20, spread/R ≤0.15, dist ≥3.5, no floor) | 2R | −0.095 / −0.030 | −0.154 / −0.090 | 6.1 | −$2,932 |
| | hold | −0.040 / +0.019 | −0.191 / −0.132 | 5.9 | −$1,202 |
| LIVE-S (what `detect()` admits: dist ≥5.0) | 2R | −0.134 / −0.068 | −0.037 / +0.029 | 3.1 | −$1,990 |
| | hold | −0.147 / −0.087 | −0.115 / −0.056 | 3.1 | −$2,177 |

The $20 floor + spread/R gate are worth ≈ +0.04R on TRAIN and cost 94% of the trades (50.7 → 3.1/wk); all six
live-population cells are TRAIN-negative, five of six VAL-negative. `min_price 20` fails the era test as a delta (H1
+0.114 / H2 −0.029). The gates are cost hygiene, not an edge. The engine's true population is LIVE-S (the detector
refuses < 5% above the open); `dist_open_pct ≥ 5` is the causal universe guarantee (99.98% of it clears
`range_so_far ≥ 5`). Deleting the floor with nothing in its place gives the old +0.164R (t 3.6) look-ahead — recorded so
nobody rediscovers it.

Anatomy ran on population A (dist ≥ 5, rv [1,5), ≤ 14:00, R ≥ 1%, ADV ≥ 100K, price ≥ 5; 4,154 signals; TRAIN book
1,303 trades, −0.1414 net / −0.0493 gross). Loss is diffuse (worst 10% of days = 90% of the loss, best 10% = +129R —
no catastrophic cohort). Median 17 min to the stop; 41.6% of stops had +0.5R on the table (breakeven already measured
negative in probe_stops). No ORB-shaped sub-population: the largest gross-positive both-halves cohort is r_pct 3–4% at
+0.10R gross on 28% of rows; 13:00–14:00 is +0.57R gross on 99 signals (~2/wk).

Availability audit: `pm_dollar_vol` present on 21% (−0.577R where present vs +0.006 absent) and news fetch keys on
71.5% (−0.392R where missing) — the D1 artefact again; both struck out.

## Candidate vetoes (mechanism each; eligibility = bucket negative in BOTH TRAIN halves AND book improves in both)
V1 prev-day range < 5% (ORB PDR: continuation needs day-1 fireworks) · V2 break bar closes in its bottom quarter
(touchgo Rule M) · V3 n_touches ≥ 2 · V4 signal in 10:00–11:30 (liquidity trough; stop rate 67.6% vs 58.2%) · V5
cumulative RTH $ < $1M · V6 spread/R ≥ 0.15 (the live gate) · V7 level < $20 (the live gate — FAILS era test) · V8
vwap_dist < 2% (fails) · V9 consol_bars ≥ 20 · V10 gap ≥ 5% · V11 range so far < 6% · V12 wrapper (fails).

## The stack on TRAIN (greedy ≤ 3, each step must raise both halves; `FREEZE.md` before VAL)
PRIMARY 2R: base −0.1414 (25.1/wk, −$18,429) → +V4 −0.0623 → +V5 −0.0120 → +V1 **+0.0102** (10.6/wk, t 0.17,
52% weeks green, MDD −190 → −33.5R, ex-top-5% −0.090). Cost term unchanged (0.092 → 0.095R): the vetoes fixed the
gross, not the cost; at zero cost the stacked book is +0.105R.
SECONDARY hold: base −0.0674 → +V4 +0.0285 → +V11 +0.1054 → +V1 **+0.1542** (t 1.39) — ex-top-5% −0.322, +3R cap
−0.211: the whole book is the top 5% of trades.

## VAL, read once (pass = mean up AND every vetoed bucket negative AND mean > 0 with ≥ 55% weeks green AND $ up)
2R: base −0.2104 → stack **−0.1956** (t −2.9, 32% weeks green); vetoed buckets V4 −0.190 / V5 −0.111 / V1 −0.126 all
negative; $ rises −$15,736 → −$7,845. Criterion 3 fails → FAIL. Monthly VAL: −21.2 / −37.1 / −14.3 / +1.4 / −7.2.
Hold: base −0.1052 → stack **+0.0032** (t 0.03, 45% green; April +49.6R is the whole book; ex-top-5% −0.437) → FAIL.

## What this says about the live engine
No gate proposed. If F5 is revisited, V4 / V5 / V1 have the best evidence (negative in both TRAIN halves AND on VAL;
slot into `_try_enter` after the `min_price` check; V1 needs the prev-day high/low ORB already loads). The hold exit
beats 2R on every gross number and dies on every tail test — the live 2R resting take-profit is the right shape.

## Power, cells, caveats
MDE of the VAL stack 0.189R/trade (3.4R/wk at 4 slots). 311 cells this stage. The spread is a MODEL (cost_curve, 2,570
samples), every net number is linear in it. No re-detection (scored off candidates4; first-break parity verified at
Stage B). Day-context keep rules (SPY down) are 80%-cut denominator rules inside the April-2025 drawdown; never taken
to VAL. The independent-rebuild leg was not run because the stack is proposed for nothing.

Files: x0_extract.py → f5_all.csv (366,864 signals) · h5core.py · x1_parity.py · x2/x2b (populations) · x3/x4/x5
(anatomy, buckets, candidates) · x6_stack.py → frozen_stack.json · FREEZE.md · x8_val.py · x9_power.py · x10_dump.py →
book_2r_frozen.csv (950 trades), book_hold_frozen.csv (1,016 trades).
