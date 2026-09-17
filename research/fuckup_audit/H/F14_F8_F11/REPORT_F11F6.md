# Stage H — book **F11 close-confirmed F6** (`fam F11`, `cfg {"base": "F6"}`)

Method `H/METHOD.md`; freeze `FREEZE_F11F6.md` (written before `--val` ran). **TEST was not read.**

## One page
| question | answer |
|---|---|
| profit on TRAIN? | **Yes.** −0.0115 -> **+0.0804 R/trade, t 1.62** (no-refill), 12.4 tr/wk, WR 43.5 -> 48.0%, weeks green 43 -> **57%**, MDD −50.4 -> −19.4 R, ex-top-5% −0.193 -> −0.118, +3R-capped −0.056 -> **+0.021**. Halves +0.149 / +0.017. Does NOT clear G1 (t 1.62). |
| profit on VAL? | **Yes on the mean — and it still FAILS the frozen rule.** VAL +0.0775 -> **+0.1517** (t 1.68, 13.1 tr/wk, 55% weeks green, MDD −23.5 -> −12.9, ex5 −0.141 -> −0.077). But the freeze required *each* vetoed bucket to be negative on VAL and **B1's is +0.0345**. (The stack's *combined* vetoed bucket is −0.0070 on 254 trades — indistinguishable from zero.) |
| filters + mechanisms | P1 `prev_day_range_pct >= 5` (a red-to-green reclaim is a CONTINUATION trade; it pays when the name was already moving yesterday — live ORB ships this as the PDR veto, "day-2 of the fireworks, not day-1") · B1 `sig_body_pct < 1.0` (F11 confirms on a CLOSE above the level and fills at the NEXT bar's open: the bigger that bar's body, the further above the level the fill lands and the wider the stop it inherits — it is a chase, and every live engine caps the chase). |
| verdict | **FAIL at METHOD step 4 on the letter of the frozen rule.** This is the book that came closest; the failing leg is B1, not P1. |
| smallest visible effect (VAL) | **0.252 R/trade** = 3.3 R/week at 4 slots. |

## 0. Parity anchor
All-day TRAIN n 1239 / −0.0115 / t −0.35 · VAL n 543 / +0.0775 / t 1.28 **and** the `>=10:00` twin TRAIN n 1018 /
−0.0045 / t −0.18 · VAL n 442 / +0.0853 / t 1.59 — all exact against `C/score5_results{,.m600}.csv`
(`dn = 0`, `|dR| = 0.0`). The METHOD table's "+0.065 (t 2.0)" is the `>=10:00` **`2R stop-1%`** cell, also
reproduced exactly (TRAIN +0.0062, VAL +0.0646, t 2.00).

## 1. Step 1 — anatomy (`h1_anatomy_F11F6.md`, `h1_buckets_F11F6.csv`)
### 1.0 Availability audit — `news_pre` REJECTED, and it is the loudest signal in the whole sub-stage
| feature | TRAIN coverage | missing mean net R (TR/VAL/TEST) | present | verdict |
|---|---:|---|---|---|
| `news_pre` (D ∪ E) | **0.9203** (0.710 in the 09:30-09:45 band) | **−0.251 / −0.251 / −0.360** | +0.001 / +0.064 / −0.033 | **REJECTED.** A 0.25-0.36 R availability indicator, the same sign on all three splits, 20% of the booked book. Coverage = "this symbol-day was in D0's key set (F6/F8 signals >=10:00) or E's causal U1∪U2 universe". Using it is trading the fact that another stage fetched you. |
| `pm_dollar_vol` | 0.3813 | +0.086 / +0.154 / +0.010 | −0.188 / −0.122 / −0.172 | **REJECTED** (D1's signature). |
| `prev_day_range_pct`, `spread_cc_bps`, `spy_at_entry` | 1.0000 | — | — | admitted |
| `adv20` | 0.9780 | — | — | admitted |

**The causal half of the news indicator is recoverable and is P1.** E's key set is `gap >= +3%` ∪
`prev_day_range_pct >= 8%`, both knowable at 09:30. F11's base (F6) opens BELOW the previous close, so the gap leg
never fires — the entire causal content of "news known" for this book is *the previous day had a big range*, which
is exactly the filter that survives below.

### 1.1 Concentration
249 booked days, 120 green / 129 red (48%), total **−14.3 R**. Losing days −271.4 · winning +257.1 ·
worst 5% (12 days) **−60.8** (22% of day-losses) · worst 10% −103.1 (38%) · best 5% +91.0 ·
**both tails removed −44.5 R** — this book is negative in its body as well as its tail, the only one of the three.
Weeks 43% green, worst −12.8, best +20.5.

### 1.2 Winners vs losers
The book's own `gap_pct` is −3.5 by construction (F6 = red-to-green), so the gap lever is unavailable.
The separations that survive as buckets are `prev_day_range_pct < 5` (booked −0.190, halves −0.188/−0.192),
`sig_body_pct > 0.824` (booked −0.067, −0.077/−0.058), `price >= 50` (booked −0.133, −0.176/−0.109) and
`dow == 0` (Monday, booked −0.155, −0.112/−0.200 — no mechanism claimed).

### 1.3 Path anatomy
eod 914 / stop 325 (**26.2%** — the highest of the three); 97 min to the stop (median 66).
**Wick stops 178 = 54.8% of stops**: more than half of this book's stops fire on a bar that closes back above the
stop level. Of the stops, **30.8% had +0.5R on the table first, 13.8% +1R, 8.3% +1.5R** — the strongest case in the
sub-stage for a breakeven / partial rule as the right SHAPE (it was outside the 3-filter budget, which METHOD
restricts to vetoes and one shape change). MAE winners 1.77% / losers 4.69%. Holding time: 0-15 −1.224 (n 62) ·
15-60 −1.108 (93) · 60-150 −0.844 (105) · 150+ **+0.259** (979). MFE: <0.5R −0.480 · 0.5-1 −0.075 · 1-2 +0.509 ·
2+ +1.784.

### 1.4 Era consistency
135 cells, **37 negative in both halves** — the richest of the three books, and the reason it is also the one that
gets furthest. All three of the mechanism filters below pass BOTH legs (population negative in both halves AND the
booked result improves in both halves).

## 2-3. Filters and the stack on TRAIN (9 single cells; `h_eval_F11F6.md`)
| filter (KEEP) | vetoed pop H1/H2 | vetoed booked H1/H2 | book H1 -> | book H2 -> |
|---|---|---|---|---|
| P1 `prev_day_range_pct >= 5` | −0.184 / −0.063 | −0.188 / −0.192 | +0.023 -> +0.094 | −0.044 -> −0.006 |
| B1 `sig_body_pct < 1.0` | −0.063 / −0.037 | −0.103 / −0.081 | +0.023 -> +0.071 | −0.044 -> −0.029 |
| PR1 `price < 50` | −0.096 / −0.053 | −0.176 / −0.109 | +0.023 -> +0.043 | −0.044 -> −0.033 |

| cell (no-refill) | n | tr/wk | net R | t | WR% | stop% | green | worst wk | MDD | ex5 | cap3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1239 | 23.4 | −0.0115 | −0.35 | 43.5 | 26.2 | 0.43 | −12.8 | −50.4 | −0.1926 | −0.0557 |
| + P1 | 955 | 18.0 | +0.0414 | 0.99 | 45.7 | 29.5 | 0.51 | −9.0 | −22.3 | −0.1589 | −0.0158 |
| **+ B1 = STACK** | 659 | 12.4 | **+0.0804** | **1.62** | 48.0 | 25.5 | **0.57** | −8.0 | −19.4 | −0.1179 | **+0.0207** |
| STACK H1 | 317 | 6.0 | +0.1489 | 1.84 | 47.6 | 24.3 | | | | −0.0840 | +0.0570 |
| STACK H2 | 342 | 6.5 | +0.0169 | 0.29 | 48.2 | 26.6 | | | | −0.1472 | −0.0130 |
| STACK refill | 1070 | 20.2 | +0.0581 | 1.69 | 47.0 | 22.1 | 0.53 | −10.8 | −23.0 | −0.1145 | +0.0188 |

**PR1 was dropped ON TRAIN**: it passes both era legs alone but takes the 3-stack to +0.0711 from +0.0804. The
frozen stack is therefore two filters, not three. Coarse grid also covered `pdr >= 8`, `body < 0.5`, `price < 20`,
`no wrapper`, `adv20 known`, `not Monday`.

## 4. VAL — read once, stack frozen
| cell | n | tr/wk | net R | t | WR% | green | worst wk | MDD | ex5 | cap3 | MDE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 543 | 24.7 | +0.0775 | 1.28 | 44.9 | 0.59 | −10.5 | −23.5 | −0.1411 | +0.0048 | 0.170 |
| + P1 | 448 | 20.4 | +0.1122 | 1.57 | 45.5 | 0.59 | −8.7 | −23.8 | −0.1226 | +0.0247 | 0.201 |
| **STACK (no-refill)** | 289 | 13.1 | **+0.1517** | 1.68 | 46.4 | **0.55** | −6.2 | −12.9 | −0.0770 | +0.0664 | 0.252 |
| STACK (refill) | 507 | 23.0 | +0.0465 | 0.82 | 44.2 | 0.45 | −9.2 | −25.9 | −0.1382 | −0.0027 | 0.159 |

Pre-registered pass rule, leg by leg: improves vs the unfiltered VAL book **PASS** (+0.0775 -> +0.1517);
VAL mean > 0 **PASS**; >= 55% weeks green **PASS** (0.55); **each vetoed bucket negative on VAL — FAIL**.

Per-filter vetoed bucket, TRAIN booked -> VAL booked (`valdiag_F11F6.csv`):
**P1 −0.1895 -> −0.0862 (holds)** · **P1b −0.0481 -> −0.0504 (holds)** · **B1 −0.0910 -> +0.0345 (flips)** ·
B1b −0.0388 -> +0.1237 (flips) · PR1 −0.1333 -> +0.1136 (flips) · D0 −0.1548 -> +0.4504 (flips).
(PR1b, W1 and V1 fail the era leg on TRAIN and are not counted.) **2 of this book's 6 era-passing buckets hold
their sign on VAL, and both of them are P1** — the same rule at its two thresholds. Combined stack-vetoed bucket
on VAL: **−0.0070 on 254 trades**.

**Why TEST was still not read.** Three arguments, all pointing the same way: (i) the freeze says *each*, and B1
fails it; (ii) the combined bucket at −0.007 R is inside noise, so the "stack as a whole" reading is a coin flip,
not a pass; (iii) the whole VAL book — filtered — still dies with its top 5% removed (−0.077), on both splits.
Reading TEST on a coin-flip interpretation is the exact discipline failure this audit exists to prevent.

## 5. TEST — **not read**

## Declared sensitivities
| sensitivity | TRAIN | VAL |
|---|---|---|
| **`2R stop-1%` exit** (the least tail-dependent cell in Stage H) | baseline +0.0095 (t 0.39); **stack +0.0623, t 1.82, H1 +0.075 / H2 +0.050, ex5 −0.0365, cap3 +0.0623, MDD −16.8** | baseline +0.0242; **stack +0.0632, t 1.15, 55% green, ex5 −0.0307, cap3 +0.0632, MDD −12.3** |
| `>= 10:00` window | baseline −0.0045; stack +0.0534 (t 1.72), H1 +0.015 / **H2 +0.090 (t 2.13)** | baseline +0.0853; stack +0.0843 (t 1.16), 50% green |
| refill booking | +0.0581 (t 1.69) | +0.0465 (t 0.82) |
| no-floor twin | combined REPORT §5 (a look-ahead on this universe) |

## Phrasing (PLAN §1)
In THIS universe, at THIS horizon, at THIS book size, over THIS window and at THIS cost, a two-veto stack on
F11(F6) raises the book from −0.0115 to +0.0804 R/trade on TRAIN and from +0.0775 to +0.1517 on VAL while halving
the drawdown on both — and one of its two vetoed buckets reverses sign on VAL, so under the rule frozen before the
VAL read the book does not pass and TEST stays unread. The smallest per-trade effect the VAL test could have seen
is **0.252 R** (~3.3 R/week at 4 slots); the observed +0.074 improvement is well under that, i.e. **this VAL result
is not resolvable at n = 289 either way**.
