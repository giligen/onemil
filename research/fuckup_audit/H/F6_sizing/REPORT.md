# H/F6_sizing — how much money the F6 "red-to-green" book can carry

Run 2026-09-17. Question from the owner: **how much money can this book carry?** This is a CAPACITY measurement,
not a strategy stage: no filter is proposed, no rule is selected, nothing here is a go/no-go on F6. It answers
"at what risk-per-trade does this book stop being executable, and what is the $ throughput at that point".

Scripts (all under this directory, nothing outside it was written):
`f6_load.py` (population + reproduction) -> `f6_bars.py` (fill-bar liquidity) -> `f6_sizing.py` (tables).
Artifacts: `pop_f6.csv` (11,906 population rows), `pop_f6_liq.csv` (+ liquidity), `trades_f6_sizing.csv`
(2,156 booked trades, per-trade shares/notional/participation at every risk level), `weekly_dollars.csv`,
`tables.md` (raw output of `f6_sizing.py`, reproduced verbatim below).

## Contract (frozen before the run, copied from B/score5.py — the reproduction in section 0 is the proof)

- Population: `C/pop_c.csv`, `fam == 'F6'`, `next_entry` present and `>= 5`, `next_entry_m <= 841` (14:01),
  `next_r_pct >= 1.0`, `range_so_far_pct >= 5` (the causal membership guarantee). Fill = next bar's open.
- Cost: `half = 0.5*(spread_cc_bps/100)/max(r_pct, 0.05)`;
  `net = rr - 0.25*half - half*{stop .875, lock .875, eod .412, target .875, none .875}[why]`.
- Exits: hold-to-close (`next_rr_hold`/`next_why_hold`/`next_exit_m_hold`) and 2R close-fill
  (`next_rr_2r`/`next_why_2r`/`next_exit_m_2r`).
- Book: `trading.hod_break.run_book(rows, 12, 4)` per day.
- Splits: TRAIN 2025 (53 weeks) / VAL 2026-01..05 (22 weeks) / TEST 2026-06..09-11 (18 weeks).

## Assumptions, stated once

| assumption | value | why |
|---|---|---|
| account equity | ~$66,000 | the owner's stated account size; every $ figure scales linearly with risk, not with equity |
| day-trading buying power | ~$264,000 (4x equity) | PDT margin on a $66K account |
| liquidity window | the 5 one-minute bars **ending at the fill bar** (t-4..t), close x volume | the position is entered at the open of bar t; the 5-minute tape around it is the pool the order competes in |
| participation limit | 1% of that 5-minute dollar volume is "clean"; 2% and 5% shown as looser bars | conventional small-cap intraday rule of thumb; it is a CHOICE, not a measurement — every table is also given at 2% and 5% |
| bar source | `data/cache.db intraday_bars_1min` first, then `research/bf_zero/bars_sip.db` | exactly `research/bf_zero/build_candidates.py::load_bars`, the precedence the population itself was built with. bars_sip.db ALONE covers only 41% of these symbol-days (it is the side store); using it alone would have silently dropped 59% of the book |
| weeks denominator | the distinct weeks present in the F6 population per split | reproduces score5's `wkR` exactly |

Coverage: symbol-day found **100.0%**, fill bar present **100.0%**, and the next-open fill lies inside its own bar
on **100.00%** of booked trades (PLAN §1 obtainability check). ADV20 is missing on 1.6% of population rows
(69 booked trades) — those rows are shown as a separate "missing" count in the ADV band table, never silently dropped.

## The answer in one paragraph

**The honest capacity of this book is about $250-$400 of risk per trade, and its dollar throughput saturates at
roughly $1,400-$2,000 per month (TRAIN+VAL pooled), not at the $3.5K-$17K per month that naive
"weekly R x risk" arithmetic gives.** The reason is in section 2b and section 6: the median booked trade can only
carry **$226** of risk at 1% of its own 5-minute tape, and the trades that CAN carry $1,000+ are the $50-100 and
$10M-50M-ADV names whose mean net R is **negative** (-0.144 and -0.012). Scaling risk therefore does not scale
P&L — it deletes the part of the book that has the edge. Pooled TRAIN+VAL, at a 1% participation cap: $100 risk
-> $474/month, $250 -> $1,426, $400 -> $1,700, $700 -> $2,013, $1,000 -> $1,526, $2,000 -> $1,705. The curve is
FLAT from $250 up; there is no risk level at which this book pays four figures a week. Day-trading buying power is
never the binding constraint (gross 4-concurrent exposure at $400 risk has a median of $21.7K and a p90 of $38.0K
against $264K of DTBP; it only touches DTBP on 2.4% of entries at $2,000 risk) — liquidity binds first, by an
order of magnitude. Three caveats that are larger than the number itself: (1) **TEST is negative** (hold -0.083R,
2R -0.048R), so the TRAIN/VAL weekly R this projection multiplies is not a forward expectation; (2) the whole
TRAIN edge is inside the cost uncertainty — at 1.5x the spread curve TRAIN hold falls to +0.035R and at 2x to
+0.018R, and the 2R exit goes negative at 2x (section 5), while the untraded thin names this book lives in are
exactly the ones likeliest to quote wider than the curve; (3) the 1% cap is a measurement convention, and it is not
even live-computable as written (the fill bar's own volume is unknown when the order is sent) — the causal version
using the four PRIOR bars is in section 4 and gives the same plateau.

## What that means, concretely

- At the risk level the F5 engine currently runs ($100), this book is a **$400-500/month** instrument with ~20
  trades a week. That is not a retirement book; it is a live-evidence instrument.
- Between $250 and $700 risk the capped throughput is $1.4K-$2.0K/month and the number of executable trades per
  week falls from 18-22 to 14-19. Beyond $700 the throughput stops rising at all.
- If the F6 edge were re-selected onto the liquid half of the population (the $20-50 price band, +0.087R, median
  capacity $461/trade; or the 2M-10M ADV band, +0.076R, median capacity $928/trade) the capacity per trade rises
  4-5x on ~21-26% of the book. **That is an observation from section 6, not a proposal** — it is a bucket chosen
  after looking at the table, it has not been era-split, and it belongs in H/F6's filter stage under the ORB veto
  rule (negative in BOTH TRAIN halves, improves both), not here.

## Cell count and what was NOT done

64 descriptive cells: 36 book re-runs (2 exits x 6 risk levels x {as-is, 1% cap, causal 1% cap}), 18 cost cells
(2 exits x 3 splits x 3 spread multipliers), 10 band cells (5 price + 5 ADV). **No selection was made**, so no
multiplicity correction applies to a decision — but if any bucket from section 6 is later adopted as a filter,
these 10 band cells count toward that filter's denominator. TEST was read for the liquidity distribution and is
reported in sections 0 and 5; every $ projection uses TRAIN and VAL weekly R only, as briefed. The 2R book's
liquidity is identical to the hold book's (the fill bar is the same); sections 1, 2, 2b, 3 and 6 are computed on
the 2,156 hold-exit booked trades, and both exits are carried through sections 0, 4 and 5.

---

## 0. Reproduction of the C numbers (must match C/score5_results.csv before anything else)

| exit | split | n | mean net R | t | ref (C) |
|---|---|---:|---:|---:|---|
| hold | TRAIN | 1242 | +0.0510 | +1.34 | +0.0510 / t 1.34 |
| hold | VAL | 536 | +0.1629 | +2.46 | +0.1629 / t 2.46 |
| hold | TEST | 378 | -0.0827 | -1.33 | (TEST - not in the C table) |
| 2R | TRAIN | 1342 | +0.0273 | +1.05 | +0.0273 / t 1.05 |
| 2R | VAL | 607 | +0.0702 | +1.72 | +0.0702 / t 1.72 |
| 2R | TEST | 419 | -0.0484 | -0.97 | (TEST - not in the C table) |

## 1. The booked set measured (hold exit, all three splits pooled): n = 2156
(TRAIN 1242 / VAL 536 / TEST 378; the 2R book is n = 2368. Liquidity is a property of the FILL BAR, identical under either exit.)

| quantity | p10 | median | mean | p90 |
|---|---:|---:|---:|---:|
| entry price $ | 5.98 | 14.80 | 30.79 | 65.82 |
| R per share $ | 0.30 | 0.93 | 1.99 | 4.27 |
| R as % of price | 3.24 | 6.06 | 6.57 | 9.77 |
| fill-bar $ volume | 2,958 | 55,458 | 788,723 | 1,425,618 |
| 5-min $ volume (bars t-4..t) | 20,645 | 361,492 | 3,978,105 | 7,376,054 |
| ADV20 (shares) | 167,245 | 880,811 | 4,826,996 | 9,678,246 |
| spread_cc_bps | 28.8 | 42.2 | 41.6 | 63.4 |

bars with a print in all 5 of the 5 minutes: 52.0%; 4 or fewer: 48.0%. Obtainability: the next-open fill lies inside its own bar on 100.00% of booked trades.

## 2. Participation = shares x entry / 5-minute $ volume (booked trades, n = 2156)

| risk $/trade | median shares | p<=1% | p<=2% | p<=5% | median participation | p90 participation | median part. of the FILL BAR alone |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 107 | 64.1% | 73.4% | 84.3% | 0.44% | 9.95% | 2.96% |
| 250 | 268 | 48.6% | 60.3% | 73.4% | 1.11% | 24.87% | 7.41% |
| 400 | 429 | 40.4% | 51.6% | 67.0% | 1.77% | 39.79% | 11.85% |
| 700 | 751 | 32.0% | 42.2% | 58.3% | 3.10% | 69.64% | 20.74% |
| 1000 | 1,072 | 26.3% | 37.2% | 51.6% | 4.42% | 99.49% | 29.63% |
| 2000 | 2,145 | 18.3% | 26.3% | 40.4% | 8.84% | 198.97% | 59.26% |

### 2b. The implied capacity of each trade: the risk $ at which participation hits exactly 1%

cap_risk$ = 1% x (5-min $ volume) / entry x (R per share). It is the largest risk-per-trade that trade could have carried at 1% of the 5-minute tape.

| set | n | p10 | p25 | median | p75 | p90 | share of trades carrying >= $400 |
|---|---:|---:|---:|---:|---:|---:|---:|
| all booked | 2156 | 10 | 45 | 226 | 1,095 | 4,938 | 40.4% |
| TRAIN | 1242 | 10 | 41 | 190 | 1,031 | 5,258 | 37.6% |
| VAL | 536 | 8 | 60 | 319 | 1,181 | 5,154 | 45.3% |
| TEST | 378 | 13 | 60 | 256 | 1,145 | 3,883 | 42.9% |
| winners (net R > 0) | 949 | 11 | 47 | 232 | 1,201 | 4,866 | 41.9% |
| losers (net R <= 0) | 1207 | 9 | 43 | 221 | 1,028 | 5,000 | 39.3% |

## 3. Notional per trade and 4-concurrent gross exposure vs day-trading buying power

Assumption: account equity ~ $66,000, pattern-day-trader margin 4x -> DTBP ~ $264,000. "4-concurrent" is the REAL gross exposure measured at every booked entry (this trade plus every booked trade still open at that minute, run_book 12/4), not 4 x the median.

| risk $/trade | notional median | notional p90 | notional max | concurrent median | concurrent p90 | concurrent max | % of entries over DTBP |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 1,650 | 3,086 | 9,682 | 5,426 | 9,506 | 21,394 | 0.0% |
| 250 | 4,126 | 7,716 | 24,205 | 13,565 | 23,765 | 53,486 | 0.0% |
| 400 | 6,602 | 12,346 | 38,727 | 21,704 | 38,024 | 85,577 | 0.0% |
| 700 | 11,553 | 21,605 | 67,773 | 37,982 | 66,542 | 149,760 | 0.0% |
| 1000 | 16,505 | 30,864 | 96,818 | 54,260 | 95,059 | 213,943 | 0.0% |
| 2000 | 33,010 | 61,728 | 193,636 | 108,520 | 190,119 | 427,886 | 2.4% |

## 4. Expected weekly $ = weekly R x risk, as-is and liquidity-capped at 1% participation

The capped book drops every POPULATION row whose participation at that risk level exceeds 1% and re-runs run_book(12,4) on the survivors, so a freed slot refills.

| exit | risk $ | TRAIN wkR | TRAIN $/wk | VAL wkR | VAL $/wk | pop kept @1% | TRAIN capped wkR | TRAIN capped $/wk | VAL capped wkR | VAL capped $/wk | capped trades/wk TRAIN / VAL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| hold | 100 | 1.19 | 119 | 3.97 | 397 | 68.3% | 0.21 | 21 | 3.23 | 323 | 20.0 / 22.5 |
| hold | 250 | 1.19 | 299 | 3.97 | 992 | 53.5% | 0.64 | 159 | 2.96 | 739 | 18.0 / 21.6 |
| hold | 400 | 1.19 | 478 | 3.97 | 1,587 | 45.3% | 0.44 | 177 | 2.28 | 913 | 16.3 / 20.5 |
| hold | 700 | 1.19 | 836 | 3.97 | 2,777 | 35.8% | 0.27 | 188 | 1.62 | 1,132 | 14.5 / 19.2 |
| hold | 1000 | 1.19 | 1,194 | 3.97 | 3,968 | 30.2% | -0.17 | -165 | 1.60 | 1,600 | 13.3 / 17.7 |
| hold | 2000 | 1.19 | 2,388 | 3.97 | 7,935 | 20.7% | -0.16 | -315 | 1.05 | 2,102 | 10.6 / 14.9 |
| 2R | 100 | 0.69 | 69 | 1.94 | 194 | 68.3% | -0.04 | -4 | 2.66 | 266 | 20.9 / 24.6 |
| 2R | 250 | 0.69 | 173 | 1.94 | 484 | 53.5% | 0.32 | 81 | 2.59 | 649 | 18.8 / 23.5 |
| 2R | 400 | 0.69 | 276 | 1.94 | 775 | 45.3% | 0.26 | 103 | 1.92 | 766 | 16.8 / 22.0 |
| 2R | 700 | 0.69 | 483 | 1.94 | 1,356 | 35.8% | 0.00 | 1 | 1.57 | 1,102 | 14.9 / 20.3 |
| 2R | 1000 | 0.69 | 691 | 1.94 | 1,937 | 30.2% | -0.24 | -244 | 1.71 | 1,707 | 13.5 / 18.5 |
| 2R | 2000 | 0.69 | 1,381 | 1.94 | 3,875 | 20.7% | -0.31 | -619 | 0.95 | 1,890 | 10.7 / 15.3 |

The same cap made LIVE-COMPUTABLE (the fill bar's own volume is unknown when the order is sent, so the rule uses the four bars t-4..t-1 scaled x1.25):

| exit | risk $ | pop kept @1% causal | TRAIN causal wkR | TRAIN causal $/wk | VAL causal wkR | VAL causal $/wk | causal trades/wk TRAIN / VAL |
|---|---:|---:|---:|---:|---:|---:|---|
| hold | 100 | 67.4% | 0.17 | 17 | 2.68 | 268 | 20.0 / 22.5 |
| hold | 250 | 53.0% | 0.69 | 172 | 3.39 | 848 | 17.8 / 21.5 |
| hold | 400 | 44.8% | 0.34 | 135 | 2.22 | 887 | 16.1 / 20.5 |
| hold | 700 | 35.5% | 0.20 | 140 | 1.21 | 847 | 14.4 / 19.3 |
| hold | 1000 | 29.8% | -0.07 | -72 | 1.39 | 1,392 | 13.2 / 17.6 |
| hold | 2000 | 20.3% | -0.13 | -258 | 1.41 | 2,815 | 10.4 / 14.5 |
| 2R | 100 | 67.4% | -0.04 | -4 | 2.20 | 220 | 20.9 / 24.6 |
| 2R | 250 | 53.0% | 0.23 | 57 | 3.00 | 749 | 18.6 / 23.5 |
| 2R | 400 | 44.8% | 0.07 | 29 | 1.80 | 722 | 16.7 / 22.0 |
| 2R | 700 | 35.5% | -0.01 | -4 | 1.39 | 973 | 14.8 / 20.3 |
| 2R | 1000 | 29.8% | -0.16 | -157 | 1.69 | 1,689 | 13.4 / 18.5 |
| 2R | 2000 | 20.3% | -0.29 | -584 | 1.34 | 2,672 | 10.6 / 15.0 |

## 5. Spread-cost sensitivity (the untraded names may quote wider than the cost curve)

run_book ignores the payload, so the booked SET is unchanged; only the charge moves.

| exit | split | mean net R x1.0 | x1.5 | x2.0 | weekly R x1.0 | x1.5 | x2.0 | mean spread charge (R) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| hold | TRAIN | +0.0510 | +0.0346 | +0.0182 | +1.19 | +0.81 | +0.43 | 0.0327 |
| hold | VAL | +0.1629 | +0.1456 | +0.1284 | +3.97 | +3.55 | +3.13 | 0.0344 |
| hold | TEST | -0.0827 | -0.1024 | -0.1221 | -2.23 | -2.76 | -3.30 | 0.0394 |
| 2R | TRAIN | +0.0273 | +0.0103 | -0.0066 | +0.69 | +0.26 | -0.17 | 0.0339 |
| 2R | VAL | +0.0702 | +0.0524 | +0.0345 | +1.94 | +1.44 | +0.95 | 0.0357 |
| 2R | TEST | -0.0484 | -0.0687 | -0.0890 | -1.45 | -2.06 | -2.66 | 0.0406 |

## 6. Where the capacity lives — participation by price band and by ADV20 band ($400 risk)

### by price band

| band | n | share of book | median 5-min $vol | median part. @$400 | p<=1% @$400 | p<=1% @$1000 | p<=1% @$2000 | mean net R | max risk $ at 1% for the MEDIAN trade |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| $5-10 | 739 | 34.3% | 125,973 | 5.45% | 23% | 12% | 7% | +0.063 | 73 |
| $10-20 | 572 | 26.5% | 253,563 | 2.57% | 34% | 22% | 15% | +0.091 | 156 |
| $20-50 | 559 | 25.9% | 732,807 | 0.87% | 53% | 34% | 23% | +0.087 | 461 |
| $50-100 | 174 | 8.1% | 1,764,128 | 0.35% | 73% | 53% | 40% | -0.144 | 1,130 |
| $100+ | 112 | 5.2% | 4,089,727 | 0.16% | 79% | 63% | 54% | -0.030 | 2,560 |

(n missing price band: 0)

### by ADV20 band

| band | n | share of book | median 5-min $vol | median part. @$400 | p<=1% @$400 | p<=1% @$1000 | p<=1% @$2000 | mean net R | max risk $ at 1% for the MEDIAN trade |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| <500K | 730 | 33.9% | 87,322 | 7.50% | 12% | 4% | 2% | +0.074 | 53 |
| 500K-2M | 695 | 32.2% | 388,494 | 1.63% | 38% | 21% | 12% | +0.064 | 245 |
| 2M-10M | 462 | 21.4% | 1,519,399 | 0.43% | 70% | 47% | 32% | +0.076 | 928 |
| 10M-50M | 172 | 8.0% | 6,613,812 | 0.10% | 87% | 78% | 68% | -0.012 | 4,098 |
| 50M+ | 28 | 1.3% | 20,632,105 | 0.04% | 82% | 82% | 82% | +0.139 | 10,935 |

(n missing ADV20 band: 69)

per-trade CSV: `research/fuckup_audit/H/F6_sizing/trades_f6_sizing.csv` (2156 booked hold-exit trades, all columns above)
