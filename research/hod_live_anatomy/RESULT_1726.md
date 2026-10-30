# Cell 1,726: passive squeeze + uncensored afternoon path + double-down (PREREG_1726)

Live HOD paper book, 231 fills (0 without cached bars: Delta 0), free Alpaca 1-min bars (cell 1,724 cache). Live replay reproduces variants_1724 live_R to max |diff| 4.4e-16. Half = alternating trade dates (A/B).

## (A) Passive squeeze

### A1 offer = 13:59 close + 1 tick, R1 squeeze side

Eligible positions 46; filled via the offer **44** = 96% of eligible, 19.0% of all 231 fills. Pooled over all fills (Delta = 0 where the offer does not fill).

| basis | pooled dR/trade | t iid / day-cl. | ex-top-5 % | half A | half B | dR on touched | verdict (bar: >= +0.02, both halves > 0, ex-top-5 % > 0) |
|---|---|---|---|---|---|---|---|
| primary (EOD flat uncosted, as in the live replay) | +0.0035 | +0.19 / +0.25 | -0.0400 | -0.0251 | +0.0349 | +0.018 (n 44) | FAIL |
| sensitivity (live EOD flat charged 12 bps) | +0.0117 | +0.63 / +0.83 | -0.0334 | -0.0157 | +0.0420 | +0.062 (n 44) | FAIL |

### A2 offer = 13:59 close + 0.5 x bar range, R1 squeeze side

Eligible positions 46; filled via the offer **43** = 93% of eligible, 18.6% of all 231 fills. Pooled over all fills (Delta = 0 where the offer does not fill).

| basis | pooled dR/trade | t iid / day-cl. | ex-top-5 % | half A | half B | dR on touched | verdict (bar: >= +0.02, both halves > 0, ex-top-5 % > 0) |
|---|---|---|---|---|---|---|---|
| primary (EOD flat uncosted, as in the live replay) | +0.0081 | +0.43 / +0.55 | -0.0369 | -0.0216 | +0.0408 | +0.044 (n 43) | FAIL |
| sensitivity (live EOD flat charged 12 bps) | +0.0161 | +0.85 / +1.09 | -0.0306 | -0.0123 | +0.0474 | +0.087 (n 43) | FAIL |

### A3 passive EOD, every position open at 15:30 (15:29 close + 1 tick)

Eligible positions 54; filled via the offer **50** = 93% of eligible, 21.6% of all 231 fills. Pooled over all fills (Delta = 0 where the offer does not fill).

| basis | pooled dR/trade | t iid / day-cl. | ex-top-5 % | half A | half B | dR on touched | verdict (bar: >= +0.02, both halves > 0, ex-top-5 % > 0) |
|---|---|---|---|---|---|---|---|
| primary (EOD flat uncosted, as in the live replay) | -0.0001 | -0.00 / -0.00 | -0.0243 | -0.0060 | +0.0065 | -0.000 (n 50) | FAIL |
| sensitivity (live EOD flat charged 12 bps) | +0.0099 | +0.95 / +0.93 | -0.0161 | +0.0054 | +0.0148 | +0.046 (n 50) | FAIL |

Verdict (primary basis): A1 FAIL, A2 FAIL, A3 FAIL. 

## (B) Uncensored 14:00 open -> 15:55 close path (stop and target ignored)

| group | n | mean bps | median bps | t (bps) | mean R | median R | mean MFE R | mean MAE R | closes above 14:00 open |
|---|---|---|---|---|---|---|---|---|---|
| R1 hold ALL | 21 | +13 | +88 | +0.32 | +0.050 | +0.494 | +0.74 | -0.68 | 62% |
| R1 hold A | 9 | +90 | +134 | +1.45 | +0.408 | +0.655 | +1.08 | -0.69 | 78% |
| R1 hold B | 12 | -45 | +5 | -0.90 | -0.218 | +0.033 | +0.49 | -0.67 | 50% |
| R1 squeeze ALL | 46 | -9 | -16 | -0.42 | +0.054 | -0.049 | +0.58 | -0.52 | 43% |
| R1 squeeze A | 23 | +26 | +19 | +0.99 | +0.142 | +0.073 | +0.55 | -0.42 | 52% |
| R1 squeeze B | 23 | -43 | -20 | -1.36 | -0.034 | -0.136 | +0.60 | -0.63 | 35% |
| R2 hold ALL | 23 | -16 | -1 | -0.38 | -0.069 | -0.003 | +0.65 | -0.80 | 48% |
| R2 hold A | 6 | +109 | +166 | +1.36 | +0.526 | +1.147 | +1.18 | -0.89 | 67% |
| R2 hold B | 17 | -60 | -101 | -1.33 | -0.280 | -0.395 | +0.46 | -0.77 | 41% |
| R2 squeeze ALL | 44 | +5 | -2 | +0.27 | +0.117 | -0.006 | +0.62 | -0.45 | 50% |
| R2 squeeze A | 26 | +29 | +53 | +1.13 | +0.146 | +0.168 | +0.59 | -0.40 | 58% |
| R2 squeeze B | 18 | -28 | -16 | -0.96 | +0.075 | -0.054 | +0.66 | -0.52 | 39% |

### B1 double down (R1 hold side adds the original share count at the 14:00 open + 12 bps)

| set | n | mean add R | t iid / day-cl. | ex-top-5 % | half A | half B |
|---|---|---|---|---|---|---|
| add leg | 21 | +0.030 | +0.17 / +0.14 | -0.077 | +0.237 (n 9) | -0.126 (n 12) |
| combined dR/trade over all 231 fills (live + add; Delta = add R in original-R units) | 231 | +0.0027 | +0.17 / +0.14 | -0.0308 | | |

Add exits by kind: {'target': 10, 'eod': 7, 'stop': 4}. Live R on the same 21 hold positions: mean +1.125; combined (live + add) mean on the 21 = +1.155 R of original risk.

**B1 verdict (bar: add-leg mean >= +0.10, both halves > 0, ex-top-5 % > 0): FAIL**

## (C) Gate for a dedicated 14:00 runner scan (cell 1,727)

Rule: R1 hold side 14:00 -> 15:55 return >= +30 bps net of 24 bps round trip (i.e. gross >= 54 bps) in BOTH halves. Mean basis (primary): A +90 bps (n 9), B -45 bps (n 12) -> net A +66, net B -69. Median basis: A +134, B +5.
**Gate outcome: NOT MET (mean basis)** (median basis: NOT MET). Cell 1,727 was NOT run here.

## Implementation caveats (my own)

- Offer placed at the 13:59 close + tick: bars carry no quotes; the real ask is unknown. A1's 'later bar' includes the 14:00 bar itself (the order exists from 14:00:00); fill price = limit even when a bar opens above it (conservative).
- Fill test is high >= limit + 0.01 on 1-min bars: a bar high can be a single print, no queue position is modelled, so fill shares are an upper bound on a real resting order.
- Passive exits pay 0 bps, the live EOD flat in the replay also pays 0 (close of the 15:55 bar, cell 1,724 convention), so the primary basis does not credit the 12 bps cost lever on EOD exits; the sensitivity charges 12 bps to every EOD flat on both sides. Stops keep their measured slip in both.
- The live 2R target stays resting beside the offer; stop-first inside a bar. R1 hold flags are the frozen 1,725 flags (features through the 13:59 bar; high-so-far = post-entry high since the cache starts at the fill).
- B path uses the 14:00 bar open and the 15:55 close (last close <= 15:55 if absent); no cost in the descriptive table (the gate subtracts 24 bps). B1 add stop 1 R below the 13:59 close is always above the original stop (open P&L >= 0.5 R), so the add stops first; add R is divided by the ORIGINAL rps (actual risk is slightly above 1 R after the +12 bps entry). A gap above the live target at 14:00 sells the add at the target price (replay convention).
- 21 hold positions: the add-leg and gate numbers are tiny-sample; cells counted: A1, A2, A3 (x2 bases), B1, path tables, gate = 7.
- A lost-bars fill contributes Delta 0 (no cached bars).
