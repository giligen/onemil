# RESULT 1,720 - QLD 'Claude AI Scalper' replay on QQQ 30-min bars (TRAIN 2016-19, VAL 2020-22; TEST sealed)

Rule as amended (Amendment 1): close > prior-30-bar high, volume >= 1.75 x SMA20, close > open; 16/60-bar exit, 0.5 % threshold at bar 16.


## TRAIN

| trades | in-market share | held bars | mean hold (bars) | win % | QLD-proxy gross comp. | QLD-proxy net comp. (1bp/side) | QQQ 1x gross comp. | max trade | min trade |
|---|---|---|---|---|---|---|---|---|---|
| 120 | 30.9 % | 3988 | 33.2 | 52.5 | 72.4 % | 68.4 % | 32.6 % | 12.6 % | -6.1 % |


Per-trade means (bps). EXCESS = QLD-proxy net trade return - exposure-matched B&H (hold bars x split mean QLD-proxy bar return). 'N1 literal' = QLD net - QQQ 1x same bars (leverage-confounded).

| cost | QLD gross | QLD net | QQQ 1x | B&H matched | EXCESS | t (trade) | ex-top-5% excess | winner-capped(+3%) excess | N1 literal |
|---|---|---|---|---|---|---|---|---|---|
| 0 bp/side | 48.9 | 48.9 | 24.4 | 37.2 | 11.6 | 0.51 | -24.2 | -22.8 | 24.5 |
| 1 bp/side | 48.9 | 46.9 | 24.4 | 37.2 | 9.6 | 0.42 | -26.2 | -24.6 | 22.5 |
| 3 bp/side | 48.9 | 42.9 | 24.4 | 37.2 | 5.6 | 0.25 | -30.2 | -28.0 | 18.5 |


Per year (1 bp/side):

| year | trades | QLD net bps/trade | B&H matched | excess bps/trade | t | abs t > 6 |
|---|---|---|---|---|---|---|
| 2016 | 29 | 29.2 | 36.6 | -7.4 | -0.18 |  |
| 2017 | 35 | 28.3 | 32.0 | -3.8 | -0.16 |  |
| 2018 | 24 | 104.6 | 42.6 | 62.0 | 0.80 |  |
| 2019 | 32 | 40.0 | 39.5 | 0.5 | 0.01 |  |


Years with excess > 0: 2 of 4.  N2 (1000 random-entry draws, same trades/year): mean -18.4 bps, 95th pct 21.3 bps, realised percentile **88.5**.  MDE at n=120 (2 x se) = 46.0 bps/trade.

Flags: none (no trade > +50 % / < -30 %).


## VAL

| trades | in-market share | held bars | mean hold (bars) | win % | QLD-proxy gross comp. | QLD-proxy net comp. (1bp/side) | QQQ 1x gross comp. | max trade | min trade |
|---|---|---|---|---|---|---|---|---|---|
| 84 | 30.1 % | 2928 | 34.9 | 45.2 | -20.7 % | -22.0 % | -8.8 % | 11.4 % | -11.7 % |


Per-trade means (bps). EXCESS = QLD-proxy net trade return - exposure-matched B&H (hold bars x split mean QLD-proxy bar return). 'N1 literal' = QLD net - QQQ 1x same bars (leverage-confounded).

| cost | QLD gross | QLD net | QQQ 1x | B&H matched | EXCESS | t (trade) | ex-top-5% excess | winner-capped(+3%) excess | N1 literal |
|---|---|---|---|---|---|---|---|---|---|
| 0 bp/side | -17.7 | -17.7 | -8.6 | 22.4 | -40.1 | -0.84 | -102.9 | -108.9 | -9.1 |
| 1 bp/side | -17.7 | -19.7 | -8.6 | 22.4 | -42.1 | -0.88 | -104.9 | -110.5 | -11.1 |
| 3 bp/side | -17.7 | -23.7 | -8.6 | 22.4 | -46.1 | -0.96 | -108.9 | -113.7 | -15.1 |


Per year (1 bp/side):

| year | trades | QLD net bps/trade | B&H matched | excess bps/trade | t | abs t > 6 |
|---|---|---|---|---|---|---|
| 2020 | 30 | 1.3 | 22.5 | -21.2 | -0.28 |  |
| 2021 | 31 | 86.4 | 22.1 | 64.2 | 1.18 |  |
| 2022 | 23 | -190.0 | 22.6 | -212.6 | -1.78 |  |


Years with excess > 0: 1 of 3.  N2 (1000 random-entry draws, same trades/year): mean -29.2 bps, 95th pct 48.3 bps, realised percentile **40.7**.  MDE at n=84 (2 x se) = 95.9 bps/trade.

Flags: none (no trade > +50 % / < -30 %).


## 2022 (bear year inside VAL), bar-level open-to-open equity, 1 bp/side

|  | return | max DD |
|---|---|---|
| Strategy (QLD proxy) | -37.6 % | -43.2 % |
| QQQ B&H | -33.8 % | -36.6 % |
| QLD-proxy B&H (reference) | -59.9 % | -62.5 % |


In-market share 2022: 24.9 %.


## Pass bar (frozen) - VAL, after TRAIN same-signed

| item | pass | evidence |
|---|---|---|
| 1a Excess>0 in >=3 of 4 TRAIN years and in VAL | FAIL | TRAIN 2 of 4 yrs positive; VAL -42.1 bps (VAL years positive 1 of 3) |
| 1b VAL excess t >= 2 | FAIL | t = -0.88 |
| 2 VAL excess > N2 95th pct | FAIL | percentile 40.7 |
| 3 2022: DD <= QQQ DD and return > QQQ | FAIL | DD -43.2 vs -36.6 %; ret -37.6 vs -33.8 % |
| 4 ex-top-5 % excess > 0 and winner-capped > 0 | FAIL | -104.9 / -110.5 bps |
| 5 fills obtainable (next-bar open) | PASS | by construction |


## VERDICT: **REJECT** (TEST remains sealed; no variants, no refit).


## Caveats about this implementation
- **N1 definition.** The task text defines N1 as QQQ 1x return over the same bars with excess = strategy - N1; that
  compares a 2x instrument with a 1x one and is dominated by leverage. The pass-bar EXCESS here is the
  exposure-matched B&H (hold bars x split mean QLD-proxy per-bar return, bar = open to next open), the PREREG's
  own words. The literal N1 column is reported beside it.
- QLD proxy = 2x per-day price change (daily reset) compounded; no expense ratio, tracking error or financing.
  The 0.5 % threshold at bar 16 is evaluated on the QLD proxy. The 2022 equity curve is bar-level (open to
  next open, 2x per bar without per-day reset), an approximation for drawdown only.
- Volume SMA20 includes the current bar and runs across days. Half-hour volume has a strong time-of-day shape
  (09:30 and 15:30 bars carry far more volume), so even at 1.75x the filter is partly a time-of-day filter.
- Days with < 380 1-min bars (early closes, holes) and days without all 13 half-hour bars are removed from the
  series, so the 30-bar lookback and the 16/60-bar holds span them (bars counted, not time).
- Signal bar i enters at open of i+1; after an exit at the open of bar e the first signal considered is bar e's
  close. Trades are assigned to a split by entry date; trades whose exit would fall beyond the data are dropped.
- N2 draws non-overlapping entries uniformly within each year (same count per year, same exit rule on the QLD
  proxy); random entries are not restricted to signal-like bars (as specified).
- Costs: 1 bp/side charged on the QLD-proxy return (sensitivity 0 and 3 bp shown).
- t statistics are trade-level; trades are non-overlapping but serially correlated through regime, so the SE is
  likely understated.
