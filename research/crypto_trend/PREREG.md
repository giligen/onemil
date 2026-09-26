# PREREG — crypto time-series trend on Alpaca (cells C1–C4), FROZEN 2026-09-26 before any number

Owner 9/26: "Find a different population or something else. Make it work." This is the second untested population
(the first, information events, has its own PREREG). Programme count: these are cells 1,470–1,473 on the desk ledger.

## Why this population
Every behavioural equity anomaly reachable with price/volume data has been refuted on this account (WEEKEND_RESULTS.md
9/26 screen). Crypto time-series momentum / trend is the one asset class where retail trend-following had documented
edge through the 2020s (Liu & Tsyvinski 2021 RFS: 1- and 4-week momentum in coins; practitioner trend replications on
BTC/ETH), it trades 24/7 on Alpaca without pattern-day-trade limits, has no survivorship problem for the majors, and a
weekly rebalance keeps the 25 bps taker fee affordable. Mechanism: slow information diffusion plus retail herding in a
market with no fundamental anchor; the risk is that the edge decayed after 2021 — the VAL window decides.

## Data
* Signals and TRAIN: daily closes (UTC) for BTC, ETH, SOL from Binance's public klines (BTCUSDT from 2017-08-17,
  ETHUSDT from 2017-08-17, SOLUSDT from 2020-08-11; no key needed; fetched once into `research/crypto_trend/bars_daily.csv`
  with the source column) and, for 2021-01-01 onward, Alpaca's own daily bars (`CryptoHistoricalDataClient`, the
  executable venue). Report the median absolute close difference Binance vs Alpaca on the overlap (must be < 0.3 %).
* Splits: TRAIN 2017-08 (coin inception) → 2023-12-31; VAL 2024-01-01 → 2025-06-30; TEST 2025-07-01 → 2026-09-26 SEALED
  (read once for the single best cell only if VAL passes).

## Rules (evaluated once per week at Monday 00:00 UTC on the prior daily closes; orders at that time at the taker price)
| cell | rule (long / flat only — Alpaca crypto has no shorting) |
|---|---|
| C1 | trend filter: long the coin while close > its 100-day simple moving average, else flat |
| C2 | Donchian: enter long on a close above the prior 20-day high, exit on a close below the prior 10-day low |
| C3 | 4-week momentum: long while the 28-day return > 0, else flat (Liu-Tsyvinski) |
| C4 | C1 ∧ C3 (both agree), else flat |
Portfolio: equal weight across the coins that are long that week (up to 3), 100 % of the allocated capital when all three
are long, cash otherwise; no leverage. Cost: 30 bps per leg (25 bps taker fee + 5 bps half-spread), charged on every
change of position; slippage beyond that not modelled at weekly frequency (disclosed).

## Pass bar (frozen; on VAL, 78 weeks)
Net weekly return series: mean ≥ +0.35 %/week (≈ +1.5 %/month, the owner's bar at $65K), weekly t ≥ 2.0 (Newey-West,
4 lags), green-week share ≥ 55 % against a count-matched null of random long/flat weeks with the same time-in-market,
max drawdown ≤ 30 %, ex-top-5 % weeks still ≥ 0, and TRAIN same sign with t ≥ 2. Report beside it: buy-and-hold BTC over
the same window (the trend rule must beat it on drawdown AND not lose more than half its return), time in market, number
of round trips, total fees paid. Every cell reported whether it passes or fails.

## Independent check and consequences
A second agent rebuilds each cell's weekly position series from this prose (never reading the builder's code); agreement
≥ 99 % of coin-weeks and VAL mean within 0.05 %/week. Refuters on any passing cell (look-ahead at the weekly boundary —
the Monday close must not be visible; data — Binance vs Alpaca prices, missing days; statistics — regime concentration:
share of the VAL return in the best 4 weeks). PASS → an owner decision (crypto is a new asset class on the account): a
paper run of 8 weeks with the exact orders logged, then a $5K allocation. FAIL → closed; no further crypto cell without a
new mechanism.

## Not allowed
Tuning any lookback; adding coins after a number exists; reading TEST for more than one cell; using intraday data.
