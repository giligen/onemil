# PREREG — cell 1,720: "QLD Claude AI Scalper" (TrendSpider share, owner 2026-10-30) replayed out of its sample

Frozen before any number. Rejection-only Stage 1 on free data already on disk. No paid pulls.

## The published claim
QLD (2× QQQ) long-only, 30-min bars, tested Jan 2025 → Oct 29 2026 (one bull leg; QLD B&H +105 %): strategy +127 %,
62 % winners, avg +1.24 %/trade, max DD −12.3 %, RR 2.23. Entry (all of): 30-min close > highest high of the prior
30 bars (`RANGE(30,bars).high (1 candle ago)`); bar volume ≥ 20-bar SMA volume **by at least 75 %, i.e. volume ≥ 1.75 ×
SMA20** (Amendment 1, 2026-10-30 20:59, from the owner's second screenshot which completed the truncated text — the
published rule, not a fit; frozen before any number was read); close > open. Exit: after 16 bars if PnL < +0.5 %, else after 60 bars.
No stop. "Trade by: open" = fill at the NEXT 30-min bar's open.

## Our prior
Index ORB (18 cells, 0 pass) and SPY/QQQ first→last half-hour momentum (1,714–1,716, reversed below cost) are closed.
A 30-bar (≈ 2.3-day) breakout with a 1–5 day time exit on a 2× bull-market ETF is expected to be BETA: the right null is
exposure-matched buy-and-hold, not 100 % B&H.

## Data
`research/index_orb/cache/QQQ_1min.parquet` (Alpaca SIP regular session 2016-01-04 → 2026-xx), resampled to 30-min bars
09:30–16:00 ET (13 bars/day; the last bar 15:30–16:00). Exclude days with < 380 1-min bars and early closes (the 1,714
precedent). QLD proxy = QQQ 30-min returns × 2 within a day, compounded per trade (QLD's daily reset is immaterial at a
1–5 day hold; report QQQ 1× beside it). Cost: 1 bp/side on QQQ (QLD spread ≈ 2 bps; sensitivity 0 and 3 bps).

## Splits
TRAIN 2016-01 → 2019-12, VAL 2020-01 → 2022-12, TEST 2023-01 → 2026-xx SEALED (not read until VAL passes). The
published window (2025–26) sits inside TEST and is NOT evidence.

## Rules replayed exactly as published (one cell, no variants)
Signal at bar t close; entry at bar t+1 open; one position at a time (TrendSpider default); exit at the open of the bar
after the 16th bar when the trade's PnL at the 16th close < +0.5 %, otherwise at the open of the bar after the 60th.
Session-only bars; a hold runs across days.

## Nulls
N1 exposure-matched B&H: QQQ return over the SAME in-market bars (the strategy's excess = its return − N1).
N2 random entry, same hold rule, same number of trades per year, 1,000 draws → percentile of the realised excess.

## Pass bar (all on VAL, after TRAIN same-signed)
1. Excess over N1 > 0 per year in ≥ 3 of 4 TRAIN years AND in VAL; VAL excess t ≥ 2 (trade-clustered).
2. VAL excess above the 95th percentile of N2.
3. 2022 (bear year inside VAL): strategy max DD ≤ QQQ B&H max DD AND strategy 2022 return > QQQ 2022 return.
4. Ex-top-5 % trades excess > 0; winner-capped (+3 %) excess > 0.
5. Fills obtainable (next-bar open) — by construction.
Fail any → REJECT, TEST stays sealed, no variants, no refit of 30/16/60/0.5 %.

## Report
`RESULT_1720.md`: trade count, in-market share, gross/net return, excess vs N1 per year, N2 percentile, 2022 line, tail
lines, flags (|t| > 6 on any year, any trade > +50 % / < −30 %), MDE at the VAL trade count.
