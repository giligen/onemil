# Weekly Cross-Sectional Momentum Sleeve — implementation spec (v1, 2026-10-03)

One rule, long only, US equities, weekly. Backtest and live share the same functions; every number below is from the
backtest 2017-01 → 2026-09 at $50K with 15–20 bp per traded dollar.

## 1. Universe (evaluated at Friday's close, data through Friday)
- US common stocks tradable on Alpaca; exclude by name: ETFs/ETNs, warrants, units, rights, preferreds, SPAC shells,
  test tickers (`^Z[A-Z]ZZT$`), symbols with `.`/`-` class suffixes you cannot trade.
- Prior close ≥ $10. 20-day average dollar volume (close × volume) ≥ $200M.
- At least 273 daily bars of history.

## 2. Hygiene guard (data-quality filter, no look-ahead)
A name is ineligible if, inside its most recent 273 bars, it has
- a one-day close-to-close move > +200 % or < −75 %, or
- a gap > 10 calendar days between consecutive bars.
Purpose: recycled tickers, bankruptcy re-issues, broken adjustments. Effect: 36 of 10,060 name-weeks removed;
CAGR 26.6 → 28.7 %, max DD −42.9 → −38.1 %.

## 3. Score and selection
- Momentum = close[t−21] / close[t−252] − 1  (12-1: skip the most recent month).
- Vol = standard deviation of daily returns over the last 252 bars (not annualised; any constant scaling is fine).
- Score = Momentum / Vol. Rank descending, take the **top 20**.

## 4. Execution (Monday)
- Target weight = 1/20 of current equity for every selected name; **all 20 are reset to target every week**
  (sell what left, buy what entered, trim/add the rest).
- Backtest trades at Monday's official open. Live trades 09:45 ET market orders (measured half-spread 9.5 bp mean,
  12 bp notional-weighted; 09:31 is ~2× worse). Turnover ≈ 45 % of the book per week.
- Fractional notional orders where allowed; whole shares otherwise.

## 5. Calm-week half-size switch (optional; on in our paper run)
- Gate value = VIX / VIX3M at Friday's close (CBOE daily files).
- Percentile = share of the prior 252 values strictly below today's + ½ × share equal; need ≥ 126 observations.
- If percentile < 20 % → every name at 1/40 of equity, remainder cash. Otherwise full size.
- Rationale: the calmest fifth of weeks returned −0.46 %/week for this book. Effect: 28.7 → 30.0 % CAGR,
  −38.1 → −36.7 % max DD. A hard cash gate at any threshold did NOT improve the drawdown — do not use one.

## 6. What to expect
| Book (2017-01 → 2026-09, $50K) | CAGR | Max DD | End equity |
|---|---|---|---|
| Plain top-20 | 26.6 % | −42.9 % | $485K |
| + hygiene guard | 28.7 % | −38.1 % | $569K |
| + calm-week half size | 30.0 % | −36.7 % | $626K |
| SPY | ~15 % | −34 % | — |

Worst episodes are momentum unwinds with the market flat or up (Feb–May 2021 −38 %, Feb–Apr 2025 −34 %, Mar 2020 −37 %).
Tested and rejected as repairs: VIX cash gates, residual momentum, path-smoothness momentum, SPY or loser-leg shorts,
own-volatility scaling, 52-week-high ranking, crypto or low-vol stacks. Position size is the only risk control.

## 7. Operational guards worth copying
- Completeness gate before trading: refuse the rebalance if > 5 % of symbols failed to fetch, if < 98 % of liquid names
  (ADV20 ≥ $200M) have Friday's bar, or if any held/selected name is missing — a stale book beats a half-universe book.
- Idempotent client order ids per (date, symbol, side); persist "rebalanced today" immediately after orders go out.
- Reconcile positions to the broker after every run; log the fill slippage vs the official open per name.
- Stale-gate check: if the latest VIX close is older than the signal date, log it loudly (CBOE files lag a day).

## 8. Pitfalls we hit
- Fetch loops that share a retry path with "invalid symbol" removal silently drop thousands of names — count LOST.
- Marking equity at signal-date closes instead of current prices set a false equity peak.
- Rank ties and the percentile convention must match between backtest and live (we test both against frozen fixtures).
