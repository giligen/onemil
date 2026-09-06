# Bull-flag day trading — expert dossier (2026-09-06)

Owner: "become the world's number 1 expert on BF. Check entry, exit, TP, early
exit, everything. Find in the simulation configs that work. Every day there
are BFs that actually work." This dossier = (A) the practitioner and
quantitative rulebook from the sweep, (B) how our stack maps to it, (C) what
the simulation says today, (D) the consistency model and the build.

## A. The rulebook (sources at the end)

### A1. Universe / stock selection — every credible source agrees
| Rule | Practitioner (Warrior/BBT/others) | Ours |
|---|---|---|
| Price | $2–$20 (some to $30) | $1–30 cache screen; tiers $2–20 live |
| Day move | ≥ 10% up on the day (gap or intraday) | 10% cache screen, 20% Stage-2 threshold |
| Relative volume | ≥ 5× 50-day average (BBT: 2× minimum) | prev-day / 20-day volume tiers, no RVOL-5× gate |
| Float | < 10M (Warrior) / < 20M (BBT, Bullish Bears) | float_max in universe builder |
| Catalyst | breaking news required | none in the BF stack (ORB has it) |
| Time | first 30–90 min; fade after 11:00; "trade almost exclusively the first two hours" | skip_midday 11:30–14:00; entries allowed to 11:30 and after 14:00 |
| Names/day | 1–3 "stocks in play", traded repeatedly | 1 trade per symbol-day (Stage-1 early-exit-after-trade) |

### A2. Pattern definition
- Pole: sharp, high-volume, "near-vertical" impulse of 3+ green 1-min candles; the high-tight variant (pole near vertical, flag tight) is the only flag with a strong record (Bulkowski: 85% success / +39% vs loose flags 45% / +9%, n=1,028; daily bars, but the shape logic transfers).
- Flag: 2–5 red/small candles (intraday), retrace 20–50% of the pole (best 20–38%), declining volume, no free-fall; loses the flag low = dead. LuxAlgo detector defaults: pole ≥ 3 ATR14 within ≤ 10 bars, 60% strong closes, flag 4–15 bars, depth ≤ 50% pole, tilt ≤ 10% pole, bar-range contraction required.
- Only the first or second flag on the stock; third-leg flags fail (momentum waning); flags after an exhausted parabolic move are distribution.

### A3. Entry trigger
- Warrior: "the first candle to make a new high" after 2–3 red candles (1-min); BBT/others: break of the flag high / prior candle high, prefer a candle CLOSE above the level, never the first wick; alternative: break + retest.
- Breakout volume must expand vs the flag (thin-volume breaks are "the most common trap").
- Never chase a spike; never buy inside the flag; avoid entering into daily resistance / half- and whole-dollar levels just above.

### A4. Stop
- Low of the pullback (flag low), or the last higher low inside the flag (tighter); sized off the stop distance; ATR-aware buffer on volatile names; "if it loses the flag low, the setup is dead"; failed follow-through = break of the breakout candle's low.

### A5. Targets / partials / trailing — the consistency machinery
- First target: high of day / measured move (pole height from breakout) / 2:1 to 3:1.
- **Sell half at the first target, stop to breakeven on the rest** (Warrior, BBT, Bullish Bears, TradingSim, all of them).
- Runner: trail under higher lows or the 9 EMA on 1-min (first candle close below the 9 EMA = out); scale again at the next level; exit on "lost momentum" (first red candle after an extension, VWAP loss).
- Quant evidence on partials: scaled exits raise WR (38 → 54%, 42 → 61% in the cited backtests), cut MDD (14.2 → 8.5%), profit factor up (1.45 → 1.62), total return down (22 → 18.5%). Exactly the trade the owner wants.

### A6. Sizing and daily discipline (where "every day" actually comes from)
- Warrior 51-day sample: 936 trades, WR 71.4%, avg win $1,800 / avg loss $761 (≈ 3:1 with the partials), winners held ~3 min, losers ~2 min. This is a scalp cadence — many trades per day on 1–3 names — not a hold-to-EOD book.
- Cushion sizing: quarter size until a quarter of the daily goal is banked; full size only on a cushion; cut size when the cushion is lost; daily max loss ends the session; no setup in 30 min → stop.
- Consistency in that world is a function of trade COUNT (law of large numbers per day) plus quick partials plus a hard daily stop; not of holding for the monster.

## B. Our stack vs the rulebook — the gaps that matter
1. **Cadence**: we take one long-hold trade per symbol-day (trail to EOD, 1.75R lock…), ~4 trades/month after the filter stack. The rulebook trades the same strong name repeatedly with quick partials. Our Stage-1 stops scanning a symbol after its first trade (`early_exit_after_trade`), so the cache cannot even show the re-entries; `BT_ALLOW_REENTRY=1` only lifts it at Stage-1 build time.
2. **Exit profile**: no partial at +N R in the unified spec (the BT `partial_profit` branch is a legacy path: fill-R, fixed target, no trail, no vol-guard; live has no +N R partial at all — only the exhaustion partial at 3R).
3. **Time window**: rulebook says first 60–90 min; ours allows to 11:30. In 2026 the late-entry tercile is the worst (−0.22R).
4. **Catalyst/RVOL-5×**: not in the BF stack.
5. **Flag quality**: our detector already encodes pole gain, retracement ≤ 50%, pullback candles, breakout volume ratio; the univariate read on 638 live-universe trades shows NO causal flag-quality feature that separates winners consistently across 2025H1/2025H2/2026 (only the EOD range does, which is lookahead). The edge is not in "better flags"; it is in cadence and management.

## C. What the simulation says (regen-7, live universe rule, faithful resim)
- Exit-only grid (9 profiles): no profile clears the consistency bar; green months stay 13/20 in every variant (4 trades/month = coin-flip months). Partial 50% at +1.5R + BE (legacy branch) is the best shape: 2026 +$6.2K, total $105K, top-5 share 65% (vs 87%).
- Capacity (positions, daily loss, buying power) is not the cap (+3 trades in 20 months).
- Entry window ≤ 60 min: WR 58.6%, 2026 +$2.8K with the partial profile; no change in month count.
- Re-entry at Stage-2: no effect (Stage-1 never produced the second trades).
- => The month count can only move with more trades per day. Hence the June-2026 lab: Stage-1 with re-entry, quick exits (partial 1R + BE + no-pop), and a looser detector (2-candle pole, 5-candle pullback), measured on trades/day, WR, green days.

## D. The consistency model to build (proposal)
"Scalp-BF": rulebook cadence on our infrastructure.
1. Entries only 9:30–11:00 (first 90 min), A+ names only (add RVOL-5× and a catalyst flag from the ORB news layer), 1–3 names/day, re-entry allowed on the same name while it holds VWAP and the day's high structure (max 2–3 flags/name/day — BBT trades "only 2 bull flags in one stock").
2. Exit = partial 50% at +1R (plan-R), stop to breakeven, remainder: trail per the unified spec (or a fixed +2.5R target — both to be tested), no-pop exit at 10 bars, hard EOD flat. Implemented ONCE on the shared trail spec: BT `simulate` main loop (not the legacy branch) and live (StopMonitor `execute_partial_exit` already exists for the exhaustion rule; add a resting limit for the partial leg so BT's touch-fill and live's fill agree), parity test = one tape both sides.
3. Sizing: risk per trade small enough that the month is the sum of many ±1R outcomes (the plan's $500–1K), daily max loss −3R, cushion rule.
4. Evidence bar (pre-committed): green days ≥ 60%, green months ≥ 70%, worst month ≥ −4× mean month, 2025 and 2026 both positive, walk-forward (fit 2025 → read 2026 and reverse). Shadow live 10 sessions before it trades.

## Sources
Warrior Trading (bull flag guide; momentum strategy; 51-day challenge stats via MyOptionsJournal), Bear Bull Traders (Aziz: ABCD, 9/20 EMA + VWAP, 2 flags per stock, first two hours), Bullish Bears (float < 20M, RVOL 2×, sell half, first/second consolidation only), TradeMomentum (valid flag vs trap: no pole = no edge, mid-flag entries, thin-volume breaks, fakeouts), TradingSim (entry variants, 2–3% stop buffer, partial 75% of measured move, 9 EMA trail, invalidation = break of breakout-candle low), DayTradingZ (measured move, first-hour candidates), LuxAlgo detector defaults, Bulkowski / LiberatedStockTrader (high-tight 85% / loose 45%, n=1,028), QuantStrategy.io (scaled-exit backtests: WR +16pp, MDD −5.7pp, return −3.5pp), SmallCapLab (gap-up fade rates; page blocked, cited from the search summary), ORB sweep sources for the time-of-day evidence.
