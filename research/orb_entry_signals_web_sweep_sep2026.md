# ORB entry signals & parameters — web/social sweep (2026-09-05)

Owner ask: "exhaustive search for relatively new params / entry signals for
ORB — don't validate, comprehensive research on everything out there."
~40 sources read (academic, quant blogs, data vendors, TradingView scripts,
retail guides, X/Telegram/Discord). Nothing here is validated on our book;
section 5 ranks what is worth testing on our pipeline and maps each item to
features we already compute.

## 1. Sources with real numbers

### 1a. Zarattini / Barbon / Aziz (Concretum, SSRN 4729284, rev. Apr 2025) — the canonical stock-ORB paper
- 7,000+ US stocks, 2016–2023. Plain 5-min ORB: +29% total, Sharpe 0.48. **Restricting to "Stocks in Play" — top-20 by relative volume of the first 5 minutes vs the 14-day average of that same 5-minute window (RV ≥ 1.0) — gives +1,637%, 41.6%/yr, Sharpe 2.81, MDD 12%.** "Abnormal opening relative volume did almost all the work."
- Entry: stop order beyond the 5-min high (long only if first candle close > open). Stop: **10% of 14-day ATR from entry** (tight!). Exit EOD. 1% risk, 4× leverage cap. 5-min range beat 15/30/60.
- QuantConnect replica: universe price > $5, 1,000 most liquid, ATR > $0.50; 68% of 25 parameter combos beat the benchmark Sharpe (2016 sample).
- Companion paper (Zarattini & Aziz 2025, QQQ/TQQQ): enter at 9:35 in the direction of the first 5-min candle, stop at the candle's opposite extreme, 10R target / EOD. WR 24%, +0.13R/trade, Sharpe 1.1–1.2; the extreme variant (5% ATR stops, no target) assumes zero slippage.
- Maróy (SSRN 5095349, Jan 2025): parameter optimisation + alternative exits for the intraday-momentum family (paper not fetchable; cited as improving exits).

### 1b. ORB Setups — 190,460 setups / 611 US stocks+ETFs, 30-day window (2026)
- Per-symbol WR: 5-min 52.2% (0.028R), 15-min 52.5% (0.004R), 30-min 52.9% (0.019R). Realized target-hit WR 35.1% overall.
- **Longs 37.4% vs shorts 32.9%** (n=93K/97K).
- **Range width (stocks): tight < $0.50 → 51.0% WR (n=45,582); $0.50–1.00 → 40.9%; $1–2 → 40.4%; > $2 → 34.6%.** Opposite sign to the index-futures finding below.
- Baseline false-breakout (stop-hit) rate 65.9%; 5-min 54.8%, 15-min 68.8%, 30-min 75.8%.
- Time of day: 9-AM breakouts 34.0% WR, 10-AM 29.9% (worst), 3-PM 51.7%.
- Half-target/full-stop 67.8% WR vs full-target/half-stop 41.8% WR but highest expectancy.
- Symbol dispersion is huge (ZS 78% vs KTOS 27%): symbol selection dominates.

### 1c. TradingStats — 6,142 days ES/NQ, 2014–Jan 2026 (index futures, not stocks)
- Continuation (first break = session direction): 5-min close confirmation adds 6–10pp over wick entry (ES 30m: 64.6% wick → 70.7% 5-min close).
- **Wide ORB (> 0.6× ATR14) → 77.5% continuation** (trend days, not mean-reversion). Narrow (< 0.3× ATR) → 53% double-break.
- **ORB internal direction (range candle closed up) is "the single strongest filter": 61.7% vs 55.3% (ES 5m), 70.3% vs 63.1% (NQ 30m).**
- Upside breaks beat downside by 8–10pp. Gap direction: no edge (58.6% vs 61.1%); flat opens chop most.
- ATR regime: no effect on continuation (within 1.7pp). Weekday: Monday cleanest, Wednesday worst.
- Combo setups: wide ORB + range-up = 83.9% (n=56); Monday + range-up 72% (n=339); anti-setup Wednesday + flat gap 51%.
- Double-break: second direction wins 55–72%; false breaks cascade to a double 80–99%.
- Year-over-year stable 2014–2025 (no decay) while median ORB width grew 4–7×.

### 1d. NQ market-by-order study (Sep 2025–Apr 2026, 159 days, tick MBO data)
- Baseline 71.1% reach 0.5× the filter range; 51.6% at 1.0×; 25% at 2.0×.
- **Opening order-flow delta (aggressive buy − sell, normalised) positive → +7.5pp** at 0.5× (75.0% vs 67.5%).
- **Strongest warning: breakout that reverses through the range midpoint before reaching 0.5× → continuation drops 71% → 22.7%** (n=44).
- **Second-episode breakouts (after a failed first) → 76.5%**, better than baseline.
- Faded with more data: breakout-candle volume tertiles (no edge) and extreme breakout-candle delta.
- Volume-profile context: breakouts from inside the opening value area extend more (47.6% at 1.5×) than from below it (17.6%).

### 1e. Edgeful (ES 15-min, 6-month): breakout 16.9% / breakdown 16.2% / double 66.9%; upside 0.5× reached 72%, 1.0× 51%; "by-retracement" sub-report for pullback entries.

### 1f. Traders Mastermind (QQQ, 15-min, 2 yrs): stop 100% of range → 40% WR; 200% of range = "sweet spot"; **NR7-day filter → 56% WR, 1.6 R:R** at much lower frequency; entry window to 11:00 beat 10:00.

### 1g. arXiv 2605.04004 (MNQ, 2021–2025, 14 signal families incl. ORB immediate/pullback/delayed): **none of the ORB variants survived a 2-point friction test**; only two non-ORB signals passed. A useful negative result for index futures.

## 2. What the 2026 practitioner consensus says (no numbers, repeated everywhere)
Candle CLOSE beyond the range (not wick); breakout-bar RVOL ≥ 1.5–2× the range bars' average (claimed +8–12pp WR); price above VWAP with positive slope; skip narrow/choppy mornings (OR/ATR floor); trade with the gap direction; one or two attempts per session; entries only 9:30–11:00; ATR-scaled stops; skip major-news minutes; "ORB is the most crowded intraday strategy of 2026 — unfiltered no longer works". Indicator soup on top: MFI, SuperTrend, Vortex, EMA stack, FVG/ICT confluence, LinReg slope + Williams %R (ORB-FVG Telegram bot script). 1-minute ORB on "A+ gappers" (gap ≥ 2%, PM volume > 100K, price > $5, stop at candle-1 low). Bulls-on-Wall-Street "3-candle rule": entry only after a 3-bar consolidation with a tight lid, RVOL ≥ 2, catalyst required, stop at VWAP.

## 3. Social / bot channels
No concrete, reproducible "new criterion" surfaced from X/Telegram/Discord/WhatsApp searches — the public bot ecosystem is crypto-dominated; ORB bot posts are TradingView scripts (LuciTech, ORBDD, Break-Retest, ORB-FVG v6.1) whose "improvements" are the filters in §2. If the WhatsApp group's rule is specific (a threshold, a feature), get the exact wording — it is almost certainly one of §5's items in disguise.

## 4. Where our ORB already stands (so we don't re-test what we have)
5-min range, pre-placed stop-limit at range_high + 30 bps (touch entry, no close confirmation), long-only gap-ups (gap ≥ 5%, prev-day vol ≥ 500K, $3–30), 7-feature composite z-score ranking (range_size_pct, range_total_volume, range_close_position, bars_green_in_range, last_bar_green, range_vwap_distance_pct, gap/prev-day/20d/SPY features), Q1 filter, PDR veto (prev-day range > 8% — day-2 of fireworks), catalyst veto (news or complex confirmation), news-gated PM$ sizing mult, touchgo Rule M (breakout-bar close in the bottom half → exit) + Rule D (first bar ≥ 0.75R adverse → exit), static lock 1.75R → +0.5R, no target, 15:45 flat, 60-min auto-cancel, max 4 slots, quintile sizing with the Q5 cap.

## 5. Candidate list, ranked for OUR book (untested; each is a single-knob A/B on the entered-inclusive features)

| # | Candidate | Source / evidence | What we have | Test shape |
|---|---|---|---|---|
| 1 | **Opening relative volume vs own history**: vol(9:30–9:35) / mean of the same 5-min window over the prior 14 days; rank/veto by it | Zarattini SIP: the dominant driver (Sharpe 0.48 → 2.81) | range_total_volume (absolute), prev_day_volume_vs_20d (daily) — NOT the 5-min-vs-own-5-min ratio | add feature `rvol_open5`; test as (a) 8th composite feature, (b) veto < 1.0, (c) rank override |
| 2 | **Range-candle direction / close position** as a hard gate (range closed in upper half) | TradingStats: strongest filter, +6–7pp; NQ MBO agrees | range_close_position, last_bar_green already in the composite (soft) | hard gate vs soft weight; check what the composite already captures |
| 3 | **Midpoint-reversal kill**: post-entry, price back through the range midpoint before +0.5R → exit | NQ MBO: 71% → 22.7% continuation | touchgo M/D cover the first 1–2 bars only; static lock arms at 1.75R | a Rule "R" in orb_touchgo_filter with parity; likely overlaps Rule D |
| 4 | **Second-episode (re-)breakout** after a failed first break | NQ MBO 76.5% vs 71%; TradingStats double-break dynamics | we never refill / re-enter; PDR/catalyst refill was toxic (MDD) | entry-only variant: allow one re-arm of the stop-limit after a tag exit, same day, same slot |
| 5 | **ORB width vs ATR14 tier** | futures: wide = trend (77–84%); stocks (ORB Setups): tight = better WR — CONFLICTING | range_size_pct in composite; SZ1 ATR floor on the stop | bucket the book by range/ATR14 and read the P&L by tier before any rule |
| 6 | **Close-confirmed entry** (enter on the first 1-min or 5-min close above range_high instead of the touch) | +6–10pp continuation on futures; retail consensus | touch entry + touchgo post-fill | delayed-entry variant; cost = worse fills on runners (our monster concentration argues against) |
| 7 | **Tighter ATR stop** (10% of ATR14 from entry, Zarattini) instead of range_low | paper's core parameter | stop = range_low (typically 3–8% of price) | changes R, sizing, fill rate; big test, high variance |
| 8 | **Premarket-high as the breakout level** (max of range_high, PM high) for gappers | gap-and-go guides; PM$ already matters for us | PM$ sizing mult, no PM-high level | alternative trigger level; needs premarket bars (scripts/orb_premarket_backfill.py exists) |
| 9 | **Weekday context** (Monday best, Wednesday worst on futures) | TradingStats; our day_of_week feature exists | day_of_week in features | read-only cut of the book by weekday; only a rule if era-consistent |
| 10 | **Opening order-flow delta** (aggressive buy − sell in the range) | NQ MBO +7.5pp | we store bid/ask sizes at entry only | needs trade-tape classification in the range window — expensive; park |
| 11 | Breakout-bar volume ≥ 1.5–2× range-bar average | retail consensus; NQ MBO found NO edge at n=159 | range_total_volume | low priority |
| 12 | NR7 prior-day filter | Traders Mastermind (QQQ) | we have the OPPOSITE evidence (PDR veto: quiet prev day is bad for us) | skip unless the RVOL work changes the picture |
| 13 | VWAP-above / slope gate; MFI/SuperTrend/Vortex/FVG | retail | range_vwap_distance_pct in composite | skip the indicator soup |
| 14 | Retest / pullback entry | "highest probability, fewest signals" | fill-limited book already (55% fill) | skip |

Owner's WhatsApp rule: match it to the table before building anything; if it is not on the table, it is new and goes to the top of the test queue.

## 6. Method / caveats
All figures are the sources' own, mostly in-sample, mixed instruments (index futures ≠ small-cap gappers), several with no cost model (Zarattini ETF variant explicitly zero slippage). Two results conflict head-on (range width). Our test discipline stays: pre-committed decision rule, entered-inclusive features, walk-forward eras, MDD and negative-month count, one knob at a time.

Sources: [Zarattini/Barbon/Aziz SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4729284), [danfin summary of the two papers](https://danfin.net/opening-range-breakout-research), [QuantConnect replica](https://www.quantconnect.com/research/18444/opening-range-breakout-for-stocks-in-play/), [Maróy SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5095349), [ORB Setups win-rate study](https://orbsetups.com/research/opening-range-breakout-win-rate/), [ORB Setups false-breakout study](https://orbsetups.com/research/how-to-identify-and-avoid-false-breakouts-a-data-driven-approach/), [ORB Setups gap-and-go](https://orbsetups.com/research/gap-and-go-trading-strategy-how-to-combine-pre-market-gaps-with-opening-range-breakouts/), [TradingStats ES/NQ guide](https://tradingstats.net/orb-breakout-strategy-guide/), [TradingStats context filters](https://tradingstats.net/orb-strategy-research/), [NQ MBO study 2026](https://researchpaperfilteropen.vercel.app/), [arXiv 2605.04004](https://arxiv.org/abs/2605.04004), [Edgeful ORB](https://www.edgeful.com/blog/posts/the-opening-range-breakout-orb-trading-strategy), [Traders Mastermind settings](https://tradersmastermind.com/best-opening-range-breakout-settings/), [BuildAlpha guide](https://www.buildalpha.com/opening-range-breakout/), [Bulls on Wall Street 3-candle rule](https://www.bullsonwallstreet.com/post/how-to-trade-the-opening-range-breakout), [TradingSim 1-minute ORB](https://www.tradingsim.com/blog/1-minute-orb), [ChartingLens 2026 guide](https://chartinglens.com/blog/opening-range-breakout-strategy), [StatsEdge intro](https://letters.statsedgetrading.com/p/exploring-opening-range-breakouts), [ORB-FVG Telegram script](https://my.tradingview.com/script/UrAxU2wE-ORB-FVG-Strategy-with-telegram-V6-1), [LuciTech ORB script](https://www.tradingview.com/script/10yYqaY7-ORB-Strategy-LuciTech/), [Trade That Swing](https://tradethatswing.com/opening-range-breakout-strategy-up-400-this-year/) (403, not read).
