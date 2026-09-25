# Route E2 — Leveraged-ETF Decay Harvest (short both legs, dollar-neutral)

Owner ask: build a passive, low-touch income sleeve by shorting both legs of a leveraged
ETF pair (harvesting volatility drag from daily rebalancing) plus a UVXY-only short
(VIX-product roll decay). Real backtest, real Alpaca data, real account constraints below.
**Headline: the mechanism is real and has a positive long-run mean, but (1) three of the
four requested pairs cannot currently be executed on this account because the inverse leg
is not shortable at Alpaca, and (2) the one thing with the best historical mean (SOXL/SOXS)
also has the worst tail — a real April-2026 month lost $32K on a $65K account at the
requested sizing. This is not free money and it is not fully "passive" as specified.**

## 1. Data & method

- Source: Alpaca `StockHistoricalDataClient`, feed=SIP, timeframe=Day, **`adjustment=ALL`**
  (split *and* dividend back-adjusted). Raw/unadjusted bars were tried first and rejected —
  leveraged/inverse ETFs do frequent reverse splits (SQQQ, SPXS, SOXL, SOXS, TZA, UVXY all
  have several); unadjusted closes fabricate a huge fake one-day return on the split date
  (CLAUDE.md price-scale-check rule). Verified: with `ALL` adjustment the largest daily
  moves in the series land on real, named macro days (2020-03-16 COVID, 2024-08-05 vol
  spike, 2025-04-09 tariff reversal, 2026-04 semis rally) — not on arbitrary dates, which is
  what a missed-split artifact looks like.
- Range: 2019-01-01 → 2026-09-24. SVIX inception 2022-03-30 (1,126 bars); everything else
  1,943 bars (11/11 symbols fetched clean, no gaps).
- Symbols: TQQQ/SQQQ, SPXL/SPXS, SOXL/SOXS, TNA/TZA (pairs); UVXY (short-only); SVXY, SVIX
  (long reference — these are *not* shorted, they're the "just buy the short-vol product
  instead" comparison).
- Mechanism: at each rebalance date, short equal dollar notional of both legs (gross short
  notional = leg_A + leg_B), hold to the next rebalance, mark-to-market on close-to-close
  adjusted return, then reset to equal-dollar-neutral again (full rebalance, both legs
  every period). Two cadences tested: **monthly** (calendar month-end) and **weekly**
  (Friday close). Position sizing always uses **today's dollar cap**, never a historical
  price level — back-adjusted historical closes for names with big reverse-split histories
  (SQQQ shows a fictitious $31,542 in Jan-2019, SOXS $25.7M, UVXY $194,800) are valid for
  *returns only*, never for implying a historical share count or notional.
- Costs: borrow 2%/yr on notional held (UVXY/short-vol names 5%/yr — harder to borrow),
  accrued on actual calendar days between rebalances; transaction cost 5 bps per leg per
  rebalance touch (both the reset trade and the un-wind are folded into one 5 bps/leg
  charge per period, a simplifying assumption — see §5).
- Caps: pairs capped at **gross short notional = 1.0× equity = $65,083** (so $32,541/leg);
  UVXY-only capped at **15% of equity = $9,762**. This is *half* of the 2× overnight margin
  Alpaca actually allows on this account (4× intraday / 2× overnight, confirmed via
  `TradingClient.get_account()` today: equity $65,083.14, multiplier 4, shorting_enabled
  True) — i.e. the requested cap already carries 2× headroom against a margin call from
  cap alone.
- All series and the code: `research/passive_income/E2_<pair>_<M|W>.csv`, fetch script and
  simulator kept in the session scratchpad (not committed to the repo per the run rules).

## 2. CRITICAL — obtainability check: half of this is not tradable today

Per CLAUDE.md's obtainability rule (a position must be reachable by an order the engine
could actually place), I queried Alpaca's live asset flags for every symbol
(`TradingClient.get_asset`, today 2026-09-25):

| Symbol | shortable | easy_to_borrow |
|---|---|---|
| TQQQ | **True** | True |
| SQQQ | **False** | False |
| SPXL | **True** | True |
| SPXS | **True** | True |
| SOXL | **True** | True |
| SOXS | **False** | False |
| TNA | **False** | False |
| TZA | **False** | False |
| UVXY | **True** | True |
| SVXY | False | False |
| SVIX | False | False |

Of the four requested pairs, only **SPXL/SPXS** has both legs shortable right now.
TQQQ/SQQQ, SOXL/SOXS and TNA/TZA all fail on the inverse leg — you cannot get short SQQQ,
SOXS or TZA at Alpaca today, so the dollar-neutral pair as specified cannot be built for
three of the four candidates, full stop. (SVXY/SVIX not being shortable doesn't matter —
they were only ever a long reference in this design, never a short leg.) UVXY-only *is*
shortable. This is a locate snapshot, not a permanent fact — borrow availability moves
day to day — but it is the real, current state of the only broker on this stack, and it
inverts the "best candidate": **the pair with the best backtested mean (SOXL/SOXS) is the
one you cannot actually put on.**

## 3. Results — all pairs, both cadences (net of borrow + txn, full cap)

| Series | n periods | mean/period | median | P10 | P90 | skew | worst period | maxDD |
|---|---|---|---|---|---|---|---|---|
| TQQQ/SQQQ monthly | 92 | $179 | $21 | -$1,680 | $1,849 | +4.3 | -$5,553 | -$10,446 |
| TQQQ/SQQQ weekly | 403 | -$1 | -$45 | -$357 | $327 | +6.3 | -$1,929 | -$15,058 |
| **SPXL/SPXS monthly (only fully-shortable pair)** | 92 | **$36** | -$95 | -$971 | $598 | +6.7 | -$2,563 | -$12,401 |
| SPXL/SPXS weekly | 403 | -$48 | -$65 | -$240 | $115 | +6.6 | -$3,083 | -$22,791 |
| SOXL/SOXS monthly (best mean, **not shortable**) | 92 | $538 | $649 | -$3,550 | $4,185 | -0.7 | **-$32,162** (Apr-2026) | -$40,467 |
| SOXL/SOXS weekly (not shortable) | 403 | $133 | $67 | -$724 | $957 | +3.4 | -$4,153 | -$9,607 |
| TNA/TZA monthly (not shortable) | 92 | $209 | $161 | -$1,381 | $1,574 | +3.1 | -$6,737 | -$13,907 |
| TNA/TZA weekly (not shortable) | 403 | -$42 | -$23 | -$387 | $298 | +0.7 | -$5,768 | -$18,976 |
| UVXY-only monthly (shortable, 15% cap) | 92 | $578 | $1,158 | -$2,384 | $2,189 | -2.7 | **-$15,217** (Mar-2020) | -$22,318 |
| UVXY-only weekly (shortable, 15% cap) | 403 | $124 | $357 | -$1,211 | $1,307 | -2.1 | -$8,392 | -$27,720 |
| SVXY long reference (buy & hold, 15% cap) | 92 | $143 | $273 | -$895 | $1,119 | -1.0 | -$3,799 (Mar-2020) | -$6,068 |
| SVIX long reference (2022-on, 15% cap) | 54 | $247 | $237 | -$1,638 | $2,120 | -0.2 | -$3,821 (Apr-2025) | -$10,798 |

P&L decomposition (sum over full backtest, monthly cadence, full cap): SOXL/SOXS —
price/decay+trend +$62,495, borrow -$9,960, txn -$2,994 → the edge is almost entirely
**trend contribution, not pure decay** (see §5). UVXY-only — price +$57,401, borrow
-$3,735, txn -$449: same story, most of the "decay" number is really the multi-year
structural bleed of VIX futures contango, which is a trend, not a stationary harvest.

## 4. The tail, in real months that actually happened

| Month | TQQQ/SQQQ | SPXL/SPXS | SOXL/SOXS | TNA/TZA | UVXY-only | SVXY (long) | SVIX (long) |
|---|---|---|---|---|---|---|---|
| Mar-2020 | +$17,664 | +$16,857 | +$26,554 | +$14,286 | **-$15,217** | -$3,799 | n/a |
| 2022 H1 (sum) | +$5,690 | +$2,752 | +$7,841 | +$4,273 | -$3,288 | -$2,215 | -$2,515 |
| Aug-2024 | +$1,182 | +$444 | +$5,825 | +$1,595 | +$1,267 | -$900 | -$2,602 |
| Apr-2025 | +$7,417 | +$5,223 | +$18,957 | +$5,076 | -$2,891 | -$1,786 | -$3,821 |
| **Apr-2026 (worst on record, found in this run)** | -$3,169* | -$2,563 | **-$32,162** | -$1,674* | +$3,200* | n/a | n/a |

*not a top-3 worst for that series but shown for the same calendar month; SOXL/SOXS's
-$32,162 is that pair's single worst month in the whole 92-month sample.

The pairs made money in every crash/vol-spike month shown (they're short realized-vol
convexity on the *pair*, so a sharp move that hits one leg hard is offset by the other) —
**except** when one side runs in an extreme, sustained, single direction for a full
month. SOXL went from $47.88 to $126.90 (+165%) in April 2026 on a semis rally; SOXS only
had -66.6% to give back (bounded near -100% for an inverse fund but nowhere near enough
to offset a +165% move on 3× leverage) → the short-both-legs structure lost on both legs
at once. **Re-run at weekly cadence, the same April 2026 event cost -$6,107 instead of
-$32,162** (three moderate bad weeks instead of one catastrophic month) — cadence is the
single biggest lever on tail size, at cost of higher transaction drag (SOXL/SOXS weekly
mean $133/wk ≈ $576/mo annualized, close to the monthly mean $538/mo, for roughly 1/5 the
tail).

UVXY-only lost hardest exactly where you'd expect — Mar-2020 (VIX spike), 2022 H1
(persistent vol), Apr-2025 (tariff shock) — the classic **short-volatility negative-skew
signature** (skew -2.7 monthly, -2.1 weekly): most months clip a modest, steady premium,
a few months give back several months' worth at once. This is the same structural risk
that terminated XIV and forced SVXY's leverage cut in the well-documented Feb-2018
"Volmageddon" event (public record, not re-derived here — flagged as background, not a
backtested number).

## 5. Skew, mechanism honesty, and cost caveats

- **Decay vs. trend**: the requested framing ("harvest volatility drag") implies a
  stationary, regime-independent edge. The P&L decomposition says otherwise — most of the
  positive mean comes from *trending* periods where realized vol was itself elevated
  (2020, 2022, 2025), and the single realized loss events (Apr-2026 SOXL/SOXS, Mar-2020
  UVXY) are also trend-driven. This is closer to **short realized-volatility exposure**
  than to a clean, path-independent decay harvest. It will not behave like an interest
  coupon; it behaves like a short-vol position that mostly pays and occasionally doesn't.
- **Borrow cost is a placeholder, not a quote.** 2%/yr and 5%/yr are assumptions from the
  task, not pulled from Alpaca's live short-locate/margin API (not queried in this pass —
  time-boxed). Real borrow on high-demand leveraged/inverse names has historically spiked
  far above 2%/yr during crowded periods; SQQQ/SOXS/TZA showing `easy_to_borrow=False`
  right now is itself evidence that a "2%" assumption could be optimistic if/when they
  ever do become shortable again. **This must be checked against Alpaca's real margin
  disclosure before sizing a dollar of this live.**
- **Margin schedule unverified.** The 1.0×/15% caps satisfy the requested "2× overnight
  margin" rule using standard maintenance-margin math, but brokers frequently apply
  *higher* maintenance requirements specifically on leveraged-ETF shorts (150–300%+ is
  common industry practice, not the plain 50% used here). Not verified against Alpaca's
  actual house margin schedule for these specific symbols in this pass — flagged as an
  open item, not resolved.
- **Sample size / regime**: this is one continuous 7.7-year draw (92 months / 403 weeks),
  not independent trials — the well-behaved months cluster in 2019/2023/calm-2024 and the
  bad months cluster around known vol events. Treat the mean as a single noisy draw from a
  fat-tailed process, not a converged expectation.
- **No independent reimplementation, no causality trace, no PREREG** — this is a first-pass
  single-agent research answer produced under a 40-tool-call/sub-90-minute budget, not a
  claim cleared for capital under CLAUDE.md's "no research claim ships without an
  independent check" rule. Treat every number above as "this is what one honest backtest
  said," not as a cleared verdict.

## 6. Verdict

**Is this passive on our stack? Partially, and only for a subset of what was asked.**

- SOXL/SOXS (the best historical mean, weekly cadence, tamed tail) **cannot be executed**
  — SOXS is not shortable at Alpaca today. Same for TQQQ/SQQQ (SQQQ not shortable) and
  TNA/TZA (TZA not shortable).
- SPXL/SPXS — the *only* fully executable pair — shows ~breakeven-to-slightly-positive
  mean monthly ($36/mo) and **negative** mean weekly (-$48/wk) at full $65K cap, net of
  costs. Not a source of "several $1,000s/month." Not recommended as a standalone sleeve.
- UVXY-only short **is executable** (shortable=True), has the highest of the confirmed-
  tradable means ($578/mo monthly cadence, $124/wk weekly, at the specified 15% cap), but
  carries the worst confirmed skew (-2.7) and a real -$15,217 single month (Mar-2020) —
  at 15% cap that's -156% of the capital allocated to the trade, ~-23% of total account
  equity, in one month. This is the most promising currently-tradable leg but it is a
  short-volatility bet with fat left-tail risk, not a decay coupon.
- Operationally, weekly-rebalance is a cron/systemd timer away from "low touch" — genuinely
  buildable on the existing stack without new infrastructure. But "low touch" is not the
  same as "safe": nothing here removes the need to watch for a running month like Apr-2026
  or Mar-2020.

**Skew and tail, plainly**: every series here is negative-skew or fat-tailed in the sense
that matters — the wins are small-and-frequent(ish), the losses are rare-and-large, and
the two largest losses found (SOXL/SOXS -$32,162, UVXY -$15,217) are each a meaningful
fraction of this $65K account in a *single* period. This is **beta to realized volatility
of the underlying**, dressed as "decay harvest." It is not free, uncorrelated income.

## 7. First experiment (bounded, cheap, fast)

1. **Paper-trade UVXY short only**, weekly rebalance, $9,762 cap, on the existing Alpaca
   paper account — $0 cost, 4 weeks to see the mechanism and realized (paper) borrow/fee
   behavior before touching real money.
2. In parallel, pull Alpaca's actual margin-requirement schedule and short-locate fee for
   UVXY and SPXL/SPXS (a documentation/API check, not a backtest) — 1 day, $0 — to replace
   the placeholder 2%/5% borrow assumption with a real number before any live sizing.
3. Re-poll `shortable`/`easy_to_borrow` for SQQQ/SOXS/TZA/SVXY/SVIX daily for two weeks to
   see whether today's non-shortable snapshot is structural or transient; only revisit the
   SOXL/SOXS candidate if SOXS locate opens back up.
4. Do not size real capital until (1)-(3) close and this clears the independent-check
   protocol (reimplementation + causality trace) required before any research number goes
   live per CLAUDE.md.

## 8. Files

- `research/passive_income/E2_TQQQ_SQQQ_M.csv`, `_W.csv`
- `research/passive_income/E2_SPXL_SPXS_M.csv`, `_W.csv`
- `research/passive_income/E2_SOXL_SOXS_M.csv`, `_W.csv`
- `research/passive_income/E2_TNA_TZA_M.csv`, `_W.csv`
- `research/passive_income/E2_UVXY_only_M.csv`, `_W.csv`
- `research/passive_income/E2_SVXY_long_ref_M.csv`, `E2_SVIX_long_ref_M.csv`

Each CSV: `period_end, days, pnl_price, pnl_borrow, pnl_txn, pnl_net` (dollars).
