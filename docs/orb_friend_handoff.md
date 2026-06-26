# ORB Strategy Handoff — Architecture, Methodology, and the Traps That Will Bite You

## What this is (and isn't)

This is the **shape** of a working opening-range-breakout (ORB) day-trading
bot, the **method** for fitting it yourself, and — most importantly — the
**list of failures that backtest clean and bleed live.**

What's **deliberately not here**: the fitted constants (quintile cutoffs,
sizing multipliers, filter thresholds, regime boundaries, z-score
parameters). Those are the output of many months of walk-forward
backtesting on a specific data cache. Handing them over would (a) give you
numbers fit to *my* universe that may not hold on yours, and (b) make our
two bots queue for the same shares on the same thin stocks at the same
second. You'll fit your own — that's the part that actually teaches you the
strategy, and it's what keeps us from competing. More on that at the end.

The trap list, by contrast, is the genuinely valuable part and it's here in
full. It's bug-knowledge, not alpha. Every item cost real money to discover.

---

## 1. The architecture (the shape)

**Opening range.** For each candidate, the first few minutes after the open
(e.g. 9:30–9:35 ET) define a high/low band. The high is your breakout
trigger; the low is your initial stop.

**Universe.** Pre-open, build a watchlist of gapped-up small/mid-caps in a
price band, above a minimum gap % and a minimum prior-day volume, from a
snapshot feed. Refresh it nightly/pre-open — don't trade a stale list.

**Entry.** Pre-place a **stop-limit buy** just above the range high (a small
buffer in basis points, so the limit is marketable but capped). Auto-cancel
it after a time window if it never triggers.

**Rank + filter.** Score each candidate from a *small set of pre-entry
features* (gap size, opening-range shape, relative volume, prior-day
positioning — **you choose and validate your own feature set**). Z-score the
features, combine into one composite, gate on a threshold, and bucket into
quintiles. Add a **correlation/family dedup** so you don't take four
variants of the same move.

**Sizing.** Risk-parity: fixed dollar risk per trade, position size = risk ÷
stop-distance. Apply a per-quintile multiplier, and **cap the top quintile
hard** (anti-overfit — your best bucket in-sample is the one most likely to
be lucky). Cap per-position and per-portfolio dollars; enforce a daily-loss
limit and a buying-power ceiling.

**Exit.** Initial stop at the range low. Then a **static lock**: once price
reaches +kR, move the stop to +mR and *freeze it* (no trailing). Otherwise
hold to a force-close time near the close. Pick k and m yourself.

**Post-fill failed-breakout filters.** In the first 1–2 minutes, bail early
if the breakout bar closed weak, or the first bar after entry reverted deep
below entry. These cut the "touch and go" fakeouts.

**Regime overlay.** Classify each day from an index (e.g. SPY) using only
*yesterday's* close, and scale size up/down or skip entirely.

That's the whole machine. None of it is secret — it's standard momentum
structure. The edge is in the constants, and you'll earn those.

---

## 2. The methodology (how to find the constants — this is the real skill)

1. **Two-stage backtest.** Stage 1: build a broad cache with a *loose*
   screen (capture everything that moved). Stage 2: apply your production
   filters on top of the cache. **Only ever report Stage-2 numbers.**
   Stage-1 numbers are raw and meaningless as results.

2. **Walk-forward, always.** Split into TRAIN / VAL / HOLDOUT. Fit z-score
   params and quintile cutoffs on **TRAIN only**. If you ever tune a
   threshold by looking at the holdout, you've burned it — start over with
   new data.

3. **The backtest is a *relative* tool, not a P&L oracle.** It's valid for
   "does filter X beat not-X?" because the unmodeled layers cancel in the
   diff. It is **not** a forecast of dollars — live slippage, regime sizing,
   and venue queue don't transfer. The only honest P&L number is
   **accumulated live data.** Never tell yourself "the BT says I'll make $X."

4. **Parity by construction.** Any logic shared between backtest and live
   (the filter, the exit rule, the regime classifier) lives in **one module
   imported by both.** If you fork it into two implementations, they *will*
   drift, and every drift is a silent loss. This single discipline prevents
   most of the trap list below.

5. **Incident-driven hardening.** When live diverges from backtest, fix the
   root cause and write **two** tests: a unit test for the component, and a
   replay test that feeds the real event sequence through the full flow.
   Bugs you've paid for should be impossible to reintroduce.

---

## 3. The traps — read this before you write a line

Each of these **passes a backtest and a paper account, then loses money
live.** Your Claude will hit the *intuitive* implementation of each, which
is the wrong one.

1. **Partial fills accepted as terminal.** *Naive:* on the first
   `partially_filled` event, record it as the fill and stop polling.
   *Reality:* the order keeps filling in the background; you record half your
   size and under-report the loss. *Fix:* only transition on terminal
   `filled`; keep polling through partials; add a stall-timeout that closes
   at observed qty if the broker hangs. Do the same on the **exit** side —
   reconcile the broker's actual position qty *before* you submit the sell.

2. **Post-fill rule keyed to the fill bar instead of the market breakout
   bar.** *Naive:* "the breakout bar is the minute I got filled." *Reality:*
   when your fill lags the breakout, you evaluate a *different* bar than the
   backtest did, and a large fraction of your early-exit decisions flip.
   *Fix:* key to the first market bar whose high exceeds the range high
   (capture it during the pending phase). Add a **late-fill guard**: if the
   fill lagged the breakout by more than N minutes, skip the post-fill
   rule — it's a stale entry, not a clean breakout.

3. **Exit limit priced with a fixed offset.** *Naive:* `limit = bid −
   max($0.03, 0.5%)`. *Reality:* on a 3¢ spread that's a full spread below
   bid → you give up the whole spread every exit. *Fix:* make it
   spread-aware — `offset = max(floor, fraction × spread)`. Keep the fixed
   formula *only* for emergency exits where you don't have a trustworthy
   live quote.

4. **"No data yet" treated as "zero signal."** *Naive:* initialize
   last-bar-volume (or any confirmation field) to `0`. *Reality:* a position
   armed mid-minute reads `0` before its first bar arrives; a
   volume-confirmation guard reads that as "low volume, hold" and suppresses
   your stop for the critical first seconds of a reversal. *Fix:* initialize
   to `None`/unknown, and make **unknown fail open** — fire the protection.
   Never let "I haven't observed anything" masquerade as "I observed nothing
   happening."

5. **Reconcile-vs-exit race writes a phantom $0 close.** *Naive:* a
   position-sync loop sees the broker position go flat and immediately
   stamps an exit. *Reality:* your exit engine is *still confirming* its
   market-close fill; the sync overwrites the real price with a $0
   placeholder, and you never learn what you made or lost. *Fix:* the exit
   engine stamps a start-time; the sync **defers** its fallback while an exit
   is in progress, with a staleness cutoff so a genuinely stuck exit still
   surfaces.

6. **Buy stop-limit rejected when stop ≤ ask — invisible in paper.**
   *Naive:* place the stop-limit, assume it rests. *Reality:* live brokers
   reject a buy stop whose stop price is at/below the current ask
   ("immediately marketable") — but **paper accepts it**, so this never
   appears in backtest or paper trading. *Fix:* a guard that converts to a
   marketable limit when the bid already confirms the breakout, or re-bumps
   the stop above the ask when the spread straddles the level. Assume
   *every* paper-only behavior is a live landmine.

7. **Look-ahead in the classifier.** *Naive:* compute the day's
   regime/features from the day's own bars. *Reality:* inflated,
   non-reproducible backtest. *Fix:* day T uses **T-1 close only**; in live,
   fetch data ending yesterday.

8. **Fitting on the evaluation set.** Covered above — it's the #1 way a
   beautiful backtest becomes a mediocre live account. TRAIN-only fit,
   untouched holdout, BT as a relative tool.

9. **Synthetic fills hide venue reality.** Paper fills are optimistic; thin
   micro-caps fill multiples worse live. Run a **real-money, half-size
   data-collection phase** before scaling, source your slippage numbers from
   *accumulated live telemetry*, and expect a haircut off the backtest.

10. **Silent fallbacks.** Every `except` / `else` / `.get(default)` that
    papers over a problem must log a WARNING/ERROR with the reason. A
    non-zero daily count of "unknown exit" rows is an **alarm**, not a
    curiosity — alert on it.

11. **One symbol carries the quarter.** P&L is outlier-dependent: a couple of
    big winners drive the year. Don't quote blended P&L without the
    contribution distribution, and size so you *survive* to be in the book
    on the day the winner shows up.

12. **State on restart.** Pending orders and positions opened before a crash
    must be **rehydrated from the broker** on startup. Never assume a clean
    slate — orphaned positions with no stop are how accounts blow up.

---

## 4. The collision pact (why we won't fit the same numbers)

If we both run the same universe at 9:35, we queue for the same shares on
the same thin names. The cost falls almost entirely on the **median trade**
(the 20–60 bps names with shallow books) — the big winners have deep enough
books that we both fill regardless.

So the clean separation is by **price band.** Fit your universe to the
**higher-priced slice** (e.g. $15–30 single names); I'll keep the full
range. Your median names will have deeper books and we'll rarely touch the
same symbol at the same minute. This isn't a handicap — the higher-priced
bucket is a real, tradeable subset with *better* fill quality. And fitting
it yourself is exactly the exercise that teaches you the strategy.

Practical pact: a shared channel where we each post our *entered-today*
symbols, and whoever's already in a name, the other skips it for the day.
Costs each of us a handful of setups; costs neither of us a working bot.

---

## 5. Build order for your Claude Code

1. Stand up the **two-stage backtest harness** and a clean TRAIN/VAL/HOLDOUT
   split first — before any live code. You can't fit anything without it.
2. Implement the **shared modules** (filter, exit, regime) as single sources
   of truth; write the backtest against them.
3. Fit your constants on TRAIN; validate on VAL; look at HOLDOUT *once*.
4. Only then write the live engine, importing the **same** shared modules.
5. For **each of the 12 traps above**, write a replay test *before* you
   trust the live path. Treat the trap list as your acceptance criteria.
6. Run a **half-size, real-money data-collection phase**. Compare live fills
   to backtest per-trade. Don't scale until the divergence is understood and
   small.

The strategy is the easy part. The discipline in sections 2 and 3 is the
whole game.
