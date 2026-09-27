# FETCH_1567 — options_vrp data fetch (PREREG_1567 FETCH stage)

Run finished: 2026-09-27T22:37:56.105107Z

## SPY underlying
* Daily bars (2024-01-02..2026-09-26): 686 rows.
* Minute bars (full RTH 09:30-16:00 ET, every session, monthly-chunked fetch): 566174 rows.

## Options grid
* Entry Mondays (first trading session of the week, 2024-02-05..2026-08-17): 133.
* Unique OCC contracts attempted: 90016.
* (monday, contract) reference pairs: 181918.
* Strikes: $1 steps, spot*(1-0.2) to spot, spot priced at the SPY minute bar nearest
  10:00 ET that Monday. **$0.50 strikes were NOT attempted** (scope reduction — doubling every
  request for a refinement not needed to bracket delta targets 0.15-0.30 on a $1 grid; flagged per
  the task's own "say so" clause, not silently dropped).
* Expiries: every calendar-weekday (Mon-Fri) date at DTE 38-52 from the entry Monday.
  SPY does not list options on every weekday; non-listed (monday,strike,expiry) combinations return
  no bars and are counted as ABSENT below, not as a fetch error.

## Option daily bars (2024-01-02..expiry, every grid contract)
* ok=45567  absent=44449  error=0
* Completeness gate (error rate vs 3.0% threshold): 0.00% -> **PASS**

## Option entry-minute bars (09:55-10:10 ET, entry Monday only — see scope reduction below)
* ok=9167  absent=172751  error=0
* Completeness gate (error rate vs 3.0% threshold): 0.00% -> **PASS**

## Documented scope reduction (say-so clause)
The task's literal ask was minute bars at 10:00-10:05 ET for every session a contract is live, or
(fallback) 09:55-10:10 ET on Mondays **and** 15:55-16:00 ET of every session. At this grid size
(90016 unique contracts, avg lifetime ~40 sessions), the every-session leg is
O(contracts x sessions) ~= several million requests — infeasible under the 200 req/min ceiling in
one research session. What was fetched instead:
* Option DAILY bars for every contract, full listing-to-expiry range — gives the SCORE stage a
  daily mark (close) to evaluate the 50%-credit / 2x-stop / 21-DTE management rule.
* Option MINUTE bars only for the entry-Monday 09:55-10:10 ET window — gives the real minute-bar
  fill price the PREREG's Fill rule requires (mid at 10:00-10:05, never approximated by a daily bar).
* NOT fetched: 15:55-16:00 ET minute bars on every non-entry session for every contract. The SCORE
  stage must either (a) treat the daily close as the management-trigger price (documented
  approximation, disclose it), or (b) request a second, much smaller fetch limited to the specific
  legs actually selected after Stage-1 strike selection (a small, bounded set vs. the full grid) —
  recommended, and cheap once legs are known.

## Files
* research/options_vrp/opt_cache/spy_daily.parquet
* research/options_vrp/opt_cache/spy_minute.parquet
* research/options_vrp/opt_cache/option_daily.parquet
* research/options_vrp/opt_cache/option_minute_entry.parquet
* research/options_vrp/opt_cache/manifest.parquet (90016 contracts)
* research/options_vrp/opt_cache/state.db (sqlite, resumable fetch state — keep)
