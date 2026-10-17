# ORB latency triage 2026-10-17 (read-only; spec `orb_latency_triage_20261017_SPEC.md`)

## Headline
The 5-10 s gap is NOT client HTTP. The spec's "engine stamp" matches the broker `created_at` (Alpaca stamps it on
receipt, within ~0.1-0.4 s of the engine's log line). The gap is `created_at -> submitted_at`: 4.3-10.7 s INSIDE
Alpaca. A second, client-side delay exists before the first order (below). H3 refuted, H2 not supported, H1 only as a
lock wait, not as per-order REST work.

## 1. Static trace (engine -> wire), current code
| step | call | per-order / per-burst | expected ms |
|---|---|---|---|
| timer 09:35:05 `_preplace_submit_at_close` (l.4194) | `with self._lock` (l.4209) | per-burst; BLOCKS while `check_entries` runs (l.3199 holds `_lock` for the whole body) | 0 .. 7,800 (measured) |
| inside the lock: `post_open_range_sweep` (l.3260) | REST bars batch (`_ensure_ranges_post_open`) | per-burst, 09:35:00 start | 7,300-7,800 (logged by the tripwire line) |
| `_submit_entry` (l.5461) | `system_state.is_account_halted` | per-order | <1 |
| | `get_latest_quote` (data client, `_call_with_timeout` = new 1-worker ThreadPoolExecutor) | per-order | ~105 |
| | `evaluate_buy_stop` (pure) | per-order | <1 |
| | `submit_stop_bracket_order` -> `_call_with_timeout` -> `trading_client.submit_order` | per-order | ~105 (+ broker accept) |
| | `_log_submit_timing` (new, log-only), `_save_pending_trade` (SQLite) | per-order | few |
Preplace burst: concurrent (ThreadPoolExecutor <= 8). Production path (09:35+): `fast_submit: false` -> SERIAL loop,
each order = vetoes + spread/quote + planner + `_has_buying_power` (account REST) + `_submit_entry` (quote + submit).
Wrapper (`alpaca_client.py` l.2309 etc.): no sleeps on the happy path; retry sleeps only on 429 (backoff) or timeout
(`TIMEOUT_RETRY_BACKOFF_SECONDS`); one `TradingClient` built at init (not per call).

## 2. Every ORB buy order 10/12-10/16 (ORB account; 10/13 and 10/15 placed none)
engine stamp = broker `created_at` (== log line second); times are ET 09:35:SS.s
| day | sym | path | created (SS.s) | submitted_at | gap s | pos |
|---|---|---|---|---|---|---|
| 10/12 | ASTX | preplace | 08.1 | 12.4 | 4.3 | 1 |
| 10/12 | ASTY | preplace | 08.4 | 12.8 | 4.4 | 2 |
| 10/12 | ASTY | resubmit | 26.3 | 31.2 | 4.9 | 3 |
| 10/12 | APUS | production | 27.0 | 31.7 | 4.7 | 4 |
| 10/14 | SNDQ | preplace | 08.2 | 18.9 | 10.7 | 1 |
| 10/16 | PMI | preplace | 07.7 | 14.7 | 7.0 | 1 |
| 10/16 | CORD | preplace | 08.0 | 15.3 | 7.3 | 2 |
| 10/16 | IREZ | preplace | 08.3 | 15.7 | 7.4 | 3 |
| 10/16 | IREZ | resubmit | 09.8 | 17.9 | 8.2 | 4 |
| 10/16 | BKYI | production | 10.4 | 19.2 | 8.8 | 5 |
| 10/16 | SMST | production | 10.9 | 19.9 | 9.0 | 6 |
| 10/16 | AIXI | production | 14.6 | 24.7 | 10.1 | 7 |
Pattern: gap GROWS with burst position within 10/16 (7.0 -> 10.1, monotone) but is FLAT on 10/12 (4.3-4.9) and 10.7 for a
single order on 10/14 -> level is per-day, not per-position. Position and time-after-open are confounded (n = 12 orders,
3 days). The growth sits after `created_at`, i.e. in Alpaca's order pipeline (a queue), not in our serial loop: our
serial loop adds only 0.3-0.6 s per order to `created_at`, while the broker adds ~+0.4 s per extra order on 10/16.
Caveat: ORB runs on the PAPER endpoint (`ALPACA_ORB_PAPER=true`); a paper OMS may queue differently from live.

## 3. REST round trip now (ORB keys, paper, weekend, n = 20 each)
| call | first | median (rest) | p90 | max |
|---|---|---|---|---|
| `get_account_info` | 329 | 105 | 107 | 329 |
| `get_latest_quote` AAPL | 335 | 107 | 115 | 335 |
| `get_order` | 113 | 109 | 110 | 113 |
| plain `requests.Session` /v2/account | 324 | 107 | 109 | 324 |
| plain `requests.get` no session | 327 | 325 | 330 | 332 |
The client already sits on the keep-alive floor (105 ms == Session); TLS warm-up costs ~220 ms once. H3 refuted.
Weekend numbers; the open-of-market RTT is unmeasured (Monday's SUBMIT TIMING `http_ms` will give it).

## 4. Contention 13:34:30-13:36:30 UTC, 10/12-10/16
Window is ~100 % `[ORB]` lines; 3-6 HOD lines per day, none between 13:35:00 and 13:35:30 (first HOD line 13:36:00).
`Engine tick`, `Cycle overrun`, `TIMEOUT`, `mover queue`, HOD broad-scan lines: ZERO hits in the window on all 5 days.
Only 27-43 log lines in 13:35:00-13:35:30 each day (quiet, not a flood); DEBUG `GAP_GATE` floods of ~2,000 lines/s occur
later (13:35:40-46, 13:36:25) AFTER the orders. `nproc` = 2. No thread count is logged. H2 is not supported by the logs.

## 5. Verdict
Two separate delays, neither is H1-per-order, H2 or H3:
(a) Broker-side `created_at -> submitted_at` 4.3-10.7 s (median ~7.3 s) = the part the spec asked about. Not reducible
    from our code; the order is "created" at Alpaca but not "submitted" until later. Open question for Alpaca / the live
    endpoint (is it paper-only?).
(b) Client-side lock wait: the preplace timer fires at 09:35:05, but `check_entries` holds `self._lock` through the 7.3-7.8 s
    `post_open_range_sweep` (tripwire line: "post_open_range_sweep 7.3s/7.6s/7.8s"), so the first order is created at
    07.7-08.2 s, ~2.7-3 s after the timer. Same pattern all 3 days.
ONE fix: in `check_entries`/`_check_entries_locked` (trading/orb_engine.py l.3199, 3260) run `_ensure_ranges_post_open`
(the REST sweep) OUTSIDE `self._lock` (or pre-sweep at 09:34:58 before the timer), so the 09:35:05 preplace burst never
waits behind it. Must not change selection, sizing, ranking, chase/guard rules or the sweep's result.
What Monday's SUBMIT TIMING separates: `quote_ms`, `size_ms`, `http_ms`, decision->http_start (lock wait shows as
decision_ts to http_start ~2.7 s) vs `broker_submitted_at - http_end` (broker lag). If broker lag is again 4-10 s, the
remaining gap is Alpaca's and needs the Alpaca support/live-endpoint check.
