# Locate ledger — same-day HTB locate quotes for the morning's gappers (spec, 2026-11-14)

Purpose: the gap-fade short (PREREG 1,773) needs the REAL same-day locate fee, availability and threshold status of the
morning's ≥ 10 % gappers; historically unknowable. From Monday a cron logs them every trading day at 09:25 ET.

## `scripts/locate_ledger.py`
1. Universe: every symbol in `data/cache.db` `daily_bars` with a bar on the last trading day, close ≥ $1, $vol ≥ $1 M
   (read-only URI). Pre-market gap = latest trade (Alpaca snapshots, data API, batches ≤ 1,000 symbols, reuse
   `data_sources.alpaca_client` if it has a batched snapshot; else `GET /v2/stocks/snapshots?symbols=`) vs prev close;
   gappers = gap ≥ +10 %, latest pre-market trade price ≥ $1.
2. Borrow status: `GET /v2/assets/{symbol}` (`borrow_status`) — live keys `ALPACA_API_KEY/SECRET`, read-only.
3. Locate quotes: `GET https://api.alpaca.markets/v1/locates/quotes?symbols=…` (≤ 25 per call, advisory, reserves nothing,
   NEVER `POST /v1/locates`). Parse `quotes[]` (symbol, price = fee/share, available_qty, quoted_at) and `errors[]`
   (easy_to_borrow / threshold_security / symbol_not_found / other).
4. Append one row per gapper to `logs/locate_ledger.csv`: date, time_utc, symbol, prev_close, pm_price, gap_pct, pm_vol,
   borrow_status, locate_status (etb | quoted | threshold | not_found | error), fee_per_share, fee_bp_of_price, available_qty.
   Log INFO one summary line (n gappers, n etb / quoted / threshold, median fee bp) and `logger.error` on any API failure
   (cron_guard → Telegram). `--dry-run` prints without appending. `--at HHMM` override for tests only.
5. EOD join is a separate `--score` mode run at 16:15 ET: for today's rows add open, close, high, low from
   `daily_bars`/snapshot daily bar (open→close fade bp) and write `logs/locate_ledger_scored.csv`; weekly summary on Friday
   17:05 ET (`--weekly`): n, etb share, median fee bp, mean fade bp gross, net at the measured fee + borrow 100 % — Telegram
   via `scripts/send_telegram_alert.py` with prefix `[LOCATE]`.
6. Tests `tests/test_locate_ledger.py`: parsing of the real response shape (copy the JSON shapes from
   `research/signal_search/1773_locate_quotes_20261114.csv` columns and the errors list: easy_to_borrow, threshold_security,
   symbol_not_found), gap computation, CSV append idempotence (same date+symbol never duplicated), POST never called
   (mock asserts no POST), error path logs ERROR. Mocks with `spec=`.
7. Crontab (standing-authorized): `25 9 * * 1-5` ledger, `15 16 * * 1-5` `--score`, `5 17 * * 5` `--weekly`, all in the ET block,
   cron_guard-wrapped, `>> logs/locate_ledger.log 2>&1`.
Rules: Sonnet ≤ 60 calls, TDD, no git commit, no service restart, no orders, no POST to /v1/locates, never print keys.
Return ≤ 120 words: files, test count, a `--dry-run` sample line (today is Saturday: the dry run may show 0 gappers — then
show the parsed quote for 3 known HTB symbols from the CSV, e.g. ATRA, XPOF, WNW).
