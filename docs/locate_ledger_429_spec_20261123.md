# Locate ledger: 607 per-symbol asset GETs hit Alpaca's rate limit; 191 ERROR lines paged Telegram; 191 rows carry borrow_status 'error' (2026-11-23)

## The incident (verified: logs/locate_ledger.log 14:26 UTC, logs/locate_ledger.csv, Telegram notifier WARNINGs)
* 09:25 ET `scripts/locate_ledger.py` found 607 gappers (post-crash Monday) and called `fetch_borrow_status` -> `fetch_asset_flags`
  (one `GET /v2/assets/{symbol}`) sequentially for every one, no pacing, no retry. From ~#400 on Alpaca answered HTTP 429;
  191 symbols failed, each logged `ERROR asset borrow_status failed for X: HTTP 429` -> the cron_guard Telegram handler sent 20
  and dropped 171 ("Telegram rate cap hit (20/60s)"). The run then appended 607 rows; 191 have `borrow_status='error'`
  (211 easy_to_borrow / 205 hard_to_borrow / 191 error; locate_status 351 etb / 222 quoted / 34 threshold is intact — the
  locate quotes go in batches of 25 and were fine). `append_rows` dedupes on (date, symbol), so a re-run cannot repair them.
* Precedent: the 11/14 spec `docs/locate_ledger_spec_20261114.md` sized the day at ~10-60 gappers; the per-symbol GET was fine then.

## The fix (TDD: failing tests first). `scripts/locate_ledger.py` only. The ledger stays read-only on the broker (GET only; NEVER
## POST /v1/locates). Frozen constants / LEDGER_COLUMNS unchanged.
1. One asset table instead of N GETs. New `fetch_asset_table(session) -> dict`: ONE `GET /v2/assets?status=active&asset_class=us_equity`
   -> `{symbol: {borrow_status, shortable, easy_to_borrow}}` using the same fields `fetch_asset_flags` reads (`borrow_status`
   may be absent on the list endpoint: VERIFY with one real read-only GET during the task and say so in the Result; if the field
   is absent there, keep the per-symbol path as the source of `borrow_status` and use the table only to skip symbols that are
   not shortable). `fetch_borrow_status(session, symbol, table=None)`: table hit -> no GET; miss -> the per-symbol GET as today.
2. Pacing + 429 backoff on the per-symbol path: module constants `ASSET_GET_MIN_INTERVAL_S = 0.35` (<= ~170/min) and
   `ASSET_GET_RETRIES = 3`; on HTTP 429 sleep `Retry-After` seconds when the header is present, else 2^attempt seconds, then
   retry; only after the last attempt return `'error'`. Each retry logs WARNING once with the attempt number.
3. ONE aggregated ERROR. Per-symbol failures no longer log ERROR each; `run_ledger` collects them and logs ONE line
   `ERROR locate ledger: borrow_status failed for N of M gappers (HTTP 429 x K, other x J): first 10 symbols ...` (the cron_guard
   handler then sends one Telegram message). `summary_line` adds `error N` when N > 0.
4. Repair mode. `--repair-borrow [YYYY-MM-DD]` (default today): read the ledger, select that day's rows with
   `borrow_status == 'error'`, re-fetch their borrow_status (table first, paced GETs after), rewrite ONLY the `borrow_status` field
   of those rows in place, atomic write (temp file + os.replace), log INFO `repaired N of M error rows for <date>; still error: K`.
   Rows that still fail stay 'error'. No other field changes, no new rows.

## Tests — append to `tests/test_locate_ledger.py` (reuse its fixtures / fake session pattern; read it first)
1. `fetch_asset_table` parses a list response into the dict; `fetch_borrow_status` with a table hit makes NO HTTP call.
2. Table miss -> exactly one GET.
3. 429 then 200: the fake session returns 429 (with `Retry-After: 0`) once then 200 -> status returned, one WARNING, no ERROR.
4. 429 x (retries+1) -> 'error', still no per-symbol ERROR record.
5. `run_ledger` with 3 gappers of which 2 fail -> exactly ONE ERROR record containing "failed for 2 of 3"; summary line has
   "error 2"; the rows carry 'error'.
6. Pacing: with `ASSET_GET_MIN_INTERVAL_S` patched to 0.05 and 3 misses, the monotonic span of the 3 GETs is >= 0.1 s.
7. `--repair-borrow`: a temp ledger with 4 rows for the day (2 'error', 1 'easy_to_borrow', 1 other day 'error') -> after repair
   the 2 same-day error rows have the re-fetched status, the other two rows are byte-identical, row count unchanged, header
   unchanged; a still-failing symbol stays 'error' and is counted in the INFO line.
8. Integration (real temp CSV through `append_rows` then `--repair-borrow` through `main()` with argv patched and the HTTP
   session/broker mocked): the file round-trips and `read_csv` returns the repaired statuses.

## Run (this file only): `python3 -m pytest tests/test_locate_ledger.py -q -p no:cacheprovider --ignore=tests/integration` -> green.

## Rules
Sonnet, <= 45 calls. Trading hours are open and the LIVE service is running: no `git commit`/`git stash`, no service action, no
orders, no writes to `logs/*` (the real ledger included — the main session runs the repair), `data/*`, `config.yaml`, `.env`.
Exactly ONE real read-only GET is allowed for item 1's field check (`GET /v2/assets?status=active&asset_class=us_equity`, print
only the key names of one element, never the keys/secrets; use the ALPACA_API_KEY / ALPACA_API_SECRET env names the script
already uses). Files you may edit: `scripts/locate_ledger.py`, `tests/test_locate_ledger.py`, and a "Result" section appended to
THIS file. Docstrings on every new function; every fallback logs WARNING or ERROR. Return <= 120 words: files changed, test counts
red-before / green-after, the field-check result, anything you could not do, verified facts separated from guesses.

## Result (2026-11-23)
* Files: `scripts/locate_ledger.py`, `tests/test_locate_ledger.py` (+12 tests; the old per-symbol-ERROR assertion in `test_borrow_status_and_error_logged` now asserts WARNING, no ERROR, per spec item 3).
* Red before: 5 failed + 7 errors (12 new tests); green after: 22 passed (spec pytest command).
* Field check (one real read-only GET): `GET /v2/assets?status=active&asset_class=us_equity` -> 200, 14,451 rows, `borrow_status` PRESENT on every row (keys: attributes, borrow_status, class, easy_to_borrow, exchange, fractionable, id, maintenance_margin_*, marginable, name, shortable, status, symbol, tradable). The table is therefore the full source of borrow_status; per-symbol GET only on a table miss/failed table.
* Added: `fetch_asset_table`, `resolve_borrow`, `log_borrow_failures`, `repair_borrow`, `--repair-borrow [DATE]`, pacing + 429 backoff constants. Not run against the real ledger (main session does the repair).
