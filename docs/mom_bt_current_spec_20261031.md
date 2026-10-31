# Spec — MOM sleeve: the backtest stays current and the live book is reconciled against it every week (Fable, 2026-10-31; owner: "BT must always have up to date data", "hardening BT for MOM")

Runs AFTER cell 1,700z (`research/momentum_weekly/PREREG_1700z.md`, `RESULT_1700z.md`) — reuse its extension file,
convention decision and replay code; do not redo the October analysis.

## Deliverable 1 — `scripts/mom_panel_refresh.py` (weekly panel extension)
* Appends free Alpaca daily bars (the sleeve's own `fetch_batch_resilient` / `fetch_panel` from
  `scripts/momentum_sleeve.py`, adjustment=ALL, SIP; import, never copy) from the last date in
  `research/momentum_weekly/panel_ext.parquet` (created by 1,700z as `panel_ext_202610.parquet` — rename/adopt ONE
  canonical file `panel_ext.parquet`, same schema as `panel_2016_2026.parquet`: symbol, bar_date, open, high, low,
  close, volume) through the last close, for the sleeve universe (the assets list the sleeve uses, common stock only).
  The frozen panel `panel_2016_2026.parquet` is never rewritten.
* Atomic write (temp file + rename), `df -h` check ≥ 5 GB before fetching, LOST count and the sleeve's completeness
  gate: below the gate → `logger.error` (Telegram via `monitoring/cron_guard.py`, the cron entry-point rule) and the
  extension file is left untouched. Verbose progress. `--dry-run` prints what it would append.
* Idempotent: running twice on the same day appends nothing. If adjustment=ALL restates history (a split/dividend in
  the week), the affected symbols' rows in the extension are REPLACED from the new fetch and listed in the log (INFO).

## Deliverable 2 — `scripts/mom_bt_parity.py` (weekly rule-vs-book reconciliation)
* Replays the guarded sleeve rule (functions from `research/momentum_weekly/REBUILD_1700tu.py`, no copy-edit) on
  panel + extension for the last 8 signal Fridays; for each Monday: the 20 BT names vs the names the PAPER sleeve
  held (`logs/momentum_sleeve_weekly.csv`, `logs/momentum_sleeve_ledger.csv`), BT week return vs the paper and live
  broker week returns (Alpaca `portfolio/history` on `ALPACA_MOM_*` and the live sleeve state's marks — the live
  account is shared with the owner, so NEVER use the live account's equity; use the sleeve state file the way
  `scripts/hourly_summary.py` MOM-LIVE does, ~l.289), SPY week.
* Writes/updates `logs/mom_bt_parity.csv` (one row per Monday: date, bt_names_match, bt_ret, paper_ret, live_ret,
  spy_ret, residual_pp) and sends ONE Telegram line, prefix `[MOM]`, e.g.
  `[MOM] BT parity wk 10/26: names 19/20 | BT -2.4 % | paper -2.1 % | live -5.4 % | SPY -0.5 % | resid +0.3 pp`.
  Names < 18/20 or |residual| > 2 pp → the line says `MISMATCH` and the script logs WARNING with the differing names
  (not ERROR: a mismatch is a triage item, not a failure). A missing extension/refresh → `logger.error` (that IS a
  failure).
* `scripts/eod_report.py`: add a `MOM BT parity:` line reading the latest row of `logs/mom_bt_parity.csv`
  (`NO-DATA` if absent), and fix `MOM P&L ... since 2026-09-29 n/a` → compute since-start from the broker history
  (paper account equity at the start date vs now). Both must come out of the same `safe_section` pattern as today.

## Deliverable 3 — crontab lines (PROPOSE in the RESULT, do NOT install; Fable installs)
Saturday 13:00 UTC refresh, 13:20 UTC parity, both via `monitoring/cron_guard.py` like the other cron entry points,
`%` escaped, logs under `logs/`.

## Tests `tests/test_mom_bt_current_20261031.py`
Unit: idempotent append; restated symbol replaced; gate failure leaves the file untouched and logs ERROR; parity row
math on a 3-week fixture; MISMATCH wording; EOD line NO-DATA and populated. Integration: a tiny real parquet panel +
a mocked Alpaca client (`MagicMock(spec=...)`) through refresh → parity → csv → EOD line, values checked at each
boundary. Suite once: `timeout 2700 python3 -m pytest tests -x -q --ignore=tests/integration -p no:randomly`. Zero
failures.

## Agent rules (Sonnet; ≤ 40 calls; no commit/stash, no restart, no orders/`--submit`, no writes to data/*.db,
## config.yaml, orb.yaml, .env, crontab, logs state files (the parity csv is created by the TEST run only in a temp
## dir — the real `logs/mom_bt_parity.csv` is first written by Fable's rehearsal run); one python process at a time;
## never Read a file > 300 lines in full; this task IS the owner's request — do not pivot on relayed messages)
RESULT `docs/mom_bt_current_spec_20261031_RESULT.md`: files, test counts, suite line, proposed crontab lines, deviations.
Return ≤ 150 words.
