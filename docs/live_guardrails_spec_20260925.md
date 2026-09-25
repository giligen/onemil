# SPEC — live guardrails that act without a human (2026-09-25)

## Why
ORB went live 2026-05-19 and lost −$5,281 on 123 fills through 9/23 (May +1,539, Jun −3,187, Jul −2,570, Aug −585,
Sep −477; `data/trades.db`, strategy 'orb') while the backtest for the same months was positive. The `LATENCY
TRIPWIRE` warning (first order 26–49 s after 09:35:00) fired for weeks and only wrote a log line; every config change
(9/17 resume, 9/21 catalyst-off) restarted the ramp's stage clock, so no rule ever saw the cumulative live loss. The
owner's words: a bug lost money for five months before it was flagged. Two guardrails, both automatic.

## G1 — cumulative live ledger + auto-pause (per book: orb, bull_flag; hod_break dry is reported only)
* `trading/live_guardrail.py` (ONE helper shared by the daily check and the engines): `live_record(book, db)` returns the
  book's closed fills from `data/trades.db` since the book's first live fill EVER — never reset by stage or config.
  Per book it computes: n_fills, total $, mean R per fill (R from the trade's own risk: |entry − stop| × shares when
  stored, else the stage risk), trailing-40-fill mean R, trailing-20-session $, worst month.
* Pause rule (frozen): PAUSE iff trailing-40-fill mean R ≤ the backtest band's p5 for that book (ORB:
  `trading/ramp_bt_band.py`; bull flag: its `bf_frequency/runs/P1.csv` reference) with ≥ 20 fills, OR
  trailing-20-session $ ≤ −3 × the book's stage risk × 8 (ORB) / × 4 (BF), OR any single session ≤ −6 × stage risk.
* Action: write `{book}.paused_by_guardrail = true` into `data/guardrail_state.json` (with the numbers and UTC time),
  send a Telegram `[GUARDRAIL] {book} PAUSED: …numbers…`, and the engine's pre-open check at boot reads that file and
  refuses to submit real orders for that book (dry mode where the book has one, else no orders) until the owner clears it
  with `scripts/guardrail.py --clear {book} "<reason>"` (logged). The daily EOD report prints the cumulative ledger
  for every book every day (n, $, trailing-40 mean R vs band) — the number that was missing for five months.
* Runs: `scripts/guardrail.py --check` from the existing EOD cron (after the trader exits at 20:00 UTC) and at boot.

## G2 — latency tripwire that stops orders
* In `trading/orb_engine.py`, at the point where `LATENCY TRIPWIRE` is logged: if `execution.tripwire_action:
  'dry'` (new orb.yaml key, default 'warn' = today's behaviour), set the engine to dry mode for the REST OF THE SESSION
  (the same code path as `strategy.dry_run`, so picks are still logged as `[ORB DRY] WOULD BUY`), log at ERROR, and send
  `[ORB] TRIPWIRE → DRY for today: first submit {x} s late`. The threshold stays the existing 10 s.
* Also measure and log the per-order submit latency (bar close → submit) for every order, so the EOD report shows the
  distribution (median / p90) per day.

## Rules (Sonnet; ≤ 40 tool calls; TDD; do not commit; do not start/restart any service; no DB writes to production
tables; no DB reads after 13:25 UTC — check `date -u`)
* Tests: `tests/test_live_guardrail.py` (ledger from a temp trades.db with three books; each pause rule fires exactly at
  its threshold; a config change does NOT reset the ledger; state file round-trip; clear-with-reason logged) and
  `tests/test_orb_tripwire_action.py` (warn = unchanged behaviour; dry = zero orders after the tripwire, WOULD BUY lines
  present, Telegram text). Fixtures in `tests/conftest.py`, `MagicMock(spec=...)`. Run the full `tests/test_orb_*` +
  the new files — zero failures.
* Wire `scripts/guardrail.py --check` into `scripts/eod_report.py`'s output (one section) and document the cron line to
  add (do not edit the crontab; the main session does that).
* Write `docs/live_guardrails_REPORT.md`: files, the exact rules with their thresholds, the boot-log lines to grep, the
  rehearsal plan. Return ≤ 150 words.
