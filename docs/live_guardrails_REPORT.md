# Live guardrails — implementation report (2026-09-25)

Spec: `docs/live_guardrails_spec_20260925.md`. This agent's scope was the CLI, the EOD section,
CLI tests, the full-suite run, and this report; `trading/live_guardrail.py`, the G2 `tripwire_action`
in `trading/orb_engine.py`, and their fixtures were already done.

## Files
* `trading/live_guardrail.py` — ledger (`live_record`), pause rule (`evaluate_pause`), state file
  I/O (`load_state`/`save_state`/`is_paused`), Telegram (`send_guardrail_telegram`), manual clear
  (`clear_pause`). ONE spec shared by the CLI and (per-spec) the engines' boot check.
* `scripts/guardrail.py` (NEW) — `--check` (all books' ledger, pauses on breach, writes
  `data/guardrail_state.json`, sends `[GUARDRAIL] ... PAUSED` Telegram only on a *new* pause) and
  `--clear BOOK "reason"` (logged; requires a non-empty reason). Resolves `stage_risk_usd` from
  `orb.yaml sizing.risk_per_trade_usd` / `config.yaml trading.risk_per_trade`, and the BT band from
  `trading/ramp_bt_band.py` via the live `catalyst_veto` / `scanner.min_daily_volume` knobs.
* `scripts/eod_report.py` — new `guardrail_section()`, wired into `assemble()`, reuses
  `trading/live_guardrail.py` + `scripts/guardrail.py`'s config wiring (no duplicated logic).
* `tests/test_guardrail_cli.py` (NEW, 6 tests) — `--check` pauses a losing book and calls the
  (mocked, `spec=`) Telegram helper; a healthy book stays unpaused, no Telegram; `hod_break` is
  reported but never paused; `--clear` stores the reason and un-pauses; the EOD `guardrail_section()`
  renders for all three books; a `--check` smoke test through `main()`.

## Rules (frozen, G1) — PAUSE iff ANY fires, per book (orb, bull_flag only; hod_break reported only)
1. trailing-40-fill mean R <= BT band p5 for that book, with >= 20 fills feeding the mean.
2. trailing-20-session $ <= -3 x stage_risk_usd x SESSION_MULT (8 orb / 4 bull_flag).
3. any single session <= -6 x stage_risk_usd.

Ledger starts at the book's first-ever live fill (never reset by a stage or config change — the
defect this spec fixes). `scripts/guardrail.py --clear {book} "<reason>"` is the only un-pause path.

## Boot-log lines to grep
* `journalctl -u onemil-trader | grep '\[GUARDRAIL\]'` — pause/clear events (ERROR on pause, WARNING on clear).
* `grep 'guardrail:' <service log>` — WARNING/ERROR fallbacks (missing config, unreadable state file,
  no BT-band reference, fill with no usable risk).
* `cat data/guardrail_state.json` — current paused flags + numbers + history per book.
* `logs/eod/<date>.md` — the `GUARDRAIL:` section (n, $, trailing-40 mean R vs band p5, paused flag).

## Cron line to add (owner/main session — not edited here)
```
# after the 20:00 UTC EOD report; also run once at boot before market open
0 20 * * 1-5 cd /home/ec2-user/onemil && /usr/bin/python3 scripts/guardrail.py --check >> logs/guardrail_cron.log 2>&1
```

## Tests
`tests/test_live_guardrail.py` (16), `tests/test_orb_tripwire_action.py` (10),
`tests/test_guardrail_cli.py` (6) — 32 pass. Full `tests/test_orb_*.py` + these three files:
**999 passed, 0 failed** (107s). No commits made, no service touched, no writes to `data/trades.db`
or `data/cache.db` (guardrail tests use temp DBs via `tests/conftest.py`'s `guardrail_trades_db`).
