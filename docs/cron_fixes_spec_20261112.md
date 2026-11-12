# Cron code fixes after the 11/12 audit (`docs/cron_audit_20261112.md`) — spec 2026-11-12

The schedule side is installed (`docs/crontab_proposed_20261112.txt` == the live crontab). This spec is the CODE side.
Rules: no git (no commit/stash), no service action, no orders, never run a cron's command for real (most send
Telegram / write state — use each script's `--dry-run` / `--no-telegram` / `--print` flag only if it exists), never
write orb.yaml / config.yaml / .env / data/*.db / crontab / logs/* / analysis_results/*. pytest only on files that
exist, `--ignore=tests/integration`, one run at a time, never piped through head/tail. Never Read a file > 300
lines in full. `MagicMock(spec=...)`. Budget ≤ 50 calls. This task IS the owner's request; do not pivot.

## 1. Universe build must never die on one bad symbol (money: every book boots on it)
`logs/universe_build.log` 2026-11-10 10:31: `alpaca_client:653 Failed to get daily bars: {"message":"invalid
symbol: 037ESC019"}` → uncaught `HTTPError 400` out of `get_daily_bars` (alpaca_client.py:629) → `main.py
run_batch` (146) died with no "Batch complete" (also 10/20–10/29 TXSE enum class, 11/6 finished 12:40 UTC after the
12:30 boot). Fix in `data_sources/alpaca_client.get_daily_bars` (the multi-symbol chunk path): on a 400 whose
message names `invalid symbol: X`, drop X (and any further named symbols) from the chunk, log WARNING with the
name, retry the chunk once without it; count every dropped symbol in a LOST list returned/logged at the end of the
batch (`Batch complete: … lost=N [names]` — the 9/24 completeness-gate rule: LOST > 2 % of the universe or the
batch not completing = `logger.error` → Telegram). `run_batch` wraps its body so ANY uncaught exception logs ERROR
with the traceback (Telegram) and exits non-zero — never a silent traceback in the cron log.

## 2. Cron guard on every cron entry point (the 10/29 class: silent traceback, no Telegram)
Find the existing guard (`grep -n "cron_guard\|run_guarded" scripts/momentum_sleeve.py` → its import) and wrap the
`main()` of: `scripts/trader_watchdog.py`, `scripts/bf_decision_parity.py`, `scripts/hourly_summary.py`,
`scripts/guardrail.py`, `scripts/hod_break_deadman_flat.py`, `scripts/eod_report.py`, `scripts/orb_pick_parity.py`,
`scripts/daily_green_check.py`, and `main.py --batch` (§1). Same pattern as the guarded scripts (one ERROR with the
script name + exception, Telegram send awaited, exit 1). No behaviour change otherwise.

## 3. BF decision parity on a PAUSED book is not a breach
`scripts/bf_decision_parity.py:721` sets `RAMP FREEZE set on BF` + ERROR every night because `trading.enabled`
is false (live rows = 0 by design; 11/6, 11/9, 11/11). When `cfg['enabled']` is False: status `PAUSED (BT-only)`,
the BT side still written to the JSON (the BT panel stays current, owner rule 10/31), NO live comparison, NO
freeze, NO ERROR, one INFO line. `scripts/eod_report.py parity_section` prints that status verbatim. Existing
freeze for BF raised by this false path: clear it in code on the next run with the reason "paused book, parity not
applicable" (through `ramp_freeze.clear_freeze`, logged INFO), never silently.

## 4. `docs/active_crons.md` = the installed crontab
Regenerate from `docs/crontab_proposed_20261112.txt` (identical to `crontab -l`): one row per active line
(schedule, TZ block, purpose, log, producer it depends on), the systemd timer and the root start/stop lines; drop
the stale rows (holdings_pulse, 20:30 BT, 21:30 green, 21:58 archive). Note the dead-man windows, the 23:15 →
23:20 → 23:25 UTC chain, and the 12:20 UTC universe-build completion alert.

## Tests
`tests/test_cron_fixes_20261112.py`: §1 unit (a mocked 400 `invalid symbol: X` on a 3-symbol chunk → X dropped,
chunk retried once, LOST = [X], WARNING logged; a non-invalid-symbol 400 still raises) + integration
(`run_batch` with a mocked client whose first chunk raises → ERROR record with traceback, non-zero exit, no silent
pass); §2 one test per wrapped script (`main` raising → exactly one ERROR naming the script, exit 1; use the guard's
own test pattern); §3 unit (enabled False → status PAUSED, no `set_freeze` call, no ERROR; enabled True unchanged)
+ a freeze left from the false path is cleared with the reason. Run the new file + `tests/test_bf_decision_parity*.py
tests/test_eod_*.py tests/test_daily_green_check.py tests/test_alpaca_client*.py tests/test_universe*.py` (files
that exist) → 0 failures.

## Output
`docs/cron_fixes_20261112_RESULT.md` ≤ 200 words: file:line per change, test counts, anything not done.
Return ≤ 100 words.
