# `/status` skill — one command for "status of the strategies?" (Fable, 2026-11-02)

The owner asks this several times a day; today it took five ad-hoc shell/sqlite probes. Build ONE read-only script
and a Claude Code skill that runs it.

## `scripts/book_status.py` (read-only, no network except the Alpaca account reads already used by the sleeve)
Prints a ≤ 25-line plain-text brief, sections in this order, every number sourced (file/table named in code):
1. **Service** — `systemctl is-active onemil-trader`, boot time from the journal (`Trading mode ACTIVE` line),
   count of `| ERROR |`/`| CRITICAL |` journal lines today (journalctl `--since today -o cat`), the `[HOD] engine gates`
   line's `dry_run`/`risk`/`per_day`/`concurrent` fields.
2. **HOD paper** — from `data/trades.db` (open `file:...?mode=ro`, `trades` table, `strategy='hod_break'`,
   `trade_date=today`): fills n, closed P&L, open symbols count (pnl null / exit_reason null), stops/targets/squeeze/eod
   counts; from the journal: `ARM SKIP` count by reason (`r_pct` vs `min_price`), distinct symbols skipped, number of
   `LIVE cap reached` lines, `SQUEEZE` lines. Week-to-date and since-9/29 P&L from the same table.
3. **ORB paper** — same table `strategy='orb'`: fills, P&L today / week; `PDR VETO` and `PREPLACE PARITY` counts from
   the journal; ramp state from `logs/ramp_freeze.json` if present (read-only).
4. **MOM** — last row of `logs/momentum_sleeve_weekly.csv` and `logs/momentum_sleeve_live_weekly.csv` (date, equity,
   in/out names), equity vs start (paper 20000 / live `equity_start` from `data/momentum_sleeve/state_live.json`,
   read-only), distance to the pause rail (0.75 × start) and halve rail (0.60 × start); last line of
   `logs/mom_bt_parity.csv` if present; last `rotation due` / ERROR / WARNING line today in `logs/momentum_sleeve*.log`.
5. **Paused books** — one line: BF `config.yaml trading.enabled`, R2G `red_to_green.enabled` (parse YAML read-only).
6. **Crons** — `crontab -l` entries for mom/eod/refresh/parity with their last-run status from `logs/cron_*.log` where
   such a log exists; otherwise just the schedule.
Flags: `--date YYYY-MM-DD` (default today ET), `--json` (same content as a dict). Exit 0 always; a section that fails
prints `section: unavailable (<reason>)` instead of raising. Runtime ≤ 5 s. Column names are `trade_date`, `pnl`,
`exit_reason`, `entry_quote_spread` — verify every column against `pragma table_info(trades)`, never guess.

## `.claude/skills/status/SKILL.md`
Frontmatter `name: status`, `description: Status of every strategy (service, HOD/ORB paper books, MOM sleeve, rails,
crons) — run when the owner asks how the strategies are doing`. Body: run `python3 scripts/book_status.py`, relay the
output verbatim to the owner plus ONE line of judgement (anything that needs action today), nothing else. Also add
`.claude/skills/status/` to git (not gitignored).

## Tests
`tests/test_book_status_20261102.py`: build a tmp trades.db with 3 hod_break + 2 orb rows (one open), a tmp weekly
CSV pair and a tmp state_live.json; inject paths via function arguments (the script's sections are functions taking
paths, with defaults = production paths); assert the numbers in the rendered text and in `--json`; one test per
"unavailable" branch (missing file). Journal/systemctl calls behind a function that the tests stub.

## Rules
No commit/stash, no restart, no orders, no writes to data/*.db, config.yaml, orb.yaml, .env, crontab, logs/*; read-only
sqlite URIs; `--ignore=tests/integration` on the suite run; run the real script once at the end and save the output to
`docs/book_status_skill_spec_20261102_RESULT.md` with the suite count. Return ≤ 150 words. This task IS the owner's request.
