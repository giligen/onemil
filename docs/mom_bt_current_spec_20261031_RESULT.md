# RESULT — MOM backtest stays current (spec `docs/mom_bt_current_spec_20261031.md`)

## Files
* `research/momentum_weekly/panel_ext.parquet` — canonical extension (renamed from `panel_ext_202610.parquet`, 567,202 rows, 2026-09-01..10-30); `cell_1700z.py` EXT path and `RESULT_1700z.md` updated to the new name. `panel_2016_2026.parquet` untouched.
* `scripts/mom_panel_refresh.py` (new) — weekly extension refresh.
* `scripts/mom_bt_parity.py` (new) — weekly rule-vs-book reconciliation, `logs/mom_bt_parity.csv`, one `[MOM]` Telegram line.
* `scripts/eod_sections.py` — `history_period()` (the `MOM P&L ... since 2026-09-29 n/a` cause: a fixed 1M portfolio history no longer reached the 9/29 base; now months-since-start + 1, `1A` from a year), `hod_paper_summary/hod_paper_line/hod_guardrail_line` (read-only sqlite `mode=ro`, `strategy='hod_break' AND account='paper'`), `promotion_section(..., hod_db=None)`; stale `HOD_LINE` removed.
* `scripts/eod_report.py` — GUARDRAIL `hod_break` line now the paper book; `freeze_summary(path, bf_on)` prints BF only when `config.yaml trading.enabled` is true; `MOM BT parity:` line (via `safe_section`) in the MOM block.
* Tests: `tests/test_mom_bt_current_20261031.py` (17 tests: 13 unit, 1 integration refresh→parity→csv→EOD line on a real tiny parquet panel with the real REBUILD rule and `MagicMock(spec=AlpacaClient)`, plus EOD/HOD/history/BF-freeze units); adjusted `tests/test_eod_sections.py` (HOD line) and `tests/test_eod_report.py` (freeze with `bf_on`).

## Real-data dry runs (no writes)
* `mom_panel_refresh.py --dry-run`: 5,757 requested (sleeve assets ∩ extension symbols, common stock), LOST 5 (0.1 %), liquid 768/771 with the 10/30 bar (99.6 %) -> "nothing to append (extension already current through 2026-10-30)".
* `mom_bt_parity.py --dry-run` (8 Mondays; weeks before the paper sleeve's first rotation 10/2 are skipped by the final code, the dry run printed them as 0/20 before that fix):
  wk 10/5 names 20/20 | BT -3.3 % | paper -2.9 % | SPY +1.2 % | resid +0.4 pp; wk 10/12 20/20 | BT -10.1 % | paper -10.6 % | resid -0.5 pp; wk 10/19 20/20 | BT -3.0 % | paper -0.9 % | resid +2.1 pp (MISMATCH, residual only); wk 10/26 17/20 | BT -2.9 % | paper -1.8 % | resid +1.0 pp (MISMATCH, names: BT-only INTC MRNA MXL, paper-only DINO FRO TRGP = the missed 10/26 rotation). Live column n/a for those weeks (see deviations).

## Proposed crontab lines (NOT installed)
```
# MOM backtest stays current: Saturday panel refresh, then the rule-vs-book parity line (spec 2026-10-31)
0 13 * * 6 cd /home/ec2-user/onemil && /usr/bin/nice -n 19 /usr/bin/python3 scripts/mom_panel_refresh.py >> logs/mom_panel_refresh.log 2>&1
20 13 * * 6 cd /home/ec2-user/onemil && /usr/bin/nice -n 19 bash scripts/research_run.sh -m 3G /usr/bin/python3 scripts/mom_bt_parity.py >> logs/mom_bt_parity.log 2>&1
```
(no `%` appears in either line; both scripts end in `run_guarded(main, ...)` = cron_guard, uncaught exceptions reach Telegram as ERROR.)

## Deviations / notes
* Refresh universe = sleeve assets (name-filtered common stock, `1700c`-style EXCL_RE + ZxZZT) **intersected with the symbols already in the extension** (+SPY). The first real dry run over all 6,116 assets counted 313 symbols that simply did not trade in the 9-day probe window as LOST (5.1 % > the sleeve's 5 % gate). A listing without 273 panel bars can never rank in the BT, so nothing is lost; new listings are not added to the extension.
* Restatement probe = fresh fetch from last extension date - 7 days compared on `close` (rel > 1e-4); restated symbols are re-fetched over their whole extension span and replaced. The frozen panel's own history is NOT re-adjusted (the 12.7 % September basis difference documented in RESULT_1700z stays).
* BT week return = equal weight, Monday open -> last close of the week, **before cost** (the 1,700z sim charges ~5-20 bp turnover cost; well inside the 2 pp bar).
* Paper week return = MOM paper account daily portfolio history (session-keyed) delta over the slice equity mark at that Monday's rotation (slice start 20,000 when absent). Live week return = live weekly marks (Monday -> next Monday rotation mark); the latest week uses the sleeve STATE marked with its own names' prices only. A week whose next rotation was skipped (live 10/26) has no mark -> `n/a`, never a guess. The shared live account equity is never read.
* Weeks before the first paper rotation are skipped (INFO), the csv holds one row per Monday that has a paper book.
* `rule module`: `REBUILD_1700tu` is imported as is (its relative `PANEL/ASSETS/D` globals are overridden with absolute paths); `features()` + the guarded selection of its `main()` are used unmodified; `extend_data` is re-implemented in the script (the 1,700z copy is hard-wired to its own EXT path).
* Not run by me: the real `logs/mom_bt_parity.csv` write, the Telegram send, `eod_report.py` against real data (it writes `logs/eod/<day>.md`) — first real write is Fable's rehearsal.

## Suite
`timeout 2700 python3 -m pytest tests -x -q --ignore=tests/integration -p no:randomly` -> **5470 passed, 6 skipped, 0 failed** (576.95 s). Also adjusted `tests/test_guardrail_cli.py` (hod_break guardrail line is now the paper book, `hod_guardrail_line(gr.TRADES_DB)`).
