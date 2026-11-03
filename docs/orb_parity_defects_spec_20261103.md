# ORB parity defects found on the 2026-11-02 EOD (Fable, 2026-11-03) — fix spec

Four separate defects produced yesterday's "ORB PAPER PARITY" noise and one false Telegram alarm. Evidence already
established (do not re-derive; verify by reading the cited lines):

* Engine 09:35 decision 11/2: final top-8 `['SKUU','BILI','HSAI','BABX','SCO','XPEV','SKHL','SKHX']`, PDR vetoes
  BILI HSAI BABX SCO XPEV, fills SKUU SKHL SKHX (all stopped, −$388). PMN was PREPLACED and submitted at 09:35:05 on the
  provisional top-4, then cancelled by the reconcile (`trades.order_status='preplace_canceled'`, never filled). FTFT
  preplace submit failed. AEYE: `GAP_GATE` passed on a phantom snapshot open (5.89 %), then
  `phantom-gap reject (production) — real gap 3.07% < 5.0%` — correct engine behaviour.
* EOD (`logs/eod_report.log` 1121–1127): `Ranked: engine 12 vs BT 8 | engine-only FTFT PMN RAM SKHX XPOF | BT-only AEYE`,
  `Picks: engine 4 vs BT 2 | engine-only PMN SKHX SKUU | BT-only AEYE`, `ORB P1 NO-DATA`.
* BT replica (`scripts/eod_sections.bt_ranked('2026-11-02')`): ranked `['AEYE','SKUU','BILI','HSAI','BABX','SCO','XPEV','SKHL']`,
  picks `['AEYE','SKHL']` (SKUU vetoed on BT range 2.209 % < 2.221 floor; live range was 2.26 % — bar drift, not a defect).
* `analysis_results/orb_features_20261102_2156.csv` (production panel, 14,031 rows): **99 rows with gap_pct < 5.0, 60
  entered, +$20,942 BT P&L; since 10/1: 9 rows, 5 entered (FLY 4.11, OKLL 4.39, BOOM 4.79, AEYE 3.07 …)**.
  `study_orb_features.py` says `MIN_GAP_PCT = 5.0` (l. 55, used l. 165–168) and stores `gap_pct = open − prev_close`
  (l. 305). The production book `analysis_results/orb_bplus_book.csv` carries AEYE 2026-11-02 −$1,802 — a trade the
  engine can never take.
* `logs/orb_observer_cron.log` 1787–1838: the observer (cron `26 13 * * 1-5` UTC) waits on HARD-CODED UTC times
  (`wait_until_utc(13,29,30)` etc., script header says "9:29:30 ET"). Since DST ended 11/1 it runs an hour early: 44
  "expected" candidates from pre-market snapshots, `LIVE ranged: 0 | scored: 0`, and it SENT a
  `[ORB-SELECTION] CAUGHT n REAL dropped candidate(s)` Telegram (no `telegram failed` line). Note 10/30 (still EDT,
  l. 1736–1740) ALSO parsed `ranged 0 | scored 0` while `submitted: ['IREZ']` — the journal parse was already failing
  (suspect: the 2.3M-line journal flood vs the 30 s `journalctl` timeout, l. 117; the flood was fixed in d1977df).
* `analysis_results/orb_bplus_book_markers.csv` has `2026-11-02,P1,20,0,ok` written 22:26 UTC; the EOD ran 21:57 → P1
  NO-DATA is cron ORDERING. Find what builds the production features at 21:56 and the P1 book at 22:26 (no script in
  `scripts/` greps for `orb_backtest`; check `daily_green_check.py`, `nightly_bt_update.sh`, the stupid-money crons,
  and `logs/*.log` for the 21:5x/22:2x writer).

## Fixes (in this order; each with a unit test; production behaviour otherwise unchanged)

1. **Observer DST + false alarm** (`scripts/orb_selection_observer.py`): every wait and the journal `--since` use
   America/New_York via `zoneinfo` (09:29:30 / 09:31:00 / 09:35:00 / 09:36:30 ET), never a UTC literal. If
   `live_range` AND `live_scored` are both empty while the engine logged any `ENTRY SUBMITTED` / `ORB SCORED` /
   `reconcile final` line, or `journalctl` failed/timed out: log WARNING `observer: engine log not parsed — no verdict`
   and send ONE Telegram line saying exactly that (ops warning, not "CAUGHT"). Never send CAUGHT on a day the live set
   is empty. Reproduce the 10/30 parse failure (run the parse function on `logs/session_archive/2026-10-30.log` or the
   journal for that day) and fix the cause. Tests: frozen-clock ET wait targets on an EST and an EDT date; empty-live
   → no CAUGHT message (mock the notifier).
2. **EOD parity counting** (`scripts/eod_sections.py` ~l. 860–892): `traded` excludes `order_status` in
   `('cancelled','canceled','rejected','preplace_canceled')` and any row without a `fill_price` AND without an open
   order; preplace names that the reconcile DROPPED (in `provisional top-N` but not in `reconcile final` nor
   `ENTRY SUBMITTED`-and-kept) never enter `eng` ranked. Re-run the 11/2 lines after the fix: expected
   `Ranked: engine 8 vs BT 8 … BT-only AEYE`, `Picks: engine 3 vs BT 2 | match 1 | engine-only SKHX SKUU | BT-only AEYE`.
   Same for `p1_parity_lines`. Tests on the 11/2 log lines as a fixture.
3. **P1 ordering**: make the ORB BT build (production features + P1 book) finish BEFORE the 21:57 EOD, or make the
   EOD's P1/production BT lines read the LATEST marker ≤ today and say `(BT of <date>)` when it is not today's. Prefer
   the first if it is only a cron time (crontab is the owner's — write the exact `!` crontab edit into the RESULT, do not
   edit the crontab); otherwise the second.
4. **BT gap gate (money-relevant, NO rebuild in this pass)**: find why 99 rows with `gap_pct < 5` are in the production
   panel (which gap the admission at l. 165–168 actually tests vs the stored l. 305 value — phantom/premarket open?
   adjusted vs raw prev close?). Make the production admission test the SAME real gap the engine's phantom-gap reject
   tests (open of the first regular-session bar vs prior regular close) and keep `gap_pct` identical to it. Add a
   parity test: no production row with `gap_pct < min_gap_pct`. Do NOT rebuild the panel or the book; write the
   affected rows (date, symbol, gap_pct, entered, pnl) to `docs/orb_parity_defects_20261103_gaprows.csv` and the P&L
   delta by month to the RESULT. The rebuild runs under a PREREG (it moves the ramp band).

## Rules
No commit/stash; no restart (onemil-trader boots 12:30 UTC and is LIVE until the close — finish the code before 12:25
UTC or leave the working tree importable at every moment: edit, test, never leave a half-written module); no orders; no
writes to data/*.db, config.yaml, orb.yaml, .env, crontab, logs/*, analysis_results/* (the gap-rows CSV goes to docs/).
Tests: the specific test files, then the full suite ONCE at the end in the background with a scratchpad log
(`python3 -m pytest tests -q -x -n 2 --ignore=tests/integration > <scratchpad>/suite.log 2>&1`), never piped; one pytest
process at a time (`pgrep -af pytest` first). Never Read > 300 lines at once. Budget ≤ 40 calls.
RESULT → `docs/orb_parity_defects_spec_20261103_RESULT.md` (what changed, the re-rendered 11/2 parity lines, the 10/30
parse cause, the P1 writer and the proposed cron line, gap-gate cause + affected-row counts + P&L delta, test counts,
suite count/time). Return ≤ 150 words. This task IS the owner's request.
