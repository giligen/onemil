# MOM sleeve — Monday 2026-11-02 readiness RESULT (2026-10-31)

Spec: `docs/mom_sleeve_monday_readiness_review_20261031.md`. Rehearsal output: `docs/mom_sleeve_monday_readiness_review_20261031_PLAN.md`.
Nothing committed. No orders, no `--submit`, no state/ledger/config/crontab writes.

## Ranked findings (file:line refer to `scripts/momentum_sleeve.py` after the edit)

### Defects that lose money or skip a rotation
1. **No catch-up (FIXED, mandatory).** The old `is_rebalance_session` (l.102) was "first session of the ISO week" only,
   so Tue 10/27 printed `not a rebalance session — no-op` after the 10/26 crash. New `rotation_due` (l.108–135): DUE iff
   today is a session AND the state's `last_rebalance` (already the completed rotation's DATE, so no new field and
   old state files keep working) is not in today's ISO week AND today is Mon–Wed. INFO
   `rotation due: catch-up (last 2026-10-19, week 44)` / `rotation due: first session of week 45`. `run()` (l.968–975)
   reads the mode's own state file (paper or live) before the check. `--force` unchanged (skips the check). Proven equal to
   the old rule on every first session of 2026 (parity test) and on 11/2.
2. **Cron runs Mon+Tue only (`* * 1,2`) — LEFT, needs the main session/owner.** A Wednesday catch-up never fires from
   cron. To use the full Mon–Wed window, change the five MOM cron lines to `1-3`. Not edited (crontab off-limits).
3. **LIVE same-day re-run double-booked fills (FIXED).** `execute_live` booked whatever `poll_fills` returned. After a
   crash once the sells were booked and saved (`on_phase`) but before the marker was saved, the 14:46 re-run planned
   again from the half-updated state. Any delta on a name already traded got the SAME client order id,
   `submit_market` returned the existing order, and its fill was booked a SECOND time (doubled qty and cash move). The
   broker never double-buys because ids are idempotent, but the slice's state goes wrong. Fix: `state['booked_coids']`
   holds today's booked ids, pruned per day. A fill whose id is already booked is skipped with a WARNING (l.808–819).
4. **LIVE marker saved late (FIXED).** `last_rebalance` was set right after `execute_live` but saved only after the ledger,
   official-opens and marks code (old l.871→894). It is now saved at once, like paper (l.900–903).
5. **LIVE re-run after a crash in the buy poll refused its own buys (FIXED).** The buying-power check counted buys that
   were already at the broker, whose cost had already come out of buying power. Result: `NO buys this rotation`, and
   those fills were never booked. The broker shares would then be invisible to the sleeve and would look like the
   owner's. Fix: the check now counts only buys whose id is not already at the broker (`_order_exists`, l.498; l.769–779).
6. **LIVE poll timeout — LEFT.** An order still open after 180 s is logged ERROR and is not booked. The marker is
   saved anyway, so a later fill is never reconciled into `state_live.json`. Paper heals itself through
   `sync_state_from_broker`. This is rare for market DAY orders at 09:46 ET. Next change: a reconcile pass that books
   any filled `moml-<today>` id missing from `booked_coids`.
7. **Paper:** `last_rebalance` is written even when every order was REJECTED (`execute` catches per order). That week
   then has no retry. The cause is ERROR-logged, so this is noted, not changed.

### Defects that misreport
8. **EST pre-open crash ERROR (FIXED).** 11/2 is the first EST rotation (DST ended 11/1). The 13:45/13:46 UTC lines run at
   08:45 ET, where `check_submit_window` raised and `run_guarded` would have sent two `momentum_sleeve crashed
   (RuntimeError)` ERRORs to Telegram. A due session before 09:31 ET is now an INFO no-op, and the 14:45/14:46 lines
   submit (l.976–982). Submitting after 15:30 ET or on a non-session still raises.
9. **LIVE refusal labelled PAPER (FIXED).** The completeness-refusal Telegram used the default `[MOM]` prefix and the ERROR
   line said `MOM REFUSED` on the live path. Now `[MOM LIVE] … REFUSED` and `MOM LIVE REFUSED` (l.1053–1057). Paper text unchanged.
10. **LEFT (cosmetic/misreport):** other ERRORs that reach Telegram through the root `TelegramErrorHandler` carry no
    PAPER/LIVE tag (same logger name on both paths). The two state paths never mix; only the label is missing.

### Checked and clean
- **TXSE path:** `fetch_assets` → `get_assets_tolerant` (raw-REST fallback, seen working on both accounts today);
  `apply_fractionable` → `get_asset_tolerant`. grep for `get_all_assets|GetAssetsRequest|.exchange` in the script,
  `trading/momentum_sleeve.py`, `trading/momentum_live.py` and `monitoring/cron_guard.py` finds no other pydantic asset parse
  on the Monday path. Universe → `fetch_panel` (LOST counted, CUSIP pre-filter) → completeness/LOST gates → selection → plan.
- **Idempotency:** client order id = `mom-|moml-<YYYYMMDD>-<SYM>-<s|b>` (plus a run stamp only under `--force`). Paper
  re-derives deltas from the broker (sync before the plan), saves the marker straight after `execute`, and a 14:45 re-run
  after a success is a no-op. It is now stopped earlier (`rotation already done this week`) as well as by `already rebalanced`.
- **Live isolation:** sells are capped at the state qty and at what the broker holds (`ml.sell_quantity`). Names not in the
  state are never sold. Buys are sized from slice equity (state cash plus marks of its own names only). Collisions
  (names the owner holds) lose their buy. The rail and `equity_start` come from the state (10,000). `paused` is absent
  (false) in both state files.
- **Gates:** completeness < 0.98 liquid or LOST > 0.05 → ERROR + Telegram, `return 2` before any save (state untouched).
  Rail 0.75 × start, kill 0.40.
- **Stale two-week book:** there is no turnover or notional cap that could refuse a catch-up. `min_trade_usd` is $5.
  Sizing uses broker marks at submit time, not last week's. The rehearsal turnover is modest (3 in / 3 out).

## Tests
- New `tests/test_mom_sleeve_readiness_20261031.py`: **15 tests**, all passing.
  - 7 unit tests of `rotation_due`: 11/2 due as before; Tuesday after a crash = catch-up with the exact log reason;
    Thursday/Friday no-op; already-done week; Monday holiday; unreadable marker; parity over every 2026 week.
  - 8 integration tests through `run()` on tmp state files:
    - paper Tuesday catch-up plans 20 orders;
    - paper Thursday no-op;
    - paper 11/2: the 13:45 UTC line is an INFO no-op with no ERROR, the 14:45 line rotates, and the retry is a no-op;
    - after-close submit still refused;
    - LIVE crash in the buy poll, then a re-run: books all 20, re-submits nothing, buying power 0;
    - LIVE re-run never double-books;
    - LIVE marker on disk before the ledger code;
    - LIVE refusal labelled `[MOM LIVE]` with the state untouched.
- Two existing tests adjusted to the new rule. `test_non_rebalance_day_is_noop` now uses a tmp state; it read the real
  `data/` file before. The C2 re-run assertion now reads `already done this week`.
- Momentum files: 102 passed.
- Full suite (`timeout 2700 python3 -m pytest tests -x -q --ignore=tests/integration -p no:randomly`): **5441 passed, 6 skipped, 0 failed** (602 s).

## Rehearsal summary (asof 10/30, real API, dry)
- Both runs: 0 ERROR.
- LOST 310 (2.3 %); liquid 770/770 (100 %); gate OFF (p83), size 100 %.
- 20 names, identical for paper and live. OUT DINO / IOVA / WDC, IN HPE / MRNA / MXL. 22 orders each.
- Paper: equity $16,905, target $845/name. Live: equity $9,447, target $472/name, rail $7,500 not hit, buying power OK.
- No refusal.

## Verdict
**READY for 11/2.** Monday is the first session of week 45, so the rotation is due under both the old and the new rule. The rehearsal on real data passes every gate with 0 ERROR. The two Telegram crash ERRORs on the EST pre-open lines and the LIVE re-run booking defects are fixed and tested. Open: cron `1,2` (Wednesday catch-up), the LIVE poll-timeout reconcile, and committing these edits plus a boot-style rehearsal on the exact cron commands (main session).
