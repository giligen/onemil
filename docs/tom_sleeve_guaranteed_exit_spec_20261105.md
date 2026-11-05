# TOM sleeve: guaranteed exit — spec 2026-11-05 (owner: "make sure we sell these 3 before exit no matter what")

## Why (supersedes part 2 of `tom_sleeve_moc_fallback_spec_20261105.md`)
Paper MOC (CLS) is NOT reliable for exits even on QQQ: `tom-202610-QQQ-out` expired unfilled 10/5 AND `-r1` 10/6
(25/26 partial), so with "leave a live MOC alone" an expired MOC carries the position overnight. The owner wants the
exit certain. A paper MARKET order fills in seconds (11/4: 1–8 s). Cost of certainty: ~6 bps vs the close (−$37 on
11/4), measured and accepted.

## Rule (exit leg only — the ENTRY leg is unchanged by this spec)
1. On the exit session, `_process_exit` NEVER uses MOC: in the MOC window (15:40–15:55 ET) it does nothing; in the
   late window (15:57–15:59 ET) it submits MARKET sells for the owed qty of every held symbol (coid `<base>`; a dead
   chain continues `-rN` as now). Remove `use_moc` from the exit path; `MOC_RELIABLE_SYMBOLS` keeps governing entries.
2. `is_live_cls` / "left alone" stays for the ENTRY leg (a live MOC BUY at 15:58 is harmless if it expires — the
   catch-up re-enters or the cycle is skipped). On the EXIT leg a live non-`-mkt` order in the late window is
   cancelled-and-replaced by MARKET exactly as before the 11/5 change (there should be none now, so this is a safety net).
3. Catch-up (`owed_exits`) is unchanged: a missed exit session sells MARKET in the first late-window tick of the next
   session (and in the MOC-window tick too — an owed exit must not wait for 15:58 a second day: submit MARKET in
   whichever tick first sees it).
4. Alarm: at every tick after the exit session's late window (the same day's later ticks, and the next sessions'
   ticks) if the broker still holds a tom position for a closed cycle → `logger.error` + Telegram
   "[TOM] POSITION STILL HELD after the exit session: SYM qty — selling MARKET now / outside market hours, will sell at
   the next tick". ERROR level, so it reaches the owner through the Telegram error handler.

## Tests (`tests/test_tom_sleeve.py`)
- exit session, MOC-window tick: no orders for any symbol (positions held).
- exit session, late-window tick: one MARKET sell per held symbol, qty = min(broker, ledger).
- exit session, late-window tick with a live DAY/CLS exit order (safety net): cancel-and-confirm then MARKET.
- owed exit on the NEXT session, MOC-window tick: MARKET sold immediately + ERROR logged.
- integration (tempdir ledger, `MagicMock(spec=AlpacaClient)`): replay the 11/4 sequence → ends flat with exactly one
  MARKET order per symbol and no MOC order ever submitted on the exit leg.
- Update any existing test that asserted an exit MOC. Zero failures in `tests/test_tom_sleeve.py` and
  `tests/test_exit_coverage_20261029.py`.

## Rules
Never submit orders or run the script against the API; no git commit; no crontab/service changes; no writes to logs/
or data/. `--ignore=tests/integration`. Budget ≤ 35 calls. RESULT → `docs/tom_sleeve_guaranteed_exit_spec_20261105_RESULT.md`
(≤ 150 words); return ≤ 100 words.
