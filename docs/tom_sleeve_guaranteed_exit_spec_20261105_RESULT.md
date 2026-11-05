# TOM guaranteed exit — RESULT 2026-11-05
`scripts/tom_sleeve.py`: `_process_exit` never uses MOC. Exit session: MOC window = no-op; late window
(15:57-15:59 ET) = MARKET per held symbol (qty = min(broker, ledger)). Owed exit (earlier session) = MARKET in the
first tick (MOC or late) + ERROR/Telegram "POSITION STILL HELD ... selling MARKET now". A live non-`-mkt` exit
order in the late window (DAY or CLS) is cancel-and-confirm then MARKET. Entry leg unchanged. New
`still_held_after_exit` + `alarm_still_held` (called in `main` after the fill checks, out-of-window ticks only):
ERROR + Telegram while the broker still holds a closed cycle (capped by ledger qty, so owner shares never alarm).
Module docstring updated. Tests: existing MOC-exit asserts converted; new tests for MOC-window no-op, per-symbol
MARKET qty, DAY/CLS safety net, owed next-session ERROR, 11/4 replay integration (flat, no MOC), alarms.
Result: tests/test_tom_sleeve.py + tests/test_exit_coverage_20261029.py = 80 passed, 0 failed.

## Addendum: exit session passed with NO exit row
New `missed_exit_sessions` (held symbol, latest entry row, no exit row since; exit session = 3rd session of the
month after the entry). `owed_exits` adds such symbols (`base` = coid of the exit date, ERROR logged) so `run` sells
MARKET in the first tick; `still_held_after_exit` (now takes `alpaca_client`) alarms once past the late window.
Tests: owed, not-yet-reached negative, alarm, run-sells. Suite: 84 passed, 0 failed.
