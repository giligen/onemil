# ORB universe gate test order-dependence (2026-11-11)

Leaker: `tests/test_forward_ledger.py` calls `research/day_breadth/forward_ledger.py::universe_for_day`, which
permanently overwrote `study_orb_broad.MIN_GAP_PCT` (5.0 -> 3.0), `MAX_OPEN_PRICE`, `DATE_START`, `DATE_END`.
`study_orb_features.builder_admission` reads `MIN_GAP_PCT` at call time, so AEYE's 3.07 % re-check passed.

Fix: `universe_for_day` saves and restores the four globals in a `finally`. Regression test added in
`tests/test_forward_ledger.py`. Full suite: 5712 passed, 0 failed.
