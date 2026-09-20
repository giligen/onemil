# frames17 — FREEZE

**TEST is `day >= 2026-06-01`. It is NOT opened in this pass, under any result.**

Splits (unchanged from frames16 PLAN §1): TRAIN `2025-01-02 … 2025-12-31` · VAL
`2026-01-01 … 2026-05-31` · TEST `2026-06-01 … 2026-09-11`.

Base state: `frames16` at commit `a01c103df0bac1f2415e3f2fa7089180443c2773` (the frame this pass
answers). Working tree at the start of this pass: `ec1dfd0361e79da36af6eb33f90dd971f9cffda3`.

How TEST is kept sealed in this pass:

* `passive_walk.py::population()` filters `frames16/sw_*.csv` to `day < TEST_FROM` (`2026-06-01`)
  before any other operation — the same cut frames16's own `sw_*.csv` builder already applied at
  signal-generation time (`short_walk.py: h = h[(h.day < TEST_FROM) & …]`), so a TEST row cannot
  exist in the source file at all. This pass adds a second, redundant filter on top of that for
  defense in depth.
* The entry walk (`passive_walk.py`) reads `bars_sip.db` only for the days present in the filtered
  population — it never queries a day `>= 2026-06-01`.
* The NBBO measurement (`nbbo17.py`) only requests legs whose `(day, symbol, exit_m)` came from the
  above walk, so it inherits the same cut.
* `score17.py` re-applies `day < '2026-06-01'` before scoring as a third, redundant guard, and prints
  the max day scored so the guard is auditable in the log, not just asserted here.

**Declared exception: none.** If one becomes necessary it is logged here, in writing, BEFORE it is
taken.
