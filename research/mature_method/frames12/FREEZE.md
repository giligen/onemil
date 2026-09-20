# FREEZE — frames12

TEST is `day >= 2026-06-01` on every object in this pass.

It is **sealed**. It is not opened, loaded, printed, joined or scored at any point in pass 12.
Every loader in `frames12/` cuts `day < 2026-06-01` at read time, before a bar is walked and before
a level is computed, and every control panel is cut on the same key.

F39 holds a position overnight in one declared exit cell (`hold to the next session's open`). The
next session's open for a **2026-05-29 / 2026-05-30** signal therefore lands on a TEST-side date.
That is a **price**, not a split: the single daily `open` of the following session is read for those
signals, no TEST-side signal is ever generated, no TEST-side bar is walked, and no TEST statistic is
computed. The count of such trades is printed in the report. (The conservative alternative — drop
the last two TRAIN/VAL signal days of the overnight cells — is also printed as a sensitivity.)

TEST may be opened exactly once, and only when a recommendation has been committed to git with the
owner's approval.

Splits (inherited, unchanged since pass 1):

| split | range |
|---|---|
| TRAIN | 2025-01-02 → 2025-12-31 (H1 < 2025-07-01, H2 >= 2025-07-01) |
| VAL | 2026-01-01 → 2026-05-31 |
| TEST | 2026-06-01 → **SEALED** |
