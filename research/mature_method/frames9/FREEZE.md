# FREEZE — frames9

TEST is `day >= 2026-06-01` on every book in this pass (HOD-break, ORB, BF-P1).

It is **sealed**. It was not opened, loaded, printed, joined or scored at any point in pass 9.
Every loader in `frames9/` filters `split in ('TRAIN', 'VAL')` at read time, and the pond
membership panels are cut to `bar_date < 2026-06-01` before any signal or control is priced.

TEST may be opened exactly once, and only when a recommendation has been committed to git with the
owner's approval. No recommendation was made by this pass.

Splits (inherited, unchanged since pass 1):

| split | range |
|---|---|
| TRAIN | 2025-01-02 → 2025-12-31 (H1 < 2025-07-01, H2 >= 2025-07-01) |
| VAL | 2026-01-01 → 2026-05-31 |
| TEST | 2026-06-01 → **SEALED** |
