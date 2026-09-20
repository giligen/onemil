# FREEZE — frames11

TEST is `day >= 2026-06-01` on every book in this pass.

It is **sealed**. It is not opened, loaded, printed, joined or scored at any point in pass 11.
Every loader in `frames11/` filters `split in ('TRAIN', 'VAL')` at read time, and the control
panel is cut to `day < 2026-06-01` before any bracket is priced.

TEST may be opened exactly once, and only when a recommendation has been committed to git with the
owner's approval.

**The one object in this pass that reads dates after 2026-06-01** is F36's producer wiring: the EOD
check's dry-run book for the live dry-run sessions (2026-09-14 onward) is appended to
`data/hod_dry_pool.csv`. That is LIVE TELEMETRY, not the sealed backtest split — no research cell
is scored on it, exactly as F33's replay was exempted in `frames10/FREEZE.md`.

Splits (inherited, unchanged since pass 1):

| split | range |
|---|---|
| TRAIN | 2025-01-02 → 2025-12-31 (H1 < 2025-07-01, H2 >= 2025-07-01) |
| VAL | 2026-01-01 → 2026-05-31 |
| TEST | 2026-06-01 → **SEALED** |
