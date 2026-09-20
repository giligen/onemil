# FREEZE — frames10

TEST is `day >= 2026-06-01` on every book in this pass (HOD-break, ORB, BF-P1, the wrapper short).

It is **sealed**. It is not opened, loaded, printed, joined or scored at any point in pass 10.
Every loader in `frames10/` filters `split in ('TRAIN', 'VAL')` at read time, and the wrapper
universe panel is cut to `bar_date < 2026-06-01` before any signal or control is priced.

TEST may be opened exactly once, and only when a recommendation has been committed to git with the
owner's approval.

F33's historical replay is the one object in this pass that reads dates after 2026-06-01: it replays
the LIVE ramp streams (ORB's stage trades and the HOD-break dry run, 2026-09-14+). That is live
telemetry, not the sealed backtest split — no research cell is scored on it, and it is used only to
print what the pooled statistic would have read.

Splits (inherited, unchanged since pass 1):

| split | range |
|---|---|
| TRAIN | 2025-01-02 → 2025-12-31 (H1 < 2025-07-01, H2 >= 2025-07-01) |
| VAL | 2026-01-01 → 2026-05-31 |
| TEST | 2026-06-01 → **SEALED** |
