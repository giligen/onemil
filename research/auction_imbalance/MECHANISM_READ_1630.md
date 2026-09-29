# MECHANISM READ — cells 1,630-1,632, Amendment 3 (2026-09-28 21:15 UTC)

One free, pre-committed, mechanism-only read on the 56 sessions purchased before the pull was stopped ($20.96).
No money number, no trade P&L, no cost model, no fill simulation — decile tables only, all TRAIN.

**Data note (read this first):** `dbn_cache/imbalance.parquet` / `ref_quotes.parquet` on disk are STALE — built
18:50 UTC 9/28, before Amendment 1 moved `SAMPLE_START` to 2024-07-01 and widened the quotes pull from a `window`
column to a continuous `minute` column. Rebuilt here directly from the 112 `imbalance` + 111 `quotes` shard files
dated 2024-07-01..2024-09-18 (no `fetch_imbalance.py` run, no `databento` import — local parquet reads only).
`2024-07-03` (half day, day before July 4) purchased both venues but wrote zero rows in the normal 15:45-16:00 ET
window — 2 empty shards, correctly excluded, leaving **55 calendar sessions with any imbalance data**.

## Coverage
| stage | n | note |
|---|---|---|
| first-publication events (symbol,date,venue) | 16,838 | 55 sessions, 308 symbols |
| + matched reference mid | 16,536 (98.2%) | 302 events (1.8%) have no quote at the exact threshold minute |
| + panel adv20/close/next_open (usable) | 13,790 (83.4% of above) | drop concentrated in 9 sessions, see caveat below |

Venue share (usable events): **XNYS.PILLAR 54.6% / XNAS.ITCH 45.4%** (pre-panel: 54.6% / 45.4%, essentially identical).
First-publication ET time-of-day: XNYS.PILLAR min 15:50:00.0004, median 15:50:00.879; XNAS.ITCH min 15:55:00.0003,
median 15:55:00.088 — both venues fire almost exactly on their threshold, no timestamp drift.

## Halves
Defined on the **55 calendar sessions on disk** (first 28 / last 28, sharing one boundary session since 55≠56):
FIRST_28 = 2024-07-01..2024-08-09, LAST_28 = 2024-08-09..2024-09-18. The panel's adv20/dvol20 is NaN for its own
first ~2 weeks (trailing-20-session warm-up from its 2024-07-01 start) — **only 19 of the 28 FIRST_28 sessions
contribute any usable event** (first usable session 2024-07-16) vs 27 of 28 for LAST_28. This is a coverage gap,
not a random split — FIRST_28 is thinner and noisier by construction, not "a different regime."

## Decile tables (mean bps; decile 1 = smallest \|I\|, 10 = largest; deciles fit ONCE, pooled, applied to both halves)
| decile | n pooled | close move, POOLED | FIRST_28 | LAST_28 | next-open reversal, POOLED | FIRST_28 | LAST_28 |
|---|---|---|---|---|---|---|---|
| 1 | 1,379 | 0.39 | 0.47 | 0.34 | -1.04 | -5.87 | 2.27 |
| 2 | 1,379 | 1.85 | 2.94 | 1.11 | -2.44 | -3.68 | -1.60 |
| 3 | 1,379 | 2.39 | 3.87 | 1.42 | -10.70 | -28.34 | 0.94 |
| 4 | 1,379 | 2.16 | 1.83 | 2.38 | -1.35 | -2.76 | -0.39 |
| 5 | 1,379 | 3.25 | 3.35 | 3.19 | -6.31 | -10.15 | -3.66 |
| 6 | 1,379 | 1.78 | -0.60 | 3.32 | 6.21 | -1.89 | 11.42 |
| 7 | 1,379 | 4.12 | 2.39 | 5.36 | -24.19 | -32.33 | -18.33 |
| 8 | 1,379 | 4.67 | 5.11 | 4.32 | -6.86 | -2.08 | -10.62 |
| 9 | 1,379 | 4.85 | 4.23 | 5.33 | -22.04 | -32.59 | -13.81 |
| 10 | 1,379 | 5.65 | 6.43 | 4.97 | -14.89 | -22.31 | -8.35 |

**Sign convention** (flagged for the independent check): next-open reversal is signed IDENTICALLY to the closing
move (aligned with the imbalance side), per the task spec's "signed the same way" — NOT `cell_1630.py`'s convention
(which negates it so positive always means "reversion"). Under this convention the mechanism predicts closing move
**> 0, rising** in \|I\|, and next-open reversal **< 0, falling** (more negative) in \|I\|.

## Top-minus-bottom decile (day-clustered t) and decile-mean monotonicity (Spearman rho of decile vs decile-mean)
| split | n | close move Δ(10-1) bps | t | rho | next-open rev Δ(10-1) bps | t | rho |
|---|---|---|---|---|---|---|---|
| POOLED | 13,790 | +5.26 | **2.63** | 0.867 | -13.85 | -1.60 | -0.612 |
| FIRST_28 | 5,761 (19 sessions) | +5.96 | 1.46 | 0.612 | -16.44 | -0.94 | -0.261 |
| LAST_28 | 8,029 (27 sessions) | +4.63 | **3.61** | 0.915 | -10.62 | -1.28 | -0.673 |

Correlation sign of \|I\| with closing move, per venue (row-level, pooled): XNAS.ITCH pearson +0.053 / spearman
+0.067 (n 6,262); XNYS.PILLAR pearson +0.044 / spearman +0.053 (n 7,528) — **same positive sign both venues**,
small magnitude (expected: one event's idiosyncratic move dominates the raw row-level correlation; the decile-mean
table above is the cleaner read).

## Read against Amendment 3's own pre-committed bar ("monotone table, \|t\| ≥ 3 on both halves")
**Does not clear it.** Closing move: right sign and decile-monotone (rho 0.61-0.92) in both halves and pooled;
significant pooled (t 2.63) and in LAST_28 (t 3.61) but NOT in FIRST_28 (t 1.46) — FIRST_28's shortfall tracks its
9-session data gap, not a sign flip. Next-open reversal: right sign in both halves and pooled (always negative) but
weak everywhere (\|t\| ≤ 1.6, rho as low as -0.26 in FIRST_28) — the reversal leg is the weaker half of the
mechanism on this population. Both halves agree in SIGN and rough monotonicity for both metrics; neither clears
\|t\| ≥ 3 on both halves. Closing move is the closer call and the shortfall is largely a coverage artifact;
next-open reversal (the actual P&L leg for 1,630/1,631) is not yet supported at this sample size.

## Caveats
Row-level Spearman (not shown per-decile) is much weaker than decile-mean rho — large idiosyncratic per-event noise,
expected. 302 first-pub events (1.8%) have no matching reference-mid quote, dropped, not imputed. Venue assignment
inherited as-is from the FETCH stage's `pit_listings.py`-based `venue` column, not independently re-verified here.
`side='N'` (no imbalance) events are included at abs_I=0 (fall in decile 1), per `cell_1630.py`'s documented
convention. Output: `research/auction_imbalance/mechanism_rows_1630.csv` (13,790 rows, one per usable symbol-session).
