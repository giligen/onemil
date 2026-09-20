# PREREG — recalibration of BF/ORB honest books at realized entry cost (2026-09-20)

Diagnostic pass, not a ship gate. Method fixed before scoring below.

**Inputs**
- BF book: `research/mature_method/frames14/f45_bf_book.csv` (56-trade P1 regen-7 book,
  per-trade `shares`, `sp_e`/`sp_x` = frames14's measured full-spread fraction at the
  entry/exit minute, `entry_price` = booked fill at flat 50bps, `raw_fill = entry_price/1.005`).
  Formula reproduces the file's own `pnl_meas` exactly (validated: recomputed total
  $67,201 == file total $67,201).
- ORB book: `analysis_results/orb_bplus_book.csv`, using `_sized_pnl` (NOT `pnl`, which is
  15x sized notional) and `pnl_pct` to back out notional = `_sized_pnl / (pnl_pct/100)`.
- Live realized-vs-quoted entry bps: BF quoted 33.92 / realized 12.55; ORB quoted 13.50 /
  realized 3.08 (`REPORT_REALIZED.md` correction section, fill-time quote).

**Settings (entry leg, 3-way)**
- P = book's current measured/table half-spread (BF: `sp_e` as-is; ORB: `_sized_pnl` as-is —
  its existing cost model is treated as approximating the quoted 13.5bps median).
- M = midpoint(quoted, realized): BF 23.235bps, ORB 8.29bps.
- O = realized only: BF 12.55bps, ORB 3.08bps.
- Implementation: for BF, `sp_e` is rescaled per trade by `target_bps / median(implied entry
  bps)` — preserves the cross-sectional shape, shifts the level. For ORB, entry cost is a
  flat bps applied to notional (`notional * (target_bps - 13.5bps) / 1e4`), since no
  per-trade spread column exists there.

**Exit leg**
- ORB: unchanged (whatever `_sized_pnl` already charges) in all 3 settings.
- BF: `sp_x` (measured exit spread) halved in all 3 settings ("50% of current charge") —
  applied uniformly, does not vary with P/M/O.

**Splits**: TRAIN = 2025-01-01..2025-12-31, VAL = 2026-01-01..2026-05-31, TEST >= 2026-06-01
sealed (not opened). MDD = running cumulative-$ trough. R/trade: BF uses `R_dollar` (shares x
stop distance) from the file; ORB has no per-trade risk column — R approximated as
`pnl_pct / range_size_pct` (stop ~= range_low, entry ~= range_high, so range_size_pct is the
stop-distance proxy); flagged as approximate. t-stat: day-clustered (mean/day, then t on the
daily series). Green-week share: ISO week P&L > 0 / weeks with >=1 trade.

**Caveat pre-committed**: BF's own `sp_e`/`sp_x` units did not resolve cleanly to a bps
convention from the source alone; the rescale-to-target-median approach is a level-shift, not
a re-derivation from raw quotes — treat all BF numbers here as a bounded reproduction, not a
fresh measurement. ORB's flat-bps-on-notional treatment ignores intra-book spread dispersion.
