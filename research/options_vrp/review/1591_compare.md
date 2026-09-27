# Independent-check comparison: cell_1591_cycles.csv (builder) vs rebuild_1591_cycles.csv (rebuild)

Keyed by `cell + entry_date` (`monday` in the rebuild file); cycle identity for set comparison is
`(cell, entry_date, expiry, short_strike, long_strike)` — short/long strike parsed from the rebuild's OCC
option symbols (`SPY240328P00492000` → 492.0).

## Population sizes
- Builder: 267 rows total, **no void rows are written at all** (the file only records cycles that actually traded).
- Rebuild: 761 rows total, 320 flagged `void=True` (skipped Mondays — no usable tick fill). 441 non-void rows.
- The rebuild carries **65% more "valid" cycles** than the builder (441 vs 267) even after dropping its own void
  rows — the two implementations disagree materially on which Mondays are tradable, not just on price.

## 1. Cycle-set Jaccard (expiry, short_strike, long_strike), keyed by cell+entry_date
- Builder set: 267, rebuild (non-void) set: 441, intersection: 231, union: 477.
- **Jaccard = 231 / 477 = 0.484.**
- 2 builder cycles have no rebuild counterpart at all (same cell/date, no matching non-void rebuild row);
  210 rebuild non-void cycles have no builder counterpart — the rebuild trades many Mondays the builder never
  entered (or entered on a different strike pair) at all.

## 2. Share of common cycles with |Δpnl| ≤ $5
- Of the 231 matched cycles, 135 have `|builder_pnl − rebuild_pnl| ≤ $5`.
- **Share within $5 = 135 / 231 = 0.584.**

## 3. TRAIN-selected cell match
- **Builder** (per `RESULT_1591.md`, frozen rule: highest monthly Sharpe, n_cycles≥12, green months≥55%):
  only cells **1597** and **1593** clear the eligibility bar; **1597** wins (Sharpe **−0.04**, n=23, green 70.6%).
- **Rebuild** (per `REBUILD_1591.md`'s own TRAIN table, same nominal rule): cells 1591/1592 fail the 55% green
  floor, and among the rest **cell 1594** wins on Sharpe (**1.16**, n=29, green 72.7%) — 1597 in the rebuild's own
  table has Sharpe 0.62, n=72, green 83.3%, and is NOT selected.
- **Result: NO MATCH.** Builder selects 1597 (Δ0.30, M=B, gate=none); rebuild selects 1594 (Δ0.20, M=B,
  gate=ivgate). `same_selected_cell = false`.
- Root cause of the selection flip: the two files disagree on how many Mondays count as "cycles" per cell for the
  n_cycles/green-month denominators (rebuild's TRAIN table shows n_cycles=72 for the `gate=none` cells vs the
  builder's 23–25 for the same cells) — a VOID/eligibility-counting divergence, not a pricing divergence, since
  the winning cells differ on the *gate* axis (none vs ivgate), which only the VOID/eligibility rule touches.

## 4. VAL mean monthly return on the (builder-)selected cell 1597
- Builder VAL (`RESULT_1591.md`): mean monthly return on B = **3.17%/mo** (n=18 cycles, Sharpe 4.25, green 78.6%).
- Rebuild VAL (`REBUILD_1591.md`'s table, same cell 1597 row): mean monthly return on B = **3.64%/mo** (n=59
  cycles listed, Sharpe 1.42, green 92.9%).
- **Difference (builder − rebuild) = 3.17% − 3.64% = −0.47 percentage points/month** (≈ −$31/mo on B=$6,500;
  ≈ −13% relative to the builder's own number). Sharpe also disagrees sharply (4.25 vs 1.42) driven by the same
  cycle-count divergence as #3 (18 vs 59 VAL cycles counted for the identical cell).

## 5. Dominant cause of the 10 largest |Δpnl| cycles (of the 231 matched)
| cell | entry_date | expiry | strikes | builder pnl | rebuild pnl | Δ |
|---|---|---|---|---|---|---|
| 1592 | 2026-05-26 | 2026-07-17 | 713/703 | 203.88 | −41.06 | 244.94 |
| 1591 | 2026-05-26 | 2026-07-17 | 713/703 | 203.88 | −41.06 | 244.94 |
| 1596 | 2025-03-03 | 2025-04-17 | 575/565 | −545.12 | −329.06 | −216.06 |
| 1595 | 2025-03-03 | 2025-04-17 | 575/565 | −545.12 | −329.06 | −216.06 |
| 1591 | 2026-06-22 | 2026-07-31 | 718/708 | −153.12 | 51.94 | −205.06 |
| 1591 | 2025-02-18 | 2025-04-04 | 584/574 | −240.12 | −80.06 | −160.06 |
| 1596 | 2025-03-31 | 2025-05-16 | 532/522 | −298.12 | −145.06 | −153.06 |
| 1595 | 2025-03-31 | 2025-05-16 | 532/522 | −298.12 | −145.06 | −153.06 |
| 1595 | 2026-07-13 | 2026-08-28 | 735/725 | −104.12 | −245.06 | 140.94 |
| 1592 | 2025-04-14 | 2025-05-30 | 495/485 | 109.88 | −27.06 | 136.94 |

**All 10 are Management-A cells (1591/1592/1595/1596) and all are `exit_reason=stop` exits — zero of the top 10
are on Management-B cells (1593/1594/1597/1598), including the selected cell 1597.** Entry credit matches exactly
between builder and rebuild in every row checked (e.g. 1.9399999999999995 for both on 1596/2025-03-03) — the
divergence is entirely in the **stop-exit mark**, not the entry fill. Builder and rebuild disagree, sometimes in
sign, on the P&L realized at the 2×-credit stop trigger day.

**Dominant cause: management precedence** — specifically, how the Management-A stop day's exit price is computed
(the builder's original method vs the rebuild's Amendment-2a daily-close-proxy-for-marks / tick-trade-for-fill
method) differs enough to flip sign on some stop exits. Because the TRAIN/VAL-selected cell is Management-B
(hold-to-expiry, deterministic settlement from strikes+credit), this specific mismatch does not touch the
frozen cell's own cycle-level pnl — but it is the reason 1591/1592/1595/1596 (the A-management neighbours used in
the rebuild's neighbour-check) cannot be trusted at face value in either file. The secondary, larger-scale
cause of disagreement (Jaccard 0.48, selection flip) is the **VOID rule**: the two implementations disagree on
which Mondays are tradable at all (441 vs 267 valid cycles; n_cycles 72 vs 23–25 for the same cells in the
TRAIN table), which is what actually moves the TRAIN-selected cell from 1597 to 1594.

## Verdict
Agreement is weak: Jaccard 0.48 (below any reasonable independent-rebuild bar), only 58% of matched cycles agree
within $5, and **the TRAIN-selected cell itself does not replicate** (1597 vs 1594) — both because of a VOID/
eligibility-counting divergence (dominant, population-level) and a Management-A stop-mark divergence (dominant,
per-trade). Per `REBUILD_1591.md`'s own verdict, its selected cell 1594 FAILS the VAL pass bar outright (1.61%/mo
vs 4% required) and fails on a 35.6% VOID share versus Amendment 2a's own 10% VOID rail. Neither file's numbers
should be relayed as a passing result without resolving the VOID-rule disagreement first.
