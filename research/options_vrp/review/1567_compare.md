# Independent-rebuild compare: cell 1567 family (PREREG_1567)

Builder: `cell_1567_cycles.csv` (1,461 cycles, 24 cells 1567–1590, no split column).
Rebuild: `rebuild_1567_cycles.csv` (2,124 cycles, same 24 cells keyed by delta/width/mgmt/gate, split=TRAIN/VAL only — no TEST, consistent with VAL not yet cleared).

Keyed by (cell, entry_date), mapping rebuild's (cell_delta, cell_width, cell_mgmt, cell_gate) to the builder's cell numbers via the builder's own combo table (verified 1:1, 24/24 combos match).

## Headline numbers
- **Cycle-set Jaccard** (same cell+entry_date+expiry+short_strike+long_strike): **0.383** (992 / 2,593 union). Only 1,394 of the (cell, entry_date) keys are common at all (builder 1,461, rebuild 2,124 — rebuild has 730 keys the builder doesn't, builder has 67 the rebuild doesn't); of those 1,394 common keys, only 992 (71%) picked the identical expiry+strikes.
- **Share of common cycles with |Δpnl| ≤ $5**: **569 / 1,394 = 40.8%**.
- **TRAIN-selected cell**: builder = **1574** (Delta=0.15, Width=$10, Mgmt=B, Gate=ON≥15%, TRAIN Sharpe 7.60). Rebuild's own TRAIN screen picks **delta=0.3, width=$5, mgmt=B, gate=gate15 = builder's cell 1586** — a **different cell**. **NOT A MATCH.**
- **VAL mean monthly return, cell 1574** (the builder's selection, both builds re-scored on it for comparability): builder = 0.01553/mo (n=10 VAL months), rebuild = 0.02522/mo (n=10 VAL months, pnl_usd / B=$6,500) → **diff (builder − rebuild) = −0.0097/mo**.

## Dominant cause of the 10 largest |Δpnl| cycles
Ranked (all in cells 1583–1586, the wide/high-delta corner, plus one 1567 and 1585/1584 repeats):

| rank | cell | entry_date | |Δpnl| | cause |
|---|---|---|---|---|
| 1,2 | 1586,1585 | 2025-11-17 | $3,343 | **sizing**: contracts 2 (builder) vs 9 (rebuild); strikes also differ (655/650 vs 649/644) |
| 3,4 | 1584,1583 | 2025-11-17 | $2,604 | same as above (profit_50 exit) |
| 5,6 | 1585,1586 | 2026-02-17 | $1,082 | **expiry/strike selection**: rebuild rolls to 2026-04-02 vs builder's 2026-03-31, same contracts (2 vs 2) — not sizing |
| 7 | 1567 | 2025-07-28 | $439 | **fill standard**: identical strikes/expiry, but the profit-target trigger fires a day apart (7/31 vs 8/1) — option convexity turns a 1-day fill-timing difference into a sign flip |
| 8 | 1583 | 2026-02-02 | $414 | same fill-standard mechanism (2/13 vs 2/17 exit) |
| 9 | 1585 | 2024-06-17 | $371 | mixed: different expiry (2024-08-02 vs 2024-07-31) + contracts (2 vs 3) |
| 10 | 1584 | 2025-04-07 | $364 | mixed: different strikes + contracts (2 vs 1), different exit rule fired (profit_50 vs stop_2x_credit) |

**Dominant cause = SIZING.** The four largest rows (67% of the top-10's summed |Δpnl|, ~$11.9K of ~$14.7K) share one root cause: the two builds compute `contracts` from different formulas. Builder: `size_position()` = `floor(alloc_dollars / ((width−credit)*100))` on a per-cycle `alloc_dollars`. Rebuild: `floor((B / N_SLOTS) / risk_per_contract)` — divides the $6,500 budget by a concurrent-slot count before sizing. On the 2025-11-17 entries the two formulas diverge by 4.5× (2 vs 9 contracts), which alone explains the $3.3K swings since `pnl` scales linearly with `contracts`. A secondary, independent cause — **fill-standard / exit-day divergence on identical cycles** (rows 7, 8, both `same_cycle=True`) — shows the daily mark-to-market convention (which price series decides the profit-target/stop trigger day) is itself not reproduced, turning identical trades into opposite-signed outcomes. A third, minor cause is **expiry/strike selection** drift (rows 5,6,9,10) where the two builds pick a different Friday/strike off the option chain for the same nominal cycle.

## Verdict
NOT in agreement: cycle-set Jaccard 0.38 (well under a reasonable ≥0.9 parity bar), only 41% of common cycles pnl-match to $5, and the TRAIN-selected cell differs (1574 vs 1586) — the independent rebuild does not reproduce the frozen builder's PREREG_1567 result. Root-caused to the sizing formula (`alloc_dollars` vs `B/N_SLOTS`) as the dominant driver of the largest dollar errors, plus a separate fill-standard (exit-timing) bug on cycles that do match on strikes/expiry. Neither cell 1574's VAL read nor the cell selection can be trusted until the sizing formula and the exit-day fill convention are reconciled to one spec and re-run.
