# Compare -- builder vs rebuild, PREREG_1617 Frame B (cell 1619, burst fade)

Builder: `cell_1619.py` / `cell_1619_fills.csv` / `RESULT_1619.md`
Rebuild: `rebuild_1619.py` / `rebuild_1619_fills.csv` / `REBUILD_1619.md` (built from PREREG prose only, per the note in REBUILD_1619.md the builder files were not opened)

## Bar (frozen in PREREG_1617.md): fill-set Jaccard >= 0.98, >= 99% of common fills within 0.01 R_f

**FAIL on both legs.**

* Fill-set Jaccard (keyed on day, symbol; cell 1619 filled==True vs rebuild status=='fill'):
  builder n=3750, rebuild n=3864, intersection=3750, union=3864 -> **Jaccard = 0.9705** (bar 0.98).
  Every builder fill is a subset of the rebuild's fills; the rebuild alone has 114 extra fills
  (rows the builder resolved to `no_fill`/`no_fill_gap`/`no_tape` on the entry search).
* Common fills (n=3750, split assignment agrees on 100% of them) with net_R_f within 0.01:
  **7 / 3750 = 0.19%** (bar 99%). Median |diff| = 0.445 R_f, mean = 0.644 R_f, max = 7.41 R_f.

Passing cells: **builder reports FAIL for cell 1619 on VAL** (RESULT_1619.md, pass bar not met);
**rebuild also reports FAIL** (REBUILD_1619.md verdict). Directionally both agree the frame is
dead, but they disagree by an order of magnitude on how negative it is (builder VAL mean net R_f
-0.075 vs rebuild VAL mean net R_f -0.600) -- **the passing/failing verdict matches, the
magnitude does not**, so this is not a clean independent-check pass; per CLAUDE.md item 1 the
reimplementation bar (Jaccard/tolerance) governs, and it fails.

## Dominant cause of the 10 largest differences

All 15 largest-diff rows checked (PBT, NGNE, CTEV, PVLA, IESC, DRUG, ANAB, MAZE, ZBIO, AMR, GRDN,
MBX, ...) are **cost-methodology, not price-walk, divergence**:

* Stripping the rebuild's cost model out (comparing builder's `net_Rf` to rebuild's
  **gross** `raw_pnl / R_f`, i.e. no entry_cost/exit_cost/borrow) collapses the mismatch:
  agreement within 0.01 jumps from **0.19% -> 79.1%** (2967/3750), median |diff| from 0.445 ->
  ~2e-6. The exit_reason/why field itself agrees on 91.4% of common fills. So the price walk
  (entry trigger, stop/retest resolution, exit price) is largely consistent; the R_f-normalized
  P&L is not, because of how cost is charged.
* Mechanism: cell_1619.py's `cost_and_r` charges **no separate entry cost** (the through-print
  limit at `level*(1+OFFER_BPS)` already embeds the entry cost in the fill price) and **no exit
  cost at all on a cover/retest exit** -- only stop-exits get `SLIP_STOP_BPS` slippage, plus
  borrow on every row. rebuild_1619.py's `simulate_short` charges `entry_cost = half_entry`
  (a separately measured half-spread) on every fill **and** `exit_cost = half + SLIP_BP*exit_price`
  on cover/retest exits, on top of the raw price P&L from the same walk -- i.e. the rebuild adds
  a spread charge on both legs that the builder never applies to a cover exit, and applies to
  entry that the builder folds into the fill price instead.
* Because this population's `R_f` (stop-to-entry distance) is tiny -- median R_f is **~0.6% of
  price** in both files -- a modest, plausible per-trade cost (median rebuild entry_cost ~0.10%
  of price) still averages **~0.39x of R_f** (median `cost_total/R_f`), so the two cost
  conventions diverge by multiple R_f per trade even when the underlying exit price and exit
  reason agree almost exactly (e.g. PBT 2026-05-15: both walks resolve to a cover exactly at the
  28.68 target, builder net_Rf=+0.31, rebuild net_R_f=-7.10, entirely from rebuild's
  entry_cost=0.695 + exit_cost=0.581 landing on an R_f of 0.172).

## Verdict

Not an independent-check pass. The two builds substantively agree on the entry/exit mechanism
(91% exit-reason agreement, 79% gross-R_f agreement within tolerance once cost is stripped) but
diverge on which cost legs get charged and where -- exactly the "no double-charged slip" /
"R must exceed the spread" failure modes CLAUDE.md flags. Before any number from either file is
relayed to the owner, the PREREG's cost recipe needs to be reread and one convention picked
(cost embedded in the through-print fill + stop-only slip, per cell_1619.py, vs a separately
measured half-spread on both legs, per rebuild_1619.py) -- at this book's ~0.6%-of-price R_f, the
choice changes the sign and magnitude of every fill, not just the tails. The 114-fill entry-set
gap (rebuild finds fills the builder's entry search didn't) is a secondary, smaller-magnitude
issue worth a follow-up but is not what drives the 0.19% tolerance failure.
