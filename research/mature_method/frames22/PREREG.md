# frames22 — PREREG (written and frozen BEFORE any scoring)

**Cell F58 — the passive mirror short, OUT OF TIME: does it survive on 2024H2?**

F55 (frames18) walked a resting SELL limit at `ref x (1+k)` on the MIR2 mirror-short signal and
cleared its own bar at `k=1.0 %`. F57 (frames20) then measured the same book against **five matched
non-signal names per signal** (P1x5) and found the excess **+0.089 / −0.018 R (w=5, TRAIN/VAL)** and
**+0.128 / +0.177 R (w=10)**, all `t < 1.4`, MDE 0.27–0.93 R — i.e. the design could not see an
effect twice the size of the one measured. F57's own verdict named the binding constraint: **too few
signal fills**, not too few controls.

frames21 pulled an out-of-time extension — **2024-07-01 .. 2024-12-31**, 6,950 mirror rows
(3x the 2025 TRAIN count in half the time), 95.4 % NBBO leg coverage — using a **RECONSTRUCTED**
candidate universe, because the original `universe.csv` builder was not found in the repo.

**F58 asks**: on 2024H2 as a NEW, previously unopened out-of-time split, is `S − P1x5` for the two
F55 cells (`k=1.0 %`, `w in {5,10}`) at least `+0.10 R` with a day-clustered `t >= 2` and an
ex-top-5 %-of-the-difference `>= 0`?

Programme cell count: **1,277 → 1,279** (2 cells, both declared here, both scored). The brief
assigns these cells the numbers 1,278–1,279; the FRAMES.md ledger's last recorded count is
**1,270** (F57) — the discrepancy is recorded here, not silently reconciled. TEST (`>= 2026-06-01`)
is never opened; 2024H2 is BEFORE TRAIN, so no sealed period is touched either way.

---

## 0. GATE 0 — universe reconciliation. If it fails, EVERYTHING below is VOID.

frames21 rebuilt the candidate universe from the three documented thresholds
(`day high / open − 1 >= 0.05`, `close >= $1`, trailing-20 `adv20 >= 100,000`, strictly prior
sessions) over `research/multiday/data/prices_by_year/raw/year=2024.parquet`, because the script
that produced `research/bf_zero/universe.csv` no longer exists. The extension is only admissible if
that reconstruction reproduces the ORIGINAL universe where the two overlap.

**Test (pre-committed, run and reported BEFORE any cell is scored):**

1. Apply the reconstructed rule, byte-for-byte the same code, to **2025-01** (a month the original
   `universe.csv` covers), sourcing `year=2025.parquet` with a 2024 warm-up tail for the trailing
   `adv20` so the January window is fully warm.
2. Compare the resulting set of `(symbol, date)` symbol-days with `universe.csv` restricted to
   2025-01.
3. **Jaccard `|A ∩ B| / |A ∪ B| >= 0.95` → the extension is admissible.** `< 0.95` → the whole
   frames22 extension is **VOID** and no cell result may be quoted as evidence for anything.
4. Report, either way: Jaccard, `|A|`, `|B|`, the count and 10 examples of names **only in the
   reconstruction**, and the count only in the original.
5. **Density check (report-only, no bar):** monthly signal-row counts (gate5 & mirror flag) for
   2024-07 .. 2025-03 side by side, so a reader can see whether the extension's 3x row density is a
   real 2024H2 regime or an artefact of the reconstructed universe.

---

## 1. Populations

**S (signal)** — `frames21/signals_ext.csv` rows with
`gate5 & f_mir & price >= $5 & ex-wrapper & symbol not ~ ^Z[A-Z]ZZT$ & ETB`, days
2024-07-01..2024-12-31. `f_mir` (`frames15/armB_intra.py`: `hrv >= 3 & |hour_ret| > 2 %`) is
**byte-identical in definition** to frames16's `f_mir2` (`frames16/short_walk.py:70`) — verified
before writing this file; the name differs, the rule does not. Wrapper class via
`hod_frames5/common5.attach_instrument`, ETB via
`research/fuckup_audit/O_halt/PASSIVE/borrow_flags.csv` (`shortable AND easy_to_borrow`, absent =
not shortable) — **the same 2026 snapshot F55/F57 used, now applied to 2024 names; declared as a
caveat, not a fix.**

**P1x5 (control)** — frames20's matched non-signal control, `NMATCH=5`, construction unchanged:
for each S row, the pool is every symbol with 1-min bars that session that is NOT a signal name that
day, not a wrapper, not a test ticker, ETB, and inside the price bucket `|log(prev_close ratio)| <=
log(1.25)` and the ADV bucket `|log2(adv20 ratio)| <= 1`; five drawn without replacement (seed
**57**, unchanged), each walked at the SAME clock hour with the same entry/exit/cost stack, kept only
if it passes the same `gate5` and `price >= $5` rails.

## 2. Entry, exit, cost — unchanged from F55/F57

Resting SELL limit at `ref x (1+k)`, `k = 1.0 %`, placed at the close of the signal-hour cut bar
`e`, live for `w` bars, filling AT the limit on the first bar whose HIGH `>=` limit. No touch in the
window = UNFILLED, 0 P&L. Entry charged **zero**. Exit: `stop = entry x 1.02`, bare arm-3 walk from
fill bar + 1 via `frames16/short_walk.py::swalk` **imported verbatim**, charged
`0.5 x measured_spread / entry / rpct` in R from `frames21/nbbo_ext.csv`; legs with no measured quote
get the pooled **price-decile median relative spread** fallback, exactly as F57.

Reg SHO 201 rail as F55/F57: an SSR-active fill (session running-min low `<= prev_close x 0.90`) is
valid only if the limit is strictly above the NBB at the fill minute; NBB unavailable or SSR
undetermined → VOID fill.

## 3. Declared deviations from the frames18/frames20 code (input paths + two forced substitutions)

Everything else is the F55/F57 code unchanged. These are the only changes, all declared here BEFORE
scoring:

1. **Date range / input paths**: bars from `frames21/raw/bars_sip_ext.db`; signals from
   `frames21/signals_ext.csv`; quotes from `frames21/nbbo_ext.csv`.
2. **`ref` recomputed from the tape.** `signals_ext.csv` carries `entry` but not `ref`, so the
   signal-hour cut-bar close is recomputed with `frames20/walk20.py::hour_rows` — the SAME
   convention the CSV itself was built with. Not a rule change.
3. **Daily panel source.** `data/cache.db::daily_bars` holds essentially NO 2024 rows (1 symbol in
   2024-09), so `prev_close` (SSR rail) and `adv20` (matching bucket) come from the point-in-time
   panel `research/multiday/data/prices_by_year/raw/year=2024.parquet`, causal by construction
   (`shift(1)` close, trailing-20 mean of prior volume, `min_periods=15`). Same formulas as
   `frames20/walk20.py::daily_panel`, different vendor — a **price-scale hazard** is therefore
   possible between the daily panel and the Alpaca 1-min tape and is reported, not assumed away:
   the share of S fills whose `ref` deviates from the panel close by more than 10 % is printed.
4. **Split label.** 2024H2 is a single new split named `EXT`. TRAIN/VAL are NOT re-scored here.

## 4. Pre-committed pass bar (primary)

Per cell (`w in {5,10}`), on the EXT split:

* `S − P1x5 >= +0.10 R` (day-clustered difference of means), **AND**
* day-clustered `t >= 2.0`, **AND**
* ex-top-5 %-of-the-difference (each population trimmed separately at its own gross-`rr` 95th
  percentile, difference recomputed) `>= 0`, **AND**
* `|fill rate(S) − fill rate(P1x5)| <= 10 pp` — **otherwise the cell is VOID** and the fill- and
  touch-conditioned differences are reported in its place.

All four required → **PASS**. Anything else → **FAIL**. No cell is re-scored under a changed bar.

## 5. Secondary (report-only, decides nothing)

S alone: net R, ex-top-5 %, fills/week, green-week share vs the count-matched null, gross, measured-
quote coverage, SSR-void count. The **80 %-power MDE** (`2.8 x |diff| / |t|`) is printed for every
cell whether it passes or fails; a FAIL without its MDE is not a result.

## 6. What a PASS would and would not buy

A PASS on 2024H2 would say the F55 book's excess over matched non-signal names replicates out of
time at a size the design can see. It would **not** re-open TEST, would **not** authorise a dry run
on its own, and would **not** retire F57's finding that the touch-conditioned difference is fill
FREQUENCY rather than fill QUALITY. A FAIL at an MDE larger than 0.10 R says the design still cannot
see the effect — "no edge was detectable in THIS universe at THIS size", never "no edge exists".
