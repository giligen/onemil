# RESULT — cells 1,552–1,561 (the EDGAR event desk): BUILDER status, 2026-09-27

Spec: `research/edgar_desk/PREREG_1552.md` (FROZEN 2026-09-26 20:40 UTC) + **Amendment 1** (2026-09-27
04:25 UTC, before any score). **No PREREG pass/fail numbers are reported below — the full-file scorer
run has not finished writing `cell_1552_stats.csv` as of this update.** This is a BUILDER status
report, not a PREREG result; do not relay any number here to the owner as a finding.

## What is done and verified this pass
1. **Amendment 1 implemented in `cell_1552.py`**: `classify_from_row(form, items, serial_issuer=...)`
   now excludes 424B2 from the OFFERING trigger set and gates OFFERING/SHELF on a per-issuer,
   per-filing-date trailing-365-day count of ALL 424B1–424B5 filings (`build_serial_index`,
   `is_serial_issuer`, vectorized as `compute_serial_flags` / `compute_class_masks_amended` for the
   full file). Every other class is untouched, matching the amendment text.
2. **Tests**: `test_cell_1552.py` now has 43 tests (31 original + 12 new — 424B2 exclusion, the
   serial-issuer cap on OFFERING and SHELF from both a 424B form and item 3.02, the cap NOT touching
   unrelated classes, the exact `> 12` boundary, the 365-day window boundary, an unknown-CIK default,
   and non-424B forms never counting toward the cap). **43/43 pass.**
3. **Amendment self-check** (`run_amendment_self_check`): the vectorized full-file classifier is
   cross-checked against the scalar, unit-tested `classify_from_row` on a random 2,000-row sample
   every run. **Result on the full 4,410,667-row file: 2000/2000 agree (100.0000%).** This only
   catches a vectorization bug, not a spec error — the real independent reimplementation is a
   separate agent's job per CLAUDE.md and has not happened yet for this amendment.
4. **Class counts, raw (pre-amendment, from `fetch_summary.txt`) vs amended, on the FULL file**
   (`research/edgar_desk/class_counts_raw_vs_amended.csv`):

   | class | raw | amended |
   |---|---|---|
   | OFFERING | 1,149,657 | 55,077 |
   | SHELF | 16,398 | 15,547 |
   | REVERSE_SPLIT | 17,629 | 17,629 |
   | AUDITOR | 4,198 | 4,198 |
   | NON_RELIANCE | 1,165 | 1,165 |
   | LATE_FILING | 8,676 | 8,676 |
   | OFFICER_EXIT | 81,459 | 81,459 |
   | CONTRACT | 68,611 | 68,611 |
   | ACTIVIST | 5,769 | 5,769 |
   | BUYBACK_OR_INSIDER (1,561) | NOT_COMPUTABLE | NOT_COMPUTABLE (needs Form 4 XML; unchanged) |

   The amendment's own diagnosis is confirmed on the real data: OFFERING drops 95.2% (424B2 pricing
   supplements were the population), SHELF drops 5.2% (the same serial-issuer cap). Every other
   class is bit-for-bit unchanged, as the amendment specifies.
5. **A real bug found and fixed during this pass, before any score**: the first implementation of
   `build_serial_index` grouped issuers with a `for c in np.unique(cik): days[cik == c]` scan —
   O(n_ciks × n_424B_rows) ≈ 7,754 × 1.15M ≈ 9 billion comparisons — which did not finish in a
   reasonable time on the full file. Replaced with a single sort + boundary-detection grouping
   (O(n log n)); the fixed version builds the index for 5,400 issuer CIKs in ~22 seconds. Caught by
   running the real scorer, not by the unit tests (all synthetic fixtures are tiny) — a gap the
   fixed version's test suite still does not close; a future pass should add a larger synthetic
   `build_serial_index` timing/regression test.
6. **Memory**: the first two full-pipeline attempts (against a 300K-row TRAIN-only sample, same
   full-size price panels) were OOM-killed / swap-thrashed on this node (~7.6GB RAM + 4GB swap):
   the original code built one per-symbol DataFrame for every one of the ~40K universe symbols
   (`build_symbol_index`) and duplicated the bars panel across `load_bars` → `prepare_bars_shared`
   → `build_symbol_index` → `build_universe_return_table`. Fixed by (a) dropping unused high/low
   columns and reading only needed parquet columns, (b) computing prior_close/prior_dvol20/the two
   raw leg returns ONCE and sharing them (`prepare_bars_shared`), (c) replacing the eager
   per-symbol dict with a lazy, memoized `SymbolFrameCache` that slices the one shared bars frame
   only for symbols an event actually touches. A 300K-row smoke test then completed end-to-end
   (peak RSS 6.14GB, 2:54 wall).

## In progress (not done)
The full scorer (`python3 research/edgar_desk/cell_1552.py`, all fixes above applied) was launched
under `nohup` in the background (log: `research/edgar_desk/cell_1552_run.log`) and was still running
— past the amended-class-count log lines above, into bars loading / entry-session resolution / the
per-cell scoring and stats loop — when this BUILDER pass's step budget ran out. It was NOT killed;
it is a detached (`disown`) background process and may still be running, or may have finished, or
may have been reaped if the parent shell session was torn down — **check `ps aux | grep cell_1552.py`
and the tail of `cell_1552_run.log` before assuming either outcome, and re-launch the same command
if the process is gone and `cell_1552_stats.csv` was not produced.**

**Not yet produced, and NOT fabricated here:** `cell_1552_stats.csv` (per cell × holdout × leg n,
events/week, net bps, day-clustered t, ex-top-5%/1%, winner-capped, median, share-in-direction,
universe placebo, placebo margin + t, count-matched null percentile, pass_bar), the leg-named-on-
TRAIN-before-VAL log (`leg_selection_train.log` — code enforces this by computing and writing the
log entry for a cell before its VAL stats are computed, in-process, but no cell's TRAIN leg had been
logged before the pass ended), the per-year table, and the pass-bar verdict per cell. `cell_1552_events.csv`
(the per-event-leg ledger, columns cell/cls/leg/split/date/symbol/entry/exit/ret_dir_net) may exist
partially or not at all depending on how far the background run got — check its row count and the
`split` column before using it (TEST rows are never written, by construction, since the scorer drops
the TEST split before scoring).

## Caveats (read as an adversary)
- **No number above is a PREREG finding.** The class-count table is a population-diagnostic count,
  not an edge estimate; it carries no direction, cost, or holdout information.
- The amendment self-check (100% agreement) validates the CODE against itself, not the SPEC against
  reality — still needs an independent agent rebuild from PREREG_1552.md + Amendment 1 prose.
- The "prospectus supplement" count for the serial-issuer cap uses ALL 424B1–424B5 filings by CIK
  (not just 424B2) per a reading of the amendment text ("424B* filings... of any of the five
  subtypes"); this reading is not independently confirmed.
- Trailing-365-day counts for filings in the first year of the fetch window (2019-01 through
  2019-12) are under-counted (no pre-2019-01 424B history was fetched) — this could make a handful
  of early-2019 issuers look non-serial when they were actually serial; negligible for VAL/TEST
  (2023+) cells but flagged for TRAIN.
- This node's memory ceiling (~7.6GB + 4GB swap, apparently with additional cgroup or allocator
  pressure — RSS around 5-6GB already caused heavy swapping) is tight for this cell's full price
  panel (19.4M bars rows) plus the full 4.4M-row events file. The fixes above got a 300K-row sample
  through, but the full run's actual peak was not observed to completion in this pass.
- Universe/population filters (price ≥ $1, 20-day dollar volume ≥ $1M on the PRIOR session, common
  stock, PIT listing membership) reuse the pre-existing `cell_1552.py` population logic from before
  this amendment; this pass did not re-audit that logic against the PREREG's "Population, timing and
  prices" section beyond what the original 31 tests already covered.

## Next steps for whoever resumes this
1. Confirm whether the background run (started under this pass) is still alive; if not and
   `cell_1552_stats.csv` is missing or incomplete, re-run `python3 research/edgar_desk/cell_1552.py`
   (log to `cell_1552_run.log`) under nohup + a background until-loop, per the standing token-
   discipline rule, and expect several minutes given the memory constraints noted above.
2. Once `cell_1552_stats.csv` exists, apply the PREREG pass bar (VAL, named leg, net ≥ +15 bps,
   t ≥ 2.5, ex-top-5% > 0, winner-capped > 0, ≥ 3 events/week, placebo margin ≥ +10 bps at t ≥ 2,
   null ≥ 99th percentile, TRAIN same-sign t ≥ 1, ≥ 3 of 4 TRAIN years positive, SHORT cells also
   net of borrow/SSR/price-floor) per cell and rewrite this file with the real per-cell tables.
3. Route the result through the independent-reimplementation + causality-trace + price-scale +
   fill-realism + tail + multiplicity + statistics checklist (CLAUDE.md "No research claim ships
   without an independent check") before any number reaches the owner.

## Judge (main session, 2026-09-27 06:30 UTC) — FAIL all nine computable cells; the structured desk is closed

* Both implementations (builder `cell_1552.py`, rebuild `rebuild_1552.py` from the prose) fail every cell on VAL: the
  best day-clustered t is 1.8 (builder) / 1.6 (rebuild). Under every refuter correction (ET timing, split windows
  removed, de-duplicated 13D groups, calendar-week frequency, intraday SSR, 30 % borrow) still 0 of 9; even the
  forbidden best-leg-on-VAL choice fails (t ≤ 2.05). Sign-flip null: 3–5 % chance that at least one of nine passes.
* The only classes in their pre-registered direction after the tail cut are the distress shorts: LATE_FILING (NT
  10-K/Q) E5 +156 / +179 bps on VAL (builder / rebuild, t 1.3 / 1.6) at 2.6 calendar events/week, NON_RELIANCE E5
  +186 (t 1.8) at 0.9/week. Real mechanism, no book: at $3K per event that is ≈ $100/week before borrow on names that
  are hard to borrow.
* The two builds agree poorly on the EVENT SETS (Jaccard 0.24–0.66 TRAIN, 0.49–0.77 VAL; three named legs differ)
  although the amendment mechanics and the timing rule match — the disagreement is the class definitions themselves:
  8-K item 5.03 is mostly bylaw/charter amendments (0 % of a sample mention a split), 424B3 mixes resale prospectuses,
  ETF and note wrappers, 13D is attributed to the filer, 1.01 is often a financing. Item codes do not identify the
  events this desk wanted; only the NT 10-K/Q and 4.02 forms are clean.
* Defects on record for any rerun (none flips a verdict): acceptanceDateTime is UTC, both builds read it as ET (9 % of
  pre-market filings enter a session late, no look-ahead); raw prices leave splits inside the E5 windows (AMZN 20:1,
  WHWK); renamed tickers double-counted; events/week divided by weeks with an event; passes_bar omits two TRAIN
  criteria; survivorship 2019–2024H1 is adverse to the SHORT cells (missing SPAC/acquired deaths, SIVB/FRC/PACW
  unmapped); borrow flat 3 % with no locate.
Consequence per PREREG: leg B (structured EDGAR desk) closes with the class table on record. The text-classified
version (an LLM reading the 8-K to find the real reverse splits, going-concern language and priced offerings) is a
separate decision: its only candidates are the distress classes at < 3 events/week, below the frequency bar before
any reading cost. Programme count 1,561.
