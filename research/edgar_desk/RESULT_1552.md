# RESULT — cells 1,552–1,561 (the EDGAR event desk): BUILDER status, 2026-09-26

Spec: `research/edgar_desk/PREREG_1552.md` (FROZEN 2026-09-26 20:40 UTC). This is the BUILDER's
deliverable for this pass. **No PREREG numbers are reported below — none exist yet.** The FETCH
stage (`fetch_submissions.py`, already running under nohup + a background until-loop, pid
3701143 / wait-loop pid 3701175, started 20:48 UTC) has not produced `events_raw.csv`, so
`cell_1552.py` has nothing to score. Re-run `python3 research/edgar_desk/cell_1552.py` once that
file exists.

## What is done this pass
1. `research/edgar_desk/cell_1552.py` — the scorer: population filter (price ≥ $1 and prior-session
   20-day dollar volume ≥ $1M, evaluated on the session BEFORE entry, never the entry day itself),
   entry-auction resolution from `acceptanceDateTime` (< 09:00 ET → same session, else next
   session — a 09:31 acceptance never gets a same-day fill), legs E1/E5, 5 bps/leg cost (10 bps
   round trip), 3%/yr pro-rata borrow on SHORT cells, SSR (prior session ≤ −10%) and sub-$5
   exclusions, day-clustered t (`cell_1445.day_clustered_t`, reused), ex-top-5%/1%, winner-capped
   mean (±20%), median, share-in-direction, a count-matched null (1,000 draws, seed 1552), and a
   TRAIN/VAL/TEST split (TEST sealed — the script never scores it without `--unseal-test`).
2. `research/edgar_desk/test_cell_1552.py` — 31 unit tests, all passing: entry-timing (incl. the
   09:31 refuter, weekend roll-forward, and the "beyond loaded bars" edge), class assignment for
   all nine directional classes plus the CONTRACT/OFFERING gating conflict the PREREG itself flags
   (1.01 co-filed with 3.02 or 2.03 must NOT also fire CONTRACT), test-ticker exclusion, direction
   sign (a SHORT cell must show a POSITIVE net return on a down move), the E5 window (exactly 5
   sessions after entry, exact arithmetic checked), SSR flagging, and the two population-floor
   exclusions.
   ```
   31 passed in 1.58s
   ```
3. Smoke-tested `cell_1552.py`'s missing-input path: it logs ERROR and exits 2 rather than
   silently scoring an empty book — confirmed by direct run.

## What is NOT done (blocking)
`events_raw.csv` does not exist. The task text asserted it existed with "0 filings, PENDING" —
that does not match what is on disk; there is no file at that path. The FETCH stage is in its
first phase (CIK mapping): SEC's own `company_tickers.json` mapped 7,645/30,858 universe symbols;
the remaining 23,213 are going through the `browse-edgar` fallback at the mandated ≤1 req/s.

**Observed after ~6 minutes of fallback lookups: 200/23,213 checked, 0 recovered.** Every symbol
in the sampled warnings (`004CVR049`, `038CVR015`, `18506U302`, `382ESC010`, `457ESC051`,
`471CVR019`, `478ESC031`, `685CVR011`) is a CUSIP-shaped identifier (digits + letters, no normal
ticker form) — these read as escrow shares / contingent-value-right (CVR) instruments carried in
`alpaca_assets_all_20260905.csv` as "symbols," not real exchange tickers, and cannot resolve to a
CIK by definition. **Flagging, not fixing**, per this task's scope (BUILDER only, no pivot): at the
observed rate this stage alone projects to roughly 11–12 hours, and a large share of that time is
being spent on identifiers that were never going to map. A future pass should pre-filter the
universe to ticker-shaped symbols (or intersect against `alpaca_assets_all`'s own `common` flag
more strictly) before the fallback loop, to cut this by what looks like a large fraction — I did
not make this change myself; it touches the running FETCH stage's universe-build step, is a
design call outside "execute the builder," and the process already has zero progress to lose if
someone chooses to restart it with a tighter prefilter.

This does not threaten *correctness* (unmapped symbols are counted as LOST and excluded, not
silently dropped from the count — the completeness gate is unaffected), only *when* real numbers
exist to report.

## Not yet run / carried forward for the next pass
- `cell_1552_events.csv` and the full per-cell/per-holdout tables (n, events/week, mean net bps,
  day-clustered t, ex-top-5%/1%, winner-capped, median, share-in-direction, universe placebo,
  count-matched null, per-year table) all require `events_raw.csv`.
- The leg (E1 vs E5) has not been named on TRAIN for any cell — nothing to name yet. `LEG_LOG`
  path (`research/edgar_desk/leg_selection_train.log`) is wired but unwritten.
- 1,561 BUYBACK_OR_INSIDER is out of scope for this build regardless of fetch status: it needs a
  Form 4 join `fetch_submissions.py` does not build. Report-only until a follow-up PREREG's fetch.
- Independent reimplementation (CLAUDE.md rule #1) has not happened — required before any number
  from this pipeline is shown to the owner.
- Item-code-mapping hand check (30 filings/class, per the PREREG's own refuter list) not done —
  no filings exist to sample yet.

## Caveats (read as an adversary, per CLAUDE.md)
- All numbers in this report are ZERO — nothing has been computed. Do not read the code as a
  proxy for a result.
- `cell_1552.py`'s SSR proxy ("prior session return ≤ −10% vs the session before it") is a coarse
  stand-in for the real SSR trigger (a 10% intraday drop from the prior CLOSE, tracked minute-by-
  minute); it will over-exclude some names and under-exclude others relative to the true rule —
  flagged for the independent-check pass, not resolved here.
- `classify_from_row` is intentionally duplicated (not imported) from `fetch_submissions.py` so a
  later edit to one does not silently change the other; if the two ever diverge, that is a bug to
  catch via the class-assignment unit tests, not a design feature to rely on.
- Disk: 21GB free (80% used) — no bulk-fetch action was taken this pass beyond what was already
  running; worth watching as `submissions/*.json.gz` accumulates (currently 0 cached files).

## Files
- `research/edgar_desk/cell_1552.py` (scorer, not yet run against real data)
- `research/edgar_desk/test_cell_1552.py` (31/31 passing)
- `research/edgar_desk/RESULT_1552.md` (this file)
- `research/edgar_desk/cell_1552_events.csv` — NOT written: there is nothing to put in it yet and
  a header-only stub would misrepresent readiness; it will be produced by `cell_1552.py`'s next
  run once `events_raw.csv` exists.
- Unchanged, still running in the background: `research/edgar_desk/fetch_submissions.py` (pid
  3701143), its wait-loop (pid 3701175), log `research/edgar_desk/fetch_submissions.log`.
