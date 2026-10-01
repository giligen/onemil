# RESULT 1,693b -- band A (gap [2,3)%) sub-pools AF1-AF6: backfill + feature build COMPLETE;
stage_pools killed by the harness's node memory-pressure protection -- still no AF1-AF6 scoring

PREREG: research/orb_freq/PREREG_1693.md (original, FROZEN). Re-run requested once the bar store
was free (sole writer, one process, `nice -n 10`). **Outcome this session: steps 1-3 (candidate
list, backfill appender, feature build) ran to completion on real data for BOTH windows. Step 4
(stage_pools) started, made partial progress, then was killed by Claude Code's own background-shell
memory-pressure reaper -- not a bug in this cell's code, not file-lock contention (the prior
session's blocker). Steps 4-5 (pipeline, 12-exit scoring) still did not run.** No pass/fail/
robust/regime_specific claim is made on AF1-AF6 below; CLAUDE.md: "a null is a claim about MY test
first."

## Step 1 -- candidate list (unchanged from the prior session, re-verified)
42,444 symbol-days (7,769 out_regime 2024-07-01..2024-12-31 + 34,675 in_regime
2025-01-01..2026-09-26), gap [2,3)% (fetch-screen buffered [1.7,3.0)%), $3-30, prior-day volume
>=500K, production/3-5% seed excluded by construction (overlap re-checked = 0).

## Step 2 -- backfill appender: COMPLETE, 561/561 days, disk floor never breached
Ran `nice -n 10 python3 .../scratchpad/backfill_fill_days.py` to completion, then once more to
mop up 3 days a newly-found appender gap had poisoned (below). **12,730,554 bars appended this
session** (12,662,274 + a 68,280-bar resume pass). Disk free: started 7.92 GB, ended 6.55 GB
(internal, `df -h` ~5.9-6.1 GB), floor 5.0 GB never breached, checked before every day.
- **New appender-gap finding (this session):** band A's own candidate population carries THREE
  Alpaca-rejected tickers never seen by the original cell 1689b's list (`ALB-A`/`BA-A`/`HPE-C`/
  `MRP#`/`ORCL-D`/`QXO-B`) -- `AXIA-C`, `SCE-K`, `WAL-A` (hyphen+letter, same shape, filtered
  pre-fetch, 4 pairs) plus a NEW shape, a literal `=` suffix (`NWAX=`, `OTAI=`) that the original
  filter did not catch: it killed 2 whole days (`2025-12-29`, `2026-01-02`, "invalid symbol:
  NWAX=") before the regex was widened (`#|=|-[A-Z]{1,2}$`) and a third day (`2026-06-12`, `OTAI=`)
  was caught live; all 3 days' OTHER symbols were recovered by the resume pass once the symbol was
  excluded (STATE-resumable, no data lost beyond the always-invalid tickers themselves). Dot-suffixed
  class shares/warrants (`BF.B`, `LGF.A`, `NWAX.U`, `ACHR.WS`, ...) were deliberately NOT filtered
  (pools_1689a_lib's own `+`->`.WS` convention; no evidence they fail) and fetched without incident.
  Net exclusion: **7 symbol-day pairs across 5 symbols**, permanent (Alpaca does not carry them).
- Also fixed in the wrapper: the fetch-list CSV's three `NA`-ticker rows (Nano Labs) were silently
  dropped by pandas' default NA handling (the exact bug `trading/orb_csv.read_orb_csv` exists for);
  switched the wrapper to that helper -- all 3 "NA" rows now survive.
- **Minute-bar coverage after this session (valid 42,437-pair universe): 42,076/42,437 = 99.15%**
  (out_regime 7,767/7,769 = 99.97%; in_regime 34,309/34,668 = 98.96%) -- clears the CLAUDE.md >=80%
  availability rail with large margin. Remaining ~0.85% gap is genuine no-trade/no-SIP-data minutes,
  not an appender defect (reported by `features_build.py` per-window, below).

## Step 3 -- feature build: COMPLETE for BOTH windows, with a live memory-safety fix
`BAND23_WINDOW=out_regime` ran whole (7,769 candidates, 2,479.7s, finished clean) ->
**6,138 feature rows** (79.0% yield), coverage 7,767/7,769 bars (99.97%).
`BAND23_WINDOW=in_regime` run WHOLE (34,675 candidates) was **killed by this agent** after `free -h`
showed 73.2% RSS / **swap 100% full (0 free)** mid-bulk-fetch -- the live trading service (a
separate, untouched process) was never at risk but the node was one allocation from an OOM event;
root cause is seam 4's bulk minute-bar fetch building one Python dict per bar across the WHOLE
requested candidate-day set at once, so peak memory scales with candidate COUNT per invocation, not
with the (unchanged, always-loaded-whole) daily/SPY panels. Fix: added `BAND23_DATE_LO`/`_HI` env
filters to the scratchpad `features_build.py` (project code untouched) and re-ran in_regime as 7
quarterly chunks (q1..q7, ~4.0-5.6K candidates each, all below out_regime's clean 7,769) via
`scratchpad/band23/run_remaining_chunks.sh`; peak RSS per chunk stayed <=1GB, available memory never
dropped below ~1.5 GB. **All 7 chunks succeeded (rc=0) -> 27,793 feature rows** (80.1% yield),
concatenated (0 duplicate (symbol,date) rows at the quarter boundaries) to
`out_in_regime/orb_features_COMBINED.csv`. Coverage 34,309/34,675 bars (98.96%, 366 symbol-days with
no bars reported, matches Step 2).
**True-up to the exact [2,3)% band on the minute-bar `gap_pct`** (recomputed directly from both
feature CSVs, no DB access, independent of stage_pools): **out_regime 3,946/6,138 (64.3%);
in_regime 18,478/27,793 (66.5%); combined 22,424** candidates that would enter F1-F6 admission.

## Step 4 -- stage_pools: STARTED, KILLED by the harness's memory-pressure protection, NOT a bug
`python3 research/orb_freq/1693b_band23.py --stage pools` logged `load_long_daily()` (8,486,691-row
daily panel, cache.db+XNAS, successful) and in_regime's true-up (18,478, matches the independent
recount above) -- then Claude Code's own background-shell OOM reaper killed the process (and this
agent's monitor loop) while the node was under memory pressure, with the explicit system note "do
not start it again on your own ... start it again only when asked." **No manifest.csv, no
`AFx_*_features.csv`, no `premkt_coverage.txt` were written -- zero partial/corrupt output exists.**
Root cause (by elimination: the two steps already proven safe above it in the log are the 8.4M-row
daily panel load and the in_regime merge): almost certainly `bulk_premarket_usd()`'s single
unchunked query across all ~22K true-up (symbol,day) pairs against the 133M+-row `bars` table --
the same risk shape as Step 3's seam-4 fetch, which date-chunking already fixed. **This was NOT
re-run or chunked this session** per the explicit instruction not to restart harness-killed work
without being asked; `research/orb_freq/1693b_band23.py` itself carries only the cache.db
retry/backoff patch (below), no stage-4 chunking. Recommended next step, NOT executed: apply the
same `BAND23_DATE_LO`/`_HI`-style chunking inside `stage_pools` (or call `bulk_premarket_usd` in
date-range batches) before the next attempt. System memory recovered fully after the kill (6.3 GB
available); disk held at 5.9 GB free throughout, floor never breached.

## Steps 4 (cont'd)-5 -- pipeline + 12-exit scoring: NOT RUN (unchanged bottom line, new cause)
No manifest/per-pool features CSVs exist, so `--stage pipeline` and `--stage score` were not
invoked. `1693_pool_exits.py` was not touched or re-verified this session (no new risk to it).

## AF1-AF6 classification, robust/regime-specific pairs, union effect
**None -- same as the prior session: no fills were reconstructed, no exit grid, no classification,
no union read exist for band A.** `1693b_reads.csv` and `1693b_pool_books.csv` are unchanged:
schema-only, zero data rows (not fabricated).

## Code changes this session (all additive, none touch config/orb.yaml/the live service/trading/*.py)
- `research/orb_freq/1693b_band23.py`: `_ro()` gained a 30s-backoff/10-retry wrapper for
  "database is locked" on cache.db/bars_sip.db reads (never raised this session; precautionary,
  matches the project's read-only-during-live-hours rule). No other change; stages/logic untouched.
- Scratchpad only (never under `research/orb_freq`): `backfill_fill_days.py` (wider invalid-symbol
  regex + `read_orb_csv` for the "NA"-ticker bug), `features_build.py` (`BAND23_DATE_LO`/`_HI`
  chunking), `run_remaining_chunks.sh` (new, the 7-chunk driver).

## What a re-run needs
1. Do NOT re-run `--stage pools` without the owner's go-ahead (explicit harness instruction this
   session). When asked: chunk `bulk_premarket_usd`/stage_pools by date range first (pattern proven
   in Step 3), watch `free -h` during the run, abort if available memory drops near ~500 MB.
2. Then `--stage pipeline`, then `--stage score`, unchanged.

## Files
`research/orb_freq/1693b_band23.py` (cache.db retry patch only), `1693b_band23.log` (this session's
stage_pools-partial log, prior session's note preserved above it), `1693b_reads.csv` /
`1693b_pool_books.csv` (schema only, 0 rows, unchanged). Scratchpad: `backfill_fill_days.py` (fixed),
`band23/build_candidates.py` (unchanged), `band23/features_build.py` (chunked),
`band23/run_remaining_chunks.sh` (new), `band23/out_out_regime/orb_features_20261001_1916.csv`
(6,138 rows), `band23/out_in_regime/orb_features_COMBINED.csv` (27,793 rows, concatenated from
`chunks/q1..q7/`), `band23/band23_backfill_state.json` (561/561 days).

---

## Appendix: prior session (infrastructure-blocked attempt, superseded above)

# RESULT 1,693b -- band A (gap [2,3)%) sub-pools AF1-AF6: candidate list and backfill built;
scoring NOT run this session (infrastructure contention, documented below)

PREREG: research/orb_freq/PREREG_1693.md (original, FROZEN). This cell was asked to build the ONE
seed cell 1,693 explicitly declared not built ("Both original missing seeds ... require NEW minute
bars via the bars_sip.db appender for a candidate universe that does not exist in any CSV on disk
... infeasible inside this cell's call budget, and no owner GO for a new data pull"). Alpaca SIP
minute bars are a FREE pull (not the Databento spend the owner's "convince first" rule covers), so
this cell had a GO to build it. **Outcome: steps 1-2 (candidate list, backfill appender) ran to
completion; step 3 (feature build) did not finish in this session's time/call budget; steps 4-5
(pipeline, 12-exit scoring) were therefore not run on real data.** This is a coverage/infrastructure
finding, not a verdict on band A -- no pass/fail/robust/regime_specific claim is made on AF1-AF6
below; CLAUDE.md: "a null is a claim about MY test first."

## What ran (disk floor 5 GB respected throughout; final free = 8.2 GB, never breached)

**Step 1 -- candidate list** (`/tmp/.../scratchpad/band23/build_candidates.py`, daily bars via
`pools_1689a_lib.py`'s own cache.db+Databento EQUS.SUMMARY loaders, same source cells 1689a/1689b
use): gap [2,3)%, price $3-30, prior-day volume >=500K, 2024-07-01..2026-09-26, excluding
production (>=5%) and the 3-5% seed (bands B/C) BY CONSTRUCTION (disjoint gap bands) -- re-checked
explicitly, overlap = 0 in both windows, confirmed not assumed.
- Fetch-list (daily-bar screen, buffered gap[1.7,3.0)% per the 1689b -1pp convention, scaled down
  from -1.0pp because band A's low edge is in the steep part of the gap distribution -- an
  unbuffered -1.0pp inflated the fetch list 3.6x, from 27,997 to 101,505 rows, measured not assumed):
  **42,444 symbol-days** (7,769 out_regime 2024-07-01..2024-12-31 + 34,675 in_regime
  2025-01-01..2026-09-26).
- True band-A [2,3)% estimate on the DAILY-bar gap (pre minute-bar true-up): **27,997 symbol-days**
  (4,971 out_regime + 23,026 in_regime).

## Step 2 -- backfill appender (`/tmp/.../scratchpad/backfill_fill_days.py`, a wrapper around
research/bf_zero/backfill_bars_sip.py -- its fetch_day()/BATCH reused verbatim, "the designed
appender" itself never edited; FEATURES/STATE monkeypatched to this cell's own scratchpad files;
bars_sip.db is APPEND-only, never rewritten)
- Already present in bars_sip.db before this cell touched it (incidental overlap with other cells'
  fetches): **10,689 / 42,444 pairs (25.2%)**.
- Needed a fresh fetch: 31,755 pairs across 561 days.
- **Bars appended this session: 0.** Root cause, found live and fixed mid-session (both documented
  in `research/orb_freq/1693b_band23.py`'s module docstring and the scratchpad wrapper's own
  docstring, not hidden):
  1. `research/orb_freq/1689b_pools.py --stage backfill` (a sibling cell, same owner ask) was
     running against the SAME bars_sip.db at this cell's start, holding write locks; the shared
     appender's own `missing_pairs()` ("select distinct symbol,day from bars", a 133M+-row scan)
     failed with "database is locked" and then ran >13 minutes without finishing under the
     contention -- replaced with an INDEXED anti-join against this cell's own ~42K-pair want-list
     (bars has `PRIMARY KEY (symbol,day,t)`), which finished in 40-50s every time it ran.
  2. One fetch call hung INDEFINITELY (not slow -- confirmed via `ss -tnp`: the TCP socket to
     Alpaca sat in CLOSE-WAIT with unread bytes, a dead connection the alpaca-py client never
     noticed) -- fixed with a SIGALRM hard wall-clock timeout around every `fetch_day()` call.
  3. After the hang fix, EVERY day still hit "database is locked" on all 3 retries (tightened
     busy_timeout 30s->8s to fail fast rather than hang) and got skipped -- the sibling's own
     `1689b_features.py` (a different, CPU+I/O-heavy stage) was still running against bars_sip.db
     for the full session; this cell's write attempts never found a gap.
- Net: this cell's own candidate population has only the pre-existing 25.2% minute-bar coverage;
  a clean re-run (sequenced AFTER the sibling cell's backfill+feature-build finish, not
  concurrent with them) is needed to actually fetch the missing 31,755 pairs.

## Step 3 -- feature build (`/tmp/.../scratchpad/band23/features_build.py`, the
study_orb_features.py loader-seam pattern identical to 1689a_features.py/1689b_features.py)
Attempted on out_regime only (smaller: 7,769 fetch-screen candidates). Loaded daily bars (199,408
rows) and SPY intraday (459,058 bars) successfully; the per-candidate minute-bar feature
extraction did not produce output within this session's remaining time/call budget (killed after
~5 min with no CSV written) -- almost certainly the 25.2% bar coverage above, not a new bug: most
candidates have no bars at all and study_orb_features.py's own per-row path still has to probe and
reject each one. in_regime (34,675 candidates, ~4.5x larger) was not attempted.

## Steps 4-5 -- pipeline + 12-exit scoring: NOT RUN
No features CSV exists for any of AF1-AF6 in either window, so `research/orb_freq/1693b_band23.py
--stage pipeline` and `--stage score` were not invoked against real data this session. The script
itself (`research/orb_freq/1693b_band23.py`) is complete, syntax-checked, and reuses
`research/orb_freq/1693_pool_exits.py`'s CachedStore/reconstruct_fill/EXITS/build_per_fill_table/
score_table/classify_pool/cb verbatim (imported read-only via importlib, never edited) -- it is
ready to run stage-by-stage (`--stage pools`, then `pipeline`, then `score`) once a features build
completes, without further code changes. It also carries the AF2 pre-market-coverage VOID gate
(CLAUDE.md's >=80% availability rail) and the production-union logic described in PREREG_1693.md.

## AF1-AF6 classification, robust/regime-specific pairs, union effect (prior session)
**None -- no fills were reconstructed, so no exit grid, no classification, and no union read
exist for band A this session.** `1693b_reads.csv` and `1693b_pool_books.csv` are written with
their intended schema and zero data rows (not fabricated placeholders).

## What a re-run needs (prior session's list, superseded by this session's "What a re-run needs" above)
1. Confirm no sibling cell is writing bars_sip.db (`lsof` / `ps aux | grep backfill`) before
   starting -- the contention was the entire blocker, not this cell's own logic.
2. `nice -n 10 python3 /tmp/.../scratchpad/backfill_fill_days.py` (STATE already tracks 0 days
   done; it resumes cleanly) until the 31,755-pair gap closes.
3. `BAND23_WINDOW=out_regime` then `BAND23_WINDOW=in_regime` through
   `/tmp/.../scratchpad/band23/features_build.py`.
4. `python3 research/orb_freq/1693b_band23.py --stage pools`, then `--stage pipeline`, then
   `--stage score` -- unchanged, no further edits expected.

## Files (prior session)
`research/orb_freq/1693b_band23.py` (the stage pools/pipeline/score script, untested end-to-end
this session -- syntax-verified only), `1693b_band23.log` (this session's stage_pools-less run
log), `1693b_reads.csv` / `1693b_pool_books.csv` (schema only, 0 rows). Scratchpad working files:
`build_candidates.py`, `backfill_fill_days.py` (STATE at `band23_backfill_state.json`),
`features_build.py`.
