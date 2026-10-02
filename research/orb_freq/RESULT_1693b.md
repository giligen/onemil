# RESULT 1,693b -- band A (gap [2,3)%) sub-pools AF1-AF6, the missing seed from PREREG_1693's own escape clause

PREREG: research/orb_freq/PREREG_1693.md (original, FROZEN). This cell builds the ONE seed that cell 1,693 explicitly declared NOT built (2-3% gap band x F1/F3/F4/F5/F6), plus F2 (pre-market dollar volume) on the same fresh population, since new minute bars had to be fetched for it anyway. Harness reused verbatim from research/orb_freq/1693_pool_exits.py (CachedStore/reconstruct_fill/EXITS/build_per_fill_table/score_table/classify_pool/cb), imported read-only via importlib -- never edited.

## Band-A sub-pools (6) -- best exit per direction, classification
- **AF1** [fails] dirA: none select | dirB: none select | mean OOS-2024H2 R across exits: +0.283 (n_fills reconstructed=55, dropped=0)
- **AF2** [fails] dirA: none select | dirB: none select | mean OOS-2024H2 R across exits: +0.058 (n_fills reconstructed=199, dropped=0)
- **AF3** [fails] dirA: none select | dirB: none select | mean OOS-2024H2 R across exits: -0.078 (n_fills reconstructed=162, dropped=0)
- **AF4** [fails] dirA: none select | dirB: none select | mean OOS-2024H2 R across exits: +nan (n_fills reconstructed=11, dropped=0)
- **AF5** [fails] dirA: none select | dirB: none select | mean OOS-2024H2 R across exits: +0.024 (n_fills reconstructed=86, dropped=0)
- **AF6** [fails] dirA: none select | dirB: none select | mean OOS-2024H2 R across exits: +0.849 (n_fills reconstructed=30, dropped=0)

## Robust pairs (both directions confirm) -- band A
- none

## Regime-specific pairs (one direction only; reported, not shipped) -- band A
- none

## Union (2025-01-01..2026-09-18, production + band-A robust pairs; fixed $375/fill)
- production ALONE: n=472, 5.29 fills/wk, mean +0.290 R/fill, weekly P10 -2.47 R ($-928), strong-week gap median/p90=3.5/12.600000000000009 wk, green 0.49295774647887325 vs null 0.4970165039511938
- UNION (production+band-A robust): n=472, 5.29 fills/wk, mean +0.290 R/fill, weekly P10 -2.47 R ($-928), strong-week gap median/p90=3.5/12.600000000000009 wk, green 0.49295774647887325 vs null 0.4970165039511938
- added frequency: +0.00 fills/wk from band A's robust pairs

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

## AF1-AF6 classification, robust/regime-specific pairs, union effect
**None -- no fills were reconstructed, so no exit grid, no classification, and no union read
exist for band A this session.** `1693b_reads.csv` and `1693b_pool_books.csv` are written with
their intended schema and zero data rows (not fabricated placeholders).

## What a re-run needs
1. Confirm no sibling cell is writing bars_sip.db (`lsof` / `ps aux | grep backfill`) before
   starting -- the contention was the entire blocker, not this cell's own logic.
2. `nice -n 10 python3 /tmp/.../scratchpad/backfill_fill_days.py` (STATE already tracks 0 days
   done; it resumes cleanly) until the 31,755-pair gap closes.
3. `BAND23_WINDOW=out_regime` then `BAND23_WINDOW=in_regime` through
   `/tmp/.../scratchpad/band23/features_build.py`.
4. `python3 research/orb_freq/1693b_band23.py --stage pools`, then `--stage pipeline`, then
   `--stage score` -- unchanged, no further edits expected.

## Files
`research/orb_freq/1693b_band23.py` (the stage pools/pipeline/score script, untested end-to-end
this session -- syntax-verified only), `1693b_band23.log` (this session's stage_pools-less run
log), `1693b_reads.csv` / `1693b_pool_books.csv` (schema only, 0 rows). Scratchpad working files:
`build_candidates.py`, `backfill_fill_days.py` (STATE at `band23_backfill_state.json`),
`features_build.py`.
