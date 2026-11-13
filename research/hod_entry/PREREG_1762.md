# PREREG 1,762 — anticipation entry on the HONEST population (child 1 of 1,752): PIT gapper days only, missing bars fetched, R ≥ 1 % floor

FROZEN 2026-11-13 20:20 UTC before any number. Programme cell count: 1,752 → 1,762 (1,753–1,761 ran in between).
Parent: 1,752 — the approach entry's gross was positive on break-conditioned arm days only and ≈ 0 on the PIT
gapper days, which had minute bars for just 40 % of symbol-days; daily-MA features were VOID at 64 % coverage.
Rules: write only `research/hod_entry/1762_*`, `1762_cache/`, `RESULT_1762.md`; no git/service/orders; network
ONLY for Alpaca minute and daily bars (free; keys from `.env`, never printed; ≤ 5 requests/s; the heavy window is
open after 20:05 UTC); `ulimit -v 2500000; nice -n 19`, peak RSS < 1.5 GB, per-day chunks, checkpoints; budget
≤ 60 calls. TEST sealed (never fetched for scoring).

## Population (the honest denominator)
`research/hod_entry/cell_1428_population.csv` (7,487 point-in-time gapper symbol-days with split, wk) — ALL of
them, no arm conditioning. Minute bars 09:30–16:00 for every symbol-day: from bars_sip.db where present, else
fetched from Alpaca (SIP feed, raw) into `1762_cache/bars/<day>.parquet`; report coverage after the fetch (≥ 90 %
or VOID), and the winner/loser missingness gap (≤ 5 pp). Daily bars for the MA features: fetch the missing symbols'
daily history (≥ 60 trading days before each event) into `1762_cache/daily.parquet`; coverage ≥ 90 % or the MA
reads are VOID (the rest of the cell still stands).

## Entries, features, stops, grid — exactly PREREG_1752 + Amendment 1, with two frozen changes
1. R floor: only entries whose strict-stop distance (entry − S) ≥ 1.0 % of price are eligible (the R-must-exceed-
   the-spread rule); report how many approaches the floor removes.
2. The grid is the 1,752 grid restricted to the APPROACH entries (the BREAK/HOLD/RETEST entries on this population
   are reported as comparators only, one row each, not searched).
Stage A on TRAIN (halves), pick on TRAIN-H2 with ≥ 3 fills/wk; Stage B one VAL read; placebo as 1,752.

## Kill (as 1,752)
VAL net ≤ 0 or day-t < 2; TRAIN halves opposite; VAL ex-top-5 % ≤ 0; < 3 fills/wk; placebo ≥ rule; P(break) under
the rule on VAL < TRAIN − 10 pp.

## Output
`1762_fetch.py`, `1762_build.py` (may import from 1752_build.py), `1762_score.py`, `1762_events.parquet`,
`1762_anatomy.csv`, `1762_grid.csv`, `1762_val.csv`, `RESULT_1762.md` ≤ 350 words ending with "CHILDREN:" (≥ 2).
Return ≤ 120 words: coverage after the fetch (minute, daily), approaches removed by the R floor, the winners-vs-
losers separators (AUC) on the honest population, P(break) TRAIN vs VAL, the picked rule, VAL net / day-t /
fills-wk / placebo, the BREAK comparator on the same population, KILL/SURVIVE, children.
