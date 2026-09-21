# PREREG — HOD-break attention-rank filter (cells 1,351–1,352). Owner 2026-09-21 "golden idea" pass

Frame: an HOD break pays when the crowd is watching the level. The 12 causal-filter cells (`CAUSAL_FILTER_REPORT.md`)
never conditioned on ATTENTION. Today's stratum work (`research/orb_seed_wide/REPORT_S1_CREATIVE.md`) showed the
one structural separator between a flat and a paying population was whether the name sits on the day's
rank-ordered gainer lists. This pass adds ONE causal feature in two definitions and scores it inside the existing
causal-filter machinery (`cells.py`, `features.csv`, the same TRAIN/VAL split and the same pass bar as
`CAUSAL_FILTER_PREREG.md` — unchanged, not re-read for thresholds).

## Feature (knowable at the signal minute, no look-ahead)
* **rank_cand** — among all rows of `candidates_full.csv` with the same `day` and `entry_m` ≤ the signal's
  `entry_m` (everything that has broken out so far that day), the signal's rank by `dist_open_pct` descending
  (1 = the strongest mover so far). Uses only rows whose `entry_m` is at or before the signal minute.
* **rank_mkt** — among the day's stream universe (`logs/hod_stream_universe_<day>.txt` where it exists; else the
  symbols present in `bars_sip.db` for that day), the signal's rank by (close at the signal minute / session open
  − 1) descending, computed from 1-minute bars up to and including the signal minute. If bar coverage of the
  universe for a day is < 80 %, that day is VOID for rank_mkt (report the share of days void).

## Cells
1,351 rank_cand ≤ 10 vs > 10 · 1,352 rank_mkt ≤ 10 vs > 10. Diagnostics (report-only): ≤ 5, ≤ 20, and the
monotone table by rank decile. The kept cohort must be ≥ 3 fills/week at the live config (C5) to matter.

## Pass bar
Exactly the causal-filter study's per-cell bar (G1) as written in `CAUSAL_FILTER_PREREG.md`, plus: kept cohort
positive on both TRAIN halves, dropped cohort ≤ 0 on both splits, cadence C4/C5 on VAL. A pass → independent
rebuild (Haiku) → ships as a filter to the DRY run (still zero orders) with `[HOD DRY]` lines carrying the rank.

## Forward check (report-only, n is small)
Parse this month's `[HOD DRY] WOULD BUY` journal lines (timestamp, symbol, "+x% from open"); rank each signal
among that day's dry signals fired at or before its minute; report mean spec R for rank ≤ 10 vs > 10 using the
spec P&L the EOD check computes. Sign must agree with the study before any claim.

## Not allowed
Thresholds other than 10 as cells; any other feature; changing the population or the split. Programme count 1,352.
