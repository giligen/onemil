# PREREG — A NEW population: the PRE-MARKET-HIGH break (PMH). Cells 1,389–1,392

Why a new population and not another variant: 37 cells on the HOD-break population (`REPORT.md`, `REPORT_PASS2.md`)
show every exit, filter, entry and slot rule within ±0.04 R of the live rule — the path after that signal carries no
information. The pre-market high is a different object: a level formed while the crowd is positioning, watched by
every gapper trader, broken once. Mechanism = attention at a SPECIFIC level with resting orders above it, which
the intraday HOD (a moving target) does not have. Frozen before any bar is walked.

## Data
* Pre-market bars 04:00–09:35 ET for the 15,656 HOD-universe symbol-days: `data/cache.db::intraday_bars_1min`
  via `scripts/orb_premarket_backfill.py --candidates research/hod_exit_lab/pm_candidates.csv` (launched 9/22 20:59
  UTC). Regular-session bars: `research/bf_zero/bars_sip.db`. Universe = the HOD-break universe's symbol-days
  (gappers with a signal that day), so the population is "PMH breaks on names that were in play" — stated as such.
* Availability rail: a symbol-day is usable only if it has ≥ 20 pre-market bars with prints; report the share.

## Signal (frozen)
PMH = max(high) of the pre-market bars 04:00–09:29 ET, with pre-market volume ≥ 50,000 shares (else no level).
Break = the first regular-session 1-minute bar (09:31 onward; the 09:30 bar excluded — auction noise) whose CLOSE
> PMH, at or before 11:30 ET. Entry = the OPEN of the next bar (obtainable). Stop = the low of the last 15 minutes
before the break bar, floored at 1.5 % of price (R-vs-spread gate: cost ≈ 0.2 R at the floor; report R as % of
price). Exit: cell-specific below; always flat at the 15:55 open. One trade per symbol-day. Cost: measured
half-spread at the entry minute if `nbbo.csv` covers the (day, symbol) — it covers the HOD signal minute, not this
one, so use the study's price×hour band ONLY as a sensitivity line and mark the base cost as "half-spread at the
HOD signal minute of the same name-day" (same name, same morning); state coverage.

## Cells
| cell | exit |
|---|---|
| 1,389 P1 | target +2 R, stop, 15:55 |
| 1,390 P2 | no target, stop, 15:55 |
| 1,391 P3 | breakeven lock at +1 R, no target |
| 1,392 P4 | P1 restricted to breaks in 09:31–10:00 ET (the crowd's window) — a CONDITION cell, reported with the dropped cohort |

## First deliverable, before any cell is scored
The drift exhibit for this population (as `DRIFT.md`): if the unmanaged path is ≤ +0.15 R at 2 h with the same
±1.7 R excursions, the population is as informationless as HOD-break and the cells are reported as such. Also the
overlap with the HOD population: what share of PMH breaks are also HOD-break signals within 5 minutes (if > 80 %,
this is the same population renamed and the pass is closed at the exhibit).

## Splits, statistics, pass bar
TRAIN 2025 (halves), VAL 2026-01..05, TEST sealed. Net mean R ≥ +0.10 both splits, VAL day-clustered t ≥ 2, halves
> 0, ex-top-5 % > 0, ≥ 3 fills/week on VAL at a first-12/day, 4-concurrent slot rule, cadence C3/C4 on VAL, and
the D1/D3 placebos (same name-day random minute; random other name) beaten by ≥ +0.10 R. Verification: three
refuter lenses + independent rebuild from this prose. Programme count 1,392.
