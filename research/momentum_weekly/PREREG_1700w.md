# PREREG — cell 1,700w: the sleeve's trading cost from measured quotes (FROZEN, before any number)

The reconciliation (RECON_1700tu.md) shows the backtest's cost is a BAND, not a measurement: 15–20 bp per traded
dollar, binding on 66 % of name-weeks, 3.3–3.7 %/yr of drag at 44.6 %/week turnover. Project rule: cost is charged
from measured NBBO at the trade minute, never a band. The live sleeve sends market orders at 09:45 ET on Monday.

## Data (free only — Alpaca historical quotes, SIP; if the API refuses historical quotes for this key, STOP and say so)
Sample: 80 rebalance Mondays drawn evenly across 2020-01 → 2026-09 (every k-th Monday, fixed before any fetch; the
quote history may not reach before 2020 — report the earliest date served). For each sampled Monday: every name the
guarded reference TRADED that day (entries, exits and the re-equalised kept names — from the first build's weekly
holdings dump), its NBBO at 09:45:00 ET (last quote at or before) and, for information, at 09:31:00 and 10:00:00.
LOST count and coverage per Monday; VOID if coverage < 80 % of traded name-weeks.

## Reads
Half-spread in bp (ask − bid) ÷ 2 ÷ mid: median, mean, P90, by year, by side (entry / exit / re-equalise), by dollar
size of the trade; share of name-weeks above 10 bp; crossed or locked quotes dropped and counted.
Cost per traded dollar = half-spread + 1 bp (fees / impact allowance at $1K–$3K orders in ≥ $200M-a-day names).

## Restatement rule (decided now)
Recompute the plain, guarded and half-size-gated books with cost per traded dollar = each year's MEAN measured
half-spread at 09:45 + 1 bp (years before the first served date use the earliest served year's mean × 1.5).
Report beside the band-cost headlines (26.6 % / 28.7 % / 30.0 %). The measured-cost row becomes the headline only if
coverage ≥ 80 % and the 09:31 spread is reported next to it (a live order sent earlier would pay that instead).
No strategy rule changes in this cell. 1 cell.

## Output
`1700w_cost.py`, `1700w_quotes.parquet` (gitignored), `1700w_spreads.csv` (one row per traded name-week sampled),
`RESULT_1700w.md` (≤ 50 lines, adversary caveats). Fetch at ≤ 3 requests/second with retries and a resume file.
