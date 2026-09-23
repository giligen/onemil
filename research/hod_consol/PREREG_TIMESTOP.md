# PREREG — Base entry + NO-BREAK time stop. Cells 1,403–1,405 (frozen before any number is computed)

Motivation (`REPORT.md`, `DECOMP.md`): base entries that later break the high earn +0.25 / +0.39 R under C1; the
18–24 % that never break lose −1.06 / −1.13 R. If breaks come early and the non-breakers are recognisable by the
clock, leaving when no break has happened cuts the −1.1 R cohort. The time-to-break distribution has NOT been
looked at; N is fixed here.

## Everything identical to cells 1,400–1,402 except the exit
Population, signal, entry, stop B_t, R, stop-at-open handling, cost (15 bps half-spread both legs + 2 bps/side
proxy), splits, fill convention (walk from the bar after the entry bar; stop before target within a bar): as
`PREREG.md` and `run_consol.py`. Data: the cached `signals.parquet` and `paths.parquet` only (no DB access).
TEST is sealed and not computed.

## Exit
C1 (target entry + 2 R, stop B_t, flat at the 15:55 open) PLUS the no-break stop: if no 1-minute CLOSE > H_t
occurs in the bars entry_m+1 … entry_m+N, exit at the OPEN of the first bar after entry_m+N. A close > H_t sets
the flag after its bar, so it acts from the next bar. **N = 15 (1,403), 30 (1,404), 60 (1,405).**

## Placebo
D1 only: same name-day, seeded random minute 10:00–14:00, entry = that bar's open, stop = prior-20-bar low, 1.5 %
floor (≤ 5 redraws), H = high of day through the previous bar, IDENTICAL exit including the no-break stop with the
same N. D3 is not run: D1 was the binding placebo in all three earlier cells (D1 ≥ D3) — disclosed.

## Primary book and pass bar
Primary = the SLOTTED book (first 12 per day, 4 concurrent — `run_consol.simulate_slots`), the book live would
trade. **Pass (all):** slotted net ≥ +0.10 R on TRAIN and on VAL; VAL trade-weighted day-clustered t ≥ 2 on the
slotted book; both TRAIN halves > 0 (slotted); ex-top-5 % > 0 on both splits (slotted); beats D1 by ≥ +0.10 R on
VAL (all-signal book vs the D1 mean); ≥ 3 fills/week on VAL (slotted). Also reported: the all-signal book, exit
mix, and the exhibit of time-to-first-break for the breakers (share within 15 / 30 / 60 / 120 min) and
P(eventual break | no break by N).

## Verification of any pass
Causality trace of the no-break rule (uses closes ≤ entry_m+N, acts at the next open), independent rebuild from
this prose, measured-NBBO re-score — before anything is reported as a pass.

## Not allowed
Any other N; stacking with the C2/C3 exits; any TEST row. Programme count 1,405.
