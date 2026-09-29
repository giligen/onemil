# PREREG — cells 1,655–1,657: THE FIRST FIVE SECONDS — is an instant ORB break a loser, and does a 5-s placement delay cost anything?

FROZEN 2026-09-29 08:20 UTC before any number (owner: "test the 5 sec issue now"). Programme count: 1,654 → 1,657.

## Question
`research/exec_quality/REPORT_20260928.md` §6: on 301 live ORB fills replayed on ticks at zero delay, breaks whose first
tick through the trigger came 0–5 s after 09:35:00 (n 42) averaged −$23.5 with 19 % winners, while 15–30 s breaks
averaged +$80. Pre-placement (live today) rests the buy-stops at 09:35:00.0 and therefore takes the instant breaks
first. One bucket of five on 42 fills is not a rule; the backtest population must show the same before anything changes.

## Population and data (on disk, no purchases)
The frozen population and cached ticks of cell 1,426 (`research/orb_latency_bt/population.csv`, `raw/`, `replay.py`:
every BT fill of the live-config ORB replayed on XNAS ticks with the engine's chase rule; out-of-sample book = 2023-24
+ 2025H2–2026-09; the May–Sep 2026 live window separately). Coverage reported: fills with ticks / fills total.
t* = seconds from 09:35:00.000 ET to the first tick at or through the trigger price.

## Cells (delay grid of 1,426 reused; exit rule unchanged = the live exit rule)
* 1,655 BUCKET TABLE: mean R, day-clustered t, n, win %, ex-top-5 % by t* bucket (0–5, 5–15, 15–30, 30–60, 60–300 s) at
  delay 0 — on the out-of-sample book, on 2025H1 (the in-sample half) and on the May–Sep 2026 live window; also the
  live-fill table from the report beside it.
* 1,656 DELAY-5: the buy-stop is armed at 09:35:05.000 instead of 09:35:00.000 — a name already through the trigger at
  09:35:05 fills at the first tick ≥ 09:35:05 subject to the same chase guard (skipped if the guard rejects); others
  fill at their t* as today. Mean R/fill, total $, n filled, n guard-skipped vs the delay-0 baseline, paired ΔR per
  fill with ex-top-5 % of ΔR (the tail check).
* 1,657 SKIP-INSTANT: fills with t* < 5 s are excluded for the day (the slot is not refilled — no refill after a veto).
  Same reporting.

## Pass bar for a live rule (frozen)
A `preplace_submit_delay_s: 5` (1,656) ships to the paper session only if: the 0–5 s bucket in 1,655 has mean R < 0
with day-clustered t ≤ −1.5 on the out-of-sample book AND the same sign on 2025H1; AND 1,656's mean R/fill ≥ the
delay-0 baseline − 0.005 R with ex-top-5 % of ΔR ≥ −0.01 R (the delay must not cost the tail winners). 1,657 is
report-only (a veto is a bigger change than a delay). If the 0–5 s bucket is not negative in the backtest population,
the live −$24 is noise on 42 fills and nothing changes; the paper session's own histogram is still recorded.

## Independent check
Report-only cell on an existing frozen harness: one build; the bucket table must reconcile with 1,426's published
delay-0 totals (same n, same total $ within $1) as the consistency check. Refuters: the t* definition (first tick AT
the trigger vs THROUGH it — report both), the chase-guard rejects at 09:35:05, ticks missing for a fill (coverage),
tails (ex-top-5 % of each bucket).

## Not allowed
Choosing the delay (5 s) or the bucket edges after a number.
