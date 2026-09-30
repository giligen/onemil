# PREREG — cell 1,673: short the predicted failure (FROZEN 2026-09-30 05:20 UTC)

Owner 05:10 UTC: "find the errors and oversights in your research, there's money there." Oversight #1: cells 1,668–1,670
established that a stop-out is predictable after entry (fast failure within 5 min at minute 1: AUC 0.76–0.81 out of
sample, precision 0.4–0.5 at P ≥ 0.7; any later stop at minute 5–10: AUC 0.63–0.67) and then only ever asked whether to
CUT the long. A predicted stop-out is a predicted decline of ≥ r_pct (≥ 1.5 %) from the entry. The direct way to
monetise that is to be SHORT it. Every mirror-short cell so far (F52–F57, 1,439) was keyed on pre-entry features; a
short keyed on the post-entry failure model has never been read.

## Population, cost, halves
The 1,663 join, primary r_pct ≥ 1.5 %; halves = `split`; the completed bar store; the out-of-sample failure
probabilities ALREADY SCORED in `1669_per_fill.csv` (FF5 and FF10 at k = 0 and 1, both scorings) and
`1670_per_fill_k.csv` (P(stop after k), k ∈ {1, 2, 5, 10}, both scorings) — reuse them, never re-fit.
Short costs: entry 7 bps (marketable sell short at the next bar's open), cover 6 bps (stop or target), EOD cover 11
bps; locate/borrow fee ignored on paper and STATED; SSR rail: a fill whose stock is down ≥ 10 % from the prior close
(daily panel) is NOT shortable that day (skipped, counted).

## Rules (fixed, no fitting)
S(label, k, τ): at the close of bar fill+k, if P ≥ τ (τ ∈ {0.6, 0.7, 0.8}), sell short one unit at the open of bar
fill+k+1; target = the long's stop level (a 1 R move in the long's units); stop = the short's entry + 1 R (the same
distance up); exits bar by bar, stop before target inside a bar; EOD cover at the 15:55 ET close. Labels/k: FF5 at
k = 1, FF10 at k = 1, P(stop after k) at k ∈ {2, 5, 10}. Variant T2: target 2 R below (the long's stop distance
twice), same stop. Variant H: the short's stop at the day's high so far + $0.01 instead of +1 R (report R in the
long's units either way).

## Reads (per rule, per scoring, both halves of the scoring half)
n shortable, share of open-at-k fills shorted, mean net R per short, iid t, day-clustered t, ex-top-5 % mean, MDE,
hit rate, average holding minutes, worst day, the SSR-skipped count; and the PORTFOLIO read: the long book + the short
overlay on the same fills (paired ΔR of adding the short overlay to the base long book).

## Pass bar
A rule ships to paper only if the short's own mean net R ≥ +0.10 R (short entries cost more than 0.05 R would cover)
with t ≥ 2.5 in BOTH scorings, ex-top-5 % > 0 in both, ≥ 3 shorts/week at the live config, placebo (label-shuffled
P, from 1,669/1,670's placebo runs) ≈ 0. A pass → independent rebuild from this prose before the owner sees a number.
A null states MDE and the hit rate first.

## Multiplicity
5 (label, k) × 3 τ × 3 variants × 2 scorings = 90 reads + 90 portfolio reads. Programme count: > 2,600.

## Not allowed
Re-fitting or re-scoring any model; choosing τ, k or the variant after seeing numbers; bars after the decision bar for
the decision; ignoring the SSR rail; pooled-only numbers.

## Output
`research/hod_entry/RESULT_1673.md` (≤ 120 lines), `1673_reads.csv`, `1673_per_short.csv`, `1673_short.py`,
`1673_short.log`. The agent returns ≤ 150 words.
