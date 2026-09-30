# PREREG — cell 1,677: the model-gated take-profit — the one action a success predictor has not been asked to do (FROZEN 2026-09-30 07:40 UTC)

1,676 (leak fixed) found a REAL post-entry shape signal for short-horizon success: P(+1 R within the next 15 min)
out of sample AUC 0.73 at k = 5 and 0.87 at k = 15 (placebos 0.51–0.66), on the rolling 5/10/15-minute candles without
the distance-to-target column. Cut, short and add keyed on it do not pay (the add at k = 15 loses −0.12 R, t −3):
a new unit bought at the current price with the original stop has the wrong geometry. The action that matches a
success predictor is the EXIT decision of the position already held: when the model says the next 15 minutes are
unlikely to add +1 R and the trade is in profit, take the profit instead of giving it back (AXTL 9/29: +$170 → −$96
at the forced close). The exit lab's locks, trails and time stops were price-only; a model-keyed take-profit has never
been read.

## Population, cost, halves
The 1,663 join, floored r_pct ≥ 1.5 %; the per-fill × k tables from 1,670 (`1670_per_fill_k.csv`: open-at-k, mtm R,
ALL-family P per scoring) and 1,676 (`1676_features.csv` G7 columns; the G7-without-close_R + ALL success model per
k and scoring — reuse the saved out-of-sample scores; NEVER re-fit). Cost: take-profit exit 6 bps (marketable sell at
the next bar's open); the base book's exits as before.

## Rules (fixed)
TP(k, m, τ_low): at the close of bar fill+k, k ∈ {5, 10, 15, 30, 60}, if the trade is open, mark-to-market ≥ m R
(m ∈ {0.5, 1.0}) and P(+1 R within the next 15 min) < τ_low (τ_low ∈ {0.3, 0.4, 0.5}), sell the whole position at
the open of bar fill+k+1; otherwise hold to the standard exit. Variant P50: sell half, keep half with the standard
exits (book in original-R units). Variant TS: the same rule evaluated at EVERY minute from 5 to 60 (the first firing
wins) — the rolling version.

## Reads (both scorings; the paired book = the base long book on the same fills)
Paired ΔR vs holding, iid and day-clustered t, ex-top-5 % ΔR, MDE, share fired, and the decomposition per fired
trade: give-back saved (base R − exit R on trades that ended below the exit) vs continuation forgone (on trades that
ended above), cost; plus the same rule with the model replaced by a coin flip at the same firing rate (the placebo
rule) and by the price-only trail from the exit lab (X6, MFE − 1 R) as the comparator.

## Pass bar
Paired ΔR ≥ +0.05 R, day-clustered t ≥ 2.5, ex-top-5 % ΔR > 0 on BOTH scorings, beating the placebo rule by ≥ 0.03 R
on both. A pass → independent rebuild before the owner sees a number; then paper as ONE mechanism.

## Multiplicity
5 k × 2 m × 3 τ × 3 variants × 2 scorings = 180 paired reads (+ comparators). Programme: > 5,000 on the HOD line.

## Not allowed
Fitting τ_low, m or k; using bars after the decision bar; re-fitting any model; pooled-only numbers.

## Output
`research/hod_entry/RESULT_1677.md` (≤ 100 lines), `1677_reads.csv`, `1677_per_fill.csv`, `1677_take_profit.py`,
`1677_take_profit.log`. The agent returns ≤ 120 words.
