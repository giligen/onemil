# PREREG — cell 1,678: make the post-entry shape signal pay — decomposition, the remaining-R regression, the ORB transfer, the flat-price subset (FROZEN 2026-09-30 15:05 UTC)

Owner 15:00 UTC: "go deeper on this, make it work, think out of the box." Facts on the table (1,676 leak-fixed, 1,677):
rolling 5/10/15-minute candles at minute k predict "+1 R within the next 15 min" out of sample (AUC 0.73 at k = 5,
0.87 at k = 15); shape ADDS ≈ 0 to the path family through 15 min and +0.03–0.07 AUC at 30–60 min; the model-gated
take-profit at k = 60 for trades ≥ +1 R is +0.30 R (t 2.3) and +0.13 R (t 0.1) on the two scorings — positive both
ways, n 43/42, MDE ≈ 0.5 R. Correction recorded: that rule did not flip sign; its t did.

## Part 1 — decomposition (what the signal is)
Per k ∈ {5, 10, 15, 30, 45, 60, 90, 120}, label "+1 R within the next 15 min from k" (forward-only) and the money
label "remaining R" (base exit R − mark-to-market at k): out-of-sample AUC / R² for (a) mark-to-market alone, (b) the
path family alone (mtm, MFE, MAE, minutes since last high), (c) G7 shape alone WITHOUT any close-vs-fill column,
(d) path + shape; the increment (d) − (b) with a 200-draw bootstrap CI; placebo at k ∈ {15, 60}. Both scorings.

## Part 2 — the right object: expected remaining R
HistGradientBoostingRegressor (defaults, max_iter 200, n_jobs 1) on features at k for the target "remaining R from
k" (base exit R − mtm_k), trained on one half, scored on the other, then swapped. Feature sets: path only; path +
shape. Decision rule X(c): at each minute k from 15 to 120 (first firing wins), if the trade is open and the
predicted remaining R < −c (c ∈ {0.05, 0.10, 0.20} R, FIXED), exit at the next bar's open (6 bps). Variant X+: only
when mtm ≥ +0.5 R (the give-back guard). Reads: paired ΔR vs the base on the whole book, iid and day-clustered t,
ex-top-5 % ΔR, MDE, share fired, the give-back-saved vs continuation-forgone decomposition, and the same rule with the
path-only model — the difference is the shape's money. Comparators: the 1,677 classifier rule at k = 60, the X6 trail,
and a placebo rule (same firing rate, seeded coin flip, 10 seeds).

## Part 3 — transfer to ORB (out of sample by construction)
ORB fills 2025-01..2026-09 from `analysis_results/orb_bplus_book.csv` (read through `trading/orb_csv.read_orb_csv`;
entry time, entry price, stop, target, exit time/price/reason). Minute bars: the symbol-days appended to
`research/bf_zero/bars_sip.db` through the designed appender (Alpaca SIP, free; wrapper pattern in the scratchpad),
coverage line first. Apply the HOD-trained Part 2 models UNCHANGED (both halves' models, reported separately) with
the ORB trade's own R units; decision rule X(c) and X+ from minute 15 after the ORB fill to its exit; paired ΔR vs the
ORB book's own exit, day-clustered t, ex-top-5 %, MDE at n ≈ 470 (state it: ≈ 0.17 R), the give-back decomposition,
and the dollar effect at the ORB live risk ($375). Also the ORB book's own give-back anatomy: share of fills that were
≥ +1 R at some minute and closed ≤ 0, and the R given back per fill.

## Part 4 — the non-tautological subset
At k ∈ {30, 45, 60}: fills with mtm ≤ +0.3 R AND P(+1 R next 15) from the shape+path model ≥ 0.6 (FIXED): the add
(1,670 R4 geometry) vs hold on that subset only, both scorings, plus the subset's own remaining R (does price follow
the shape when it has not moved yet?).

## Pass bar
HOD: paired ΔR ≥ +0.05 R, day-clustered t ≥ 2.5, ex-top-5 % > 0 on BOTH scorings, beating the placebo rule by
≥ 0.03 R on both. ORB: paired ΔR ≥ +0.05 R with t ≥ 2.0 on BOTH HOD-trained models and ex-top-5 % > 0 (n limits
power; a pass here goes to ORB PAPER as the exit rule with the forward read at 100 fills). Any pass → independent
rebuild from this prose before the owner sees a number. A null reports the decomposition first, then the MDE.

## Multiplicity
Part 1: 8 k × 4 sets × 2 labels × 2 scorings = 128; Part 2: 2 sets × 3 c × 2 variants × 2 scorings = 24 paired reads
(+ comparators); Part 3: 24 on ORB; Part 4: 3 k × 2 scorings × 2 = 12. Programme count: > 5,200 on the HOD line.

## Not allowed
Fitting c, k or τ; features that use bars after the decision bar; re-labelling with the exit type; pooled-only numbers;
any Databento spend; modifying the ORB book CSV.

## Output
`research/hod_entry/RESULT_1678.md` (≤ 180 lines: Part 1 table first), `1678_decomp.csv`, `1678_reads.csv`,
`1678_orb_per_fill.csv`, `1678_remaining_r.py`, `1678_remaining_r.log`, models under `models/1678_*`. The agent
returns ≤ 150 words.
