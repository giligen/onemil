# PREREG — cell 1,668: post-entry failure detection — cut the trade early when the first minutes say it will stop out (FROZEN 2026-09-29 18:31 UTC)

Owner, 2026-09-29: "we should also invest in failure detections… we entered, but then we find out that we better cut our
losses now due to indicators that increase the likelihood that this trade will stop out."

## What is already known (so this cell tests only what is not)
* Exit lab 9/22 (35 cells, `research/hod_exit_lab/REPORT.md`): breakeven lock, MFE trail, 30/60-min time stops, dip
  exits — all within ±0.04 R of the base; the post-entry path in R units behaves as a random walk around a small drift,
  so price-path-only exits neither add nor subtract. Re-read at the measured cost (cell 1,660): no candidate.
* Stops under 1.5 % (cell 1,658) lose −0.11 R net on both halves: any early cut is a tighter stop in disguise and pays
  the 13-bps round trip more often; a driftless first-passage bet is not improved by cutting it short.
* What has NOT been tested as an exit trigger: information that is not the price path — volume after the break, the
  market's move after entry, VWAP position — and a trained post-entry classifier scored out of sample.

## Population and cost
`1663_features.csv` join (fills_1658 ⋈ causal_arming_causal), primary book r_pct ≥ 1.5 % (n 5,506), unfloored beside.
Post-entry path from `research/bf_zero/bars_sip.db` (same-day bars after the fill bar; coverage ≈ 81 %, rail applies:
≥ 80 % of the book and winner/loser missingness gap ≤ 5 pp, else VOID). Cost: entry 7 bps (already in net_R), stop
exit 6 bps, target fill 0, EOD 11 bps, EARLY CUT 6 bps (a marketable sell at the next bar's open). Halves: `split`.

## Part A — pre-declared rules (an early cut at minute k after the fill, else the standard exit)
Horizons k ∈ {3, 5, 10, 15} minutes after the fill bar. At the close of bar fill+k, exit at the open of bar fill+k+1 if:
 S1 withdrawal: close(fill+k) < level                        S2 no new high: max high(fill+1..fill+k) ≤ fill price
 S3 negative return: close(fill+k) < fill price              S4 volume fade: mean v(fill+1..fill+k) < 0.5 × v(break bar)
 S5 market: SPY return fill→k < −0.2 % (bars_sip SPY; VOID if absent)   S6 below VWAP: close(fill+k) < VWAP(open..fill+k)
 S7 drawdown: min low(fill+1..fill+k) ≤ fill − 0.5 × (fill − stop)      S8 S2 and S4 together
Exits are evaluated bar by bar; a stop or target hit before minute k takes precedence (stop before target in a bar).
Reads per rule × horizon × half: n fired, share fired, paired ΔR vs the base on the whole book, iid t, day-clustered
t, ex-top-5 % ΔR, MDE beside every t; plus the ΔR on the fired subset alone (what the cut saved or cost).

## Part B — a trained failure classifier, scored out of sample
Features at k = 5 and k = 10: the continuous versions of S1–S7 (distance to level, MFE, return, volume ratio, SPY return,
distance to VWAP, MAE in R) plus r_pct and ATR %. Label: the trade later exits at the stop (vs target or EOD).
Gradient boosting (sklearn HistGradientBoosting, default depth, 200 rounds, no tuning). Train on TRAIN-H2, score VAL;
then SWAP (train on VAL, score TRAIN-H2). Exit rule: cut at minute k when P(stop) > τ, τ chosen on the training half
to maximise paired ΔR (τ is fitted; the training-half ΔR is in-sample and reported only as a caveat). Reads per k:
out-of-sample AUC, out-of-sample paired ΔR with t (iid and day-clustered), ex-top-5 % ΔR, share cut, and a placebo
(labels shuffled within day → AUC and ΔR must be ≈ 0.5 and ≈ 0).

## Pass bar
Part A: a rule × horizon ships to paper only if paired ΔR ≥ +0.05 R with t ≥ 2.5 in BOTH halves, ex-top-5 % ΔR > 0 in
both, and the base book's fills/week unchanged (an exit rule removes no fills). Part B: out-of-sample ΔR ≥ +0.05 R,
t ≥ 2.5, ex-top-5 % > 0 on BOTH out-of-sample scorings (VAL and the swap), placebo ≈ 0. A pass triggers an independent
rebuild from this prose before any number reaches the owner. A null states MDE and coverage first.

## Multiplicity
Part A: 8 rules × 4 horizons × 2 halves = 64 paired reads (+ 64 fired-subset reads). Part B: 2 horizons × 2 scorings
× 3 reads = 12. Programme count on the HOD line after 1,667: ≥ 1,900 → ≥ 2,000 with this cell.

## Not allowed
Adding rules or features after seeing numbers; tuning k, the 0.5 × volume ratio, the −0.2 % SPY move or the 0.5 R
drawdown; any read that uses bars after the cut minute to decide the cut; pooled-only numbers; reporting the
training-half ΔR of Part B as a result.

## Output
`research/hod_entry/RESULT_1668.md` (≤ 150 lines: coverage first, Part A table of every rule × horizon with both
halves, Part B table, verdicts, adequacy review), `1668_reads.csv`, `1668_per_fill.csv` (fill_id, split, base exit,
per-rule fire flags at each k, P(stop) at k = 5/10), `1668_failure.py`, `1668_failure.log`. The agent returns ≤ 150 words.

## Amendment 1 (2026-09-29 18:52 UTC, before any number; owner's question on price movement vs volume)
Add rule S9 "effort without result": at minute k, the high since the fill has not exceeded fill + 0.25 × (fill − stop)
AND the mean volume of bars fill+1..fill+k ≥ 1.5 × v(break bar) — heavy volume that produced no upward progress (buying
absorbed). S8 is its mirror (no progress on thin volume). Part B gains the continuous feature "progress per unit volume"
= (close(fill+k) − fill) / (fill − stop) ÷ (Σ v(fill+1..fill+k) / v(break bar)). Multiplicity: Part A becomes 9 × 4 × 2
= 72 paired reads. Thresholds 0.25 R and 1.5× are fixed here and never tuned.
