# PREREG — cell 1,675: short the HOD-break failure — sealed 3-month forward test, week by week, borrowability (FROZEN 2026-09-30 06:40 UTC)

Owner 06:30 UTC: "do an in-depth test of the short HOD break failures, show me a week-by-week breakdown of the past
3 months; if it looks good (borrowable symbols etc.) implement it for today for paper trade."

## What is being tested
Cell 1,673's best rule: at the close of bar fill+1 of a HOD-break fill, if the FF10 model (stop-out within 10 min,
trained in 1,669 on TRAIN-H2 and, separately, on VAL) gives P ≥ 0.6, sell short one unit at the open of bar fill+2;
target = the long's stop level (1 R in the long's units); stop = the day's high so far + $0.01; cover at the 15:55 ET
close otherwise. Costs: 7 bps entry, 6 bps cover, 11 bps EOD cover. The 1,673 read was +0.065 / +0.062 R per short
(t 1.1 / 1.6) on 2025-07..2026-05. The months 2026-06-01..2026-09-26 were never used by any HOD cell or model: they
are a SEALED forward test. The models are NOT re-fit; both trained versions are scored and reported separately.

## Population for the new months
The same HOD-break rule (the 1,438 definition: the live scanner predicate with correct levels, resting stop-limit at
level + $0.01, limit +0.15 %, first cross per symbol-day, stop ≥ 1.5 % of the fill) built by the existing pipeline
(`research/hod_entry/causal_arming.py` on candidates from `research/bf_zero/build_candidates.py`; daily universe from
`data/cache.db daily_bars`; minute bars appended to `research/bf_zero/bars_sip.db` through the designed appender
`research/bf_zero/backfill_bars_sip.py` for the candidate symbol-days and each fill's own day — Alpaca SIP, free).
Report: candidate symbol-days, fills, coverage; if the pipeline cannot be run for new dates within budget, say so and
STOP — no approximation with a different population.

## Reads
1. Week-by-week table (13 weeks): shorts fired, shortable-and-easy-to-borrow share (Alpaca asset flags as of today,
   stated as a proxy), hit rate, mean net R, sum R, $ at $150 risk per short, worst day, max concurrent shorts, SSR
   skips; for BOTH trained models. Pooled: mean net R, week-clustered t, ex-top-5 %, MDE at n.
2. The same with the ETB filter applied (only shortable + easy-to-borrow symbols).
3. Splits: fills the live cap would have taken (first 12 per day) vs the rest; stop bucket 1.5–3 % vs ≥ 3 %; time of
   day; the reversal variant (for held longs: sell 2 units = cut + short, P&L vs holding the long).
4. Gapped-through share and the same table with gapped exits priced at the next bar's open (pessimistic).
5. Frequency at the live config: shorts/week under the 12-fills-per-day cap.

## Go / no-go for today's paper session (fixed now)
GO only if, on BOTH trained models, with the ETB filter: mean net R ≥ +0.05 per short, week-clustered t ≥ 2.0,
≥ 8 of 13 weeks non-negative, worst week ≥ −3 R, ETB share ≥ 70 %, ≥ 3 shorts/week under the cap, and the pessimistic
gap pricing keeps the mean ≥ +0.03 R. Otherwise the mechanics ship OFF (flag false) and the read continues on paper as
telemetry only (the engine logs the signal without trading it).

## Multiplicity
This is a single pre-declared rule on sealed months; the reads are descriptive splits of one book. Programme: > 2,700.

## Output
`research/hod_entry/RESULT_1675.md` (the week table first), `1675_weeks.csv`, `1675_per_short.csv`,
`1675_forward.py`, `1675_forward.log`; the serialized models + feature spec for the engine:
`research/hod_entry/models/ff10_k1_trainh2.joblib`, `ff10_k1_val.joblib`, `ff10_k1_features.json`, and the feature
code factored into an importable module `trading/hod_failure_features.py` (pure functions on bar arrays: the research
scoring and the live engine MUST import the same function — parity by construction). The agent returns ≤ 150 words.
