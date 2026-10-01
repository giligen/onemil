# PREREG — cell 1,696: the missed runners at full coverage, and what "rank_not_selected" really is (FROZEN 2026-10-01 16:45 UTC)

From 1,694 Part C: 13,163 admitted ORB candidates 2025–26 = 476 taken + 7,021 vetoed + 5,666 no-fill. The bucket
"rank_not_selected" (1,937 candidates) reads +0.284 R with a runner share of 6 % and ≈ $19.7K/yr — on 16 % bar
coverage that is probably not random. The G1 veto bucket reads +0.55 R on n 82. Neither is a claim at that coverage.

## Steps (sequential, one writer on the bar store; runs after cell 1,689b's appender has finished)
1. Append the minute bars of every candidate in the rank_not_selected and G1-vetoed buckets (≈ 2,600 symbol-days)
   through the designed appender via the scratchpad wrapper; `df -h /` ≥ 5 GB first; report coverage after.
2. Split "rank_not_selected" into its real reasons by replaying the pipeline's per-day ranking (score quintile /
   skip_q1 / slot cap / dedup) — the study script's own functions, read-only; report the count per reason.
3. Counterfactual outcome per candidate under the production entry (break of the range high, chase cap) and exit,
   walked on the bars: mean R, day-clustered t, runner share (≥ 3 R MFE), $/yr at $375, per reason and per half
   (2025 / 2026), with the taken book beside it. Also the coverage-bias check: the 16 % covered earlier vs the newly
   covered rest, same stats.
4. Pre-declared promotion read: for each reason bucket with full coverage, mean R ≥ +0.15 in BOTH halves, runner
   share ≥ 15 %, and day-t ≥ 2.0 pooled → a gate-relaxation cell is written (e.g. "take the next-ranked candidate up
   to the slot cap", or "G1 veto off") with the union read (frequency, union mean, P10 per fill, strong-week gap);
   nothing changes in orb.yaml without that cell passing both directions and a rebuild.

## Not allowed
Reading the buckets before full coverage; pooling reasons; changing the production entry/exit model in the replay.

## Output
`research/orb_freq/RESULT_1696.md` (≤ 100 lines), `1696_missed_full.csv`, `1696_reasons.csv`, `1696_missed.py`,
`1696_missed.log`. The agent returns ≤ 150 words.
