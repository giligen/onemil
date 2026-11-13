# PREREG 1,761 — B3 (daily form): 52-WEEK-HIGH / 20-DAY-HIGH breakout on daily bars (George–Hwang anchoring) — Stage 1 rejection-only

FROZEN 2026-11-13 19:40 UTC before any number. Programme cell count: 1,760 → 1,761. Rules as PREREG_1754 (write
only `research/signal_search/1761_*`, `RESULT_1761.md`; daily bars on disk, loader as 1,754 with RAW opens;
`ulimit -v 2500000; nice -n 19`, peak RSS < 1 GB; budget ≤ 40 calls). TRAIN 2016–20 (halves 2016–18 / 2019–20),
VAL 2021–23 once, TEST 2024 → sealed. Survivorship caveat as 1,746.

## Claim (George–Hwang JF 2004: nearness to the 52-week high predicts returns; anchoring; the effect is strongest
## at the FIRST close above the prior high and in small caps)
Universe at t−1: ranks 1–3,000 by 20-day dollar volume, price ≥ $3. Event: close_t is the first close above the
prior 252-day high (primary) / 20-day high (secondary) after ≥ 20 days below it; volume_t ≥ 1.5× its 20-day mean
(attention, causal at the close). Entry at the OPEN of t+1 (auction, no spread); exits at the close of t+5
(primary) and t+20; cost 2 bp per leg + the measured half-spread sensitivity on the exit only. Reads: 252-d vs
20-d; by rank tier (1–1,000 / 1,001–3,000); by price tier; with vs without the volume condition.

## Numbers (per split, per read)
n/wk, mean gross excess vs the universe over 5 d and 20 d, day-clustered t, ex-top-5 %, winner-capped +20 %,
monthly green share, year signs, net, MDE at 80 % power.

## Kill (primary = 252-d, rank 1,001–3,000, 5 d, with volume)
VAL gross excess ≤ +30 bp or |t| < 2; TRAIN halves opposite sign; ex-top-5 % ≤ 0 on VAL; 2022 negative; net ≤ 0;
< 3 events/wk. Survive → TEST once on PIT → independent rebuild → paper sleeve spec.

## Output
`1761_score.py`, `1761_summary.csv`, `RESULT_1761.md` ≤ 350 words ending with "CHILDREN:" (≥ 2). Return ≤ 120 words.
