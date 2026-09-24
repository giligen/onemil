# PREREG — the LIVE ORB book on 18 more months of a different regime: 2023-01-03 .. 2024-06-28. Cells 1,418–1,419

Why: on the untouched 2024H2 half-year the live ORB config was flat (−0.007 R/fill, n 59; `research/orb_2024/`), which
rejects the 2025 level (+0.272) but cannot exclude a modest edge (+0.10 R, ~1.4 SE). Owner GO 2026-09-24 for the
data. 18 months ≈ 3× the fills → SE ≈ 0.05 R: this pass decides whether ORB has an edge outside 2025–26. Frozen
before any 2023 minute bar exists.

## Universe (point-in-time, parity with 2025–26 and 2024H2)
Tickers: every symbol that traded on XNAS.ITCH 2023-01..2024-06 (Databento ohlcv-1d, $5.51, delisted included; `-`
preferreds and `^Z[A-Z]ZZT$` test tickers excluded). Thresholds on CONSOLIDATED Alpaca SIP daily bars
(`research/orb_2023/daily_alpaca.parquet`): open ≥ 1.03 × prior close, open $3–50, prior-day volume ≥ 500,000; then
09:30–09:35 RTH volume ≥ 15,000 from the minute bars. Cell 1,418 = production seed (gap ≥ 5 %, open $3–30); cell 1,419
= the `addon_p30` pool (gap 3–5 %, open $30–50), each walked alone. Availability rail ≥ 80 % of candidates with bars.

## Book
Identical to `research/orb_2024/PREREG.md`: `study_orb_features.py` + `study_orb_pipeline_static_lock.py` with the
`orb.yaml` literals as of 2026-09-23 (no refit), 8 slots, $375, catalyst veto OFF, PDR / G1 / range-size vetoes ON,
static lock + touchgo + ATR floor + scale-out as configured. 20-day lookbacks from the consolidated daily file (the
2024H2 run's July lookback defect does not recur: daily bars start 2022-11-15).

## Verdict (per cell) — frozen
EDGE iff net R per fill ≥ +0.10 AND day-clustered t ≥ 2 AND ex-top-5 % > 0 AND both calendar halves (2023, 2024H1)
> 0. FLAT iff |net R per fill| < 0.10 with t < 2. NEGATIVE iff net R per fill ≤ −0.10. Pooled with 2024H2 (same code),
report the out-of-regime ORB estimate with its SE. Also: fills/week, monthly $, worst month, no-fill share.

## Consequence (pre-committed)
EDGE → ORB's out-of-regime edge is real; the ramp may advance on realized stage P&L as designed. FLAT → keep ORB at the
current stage indefinitely (no advance even on realized profit until 40 live fills confirm), tell the owner the
book is a coin flip outside 2025–26. NEGATIVE → recommend pausing ORB live.

## Not allowed
Refitting anything; changing the thresholds, the verdict or the consequence after a 2023 number exists.
Programme count 1,419.
