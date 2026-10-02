# PREREG — cell 1,700s: sit out the calm — a LOW-VIX gate on the sleeve (FROZEN 2026-10-02 18:25 UTC)

Owner 10/2: "I didn't tell you to exit on high VIX, maybe exit on low VIX." Honest provenance: this is read off
1,700q's table AFTER seeing it (selection from 36 tests), so it is a LEAD, not a result: in the calmest fifth of weeks
the sleeve averaged −0.26 %/week by VIX level (other fifths +0.35..+1.34) and −0.46 %/week by VIX ÷ VIX3M (up weeks
51 %), spread t 2.2 / 2.7, same sign in both halves; SPY 1993–2026 lowest fifth +0.06 %/week vs +0.54 % highest
(t 2.4). Mechanism candidate: complacency — low implied volatility and a steep term structure mark crowded, fully
priced momentum; the risk premium is paid when fear is high. The neighbour family decides whether it is real.

## Engine
The reconciled sleeve on the 1700j/1700l DAILY engine (REF must reproduce 27.18 % / −44.5 % / $507,823). Gate decided
at the Friday close, applied for the week from Monday's open; switching costs as 1,700c on the traded notional; cash
earns 0. Gauge series from `1700q_series.parquet` (re-verify the shift test: 0 mismatches).

## Family (40 cells, declared now)
gauge {VIX level, VIX ÷ VIX3M} × percentile window {trailing 252 days, trailing 504 days} × threshold: gate when the
gauge's trailing percentile is below {10, 15, 20, 25, 30} % × action {all cash, half size}.

## Reads per cell (2017-01..2026-09, $50K; halves 2017–2021 / 2022–2026)
CAGR, max DD, ratio, end $, share of weeks gated, number of gated spells, the sleeve's mean weekly return inside gated
weeks (whole, each half, ex the worst 5 % of gated weeks), the five 1,700j episode depths, years beating SPY /10,
worst year, paired weekly difference vs REF (mean, t, ex-top-5 %), by-year gated share (is it one era?).

## Pass rule
A cell "improves" if CAGR ≥ REF's AND max DD is better than REF's by ≥ 3 points. The family is REAL only if ≥ 30 of
40 improve AND the ratio beats REF's in BOTH halves in ≥ 30 of 40 AND the gated-week mean return is negative in both
halves and still ≤ 0 ex its worst 5 % in ≥ 30 of 40 AND no single calendar year holds > 40 % of the gated weeks in
the median cell. If real: recommend the MEDIAN cell (by ratio), then an independent rebuild, then a default-off flag
on the paper sleeve for the owner's go. 15–29 improve: "partial" — report the separating axis, no recommendation.
Also reported, not a pass condition: the same gate on SPY buy-and-hold 1993–2026 (VIX level only) as the
out-of-sample-era mechanism check. Cells on this line: +40. Nothing outside the family after numbers.

## Output
`1700s_lowvix.py`, `1700s_cells.csv`, `RESULT_1700s.md` (≤ 70 lines, adversary caveats). ONE process through
`bash scripts/research_run.sh -m 2500M`. The agent returns ≤ 150 words.
