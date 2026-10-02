# PREREG — cell 1,700u: the term-structure gate on the GUARDED sleeve (FROZEN 2026-10-02 18:00 UTC, before any number)

1,700s was PARTIAL (23 of 40): the VIX ÷ VIX3M gate improved 19 of 20 cells, the VIX-level gate 4 of 20. But those
cells ran on the unguarded reference, and 1,700t showed the reference's deepest drawdown was partly one name (AMC,
2021: the guard alone lifts 27.2 % / −44.5 % to 29.3 % / −38.3 %). The gate's median cell (28.5 % / −38.6 %) may be
the same AMC weeks avoided a second way. Also my 1,700s mechanism condition was mis-specified (dropping the worst 5 %
of gated weeks always raises their mean) — corrected below. This cell asks: is anything left on the guarded book?

## Engine
1700s_lowvix.py's daily engine WITH the 1,700t guard = the new reference GREF (must reproduce 29.34 % / −38.3 % /
$596,394). Gate decided at the Friday close, applied from Monday's open, costs on the traded notional, cash earns 0.

## Family (20 cells, the axis 1,700s separated; declared now)
VIX ÷ VIX3M trailing percentile, window {252, 504} × gate below {10, 15, 20, 25, 30} % × action {cash, half size}.

## Reads per cell (2017-01..2026-09, $50K; halves 2017–2021 / 2022–2026)
CAGR, max DD, ratio, end $, share of weeks gated, spells, the sleeve's mean weekly return in gated vs ungated weeks
(whole and per half, with the difference's t), GREF's three deepest episodes and each cell's depth in them, by-year
gated share, paired weekly difference vs GREF (mean, t, ex-top-5 % — information only).
Out-of-sample era (pass condition): SPY weekly returns 2008–2016 (before the sleeve's sample; VIX3M history starts
2007-12), gated vs ungated mean for each of the 10 distinct gates.

## Pass rule
A cell "improves" if CAGR ≥ GREF's AND max DD is ≥ 3 points better. The gate is REAL only if ≥ 15 of 20 improve AND
the ratio beats GREF's in BOTH halves in ≥ 15 of 20 AND the gated-week mean is below the ungated-week mean in BOTH
halves in ≥ 15 of 20 AND the median cell is shallower by ≥ 3 points in ≥ 2 of GREF's 3 deepest episodes AND no
calendar year holds > 40 % of the median cell's gated weeks AND on SPY 2008–2016 the gated-week mean is below the
ungated mean for ≥ 8 of the 10 gates. If real: the MEDIAN cell goes to an independent rebuild, then a default-off
flag on the paper sleeve for the owner's go. Otherwise the lead is closed as the AMC episode seen twice.
Cells on this line: +20. Nothing outside the family after numbers.

## Output
`1700u_gate_guarded.py` (import/reuse 1700s_lowvix.py; one process, panel loaded once), `1700u_cells.csv`,
`RESULT_1700u.md` (≤ 60 lines, adversary caveats). Through `bash scripts/research_run.sh -m 2500M`. Agent returns ≤ 150 words.
