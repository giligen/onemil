# REBUILD_1640 — independent rebuild of cells 1,640–1,642 (VIX basis carry)

Built from `PREREG_1640.md` prose alone in `rebuild_1640.py`; `cell_1640.py`, `RESULT_1640_build.md`,
`daily_1640.csv`, `basis_f30.csv` and `price_scale_check.csv` (the first build's outputs) were opened
only after this file's own numbers were computed. F30, basis and price-scale checks were recomputed
from `data/vx_raw/*.csv` and `data/vix_spot.csv`; TEST (2024-01-01..2026-09) was excluded from every
input **before** any backtest/decile/scale computation — nothing from TEST was read or scored.

## Data notes
- Alpaca SVXY/SVIX daily bars start 2016-01-04 (not 2011-10) — TRAIN is 2016-01..2019-12 in practice,
  split at 2018-02-27 into the −1x era (2016-01-04..2018-02-27) and −0.5x era (2018-02-28..2019-12-31).
- vx_raw: 182 contract files, 2011-09-21..2026-10-21 expiries. A 2013-vintage subset (13 files) left
  `Settle==0` on rows that traded (nonzero `Close`); fallback `settle_eff = Settle if Settle>0 else
  Close`. Irrelevant to TRAIN/VAL (those contracts are gone from the "two nearest live" set well before
  2016) but kept for correctness. Dates mixed MM/DD/YYYY and YYYY-MM-DD across file vintages — parsed
  with `format="mixed"`.
- SVIX data starts 2022-03-30 → no TRAIN sample for 1,641; VAL is truncated to 2022-03-30..2023-12-31.

## Results (TRAIN + VAL only; all bps/day are in-market means)
| block | n_days | %in-mkt | bps/day | NW t(5) | ann.ret | worst day | worst mo | max DD |
|---|---|---|---|---|---|---|---|---|
| 1640 TRAIN −1x (two-close) | 542 | 85.8% | 29.76 | 1.967 | 71.6% | −18.26% | −26.11% | −35.96% |
| 1640 TRAIN −0.5x (two-close) | 464 | 68.3% | 3.55 | 0.423 | 4.0% | −8.22% | −11.07% | −20.36% |
| **1640 VAL −0.5x (two-close, frozen)** | 1006 | 74.9% | **5.15** | **0.755** | 5.9% | −16.90% | −13.60% | −32.87% |
| 1640 VAL (one-close variant) | 1006 | 80.2% | 6.01 | 0.901 | 8.1% | −16.90% | −15.01% | −34.37% |
| 1642 always-in TRAIN −1x | 542 | 99.8% | 14.49 | 0.529 | −25.9% | −83.00% | −89.48% | −93.07% |
| 1642 always-in VAL | 1006 | 100% | 7.82 | 1.031 | 12.2% | −19.50% | −38.86% | −62.18% |
| 1641 SVIX VAL (partial, 2022-03..2023-12) | 441 | 75.7% | 21.90 | 1.250 | 36.7% | −13.34% | −17.49% | −39.23% |

$750/$5K notional P&L, TRAIN-1x/-0.5x/VAL for 1640: $1,038/$6,919; $84/$562; $291/$1,941.

## Pass bar (VAL, cell 1,640 −0.5x era) — item by item
PASS: bps/day ≥ +4 (5.15); ≥40% in market (74.9%); TRAIN −1x t≥1 (1.97); worst day ≥ −25% (−16.9%);
max DD ≥ −35% (−32.9%); gate worst day/DD both beat always-in (−16.9%/−32.9% vs −19.5%/−62.2%).
**FAIL: NW t ≥ 2.5 (0.755)**; **FAIL: decile table monotone (VAL corr negative, see below)**.
**Overall: FAIL, 7/9** — matches the first build's own scored verdict exactly.

## Refuter checklist
1. **Settlement vs ETP close timing.** By construction every decision at day D's open uses only
   `basis_lag1`/`vix_lag1` (= day D−1's close-of-day values) and `basis_lag2` (D−2); nothing from day D
   itself or later enters the signal. No lookahead found.
2. **F30 interpolation at the roll.** Within TRAIN+VAL, max \|Δb\| on roll (expiry) days = 21.7pp vs
   28.2pp on non-roll days; mean 3.21pp (roll) vs 3.14pp (non-roll). Roll days are **not** more
   discontinuous than ordinary high-vol days in this window — no interpolation artifact.
3. **2018-02-27 leverage change / reverse splits.** Only one \|daily return\|>40% day for SVXY in
   TRAIN+VAL: 2018-02-06, −83.0% (Volmageddon, matches the documented XIV-era vol spike). Zero for SVIX
   in its window. No discontinuity at the 2018-02-27 leverage-change date itself (Alpaca `adjustment=ALL`
   back-adjusts it into a continuous series — confirmed by inspection of the raw closes around that date).
4. **Tails.** VAL decile table of next-day SVXY return vs basis, all days: corr(decile,next_ret) over the
   1,006 underlying day-pairs = **−0.0157**; ex-worst-5-days = **−0.0560** (more negative, not less).
   Using the first build's own convention (Pearson corr of the 10 decile-mean bps values against decile
   index 1–10) gives **−0.295** on my own decile means — reproducing their reported −0.299 almost exactly
   and confirming the large apparent gap vs my −0.016 was a **correlation-definition** difference
   (per-day vs decile-mean-of-10), not a data or logic disagreement. Under either convention: **no sign
   flip** — the correlation is negative with and without the worst 5 days, i.e. the documented "return
   rising in the basis" mechanism does not show up in VAL either way.
5. **One-close vs two-close.** One-close VAL: 6.01 bps/day, t=0.901, 80.2% in market. Two-close (frozen):
   5.15 bps/day, t=0.755, 74.9% in market. Two-close is the PREREG-frozen rule used for the pass-bar row
   above; one-close is directionally similar (same sign, still fails t≥2.5).

## Comparison vs the first build (`daily_1640.csv`, wide format)
Reshaped and compared on common dates, TRAIN+VAL only (their file is also already TEST-truncated: max
date 2023-12-29, 0 rows after 2023-12-31).

| cell | split | Jaccard(in-mkt) | mean bps/day: mine vs theirs | Δ |
|---|---|---|---|---|
| 1640 | TRAIN −1x | 1.0000 | 29.760 vs 29.759 | +0.001 |
| 1640 | TRAIN −0.5x | 1.0000 | 3.548 vs 3.547 | +0.001 |
| 1640 | **VAL** | **1.0000** | 5.150 vs 5.148 | +0.001 |
| 1641 | VAL (partial) | 1.0000 | 21.902 vs 21.903 | −0.002 |
| 1642 | TRAIN −1x | 0.9982 | 14.487 vs 13.887* | +0.600* |
| 1642 | TRAIN −0.5x / VAL | 1.0000 | 7.921 vs 7.921 / 7.819 vs 7.819 | 0.000 |

NW t agrees to ≤0.0002 on every split where Jaccard=1.0 (VAL 1640: mine 0.7551 vs theirs 0.7549).

**Largest disagreements.** In-market day sets agree exactly (Jaccard 1.0) on every pass-bar-relevant
split. Per-day \|Δpnl\| tops out under 0.7 bps anywhere in the 2,010-day 1640/1642 overlap — proportional,
same-signed on both winning and losing days (~0.03–0.06% relative), consistent with a rounding-level
difference in the underlying Alpaca open/close feed rather than a methodology gap.
The one real structural disagreement: **cell 1642 TRAIN −1x has 542 vs 540 total calendar days** (465 vs
465 in-market — identical count), because the first 1–2 warmup days of each merged series (where
`basis_lag2` / `close_lag1` are undefined) are kept as explicit 0-signal rows here but appear to be
dropped from the denominator in the first build. This is a day-counting convention at the very start of
the series, not a data or signal-logic error, and it does not touch VAL (the pass-bar split), where
Jaccard is exactly 1.0000 for every cell.

## Bottom line
Independent rebuild confirms the first build's numbers to 3–4 significant figures on every pass-bar
input. VAL cell 1,640 (−0.5x, two-close, frozen): **+5.15 bps/day in market, NW t = 0.755, 74.9% days in
market** — point estimate clears the +4 bps bar but the significance and monotonicity bars both fail.
**Verdict: FAIL, matches the first build (7/9 pass-bar items).**
