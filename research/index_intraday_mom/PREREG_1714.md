# PREREG — cells 1,714–1,716: market intraday momentum on SPY/QQQ (first half-hour → last half-hour). FROZEN 2026-10-29 21:20 UTC (Fable), before any number.

Owner's ask 10/29: "another strat". Survey (`research/sleeves/SURVEY_20261029.md`) + ideas list (`research/IDEAS_20260928.md`
#18, "cheap re-check") + closures: 4 factor sleeves (1,710–1,713), PEAD on the reaction (1,633–1,636), pre-earnings
run-up (1,646–1,648), multiday F1–F6/A1–A4 — all FAIL on daily-panel designs. This cell is a DIFFERENT design: one
liquid instrument, one 30-minute hold per day, ~250 events/yr (power), no survivorship, cost ≈ 1 bp, fires at 15:30 ET
(distinct from MOM Mon 09:46, TOM month-end, HOD/ORB mornings). Programme count 1,713 → 1,716.

## Mechanism (published)
Gao, Han, Li, Zhou (2018, JFE) "Market intraday momentum": the first half-hour return of SPY predicts the last
half-hour return (1993–2013, t ≈ 4, timing strategy ≈ +3–4 bps/day before cost). Baltussen, Da, Lammers, Martens
(2021, JFE) "Hedging demand and market intraday momentum": persists across 16 indices through 2020; strongest when
option dealers are short gamma — proxied here by a big move so far in the day (hedging flow into the close). Both are
on liquid index products, so the result cannot be a small-cap microstructure artefact.

## Data (on disk, verified 10/29)
`research/index_orb/cache/SPY_1min.parquet`, `QQQ_1min.parquet`: Alpaca SIP 1-min regular-session bars 2016-01-04 ..
2026-05-29, columns timestamp (America/New_York), open/high/low/close/volume, raw (no splits in SPY/QQQ in the window;
dividends do not touch intraday returns). Exclusions: days with < 380 bars (the index-ORB precedent) and early-close days
(no 15:30 bar). Report the excluded count per split.
Half-hour grid: r1 = close(09:59 bar) / open(09:30 bar) − 1; r12 = close(15:29) / close(09:59) − 1 (the day-so-far after
the first half-hour); rday = close(15:29) / open(09:30) − 1; TARGET r13 = close(15:59) / open(15:31 bar) − 1.
Obtainability: a market order sent at 15:30:30 fills at ≈ the 15:31 open (SPY spread 1 bp); exit = MOC at the official
close (auction, no spread); the proxy for the official close is the 15:59 bar close — report |15:59 close − next-day
official close proxy| is NOT available here, so state the proxy explicitly in the result.

## Cells (long and short; a 30-minute SPY short is bounded and cheap — report long-only beside long-short)
* **1,714 — sign timing (Gao et al.):** position = sign(r1) from 15:31 to the close. Variants reported, not chosen:
  sign(r1 + r12) (whole day so far), and r1 only when |r1| > its TRAIN median.
* **1,715 — hedging-flow variant (Baltussen et al.):** trade only when |rday| > the TRAIN 70th percentile of |rday|
  (frozen as a TRAIN-derived number, written in the result before VAL is read); position = sign(rday). Sensitivity: 50th
  and 85th percentiles.
* **1,716 — report-only:** per split, OLS of r13 on r1 and r12 (Newey-West not needed: one obs/day), the decile table
  r1 → mean r13 (monotonicity = the mechanism), per-year mean and t, by VIX-proxy tercile (20-day realised vol of
  SPY daily closes from the same bars), QQQ beside SPY throughout.
Costs: 1 bp per side at 1x (SPY/QQQ spread + nothing for MOC); sensitivity 3 bps. Leverage is reported as 3× the
return minus 3 bps/side (UPRO/TQQQ), never traded in the BT.

## Splits
TRAIN 2016-01 .. 2019-12; VAL 2020-01 .. 2022-12; TEST 2023-01 .. 2026-05 SEALED (not read until VAL passes).

## Pass bar (frozen; VAL, per cell, SPY is the sleeve, QQQ must agree in sign)
mean net ≥ +2 bps/trade AND iid t ≥ 2.0 (one trade per day; also report year-block bootstrap CI) AND ex-top-5 % of
trades > 0 AND winner-capped (+50 bps) > 0 AND ≥ 2 trades/week on average AND green-week share ≥ 55 % and above a
count-matched random-direction null (1,000 draws) AND TRAIN same sign with t ≥ 1 AND QQQ VAL mean net > 0.
Power: sd(r13) ≈ 25 bps, ~750 VAL days → se ≈ 0.9 bps → MDE ≈ 1.8 bps at t 2. The bar is detectable, unlike the
monthly sleeves (MDE 1.4 %/mo). Cadence bar (`docs/cadence_bar.md`): R := 25 bps (one sd of r13); report
`scripts/cadence_bar.py` on weekly R sums if the scorer accepts a series; else the weekly P10/green/strong-week table.
Undetermined: mean net > 0 but t < 2 → reported, no ladder. Reject: mean net ≤ 0 or ex-top-5 % ≤ 0.

## Ladder (pre-committed)
Pass → independent rebuild from this prose by an agent that has not seen cell_1714.py, compared day by day (date,
position, r13) → TEST read once, reported beside VAL → paper 10 sessions on the ORB paper account (idle at 15:30 ⟶
15:45 force-close does not collide; one SPY order/day, $20K 1x) → live $10K 1x on the live account after 10 clean paper
sessions → 3× via UPRO/TQQQ at $10K only after 20 live sessions realized > 0. Kill: 20 sessions realized < 0 or one
session < −1.5 %. Nothing to the owner before the rebuild.

## Builder contract
Script `research/index_intraday_mom/cell_1714.py` (argparse: --split train|val, --symbol SPY|QQQ, --cost-bps 1), writes
`RESULT_1714.md` (sections: data check incl. excluded days, frozen TRAIN thresholds, per-cell per-split table with every
pass-bar item, decile table, per-year table, weekly table, variants, caveats read as an adversary) and
`trades_1714_<cell>_<split>_<sym>.csv` (date, r1, r12, rday, position, r13_gross, net). TEST must not be run or read.
No Databento, no network, no writes outside `research/index_intraday_mom/`.
