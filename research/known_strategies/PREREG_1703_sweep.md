# PREREG — cells 1,703a–e: the documented-strategy sweep (FROZEN 2026-10-02 evening, before any number)

Owner 10/2: "I brought the top 10/20 idea, why? why wasn't it you? it is a known strategy, how many more are we
missing?" Answer owed as cells, not prose. Candidates = strategies with published evidence, implementable long-only
with FREE daily data on Alpaca, not yet tested here (prior work: research/momentum_weekly/SURVEY_multiday_prior_work.md;
closed: PEAD ×2, overnight, pre-FOMC, ETF asset-class rotation as a diversifier, residual and continuous-path momentum).

| Cell | Strategy (published source) | Spec (fixed now) |
|---|---|---|
| a | 52-week-high momentum (George & Hwang 2004) | U2 universe; rank by close ÷ 52-week high; top 20; Monday open; 1/N reset; costs as the sleeve |
| b | Dual momentum (Antonacci 2014) | monthly: SPY vs EFA by 12-month return; hold the winner if its 12-month return > 1-month T-bill proxy (SHY), else IEF; from the panel's ETFs |
| c | Weekly short-term reversal (Lehmann 1990; Jegadeesh 1990) | liquid names (prior close ≥ $10, ADV20 ≥ $200M); buy the 20 worst prior-week returns, Monday open, hold one week; measured cost 10.5 bp per traded dollar (1,700w) |
| d | Low-volatility (Blitz & van Vliet 2007) | U2; 20 lowest 252-day volatility names with positive 12-month return; monthly; 1/N |
| e | Crypto time-series momentum (Moskowitz–Ooi–Pedersen 2012 applied to BTC/ETH; Liu & Tsyvinski 2021) | BTC and ETH daily bars from Alpaca crypto (free); long when the 20-day return > 0, else cash; daily; 10 bp cost; 2018-01 → 2026-09 |

Each cell is ONE fixed specification — no grid, no tuning; the point is the strategy's documented form.

## Reads (2017-01..2026-09, $50K; halves 2017–2021 / 2022–2026; e from 2018)
Stand-alone: CAGR, max DD, CAGR/DD, Sharpe, worst year, years beating SPY, turnover, cost drag. As a STACK component
with the guarded sleeve (GREF 28.7 % / −38.1 %): weekly-return correlation (whole and inside GREF's three deepest
episodes), the 50/50 portfolio's CAGR, max DD and ratio, and the portfolio that holds the sleeve at 100 % and adds
the component at 50 % (the "stacking" read, leverage 1.5 — reported, not recommended).

## Pass rule (a component is recommended for a paper sleeve only if)
Stand-alone CAGR ≥ 10 % with max DD better than −30 % AND correlation with GREF ≤ 0.5 AND the 50/50 portfolio's
CAGR/DD ratio ≥ GREF's (0.75) + 0.10 AND both halves agree in sign on the component's return. A stand-alone that
beats GREF on both CAGR and DD is reported as a candidate replacement (none expected). Cells: +5 (1,703a–e).

## Output
`research/known_strategies/1703_sweep.py` (reuse the momentum engine's memory-safe panel load; crypto fetched to
`1703e_crypto.parquet`), `1703_cells.csv`, `RESULT_1703.md` (≤ 70 lines, adversary caveats, one verdict per cell).
Through `bash scripts/research_run.sh -m 4000M` (service down; one process). Agent returns ≤ 180 words.

## Amendment 1 (after the first run, before any re-read): cells a and b are inconsistent with their literature
First run: a (52-week high) −4.8 % / −58.7 %, b (dual momentum) 6.0 % / −38.7 %. Published forms: a ≈ market-plus
with momentum-like drawdowns; b ≈ 10–15 % with max DD near −20 % (2017–2026 it held SPY most of the time). A null
is a claim about my test first: both are re-implemented from THIS prose by a builder who has not read 1703_sweep.py.
a: ratio = prior close ÷ max(high over the trailing 252 sessions); ties (ratio = 1.0) broken by the 126-session
return (the PREREG left ties undefined — that is the suspected defect); also the published form: monthly, top 30 by
ratio, 6-month overlapping hold, as the reference. b: on the last session of each month, 12-month total return of SPY,
EFA, SHY; hold SPY or EFA (the higher) if its return > SHY's, else IEF; print the holding per month and the monthly
return; hand-check 2020-03, 2020-04, 2022-01..2022-12. Cells c, d, e stand as run. +0 cells (re-implementation).
