# PREREG — published-strategy sleeve survey (frozen 2026-10-29, Fable; owner "get me money", "do it all")

Why: the one book running live with a documented prior (MOM sleeve) came from the owner's reading, not from 1,700
intraday cells (feedback 10/2: survey known strategies FIRST). Target: 3–5 sleeves at 1–2 %/month each that trigger on
DIFFERENT days (stacking, feedback 9/29), judged on the portfolio sum and the shared tail.

## Survey (Haiku/Sonnet agent, ≤ 40 calls, no data purchase, no code)
List ≤ 12 published, implementable US-equity/ETF strategies with (a) a citable source (paper or widely replicated
book/blog with a public backtest), (b) daily or lower frequency, (c) implementable on Alpaca with our data
(`daily_bars`, Databento EQUS.SUMMARY daily incl. delisted, 1-min bars), (d) a stated historical Sharpe/return and the
year the paper's sample ends. Columns: name, mechanism (one line), trigger day/frequency, holding period, universe,
published CAGR / Sharpe / max DD, post-publication evidence (decay?), data we already hold, cost sensitivity
(turnover × spread), correlation guess with MOM/TOM (Monday / month-end). Candidates to cover at minimum:
post-earnings-announcement drift (PEAD), turn-of-month (we run TOM — report its published prior for calibration),
52-week-high momentum (George & Hwang), short-term reversal (weekly), overnight-vs-intraday return (Lou et al),
dual momentum / GEM (Antonacci), sector rotation (Faber), low-vol anomaly, time-series momentum on ETFs (Moskowitz),
end-of-day/first-hour ETF effects, pre-FOMC drift (Lucca–Moench), index-rebalance / S&P inclusion.
Return the table in `research/sleeves/SURVEY_20261029.md` (≤ 120 lines) and ≤ 150 words.

## Selection rule (pre-committed, no owner menu)
Rank by: published Sharpe × post-publication survival (1 if OOS evidence exists, 0.5 if none, 0 if refuted) ÷
(1 + turnover-cost penalty) — then take the top 3 that fire on days different from Monday (MOM) and month-end (TOM).
Each selected sleeve gets its own PREREG (rule, universe, cost from measured NBBO, cadence bar `docs/cadence_bar.md`,
sealed TEST), a point-in-time backtest on Databento daily bars (survivorship-free), an independent rebuild, then
4 weeks paper at $10K, then live at $5–10K if realized paper P&L > 0 — the exploration-tier rule (feedback 9/18).
Kill: a sleeve is cut after 8 cycles with realized P&L < 0, or any single cycle < −3 % of its sleeve.
Schedule: survey by 11/2; sleeve #1 PREREG 11/3; backtest + rebuild by 11/9; paper from 11/10.
