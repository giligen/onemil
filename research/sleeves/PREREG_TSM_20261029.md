# PREREG — Sleeve #2: ETF time-series momentum (Moskowitz–Ooi–Pedersen 2012 / Faber 2007), frozen 2026-10-29 (Fable)

Sleeve #1 (52-week-high, cell 1,710) was REJECTED at Stage 1 (VAL −0.85 %/mo even with survivorship in its favour).
Next by the survey rule: time-series momentum on a FIXED ETF basket — no survivorship problem, no PIT data needed, so
the Alpaca daily panel (`research/momentum_weekly/panel_2016_2026.parquet` or Alpaca daily bars already cached) is adequate.

## Rule (one implementation, `research/sleeves/tsm_etf.py`; cell 1,711)
* Basket (fixed, frozen): SPY, QQQ, IWM, EFA, EEM, TLT, IEF, LQD, HYG, GLD, DBC, VNQ (12). Any ETF absent from local data
  is dropped and reported; never substituted.
* Signal on the last session of each month at the close: `mom12 = close / close 252 sessions ago − 1` (excess of the
  12-month T-bill return is ignored at today's rates — report the sensitivity with a 4 %/yr hurdle). Long an ETF when
  mom12 > 0, else cash (0 %). Equal weight across the longs: weight = 1/12 per ETF (cash fills the rest) — the frozen
  variant; sensitivity: weight 1/N_longs (fully invested in the longs). Orders at the next session's official open;
  live script at 09:46 ET like MOM; monthly turnover is low (≈ 10–20 % of notional).
* Cost: 5 bps round trip placeholder (ETF spreads ~1–3 bps), sensitivity 15 bps.
* Trigger day: first session of the month (same day as sleeve-#1 would have been; TOM EXITS the same day; MOM is
  Monday) — report the position overlap with TOM and the monthly return correlation with MOM / TOM / the live account.

## Evaluation
TRAIN 2017-01..2019-12 (252-session lookback eats 2016), VAL 2020-01..2022-12, TEST 2023-01..2026-09 SEALED. Per split:
mean monthly return, excess vs 1/12-weighted buy-and-hold of the same basket (the honest benchmark — TSM must beat
holding the basket, not cash), Sharpe, max DD, ex-top-5 % months, green-month share vs a count-matched null that holds
a random subset of the same size each month (1,000 draws), year-block bootstrap CI. Price-scale check per ETF (splits —
e.g. none expected; distributions ignored — report whether the panel is adjusted).
Pass on VAL (frozen): excess vs basket buy-and-hold > 0 with CI > 0 OR (max DD ≤ half the basket's DD AND mean return ≥
basket mean − 0.1 %/mo); AND ≥ 50 % green months; AND max DD ≤ 15 %. (TSM's published value is drawdown control more
than return; the OR clause encodes that — frozen before any number.)

## Ladder
Pass → independent rebuild → paper $10K from the next month-start → live $5K after 2 paper months realized > 0 → $10K
after 4. Kill: 8 cycles realized < 0, or one cycle < −3 %. Nothing to the owner before the rebuild. Cell count 1,711.
