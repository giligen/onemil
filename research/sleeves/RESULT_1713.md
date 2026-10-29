# RESULT 1,713 — Sleeve #4 weekly short-term reversal (PREREG_STREV_20261029), Stage 1 on the Alpaca panel

Code `research/sleeves/strev.py` (reuses lowvol/fiftytwo_wh loader, symbol + fund-name screens, split_metrics, block_ci); weekly rows
`weekly_1713.csv`; log `cache/run_1713.log`. Rule as frozen: top-500 by 60d median $vol, price >= $10, 20 lowest prior-5-session returns
at the Friday close, enter first session of next week at the open, exit the following week's open, 20 bps RT charged on 100 % turnover
(40 bps sensitivity), long-only. TRAIN 2017-01..2019-12 (157 wk / 36 mo), VAL 2020-01..2022-12 (155 wk / 36 mo). TEST 2023+ never read
(parquet filter stops 2022-12-31; the week whose exit open falls in 2023 is dropped). Single cell, no tuning, 1,713 on the programme count.

**Stage 1 is rejection-only: a pass here is not a pass.**

## Headline (monthly = compounded weeks; excess vs equal-weight universe, which pays 20 bps on its ~3 % weekly drift turnover)
| split | gross/mo | net/mo | univ/mo | net excess [yr-block 95 % CI] | Sharpe | max DD (univ, SPY) | green wk (null mean / p95) |
|---|---|---|---|---|---|---|---|
| TRAIN | +1.93 % | +1.05 % | +1.04 % | +0.01 % [-0.80, +1.23] | 0.71 | 11.7 % (15.2, 13.0) | 52.9 % (54.2 / 58.6; p(null>=act) .73) |
| VAL | +0.91 % | +0.04 % | +0.75 % | -0.71 % [-3.04, +2.44] | 0.01 | 60.2 % (27.1, 23.3) | 46.5 % (51.3 / 54.8; p(null>=act) .99) |

Gross vs net (weekly): TRAIN +0.452 % gross -> +0.252 % net; VAL +0.251 % -> +0.051 %; cost 0.200 %/wk (about 0.87 %/mo). Gross excess vs
universe VAL +0.13 %/mo (CI -2.22..+3.31), i.e. the reversal premium is gone before cost on VAL; random-20 null net is -0.005 %/wk.
Turnover: 0.91 / 0.89 of the book replaced per week (losers do not persist), so the full-turnover charge is nearly exact; universe EW 0.03.

## Pass bar on VAL, line by line
- net excess vs universe > 0: **FAIL** (-0.71 %/mo; also -0.62 at actual-turnover cost, -1.56 at 40 bps)
- excess CI > 0: **FAIL** (lo -3.04 %)
- max DD <= 20 %: **FAIL** (60.2 %; Mar-2020 and 2021-22 growth-loser drawdowns, universe 27 %)
- >= 52 % green weeks: **FAIL** (46.5 %); above the count-matched random-20 null: **FAIL** (below its mean 51.3 %, p(null>=act) 0.99)
- **Verdict: REJECT** (net excess <= 0). Same verdict at 40 bps and at actual-turnover cost. TRAIN alone would read UNDETERMINED
  (excess +0.01, CI spans 0, green below null) — not a pass, and TRAIN is not the gate.

## Sensitivities (VAL net excess vs universe %/mo; all negative)
top-50 losers -0.24 (DD 51.6); price >= $20 -0.71; skip-last-session (Thursday signal) -0.79; causal jump screen (>50 % one-day move in the signal
window) -0.71; as-written asset-map screen without the fund-name screen -0.96. TRAIN signs: top-50 -0.59, price20 -0.38, skipLast -0.05.

## Tail, scale and overlap
- VAL weekly net ex-top-1 % -0.317 %/wk, ex-top-5 % -0.994 %/wk; best +28.5 %, worst -28.2 %. The mean is carried by 2020 (+4.89 %/mo)
  against 2021 -1.65 and 2022 -3.11 %/mo (universe +1.39, -1.59): a one-year, tail-carried path, not a stable premium.
- Price-scale: 0 of 6,240 name-weeks beyond +500 % / -80 % (max +158 %, min -56 %); jump-screen run is identical to the base within 0.01 %.
- Monday overlap with MOM: MOM's ledger (`1700c_monthly.csv`) is monthly and carries no pick names, so name-level position overlap is NOT
  computable; 72 of 312 STREV entry weeks share an ISO week with a MOM entry date. Monthly net-return correlation with MOM: TRAIN 0.71,
  VAL 0.81, ALL 0.78 (n 71) — same-sign beta, little diversification.

## Adversarial caveats
- Alpaca panel is survivorship-exposed (present-day asset map as the exclusion list; delisted losers that never traded again are the
  worst case for a loser-buying sleeve, so survivorship FLATTERS this rule — the reject is robust to it, a TRAIN-style "undetermined" is not).
- 2,341 name-weeks had no exit open and used the last close before it (WARNING logged); 461 symbols lack an asset-map name so the fund screen
  cannot apply to them. Holiday weeks use the first/last session of the ISO week, not literally Monday/Friday.
- "Above the null" for green weeks is read as green > null p95 (also reported vs null mean); both fail on VAL, so the reading is immaterial.
- 20 bps RT for a 500-name, >= $10 universe is a placeholder, not measured NBBO; gross is shown separately. Equal-weight universe benchmark
  uses actual drift turnover while the sleeve pays the PREREG's full-turnover charge (turnover is 0.9 anyway, so the asymmetry is small).
- MDE ~1.4 %/mo: the VAL CI is far wider than any plausible effect, so the reject rests on the SIGN of the point estimate plus DD/green, per the PREREG.
- 36 months, three years in a COVID-dominated window: the result says nothing about quieter regimes; TEST stays sealed and is not opened.
