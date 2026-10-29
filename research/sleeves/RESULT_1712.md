# RESULT cell 1,712 -- sleeve #3 low-volatility (PREREG_LOWVOL_20261029), built by Sonnet agent, NOT yet independently rebuilt

Rule: `research/sleeves/lowvol.py` (`nice -n 19 python3 research/sleeves/lowvol.py`, 44 s); ledger `monthly_1712.csv`; log `cache/run_1712.log`.
Stage 1 is rejection-only: a pass here is not a pass. (Result: no pass.)
Data: local `panel_2016_2026.parquet`, scan `bar_date <= 2022-12-31` (TEST never read). Universe at each signal: top-1,000 by 60d median $vol
(of ~5,190 name-screened symbols; always exactly 1,000 eligible), 30 lowest 252d-vol, equal weight, quarterly (Jan/Apr/Jul/Oct entry open),
buy-and-hold drift in-quarter, 15 bps RT on traded turnover. Q4-2022 cohort (exit open 2023-01-03) DROPPED -> TRAIN 2017-01..2019-12 n 36 (12 cohorts),
VAL 2020-01..2022-09 n 33 (11 cohorts). CAPM on raw monthly returns (no rf in panel).

## Headline (monthly, net 15 bps; excess vs top-1,000 EW universe, year-block CI)
| split | mean | univ | SPY | excess | CI | alpha/mo (ann) | alpha CI | beta | Sharpe | maxDD strat/univ/SPY | green strat/univ/SPY | ex-top5% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN | +1.02 % | +1.08 | +1.24 | -0.06 | [-0.81,+0.82] | +0.43 (+5.2 %) | [-0.71,+0.84] | 0.47 | 1.36 | 6.4 / 18.6 / 15.3 % | 69 / 72 / 78 % | +0.65 % |
| VAL | +0.19 % | +0.66 | +0.67 | -0.47 | [-2.38,+1.54] | -0.24 (-2.9 %) | [-1.30,+0.29] | 0.64 | 0.13 | 24.3 / 31.3 / 23.3 % | 52 / 61 / 61 % | -0.32 % |
Count-matched random-30 null (1,000 draws, 15 bps/qtr): green TRAIN 69.8 % (p95 77.8) vs actual 69.4 % (p(null>=actual) 0.66);
VAL 57.1 % (p95 63.6) vs 51.5 % (p 0.93). Null mean +1.04 / +0.63 %/mo vs strat +1.02 / +0.19: the 30 low-vol names do not beat a random 30.
Per-year mean monthly % (strat/univ/SPY): 2017 1.41/1.59/1.63, 2018 -0.06/-0.88/-0.47, 2019 1.70/2.51/2.55, 2020 0.01/2.39/1.74, 2021 1.48/1.54/2.16, 2022 -1.30/-2.85/-2.74.
Defensive in 2018 and 2022 (as published), 2020 -2.4 %/mo behind (AGNC, NLY -41/-49 % in 2020Q1), lagging every up year.

## Frozen VAL pass bar, line by line
* alpha > 0 with CI > 0: FAIL (-0.24 %/mo, CI [-1.30,+0.29]; iid t -0.48).
* maxDD <= 0.75 x SPY DD: FAIL (24.3 % vs 17.5 % limit; SPY 23.3 %; strat DD is HIGHER than SPY's).
* green >= 50 %: PASS (52 %, but below universe 61 %, SPY 61 % and the null 57.1 %).
* mean >= universe mean - 0.2 %/mo: FAIL (+0.19 vs +0.46 floor).
**VERDICT: FAIL (1 of 4). Sleeve #3 REJECTED at Stage 1; no PIT purchase, no rebuild, no paper.** TRAIN also has alpha CI spanning 0.

## Sensitivities (mean / excess vs univ / alpha / maxDD / green) TRAIN | VAL; all FAIL the VAL bar
* RT 30 bps: +1.00 / -0.08 / +0.41 / 6.4 % / 69 % | +0.17 / -0.48 / -0.25 / 24.3 % / 52 %.
* 60-session vol: +0.64 / -0.44 / -0.01 / 10.0 % / 75 % | +0.38 / -0.28 / -0.03 / 20.1 % / 55 %.
* monthly rebalance: +0.93 / -0.14 / +0.33 / 7.2 % / 69 % | +0.26 / -0.40 / -0.17 / 23.3 % / 52 %.
* top 50: +1.04 / -0.04 / +0.44 / 7.3 % / 75 % | +0.30 / -0.36 / -0.19 / 26.1 % / 52 %.
* universe as written (asset-map only, funds leak): +0.90 / -0.18 / +0.37 | +0.17 / -0.49 / -0.21 (picks were MINT, GLD, IAU, VTEB).
* split-artefact screen (drops NVS 2019-04): TRAIN +1.09 / +0.02 / +0.49; VAL unchanged. Does not change the verdict.
Turnover: mean one-way 0.32 (TRAIN) / 0.33 (VAL) per quarterly rebalance (universe EW 0.13); cost ~2.4 bp/mo -- cost is irrelevant to the verdict.
Picks: 101 (TRAIN) / 99 (VAL) distinct names, median price $54 / $71, median vol 0.88 / 1.16 %/day; staples, utilities, telcos, waste (RSG, WM, KO, DUK, VZ, JNJ).

## Overlap (monthly net corr, n; sealed months cut)
SPY +0.73 / +0.83 (TRAIN/VAL); TOM -0.01 / +0.02; MOM +0.57 / +0.53; TSM (monthly_1711) +0.64 / +0.78. Not a diversifier to MOM or TSM; ~SPY at beta 0.5-0.6.

## Adversarial caveats
* Deviation: asset-map labels bond / gold / T-bill funds 'stock', so a low-vol sort picked MINT/GLD/IAU/VTEB; I added a name regex (`FUND_RE`) for the headline. 461 symbols
  have no map name (unscreened). The regex also removes some legitimate names containing 'Index'/'Treasury' etc.; both versions fail.
* Panel is survivorship-exposed and dividend-adjusted; unadjusted split artefact found (NVS 2019-04: 406 -> 68, -82 % day, picked at 1/30 weight, -71 %); low-vol longs rarely delist,
  the universe benchmark is biased up by dead-name omission less than a random book, so survivorship is not the explanation for the failure.
* Power: 11 VAL cohorts; CIs are +-1.5 %/mo wide (MDE far above a published 3-5 %/yr alpha). A Stage 1 FAIL is rejection of THIS rule on THIS window (2020 COVID crash + 2022), not proof of no anomaly.
* Universe is a fixed 1,000 each quarter; entry tradability (finite open) is conditioned at the entry, as in 52WH (tiny look-ahead). No rf subtracted in CAPM. 252-session history uses >= 245 valid closes.
* Cells run on this programme's low-vol line: 1 (plus 6 sensitivities, same cell). TEST 2023+ untouched.
