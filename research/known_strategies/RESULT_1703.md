# RESULT 1,703a-e: documented-strategy sweep (PREREG_1703_sweep.md, frozen; one run, nothing tuned)
Script `1703_sweep.py`, log `1703.log`, cells `1703_cells.csv`, crypto bars `1703e_crypto.parquet`. 2017-02..2026-09, $50K, daily equity marked at the open.
GREF = `1700u_curves_daily.csv` column `guard` (daily equity of the guarded sleeve, 2017-02-06..2026-09-28; recomputed 29.3 % / -38.3 %, vs the quoted 28.7 / -38.1).
Stack reads use WEEKLY (W-FRI) equity on the common weeks; GREF on that basis = 29.1 % / -34.9 % / ratio 0.83 (the literal 0.75+0.10 = 0.85 bar is used).
Universe: U2 (price >= $10, ADV20 >= $200M, 273-day history) plus the 1700t hygiene guard (names with a +200 % / -75 % day or a >10-day gap in the last 273 sessions excluded) for a, c, d.
Costs: a, d = the sleeve's spread model (0.05 % + half a range-based spread, cap 0.2 %); c = flat 10.5 bp; b = 2 bp (my assumption); e = 10 bp per traded dollar.

| Cell | CAGR | Max DD | CAGR/DD | Sharpe | Worst yr | Yrs > SPY | Turn/yr | Cost/yr | H1 / H2 CAGR |
|---|---|---|---|---|---|---|---|---|---|
| a 52w-high | -4.8 % | -58.7 % | -0.08 | -0.21 | -28.8 % | 0/10 | 67x | 8.8 % | +0.1 % / -9.5 % |
| b dual mom | 6.0 % | -38.7 % | 0.16 | 0.44 | -25.0 % | 1/10 | 4.7x | 0.1 % | 6.2 % / 5.8 % |
| c ST reversal | 2.6 % | -66.5 % | 0.04 | 0.27 | -30.0 % | 3/10 | 92x | 9.6 % | 2.2 % / 3.0 % |
| d low-vol | 7.3 % | -29.1 % | 0.25 | 0.60 | -1.4 % | 2/10 | 6.7x | 0.9 % | 9.0 % / 5.5 % |
| e crypto TSM (2018-01..) | 38.8 % | -59.5 % | 0.65 | 0.95 | -42.6 % | 4/9 | 38x | 3.8 % | 98.5 % / 2.7 % |
SPY over the same span: 15.1 % / -32.0 %. e buy-and-hold 50/50 BTC/ETH: 22.7 % / -87.9 %.

| Cell | corr GREF (weekly) | corr in GREF's 3 deepest episodes | 50/50 CAGR / DD / ratio | GREF 100 % + 50 % comp: CAGR / DD / ratio |
|---|---|---|---|---|
| a | 0.53 | 0.52 | 12.0 % / -30.8 % / 0.39 | 24.6 % / -43.8 % / 0.56 |
| b | 0.57 | 0.66 | 18.1 % / -31.6 % / 0.57 | 31.3 % / -44.8 % / 0.70 |
| c | 0.66 | 0.73 | 16.1 % / -46.6 % / 0.34 | 27.5 % / -58.1 % / 0.47 |
| d | 0.35 | 0.54 | 19.2 % / -28.1 % / 0.68 | 33.0 % / -42.3 % / 0.78 |
| e | 0.18 | 0.49 | 37.8 % / -35.7 % / 1.06 | 53.2 % / -44.9 % / 1.19 |
Episodes (peak to trough): 2021-02-16..05-11 (-38.3 %), 2020-02-14..03-19 (-37.1 %), 2025-02-13..04-07 (-33.6 %); 27 weeks inside.

## Pass rule, line by line (all four must hold)
| Cell | CAGR >= 10 % and DD better than -30 % | corr <= 0.5 | 50/50 ratio >= 0.85 | both halves positive | PASS |
|---|---|---|---|---|---|
| a | NO (-4.8 %, -58.7 %) | NO (0.53) | NO (0.39) | NO (H2 -9.5 %) | NO |
| b | NO (6.0 %, -38.7 %) | NO (0.57) | NO (0.57) | yes | NO |
| c | NO (2.6 %, -66.5 %) | NO (0.66) | NO (0.34) | yes | NO |
| d | NO (7.3 % < 10 %; DD -29.1 % ok) | yes (0.35) | NO (0.68) | yes | NO |
| e | NO (38.8 % ok, DD -59.5 %) | yes (0.18) | yes (1.06) | yes (sign only) | NO |
No cell beats GREF on both CAGR and DD (no replacement candidate). Zero of five pass; 1,703 adds five cells to the multiplicity count.

## Verdicts
* a: closed. Negative, cost drag 8.8 %/yr on 67x turnover; the published edge is a monthly, all-cap, pre-2004 result.
* b: closed. Spends most of the time in SPY at lower return (SPY 15.1 %); holdings SPY 71 / EFA 29 / IEF 16 months.
* c: closed. Gross edge cannot carry 92x turnover at 10.5 bp on the worst-performing liquid names; DD -66 %.
* d: closed as a component (7.3 %, half the bar) but the best diversifier by correlation among equities; 50/50 improves DD (-28 %) not ratio.
* e: fails ONLY the stand-alone DD bar (-59.5 % vs -30 %), the only cell with a high 50/50 ratio. Not recommended; see caveats. A cadence/vol-scaled variant would be a NEW cell with its own PREREG, not a rescue of this one.

## Causality and data
* Signal date = the session before the rebalance (asserted for every rebalance). Shift test on 2023-06-05 (signal 2023-06-02): top name of a / c / d recomputed from panel rows strictly before the date: RCL 0.9957, AAP -0.394689, PEP 0.010305 = stored values. b: closes before the rebalance. e: position over day t uses close[t-1]/close[t-21]; the same-day (look-ahead) version gives 272 % CAGR, shown only as the contrast.
* ETFs SPY/EFA/SHY/IEF all present in the panel (2016-01..2026-09, 2,700 days), adjusted. Crypto: Alpaca serves BTC/USD and ETH/USD only from 2021-01-01 (2,100 days each). 2017-06..2020-12 (BTC) and 2017-11..2020-12 (ETH) come from yfinance (free fallback, flagged in the parquet `source` column); overlap check Jan 2021 median Alpaca/yfinance close ratio 0.9997 (BTC), 1.0001 (ETH). LOST days: BTC 0, ETH 161 calendar days (forward-filled, position unchanged).

## Caveats (adversary)
* e is two regimes in one number: 98.5 % CAGR in 2018-2021 (the 2020-21 run), 2.7 % in 2022-2026; the "halves agree in sign" rule passes on a 2.7 % half. About 5.7 of 9 years are Alpaca-served; the first 3 are a different data vendor. Only two assets, one cost assumption (10 bp; real crypto cost is spread plus fee and Alpaca crypto is not the same venue as the index). Equal-weight daily re-mix drift trades ignored.
* a, c, d inherit the 1700 panel's biases: adjusted prices (survivorship handled by the point-in-time panel, price-scale not re-audited here), ADV >= $200M leaves 1,115 names across the programme; the hygiene guard is applied to all three (the PREREG says U2 only) because unadjusted-action artefacts otherwise fabricate rank-1 names; this is a deviation I chose and state, not tested the other way.
* a ranks close / 252-day max of the daily high (ties at 1.0 broken alphabetically); c's prior-week return is the 5-trading-day close ratio, not calendar-week; d uses the 12-month return close/close[-252]. One definition each, none tuned.
* b's 2 bp ETF cost is my assumption; the dual-momentum paper uses a global equity index (not EFA) and monthly T-bill, here SHY total return. b is regime-sparse: ~116 monthly decisions, ~2 switches per year.
* Stack numbers are weekly-equity based, the 1.5x overlay ignores financing cost and margin; correlations are on ~500 weeks (SE ~0.04), episode correlations on 27 weeks (SE ~0.2, not significant).
* Tail dependence (ex-top-1/5 % ) NOT run for these cells; closure rests on the stand-alone bars, so it is a conservative negative for a-d. The e verdict would need it before any reconsideration.
