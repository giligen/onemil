# RESULT 1,703f — leveraged index trend (PREREG_1703_sweep.md Amendment 2). Script 1703f_levtrend.py, log 1703f_run.log
Spec as frozen: QQQ (SPY for UPRO) close > 200-day SMA through day d-1 -> hold TQQQ (UPRO) from open d, else cash; open-to-open
returns; 5 bp per switch. Shift test 2022-03-15: close 318.32 < SMA 356.28 -> signal False, applied only at next open. +1 cell.
Data: panel 2016+, yfinance 2010-2015 spliced by level at the first panel date; daily-return corr vs panel on the overlap:
TQQQ .99998, QQQ .99999, UPRO .99302 (.99986 excl. 2020-03: panel-vs-yf close differences of up to 13 pp on 2020-03-12..19),
SPY .99610 (.99965 excl. 2020-03; open corr .9985, SPY/QQQ opens unused). GREF = RECON_1700tu_weeks.csv column net_A (weekly,
Monday-open basis, 2017-02-06..): CAGR 29.2 % / DD -36.9 % / ratio 0.79 (PREREG quoted 28.7/-38.1/0.75; bar below uses 0.85).

| Row | Window | CAGR | Max DD | End $ (from $50K) | Sharpe | Switches/yr |
|---|---|---|---|---|---|---|
| TQQQ-trend | 2017-26 | 40.3 % | -58.1 % | 1.34M | 0.92 | 4.1 |
| TQQQ-trend | 2011-26 | 28.9 % | -60.3 % | 2.69M | 0.77 | 5.5 |
| UPRO-trend | 2017-26 | 22.4 % | -53.3 % | 356K | 0.73 | 5.1 |
| UPRO-trend | 2011-26 | 21.2 % | -53.3 % | 1.02M | 0.72 | 5.0 |
| TQQQ hold | 2017-26 | 42.2 % | -81.6 % | 1.52M | 0.87 | 0 |
| TQQQ hold | 2011-26 | 40.5 % | -81.6 % | 10.4M | 0.87 | 0 |
| UPRO hold | 2017-26 | 28.4 % | -74.8 % | 567K | 0.74 | 0 |
| UPRO hold | 2011-26 | 29.0 % | -74.8 % | 2.74M | 0.77 | 0 |

## Stacking with GREF (2017-02..2026-09, weekly)
| Component | corr (whole) | corr in GREF's 3 deepest episodes (2021-24, 2020, 2025) | 50/50 CAGR / DD / ratio | GREF 100 % + 50 % comp (CAGR / DD / ratio) |
|---|---|---|---|---|
| TQQQ-trend | 0.59 | 0.52 / 0.59 / 0.51 | 43.4 % / -39.3 % / 1.10 | 59.5 % / -51.0 % / 1.17 |
| UPRO-trend | 0.52 | 0.54 / 0.16 / 0.49 | 34.2 % / -34.9 % / 0.98 | 50.8 % / -47.8 % / 1.06 |
| TQQQ hold | 0.66 | 0.50 / 0.89 / 0.88 | 42.5 % / -57.6 % / 0.74 | 56.5 % / -61.0 % / 0.93 |
| UPRO hold | 0.64 | 0.55 / 0.82 / 0.89 | 35.2 % / -54.4 % / 0.65 | 49.7 % / -64.1 % / 0.78 |
(stacking read adds no financing cost on the 1.5x leverage; reported, not recommended.)

## Pass rule (stand-alone CAGR >= 10 % AND DD better than -30 % AND corr <= 0.5 AND 50/50 ratio >= 0.85 AND halves agree)
TQQQ-trend: CAGR 40.3 % PASS; max DD -58.1 % FAIL; corr 0.59 FAIL; 50/50 ratio 1.10 PASS; halves +772 % / +207 % agree PASS -> FAIL.
UPRO-trend: CAGR 22.4 % PASS; DD -53.3 % FAIL; corr 0.515 FAIL (just); ratio 0.98 PASS; halves agree -> FAIL.
Holds fail on DD (-82 %/-75 %) and ratio. Beats GREF on both CAGR and DD? No (DD worse in every row). Not a replacement.

## TQQQ-trend by year (full history 2010-26; hold for comparison)
2010 -10.4 (hold +94.3, partial yr) | 2011 -33.2 (-5.4) | 2012 +19.9 (+54.6) | 2013 +120.5 (+120.6) | 2014 +61.6 (+61.6) | 2015 -25.5 (+7.8)
2016 -5.6 (+21.7) | 2017 +117.1 (+117.1) | 2018 -17.9 (-25.8) | 2019 +41.7 (+155.3) | 2020 +89.3 (+107.5) | 2021 +82.3 (+82.3)
2022 -44.9 (-78.8) | 2023 +98.9 (+182.9) | 2024 +65.5 (+65.5) | 2025 +24.7 (+35.6) | 2026 +36.0 (+45.1). Switches/yr: 15,18,9,1,0,6,12,0,9,7,4,0,9,5,0,4,2.

## Adversary caveats
- Whipsaw years: 2010-11 (33 switches, -33 % in 2011), 2015-16 (18 switches, -25 %/-6 %), 2018 (9), 2022 (9 switches, still -45 %): the
  200-day exit is late in a 3x ETF, so the in-year DD is -46..-58 % even in good years (2020 -58 %, 2024 -42 %).
- The 2017-26 CAGR is carried by 2017, 2020, 2021, 2023, 2024 (+65..+117 %); 2017-26 sits in the best leveraged-Nasdaq decade; 2011-26 CAGR 28.9 % with the 2011 whipsaw.
- Switch cost 5 bp is an assumption; TQQQ spreads in 2011/2020 were wider; no tax, no financing. One fixed spec, no tuning (SMA200 only), so selection bias is low but the form itself was chosen because it is popular after the fact.
- Open-to-open fills assume the open is obtainable (liquid ETFs, MOO fine). Panel/yfinance close mismatches in 2020-03 (UPRO/SPY) are
  data-source differences, not used to splice (2016+ is panel only).
- Stack numbers use 2017-02+ weeks only; the stacking row's leverage of 1.5x has no borrowing cost.
