# RESULT 1,649-1,651: index overnight premium + turn-of-month

Data: SPY/QQQ/IWM minute+daily, Alpaca adjustment=all SIP. 20 early-close sessions excluded (data-driven: >=2/3 symbols, 13:00-16:00 ET vol share < 0.5x own median). Cost 1.0 bp/round-trip (flat). TEST 2024-01..2026-09 SEALED: 0 events computed past that date.

## Cell 1,649: unconditional overnight (SCORED per Amendment 1, 2016-2023 pooled)
- **1649 overnight MOC->MOO TRAIN**: n=3018 (1006 clusters), mean +3.69 bps, clustered t=2.29, MDE=4.18 bps, ex-top5% -2.24, ex-worst5% +10.76, worst 2016-06-23 (-375.2)
- **1649 overnight MOC->MOO VAL**: n=3015 (1005 clusters), mean +2.92 bps, clustered t=0.92, MDE=8.32 bps, ex-top5% -9.43, ex-worst5% +16.59, worst 2020-03-13 (-984.5)
- **1649 overnight MOC->MOO POOLED**: n=6033 (2011 clusters), mean +3.30 bps, clustered t=1.85, MDE=4.65 bps, ex-top5% -6.23, ex-worst5% +14.17, worst 2020-03-13 (-984.5)
  - per-ETF: IWM:+4.81(2011), QQQ:+2.98(2011), SPY:+2.12(2011); ann/dd: SPY:+4.74%/dd-32.86%, QQQ:+6.79%/dd-28.54%, IWM:+11.75%/dd-31.44%
- **1649 intraday MOO->MOC TRAIN**: n=3018 (1006 clusters), mean +0.49 bps, clustered t=0.20, MDE=6.43 bps, ex-top5% -8.96, ex-worst5% +11.37, worst 2018-02-08 (-377.0)
- **1649 intraday MOO->MOC VAL**: n=3015 (1005 clusters), mean +0.97 bps, clustered t=0.27, MDE=9.60 bps, ex-top5% -12.83, ex-worst5% +15.88, worst 2020-03-20 (-540.0)
- **1649 intraday MOO->MOC POOLED**: n=6033 (2011 clusters), mean +0.73 bps, clustered t=0.34, MDE=5.77 bps, ex-top5% -11.30, ex-worst5% +14.07, worst 2020-03-20 (-540.0)
  - per-ETF: IWM:-2.21(2011), QQQ:+2.92(2011), SPY:+1.47(2011); ann/dd: SPY:+2.86%/dd-26.59%, QQQ:+5.96%/dd-25.67%, IWM:-6.92%/dd-58.64%
  - overnight Sharpe vs 24h buy-and-hold Sharpe (same ETF, 2016-2023): SPY 0.45 vs 0.78, QQQ 0.55 vs 0.88, IWM 0.85 vs 0.50
  - worst SINGLE event: SPY 2020-03-13 -1075.2 bps = $-2,150 at $20,000/night notional
  - **Pass bar 1,649 (FAIL)**:
    - [ ] MDE=4.65>2bps -> alt decides (mean/t informational): mean=+3.30 t=1.85 | years_pos=0.75 etfs_pos=True | overnight-vs-BH Sharpe SPY:0.45vs0.78, QQQ:0.55vs0.88, IWM:0.85vs0.50
  - **Stacking**: 62.9 events/month pooled -> $415/month expected at $20,000/night-or-event (gross of any correlation between simultaneous same-night fills). Capital window: overnight only (~17h flat position, flat intraday every session). Shared-tail: worst night 2020-03-13 (SPY, -1075bps) is a broad market gap event (e.g. COVID crash week) -- correlated with, not diversifying from, the account's day-trading books on the same date

## Cell 1,650: conditional overnight (heavy close -> buy MOC, sell MOO)
- **1650 heavy TRAIN**: n=576 (400 clusters), mean +2.97 bps, clustered t=1.63, MDE=6.77 bps, ex-top5% -2.97, ex-worst5% +10.35, worst 2016-06-23 (-375.2)
- **1650 heavy VAL**: n=464 (324 clusters), mean +8.82 bps, clustered t=2.57, MDE=11.12 bps, ex-top5% -2.01, ex-worst5% +18.76, worst 2022-02-23 (-281.1)
  - TRAIN excess over unconditional (complement nights): heavy +3.96 (n=400) - complement +4.39 (n=905) = -0.44 bps, t=-0.15
  - VAL excess over unconditional (complement nights): heavy +11.62 (n=324) - complement +2.80 (n=967) = +8.82 bps, t=1.57
  - TRAIN s_t tercile: T1_low: +3.33bps (n=998) | T2_mid: +3.56bps (n=998) | T3_high: +3.71bps (n=998)
  - VAL s_t tercile: T1_low: +4.74bps (n=999) | T2_mid: -1.11bps (n=999) | T3_high: +5.00bps (n=999)
  - VAL per-ETF: IWM:+8.73(160), QQQ:+8.52(166), SPY:+9.28(138)
  - VAL per-year: 2020:+16.1(119), 2021:+14.7(131), 2022:-8.4(114), 2023:+12.0(100)
  - VAL ann/dd (fully deployed, per ETF): SPY:+3.19%/dd-6.18%, QQQ:+3.46%/dd-10.34%, IWM:+3.47%/dd-7.81%
  - **Pass bar 1,650 (FAIL)**:
    - [ ] MDE=11.12>8bps -> alt test decides (mean/t informational): mean=+8.82 t=2.57 (informational=True) | excess=+8.82 excess_t=1.57 mono_tr=True mono_val=False
    - [ ] ex-top-5% > 0 (VAL): -2.0080136290816895
    - [x] TRAIN same sign, |t| >= 1: 1.6299687007449764
    - [x] positive in all 3 ETFs (VAL): {'IWM': np.float64(8.72754796027927), 'QQQ': np.float64(8.515960655698413), 'SPY': np.float64(9.281121648157125)}
  - **Stacking**: 9.7 events/month pooled -> $170/month expected at $20,000/night-or-event (gross of any correlation between simultaneous same-night fills). Capital window: overnight only, heavy-close nights only (a subset of all nights). Shared-tail: worst heavy night 2022-02-23 (-281bps) -- heavy-close nights are not a random subset of all nights (they cluster around index-rebalance and high-volatility sessions), so this sleeve's tail is not obviously diversifying from 1649's or the account's other books

## Cell 1,651: turn-of-month (2016-2023 pooled)
- **1651 pooled**: n=285 (95 clusters), mean +32.08 bps, clustered t=1.41, MDE=60.22 bps, ex-top5% +4.35, ex-worst5% +61.22, worst 2018-01 (-614.3)
  - per-ETF: IWM:+29.41(95), QQQ:+31.50(95), SPY:+35.34(95)
  - per-year: 2016:-1.3(36), 2017:+19.7(36), 2018:-22.2(36), 2019:+33.2(36), 2020:+150.9(36), 2021:+35.4(36), 2022:+5.8(36), 2023:+35.6(33)
  - ann/dd (fully deployed, per ETF): SPY:+4.14%/dd-7.76%, QQQ:+3.51%/dd-10.09%, IWM:+3.20%/dd-11.64%
  - overlap with 1650 heavy-close nights: 25.8% of 855 TOM trigger-nights (entry + intermediate sessions) are also 1650 heavy-close nights
  - **Pass bar 1,651 (PASS)**:
    - [x] MDE=60.22>30bps -> alt test decides (mean/t informational): mean=+32.08 t=1.41 (informational=True) | years_pos=0.75 etfs_pos=True
    - [x] ex-top-5% > 0: 4.353287069600375
  - **Stacking**: 3.0 events/month pooled -> $192/month expected at $20,000/night-or-event (gross of any correlation between simultaneous same-night fills). Capital window: ~4.7 calendar days tied up, once a month per ETF (entry MOC last day of month through exit MOC 3rd session of the next). Shared-tail: worst event 2018-01 (-614bps); 26% of its trigger-nights are also 1650 heavy-close nights -- partial overlap, not independent of 1650's tail

## Verdict
- 1649 (unconditional overnight, stacking sleeve): FAIL
- 1650 (conditional overnight): FAIL
- 1651 (turn-of-month): PASS

## Caveats (read as an adversary before relaying)
- Cost is a flat 1.0 bp/round-trip deduction (0.5bp/auction leg x 2 legs); the PREREG's "0.5bp per auction leg plus 0.5bp half-spread" is read as one number via its own check "(1bp per round trip is generous)", not decomposed into 2bp by adding both terms per leg.
- Early-close dates are a DATA-DRIVEN volume-ratio fallback (pandas_market_calendars not installed), reused unchanged from research/lev_flow/cell_1643.py, which verified this method recovers the known NYSE calendar exactly on the same data source; no separately hard-coded date list was kept, so an undetected non-standard early close cannot be cross-checked here.
- MDE uses SD(pooled events)/sqrt(N_CLUSTERS)*2.5; reproduces the PREREG's own worked numbers (~11bps 1650 VAL, ~55bps 1651) only with clusters = distinct nights/months, not distinct symbol-events -- verify the printed MDE against those before trusting the bar.
- 16:00:00-labeled minute bars (if present) are included in both the s_t numerator and denominator (closing-auction print), consistent with cell_1643's RTH convention.
- "Heavy" is NaN (not False) for the first ~60 valid sessions per symbol (no trailing-median baseline yet) and is excluded from the heavy/complement split and from the pass-bar per-ETF checks, but s_t itself still feeds the tercile table for those rows.
- TOM overlap uses entry + intermediate trigger-sessions (excludes the final exit session, which starts no new overnight leg) checked against 1650's heavy flag pooled TRAIN+VAL; an unclassifiable trigger date (warmup/early-close) defaults to "not heavy" for this count only.
- Annualised return/drawdown compounds each ETF's own event stream sequentially at 100% notional (no cross-ETF netting or margin sharing); a real sleeve running all 3 signals at once would need $20K x however many fire the same night, not $20K flat.
- Month boundaries are the trading-session calendar built from the daily-bar dates themselves (no external calendar); this already excludes holidays by construction.
- Amendment 1 interpretation calls, stated explicitly since the amendment leaves them open: "$20K per night/event" is read as $20K per (symbol, night) -- NOT $60K if all 3 ETFs fire the same night -- so stacking $/month sums 3 independent $20K sleeves, one per ETF; "worst night in dollars" uses the single worst (symbol, date) row, not the cross-ETF date-clustered mean (which is smaller); events/month for 1650 uses the VAL rate (scored split) and for 1649/1651 the full 2016-2023 pooled rate; the Sharpe comparison requires ALL 3 ETFs to beat their own buy-and-hold Sharpe (mirrors "all three ETFs positive" in the same sentence), not a pooled/average Sharpe.
