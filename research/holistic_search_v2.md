# Phase 4b — stability-aware joint grid search

Trades: TRAIN=536, VAL=496, HOQ1=386, HOAPR=53

Baseline (current ship config @ conv>=1.4, current tiers):

  TRAIN: n=356 WR=42.7% PnL=$+31,905 DD=$-15,545
  VAL: n=276 WR=36.6% PnL=$+14,710 DD=$-14,365
  HOQ1: n=262 WR=40.8% PnL=$+7,589 DD=$-12,919
  HOAPR: n=49 WR=61.2% PnL=$+15,035 DD=$-2,053

Constraints: TRAIN gain ≥ +3%, VAL gain ≥ +3%, |TRAIN%-VAL%| ≤ 20pt, trade count 0.7x-1.5x of baseline.

Phase S1: scanning 9216 weight configs @ th=1.4 tiers=current...

## Stage 1 — Top 30 weight configs (fixed th=1.4, tiers=current)

| # | T+V gain | TRAIN% | VAL% | |Δ| | TRn | VLn | HOn | HOLDOUT | weights | tier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | $+63,992 | +32.7% | +47.2% | 14.5% | 424 | 342 | 357 | $+25,753 | w_r1=0.4 w_r2p=0.4 w_r2n=0.0 w_r3=0.45 w_r5=0.1 w_r7=0.3 w_r9=0.7 | t=current |
| 2 | $+63,797 | +40.5% | +29.0% | 11.5% | 436 | 366 | 371 | $+24,319 | w_r1=0.4 w_r2p=0.4 w_r2n=-0.15 w_r3=0.45 w_r5=0.3 w_r7=0.3 w_r9=0.7 | t=current |
| 3 | $+63,479 | +38.5% | +31.1% | 7.4% | 432 | 356 | 363 | $+26,510 | w_r1=0.3 w_r2p=0.4 w_r2n=0.0 w_r3=0.45 w_r5=0.2 w_r7=0.3 w_r9=0.7 | t=current |
| 4 | $+63,470 | +34.5% | +39.8% | 5.3% | 421 | 341 | 354 | $+25,941 | w_r1=0.4 w_r2p=0.3 w_r2n=0.0 w_r3=0.45 w_r5=0.1 w_r7=0.3 w_r9=0.7 | t=current |
| 5 | $+63,091 | +39.7% | +25.9% | 13.8% | 423 | 352 | 360 | $+23,316 | w_r1=0.4 w_r2p=0.4 w_r2n=-0.15 w_r3=0.45 w_r5=0.3 w_r7=0.2 w_r9=0.7 | t=current |
| 6 | $+62,945 | +37.9% | +28.8% | 9.1% | 427 | 351 | 361 | $+20,694 | w_r1=0.4 w_r2p=0.4 w_r2n=-0.15 w_r3=0.45 w_r5=0.2 w_r7=0.3 w_r9=0.7 | t=current |
| 7 | $+62,854 | +31.4% | +42.2% | 10.7% | 424 | 342 | 357 | $+25,330 | w_r1=0.4 w_r2p=0.4 w_r2n=0.0 w_r3=0.45 w_r5=0.1 w_r7=0.3 w_r9=0.6 | t=current |
| 8 | $+62,737 | +34.9% | +33.9% | 1.0% | 400 | 321 | 341 | $+20,434 | w_r1=0.3 w_r2p=0.4 w_r2n=-0.3 w_r3=0.45 w_r5=0.2 w_r7=0.3 w_r9=0.7 | t=current |
| 9 | $+62,669 | +31.2% | +41.5% | 10.4% | 424 | 342 | 357 | $+25,629 | w_r1=0.4 w_r2p=0.4 w_r2n=0.0 w_r3=0.45 w_r5=0.2 w_r7=0.2 w_r9=0.7 | t=current |
| 10 | $+62,577 | +38.7% | +24.6% | 14.1% | 405 | 334 | 348 | $+18,181 | w_r1=0.4 w_r2p=0.4 w_r2n=-0.3 w_r3=0.45 w_r5=0.3 w_r7=0.2 w_r9=0.7 | t=current |
Phase S2: on 10 weight configs × 7 thresholds × 4 caps × 6 macd-pairs × 8 tiers = 13440 configs

## Stage 2 — Top 30 full configs (weights × th × cap × macd × tier)

| # | T+V gain | TRAIN% | VAL% | |Δ| | TRn | VLn | HOn | HOLDOUT | config | tier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | $+88,044 | +89.4% | +87.6% | 1.8% | 461 | 389 | 391 | $+36,526 | w_r1=0.4 w_r2p=0.4 w_r2n=-0.15 w_r3=0.45 w_r5=0.3 w_r7=0.2 w_r9=0.7 th=1.2 cap=4.0 macd_norm=1.0 macd_strong=2.0 | t=+T3_small_2.0 |
| 2 | $+86,423 | +83.1% | +90.4% | 7.4% | 461 | 389 | 391 | $+35,263 | w_r1=0.4 w_r2p=0.4 w_r2n=-0.15 w_r3=0.45 w_r5=0.3 w_r7=0.2 w_r9=0.7 th=1.2 cap=4.0 macd_norm=0.75 macd_strong=1.8 | t=+T3_small_2.0 |
| 3 | $+86,295 | +80.8% | +94.5% | 13.8% | 457 | 384 | 386 | $+38,891 | w_r1=0.4 w_r2p=0.4 w_r2n=-0.15 w_r3=0.45 w_r5=0.3 w_r7=0.3 w_r9=0.7 th=1.3 cap=4.0 macd_norm=1.0 macd_strong=2.0 | t=+T3_small_2.0 |
| 4 | $+86,006 | +83.4% | +86.9% | 3.6% | 454 | 371 | 381 | $+29,360 | w_r1=0.4 w_r2p=0.4 w_r2n=-0.3 w_r3=0.45 w_r5=0.3 w_r7=0.2 w_r9=0.7 th=1.2 cap=4.0 macd_norm=1.0 macd_strong=2.0 | t=rescue_T1b |
| 5 | $+86,006 | +83.4% | +86.9% | 3.6% | 454 | 371 | 381 | $+29,360 | w_r1=0.4 w_r2p=0.4 w_r2n=-0.3 w_r3=0.45 w_r5=0.3 w_r7=0.2 w_r9=0.7 th=1.2 cap=4.0 macd_norm=1.0 macd_strong=2.0 | t=filter_large_mid |
| 6 | $+85,419 | +83.3% | +83.1% | 0.2% | 461 | 389 | 391 | $+32,137 | w_r1=0.4 w_r2p=0.4 w_r2n=-0.15 w_r3=0.45 w_r5=0.3 w_r7=0.2 w_r9=0.7 th=1.2 cap=4.0 macd_norm=1.0 macd_strong=1.8 | t=+T3_small_2.0 |
| 7 | $+84,755 | +78.7% | +88.6% | 9.9% | 461 | 389 | 391 | $+34,948 | w_r1=0.4 w_r2p=0.4 w_r2n=-0.15 w_r3=0.45 w_r5=0.3 w_r7=0.2 w_r9=0.7 th=1.2 cap=3.5 macd_norm=1.0 macd_strong=2.0 | t=current |
| 8 | $+84,748 | +86.0% | +72.7% | 13.3% | 461 | 389 | 391 | $+34,069 | w_r1=0.4 w_r2p=0.4 w_r2n=-0.15 w_r3=0.45 w_r5=0.3 w_r7=0.2 w_r9=0.7 th=1.2 cap=4.0 macd_norm=0.75 macd_strong=1.8 | t=rescue_T1b |
| 9 | $+84,748 | +86.0% | +72.7% | 13.3% | 461 | 389 | 391 | $+34,069 | w_r1=0.4 w_r2p=0.4 w_r2n=-0.15 w_r3=0.45 w_r5=0.3 w_r7=0.2 w_r9=0.7 th=1.2 cap=4.0 macd_norm=0.75 macd_strong=1.8 | t=filter_large_mid |
| 10 | $+84,744 | +77.4% | +91.3% | 13.9% | 452 | 380 | 382 | $+33,635 | w_r1=0.4 w_r2p=0.4 w_r2n=-0.15 w_r3=0.45 w_r5=0.2 w_r7=0.3 w_r9=0.7 th=1.3 cap=4.0 macd_norm=1.0 macd_strong=2.0 | t=rescue_T1b |
| 11 | $+84,744 | +77.4% | +91.3% | 13.9% | 452 | 380 | 382 | $+33,635 | w_r1=0.4 w_r2p=0.4 w_r2n=-0.15 w_r3=0.45 w_r5=0.2 w_r7=0.3 w_r9=0.7 th=1.3 cap=4.0 macd_norm=1.0 macd_strong=2.0 | t=filter_large_mid |
| 12 | $+84,478 | +79.4% | +85.1% | 5.6% | 461 | 389 | 391 | $+34,483 | w_r1=0.4 w_r2p=0.4 w_r2n=-0.15 w_r3=0.45 w_r5=0.3 w_r7=0.2 w_r9=0.7 th=1.2 cap=3.0 macd_norm=1.0 macd_strong=2.0 | t=current |
| 13 | $+84,396 | +83.4% | +75.8% | 7.6% | 457 | 384 | 386 | $+35,201 | w_r1=0.4 w_r2p=0.4 w_r2n=-0.15 w_r3=0.45 w_r5=0.3 w_r7=0.3 w_r9=0.7 th=1.3 cap=4.0 macd_norm=1.0 macd_strong=2.0 | t=rescue_T1b |
| 14 | $+84,396 | +83.4% | +75.8% | 7.6% | 457 | 384 | 386 | $+35,201 | w_r1=0.4 w_r2p=0.4 w_r2n=-0.15 w_r3=0.45 w_r5=0.3 w_r7=0.3 w_r9=0.7 th=1.3 cap=4.0 macd_norm=1.0 macd_strong=2.0 | t=filter_large_mid |
| 15 | $+84,331 | +78.7% | +85.7% | 7.0% | 461 | 389 | 391 | $+35,530 | w_r1=0.4 w_r2p=0.4 w_r2n=-0.15 w_r3=0.45 w_r5=0.3 w_r7=0.2 w_r9=0.7 th=1.2 cap=4.0 macd_norm=1.0 macd_strong=2.0 | t=current |

## Stage 3 — One-shot HOLDOUT validation for top-5 stable configs

Same top 5 from Stage 2, reporting HOLDOUT Q1 + April deltas vs baseline:

| # | config | TRAIN Δ | VAL Δ | HOQ1 Δ | HOAPR Δ | HOLDOUT Δ | Grand Δ | % of base |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | r1=0.4 r2p=0.4 r2n=-0.15 r3=0.45 r5=0.3 r7=0.2 r9=0.7 th=1.2 cap=4.0 macd_norm=1.0 macd_strong=2.0 t=+T3_small_2.0 | $+28,535 | $+12,893 | $+7,610 | $+6,292 | $+13,902 | $+55,331 | +79.9% |
| 2 | r1=0.4 r2p=0.4 r2n=-0.15 r3=0.45 r5=0.3 r7=0.2 r9=0.7 th=1.2 cap=4.0 macd_norm=0.75 macd_strong=1.8 t=+T3_small_2.0 | $+26,502 | $+13,305 | $+8,717 | $+3,923 | $+12,640 | $+52,447 | +75.7% |
| 3 | r1=0.4 r2p=0.4 r2n=-0.15 r3=0.45 r5=0.3 r7=0.3 r9=0.7 th=1.3 cap=4.0 macd_norm=1.0 macd_strong=2.0 t=+T3_small_2.0 | $+25,774 | $+13,906 | $+9,423 | $+6,845 | $+16,268 | $+55,947 | +80.8% |
| 4 | r1=0.4 r2p=0.4 r2n=-0.3 r3=0.45 r5=0.3 r7=0.2 r9=0.7 th=1.2 cap=4.0 macd_norm=1.0 macd_strong=2.0 t=rescue_T1b | $+26,602 | $+12,788 | $+1,930 | $+4,807 | $+6,736 | $+46,127 | +66.6% |
| 5 | r1=0.4 r2p=0.4 r2n=-0.3 r3=0.45 r5=0.3 r7=0.2 r9=0.7 th=1.2 cap=4.0 macd_norm=1.0 macd_strong=2.0 t=filter_large_mid | $+26,602 | $+12,788 | $+1,930 | $+4,807 | $+6,736 | $+46,127 | +66.6% |

## Recommended ship config (rank 1 of stability-scored search)

```
Rule weights:
  w_r1 (pole_gain):      0.4  (baseline 0.3)
  w_r2+ (flag_tight):    0.4  (baseline 0.3)
  w_r2- (flag_loose):    -0.15  (baseline -0.3)
  w_r3 (vol_ratio):      0.45  (baseline 0.3)
  w_r5 (retracement):    0.3  (baseline 0.2)
  w_r7 (vwap_dist):      0.2  (baseline 0.2)
  w_r9 (v_reversal):     0.7  (baseline 0.4)
min_threshold:           1.2  (baseline 1.4)
cap:                     4.0  (baseline 3.0)
macd_normal:             1.0  (baseline 1.0)
macd_strong:             2.0  (baseline 1.5)
tier variant:            +T3_small_2.0
```

- TRAIN: 461 trades / $+60,440 (+89.4%)
- VAL:   389 trades / $+27,603 (+87.6%)
- HOQ1:  339 trades / $+15,200
- HOAPR: 52 trades / $+21,326
- **Grand total PnL: $+124,570 vs baseline $+69,239 (+$+55,331, +79.9%)**
