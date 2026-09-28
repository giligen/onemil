# REBUILD_1633.md -- independent rebuild of PREREG_1633.md (cells 1,633-1,635)

Built from the PREREG prose only. cell_1633.py, cell_1633_events_pead.csv, RESULT_1633.md and cell_1552.py were never opened while writing rebuild_1633.py -- this is the independent-reimplementation leg of the CLAUDE.md pre-ship check, not a comparison against the original (a trade-by-trade Jaccard/bps diff against cell_1633_events_pead.csv must be done separately by whoever can see both files).

## Headline
Cell 1,633 (long top decile, 10-session hold) FAILS the frozen VAL pass bar on every criterion
but frequency: VAL mean net = +9.0 bps/event (bar: >=+50), day-clustered t = 0.19 (bar: >=2.5),
ex-top-5% = -135.0 bps (bar: >0, i.e. the already-small mean is entirely tail-carried), TRAIN
is same-sign but also far below its own t>=1 bar (t=0.38), and the decile table is not
monotone on either split (deciles 6-9 do not rank above deciles 2-3 in forward return on
TRAIN or VAL). Cell 1,634 (20-session hold) is directionally better (VAL t=1.10) but still
short of any of the frozen bars. Cell 1,635 (short) is negative on VAL once shortable-filtered
(-14.2 bps, t=-0.30). On this independent rebuild, R0-decile-conditioned PEAD on the 8-K/2.02
population does not clear the pre-registered bar in TRAIN or VAL -- consistent with the 1,552
desk's finding that the unconditioned 2.02 class shows nothing, extended here to show the
decile-conditioning does not rescue it either.

## Independent-check evidence
- **Event-count cross-check against the PREREG prose** (the only independent check possible
  without opening the original files, per this task's restrictions): the PREREG text says
  "form 8-K with item 2.02 = 116 k earnings releases". This rebuild's own item/form parsing
  (streamed from events_raw.csv, item token '2.02' exactly matched on the ';'-split `items`
  field, form=='8-K' exact) found 128,645 raw matches, and 116,492 after dedup (same-symbol
  same-ET-date collapse + exact-duplicate-row drop) -- a 0.4% difference from "116k". This is
  strong evidence the item/form parsing logic in this independent rebuild agrees with
  whatever produced the PREREG's own population-size figure, even though no trade-by-trade
  Jaccard against cell_1633_events_pead.csv was possible under this task's rules.
- 1,069 8-K/A amendments (and 9 ABS-15G, 1 8-K12B) carrying item 2.02 were excluded by the
  exact form=='8-K' match -- a direct, mechanical answer to the PREREG's own "duplicate 8-Ks
  (amendments)" refuter.

## Event population
- Eligible (universe-filtered) events, TRAIN+VAL: 55889 (TRAIN 39532, VAL 16357). TEST (reaction_session >= 2024-07-01) was dropped immediately after split assignment and never scored, per SEALED.
- TRAIN R0 decile cutoffs (10th..90th pct): [-0.0837, -0.0461, -0.025, -0.0104, 0.0013, 0.0131, 0.0274, 0.0479, 0.0849]

## TRAIN decile table (R0 and forward gross return by TRAIN-fit decile)
|   decile |      n |   mean_R0_bps |   mean_gross10_bps |   mean_gross20_bps |
|---------:|-------:|--------------:|-------------------:|-------------------:|
|      1.0 | 3954.0 |       -1520.7 |              -54.0 |              -67.7 |
|      2.0 | 3953.0 |        -624.2 |               17.2 |               79.1 |
|      3.0 | 3952.0 |        -348.1 |               14.9 |               50.1 |
|      4.0 | 3954.0 |        -174.3 |               -1.2 |               40.4 |
|      5.0 | 3953.0 |         -43.4 |               -2.3 |               48.2 |
|      6.0 | 3953.0 |          70.8 |               34.8 |               65.2 |
|      7.0 | 3953.0 |         200.3 |               44.6 |              106.7 |
|      8.0 | 3953.0 |         368.5 |               47.3 |               88.9 |
|      9.0 | 3953.0 |         638.5 |                9.6 |               89.0 |
|     10.0 | 3954.0 |        1528.4 |               24.9 |               99.2 |

## VAL decile table (same TRAIN cutoffs applied out-of-sample)
|   decile |      n |   mean_R0_bps |   mean_gross10_bps |   mean_gross20_bps |
|---------:|-------:|--------------:|-------------------:|-------------------:|
|      1.0 | 2045.0 |       -1584.2 |              -24.3 |              -47.2 |
|      2.0 | 1673.0 |        -628.7 |              -44.2 |              -31.8 |
|      3.0 | 1635.0 |        -347.9 |              -20.6 |               30.8 |
|      4.0 | 1505.0 |        -172.6 |                6.2 |               27.7 |
|      5.0 | 1455.0 |         -45.2 |              -22.5 |              -27.9 |
|      6.0 | 1424.0 |          69.1 |              -41.8 |              -32.8 |
|      7.0 | 1417.0 |         199.5 |              -20.3 |               16.3 |
|      8.0 | 1525.0 |         370.5 |                9.9 |               27.8 |
|      9.0 | 1653.0 |         640.8 |                7.2 |               70.5 |
|     10.0 | 2025.0 |        1608.9 |               19.0 |               83.9 |

## Cells 1,633 / 1,634 / 1,635 -- per split
| split   | cell                                  |    n |   events_per_week |   mean_net_bps |   day_clustered_t |   ex_top5pct_bps |   winner_capped_30pct_bps |   median_bps |   pct_positive |
|:--------|:--------------------------------------|-----:|------------------:|---------------:|------------------:|-----------------:|--------------------------:|-------------:|---------------:|
| TRAIN   | 1633_long_top_h10                     | 3954 |             18.96 |          14.93 |              0.38 |          -160.83 |                    -16.66 |        -3.83 |           0.50 |
| TRAIN   | 1634_long_top_h20                     | 3954 |             18.96 |          89.18 |              1.30 |          -174.65 |                    -15.76 |        15.89 |           0.50 |
| TRAIN   | 1635_short_bot_h10_FILTERED_shortable | 3172 |             15.21 |          20.41 |              0.47 |          -126.26 |                      6.96 |        53.18 |           0.53 |
| TRAIN   | 1635_short_bot_h10_UNFILTERED         | 3954 |             18.96 |          33.48 |              0.76 |          -123.88 |                     15.65 |        60.04 |           0.53 |
| VAL     | 1633_long_top_h10                     | 2025 |             25.96 |           9.03 |              0.19 |          -135.03 |                     -9.48 |        -2.60 |           0.50 |
| VAL     | 1634_long_top_h20                     | 2025 |             25.96 |          73.93 |              1.10 |          -137.60 |                     14.27 |        56.83 |           0.51 |
| VAL     | 1635_short_bot_h10_FILTERED_shortable | 1710 |             21.92 |         -14.18 |             -0.30 |          -140.50 |                    -20.81 |        14.19 |           0.51 |
| VAL     | 1635_short_bot_h10_UNFILTERED         | 2045 |             26.22 |           3.74 |              0.08 |          -128.46 |                     -3.92 |        26.45 |           0.52 |

## Cell 1,633 per-year (top decile, 10-session hold, net bps)
| split   |   year |   count |   mean |
|:--------|-------:|--------:|-------:|
| TRAIN   |   2019 |     783 |  -73.9 |
| TRAIN   |   2020 |     956 |  130.5 |
| TRAIN   |   2021 |     958 |  -62.8 |
| TRAIN   |   2022 |    1257 |   41.7 |
| VAL     |   2023 |    1288 |  -26.1 |
| VAL     |   2024 |     737 |   70.5 |

## Pass-bar check (frozen, VAL, cell 1,633)
- Mean net >= +50 bps/event: VAL = 9.0 bps -> FAIL
- day-clustered t >= 2.5: VAL t = 0.19 -> FAIL
- ex-top-5% > 0: VAL = -135.0 bps -> FAIL
- >= 5 events/week in season: VAL = 25.96/wk -> PASS
- TRAIN same sign, t >= 1: TRAIN mean = 14.9 bps, t = 0.38 -> FAIL
- decile table monotone on VAL (10-session gross): False

## Caveats (read as an adversary, per CLAUDE.md)
- **borrow_flags.csv is not point-in-time.** It is a single current-day snapshot (no date column) joined onto TRAIN/VAL events by symbol only; a name's shortability in 2019-2023 may differ from today's snapshot. Cell 1,635 is reported both FILTERED (shortable==True in the snapshot) and UNFILTERED (all bottom-decile events), per the task's own instruction for when the borrow data is imperfect/absent.
- **SSR is not represented at all** in borrow_flags.csv (columns: symbol, tradable, shortable, easy_to_borrow, exchange). No SSR exclusion is applied anywhere -- this is a real gap against the PREREG's 'shortable/SSR excluded', not a filter that was silently skipped.
- **Delistings inside the hold are kept at -100%** for the long legs (1,633/1,634) when a symbol has core prices (prior/reaction/entry) but no exit price; the short leg (1,635) EXCLUDES those events instead of assuming a clean +100% cover, since a delisting's actual short P&L depends on the bankruptcy/wind-down process and is not reliably inferable from daily bars.
- **Known 1-day data hole: 2024-06-28.** alpaca_daily_2019_2024H1.parquet ends 2024-06-27; panel_2024_2026.parquet starts 2024-07-01. 2024-06-28 was a real Friday trading session missing from both files, so it is simply absent from the SPY calendar built here; any event landing exactly on it is invisible to this rebuild (logged, not silently large -- see the run log).
- **No market-cap split.** Cell 1,636's small-cap (<=$1B) vs larger cut needs shares outstanding, which is not in any of the datasets handed to this task; out of scope here and not attempted (no market-cap number is reported).
- **Obtainability:** both legs are MOO/MOC auction orders at prices the market actually printed, not a level touch, so this satisfies the CLAUDE.md obtainability check by construction.
- **Reaction-session mapping for non-trading-day (weekend/holiday) filings** is not stated in the PREREG prose; this rebuild rolls them forward to the next real session under the same rule as a pre-market filing (see the module docstring in rebuild_1633.py). This is a genuine judgment call an independent implementation could reasonably make differently.
- Dedup collapsed same-symbol/same-ET-date 8-K/2.02 filings to the earliest acceptance (handles amendments/multi-part same-day filings); exact form=='8-K' matching (vs '8-K/A') independently excludes amendment forms.

Runtime: 70.2s.