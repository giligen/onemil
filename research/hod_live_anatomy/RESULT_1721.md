# RESULT 1,721 — loser anatomy of the live HOD paper book

Window 2026-09-29 .. 2026-10-30; closed fills 232 (PREREG said 241; the DB has 232 closed + 2 canceled CONI rows = 234 rows). R-valid fills 231 (rows with total_risk <= 0 excluded from R stats: ids [433]). R = pnl / (total_risk * filled_qty/shares) (actual risk per fill; 25 rows have filled_qty != shares and the DB pnl follows filled_qty; mean $34.1, not the nominal $50).

## Step 1 — decomposition (per fill; R = / total_risk, valid rows)

| group | n | gross@levels R | entry slip R (cost) | exit slip R (cost) | residual R | NET R/trade | gross $ | entry cost $ | exit cost $ | resid $ | net $ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 232 | +0.075 | +0.055 | +0.037 | +0.000 | -0.026 | +340 | +333 | +74 | +0 | -275 |
| stop exits | 124 | -0.936 | +0.063 | +0.091 | +0.000 | -1.099 | -3840 | +178 | +359 | +0 | -4458 |
| target exits | 55 | +2.052 | +0.050 | -0.051 | -0.000 | +2.053 | +3497 | +77 | -285 | -0 | +3525 |
| eod exits | 53 | +0.374 | +0.039 | +0.000 | +0.000 | +0.364 | +684 | +78 | +0 | +0 | +658 |

Identity: net = gross@levels - entry slip - exit slip + residual (residual = partial-exit row, fill fallback, fees; eod exits carry exit_level = actual exit so zero exit slip by construction). Entry slip = fill - pattern level; exit slip = intended exit level - actual exit (stop/target exits only).
All: gross@levels +0.075 R/trade, total cost +0.091 R/trade -> COST >= |GROSS|: the lever is EXECUTION first.
Entry slip: mean 8.1 bps, median 8.6, p90 14.8. Exit_slippage column (as stored, unit price): mean -0.075 (n 225).
Exit-method counts: {'stop_bid': 96, 'bracket_target': 55, 'eod_market': 53, 'bracket_stop_limit': 18, 'stop_bid_rest': 9, None: 1}

### EOD exits

n 53 (49 eod + 4 eod_ops_ah); P&L $+658; R/trade +0.364; positive 33 / negative 20. R quantiles (min, p10, p25, med, p75, p90, max): -1.12, -0.35, -0.10, +0.27, +0.77, +1.23, +2.45. MFE NOT computable: no 1-min bars for Sep-Oct 2026 in cache.db (0 rows for SPY/AAOI since 2026-09-01); only sign/distribution reported.

Book per-fill R: mean -0.026, sd 1.370, n 231, $-275 total; ex-top-5 % mean -0.151.

## Step 2 — the 12 cuts (cell = n / R per trade / $ ; kept vs dropped)

### Cut 1: fill time ET (keep <= median)
| half | kept n/R/$ | dropped n/R/$ | kept - dropped R |
|---|---|---|---|
| A | 53 / +0.089 / +71 | 62 / +0.143 / +335 | -0.054 |
| B | 63 / -0.202 / -473 | 53 / -0.129 / -268 | -0.073 |
| pooled | 116 / -0.069 / -402 | 115 / +0.017 / +67 | -0.087 |

kept 50% of fills (25.4/wk); kept ex-top-5 % -0.184; p one-sided: day-clustered 0.696 (Bonferroni x12: 1.000), iid Welch 0.685 (Bonf 1.000); mechanism listed: True. **Verdict: FAIL**

Bucket table (pooled): 09:30: 16 / +0.181 / +133; 09:40: 60 / -0.080 / -238; 09:50: 24 / -0.536 / -555; 10:00: 35 / +0.319 / +340; 10:10: 15 / -0.039 / -50; 10:20: 8 / -0.014 / +8; 10:30: 9 / -0.365 / -85; 10:40: 9 / +0.341 / +190; 10:50: 2 / -0.828 / -51; 11:00: 6 / +0.948 / +109; 11:10: 5 / -0.546 / +11; 11:20: 9 / -0.720 / -96; 11:30: 6 / -0.229 / +11; 11:40: 4 / -1.084 / -82; 11:50: 3 / +0.303 / +7; 12:00: 5 / +0.817 / +70; 12:20: 1 / +1.243 / +17; 12:30: 1 / +2.003 / +22; 12:40: 1 / +1.978 / +27; 12:50: 3 / +1.426 / +7; 13:10: 2 / +1.604 / +32; 13:30: 2 / -1.108 / -23; 13:40: 2 / -1.392 / -52; na: 3 / -1.114 / -86

### Cut 2: R as % of price (keep >= median)
| half | kept n/R/$ | dropped n/R/$ | kept - dropped R |
|---|---|---|---|
| A | 60 / +0.080 / +198 | 55 / +0.159 / +208 | -0.079 |
| B | 56 / -0.153 / -342 | 60 / -0.184 / -399 | +0.032 |
| pooled | 116 / -0.032 / -144 | 115 / -0.020 / -191 | -0.012 |

kept 50% of fills (25.4/wk); kept ex-top-5 % -0.146; p one-sided: day-clustered 0.525 (Bonferroni x12: 1.000), iid Welch 0.526 (Bonf 1.000); mechanism listed: True. **Verdict: FAIL**

Bucket table (pooled): (-2.608, 1.191]: 57 / -0.051 / -69; (1.191, 1.866]: 58 / +0.010 / -123; (1.866, 2.854]: 58 / -0.103 / -291; (2.854, 4.106]: 58 / +0.039 / +147

### Cut 3: entry spread bps
NOT COMPUTABLE: entry_quote_spread / entry quote columns are NULL for all rows (no entry NBBO stored).

### Cut 4: price (keep >= median)
| half | kept n/R/$ | dropped n/R/$ | kept - dropped R |
|---|---|---|---|
| A | 64 / +0.039 / -62 | 51 / +0.217 / +468 | -0.178 |
| B | 52 / -0.286 / -493 | 64 / -0.074 / -248 | -0.212 |
| pooled | 116 / -0.107 / -555 | 115 / +0.055 / +220 | -0.162 |

kept 50% of fills (25.4/wk); kept ex-top-5 % -0.239; p one-sided: day-clustered 0.846 (Bonferroni x12: 1.000), iid Welch 0.815 (Bonf 1.000); mechanism listed: False. **Verdict: FAIL (no mechanism listed; cannot pass)**

Bucket table (pooled): (10.0, 20.0]: 47 / -0.076 / -70; (20.0, 1000000000.0]: 184 / -0.013 / -265

### Cut 5: minutes first arm->fill (keep <= median)
| half | kept n/R/$ | dropped n/R/$ | kept - dropped R |
|---|---|---|---|
| A | 55 / +0.046 / +19 | 60 / +0.184 / +388 | -0.138 |
| B | 60 / -0.160 / -342 | 56 / -0.179 / -399 | +0.019 |
| pooled | 115 / -0.061 / -323 | 116 / +0.009 / -12 | -0.070 |

kept 50% of fills (25.2/wk); kept ex-top-5 % -0.177; p one-sided: day-clustered 0.651 (Bonferroni x12: 1.000), iid Welch 0.651 (Bonf 1.000); mechanism listed: False. **Verdict: FAIL (no mechanism listed; cannot pass)**

Bucket table (pooled): (-0.743, 3.062]: 58 / -0.057 / -152; (26.444, 57.808]: 56 / -0.065 / -171; (3.062, 26.444]: 57 / -0.066 / -171; (57.808, 246.474]: 57 / +0.140 / +246; nan: 3 / -1.114 / -86

### Cut 6: fill rank in day (keep <= median rank)
| half | kept n/R/$ | dropped n/R/$ | kept - dropped R |
|---|---|---|---|
| A | 69 / +0.039 / +85 | 46 / +0.237 / +322 | -0.199 |
| B | 66 / -0.206 / -486 | 50 / -0.121 / -255 | -0.085 |
| pooled | 135 / -0.081 / -401 | 96 / +0.051 / +66 | -0.131 |

kept 58% of fills (29.5/wk); kept ex-top-5 % -0.200; p one-sided: day-clustered 0.739 (Bonferroni x12: 1.000), iid Welch 0.757 (Bonf 1.000); mechanism listed: True. **Verdict: FAIL**

Bucket table (pooled): 1st: 23 / -0.118 / -58; 2nd: 23 / -0.105 / -87; 3rd+: 185 / -0.005 / -190

### Cut 7: fills that day (keep <= median)
| half | kept n/R/$ | dropped n/R/$ | kept - dropped R |
|---|---|---|---|
| A | 67 / +0.036 / +114 | 48 / +0.233 / +293 | -0.198 |
| B | 65 / -0.147 / -228 | 51 / -0.197 / -514 | +0.050 |
| pooled | 132 / -0.054 / -114 | 99 / +0.012 / -221 | -0.066 |

kept 57% of fills (28.9/wk); kept ex-top-5 % -0.183; p one-sided: day-clustered 0.614 (Bonferroni x12: 1.000), iid Welch 0.639 (Bonf 1.000); mechanism listed: True. **Verdict: FAIL**

Bucket table (pooled): 1-5: 9 / -0.012 / +67; 11+: 143 / -0.007 / -220; 6-10: 79 / -0.063 / -182

### Cut 8: SPY ret 09:30->fill
NOT COMPUTABLE: SPY/QQQ 1-min parquet ends 2026-05-29 and cache.db has no Sep-Oct intraday bars; the daily_bars close-to-close SPY return would include post-fill information (look-ahead), so not substituted.

### Cut 9: today's gap % (keep >= median)
| half | kept n/R/$ | dropped n/R/$ | kept - dropped R |
|---|---|---|---|
| A | 54 / -0.069 / -74 | 61 / +0.283 / +481 | -0.352 |
| B | 58 / -0.360 / -590 | 58 / +0.022 / -152 | -0.382 |
| pooled | 112 / -0.219 / -664 | 119 / +0.156 / +329 | -0.375 |

kept 48% of fills (24.5/wk); kept ex-top-5 % -0.347; p one-sided: day-clustered 0.957 (Bonferroni x12: 1.000), iid Welch 0.982 (Bonf 1.000); mechanism listed: True. **Verdict: FAIL**

Bucket table (pooled): (-1.909, 0.0]: 56 / +0.057 / +60; (-15.599, -1.909]: 55 / +0.274 / +358; (0.0, 1.697]: 53 / +0.283 / +523; (1.697, 11.598]: 55 / -0.639 / -1085; nan: 12 / -0.348 / -191

### Cut 10: R/(level-consol_low) (keep <= median)
| half | kept n/R/$ | dropped n/R/$ | kept - dropped R |
|---|---|---|---|
| A | 59 / +0.296 / +243 | 56 / -0.069 / +163 | +0.365 |
| B | 55 / -0.312 / -714 | 61 / -0.040 / -27 | -0.271 |
| pooled | 114 / +0.003 / -471 | 117 / -0.054 / +136 | +0.057 |

kept 49% of fills (24.9/wk); kept ex-top-5 % -0.132; p one-sided: day-clustered 0.367 (Bonferroni x12: 1.000), iid Welch 0.377 (Bonf 1.000); mechanism listed: True. **Verdict: FAIL**

Bucket table (pooled): (0.391, 1.023]: 57 / +0.012 / -256; (1.023, 1.048]: 57 / -0.007 / -215; (1.048, 1.086]: 56 / +0.267 / +478; (1.086, 1.429]: 57 / -0.327 / -302; nan: 4 / -0.668 / -40

### Cut 11: symbol repeat in ISO week (keep first fill)
| half | kept n/R/$ | dropped n/R/$ | kept - dropped R |
|---|---|---|---|
| A | 110 / +0.130 / +431 | 5 / -0.136 / -25 | +0.266 |
| B | 111 / -0.156 / -646 | 5 / -0.470 / -96 | +0.314 |
| pooled | 221 / -0.014 / -214 | 10 / -0.303 / -121 | +0.289 |

kept 96% of fills (48.3/wk); kept ex-top-5 % -0.144; p one-sided: day-clustered 0.197 (Bonferroni x12: 1.000), iid Welch 0.257 (Bonf 1.000); mechanism listed: True. **Verdict: FAIL**

Bucket table (pooled): first: 221 / -0.014 / -214; repeat: 10 / -0.303 / -121

### Cut 12: day of week (keep Tue-Thu)
| half | kept n/R/$ | dropped n/R/$ | kept - dropped R |
|---|---|---|---|
| A | 76 / +0.075 / +248 | 39 / +0.202 / +159 | -0.127 |
| B | 72 / -0.089 / -191 | 44 / -0.301 / -550 | +0.212 |
| pooled | 148 / -0.005 / +57 | 83 / -0.065 / -392 | +0.060 |

kept 64% of fills (32.4/wk); kept ex-top-5 % -0.136; p one-sided: day-clustered 0.402 (Bonferroni x12: 1.000), iid Welch 0.376 (Bonf 1.000); mechanism listed: False. **Verdict: FAIL (no mechanism listed; cannot pass)**

Bucket table (pooled): Fri: 47 / -0.287 / -622; Mon: 33 / +0.348 / +317; Thu: 54 / -0.001 / +209; Tue: 43 / -0.177 / -269; Wed: 51 / +0.137 / +116

## Verdict summary

| cut | verdict | A diff R | B diff R | Bonf p (cluster) |
|---|---|---|---|---|
| 1 fill time ET (keep <= median) | FAIL | -0.054 | -0.073 | 1.000 |
| 2 R as % of price (keep >= median) | FAIL | -0.079 | +0.032 | 1.000 |
| 3 entry spread bps | NOT COMPUTABLE | +nan | +nan | nan |
| 4 price (keep >= median) | FAIL (no mechanism listed; cannot pass) | -0.178 | -0.212 | 1.000 |
| 5 minutes first arm->fill (keep <= median) | FAIL (no mechanism listed; cannot pass) | -0.138 | +0.019 | 1.000 |
| 6 fill rank in day (keep <= median rank) | FAIL | -0.199 | -0.085 | 1.000 |
| 7 fills that day (keep <= median) | FAIL | -0.198 | +0.050 | 1.000 |
| 8 SPY ret 09:30->fill | NOT COMPUTABLE | +nan | +nan | nan |
| 9 today's gap % (keep >= median) | FAIL | -0.352 | -0.382 | 1.000 |
| 10 R/(level-consol_low) (keep <= median) | FAIL | +0.365 | -0.271 | 1.000 |
| 11 symbol repeat in ISO week (keep first fill) | FAIL | +0.266 | +0.314 | 1.000 |
| 12 day of week (keep Tue-Thu) | FAIL (no mechanism listed; cannot pass) | -0.127 | +0.212 | 1.000 |

## MDE (per-fill R sd 1.370 from this book)

50 % cut pooled (116 vs 116 of 231): one-sided alpha 0.05 power 0.8 -> 0.449 R/trade; Bonferroni alpha 0.05/12 -> 0.629 R. Per half (~57 vs 57): alpha 0.05 -> 0.638 R; the frozen +0.10 R edge is 0.16 of that per-half MDE, i.e. each half can only resolve effects of that size, and the +0.10 R bar in both halves is mostly noise-gated.

## Ship recommendation

No cut passes the frozen bar; nothing to ship from this cell. See the MDE paragraph for what 232 fills can and cannot detect.

## Caveats on this implementation

* Fill count 232 closed (not 241); 2 canceled rows ignored. R unit = total_risk rescaled by filled_qty/shares (not the nominal $50); the first run without that rescale left a +0.045 R residual from 25 rows where filled_qty != shares.
* Direction and threshold (pooled median) of every cut were fixed in code before computing outcomes; categorical tables are descriptive. Cut 7 uses the full-day fill count (not known at fill time -> not directly implementable; a running count would be needed). Cut 5 minutes are fill minus the day's first created_at (any row, including canceled). Cut 10 follows the PREREG formula literally (risk_per_share uses entry_price, which includes the expected slip). Cuts 4, 5, 12 had no mechanism written -> cannot pass.
* Cuts 3 and 8 not computable (no entry quotes; no Sep-Oct index bars). Cut 9 gap uses daily_bars open vs prior close; fills on dates/symbols missing from daily_bars drop out of the kept side (treated as not kept); coverage logged in the run.
* Gross@levels for eod exits equals the actual exit (no intended level). Residual column absorbs the partial-exit row and rows with fill_price missing (entry_price used).
* Day-clustered one-sided p from a sandwich OLS slope with 22-ish clusters is itself approximate; the iid Welch p is also shown.
* 232 fills across ~22 dates, one regime month; this is an in-sample anatomy of a single window.

## Cell 1,722 — execution levers (exploratory, post hoc)

Not pre-registered; descriptive, no pass bar. Buckets use entry slip = fill - pattern level in bps (n=231 R-valid).

### (a) outcome by entry-slip bucket
| half | slip bps | n | gross@levels R | net R | win % | target-hit % |
|---|---|---|---|---|---|---|
| pooled | <=5 | 55 | +0.119 | +0.131 | 42% | 31% |
| pooled | 5-10 | 72 | +0.035 | -0.051 | 40% | 21% |
| pooled | 10-15 | 78 | +0.145 | -0.002 | 38% | 23% |
| pooled | >15 | 22 | -0.149 | -0.308 | 27% | 18% |
| A | <=5 | 28 | +0.204 | +0.316 | 46% | 36% |
| A | 5-10 | 39 | +0.186 | +0.096 | 44% | 23% |
| A | 10-15 | 38 | +0.304 | +0.156 | 45% | 26% |
| A | >15 | 9 | -0.562 | -0.753 | 11% | 0% |
| B | <=5 | 27 | +0.031 | -0.062 | 37% | 26% |
| B | 5-10 | 33 | -0.144 | -0.224 | 36% | 18% |
| B | 10-15 | 40 | -0.006 | -0.151 | 32% | 20% |
| B | >15 | 13 | +0.136 | +0.000 | 38% | 31% |

slip quantiles p10/p50/p90: 1.5/8.6/14.8

### (b) exit slip by exit_reason (exit slip = intended level - actual exit; positive = cost; eod has no intended level)
| exit_reason | n | mean slip bps | mean slip R | exit_pricing_method |
|---|---|---|---|---|
| eod | 49 | +0.0 | +0.000 | {'eod_market': 49} |
| eod_ops_ah | 4 | +0.0 | +0.000 | {'eod_market': 4} |
| stop | 18 | +34.5 | +0.187 | {'bracket_stop_limit': 18} |
| stop_loss | 105 | +13.0 | +0.075 | {'stop_bid': 95, 'stop_bid_rest': 9, nan: 1} |
| stop_loss_unconfirmed | 1 | +2.7 | +0.029 | {'stop_bid': 1} |
| target | 55 | -28.2 | -0.051 | {'bracket_target': 55} |

By kind x method (sum R = contribution to the 0.037 R/trade total):
| kind | method | n | mean bps | mean R | sum R |
|---|---|---|---|---|---|
| stop | bracket_stop_limit | 18 | +34.5 | +0.187 | +3.4 |
| stop | stop_bid | 96 | +11.8 | +0.069 | +6.6 |
| stop | stop_bid_rest | 9 | +25.7 | +0.135 | +1.2 |
| stop | nan | 1 | +6.3 | +0.080 | +0.1 |
| target | bracket_target | 55 | -28.2 | -0.051 | -2.7 |

### (c) EOD exits (n=53)
R at exit p10/p50/p90: -0.35/+0.27/+1.23; above +1 R: 10 of 53 (mean R of those +1.48, $+449); mean minutes held 457 (median 367, n 53).

### (d) entry quote recording (hod_break rows, all dates)
| account | n | bid | ask | spr | fbid | fask | xbid |
|---|---|---|---|---|---|---|---|
| None | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| live | 7 | 0 | 0 | 0 | 0 | 0 | 3 |
| paper | 234 | 0 | 0 | 0 | 0 | 0 | 157 |
ORB rows for comparison (n / non-null bid / ask): 219 / 146 / 146
