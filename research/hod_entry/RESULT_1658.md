# RESULT — cell 1,658: HOD-break net R by stop-distance bucket, measured cost (13 bps/RT)

PREREG: `PREREG_1658.md` (frozen 2026-09-29 15:40 UTC). Source: cell 1,438's 9,911-fill population
(`causal_arming_causal.csv`, variant=causal, status=fill; TRAIN-H2=27 wk /
VAL=22 wk). No re-selection — every fill keeps its outcome; cost re-expressed as
(7 bps entry + 6 bps exit) x price / R$, EOD exits pay the same 13 bps. TEST split absent from this file
(TEST NOT read, per REPORT_1438.md). n = 9911 (TRAIN-H2 4,398 + VAL 5,513, matches REPORT_1438.md).

## What `cap` and `min_r_pct` do to the stop (trading/hod_break_engine.py)
- `cap = 0.006` (line 1763) bounds the ENTRY, not the stop: `limit = level * (1 + cap)` — a 60 bps chase
  ceiling on the breakout level. An ask above that limit is refused ("no chase"). It never touches the stop.
- `min_r_pct = 1.0` (lines 1792-1793) is a ONE-TIME admission floor: at signal time, `r = entry_est - stop`
  (`entry_est` = the ask observed at that instant, `stop` = the consolidation low); the candidate is skipped
  if `r / entry_est * 100 < min_r_pct`. It is never re-checked against the actual fill. Under
  `entry_mode: resting_stop_limit` (live since 9/25) the order rests and can fill later, at a different
  price than `entry_est`, while `stop` stays fixed at the consolidation low — so the REALIZED R% at fill
  can land below the nominal 1.0% floor (a fill closer to the fixed stop than the ask was at admission) or
  above it. That is consistent with today's live spread (0.66%-4%; TTAN 0.66% was the paper account, see
  `spec_no_trade` at line 1952, which flags fills where the realized `r_fill/px` would have failed
  `min_r_pct` after the fact). A separate fallback (line 1346, boot-only) synthesizes a stop at exactly
  `entry_price * (1 - min_r_pct/100)` when no real stop is known — a data artifact, not a trading rule.

## Bucket table — R% = (entry - stop) / entry, by split and pooled
| split | bucket | n | share | fills/wk | win% | gross R | net R (13bps) | t (day-clust, ndays) | ex-top5% net R | $ @ $50 risk | MDE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN-H2 | <0.75% | 1 | 0.0% | 0.0 | 0.0% | -1.000 | -1.475 | n/a (1) | n/a | $-73.73 | n/a |
| TRAIN-H2 | 0.75-1.5% | 2048 | 46.6% | 75.9 | 36.0% | -0.001 | -0.109 | -3.96 (128) | -0.216 | $-5.44 | 0.060 |
| TRAIN-H2 | 1.5-3% | 1930 | 43.9% | 71.5 | 38.8% | 0.056 | -0.009 | -0.83 (127) | -0.113 | $-0.46 | 0.062 |
| TRAIN-H2 | >=3% | 419 | 9.5% | 15.5 | 36.0% | -0.038 | -0.076 | -1.56 (108) | -0.183 | $-3.78 | 0.123 |
| VAL | <0.75% | 1 | 0.0% | 0.0 | 100.0% | 2.000 | 1.801 | n/a (1) | n/a | $90.03 | n/a |
| VAL | 0.75-1.5% | 2355 | 42.7% | 107.0 | 36.6% | -0.007 | -0.115 | -4.43 (102) | -0.222 | $-5.76 | 0.056 |
| VAL | 1.5-3% | 2573 | 46.7% | 117.0 | 37.2% | -0.003 | -0.068 | -2.75 (102) | -0.174 | $-3.38 | 0.053 |
| VAL | >=3% | 584 | 10.6% | 26.5 | 45.0% | 0.176 | 0.138 | 0.50 (101) | 0.039 | $6.89 | 0.108 |
| POOLED | <0.75% | 2 | 0.0% | 0.0 | 50.0% | 0.500 | 0.163 | 0.10 (2) | -1.475 | $8.15 | 3.275 |
| POOLED | 0.75-1.5% | 4403 | 44.4% | 91.7 | 36.3% | -0.005 | -0.112 | -5.75 (230) | -0.219 | $-5.61 | 0.041 |
| POOLED | 1.5-3% | 4503 | 45.4% | 93.8 | 37.9% | 0.023 | -0.043 | -2.23 (229) | -0.148 | $-2.13 | 0.040 |
| POOLED | >=3% | 1003 | 10.1% | 20.9 | 41.3% | 0.086 | 0.049 | -0.73 (209) | -0.054 | $2.43 | 0.081 |

## Same table by TARGET distance (2 x R%, since target_r = 2.0 is constant across this population)
This is a mechanical rescale of the table above (target = entry + 2R, so target-distance bucket membership
is identical to the R% bucket above, just doubled labels) — not an independent cut.
| split | bucket | n | share | fills/wk | win% | gross R | net R (13bps) | t (day-clust, ndays) | ex-top5% net R | $ @ $50 risk | MDE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN-H2 | <1.5% | 1 | 0.0% | 0.0 | 0.0% | -1.000 | -1.475 | n/a (1) | n/a | $-73.73 | n/a |
| TRAIN-H2 | 1.5-3% | 2048 | 46.6% | 75.9 | 36.0% | -0.001 | -0.109 | -3.96 (128) | -0.216 | $-5.44 | 0.060 |
| TRAIN-H2 | 3-6% | 1930 | 43.9% | 71.5 | 38.8% | 0.056 | -0.009 | -0.83 (127) | -0.113 | $-0.46 | 0.062 |
| TRAIN-H2 | >=6% | 419 | 9.5% | 15.5 | 36.0% | -0.038 | -0.076 | -1.56 (108) | -0.183 | $-3.78 | 0.123 |
| VAL | <1.5% | 1 | 0.0% | 0.0 | 100.0% | 2.000 | 1.801 | n/a (1) | n/a | $90.03 | n/a |
| VAL | 1.5-3% | 2355 | 42.7% | 107.0 | 36.6% | -0.007 | -0.115 | -4.43 (102) | -0.222 | $-5.76 | 0.056 |
| VAL | 3-6% | 2573 | 46.7% | 117.0 | 37.2% | -0.003 | -0.068 | -2.75 (102) | -0.174 | $-3.38 | 0.053 |
| VAL | >=6% | 584 | 10.6% | 26.5 | 45.0% | 0.176 | 0.138 | 0.50 (101) | 0.039 | $6.89 | 0.108 |
| POOLED | <1.5% | 2 | 0.0% | 0.0 | 50.0% | 0.500 | 0.163 | 0.10 (2) | -1.475 | $8.15 | 3.275 |
| POOLED | 1.5-3% | 4403 | 44.4% | 91.7 | 36.3% | -0.005 | -0.112 | -5.75 (230) | -0.219 | $-5.61 | 0.041 |
| POOLED | 3-6% | 4503 | 45.4% | 93.8 | 37.9% | 0.023 | -0.043 | -2.23 (229) | -0.148 | $-2.13 | 0.040 |
| POOLED | >=6% | 1003 | 10.1% | 20.9 | 41.3% | 0.086 | 0.049 | -0.73 (209) | -0.054 | $2.43 | 0.081 |

## Both-halves agreement (frozen reading rule: net >= +0.05 R, t >= 2, ex-top-5% > 0, >= 3 fills/wk, on BOTH halves)
- <0.75%: TRAIN-H2 net -1.475 (t n/a, 0.0/wk) | VAL net 1.801 (t n/a, 0.0/wk) -> does not qualify
- 0.75-1.5%: TRAIN-H2 net -0.109 (t -3.96, 75.9/wk) | VAL net -0.115 (t -4.43, 107.0/wk) -> does not qualify
- 1.5-3%: TRAIN-H2 net -0.009 (t -0.83, 71.5/wk) | VAL net -0.068 (t -2.75, 117.0/wk) -> does not qualify
- >=3%: TRAIN-H2 net -0.076 (t -1.56, 15.5/wk) | VAL net 0.138 (t 0.50, 26.5/wk) -> does not qualify

**Verdict: no bucket qualifies on both halves.** Every bucket's net R is negative on both TRAIN-H2 and VAL
under the measured 13-bps cost; the >=3% bucket is least negative but still fails net >= +0.05 R on both
splits. This is a table, not a rule (per PREREG) — cell 1,438's population-level closure stands.

## Live fills, HOD-break, 9/25-9/29 (data/trades.db; no verdict)
| date | symbol | R% | bucket | gross R | status |
|---|---|---|---|---|---|
| 2026-09-25 | CDNA | 1.32% | 0.75-1.5% | -1.805 | eod / closed |
| 2026-09-29 | AXTX | 4.09% | >=3% | 0.553 | ops_flatten / pending_new |
| 2026-09-29 | BE | 0.67% | <0.75% | 0.935 | ops_flatten / pending_new |
| 2026-09-29 | DUOL | 1.27% | 0.75-1.5% | 0.046 | ops_flatten / pending_new |
| 2026-09-29 | MRNA | 3.48% | >=3% | -1.031 | stop_loss / closed |
| 2026-09-29 | NBIG | 3.53% | >=3% | -0.030 | ops_flatten / pending_new |
| 2026-09-29 | NBIL | 3.86% | >=3% | -1.031 | stop_loss / closed |
| 2026-09-29 | TWST | 3.35% | >=3% | -0.999 | stop_loss / closed |
Caveat: AXTX/BE/DUOL/NBIG are `order_status=pending_new` with `exit_reason=ops_flatten` — the EOD flatten
was in flight when trades.db was read (not a final closed fill); MRNA/NBIL/TWST/CDNA are `closed`. CDNA's
`account` field is blank (pre-dates the account column's live/paper split). n=8 is far below any MDE.

## Caveats
Win rate uses gross raw_R > 0 (target vs stop, independent of the cost convention). Day-clustered t treats
each trading day as one cluster (bucket n as low as single digits in some TRAIN-H2/VAL cells makes several
t-stats and MDEs unstable — read the n and fills/wk columns before the t column). ex-top-5% drops the top
ceil(5%) of fills by net R within each bucket/split cell, not the pooled top 5%.
