# RESULT — Cell 1,625: the index (SPY/IWM) as the instrument
PREREG_1623.md lines 27-34 (FROZEN). Verdict: **FAIL**.
## CAVEAT — read first (data-availability deviation from the literal spec)
PREREG says B30 = ARM events of **any status**. causal_arming_causal.csv, causal_arming_tick_rv.csv, features_1478_A.csv and bars_fills_1478.db were all inspected; none carries a timestamp for 'nofill'/'not_armed' rows (verified: bars_fills_1478.db and features_1478_A.csv are each scoped to exactly the 9,911 fills, 0 nofill symbol-days). B30 here is **B30_FILL_PROXY**: built from the 9,911 fills' own arm_m (features_1478_A.csv), not fill_min. This is a real causality weakening (at the true arm moment you do not yet know whether the order will fill), not just an undercount — a live system's true 'any status' B30 would run higher and possibly cross the decile earlier. A PASS below is evidence for the dry-instrument path only, not a clean test of the frozen spec; a true rebuild needs a full-universe crossing re-derivation from causal_arming.py's arm logic, out of scope for this cell.
Separately: the task's assumption that data/cache.db covers SPY *and* IWM through 2026-03-20 was checked and found wrong for IWM — cache.db had only 1,094 IWM rows (2025-04-07..09). IWM was pulled from Alpaca SIP for effectively the full 230-day range (226 days), SPY for the 48 days cache.db was missing (mostly 2026-03-21..05-29, plus scattered earlier gaps); both now show 230/230 required days with 0 lost (research/hod_entry/fetch_index_bars_1625.py log). Source is 'cache' vs 'alpaca' per row in index_bars_1625.parquet.
## Data
- Base fills + arm minute: causal_arming_causal.csv (status==fill, n=9,911) INNER JOIN features_1478_A.csv on (day,symbol,fill_min) for arm_m. holdout = VAL if split==VAL else TRAIN-H2 (n_days TRAIN-H2=128, VAL=102).
- SPY/IWM minute bars: research/hod_entry/index_bars_1625.parquet (built by fetch_index_bars_1625.py from data/cache.db READ-ONLY + Alpaca SIP).
- TRAIN-H2 top-decile threshold tau = **7.000** (B30 >= tau counts as "top decile"; pooled over every (day, minute) pair, minute grid 09:30-15:59 ET, TRAIN-H2 days only).
- Signal days found: 209; days with a fill but B30 never reached tau: 21.

## Time-of-day of m* (report-only — the confound PREREG names for 1,624 applies structurally to 1,625 too, since B30 rises through the morning; not hour-adjusted here because the 1,625 prose does not call for it, and thresholds cannot be tuned post-hoc)

| split | n | first m* (min) | median m* | last m* |
|---|---|---|---|---|
| TRAIN-H2 | 108 | 09:41 | 10:00 | 12:37 |
| VAL | 101 | 09:38 | 09:51 | 11:32 |

## Per-holdout stats — primary (60-minute hold)

| holdout | leg | n | mean net bps | t (day-clust) | t (iid) | ex-top-5% | winner-capped +1% | signals/wk |
|---|---|---|---|---|---|---|---|---|
| TRAIN-H2 | SPY | 108 | -4.17 | -1.63 | -1.63 | -6.90 | -4.17 | 4.00 |
| TRAIN-H2 | IWM | 108 | -6.95 | -1.55 | -1.55 | -11.86 | -7.55 | 4.00 |
| VAL | SPY | 101 | 1.34 | 0.39 | 0.39 | -2.56 | 1.11 | 4.59 |
| VAL | IWM | 101 | 6.19 | 1.05 | 1.05 | 0.12 | 5.07 | 4.59 |

## Placebo (random minute, same day, seed 1625) — 60-minute hold

| holdout | leg | n | real mean | placebo mean | margin | t (margin) |
|---|---|---|---|---|---|---|
| TRAIN-H2 | SPY | 108 | -4.17 | -1.63 | -2.53 | -0.79 |
| TRAIN-H2 | IWM | 108 | -6.95 | 3.23 | -10.18 | -2.01 |
| VAL | SPY | 101 | 1.34 | -4.56 | 5.90 | 1.82 |
| VAL | IWM | 101 | 6.19 | -5.37 | 11.56 | 2.10 |

## 30- / 120-minute holds (report-only, SPY + IWM, not pass-bar-graded)

| holdout | leg | n | mean net bps | t (day-clust) | ex-top-5% |
|---|---|---|---|---|---|
| TRAIN-H2 | SPY 30m | 108 | -3.76 | -1.92 | -5.82 |
| TRAIN-H2 | SPY 120m | 108 | -6.29 | -1.55 | -10.42 |
| TRAIN-H2 | IWM 30m | 108 | -6.12 | -1.78 | -9.76 |
| TRAIN-H2 | IWM 120m | 108 | -7.01 | -1.06 | -14.49 |
| VAL | SPY 30m | 101 | -1.67 | -0.67 | -4.10 |
| VAL | SPY 120m | 101 | 2.03 | 0.46 | -2.47 |
| VAL | IWM 30m | 101 | 2.09 | 0.49 | -2.26 |
| VAL | IWM 120m | 101 | 4.87 | 0.67 | -2.65 |

## Pass-bar checklist (graded on VAL, SPY 60-min leg — PREREG_1623.md line 33-34)

| check | value | pass |
|---|---|---|
| mean net (VAL, SPY 60-min) >= +8 bps | 1.34 | FAIL |
| day-clustered t (VAL) >= 2.5 | 0.39 | FAIL |
| ex-top-5% mean (VAL) > 0 | -2.56 | FAIL |
| placebo margin (VAL) >= +5 bps | 5.90 | PASS |
| placebo margin t (VAL) >= 2.0 | 1.82 | FAIL |
| signals/week (VAL) >= 2 | 4.59 | PASS |
| TRAIN-H2 same sign as VAL | TRAIN-H2=-4.17, VAL=1.34 | FAIL |

**Overall: FAIL** (2/7 checks passed).

## Consequence (per PREREG line 46-48)
FAIL — this frame closes on this population as tested. Note the proxy caveat above: a FAIL here does not by itself refute the literal any-status spec, since the signal tested is a weaker (fill-only) proxy for it; nothing on this population says the any-status version would behave the same, better, or worse.

## Not allowed (honored)
Decile threshold (90th pct), hold length (60 min), cost (2 bps) and tolerance (5 min) were set from the frozen PREREG/task text before any number was computed and were not adjusted after seeing results.
