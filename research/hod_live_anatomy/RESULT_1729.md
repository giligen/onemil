# RESULT 1,729 — tight-R cohort (R_pct < 1.5 %) on cell 1,438's 9,911-fill population

Script `cell_1729.py`, per-fill `cohort_1729.csv`. PREREG `PREREG_1729.md` (frozen). Cost: entry level x (1+8 bps); marketable exits (stop/stop_bar/eod/eod_fallback) 12 bps off the recorded exit; TARGET prints pay NOTHING (resting limit). R unit = level − stop; R_pct = (level − stop)/level. Entry charged from the level, so the file's own modelled slip is not double-charged.

**TEST WAS NOT READ.** C1 did NOT pass on both TRAIN-H2 and VAL, so TEST stays sealed and was not read.

Weeks: TRAIN-H2 27, VAL 22 (weeks with any armed row). Cells run here: C1 (2 cohorts x 2 splits), C2, C3 (2 risks x 2 splits) + a sensitivity — this is the 14th cut on the HOD live book tonight; TRAIN-H2/VAL are a replication of a SEEN hypothesis.

## C1 / C2 — TRAIN-H2

| cohort | n | fills/wk | net R | day-clust t | ex-top-5 % | winner-capped +2R | gross R | alt-week halves A / B |
|---|---|---|---|---|---|---|---|---|
| all fills | 4398 | 162.9 | -0.048 | -1.19 | -0.167 | -0.070 | +0.060 | -0.070 / -0.027 |
| kept R_pct < 1.5 % | 2219 | 82.2 | -0.088 | -2.01 | -0.211 | -0.115 | +0.053 | -0.076 / -0.099 |
| complement >= 1.5 % | 2179 | 80.7 | -0.008 | -0.15 | -0.122 | -0.024 | +0.068 | -0.064 / +0.049 |
| C2 R_pct < 0.75 % | 0 | 0.0 | n/a | n/a | n/a | n/a | n/a | n/a / n/a |
| 0.75-1.5 % | 2219 | 82.2 | -0.088 | -2.01 | -0.211 | -0.115 | +0.053 | -0.076 / -0.099 |

MDE (2.8 x day-clustered SE, kept cohort): 0.122 R; complement 0.152 R; C2 nan R.

## C1 / C2 — VAL

| cohort | n | fills/wk | net R | day-clust t | ex-top-5 % | winner-capped +2R | gross R | alt-week halves A / B |
|---|---|---|---|---|---|---|---|---|
| all fills | 5513 | 250.6 | -0.055 | -1.41 | -0.175 | -0.075 | +0.051 | -0.097 / -0.022 |
| kept R_pct < 1.5 % | 2542 | 115.5 | -0.097 | -2.64 | -0.222 | -0.123 | +0.044 | -0.127 / -0.074 |
| complement >= 1.5 % | 2971 | 135.0 | -0.020 | -0.38 | -0.135 | -0.035 | +0.057 | -0.072 / +0.023 |
| C2 R_pct < 0.75 % | 0 | 0.0 | n/a | n/a | n/a | n/a | n/a | n/a / n/a |
| 0.75-1.5 % | 2542 | 115.5 | -0.097 | -2.64 | -0.222 | -0.123 | +0.044 | -0.127 / -0.074 |

MDE (2.8 x day-clustered SE, kept cohort): 0.103 R; complement 0.146 R; C2 nan R.

C2 (R_pct < 0.75 %) is EMPTY in this population: cell 1,438 enforced a 0.75 % R floor, so kept (< 1.5 %) = the 0.75-1.5 % bucket only; the live < 0.75 % cohort (n 29) has no BT counterpart.

Pooled kept (TRAIN-H2+VAL): n 4761, net -0.093 R, day-clustered t -3.28.

## C3 — the book "keep only R_pct < 1.5 %" (weekly P&L, zero-fill weeks included; $ = net R x risk, uncapped notional)

| split | risk | total $ | mean $/wk | green-week share | null mean (1,000 count-matched draws) | P(null >= obs) | P10 wk | worst wk | max DD $ | alt-week halves $/wk A / B | notional-capped total / DD / green |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN-H2 | $50 | -9730 | -360.4 | 19% | 40% | 0.999 | -1085 | -1606 | 11175 | -296.0 / -429.7 | -8745 / 10055 / 19% |
| TRAIN-H2 | $150 | -29189 | -1081.1 | 19% | 39% | 1.000 | -3254 | -4817 | 33526 | -887.9 / -1289.2 | -26235 / 30164 / 19% |
| VAL | $50 | -12322 | -560.1 | 27% | 32% | 0.828 | -1434 | -2735 | 12353 | -644.1 / -476.1 | -11182 / 11383 / 27% |
| VAL | $150 | -36966 | -1680.3 | 27% | 32% | 0.812 | -4303 | -8204 | 37060 | -1932.3 / -1428.3 | -33546 / 34148 / 27% |
| TRAIN-H2+VAL chronological | $50 | -22322 | -455.5 | 22% | | | | -2735 | 23798 | | |
| TRAIN-H2+VAL chronological | $150 | -66965 | -1366.6 | 22% | | | | -8204 | 71395 | | |

## Live paper cohort (fills_1721, valid_r)

* R_pct < 1.5 %: n 82, net +0.064 R, 95 % CI [-0.325, +0.454] (day-clustered, t 0.32), ex-top-5 % -0.095.
* R_pct < 0.75 %: n 29, net +0.105 R. Complement: n 149, net -0.076 R.
* (the ledger-based count 82 in the PREREG vs this n differ only by the R_pct definition/valid_r rows; both reported above as computed.)

Live/BT fill-rate ratio 0.210 (live 5.4 wk).

## Frozen pass bar, item by item

* TRAIN-H2: kept net >= +0.05: FAIL (-0.088); ex-top-5 % > 0: FAIL (-0.211); alt-week halves same sign (and positive): FAIL (-0.076/-0.099); complement <= 0: PASS (-0.008).
* VAL: kept net >= +0.05: FAIL (-0.097); ex-top-5 % > 0: FAIL (-0.222); alt-week halves same sign (and positive): FAIL (-0.127/-0.074); complement <= 0: PASS (-0.020).
* pooled day-clustered t >= 2: FAIL (-3.28).
* >= 3 fills/week at the live rate (BT rate x 0.21): TRAIN-H2 17.3, VAL 24.3 -> PASS.
* live tight fills >= 0: PASS (+0.064, CI [-0.325, +0.454]).
* TEST net >= +0.05 / ex-top-5 % > 0: NOT EVALUATED (TEST not read).

**C1 on TRAIN-H2 and VAL both pass: False.** Overall verdict: FAIL on the replication splits; TEST sealed and unread; per PREREG the parity defect is fixed the other way (the twin's gate goes live) and the cohort is reported as noise with its CI.

## Caveats

* The 9,911 file is cell 1,438's population: exits come from its recorded exit_price (NBBO / fill-instant exit-half sources, 76+58 bar-fallback exits); the cost model here is the task-frozen 8/12 bps, not the file's banded cost_R (0.23 R/fill) — with R_pct < 0.75 % a flat 20 bps round trip is itself 0.27+ R, so the cohort's net is very sensitive to the true spread of these (mostly large-cap) names; the live measured exit slip for them is not separately known.
* Tight-R fills are sized by the live $4,000 notional cap, so live dollar weight differs from the per-fill R mean; the notional-capped column in C3 shows it.
* Hypothesis seen in live data first (13 prior cuts), so TRAIN-H2/VAL are a replication, not a discovery; the sealed TEST is the confirmation and is absent from this population.
* C3 $ totals are uncapped by the live 4-concurrent / 12-per-day caps (BT cohort fires ~80-115 fills/wk vs ~49/wk live all-in), so they overstate dollars; the sign and green-week shares are the content.
* Trigger taken as the level (live trigger sits a tick above); R_pct difference is immaterial except below ~0.3 %.
