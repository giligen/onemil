# DRIFT.md -- base-under-the-high descriptive exhibit

PREREG.md "Exhibits first" #1, format of research/hod_exit_lab/DRIFT.md. Universe = research/hod_pmh_causal/pm_candidates.csv (23,767 ORB wide-seed causal symbol-days). No pre-market data used (dropped per PREREG). TRAIN/VAL only below; TEST is sealed.

## Availability rail (all candidates, RTH 1-min bars)

- Candidates with usable RTH 1-min bars: 23575/23767 (99.2%) -- PASS (>=80% rail)
- No usable bars: 192 | no qualifying signal bar: 15499 | stop-at-open (kept in the cells at -cost, excluded from this exhibit): 0 | no bar after signal: 0

## Bar-density-in-hold rail (signals only)

- Total signals (signal bar found, entry/stop obtainable): 8076
- Dropped from PRIMARY arm (bar_density < 80%): 1312 (16.2% of signals)
- Bar source: {'rth': 7229, 'sip': 847}
- Winner/loser missingness gap (conservative-arm raw_rr>0 vs <=0, share with bar_density<80%): winners=1.9%, losers=20.6%, gap=18.7pp (>5pp -- flag)

## Cost

- No nbbo.csv coverage for these minutes (PREREG): proxy 15bp half-spread on BOTH legs + 2bp/side slip, for 100% of signals. Gross (raw_rr) vs net (proxy-cost) both reported below (C1 exit, used descriptively for this exhibit).

## PRIMARY arm by split (gross vs net, proxy cost, C1 exit)

| split | n | mean gross R | median gross R | mean net R | median net R |
|---|---|---|---|---|---|
| TEST | 1467 | 0.233 | -0.096 | 0.090 | -0.222 |
| TRAIN | 3423 | 0.115 | -0.651 | -0.022 | -0.811 |
| VAL | 1874 | 0.305 | -0.011 | 0.168 | -0.134 |

## Minute-since-entry table (unmanaged path, TRAIN+VAL, close-vs-entry in R, no stop/target/EOD applied)

| minute | n | mean R | median R | p10 R | p90 R |
|---|---|---|---|---|---|
| 0 | 6275 | 0.008 | 0.000 | -0.181 | 0.195 |
| 5 | 6275 | 0.021 | 0.000 | -0.432 | 0.515 |
| 10 | 6275 | 0.034 | 0.000 | -0.571 | 0.683 |
| 15 | 6275 | 0.042 | 0.000 | -0.678 | 0.824 |
| 20 | 6275 | 0.043 | 0.000 | -0.762 | 0.890 |
| 30 | 6275 | 0.048 | 0.000 | -0.929 | 1.063 |
| 45 | 6275 | 0.057 | 0.000 | -1.115 | 1.276 |
| 60 | 6275 | 0.053 | 0.000 | -1.303 | 1.429 |
| 90 | 6275 | 0.051 | 0.022 | -1.533 | 1.690 |
| 120 | 6275 | 0.061 | 0.035 | -1.707 | 1.884 |
| 150 | 6275 | 0.098 | 0.071 | -1.781 | 2.042 |
| 180 | 6275 | 0.108 | 0.074 | -1.913 | 2.139 |
| 210 | 6275 | 0.121 | 0.077 | -2.016 | 2.250 |
| 240 | 6275 | 0.134 | 0.095 | -2.052 | 2.381 |
| 270 | 6275 | 0.161 | 0.122 | -2.100 | 2.500 |
| 300 | 6275 | 0.164 | 0.133 | -2.167 | 2.548 |
| 330 | 6275 | 0.155 | 0.120 | -2.219 | 2.611 |
| 360 | 6275 | 0.166 | 0.121 | -2.219 | 2.637 |

Unmanaged mean R at 2h (minute=120) = 0.061R (n=6275) -- as informationless as HOD-break -- reported as such.

## ORB-overlap (share of signals that are also ORB picks, same day+symbol)

Any-row match: **2.8%** (176/6275). entered==1-only match: 2.5%. <=50% -- a distinct population from the ORB production book.

