# RESULT -- cells 1,717 / 1,718a / 1,718b / 1,719 (one-day cross-sectional reversal), builder output

Spec: `PREREG_1717.md` (frozen 2026-10-29). Script: `cell_1717.py` (`--split train|val --cost-bps 5`). TEST (2023+) was never loaded. Net = gross - 4 x cost per side (1x long / 1x short book, 5 bps/side placeholder = 20 bps/day literal reading). Not independently rebuilt yet: nothing here goes to the owner before the rebuild (ladder).


## Split TRAIN (2016-01-01..2019-12-31), cost 5 bps/side

### Data check

* Signal sessions: **990** (2016-01-25..2019-12-27); last 2 sessions of the split dropped (no t+2 exit).
* Mean universe size (top-1000 rule, eligible names): **987**; mean names ranked after the signal screens (1717): **1000**; decile size **100**.
* Symbol screen (`fiftytwo_wh.symbol_screen`): 4585 of 7849 panel symbols allowed. Exclusions:

| reason | symbols | share |
|---|---|---|
| wrapper_etf | 1964 | 25.0% |
| name_pattern | 487 | 6.2% |
| asset_unknown | 401 | 5.1% |
| punctuation | 382 | 4.9% |
| not_in_asset_map | 30 | 0.4% |


Signal exclusions (|1-day close move| > 50 % -> out of the ranking; no-signal = missing close_{t-1}/open_t) and picked name-days with no price (counted as 0):

| cell | jump excl (total) | /day | no-signal (total) | /day | entry open missing | exit missing MOO | exit missing MOC |
|---|---|---|---|---|---|---|---|
| 1717 | 37 | 0.04 | 0 | 0.0 | 0.002% | 0.000% | 0.000% |
| 1718a | 37 | 0.04 | 0 | 0.0 | 0.002% | 0.001% | 0.000% |
| 1718b | 37 | 0.04 | 0 | 0.0 | 0.001% | 0.000% | 0.000% |


Price-scale check on picks (beyond +500 % / -80 % = unadjusted corporate action):

| cell | return | picks w/ price | > +500% | < -80% | max | min |
|---|---|---|---|---|---|---|
| 1717 | ret1 (open t+1 -> open t+2) | 197922 | 1 | 2 | +1289% | -86% |
| 1717 | ret2 (open t+1 -> close t+1) | 197922 | 0 | 0 | +70% | -49% |
| 1718a | ret1 (open t+1 -> open t+2) | 197920 | 0 | 2 | +88% | -86% |
| 1718a | ret2 (open t+1 -> close t+1) | 197922 | 0 | 0 | +70% | -42% |
| 1718b | ret1 (open t+1 -> open t+2) | 197924 | 1 | 3 | +1289% | -89% |
| 1718b | ret2 (open t+1 -> close t+1) | 197924 | 0 | 0 | +70% | -49% |


### Per-cell results (primary exit = open t+2; units bps/day, 1x long / 1x short, days are the observations)

| cell | N | gross LS | net LS | sd | iid t | CI year-block | CI month-block | ex-top-5% | capped +2% | long gross | long net | long excess vs univ | short net | green wk net | green wk gross |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1717 | 990 | 4.3 | -15.7 | 100 | -4.91 | [-20.9, -10.3] | [-22.4, -9.2] | -27.3 | -17.0 | 8.9 | -1.1 | -7.4 | -14.6 | 30.7% | 57.1% |
| 1718a | 990 | -0.6 | -20.6 | 100 | -6.48 | [-23.5, -17.2] | [-26.5, -14.9] | -32.5 | -22.5 | 5.4 | -4.6 | -10.9 | -16.0 | 28.3% | 50.2% |
| 1718b | 990 | 3.9 | -16.1 | 95 | -5.35 | [-21.7, -11.8] | [-22.6, -9.6] | -26.9 | -17.3 | 8.4 | -1.6 | -7.9 | -14.5 | 31.2% | 57.1% |


Secondary exit (close t+1, MOC):

| cell | gross LS | net LS | iid t | ex-top-5% | long gross | long net | long excess | short net | green wk |
|---|---|---|---|---|---|---|---|---|---|
| 1717 | 3.8 | -16.2 | -6.37 | -25.6 | 2.1 | -7.9 | -9.8 | -8.3 | 31.7% |
| 1718a | 2.4 | -17.6 | -6.87 | -27.4 | 1.6 | -8.4 | -10.3 | -9.2 | 27.3% |
| 1718b | 2.5 | -17.5 | -7.40 | -26.2 | 1.5 | -8.5 | -10.4 | -9.0 | 28.3% |


Random count-matched null (1717 days, 1000 draws of random k/k books, same costs): green-week share net mean 5.2% (p95 7.8%), gross mean 50.1% (p95 55.6%). Mean universe EW next-day open-to-open return (long beta reference) 6.3 bps/day.


### 1,719 decile tables (mean next-day open-to-open return by signal decile, bps; decile 1 = lowest signal; minus = excess over universe mean)

| cell | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 | rank corr (-1 = monotone reversal) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1717 | 2.6 | 1.1 | 1.2 | -0.9 | 0.0 | -0.1 | -0.1 | -0.7 | -1.5 | -1.7 | -0.87 |
| 1718a | -0.9 | -1.8 | -0.8 | -0.1 | -0.4 | 0.1 | 0.9 | 2.4 | 1.0 | -0.3 | 0.77 |
| 1718b | 2.1 | 1.6 | 1.1 | 0.5 | -0.2 | -0.4 | -0.0 | -1.5 | -1.4 | -1.7 | -0.95 |


### Per-year net LS (bps/day) and calendar-year t

| cell | year | days | gross | net | t | long net |
|---|---|---|---|---|---|---|
| 1717 | 2016 | 238 | 12.3 | -7.7 | -1.03 | 8.8 |
| 1717 | 2017 | 251 | -2.6 | -22.6 | -3.28 | -2.3 |
| 1717 | 2018 | 251 | 7.2 | -12.8 | -2.17 | -10.7 |
| 1717 | 2019 | 250 | 0.8 | -19.2 | -3.76 | 0.4 |
| 1718a | 2016 | 238 | 4.6 | -15.4 | -1.97 | 6.3 |
| 1718a | 2017 | 251 | 0.1 | -19.9 | -4.14 | -3.9 |
| 1718a | 2018 | 251 | -2.0 | -22.0 | -3.14 | -16.6 |
| 1718a | 2019 | 250 | -4.7 | -24.7 | -4.53 | -3.8 |
| 1718b | 2016 | 238 | 7.9 | -12.1 | -1.74 | 4.7 |
| 1718b | 2017 | 251 | -4.8 | -24.8 | -3.68 | -2.9 |
| 1718b | 2018 | 251 | 8.6 | -11.4 | -2.07 | -9.8 |
| 1718b | 2019 | 250 | 4.0 | -16.0 | -3.40 | 2.0 |


### Realised-vol state (20-session vol of the universe EW move, terciles within the split; Nagel: reversal stronger in high vol)

| cell | vol tercile | days | gross | net | t |
|---|---|---|---|---|---|
| 1717 | low | 326 | -1.7 | -21.7 | -5.50 |
| 1718a | low | 326 | 1.6 | -18.4 | -4.94 |
| 1718b | low | 326 | -3.4 | -23.4 | -6.15 |
| 1717 | mid | 325 | 5.1 | -14.9 | -2.41 |
| 1718a | mid | 325 | -2.2 | -22.2 | -4.35 |
| 1718b | mid | 325 | 3.3 | -16.7 | -2.80 |
| 1717 | high | 325 | 10.6 | -9.4 | -1.55 |
| 1718a | high | 325 | -0.1 | -20.1 | -2.88 |
| 1718b | high | 325 | 12.3 | -7.7 | -1.37 |


### Sensitivities (primary exit)

| cell | bps/side | net LS | t | ex-top-5% | long net |
|---|---|---|---|---|---|
| 1717 | 2.5 | -5.7 | -1.78 | -17.3 | 3.9 |
| 1717 | 3 | -7.7 | -2.40 | -19.3 | 2.9 |
| 1717 | 5 | -15.7 | -4.91 | -27.3 | -1.1 |
| 1717 | 10 | -35.7 | -11.17 | -47.3 | -11.1 |
| 1718a | 2.5 | -10.6 | -3.33 | -22.5 | 0.4 |
| 1718a | 3 | -12.6 | -3.96 | -24.5 | -0.6 |
| 1718a | 5 | -20.6 | -6.48 | -32.5 | -4.6 |
| 1718a | 10 | -40.6 | -12.77 | -52.5 | -14.6 |
| 1718b | 2.5 | -6.1 | -2.03 | -16.9 | 3.4 |
| 1718b | 3 | -8.1 | -2.69 | -18.9 | 2.4 |
| 1718b | 5 | -16.1 | -5.35 | -26.9 | -1.6 |
| 1718b | 10 | -36.1 | -11.98 | -46.9 | -11.6 |


(2.5 bps/side equals the PREREG's "hurdle ~10 bps/day" reading, i.e. cost on half-weights; the literal 5 bps/side on a 1x/1x book is 20 bps/day.)

| cell | variant | days | gross LS | net LS | t | ex-top-5% | long net | long excess |
|---|---|---|---|---|---|---|---|---|
| 1717 | top500 | 990 | 6.2 | -13.8 | -4.01 | -27.4 | -1.3 | -7.4 |
| 1717 | quintile | 990 | 3.5 | -16.5 | -7.13 | -25.3 | -1.9 | -8.1 |
| 1718a | top500 | 990 | -0.5 | -20.5 | -5.44 | -34.5 | -5.0 | -11.0 |
| 1718a | quintile | 990 | -1.7 | -21.7 | -8.94 | -30.9 | -5.1 | -11.4 |
| 1718b | top500 | 990 | 7.7 | -12.3 | -3.77 | -24.4 | -0.9 | -7.0 |
| 1718b | quintile | 990 | 3.4 | -16.6 | -7.72 | -24.6 | -1.8 | -8.1 |
| 1717 | ex_1718a_overlap | 990 | 0.7 | -19.3 | -5.81 | -30.2 | -2.4 | -8.6 |


### Weekly table (cadence; R = sd of daily net LS of the cell)

| cell | weeks | weekly P10 (R) | green wk | strong wk (>= +5R) | gap median (wk) | gap P90 (wk) | bleed P90 (R, worst decile of inter-strong cum) |
|---|---|---|---|---|---|---|---|
| 1717 | 205 | -3.23 | 30.7% | 1 | n/a | n/a | n/a |
| 1718a | 205 | -3.58 | 28.3% | 0 | n/a | n/a | n/a |
| 1718b | 205 | -3.21 | 31.2% | 2 | 139.0 | 139.0 | -142.2 |


## Split VAL (2020-01-01..2022-12-31), cost 5 bps/side

### Data check

* Signal sessions: **754** (2020-01-02..2022-12-28); last 2 sessions of the split dropped (no t+2 exit).
* Mean universe size (top-1000 rule, eligible names): **1000**; mean names ranked after the signal screens (1717): **1000**; decile size **100**.
* Symbol screen (`fiftytwo_wh.symbol_screen`): 5725 of 10251 panel symbols allowed. Exclusions:

| reason | symbols | share |
|---|---|---|
| wrapper_etf | 2769 | 27.0% |
| name_pattern | 1006 | 9.8% |
| punctuation | 660 | 6.4% |
| not_in_asset_map | 50 | 0.5% |
| asset_unknown | 41 | 0.4% |


Signal exclusions (|1-day close move| > 50 % -> out of the ranking; no-signal = missing close_{t-1}/open_t) and picked name-days with no price (counted as 0):

| cell | jump excl (total) | /day | no-signal (total) | /day | entry open missing | exit missing MOO | exit missing MOC |
|---|---|---|---|---|---|---|---|
| 1717 | 87 | 0.12 | 0 | 0.0 | 0.018% | 0.016% | 0.000% |
| 1718a | 87 | 0.12 | 0 | 0.0 | 0.027% | 0.017% | 0.000% |
| 1718b | 87 | 0.12 | 0 | 0.0 | 0.017% | 0.010% | 0.000% |


Price-scale check on picks (beyond +500 % / -80 % = unadjusted corporate action):

| cell | return | picks w/ price | > +500% | < -80% | max | min |
|---|---|---|---|---|---|---|
| 1717 | ret1 (open t+1 -> open t+2) | 150607 | 0 | 0 | +301% | -72% |
| 1717 | ret2 (open t+1 -> close t+1) | 150631 | 0 | 0 | +498% | -63% |
| 1718a | ret1 (open t+1 -> open t+2) | 150592 | 1 | 0 | +538% | -72% |
| 1718a | ret2 (open t+1 -> close t+1) | 150618 | 0 | 0 | +293% | -63% |
| 1718b | ret1 (open t+1 -> open t+2) | 150618 | 1 | 1 | +538% | -90% |
| 1718b | ret2 (open t+1 -> close t+1) | 150633 | 0 | 0 | +498% | -63% |


### Per-cell results (primary exit = open t+2; units bps/day, 1x long / 1x short, days are the observations)

| cell | N | gross LS | net LS | sd | iid t | CI year-block | CI month-block | ex-top-5% | capped +2% | long gross | long net | long excess vs univ | short net | green wk net | green wk gross |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1717 | 754 | -0.9 | -20.9 | 229 | -2.50 | [-37.4, -11.4] | [-34.8, -7.5] | -49.0 | -39.1 | 0.6 | -9.4 | -13.2 | -11.5 | 38.2% | 49.7% |
| 1718a | 754 | 2.6 | -17.4 | 219 | -2.18 | [-21.3, -15.0] | [-34.7, -0.8] | -43.4 | -34.5 | 2.7 | -7.3 | -11.1 | -10.1 | 47.1% | 54.8% |
| 1718b | 754 | -2.9 | -22.9 | 209 | -3.02 | [-44.6, -9.4] | [-36.1, -10.7] | -49.3 | -38.2 | 0.8 | -9.2 | -13.1 | -13.7 | 36.3% | 51.0% |


Secondary exit (close t+1, MOC):

| cell | gross LS | net LS | iid t | ex-top-5% | long gross | long net | long excess | short net | green wk |
|---|---|---|---|---|---|---|---|---|---|
| 1717 | -1.0 | -21.0 | -3.04 | -44.4 | -6.6 | -16.6 | -15.2 | -4.4 | 36.9% |
| 1718a | 5.5 | -14.5 | -2.47 | -34.2 | -3.8 | -13.8 | -12.4 | -0.8 | 46.5% |
| 1718b | -4.1 | -24.1 | -3.66 | -46.7 | -6.6 | -16.6 | -15.2 | -7.5 | 35.7% |


Random count-matched null (1717 days, 1000 draws of random k/k books, same costs): green-week share net mean 12.6% (p95 16.6%), gross mean 50.0% (p95 56.7%). Mean universe EW next-day open-to-open return (long beta reference) 3.8 bps/day.


### 1,719 decile tables (mean next-day open-to-open return by signal decile, bps; decile 1 = lowest signal; minus = excess over universe mean)

| cell | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 | rank corr (-1 = monotone reversal) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1717 | -3.2 | 2.2 | 3.3 | 2.2 | 1.5 | -0.4 | -1.2 | -1.7 | -0.5 | -2.3 | -0.38 |
| 1718a | -1.1 | 0.7 | 0.9 | 1.3 | 1.8 | 1.6 | 0.7 | -0.2 | -2.0 | -3.6 | -0.43 |
| 1718b | -3.1 | 1.3 | 2.2 | 2.3 | 1.5 | 0.3 | -1.5 | -1.0 | -2.0 | -0.1 | -0.28 |


### Per-year net LS (bps/day) and calendar-year t

| cell | year | days | gross | net | t | long net |
|---|---|---|---|---|---|---|
| 1717 | 2020 | 253 | 5.9 | -14.1 | -0.86 | 6.2 |
| 1717 | 2021 | 252 | 8.6 | -11.4 | -0.92 | 1.3 |
| 1717 | 2022 | 249 | -17.4 | -37.4 | -2.63 | -35.9 |
| 1718a | 2020 | 253 | 4.2 | -15.8 | -0.94 | 1.2 |
| 1718a | 2021 | 252 | -1.3 | -21.3 | -1.96 | -6.1 |
| 1718a | 2022 | 249 | 5.0 | -15.0 | -1.13 | -17.1 |
| 1718b | 2020 | 253 | 4.9 | -15.1 | -1.10 | 5.3 |
| 1718b | 2021 | 252 | 10.6 | -9.4 | -0.80 | 2.8 |
| 1718b | 2022 | 249 | -24.6 | -44.6 | -3.21 | -36.1 |


### Realised-vol state (20-session vol of the universe EW move, terciles within the split; Nagel: reversal stronger in high vol)

| cell | vol tercile | days | gross | net | t |
|---|---|---|---|---|---|
| 1717 | low | 252 | 0.8 | -19.2 | -1.94 |
| 1718a | low | 252 | 3.4 | -16.6 | -1.73 |
| 1718b | low | 252 | 4.3 | -15.7 | -1.79 |
| 1717 | mid | 251 | 9.6 | -10.4 | -0.86 |
| 1718a | mid | 251 | 22.2 | 2.2 | 0.19 |
| 1718b | mid | 251 | -7.6 | -27.6 | -2.30 |
| 1717 | high | 251 | -13.1 | -33.1 | -1.69 |
| 1718a | high | 251 | -17.7 | -37.7 | -2.01 |
| 1718b | high | 251 | -5.5 | -25.5 | -1.47 |


### Sensitivities (primary exit)

| cell | bps/side | net LS | t | ex-top-5% | long net |
|---|---|---|---|---|---|
| 1717 | 2.5 | -10.9 | -1.31 | -39.0 | -4.4 |
| 1717 | 3 | -12.9 | -1.54 | -41.0 | -5.4 |
| 1717 | 5 | -20.9 | -2.50 | -49.0 | -9.4 |
| 1717 | 10 | -40.9 | -4.90 | -69.0 | -19.4 |
| 1718a | 2.5 | -7.4 | -0.92 | -33.4 | -2.3 |
| 1718a | 3 | -9.4 | -1.17 | -35.4 | -3.3 |
| 1718a | 5 | -17.4 | -2.18 | -43.4 | -7.3 |
| 1718a | 10 | -37.4 | -4.68 | -63.4 | -17.3 |
| 1718b | 2.5 | -12.9 | -1.70 | -39.3 | -4.2 |
| 1718b | 3 | -14.9 | -1.97 | -41.3 | -5.2 |
| 1718b | 5 | -22.9 | -3.02 | -49.3 | -9.2 |
| 1718b | 10 | -42.9 | -5.65 | -69.3 | -19.2 |


(2.5 bps/side equals the PREREG's "hurdle ~10 bps/day" reading, i.e. cost on half-weights; the literal 5 bps/side on a 1x/1x book is 20 bps/day.)

| cell | variant | days | gross LS | net LS | t | ex-top-5% | long net | long excess |
|---|---|---|---|---|---|---|---|---|
| 1717 | top500 | 754 | 1.5 | -18.5 | -2.00 | -49.1 | -6.6 | -10.8 |
| 1717 | quintile | 754 | 0.9 | -19.1 | -2.88 | -41.4 | -6.6 | -10.5 |
| 1718a | top500 | 754 | 6.7 | -13.3 | -1.53 | -41.4 | -5.2 | -9.3 |
| 1718a | quintile | 754 | 2.7 | -17.3 | -2.64 | -38.8 | -6.4 | -10.2 |
| 1718b | top500 | 754 | -1.3 | -21.3 | -2.55 | -50.5 | -8.1 | -12.2 |
| 1718b | quintile | 754 | 0.2 | -19.8 | -3.34 | -40.5 | -7.0 | -10.9 |
| 1717 | ex_1718a_overlap | 754 | -4.3 | -24.3 | -3.07 | -51.4 | -7.2 | -11.0 |


### Weekly table (cadence; R = sd of daily net LS of the cell)

| cell | weeks | weekly P10 (R) | green wk | strong wk (>= +5R) | gap median (wk) | gap P90 (wk) | bleed P90 (R, worst decile of inter-strong cum) |
|---|---|---|---|---|---|---|---|
| 1717 | 157 | -2.57 | 38.2% | 1 | n/a | n/a | n/a |
| 1718a | 157 | -2.78 | 47.1% | 2 | 102.0 | 102.0 | -54.0 |
| 1718b | 157 | -3.08 | 36.3% | 2 | 52.0 | 52.0 | -20.1 |


### VAL pass bar (frozen) -- judged per cell


**Cell 1717 (close-to-close) -- verdict REJECT**; year-block 95 % CI of net LS [-37.4, -11.4] bps (3 blocks, crude), month-block [-34.8, -7.5]

| pass-bar item | value | result |
|---|---|---|
| mean net LS >= +5 bps/day | -20.89 | FAIL |
| iid t >= 2.0 | -2.50 | FAIL |
| ex-top-5% of days > 0 | -49.00 | FAIL |
| winner-capped (+2%/day) > 0 | -39.09 | FAIL |
| LONG leg alone net > 0 (raw) | -9.35 (excess vs univ -13.19) | FAIL |
| TRAIN same sign, t >= 1 | TRAIN net -15.67, t -4.91 | PASS |
| >= 3 of 4 TRAIN years positive | 0 of 4 | FAIL |
| green-week share >= 55 % | 38.2% | FAIL |
| green-week share > random-decile null mean | 38.2% vs 12.6% (p95 16.6%) | PASS |

Cadence bar lines (docs/cadence_bar.md; NaN gap = fewer than 2 strong weeks = FAIL):

| cadence item | value | result |
|---|---|---|
| weekly P10 >= -2R | -2.57 | FAIL |
| strong-week gap median <= 3 wk | n/a | FAIL |
| strong-week gap P90 <= 6 wk | n/a | FAIL |
| bleed between strong weeks >= -4R at P90 | n/a | FAIL |
| >= 3 fills/week | by construction (200 names/day) | PASS |


**Cell 1718a (overnight component) -- verdict REJECT**; year-block 95 % CI of net LS [-21.3, -15.0] bps (3 blocks, crude), month-block [-34.7, -0.8]

| pass-bar item | value | result |
|---|---|---|
| mean net LS >= +5 bps/day | -17.38 | FAIL |
| iid t >= 2.0 | -2.18 | FAIL |
| ex-top-5% of days > 0 | -43.45 | FAIL |
| winner-capped (+2%/day) > 0 | -34.52 | FAIL |
| LONG leg alone net > 0 (raw) | -7.29 (excess vs univ -11.13) | FAIL |
| TRAIN same sign, t >= 1 | TRAIN net -20.60, t -6.48 | PASS |
| >= 3 of 4 TRAIN years positive | 0 of 4 | FAIL |
| green-week share >= 55 % | 47.1% | FAIL |
| green-week share > random-decile null mean | 47.1% vs 12.6% (p95 16.6%) | PASS |

Cadence bar lines (docs/cadence_bar.md; NaN gap = fewer than 2 strong weeks = FAIL):

| cadence item | value | result |
|---|---|---|
| weekly P10 >= -2R | -2.78 | FAIL |
| strong-week gap median <= 3 wk | 102.0 | FAIL |
| strong-week gap P90 <= 6 wk | 102.0 | FAIL |
| bleed between strong weeks >= -4R at P90 | -54.0 | FAIL |
| >= 3 fills/week | by construction (200 names/day) | PASS |


**Cell 1718b (intraday component) -- verdict REJECT**; year-block 95 % CI of net LS [-44.6, -9.4] bps (3 blocks, crude), month-block [-36.1, -10.7]

| pass-bar item | value | result |
|---|---|---|
| mean net LS >= +5 bps/day | -22.93 | FAIL |
| iid t >= 2.0 | -3.02 | FAIL |
| ex-top-5% of days > 0 | -49.33 | FAIL |
| winner-capped (+2%/day) > 0 | -38.18 | FAIL |
| LONG leg alone net > 0 (raw) | -9.22 (excess vs univ -13.06) | FAIL |
| TRAIN same sign, t >= 1 | TRAIN net -16.12, t -5.35 | PASS |
| >= 3 of 4 TRAIN years positive | 0 of 4 | FAIL |
| green-week share >= 55 % | 36.3% | FAIL |
| green-week share > random-decile null mean | 36.3% vs 12.6% (p95 16.6%) | PASS |

Cadence bar lines (docs/cadence_bar.md; NaN gap = fewer than 2 strong weeks = FAIL):

| cadence item | value | result |
|---|---|---|
| weekly P10 >= -2R | -3.08 | FAIL |
| strong-week gap median <= 3 wk | 52.0 | FAIL |
| strong-week gap P90 <= 6 wk | 52.0 | FAIL |
| bleed between strong weeks >= -4R at P90 | -20.1 | FAIL |
| >= 3 fills/week | by construction (200 names/day) | PASS |


## Caveats read as an adversary

**Verdict: all three cells REJECT on VAL (net LS -17 to -23 bps/day, long leg net < 0, 0 of 4 TRAIN years positive); the gross LS is ~0 (-2.9 to +2.6 bps/day VAL, -0.6 to +4.3 TRAIN), i.e. the 20 bps/day cost (literal 4 sides x 5 bps) is not the cause of the failure. Even at the 2.5 bps/side "half-weight" reading of the PREREG hurdle, net stays negative (-7 to -13 bps, t -0.9 to -1.7). Cost sensitivity: no cost level in 2.5-10 bps/side makes any cell positive.**

* **Detected absence, not an underpowered null:** VAL se of the daily net LS is about 8 bps (sd 209-229 bps over 754 days), NOT the PREREG's 4.4 bps (it assumed sd 1.2 %/day; the realised daily LS sd is 2.1-2.3 % in VAL and 1.0 % in TRAIN). So the VAL MDE is ~16 bps/day at t 2, not ~9; the VAL gross CI (about +-16 bps) comfortably contains the +10 to +25 bps a net pass would need to be reachable, but VAL alone cannot resolve a +5 bps net edge. TRAIN (sd ~100 bps, se ~3 bps) is tight and says gross +4.3 / -0.6 / +3.9 bps, i.e. below the 20 bps cost by a wide margin; TRAIN and VAL agree in sign of the net (negative) in every cell.
* **VAL sd is driven by the 2020 crash/June-2020 retail-squeeze days** (worst 1717 gross days: 2020-03-16/17, 2020-06-03, 2020-06-09, each about -10 %/day on a 100/100 book; best 2020-03-18 +13 %). 1717 gross ex March 2020 is +3.1 bps; 1717 ex both-tail 2 % of days is +0.7 bps. Nothing is rescued by trimming: no truncation produces a net-positive cell (ex-top-5 % net is -43 to -49 bps).
* **The 1,719 decile tables show no usable monotone reversal:** the lowest decile is the WORST (D1 -3.2 bps vs universe on 1717, -3.1 on 1718b), the middle deciles are best; rank correlation only -0.28 to -0.43 and entirely from the top end (1718a D10 -3.6 bps). The published mechanism (lowest decile outperforms) is absent in this universe/era on the long side; the short side loses too (short net -10 to -14 bps VAL).
* **Vol-state split:** no Nagel pattern -- the HIGH-vol tercile has the worst gross LS on 1717 and 1718a (-13.1, -17.7 bps), the opposite of the paper's high-VIX reversal; 1718a mid-vol is the only positive net (+2.2 bps, t 0.19) cell in the report -- a one-of-nine-cell noise level.
* **Pass-bar lines that print PASS are artefacts:** "TRAIN same sign, t >= 1" is satisfied by both being negative (the bar presupposes a positive VAL); "green-week share > random-decile null mean" is trivial because the null pays the same 20 bps/day and sits at 12.6 % net green weeks. Beside the gross null (50.0 %, p95 56.7 %) the cells' gross green-week shares are 49.7 / 54.8 / 51.0 % -- at the null. The 3-year year-block bootstrap is crude (27 distinct resamples); month-block CIs exclude 0 on the negative side for all three cells.
* **Cadence lines:** the strong-week (>= +5R) count is 1-2 in 157 VAL weeks, so gap/bleed rows are n/a or enormous -- all FAIL.
* **Survivorship / panel:** the Alpaca daily panel is survivorship-exposed (delisted names absent) and the symbol screen uses a PRESENT-DAY asset map (it excludes 27-32 % of panel symbols as wrappers, 10 % by name, and 0.5 % not in the map -- a not-in-map delisted stock is silently dropped, which removes exactly the names that fell). For a reversal strategy long losers this biases the long leg UP (the biggest-loser-then-delisted cases are missing), so the absent edge is, if anything, overstated here. Mean universe is a constant 1,000 (the 15-of-20-bars and $10 screens never bind below the cap in these years).
* **Cost is a placeholder:** 5 bps/side at the open/close auction for ranks up to 1,000 by $vol, not a measured NBBO; MOO/MOC fills for the most extreme decile names (just-moved losers/winners) are likely worse, and the short leg ignores borrow and locate (only ~11 of 539 leveraged ETFs are ETB; single-stock hard-to-borrow share of the top decile is unknown). Both biases go against the strategy; they cannot create the null.
* **Exit-missing shares (counted 0 return):** entry open missing 0.017-0.027 % of picked name-days, MOO exit missing 0.010-0.017 %, MOC exit 0 % (VAL); TRAIN similar (see the Data check). Negligible, and symmetric between winners/losers is untested.
* **Price-scale:** 1718a ret1 has one pick > +500 % (CENN 2021-01-26, +538 % on a short, about -54 bps on that day) and 1718b has CENN (+538 % on a long) and UHAL 2022-11-08 (-90 % on a long, a likely unadjusted corporate-action/share-class artefact, about -90 bps on that day; mean effect on 754 days about 0.1 bps). Per the project's price-scale rule these are "flagged, not adjusted"; neither can flip a REJECT (net is -17 to -23 bps), but a formal run-through would exclude them.
* **Deviations / interpretation choices (all declared):** (1) price floor and bar-count screens applied BEFORE the top-1,000 dollar-volume ranking; (2) the |close move| > 50 % exclusion is applied to the close-to-close move for ALL three cells (87 name-days in VAL, 0.12/day; no-signal exclusions 0); (3) the LS book's cost is 4 x 5 bps (both legs, both sides), the PREREG's "hurdle ~10 bps/day" corresponds to 2.5 bps/side and is reported as a sensitivity row; (4) long-leg pass line uses the RAW long return (excess over the universe EW mean is printed beside it and is worse); (5) tercile cuts for the vol state are within-split (report-only); (6) weeks are ISO weeks of the signal date; (7) the last 2 sessions of each split are dropped (no t+2 open); (8) TRAIN signal days start 2016-01-25 (990 days: the first ~13 sessions have fewer than 15 bars of history, so the universe is below the 50-name minimum and skipped); (9) one random null (k/k from the 1717 days, same draws for all cells).
* **Not done by design:** no independent rebuild (ladder: only needed on a PASS), no TEST. The cells are rejected on VAL, so TEST stays sealed.
