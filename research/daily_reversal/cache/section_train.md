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
