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
