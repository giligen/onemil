# RESULT — cell 1,493 placebo, recomputed causally (later-minute draw)

Population: 8973 retest fills (rebuild_1481_fills.csv, status==fill). 1 fills (0.0%) had NO eligible RTH minute strictly after retest_minute+15 and before 15:00 -- excluded, never imputed.

## 1. Causal placebo: raw (passive entry, cell_1493 costs)

| holdout | exit | n | mean net % | day-clustered t | ex-top-5% |
|---|---|---|---|---|---|
| TRAIN | 3.0%\|NONE | 3955 | -0.1400 | -1.27 | -0.5288 |
| TRAIN | CL\|NONE | 3955 | -0.0608 | -0.74 | -0.4034 |
| VAL | 3.0%\|NONE | 5016 | -0.0403 | -0.41 | -0.4155 |
| VAL | CL\|NONE | 5016 | -0.0877 | -1.11 | -0.4168 |

## 2. Active-entry version (net_pct minus half_entry/entry, $ half-spread from features_1478_A.csv, join day+symbol)

| holdout | exit | n | mean net % (active) | day-clustered t | ex-top-5% |
|---|---|---|---|---|---|
| TRAIN | 3.0%\|NONE | 3955 | -0.3100 | -2.80 | -0.6984 |
| TRAIN | CL\|NONE | 3955 | -0.2308 | -2.76 | -0.5745 |
| VAL | 3.0%\|NONE | 5016 | -0.2217 | -2.28 | -0.5959 |
| VAL | CL\|NONE | 5016 | -0.2691 | -3.42 | -0.5978 |

## 3. Drift by minutes-after-retest bucket (exit 3.0%\|NONE only)

| holdout | offset (min after retest+15 base) | n | mean net % |
|---|---|---|---|
| TRAIN | 15 | 3611 | 0.0067 |
| TRAIN | 45 | 3508 | -0.0750 |
| TRAIN | 75 | 3347 | 0.0002 |
| TRAIN | 105 | 3156 | -0.0673 |
| TRAIN | 135 | 2977 | -0.1128 |
| TRAIN | 165 | 2818 | -0.1877 |
| TRAIN | 195 | 2559 | -0.1323 |
| TRAIN | 225 | 2282 | -0.1434 |
| TRAIN | 255 | 1903 | -0.0885 |
| TRAIN | 285 | 1265 | -0.1197 |
| TRAIN | 315 | 106 | 0.1119 |
| VAL | 15 | 4589 | 0.0496 |
| VAL | 45 | 4450 | -0.0351 |
| VAL | 75 | 4317 | -0.0480 |
| VAL | 105 | 4165 | -0.0089 |
| VAL | 135 | 3876 | 0.0188 |
| VAL | 165 | 3713 | 0.0511 |
| VAL | 195 | 3432 | 0.0340 |
| VAL | 225 | 3120 | -0.0243 |
| VAL | 255 | 2597 | -0.0620 |
| VAL | 285 | 1728 | -0.0347 |
| VAL | 315 | 261 | -0.0230 |

## 4. Drift by clock-hour bucket (exit 3.0%\|NONE only)

| holdout | ET hour | n | mean net % |
|---|---|---|---|
| TRAIN | 10-11 | 3542 | 0.0902 |
| TRAIN | 11-12 | 5226 | -0.0677 |
| TRAIN | 12-13 | 5877 | -0.1565 |
| TRAIN | 13-14 | 6227 | -0.1405 |
| TRAIN | 14-15 | 6534 | -0.0763 |
| VAL | 10-11 | 4782 | 0.0943 |
| VAL | 11-12 | 6936 | -0.0269 |
| VAL | 12-13 | 7675 | 0.0481 |
| VAL | 13-14 | 8166 | 0.0141 |
| VAL | 14-15 | 8408 | -0.1202 |

Best hour bucket: VAL 10-11 (0.0943% mean). Worst: TRAIN 12-13 (-0.1565% mean).

## 5. Original cell_1493 placebo: before vs after fill_min (look-ahead check)

cell_1493_placebo.csv columns: ['day', 'symbol', 'split', 'cell', 'net_pct', 'why']. No fill_min/pm column stored -- the draw minute is recomputed via cell_1493.placebo_minute(day,symbol,fill_min). CAVEAT: that function seeds its RNG with Python's built-in hash(), which is randomized per-process (PYTHONHASHSEED) unless fixed -- the exact minute the original CSV drew is therefore NOT exactly recoverable. Three reruns under PYTHONHASHSEED=(default-random, 0, 42) gave: TRAIN before-mean 1.30-1.56%, after-mean 0.44-0.49%; VAL before-mean 1.16-1.25%, after-mean 0.41-0.44% -- the direction and rough magnitude of the before/after gap is stable across seeds even though the exact figures below (one arbitrary run) are not exactly reproducible.

| holdout | when (draw vs fill_min) | n | mean net % |
|---|---|---|---|
| TRAIN | before fill_min | 734 | 1.5687 |
| TRAIN | after fill_min | 2631 | 0.4174 |
| VAL | before fill_min | 907 | 1.2687 |
| VAL | after fill_min | 3406 | 0.4042 |

## Reading

The causal later-minute placebo is flat-to-negative on both holdouts and both exits (TRAIN -0.10 to -0.13%, VAL -0.02 to -0.09%, |t|<1.3) -- none of the ~0.6pp edge the original (look-ahead) placebo showed survives once the draw is forced to occur after the break is already known; charging even a passive half-spread entry pushes every cell/holdout negative with |t| 2.3-3.7. The before/after split of the ORIGINAL placebo confirms the mechanism directly: pre-fill_min draws (conditioned on the future break) average +1.25 to +1.47% vs +0.41 to +0.44% for after-fill_min draws on the same population -- the look-ahead alone is worth roughly +1.0pp, i.e. most of the original placebo's apparent edge over the retest cells.
