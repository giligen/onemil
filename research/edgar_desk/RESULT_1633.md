# RESULT — cells 1,633-1,636: post-earnings drift on the 8-K item-2.02 reaction

Spec: `research/edgar_desk/PREREG_1633.md` (FROZEN 2026-09-28 18:40 UTC). Builder run: this script, `research/edgar_desk/cell_1633.py`.

## Pipeline counts
- item-2.02 8-K (filing,symbol) rows after parse/test-ticker filters: 128645
- resolved against the price panel (has a symbol match + full index bounds), TRAIN+VAL only --
  TEST (reaction_session >= 2024-07-01, 37614 rows) is already sealed out of this count: 82134
- pass universe filter (price >= $3, 20-day $ volume >= $1M at the session strictly before the announcement date): 58863 (71.7%)
- TRAIN decile count realized: 10 (10 requested; fewer means the TRAIN R0 distribution had duplicate qcut edges)

## Timezone conversion (departure from an earlier cell-1,552 convention)
`acceptance_datetime` carries an ISO 'Z' (UTC) suffix. This script converts it to America/New_York with `zoneinfo` (DST-aware), per PREREG_1633's explicit instruction: "acceptance datetime UTC ... convert properly." `research/edgar_desk/rebuild_1552_full.py` instead stripped the 'Z' and treated the raw digits as already-ET wall-clock (its own comment: "tz_localize(None) drops the (spurious) UTC label without shifting the clock"). The two conventions disagree by 4-5 hours (the DST offset) on every event, which can flip an event between the pre-market/intraday/after-close buckets and therefore change its reaction_session. **This is the single highest-priority item for the independent-check agent to verify against the raw SEC EDGAR convention directly** (SEC's own acceptance-datetime is documented as Eastern time in the submissions header; if that is also true of this vendor's export despite the 'Z' suffix, this script's conversion — not cell 1,552's — would be the bug).

## Cell stats (n, events/week, mean net bps, day-clustered t [entry session], ex-top-5%/ex-top-1% bps, winner-capped [+30%] bps, SPY-adjusted raw/net bps, per-year sign)

|   cell | split   |    n |   events_wk |   mean_net_bps |     t |   ex_top5_bps |   ex_top1_bps |   winner_capped_bps |   spy_adj_bps |   spy_adj_net_bps | years_positive              |
|-------:|:--------|-----:|------------:|---------------:|------:|--------------:|--------------:|--------------------:|--------------:|------------------:|:----------------------------|
|   1633 | TRAIN   | 4156 |       21.1  |           19.7 |  0.5  |        -155.2 |         -37.4 |                -1.6 |          -9.6 |             -19.6 | 2019:-/2020:+/2021:-/2022:+ |
|   1633 | VAL     | 2093 |       27.18 |            9.1 |  0.19 |        -127.9 |         -33.6 |                 6.7 |        -109.2 |            -119.2 | 2023:-/2024:+               |
|   1634 | TRAIN   | 4153 |       21.08 |           96.3 |  1.39 |        -163.8 |           7.1 |                 4.1 |           8.6 |              -1.4 | 2019:-/2020:+/2021:-/2022:+ |
|   1634 | VAL     | 2089 |       27.13 |           85.3 |  1.28 |        -112.1 |          19.3 |                47.4 |        -143.4 |            -153.4 | 2023:+/2024:+               |
|   1635 | TRAIN   |  893 |        5.58 |          -13.2 | -0.24 |        -143.5 |         -52.6 |               -31.8 |          23.4 |               2.6 | 2019:+/2020:+/2021:-/2022:- |
|   1635 | VAL     |  425 |        6.75 |          -34.1 | -0.57 |        -128   |         -57.8 |                -8.2 |         114.6 |              93.9 | 2023:-/2024:-               |

## Pass-bar checklist (frozen; VAL, per cell)
Mean net >= +50bps | day-clustered t >= 2.5 | ex-top-5% > 0 | TRAIN same-sign t >= 1 | decile table monotone both halves | SPY-adjusted (net) >= +30bps

- Decile-table monotonicity (net bps, non-decreasing decile 1->10): +10 TRAIN=NOT MONO, +10 VAL=NOT MONO, +20 TRAIN=NOT MONO, +20 VAL=NOT MONO

### Cell 1633 (VAL n=2093, 27.18 events/wk): FAIL
  - mean_net>=+50bps: FAIL
  - t>=2.5: FAIL
  - ex_top5>0: FAIL
  - TRAIN_same_sign_t>=1: FAIL
  - decile_monotone_both_halves: FAIL
  - spy_adj_net>=+30bps: FAIL

### Cell 1634 (VAL n=2089, 27.13 events/wk): FAIL
  - mean_net>=+50bps: PASS
  - t>=2.5: FAIL
  - ex_top5>0: FAIL
  - TRAIN_same_sign_t>=1: PASS
  - decile_monotone_both_halves: FAIL
  - spy_adj_net>=+30bps: FAIL

### Cell 1635 (VAL n=425, 6.75 events/wk): FAIL
  - mean_net>=+50bps: FAIL
  - t>=2.5: FAIL
  - ex_top5>0: FAIL
  - TRAIN_same_sign_t>=1: FAIL
  - decile_monotone_both_halves: FAIL
  - spy_adj_net>=+30bps: PASS

## 1,636 full decile table (report-only; TRAIN-defined edges applied to both halves)

|   hold | split   |   decile |    n |   mean_ret_bps |   mean_net_bps |   spy_adj_bps |
|-------:|:--------|---------:|-----:|---------------:|---------------:|--------------:|
|    +10 | TRAIN   |        1 | 4171 |          -60.9 |          -70.9 |         -93.6 |
|    +10 | TRAIN   |        2 | 4179 |           16.8 |            6.8 |          -6.7 |
|    +10 | TRAIN   |        3 | 4179 |           13.9 |            3.9 |         -22.7 |
|    +10 | TRAIN   |        4 | 4179 |            0.8 |           -9.2 |         -32.6 |
|    +10 | TRAIN   |        5 | 4180 |           -0.5 |          -10.5 |         -35.6 |
|    +10 | TRAIN   |        6 | 4176 |           16.3 |            6.3 |         -21.9 |
|    +10 | TRAIN   |        7 | 4179 |           45.8 |           35.8 |           0.9 |
|    +10 | TRAIN   |        8 | 4181 |           59   |           49   |          15.7 |
|    +10 | TRAIN   |        9 | 4179 |           14.1 |            4.1 |         -21.2 |
|    +10 | TRAIN   |       10 | 4156 |           29.7 |           19.7 |          -9.6 |
|    +10 | VAL     |        1 | 2106 |          -33.5 |          -43.5 |        -153.4 |
|    +10 | VAL     |        2 | 1764 |          -45.3 |          -55.3 |        -159.9 |
|    +10 | VAL     |        3 | 1688 |          -23.2 |          -33.2 |        -134.7 |
|    +10 | VAL     |        4 | 1577 |            7.1 |           -2.9 |        -120.4 |
|    +10 | VAL     |        5 | 1512 |          -17.7 |          -27.7 |        -120.1 |
|    +10 | VAL     |        6 | 1508 |          -40.5 |          -50.5 |        -150.3 |
|    +10 | VAL     |        7 | 1478 |          -10.1 |          -20.1 |        -137.5 |
|    +10 | VAL     |        8 | 1580 |            5.7 |           -4.3 |        -116.5 |
|    +10 | VAL     |        9 | 1727 |            9.2 |           -0.8 |        -110.7 |
|    +10 | VAL     |       10 | 2093 |           19.1 |            9.1 |        -109.2 |
|    +20 | TRAIN   |        1 | 4171 |          -69.6 |          -79.6 |        -138.1 |
|    +20 | TRAIN   |        2 | 4179 |           79.1 |           69.1 |          -1.7 |
|    +20 | TRAIN   |        3 | 4177 |           50   |           40   |         -29.6 |
|    +20 | TRAIN   |        4 | 4176 |           39   |           29   |         -39.1 |
|    +20 | TRAIN   |        5 | 4176 |           38.6 |           28.6 |         -45.9 |
|    +20 | TRAIN   |        6 | 4174 |           55.1 |           45.1 |         -17.8 |
|    +20 | TRAIN   |        7 | 4178 |          114   |          104   |          24.3 |
|    +20 | TRAIN   |        8 | 4179 |          100.2 |           90.2 |           3.2 |
|    +20 | TRAIN   |        9 | 4177 |           99   |           89   |         -11.5 |
|    +20 | TRAIN   |       10 | 4153 |          106.3 |           96.3 |           8.6 |
|    +20 | VAL     |        1 | 2105 |          -57.9 |          -67.9 |        -287.2 |
|    +20 | VAL     |        2 | 1764 |          -22   |          -32   |        -243.3 |
|    +20 | VAL     |        3 | 1688 |           22.3 |           12.3 |        -202.5 |
|    +20 | VAL     |        4 | 1577 |           31.6 |           21.6 |        -215.4 |
|    +20 | VAL     |        5 | 1509 |          -27.8 |          -37.8 |        -241.2 |
|    +20 | VAL     |        6 | 1508 |          -27.4 |          -37.4 |        -247.8 |
|    +20 | VAL     |        7 | 1478 |           22.2 |           12.2 |        -214.8 |
|    +20 | VAL     |        8 | 1580 |           28.6 |           18.6 |        -203.8 |
|    +20 | VAL     |        9 | 1727 |           61.4 |           51.4 |        -164.1 |
|    +20 | VAL     |       10 | 2089 |           95.3 |           85.3 |        -143.4 |

## Short eligibility (cell 1,635, bottom decile population)
- bottom-decile rows (pre shortability/SSR filter, universe_ok only): 6301
- excluded: not shortable / not easy-to-borrow per borrow_flags.csv (CURRENT SNAPSHOT, no date column): 492
- absent from borrow_flags.csv entirely -> KEPT UNFILTERED per the task spec: 745
- borrow_flags.csv total coverage: 14355 symbols
- **Caveat**: borrow_flags.csv has no date column -- it is a present-day (~2026-09-18) snapshot of shortability, applied retroactively to 2019-2024 short entries. A name that is easy to borrow today may not have been in 2020, and vice versa; this cannot be corrected without a historical borrow-flag source. The SSR exclusion is itself a proxy (prior-session close < $5 OR R0 <= -10%, the research/edgar_desk/rebuild_1552_full.py convention), not the real intraday-triggered SSR rule (which needs intraday data not fetched here).

## Small-cap (<=$1B) vs larger split -- cell 1,636
**UNAVAILABLE.** No shares-outstanding or market-cap source exists on disk for this task's inputs (checked: no market_cap/shares_outstanding file under `research/` or `data/research/`). Fabricating a market-cap proxy from price or dollar volume would misrepresent size and was not done. The full decile table above is unsplit by size; this is a data gap, not a finding, and should be filled by fetching a shares-outstanding source (e.g. an EDGAR company-facts pull) before this line item can be reported.

## Split-artifact guard (unadjusted-split defense)
Daily bars here are RAW (unadjusted). Any event whose price path from the prior-reaction session through the exit session contains a session-to-session close ratio outside [0.4, 2.5] is excluded from scoring (guard10_ok / guard20_ok = False) rather than left in as a fabricated extreme return. Excluded by this guard (universe_ok rows): +10 hold 71, +20 hold 98.

## Independent-check flags for the next agent
1. **Timezone conversion** (see section above) -- the single most consequential methodological choice in this cell; verify against raw SEC EDGAR behavior.
2. **'Prior session' for the universe filter** was read literally as *the session strictly before the announcement's own calendar day* (date_et - 1 session), uniformly across after-close/pre-market/intraday. An equally defensible reading uses date_et's own close for after-close filings (one session later, still fully causal) -- rebuild independently and compare the event set under both readings.
3. **Small-cap split is unavailable** (see above) -- confirm no market-cap source was missed before reporting this gap as final.
4. **Borrow/SSR filtering is a proxy** applied with a present-day snapshot; confirm the SSR proxy formula and its constants against rebuild_1552_full.py directly rather than trusting this script's transcription.
5. Rebuild the event set independently from `events_raw.csv` prose (Jaccard >= 0.98 on (symbol, reaction_session) pairs) and R0/net_bps within 2 bps, per the CLAUDE.md independent-check protocol, BEFORE this result is shown to the owner.

## Judge (main session, 2026-09-28 19:00 UTC) — FAIL; post-earnings drift on the reaction is not there for this universe
* Long top decile: +10 sessions VAL +9 bps (t 0.2), +20 sessions +85 bps (t 1.3) — both tail-carried (ex-top-5 %
  −128 / −112 bps), SPY-adjusted negative (−109 / −143 bps); TRAIN the same shape (+20 / +96 bps, t 0.5 / 1.4,
  ex-top-5 % ≈ −160). Short bottom decile: −34 bps VAL. The decile table is not monotone in any split or hold: the
  mechanism's staircase (bigger reaction → bigger drift) is absent. Rebuild agrees (VAL +9.0 bps t 0.19; +74 bps t 1.1
  at +20; Jaccard on the event set high; its decile table non-monotone too). Timezone handled properly this time
  (UTC → ET with DST); market-cap split unavailable (no shares data on disk).
Consequence per PREREG: PEAD closes on this population with the decile table on record; the literature's post-2010
decay is what we see. Programme count 1,636.
