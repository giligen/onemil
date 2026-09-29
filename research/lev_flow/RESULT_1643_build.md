# RESULT 1,643-1,645: leveraged-ETF rebalancing flow into the close

Data: 13,840,850 minute bars, 24,192 daily sessions across 9 symbols (SMH:1,185,374, QQQ:2,064,414, IWM:1,738,524, XLF:1,318,881, XLE:1,306,982, GDX:1,464,495, XBI:1,163,230, TLT:1,458,173, SPY:2,140,777). 21 early-close sessions excluded (data-driven fallback, no pandas_market_calendars on this node). TEST 2024-01..2026-09 SEALED: 0 events computed.

## Cell 1643: 1643 LONG (up days)
- **1643 LONG (up days) TRAIN**: n=2187 (847 day-clusters), 8.41 ev/wk, mean +0.81 bps, clustered t=0.14, MDE=4.13 bps, ex-top5% -6.16, ex-top1% -1.94, winner-capped -3.53, worst day 2020-03-16 (-1191.3), SPY-only +2.31, FOMC-excl +1.00
- **1643 LONG (up days) VAL**: n=1630 (549 day-clusters), 10.50 ev/wk, mean -3.15 bps, clustered t=-2.56, MDE=2.04 bps, ex-top5% -7.56, ex-top1% -4.46, winner-capped -4.67, worst day 2023-03-22 (-160.4), SPY-only -3.27, FOMC-excl -3.03
  - VAL |r| tercile: T1_low: -3.5bps (n=543) | T2_mid: -2.0bps (n=543) | T3_high: -4.0bps (n=544)
  - TRAIN |r| tercile: T1_low: +0.8bps (n=729) | T2_mid: +0.2bps (n=729) | T3_high: +1.4bps (n=729)
  - VAL per-underlying: GDX:-8.0(210), IWM:-2.2(183), QQQ:-2.6(174), SMH:-1.5(219), SPY:-3.3(117), TLT:+5.5(119), XBI:+2.1(224), XLE:-8.6(234), XLF:-6.8(150)
  - VAL per-year: 2021:-5.6(500), 2022:+0.3(630), 2023:-5.0(500)
  - **Pass bar (FAIL)**:
    - [ ] mean net >= +5 bps (VAL): -3.1509539784274003
    - [ ] day-clustered t >= 2.5 (VAL): -2.55715583148471
    - [ ] ex-top-5% > 0 (VAL): -7.559775932868145
    - [x] >= 3 events/week pooled (VAL): 10.496780128794848
    - [ ] TRAIN same sign, |t| >= 1: 0.1421573148957154
    - [ ] |r| tercile table monotone, both halves: TRAIN mono=False, VAL mono=False
    - [ ] positive in >= 5/9 underlyings (VAL): 2

## Cell 1645: 1645 SHORT mirror (down days)
- **1645 SHORT mirror (down days) TRAIN**: n=1805 (740 day-clusters), 6.93 ev/wk, mean +2.76 bps, clustered t=0.57, MDE=3.73 bps, ex-top5% -5.04, ex-top1% +0.04, winner-capped -3.51, worst day 2020-03-18 (-268.0), SPY-only -1.83, FOMC-excl +2.21
- **1645 SHORT mirror (down days) VAL**: n=1563 (523 day-clusters), 10.05 ev/wk, mean -1.61 bps, clustered t=-1.19, MDE=2.47 bps, ex-top5% -5.98, ex-top1% -2.88, winner-capped -3.26, worst day 2022-05-20 (-133.3), SPY-only -1.25, FOMC-excl -2.23
  - VAL |r| tercile: T1_low: -2.4bps (n=521) | T2_mid: -0.2bps (n=521) | T3_high: -2.3bps (n=521)
  - TRAIN |r| tercile: T1_low: +3.9bps (n=602) | T2_mid: +4.0bps (n=601) | T3_high: +0.4bps (n=602)
  - VAL per-underlying: GDX:-7.9(226), IWM:-0.8(177), QQQ:+0.0(153), SMH:-0.1(211), SPY:-1.3(95), TLT:-0.8(141), XBI:+5.0(244), XLE:-4.8(181), XLF:-5.1(135)
  - VAL per-year: 2021:+1.6(418), 2022:-3.1(689), 2023:-2.3(456)
  - **Pass bar (FAIL)**:
    - [ ] mean net >= +5 bps (VAL): -1.6131550249367488
    - [ ] day-clustered t >= 2.5 (VAL): -1.1860976812827044
    - [ ] ex-top-5% > 0 (VAL): -5.981717844726466
    - [x] >= 3 events/week pooled (VAL): 10.046831955922864
    - [ ] TRAIN same sign, |t| >= 1: 0.574753681140057
    - [ ] |r| tercile table monotone, both halves: TRAIN mono=False, VAL mono=False
    - [ ] positive in >= 5/9 underlyings (VAL): 2

## Cell 1644 (report-only, not scored against the pass bar, whole |r| >= 1% population)
- **1644a hold-to-next-open (reversal read) TRAIN**: n=3992 (1122 day-clusters), 15.34 ev/wk, mean +4.89 bps, clustered t=1.73, MDE=5.77 bps, ex-top5% -14.21, ex-top1% -1.87, winner-capped -24.06, worst day 2020-03-12 (-499.8), SPY-only -9.64, FOMC-excl +4.54
- **1644a hold-to-next-open (reversal read) VAL**: n=3193 (728 day-clusters), 20.52 ev/wk, mean -2.31 bps, clustered t=-0.62, MDE=4.90 bps, ex-top5% -15.49, ex-top1% -5.95, winner-capped -23.46, worst day 2022-11-09 (-405.0), SPY-only -5.10, FOMC-excl -3.13
- **1644b 15:45 entry variant TRAIN**: n=3992 (1122 day-clusters), 15.34 ev/wk, mean +0.80 bps, clustered t=0.87, MDE=2.43 bps, ex-top5% -4.94, ex-top1% -1.37, winner-capped -2.27, worst day 2020-03-18 (-257.4), SPY-only -0.77, FOMC-excl +0.72
- **1644b 15:45 entry variant VAL**: n=3193 (728 day-clusters), 20.52 ev/wk, mean -0.41 bps, clustered t=-0.65, MDE=1.25 bps, ex-top5% -4.01, ex-top1% -1.43, winner-capped -1.32, worst day 2022-12-30 (-60.8), SPY-only -2.25, FOMC-excl -0.62

## Verdict
- 1643 (long): FAIL
- 1645 (short mirror): FAIL

## Caveats (read as an adversary before relaying)
- FOMC 2016-2023 dates are hard-coded from memory of the published Fed calendar (https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm), NOT machine-fetched this run -- verify against the source before this line is load-bearing for anything else.
- Cell 1644a next-open exit cost (half-spread) is an ASSUMPTION: the PREREG gives a cost for the 15:31/15:45 entries and the same-day MOC exit only, not for a next-open exit.
- Early-close list is a data-driven fallback (>=6/9 symbols with 13:00-16:00 ET volume share < 0.5x own median), not the NYSE calendar package (not installed on this node). The PREREG's literal "last bar before 15:59 ET" test fires on ZERO sessions on this full-SIP extended-hours data (verified) and was replaced by this volume-based test; the recovered dates match the known NYSE early-close calendar exactly (Thanksgiving Friday every year, July 3 / Dec 24 when those are trading days) -- see build_early_close_set().
- 1644 pools both directions (long legs on up days, short legs on down days) into one signed net-bps series per the "same entry"/"same population" reading of the PREREG; it is report-only so no pass bar was applied regardless.
- Tercile monotonicity is checked as non-decreasing T1<=T2<=T3, not strictly increasing.
