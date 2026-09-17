# F6 red-to-green — INDEPENDENT REBUILD from prose

Built from the prose specification only. No file under `research/fuckup_audit/` other than
this directory was read; `research/bf_zero/build_candidates*.py`, `B/build_candidates4.py`
and `B/score5.py` were not read. Code: `prefilter.py` (daily gates), `scan.py` (tape walk +
exit walk), `book.py` (costs + 12/4 book + stats), `monthly.py`, `obtain_check.py`,
`open_bar_signal.py`. Nothing outside this directory was written; no config, service, cache
or order touched.

Window 2025-01-02 .. 2026-09-11. The universe file ends 2026-09-04, so TEST is 2026-06-01..09-04
in practice.

---

## 1. Funnel

| step | rows |
|---|---|
| universe.csv rows in window | 647,796 |
| open >= 5 | 396,210 |
| prior trading day found in the Databento panel | 394,289 (1,921 dropped) |
| prior-day range % >= 8 | 140,144 |
| — prior-day OHLC taken from universe.csv / from the panel | 254,193 / 140,096 (of 394,289) |
| 09:30 bar present and open < prior close (red open) | 67,178 symbol-days with a tape (573 had no RTH tape at all) |
| signal bar found (high >= prior_close x 1.003), primary variant | 42,522 |
| — killed by the range-so-far floor (< 5% before the signal bar) | 30,256 |
| — no next bar in the tape | 54 |
| — entry minute > 14:01 ET | 1,436 |
| — fill > level x 1.006 (the cap) | 3,068 |
| — fill price < $5 | 5 |
| — R < 1% of entry | 93 |
| candidates surviving to the book | **7,606** (417 days, 2,207 symbols) |
| booked (12/day, 4 concurrent) — hold / 2R / partial | 2,011 / 2,214 / 2,038 |

The book differs by exit rule because exit minutes free slots at different times.

Daily-file `open` vs the 09:30 bar open disagree about the red open on 790 of 140,144
symbol-days (0.56%). The 09:30 bar is used (see section 5).

---

## 2. Results — PRIMARY variant (signal searched from 09:31; stop = lowest low 09:30..signal bar inclusive)

net R is per trade, after the spread contract. `t` is on net R. WR is net R > 0. stop rate is
the share of trades whose terminal exit type is a stop. Weeks are calendar weeks containing
at least one trade.

| exit | split | n | tr/wk | mean net R | mean gross R | t | WR | stop rate | weekly R | weeks | weeks green | worst week | total net R |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hold | TRAIN | 1114 | 21.02 | **+0.0626** | +0.0925 | 1.51 | 43.8% | 29.6% | +1.32 | 53 | 58.5% | -9.96 | +69.8 |
| hold | VAL | 531 | 24.14 | **+0.2066** | +0.2388 | 2.51 | 47.3% | 32.0% | +4.99 | 22 | 77.3% | -12.85 | +109.7 |
| hold | TEST | 366 | 26.14 | **-0.0420** | -0.0082 | -0.63 | 39.6% | 32.5% | -1.10 | 14 | 42.9% | -11.28 | -15.4 |
| hold | ALL | 2011 | 22.85 | +0.0816 | +0.1128 | 2.41 | 44.0% | 30.8% | +1.87 | 88 | 60.2% | -12.85 | +164.2 |
| 2R | TRAIN | 1206 | 22.75 | **+0.0726** | +0.1037 | 2.45 | 47.0% | 25.6% | +1.65 | 53 | 60.4% | -9.70 | +87.6 |
| 2R | VAL | 600 | 27.27 | **+0.0764** | +0.1100 | 1.80 | 49.7% | 28.7% | +2.08 | 22 | 77.3% | -11.53 | +45.8 |
| 2R | TEST | 408 | 29.14 | **-0.0220** | +0.0131 | -0.42 | 42.6% | 29.2% | -0.64 | 14 | 35.7% | -7.27 | -9.0 |
| 2R | ALL | 2214 | 25.16 | +0.0562 | +0.0887 | 2.55 | 46.9% | 27.1% | +1.41 | 88 | 60.2% | -11.53 | +124.5 |
| partial | TRAIN | 1128 | 21.28 | **+0.0783** | +0.1094 | 2.30 | 47.0% | 31.1% | +1.67 | 53 | 56.6% | -9.70 | +88.3 |
| partial | VAL | 541 | 24.59 | **+0.1147** | +0.1485 | 2.04 | 48.6% | 33.6% | +2.82 | 22 | 72.7% | -12.20 | +62.1 |
| partial | TEST | 369 | 26.36 | **-0.0163** | +0.0192 | -0.28 | 42.0% | 34.4% | -0.43 | 14 | 42.9% | -7.50 | -6.0 |
| partial | ALL | 2038 | 23.16 | +0.0708 | +0.1034 | 2.70 | 46.5% | 32.4% | +1.64 | 88 | 57.9% | -12.20 | +144.3 |

TEST is negative on all three exits. Costs are 0.030-0.033 R/trade (mean `half` = 0.0366 R,
median 0.0306 R).

Per-trade CSVs (day, symbol, entry minute, entry, stop, exit minute, exit type, gross R,
net R, plus `half`, `sig_min`, `src`, leg list):
`trades_hold_ai.csv`, `trades_r2_ai.csv`, `trades_partial_ai.csv`.
Stats: `stats_ai.csv`. Alternates: `*_ax.csv`, `*_bi.csv`, `*_bx.csv`.

### Monthly net R (primary variant)

| month | hold n | hold R | 2R n | 2R R | partial n | partial R |
|---|---|---|---|---|---|---|
| 2025-01 | 91 | -4.88 | 98 | -15.89 | 93 | -8.75 |
| 2025-02 | 79 | +14.25 | 83 | +8.71 | 79 | +12.84 |
| 2025-03 | 87 | +12.75 | 91 | +2.35 | 87 | +7.85 |
| 2025-04 | 98 | +1.89 | 107 | +16.12 | 99 | +10.85 |
| 2025-05 | 90 | +3.39 | 98 | +11.41 | 91 | +9.37 |
| 2025-06 | 92 | -6.73 | 98 | +3.83 | 93 | +1.09 |
| 2025-07 | 99 | +12.66 | 107 | +15.47 | 100 | +14.98 |
| 2025-08 | 87 | +15.65 | 97 | +12.58 | 89 | +14.00 |
| 2025-09 | 89 | +18.30 | 95 | +19.54 | 90 | +16.34 |
| 2025-10 | 114 | -3.83 | 123 | -1.95 | 115 | -1.91 |
| 2025-11 | 89 | +10.84 | 100 | +14.16 | 90 | +11.71 |
| 2025-12 | 99 | -4.50 | 109 | +1.26 | 102 | -0.08 |
| 2026-01 | 103 | +13.54 | 115 | +5.67 | 105 | +2.54 |
| 2026-02 | 99 | +2.49 | 111 | +7.37 | 101 | +3.28 |
| 2026-03 | 110 | +19.33 | 127 | +1.15 | 112 | +6.82 |
| 2026-04 | 104 | +45.43 | 117 | +16.10 | 105 | +29.58 |
| 2026-05 | 115 | +28.94 | 130 | +15.54 | 118 | +19.84 |
| 2026-06 | 116 | -12.55 | 125 | -8.51 | 117 | -5.69 |
| 2026-07 | 121 | -11.11 | 136 | -3.27 | 122 | -5.26 |
| 2026-08 | 110 | +6.15 | 126 | +1.61 | 111 | +2.89 |
| 2026-09 | 19 | +2.16 | 21 | +1.20 | 19 | +2.05 |

Red months: hold 6/21, 2R 4/21, partial 5/21. VAL's +0.207 R/trade on hold is carried by
2026-04 and 2026-05 (+74.4 R of the split's +109.7).

---

## 3. The two ambiguous switches, run both ways (mean net R)

| variant | exit | TRAIN | VAL | TEST |
|---|---|---|---|---|
| a/i signal from 09:31, stop incl. signal bar (PRIMARY) | hold | +0.0626 | +0.2066 | -0.0420 |
| | 2R | +0.0726 | +0.0764 | -0.0220 |
| | partial | +0.0783 | +0.1147 | -0.0163 |
| a/x signal from 09:31, stop excl. signal bar | hold | +0.0607 | +0.1813 | -0.0420 |
| | 2R | +0.0681 | +0.0699 | -0.0220 |
| | partial | +0.0740 | +0.0982 | -0.0163 |
| b/i 09:30 bar may be the signal, stop incl. | hold | +0.1336 | +0.0750 | -0.0351 |
| | 2R | +0.1044 | +0.0540 | -0.0338 |
| | partial | +0.1323 | +0.0468 | -0.0408 |
| b/x 09:30 bar may be the signal, stop excl. | hold | +0.1336 | +0.0775 | -0.0351 |
| | 2R | +0.1044 | +0.0507 | -0.0338 |
| | partial | +0.1323 | +0.0458 | -0.0408 |

The stop window (i vs x) moves the book by <= 0.03 R. The signal-start switch (a vs b) moves
TRAIN by +0.07 R and VAL by -0.13 R, i.e. it flips which split looks strong; the two splits
disagree about the sign of the effect. Mechanically, allowing the 09:30 bar to be the signal
never ADDS a trade — the range-so-far floor has no bars to measure before a 09:30 signal, so
those days are killed outright (rows with sig at 09:30 in `scan_bi.csv`: 0). Variant b is
variant a minus 622 candidates.

---

## 4. Obtainability (`obtainability_ai.csv`, re-read from the tape)

| check | hold | 2R | partial |
|---|---|---|---|
| entry fill inside its bar (low <= fill <= high) | 2011/2011 | 2214/2214 | 2038/2038 |
| fill bar is the bar immediately after the signal bar in the tape | 2011/2011 | 2214/2214 | 2038/2038 |
| stop exits | 619 | 600 | 660 |
| — stop fill inside the exit bar | 388 (62.7%) | 376 (62.7%) | 414 (62.7%) |
| — stop fill BELOW the exit bar's low | **231 (37.3%)** | 224 (37.3%) | 246 (37.3%) |
| target legs | 0 | 314 | 306 |
| — exit bar closed >= target | — | 314/314 | 306/306 |
| — target inside the exit bar | — | 224 (71.3%) | 217 (70.9%) |
| — exit bar low ABOVE target (we sold cheaper than the market) | — | 90 | 89 |
| — exit bar high BELOW target (unobtainable) | — | **0** | **0** |
| eod exits taken at the first bar at/after 15:55 | 1392/1392 | 1300/1300 | 1378/1378 |

Direction of the two failures, both CONSERVATIVE for a long book:
- stop fill = `min(stop, bar open) x 0.999`; when the bar opens at or below its low the 0.1%
  slip pushes the fill under the bar's range, i.e. we book a worse price than was available.
  37.3% of stop exits.
- the target leg fills AT the target on a bar that gapped entirely above it in 29% of cases,
  i.e. we sell below the whole bar. Zero target fills are above anything the bar offered.

Other obtainability numbers:
- **Share of signals lost to the 0.6% cap: 3,068 / 10,772 = 28.5%** of on-time signals that
  reached the fill test (25.1% if the 1,436 post-14:01 signals are counted in the denominator).
- **Share of red-open candidate days where the 09:30 bar itself already reached the level:
  8,945 / 65,695 = 13.6%** (`open_bar_signal.py`; 3,613 of the 140,144 prefiltered days had no
  09:30 bar at all). Under the primary variant every one of those days signals later, if at
  all; under variant b every one of them is dropped.
- The fill bar is not the next clock minute for 1,097 / 7,606 = 14.4% of candidates
  (331 / 2,011 of booked hold trades) — the tape skips minutes on thin names, so "the next
  bar" is the next bar that exists, not signal + 1 on the clock.

---

## 5. Every choice made where the prose was ambiguous

1. **Red open measured on the 09:30 1-min bar**, not the daily file's `open` (the prose says
   "the day's 09:30 open"). They disagree on 0.56% of symbol-days. `prefilter.csv` (the
   daily-file version, 67,360 rows) is kept for reference; the walk uses `prefilter_all.csv`
   (140,144 rows) and decides red on the tape.
2. **Prior trading day** = the immediately preceding row for that symbol in the Databento
   panel (the only complete daily source); its OHLC is taken from `universe.csv` when that
   (symbol, prior_day) row exists there (64.5% of cases) and from the panel otherwise. 1,921
   symbol-days have no prior panel row and were dropped.
3. **Signal search starts at 09:31** (primary). Variant b run in section 3.
4. **Stop = lowest low 09:30..signal bar inclusive** (primary). Variant x run in section 3.
5. **"Entry minute <= 14:01 ET" is applied to the FILL bar**, not the signal bar.
6. **"The next bar"** = the next bar PRESENT in the RTH tape, not the next clock minute
   (14.4% of candidates differ; a resting capped limit order fills at the next print).
7. **Bar source precedence: `bars_sip.db` first, `data/cache.db` only when the (symbol, day)
   key is absent**, as the prose states. Coverage is 100% of prefiltered days (573 keys had a
   row in neither, dropped). 75% of booked trades come from `cache.db`
   (1,518 of 2,011 hold trades), so this book is mostly NOT on the SIP tape — a different
   precedence would produce a different book on the days present in both stores.
8. **"Flat at the 15:55 bar's open"** = the open of the first bar at or after 15:55 ET; if a
   tape has no such bar, the last RTH bar's close is used (no booked trade needed the fallback).
9. **The 2R target is checked AFTER the stop within the same bar** (the prose says stop first);
   the partial leg is skipped in any bar where the stop fires.
10. **E-partial stop-to-entry** is a true breakeven stop at the entry price; the remainder's
    stop fill uses the same `min(level, open) x 0.999` rule, and its exit type is recorded as
    `pp+stop` / `pp+eod` but costed as `stop` / `eod`.
11. **R% in the cost formula** = R / entry x 100. Spread S = median `spread/price x 1e4` over
    `n_q > 0` rows of the band cell, divided by 100 (a percent of price). All 25 (pb, hb)
    cells are populated (n 50-120).
12. **WR is on net R**, stop rate is on the terminal exit type, weeks are calendar weeks with
    at least one trade, weeks green is the share of those weeks with positive net R.
13. **Book slot release**: an open trade frees its slot only if its exit minute is strictly
    less than the candidate's entry minute; the per-day cap (12) counts entries and is checked
    before the concurrency cap.
14. TEST ends 2026-09-04 because the universe file does, not 09-11.

---

## 6. Files

`prefilter.py`, `prefilter.csv`, `prefilter_all.csv`, `scan.py`, `scan_{ai,ax,bi,bx}.csv`,
`scan.log`, `book.py`, `trades_{hold,r2,partial}_{ai,ax,bi,bx}.csv`,
`stats_{ai,ax,bi,bx}.csv`, `monthly.py`, `monthly_ai.csv`, `obtain_check.py`,
`obtainability_ai.csv`, `obtain.log`, `open_bar_signal.py`, `open_bar.log`.
