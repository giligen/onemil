# PREREG 1,772 — the OVERNIGHT leg of the attention gappers: buy the closing auction (MOC) on gap day D, sell the opening auction (MOO) on D+1

FROZEN 2026-11-14 (Saturday) before any number. Programme cell count 1,771 → 1,772. Rules: write only
`research/signal_search/1772_*` and `RESULT_1772.md`; no git / service / orders / network except NOTHING (all data is on
disk); read-only DB URIs (`file:...?mode=ro`); `ulimit -v 2500000; nice -n 19`; budget ≤ 50 calls. Saturday: no
RTH blackout applies.

## Why this cell (mechanism first)
Every intraday long on this population is gross ≈ 0 and dies on the spread (2,400+ cells). Two facts are solid:
(1) ≥ 10 % gappers FADE −185 bp open-to-close, 8/8 years (1,754/1,767); (2) the literature's tug-of-war
(Lou–Polk–Skouras 2019; Aboody–Even-Tov–Lehavy–Trueman 2018; Bogousslavsky 2021): retail-attention names earn
NEGATIVE intraday and POSITIVE, short-run-persistent OVERNIGHT returns — the gap itself is one overnight return.
The untested leg is the NEXT overnight. Auction fills carry no quoted spread (CLAUDE.md rule 4), so this is the
only entry on this population whose cost is not the wall. Prior related cells: 1,754 (MOO→MOC long, KILL), 1,757 F2
(overnight on HOD arms, VOID by coverage) — neither is this read.

## Data (all on disk)
* Population `research/hod_entry/cell_1428_population.csv` (7,487 PIT gapper symbol-days; `split` TRAIN = 2025, VAL =
  2026; report the gap distribution from daily bars so the reader knows what "gapper" means here).
* `data/cache.db` table `daily_bars(symbol, bar_date, open, high, low, close, volume)` — day-D close and D+1 open
  (next TRADING day: the next `bar_date` present for SPY, not BDay — holidays). Sampled coverage of D+1 = 94.5 %.
* `research/hod_entry/1762_cache/bars/<day>.parquet` — day-D minute bars (16:00 ET bar = closing-print volume →
  capacity; 09:30 bar volume on D+1 is NOT on disk: use 2 % of D+1 daily volume as the MOO capacity proxy, say so).
* SPY same-night overnight from `daily_bars` (symbol SPY) for excess; if SPY is absent there, from
  `research/index_orb/cache/SPY_1min.parquet` (15:59 close → next 09:30 open).

## Definitions
r_on = open(D+1) / close(D) − 1 (bp). excess = r_on − SPY r_on same night. Fees: 0 commission; SEC + TAF on the
sell ≈ 0.3 bp; report net at auction-haircut 0 / 10 / 25 bp per side (sensitivity, NOT a band cost — auction fills
have no quoted spread). Capacity rail: $150 position ≤ 5 % of the 16:00-bar dollar volume on D AND ≤ 2 % of D+1 daily
dollar volume; names failing it are reported but excluded from the executable book.
Availability rail: D+1 open coverage ≥ 80 % overall and winner/loser missingness gap ≤ 5 pp (winner = day-D
open-to-close > 0), else VOID.

## Reads (multiplicity declared: 1 primary + 6 secondary)
PRIMARY: mean excess r_on (bp/name-night) on VAL, night-clustered t (every name on the same night shares the
night), with ex-top-5 %, winner-capped at +50 % per name, worst-1 % and the weekly P10 of an equal-$ book.
Secondary (TRAIN first, then VAL, same sign required): (a) day-D fade cohort (close < open) vs held cohort (close ≥
open); (b) price tier $1–5 / 5–20 / > 20; (c) close location in the day range (top / middle / bottom third);
(d) by gap size tercile; (e) TRAIN halves H1/H2 sign; (f) the short mirror (report-only, not executable: HTB).
PLACEBOS (binding lesson 11/13): (i) count-matched random symbol-nights from the whole `daily_bars` universe with
close ≥ $1 and $vol ≥ $1 M on the SAME nights — 200 draws, report p95 of the placebo mean; (ii) same-name placebo:
the same symbol's overnight 5 trading days later (unconditional name overnight). The primary must beat BOTH p95s.
Index frame (FACT, no pass bar): SPY and QQQ close→open vs open→close 2016–2026 from the 1-min parquets, per year
(mean bp/day, Sharpe) — states what the plain overnight premium is so the gapper number is read against it.

## Pass bar (pre-committed)
VAL primary excess ≥ +30 bp/name-night net at the 10 bp haircut, night-t ≥ 2, same sign in TRAIN H1 and H2,
ex-top-5 % > 0, beats both placebo p95s, availability rail met, executable (capacity) share ≥ 60 %, median ≥ 3
names/night. PASS → paper instrument spec: MOC buy / MOO sell on the paper HOD account, $150/name, cap 10 names,
scored on the cadence bar (`docs/cadence_bar.md`) from the weekly series — never live from the cell.
FAIL → FACT recorded; CHILDREN ≥ 2 (e.g. D+1 intraday continuation of the fade for the owner list; week-ahead
overnight persistence; the opposite leg on the held cohort).

## Output
`1772_score.py` (one file, docstrings, verbose progress), `1772_events.parquet` (one row per symbol-night with every
field used), `1772_summary.csv`, `1772_placebo.csv`, `1772_index_frame.csv`, `RESULT_1772.md` ≤ 350 words ending with
"CHILDREN:". Return ≤ 150 words: coverage, n TRAIN/VAL, primary (mean, night-t, ex-top-5 %, capped), both placebo p95,
cohort (a) split, index frame one line, PASS/FAIL, children. An independent rebuild from this prose follows.
