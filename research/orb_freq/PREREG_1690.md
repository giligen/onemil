# PREREG — cell 1,690: can ORB add-on pool P1 (idea1, gap 3–5%) be improved? (FROZEN 2026-10-01)

Owner's ask: "the new pool that is negative this quarter, can we improve it?" P1 = cell 1,684 idea1 (gap
3–5% at open, 5-min range high ≥ prev_close+5% by 09:35, price $3–30, prior-day volume ≥ 500K). In-sample
(2025-01..2026-09-18) P1 is +0.047 R (cell 1,684 FAIL verdict stands); Q3 2026 alone it is **−0.008 R / 64
fills / −$185** (`WEEKLY_P1_vs_PROD_2026Q3.md`). This cell asks whether a gate, exit, or sizing change
fixes Q3 without breaking TRAIN/VAL, using the SAME frozen reference book, never re-fit on held-out data.

## Data (read-only; no new minute-bar work, no fetch)
- **P1 base book**: `research/orb_freq/1684_pool_books.csv`, `pool=='idea1' & window=='in_regime'` — n=394,
  2025-01-02..2026-09-18 (entered fills only; columns date,symbol,entry_price,_sized_pnl,R,_composite;
  R = _sized_pnl/375). This is the reference for plain-P1 and the base for every variant's R unless a
  variant explicitly re-walks the exit (b) or rescales the size (c).
- **Variant (a) gates**: `research/orb_freq/subpools_1685/{B,C}F{1,3,4,5}_in_regime_true.csv` and
  `{B,C}F6_in_regime_true.csv` (cell 1,685, entered==1 rows only). These sub-pools admit on gap-BAND alone
  (B=[3,4)%, C=[4,5)%) × one feature, at a looser vol floor (300K) than P1 (500K) — so "P1 × F_i" is built
  as the INTERSECTION on (date,symbol) of P1's own base book with the UNION of that feature's B and C
  true-files, keeping R from the P1 base book (same production pipeline/mechanics; a symbol-day both
  chains independently chose to enter has one real fill, one real R — intersecting only keeps days both
  agreed on, so this is exact, not approximate).
- **Variants (b)/(c) bars**: `data/cache.db::intraday_bars_1min` (read-only, `?mode=ro`; indexed on
  (symbol,bar_date) — verified fast, e.g. VUZI 2025-01-02 → 444 bars). **NOT** `research/bf_zero/bars_sip.db`:
  measured coverage of P1's own (date,symbol) population in bars_sip.db is **13/40 (32.5%)** on a random
  sample — below the 80% availability rail — because idea1 was built from a separate wide-seed minute
  extraction, not the bars_sip appender. cache.db is also the PRODUCTION bar source
  (`study_orb_pipeline_static_lock.BARS_DB_PATH` default), so it is the correct store, not a fallback.
  Reused, unmodified: `research/orb_exit/1679_orb_exit.py` (`find_range_and_breakout`, `_lock_walk`,
  `_trail_walk`, constants `LOCK_TRIGGER_R_LIVE=1.75`, `LOCK_STOP_R_LIVE=0.5`, `EXIT_SLIP_BPS=10`,
  `ORB_EOD_M=945`) and `research/hod_entry/1668_failure.py` (`minute_of_day`, `et_offset_minutes`) via the
  project's `importlib` convention (digit-prefixed filenames). My own loader queries
  `intraday_bars_1min` per (symbol,bar_date) (indexed, fast) and reshapes rows into the same
  `{minarr,o,h,l,c}` dict the reused walkers expect — same ISO-UTC+offset timestamp format as bars_sip, so
  `minute_of_day` applies unchanged.

## F7 (pre-market-high break): VOID
Pre-market bars (before 09:30 ET) ARE present on the covered subset of bars_sip.db (13/13 sampled days),
but overall store coverage for P1's population is 32.5% (above) — building a new admission feature on a
<80%-covered, non-random subset (bars_sip.db was populated for other cells' purposes, not a random draw)
risks exactly the availability-gap bias CLAUDE.md's rail exists to catch. VOID per the task's own fallback
clause, stated plainly rather than silently dropped.

## Windows (fixed, no re-fit)
TRAIN 2025-01-01..2025-12-31. VAL 2026-01-01..2026-06-30. HELD-OUT 2026-07-01..2026-09-26, **read ONCE at
the end**. Data gap disclosed: the P1 base book (and every CSV built from it) runs through 2026-09-18 only
(latest existing build, same cutoff `WEEKLY_P1_vs_PROD_2026Q3.md` already discloses as "data available
through 2026-09-18, W38 partial") — 2026-09-19..09-26 is NOT built and is NOT fetched here (no new
minute-bar work, no spend). HELD-OUT below means 2026-07-01..2026-09-18 throughout; this is stated once
here and not repeated as a caveat on every number.

## Variants
**(a) P1 × one 1,685 feature gate**, for F in {F1 rel-vol≥3×, F3 above-VWAP+top-half, F4 within 5% of
52wk-high, F5 prev-day range≥1.5×ATR14, F6 day-2 of a ≥10% gapper}: P1 base rows whose (date,symbol) is in
`BF{n}_in_regime_true ∪ CF{n}_in_regime_true` (entered==1). F7 VOID (above).

**(b) P1 × exit menu**, each fully replacing the post-entry exit (R-floored at 0.5% of entry price, fills
below excluded from R-unit reads and counted separately, exactly 1,679's convention), walked on
cache.db bars with i0/range_high/range_low from `find_range_and_breakout` (09:30-09:34 range, breakout bar
= first bar in [09:35,10:35) ET with high>range_high), entry=book's own entry_price, stop=range_low,
R_unit=entry-stop, -10bps touch slippage throughout, 15:45 ET force-close:
  - `live_rule`: `_lock_walk(entry,stop,R_unit,trigger_r=1.75,lock_stop_r=0.5)` — reused unchanged; this is
    the SANITY CHECK (expected ≈ the base book's own R, since it reproduces live mechanics on an
    independently-reconstructed i0/stop) and NOT itself a candidate.
  - `scale50_1R`: 50% exits at entry+1R (touch+slip) if ever reached, the other 50% independently rides
    `live_rule`'s walk over the full path; blended 0.5×leg1+0.5×leg2 (identical construction to 1679's
    `scale50_1R_plus_live`, reused by formula).
  - `noexit2R_half3R_trail1R`: NEW function (not in 1679's menu — PREREG_1684 item 41's "no exit at 2R,
    half at +3R, trail the rest MFE−1R"), same bar-walk precedence as `_trail_walk`: stop stays at the
    ORIGINAL stop (no tightening at 2R — the "no exit" is literal, 2R is a waypoint not a trigger) until
    the first bar whose high ≥ entry+3R; at that bar 50% exits at entry+3R (touch+slip); the other 50%
    trails from then on at running-high−1R, floored at the original stop, to the stop/EOD; if 3R is never
    reached, 100% rides the original stop to stop/EOD (no partial, no trail — nothing in the spec triggers
    one). Blended R = 0.5×leg1+0.5×leg2 when the target fires, else the single full-size outcome.

**(c) P1 × cost-aware sizing**: range_pct = (range_high−range_low)/entry_price×100 from the SAME
reconstruction as (b). For fills with range_pct < 0.75: a full-size $375-risk trade's round-trip cost in R
is 13bps / (range_pct/100) (cost and position size cancel in the R ratio, so "sizing down" must be read as
capping DOLLAR exposure, not as a free lunch); solve for the size factor s=min(1, 0.10×range_pct/100 /
0.0013) that caps the cost contribution at 0.10R of the STANDARD $375 basis; new_R = s×R_base, new_$ =
s×R_base×375 ("$ at the live sizing" = this reduced $). Fills with range_pct ≥ 0.75 are unchanged (s=1).
This changes magnitude only, not which fills are in the book (same n as plain P1).

## Scoring (reused from `research/orb_freq/1684_score.py::stats`, not re-derived)
Per variant × window: n, fills/week, mean R, day-clustered t (one-sample t on the per-day mean R — days,
not trades, are the clustering unit), ex-top-5%, weekly P10, worst week. Plain P1 read on all three windows
as the reference row.

## Selection rule (fixed before any number is read)
A variant is a **candidate** only if mean R ≥ +0.05 **and** day-clustered t ≥ 1.5 on **BOTH** TRAIN and
VAL. Candidates are then read ONCE on HELD-OUT (mean R, t, fills/week, green weeks out of n weeks, worst
week). **Winner** = the candidate with the best HELD-OUT mean R that is ≥ 0. If no variant is a candidate,
or every candidate's HELD-OUT mean R is < 0: **"no improvement"**. `live_rule` (b's sanity check) is never
eligible as a candidate regardless of its numbers — it is not a change.

## Multiplicity
5 (a, F1/F3/F4/F5/F6) + 3 (b, incl. the live_rule sanity check which is not a candidate) + 1 (c) = 9 variant
reads × 3 windows = 27 cells, plus plain P1 × 3 windows as the reference. No stage 2, no tuning of the 0.05/
1.5/0.75%/0.10R/3R/1R constants — every threshold above is fixed by this PREREG or inherited unchanged from
1679/1684/1685.

## Constraints (restated, binding)
≤55 tool calls this session; `nice -n 10`, one process, no parallel runs; never Read any file >300 lines in
full; config.yaml/orb.yaml/the live service/trading/*.py/data/*.db untouched (data/*.db and the bar store
read via `?mode=ro` only — reading/importing trading/*.py and study_orb_pipeline_static_lock.py for shared
formulas is the established project convention, writing to them is not done); no git commit; no spend (no
Databento/API calls — every source above already exists on disk).

## Output
`PREREG_1690.md` (this file), `RESULT_1690.md` (≤100 lines: variant table TRAIN/VAL first, then HELD-OUT
for candidates), `1690_reads.csv` (every variant×window row), `1690_variants.py`, `1690_variants.log`.
Agent returns ≤150 words: candidates with TRAIN/VAL numbers, their HELD-OUT read, the winner or "no
improvement", file names.
