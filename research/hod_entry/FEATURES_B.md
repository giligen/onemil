# FEATURES_B — Feature Set B ("the crowd") for PREREG_1478 cell 1,478

Owner framing (PREREG_1478.md, frozen 2026-09-26): *"How can you predict the lookahead you had
last night that led to the 0.3 R. Be creative."* Rounds 1–2 used single-symbol features; this set
zooms OUT — market breadth, sector co-movement and SPY, i.e. whether the CROWD (not just the one
name) was already moving before the break. Built by `research/hod_entry/build_features_B.py`,
run start-to-finish 2026-09-26 11:33:32 → 12:06:33 UTC (1,981 s, single process, `nice -n 19`).
Output: `research/hod_entry/features_1478_B.csv`, 9,911 rows = every `status=='fill'` row of
`causal_arming_causal.csv` (TRAIN-H2 4,398 / VAL 5,513), joined on (day, symbol, fill_min, split).

## Arm-minute convention
`arm_minute = int(fill_min) - 1` for every feature below (PREREG_1478 wording, literally
"fill_min − 1"). Cell 1,445/1,457's own arm bar is `floor(fill_min)` (`cell_1445.py
arm_bar_features`: bars with `m < fill_min`, last one kept) — so every crowd/SPY value here is
read off a bar that closed a full minute *before* that arm bar even opens. This is strictly more
conservative than the base book's own causal cutoff; 0/9,911 fills had an out-of-range arm minute
(all in [570, 959]).

## Universe used for breadth/sector/pool (per day)
`research/bf_zero/universe.csv` rows for that `bar_date`, `adv20 >= 100,000`, test tickers
(`^Z[A-Z]ZZT$`) excluded — same read pattern as `causal_arming.load_population` minus its
pattern-specific filters (`min_dist_open_pct`, `min_price`), per the task's instruction to build
the general HOD universe, not the narrower armed-crossing population. Mean 1,428 names/day
(min 478, max 2,418; `n_universe_j` column, one row per fill = the breadth denominator at that
fill's own arm minute).

## Feature-by-feature

### 1. `breadth_count_j`, `breadth_share_j` — universe breadth at the arm minute
**Mechanism**: for every name in the day's HOD universe, forward-fill its 1-min close from
`causal_arming.load_day_bars` (cache.db `intraday_bars_1min` / `bars_sip.db`, SIP-preferred on a
tie — the base book's own bar source) onto the full RTH minute grid [570, 959], and mark it "up"
at minute m if `close_ffill[m] >= 1.05 * open` (`open` from `universe.csv`, the official session
open, known at 09:30). `breadth_count_j` = count of "up" names at `arm_minute`; `breadth_share_j`
= that count / count of names with ANY bar at or before `arm_minute` (the denominator, so a name
with only sparse coverage doesn't silently count as "not up").
**Timestamp proof**: `ffill` only propagates a value forward from bars with `m' <= m` — the value
at column `arm_minute` is a function of bars with `m' <= arm_minute` only, never a later bar.
Cached in `breadth_cache/<day>_breadth.csv` (m, count, denom, share) and
`breadth_cache/<day>_matrix.npz` (symbols, is5, have, opens) so the whole crowd matrix is
inspectable per day without re-querying either database.
**Coverage**: 100.000% (9,911/9,911) — every fill's own day was itself built (fill days are
always in the "needed" day set). Range: share 1.3%–46.5%, mean 11.7%.

### 2. `breadth_5d` — the regime (5-session average of 11:00 ET breadth)
**Mechanism**: for the fill's day `d`, take the trading-day calendar from `universe.csv`'s own
distinct `bar_date` values (420 days, 2025-01-02..2026-09-04), find the FIVE sessions strictly
before `d` (`trading_days[i-5 .. i-1]`, `i` = `d`'s index), and average each one's
`breadth_share` at minute 660 (11:00 ET) from that day's own cached breadth series.
**Timestamp proof**: only sessions with calendar index `< i` are used; the 11:00 value of a prior
CLOSED session is known in full well before today's open, let alone before today's arm bar.
**Coverage**: 100.000% — every fill day had 5 valid prior sessions in range. Values 6.2%–19.2%,
mean 11.2% (close to the instantaneous breadth mean, as expected of a slow-moving regime average).

### 3. `sector_peers_j`, `sector_peers_pool_j` — SIC-2 sector co-movement at the arm minute
**Mechanism**: `sic2` per symbol from `research/multiday/data/panel_f3f4.npz` (`symbols`/`sic2`
arrays, 4,871 symbols, loaded once via `numpy.load`). For a fill's symbol with a known `sic2`,
`sector_peers_j` = count of OTHER names in that day's HOD universe with the same `sic2` that are
"up" (same 5%-above-open test) at `arm_minute`; `sector_peers_pool_j` = count of same-sic2 names
with ANY data at that minute (the denominator/pool size, reported for context). The fill's own
symbol is excluded from its own peer count.
**Timestamp proof**: identical `is5`/`have` matrix as breadth, so the same ffill-only-uses-the-
past argument applies; `sic2` is a static reference-panel lookup (not a per-day observation), so
it carries no forward-looking information by construction.
**Coverage**: 55.908% (5,541/9,911). Cause verified directly: the other 44.1% (4,370 fills, 515 of
1,625 distinct fill symbols) are symbols absent from the 4,871-symbol panel — e.g. leveraged/
inverse ETFs (`LABU`, `NAIL`), 2x single-stock wrappers (`CONL`, `HOOX`), SPACs/small caps
(`ICHR`, `MRAL`, `PGY`) — reported NaN per the PREREG's explicit rule ("NaN if the symbol is
absent"), never imputed. Among covered fills: 0–161 peers (mean 11.0) out of a pool of 0–319
same-sector names (mean 82.6).

### 4. `spy_ret_open_to_j` — SPY return, session open to the arm minute
**Mechanism**: SPY's own 1-min bars for that day from `data/cache.db intraday_bars_1min` ONLY (no
`bars_sip.db` fallback, per the task's instruction), RTH-filtered and forward-filled the same way
as every other name; `open` from `universe.csv`'s SPY row for that day (fallback: the first
available intraday bar's open, if `universe.csv` lacked the row). Return = `close_ffill[arm_minute]
/ open - 1`.
**Timestamp proof**: same ffill argument; `open` is fixed at 09:30, well before any arm minute.
**Coverage**: 70.013% (6,939/9,911). Cause verified directly, not assumed: cache.db has ZERO SPY
rows at all on 51 of the 230 fill days (e.g. 2025-11-20/21/24, 2026-03-23..31 — full missing
weeks), and 100% of the NaN fills fall on exactly those 51 zero-SPY-bar days (checked by set
membership, not inferred). This is a data gap in cache.db's SPY coverage, not a logic bug — the
task specified cache.db only for this feature, so no `bars_sip.db` fallback was added; flagged
here as a genuine ~30% missingness gap the model must handle via HistGradientBoostingClassifier's
native NaN support (per PREREG_1478's model spec), not by imputation. Range (where present):
−2.0% to +1.5%, mean +0.13%.

### 5. `spy_ret_5d` — SPY's own 5-session return (regime)
**Mechanism**: `data/research/databento/equs_daily_2025_2026.parquet`, rows where `symbol=='SPY'`
(direct symbol column, not requiring the instrument-id map — verified populated for every SPY row
in range), sorted by `bar_date`; for fill day `d`, take all parquet dates strictly `< d`, and
compute `close[-1] / close[-6] - 1` (yesterday's close vs. the close 6 sessions back = a trailing
5-session return, fully realized before `d` opens).
**Timestamp proof**: only dates `< d` are ever indexed; the daily parquet is PIT by construction
(reference `reference_pit_listings.md`).
**Coverage**: 100.000% (9,911/9,911) — every fill day had ≥ 6 prior PIT daily sessions. Range
−3.6% to +5.3%, mean +0.31%.

## What this does NOT claim
This is a feature build, not a result. No model has been fit on these columns; PREREG_1478's own
pass bar (VAL AUC ≥ 0.60 for the label, ≤ 0.53 on a label-shuffled placebo, kept-set VAL mean net
R ≥ +0.15 at t ≥ 2.5) is unevaluated here. `sector_peers_j` and `spy_ret_open_to_j` carry real,
data-driven missingness (44% and 30%) that a second reader should re-derive from the two counts
above before trusting any feature-importance ranking that leans on them.

## Independent-check hooks (PREREG_1478 "Independent check" section)
A second agent rebuilding this from prose alone should reproduce, independently: `n_universe_j`
mean 1,428 (min 478 / max 2,418); `breadth_share_j` mean 11.7%; `breadth_5d` mean 11.2%;
`sector_peers_j` coverage 55.9% with the specific 515 missing symbols listed above obtainable from
`set(df.symbol) - set(panel['symbols'])`; `spy_ret_open_to_j` coverage 70.0% with the 51 zero-bar
SPY days obtainable from `select bar_date, count(*) from intraday_bars_1min where symbol='SPY' ...
group by bar_date` against `data/cache.db`.
