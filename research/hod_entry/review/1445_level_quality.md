# TASK B — are the HOD-break levels reliable? (diagnostic, not a research cell)

Diagnostic for why the PREREG_1445 placebo failed. Base book = `research/hod_entry/causal_arming_causal.csv`
rows `status=='fill'` (9,911 rows, 230 unique days, 9,911 unique symbol-day pairs — no repeats), joined to
`research/hod_entry/cell_1445_features.csv` on `(day,symbol,fill_min)` (`validate='one_to_one'`, 9,911/9,911
matched; feat's `split` label `TRAIN-H2` is a rename of base's `TRAIN`, same 4,398 rows — confirmed by exact
count match, not filtered further).

SIPZERO = fills whose `(symbol,day)` has zero rows in `research/bf_zero/bars_sip.db` table `bars`
(`select symbol, t, h from bars where day=? and symbol in (...)`, then `count==0`). **n = 1,928**, matching the
task's context number exactly. REST = the other 7,983.

**Caveat on the context numbers**: recomputing mean `net_R_corr` on this exact SIPZERO/REST split gives REST
= **-0.345** (TRAIN -0.368 n=3,499 / VAL -0.327 n=4,484), which matches the task's quoted "~-0.35" almost
exactly — the cohort definition and join are validated. SIPZERO gives **TRAIN +0.226 (n=899) / VAL +0.060
(n=1,029)** (pooled +0.137), lower than the quoted +0.345 / +0.195. `net_R_corr_flat30` gives +0.245 / +0.088,
also short. Flagged, not resolved: outside this diagnostic's scope, but the gap means the headline SIPZERO
number may come from a further filter or a different cost column not reproduced here.

## Source selection (mirrors `causal_arming.load_day_bars` lines 192-212)

Per symbol-day: SIP wins ties (`>=`); cache wins only if it has strictly more RTH bars. Checked directly on
the 7,983 REST symbol-days (SIP has ≥1 row): **`n_sip - n_cache` is never negative** (min diff = 0, 48.0% of
REST symbol-days have `n_sip == n_cache`). Consequence: **source actually used is 100% cache.db for SIPZERO
(1,928/1,928, by construction — SIP has 0 rows) and 100% SIP for REST (7,983/7,983)** — cache.db never wins a
symbol-day where SIP has any data at all in this population.

RTH = `t`/`timestamp` converted UTC→America/New_York, bars with `09:30 <= m < 16:00` (`m` = minute-of-day),
deduped on `m`. Databento daily high: `data/research/databento/equs_daily_2025_2026.parquet` joined to
`equs_instrument_symbol_map.csv` on `instrument_id` with `d0 <= day <= d1`; **all 9,911 symbol-days resolved
an instrument_id and a daily high** (0 missing).

## Results table

| metric | SIPZERO (n=1,928, source=cache.db) | REST (n=7,983, source=SIP) |
|---|---|---|
| (i) RTH bar count: mean / median / p10 / p90 | 322.9 / 359 / 193 / 390 | 337.0 / 374 / 219 / 390 |
| (i) share of symbol-days with < 300 RTH bars | **31.2%** (601/1,928) | **23.6%** (1,884/7,983) |
| (ii) share where source RTH max-high < Databento daily high by > 0.5% | **0.10%** (2/1,928) | **0.03%** (2/7,983) |
| (ii) mean (db_high − src_max_high)/db_high, all rows | 0.0012% | 0.0002% |
| (iii) share with a "level error" (level < running max-high through fill_min-1, by > 1 tick) | 88.5% (1,707/1,928) | 91.0% (7,266/7,983) |
| (iv) mean net_R_corr, level-error rows | **+0.179** (n=1,707) | **-0.332** (n=7,266) |
| (iv) mean net_R_corr, no-level-error rows | **-0.183** (n=221) | **-0.482** (n=717) |

10 random SIPZERO fills (`symbol, day, level, cache.db RTH bar count, Databento daily high`), seed 42:

| symbol | day | level | cache.db bars | Databento high |
|---|---|---|---|---|
| HLNE | 2026-02-03 | 147.46 | 355 | 149.44 |
| OKLO | 2026-04-30 | 68.83 | 390 | 72.84 |
| HOV | 2026-05-21 | 107.45 | 149 | 116.08 |
| PLTU | 2026-01-07 | 79.93 | 371 | 80.84 |
| RGTIW | 2025-09-25 | 20.2323 | 305 | 21.33 |
| SSRM | 2025-10-15 | 25.55 | 390 | 25.98 |
| APLD | 2026-03-09 | 26.43 | 390 | 27.1661 |
| NVDL | 2025-09-22 | 87.46 | 390 | 92.00 |
| WGMI | 2026-02-09 | 42.79 | 331 | 44.1685 |
| MARA | 2025-10-20 | 21.75 | 390 | 22.0899 |

(All 10 sampled rows: source RTH max-high == Databento daily high exactly, i.e. cache.db captured the day's
print even where bar count was well under 390 — e.g. HOV at 149/390 bars.)

## Reading the numbers (adversarial pass on my own query, per CLAUDE.md)

- **(iii)/(iv) is not evidence of bad levels.** `level` is the running max-high through the *arm* bar
  (`causal_arming.py:73`), and fill happens later on a resting order once price trades through it — so by
  construction the price is expected to make a new high between arming and fill on most trades. 88-91%
  "error" in BOTH cohorts (nearly identical) shows this metric is dominated by normal breakout follow-through,
  not a data defect, and it does **not** differentiate SIPZERO from REST. The R-by-error-status pattern
  (more follow-through → less-bad R) points the same direction in both cohorts too.
- **(ii) is the real check on "is the level built from a truncated bar set actually wrong,"** and it comes
  back essentially null: source RTH max-high understates the Databento daily high by >0.5% on only 0.10%
  of SIPZERO symbol-days and 0.03% of REST — even though 31% of SIPZERO symbol-days have <300 of a possible
  390 RTH bars in cache.db. Missing bars in cache.db are concentrated where price isn't printing new
  extremes; the day's high (and hence the level) is preserved almost every time regardless of density.
- **Net read: this diagnostic finds no support for "SIPZERO outperforms because its cache.db-sourced levels
  are corrupted by missing bars."** Bar-count coverage for SIPZERO is thinner than REST (31% vs 24% below
  300 bars) but the high-value fidelity implied by (ii) is comparable and near-perfect in both cohorts. The
  PREREG_1445 placebo failure needs a different explanation than mismeasured HOD levels — the SIPZERO/REST
  mean-net_R_corr gap itself did not fully reproduce off `net_R_corr` here either (see caveat above), so the
  next step is resolving *that* gap before ruling levels in or out entirely.

## Reproduction

Script and intermediate pickles: `/tmp/claude-1000/-home-ec2-user-onemil/257c3e2d-cf38-45d5-94e7-4877f8170f44/scratchpad/level_quality{,2,3,4}.py`
(session scratchpad, not committed). All DB access via `?mode=ro` URIs; no cache.db/bars_sip.db writes; no
git commit made.
