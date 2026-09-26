# 1445 SIPZERO diagnostic — which builder wrote it, and does any cache-builder rule explain it

Diagnostic only (per task framing), not a research cell — numbers below are reported as facts with the
query/script that produced them. Read-only throughout: all sqlite connections opened `?mode=ro`; nothing
written to `bars_sip.db`, `cache.db` or any production file. Scripts used (kept in scratchpad, not committed):
`sipzero_analysis.py` (stage 1: cohort build + fetch_log), `sipzero_stage2.py` (stage 2: PIT rule membership),
`sipzero_stage3.py` (stage 3: tables + composition).

## Cohort build

Base = `research/hod_entry/causal_arming_causal.csv` rows with `fill_min` not null (9,911 — this is the
`status=='fill'` population; `fill_min` is only populated on fills). Joined to
`research/hod_entry/cell_1445_features.csv` (9,911 rows, one per fill) on `(day, symbol, fill_min)` — **not**
on `split`: the base CSV's own `split` column is a stale pre-1445 label (every fill row shows
`split_orig=='TRAIN', half=='H2'`); the feature file's `split` (`TRAIN-H2`/`VAL`/`TEST`) is the one actually
used everywhere below. The join is 1:1 (max 1 row per key) and recovers all 9,911 rows.

SIPZERO = fills whose `(symbol, day)` has zero rows in `research/bf_zero/bars_sip.db` table `bars`, computed by
querying `SELECT DISTINCT symbol, day FROM bars WHERE day IN (<our 230 fill days>)` (index on `day` makes this
cheap — 169,889 covered pairs found, vs a full-table distinct scan which the codebase itself notes takes ~4 min).

**n(SIPZERO) = 1,928, n(REST) = 7,983** — matches the task's stated 1,928 exactly.

Corrected mean `net_R_corr` from my query:
| split | SIPZERO n | SIPZERO mean net_R_corr | REST n | REST mean net_R_corr |
|---|---|---|---|---|
| TRAIN-H2 | 899 | **+0.226** | 3,499 | |
| VAL | 1,029 | **+0.060** | 4,484 | |
| overall | 1,928 | +0.137 | 7,983 | **−0.345** |

My REST figure (−0.345) matches the task's "≈ −0.35" closely. My SIPZERO split means (+0.226 TRAIN-H2 /
+0.060 VAL) do **not** match the task's quoted +0.345 / +0.195 — both of mine are lower by almost exactly the
same ~0.12–0.135 R, which looks systematic rather than random (tried `net_R_corr_flat30` too: +0.245 / +0.088,
still off). Flagging this as a fact, not resolving it: the cohort size (1,928) and the REST anchor both
reproduce exactly, so the discrepancy is most likely in the split-mean definition (e.g. a weighting or
tail-handling step upstream of this file) rather than in cohort membership. Read this report's SIPZERO-mean
numbers as **directionally confirmed, magnitude unresolved**.

## Task A(1) — which builder wrote SIPZERO (or rather, didn't)

**`bars_sip.db` is not an HOD-entry artifact at all.** Its `fetch_log` table (`src` column) shows it was built
for the **bull-flag zero-study's own candidate population**, by two scripts, neither of which has ever heard of
`causal_arming_causal.csv`:

- `research/bf_zero/refetch_thin_tape.py` (2026-09-15, "parity review") — re-fetches, from Alpaca SIP, the
  **bf_zero causal-superset symbol-days** (its own docstring: *"day high >= open × 1.05, all prices"*) that
  `build_candidates.load_bars` had served from a thin (EQUS.MINI) store. Its key list is
  `research/bf_zero/parity_review/superset_provenance.csv` — a bull-flag artifact. This is the bulk source
  (`fetch_log.src` values `bars.db`/`pit_bars_1min.db`/`topup.db(copy)`/`none`, 307,343 rows repo-wide).
- `research/bf_zero/backfill_bars_sip.py` — appends rows for `research/bf_zero/causal_filter/features.csv`'s
  population (the **failed-break-short study**, cells 1,357–1,358). It has **no fetch_log write at all** and
  no "skip if cache.db already has ≥ N bars" rule of any kind — it just diffs its FEATURES population against
  existing `(symbol, day)` pairs in `bars`.

Both target populations are bf_zero-specific, not the HOD arming-bar population.

**20-symbol sample check** (`SELECT src, n_bars, fetched_at FROM fetch_log WHERE symbol=? AND day=?`, first 20
distinct SIPZERO pairs by day): **20/20 ABSENT from fetch_log** — never requested, not "requested with
n_bars=0". Sample: ACMR/ASTS/CLSX/CRWV/CWVX/INTW/LEU/NNE/PLTZ/POWL/SKYT/SNDK/SOXL (2026-01-02),
ALGM/APPX/ASTX/CIFU/FIGR/HOOX/IONX (2026-01-05) — all absent.

**Population-overlap check**: of the 1,928 distinct SIPZERO `(symbol, day)` pairs, **0 (0.0%)** appear in
`research/bf_zero/causal_filter/features.csv` (`backfill_bars_sip.py`'s own target list) — confirming that
script never had the chance to fetch any of them; it wasn't skipping them, it was never pointed at them.

**Day-level check** (does the gap look like a per-day outage, or a per-symbol selection?): grouped fills by
`day`, computed the SIPZERO share per day. **226 of 230 days are "mixed"** (share strictly between 0 and 1);
only 4 days are 0%-SIPZERO and **0 days are 100%-SIPZERO**. So the gap is not a dead day or a broken backfill
run — every day has *some* symbols covered and some not, consistent with per-symbol candidate-list membership
in a different study's universe, not a completeness defect in this one.

**Bottom line for A(1): no rule "wrote" SIPZERO — SIPZERO is the complement of the bf_zero study's own
candidate universe.** The real fetch predicate quoted above (`high >= 1.05 × open`) is satisfied by **100% of
both SIPZERO and REST** (every HOD arming-bar fill is, trivially, some kind of intraday mover) — so it has zero
discriminating power here; the actual gate that put a symbol-day on bf_zero's list was narrower (float/price/
liquidity terms inside `build_candidates.py`, not reconstructed in this diagnostic — out of the 35-call budget).

## Task A(2) — the four proxy rules, evaluated point-in-time

PIT source: `data/research/databento/equs_daily_2025_2026.parquet` (5,047,530 rows) joined to
`equs_instrument_symbol_map.csv` on `symbol` with `d0 <= day <= d1` to get the PIT `instrument_id`, then
`prev_close`/`prev_volume` = that instrument's prior parquet row (`groupby(instrument_id).shift(1)`, sorted by
`bar_date`). **9,911/9,911 fills matched an instrument, and 9,911/9,911 have a same-day PIT daily row — SIPZERO's
share with NO daily row in the parquet at all = 0.0%.** (SIPZERO is missing 1-minute SIP bars, not missing
daily bars — the two are unrelated in this population.)

Rules: ORB = `(open-prev_close)/prev_close>=5% & 3<=open<=30 & prev_volume>=500K` (matches `orb_backtest.py:55-58`
constants exactly). MACD = `(high-low)/low>=10% & close>=10 & prev_volume>=1M` (matches
`macd_wave_backtest.find_movers`'s filter shape). BF = the actual production predicate
`trading.bf_selection.mover_day_qualifies` (imported, not reimplemented), `threshold_pct=0.10`, two price bands:
2–20 (older) and 1–30 (current `config.yaml`: `scanner.price_min: 1.0`, `scanner.price_max: 30`).
WRAPPER = union of three sources: (a) `data/research/orb_asset_class_map_20260711.csv` `asset_class=='wrapper'`
(309/1,625 of our symbols), (b) `trading.bf_universe_filter`'s name rule inverted — fails
`AlpacaClient._is_common_stock` by default but passes it with `exclude_leveraged=False` (252/1,625), (c) the
legacy `AlpacaClient._LEVERAGED_ETF_SYMBOLS` hard list (45/1,625); union = 312/1,625 symbols.

| rule | SIPZERO share kept | SIPZERO kept mean net_R_corr | REST share kept | REST kept mean net_R_corr |
|---|---|---|---|---|
| ORB | 3.9% | −0.419 | 2.2% | −0.058 |
| MACD | 47.3% | +0.408 | 29.1% | +0.303 |
| BF (2–20) | 4.5% | −0.980 | 0.6% | −0.702 |
| BF (1–30) | 38.8% | −0.139 | 16.5% | −0.014 |
| WRAPPER | 41.5% | +0.043 | 32.8% | −0.257 |

Union coverage of SIPZERO, added in the order **ORB → WRAPPER → MACD → BF(1–30)**:
ORB alone 3.9% (75/1,928) → +WRAPPER 42.8% (825) → +MACD 73.9% (1,425) → +BF(1–30) 83.0% (1,600). **17.0%
(328 fills) are covered by none of the four rules**, and that leftover slice has a near-zero mean
(`net_R_corr` = +0.007) — i.e. the discriminating part of SIPZERO's edge sits inside the 83% the rules DO
touch, not in the residual.

None of the four rules is a strong single explanation (each keeps a small-to-moderate, overlapping slice of
both SIPZERO and REST at a somewhat higher kept-share for SIPZERO than REST on every rule) — consistent with
A(1)'s conclusion that the real gate is bf_zero's own candidate screen, which these four rules only partially
approximate.

## Task A(3) — SIPZERO composition

**Top-15 symbols by fill count** (`symbol, n, mean net_R_corr, mean raw_R, is_wrapper`):

| symbol | n | mean net_R_corr | mean raw_R | wrapper |
|---|---|---|---|---|
| ASTX | 21 | +0.065 | +0.313 | yes |
| OKLL | 19 | +0.806 | +0.895 | yes |
| NEBX | 19 | −0.096 | +0.103 | yes |
| CWVX | 17 | −0.585 | −0.378 | yes |
| RGTX | 16 | +0.213 | +0.366 | yes |
| RKLX | 16 | +0.255 | +0.447 | yes |
| NVTX | 15 | −0.115 | +0.295 | yes |
| IREN | 15 | +0.727 | +0.800 | no |
| QBTX | 14 | −0.391 | −0.177 | yes |
| OKLO | 14 | +1.148 | +1.264 | no |
| CLSX | 13 | −0.902 | −0.609 | yes |
| IONX | 13 | −0.248 | −0.078 | yes |
| IONL | 12 | −0.126 | +0.064 | yes |
| INTW | 12 | +0.777 | +0.896 | yes |
| APLX | 12 | −0.536 | −0.188 | yes |

13 of the top 15 are wrapper-flagged (single-stock 2x/3x names — the "X"/"L"/"U" suffix pattern is visible:
ASTX/OKLL/NEBX/CWVX/RGTX/RKLX/NVTX/QBTX/CLSX/IONX/IONL/APLX). No single symbol dominates the cohort's total
R (max 21 fills of 1,928 — this is a broad population effect, not one carried trade).

- **Wrapper share**: SIPZERO 41.5% of fills vs REST 32.8% (`WRAPPER.mean()` on each subset).
- **Price (`level`) quartiles, SIPZERO**: min $20.0, P25 $23.7, median $33.2, P75 $58.5, max $1,075, mean $59.9.
  (Materially above bf_zero's own $1–30/$2–20 bands — consistent with A(1)/A(2): a chunk of SIPZERO is simply
  priced outside the universe bf_zero's builders ever looked at.)
- **Cost effect** (`raw_R` uncorrected vs `net_R_corr`): SIPZERO raw +0.388 → net +0.137 (cost = 0.250 R/fill).
  REST raw −0.073 → net −0.345 (cost = 0.273 R/fill). The cost correction is similar in size on both sides
  (~0.25–0.27 R/fill) — **the SIPZERO/REST gap is present in the gross (`raw_R`) numbers too** (+0.388 vs
  −0.073), so it is not an artifact of the cost model; it pre-dates the correction.

## Caveats (read as an adversary)

- SIPZERO's split-level means here (+0.226 TRAIN-H2 / +0.060 VAL) don't reproduce the task's quoted
  (+0.345 / +0.195) even though n and the REST anchor match exactly — unresolved, flagged above, not smoothed
  over.
- The four proxy rules are approximations of "what a cache-builder would have fetched"; the actual bf_zero
  `build_candidates.py` screen (float/liquidity/name terms beyond a bare mover threshold) was not reconstructed
  here — out of budget for this diagnostic. The one predicate I could pull verbatim from the true builder's own
  docstring (`high >= 1.05×open`) turned out to have zero separating power because the entire HOD fill
  population already satisfies it.
- "Union coverage" uses BF(1–30) only (the live config band) for the cumulative pass; BF(2–20) is reported
  separately in the per-rule table and would add less (it keeps only 4.5% of SIPZERO on its own).
