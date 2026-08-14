# ORB clean re-derivation (2026-08-14, post-DST-bug emergency rebuild)

Owner-ordered rebuild after the session-open DST bug (fixed in 3fab1f9) voided
every prior ORB research verdict. Every number in this document is computed on
DST-clean data only: Phase A on the uncontaminated EDT-season slice of the
pre-fix features CSV (2025-03-10..2025-10-31 + 2026-03-09..2026-08-13); Phase B
on the fully regenerated features CSV.

Harness: `research/scripts/orb_clean_harness.py` — rebuilds per-trade physics
(exact 9:30 ET anchor) from cache.db bars, drops any row whose CSV trigger
disagrees with the reconstructed trigger by >10bps, and re-simulates exits /
selection / sizing / vetoes with every knob parameterized. Validated against
the shipped pipeline: on identical rows the monthly P&L table reproduces
`analysis_results/orb_monthly_static_lock.csv` (diffs only from dropping the
2 contaminated missing-9:30-bar rows the pipeline keeps via CSV-pnl fallback).

---

## 1. Bug-class audit (repo-wide, all strategies)

Pattern hunted: session/range windows selected via naive-UTC hour comparisons
(`hour == 13/14`, `minute == 30`, hardcoded '13:30'/'20:00' on UTC timestamps).

**BUGGY — residual, NOT fixed by 3fab1f9:**

1. **`study_orb_features.py:321-324` — CRITICAL.** The SPY opening-range
   features (`spy_range_pct_5min`, `spy_return_5min_pct`) use the exact
   pre-fix pattern (`dt.hour.isin([13, 14]) & minute==30` on UTC SPY bars).
   Every EST-season row's SPY 5-min features are the 8:30 premarket window —
   including in the freshly regenerated CSV. NOT in the shipped 7-feature
   composite, so the book/selection is unaffected, but **no rebuilt candidate
   may use SPY 5-min features until this is fixed + regenerated**.
2. `research/study_macd_day_from_high.py:55-56,121-123` — RTH window as fixed
   UTC 13:30-20:00; winter `day_high`/`pct_of_day_high` include premarket
   8:30-9:30 and exclude 15:00-16:00 ET. MACD research feature contaminated.
3. `study_orb_exit_variants.py:79-82` — V4 "11:00 ET" force-close matches both
   15:00Z and 16:00Z; in EST it closes at 10:00 ET. Research-only.
4. `diagnose_morning.py`, `diagnose_morning2.py` — EDT-hardcoded UTC bucket
   edges (13.5/14.0/14.5); winter buckets off by 1h. Diagnostic-only.

**CORRECT (verified tz-aware):** all live paths — `trading/orb_engine.py`
(`_first_session_open_ts_utc` via ZoneInfo), `trading/macd_wave_engine.py`,
`trading/ignition_shadow.py`, `trading/trading_engine.py`, `main.py`,
`macd_wave_backtest.py`, `batch_backtest.py` premarket fetch, BF/ignition
research scripts (all `tz_convert('America/New_York')` before comparing), and
`study_orb_pipeline_static_lock.py`'s own session-open copy.

**FIXED-IN-3fab1f9:** `study_orb.py::_session_open_timestamp` + ~34
`study_orb_*` importers (code now correct; their pre-fix CSV outputs remain
void until regenerated).

Verdict: **the bug class is ORB-research-specific.** Bull-flag and MACD-wave
backtests and all live engines are tz-correct; one MACD research script and
two diagnostics carry the bug; one residual ORB feature-generation bug (SPY
5-min features) must be fixed before any SPY-feature-based candidate work.

---

## 2. What the bug did to the book (from research/orb_bar1_fill_study.md)

- 40/236 book trades (38 winter-DST + 2 missing-9:30-bar) = **$155,599 of the
  $251,647 headline (62%)**, including the two biggest trades ever booked
  (CRNC +$40.9K, AMCI +$46.0K) — computed off 8:30-8:34 premarket ranges live
  could never trade.
- Clean-subset era totals flip: published 2025H1 +$60.0K → clean **−$14.2K**.

## 3. Phase A — clean-EDT baseline (shipped config, $100K book, 14 EDT months)

Shipped pipeline restricted to EDT windows (both via book-filter and via
harness resim; harness numbers below):

| metric | value |
|---|---|
| Total P&L | **+$80,206** (book-filter view: +$77,529) |
| Trades | 137 |
| Trade WR / day WR | 41.6% / 42.5% |
| MDD (daily cum) | −$11,237 |
| Positive months | **5 / 14 (36%)** |
| Worst / best month | −$5,296 / **+$61,765 (2026-03)** |
| Top-5 trades' share of total | **97%** (ANNA 3/20/26 +$41.0K alone = 51%) |
| Eras (EDT slices) | 2025H1 −$4.9K · 2025H2 +$30.2K · 2026 +$54.9K |

Monthly: -1.6K, -0.1K, -2.3K, -1.0K, -0.8K, -1.0K, +8.3K, +23.7K | +61.8K,
+0.6K, -0.6K, +1.5K, -5.3K, -3.0K (Mar'25→Oct'25 | Mar'26→Aug'26).

**Honest read: on clean data the shipped ORB is a lottery-ticket book.** The
entire P&L is five trades; the median month loses money; 9 of 14 months are
red. The $342K/Calmar-18.9 and $251.6K headlines are void.

## 4. Phase A — clean-data lever findings (hypotheses for Phase B walk-forward)

All at shipped $100K/N=4/risk$3K unless noted; same 14 EDT months; relative
comparisons (unmodeled layers cancel in diffs).

- **Touchgo (Rule M 0.5 / Rule D 0.75) re-validates**: removing it costs
  −$21.8K, doubles MDD (−$11.2K → −$21.0K), worst month −$5.3K → −$7.6K.
- **PDR veto re-validates and strengthens**: on the $10K shape the clean-data
  response is monotone through threshold 10-12 (old 8.0 was not the optimum):
  None +$1.3K → 8.0 +$6.5K → 10.0 +$7.3K (11/14 months+) → 12.0 +$7.4K
  (12/14 months+, tWR 45.7%, top5 33.7%).
- **Catalyst veto (shipped 7/18 on contaminated evidence) HURTS clean-EDT
  totals**: without it $120.8K vs $80.2K, 10/14 months+, top-5 share 69%.
  BUT the un-vetoed cohort is itself monster-driven (+$40.6K net, only
  +$3.6K ex-top-5) and bleeds in Jul-Aug 2026 (−$14.6K). Not a slam-dunk
  either way → Phase B decides.
- **Composite threshold 0.25 > 0.0** on every owner bar (all eras positive,
  MDD −$5.3K vs −$11.2K, 8/14 months+) for −$8.8K total. 0.5 is too strict
  (starves the book). Q4Q5-only ≈ threshold 0.25.
- **Exits**: static-lock maximizes total but is pure lottery (top5 97%).
  Profit targets trade total for consistency: +1.5R full exit → top5 56%,
  9/14 months+, all eras positive, MDD −$8.4K. +2R similar. Quick targets
  (≤1R) lift WR (to ~49-59%) but destroy P&L (+$0.75R target: $1.3-1.7K
  total) — the edge lives in the 1.5R+ tail, ORB cannot be made into a
  high-WR scalper without killing it. Partial 50%@1.5R + breakeven + lock
  on the remainder is the best hybrid ($62.7K, top5 73%).
  Time-boxing static-lock at 11:30 keeps $59.7K with the best small-book WR
  (tgt WR 50%, dWR 49% at $10K) but stays lottery-shaped (top5 96-101%).
- **PM news-gated 2.0x mult**: +$17.7K on clean-EDT, but concentrated in the
  single ANNA monster; on the $10K book it mostly amplifies tail risk.

## 5. Phase A — $10K-book candidates (clean-EDT, 14 months, risk $375, N=2, no PM mult, uniform mults)

| config | total | tWR | dWR | MDD | months+ | worst m | top5 |
|---|---|---|---|---|---|---|---|
| SL + th.25 + cv | +$7.1K | 44.8% | 45.6% | −$1.5K | 9/14 | −$0.9K | 106% |
| tgt1.5R th.25 cvN pdr8 | +$6.5K | 38.9% | 44.6% | −$3.2K | 10/14 | −$1.9K | 40% |
| tgt1.5R th.25 cvN pdr10 | +$7.3K | 42.1% | 48.5% | −$3.0K | 11/14 | −$1.8K | 34% |
| tgt1.5R th.25 cvN pdr12 | +$7.4K | 45.7% | 51.8% | −$2.7K | 12/14 | −$1.6K | 34% |
| tgt2R th.25 cvY | +$3.9K | 43.1% | 43.9% | −$0.9K | 10/14 | −$0.6K | 70% |
| SL-11:30 th.25 cvY | +$6.1K | 50.0% | 49.1% | −$0.9K | 10/14 | −$0.6K | 101% |

Risk sensitivity (tgt1.5R th.25 cvN): risk $250 → +$5.5K / top5 33%;
$375 → +$6.5K; $500 → +$6.6K (cap $5K/pos binds).

**PHASE A VERDICT (pre-walk-forward)**: the only shapes that approach the
owner's bars (top5 <40%, ≥70% positive months, bounded worst month) are
profit-target exits (+1.5R/+2R) with threshold 0.25, PDR 10-12, catalyst veto
OFF. They project ~$6-7K per 14 EDT-months on a $10K book (~55-65%/yr) with
tWR ~42-46% — the high-WR bar is NOT met by any P&L-positive variant.

---

## 6. Phase B — regenerated-data validation (PENDING regen completion)

_(sections below to be filled when orb_features_20260814_*.csv lands)_

- Regen validation: CRNC 2025-01-03 range anchor check + EST spot-checks.
- Clean full-book baseline (shipped params, all months incl. winter).
- Walk-forward: fit on TRAIN=2025H1 only, validate 2025H2, report OOS 2026.
- Final candidate table + ranked recommendation.

## Caveats (apply to every number above)

- Fill reality: BT assumes the 9:35 stop-limit fills at trigger×1.003 when
  touched. The bar-1 fill study measured ~96.7% P&L capture at ≤5s placement
  latency but ~85.5% at the current degraded ~74s first-order latency —
  **deployment of ANY rebuilt config assumes the latency regression (since
  2026-07-07 news prefetch) is fixed first**.
- Phase A sweeps reuse the same 14 EDT months (selection bias); treat as
  hypothesis-generation only. Phase B applies TRAIN/VAL/OOS discipline.
- 2025H2-EDT (Jul-Oct 2025) is consistently the weakest era for target-exit
  variants — several configs are barely positive there.
