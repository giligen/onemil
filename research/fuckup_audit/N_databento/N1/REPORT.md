# N1 — order flow at the ORB breakout minute (Databento EQUS.MINI `tbbo`)

Run 2026-09-18 against `N_databento/PREREG.md` (written before any pull). One python process at a
time, `nice -n 10`, `ulimit -v 3000000`. Read-only outside `N_databento/N1/`; `data/cache.db` opened
`mode=ro`; no config, order, cron or production artefact written. **STATUS: complete.**

**Cost spent: $2.78** (whole 427-date plan priced with `metadata.get_cost` before the first byte was
pulled; budget was $150). 13,033 candidate symbol-days, 09:29:30–09:46:00 ET, one request per date.

---

## 1. What was built

* `fetch_tbbo.py` → `N1/tbbo/YYYY-MM-DD.parquet` (427 dates, ~154 MB, gitignored, resumable).
* `features.py` → `N1/features.parquet` + `N1/sidecar.csv`, keyed `(symbol, date)`:
  `ofi_range`, `tai_range` (09:30:00–09:34:59.999 ET), `ofi_break60`, `tai_break60`,
  `spread_at_break_bps`, `n_trades_range`, `n_trades_break60`, plus `range_high` / `range_low` from
  the Alpaca 09:30–09:34 bars (`study_orb_features.py:248`'s definition) and the breakout timestamp.
* `cells.py` → the 12 books, each produced by the **D1 code path**
  (`study_orb_pipeline_static_lock.py`, `ORB_BT_RESIM_CACHE=D1/candidates_dump.csv`, N=8,
  `per_pos_cap = account/N = 3,333.33`, Q1 filter on) — the same command `run_grid.sh` used for
  `book_n8_q1on.csv`. The pipeline gained two hooks that are **inert unless their env var is set**:
  `ORB_N1_COMPOSITE_FEATURE`/`_SIGN` (8th signed-z feature; threshold and quintile cutoffs rescaled
  by 7/8, the same affine factor the composite mean picks up, so an all-NaN column reproduces the
  baseline exactly) and `ORB_N1_VETO_FEATURE`/`_SIDE`/`_Q` (bottom-quintile veto, **post-ranking, no
  refill** — the PDR/G1 form).
* **Parity control**: hooks off, sidecar joined → **215 picks / $14,428.62**, identical to
  `D1/book_n8_q1on.csv`. Refusal to run the cells is wired in if that check fails.

Declared signs (fixed before the runs, so the grid stays at 12 cells): `ofi_range` +1, `tai_range`
+1, `ofi_break60` +1, `tai_break60` +1, `spread_at_break_bps` −1. "Bottom quintile" = the worst
quintile under that sign, cut at the 20th (or 80th) percentile of the **TRAIN 2025 candidate
universe**. NaN fails open everywhere.

---

## 2. The table — 12 cells + the baseline

R/pick = `(pnl_pct/100 × entry_price) / (range_high − range_low) × quintile_mult`, $0 for a
modelled non-fill. `$` columns are the 21-month book at $10K-stage sizing.

| cell | causal | TRAIN R | TRAIN t | VAL R | VAL wk-green | TEST R | picks | book $ | MDD | worst mo | red mo | ex-top-5% $ | +3R-cap $ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **baseline (book_n8_q1on)** | — | 0.486 | 3.21 | 0.862 | 38.9% | 0.172 | 215 | **14,429** | −576 | −148 | 2 | 4,403 | 8,583 |
| comp ofi_range | yes | 0.419 | 2.71 | 0.726 | 38.9% | 0.189 | 215 | 12,507 | −915 | −252 | 6 | 2,598 | 6,961 |
| **veto ofi_range** | yes | **0.540** | **3.37** | 0.879 | 38.9% | 0.175 | 206 | **14,729** | −576 | −148 | 2 | 4,703 | 8,883 |
| comp tai_range | yes | 0.373 | 2.40 | 1.095 | 47.4% | 0.069 | 207 | 14,085 | −977 | −400 | 7 | 1,112 | 5,402 |
| veto tai_range | yes | 0.468 | 2.78 | 0.759 | 29.4% | 0.208 | 158 | 10,939 | −838 | −505 | 6 | 3,473 | 6,914 |
| comp ofi_break60 | **DIAGNOSTIC** | 0.443 | 2.97 | 0.958 | 47.1% | 0.288 | 215 | 17,270 | −508 | −266 | 2 | 5,143 | 9,790 |
| veto ofi_break60 | **DIAGNOSTIC** | 0.537 | 3.29 | 0.881 | 41.2% | 0.180 | 203 | 14,653 | −533 | −148 | 1 | 4,627 | 8,807 |
| comp tai_break60 | **DIAGNOSTIC** | 0.369 | 2.41 | 1.398 | 47.4% | 0.164 | 213 | 15,566 | −1,090 | −677 | 4 | 1,821 | 6,032 |
| veto tai_break60 | **DIAGNOSTIC** | 0.467 | 2.83 | 0.936 | 43.8% | 0.081 | 178 | 11,563 | −791 | −510 | 5 | 2,841 | 6,345 |
| comp spread_at_break_bps | yes | 0.428 | 2.87 | 0.873 | 44.4% | 0.201 | 215 | 14,056 | −832 | −436 | 3 | 4,030 | 8,210 |
| veto spread_at_break_bps | yes | 0.427 | 2.60 | 0.948 | 38.9% | 0.214 | 178 | 12,102 | −977 | −496 | 4 | 3,688 | 6,889 |
| hold ofi_break60 (fill+10m) | yes | 0.458 | 3.08 | 0.873 | 44.4% | 0.197 | 215 | 14,386 | −729 | −226 | 2 | 4,361 | 8,566 |
| hold tai_break60 (fill+10m) | yes | 0.413 | 2.81 | 0.870 | 44.4% | 0.180 | 215 | 13,360 | −748 | −245 | 2 | 3,335 | 7,587 |

`DIAGNOSTIC` = the feature is measured in the 60 s **after** the breakout trade, i.e. after the
resting stop-limit has filled. It cannot select a pick. Those four cells are in the grid because
PREREG declared them; they are **not shippable under any result** and are reported only to show what
a non-causal version of this feature looks like (`comp ofi_break60` is the best book in the table at
$17,270 — precisely the shape of finding this program has been burned by four times). The two
break60 **hold** rules are live-usable (exit at fill+10 min when the 60 s imbalance ≤ 0 and the trade
is under +0.25R); both lose money vs the baseline.

**Nothing causal clears the baseline by more than noise.** One cell is nominally better —
`veto ofi_range`, +$300 over 21 months (+2.1%), same MDD, same worst month, same red-month count. It
does that by removing **9 of 215 picks** whose realised P&L was −$300 (mean −0.21R). Every other
causal cell is worse than doing nothing (`veto_removed.csv`: the tai_range / tai_break60 /
spread vetoes each cut 37–57 picks worth +$2.3K to +$3.5K).

## 3. Gates, permutation, power

* **G1 (TRAIN t ≥ 2.0)** is met by all 13 rows *including the baseline* — the gate as written in
  PLAN §1 scores the book, and this book already has an edge. The only meaningful statistic is the
  **increment over the baseline**, which is what §2 and the permutation below measure. The gate's
  frequency arm (**≥ 5 trades/week**) **fails for every row and for the baseline**: the B+ book runs
  2.6 / 2.9 / 3.6 picks per week on TRAIN / VAL / TEST. This stage cannot clear PLAN §1 on volume.
* **G2 (VAL sign + ≥ 55% weeks green)**: **no row reaches 55%** — the baseline itself is 38.9%, best
  in the grid 47.4%. Weekly green rate is structurally low at ~3 picks/week.
* **Permutation** (`perm.py`, the best causal cell `veto_ofi_range`, feature shuffled **within
  date**, 200 re-runs of the identical pipeline): observed TRAIN t **3.368**; null t mean 2.973,
  sd 0.257, p95 3.372 → **p = 0.060 unadjusted for one cell**. Twelve cells were looked at, so the
  search-adjusted p is ≈ 0.5 (1 − (1 − 0.060)^12 = 0.52). Not evidence.
* **Power / MDE** (baseline picks, two-sided α 0.05, 80% power, `mde.csv`):

  | split | picks | sd(R) | MDE R/pick | MDE as book $ |
  |---|---|---|---|---|
  | TRAIN | 105 | 1.55 | **0.424** | ~$5,800 |
  | VAL | 53 | 2.81 | **1.081** | ~$8,000 |
  | TEST | 57 | 1.02 | **0.379** | ~$3,000 |

  The smallest per-pick effect TRAIN could resolve is **0.424 R — 87% of the book's entire existing
  edge (0.486 R)**. At 8 slots and 215 picks this test is close to blind to anything short of a
  second edge the size of the first one.
* **TEST**: computed in the single `--analyse` pass that produced the table above, together with
  TRAIN and VAL. It was **not** used to pick a cell — the pre-registered selection statistic is the
  TRAIN t, and the reported best causal cell (`veto ofi_range`) is the TRAIN-t maximum. Stating this
  plainly rather than claiming a sequencing that did not happen.

## 4. Data quality — coverage, availability, price scale

* **Availability assertions executed and passed**: **12,451** (no timestamp used for a `*_range`
  feature is at or after 09:35:00 ET) and **4,706** (no timestamp used for a `break60` feature is
  later than `break_ts` + 60 s). Zero failures; an assertion failure aborts the build.
* **Price-scale check**: 200 random candidates, the Databento breakout trade price vs the Alpaca
  1-min bar high/low of that same ET minute — **4 of 200 (2.0%) fall outside the bar**, all marginal
  (the Alpaca consolidated minute bar and the EQUS.MINI venue subset do not have identical extremes).
  No systematic scale error (no 2× / 10× / split-shaped disagreement).
* **Coverage — this is the finding that matters more than the table.** EQUS.MINI is a venue subset:

  | | all 13,033 candidates | the 215 baseline **picks** |
  |---|---|---|
  | zero tbbo trades in the 5-min range | 582 (4.5%) | — |
  | **< 20 trades in the range window** | **4,446 (34.1%)** | **54.4%** |
  | `ofi_range` computable | 12,170 (93.4%) | 90.2% |
  | a breakout print above `range_high` seen by 09:46 | 4,706 (36.1%) | — |
  | `ofi_break60` computable | 4,360 (33.5%) | **51.6%** |
  | `spread_at_break_bps` computable | — | **56.3%** |

  PREREG's own stop rule fires here: *"a feature that is empty on the thin half is not a feature."*
  On the picks the book actually takes, more than half have fewer than 20 prints in the whole
  opening range on this tape, and the break-instant features are missing on ~45–48% of them. The OFI
  is additionally a **sampled** CKS sum: `tbbo` carries the BBO *at each trade*, not every BBO
  update, so the quote sequence is observed at trade times only — on a name with 8 prints in five
  minutes that is 7 increments, not an order-flow measurement.

## 5. Cells looked at

**12 pre-registered cells** (5 features × {composite, veto} + 2 break60 hold rules), **1 parity
control**, **1 permutation family** (200 re-runs of one cell). No cell outside PREREG was scored, no
threshold, z-param, quintile cutoff or sizing mult was re-tuned, and the veto quantile stayed at the
declared 0.20. Inherited multiplicity to carry forward: the baseline book `book_n8_q1on` was itself
the survivor of D1's 14-book slot/pool grid.

## 6. Verdict

> **No edge was detectable in THIS population** (the 13,033 entered-inclusive ORB candidates,
> 2025-01-02 → 2026-09-16), **at THIS horizon** (the 09:30–09:35 opening range and the 60 s after the
> breakout print), **on THIS tape** (Databento EQUS.MINI `tbbo`, a venue subset that shows fewer than
> 20 range prints on 54% of the book's own picks), **at THIS book size** (8 slots, 215 picks, $10K
> stage sizing), **at THIS cost contract**. The best causal cell, a bottom-quintile `ofi_range` veto,
> is +$300 on 21 months from removing 9 picks, permutation p = 0.060 for one cell and ≈ 0.5 adjusted
> for the 12 looked at. **The smallest effect this test could have seen is 0.424 R/pick on TRAIN —
> 87% of the book's entire existing edge** — so the null here is close to uninformative about effects
> of a plausible size. Nothing goes to an independent rebuild; nothing goes near the engine.

**What would make the question answerable** (not run, not costed here): the same features on a
consolidated tape (XNAS.ITCH for Nasdaq names, or EQUS full), which would fix the 54% thin share and
give a true message-level OFI; and a population with enough picks per week that the MDE falls under
~0.1R — i.e. the CKS filter is worth re-asking on the HOD-break book (25–30 signals/week) rather than
on ORB's 3/week.

**Standing note for the plan**: the strongest-looking book in the grid (`comp ofi_break60`, $17,270,
+20% over baseline, best TEST R in the table) is non-causal by construction. It is recorded here as a
worked example of the failure mode, not as a result.
