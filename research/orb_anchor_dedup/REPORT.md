# REPORT — ORB anchor dedup (one pick per underlying per day)

Pre-registration: `PREREG.md` (written before any code or any number).
Run: 2026-09-18. **Verdict: DOES NOT SHIP.**

## 1. Reproduction gate (PREREG §3) — PASSED

As-is at N=8 reproduces the reference book to the cent:

| | picks | P&L |
|---|---|---|
| `research/fuckup_audit/D1_orb/book_n8_q1on.csv` | 215 | $14,428.617 |
| this run `runs/book_n8_asis.csv` | 215 | $14,428.617 |

Identical (symbol, date) set; max per-trade |ΔP&L| = 0.0.

## 2. The cells

`{as-is, by_anchor} × {N=8, N=3}`, honest entered-inclusive features
(`analysis_results/orb_features_20260916_2053.csv`) replayed off the D1
candidate dump, $10K-stage sizing, Q1 filter ON, every shipped veto ON.
Recipe: `run_cells.sh`. No threshold exists to tune — the rule has no
parameter. Cells looked at in this program: 4.

## 3. Results

P&L in dollars at $10K-stage sizing; MDD on the book's daily equity curve.

| N | split | as-is P&L | anchor P&L | as-is MDD | anchor MDD | as-is worst mo | anchor worst mo | as-is picks | anchor picks |
|---|---|---|---|---|---|---|---|---|---|
| 8 | TRAIN 2025 | 6,662 | 5,267 | -528 | -387 | -25 | -221 | 105 | 90 |
| 8 | VAL 26-01..05 | 6,386 | 6,496 | -489 | -489 | +159 | +159 | 53 | 48 |
| 8 | TEST 26-06+ | 1,380 | 947 | -515 | -473 | -148 | -148 | 57 | 48 |
| 8 | **ALL** | **14,429** | **12,710 (-11.9%)** | -576 | -489 | -148 | **-221** | 215 | 186 |
| 3 | TRAIN 2025 | 3,650 | 3,618 | -617 | -480 | -203 | -295 | 47 | 44 |
| 3 | VAL 26-01..05 | 2,681 | 2,782 | -409 | -322 | -223 | -223 | 19 | 18 |
| 3 | TEST 26-06+ | 855 | 855 | -193 | -193 | -122 | -122 | 22 | 22 |
| 3 | **ALL** | **7,187** | **7,254 (+0.9%)** | -714 | **-759** | -223 | **-295** | 88 | 84 |

Negative months are unchanged in every cell (N=8: 2 and 2; N=3: 6 and 6).

**Picks removed and their P&L** (`runs/dropped_n*_anchor.csv`):

* N=8 — 54 picks removed (13 winners / 31 losers / 10 flat-or-no-fill),
  pre-veto P&L +$1,445; book delta **-$1,719** (larger than the pre-veto
  number because several removed picks were losers the PDR/G1/catalyst vetoes
  would have dropped anyway, while the survivors it removed were winners:
  CRCG +$561 and CRCA +$559, both 2025-10-02, and SMCL +$527 2025-05-13).
* N=3 — 7 picks removed (1 winner / 4 losers / 2 flat), pre-veto P&L -$137;
  book delta **+$67**.

## 4. Verdict against the pre-committed ship rule

The rule required, at **BOTH** N=8 and N=3: P&L >= 98% of as-is, no split
flipping positive->negative, **and** MDD <= as-is **and** worst month >= as-is.

| criterion | N=8 | N=3 |
|---|---|---|
| P&L >= 98% | **FAIL** (88.1%) | pass (100.9%) |
| no split flips negative | pass | pass |
| MDD <= as-is | pass (-576 -> -489) | **FAIL** (-714 -> -759) |
| worst month >= as-is | **FAIL** (-148 -> -221) | **FAIL** (-223 -> -295) |

**DOES NOT SHIP.** The one thing the rule was supposed to buy — a smaller
tail from doubling up on one underlying — is not what it delivers: the worst
month gets WORSE at both slot counts, because the pick it keeps is not
reliably the better of the pair (it keeps the best-RANKED one, and the
ranking does not separate siblings). At N=8 it also costs 12% of the book.
At the live N=3 the whole effect is 7 picks in 21 months worth +$67 — inside
the noise, decided by two ties.

Nothing is implemented in `trading/orb_engine.py`. `orb.yaml` is unchanged;
there is no `dedup.by_anchor` knob to flip. The live dedup stays
`by_family + by_super_group`.

## 5. What this does NOT say

* It does not say sibling double-fills are harmless. CIFG/CIFU on 2026-09-18
  (both 2X CIFR wrappers, both filled, both stopped in the same second) is
  real and is exactly the exposure the rule targets. It says that **this**
  rule — keep the best-ranked sibling, no refill — does not pay for itself on
  21 months of the honest book, at either slot count, on any of the three
  pre-committed measures.
* Power: at N=3 the rule touches 7 of 88 picks. A test that moves 8% of the
  picks cannot resolve an effect smaller than roughly a month's P&L; the
  honest statement is "no improvement detectable in THIS book, at THIS slot
  count, over THIS window", not "correlation does not matter".
* An untested alternative that would NOT have failed criterion 1 the same way:
  dedup that keeps both siblings and instead HALVES each one's size (a shared
  risk budget per anchor) rather than dropping a slot. That is a different
  rule with a different mechanism and needs its own pre-registration — it was
  not tested here and must not be inferred from these numbers.

## 6. Code left behind (research infrastructure, default OFF)

Per PREREG §4 nothing enters the engine or the pipeline's default path. Kept
so the next person re-runs instead of re-implementing:

* `trading/orb_anchor_dedup.py` — the rule as ONE spec (rank order, no refill,
  unknown anchor fails open). Imported only by the study path.
* `study_orb_pipeline_static_lock.py` — env-gated block, `ORB_ANCHOR_DEDUP=1`
  to enable; default OFF and byte-identical to before (proved by the
  reproduction gate above and by `tests/test_orb_anchor_dedup.py`).
* `run_cells.sh`, `runs/`, `table.txt` — the four cells and their logs.
