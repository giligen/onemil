# PRE-REGISTRATION — ORB anchor dedup (one pick per underlying per day)

Written **before** any code or any number was produced. 2026-09-18.

## 1. The observation that motivates it (not evidence — motivation)

2026-09-18 live: **CIFG** and **CIFU** were both selected, both filled, and both
stopped in the same second (13:44:23 UTC). Per
`data/research/orb_asset_class_map_20260711.csv` both are 2X long wrappers on
**CIFR** — one underlying, two slots, one outcome, twice the size.

They passed dedup because `orb.yaml dedup: by_family + by_super_group` resolves
families from the hand-kept table in `study_orb_correlation_filter.py` (14
families / 91 symbols) which predates these tickers. The engine ALREADY computes
`underlying_anchor` per pick (`trading/orb_asset_class.py`) for the catalyst
veto — the information was present and unused. Per the deliberate-rules
doctrine this is an ACCIDENT, not a rule: it must be tested and then either
shipped as explicit code or documented as deliberately not shipped.

## 2. The rule under test (exactly one cell)

> **by_anchor**: at most ONE pick per `underlying_anchor` per day. Rank order
> decides which one survives (the day's ranking: `_q_rank` asc, `_composite`
> desc). Every later pick sharing that anchor is REJECTED.
> **NO REFILL** — the rejected pick's slot stays empty, exactly like the PDR,
> G1, range-size and catalyst vetoes. A refilling form is NOT tested and will
> not be shipped (refill re-tested toxic for PDR: MDD −$29K→−$50K).

Placement: immediately after top-K selection, BEFORE the PDR/G1/range-size/
catalyst vetoes. An anchor is marked seen at its rank position whether or not
that pick is later vetoed — so the surviving pick is a function of the ranking
alone, not of veto ordering. The live engine must apply the identical rule at
the same point (first check in the submit loop), sharing one helper.

Symbols with no resolvable anchor (`None`) are never deduped against each
other — unknown fails OPEN, like every other anchor consumer.

## 3. Design

* Book: the honest entered-inclusive features
  `analysis_results/orb_features_20260916_2053.csv` replayed off
  `research/fuckup_audit/D1_orb/candidates_dump.csv` (identical exit physics),
  `study_orb_pipeline_static_lock.py`, $10K-stage sizing
  (`per_pos_cap = account/N = 3333.33`, `ORB_BT_RISK=375`), Q1 filter ON, all
  shipped vetoes ON — i.e. the exact `run_grid.sh` recipe.
* **Reproduction gate (run first)**: the as-is run at N=8 must reproduce
  `research/fuckup_audit/D1_orb/book_n8_q1on.csv` to the cent. If it does not,
  the study STOPS and nothing is shipped.
* Cells: `{as-is, by_anchor} × {N=8, N=3}` = 4 runs. Nothing else is varied.
  N=8 is the grid's reference book; N=3 is the live `max_concurrent`.
* Splits, fixed now: **TRAIN** 2025-01..2025-12, **VAL** 2026-01..2026-05,
  **TEST** 2026-06..2026-09.
* Reported per split and overall: P&L, number of picks, max drawdown (on the
  daily equity curve of the book), worst month, negative months.
* Also reported: how many historical picks the rule removes and their P&L.

## 4. Ship rule (committed before seeing any result)

Ship `dedup.by_anchor: true` iff, at **BOTH** N=8 and N=3:

1. **Not worse on P&L**: total P&L ≥ 98% of as-is (a ≤2% give-back is accepted
   as the price of a correlation rule — larger is a real cost), AND no split
   flips from positive to negative; AND
2. **Better or equal on risk**: max drawdown ≤ as-is AND worst month ≥ as-is
   (both at the book level).

If the rule fails either test, it does NOT ship: no code is added to the engine
or the pipeline's default path, and `REPORT.md` records why, so the next person
does not re-litigate it from the CIFG/CIFU anecdote.

A single-day anecdote (CIFG/CIFU) is NOT evidence and is excluded from the
verdict — it only motivated the question.

## 5. Multiplicity

Cells looked at in this study: 4 (2 rules × 2 slot counts). No threshold is
tuned — the rule has no free parameter. This is the first and only anchor-dedup
form tested.
