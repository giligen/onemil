# PREREG — live ORB add-on pools (the "union rung", cell 1,328) at exploration size. Owner GO 2026-09-21

**Evidence** `REPORT_S1_CREATIVE.md` §2: production ∪ (gap 4–5 %, open $3–30) ∪ (gap 3–5 %, open $30–50), each
pool evaluated by the unchanged ORB selection chain on its own, sharing the 8 slots. Union book TRAIN $9,190 vs
$6,561, VAL $10,207 vs $6,398; added cohort +0.09 R (n 77) / +0.15 R (n 68), positive in both 2025 halves and
ex-top-5 %; largest added trade +2.25 R. Fails one research criterion (TRAIN weekly MDD 2.05× production).
**Tier:** owner's exploration tier (9/18) — a research gate is for claims, not capital. This is NOT a claim of edge.
**Independent rebuild:** `INDEPENDENT_1328.md` must agree with `REPORT_S1_CREATIVE.md` §2 on totals within 1 % and
on Task-A membership before any code ships. If it does not, this PREREG is void.

## Mechanism
Same 5-minute opening-range breakout, same features, filters, ranking, exits and cost, applied to two adjacent
strata the production seed excludes by threshold (gap ≥ 5 % / open ≤ $30). The strata are cheaper per trade
(a third of production's per-trade edge) and more frequent (+2.3 fills/wk). Live implementation must mirror the
BT construction: the selection chain runs ONCE PER POOL (never one shared pool — cell 1,327 showed a shared pool
displaces a third of production's picks through a pool-dependent stage), the picks are unioned, and the slot cap
is shared. Production's pool, rules and picks are byte-identical to today.

## Live rules (all explicit, config-driven, default OFF)
* `orb.yaml::universe.addon_pools` — list of `{min_gap_pct, max_gap_pct, min_price, max_price}`; shipped with the two
  strata above; `enabled: false` until the dry day passes.
* Add-on picks are tagged `pool: addon_gap4|addon_p30` in `pattern_data` and Telegram (`[ORB+]` prefix) so the
  live ledger separates them from production from the first fill.
* Sizing: the same stage risk as production (R = stage R); no multiplier of any kind.
* Slot arithmetic: production picks are submitted first each day; add-ons take remaining slots in composite order.

## Pass / kill (live, pre-committed, on the ADD-ON cohort only, scored by `scripts/cadence_bar.py --live`)
* Dry day (zero orders, `[ORB+ DRY] WOULD BUY` telegrams) must show ≥ 1 add-on candidate and 0 production
  differences vs the previous day's selection observer before `enabled: true`.
* Kill: add-on cohort mean R ≤ −0.15 after 40 fills, OR add-on weekly P&L ≤ −4 R in any week, OR any production
  selection difference attributable to the add-on pools (the observer's daily parity must stay clean).
* Review at 40 add-on fills (~13 weeks): keep if cohort mean R > 0 and production cohort unchanged in mean R
  within 1 SE; otherwise disable and record the cell as a live null with its MDE.

## Not allowed
Refitting z-params/quintiles on the add-on pools; any change to production thresholds; a shared pool; enabling
without the dry day and the owner's word.
