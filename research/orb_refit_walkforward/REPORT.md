# ORB weekly refit — walk-forward on the HONEST dump (2026-09-07, owner question "train weekly on the past 2 months?")

Harness: `run.py`. Every Monday from 2025-03-03 the composite's z-params, quintile cutoffs and adaptive mults are re-fit on the trailing window (strictly before the week), then the week trades through the same selection + veto stack (PDR 11, G1 + short-history, range-size, catalyst; top-3, family/super dedup, Q1 skip). Exits = static-lock candidate dump (identical physics). Stage sizing $10K / 3 / $375. **Frozen (orb.yaml literals) reproduces the honest book from 2025-03-03: $5,669** — the replay is valid.

| variant (refit what, window) | fills | total | MDD | red months | worst mo | 2025 | 2026 |
|---|---|---|---|---|---|---|---|
| frozen (live today) | 60 | 5,669 | −551 | 6 | −185 | 3,026 | 2,643 |
| **refit everything, 8 weeks** (the 2-month idea) | 68 | **4,368** | **−1,254** | 7 | −612 | 2,615 | 1,753 |
| refit everything, 13w | 66 | 5,834 | −518 | 3 | −518 | 3,884 | 1,950 |
| refit everything, 26w | 71 | 8,135 | −106 | 3 | −106 | 4,928 | 3,207 |
| refit everything, expanding | 68 | 11,041 | −161 | 4 | −161 | 3,766 | 7,275 |
| refit mults only, 26w | 60 | 4,758 | −603 | 7 | −412 | 3,561 | 1,197 |
| refit mults only, expanding | 60 | 10,491 | −347 | 6 | −277 | 3,473 | 7,017 |
| **selection only (z + cutoffs), 20w** | 70 | 6,279 | −131 | 5 | −131 | 2,824 | 3,455 |
| **selection only, 26w** | 71 | **7,588** | **−223** | **4** | −223 | 4,228 | 3,359 |
| selection only, 39w | 65 | 6,834 | −223 | 2 | −223 | 3,577 | 3,257 |
| selection only, 52w | 67 | 6,989 | −223 | 2 | −223 | 3,577 | 3,412 |
| selection only, expanding | 68 | 7,733 | −223 | 3 | −223 | 3,577 | 4,156 |

## Read
1. **The 2-month refit loses on every metric** — top-5 fills are 98% of its P&L; it is noise-fitting.
2. **"Refit everything, expanding" is one trade**: ANNA 2026-03-20 sized at $5,649 vs $1,883 frozen (the refit adaptive mult at the 3.0 cap). Mults-only refit carries almost all of it ($10,491). That is sizing amplification on a monster, not selection skill — the July whipsaw class, now with a lucky sign.
3. **Selection-only refit (z-params + quintile cutoffs, sizing mults FROZEN) is a plateau, not a spike**: every window from 20 weeks to expanding beats frozen on total (+11% to +36%), MDD (−131 to −223 vs −551), red months (2–5 vs 6) and 2026 (+$0.6K to +$1.5K). 2025 is up at 26w and above, slightly down at 20w. Pick overlap with frozen ~68%; fills 65–71 vs 60.
4. This supersedes the July audit's "never refit" (measured on the lookahead-inflated features): the honest answer is **refit the SELECTION on a rolling ≥ 26-week window, never the sizing mults** — refitting mults is where the whipsaw lives.

## Proposal (owner decision, not shipped)
Weekly (Sunday) refit of `orb.yaml::filter.features` z-params and `quintile_cutoffs` on the trailing 26 weeks of candidates, `adaptive_mults` frozen; the nightly pipeline replays the same walk-forward so BT stays parity with live (the "frozen fit is canonical" doctrine becomes "the walk-forward fit is canonical, written to yaml and logged"). Kill switch = stop the weekly job (yaml keeps the last fit). Evidence bar before ship: the same plateau on a second seed of the week phase (refit on Wednesdays) and a live parity test of one refit cycle.
