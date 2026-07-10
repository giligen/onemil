# ORB selection & prioritization re-audit — VERDICT (2026-07-10)

Owner mandate: fresh eyes on stock picking + prioritization/multipliers
after all the July changes; take nothing for granted.

## 🚨 THE HEADLINE: production's sizing layer was inverted
The frozen adaptive mults (Q4=1.842 amplified, Q2/Q3=0.25 floored,
"Q4 empirically best across 3 splits") were fit on **15:59-exit physics
that live never traded**. Under the CORRECT 15:45 exits the structure
INVERTS and the inversion is ROBUST:
- Leave-out-stable: Q4's TRAIN mean stays NEGATIVE removing top-1/3/5
  trades; Q2/Q3 stay strongly positive (top-5 spread across ALL quintiles
  — not one monster's placement).
- Era-consistent: Q2 best per-trade in 25H1, 25H2, AND 2026; Q4≈0/neg in all.
- Old config on corrected physics: **2026 = −$21,214, negMo 9/19, MDD
  −$31K** — consistent with live's actual mediocre ramp performance.

## What was compared (full final stack: veto+PM+15:45, replica)
| config | TOT | 2026 | MDD | ex-top3 | negMo |
|---|---|---|---|---|---|
| LIVE frozen (old) | $181K | **−$21K** | −$31K | $66K | 9/19 |
| REFIT-15:45 (Q2=3.0) | $339K | +$177K | −$26K | $151K | 6/19 |
| **SHRUNK [0.5,1.5] ← SHIPPED** | **$251K** | **+$109K** | **−$18.8K** | $133K | 6/19 |
| FLAT (all 1.0) | $227K | +$60K | −$24K | $131K | 5/19 |

SHRUNK over REFIT: REFIT's extra +$85K is almost entirely top-3 monster
amplification via Q2=3.0 (ex-top3 gap only +$15K) — rejected per the
Q5-cap anti-overfit precedent. SHRUNK = the same doctrine applied
symmetrically: clip all ratios to [0.5, 1.5].

## Ranking order: NO CHANGE (audited, not assumed)
Per-quintile inversion does NOT make ordering the lever: slots rarely
bind (≤4 qualified most days), so counterfactual re-orderings move the
book <2% while Q2-first orderings DOUBLE the MDD (crowded-day Q2s are
junk — selection-bias confirmed). Current order retains best MDD.

## Also shipped: BT↔live mult parity fix
The pipeline REFIT mults per run; pre-parity that reproduced the yaml.
The 15:45 fix silently flipped the refit (BT ran Q2=3.0/Q4=0.25 while
live ran the opposite) — every BT number since 7/4 embedded a config
production doesn't trade. Pipeline now reads orb.yaml literals (refit
only as fallback). **Rebased baseline** with shipped mults: ~$251K/18mo
(prior $344,766 quotes were REFIT-mults — retire that number).

## Other audit parts (checked, no change)
- Slot count, dedup: unchanged (slots rarely bind post-veto).
- Composite features/threshold: quintile CUTOFFS retained; feature-level
  re-examination deferred — the mult correction captures the first-order
  error; feature surgery needs its own walk-forward campaign.
