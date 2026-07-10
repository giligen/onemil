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

## PART 2 (2026-07-10 evening): the owner's split hypothesis — tested to the bottom
Owner: "Q2 has monsters AND garbage — split it (Q2a/b/c), cut DD, max profit."

**The bench audit vindicated the intuition**: 17 monsters >$5K each
(~$115K: SLDP 12.9K, RGTU 8.3K...) were benched on crowded days while
Q4/Q5 seat-takers earned +$14.8K. BUT the crowded-day Q2 bench nets
−$38K/268 trades — monsters swim in a junk river.

**The split EXISTS — PM dollar volume is Q2a/b/c**: within Q2+Q3, PM$
terciles (TRAIN cuts $416K/$4.3M) are monotone in ALL THREE eras
(lo −180/−145/+23 per trade; hi +376/+223/+380). The ONLY
era-consistent within-zone separator (six price/volume features all
flip: crowding, pdr, gap, rtv, range size, vol20).

**But every conversion beyond sizing FAILS the owner's own rule
(profit up AND MDD down)**:
- low-PM filter (cut Q*c): −$21K TOT, MDD worse — the low tercile still
  hides monsters (4 found); exclusion pays capped savings for uncapped
  forfeits. Monsters-kept violated.
- 2 PM-gated extra slots (full PM data): −$2.3K, MDD worse — benched
  hot names bring balancing junk.
- PM ladder 0.75/1.0/1.5 sizing: −$5.4K, MDD −$1.6K better — a wash.
- (Earlier: reorderings flat-to-worse; 6 slots worse; EV-order doubles MDD.)

**VERDICT: the gradient is real and is ALREADY fully monetized by the
shipped binary PM mult (upsize-only 1.5×)** — the one mechanism that
harvests it with zero exclusion risk. The selection layer is at its
information ceiling; further separation requires NEW information
(point-in-time float, catalysts, tape), same frontier as BF.
Data asset: full-universe PM coverage now exists
(/tmp/orb_pm_universe_extra.csv + data/research CSV, 6,221 pairs) —
persist and append nightly.
