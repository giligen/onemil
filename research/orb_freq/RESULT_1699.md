# RESULT 1,699 — second layer batch on the ORB stack (base → RVOL tilt → add @+1R)

PREREG: `research/orb_freq/PREREG_1699.md` (FROZEN). Owner 10/1: "push harder." Book: production
(`analysis_results/orb_bplus_book.csv`, entered==1, n=478, recon 5 `no_bars`/473 ok... full 478 after
retry = 478 ok, 5 excluded as `no_bars`), real entry minutes via `1693_pool_exits.py.reconstruct_fill`
on **bars_sip.db** (read-only, `?mode=ro`), live-lock exit (1.75R arm → 0.5R stop). $375/unit fixed risk
except L14. "Both directions" = independent reads on TRAIN2025 and VAL2026 separately for fixed-level
layers (L9/L10/L11/L13 — nothing is fit, so nothing needs swapping); for the one *fitted* layer (L12) it
means fit-edges-on-one-half/test-on-the-other, done both ways. OOS2024H2 = 0 fills on this book (same
disclosed gap as 1687/1693/1698) — reported, never gates.

**Independent-reimplementation check (CLAUDE.md rule 1):** this script's own freshly-walked r1 matches
`1694_runners.csv`'s `A3_noTarget_liveLock` on **478/478** fills, mean|diff|=0.00000. **Discrepancy found
and resolved:** `WEEKLY_Q3_base_tilt_add.md`'s Q3 base total (+$283/13wk) undercounts by exactly $587 —
week-of-06/29 has 3 real fills (MSTX 6/29, MSTZ 6/30, CAST 6/30; present in the parity-matched
`1694_runners.csv` too) that doc reported as a zero-fill week. This cell's Q3 base = **−$304**, used
throughout below as the verified reference; the older doc's $283 should be treated as stale.

## L9 — add-level sweep (on top of base+tilt), paired ΔR vs base+tilt
| Variant | TRAIN2025 ΔR (t) | VAL2026 ΔR (t) | ex5 TRAIN/VAL | Verdict |
|---|---|---|---|---|
| add +0.5R | +0.155 (1.11) | +0.419 (2.54) | −0.096 / +0.024 | FAIL (TRAIN ex5 < −0.02) |
| **add +1.0R (reference, =1698's L2)** | +0.061 (0.32) | +0.323 (2.11) | **−0.164 / −0.045** | FAIL own ex5 bar — tail-carried |
| add +1.5R | −0.117 (−1.16) | +0.207 (1.80) | −0.322 / −0.136 | FAIL (sign flips) |
| 2 units @+1.0R | +0.121 (0.32) | +0.645 (2.11) | −0.328 / −0.091 | FAIL (worse tail than 1 unit) |

No candidate clears the ≥+0.03R/both-signed/ex5≥−0.02 bar. **Finding, not just a null:** even the
already-shipped +1R add is tail-carried under this stricter paired/ex-top-5% read in BOTH windows —
its incremental lift over base+tilt leans on the top 5% of fills (paired-lift tail check precedent).
Stack keeps add@+1R (L9 does not change the stack).

## L10 — lock re-read on the combined 2-unit position (8 combos: arm∈{1.5,1.75,2,2.5}×stop∈{0.5,1})
All 8 combos read negative-to-flat paired ΔR vs the add@+1R reference in VAL2026 (best: arm2.0/stop1.0
TRAIN +0.026/VAL −0.502; worst: arm1.75/stop1.0 VAL −0.605, t −2.24). **None join.** Stack keeps the
live 1.75R→0.5R per-unit lock.

## L12 — gapper-count tilt (day candidate count, entered-inclusive, mean 2.1/day over 300 days)
Direction A (fit TRAIN, test VAL): map low=0.5/mid=1.5/high=1.0, EV/risk gain on VAL +3.8%.
Direction B (fit VAL, test TRAIN): map **high=0.5**/mid=1.5/**low=1.0**, EV/risk gain on TRAIN +1.8%.
Ordering FLIPS between directions (low and high swap which gets 0.5×) and both EV/risk gains are below
the +10% bar. **Does not join.**

## L11 — P1 pool (idea1, gap 3–5%) as a frequency layer
Bar-walk coverage on bars_sip.db: **126/394 (32.0%)** — below the 80% rail (same gap RESULT_1690
flagged for cache.db; this read uses bars_sip.db instead, consistent with the rest of the stack, but
coverage is still low — reported, not hidden). Regime-specific half-out (50%@+1R, rest on live lock)
paired vs plain P1, on the 126 covered fills: TRAIN ΔR **−0.111** (t −0.98), VAL ΔR **−0.012** (t +0.41)
— **both negative, does not join** (contradicts the "frequency bonus" framing; a direct paired read,
which 1690/1692 never ran, says the half-out is not an improvement on this reduced population).
Tilt-transfer to P1 ("does the stack's RVOL tilt apply to P1?"): **NOT COMPUTED** — no `rvol_0935`
feature exists for the P1 population; building one needs a new 20d-avg-volume + 09:35-cum-volume fetch,
out of this cell's budget and no owner GO for a new pull. Flagged as a gap, not skipped silently.

## L13 — add @+1R on P1
P1+add vs P1 plain (same 126-fill population): TRAIN ΔR +0.208 (t 0.92, n.s.), VAL ΔR +0.012 (t −0.51,
flat). **Does not join.** Base-side "add at the chosen level" is L9, already in the stack (unchanged).

## Final stack — IDENTICAL to 1698's (base × RVOL tilt × add@+1R): no 1699 layer cleared the bar

## Q3 2026 weekly, $ (`1699_weekly_q3.csv`; full table there; fills = base/+P1)
| Wk(Mon) | 27(6/29) | 28(7/6) | 29(7/13) | 30(7/20) | 31(7/27) | 32(8/3) | 33(8/10) | 34 | 35(8/24) | 36(8/31) | 37 | 38(9/14) | 39(9/21) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| $Base | −587 | −2077 | 1015 | −842 | 2576 | −1199 | −494 | 0 | 1042 | −96 | 0 | 914 | −556 |
| $Stack(=1698=1699) | −605 | −4739 | 3215 | −450 | 729 | −1183 | −1068 | 0 | 4543 | 629 | 0 | 4000 | −654 |
| $+P1(32% cov) | −986 | −4557 | 2065 | −752 | 729 | −908 | −1366 | 0 | 4402 | 629 | 0 | 3611 | −654 |

**Quarter totals (13 wk):** Base $−304, 4/11 green, worst −$2,077, max DD $2,077. Stack $4,417, **5/11
green**, worst −$4,739, max DD $4,739. Stack+P1(low-coverage) $2,213, 5/11 green, worst −$4,557, max DD
$4,789 — P1 is net NEGATIVE in Q3 on its covered fills, lowering the union versus the stack alone.

## L14 — compounding equity, risk = 0.5% of equity, weekly, above-water rule, from $65,000 (2025-01-01)
Full period (2025-01-01 → latest fill, 86 weeks): **end equity $199,132 (3.06×), max drawdown $22,222**.
Q3 2026 alone: +$11,092 P&L, end equity $198,808, within-Q3 drawdown $11,862. Informational (scale
question only) — uses the core stack's R series (base×tilt×add), not the low-coverage P1 union.

## Caveats (read as an adversary before relaying)
P1's 32% bars_sip.db coverage makes every P1 number (L11/L13/the union row) a reduced, non-random
sample — directional only. L12's 3-point tercile ordering check is coarse. The existing +1R add's own
tail-dependence (L9) is a finding about the ALREADY-SHIPPED 1698 stack, not new code — not unshipped
here (out of this cell's scope). Single quarter, n small per cell; not a new independent edge claim.

Files: `1699_layers2.py`, `1699_layers2.log`, `1699_reads.csv`, `1699_weekly_q3.csv`, `1699_equity.csv`.
