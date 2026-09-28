# Refuter: PREREG_1617 frame C (cells 1,621 W=3 and 1,622 W=5), the short-window confirmation entry

Verdict: **NOT REFUTED. Both cells FAIL, and the FAIL holds under every lens.** No defect I found moves either cell toward
the pass bar. The defects I did find all make the book look *better* than it is, so they only strengthen the FAIL.
Check script: `review/1617_confirm_refuter_chk.py`. It reuses the builder's own `eligibility_and_entry_w` and the
cell-1,487 loaders and changes one lens at a time. Runtime was about 2 minutes.

## 0. Calibration reproduction (is the W parameterisation right?)
Running the builder's function at W=15 reproduces RESULT_1487 to 4 decimals:
- eligible share 0.1021 / 0.0985
- calibration +0.6896 / +0.6451
- primary book −0.1287 R t −1.54 (TRAIN-H2) and −0.0450 R t −0.52 (VAL)

So the only thing that changed is W. The cohort's base outcome falls as W shortens:
- W=15: +0.69 / +0.65
- W=5: +0.42 / +0.35
- W=3: +0.24 / +0.24

The builder's "less-selected cohort" reading holds.

## 1. Window boundary and look-ahead
`fill_min` is a fractional ET minute. Bar `m` is the bar's start minute (the SIP timestamp is `t`).
- The window is `m ∈ (fill_min, fill_min+W]`, which means bars floor+1 … floor+W. These bars end exactly at the open of
  the entry bar floor+W+1.
- The base-exit test `exit_m > fill_min+W` uses integer exit bars and follows the same boundary.
- **No bar after minute F+W decides eligibility.**

There is one mild look-ahead: rows with no bar at minute F+W+1 (`no_entry_bar`: 292 at W=3, 212 at W=5) are dropped. I
entered them instead at the first later bar (median delay 1 min). They are worse than the kept rows:
- W=3 added-row mean net: −0.48 R (TRAIN-H2) and −0.56 R (VAL)
- W=3 VAL book with them included: −0.285 R, t −4.79 (the builder reported −0.254)
- W=5 VAL book with them included: −0.216 R, t −2.98

So the exclusion flatters the result.

Window sparsity: 12–17 % of primary rows pass the "no dip" test on fewer than W window bars. These rows are the worst:
- VAL W=3 sparse rows: −0.55 R
- VAL W=3 full-window book: −0.21 R, t −3.2
- VAL W=5 full-window book: −0.17 R, t −2.2

This is another flattering bias, and it cannot flip the verdict.

## 2. Ask proxy and cost decomposition (VAL primary)
| W | net | raw (ask entry, 0 exit cost) | gross at mid | entry half-spread | exit cost |
|---|---|---|---|---|---|
| 3 | −0.254 | −0.169 (t −2.97) | **−0.005 (t −0.1)** | 0.163 R | 0.085 R |
| 5 | −0.208 | −0.131 | **+0.013 (t 0.18)** | 0.143 R | 0.077 R |

At the mid with zero exit cost the TRAIN-H2 book is +0.087 R (t 1.44) at W=3 and +0.026 R at W=5.

The path after a W-minute hold has no drift at the mid. That leaves the ask proxy as the only way to flip the verdict,
and it cannot:
- The fill-instant half-spread may overstate the minute-W+1 spread.
- Even at **zero** cost the mean is about 0, far below the +0.15 R bar.

With a VAL SE of about 0.06 R, a +0.15 R book would have been detected. The null is adequate.

## 3. Run-up cost
- Entry sits a median of 101 bps (W=3) and 118–123 bps (W=5) above the level, with a p90 of about 220–260 bps.
- R″/R_base median is 0.63–0.75. The level−$0.01 stop is *tighter* than the base stop, so R″ is smaller, not larger.
- The target share is 0.28–0.31. The cost-free breakeven for a 2 R target is 0.333.
- 3–7 % of trades stop inside the entry bar.
- Runners lost are 0.4–0.9 % of base fills (42 at W=3), so they are not the driver. This confirms the builder.

## 4. Tails
| cell / holdout | dropping the best 2 days | months positive |
|---|---|---|
| 1,621 VAL | −0.297 R, t −5.55 | 0/5 |
| 1,622 VAL | −0.280 R, t −4.58 | 1/5 |

- Ex-top-1 % is −0.28 (1,621 VAL) and −0.23 (1,622 VAL).
- Winners are capped at +2 R by construction.
- The losses are broad and not tail-driven.

## 5. Kept cache-only share vs 19.5 % (a 1,487 pass criterion the builder omitted)
| cell | TRAIN-H2 | VAL |
|---|---|---|
| 1,621 | 0.309 | 0.278 |
| 1,622 | 0.317 | 0.276 |

All four are outside 19.5 ± 5 pp. RESULT_1487 was also outside on TRAIN at 0.311. This is a reporting omission, since
PREREG_1617 C says "as 1,487". It would add one more failed criterion and does not change the verdict.

## 6. Reporting nits (no verdict impact)
- **Eligible share, calibration and fills/wk.**
  - The builder counts `no_entry_bar` and `nonpositive_R2` rows as eligible. This gives calibration 0.24 and eligible
    share 0.26/0.25.
  - The rebuild counts only entered rows. This gives calibration 0.32 and share 0.23/0.22.
  - The rebuild's fills/wk is not slot-capped (27.9/41.0 against 25.4/35.0).
  - The compare step passed on net R only and did not flag these differences. Both definitions are negative-consistent.
- **The all_eligible VAL mean of −2.13 R is not meaningful.** It is driven by R″ = $0.0001 rows (IREZ 2026-03-31, ROBN
  2026-01-21, each about −1,000 R), where the ask lands within a cent of level−$0.01. The 0.5 % floor removes these
  rows from the primary book, as it should.
- **The paired Δ (−0.68 / −0.74 R) mixes R units.** R″ is about 0.65× R_base, so this is not a dollar comparison. It is
  a descriptive statistic outside the pass bar.

## Conclusion
Cells 1,621 and 1,622 FAIL the frozen bar C decisively. The FAIL holds under:
- the boundary/look-ahead audit
- the entry-bar and sparse-window corrections (both make it worse)
- a zero-cost ask proxy (gross at the mid ≈ 0)
- the tail cuts
- the cache-only criterion

Setting the scope honestly: on the 9,911 cell-1,438 fills, the no-dip confirmation entry at W ∈ {3, 5, 15} has about
zero gross drift at the mid. The ~0.2–0.25 R of entry-plus-exit cost is the whole loss. The MDE is about 0.12 R at VAL
n ≈ 800–900.
