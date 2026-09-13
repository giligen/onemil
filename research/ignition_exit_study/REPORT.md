# Ignition exit-design study — 2026 (2026-09-13)

Owner: "relying on 3 monsters is not a strategy." Question: is there an exit with positive per-trade expectancy EXCLUDING R>=2 trades? Same entries/bars/friction as capsim (DAY_DOLLAR gate retired), only the exit walk changes; V0 reproduces capsim to 4 decimals (0 reason mismatches). Pre-committed bar: meanR ex-monster >= +0.05 AND >= 5/8 months green AND total > V0; passers re-run on 2025 (in progress).

## Whole book (4,757 trades): NO variant passes
Best ex-monster meanR is +0.004 (partial+trail); every variant that protects the tail keeps meanR ~0, every variant that cuts early destroys the tail and the total. The +10% cross without a cohort is not a strategy under any of these exits — that is settled.

## Complex-confirmed cell (710 trades): the hold is the problem, not the entry
| variant | meanR | WR | P&L | meanR ex-monster | P&L ex-monster | monsters | green | worst mo |
|---|---|---|---|---|---|---|---|---|
| V0 hold to close (live) | +0.093 | 52% | 137K | +0.036 | 54K | 16 | 5/8 | −49K |
| V2 partial 50% @1R, BE | +0.099 | 53% | 138K | +0.091 | 128K | 3 | 6/8 | −39K |
| **V6 exit at trigger+60 min** | **+0.141** | **55%** | **178K** | **+0.104** | 127K | 11 | 5/8 | **−26K** |
| V7 partial@1R + cut30 | +0.110 | 28% | 140K | +0.102 | 130K | 3 | 6/8 | −18K |

Read: the complex-confirmed entry has real per-trade edge (+0.10R on 700 trades without a single monster) that the hold-to-close exit gives back by the bell. A 60-minute hold keeps the intraday move, improves the total by 30%, halves the worst month, and keeps 11 of 16 monsters. Passes the bar on 2026. **Not a strategy until 2025 H2 and H1 agree** — running.

## Out of sample (2025 H1, H2) — verdict on the complex-confirmed cell (2026-09-13 evening)
| exit | era | n | meanR | WR | P&L | meanR ex-monster | monsters | green | worst mo |
|---|---|---|---|---|---|---|---|---|---|
| V0 hold to close | 25H1 / 25H2 / 2026 | 177 / 202 / 710 | +0.22 / +0.10 / +0.09 | 51 / 52 / 52% | 95K / 37K / 137K | +0.01 / −0.01 / +0.04 | 12 / 9 / 16 | 5/6, 4/6, 5/8 | −19K / −11K / −49K |
| **V2 partial 50% @ +1R, stop → breakeven** | 25H1 / 25H2 / 2026 | same | +0.15 / +0.12 / +0.10 | 53 / 55 / 53% | 64K / 48K / 138K | **+0.08 / +0.11 / +0.09** | 5 / 1 / 3 | 5/6, 4/6, 6/8 | −19K / −4K / −39K |
| V6 exit at trigger+60 | 25H1 / 25H2 / 2026 | same | +0.06 / +0.17 / +0.14 | 41 / 58 / 55% | 31K / 74K / 178K | +0.03 / +0.13 / +0.10 | 3 / 3 / 11 | 3/6, 5/6, 5/8 | −12K / −2K / −26K |

- **V6 (60-minute hold) FAILS**: it passed 2026 and 25H2 but collapses in the monster-rich 25H1 (WR 41%, 3/6 green, a third of the total). Regime-dependent. Rejected.
- **V2 (partial @1R + breakeven) is the only exit with positive per-trade expectancy WITHOUT the tail in all three eras** (+0.08 / +0.11 / +0.09 R, WR 53–55%). It gives up most monsters (12→5, 9→1, 16→3), so its total is 68% of V0's in 25H1, +29% in 25H2, equal in 2026, and its worst months are equal or better. It fails the "total > V0" leg in 25H1 only — the tail-vs-consistency trade-off the owner already chose on BF.
- Whole book (no cohort gate): nothing passes in any era.

**Relaunch candidate (owner decision)**: complex-confirmed entries only, resting orders at the level sized AT fill, exit = 50% at +1R with stop to breakeven and the remainder on the existing lock/EOD rule. Expectation at the model size: ~+0.09 R/trade on ~85 candidates/month in 2026 (live cohort rate is far lower — 2 fills in 4 weeks). This is a consistency book, not a monster book.

## Stop tightening (owner 9/13: "R:R should be 1:2 — move risk to 0.5R?") — `run_stop.py`, three eras, complex-confirmed cell, dollars at equal $ risk
| stop | exit | 25H1 P&L / ex-tail | 25H2 P&L / ex-tail | 2026 P&L / ex-tail | stop-outs (25H1/25H2/2026) |
|---|---|---|---|---|---|
| **1.0R** | **partial @+1R, BE** | **64K / +32K** | **48K / +43K** | **138K / +128K** | 24 / 29 / 91 |
| 0.75R | partial @ old +1R | 49K / −2K | 60K / +31K | 128K / +69K | 48 / 53 / 187 |
| 0.5R | partial @ old +1R (= +2R new) | 31K / −49K | 56K / −11K | 99K / −118K | 90 / 85 / 310 |
| 0.5R | hold | 46K / −66K | 53K / −35K | 104K / −162K | 90 / 85 / 310 |

**Verdict: NO to 0.5R.** Total dollars: 2025H1 −52%, 2025H2 +16%, 2026 −28% vs the 1.0R partial book; and in ALL three eras the tighter stop makes the book tail-only (ex-tail negative everywhere). Stop-outs 3.4×; the structural low is doing real work — the tighter stop is hit by noise before the 2:1 payoff arrives. Per-unit-risk looks better only because a doubled position turns the same move into 2× R, and the position caps prevent deploying that risk (total $ falls). 0.75R is a strict middle: worse than 1.0R ex-tail in every era. The 1.0R stop + partial@1R/BE remains the only cell positive without the tail in all three eras.
