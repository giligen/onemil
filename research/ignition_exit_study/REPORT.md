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
