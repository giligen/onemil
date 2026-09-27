# REBUILD_1591 — independent rebuild of PREREG_1567 cells 1,591–1,598 (Amendment 2 / 2a)

Built from `PREREG_1567.md` prose + Amendment 2 + Amendment 2a **only** — did not open `cell_1591.py`,
`test_cell_1591.py`, `cell_1591_cycles.csv`, `cell_1591_monthly.csv`, `RESULT_1591.md`, or the v1 `cell_1567.py`.
Code: `research/options_vrp/rebuild_1591.py`. Cycle-level output: `research/options_vrp/rebuild_1591_cycles.csv`
(761 rows = 8 cells × up to 131 (monday, delta_target) legs, see `void`/`skip`/`pnl` columns). New tick-trade cache
built for this rebuild: `research/options_vrp/opt_cache/ticks_rebuild/ticks_rebuild.db` (1,021 (symbol,date) pairs
fetched, 671 ok / 350 absent / 0 errors — LOST=0, log `fetch_1591.log`).

## Disclosed methodology (say-so clause — read before trusting any number below)
1. **Strike selection** (which strike is nearest target Δ) used the existing `option_minute_entry.parquet` bar
   close nearest 10:00 ET (already cached from the v1 fetch) as the IV-bisection input, not a fresh tick fetch
   across the whole $1 strike grid every Monday (that would be an order of magnitude more requests than pricing
   the two legs actually traded). The **fill price** used for every dollar of P&L is a real tick trade, freshly
   fetched into `ticks_rebuild/` for exactly the legs this rebuild selected.
2. The **IV gate** (45-DTE ATM IV ≥ 15% at 10:00) uses the same bar-close proxy — a go/no-go filter on the week,
   not a fill price.
3. **Management-A daily marks** (which day triggers the 50%-credit / 2×-stop / 21-DTE close) use the cached
   `option_daily.parquet` closes, per Amendment 2a; only the actual **exit** tick trade (next session,
   10:00:00–10:00:30, fallback = that session's daily open, counted) was freshly fetched.
4. **Monthly P&L** is realized-basis: a cycle's full P&L books to the calendar month of its **exit** (management-A
   close, or expiry under B); calendar months with zero exits inside the split's own [min, max] exit-month span
   are included at $0.
5. **Fees**: $0.03/contract charged once at entry and once at exit (two transactions per PREREG's per-contract
   language).
6. **Sizing**: contracts = `floor((B/6) / ((W−credit)×100))`, B = $6,500 (10% of $65,000), asserted before every
   open; W=$10 fixed for both Δ per Amendment 2's reduced grid.

## Grid executed
Δ ∈ {0.20, 0.30} × M ∈ {A, B} × G ∈ {none, ivgate ≥15%} = 8 cells, numbered 1,591–1,598 (Δ outer, then M, then G) —
same nesting order as the PREREG. 131 entry Mondays had a resolvable expiry + priced chain (of 133 calendar
Mondays in range; 2 had no entry-minute data in the existing cache and were skipped, logged).

## TRAIN table (2024-02-05..2025-06-30 entries)
| cell | Δ | M | gate | n cycles | green % | monthly Sharpe | mean ret/mo on B |
|---|---|---|---|---|---|---|---|
| 1591 | .20 | A | none | 72 | 52.9% | −0.15 | −0.11% |
| 1592 | .20 | A | ivgate | 29 | 54.5% | 2.05 | 0.84% |
| **1594** | **.20** | **B** | **ivgate** | **29** | **72.7%** | **1.16** | **1.80%** |
| 1593 | .20 | B | none | 72 | 82.4% | 0.58 | 1.38% |
| 1595 | .30 | A | none | 72 | 66.7% | 0.03 | 0.03% |
| 1596 | .30 | A | ivgate | 29 | 63.6% | 0.66 | 0.72% |
| 1597 | .30 | B | none | 72 | 83.3% | 0.62 | 1.69% |
| 1598 | .30 | B | ivgate | 29 | 66.7% | 0.68 | 1.53% |

**TRAIN selection** (highest monthly Sharpe subject to n≥12 and green%≥55%): 1591 and 1592 fail the 55% green
floor; among the rest, **cell 1594 (Δ0.20, hold-to-expiry, IV gate ON) wins on Sharpe (1.16)**.

## VAL table (2025-07-07..2026-08-17 entries), selected cell 1594 highlighted
| cell | n cycles | green % | monthly Sharpe | mean ret/mo on B | worst month |
|---|---|---|---|---|---|
| **1594** | **30** | **83.3%** | **1.05** | **1.61%** | **−$883** |
| 1592 (Δ neighbour-M) | 30 | 66.7% | −0.28 | −0.26% | −$632 |
| 1598 (M neighbour-Δ) | 30 | 91.7% | 1.75 | 2.71% | −$813 |
| 1593 | 59 | 92.9% | 1.12 | 2.67% | −$1,663 |
| 1597 | 59 | 92.9% | 1.42 | 3.64% | −$1,702 |
| 1591 | 59 | 66.7% | −0.18 | −0.21% | −$828 |
| 1595 | 59 | 66.7% | 0.43 | 0.54% | −$808 |
| 1596 | 30 | 69.2% | 0.41 | 0.37% | −$549 |

## Pass bar on the selected cell (1594), VAL
| test | value | bar | result |
|---|---|---|---|
| mean monthly return on B | **1.61%** ($105/mo) | ≥ 4% | **FAIL** |
| monthly Sharpe | 1.05 | ≥ 1.0 | pass |
| green months | 83.3% (10/12) | ≥ 60% | pass |
| ex-top-5% of cycles positive | mean $57.05 (n=19 realized, 1 excluded) | >0 | pass |
| ex-top-5% of months positive | mean $70.37 (1 of 12 excluded) | >0 | pass |
| TRAIN same sign | TRAIN +1.80%, VAL +1.61% | same sign | pass |
| worst month ≥ −B | −$883 ≥ −$6,500 | assert | pass |
| max drawdown ≤ 1.5B | $883 ≤ $9,750 | ≤ 1.5B | pass |
| beats SPY buy-and-hold on B per unit of drawdown | option $1,255 / $883 dd = **1.42**; SPY $1,592 / $665 dd = **2.39** | option ≥ SPY | **FAIL** |
| neighbour check: other Δ (1598) same-signed | VAL +2.71% | same sign as 1594 | pass |
| neighbour check: other M (1592) same-signed | VAL **−0.26%** | same sign as 1594 | **FAIL** |

**Verdict: FAIL as rebuilt** — three of eleven checks fail, including the primary magnitude bar (return is
0.40× the 4% floor) and the M-axis neighbour check (holding to expiry earns a positive edge at Δ0.20 but
managing the same legs actively (A) loses money — the "cell" is not a robust corner of the grid, it is one point
that happens to avoid a bad exit-timing choice).

## The rail this rebuild's own numbers raise, before either of the above
**VOID share on the selected cell (1594) is 35.6% overall / 36.7% on VAL** (missing an entry tick trade for the
short or the long leg within 10:00:00–10:05:00 ET on the entry Monday) — **more than 3× Amendment 2a's own 10%
VOID rail, which reads: "> 10% → the cell is VOID."** By that rule 1594 is void on its own construction terms
independent of the pass-bar arithmetic above, and every management-A cell (1591/1592/1595/1596) carries a similar
or worse illiquid-leg problem (their VOID entries feed into the `EXPIRY_INTRINSIC`/`mgmtA` exit-day bookkeeping
even when the entry itself never printed, which is why several TRAIN win rates and Sharpes for the A cells look
erratic — n_cycles counts every Monday attempted, not every Monday that actually got a fill). This mirrors the
v1 judge's finding almost exactly: Δ0.20–0.30, 45-DTE, deep-OTM SPY puts trade too thinly in a half-second-to-
five-minute entry window for tick prints to reliably price both legs — the mechanism (index put insurance is
overpriced on average) may still be real, but **this fetch/pricing method cannot execute it at this Δ/DTE/window
combination**, exactly the failure mode Amendment 2a was written to fix for a different reason (no OPRA quotes)
without changing the underlying liquidity problem.

## Consequence
FAIL, on two independent grounds: (1) the pass bar itself (mean return 1.61% vs 4% required, and the option
book's drawdown-adjusted return is worse than passively holding SPY on the same capital); (2) the VOID rail on
the selected cell's own entries (35.6%, over 3× the 10% ceiling) — the tick-trade pricing method this rebuild
(and, per Amendment 2a, the frozen v2) was built on cannot fill Δ0.20–0.30 45-DTE SPY puts often enough inside a
5-minute entry window to trust the P&L it produces. Per PREREG_1567's own FAIL clause: defined-risk premium
selling on this ladder does not clear the bar a second time; a further amendment would need either OPRA quotes
(not on this data plan) or a materially wider entry/exit pricing window before more capital-facing numbers are
produced from this population.
