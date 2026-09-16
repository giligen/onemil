# Stage A — the corrected cost contract, the re-gate, time bands, intraday market state

Executed 2026-09-16 per `research/fuckup_audit/PLAN.md` §4 row A. Diagnostic stage: it always completes, it claims no
book. Everything written by this stage is under `research/fuckup_audit/A/`; everything outside it was read only.

---

## 0. PRE-REGISTRATION (written before any run of this stage)

### 0.1 What will be computed

**A0 — the corrected cost contract (H7).** Re-score `research/bf_zero2/candidates3.csv` on score4's population
(`price >= 5`, `entry_m <= 841`, `r_pct >= 1`, and `range_so_far_pct >= 5` for F5-F10), for EVERY family-config x exit
(`hold`, `2r`), under score4's own book — `from trading.hod_break import run_book`,
`rows = (day, entry_m, exit_m, symbol, net, wk)`, `run_book(rows, 12, 4)` — reporting side by side:

| contract | entry charge | stop exit | 15:55 exit | +2R target exit | spread source |
|---|---|---|---|---|---|
| (a) GROSS | 0 | 0 | 0 | 0 | — |
| (b) score4 | `half` | `0.875*half` | `0.412*half` | 0 | `spread_pct` column: 1.90/1.20/0.80/0.60/0.50 % by price band |
| (c) CORRECTED | `0.25*half` | `0.875*half` | `0.412*half` | `0.875*half` | `lit_review_2026/cost_curve.csv` median NBBO spread of THIS population, per (price band x time-of-day band) |
| (c') CORRECTED, resting TP | `0.25*half` | `0.875*half` | `0.412*half` | 0 | as (c) |
| (d) = (c) + live gate | as (c) | as (c) | as (c) | as (c) | as (c), plus drop every candidate with `spread_pct / r_pct > 0.15` BEFORE the book |
| (d') = (c') + live gate | as (c') | as (c') | as (c') | 0 | as (c'), same gate |

`half = 0.5 * spread_pct / max(r_pct, 0.05)` in R units, exactly score4's definition; only the spread table and the
multipliers change. The 0.25 entry coefficient is `probe_costs.md`'s measurement: against the quote at the MOMENT of the
fill our entries pay a median 0.0 bps (46% at or below the mid), and the next-bar-open fill already contains the drift
(`bf_zero/REPORT.md` §8: the last ask of the signal minute is +5.9 bps ABOVE the next bar's open), so charging a full
half spread on top double-counts the crossing. 0.25 is the conservative quartile, not the median.
Target = `0.875*half` in (c) even though the live HOD bracket rests its take-profit leg, because the sim's target fills
on a bar CLOSE; (c') is the engine-accurate version.

Splits: TRAIN 2025-01-02..2025-12-31, VAL 2026-01-01..2026-05-31, TEST 2026-06-01..2026-09-11. **A0 reports all three
because A0 selects nothing** — it is a restatement of an already-published table (`bf_zero2/score4_tables.md` printed
TEST for all 52 cells on 9/16, so TEST is already burned for this exact grid). Selection happens in A1 and reads TEST
only after G2 is frozen in writing.

Per cell: n, trades/week, gross mean R, score4 net, corrected net (c), corrected net (c'), corrected+gate net (d),
t of the corrected net, WR, weekly R, % weeks green, worst week, exit mix.

**A1 — the re-gate (H10).** PLAN §1's gate applied to contract (c):
- **G1 (TRAIN):** mean net R > 0 AND t >= 2.0 on the booked trades AND >= 5 trades/week.
- **G2 (VAL):** mean net R > 0 AND t >= 1.0 AND >= 55% of weeks green. The bar is raised by 1 SE of weekly R for every
  10 cells that passed G1.
- **G3 (TEST):** read ONCE, only for G2 survivors, reported whatever it says, week by week.
- Economic bar, reported and not gated: >= 3R/week at 4 slots.
- Every G2 survivor also gets: tail test (top 1% and top 5% of booked trades removed; winners capped at +3R), a
  per-month table, and a permutation p — day labels shuffled within the split 500x, the null taken as the MAX weekly R
  over ALL cells of the stage, so the p is search-adjusted.

**A2 — time bands (H5).** Base families `F8 {"N":5}`, `F8 {"N":15}`, `F8 {"N":30}`, `F6 {}`, `F1 {"P":0.12}` (F5 is
dropped as a base family — gross-negative at zero cost, PLAN §3 — but `F5 {"K":5,"X":0.04}` is carried as a reference
row), contract (c), the book re-run with entries restricted to `entry_m >= 600` (10:00), `entry_m >= 630` (10:30), and
`entry_m < 600` (the 09:30-10:00 window only). Same columns as A0. TRAIN and VAL; TEST not read here.

**A3 — intraday market state (H3+H5).** Entries `entry_m >= 600` only. Booked trades split by
(i) the IWM return from its 09:30 open to the trade's ENTRY minute, (ii) the same for SPY (`etf_1min.db`), in terciles
(cut on TRAIN, applied to VAL) and by sign; and (iii) breadth-so-far — the share of THAT day's scoring-population
candidates that signalled at an EARLIER minute and had `dist_open_pct > 0`. **(iii) is an approximation and is labelled
as one**: it is breadth inside the study's own candidate population, not the market's, and each candidate's
`dist_open_pct` is measured at its own entry minute, not at the split minute. Report the TRAIN->VAL sign agreement and
the cell count. Nothing here is adopted in Stage A; per PLAN §3 H3 the decision rule for a day/state filter is: the
excluded bucket must be negative on TRAIN AND VAL, the kept bucket's mean net R must improve by >= 0.05R, and the
bucket must be defined by data available at 09:30 (or at the entry minute for an intraday index return).

**A4 — one-page summary at the top of this report**, with the power (SE of mean R, and the minimum detectable effect
`MDE = 2.8*SE` for 80% power at alpha 0.05 two-sided) beside every headline number, and the PLAN §1 phrasing rule:
never "no edge exists"; always "not detectable in THIS universe / window / book / cost, smallest effect visible X".

### 0.2 Cells I will look at (declared in advance)

| block | cells |
|---|---|
| A0 | 26 family-configs x 2 exits x 6 contracts (a, b, c, c', d, d') x 3 splits — 52 decision-relevant cells (contract (c), TRAIN) |
| A1 | the same 52 under G1; G2 only for G1 survivors; TEST only for G2 survivors |
| A2 | 6 keys x 2 exits x 3 time windows = 36 |
| A3 | 4 keys x 2 exits x (2 indices x [3 terciles + 2 signs] + 3 breadth terciles) = 4 x 2 x 13 = 104 bucket-cells |
| total declared | 52 + 36 + 104 = 192 new cells, on top of the program's 52 + probe_stops' ~100 + probe_days' 156 |

### 0.3 What would make me say the corrected contract changes the verdict

A family-config x exit whose corrected (c) TRAIN mean net R is > 0 with t >= 2 and >= 5 trades/week. If none exists, the
reported conclusion is the closest miss per family with its own MDE, and the corrected contract is still adopted for
Stages B-E (it is the more accurate cost model whether or not it flips anything).

(results appended below after the runs)
