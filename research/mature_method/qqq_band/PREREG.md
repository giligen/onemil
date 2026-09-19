# PREREG — candidate #2 of `research/mature_method/RUNBOOK.md`: the QQQ noise-band sleeve (M6)

Written and committed **before any cell below was scored**. Everything here is RESEARCH: no
`config.yaml`, no `orb.yaml`, no order, no service, no cron is touched. Stores opened read-only.

## 1. The book under test

`research/lit_review_2026/RESULTS.md` rows 1/1b — the Zarattini-Aziz intraday noise-area rule on
**QQQ**, at the LIVE fill convention already established by `research/fuckup_audit/Q/REPORT.md`
(scenario C): decide on a CLOSED 1-min bar, market order, fill at the NEXT bar's open.

Prose spec (from `Q/REPORT.md` §2, unchanged; all times ET, `k` = minute index from 09:30):

1. `O0` = open of today's 09:30 bar. `Cp` = prev session's 15:59 RTH close.
2. `sigma[k]` = mean over the previous 14 sessions (≥ 10 present, strictly before today) of
   `abs(close(k)/open_0930 - 1)`.
3. `UB[k] = max(O0,Cp)*(1+VM*sigma[k])`, `LB[k] = min(O0,Cp)*(1-VM*sigma[k])`, VM = 1.
4. `VWAP[k]` = cumulative RTH volume-weighted price 09:30..k.
5. Decisions ONLY at `k ∈ {30,60,...,360}` = 10:00,10:30,…,15:30, on that bar's CLOSE.
   Flat → long if close > UB[k]; short if close < LB[k].
6. Stop, checked at the same 12 minutes: long exits if close < max(UB[k],VWAP[k]); short exits if
   close > min(LB[k],VWAP[k]). On a stop the OPPOSITE side may open at the same check.
7. Flat at the close (15:59 bar close).
8. Sizing: 1x = equity/O0 shares.

## 2. Splits (the ETF store's own, as the July program set them)

| split | window | status |
|---|---|---|
| **TRAIN (= IS)** | 2016-01-04 → 2023-12-31 | choose here |
| **VAL (= OOS-A)** | 2024-01-01 → 2025-12-31 | confirm here |
| **TEST (= OOS-B)** | 2026-01-01 → end of store (2026-09-15) | **SEALED — `FREEZE.md`** |

**Honest disclosure, written before scoring**: the July program and `Q/REPORT.md` reported
OOS = 2024-01→2026-09 as ONE window and published a 2026-YTD row for the BASE cell (`Q/step4_years.csv`:
2026 YTD +3.45 bps/traded day, t 0.64). So TEST is **not virgin for the base cell**. It IS virgin for
every new cell below (the frontier ladder scored on green weeks, the null, the additivity stack), and
it is opened only if a cell clears G1 **and** G2 in §5. This is stated as a limitation, not repaired.

## 3. Cells declared (scored in this order; nothing else is a decision cell)

Base:
- **B0** — the spec above, VM 1.0, semi-hourly 10:00–15:30, VWAP/band stop, flat 15:59 close.

There is no gate cascade in this book. The "gates" are the four knobs the runbook names for this
candidate. Single-knob ladder:

| id | knob | value |
|---|---|---|
| V1 | band multiplier VM | 0.8 |
| V2 | band multiplier VM | 1.2 |
| V3 | band multiplier VM | 1.5 (the paper's own FAQ "optimum") |
| A1 | band anchor | open only (drop the prev-close leg) |
| T1 | decision cadence | every 15 min, 10:00–15:45 |
| T2 | decision cadence | every 60 min, 10:00–15:00 |
| T3 | decision cadence | every 1 min from 10:00 (**structural ceiling**) |
| W1 | new entries only at k ≤ 150 (12:00); stops keep running |
| W2 | new entries only at k ≤ 240 (13:30) |
| W3 | first decision at 11:00 (k ≥ 90) |
| H1 | hold rule | stop = opposite band only (the paper's BASE model, no VWAP/current-band trail) |
| H2 | hold rule | no stop at all — first signal, hold to the flat |
| H3 | flat at 15:30 instead of 15:59 |
| H4 | flat at the 15:59 bar's OPEN (the exit our order types can actually place — see §6) |

Declared combined points (3, no others):
- **C1** = V2 + W1 (fewer, wider-confirmed signals, morning only)
- **C2** = V2 + H4 (the best single knob at the realistic exit)
- **C3** = T3 + V1 (the maximum-frequency corner = the structural ceiling of this family)

18 decision cells × 2 read splits = 36 cell-instances. Prior cell count on this sleeve is **66
variants / 89 cell-instances** (`H/QQQ/REPORT.md` §6); this stage takes the running total to
**84 variants / 125 cell-instances** and the multiplicity statement in the report uses that number.

Reported but NOT decision cells (slices/rescalings of one rule): leverage rows (1x/2x are linear
rescalings), the tail treatments, the per-year rows, the additivity tables, the availability audit.

## 4. Metrics, fixed now

- **PRIMARY — % green WEEKS** over EVERY market week in the split (W-FRI). A week with no traded
  day counts FLAT and stays in the denominator.
- Then: longest red streak (weeks), worst week, % green months, MDD.
- **Dollars beside every ratio**, at two sizes, both stated: **unlevered $60,000** (the figure
  `Q/REPORT.md` and `H/QQQ/REPORT.md` used) and **2× Reg-T = $120,000 notional**. 4× day-trading
  buying power exists for a PDT account ≥ $25K and the book is flat at every close, so 4× is
  *available*; 2× is the rail this pre-registration reports and nothing above it is quoted.
- Total P&L is TERTIARY.
- Gross (no cost) is reported for every cell BEFORE net (runbook step 2).

## 5. The two bars, pre-committed

**Claim bar.** G1: TRAIN net mean > 0 with **t ≥ 2** on traded days. G2: VAL same sign **and
≥ 55 % green weeks**. TEST is opened once, only for cells passing G1 ∧ G2, and reported whatever it says.

**Live-exploration bar.** (a) positive point estimate on green weeks AND on dollars at live size in
BOTH read splits; (b) a stated mechanism; (c) bounded downside with a pre-committed stop;
(d) **resolution inside a quarter at the book's own frequency** — computed, not asserted.

## 6. Adaptations this candidate forces (declared, not skipped)

1. **Cost.** Not a band table: QQQ's quoted spread is ~1 cent on ~$500–600 = ~0.17 bp full /
   ~0.09 bp half. It is MEASURED here — a random sample of Alpaca SIP NBBO at the exact decision
   instants of the book's own legs — and charged per leg, replacing the 0.5 bp assumption of
   `Q`/`H`. Auction vs intraday: **every entry and every intra-day stop is an intraday market order**
   (10:00…15:30 + the next bar's open); only the flat can be an auction print.
2. **Order types.** `data_sources/alpaca_client.py` uses `TimeInForce.DAY` at every submit site
   (7 of 7) and never `CLS`/`OPG`; there is no MOC/LOC path in this repo. The report must say whether
   that blocks the published construction. Cell **H4** exists to price the answer.
3. **Gates → the four knobs.** §3 is the frontier; there is no cascade to map, and the report says so.
4. **Additivity is the point.** Week-level return correlation of the sleeve with the live ORB path
   (`research/orb_gates2/book_G3_meas.csv`, `_sized_pnl`) and the BF path
   (`research/bf_frequency/runs/VOL_OFF.csv`, `pnl × 0.075`), plus the COMBINED weekly table with the
   sleeve stacked on. Pre-committed reading: a sleeve that **raises combined green-week % and does not
   worsen the worst combined week** is additive; one that only raises variance is not. Descriptive,
   not a selection cell.

## 7. Null

Count-matched permutation: for each cell × split, shuffle that cell's own daily P&L across its own
weeks with each week's **traded-day count held fixed**, 2,000 draws, seed 20260919 → the [p5, p95]
band on green-week %. A cell inside its band bought its week shape with trade count, not skill.

## 8. Rails

One python process, `nice -n 10`, `ulimit -v 3000000`. `research/lit_review_2026/etf_1min.db`
read-only. No Databento pull (price only, if it would decide anything). Independent blind rebuild
from the §1 prose by a second agent before any recommendation stronger than STAY DEAD.

## 9. Possible verdicts

SHIP-TO-DRY (with what a dry run would need — there is no engine for this book) · STAY DEAD ·
NOT DECIDABLE with what is on disk (with the data that would decide it, priced).
