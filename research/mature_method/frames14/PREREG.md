# frames14 — PREREG · F45 the cost model at the clocks the books trade · F43 the geometry floor map · F44 is the overnight tail selectable at 15:55

Committed BEFORE any cell is scored. Pass 14. Programme cell count entering this pass: **1,191**.
This pass declares **6 (F45) + 12 (F43) + 8 (F44) = 26 scored cells**, plus the labelled diagnostics
named below (a diagnostic is never promoted to a finding). Programme count on exit: **1,217**.

Run order is F45 → F43 → F44, as `hod_frames/FRAMES.md` recommends: F45 is pure measurement and can
move the cost number every other cell uses; F43 consumes F45's cost; F44 is built on F43's and F40's
maps rather than on an imputation.

Rails inherited unchanged from passes 6–13 and `mature_method/RUNBOOK.md`: reproduction gate asserted
in code and raising before any number is read · TEST sealed (`FREEZE.md`) · both TRAIN halves AND VAL
on every claim · day-clustered SE · count-matched permutation null wherever a green-week or a
tail-LABEL claim is made · booked cost measured per cell, never assumed · the pass-6 control rule (a
control is a NON-signal name; no detector anywhere in its construction) · availability audit on every
field with an 80 % rail · **% of price printed beside every R** · ex-top-5 % only on UNCAPPED exits
(F35's rule: a +2R bracket and a +3R scale-out are already capped, so an ex-top-5 % read on them is
meaningless) · one python process at a time, `nice -n 10`, `ulimit -v 3000000`, checkpointed · every
store outside `frames14/` opened READ-ONLY.

---

# F45 — THE COST MODEL AT THE CLOCKS THE BOOKS ACTUALLY TRADE (6 cells + the minute-of-day table)

## 1.1 The object under audit

`hod_break/score.py::build_impute` fits `IMPUTE[(price band, hour band)]` = the **median NBBO
spread as % of price** of the MEASURED sample `research/bf_zero/causal_filter/nbbo.csv`. That sample
is 15,656 rows whose `entry_m` runs **577 (09:37) → 841 (14:01)** — verified before writing this file.
The table nevertheless carries an hour band labelled `0930-0945` (fit on 09:37–09:45 only) and one
labelled `1300+` (fit on 13:00–14:01 only, and used to price every clock out to 16:00). That table
has priced most of the 1,191 cells of this programme, both live books' band tables included.

An imputation is a hypothesis about a clock it was not measured at. This frame AUDITS it: the
deliverable is a measured-vs-imputed table with the sign and size of every gap, and the list of prior
per-book numbers whose level moves.

## 1.2 The minute-of-day table (measurement, the frame's first deliverable — not a scored cell)

**Universe.** The HOD measured-NBBO population itself (2,601 symbols × 412 sessions, TRAIN/VAL only)
— the population the imputation table was fit on, so measured-vs-imputed is like-for-like and any gap
is a CLOCK effect and not a population effect. Sampling: **`SEED = 20260920`, 50 symbol-days per price
band** over `PB_LAB = ['<$5','$5-10','$10-20','$20-30','$30-50','$50-100','$100+']` (the table's own
bands), capped at the cell's size.

**Clocks, declared in full before the pull** (ET, each a one-minute window, MEAN ask−bid over the
minute — the Stage-P/frames13 convention):

`09:30` (the opening-cross minute) · `09:31` · `09:35` (**ORB's submit minute**) · `09:37` ·
`09:40` · `09:45` · `10:00` · `11:00` · `12:00` · `13:00` · `14:00` · `15:00` · `15:45`
(**ORB's flat**) · `15:55` (**HOD's and BF's flat**) — **14 clocks**.

Alpaca **SIP** consolidated quotes, the same source the imputation table and Stage P/Q used. Nothing
already on disk is refetched: `frames13/openspread.csv` (09:30/09:31/15:55 on 500 stratified
name-nights) and `P_cost/spread_rows.csv` (ORB's own 7,402 entry + 7,402 exit instants) are read as
they stand and reported beside the new pull as independent confirmations.

**Availability rail:** a clock whose coverage falls below 80 % of the sampled symbol-days is reported
as under-covered and its row is not used to move a book number.

## 1.3 The six scored cells

| id | question | the pre-committed bar |
|---|---|---|
| **E1** | **ORB @ 09:35.** Stage P measured per-trade SIP NBBO at ORB's OWN fill instants; is that confirmed, and what is the residual at 09:35 specifically? | The claim is CONFIRMED iff Stage P's fetch is at the fill instant (code-read, not assumed) and its `09:30-09:35` entry cell is reproduced from `P_cost/spread_rows.csv`. Residual = (fresh 09:35 measurement) − (Stage P's own 09:35 cell), in bps and in R at ORB's median `r_pct`. |
| **E2** | **ORB @ 15:45 flat.** ORB's exit leg is a flat 10 bps `EXIT_SLIP_BPS`; what does the 15:45 NBBO actually quote on its own names? | Residual = measured 15:45 half-spread − 10 bps, expressed in R at ORB's median `r_pct` and as a % of the 21-month book. |
| **E3** | **BF entry, measured at BF's OWN detection minutes.** The shipped Stage-2 charges a flat **50 bps**, spread-blind. Measure the SIP NBBO at the entry minute of the **56 P1 trades** and of the **896 raw regen-7 detections**, and report the distribution of BF's detection clock. | Descriptive; the bar is that the measurement exists on ≥ 80 % of each population. |
| **E4** | **The honest BF book under measured cost** — both legs (entry at the detection minute, exit at the exit minute), in **R and in % of price**, on **both TRAIN halves and VAL**. | No bar (BF is a shipped book, not a candidate). The REPORTABLE is whether the book's honest number moves by **> 10 %** vs its reference. |
| **E5** | **HOD dry @ 09:37–14:01.** Is the imputation right for the book it was fit on? | CONFIRMED iff the fresh measurement at 09:37/09:40/09:45/10:00/11:00/12:00/13:00/14:00 is within **±25 %** of the table's imputed value for the same (price band, hour band) at every clock with adequate coverage. |
| **E6** | **HOD @ 15:55.** The contract charges the EOD leg as `ratio(eod) = 0.412 × the SIGNAL-minute half-spread`. What does 15:55 actually quote? | Residual = measured 15:55 half-spread × 0.412 − charged, in R at the HOD book's median `r_pct`, applied to the 27.9 % / 33.0 % of HOD trades that force-close (frames13 §3.2). |

## 1.4 Predictions, stated before the pull

* **P45.1** The 09:30 and 09:31 clocks are **wider** than the table's `0930-0945` cell by a factor
  ≥ 1.5 (frames13 already measured 0.696 % stratified / 0.908 % liquid-common at 09:30 against an
  imputation global median of 0.345 %).
* **P45.2** 09:35 is wider than 09:37–09:45 but by less than 09:30 is — the intraday spread curve is
  monotone decreasing through the morning.
* **P45.3** 15:45 and 15:55 are **narrower** than the `1300+` cell's fit window (13:00–14:01), i.e.
  the spread curve keeps falling into the close. If so, every book that force-closes has been
  **over**-charged, not under-charged, and the direction of the error is conservative.
* **P45.4** ORB is NOT exposed: its honest book is priced by a per-trade measurement at its own
  instants, so its number moves by **< 2 %**.
* **P45.5** BF IS exposed: its flat 50 bps is spread-blind and `entry_cost_audit` already measured a
  72 bps median half-spread on the P1 names, so BF's honest number moves by **> 10 %**.
* **Falsifier of the frame:** if every clock's measurement lands within ±25 % of the table's
  imputation, the table is vindicated at every clock and F45 has found nothing.

---

# F43 — THE GEOMETRY FLOOR MAP (12 cells + the declared diagnostics)

## 2.1 The population — detector-free, twice

**Primary:** pass 6's **arm-d controls**, `hod_frames6/pd6.csv`, **288,174 keys** = a matched
NON-signal name at a random eligible minute on the PIT HOD universe (prev close ≥ $17, ADV20 ≥ 100K),
TEST cut off. No detector anywhere in the construction (the pass-6 control rule).
**Replication:** pass 12's floor keys, `frames12/floor38.csv`, **250,171 keys** (every eligible
PIT-panel non-signal name of every session at 2 random minutes).

A geometry is priced with **NO admission rule** — that is the whole point of the frame.

## 2.2 The seven geometries, each written from its shipped spec

All entered at the OPEN of the control's own bar; exit convention is `hod_frames6.common6.walk_from`'s
verbatim (priority EOD → stop → target from bar e+1; a stop fills at `min(stop, that bar's open)`
minus one slip; a target fills AT the target; EOD fills at that bar's open). Stop width is a pure
function of price, `s ∈ {2 %, 3 %, 4 %}` — so each cell is a property of the UNIVERSE and the clock,
never of a signal.

| id | geometry | spec source |
|---|---|---|
| **X1** | **+2R bracket**, flat 15:55 | HOD-break (`trading/hod_break.py`); reproduces F34 |
| **X2** | **ORB static lock** — no target; stop at `s`; when the high reaches **+1.75R** the stop moves to **+0.5R forever**; flat **15:45** | `orb.yaml exit.lock_arm_at_r 1.75 / lock_stop_r 0.5`; `frames7/c7.walk_orb` |
| **X3** | **BF R-trail, no partial** — stop at `s`; trail arms at **+1R** and ratchets on CLOSED-bar highs to `high − 1R`; flat 15:45 | `trading/bf_trail.py`; `frames7/c7.walk_bf(partial=False)` |
| **X4** | **BF R-trail + the shipped 50 % @ +2R partial**, stop → breakeven on the partial | `trading/bf_profit_partial.py`; `frames13/f41.py` re-walk |
| **X5** | **bare stop to 15:55** — stop at `s`, NO target, flat 15:55 | the null geometry: pure downside truncation |
| **X6** | **hold to next open** — no stop, no target, exit at the NEXT session's open | F40's overnight leg attached to an intraday entry |
| **X7** | **MOC** — no stop, no target, exit at the **official daily close** (`daily_bars`, the price-scale-verified join of frames13 §3.1) | F42's leg, priced unconditionally for the first time |

## 2.3 The cost, MEASURED, per geometry

The leg that pays is declared per geometry before scoring, and it is **F45's measured number, not the
imputation** (the frame pre-commits this):
* every entry is marketable → **half-spread at the entry clock**;
* a **target** fill is a resting limit → **free**;
* a **stop**, an **eod/flat** and an **MOC** fill are marketable → **half-spread at the exit clock**
  (MOC: the auction is a single-price cross → **ratio 0**, per frames13 §1.4 and F42);
* **X6's** overnight exit at the next open pays the **measured opening-minute** spread from F45's
  09:30/09:31 rows, never zero — F40's standing rule.

Unit: **% of entry price** (F31's unit), with R printed beside it using each cell's own `s`.

## 2.4 The 12 scored cells

**X1..X7 at s = 2 %, pooled over the whole entry window** (7 cells) and **X2, X4, X5, X6, X7 at
s = 3 %** (5 cells) = **12**. Declared DIAGNOSTICS, never promoted: the same geometries split by
entry-hour band (09:37–10:30 / 10:30–11:30 / 11:30–13:00 / 13:00–14:01), by price band and by ADV$
band; s = 4 %; and the pass-12 replication population.

## 2.5 The bar (pre-committed, identical for every cell)

A geometry **counts as a positive unconditional floor** iff, net of the measured cost:
1. it is **positive on both TRAIN halves (2025-H1, 2025-H2) AND on VAL**, and
2. it **survives ex-top-5 %** — applied only to X5/X6/X7, the three UNCAPPED geometries; X1 is capped
   at +2R, X2 at its lock, X3/X4 at their trail, so F35's rule forbids the read there and it is
   reported as N/A rather than as a pass.

## 2.6 Predictions

* **P43.1** X1 reproduces F34 (−0.39 % of price at s = 2 %) to within 0.02 pp. **This is also the
  reproduction gate** — if it does not, the walk is wrong and nothing else is read.
* **P43.2** No geometry is positive unconditionally on this universe. The programme's null is then
  STRUCTURAL: thirteen passes filtered signals on top of a negative floor.
* **P43.3** If one IS positive it will be **X6** (the overnight hold), because F40 measured exactly
  that premium — and it will fail the ex-top-5 % clause, because F40 measured that too.
* **Falsifier of the frame:** a geometry positive on both halves AND VAL AND ex-top-5 %. Were that to
  fire, the 1,191 signal cells were filtering on top of a positive floor and the question becomes why
  no signal ever beat it.

---

# F44 — IS THE OVERNIGHT TAIL SELECTABLE AT 15:55 (8 cells)

## 3.1 Population and the label

`frames13`'s overnight panel, declared cell **A13** (common, close ≥ $20, ADV$ ≥ $10M,
**3,310,743 name-nights**, corporate-action rail applied, TEST cut at 2026-06-01). The reproduction
gate is A13 itself: **n = 3,310,743, net +0.0306 %, gross +0.045 %, ex-top-5 % −0.1385 %** must
reproduce before any conditioner is read.

`on_pct` = adjusted `open[t+1] / close[t] − 1` in % of price, net of Reg-T margin at APR 7.0 %.
The **tail label** is `on_pct` in the **top 5 %** of its own split.

## 3.2 The conditioners — every non-downstream field the programme has built, evaluated at 15:55

All are computable from the daily panel strictly at or before the 16:00 cross of day *t*:

| id | field | definition (day *t*, all from the PIT daily panel) |
|---|---|---|
| **V1** | day's return | `close/prev_close − 1` |
| **V2** | day's range | `(high − low)/prev_close` |
| **V3** | close position in range | `(close − low)/(high − low)` |
| **V4** | rv (relative volume) | `volume / mean(volume, 20 sessions strictly before t)` |
| **V5** | dollar_frac | `close × volume / adv20` |
| **V6** | SPY day | SPY's own `close/prev_close − 1` that session |
| **V7** | gap | `open/prev_close − 1` |
| **V8** | 20-day high proximity | `close / max(high, 20 sessions ending t)` |

`wrapper/common` is **degenerate inside A13** (which is common-only) and is therefore reported as a
DIAGNOSTIC on the liquid panel with the common filter removed, never as one of the 8 cells.

## 3.3 What is scored, and the bar

Each field is cut at its own **top and bottom quintile within (split × day)** — a within-day cut, so
the selection is causal and cannot drift with the market. For each of the 8 cells the scored
quantities are: the tail rate (share of the selected subset in the top 5 % of the night distribution)
against a **count-matched permutation null on the tail LABEL** (2,000 draws, per-day tail counts
preserved — the frames11 instrument), and, **the only result that counts:**

> **the ex-top-5 % mean of the selected subset, net of margin AND of F45's measured execution
> (15:55 marketable sell-side + 09:31 marketable buy-side, the `rt_pct` frames13 measured at
> 0.368 % of price), positive on BOTH TRAIN halves AND VAL.**

A conditioner that only lifts the uncapped mean is reported as **NOT a finding** — that is what F40
already knows. An era check on **2016–2024** is printed for every cell that passes the split test.

**The short side.** If a field separates the BOTTOM tail, the short is scored the same way with two
extra charges that are measured, not assumed: `easy_to_borrow` availability from the Alpaca asset
table for the selected names, and the overnight borrow cost. A short that is not borrowable is not a
finding.

## 3.4 Predictions

* **P44.1** At least one field raises the tail RATE above its count-matched null (the tail is not
  uniformly distributed across the universe — V2 and V4 are the candidates).
* **P44.2** **No** field turns the ex-top-5 % mean non-negative on both splits. The premium is paid
  for bearing gap risk and the conditioners available at 15:55 are not the risk.
* **P44.3** The bottom tail is separated by the same fields as the top tail (they are the same
  variance), so the short is a mirror and dies on the same cap.
* **Falsifier:** a field whose selected subset is ex-top-5 % positive on both TRAIN halves and VAL
  net of the measured execution. Then the overnight premium is SELECTABLE and a new object exists.

## 3.5 The deliverable

One sentence either way, as the frame requires.

---

# 4. Multiplicity

26 scored cells this pass (**6 + 12 + 8**), all listed above before any of them was read, all printed
in the REPORT whatever they say, none selected after the fact. Programme count **1,191 → 1,217**.
Diagnostics are labelled as such in every table and are never promoted to findings.

# 5. What this pass will NOT do

No config, `orb.yaml`, `config.yaml`, checker, systemd unit, cron, cache or order is touched. No
engine file is changed. The ONE file outside `frames14/` that this pass may edit is **CLAUDE.md**, and
only the single ORB attribution sentence F41 §2.5 asked for (owner-approved direction). `hod_break`
stays `enabled: true, dry_run: true`; BF and ORB stay paused exactly as the owner left them.
