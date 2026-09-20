# PREREG — frames12 (pass 12): F39 the multi-day high · F38 the last hour · F37 the 5-/15-minute bar

**Committed before any cell of F39, F38 or F37 is scored.** Nothing in this file is changed after a
number is read; an amendment, if one is needed, is a separate commit that names what it changes and
is made before the cell it governs is read.

Programme cell count before this pass: **1,130** (`frames11/REPORT.md` §4). This pass declares
**11 (F39) + 10 (F38) + 10 (F37) = 31**, taking the programme to **1,161**. Every declared cell is
scored and printed whatever it says; nothing is selected after the fact.

TEST is sealed — see `FREEZE.md`.

---

## 0. Why these three objects and not a 1,131st filter

`frames11` F34 priced the *pond*: the unconditional long bracket on the PIT HOD universe, at a fixed
2/3/4 % stop, 09:37–14:01, is **negative in 39 of 39 rows on GROSS** — best gross cell −0.051 % of
price, measured cost 0.20–0.31 % of price. No admission rule written on that population, at that
clock, on that bar, can be positive net. So a genuinely new object must change at least one of
**population**, **clock** or **bar size**. F39 changes the population, F38 the clock, F37 the bar.

`frames11` F35 fixed the second rail: **ex-top-5 % is applied ONLY to an uncapped exit.** On a
+2 R-capped book it is a mechanical `0.05 × 1.98 ÷ 0.95 = −0.104 R` toll, not a tail test. In this
pass the clause is therefore evaluated on the **bare-stop**, **static-lock** and **next-open** exits
(uncapped) and **NOT** on the +2 R exits; on a capped cell the honest form `net > 0.05 × cap ÷ 0.95`
is printed in its place, and labelled as such.

`frames10` F31 fixed the third: **every number is reported in % of entry price BESIDE R.**

`hod_frames6` F20 fixed the fourth: **every new object gets its own matched non-signal control and
its own same-name-day-later-minute control**, and the pass-6 causality rule holds — *a control
minute is never drawn BEFORE the signal minute on a day known to signal*, because day membership is
then chosen with information the trader did not have.

---

## 1. The bar every cell is judged against (identical for all three objects)

A cell CLEARS only if **all** of:

1. **positive weekly $** at $100 risk on **both** TRAIN and VAL, and
2. **green weeks ≥ 50 %** on **both** splits, at
3. **≥ 10 trades a week** on both splits, and
4. **day-clustered t ≥ +2** on TRAIN, and
5. **both TRAIN halves (H1 < 2025-07-01, H2 ≥) same-signed positive** on gross.

A cell that clears goes to **SHIP-TO-DRY** with the exact engine diff. If none clears, the verdict
is **STAY-DRY** plus the per-object MDE and the next three frames.

Reported for every cell, no exceptions: **gross and net in R and in % of entry price**; the **cost
booked per cell from its own exit mix** (never carried from another cell); the four-row placebo
decomposition; green weeks, longest red streak, worst week, weekly $ at $100 risk, trades/wk, MDD;
both TRAIN halves; VAL; day-clustered t beside the iid t; the 2,000-draw count-matched permutation
null on green weeks; MDE; the imputed-spread share; ex-top-5 % **on uncapped exits only**.

### 1.1 Cost, per exit leg — declared before any cell is walked

`cost_R = 0.5·sp% / r% + 0.5·sp%/r% · ratio(exit)`, the programme's own model, with the
**pre-existing** `RATIO` extended by the two auction legs this pass introduces:

| exit leg | ratio | why |
|---|---|---|
| `target` (+2 R resting limit) | 0.000 | inherited — a resting limit pays no spread |
| `eod` (15:55 force close, marketable) | 0.412 | inherited |
| `stop` / `lock` (marketable) | 0.875 | inherited; the lock leg is a stop and is charged as one |
| **`moc`** (closing auction) | **0.000** | RUNBOOK step 3: an auction-executed trade pays no quoted spread |
| **`nextopen`** (next session's opening auction) | **0.000** | same rule, opening auction |

`sp%` is the measured NBBO spread where one exists for that (day, symbol, minute) and otherwise the
programme's imputation table (median measured spread % by price band × hour band). The **imputed
share is printed per cell**; a cell whose imputed share is 100 % is still scored, and said so.

### 1.2 The placebo decomposition — four rows under every object

| row | what |
|---|---|
| **universe bound** | the unconditional bracket on the PIT panel at the object's own clock (F34's map, extended by this pass where F34 did not walk it) |
| **matched non-signal** | the 10 nearest non-signal symbols of the same session (matched on prior close, ADV20, wrapper-vs-common), entered at the **signal's own minute**, same R geometry |
| **same name-day, later minute** | up to 10 minutes on the SAME symbol-day **strictly after** the signal minute, same R geometry (pass-6 rule: never before) |
| **the signal** | the object itself |

### 1.3 Obtainability and fill (RUNBOOK step 4 / CLAUDE.md rail 1b)

Every entry is the **next bar's open under a cap**: `fill iff next_open <= level × 1.006`. A signal
whose next open gaps through the cap is a **non-fill**, dropped, and the non-fill share is printed.
No touch-fills anywhere. Stops fill at `min(stop, that bar's open) × (1 − 0.001)`; targets fill at
the target; force closes at that bar's open. This is `hod_frames6.common6.walk_from`'s convention and
the walker is required to reproduce `book6.rr` on 120 booked trades to machine epsilon before any
cell is read (the pass-6 R3 rail, asserted in code, raising).

### 1.4 Price-scale check (CLAUDE.md rail 3) — mandatory, F39 and F38 and F37

F39's levels and F38/F39's auction fills come from a **daily** file (`research/bf_zero/universe.csv`,
Databento) while every bar is **intraday** (Alpaca/SIP). For every scored trade the ratio
`daily_close / intraday_last_close` on the signal day is computed; a symbol-day with
`|ratio − 1| > 1 %` is **dropped** and the dropped share is printed per object. A pass whose drop
share exceeds 5 % has its conclusion re-read as a data question first.

### 1.5 Availability rail

Every object states its bar-join coverage. **Below 80 % the cell is a diagnostic, not a cell**, and
that is decided by the rail before the number is looked at (the pass-6 arm-c precedent).

---

## 2. F39 — THE MULTI-DAY HIGH (a different POPULATION). 11 cells.

### 2.1 The object

**Population**: the PIT daily panel `research/bf_zero/universe.csv`, `prev_close >= $17`,
`ADV20 >= 100,000`, test tickers and names absent from `daily_bars` excluded. This is F34's own
universe rule, and it is **not** conditioned on the day having moved — which is the whole point:
every one of 1,130 cells so far lived on a population pre-screened to `day_high >= open × 1.05`.

**Level**, from prior sessions only (availability: the level uses daily bars strictly before the
signal day, never the signal day's own high):

* `H5` = max daily high of the prior **5** sessions;
* `H20` = max daily high of the prior **20** sessions;
* `H252` = max daily high of the prior **252** sessions (the 52-week high).

**Signal** = the **first 1-minute bar whose CLOSE exceeds the level**, entry at the **next bar's
open under the 0.6 % cap**, in the window **09:37–14:00** (entry minute `<= 841`), with the shipped
cascade applied unchanged: `rv_profile >= 1` (cumulative volume ÷ ADV20 × clock fraction, the
engine's own `profile_fraction`), `next_open >= $20`, spread `<= 100 bps` **and** `<= 15 % of R`,
obtainable. One signal per symbol-day (the first), the engine's first-break rule.

**Stop**, declared per cell:

* `S20` = the low of the 20 bars up to and including the break bar (the intraday analogue of the
  consolidation low, always computable) — the **primary**;
* `SPD` = the **prior day's low** (daily panel) — the declared robustness arm; a signal whose prior
  day low is not below the entry is dropped in that arm.

**Exits**, declared per cell:

* `X2R` — the shipped +2 R / stop, flat 15:55 (**capped**);
* `XLK` — ORB's static lock: no target, at +1.75 R the stop moves to +0.5 R forever, flat 15:55
  (**uncapped**);
* `XBR` — the bare stop ridden to 15:55 (**uncapped**);
* `XNO` — **hold to the next session's open**, the gap charged in full: the stop is walked intraday
  as usual, and a position still open at 15:59 is carried overnight with **no overnight stop** and
  sold into the next session's **opening auction** at the daily `open`, no quoted spread. A swing
  breakout is a multi-day claim and the horizon is tested with the object, not assumed away.

**Book**: `trading.hod_break.run_book`'s rule — 12 a day, 4 concurrent, first-come by entry minute,
ties by symbol — via `common4.book_ranked(12, 4)`, causal slot freeing. For `XNO` a position is open
until 15:59 of its own day for slot purposes (the overnight leg does not block the next day's slots;
stated, because it is a choice).

### 2.2 The eleven cells

| # | cell | level | stop | exit |
|---|---|---|---|---|
| 1 | `H5·S20·X2R` | H5 | S20 | +2 R |
| 2 | `H5·S20·XLK` | H5 | S20 | static lock |
| 3 | `H5·S20·XBR` | H5 | S20 | bare stop |
| 4 | `H5·S20·XNO` | H5 | S20 | next open |
| 5 | `H20·S20·X2R` | H20 | S20 | +2 R |
| 6 | `H20·S20·XLK` | H20 | S20 | static lock |
| 7 | `H20·S20·XBR` | H20 | S20 | bare stop |
| 8 | `H20·S20·XNO` | H20 | S20 | next open |
| 9 | `H252·S20·X2R` | H252 | S20 | +2 R |
| 10 | `H5·SPD·X2R` | H5 | prior-day low | +2 R |
| 11 | `H20·SPD·X2R` | H20 | prior-day low | +2 R |

### 2.3 PREDICTION (stated before the run)

The multi-day-high break is **positive on gross in % of price** where F34's same-clock unconditional
cell is −0.09 to −0.14 % — specifically, `H20` gross ≥ **+0.10 % of price** on both splits, with the
`XNO` overnight cell the **best** of the four exits because the mechanism (multi-day trapped supply,
a buyer on a daily chart) is a multi-day claim that a 15:55 flat truncates.

### 2.4 FALSIFIER (stated before the run)

The object is **refuted** if gross in % of price is **≤ 0 on either split** for both `H5` and `H20`
at the primary stop, **or** if the overnight cell is not better than its own intraday sibling on
both splits (which would say the multi-day claim is not in the object), **or** if the frequency
floor of 10 trades/wk fails on both levels. `H252` is expected to fail the frequency floor and its
result is reported as a diagnostic if it does.

---

## 3. F38 — THE LAST HOUR (a different CLOCK). 10 cells.

### 3.1 The object

The **shipped** HOD-break detector, unchanged, on the **B2** stream (`hod_frames2/breaks2.csv`, the
`n` tag — the 5-bar plain low, the programme's reference geometry), with the entry window moved to
**14:00–15:30** (entry minute in `[840, 931]`) and the flat at 15:55. Population, gates, book and
cost model identical to B2; the ONLY thing that moves is the clock. One signal per symbol-day: the
**first break inside the window**.

**The frame's first deliverable, walked BEFORE any detector is put on top**: F34's floor map does
not extend past 14:01. This pass walks the **unconditional long bracket at a fixed 2 % and 3 % stop,
entered at a random eligible minute in 14:00–15:30**, on matched non-signal names of the PIT panel —
the same construction as F34's arm-d — plus the **MOC** variant of the same bracket. Those are cells
F38-U1..U3 and they are read first.

### 3.2 The ten cells

| # | cell | what |
|---|---|---|
| U1 | floor 14:00–15:30, 2 % stop, +2 R bracket, flat 15:55 | the floor, walked first |
| U2 | floor 14:00–15:30, 3 % stop, +2 R bracket, flat 15:55 | the floor, second width |
| U3 | floor 14:00–15:30, 2 % stop, **MOC** exit (auction, no spread) | the floor at the close |
| L1 | last-hour HOD break, +2 R / stop, flat 15:55 | **capped** |
| L2 | last-hour HOD break, bare stop to 15:55 | uncapped |
| L3 | last-hour HOD break, hold to next open, gap charged | uncapped |
| L4 | last-hour HOD break, **MOC** (stop honoured intraday, else the closing auction) | uncapped |
| L5 | last-hour HOD break, static lock (+1.75 R → +0.5 R) | uncapped |
| O1 | **ORB-style**: break of the 14:00–14:30 range high, entry 14:31–15:30, stop = the range low, +2 R / flat 15:55 | capped |
| O2 | the same object, **MOC** exit | uncapped |

### 3.3 PREDICTION

The last-hour floor (U1) is **less negative** than F34's 13:00–14:01 cell (−0.402 % TRAIN /
−0.255 % VAL) — the clock gradient continues — and **U3 (MOC) beats U1 by at least 0.10 % of price**,
because the auction leg pays no quoted spread and the cost, not the drift, is F34's whole story.
The detector cells L1..L5 are **not** predicted positive: the prediction is that **the MOC exit is
the best of the five**, and that if any last-hour cell is net-positive on both splits it is L4.

### 3.4 FALSIFIER

The clock frame is **refuted** if U1/U2 are **not** less negative than F34's 13:00–14:01 row on both
splits (the gradient does not continue into the close), **or** if the MOC leg does not improve net
by at least the cost it removes (which would say the auction-cost finding does not transfer),
**or** if every detector cell is negative net on both splits — in which case the sentence is that a
different clock does not rescue this detector, and the object is closed.

---

## 4. F37 — THE 5-MINUTE AND 15-MINUTE BAR (a different BAR SIZE). 10 cells.

### 4.1 The object

The shipped spec re-expressed on **5-minute** and **15-minute** aggregates of the same SIP tape,
same universe (`dist_open >= 5 %` from the running high, ADV20 ≥ 100K, the B2 population), same
clock (entry 09:37–14:01), same book (12/4), same cost model.

Aggregation is **clock-aligned** from 09:30: a 5-minute bar covers minutes `[09:30+5k, 09:30+5k+5)`,
open = first 1-min open, high/low = max/min, close = last 1-min close, volume = sum. A bar is used
only when **complete** (the detector sees closed bars only).

**Detector** on the coarse bars: the running high-of-day over coarse bars; a consolidation of `k`
closed coarse bars all holding within **4 %** of the running high; the signal is the coarse bar whose
**high** reaches the running high; entry at the **next coarse bar's open** under the 0.6 % cap; stop
= the consolidation low; `rv_profile ∈ [1, 5)` at the signal bar; `next_open >= $20`; the two spread
gates; obtainable. `k ∈ {3, 5}` — 3 because a 15-minute consolidation of 5 bars is 75 minutes and
the frequency floor would bind on arithmetic alone, 5 because it is the shipped number; both are
declared, neither is chosen after the fact.

Exits are walked on the **1-minute** tape (the engine's stop and target are live continuously; only
the *decision* is coarse) — stated, because the alternative (coarse-bar exits) is a different object.

### 4.2 The ten cells

| # | cell | bar | k | exit |
|---|---|---|---|---|
| B0 | 1-min baseline (B2 reference, re-printed for the comparison) | 1 | 5 | +2 R |
| M1 | `5m·k3·X2R` | 5 | 3 | +2 R |
| M2 | `5m·k5·X2R` | 5 | 5 | +2 R |
| M3 | `15m·k3·X2R` | 15 | 3 | +2 R |
| M4 | `15m·k5·X2R` | 15 | 5 | +2 R |
| M5 | `5m·k5·XBR` | 5 | 5 | bare stop |
| M6 | `15m·k3·XBR` | 15 | 3 | bare stop |
| M7 | `5m·k5·XLK` | 5 | 5 | static lock |
| M8 | `15m·k3·XLK` | 15 | 3 | static lock |
| M9 | `5m·k3·XBR` | 5 | 3 | bare stop |

### 4.3 PREDICTION

Signals per week fall monotonically with bar size (1-min ≫ 5-min ≫ 15-min) and the **15-minute
cells fail the 10 trades/wk floor** — that is the first thing checked, before any P&L. The stop
share falls with bar size (the 1-minute detector's ~52 % stop rate is partly noise breaks a coarser
bar never signals), and **gross in % of price improves monotonically with bar size** on both splits.

### 4.4 FALSIFIER

The bar-size frame is **refuted** if gross in % of price does **not** improve with bar size on both
splits, **or** if the coarse cells are negative gross like their 1-minute parent — in which case the
sentence is that the bar size changes the frequency and not the edge, and the 1-minute-detector line
is closed on this universe at every bar size tested.

---

## 5. Rails (all three objects)

* Reproduction gate **asserted in code and raising** before any cell is read: `common6.walk_from`
  reproduces `book6.rr` on 120 booked trades to `< 1e-9`; the B2 book reproduces
  1,622 / −$17,346 (TRAIN) and 706 / +$893 (VAL).
* Test tickers (`^Z[A-Z]ZZT$`) and names absent from `daily_bars` excluded from every population.
* Early closes excluded (`S.EARLY_CLOSE`).
* Availability audit per object; < 80 % ⇒ diagnostic, decided before the number is read.
* Price-scale check per object (§1.4), drop share printed.
* Day-clustered t beside the iid t on every cell.
* 2,000-draw count-matched permutation null on green weeks for every book cell.
* MDE on every cell.
* Both TRAIN halves + VAL on every cell.
* ex-top-5 % on **uncapped** exits only; on capped cells the `net > 0.05 × cap ÷ 0.95` form instead.
* Cost booked per cell from its own exit mix.
* Every store read-only; one python process; `nice -n 10`, `ulimit -v 3000000`; walks checkpointed
  per session; nothing written outside `frames12/`.
* **TEST never opened**, with the single declared exception in `FREEZE.md` (the next-session daily
  `open` for the overnight exit cells of the last two VAL signal days), whose trade count is printed.
* **31 cells declared here.** Programme **1,130 + 31 = 1,161**.
