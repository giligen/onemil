# hod_frames5 — PRE-REGISTRATION (F16 portfolio · F18 population · F17 horizon)

Committed **before any cell in this pass is scored**. Pass 5 of the HOD-break frame programme
(`research/mature_method/hod_frames/FRAMES.md`). Run order fixed by the queue: **F16 → F18 → F17**.

## 0. The arithmetic that frames the whole pass — stated up front, before any number

The base book's GROSS is **−0.002 R (TRAIN) / +0.015 R (VAL)** on the pre-book signal set and
**−0.039 / +0.083** after the slot rule; the booked-set measured cost is **0.061–0.065 R**. In 958
cells the best obtainable gross is about **+0.1 R**. **Therefore no cost frame, no fill frame and no
sizing frame can rescue this book — even free fills leave it at zero.** The three frames in this
pass are admitted to the queue only because each one changes the OBJECT (which positions are held
together, which instruments are in the universe, how long the position is held), not the cost.

This sentence is pre-committed to appear at the top of `REPORT.md` regardless of the results.

## 1. Rails (identical to passes 2–4, non-negotiable)

1. **Reproduction gate.** `B2` = `sigset5(admit(br, all))` booked 12/day × 4 concurrent must read
   TRAIN 1,622 · 30.6/wk · gross −0.039 · net −0.107 · 32.1 % green · **−$17,346** and VAL 706 ·
   30.7 · +0.083 · +0.013 · 43.5 % · **+$893**. *(Run in `diag5.py` before this file was committed:
   **EXACT**, Δ$ 0.)* The shipped 12/4 slot machine is asserted in code against
   `trading.hod_break.run_book` via `common4.book_ranked(score=None)`; `common5.book_portfolio` must
   reproduce `book_ranked` row-for-row when `key=None` — **asserted in code, the run aborts
   otherwise.**
2. **TEST is sealed** behind `FREEZE.md`. No TEST number is computed in this pass.
3. **Both halves.** Every cell reports H1-2025, H2-2025 and VAL gross separately; "era-consistent"
   means same-signed positive in all three.
4. **Day-clustered SE** beside the iid t on every cell (clusters = trading days).
5. **Count-matched permutation null** (2,000 draws, pick count held fixed per week) on green weeks
   for every scored cell × split.
6. **Booked cost measured per cell** (mean `rr − net`), and the imputed share printed. Where a
   declared cell's booked imputed share exceeds 50 %, a **dedicated Alpaca-SIP NBBO fetch** at that
   cell's own signal minutes is run and the cell is re-scored on measured cost (§4.3).
7. **Availability audit** on every field used in a decision: coverage, and the missingness gap
   between winners and losers. A field under 50 % coverage is VOID; a gap over 5 pp is VOID.
8. **Already-there / arrived-after diagnostic** (`hod_frames` §2.3) reported for every cell that is
   a candidate for the selector — a REPORTED diagnostic, never an exclusion (the F10 re-spec).
9. **Obtainability.** Every fill is the open of the minute after the break bar, under the 60 bps cap.
   No cell re-prices a deferred entry. For F17 every multi-day exit is a price the market offered on
   that day (a daily open or a daily close from `daily_bars`), never a touch of a level.
10. **Price-scale check** (F17 only, mandatory — CLAUDE.md rule 3): `daily_bars` (possibly adjusted)
    is joined to intraday Alpaca bars. For every carried trade the entry-day daily bar must bracket
    the intraday entry price (`low ≤ entry ≤ high`, 0.5 % tolerance). Symbol-days that fail are
    dropped from the horizon cells and the dropped share is reported.
11. **MDE** (80 % power, per trade, net) printed on every rejection.
12. **Cell count** declared here, counted in the report, added to the programme total (958).
13. Read-only on every store. One python process. `nice -n 10`, `ulimit -v 3000000`. No config,
    `orb.yaml`, unit file, cron, order or cache is written. The dry run is not touched.

## 2. The bars

* **Claim bar** — G1: TRAIN net R > 0 with iid **and** clustered t ≥ 2 at ≥ 10 trades/week;
  G2: VAL same sign and ≥ 55 % green weeks; then TEST, once, behind `FREEZE.md`.
* **Live-exploration bar** — positive weekly **$** AND ≥ 50 % green weeks on **both** splits at
  ≥ 10 tr/wk, clustered t ≥ 2, halves same-signed. A cell clearing it → **SHIP-TO-DRY** with the
  exact engine diff. Nothing else ships.

## 3. F16 — THE PORTFOLIO (10 cells: 1 diagnostic + 9 scored)

Population: the shipped B2 pre-book set. **The admission is untouched in every cell** — same
signals, same prices, same stops, same exits. Only the slot allocator changes. Instrument fields
(`asset_class`, `anchor`) are static-at-the-open attributes built by `common5.attach_instrument`
from the offline 33K class map (`trading/orb_asset_class.py`), the same construction as F15;
`venue` is joined from `hod_frames4/sig4_inst.csv` (Databento PIT definitions, 100 % coverage);
`adv_dollar` is the 20-prior-session mean dollar volume, strictly prior.

**F16-diag (reported FIRST, before any cell): the concentration diagnostic.** Over every
slot-minute of the booked B2 book: the share of occupied slot-minutes in which ≥ 2 open positions
share an `anchor`; the same per `venue` and per ADV$ bucket; the maximum same-anchor concurrency
seen; the distribution of open positions per anchor. No bar applies.

| cell | rule |
|---|---|
| **F16-a1** | at most **1** open position per `underlying_anchor`, 4 slots (a rejected candidate does NOT consume a slot — the later candidate may take it) |
| **F16-a2** | at most **2** open positions per `anchor`, 4 slots |
| **F16-a1n** | at most 1 per `anchor`, 4 slots, **NO REFILL** — a rejected candidate consumes one of the day's 12 and the slot stays empty (ORB's pre-committed invariant, which its own research found load-bearing) |
| **F16-a6** | at most 1 per `anchor`, **6 slots** |
| **F16-a8** | at most 1 per `anchor`, **8 slots** |
| **F16-v2** | at most **2** open positions per listing **venue**, 4 slots |
| **F16-b2** | at most **2** open positions per **ADV$ bucket**, 4 slots. Buckets fixed in advance, never fitted: `< $10M`, `$10–50M`, `$50–200M`, `$200M–1B`, `≥ $1B` |
| **F16-m1** | **the MIRROR — the CONCENTRATED book.** Admit a break only if a sibling sharing its `anchor` has **ALREADY broken at or before this minute** (`cohort_causal ≥ 2`, the causal form of pass 4's look-ahead). Pre-committed expectation: pass 4's causal number, **−0.013 / +0.154 gross, −$2,284 / +$2,099, H1 −0.120** — a reproduction, not a discovery |
| **F16-m2** | F16-m1 **×** F16-a1 (the concentrated book, de-duplicated: take only the first member of a running complex) |

Declared check, exactly as F4 declared it: **the per-trade gross of a de-dup cell is NOT required to
equal B2's** — de-dup frees a slot and a different trade fills it. What IS required and asserted:
the de-dup cell's booked set is a subset of the pre-book set with no re-priced fill, and
`book_portfolio(key=None)` reproduces `book_ranked` row-for-row. The de-dup's effect is reported on
week shape, MDD, red streak, worst week and green-week share, with the gross printed beside them.

**Selector**: the live-exploration bar of §2. Nothing else is promoted.

## 4. F18 — THE POPULATION (14 scored cells + 2 structural reports)

### 4.1 Sub-question 1 — wrappers vs common vs mixed

`F15-c1/c2` scored these once under the shipped book; this pass gives the two populations the FULL
treatment (both halves, VAL, day-clustered t, count-matched null, ex-top-5 %, MDE, weekly dollar
table) because pass 4's finding — **common stock alone is era-consistently NEGATIVE
(−0.060 / −0.046 / −0.055) and the book's positive VAL is entirely its leveraged wrappers** — makes
the wrapper population the only remaining honest candidate in 958 cells.

| cell | rule |
|---|---|
| **F18-p1** | **WRAPPERS ONLY** (`asset_class == 'wrapper'`), shipped floor and gates |
| **F18-p2** | **COMMON ONLY** (`asset_class == 'stock'`), shipped floor and gates |

*(mixed = B2, the reference row, not a cell.)*

**Pre-committed promotion rule**: if `F18-p1` is same-signed positive in H1, H2 and VAL at
≥ 10 trades/week, it receives the full ship-bar treatment and, if it also clears the
live-exploration bar of §2, a SHIP-TO-DRY diff is written. Declared mechanism (written before the
number is looked at): a 2×/3× wrapper rebalances its exposure daily, so a wrapper on an underlying
up 5 % is up ~10 % and its own rebalancing is **forced buying into the close** — a "who is buying"
answer of the kind F15 looked for in short interest and did not find. If the population is positive
but the mechanism's signature is absent, that is reported as a caveat, not hidden.

### 4.2 Sub-question 2 — the price floor (and the live-parity question)

**Parity finding, established in `diag5.py` before this file was committed and reported regardless
of every other result:** `config.yaml hod_break.min_price = 20.0`. The live dry run is on the
**$20** population, not the $5 population that `FRAMES.md` F18 asserted; the four sessions of
forward data are therefore on the SAME population as every study. The one real (small) difference
is definitional and is reported: the study floors on `next_open`, the engine floors on the break
**level** (`hod_break_engine.py:665`) — 8 pre-book signals engine-only and 11 study-only out of
7,027 (**0.3 %**). `CLAUDE.md`'s "price ≥ $5 (cost rule 9/14)" is stale relative to the shipped
config and is flagged in the report. **No config is changed by this pass.**

| cell | rule |
|---|---|
| **F18-f5 / f10 / f30 / f50** | floor `next_open ≥ {5, 10, 30, 50}`, shipped gates, MIXED population (4 cells; `f20` = B2) |
| **F18-w5 / w10 / w30 / w50** | the same ladder on **WRAPPERS ONLY** (4 cells; `w20` = `F18-p1`) |

A known cost hazard, declared before scoring: the imputation model
(`hod_break/score.build_impute`, fitted on `hod_filter_stack/pop.csv`) **has no cell below $10** —
its price bands start at `$10-20` — so every sub-$10 signal falls back to the global median
(0.345 %), which is a $20-plus number. Any $5 or $10 rung is therefore scored **twice**: once on the
imputed model (the as-is arm) and once on the dedicated measured fetch of §4.3. The as-is arm alone
is not allowed to support any conclusion about the low floors.

### 4.3 Sub-question 3 — the two spread gates, with the cost MEASURED

The gates `spread ≤ 100 bps` and `spread ≤ 15 % of R` remove ~46 % of the $20-floor first-break
stream (13,299 → 7,027 pre-book signals) and were set from a cost table this programme has since
shown wrong in both directions.

| cell | rule |
|---|---|
| **F18-s08 / s25 / s40** | `max_frac_r ∈ {0.08, 0.25, 0.40}`, `max_bps = 100`, shipped floor (3 cells; `s15` = B2) |
| **F18-soff** | **both** spread gates OFF, shipped floor (1 cell) |

**The measured-cost arm.** A dedicated Alpaca-SIP NBBO fetch (`fetch_nbbo5.py`, identical source,
convention and two causal instants as `research/bf_zero/causal_filter/fetch_nbbo.py` and
`hod_frames3/fetch_nbbo3.py`: the mean ask−bid over the SIGNAL minute `entry_m − 1`, and the last
quote at or before the open of `entry_m` for the obtainability test) is run at the **own signal
minutes** of the union of (a) every pre-book row of the $20-floor gates-OFF set and (b) every row
booked by any declared F18 cell, that is not already in `research/bf_zero/causal_filter/nbbo.csv`.
Declared in advance: **measuring the spread changes membership in both directions** — a measured
spread can fail a gate an imputed one passed, and a measured `ask_dec` can make a fill
UNOBTAINABLE where a missing quote defaulted to obtainable. Both effects are counted and reported;
neither is allowed to be presented as a result of the gate rung itself.

### 4.4 The structural reports (no bar)

* **F18-R1**: the base GROSS of every population choice **before any admission rule is put on top**,
  with the era split — the question `FRAMES.md` says nine passes never asked ("is B2 the pattern?").
* **F18-R2**: the availability audit for `asset_class` on the enlarged populations.

## 5. F17 — THE HORIZON (8 scored cells + 2 diagnostics)

Population: the B2 **booked** set. Entry, stop, target and the intraday rules are **unchanged**; the
only change is what happens to a position still open at 15:55. A trade that hit its stop or its
+2R target intraday is closed and is carried into every horizon cell unchanged — the horizon only
re-writes the `eod` exits. Multi-day P&L is computed from `cache.db daily_bars` for the trade's own
symbol; for a wrapper that is the **wrapper's own** daily bars, so its daily-rebalancing decay is
in the price by construction (verified, not assumed, by the F17-D2 diagnostic below).

| cell | horizon | overnight stop |
|---|---|---|
| **F17-h1a** | exit at the **next session's OPEN** | none (position rides overnight) |
| **F17-h2a** | exit at the close of **+1 session** | none |
| **F17-h3a** | exit at the close of **+2 sessions** | none |
| **F17-h5a** | exit at the close of **+5 sessions** | none |
| **F17-h1b / h2b / h3b / h5b** | the same four horizons | **the prior session's CLOSE is the stop**: on each subsequent session, if the low ≤ prior close the position exits there; if the OPEN gaps below it, the exit is the **open** (the gap is charged, never netted) |

**Charges, declared before scoring.** (i) The overnight gap is real risk and is taken at the open,
never at the prior close. (ii) The carried exit pays a full half-spread (`0.5 × sp_pct / r_pct`,
ratio 1.0 — a marketable exit at an open or a close), which is *more* than the shipped `eod`
exit's 0.412 ratio; the carried book is therefore charged more per trade than the shipped one.
(iii) Every carried position is also reported per unit of time (net R ÷ calendar days held), because
a book that earns the same R over five days is not the same book.

**F17-D1 — the two-cohort diagnostic**: every horizon cell split by `day_range_pct ≥ 10 %` vs the
rest (the `bf_zero` §6b cohort). A multi-day hold on a mover day is the continuation bet this book
claims to be; if the horizon pays anywhere it should pay there.

**F17-D2 — the wrapper-decay measurement** (mandatory, do not assume): for every wrapper in the
carried set, the realised daily log return of the wrapper minus its stated leverage × the
underlying's daily log return, over the horizon window — the empirical decay, in R and in %/day,
reported beside the cells so that the claim "the decay is already in the price" is a measurement
and not an assertion.

## 6. Cell count

F16 **10** (1 diagnostic + 9 scored) · F18 **14** scored + 2 structural reports · F17 **8** scored +
2 diagnostics. **Declared decision cells: 9 + 14 + 8 = 31.** Programme total after this pass:
**958 + 31 = 989** (diagnostics, structural reports, reproduction rows and the parity assertion are
counted separately and carry no decision).

Expected largest |t| under a pure null over 31 × 2 ≈ 2.9. Any cell reaching the claim bar must be
read against that number.

## 7. What this pass will NOT do

* No frozen trade-level admission or exit cell (pre-refuted by F12 — the separations are noise).
* No refit of anything.
* No TEST number.
* No config, `orb.yaml`, unit-file, cron or order change; the dry run keeps running untouched.
* No new rule is put in front of the owner without an independent check: every field used in a
  decision passes the already-there / arrived-after diagnostic and the availability audit, and every
  fill is obtainable.
