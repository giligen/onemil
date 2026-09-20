# PREREG — frames11 (pass 11)

**Committed before any cell of F35 or F34 is scored.** F36 is code, not a scored frame, and carries
no cells; it is specified here only so its acceptance tests are pre-committed too.

Programme cell count before this pass: **1,104** (frames10 REPORT §4). This pass declares
**11 (F35) + 13 (F34) = 24**, taking the programme to **1,128**. Every declared cell is scored and
printed whatever it says; nothing is selected after the fact.

TEST is sealed — see `FREEZE.md`.

---

## F36 — THE POOLED INSTRUMENT FED, AND THE BF BAND'S R BASIS FIXED (code; 0 cells)

### (1) The producer

`scripts/hod_break_eod_check.py` appends the session's **DRY-RUN EXECUTABLE book** — the engine's
own logged `[HOD DRY] WOULD BUY` signals, filled at the next open under the logged limit and walked
forward on the day's bars, then cut by `trading.hod_break.run_book(12, 4)` — to
`data/hod_dry_pool.csv` via `trading.ramp_pool.append_dry_trades`.

That book, not the spec-on-REST-bars book, is the stream frames10 F33 replayed (`r33.log`: 9/14 11
trades −4.6R, 9/16 7 −4.2R, 9/17 7 −4.5R, 9/18 6 +9.4R), and it is the right one: it measures what
the ENGINE signalled, not what the spec would have signalled on a different bar source.

**Acceptance (pre-committed):**

* schema documented in `ramp_pool`'s module docstring: `day, symbol, entry_minute, r`;
* **idempotent** on `(trade_date, symbol, entry_minute)` — a re-run of a session already in the file
  appends 0 rows and leaves the file byte-identical;
* the pooled band reads the appended rows (`load_dry_trades` → `pooled_z` → `pooled_band`);
* the CSV is gitignored;
* LIVE mode and the red-to-green book write nothing;
* a failing pool write can never break the EOD check the owner's cron runs.

### (2) The BF band's R basis — the defect frames10 F31 §1.4.2 found

`research/bf_frequency/runs/P1.csv` carries the BT's **SIZED** pnl. `pnl / $2,000` is therefore not
a per-trade R: the cache's `shares` embed the conviction / MACD-zone / regime multipliers, so the
BT book risks ~1.8× its nominal $2K on the average trade. The live side divides by a **flat**
`trading.risk_per_trade`. The gate was comparing a multiplied book with a flat one.

**Column audit, stated before the fix is used** (`bf_trade_risk_usd` preference order):
`risk_per_share × shares` → `shares × |planned_entry − stop_loss|` → `shares × |entry_price − stop_loss|`.
On `P1.csv`: **`risk_per_share` is ABSENT; `planned_entry` is present as a column but EMPTY on all
56 rows; `shares`, `entry_price`, `stop_loss` are present on all 56** — so the file resolves to the
third form, and a test asserts that (it will fail loudly if `planned_entry` is ever backfilled).

**Acceptance (pre-committed):** the BF band's mean R on the reference changes from **≈ +1.65 to
≈ +0.91 on TRAIN** (F31's O6 row, 34 trades) and from +1.242 to +0.701 over all 56; the fixed basis
equals `pnl_pct / stop_pct` row by row to the CSV's own 2-dp precision; a synthetic live sample at a
**flat** stage base classifies IN-BAND where the legacy basis read BELOW-p10; `BF_BAND_R_BASIS=notional`
reproduces the retired band for one week (to 2026-09-27) and shouts at WARNING each time.

**Honest caveat, pre-committed:** the legacy basis's SD is 3.151 against the fixed 1.574, so at a
handful of trades the legacy band is so WIDE it classifies almost anything as IN-BAND — its failure
at small n is that it cannot reject, not that it rejects wrongly. The classification flip is
therefore demonstrated at the n where the band is informative (a completed stage, n = 30), and that
is stated rather than quietly chosen.

`config.yaml`, `orb.yaml`, the systemd unit, the crons and the order path are NOT touched.

---

## F35 — THE TAIL IS THE BOOK: is the tail IDENTIFIABLE at the entry bar? (11 cells)

### The question, and why it is the honest one

Three objects in 1,104 cells have been positive-net on both splits at ≥ 10 trades a week —
`hod_fresh` C1, pass-9 SUPP A, pass-10 F32-L (the wrapper long) — and **every one died on the
binding ex-top-5 % clause**. Under the owner's own metric (RUNBOOK step 7) *monsters are permitted
if green weeks dominate*, so the ex-top-5 % trim is a **diagnostic**, not a rejection reason. The
decision-relevant question is therefore not "does the book survive without its tail" but:

> **Is the tail identifiable at the entry bar?**

If some causal entry-bar field separates the top 5 % from the rest on BOTH splits, that field is a
monster-finder and earns ONE declared admission cell. If none does, then the tail is luck the book
cannot aim at, the ex-top-5 % clause has been protecting against luck rather than killing real
books, and every tail-carried object stays dead. Either answer is a deliverable sentence.

### Population

`hod_frames6/book6.csv` — the **B2 booked book**, 2,328 trades (TRAIN 1,622 / VAL 706), the
reference six passes are measured against. Reproduction gate **G-B2** is asserted in code before
any number is read: 1,622 TRAIN / **−$17,346**; 706 VAL / **+$893**.

### The label

**TAIL = the top 5 % of trades by `net` R WITHIN ITS OWN SPLIT** (rank-trimmed, as the programme's
ex-top-5 % clause is: `hod_frames6` rank trimming, because the +2 R point mass breaks quantiles).
TRAIN tail n ≈ 81, VAL tail n ≈ 35. REST = everything else in that split.

### The fields — declared in full BEFORE any is scored (11 cells, one per field)

Every field must be computable at or before the decision bar; each carries its provenance and is
subject to the **80 % availability rail** (a field below it is demoted to a diagnostic and said so).

| # | field | source | causal? |
|---|---|---|---|
| **C1** | `rv_profile` — cum volume ÷ (ADV20 × clock fraction) at the break bar | the admitted signal set (`sigset5` / `load_breaks4`) | the engine's own gate input |
| **C2** | `dollar_frac` — the bar's dollar volume as a fraction of ADV$ | `book6.dollar_frac` | closed bars only |
| **C3** | `sp_pct / r_pct` — the imputed spread as a share of R (the cost the trade pays) | `book6.sp_pct`, `book6.r_pct` | quoted at the decision minute |
| **C4** | `dist_open` — % the level sits above the session's 09:30 open | break table / PIT panel | closed bars only |
| **C5** | `gap_pct` — the session's gap vs prev close | PIT panel (`frames9.c9.panel`) | known at 09:30 |
| **C6** | `asset_class` — wrapper vs common | `trading/orb_asset_class` (`book6.asset_class`) | static |
| **C7** | price band of the entry price | `book6.price` | known at the bar |
| **C8** | entry minute band | `book6.entry_m` | the clock |
| **C9** | `spy_r5_pct` — SPY's 5-minute return into the decision bar | `book6.spy_r5_pct` | closed bars only |
| **C10** | `r_pct` — the stop width as % of price (F31's live-relevant unit) | `book6.r_pct` | known at the bar |
| **C11** | `adv20$` — ADV20 × price, the liquidity band | `book6.adv20`, `book6.price` | prior sessions |

### The pre-committed decision rule

A field **SEPARATES THE TAIL** — and only then earns one admission cell — iff ALL FOUR hold:

1. the TAIL-minus-REST difference is **same-signed on TRAIN and on VAL**;
2. **|day-clustered t| ≥ 2 on TRAIN** (one cluster per session, the programme's `clustered_t`);
3. **VAL same sign with |t| ≥ 1** (VAL is a third of TRAIN, so the bar is directional, not powered);
4. the observed |difference| is **outside the p95 of a 2,000-draw count-matched null** that relabels
   the same NUMBER of tail trades at random within the split, preserving the per-day pick counts.

For the categorical fields (C6, and the banded reads of C7/C8/C11) the statistic is the tail's SHARE
in each level against the same null.

**If ≥ 1 field passes**: it is named, ONE admission cell is declared and scored **per passing
field** — and the honest phrasing rule applies to whatever it reads.

#### AMENDMENT, committed 2026-09-20 BEFORE any admission cell was scored

The eleven separation cells have been scored and **two fields passed** (`C1 rv_profile`,
`C3 spread_over_r`), so the admission cells are now specified — parameter-free, so nothing is fitted:

* **The cut is the MEDIAN of that field over the whole TRAIN book.** No threshold is searched. The
  kept side is whichever side the TAIL sits on (both passing fields are LOWER in the tail, so both
  keep the low half).
* The kept signals are re-booked from the FULL admitted signal set with `run_book(12, 4)` — the
  cell is an admission rule, not a post-hoc trade filter, so slots freed by a rejected candidate are
  refilled by the next admitted one exactly as the live engine would.
* Scored on both TRAIN halves, VAL, with the booked cost, green weeks against its own 2,000-draw
  count-matched null, dollars at live size, day-clustered t, ex-top-5 % beside the headline, MDE.
* Both admission cells are COUNTED: F35 is therefore **11 + 2 = 13 cells**, programme
  1,104 + 13 + 13 (F34) = **1,130**.

#### The circularity check, committed at the same moment and for the same reason

`C3 spread_over_r` is **mechanically inside `net`**: the programme's cost is a monotone function of
spread ÷ R, so a low-spread/R trade has a higher net BY CONSTRUCTION. The label is therefore
re-derived on **GROSS `rr`** — which contains no cost term at all — and every one of the eleven
cells is re-scored against that label. A field that separates the tail on the net label but NOT on
the gross label is **an artefact of the cost model, not a monster-finder**, and is reported as such.
This check is run before either admission cell is read.

**If 0 fields pass**: the deliverable sentence is *"the monsters are not identifiable at entry on
this book"*, every tail-carried object (hod_fresh C1, SUPP A, F32-L) **STAYS DEAD**, and the
ex-top-5 % clause is recorded as **protecting against luck, not killing books** — with its MDE
stated, because a null at this n says only so much.

### Rails (F35)

G-B2 asserted in code · both TRAIN halves + VAL on every cell · day-clustered t · 2,000-draw
count-matched null on the tail LABEL (not on P&L) · availability audit per field with the 80 % rail ·
MDE per cell · multiplicity 11 declared, all printed · TEST never opened.

---

## F34 — PRICE THE UNIVERSE, NOT THE SIGNAL: the floor map (13 cells)

### The question

1,104 cells have asked *which signals to take*. None has asked what the **bare instrument** on this
universe is worth. If the unconditional long bracket is negative EVERYWHERE on the PIT HOD universe,
then every admission rule the programme can write is choosing the least-negative region of a
negative field — which is exactly what 1,104 cells found, and it would be the floor under all of
them. If some cell of the bare instrument is positive and era-stable at book frequency, that cell is
a universe rule (which names to put in play), which is worth more than any filter — F29's finding,
applied to HOD's own universe.

### Population and instrument

* **Population**: pass 6's **arm d — 288,174 detector-free control trades** (`hod_frames6/pd6.csv`),
  a matched NON-signal name at a random eligible minute on the PIT HOD universe
  (prev close ≥ $17, ADV20 ≥ 100K). **No detector anywhere in the construction.** TEST cut off.
* **Instrument (NEW — this is the re-walk)**: a long bracket entered at the OPEN of the control's
  own bar, with a **FIXED stop at `s` % of the entry price**, `s ∈ {2 %, 3 %, 4 %}`, a target at
  `+2R = +2s %`, and a flat at 15:55. Exit priority, slippage and fill convention are
  `hod_frames6.common6.walk_from` verbatim (EOD → stop → target from bar e+1; stop fills at
  `min(stop, that bar's open) × (1 − SLIP)`; target fills AT the target; EOD at that bar's open).

  This is NOT F22/F25's object: those walked the control at **the booked trade's own `r_pct`**, so
  the stop width carried the signal's information. Here the stop is a pure function of price, so the
  cell is a property of the UNIVERSE and the clock alone.

* **Unit**: **% of entry price** (F31's unit), i.e. `rr × s`. R is deliberately not the headline —
  F31 proved R flatters a tight stop.
* **Cost**: the programme's measured per-trade cost, booked **per cell** from that cell's own exit
  mix and its own imputed spread (`S.build_impute` / `attach_cost`, the same object the book uses),
  expressed in % of price.

### The 13 declared cells (marginals, not the 72-way cross)

Each cell is scored at all three stop widths and on both splits; the cross-tabs are printed as
diagnostics, never promoted.

| # | cell |
|---|---|
| **U0** | unconditional (the floor itself) |
| **U1–U4** | entry hour: 09:37–10:30 · 10:30–11:30 · 11:30–13:00 · 13:00–14:01 |
| **U5–U7** | price band: $17–30 · $30–100 · ≥ $100 |
| **U8–U10** | ADV$ band: < $25M · $25–150M · ≥ $150M |
| **U11–U12** | wrapper · common |

### The pre-committed bar

A cell is **POSITIVE AND ACTIONABLE** iff ALL of:

1. **net > 0 in % of price** on TRAIN **and** on VAL (and same-signed in both TRAIN halves);
2. **≥ 10 book-sized opportunities a week** in that cell (the programme's live-exploration floor);
3. **|day-clustered t| ≥ 2** on TRAIN;
4. outside the p95 of a **2,000-draw permutation across ALL 13 cells** (labels shuffled within day,
   so the multiplicity of the whole map is paid once).

**If no cell clears it**, the pre-committed sentence is: *the floor under every admission rule on
this universe is ≤ 0, so an admission rule can only pick the least-negative region* — with the map's
MDE stated, and with the honest phrasing rule (no edge was detectable in THIS universe, at THIS
horizon, at THIS book size, over THIS window, at THIS cost).

### Rails (F34)

Reproduction: the walker is required to reproduce `book6.rr` on the booked trades when handed the
booked stop (the same gate pass 6 and pass 8 used, asserted before any control is priced) ·
availability audit on the bar join · both TRAIN halves + VAL · day-clustered t · cost booked per
cell · 2,000-draw permutation across all 13 · MDE per cell · multiplicity 13 declared ·
TEST never opened · every store read-only.

---

## Rails common to the pass

* One python process at a time, `nice -n 10`, `ulimit -v 3000000`.
* `cache.db`, `bars_sip.db`, the Databento stores, `daily_bars` and `trades.db` opened READ-ONLY.
* The ONLY file written outside `frames11/` is `data/hod_dry_pool.csv` (F36's producer) and the
  code/tests F36 declares.
* No config, `orb.yaml`, systemd unit, cron or order is written.
* Verdict per frame in PLAN §1 phrasing; three LOG lines; `FRAMES.md` gains this pass's rows and the
  next queue (F37/F38/F39), declared but NOT run.
