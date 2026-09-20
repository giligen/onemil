# frames13 — PREREG (written and committed BEFORE any cell was scored, 2026-09-20)

Pass 13 of the frame programme. Three frames from the queue `hod_frames/FRAMES.md` wrote after pass 12:
**F40** the overnight floor, **F41** the overnight control as the DENOMINATOR for the three live books,
**F42** the closing auction as an exit leg.

Programme cell count entering this pass: **1,161**. This pass declares **30** cells (F40 16, F41 8,
F42 6) → **1,191** on exit. Nothing is selected after the fact; every cell below is scored and printed
whatever it says.

Rails carried, unchanged: reproduction gates asserted in code before any number is read (ORB
`book_G3_meas.csv` 282/177 picks, BF `runs/P1.csv` 56 trades / $139,113.67, HOD B2 1,622/706 and
−$17,346/+$893); **TEST (`day >= 2026-06-01`) stays sealed** (`FREEZE.md`); both halves of TRAIN
printed for every scored cell; day-clustered SE; count-matched permutation null where a green-week
claim is made; booked cost per cell and MEASURED for every auction leg; the pass-6 control rule
(a control shares the clock, the geometry and the universe of the thing it controls); an availability
audit on every field; **% of entry price beside R everywhere** (F31).

Splits: TRAIN `2025-01-01 .. 2025-12-31`, VAL `2026-01-01 .. 2026-05-31`, TEST `>= 2026-06-01` sealed.
TRAIN halves: H1 `.. 2025-06-30`, H2 `2025-07-01 ..`.

---

## 1. F40 — THE OVERNIGHT FLOOR (16 cells)

### 1.1 The object
The unconditional **close-to-next-open** return of a long position, in **% of entry price**, over the
whole PIT daily panel (`research/multiday/data/prices_by_year/{raw,all}/year=*.parquet`,
18,221,178 rows × 2 adjustments, 2016-01-04 → 2026-09-18, 11,823 symbols). This is a FLOOR MAP, read
before any detector, exactly as F34 mapped the intraday floor.

**Membership (causal).** A symbol-day is eligible iff: it is not a test ticker
(`research/scripts/pit_listings.is_test_ticker`); it has a close on day `t` and an open on the next
session `t+1` for the SAME symbol; `adv20` (20-session mean of raw dollar volume, strictly prior to
`t`, `min_periods=20`) exists; `close_t >= $1`. Price band, ADV$ band and class are all read from
data **at or before `t`** — the decision instant is the closing cross of day `t`.

**The return.** Computed on the ADJUSTED panel (`adjustment='all'`), which is the economically correct
overnight return through splits and dividends:
`on_pct = (open_adj[t+1] / close_adj[t] - 1) * 100`. The RAW panel's same quantity is computed
alongside; rows where `|on_pct_raw - on_pct_adj| > 1.0 pp` are flagged as corporate actions, counted,
and the map is printed **both** with and without them (price-scale rail 3).

### 1.2 The two legs, and how each is priced (declared before measurement)
* **Leg (a) — the auction leg.** Buy in the **closing cross** of `t` (`cls`), sell in the **opening
  cross** of `t+1` (`opg`). A cross is a single price: **no quoted spread is charged on either leg**
  (RUNBOOK step 3, the same ratio-0 treatment pass 12 used). The frame's own warning is honoured:
  *"opening auctions are illiquid"* — so the **opening cross's own liquidity risk is MEASURED, not
  assumed**, as the NBBO spread of the first RTH minute (09:30–09:31, Alpaca SIP quotes, mean of
  ask−bid over the minute) on a declared random sample of eligible name-nights, stratified by the
  price × ADV$ bands. That measured number is reported as the cost a participant pays if the cross
  does not fill him, and it is charged in full in leg (b).
* **Leg (b) — the marketable leg (the engine's current legs).** Buy marketable at 15:55 ET, sell
  marketable at 09:31 ET. Charged the measured half-spread at BOTH ends (full round-trip spread),
  from the same SIP quote sample.

### 1.3 The margin charge (declared before measurement)
An overnight position is Reg-T **2×**, not 4× intraday: half the position is financed. Per name-night
the charge in % of position is
`margin_pct = APR/360 * 0.5 * calendar_nights`, `calendar_nights` = actual calendar days from `t` to
`t+1` (3 over a weekend, 4 over a long weekend). Evaluated at **APR 6.0 / 7.0 / 8.0 %** and reported as
a band; the exact Alpaca rate is an ASSUMPTION to be confirmed from the broker statement before any
object is built, and the report says so.

### 1.4 The 16 scored cells
All on the auction leg (a), net of margin at APR 7.0 % (the middle of the band), unless stated.

| id | cell |
|---|---|
| A1 | the whole eligible panel, unconditional |
| A2 | price < $5 |
| A3 | $5 ≤ price < $20 |
| A4 | $20 ≤ price < $50 |
| A5 | $50 ≤ price < $200 |
| A6 | price ≥ $200 |
| A7 | ADV$ < $1M |
| A8 | $1M ≤ ADV$ < $10M |
| A9 | $10M ≤ ADV$ < $100M |
| A10 | ADV$ ≥ $100M |
| A11 | `kind == 'wrapper'` (leveraged / inverse) |
| A12 | `kind == 'common'` |
| **A13** | **the declared cross: `common` AND price ≥ $20 AND ADV$ ≥ $10M** — the only cell a real book could hold overnight at size, declared here BEFORE any number is read |
| A14 | A13 priced on **leg (b)** (15:55 → 09:31 marketable, measured round-trip spread) |
| A15 | A13 **ex-top-5 %** (F35's cap rule: the hold is uncapped, so the tail must be removed) |
| A16 | A13 over **2016-01 → 2024-12** (the era check the frame demands) |

**Map rows (diagnostics, NOT scored cells, no bar applied):** A13 × weekday (5), A13 × SPY regime
A/B/C1/C2 (`trading/regime_helpers`, features from SPY daily closes strictly before `t`), A13 × month
(12), and the FULL gap-risk distribution per scored cell — p1 / p5 / p50 / p95 / p99 of `on_pct`, the
share of name-nights beyond −5 % and −10 %, and the single worst name-night.

### 1.5 The bar, pre-committed
A cell is a **CANDIDATE** (and only then gets the ship-bar treatment) iff **all** of:
1. net (after the leg's measured cost **and** margin at APR 7.0 %) `> 0` in **every** era:
   2025-H1, 2025-H2, 2026-01→05, **and** 2016–2024;
2. ≥ 10 eligible name-nights a week (availability, trivially met by construction for the broad cells —
   printed anyway);
3. positive **ex-top-5 %** (the A15 form), because the hold is uncapped;
4. the cell's p5 gap is inside a stated, bounded downside.
Anything short of all four is reported as a FLOOR, not a candidate.

### 1.6 Predictions and falsifiers (stated before scoring)
* **P40.1** The unconditional panel return (A1) is POSITIVE and small — between **0 and +0.10 % of
  price** — and its sign is era-stable back to 2016.
* **P40.2** The premium is LARGER in the low-price and low-ADV$ cells (A2, A7) than in the liquid ones,
  because it is compensation for risk; and those same cells go NEGATIVE ex-top-5 %.
* **P40.3** The declared liquid cell A13 is positive gross but **fails the bar**, because
  margin + the opening cross's own liquidity risk eat a premium that is measured in single-digit basis
  points.
* **Falsifier of P40.3 (the interesting outcome):** if A13 is net-positive in all four eras AND
  positive ex-top-5 %, F40 has produced a CANDIDATE and the report must say so and apply the ship bar.
* **Falsifier of the frame:** if A1 is NEGATIVE net on the auction leg in the 2025+2026 window, pass
  12's incidental finding was an artefact of its matched-control population and F40 is REFUTED.

---

## 2. F41 — THE OVERNIGHT CONTROL AS THE DENOMINATOR FOR THE LIVE BOOKS (8 cells)

### 2.1 What is and is not rebuilt
**No new walk for ORB.** Pass 7's F24 already built ORB's matched non-signal control under ORB's OWN
exit spec (`c7.walk_orb`: static lock, arm +1.75 R, stop → +0.5 R, flat 15:45), at ORB's own clock (the
pick's own break minute + 1) and on ORB's own universe (matched on price, ADV20, class, ±5 pp gap).
That is exactly what F41 asks for, and it is reused verbatim (`p24.csv`).

**One gap is closed by a small re-walk.** Pass 7 walked BF with `partial=False`. The **shipped P1 book
carries the 50 % @ +2 R profit partial**, so BF's control was NOT under BF's own exit spec. BF's arms
are re-walked with `partial=True, partial_r=2.0, partial_frac=0.5, move_to_breakeven` — 2,400 walked
trades, the same bars, the same clock, the same pools (`poolb_bf.csv` / `poolu_bf.csv`), the same seed.
Nothing else changes. Both versions are printed.

**HOD-break** is carried for contrast only (its control is already known: −0.058 / −0.042 R universe
bound, −0.161 / −0.160 matched) and is NOT a scored cell here — it is dry, it earns nothing.

### 2.2 The 8 scored cells
| id | book | control arm | split |
|---|---|---|---|
| B1 / B2 | ORB (G3-meas, the config that boots) | **b** — matched non-signal name, same clock, same exit | TRAIN / VAL |
| B3 / B4 | ORB | **u** — random universe name, same clock, same exit | TRAIN / VAL |
| B5 / B6 | BF (P1 shipped, partial ON) | **b** | TRAIN / VAL |
| B7 / B8 | BF (P1 shipped, partial ON) | **u** | TRAIN / VAL |

Each cell prints, in this order (the frame's pre-commit: *the control's own return is printed for every
book BEFORE any difference is quoted*): the control's own mean R and % of price; the book's headline;
the paired day-clustered difference and its t; the MDE; both TRAIN halves.

### 2.3 The three readings, pre-committed with thresholds
Let `C` be the control's own mean R on a split and `H` the book's headline R on the same split.
* **R-ZERO** — `|C| < 0.25 × H` on **both** splits → the book's headline stands as an absolute edge and
  nothing downstream changes.
* **R-POS** — `C ≥ 0.25 × H` on **either** split → part of the headline is the universe under that exit
  geometry. The report must state the SHARE (`C/H`) and answer, explicitly, what the ramp band should be
  built on — **the difference or the level** — with the reason.
* **R-NEG** — `C < 0` on **both** splits → the book's edge is understated by its own floor.

### 2.4 Standing pre-commitment on the ramp band (stated before the numbers)
`trading/ramp_bt_band.py` compares **live** R/trade with the **BT reference's** R/trade. Both sides are
the same book under the same exit geometry, so a control return common to both sides **cancels inside
that gate**. The pre-committed position is therefore: a positive control is a finding about
**attribution and about the expected P&L level**, NOT automatically a reason to rebuild the band on the
difference — and the report must say which of the two it is for each book, rather than asserting a
change. **Nothing in `trading/`, `docs/`, or any checker is modified by this pass.** The deliverable is
a written recommendation for the owner.

### 2.5 Predictions and falsifiers
* **P41.1** Both live books resolve **R-POS** on at least one split (pass 7 measured ORB's universe
  bound at +0.070 / +0.024 R and BF's at +0.053 / +0.010 R, against headlines of +0.22 / +0.47 and
  +0.69 / +0.82 walker-internal).
* **P41.2** Adding BF's 50 % @ +2 R partial LOWERS both the BF signal and the BF control, and lowers the
  signal by MORE (the partial caps the right tail, and the signal is where the right tail lives), so
  BF's `C/H` share RISES.
* **Falsifier of P41.1:** if either book's control is ≤ 0 on both splits, that book resolves R-NEG and
  its headline is understated — the opposite finding, and it must be reported as plainly.

---

## 3. F42 — THE CLOSING AUCTION AS AN EXIT LEG (6 cells)

### 3.1 The object
For each book, the force-close leg is replaced by a **market-on-close order** submitted before the
15:50 imbalance publication and filled at the session's **official close**. Everything else — entry,
stop, lock/trail, partial — is unchanged, and **the stop stays live through 16:00**: a position that
would have been flattened at 15:45 / 15:55 can still be stopped out at 15:52 under MOC. That is
re-walked on the tape, never assumed.

* **ORB** force close 15:45 → MOC. **BF** force close 15:45 (`config.yaml trading.force_close_time`)
  → MOC. **HOD-break** (dry) force close 15:55 → MOC.

### 3.2 The 6 scored cells
| id | book | split |
|---|---|---|
| C1 / C2 | ORB (G3-meas) | TRAIN / VAL |
| C3 / C4 | BF (P1, partial ON) | TRAIN / VAL |
| C5 / C6 | HOD-break B2 (dry) | TRAIN / VAL |

Each prints: the share of the book's trades that are force-closed at all (the leg only matters for
those); Δ net R **per force-closed trade** and **per trade of the whole book**; Δ in % of entry price;
Δ per week in dollars at the book's own live size; the share of force-closed trades whose exit REASON
changes (stopped between the old flat and 16:00); and the distribution of
**(official close − force-close print)** on the book's own names — **median, p5, p95**, never the mean
alone.

### 3.3 The pre-registered falsifier — pass 12's lesson, applied
Pass 12 found the MOC "edge" on the HOD floor was a 15:55→16:00 tail living entirely in H1-2025 with a
**median of −0.006 %**. So, pre-committed here:
* the leg **improves** a book only if the Δ is positive on **TRAIN-H1, TRAIN-H2 and VAL**, and only if
  the **MEDIAN** of (official close − force-close print) is positive on both splits;
* **a book whose entire improvement is the mean and not the median is reported as NOT improved**;
* the reliable, mechanical part — the spread the auction does not pay — is reported SEPARATELY from the
  drift part, because only the first is repeatable.

### 3.4 If it clears
A clearing book is an **engine diff**, not a ship: `TimeInForce.CLS` is unused in this repo
(`data_sources/alpaca_client.py` carries `TimeInForce.DAY` only). The report must state exactly what
would change and that it requires an **independent rebuild from prose** before any owner decision.
**No engine code is touched in this pass.**

### 3.5 Predictions
* **P42.1** The mechanical saving (the exit spread the auction does not pay) is ~0.06–0.07 % of price
  per force-closed trade on every book — pass 12's measurement, re-confirmed on each book's own names.
* **P42.2** The drift part fails its own falsifier on at least two of the three books (median ≈ 0).
* **P42.3** ORB is the book most affected, because it flats 15 minutes earlier than the others and
  therefore force-closes the largest share of its trades.

---

## 4. Node rails for this pass
One python process at a time, `nice -n 10`, `ulimit -v 3000000`, every panel scan checkpointed per
year. `cache.db`, `research/bf_zero/bars_sip.db`, the Databento / multiday stores and `daily_bars`
opened **read-only**. Nothing written outside `research/mature_method/frames13/`. No `config.yaml`,
`orb.yaml`, systemd unit, cron or order is touched. The live service boots 12:30 UTC Monday and the
pre-boot suite runs 11:27 UTC — no repo code outside `frames13/` is modified by this pass.
