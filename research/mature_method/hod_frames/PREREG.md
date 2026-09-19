# hod_frames — PRE-REGISTRATION (written and committed BEFORE any cell was scored)

Owner 2026-09-19: *"continue and iterate and look at it from scratch every time, with fresh eyes,
using different ways to extract $$$ from this. You won't stop till this happens."* and, mid-pass,
*"short is a valid move if that helps."*

The first 799 cells of this programme were **one frame**: filter the signal minute, shape the exit.
This pass opens three structurally different frames. `FRAMES.md` is the standing ledger.

## 0. Rails that apply to every cell

* **Splits**: TRAIN `2025-01-02 … 2025-12-31`, VAL `2026-01-01 … 2026-05-31`, TEST
  `2026-06-01 … 2026-09-11` **SEALED** behind `FREEZE.md` and opened at most once, for a cell that
  has already cleared G2. No script in this directory reads a TEST-dated bar unless `--test` is
  passed AND `FREEZE.md` exists.
* **Membership**: early-close days out (`2025-07-03`, `2025-11-28`, `2025-12-24`); test tickers out
  (`research/scripts/pit_listings.is_test_ticker`); names absent from `daily_bars` out.
* **Book**: 12 picks/day, 4 concurrent, first-come (`trading.hod_break.run_book`), `risk_usd` $100,
  `<= 14:00` last entry — identical to the live spec, except where a frame explicitly changes it
  (Frame 3 changes only the dollar size; the pick set is untouched).
* **Cost**: the programme's measured-NBBO model (`hod_break/score.attach_cost`): half the measured
  spread in, plus `RATIO[why]` × half out (stop 0.875, eod 0.412, target 0.0). Where the decision
  minute has no measured NBBO row the declared imputation (median spread-% by price band × hour
  band of the measured sample) is used and the **imputed share is reported per cell**. Net = gross −
  that cost. The break-even is the **booked-set** cost of §2 of `hod_fresh/REPORT.md`
  (**0.061 TRAIN / 0.065 VAL**), NOT the retired 0.2151 constant.
* **Statistics per cell**: TRAIN and VAL, both TRAIN halves (H1 = `< 2025-07-01`, H2 = `>=`),
  iid t AND **day-clustered t** (clusters = trading days, `hod_preopen_regime` §4's rule), a
  **count-matched permutation null** (2,000 draws, pick count per week held fixed, cell's own P&L
  shuffled), green-week % over EVERY market week (no-trade = flat, in the denominator), worst week,
  max red streak, MDD, weekly $ at $100 risk, ex-top-5 %.
* **Bars**: 1-min from `data/cache.db` then the SIP side store, via
  `hod_filter_stack/pass2.load_bars` (byte-parity loader). Read-only on every DB. One python
  process at a time, `nice -n 15`, `ulimit -v 1500000`. Nothing outside this directory is written;
  `config.yaml`, `orb.yaml`, the systemd unit, the crons and every order path are untouched.

## 1. Bars — the two bars a cell is held to

* **Claim bar (G1/G2)** — G1: TRAIN net R > 0, iid t >= 2.0 **and** clustered t >= 2.0, >= 10
  trades/week. G2: VAL same sign **and** VAL green weeks >= 55 %. TEST once, behind `FREEZE.md`.
* **Live-exploration bar (the ship bar for this pass)** — **positive weekly $ AND green weeks
  >= 50 % on BOTH splits, at >= 10 trades/week, clustered t >= 2 on TRAIN, both TRAIN halves
  same-signed.** A cell clearing this is written up as SHIP-TO-DRY with the exact engine diff.

## 2. FRAME 1 — SHORT the failed break  (26 cells)

**Mechanism.** `hod_losers` §4: the losing long's path is +0.45 R by minute 6 and −1.06 R by minute
28. That path is a trade on the other side. The long book cannot monetise it (the bleed is already
booked by minute 10 — `hod_losers` Part 2 §1); a **short** entered at the confirmation is a
different instrument on the same information.

**Long facts per signal** (the B2 pre-book set, `score2.sig_set(tag='n', band=0)` — reproduction
gate R1 below): fill at `next_open` of bar `entry_m`; `level` = the HOD; long stop = the last-5-bar
low; `R_long = next_open − stop`.

**Triggers**, all on CLOSED 1-min bars strictly after the fill bar:

| id | trigger |
|---|---|
| `a01` | long ARMED (max high since fill >= `next_open + 0.4·R_long`) **and then** a bar closes <= `next_open + 0.1·R_long`, **within 15 minutes of the fill** |
| `a00` | same, close <= `next_open + 0.0·R_long` |
| `am1` | same, close <= `next_open − 0.1·R_long` |
| `fb` | the FIRST bar that closes **below `level`** (the HOD) after the fill — no arming condition, no 15-minute window |

**Short entry**: the NEXT bar's open after the trigger bar. **Obtainable** iff that open is
>= `trigger_bar_close × 0.994` — the mirror of the long's `ask <= level × 1.006` no-chase cap. The
unfilled counterfactual (what the un-obtainable rows would have done) is reported.

**Short stops** (above the entry): `mfe` = the highest high since the long's fill through the
trigger bar · `h05` = `level + 0.5·R_long` · `h10` = `level + 1.0·R_long`.
**Short targets** (below the entry): `t1` = entry − 1·R_short · `t2` = entry − 2·R_short ·
`tls` = the long's own stop level.
`R_short = stop − entry`. Exit walk from the bar after the short fill, priority EOD (15:55) → stop
(`high >= stop`, fill `max(stop, open)·(1+0.001)`) → target (`close <= target`, fill at the target)
— the sign-flipped twin of `hod_fresh/pass3.walk`.

**Cascade on the short**, the long cascade's mirror: `r_pct_short >= 1.0`, entry price >= $20,
spread <= 100 bps, spread/R <= 0.15, obtainable.

**The 26 cells**

| group | population | cells |
|---|---|---|
| **(a)** reversal on a long the book TOOK | the booked B2 long set | {`a01`,`a00`,`am1`} × `mfe` × {`tls`,`t2`} = **6** |
| **(b)** failed break on a long the book TOOK | the booked B2 long set | `fb` × `mfe` × {`tls`,`t2`} = **2** |
| **(c)** PURE short book, never long | the WHOLE B2 pre-book signal set, booked 12/4 on its own | `fb` × {`mfe`,`h05`,`h10`} × {`t1`,`t2`,`tls`} = **9** |
| **(d)** the mirror of D2 | (c) ∧ `spy_r5_pct < 0` (SPY 09:30→09:35 DOWN, known 09:35:00) | **9** |

**Borrow**: `shortable AND easy_to_borrow` from
`research/fuckup_audit/O_halt/PASSIVE/borrow_flags.csv` (Alpaca assets, TODAY's snapshot — the
survivorship caveat is stated, a delisted name is absent and counts as NOT borrowable). Every cell
is scored on the **full set AND the tradeable subset**; the tradeable share is reported.

**Cost on the short side**: measured NBBO is merged on the SHORT entry minute where
`bf_zero/causal_filter/nbbo.csv` has that exact `(day, symbol, minute)`; otherwise the declared
imputation. The measured share is reported per cell. A cell that clears the ship bar gets a
dedicated NBBO fetch before anything is recommended.

## 3. FRAME 2 — the noon conditional-mover book  (9 cells + one descriptive table)

**Mechanism.** `bf_zero` §6b: on symbol-days whose EOD range is >= 10 % the book reads **+0.43 R**
and on the others **−0.55 R**, in every split. EOD range is unobservable at 10:00 — but range is
monotone within a day, so "range so far >= X % at time T" is a causal proxy for it.

**Cells**: T ∈ {11:00, 12:00, 13:00} (minutes 660 / 720 / 780) × X ∈ {6, 7, 8} % = **9**.
**Population**: the FIRST break row at a minute strictly after T and <= 14:00 per symbol-day
(`pop.csv`, B2 detection gates: `stop_n` present, `dist_open_pct >= 5`, no rv band), the shipped
cascade (fill cap, `r_pct >= 1`, price >= $20, 100 bps, 15 % of R, obtainable), the B2 stop and the
+2 R target, on symbol-days whose **09:30→T session range** is >= X %. Book 12/4.
**Reported per cell**: trades/wk, gross, booked cost, net, green %, weekly $, both TRAIN halves,
clustered t, null band — and the frame's own decider:

> **P(EOD range >= 10 % | range >= X at T)** on the signal population, per (T, X).

**Pre-committed frame verdict**: the frame "has legs" iff some (T, X) shows conditional hit rate
>= 70 % AND >= 10 trades/week on both splits. That is a statement about the frame, not a ship.

## 4. FRAME 3 — size by regime, frequency preserved  (4 cells)

**Mechanism.** Every gate in 799 cells threw trades away and every one of them lost week shape to
arithmetic (`hod_losers` §4, `hod_preopen_regime` §5). A sizing rule keeps all ~30 picks/week and
changes only the dollars. `hod_preopen_regime` D2 (`spy_r5_pct > 0`, known 09:35:00, two minutes
before the earliest possible decision) separates the book by +0.388 / +0.189 R.

**Cells** (B2 book, pick set IDENTICAL to the shipped book in every cell):

| id | rule |
|---|---|
| `S1` | `risk = 100 × 1.5` on SPY-up days, `× 0.5` on SPY-down days |
| `S2` | `× 2.0` up, `× 0.5` down |
| `S3` | `× 2.0` up, `× 0.0` down — the gate itself, as the CONTROL |
| `S4` | the Frame-2 proxy as the key: `× 2.0` when the symbol-day's 09:30→12:00 range >= 7 %, else `× 0.5` |

**Check that must hold**: gross R/trade is IDENTICAL to the base in every cell (a sizing rule
cannot change it). It is printed as the check, and a cell whose gross moves is a bug.
**Scored on**: weekly $ at base $100, green weeks, worst week, MDD, max red streak.
**Null**: the sizing KEY is shuffled across days (2,000 draws), pick set held fixed.

## 5. Multiplicity

26 + 9 + 4 = **39 decision cells** × 2 splits, plus 1 descriptive table (Frame 2's conditional hit
rate, 9 entries, no decision attached) and the reproduction gate. Programme cumulative before this
pass: **799**. After: **838**. Expected largest |t| under a pure null over 39 cells × 2 splits
≈ 2.9–3.1 — stated beside every t.

## 6. Declared and NOT run

* A short book on the ARMED-long triggers over the whole pre-book population (not only the booked
  longs) — (a) is book-coupled by construction; the pure-short arm is `fb` only, per (c).
* Any refit of the long book's gates. This pass changes admission (Frame 2), side (Frame 1) or
  size (Frame 3) — never the 799 cells' knobs.
* F4–F8 (`FRAMES.md` queue) — the NEXT pass takes them, one heavy job at a time.
