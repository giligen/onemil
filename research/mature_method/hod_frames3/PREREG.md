# hod_frames3 — PREREGISTRATION (written and committed BEFORE any cell was scored)

Pass 3 of the frame programme on HOD-break. Three frames, in the order `hod_frames/FRAMES.md`
queued them: **F11 identify `consol_bars >= 20`**, **F10 the quarantined footprint field with the
gate re-specified**, **F12 ADAPTIVE vs FROZEN**.

Rails carried verbatim from `RUNBOOK.md` and the five previous PREREGs:
splits TRAIN = 2025-01-02..2025-12-31, VAL = 2026-01-01..2026-05-31, **TEST = 2026-06-01..2026-09-11
SEALED** (`FREEZE.md`; no script here reads a TEST-dated bar without `--test`), early closes removed,
test tickers removed (`research/scripts/pit_listings.is_test_ticker`), `daily_bars` membership
enforced, measured-NBBO cost with the declared price-band × hour-band imputation (never the retired
0.2151 constant — the booked-set cost is 0.061 TRAIN / 0.065 VAL and is **re-measured per cell**),
the engine's capped-limit fill with the obtainability test, book = **12/day × 4 concurrent at $100
risk**, count-matched permutation null (2,000 draws) on green weeks, **day-clustered t beside the iid
t** (`hod_preopen_regime` §4), both TRAIN halves reported with the same-signed-positive flag, MDE
stated on every rejection, availability audit on every new field with the >5 pp winner/loser
missingness drop rule.

---

## 0. Reproduction gate (run first, before any cell)

| id | what | reference |
|---|---|---|
| R1 | `B2` shipped book, TRAIN | 1,622 · 30.6/wk · −0.039 gross · −0.107 net · 32.1 % green · **−$17,346** |
| R2 | `B2` shipped book, VAL | 706 · 30.7/wk · +0.083 · +0.013 · 43.5 % green · **+$893** |
| R3 | `hod_fresh` **`consol_bars >= 20`** rung × the n5 stop | TRAIN 1,411 · 26.6 · +0.028 · −0.041 · 34.0 % · **−$5,749** · VAL 714 · 31.0 · +0.050 · −0.023 · 47.8 % · **−$1,629** |
| R4 | `hod_fresh` **C1** = that rung × `spy_r5_pct > 0` | TRAIN 731 · 13.8 · +0.100 · +0.033 · 43.4 % · **+$2,391** · VAL 368 · 16.0 · +0.123 · +0.050 · 47.8 % · **+$1,827** |

A mismatch on R1/R2 stops the pass. R3/R4 must reproduce from **this pass's own break rows**
(`hod_frames2/breaks2.csv` + `walk3.py`'s attached fields), which is the independent-rebuild
requirement of `feedback_independent_check_before_claims` applied to the object F11 is dissecting.

## 1. The bar pass — `walk3.py`, what is emitted and what is causal

`walk3.py` re-walks every symbol-day carrying at least one qualifying break in TRAIN+VAL (98,203
symbol-days / 344 sessions) and emits, for **every candidate break bar `i`**, four fields computed
from bars **strictly before `i`** (the decision is acted on at `m[i+1]`'s open):

* **(a) `hl_n20`** — the count of higher lows among the 20 bars ending at `i-1`
  (`#{j ∈ [i-20, i-1] : low[j] > low[j-1]}`). NaN if `i < 21`.
* **(a) `lo_slope20`** — the OLS slope of those 20 lows against bar index, expressed in **% of the
  break level per bar**. NaN if `i < 20`.
* **(b) `atr_now`** — mean 1-minute true range over bars `[i-14, i-1]`; **`atr_prev`** — the same over
  `[i-28, i-15]`; **`atr_ratio` = `atr_now / atr_prev`** (< 1 = the coil compressing into the break).
  NaN if `i < 29`.

Exits, stops and the qualification rule are NOT recomputed: they come from `hod_frames2/breaks2.csv`
(one row per qualifying break, already validated by that pass's exact independent rebuild, max
|Δrr| 3.55e-15). `walk3.py` only ATTACHES fields, keyed on (day, symbol, break_m).

**Declared structural confound, stated before scoring**: 20 held bars cannot occur before bar index
20, so `consol_bars >= 20` implies `break_m >= 590` (09:50); `hl_n20`/`lo_slope20` imply
`break_m >= 590`; `atr_ratio` implies `break_m >= 598`. **That is precisely what F11 (c) tests** and
it is why the clock cells are scored as rungs, not as caveats.

## 2. THE CELLS — 24 declared, scored exactly as written

### F11 — identify `consol_bars >= 20` (10 cells)

Cascade for every rung: the **B2 base** (`tag='n'`, `rv_profile >= 1`), the **shipped stop**
(plain last-5-bar low, `n5`), the **shipped exit** (eod → stop → target `entry + 2R`), the shipped
pre-book gates (fill cap, `r_pct >= 1 %`, price ≥ $20, spread ≤ 100 bps, spread ≤ 15 % of R,
obtainable), then the 12/4 book at $100 risk. Scan rule **KEEP-SCANNING** on a1–c2 (the admission is
the FIRST qualifying break satisfying the rung, a failing break does not retire the day) — the same
scan rule under which `hod_fresh` measured `consol_bars >= 20`.

| id | admission | component |
|---|---|---|
| F11-a1 | `hl_n20 >= 12` | (a) rising lows |
| F11-a2 | `hl_n20 >= 16` | (a) rising lows |
| F11-a3 | `lo_slope20 > 0` | (a) rising lows |
| F11-b1 | `atr_ratio <= 0.8` | (b) range compression |
| F11-b2 | `atr_ratio <= 0.6` | (b) range compression |
| F11-c1 | `break_m >= 590` (09:50 — the clock `consol_bars >= 20` structurally implies) | (c) the clock |
| F11-c2 | `break_m >= 600` (10:00) | (c) the clock |
| F11-d1 | `next_open >= P20` — **scan rule: FIRST-BREAK FILTER** on the base admission, where `P20` = the TRAIN median `next_open` of the R3 book, computed and printed before the cell is scored | (d) price band |
| F11-d2 | `sp_pct / r_pct <= S20` — **FIRST-BREAK FILTER**, `S20` = the TRAIN median spread-fraction-of-R of the R3 book, computed and printed before the cell is scored | (d) liquidity |
| F11-x1 | **conditional**: the ONE best (a)/(b) cell × `spy_r5_pct > 0` (the SPY 09:35 gate C1 uses). Scored only if some (a)/(b) cell is same-signed-positive on gross in H1/H2/VAL at ≥ 10 trades/week; if none is, the cell is **not scored** and that is reported. | the cross |

**Declared difference test**: `consol_bars >= 20` = `break_m >= 590` ∧ `consol_bars >= 20` by
construction. The difference is therefore reported as a **separation** inside the F11-c1 population
(kept vs rejected by `consol_bars >= 20`): Δ gross, n each side, iid t, **clustered t**, per half.

**Pre-committed verdict rule for F11** (written before any number is seen):
* **C1 is RETIRED as a clock/liquidity artefact** if any of F11-c1/c2/d1/d2 reproduces C1's
  qualitative signature (same-signed-positive gross in H1, H2 and VAL at ≥ 10 trades/week) while no
  (a)/(b) cell does.
* **C1 is identified as a MECHANISM** if some (a) or (b) cell is same-signed-positive on gross in
  H1, H2 and VAL at ≥ 10 trades/week while no (c)/(d) cell is — in which case F11-x1 is scored and
  the frame extends the mechanism.
* If **both** families or **neither** qualify, the verdict is **UNRESOLVED**, stated as such, with
  the difference test's clustered t as the tie-breaking evidence and no rule promoted either way.

### F10 — the quarantined footprint field, gate re-specified (8 cells)

`dollar_frac` = cumulative $ volume 09:30→break bar ÷ ADV$ (the mean of `close × volume` over the
symbol's 20 strictly-prior sessions in `research/bf_zero/universe.csv`), × 100. Population = the B2
pre-book **first-break** set. Rungs are the **TRAIN** percentiles of that set, computed once and
applied unchanged to both splits.

| id | admission (a FILTER on the B2 first break, the scan rule named) |
|---|---|
| F10-1..5 | `dollar_frac >= p50 / p60 / p70 / p80 / p90` of the TRAIN pre-book distribution |
| F10-6 | the selected rung **× `spy_r5_pct > 0`** |
| F10-7 | the selected rung **× `consol_bars >= 20`** |
| F10-8 | the selected rung **× C1** (`consol_bars >= 20` ∧ `spy_r5_pct > 0`) |

**The re-specified gate (the point of the frame).** Pass 2's exclusion rule
(`Δ mean rng_after <= 0` ∧ `Δ mean rng_sig > 0` ⇒ drop) is **withdrawn as an exclusion and retained
as a reported diagnostic**: per rung, ΔP(EOD range ≥ 10 %), Δ mean `rng_sig` (already-there),
Δ mean `rng_after` (arrived-after), and §2.3's own within-rung split (already-wide vs arrived-after
gross and WR). Nothing is excluded on it; it is printed beside every cell.

**The ONLY selector** is era-consistency: the **selected rung** is the rung whose **gross** is
positive in H1-2025, H2-2025 **and** VAL, at ≥ 10 trades/week on both splits, ranked by
`min(H1, H2, VAL)` gross. **If no rung is era-consistent, F10 is DEAD and F10-6/7/8 are NOT scored**
— declared here so the crosses cannot be fished.

**The dedicated NBBO fetch.** `dollar_frac`'s cost was 20–40 % imputed. After the rung is selected
(and only then), `fetch_nbbo3.py` fetches Alpaca **SIP** quotes at the selected rung's own signal
minutes (`entry_m - 1` mean spread; last quote at or before `entry_m` for the obtainability test) —
the same tooling, source and convention as `research/bf_zero/causal_filter/fetch_nbbo.py` — for
every row of the rung's TRAIN+VAL pre-book set that has no measured quote. The rung is then
**re-scored on the measured cost** and both numbers are reported.

### F12 — ADAPTIVE vs FROZEN (6 cells)

Population = the B2 pre-book signal set over TRAIN+VAL (the 74 market weeks of
2025-01-02..2026-05-31, W-FRI). Fields available to the refit: `entry_m`, `rv_profile`,
`dollar_frac`, `consol_bars`, `spy_r5_pct` — the short list declared by the queue, nothing else.

**The refit procedure, declared in full before any OOS week is booked.** For each arm
L ∈ {20, 26, 34} weeks and each OOS week `w` with at least L complete weeks before it:

1. `TRAINW` = every signal whose day is in the L weeks strictly before `w`.
   **Rail, asserted in code**: `max(day ∈ TRAINW) < min(day ∈ w)`; the run aborts if violated.
2. **Hour band**: deciles of `entry_m` on `TRAINW`; candidate bands = every contiguous decile range
   of width ≥ 4 (28 candidates) plus the full range ("no band"); score = mean **net R** on `TRAINW`;
   keep the argmax.
3. **One ranked feature cut**: for each feature in the short list, candidates are `>= p33` and
   `>= p67` of its band-filtered `TRAINW` distribution (`spy_r5_pct` additionally `> 0`), plus the
   null candidate "no cut"; a candidate is admissible only if it keeps ≥ 40 % of the band-filtered
   `TRAINW` signals; score = mean **net R** on the band-filtered `TRAINW`; keep the argmax.
4. Book week `w`'s signals under (band, cut) with the shipped 12/4 book at $100 risk.

| id | cell |
|---|---|
| F12-f | **FROZEN** B2 over the same OOS weeks (the reference) |
| F12-r20 / r26 / r34 | the **selection** refit, L = 20 / 26 / 34 weeks |
| F12-s26 | the **SIZING** refit control, L = 26: frozen admission, per-`entry_m`-tercile multiplier {1.5 best, 1.0 middle, 0.5 worst} fitted on `TRAINW` (ORB's shape, Q5-capped), applied to week `w` |
| F12-s34 | the same sizing control at L = 34 |

Reported per cell: % green weeks over **every** OOS market week (no-trade = flat, in the
denominator), longest red streak, worst week $, total $ at $100 risk, trades/week, MDD $, iid and
**day-clustered** t, and the **churn** — the share of consecutive OOS week pairs whose chosen
(band, cut) differs, plus the mean weekly $ of weeks that follow a change vs weeks that do not.

**Declared deviation**: the first L weeks are consumed by the initial window, so **no OOS week falls
in H1-2025**. The "both TRAIN halves" rail is therefore read on the OOS period's own halves — the
OOS weeks inside TRAIN (H2-2025) and the OOS weeks inside VAL — and this is stated wherever the
rail is invoked. TEST stays sealed; no OOS week is dated after 2026-05-31.

**Pre-committed reading rule.**
* **Reading (i), REGIME**: some refit arm beats FROZEN on **both** the owner metric (% green weeks)
  **and** total $ in **both** OOS halves, at ≥ 10 trades/week, with day-clustered t ≥ 2 on its own
  net R. Then the verdict names the live implementation (a weekly cron in the shape of
  `scripts/orb_weekly_refit.py` writing `HodBreakParams`) and the recommendation is SHIP TO DRY.
* **Reading (ii), NOISE**: otherwise. Then the pass states, in those words, that the separations
  this programme has found are noise and **every future frozen-rule pass on this book is
  pre-refuted**, and the sizing-control's churn is reported beside it so the two are separated.

## 3. What is reported per cell (no metric added after the fact)

n, trades/week, gross R, booked cost (measured per cell), net R, % green weeks, longest red streak,
worst week $, total $ at $100 risk, MDD $, ex-top-5 % net, imputed-cost share, iid t, **day-clustered
t**, H1/H2/VAL gross with the same-signed-positive flag, the count-matched null band, and the
already-there / arrived-after diagnostic (F10, and for the F11 cells that select a cohort).

## 4. The bars

* **Claim bar G1**: TRAIN net R > 0 with iid t ≥ 2.0 **AND** day-clustered t ≥ 2.0 at ≥ 10
  trades/week. G2 (VAL sign + ≥ 55 % green weeks) is evaluated only for a cell that clears G1.
  TEST opens only behind a committed recommendation.
* **Live-exploration bar**: positive weekly $ AND ≥ 50 % green weeks on BOTH splits at ≥ 10
  trades/week, clustered t ≥ 2 on TRAIN, halves same-signed positive.
* A cell that clears the live-exploration bar → **SHIP-TO-DRY** with the exact `HodBreakParams` diff.
* If none clears → **STAY DRY**, the MDE, and the next three frames appended to `FRAMES.md` with
  mechanisms.

## 5. Cell count / multiplicity

Programme cumulative through `hod_frames2`: **896**. This pass declares **24** decision cells
(10 F11 + 8 F10 + 6 F12) → **920**. Reproduction rows, the availability audit, the downstream
diagnostic and the difference/separation tables carry no decision and are not counted as cells.

## 6. Node rails

One python process, `nice -n 10`, `ulimit -v 3000000`, checkpointed per day. `cache.db`,
`bars_sip.db`, `daily_bars` opened **read-only**. No config, `orb.yaml`, systemd unit, cron, order or
cache is written. The dry run is not touched.
