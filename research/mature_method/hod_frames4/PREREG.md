# hod_frames4 — PREREG (F13 the slot rule · F14 the day as the unit · F15 who is breaking out)

**Committed BEFORE any cell was scored.** Pass 4 of the HOD-break frame programme
(`research/mature_method/hod_frames/FRAMES.md`). Frames exactly as the queue written by pass 3
declares them. TEST sealed (`FREEZE.md`).

**Standing constraint carried from F12 (pass 3):** *no cell in this pass may be a frozen trade-level
admission or exit rule — that class is pre-refuted.* Each frame below changes the **object**: F13 the
slot choice, F14 the day, F15 the name. Where a cell would degenerate into a trade-level admission
rule it is declared as a **diagnostic** and carries no decision.

---

## 0. Population, rails, and what is fixed before anything is scored

* **Population**: `sig4.csv` — the B2 pre-book signal set (7,027 rows, 344 sessions, 1,916 symbols,
  TRAIN+VAL only), built by `build4.py` from `hod_frames2/breaks2.csv` through the shipped cascade
  (`common3.sigset`: `entry_m <= 841`, `fill_capped`, `r_pct >= 1 %`, `next_open >= $20`, spread
  <= 100 bps, spread <= 15 % of R, obtainable under the no-chase cap). One row per symbol-day (the
  first qualifying break — the shipped `stale_break` scan rule).
* **Reproduction gate, run first and printed**: `B2` = TRAIN 1,622 / 30.6 per wk / −0.039 gross /
  −0.107 net / 32.1 % green / **−$17,346**; VAL 706 / 30.7 / +0.083 / +0.013 / 43.5 % / **+$893**;
  and the same book rebuilt from `breaks2.csv` (R2b). **Both reproduced EXACTLY before any cell**
  (`build4.log`).
* **Declared population difference from pass 3**: `hod_frames3/nbbo3.csv` (the F10 dedicated fetch,
  which did not exist when pass 3 reproduced the reference) is **NOT merged**; merging it moves the
  B2 rebuild by 17 trades. `use_nbbo3=True` is a declared sensitivity arm, reported if any cell
  clears.
* **Slot-machine parity**: this pass's generalised slot machine `common4.book_ranked(score=None)`
  must reproduce `trading.hod_break.run_book` row for row on the shipped book. **Asserted in code**
  (`build4.py`); the run aborts if it does not.
* **Rails on every cell**: both TRAIN halves (H1-2025 / H2-2025) and VAL printed; day-clustered t
  beside iid t (clusters = trading days, `hod_preopen_regime` §4's rule); count-matched permutation
  null on green weeks (2,000 draws, pick count held fixed); booked cost per cell (never the retired
  0.2151); MDE at 80 % power; availability audit with the 5 pp outcome-dependent-missingness drop
  rule on every NEW field; the already-there / arrived-after diagnostic (`hod_frames` §2.3) reported,
  never an exclusion.
* **Bars**. **Claim bar G1**: TRAIN net > 0 with iid *and* clustered t >= 2 at >= 10 trades/week.
  **Live-exploration bar**: positive dollars AND >= 50 % green weeks on BOTH splits at >= 10
  trades/week, clustered t >= 2 on TRAIN, halves same-signed positive.
* **Cell count**: 38 declared decision cells (16 F13 + 10 F14 + 12 F15). Programme cumulative
  **920 + 38 = 958**.

---

## 1. F13 — RANK, DON'T RACE (the slot rule). 16 cells.

### 1.1 The availability rail, declared before the cells

A signal's ONLY obtainable fill is the open of the minute after its break bar (the engine's capped
limit, `trading/hod_break.py`). A signal from an earlier minute is **not buyable later** at the price
this dataset prices it at. Therefore **the ranking operates among the signals arriving in the SAME
minute**, and a slot that frees at minute *m* is contested only by the signals of minute *m*. Any
"K-minute queue" that enters a stale candidate at its original price is an **unobtainable fill**
(CLAUDE.md's rule 1b) and is NOT scored as a book. **48.6 % (TRAIN) / 41.3 % (VAL) of signals share
their entry minute with at least one other signal** (measured before the cells, counts only, no
P&L — `build4.log`), so the rail leaves the frame real scope.

### 1.2 The ORACLE ceiling — bounds, NOT strategies (4 cells, excluded from both bars by construction)

Rank each day's signals by **realised net R** and take the top N. Declared *before* the causal cells
because the pre-committed rule is: **if the oracle book at 12/day is itself below the cost line, F13
is dead before any ranking is tried.**

| cell | rule |
|---|---|
| **F13-O1** | oracle top-4 per day, concurrency unconstrained |
| **F13-O2** | oracle top-8 per day, concurrency unconstrained |
| **F13-O3** | oracle top-12 per day, concurrency unconstrained |
| **F13-O4** | oracle top-12 per day **under 4 concurrent** (the ceiling a perfect ranker could actually book) |

### 1.3 Controls (2 cells)

| cell | rule |
|---|---|
| **F13-C0** | first-come by entry minute, ties by SYMBOL alphabetical — the shipped `run_book` (= B2) |
| **F13-C1** | first-come, **random** tie-break within the minute, 200 seeded draws — the tie-break's own noise band. **A ranking cell that sits inside this band is the alphabet, not skill.** |

### 1.4 Causal rankings (6 cells) — within-minute, fields known at that minute

| cell | score (best first) |
|---|---|
| **F13-r1** | `rv_profile` descending |
| **F13-r2** | `dollar_frac` descending (the F10 field; its cost is measured at p80) |
| **F13-r3** | `dist_open_pct / med_rng` descending (distance above the open normalised by the symbol's OWN 20-day median daily range) |
| **F13-r4** | `sp_pct / r_pct` **ascending** (cheapest first) |
| **F13-r5** | equal-weight composite z of (`rv_profile`, `dollar_frac`, `dist_open_pct/med_rng`, −`sp_pct/r_pct`), z-scored on **TRAIN only** |
| **F13-r6** | SPY-state × entry-minute. **Declared, and declared VOID if degenerate**: both `spy_r5_pct` and the entry minute are constant within a minute of a day, so this score cannot order simultaneous candidates. Reported as void rather than dropped silently. |

### 1.5 The slot COUNT and the reserve (4 cells)

| cell | rule |
|---|---|
| **F13-n8** | best causal ranker, 12/day, **8 concurrent** |
| **F13-n12** | best causal ranker, 12/day, **12 concurrent** |
| **F13-rs1** | **reserve 1** of the 4 slots until 10:30 (first-come otherwise) — a pure slot rule: it never looks at the candidate |
| **F13-rs2** | **reserve 2** of the 4 slots until 10:30 |

*(A score THRESHOLD that skips a candidate and keeps the slot for later is a trade-level admission
rule in slot clothing and is pre-refuted by F12. It is NOT declared as a cell; if it is computed at
all it appears in the supplementary section as a diagnostic with no decision attached.)*

### 1.6 F13's pre-committed selector

A ranking beats the race only if, on BOTH splits, it beats **F13-C0** on green weeks **and** on
dollars, sits **outside** the F13-C1 random-tie-break band, and is same-signed positive in H1, H2 and
VAL at >= 10 trades/week. Otherwise: **the ordering among simultaneous candidates is non-informative
and the slot rule is not the leak.** Reported either way with the oracle gap the frame leaves unclaimed.

---

## 2. F14 — THE DAY AS THE UNIT. 10 cells.

**The observation is the trading DAY**, not the trade: `y_d` = Σ pnl of the shipped 12/4 B2 book on
day *d*, at $100 risk, with **no-trade days entered as 0** and kept in the denominator. n = 242
TRAIN (H1 121 / H2 121 by date) + 102 VAL days. All inference is day-level by construction; the iid
t on days IS the clustered t (printed side by side with the trade-level numbers so the two are never
confused).

Day fields, all realised by **09:35:00** and all from `hod_preopen_regime/{day_fields.csv,
idx_1min.csv}` (Alpaca SIP 1-min SPY/QQQ) + `cache.db daily_bars` (read-only):

* `spy_r5_pct` — SPY 09:30 bar open -> 09:34 bar close (the D2 gate)
* `qqq_r5_pct` — the same on QQQ
* `spy_rng5_atr` — SPY 09:30–09:34 high-low, divided by SPY's own 20-session mean daily
  (high−low)/open ending T−1
* `breadth_0935` — the count of the book's OWN pre-book signals with `entry_m <= 576` (causal: they
  have already broken)

**IWM and the VIX open gap are NOT available in this repo's index tape (SPY and QQQ only) and are
declared NOT SCORED rather than fetched.**

| cell | day gate |
|---|---|
| **F14-d0** | base — every day (the distribution, the day-level MDE at n = 242 / 102) |
| **F14-d1** | `spy_r5_pct > 0` (the D2 gate, re-measured with the day as the unit) |
| **F14-d2** | `spy_r5_pct` TRAIN terciles (down / up-small / up-big) — **magnitude, not only sign** |
| **F14-d3** | `qqq_r5_pct > 0` |
| **F14-d4** | `spy_r5_pct > 0 AND qqq_r5_pct > 0` (agreement) |
| **F14-d5** | `spy_rng5_atr` TRAIN terciles |
| **F14-d6** | `breadth_0935` TRAIN terciles |
| **F14-d7** | **the H2-2025 question**: within H2-2025 alone, the best day state from d1–d6 — declared as a SEARCH over 6 states, with the multiplicity stated and a VAL read of whatever it finds. A state that is positive in H2 only is reported as a description, never as a rule. |
| **F14-d8** | the day gate (d1) **crossed with F13's best causal ranking** — day selection × within-day selection, the two-object book |
| **F14-d9** | the day gate (d1) crossed with the **oracle** day set (days whose realised book P&L > 0) — a bound on what any day gate can be worth |

**Pre-committed selector**: a day gate counts only if its day-level mean is positive in H1-2025, in
H2-2025 and in VAL, with a day-level t >= 2 on TRAIN and >= 50 % green weeks on both splits at the
frequency it leaves. The frame's stated question — *is there any 09:35 state under which H2-2025 is
positive?* — is answered YES only if d7's state is also non-negative on VAL.

---

## 3. F15 — WHO IS BREAKING OUT. 12 cells.

Instrument attributes, all static-at-the-open and point-in-time. Every field goes through the
availability audit FIRST (coverage, and missingness on winners vs losers with the 5 pp drop rule); a
field failing it is reported as a diagnostic and not scored.

**Sources** (all read-only, all already causal or made causal here):

* **Short interest** — FINRA consolidated short interest, public API, re-fetched by
  `fetch_finra.py` into `short_interest.csv` (settlement 2024-11-15 -> 2026-08-31). Keyed on
  **`usable_from` = settlementDate + 13 calendar days** (FINRA disseminates ~T+9 trading days), i.e.
  a signal on day *D* may only see reports with `usable_from <= D`; the most recent such report is
  used. `daysToCoverQuantity` is FINRA's own SI ÷ its own ADV; `si_ratio` = SI ÷ our `adv20`.
  *(This field has been dropped on availability twice before; the fetch is part of this pass.)*
* **Shares outstanding / float turnover** — `research/multiday/data/{shares_facts,cik_map}.parquet`
  (EDGAR `dei:EntityCommonStockSharesOutstanding`, keyed on the FILING date). **Declared VOID if
  coverage on the pre-book signal set is < 50 %.**
* **Asset class** — `trading/orb_asset_class.py` + `data/research/orb_asset_class_map_20260711.csv`
  + `data/research/alpaca_assets_all_20260905.csv` (symbol -> name), and `underlying_anchor` for the
  wrapper/underlying cohort.
* **Listing venue** — `research/scripts/pit_listings.py` (Databento PIT definitions), point-in-time.
* **Premarket news** — `data/research/orb_news_catalyst_nightly.csv` (own-ticker articles, window
  prev-day 15:00 ET -> 09:31, already causal). **Coverage of this file is the ORB candidate set, not
  our universe; the availability audit decides whether it is a cell or a diagnostic.**

| cell | admission on the shipped B2 book |
|---|---|
| **F15-a1** | `daysToCover >= TRAIN median` |
| **F15-a2** | `daysToCover >= TRAIN p75` |
| **F15-a3** | `si_ratio (SI / adv20) >= TRAIN p75` |
| **F15-b1** | float turnover `cumv / shares_out >= TRAIN p75` (VOID if coverage < 50 %) |
| **F15-b2** | `shares_out <= TRAIN p25` (low float) (VOID if coverage < 50 %) |
| **F15-c1** | common stock only (`orb_asset_class.classify_asset == 'stock'`) |
| **F15-c2** | leveraged/inverse wrapper only |
| **F15-c3** | wrapper OR stock whose **underlying anchor** has >= 2 signals the same morning (the ORB complex-confirmation cohort) |
| **F15-d1** | listing venue NASDAQ |
| **F15-d2** | listing venue NYSE / NYSE-Arca / AMEX |
| **F15-e1** | own-ticker premarket news present (`n_articles >= 1`) |
| **F15-e2** | own-ticker premarket news **recent** (`latest` within the 09:30 session's own premarket, i.e. same calendar morning) |

**Pre-committed selector**: era-consistency (same-signed positive gross in H1-2025, H2-2025 and VAL)
at >= 10 trades/week is the ONLY selector, exactly as F10's was. A cell passing it goes to the two
bars; nothing else is promoted. The already-there / arrived-after split is printed per cell as a
diagnostic.

---

## 4. Deliverable and the verdict rule

One `REPORT.md`: the reproduction line, the three frames' tables, the nulls, both bars, the MDE per
frame, the multiplicity, and ONE of **SHIP-TO-DRY** (with the exact diff — F13 would be a change to
`trading/hod_break.py::run_book` and the engine's candidate queue ordering; F14 a day gate in the
engine's session setup; F15 an admission field plus a data feed and its nightly job) or **STAY-DRY**
(with the MDE per frame and the NEXT THREE FRAMES with mechanisms appended to `FRAMES.md`).

No config, `orb.yaml`, systemd unit, cron, order or cache is written. The dry run is not touched.
**TEST is not opened.**
