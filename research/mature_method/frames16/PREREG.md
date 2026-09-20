# frames16 — PREREGISTRATION

Pass 16 of the frame programme. Written and committed **before any cell was scored**.
Programme cell count **1,240 → 1,252** (12 declared cells below; arm 1 is a DIAGNOSTIC and
declares no cell, and every diagnostic in arms 2 and 3 is named here so it cannot be promoted to a
cell after the fact).

Three arms, in the order they run:

1. **ARM 1 — instrument calibration of the OFI measurement** (diagnostic, 0 cells).
2. **ARM 2 — the lambda residual at the break** (4 cells; runs ONLY if arm 1 passes its rule).
3. **ARM 3 — F49, the discovered mirror priced as a SHORT** (8 cells; runs unconditionally).

---

## 0. WHY THIS PASS EXISTS, AND WHAT WOULD FALSIFY EACH ARM

**Arm 1.** An outside reviewer's critique of `hod_filter_stack` arm O: the CKS order-flow-imbalance
number was computed from `bbo-1s` — a **1-second-sampled** best quote from **EQUS.MINI, one
publisher**. The indicator arithmetic in `hod_filter_stack/ofi.py` has been read and its four
Cont–Kukanov–Stoikov indicator cases are correct. The open question is the INSTRUMENT. CKS's own
contemporaneous result is that a 10-second mid-price change is close to linear in depth-normalised
OFI over the same window, with R² of roughly 0.6–0.7 on liquid names **when every L1 update is
seen**. If 1-second sampling destroys that relation, arm O measured a different object and its
verdict is not a test of the OFI hypothesis — it is a test of a mutilated one.

*Pre-committed decision rule, written before any pull:*

| condition | conclusion |
|---|---|
| full-depth `mbp-1` R² **≥ 0.30** on a name AND `bbo-1s` on the SAME name/session **< 0.10** | **arm O is NOT-A-TEST** (instrument failure). The honest statement becomes: *"OFI has not yet been measured on this book; a real test needs `mbp-1`, priced at ≈ $450 for the B2 population — the owner's call."* |
| both instruments **≥ 0.30** | arm O **stands as measured** (order flow is not a filter on this book). |
| `mbp-1` itself **< 0.30 on the large-cap** | **the code is suspect.** Debug before anything else: print the first 20 quote events by hand and re-derive the increments. No other arm's conclusion is reported until this is resolved. |
| `mbp-1` ≥ 0.30 on the large-cap but `bbo-1s` in [0.10, 0.30) | partial degradation: reported as such, arm O labelled **WEAKENED, not void**, and arm 2 is judged on the small-cap number alone. |

The decision turns on the **small-cap** name (the HOD-break population) for arm 2's runnability and
on the **large-cap** name for the code check; the mid-cap is the interpolation point.

**Arm 2.** The reviewer's second point: raw OFI is **range-restricted at a break** — by construction
a HOD break happens when buyers lift the offer, so nearly every signal has positive OFI and the
cross-section of OFI at the break carries little information. The discriminator he proposes is the
**residual**: price impact per unit of flow, `λ = Δp / (OFI/depth)`. A high λ means the price moved a
long way on little flow — a break into a **thin book** (a vacuum, which retraces); a low λ means the
same price move required real size (genuine aggression). This is a different functional of the same
data and has never been computed here.

*Falsifier.* The arm is REFUTED if no λ or residual tercile rule clears the bar in §4 on TRAIN and
VAL together. The arm is **UNRUNNABLE** (and says nothing about λ) if arm 1's rule fires
NOT-A-TEST — a residual computed from an instrument that does not reproduce the contemporaneous
relation is not a measurement of λ, and I will not report one.

**Arm 3 (F49).** `frames15` B12 is the largest |t| in 1,240 cells: a name already ≥ 5 % above its
session open, trading ≥ 3× its own hourly normal, **with a large price move in that hour**, books
**−0.398 R TRAIN / −0.323 R VAL (day-clustered t −13.3 / −10.5)** on 3,601 long trades in both TRAIN
halves. It has never been priced on its natural side. The mechanism claim is that the hour's move
has already paid for the volume ("already discovered") and the next bar is a fade.

*Falsifier, pre-committed, any ONE of which kills the frame:*
1. the names that carry it are **not borrowable** (primary book = Alpaca `shortable AND
   easy_to_borrow` only; unknown symbol ⇒ NOT shortable);
2. the assumed borrow cost ≥ the measured edge;
3. the edge lives in the same **top 5 %** of trades that kills every long cell in this programme
   (ex-top-5 % on the uncapped exit must stay positive);
4. the **SSR** uptick rule blocks the fills that carry it;
5. the mirror is **not worse than its two placebos** — the claim is specifically that the MIRROR
   (volume WITH price) is the expensive long, not that high volume generally is.

*Expectation stated in advance so the pass cannot be written to its result.* Frame F2 (`frames7`,
2026-09-19) already priced a short book on this same tape — "SHORT the failed break", 26 cells,
**DEAD**, and it found that (a) borrow is **not** the binding constraint (62.3 % of short signals
were shortable ∧ ETB) and (b) **short R is 0.56× the long R**, so the same quoted spread costs about
**1.8× more per R on the short side**. B12's long loss is −0.398 R at a 2 % stop = **−0.80 % of
price**; the measured round-trip cost at these clocks is 0.23–0.39 % of price, which at the short's
own R is a materially larger R-charge. My prior is therefore that the short is gross-positive and
net-marginal, and that the pass turns on the cost, the borrow rail and the tail — not on the sign.

---

## 1. SHARED RAILS (every arm)

1. **Reproduction gate, asserted in code, raising, before any cell is read.** `frames15`'s intraday
   population must reproduce: 42,224 walked long signals, `gate5` share **26.93 %**, and on the
   gate5 population the mirror cell B12 must read TRAIN n = 2,238 gross **−0.398 R**, VAL n = 1,363
   gross **−0.323 R** (day-clustered t −13.33 / −10.51). Arm 2's gate is `hod_filter_stack` arm O:
   `ofi.csv` 7,027 rows, 7,001 merged onto B2, 6,617 with ≥ 20 updates.
2. **TEST sealed** (`FREEZE.md`). TRAIN = 2025-01-02…2025-12-31, VAL = 2026-01-01…2026-05-31,
   TEST = 2026-06-01…2026-09-11, opened by NO result in this pass.
3. **Availability audit on every field**: coverage, and winner-vs-loser missingness. Coverage < 80 %
   on the population scored, or a winner/loser missingness gap > 5 pp ⇒ the cell is **VOID by the
   rail**, decided before its number is read (`hod_frames2`'s `add30_ratio` rule).
4. **Day-clustered SE printed beside iid** on every claim; **% of price printed beside R** on every
   book; both TRAIN halves (H1 = 2025-01…06, H2 = 2025-07…12) printed and required same-signed.
5. **Count-matched permutation null**, 2,000 draws, per-week pick count held fixed, on every
   green-week claim (`common15.null_green`).
6. **Ex-top-5 %** on every **uncapped** exit; reported as a diagnostic on capped exits.
7. **Universe hygiene**: test tickers (`^Z[A-Z]ZZT$`) excluded; leveraged wrappers excluded from
   arm 3 by `trading/orb_asset_class` (their inverse already exists, so shorting them is a different
   trade) and reported as their own diagnostic split; price ≥ $5.
8. **Multiplicity**: 1,240 prior cells + 12 declared here = **1,252**, printed in the REPORT.
9. **Node**: ONE python process, `nice -n 10`, `ulimit -v 3000000`; every store opened read-only
   (`file:...?mode=ro`); nothing written outside `frames16/`; no config, `orb.yaml`, engine,
   checker, systemd unit, cron or order touched. No subagents.
10. **Databento**: every pull priced with `metadata.get_cost` BEFORE it is made and the price
    printed; **hard cap $12.00 for the pass**; raw quote payloads for arm 1 are **kept on disk**
    under `frames16/raw/` (arm O discarded its raw quotes, which is why this calibration has to
    re-pull).

---

## 2. ARM 1 — INSTRUMENT CALIBRATION (diagnostic, 0 cells)

**Session**: 2026-03-11 (a VAL-period session; arm 1 scores no book, so no split is consumed).
**Names**, one per liquidity tier, all three signalling in the B2 / HOD population band:

| tier | symbol | why |
|---|---|---|
| large-cap | **AAPL** | the code check — the contemporaneous relation must appear here or the implementation is wrong |
| mid-cap | **FSLY** | $24 price, ADV 21.6 M shares, a B2 signal on this session (break 10:04) |
| small-cap (HOD population) | **USAX** | $26 price, ADV 186 K shares ≈ $4.8 M/day, a B2 signal on this session (break 10:22) — **this is the tier arm 2's runnability is decided on** |

**Instruments priced (all nine priced before any pull; `price.csv`)**:

| dataset | schema | cost for 3 names × 1 session | records |
|---|---|---|---|
| EQUS.MINI | `mbp-1` | $0.1329 | 1,486,874 |
| EQUS.MINI | `bbo-1s` | $0.0120 | 40,123 |
| EQUS.MINI | `tbbo` | $0.0089 | 19,972 |
| XNAS.ITCH | `mbp-1` | $0.1105 | 1,236,096 |
| XNAS.ITCH | `bbo-1s` | $0.0148 | 49,578 |
| XNAS.BASIC | `cmbp-1` | $0.1422 | 1,590,767 |
| XNAS.BASIC | `cbbo-1s` | $0.0166 | 55,764 |
| XNYS.PILLAR | `mbp-1` | $0.0262 | 292,853 |
| XNYS.PILLAR | `bbo-1s` | $0.0085 | 28,511 |

**All nine total $0.4726** — inside the $12 cap, so all nine are authorised and pulled. `cmbp-1` /
`cbbo-1s` on XNAS.BASIC are the **consolidated** comparison the task asked to be priced; XNAS.ITCH
is the **full single-venue** comparison (is EQUS.MINI's publisher sparsity the problem, or the
1-second clock?); XNYS.PILLAR is the third venue.

**Measurement.** For each (name, instrument): take the RTH session 09:30–16:00 ET; form
non-overlapping **10-second** windows; within each window accumulate the CKS increment from
successive best-quote snapshots

    e_n = 1{P^b_n >= P^b_{n-1}} q^b_n - 1{P^b_n <= P^b_{n-1}} q^b_{n-1}
        - 1{P^a_n <= P^a_{n-1}} q^a_n + 1{P^a_n >= P^a_{n-1}} q^a_{n-1}

(the same four cases as `hod_filter_stack/ofi.py`, re-derived here from the paper's prose rather
than copied), set `OFI_w = Σ e_n` and `D_w = mean (q^b + q^a)/2` over the window, and regress

    Δmid_w  =  α + β · (OFI_w / D_w) + ε          (Δmid in dollars, OLS, per name per instrument)

Reported per name per instrument: **R², β, β's t, n windows, the number of quote-change events the
instrument delivered, and the fraction of `mbp-1` quote events lost at 1-second sampling**
(`1 − n_events(bbo-1s) / n_events(mbp-1)` on the same name/session/venue).

**Windows with zero flow** (`D_w = 0` or no quote change) are excluded and their count reported —
excluding them is favourable to the instrument under test, which is the conservative direction for a
NOT-A-TEST finding.

---

## 3. ARM 2 — THE LAMBDA RESIDUAL AT THE BREAK (4 cells, conditional)

**Runs only if** arm 1's small-cap `bbo-1s` R² ≥ 0.30 **and** the priced B2 pull fits the budget
remaining under the $12 cap. Otherwise the arm is declared **UNRUNNABLE AT THIS PRICE**, the reason
and the price are printed, and **no λ number is reported** — an unmeasurable quantity is not
reported as a null (PLAN §1 phrasing rule).

**Population**: `hod_filter_stack/b2.pkl`, TRAIN + VAL, the HOD-break booked signals (the same
population arm O used, so the two are directly comparable).

**Construction.** Per name-day, fit `β_nd` on the **general population of that name-day**: all 10-s
windows in the **30 minutes ending at the start of the consolidation** (i.e. strictly before the
break bar and before the bars that formed the level), by the same regression as arm 1. Then at the
break:

    λ_break      = Δp_break / (OFI_break / D_break)            -- per-event price impact
    resid_break  = Δp_break - β_nd · (OFI_break / D_break)     -- the residual on the name-day's own β

where `Δp_break` is the mid change over the break bar and `OFI_break / D_break` the depth-normalised
flow over the same minute. **Causality**: every input is timestamped at or before the break bar's
close; `β_nd` uses only windows ending ≥ 1 minute before the consolidation starts. A name-day with
fewer than 30 usable fitting windows, or with `|OFI_break/D_break|` below a floor of 1e-6, is
**missing**, and the availability rail of §1.3 applies.

**Declared cells** (terciles cut on TRAIN edges, keep the best tercile, the same selection rule the
causal-filter study used):

| # | cell |
|---|---|
| **L1** | λ terciles, keep best, HOD's shipped bare/stop exit |
| **L2** | λ terciles, keep best, HOD's +2R bracket |
| **L3** | residual terciles, keep best, bare/stop exit |
| **L4** | residual terciles, keep best, +2R bracket |

**Declared diagnostics (never promoted to cells)**: forward **mid** returns at 10 s / 30 s / 60 s /
5 min after the break, by λ tercile and by residual tercile — the **DECAY STRUCTURE**. This is
explicitly *not* a money horizon: at a 20 bps quoted spread a bracket cannot monetise a 10-second
move, and any positive number there is reported as structure, not as a book.

**Coverage rail**: ≥ 80 % of B2 signals must carry λ, with a winner/loser missingness gap ≤ 5 pp, or
the arm is VOID by §1.3.

---

## 4. THE BAR (identical for arms 2 and 3, unchanged from pass 15, pre-committed)

A cell **CLEARS** only if, on **BOTH** splits: positive weekly $ at $100 risk, green weeks ≥ 50 %,
≥ 10 trades a week, day-clustered t ≥ 2, and the two TRAIN halves same-signed; plus, for uncapped
exits, ex-top-5 % still positive. A cell that clears goes to **SHIP-TO-DRY** with its exact spec.
Anything else is **STAY-DRY** and the **MDE** is printed so the null is quantified rather than
asserted.

Secondary, reported not gated (the live-exploration bar of `RUNBOOK` step 10): positive point
estimate on green weeks AND on dollars at live size on both splits, a stated mechanism, bounded
downside with a pre-committed stop, resolution inside a quarter at the book's own frequency.

---

## 5. ARM 3 — F49: THE DISCOVERED MIRROR AS A SHORT (8 cells)

### 5.1 The signal (unchanged from `frames15` B12, re-derived here)

At the **close of a session hour** h ∈ {H9 … H14} (H9 = 09:30–09:59, H14 = 15:00–15:59 is excluded
because it leaves no session to trade — h ∈ 9…14 means the hour whose close is 10:00 … 15:00):

* `hrv_h ≥ 3` — the hour's volume ≥ 3× `adv20 × share_h`, the symbol's own hour shape
  (`frames15/hourly15.parquet`, built from prior sessions only);
* **the mirror cut on the hour's return**, in two declared forms:
  * **`MIR2`** (the `frames15` B12 object verbatim): `|hour_ret| > 2 %`;
  * **`UP2`** (the natural short leg): `hour_ret > +2 %`;
* `gate5` **causal membership**: the session high **up to that hour's close** already reached
  session open × 1.05 — computed from the tape at the decision bar, never end-of-day;
* price ≥ $5, not a leveraged wrapper, not a test ticker.

### 5.2 The entry — the mirror of the engine's no-chase cap, never a touch

A resting **sell limit** at `ref × (1 − cap)`, `ref` = the last 1-minute close before the hour cut,
`cap = 0.6 %` (the same cap the long side uses). It fills **at the next bar's OPEN** iff that open
`≥ ref × (1 − cap)`; if the bar opens below the limit the order does not fill and the signal is a
**SKIP with 0 P&L, never a loss**. This is the short-side statement of RUNBOOK step 4 and it is the
only fill convention this house uses. **Declared diagnostic**: cap ∈ {0.3 %, 0.6 %, 1.2 %, none},
plus the unfilled counterfactual (what the skipped signals would have returned) — the halt-resume
adverse-selection check, run on this book because a sell limit below the market is structurally the
same object that selected badly there.

### 5.3 The stop and R — two declared specs

* **Spec A (`2pct`)** — stop at `entry × 1.02`, `R = 2 % of price`. The direct mirror of B12, so the
  short's gross R is comparable to the long's number cell for cell.
* **Spec B (`hrhigh`)** — stop at `max(signal hour's high, entry × 1.01)`, `R = stop − entry`. The
  natural structural stop; the 1 % floor is declared here and is the standing `r_pct ≥ 1 %` rail.

### 5.4 The exits

* **bare** — cover at the stop, or flat at 15:55 (`EOD_M = 955`), whichever first. **Uncapped**, so
  ex-top-5 % is a gating diagnostic on it.
* **bracket** — `+2R` resting buy-limit target at `entry − 2R`, filled on a bar **close** at or
  below it (never a touch), stop as above, flat at 15:55.

Walk priority, mirroring `frames15/armB_intra.walk` exactly: from bar e+1 — EOD first, then stop,
then target. Stop fills at `max(stop, that bar's open) × (1 + 0.001)` (one slip against us, the same
0.1 % the long side is charged). EOD fills at that bar's open.

### 5.5 HONEST SHORT COSTS — every one, declared before scoring

1. **Spread.** The `frames14` F45 **measured** minute-of-day NBBO median (`f45_minute_table.csv`),
   half at the entry minute + half at the exit minute, in % of price, converted to R at the cell's
   own `r_pct`. The old `S.IMPUTE` model is valid only 09:37–14:01 and is **not** used. Charging a
   half spread on the bracket's resting target leg is **conservative** (the per-outcome contract
   charges a resting limit nothing) and that is stated with the number.
2. **Borrow availability.** Alpaca's own asset flags (`research/fuckup_audit/O_halt/PASSIVE/
   borrow_flags.csv`, 14,355 symbols, `shortable` ∧ `easy_to_borrow`). **Primary book = ETB only.**
   A symbol **absent** from the list is treated as **NOT shortable**. This is today's snapshot, not
   the borrow state on the trade date — a **survivorship caveat stated with every number**, and it
   is the reason the non-ETB split is reported separately rather than estimated.
3. **Borrow fee.** ETB, intraday, flat before 15:55: **assumed 0** (no overnight borrow accrues on a
   position closed the same session, and Alpaca charges no locate on ETB names). HTB names are
   **excluded, not estimated** — a fee we cannot measure is not a number.
4. **The SSR uptick rule.** A name whose session low reaches **prior close × 0.90** triggers Reg SHO
   Rule 201 for the remainder of that session **and all of the next**. Prior close from
   `frames15/daily15_*.parquet` (`p_close`). Under SSR a short may only be displayed above the
   national best bid, so a marketable sell limit is modelled as filling **only if the entry bar's
   open > the previous bar's close** (an uptick); otherwise **no fill, 0 P&L**. The **fraction of
   signals SSR-blocked** is reported; a diagnostic book ignoring SSR is printed beside it.
5. **Wrappers excluded** (their inverse already exists, so the trade is a different instrument);
   their split is reported.
6. **Price ≥ $5.**

### 5.6 The declared cells

| # | signal | stop spec | exit |
|---|---|---|---|
| **S1** | `MIR2` (B12 verbatim) | `2pct` | bare |
| **S2** | `MIR2` | `2pct` | bracket +2R |
| **S3** | `UP2` (the natural short leg) | `2pct` | bare |
| **S4** | `UP2` | `2pct` | bracket +2R |
| **S5** | `MIR2` | `hrhigh` | bare |
| **S6** | `MIR2` | `hrhigh` | bracket +2R |
| **P1** | **placebo** — high RV, NO price: `hrv ≥ 3`, `\|hour_ret\| ≤ 1 %` (the absorption cell) | `2pct` | bare |
| **P2** | **placebo** — a RANDOM other hour close on the same gate5 signal name-days (seed 16) | `2pct` | bare |

The F49 claim is specifically that **S1/S3 beat both P1 and P2**. If the mirror short is no better
than shorting any high-volume hour, or than shorting a random hour of the same name-day, the frame
is refuted even if S1 is positive — the object would be "short gap-up names", not "short the
discovered mirror", and that is a different (and already-crowded) claim.

### 5.7 Reported for every cell

n · per-week count · gross R · gross % of price · cost R · net R · **t iid AND day-clustered** ·
both TRAIN halves · VAL · ex-top-5 % (bare only) · green weeks vs the count-matched null band ·
weekly $ at $100 risk · worst week · longest red streak · **share of signals unshortable** · **share
SSR-blocked** · **share skipped by the cap** and the unfilled counterfactual · wrapper share · MDE.

---

## 6. WHAT THIS PASS WILL NOT DO

* It will not open TEST, under any result (`FREEZE.md`).
* It will not touch `config.yaml`, `orb.yaml`, any engine, checker, systemd unit, cron or order.
  A SHIP-TO-DRY verdict is a recommendation to the owner, not a change.
* It will not report a λ number if arm 1 says the instrument cannot carry λ.
* It will not report "no edge exists". Every null is phrased as PLAN §1 requires, with the MDE.
