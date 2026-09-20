# frames15 — PREREGISTRATION: INSTITUTIONAL VOLUME AS A PATTERN OVER TIME

Written and committed **before any cell was scored**. Pass 15 of the frame programme.
Programme cell count **1,217 → 1,240** (23 scored cells declared below, plus 7 declared
diagnostics that are named here so they cannot be promoted to cells after the fact).

Owner 2026-09-20: *"the institutional volume is a big one imo — by collecting relative hourly
volume of the stocks and identifying a real interest, this is huge, money must be there."*
Owner addendum, same day: *"identify the interest — might be multi-day as well."*

---

## 0. THE PREDICTION AND THE FALSIFIER (written first, before any number was read)

**Prediction.** If institutions accumulate a position over hours or days, the footprint is
*volume elevated against that stock's own normal at the same clock, sustained, and not yet paid for
in price*. The three testable consequences: (i) a symbol-day carrying sustained elevated volume is
a better name-day than one that does not, so the field should separate HOD-break's booked picks by
gross R; (ii) the footprint should stand on its own as an admission rule with a positive book;
(iii) because accumulation precedes the move, the ABSORPTION cell (high volume, small price change)
should beat its mirror (high volume, large price change = already discovered).

**Falsifier, pre-committed.** The frame is REFUTED if, on TRAIN and VAL together, no declared cell
clears the bar below, AND the absorption cell does not beat its mirror on both splits. A field whose
separation is positive on one split and negative on the other is NOISE and is reported as such — no
arm is selected after the fact, exactly as `hod_frames6` §F20 required.

**Expectation stated in advance (so the pass cannot be written to its result).** Thirteen previous
passes have found exactly one significant object on this tape — NAME-DAY selection, +0.18 / +0.25 R
(`frames7` F24, `frames13` F41) — and `hod_frames2` F9 showed that the cheap forms of cohort
membership carry no return. `rv_profile` (cumulative volume ÷ ADV20 × clock fraction) was the best
single feature of `hod_filter_stack` at +0.28 / +0.25 TRAIN and **sign-flipped to −0.096 / −0.115 on
VAL**. My prior is therefore that V1 (a spot hourly RV) repeats that failure, and that if anything
survives it is V3 (multi-day) or V4 (absorption), because those are the only forms that are NOT a
restatement of "today is a busy day", which the programme has already priced at zero.

## 0b. HOW THIS DIFFERS FROM WHAT HAS ALREADY DIED (required, before the fields)

| retired form | what it was | why this pass is not it |
|---|---|---|
| `rv_profile` (`hod_filter_stack`, `hod_frames2`) | cum RTH volume ÷ (ADV20 × **universal clock fraction**), in [1,5); the ≥5 arm | denominator is a **cross-sectionally shared** clock curve and the numerator is **cumulative** (dominated by the open). V1 here is **bar-level per hour** against the **symbol's OWN hour shape**. |
| `bar_vol_x` | the breakout bar's own volume ÷ its recent bars | one minute; this pass never scores a one-minute form. |
| `dollar_frac` | cum $vol ÷ ADV$, era-consistent only at p90 under a frequency floor | same cumulative-since-open object as `rv_profile`, in dollars. |
| volume as an **exit** trigger (`hod_frames2` F6) | inert | this pass scores volume only as **name-day selection** and as **admission**, never as an exit. |
| frozen trade-level rules on HOD (`hod_frames3` F12) | noise | no cell here is a trade-minute filter on the shipped break. |

## 1. THE FIELDS — definitions, decision times, availability rails

All fields are computed from data strictly BEFORE their decision time; every one is availability
audited, and a field whose coverage is < 80 % on the population it is scored on, or whose
missingness differs by > 5 pp between winners and losers, is **VOID by this rail** (the
`hod_frames2` `add30_ratio` rule), decided before its number is read.

**The per-stock hourly profile (the object the owner asked for).**
Session hours in ET: `H9` = 09:30–09:59, then `H10 … H15` = each clock hour to 15:59.
For symbol `s`, session `t`, hour `h`:

    share_h(s,t)   = mean over the symbol's PRIOR available intraday sessions in the last 60
                     sessions (>= 3 required) of  [hour-h RTH volume / that session's RTH volume]
    hourmean_h(s,t)= ADV20(s,t) * share_h(s,t)          # ADV20 from the DENSE daily panel,
                                                        # 20 sessions strictly before t
    hrv_h(s,t)     = hour-h volume on session t / hourmean_h(s,t)

`share_h` is a SHAPE (a fraction of the symbol's own day) and `ADV20` is a LEVEL from the dense
daily panel. This split is deliberate and is declared here as the pass's main measurement risk: the
only intraday tape we own (`research/bf_zero/bars_sip.db`, 305,547 symbol-days) covers mostly
*candidate* days, so a level estimated from it would be biased high; a shape is far less exposed to
that bias. The **secondary** form `hrv_raw` (denominator = the symbol's own mean hour-h volume over
the same prior sessions, no ADV20 rescale) is computed and reported beside it as a sensitivity, and
if the two disagree in sign on any scored cell the cell is reported as NOISE.

| field | definition | decision time | 
|---|---|---|
| **V1** `hrv` | `hrv_h` for the hour just closed | end of hour h |
| **V2** `sus(k,N)` | `hrv >= k` for each of the last N closed hours (N = 2, 3) | end of hour h |
| **V3** `rvd_j` | session RV = daily volume ÷ trailing-20-session mean volume (strictly before that session), for each of the prior j = 1, 2, 3, 5 sessions; **`interest5` = the number of the last 5 sessions with `rvd >= 1.5` (0–5)** | prior session's close |
| **V4** `abs` | high RV (V1 or V2) **with** small price change over the same window: \|return\| <= 1 % (intraday hour) / <= 3 % (5-session). **Mirror control**: the same RV with \|return\| above 2 % / 6 % | same as its RV leg |
| **V5** `weak` | `rvd_1 >= 1.5` AND prior-session return < 0 AND prior-session close in the bottom third of its own high–low range | prior session's close |
| **V6** | (V2 sustained **or** `interest5 >= 3`) AND a HOD break today, i.e. the shipped detector on B2 | the break bar |

## 2. THE 23 SCORED CELLS (declared; nothing added or reweighted afterwards)

**Arm A — as a NAME-DAY SELECTOR on HOD-break's B2 booked population.** Each cell is the field
applied to the pre-book signal set at ranking time; reported as (i) kept-minus-rejected gross R with
n each side, day-clustered t, both TRAIN halves and VAL; and (ii) the re-booked book's owner metrics
(trades/wk, green weeks, weekly $ at $100 risk, worst week, red streak) with the count-matched
permutation null, and the downstream check of `hod_frames2` §2.3 (does the filter merely move the
book's clock / cohort?).

| # | cell |
|---|---|
| A1 | V1 `hrv >= 2` at the signal's hour |
| A2 | V1 `hrv >= 3` at the signal's hour |
| A3 | V2 `sus(2, 2)` |
| A4 | V2 `sus(2, 3)` |
| A5 | V3 `interest5 >= 3` |
| A6 | V3 `rvd_1 >= 1.5` |
| A7 | V4 absorption: `hrv >= 2` and \|hour return\| <= 1 % |
| A8 | V5 accumulation on weakness |
| A9 | V6 the owner's interaction: `sus(2,2)` OR `interest5 >= 3` |

**Arm B — as a STANDALONE DETECTOR on the PIT universe.** Universe: the dense point-in-time daily
panel (`research/multiday/data/prices_by_year`), test tickers and names absent from `daily_bars`
removed, price >= $5 (the standing cost rule), ADV$ >= $1M. Two entry legs, both declared:

* **multi-day cells (B1–B8)** — the field is known at the session close, so the entry leg is the
  **closing auction** of the decision session and the exit leg the closing auction of session t+h
  (`frames13` F42: an auction fill pays no quoted spread). Holds h = 1, 2, 5 sessions. Overnight
  financing charged at **7.0 % APR per night** (`frames13` F40's convention). Results in **% of
  price** first, R second (R = the 2 % stop the intraday cells use, for comparability only).
  Wrappers are reported as their own split and the primary number is **ex-wrapper** (the wrapper
  decay charge is measured, not assumed — §5).
* **intraday cells (B9–B14)** — entry at the **next bar's open under a cap** (the engine's only
  convention; a fill above the cap is a SKIP, never a touch fill), exits as declared per cell.

| # | cell | entry leg | exit |
|---|---|---|---|
| B1 | `interest5 >= 3` | auction | hold +1 |
| B2 | `interest5 >= 3` | auction | hold +2 |
| B3 | `interest5 >= 3` | auction | hold +5 |
| B4 | `interest5 >= 4` | auction | hold +5 |
| B5 | V3×V4 `interest5 >= 3` and \|5-session return\| <= 3 % | auction | hold +5 |
| B6 | V3×V4 as B5 | auction | hold +2 |
| B7 | V5 accumulation on weakness | auction | hold +2 |
| B8 | **control** `rvd_1 >= 3` single-session spike (already discovered) | auction | hold +1 |
| B9 | V1 `hrv >= 3` at an hour close | next bar open, cap +0.6 % | bare 2 % stop to 15:55 |
| B10 | V2 `sus(2,3)` | next bar open, cap +0.6 % | bare 2 % stop to 15:55 |
| B11 | V4 absorption `hrv >= 3`, \|hour return\| <= 1 % | next bar open, cap +0.6 % | bare 2 % stop to 15:55 |
| B12 | **mirror control** `hrv >= 3`, \|hour return\| > 2 % | next bar open, cap +0.6 % | bare 2 % stop to 15:55 |
| B13 | the best of B9–B12 **on TRAIN only** | next bar open, cap +0.6 % | 2 % stop **+ 2R target** |
| B14 | the same TRAIN pick as B13 | next bar open, cap +0.6 % | ORB's static lock (arm +1.75R → stop +0.5R) |

B13/B14 select their signal on TRAIN and are reported on VAL as a held-out reading; the selection is
declared here so it is one declared degree of freedom, not a search.

**The 7 declared diagnostics** (named now so they cannot become cells): D1 the universe bound (a
random universe name at the same clock/session), D2 the matched non-signal name, D3 the same
name-day at a later minute (causal — after the signal), for each of the two B families (6), and D4
the `hrv_raw` sensitivity on every A cell.

## 3. THE BAR (unchanged, pre-committed)

A cell CLEARS only if: **positive weekly $ AND green weeks >= 50 % on BOTH splits at >= 10 trades a
week, day-clustered t >= 2, and the two TRAIN halves same-signed.** A cell that clears goes to
**SHIP-TO-DRY** with its exact spec. Anything else is **STAY-DRY** and the MDE is reported so the
null is quantified rather than asserted. TEST is sealed (`FREEZE.md`) and is not opened by this pass
under any result.

## 4. RAILS

1. **Reproduction gate**, asserted in code before any cell is read: B2 TRAIN 1,622 trades / gross
   −0.039 / **−$17,346**; VAL 706 / +0.083 / **+$893**.
2. **Availability audit** on every field: coverage, and winner-vs-loser missingness (> 5 pp ⇒ VOID).
   The hourly profile uses **only the prior 20 available sessions, strictly before the decision
   hour**; a same-day hour never enters its own denominator.
3. **Day-clustered SE beside iid** on every claim; **% of price beside R** on every book.
4. **Count-matched permutation null** (2,000 draws, pick count held fixed) on every green-week claim.
5. **Ex-top-5 %** on every uncapped exit (B1–B8, B9–B12 have no target; B13/B14 are capped and the
   trim is reported as a diagnostic only).
6. **Wrapper enrichment check** on every field (pass 9's confound): the share of each cell's picks
   that are leveraged wrappers against the universe base rate.
7. **Base rate printed before any return**: the share of PIT universe symbol-days with
   `interest5 >= 3`, and the per-day share of names each field fires on.
8. Stores read-only; nothing written outside `frames15/`; no config, `orb.yaml`, order, service or
   cron touched; one python process, `nice -n 10`, `ulimit -v 3000000`; the hourly build checkpointed.
