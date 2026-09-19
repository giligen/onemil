# Red-to-green (F6-PDR) through the mature method — PRE-REGISTRATION

Candidate #3 of `research/mature_method/RUNBOOK.md`. **Committed before any cell was scored.**
Everything written lives under `research/mature_method/red_to_green/`. `data/cache.db`,
`research/bf_zero/bars_sip.db` and `data/trades.db` are opened **read-only**; no config, `orb.yaml`,
systemd unit, cron, order or cache is written. One `nice -n 10` python process at a time,
`ulimit -v 3000000`.

---

## 0. What is already settled and is NOT re-done

`research/fuckup_audit/H/F6_reconcile/REPORT.md` (three independent implementations A, B and a third
built from the parameterised `pipeline.py`) established:

* implementation A's TEST profit was **ZVZZT**, a NASDAQ test ticker: +50.80 R of a +39.00 R total;
* A's level was the prior close **exactly** (`BFZ_SLIP=0`), not the declared `prior close x 1.003`;
* the two studies took the **first level break and killed the day if that bar failed the 5 % floor**,
  while the shipped `trading/red_to_green.detect` **skips** floor-failing bars and keeps scanning;
* under the studies' rule the book is **+0.062 / +0.207 / −0.042 R** (TRAIN/VAL/TEST, hold, net);
  under the engine's rule **−0.027 / −0.012 / −0.102 R**.

Those are **net** numbers on the `cost_curve.csv` band table and they were never scored on green
weeks, never given a gate map, never given a measured cost and never given a frequency frontier.
This pre-registration does all six runbook things on the honest population.

## 1. The book, exactly

`trading/red_to_green.py` at the `config.yaml red_to_green:` block, run through
`HodBreakEngine(book='red_to_green')`:

| | |
|---|---|
| universe | `daily_bars` names with `adv20 >= 100,000` and last close `>= $5`, screened at 09:30 to `prior_day_range_pct >= pdr_min_pct` |
| precondition | the first RTH bar's open `o[0] < prior_close` (a gap-down / red open) |
| level | `prior_close x (1 + level_buffer)`, `level_buffer = 0.003` |
| floor | on bars **strictly before** the signal bar, `(run_high − run_low) / run_low >= 5 %` (the causal membership guarantee of the `>=5 %`-range universe file) |
| signal | a closed 1-min bar whose high reaches the level, signal minute `<= 840` (14:00 ET) |
| stop | the lowest low 09:30 **through** the signal bar |
| fill | the next printed bar's **open**, iff that open `<= level x 1.006` (no chase) |
| floors | `min_price` $5 applied by the engine to the **level**; `r_ok`: `(entry − stop)/entry >= 1 %` |
| exit | broker bracket: stop at the consolidation low, take-profit `entry + 2R`; flat 15:55 |
| book | `run_book(rows, max_per_day=12, max_concurrent=4)`, first-come |
| size | `risk_usd = 100`, `max_notional_usd = 10,500` |

`max_notional_usd` is **inert by construction**: with `min_r_pct = 1 %` the notional at $100 risk is
`10,000 / r_pct <= $10,000 < $10,500`. Stated here so it is not re-measured.

## 2. THE CRUX — the scan rule is a declared cell, not a choice

Both rules are run on the same population and reported **side by side at every step**:

* **S1 — first-break-then-floor.** The first bar whose high reaches the level is the only candidate;
  if that bar's pre-signal range is below the floor (or its running low is at/above the level) the
  symbol-day is dead. This is what implementations A and B measured.
* **S2 — floor-and-break, keep scanning.** The first bar at which the floor holds **and** the level
  breaks **and** the running low is below the level. Floor-failing bars are skipped. This is what
  `trading/red_to_green.detect` implements and therefore what the live engine does today.

If S1 is the survivor the engine change is **one line** in `detect`: the `continue` on a
floor-failing bar becomes `return None`. That is named in the report whatever the answer.

Both rules are run with the **declared** level (`x 1.003`, not A's `x 1.000`) and both get the
engine's cut (signal minute `<= 840`), so S1 vs S2 differs in the scan and nothing else.

## 3. Population, membership, splits

Population = `H/F6_reconcile/prev_table.csv` (every `research/bf_zero/universe.csv` symbol-day
2025-01-03 → 2026-09-04 with a prior row in the Databento EQUS daily panel), minus:

* NASDAQ test tickers `^Z[A-Z]ZZT` (`research/scripts/pit_listings.is_test_ticker`) — 390 rows;
* every symbol absent from `cache.db daily_bars` — 27,307 rows, 782 symbols.

→ **616,883 symbol-days, 7,583 symbols** (`pop.csv`). `adv20` and `prev_close` are built **causally**
from the panel (trailing 20 sessions strictly before the day).

Splits, as the book already uses them:

| split | window | note |
|---|---|---|
| TRAIN | 2025-01-03 → 2025-12-31 | |
| VAL | 2026-01-01 → 2026-05-31 | |
| **TEST** | 2026-06-01 → 2026-09-04 | **SEALED** — see `FREEZE.md` |

TEST is opened only for cells that pass **both** claim gates on TRAIN and VAL (section 6), and for
nothing else. The reconciliation's already-published TEST figures may be quoted as prior art; no new
TEST cell is scored without passing the gates first.

## 4. Metric order (the owner's, per the runbook)

1. **% green weeks** over every W-FRI week in the split. A week with no trade counts **FLAT** and
   stays in the denominator.
2. longest red streak · worst week · % green months · max drawdown.
3. total P&L (TERTIARY), always printed in dollars at the live `risk_usd = 100`.

Ex-top-1 % / ex-top-5 % and a +3R winner cap are reported as diagnostics, never as rejection reasons.

## 5. Cost and fill

* **Cost — measured, never the band.** Alpaca **SIP NBBO** at the **signal minute** (the decision
  instant) on a stratified random sample of the book's own signals, TRAIN and VAL only (TEST legs
  are not sampled). A measured half-spread curve keyed (price band x hour band) replaces
  `research/lit_review_2026/cost_curve.csv`. Leg weights are the reconciliation's contract (c) so the
  numbers stay comparable: entry `0.25 x half`, stop `0.875 x half`, eod `0.412 x half`, target
  `0.875 x half`, all as a fraction of R. A conservative variant (full half-spread on both legs) is
  reported beside it. The band table's own charge is reported for the same trades so the direction
  and size of its error is stated. **Expectation to be tested, not assumed:** these are PDR >= 8 %
  small caps that gap down and then run — halt-resume prone — so the band is expected to be **too
  NARROW** here (the opposite direction to HOD-break's $20+ names).
* **Fill — the engine's.** A capped limit at `level x 1.006` filling at the next printed bar's open
  iff that open is at or under the cap. A capped-out signal is **not a trade** (no order fills, no
  P&L, and the reconciliation's convention: the symbol-day produces nothing). The **unfilled
  counterfactual** — the same signals filled at the open anyway — is scored as its own population and
  classified `chase-guard` (fills better than non-fills) vs `dip-buy` (fills worse).
* **Obtainability.** Every modelled fill is a bar's OPEN, hence a price the tape printed. The
  fraction of fills whose next printed bar is **not** the next clock minute (the engine's order lives
  ~20 s) is reported as a sensitivity, both scan rules.

**Declared deviations from live, stated now:**
(a) live builds `adv20`/last-close from the LAST row of `daily_bars` (a static snapshot); this study
builds them causally as of the day. (b) live applies `max_spread_bps: 300` against a real-time quote;
this study has NBBO on a sample only and reports the share of sampled signals above 300 bps rather
than gating on it. (c) live's take-profit is a resting broker limit that fills on a **touch**; the
simulator fills the target on a bar **close** (conservative). (d) the universe file is the
`>= 5 %` daily-range screen; the causal 5 % pre-signal floor is what makes it superset-exact, so the
floor ladder runs **upward only** and no cell drops it.

## 6. Bars

* **Claim bar.** G1: TRAIN mean net R > 0 with `t >= 2` and `>= 5` trades/week. G2: VAL same sign
  **and** `>= 55 %` green weeks. TEST is opened only for a cell passing both.
* **Live-exploration bar.** A positive point estimate on green weeks **and** on dollars at
  `risk_usd = 100` in **both** TRAIN and VAL; a stated mechanism; bounded downside with a
  pre-committed stop; resolution inside one quarter at the book's own frequency.

## 7. The declared cells (scored for **both** scan rules S1 and S2, primary exit = the engine's 2R bracket)

`B0` = the live config block above.

| id | cell |
|---|---|
| **B0** | shipped: pdr >= 8, floor 5 %, level x1.003, cap 60 bps, level >= $5, r >= 1 %, 14:00, adv20 >= 100K, prev close >= $5, 12/4 |
| P1 / P3 / P4 | PDR ladder: `pdr_min_pct` = 6 / 10 / 12 (B0 = 8) |
| F2 / F3 | floor ladder **upward**: `range_floor_pct` = 8 / 10 |
| X2 / X3 | price floor on the level: $10 / $20 |
| R2 / R3 | `min_r_pct` = 2 % / 3 % |
| T1 / T2 / T4 | last entry minute 660 (11:00) / 780 (13:00) / 930 (15:30) |
| A2 / A3 | `min_adv20` = 500,000 / **off** |
| L2 / L3 | level buffer `x1.000` (what A ran) / `x1.006` |
| K2 | slots 20/day, 8 concurrent |
| V1 | + `rv_profile` in [1, 5) at the signal bar (HOD-break's band; the r2g spec computes rv and never gates on it) |
| G1c | + obtainable only: the next printed bar is the next **clock** minute |
| C1 | pdr >= 10 **and** level >= $10 |
| C2 | pdr >= 12 **and** r >= 2 % |
| C3 | floor >= 8 % **and** last entry 13:00 |
| C6 | B0 + V1 (rv band) + G1c (obtainable) |
| **C4** | **structural ceiling A**: pdr off, adv20 off, last entry 15:30, slots 20/8, r >= 1 %, level >= $5 kept |
| **C5** | **structural ceiling B**: C4 with the price floor down to $1 |

**22 cells x 2 scan rules x 2 open splits = 88 declared cell-scores.** Exits `hold` and `partial`
are reported for `B0` only, as secondary shape evidence. Every additional table in the report
(gate map, cost, fill, null) is descriptive, and the count of everything looked at is restated in
the report's multiplicity paragraph.

## 8. Gate-separation map (runbook step 5)

Each gate is measured at its **cascade position**: the population is the whole live stack **minus
that gate**, and the gate partitions it into kept and rejected. Reported per gate: gross delta, net
delta, `t`, `n` each side, TRAIN and VAL separately, and **median position notional** kept vs
rejected (`risk_usd = 100`, `shares = 100 / R`), because a sizer can hide a wrong-side gate. A gate
wrong-side or insignificant in **both** years is named as a removal candidate.

## 9. Null

Count-matched permutation: each cell's own trade P&L is shuffled across its own weeks with the
per-week pick count held fixed, 2,000 draws, seed 20260919. A green-week share inside the resulting
[p5, p95] band is **pick count, not week-level skill**, and is said so.

## 10. Verdicts available

**SHIP-TO-DRY** (`red_to_green.enabled: true, dry_run: true` — a config flip the **owner** makes,
not this study) · **STAY DEAD** · **NOT DECIDABLE** with the thing that would decide it named.
