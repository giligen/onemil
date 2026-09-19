# green_weeks — pre-registration

Written 2026-09-19, **before any cell was scored**. Owner's ask, same day:

> "Do whatever it takes to lift frequency! And then to get those green weeks in."

Frequency is settled elsewhere (`research/bf_frequency/REPORT.md` — F7, 4.2x). This
stage tests the one lever nobody has tested: **exit design**, ranked on **week shape**
instead of total P&L.

Nothing ships from this stage. A survivor needs its own pre-registration and the
owner's word.

---

## 1. The thesis, stated so it can fail

Week shape is set by the DISTRIBUTION of trade outcomes, not by selection. A book of
many small wins and rare larger losses produces green weeks; a book of rare large wins
and frequent small losses produces flat/red weeks punctuated by monsters. Every book we
run is built the second way, and every exit study we have run was ranked on TOTAL P&L.

**If the thesis holds**, earlier/smaller profit-taking converts flat and mildly-red
weeks into small green weeks, and the green-week share rises materially (≥ the MDE of
§7) on TRAIN *and* VAL for at least one book.

**If it fails**, earlier taking converts monsters into breakeven and leaves the red
weeks red: green-week share moves inside the noise band while total P&L falls. That is
a legitimate, reportable outcome and the recommendation is then "shipped exit stands".

## 2. The metric — declared, in order (owner 2026-09-19)

1. **PRIMARY: % of GREEN WEEKS.** Denominator = every market week in the split,
   whether or not the book traded. A week with no trade is **FLAT**, not green.
   Week = `W-FRI` period (Sat→Fri, the convention already used by `bf_frequency` and
   `bf_zero`). Green = week P&L (or week R) **> 0**.
2. Longest **red-week streak** (consecutive weeks < 0).
3. **Worst week**.
4. **% green months** (calendar months carrying ≥1 trade in the denominator).
5. **MDD** on the daily cumulative curve.
6. **FLAT-week share** — reported beside green, because converting a flat week to a
   small green week is exactly the win condition.
7. **TERTIARY: total P&L.** Reported always, **never ranked on**.

**Tail concentration is NOT penalised.** Owner, explicitly: *"Monster is ok if green
weeks dominate even if near breakeven."* The P&L share of the top 1 / 5 / 10 trades is
reported as a **diagnostic only** and never enters a ranking or a gate.

**Ranking rule, pre-committed.** Cells are ordered by `min(green% TRAIN, green% VAL)`
— a cell must be good on both, not on one. Ties broken by (a) lower longest red-week
streak on the worse split, then (b) higher flat→green conversion (flat-week share
reduction vs E0), then (c) total P&L.

## 3. Books, populations, selection held FIXED

Selection is **frozen per book** and is never a variable in this stage. No threshold,
z-param, quintile cutoff, veto level, gate or multiplier is re-fitted anywhere.

| book | population (honest reference) | selection frozen at | slots |
|---|---|---|---|
| **ORB** | `analysis_results/orb_features_20260916_2053.csv` — entered-inclusive, 13,033 candidates / 427 sessions, via Stage M's dumps | shipped B+ stack read from `orb.yaml` (composite, Q1 filter, dedup, Q4-preferred ranking, 4 post-ranking no-refill vetoes), `ORB_BT_RISK=375`, per-position cap $3,333.33 | N = 8 (live). N = 3 / 12 reported as flanks for E0 and the recommended cell only |
| **HOD** | `research/bf_zero/spec_trades.csv` — the SIP-tape honest population (60,461 signals, §6a/§6b), i.e. the population on which the edge was REFUTED | the live config: price ≥ $20, last entry 14:00, first-come 12/day, 4 concurrent, `trading/hod_break.run_book` | 4 concurrent |
| **BF** | `data/bull_flag_cache_causal_full_20260905.csv` (regen-7, 896 rows) | **F7** = shipped P1 minus the ADV20 ≥ 200K gate minus the conviction ≥ 1.8 gate (`bf_frequency` §11). Everything else at P1: price ≤ $20, pole ≥ 5%, VWAP gate, two-tier, regime sizing off, `max_positions 3`, `max_trades_per_day 5`, rail −5u | 3 |

**Splits, identical across the three books** (each book's own already-used split, which
happen to align):

* **TRAIN = 2025-01-01 → 2025-12-31**
* **VAL = 2026-01-01 → 2026-05-31**
* **TEST = 2026-06-01 → end of that book's honest data** (BF 2026-08-31; ORB 2026-09-16;
  HOD 2026-09-11). **SEALED** — see `FREEZE.md`.

## 4. The cells — declared in full, counted before running

`R` per book: **ORB** `range_high − range_low` (the R the shipped lock and Rule D use);
**HOD** `entry − stop`; **BF** plan-R, `planned_entry − stop_loss`
(`trading.trailing_stop.r_basis: plan`). The level baseline is the book's entry
(`entry_price` / `entry` / `r_baseline` from `trading/bf_trail.r_baseline_and_unit`).

| id | shape | ORB | HOD | BF |
|---|---|---|---|---|
| **E0** | the shipped exit — the reference | static lock 1.75R→+0.5R, touchgo M/D, ATR floor, 40%@+3R scale-out, 15:45 flat | +2R resting target (close-through), stop = consolidation low, 15:55 flat | 50% at +2R → stop to the fill (breakeven), remainder on the unified trail + exhaustion + vol-guard |
| **E1** | fixed target, WHOLE position, at **+0.5R / +0.75R / +1.0R / +1.5R** (4 cells) | replaces the lock; stop + touchgo + flat unchanged | replaces the +2R target | see §5 (fraction 0.999) |
| **E2** | **50% at +0.5R / +1.0R / +1.5R**, stop → breakeven, remainder on the shipped exit (3 cells) | remainder keeps lock + scale-out | remainder keeps the +2R target | the shipped partial knob at a lower `r_multiple` |
| **E3** | 50% at **+1.0R**, remainder **trailed** instead of held to the shipped exit (1 cell) | trail 0.5R under the running closed-bar high, replacing the lock | trail 0.5R under the running closed-bar high, replacing the target | **collapses to E2@+1.0R** — the BF shipped remainder IS a trail. Declared, not double-counted |
| **E4** | time-boxed: at the first bar at/after entry + N minutes, if `(bar open − entry)/R < +0.5`, exit at that open (2 cells) | N = 10 / 20 | N = 15 / 30 | N = 20 / 40 |
| **E5** | breakeven-stop-only: stop → entry once a closed bar's high reaches entry + 0.75R; **no target**; otherwise the shipped exit (1 cell) | lock kept | the +2R target REMOVED | partial off |

**Declared cell count.**

* ORB: 12 new (E0 + 4 + 3 + 1 + 2 + 1) **+ 5 already-existing Stage-M challengers
  (X1…X5) re-ranked on week shape, not re-run** = **17** scored at N = 8.
* HOD: **12**.
* BF: **11** (E3 ≡ E2@+1.0R).
* **TOTAL = 40 cells** on the primary metric, plus **4 robustness reads** (ORB E0 and
  the ORB recommendation at N = 3 and N = 12 — no new fitting).

Nothing outside this list is scored. If a cell cannot be simulated faithfully it is
reported as **NOT SIMULABLE** with the reason — never guessed.

## 5. Fills, costs and obtainability — binding

* **Profit-take fill convention (binding, owner 2026-09-19).** A target or partial
  level fires only on a bar that **CLOSES at or above the level**, and fills **at the
  level**. **No touch fills, no wick fills.** A touch-filled target is the single most
  common way an exit study lies and it inflates exactly the small-target cells this
  study is built to test. (The first draft of this file said "high ≥ level, fill at
  `min(close, level)`"; that was corrected to the close-through rule below **before any
  cell was scored** — it is both stricter on the trigger and free of the artefact where
  a spike-and-collapse bar fabricates an exit at a bad close.)
  * This IS `trading/hod_break.walk_exit`'s shipped rule, so HOD's E0 is unchanged and
    every HOD cell inherits it.
  * ORB's new target/partial cells use the same rule. The shipped 10 bps ORB exit slip
    is applied to the profit leg like every other exit — conservative for a resting
    limit, and identical across cells.
  * **BF is the one declared deviation**: the shipped partial
    (`trading/bf_profit_partial.py`) triggers on the closed-bar high and fills at that
    **bar's close**, uncapped by the level — obtainable (a market sell after the bar
    closes) but better than the level when `close > level`. E0 and the E2 ladder use
    the SHIPPED spec, because changing it would make E0 not-the-shipped-exit. The share
    of partial fills with `close > level`, and the P&L that share carries, is reported.
* **Stop fills**: as shipped per book. ORB — stop level less 10 bps (`EXIT_SLIP_BPS`);
  a gap-through fills at the level (the shipped simulator's inherited deviation, equal
  in every cell). HOD — `min(stop, open) × (1 − 10 bps)`. BF — the shipped stop-fill
  model. Not re-modelled here.
* **Time-box fills** (E4): the bar's **OPEN**, obtainable by a market order sent at the
  bar boundary — Stage M's declared convention, and ORB's E4 is therefore comparable to
  Stage M's X1/X2/X4.
* **Costs**: as shipped and as measured, per book. ORB = Stage M's arm (10 bps exit
  slip, entry at the recorded fill). HOD = the §8 cost model on the SIP tape, applied
  identically to every cell. BF = the regen-7 / resim slippage model. **Costs do not
  vary across cells within a book**, so they cancel in the E0-vs-En diff; the absolute
  level is a relative tool, never a forecast.
* **BF's exit-machinery caveat (BT_STATUS §2a).** Regen-7's OWN exits are the faithful
  reference; the resim reproduces them only to **−$10.3K / −7.4%** (55 of 56 rows
  differ; the exhaustion rule fails to re-fire and `post_fill_exit` is unmodelled).
  Every BF variant requires the resim. Therefore: **all BF cells including E0 are run
  through the resim**, so the comparison is resim-vs-resim and the drift is common-mode;
  the cache's own-exit number is reported alongside as a level anchor ONLY. Any BF cell
  that cannot be expressed through the shipped simulator's own levers is reported NOT
  SIMULABLE. BF **E1** is expressed as the shipped partial at `fraction = 0.999`
  (integer share rounding leaves ≤ 1 share riding the shipped exit); the residual share
  count and its P&L are reported, and if the residual exceeds 1% of any position's P&L
  the cell is downgraded to NOT SIMULABLE.
* **Availability audit** on every field used by a decision: coverage %, and whether
  missingness depends on the outcome. Reported per book before any cell table.

## 6. Verification gates — run before any cell is read

1. **E0 parity.** Each book's E0, computed by this stage's walker, must reproduce the
   book's existing published reference trade-for-trade:
   * ORB — Stage M's `dump_X0.csv` / `book_X0_n8.csv` to `max |Δpnl| < 1e-9`.
   * HOD — `spec_trades.csv` `rr` on every signal to `< 1e-9`, and the live-config book
     to §6a's `24/53 · 9/23 · 9/15` green weeks.
   * BF — the resim E0 against BT_STATUS run B (56-trade P1 selection) as a shape check,
     and the F7 selection against `bf_frequency` grid row F7.
   A failure aborts the stage; it is not "explained".
2. **Week-grid integrity.** The week denominator per split is the count of `W-FRI`
   periods in the split range, generated independently of the trades. Printed and
   eyeballed once: TRAIN 53, VAL 22, TEST per book.
3. **Monotonicity read.** The E1 and E2 ladders are ordered by R; a metric that is
   non-monotone across the ladder is reported as such and its best point is treated as
   noise, not as a discovery.

## 7. Power — stated BEFORE the numbers

Green-week share is a proportion over a small number of weeks. Unpaired 95% half-width
at p ≈ 0.5: **TRAIN (53 weeks) ±13.5pp · VAL (22) ±20.9pp · TEST (14–15) ±25.3pp.**

Cells are **paired** (identical weeks, identical picks, only the exit moves), so the
real test is on **discordant weeks** (green in one cell, not the other). The count of
discordant weeks is reported with every headline comparison; a difference carried by
fewer than **8 discordant weeks** on TRAIN, or fewer than **5** on VAL, is reported as
**not resolved** regardless of its size.

**Which books can resolve anything at all** (pre-committed, from trade frequency):

* **HOD — yes.** ~44 signals/week, ~2,300 book trades on TRAIN: almost every week
  trades, so flat weeks are rare and the green/red split is genuinely measurable.
* **ORB — marginal.** 215 picks / 162 fills over 74 weeks at N = 8 ≈ 2.2 fills/week;
  Stage M already measured 44.6% green weeks with a large flat share. Differences under
  ~14pp on TRAIN will not resolve.
* **BF — weakest.** F7 is 12.4 trades/month ≈ 2.9/week on TRAIN (2.4 on VAL). At P1 it
  was 2.8/month and 56.6% of weeks were FLAT. Even at F7 this book resolves only very
  large shifts. This is stated up front so no BF cell is over-read.

## 8. Multiplicity

40 cells + 4 robustness reads, on 3 books × 2 open splits = **80 primary readings**
before TEST. At α = 0.05 roughly 4 would clear by chance. Therefore:

* no single best point is a discovery — the **rank distribution across the ladder** and
  **direction consistency between TRAIN and VAL** are what count;
* a recommendation requires the cell to be top-3 on green weeks on **both** TRAIN and
  VAL, with the discordant-week count of §7 met;
* if no cell in a book meets that, the finding for that book is **"no variant beats
  shipped on week shape"**, which is a legitimate outcome and will be written as such.

The program-level cell count is carried forward too: this stage's 40 sit on top of the
cells already counted in `research/fuckup_audit/` (Stage M's 18 among them, 5 of which
are re-used here rather than re-run).

## 9. TEST

Sealed per `FREEZE.md`. Opened **once**, **after** the recommendation is committed, and
only for **E0 and the single recommended cell per book**. If TEST disagrees, the
disagreement is reported, not engineered away.

## 10. Resources

One python process at a time, `nice -n 10`, `ulimit -v 3000000`, foreground, columnar
reads. `cache.db` / `trades.db` / `bars_sip.db` opened **read-only**. `config.yaml`,
`orb.yaml`, production caches, orders, the service and the crons are never written —
BF cells drive the resim through scratch copies of `config.yaml` under
`research/green_weeks/`.
