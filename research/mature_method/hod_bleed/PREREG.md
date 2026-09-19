# hod_bleed — PRE-REGISTRATION

Written and committed **before any Part 1 table and before any Part 2 cell was scored.** Owner brief
2026-09-19:

> *"The losses live in the first ten minutes. A losing trade runs +0.45R by minute 6, then bleeds to a
> full stop at minute 28. If we got to +0.4 we can have a full stop/exit at −0.1 or whatever. Look at
> those losers and find the rule. I'm sure one can also see it on the volume, or MACD, or VWAP —
> seeing the sentiment is likely to turn south. Maybe we will lose some of the winners, but will cut
> out the losers."*

## 0. Why this is not a repeat of `hod_losers` P8

`hod_losers` scored **breakeven at +0.5 R** (`P8`) and it was the worst cell in that pass
(VAL −$12,317 against the shipped −$4,128). The anatomy in the same report says why the level was
wrong: the **median bleed-loser's MFE is +0.45 R**, so a +0.5 R arm never fires on the trades it is
meant to catch, while every eventual winner passes +0.5 R on its way to +2 R and is then scratched on
its first retrace. The arm level was above the loser distribution and inside the winner path.

There are **two loser classes** and they need separate treatment:

1. **straight-down losers** — never reach +0.25 R, at −0.50 R by minute 10. `hod_losers` P1/P3 already
   measured these: a time stop saves **+0.012 / +0.016 R**. Already-dead money. **Not re-run here.**
2. **bleed losers** — reach ≈ +0.45 R at minute 6, roll over, stopped at minute 28. These are the
   owner's target and the largest separation in the anatomy. **This pass is about class 2 only.**

The unmeasured quantity, and the whole reason for the pass: **after a trade has reached a, what
fraction of eventual WINNERS retrace to s before reaching +2 R?** If winners go +0.4 → +0.8 → +2.0
without revisiting entry, a tight stop after +0.4 R is free. If they retrace through it 40 % of the
time, it kills the book. Nobody has measured it. Part 1 measures it; Part 2 scores it.

## 1. Population, book, cost — unchanged from the four prior HOD passes

`hod_filter_stack/pop.csv` → `score2.sig_set` at the declared base knobs; `trading.hod_break.run_book`
at 12/day, 4 concurrent; `$100` risk; the measured per-trade SIP-NBBO cost model of
`hod_break/score.py::attach_cost` with the `score4` exit-side ratio contract
`{stop 0.875, eod 0.412, target 0.0}`. Membership cuts (early closes, test tickers, non-`daily_bars`
names) as before. Splits: **TRAIN 2025** (halves H1 = 2025-01-01…06-30, H2 = 2025-07-01…12-31),
**VAL 2026-01→05**. **TEST 2026-06-01→09-11 is SEALED** behind `FREEZE.md` and is not opened unless a
cell passes the ship bar.

**Bases**: `B0` = the shipped population (what is in `config.yaml` today) — the primary base for every
cell. `B2` = `hod_filter_stack` §2's corrected base, reported for the selected cells only.

**Cost of a NEW exit, declared here so it is not chosen after seeing the numbers**: every exit this
pass introduces (a ratchet stop, a peak-retrace exit, a signal-triggered exit) is charged the
**marketable** exit ratio `0.875` — the same as a stop, the conservative choice.

**The cost line, corrected (`hod_fresh` REPORT §2, landed 2026-09-19).** The **+0.2151 R** every prior
HOD report quoted as the break-even is the cost of the **un-gated** population. A **BOOKED** trade in
the shipped 12/4 book pays **0.061 R (TRAIN) / 0.065 R (VAL)** — the 100 bps and 15 %-of-R spread
gates remove the expensive names before the book sees them (B0's own gross −0.027 vs net −0.088 =
0.061). Every prior pass held this book to a bar **3.2× its real break-even**. This pass therefore:

1. **re-measures `cost/R = mean(rr − net)` on the BOOKED set of every cell it reports** and never
   carries a constant — an exit rule changes the `why`-mix, so it changes the cost term;
2. states the arithmetic explicitly: B0 shipped is gross **−0.027** / net **−0.088** (TRAIN). An exit
   rule that adds **+0.10 R net/trade** puts the book at roughly **+0.01 R net** — **at** break-even,
   **not through it**. The claim that clearing the ship bar "lifts the book across the cost line" is
   therefore **wrong at +0.2151 R and barely true at 0.065 R**; the honest statement is that a
   +0.10 R exit takes the book from −0.088 to about zero;
3. reports, for the best cell, whether it takes the shipped book **net-positive on BOTH TRAIN halves
   and VAL with a day-clustered t**.

**A cell's headline number is the DIFFERENCE from the shipped exit on the same trades**, which is
first-order invariant to the cost level.

## 2. Obtainability — the fill convention for every exit in this pass

| exit kind | trigger | fill |
|---|---|---|
| the shipped stop | a bar's low ≤ `stop` | `min(stop, o[k]) × (1 − 0.001)` — the walk-parity convention of `hod_losers/walk2.py` |
| **ratchet stop** (E1, E2) | a bar's low ≤ the ratcheted stop | same: `min(stop, o[k]) × (1 − 0.001)`. A ratchet is a resting stop leg; the market comes to it |
| **peak-retrace** (E3) | a CLOSED bar's close ≤ `mfe_running − d` | **the NEXT bar's open**, never the trigger level |
| **signal exits** (E4: volume / VWAP / MACD) | a CLOSED bar satisfies the rule | **the NEXT bar's open**, never the trigger level |
| the +2 R target | a bar's CLOSE ≥ `E + 2R` | `E + 2R` (the shipped convention — a resting limit) |
| 15:55 flat | `m ≥ 955` | that bar's open |

**Arming is on CLOSED bars only** and takes effect on the FOLLOWING bar: inside the walk the low is
checked against the *current* stop **before** the bar's own high updates the running max. A bar that
both reaches `a` and falls to `s` therefore exits at `s` only on a LATER bar. This is the causal
choice and it is the same structure `hod_losers/walk2.py::walk` used for its breakeven arm.

The fill bar `e` (the bar whose open we buy) is never checked for an exit — the shipped spec starts at
`e + 1` — but its high DOES seed the running max, exactly as in `walk2`.

## 3. PART 1 — the path map (DESCRIPTIVE, no cell, no selection)

From a single re-walk of every `B0 ∪ B2` TRAIN+VAL signal, emitting the whole post-fill bar path in R
units plus per-bar volume, session VWAP and 1-min MACD (`pathbars.parquet`). Reported:

* for eventual winners / losers / eod-exits, per split: the distribution of **current R** (the bar's
  close, not the MFE) at minutes 1, 3, 5, 10, 15; the MFE ladder by minute; the retrace from MFE by
  minute.
* **the key table**: for each arm level `a ∈ {0.3, 0.4, 0.5, 0.6} R` — of the trades that reach `a`,
  what share are eventual winners vs losers; and of the WINNERS that reach `a`, what share
  subsequently retrace to each stop level `s ∈ {−0.3, −0.2, −0.1, 0, +0.1, +0.2} R` **before** the
  trade reaches +2 R. Reported in full, `4 × 6`, both splits.

Part 1 is descriptive. It is reported whatever it says, and nothing in it is a claim.

## 4. PART 2 — the declared cells, **cap 30**

Every cell is an EXIT rule. Entries are untouched, so the signal stream is identical; what changes is
when a trade releases its slot, which changes the booked set. **Both measurements are reported for
every cell:**

* **FIXED-COHORT** (the primary trade-off measure): the new exit applied to **exactly the trades the
  shipped book took**. Same n, same days, same weeks. This is the clean "winners lost vs losers cut"
  and it is where the ship bar is evaluated.
* **RE-BOOKED** (the honest book): `run_book(12, 4)` re-run on the new exit minutes, so earlier exits
  free slots earlier and the trade count moves. Reported beside it, because `hod_losers` §1 showed the
  added trades can be negative.

### E1 — the ratchet (≤ 12 cells)

*Arm at `a`; the stop moves once to `E + s·R`; it never loosens; the +2 R target and the 15:55 flat are
unchanged.* The candidate grid is `a ∈ {0.3, 0.4, 0.5, 0.6}` × `s ∈ {−0.3, −0.2, −0.1, 0, +0.1, +0.2}`
= 24 combinations. **The 24 are computed in the Part-1 walk as a descriptive matrix. The DECISION
cells are chosen from them by this rule, fixed now:**

> **E1 selection rule.** Compute, from Part 1's matrix on **TRAIN only**, the expected trade-off in R
> per booked trade: `(losers cut × R saved) − (winners lost × R given up)`. Rank the 24 by that
> quantity. The **top 12** are the declared E1 decision cells. If fewer than 12 are positive on TRAIN,
> only the positive ones are declared and the cell count drops accordingly. No other criterion may
> promote an (a, s) pair into E1.

### E2 — the time-conditioned ratchet (4 cells)

The selected-best E1 `(a*, s*)`, and the second-best `(a', s')`, each with the arm additionally
required to happen **within the first N minutes of the trade** (`N ∈ {6, 10}`). If `a` is reached
later than N, the trade runs the shipped exit. Mechanism: the anatomy says losers peak at minute 6, so
a late arm is a different animal.

### E3 — retrace-from-peak (4 cells)

*Armed once the running max (closed bars) reaches `a*`; then exit at the next bar's open when a closed
bar's close ≤ `mfe_running − d`.* `d ∈ {0.3, 0.4, 0.5} R` at `a = a*`, plus `d = 0.4` at `a = 0.3`.

### E4 — the owner's signals, as exits on an ARMED trade (6 cells)

Armed at `a*` (the best E1 arm). Each is one declared, causal rule read off closed bars ≤ the decision
bar; the exit fills at the next bar's open.

| # | rule |
|---|---|
| E4a | **volume — distribution selling**: a post-entry bar with `vol ≥ 2 × the BREAKOUT bar's vol` whose close is below the prior bar's close |
| E4b | **volume — momentum gone**: three consecutive post-entry bars with `vol < 0.5 × the breakout bar's vol` |
| E4c | **VWAP**: a post-entry bar closes below the session VWAP (cumulative `(h+l+c)/3 × v` from 09:30). The break is above VWAP by construction, so losing it is the owner's "sentiment turns south" |
| E4d | **MACD histogram**: the 1-min MACD histogram (12/26/9 EMAs on 1-min closes from 09:30, `adjust=False`) is negative at a post-entry bar's close |
| E4e | **MACD signal cross**: `macd` crosses below `signal` at a post-entry bar's close |
| E4f | **any two of three agree**: at a post-entry bar's close, ≥ 2 of {E4a fired at or before this bar, E4c fired, E4d fired} |

### E5 — the combination the owner is describing (2 cells)

`E1(a*, s*)` + the best E4 trigger, on `B0`; and the same on `B2`. Exit = whichever fires first.

### E6 — the exit on the only net-positive admission in 799 cells (1 cell, run LAST)

`hod_fresh`'s `C1` = **`consol_bars ≥ 20` admission × last-5-bar stop × `spy_r5_pct > 0`** (SPY's
09:30 open → 09:34 close, known 09:35:00) is the first cell in 799 that is net-positive on H1-2025,
H2-2025 and VAL (+0.033 / +0.050 net, 13.8 / 16.0 trades a week) — and it fails its own bars
(clustered t +0.54 / +0.66, one week = 85 % of the TRAIN year, ex-top-5 % negative).

**E6 = the single best exit cell of E1–E5, applied to C1's population.** Declared here as ONE extra
cell; the grid is **not** expanded otherwise, and E6 is scored only after every declared cell above
has been scored. Mechanism: the only net-positive admission crossed with the only exit that can cut
the bleed is the single most informative combination left on this book.

### Reference rows (not cells)

`B0 shipped` and `B2 shipped` reproduce `hod_break` §6 / `hod_filter_stack` §2 / `hod_losers` to the
dollar. The walk's `base` variant vs `pop.csv` is the parity gate (`why` match, exit-minute match,
`max|Δrr|`).

**TOTAL DECLARED DECISION CELLS: ≤ 12 (E1) + 4 (E2) + 4 (E3) + 6 (E4) + 2 (E5) + 1 (E6) = ≤ 29**,
under the cap of 30. Programme cumulative: **799** (through `hod_fresh`, its own PREREG §2)
**+ ≤ 29 = ≤ 828.**
Part 1's descriptive tables — including the full 4 × 6 (a, s) matrix — are **not** cells and are not
counted; they are reported in full so the reader sees everything that was looked at.

## 5. Scoring, for every cell

* PRIMARY (the programme's metric): **% green weeks** over every market week of the split (no-trade
  weeks count as flat and are in the denominator), then longest red streak, worst week, weekly $ at
  $100 risk, trades/week.
* THE OWNER'S QUESTION: **net R per trade vs the shipped exit, at unchanged frequency** — the
  fixed-cohort measure. Reported per cell as: **winners lost (n, R given up)**, **losers cut
  (n, R saved)**, **net ΔR/trade**.
* `t` on the paired per-trade difference (new − shipped on the same trade), **iid** and
  **day-clustered** (`se = sqrt(Σ_d (Σ_{i∈d} (x_i − x̄))²) / n`) side by side.
* **Both halves of TRAIN** (H1, H2) and VAL, every cell.
* **Count-matched permutation null**: shuffle the cell's own per-trade P&L across its own weeks
  (2,000 draws, per-week pick count held fixed) — green weeks inside that band are pick COUNT, not
  skill.
* Ex-top-1 % / ex-top-5 % reported as a diagnostic, never a rejection reason.

## 6. The bars — fixed now

**SHIP BAR (the owner's, for SHIP-TO-DRY).** All four, on the fixed-cohort measure:

1. net R per trade improves by **≥ +0.10 R** on TRAIN, with **both halves the same sign**;
2. net R per trade improves on **VAL**;
3. the paired **day-clustered t ≥ 2.0** on TRAIN;
4. **green weeks not worse** than the shipped book on both splits.

**Also reported** (the brief's second question, answered against the CORRECTED line): the cell's own
re-measured booked `cost/R`, its gross, and whether the shipped book's **net** goes positive. The
arithmetic is stated in §1: B0 is net −0.088 (TRAIN) / −0.050 (VAL), so a cell that just clears the
+0.10 R ship bar lands the book **at** break-even, not through it. "Crossing the +0.2151 R line" is
not a meaningful statement about this book and is not made.

**STAY-DRY** is the verdict if no cell clears the ship bar, and it is delivered as the (a, s) matrix
that shows why (winners retrace too often), with the **MDE** — the smallest per-trade effect the pass
had 80 % power to see.

**TEST stays sealed** unless a cell clears the ship bar on TRAIN **and** VAL.

## 7. Causality trace — every field every cell reads

| field | constructed from | known at |
|---|---|---|
| running max high (E1, E2, E3 arm) | closed bar highs from the fill bar `e` to bar `k` | the close of bar `k`; the stop it sets is effective from bar `k+1` |
| `arm_m` (E2) | the first closed bar whose high reaches `E + a·R` | that bar's close |
| `mfe_running` (E3) | same running max | the close of the bar the rule acts on; the exit fills at the NEXT bar's open |
| breakout-bar volume (E4a, E4b) | the break bar `i`, one bar BEFORE the fill | the break bar's close |
| per-bar volume (E4a, E4b) | the bar's own `v` | that bar's close |
| session VWAP (E4c) | cumulative `(h+l+c)/3 × v` over bars `m ≥ 570` up to and including bar `k` | bar `k`'s close |
| MACD / signal / histogram (E4d, E4e, E4f) | EMA(12), EMA(26), EMA(9) `adjust=False` over 1-min closes from `m = 570` up to bar `k` | bar `k`'s close |
| the book's slot state (re-booked measure) | exits with `exit_m <` the candidate's `entry_m` | the candidate's entry minute (`run_book`'s causal freeing) |

**No cell reads a field dated after the minute it acts on.** No cell reads a daily bar of `day` or
later. No TEST-dated bar is read by the walk.

## 8. Resource rules

One python process at a time, `nice -n 15`, `ulimit -v 1500000`; `cache.db` and `bars_sip.db` opened
**read-only**; the heavy bar walk runs only when `pgrep -f hod_fresh` is empty and is checkpointed per
day. No config, `orb.yaml`, systemd unit, cron, order or cache is written. The dry run is not touched.
