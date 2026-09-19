# green_weeks — does EXIT DESIGN buy green weeks?

Pre-registered in `PREREG.md` (written before any cell was scored). TEST sealed by
`FREEZE.md`. Run 2026-09-19, one `nice -n 10` process at a time under
`ulimit -v 3000000`; `cache.db` and `bars_sip.db` opened read-only. No config, order,
service, cron or production artefact was written — BF cells drove the shipped
`batch_backtest.py` through scratch copies of `config.yaml`.

Owner, 2026-09-19: *"Do whatever it takes to lift frequency! And then to get those
green weeks in."* Frequency was settled in `research/bf_frequency`. This stage tests
the lever nobody had tested — **exit design** — on the metric nobody had ranked on:
**% of green weeks**.

---

## PAGE ONE

### The answer: the thesis is REFUTED. On all three books the shipped exit is at the top of the pre-committed ranking, and earlier profit-taking makes week shape WORSE.

| book | best cell by `min(green% TRAIN, green% VAL)` | shipped exit's rank |
|---|---|---|
| **HOD-break** | **E0 = shipped** (45.3% / 39.1%) | **1 of 12** |
| **ORB** | **X0 = shipped**, tied with 5 others at 30.4% VAL | **1 of 17** (tied) |
| **BF** | **E0 = shipped** (52.8% / 52.2%) | **1 of 10** |

**No cell in any book passes PREREG §8's gate** (top-3 on both splits *and* the
discordant-week count of §7). Under `FREEZE.md` §5 that means **TEST was not opened
for any book.** It is still sealed.

### The falsification, in one book

HOD-break is the only book that trades often enough to resolve anything (~44
trades/week; only 2–4% of weeks are flat). On TRAIN, moving the take-profit from the
shipped **+2R to +0.5R**:

| | shipped +2R | +0.5R | direction |
|---|---|---|---|
| **win rate** | 36.8% | **61.8%** | thesis says this should buy green weeks |
| **green weeks** | **45.3%** (24/53) | **18.9%** (10/53) | it bought the OPPOSITE |
| longest red-week streak | 7 | **18** | |
| average win | +1.68R | +0.50R | |
| average loss | −1.03R | −1.06R | **unchanged — the losers were never touched** |
| profit factor | 0.95 | **0.77** | |

And it is a clean dose-response, not a single point. The whole E1 ladder is monotone
on TRAIN — green weeks **18.9 → 30.2 → 37.7 → 43.4 → 45.3%** as the target goes
**+0.5 → +0.75 → +1.0 → +1.5 → +2.0R** — with profit factor monotone alongside
(0.77 → 0.83 → 0.87 → 0.91 → 0.95) and win rate monotone *against* (61.8 → 54.1 →
48.0 → 40.5 → 36.8%). Discordant weeks 11–20, so this is resolved, not noise.

### Why it fails, mechanically — and the number that flips with frequency

Week shape is not set by the hit rate. It is set by **expectancy at the weekly
aggregation**. A shorter target shrinks every winner and leaves every loser at −1R:
the week's losers still cost the same, and the winners that used to cover them no
longer do. At ~44 trades a week a +0.5R/−1R book needs a 67% hit rate to break even
and gets 62%. The flat and red weeks the change was supposed to rescue stay red —
they only get *shallower* — while the green weeks that were carried by one runner go
flat. **Earlier taking does not convert flat weeks to green; it converts green weeks
to flat.**

The rank correlation across cells says the same thing, and shows where the thesis
*does* have a grain of truth:

| book | trades/week | rho(win rate, green%) | rho(total P&L, green%) |
|---|---|---|---|
| **HOD** | ~44 | **−0.47 TRAIN / −0.53 VAL** | **+0.50 / +0.75** |
| **ORB** | ~2.0 | +0.76 / +0.42 | −0.30 / −0.04 |
| **BF** | ~2.5 | +0.00 / +0.66 | +0.15 / +0.77 |

**The sign flips with frequency, and that is the real finding.** When a week is 44
trades, the week's sign is the sign of *expectancy* — win rate is irrelevant and
green weeks track total P&L. When a week is 2 trades, the week's sign is very nearly
the sign of one trade, so win rate does drive green weeks. But at 2 trades a week the
binding constraint is not the exit at all: **26–36% of ORB's weeks have no fill, and
no exit rule can make a week green that never traded.**

### The lever that actually moves week shape is frequency, not exits

The only intervention in this program that has moved green-week share by more than
the noise band is `research/bf_frequency`'s **F7** (drop the ADV20 and conviction
gates): BF green weeks **30.2% → 52.8%** on TRAIN and **40.9% → 54.5%** on VAL,
achieved by converting **flat** weeks (56.6% → 9.4%) into small green ones. That is
the mechanism that works. Exit design cannot create a week; it can only redistribute
the P&L of weeks that already trade — and this stage shows that when it redistributes
toward small early wins it makes the redistribution worse.

### One recommendation per book

* **HOD-break — KEEP the shipped +2R target. No variant beats it, and the small-target
  family is decisively worse.** This is the strongest result in the stage: monotone,
  resolved (11–20 discordant weeks), and consistent on TRAIN and VAL. It also settles
  a live question: the engine is running dry with `target_r: 2.0`, and nothing here
  justifies changing it. (The book is negative in every cell on the honest SIP
  population — this stage re-confirms `bf_zero/REPORT.md` §6b and does not revive it.)
* **ORB — KEEP the shipped static lock.** No variant beats it on the pre-committed
  ranking; every difference is unresolved (discordant weeks 0–7 against a bar of 8).
  The actionable fact is not an exit: **ORB's flat-week share is identical (35.8%
  TRAIN / 26.1% VAL) in all 17 cells** because 13 of 19 flat TRAIN weeks had no pick
  at all and 6 had only modelled non-fills. If ORB's week shape is to improve, the
  work is frequency and fill rate, not exits.
* **BF — KEEP the shipped 50%-at-+2R partial.** It is top of the min-ranking. The two
  cells that beat it on one split (E2a on TRAIN at 58.5%, E2c on VAL at 56.5%) lose on
  the other, which is the split-inconsistency signature this program has learned to
  discard. BF remains the weakest book for this question: 2.2–2.8 trades/week and a
  ±13.5pp / ±20.9pp unpaired band.

**Nothing ships from this stage.** The three recommendations are all "leave it alone".

---

## 1. What was run

40 pre-declared cells (PREREG §4): **ORB 17** (12 new + Stage M's 5 challengers
re-ranked, not re-run) · **HOD 12** · **BF 10** (E3 collapses to E2@+1.0R; E4
NOT SIMULABLE, §5). Plus the ORB N=3/N=12 flanks, which were not needed once no ORB
cell separated.

| book | instrument | what moved | what did not |
|---|---|---|---|
| ORB | `orb_exits.py` — Stage M's walker plus three new levers (fixed target, profit partial, trail on the remainder), then the shipped selector once per cell via `ORB_BT_RESIM_CACHE` | the per-candidate exit | the 13,033-candidate entered-inclusive population, the whole B+ selection stack, sizing, N=8 |
| HOD | `hod_exits.py` — a parametrised `walk_exit` over the SIP-tape honest population, then `trading.hod_break.run_book` per cell (a changed exit changes slot freeing, so the book is rebuilt) | the exit only | entries, fills, the live config (price ≥ $20, last entry 14:00, 12/day, 4 concurrent) |
| BF | `bf_exits.py` — the shipped `batch_backtest.py`: `--resim-exits` with the cell's exit knobs, then Stage-2 with the F7 gate set at the live rails | the exit knobs | the F7 selection (verified byte-identical to `bf_frequency/runs/F7_L.csv`) |

## 2. Verification gates (PREREG §6) — all passed before any cell was read

1. **ORB E0 parity.** The extended walker with every new lever neutral reproduced the
   SHIPPED `simulate_winner_stack` price *and* reason on **7,402 / 7,402 fills**
   (hard assert, aborts on first mismatch), and its X0 dump equals Stage M's committed
   `dump_X0.csv` at `max |Δpnl| = 0.000000000000` over 13,033 rows. The X0 book at
   N=8 reproduces Stage M to the cent: **$14,428.62 / 215 picks**.
2. **HOD E0 parity.** The parametrised walk reproduced `hod_break.walk_exit` on
   **15,938 / 15,938** walked signals (`max |dev| = 0.0`) and `spec_trades.csv`'s own
   `rr` at `max |dev| = 4.4e-16`. The E0 book is **4,025 trades** — the row count of
   the committed `spec_book_live_config.csv` — and its green weeks are **24/53 (45.3%)
   and 9/23 (39.1%)**, exactly `bf_zero/REPORT.md` §6a.
3. **BF F7 parity.** `E0_CACHE` is byte-identical to `bf_frequency/runs/F7_L.csv` on
   (symbol, date, entry_time, pnl) — 241 trades.
4. **Week-grid integrity.** Denominators generated from the split ranges independently
   of the trades: TRAIN **53** weeks, VAL **23**, TEST 14–15 (sealed). These match the
   `NW` table `bf_zero/spec_sim.py` already used.

## 3. Availability audit (PREREG §5)

| book | field coverage | note |
|---|---|---|
| ORB | bars present for **13,033 / 13,033** candidate symbol-days (0 kept-recorded); ATR14 for the stop floor on **12,707 / 13,033 (97.5%)** | the ATR gap fails open to `range_low` with a WARNING, identically in all 17 cells |
| HOD | tape present for **15,938 / 15,938** pool signals (`missing tape 0`) | cache.db first, then the Alpaca-SIP re-fetch; no thin-tape store is read |
| BF | 896 cache rows; `planned_entry` present on the regen-7 cache, rich master joined from `backtest_results/backtest_full_2025_01_to_2026_08.csv` | the resim's own unfaithfulness is quantified in §5 and is common-mode across cells |

Missingness does not depend on the outcome in any of the three: ORB's 326 ATR gaps and
BF's rich-master misses are fixed properties of the symbol-day, identical in every cell,
so they cancel in the E0-vs-En diff.

## 4. The cell tables — ranked on GREEN WEEKS

`disc` = weeks green in exactly one of {this cell, shipped}. **PREREG §7: fewer than
8 on TRAIN or 5 on VAL = NOT RESOLVED**, whatever the size of the difference.
`top5%` is a diagnostic only and never enters a ranking (PREREG §2).

### 4.1 HOD-break — the one book that can resolve this

TRAIN, 53 weeks, totals in R:

| cell | green% | flat% | red% | red-streak | worst wk | mo green% | MDD | WR% | total R | disc |
|---|---|---|---|---|---|---|---|---|---|---|
| **E0 shipped +2R** | **45.3** | 1.9 | 52.8 | 7 | −17.5 | 50.0 | −148.6 | 36.8 | −69.4 | — |
| E1d target +1.5R | 43.4 | 1.9 | 54.7 | 5 | −20.3 | 25.0 | −151.2 | 40.5 | −134.8 | 11 |
| E2c 50% @ +1.5R | 43.4 | 1.9 | 54.7 | 7 | −19.4 | 41.7 | −131.5 | 41.1 | −64.2 | 7 |
| E2a 50% @ +0.5R | 41.5 | 1.9 | 56.6 | 5 | −16.7 | 16.7 | −162.8 | 62.5 | −135.7 | 12 |
| E4a time box 15m | 41.5 | 1.9 | 56.6 | 8 | −14.7 | 16.7 | −120.8 | 39.3 | −88.2 | 16 |
| E3 50% @ +1R + trail | 41.5 | 1.9 | 56.6 | 6 | −17.7 | 16.7 | −132.8 | 48.2 | −113.7 | 8 |
| E1c target +1.0R | 37.7 | 1.9 | 60.4 | 7 | −18.6 | 16.7 | −197.7 | 48.0 | −191.3 | 12 |
| E4b time box 30m | 37.7 | 1.9 | 60.4 | 6 | −17.5 | 33.3 | −135.9 | 39.6 | −78.7 | 8 |
| E5 BE @ +0.75R | 37.7 | 1.9 | 60.4 | 5 | −22.6 | 58.3 | −136.2 | 17.3 | −24.8 | 20 |
| E2b 50% @ +1.0R | 34.0 | 1.9 | 64.2 | 7 | −17.3 | 41.7 | −155.0 | 48.3 | −98.1 | 10 |
| E1b target +0.75R | 30.2 | 1.9 | 67.9 | 7 | −20.5 | 8.3 | −238.6 | 54.1 | −220.6 | 16 |
| E1a target +0.5R | **18.9** | 1.9 | 79.2 | **18** | −18.4 | 0.0 | −262.6 | 61.8 | −261.4 | 20 |

VAL, 23 weeks: E1c 43.5 · **E0 39.1** · E4a 39.1 · E2c 39.1 · E4b 39.1 · E5 34.8 ·
E3 34.8 · E1d 34.8 · E2b 30.4 · E1a 21.7 · E1b 17.4 · E2a 17.4.
E1c's VAL win (43.5 vs 39.1, disc 5) is contradicted by its TRAIN loss (37.7 vs 45.3,
disc 12) — the split-inconsistency signature, discarded by PREREG §8.

**Mechanism table (TRAIN), the reason the ranking looks like it does:**

| cell | n | WR% | avg win | avg loss | PF | R/trade | %exits at target |
|---|---|---|---|---|---|---|---|
| E0 +2R | 2336 | 36.8 | +1.68 | −1.03 | **0.95** | −0.030 | 27.0 |
| E1d +1.5R | 2503 | 40.5 | +1.38 | −1.03 | 0.91 | −0.054 | 35.2 |
| E1c +1.0R | 2664 | 48.0 | +0.97 | −1.03 | 0.87 | −0.072 | 45.6 |
| E1b +0.75R | 2743 | 54.1 | +0.74 | −1.05 | 0.83 | −0.080 | 53.0 |
| E1a +0.5R | 2772 | 61.8 | +0.50 | −1.06 | **0.77** | −0.094 | 61.5 |
| E5 BE-only | 2327 | 17.3 | +2.71 | −0.58 | 0.98 | −0.011 | 0.0 |

The average LOSS is −1.03R in every target cell. That is the whole story: these exits
only ever touched the winners.

### 4.2 ORB — nothing separates, and the flat weeks are structural

TRAIN, 53 weeks, N=8, totals in $ (17 cells; the five `X*` are Stage M's, re-ranked):

| cell | green% | flat% | red% | red-streak | worst wk | mo green% | MDD | WR% | total $ | disc |
|---|---|---|---|---|---|---|---|---|---|---|
| E2a 50% @ +0.5R | 41.5 | 35.8 | 22.6 | 3 | −318 | 83.3 | −589 | 41.9 | 2,339 | 4 |
| E3 50% @ +1R + trail | 39.6 | 35.8 | 24.5 | 2 | −263 | 83.3 | −688 | 38.1 | 2,577 | 3 |
| E1d target +1.5R | 39.6 | 35.8 | 24.5 | 2 | −418 | 83.3 | −632 | 35.2 | 3,665 | 1 |
| E2b 50% @ +1.0R | 39.6 | 35.8 | 24.5 | 2 | −263 | 91.7 | −551 | 38.1 | 3,542 | 3 |
| E2c 50% @ +1.5R | 39.6 | 35.8 | 24.5 | 2 | −373 | 83.3 | −501 | 35.2 | 5,178 | 1 |
| **X0 shipped** | **37.7** | 35.8 | 26.4 | 3 | −418 | 91.7 | −528 | 35.2 | **6,662** | — |
| E1a target +0.5R | 37.7 | 35.8 | 26.4 | 3 | −290 | 50.0 | −598 | 41.9 | −111 | 6 |
| E4b time box 20m | 37.7 | 35.8 | 26.4 | 3 | −418 | 91.7 | −741 | 34.3 | 4,440 | 2 |
| E1c / E1b | 35.8 | 35.8 | 28.3 | 5 / 3 | −263 | 75.0 / 50.0 | −733 / −582 | 38.1 / 39.0 | 1,940 / 754 | 5 / 5 |
| E5 BE @ +0.75R | 34.0 | 35.8 | 30.2 | 3 | −381 | 91.7 | −461 | 25.7 | 4,587 | 2 |
| E4a time box 10m | 32.1 | 35.8 | 32.1 | 3 | −226 | 75.0 | −537 | 32.4 | 3,767 | 5 |
| M-X1 / X2 / X3 | 32.1 | 35.8 | 32.1 | 3 | — | — | — | — | 3,926 / 6,158 / 6,960 | 5 / 3 / 5 |
| M-X4 / X5 | 30.2 / 28.3 | 35.8 | 34.0 / 35.8 | 4 | — | — | — | — | 4,745 / 2,717 | 6 / 7 |

On VAL (23 weeks) **six cells tie at the top on 30.4% including the shipped X0**, and
the best discordant count anywhere in the ORB grid is 7 — under the resolution bar of
8. E2a's +3.8pp on TRAIN rests on 4 discordant weeks. **Nothing here is resolved.**

**The flat weeks are not an exit problem** (`analysis_extra.py`):

| split | weeks | flat | no pick at all | only modelled non-fills | filled and netted exactly 0 |
|---|---|---|---|---|---|
| TRAIN | 53 | 19 (35.8%) | **13** | **6** | 0 |
| VAL | 23 | 6 (26.1%) | **5** | **1** | 0 |

Identical in all 17 cells, by construction — which is why the `flat%` column never
moves. Converting these weeks needs picks and fills, i.e. frequency, not an exit.

### 4.3 BF — reported, but the weakest instrument

TRAIN, 53 weeks, totals in $ (F7 selection, 241 trades, 2.81/week):

| cell | green% | flat% | red% | red-streak | worst wk | mo green% | MDD | WR% | total $ | disc |
|---|---|---|---|---|---|---|---|---|---|---|
| E2a 50% @ +0.5R | **58.5** | 9.4 | 32.1 | 3 | −6,390 | 75.0 | −17,435 | 55.0 | 66,480 | 11 |
| E0_CACHE (anchor) | 52.8 | 9.4 | 37.7 | 3 | −13,098 | 75.0 | −19,091 | 53.0 | 169,954 | 0 |
| **E0 shipped +2R/50%** | **52.8** | 9.4 | 37.7 | 3 | −12,947 | 75.0 | −19,516 | 51.0 | 128,982 | — |
| E1a whole @ +0.5R | 52.8 | 9.4 | 37.7 | 3 | −6,390 | 66.7 | −19,029 | 55.0 | 27,373 | 14 |
| E2b 50% @ +1.0R | 50.9 | 9.4 | 39.6 | 3 | −12,232 | 75.0 | −18,960 | 59.7 | 73,739 | 5 |
| E1d / E2c | 49.1 | 9.4 | 41.5 | 3 | −12,298 / −12,236 | 75.0 | −16,719 / −16,232 | 53.0 / 54.4 | 77,499 / 109,951 | 4 / 4 |
| E1b / E1c | 47.2 | 9.4 | 43.4 | 3 | −13,332 / −12,110 | 58.3 / 66.7 | −18,870 / −17,668 | 56.4 / 58.4 | 23,297 / 35,298 | 9 / 7 |
| E5 BE @ +0.75R | 45.3 | 9.4 | 45.3 | 4 | −12,400 | 75.0 | −16,275 | 40.9 | 122,816 | 8 |

VAL, 23 weeks: E2c 56.5 · **E0 52.2** = E0_CACHE 52.2 = E1b = E1d = E1c = E2b 52.2 ·
E1a 34.8 · E2a 34.8 · E5 30.4.

**E2a is the one cell in the whole stage that beats shipped on a split with the
discordant count met** (TRAIN 58.5% vs 52.8%, 11 discordant weeks) — and it is the
same cell that **loses on VAL by the same margin with 8 discordant weeks** (34.8% vs
52.2%). Opposite signs, both resolved. That is not a finding; that is a coin.

## 5. Faithfulness — where a cell is NOT what it says on the tin

* **BF's resim is not exit-faithful, so BF is compared resim-to-resim.** BT_STATUS
  §2a measures the gap at **−$10.3K / −7.4%** (the exhaustion rule does not re-fire;
  `post_fill_exit` is unmodelled). Every BF cell here including E0 runs through the
  resim so the drift is common-mode; `E0_CACHE` (regen-7's own exits, $169,954) is the
  level anchor and is **not** comparable to the other cells' totals.
* **BF E4 (time box) is NOT SIMULABLE** and was not run. The only time-based lever the
  shipped simulator has is `no_pop_exit`, whose threshold is a **percentage of price**,
  not R. Across this book R is 2–15% of price, so a fixed percentage is a different
  rule on every trade. Guessing it was not an option (PREREG §4).
* **BF E1 is the shipped partial at `fraction = 0.999`**, because the shipped config
  validator requires `0 < fraction < 1`. Audited on the resim caches: the residual is
  **≤ 1.35% of shares** (median 7 shares), **≤ $35 on any trade**, and **0.1–0.3% of
  the whole raw book**. PREREG §5's strict per-trade wording ("1% of any position's
  P&L") is exceeded on a handful of trades whose own P&L is near zero — a denominator
  artefact, not a distortion — so the cell is reported, with the numbers, rather than
  silently passed or silently dropped.
* **Profit-take fills are close-through, never touch.** A target or partial fires only
  on a bar that CLOSES at or above the level and fills AT the level — `hod_break`'s
  own shipped convention, applied to ORB's new cells too. PREREG §5 was corrected from
  an earlier `min(close, level)` draft to this rule **before any cell was scored**;
  the earlier rule would have fabricated exits at bad closes on spike-and-collapse
  bars. ORB additionally charges the shipped 10 bps exit slip on the profit leg, which
  is conservative for a resting limit and identical across cells. **BF is the declared
  deviation**: its shipped partial fills at the trigger bar's CLOSE through the
  stop-fill model, uncapped by the level — obtainable, but better than the level when
  the close is above it. It is E0's own spec, so it is common-mode across BF cells.
* **Inherited, declared, unchanged**: ORB's gap-through stop fills at the level;
  entry fills are the recorded ones; the 40% @ +3R scale-out and the ATR stop floor
  stay ON in every ORB cell (single-lever rule). When an ORB partial and the scale-out
  both apply, the partial's fraction is of the active position and the scale fraction
  stays on the original shares, capped so the legs never exceed 100%.

## 6. Power and multiplicity — stated before the numbers, restated after

Unpaired 95% half-width on a green-week share at p ≈ 0.5: **TRAIN ±13.5pp · VAL
±20.9pp · TEST ±25.3pp**. Cells are paired, so the real currency is discordant weeks.

* **HOD resolves.** Discordant counts 7–20; the E1 ladder is monotone across five
  points on TRAIN. This is a real, powered result.
* **ORB does not.** Max discordant count 7 against a bar of 8; and a third of its
  weeks cannot be moved by any exit.
* **BF barely.** Only E2a, E1a and E2b clear the bar on TRAIN; E2a and E1a clear it on
  VAL with the opposite sign.

40 cells × 3 books × 2 open splits = **80 primary readings**; at α = 0.05 about 4
would clear by chance, and the two that did (E2a on TRAIN, E2a on VAL) point opposite
ways. This is exactly the pattern PREREG §8 pre-committed to reading as noise. The
program-level cell count carries forward: these 40 sit on top of `research/fuckup_audit`'s
existing cells, 5 of which (Stage M's X1–X5) were re-used here rather than re-run.

## 7. TEST

**Not opened.** `FREEZE.md` §5: the gate selected no cell for any book, so TEST stays
sealed for all three. Nothing in this report contains a TEST number.

## 8. Files

| path | what |
|---|---|
| `PREREG.md` / `FREEZE.md` | pre-registration and the TEST seal |
| `weekshape.py` | the one week-shape scorer (green/flat/red, streak, MDD, discordance) |
| `orb_exits.py` | ORB walker: Stage M's plus target / partial / trail levers |
| `run_orb_cells.sh` | the shipped ORB selector, once per cell at N=8 |
| `orb_rerank_M.py` | Stage M's X0–X5 re-ranked on week shape (no backtest run) |
| `hod_exits.py` | HOD parametrised `walk_exit` + per-cell `run_book` |
| `bf_exits.py` | BF resim + Stage-2 per cell through scratch configs |
| `score.py` | the ranked cell tables, TEST-sealed |
| `analysis_extra.py` | the ORB flat-week decomposition and the WR-vs-green rank test |
| `cells.csv` | every metric for every cell and split |
| `orb_book_*_n8.csv`, `hod_book_*.csv`, `bf_runs/*.csv` | the per-cell books |

The 12 ORB per-candidate dumps (5 MB each) are gitignored: they are reproducible by
one `orb_exits.py` walk, which re-asserts parity against Stage M's `dump_X0.csv`. The
per-cell **books**, which every number above is built from, are committed.
