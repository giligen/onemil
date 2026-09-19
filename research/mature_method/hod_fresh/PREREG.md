# hod_fresh — PRE-REGISTRATION (written and committed BEFORE any cell was scored)

The LAST pre-registered pass on HOD-break. Owner 9/19: *"we will not leave it till you find it."*

`hod_losers/REPORT.md` §5 inverted the level-quality hypothesis: an OLD, much-tested HOD is the
WORSE break in all four base×split. The only positive-dollar direction the anatomy produced was the
**FRESH high** — `consol_bars < 8` reads **VAL +0.207 R gross, +$2,977, 52.2 % green weeks** — and it
died at **5.6 trades a week** because it was run as a FILTER on the first-qualifying break. Its own
closing line: *"a different admission rule, not a different filter on this one"*. And the loser path
(MFE +0.45 R at minute 6 → bleed → stop at minute 28, `hod_losers` §4) says the shipped
consolidation-low stop is the wrong stop for a momentum-continuation book.

So this pass changes **the admission** and **the stop**. It is a different book on the same tape.

---

## 0. Rails (every one, no exceptions)

* **TEST SEALED.** No TEST-dated bar is read by the bar pass (`pass3.py` walks 2025-01-02 →
  2026-05-31 only) and no TEST number is computed unless `FREEZE.md` exists AND `--test` is passed.
  TEST is opened **once**, for a G2 survivor only.
* **Reproduction gate, before any new number is quoted.** Three rows must match the printed digits:
  | id | what | reference |
  |---|---|---|
  | **R1** | `B0` shipped from `hod_filter_stack/pop.csv` | 1,688 / −0.027 / −0.088 / 41.5 % / **−$14,835** · 820 / +0.016 / −0.050 / 43.5 % / **−$4,128** |
  | **R2** | `P13` = B0 ∧ `consol_bars < 8` (the anatomy's lead) | 297 / 5.6 wk / −0.038 / −0.104 / 35.8 % / **−$3,099** · 219 / 9.5 wk / **+0.207** / +0.136 / **52.2 %** / **+$2,977** |
  | **R3** | `pass3.py`'s own base = `B2` (consolidation OFF, rv cap OFF, no age condition) | 1,622 / −0.039 / −0.107 / 32.1 % / **−$17,346** · 706 / +0.083 / +0.013 / 43.5 % / **+$893** |
  R3 is the independent-rebuild check: a **new bar pass** must reproduce `hod_filter_stack/pass2.py`'s
  book to the dollar before its new admission rows are trusted. A miss on any of the three stops the
  pass.
* **Cost is MEASURED PER STOP DESIGN.** The +0.2151 R constant was measured at the shipped R and is
  NOT carried. For every cell the report prints the mean per-trade cost in R
  (`half + half×ratio`, `half = 0.5 × spread% / R%`, `ratio` = the score4 exit contract
  stop 0.875 / eod 0.412 / target 0.0) and the median **R as % of price**, so cost/R is visible and
  the mechanism "a wider stop lowers cost/R" is testable rather than asserted.
* **Fill model unchanged** — the engine's capped limit at `level × 1.006`, filled at the NEXT bar's
  open iff that open is at or under the cap; `obtainable` (quoted SIP ask ≤ cap) enforced; the
  unfilled counterfactual is reported for the selected admission rung.
* **Owner metric primary**: % green weeks over every market week (no-trade = flat, in the
  denominator), then longest red streak, worst week, weekly $ at the live $100 risk, trades/week.
* **Count-matched permutation null** (2,000 draws, per-week pick count fixed) on every scored cell.
* **DAY-CLUSTERED SE beside every iid t** (`hod_preopen_regime` §4's standing rule; clusters =
  trading days). Any day-level gate is judged on the clustered number.
* **BOTH HALVES OF TRAIN** (H1-2025 `day < 2025-07-01`, H2-2025) reported for every cell. **A cell is
  a candidate only if same-signed in H1, H2 AND VAL.**
* **Two-cohort diagnostic**: the share of each cell's trades falling on symbol-days whose daily range
  is ≥ 10 % — so we can see whether the fresh-high admission is simply enriching for mover days.
* **Availability audit** on the two new decision fields: `consol_bars` (bars strictly BEFORE the
  break bar) and `atr20d` (the symbol's 20 prior DAILY sessions, strictly before `day`).
* **Membership** identical to every prior pass: early closes (2025-07-03, 11-28, 12-24) out, test
  tickers out (`research/scripts/pit_listings.is_test_ticker`), non-`daily_bars` symbols out.
* Node: one python process, `nice -n 10`, `ulimit -v 3000000`; `cache.db`, `bars_sip.db`,
  `daily_bars` opened **read-only**. No config, `orb.yaml`, systemd unit, cron, order or cache is
  written. The dry run is not touched.

## 1. The base and the scan rule (named, per the F6 standing rule)

**Base `B2`** = `hod_filter_stack` §2's corrected population: consolidation-tightness filter **OFF**
(stop = the plain low of the last 5 bars, no HOD-proximity test), rv **≥ 1** with **no upper cut**.
Live cascade, unchanged and in the engine's order: first-qualifying break → `fill_capped` →
`r_pct ≥ 1 %` → `price ≥ $20` → spread ≤ 100 bps → spread ≤ 15 % of R → obtainable → entry ≤ 14:00.
Book **12/day, 4 concurrent, $100 risk**.

**Scan rule: KEEP-SCANNING.** A break bar that fails the pass's admission condition does NOT retire
the symbol-day; the scan continues to the next break bar. (The shipped engine's first-break rule is
`stale_break`; the admission rules below are a *different admission*, which is the whole point — a
filter on the first break is what died at 5.6 trades a week.) Every table names its scan rule.

`consol_bars` = the `hod_losers/walk.py` definition, verbatim: the number of consecutive 1-minute
bars immediately before the break bar whose LOW is ≥ `level × 0.96`. Computed from bars strictly
before the break bar.

## 2. Declared cells — **15**, in four families

### A. Admission — the fresh-high ladder (5 cells), stop = the B2 stop (plain 5-bar low)

| cell | admission (first break with …) |
|---|---|
| **A1** | `consol_bars ≤ 3` |
| **A2** | `consol_bars ≤ 5` |
| **A3** | `consol_bars ≤ 8` |
| **A4** | `consol_bars ≤ 12` |
| **A5** | `consol_bars ≥ 20` — the **control**; expected NEGATIVE |

Reported per rung: signals/week (the ≥ 10/wk goal), gross, net, cost/R, green %, both TRAIN halves,
VAL, iid t and day-clustered t, null band, ≥10 %-range-day share.

**Selection of the A rung, pre-committed here:** eligible = ≥ 10 trades/week on BOTH splits AND gross
same-signed in H1-2025, H2-2025 and VAL. Among the eligible, rank by
`min(green % TRAIN, green % VAL)`; ties broken by `min(TRAIN gross, VAL gross)`. If NO rung is
eligible, the pass takes the rung with the highest `min(TRAIN gross, VAL gross)` among those at
≥ 10/wk, carries it through B/C/D **for completeness**, and the report states that the eligibility
test failed — a cell carried this way can never clear a bar.

### B. Stop design (5 new cells), crossed with the selected A rung

The target stays `entry + 2 × R` of whatever R the stop defines; the exit spec is otherwise the
shipped one (`hod_losers` settled the exits: the shipped exit is rank 1 of 12).

| cell | stop |
|---|---|
| **B-i** | the shipped **consolidation low** (K=5 bars within 4 % of the HOD); rows with no such zone are dropped |
| **B-ii3** | the low of the last **3** bars before the break |
| *(ref)* | the low of the last **5** bars — this IS the A-rung cell, printed as the reference row, not a new cell |
| **B-iii05** | `entry − 0.5 × ATR20d` |
| **B-iii10** | `entry − 1.0 × ATR20d` |
| **B-iv** | the **breakout bar's own low** |

`ATR20d` = the mean true range of the symbol's 20 DAILY sessions strictly before `day`
(`max(h−l, |h−pc|, |l−pc|)`), in dollars, from `cache.db::daily_bars`. Rows without 20 prior
sessions are dropped from the two ATR cells only (coverage printed).

Reported per stop: **median R as % of price**, **mean cost/R**, gross, net, $, green %, trades/week —
so the cost-wall mechanism is measured, not assumed.

**Selection of the A×B cell**: the same pre-committed rule as A, applied over the 6 stop rows
(5 new + the reference).

### C. The day gate (3 cells), crossed with the selected A×B cell

`hod_preopen_regime`'s D2 — SPY's own 09:30 open → 09:34 close, known at 09:35:00 — the best day gate
in the programme (+0.388 / +0.189 R separation, day-clustered t +2.00 / +2.37).

| cell | gate |
|---|---|
| **C1** | `spy_r5_pct > 0` |
| **C2** | `spy_r5_pct ≥ +0.2 %` |
| **C3** | `spy_r5_pct ≥ +0.4 %` |

Judged on the DAY-CLUSTERED SE. `hod_preopen_regime` §5 already records that D2's TRAIN half is
entirely H1-2025 — if that holds here, C fails the half-consistency rail and is reported as failing.

### D. Interactions (2 cells) — only two, declared here, no others

| cell | rule |
|---|---|
| **D1** | selected A×B **× `rv_profile ≥ 5`** (the strongest positive trade gate in the gate map, `hod_break` §5.2: +0.177 R, t 4.24 — here tested as a KEEP, i.e. the high-rv side) |
| **D2** | selected A×B×C **× `spread ≤ 8 % of R`** (the B3 cost arm) |

**TOTAL DECLARED CELLS: 15.** Programme cumulative: **784** (through `hod_losers`) **+ 15 = 799.**
Reproduction rows R1–R3, the availability audit, the per-stop cost table and the unfilled
counterfactual are diagnostics, not cells.

## 3. Bars

**Claim bar G1** (TRAIN): net R > 0 with **iid t ≥ 2.0 AND day-clustered t ≥ 2.0**, ≥ 10 trades/week,
and selected-subset **TRAIN gross ≥ +0.25 R**.
**G2** (VAL, only if G1 passes): same sign, ≥ 50 % green weeks, positive weekly dollars.
TEST opened once, behind `FREEZE.md`, for a G2 survivor only.

**Ship bar (the owner's, unchanged)**: selected subset **gross ≥ +0.25 R on TRAIN and VAL** with the
cost/R for THAT stop stated; **≥ 10 trades/week**; **green weeks ≥ 50 % on VAL**; **positive weekly
dollars on both splits**; **day-clustered t ≥ 2 on TRAIN**.

* If a cell clears → **SHIP-TO-DRY** with the exact `HodBreakParams` / config diff, and whether each
  piece is a knob or a code change.
* If nothing clears → **STAY-DRY**, the MDE, and the one-paragraph closure of the HOD-break line.

## 4. Declared and deliberately NOT run

* No new feature search — `hod_filter_stack` scored 35 features and its learner lost to its own
  shuffled control.
* No week gate — `hod_losers` §3: red weeks pass a runs test in all four base×split.
* No exit-after-the-stop variants — `hod_losers` §1: every declared exit moves the book < ±0.03 R
  and the breakeven move is the worst of them.
* No re-test of `touch_n`, Rule M or Rule D — settled in `hod_losers` §5 / §2 of Part 2.
