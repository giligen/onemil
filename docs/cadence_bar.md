# The cadence bar — what "a book that makes money" means (owner, 2026-09-20)

Owner's requirement, verbatim in spirit: *prefer weekly green; shallow reds and a STRONG green every few weeks is
fine; a 10R month with the rest flat is fine; what is NOT fine is a book that quotes monsters in the backtest and
leaves the account bleeding while the monster never arrives.*

Every bar we used before scored TRADES (mean R, t-stat, ex-top-5%). This one scores WEEKS and the CYCLE between
strong weeks. A book is evaluated as a renewal process: strong weeks are the renewals; the cycle is everything
between two of them. The book passes if the renewals come often enough, regularly enough, and the bleed between
them is shallow enough that the account is above water most of the time.

## 0. Units and walk

* **R** = the risk per trade at the CURRENT ramp stage of that book (BF $150, ORB $375 …). Dollar numbers are
  derived, never the other way round.
* **Week** = Mon–Fri session week. A week with |P&L| < 0.5 R is FLAT (neither green nor red).
* The backtest is walked at the **LIVE configuration**: live slot count, live fill model (obtainable fills only:
  next bar's open under a cap, never the touch of a level), measured NBBO cost, live vetoes. A full-timeline
  8-slot walk of a book that runs live at 4 slots is not evidence of anything.
* Splits as everywhere else: TRAIN 2025 (both halves), VAL 2026-01..05, TEST sealed. The bar must hold on TRAIN
  and VAL separately.

## 1. Definitions

| Term | Definition |
|---|---|
| **Strong week** | net week P&L ≥ **+5 R** |
| **Gap** | number of weeks from one strong week to the next (strong week itself excluded) |
| **Cycle** | the weeks strictly between two strong weeks, plus the closing strong week |
| **Bleed** | sum of weekly P&L over the weeks strictly between two strong weeks (≤ 0 by nature, can be > 0) |
| **Cycle net** | bleed + the closing strong week |
| **Under water** | consecutive weeks with equity below its high-water mark |

## 2. The bar (all seven, both splits)

| # | Criterion | Threshold | What it catches |
|---|---|---|---|
| C1 | **Cadence** — gap between strong weeks | median ≤ **3 wk**, 90th percentile ≤ **6 wk** | the once-a-quarter monster |
| C2 | **Bleed bound** | 90th-percentile bleed ≥ **−4 R**; cycle net > 0 in ≥ **75 %** of cycles | a base that digs a hole one strong week cannot fill |
| C3 | **Shallow reds** | weekly P10 ≥ **−2 R**, weekly min ≥ **−4 R**, max drawdown ≤ **8 R**, longest under-water run ≤ **6 wk** | the "poor position" |
| C4 | **Green share** | green weeks ≥ **55 %** AND ≥ count-matched null + 10 pp (flat weeks excluded from both) | a coin flip dressed as a book |
| C5 | **Frequency at live config** | ≥ **3 fills/week** | cadence is arithmetically impossible at 0.8 fills/wk (ORB today) |
| C6 | **Tail realism** | every trade ≥ +3 R passes the obtainability audit trade by trade (fill inside the bar that fills it, reachable by the order the engine would have had resting; exit reachable — no touch-fill at a target) | the monster that was never available |
| C7 | **Power** | ≥ **10 cycles** per split, and a 4-week block bootstrap of the weekly series puts the 75 % upper bound of the P90 gap ≤ 8 wk | fifteen cycles is a small sample; say so |

No criterion caps winners or removes the tail. The owner accepts a fat right tail; what is bounded is how often
it pays and how much it costs to wait for it. Ex-top-5 % and the capped book are REPORTED as diagnostics, never
as pass/fail.

## 3. Arithmetic sanity (why these numbers)

Median cycle 3 weeks, strong week ≥ +5 R, median bleed around −1.5 R → ≈ +3.5 R per cycle ≈ **+1.2 R/week**.
At ORB stage risk $375 that is ≈ $450/week ≈ 0.7 %/week on $66K; at ramp stage 3 ($1,000) ≈ 1.8 %/week. The
bar is consistent with the 1 %/week goal only up the ramp — which is the point of the ramp.

## 4. Live tracking — the tripwire that formalizes "the monster never arrived"

The same statistics are computed every Friday on LIVE fills (`scripts/cadence_bar.py --live --book X`). Two
pre-committed pause rules, no discretion:

* **Cadence miss**: the live gap since the last strong week exceeds the backtest's P90 gap + 2 weeks → PAUSE the
  book, re-verify the backtest cadence at the live config before resuming.
* **Bleed miss**: the live bleed since the last strong week is below the backtest's P90 bleed − 1 R → PAUSE.

With a P90 gap of 6 weeks, 12 live weeks without a strong week rejects the backtest cadence at ~99 %. Each book
therefore has a bounded evaluation horizon (the "cadence clock") instead of an open-ended wait.

## 5. Reporting format (every research report from now on)

```
CADENCE BAR  (book, split, live config: N slots, fill model, R = $X)
C1 gap       median a wk  P90 b wk        [pass/fail]   gaps: [..list..]
C2 bleed     P90 −c R     cycles net>0 d% [pass/fail]
C3 reds      P10 −e R  min −f R  MDD −g R  under-water max h wk   [pass/fail]
C4 green     i%  null j%                   [pass/fail]
C5 fills/wk  k                             [pass/fail]
C6 tail      m of n ≥3R trades obtainable  [pass/fail]   (list the failures)
C7 power     cycles p   bootstrap P90-gap 75% UB q wk    [pass/fail]
diagnostics  ex-top-5% r R   capped s R   top-5 share t%   weekly P&L histogram
```

## 6. What this would have said about the books we ran

* **ORB B+ honest book** ($6,085 / 21 mo at the $10K stage): fails **C5** outright (~0.8 fills/wk); at that
  frequency a strong week needs one +5 R trade, and the gap distribution is a matter of months. Frequency rungs
  first (research/orb_frequency), cadence bar second.
* **Bull flag P1**: 79 trades / 20 months, worst month −$11K at $2K base: fails C3 (weekly min) and, at ~1
  trade/week, C5. The 50 %-at-2R partial improved green share and shrank reds — a move in the direction of this
  bar, chosen before the bar existed.
* Any book passing the old per-trade bar but failing C1/C2 is a lottery ticket, which the owner has already
  rejected twice.

## 7. Rules of use

* PREREG the cadence bar numbers before scoring; the thresholds above are the defaults, an owner-set change is
  recorded in the PREREG with the reason.
* The cadence bar is computed on the SAME honest walk as everything else (rails in CLAUDE.md § research claims);
  it adds to those rails, it does not replace obtainability, causality, cost, or the independent rebuild.
* Scorer: `scripts/cadence_bar.py` — input a trade CSV (`date, pnl_R` or `date, pnl, risk`), prints the block
  above; `--live --book` reads the trades table. One implementation for BT and live.
