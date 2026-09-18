# Stage O / S1-REVIVE — the halt-resume book, re-opened properly and buried properly

Pre-registered in `REVIVE/PREREG.md` (frozen before any scoring, incl. the two 9/18 amendments);
gate order frozen in `REVIVE/FREEZE.md`. Phrasing per `research/fuckup_audit/PLAN.md` §1.

**STATUS: 0 of 24 pre-declared cells is positive on TRAIN — not "fails the t-test", *negative*. The
best of the 24 is −0.011 R. Neither Bar A (the claim bar) nor Bar B (the live-exploration bar) is
reached, and TEST was not read for any cell. The owner was right that the closure was premature —
but the two things that were actually wrong with it (the entry-spread charge and the outlier-driven
cost) both make the book LOOK BETTER when fixed, and it is still negative. The one genuinely
untested idea, the long side at a pullback limit, is the worst arm in the study, and the measurement
says why.**

---

## 1. What was actually wrong with the closure — three corrections, all applied

| # | the closure said | the truth | effect on the book |
|---|---|---|---|
| 1 | O_halt charged `0.25·half_entry` on a limit entry | **We control the entry price.** A limit never crosses; a non-fill is 0 P&L, not a loss. The entry-side cost is adverse selection + opportunity cost, both MEASURED here (§4) | helps (small — PASSIVE had already zeroed it; O_halt had not) |
| 2 | PASSIVE: "−0.840 R at t −3.31 vs MDE 0.711 — a **powered rejection**" | **Overstated as written.** That mean was driven by broken prints — max cover spread 474% of price, and 19–22% of charged exit quotes are >30 s stale. A 474% spread is not a cost | helps: the same cell reads **−0.410 (t −2.28, MDE 0.503)** once unusable measurements are replaced by the split median, and **−0.224** under a flat median charge |
| 3 | nothing was said about the tail of the *no-stop* arm | every positive number in this book is a tail: ex-top-5% the no-stop +5m cell is **−0.92 / −0.06 / −0.08** (TRAIN/VAL/TEST); winners capped at +3R, **−1.07 / −0.47 / −0.31** | decisive, and against |

So: **the closure's stated grounds were wrong as written** — a null was reported as a powered
rejection, on a mean one broken print could move. Corrections 1 and 2 were applied here in the
book's favour and the verdict did not change. Correction 3 is new and is what actually settles it.

**Also corrected here:** the quarantine instinct itself, applied literally, is a look-ahead. Dropping
a trade because its *exit* quote turned out stale selects on something unknowable at entry, and it
correlates with the exit type (a passive target exit needs no quote at all) — it removed 30–35% of
rows. This stage drops only on a bad **entry** quote (0.0% of events) and replaces an unusable exit
*measurement* with the split's median charged spread, keeping every trade.

---

## 2. The fill model (owner, 9/18) — and the construction check that matters

```
entry   a resting LIMIT at a price we declare before the resume prints.  Cost: ZERO.
fill    iff the market trades at or through it in [entry_t, resume+5m].  Fill price = the limit.
nofill  NO TRADE, exactly 0 P&L, counted — never booked as a loser.
exit    stop 1R / target 2R / horizon; entry bar EXCLUDED on both legs (the repo's own
        `trading/bf_trail.entry_bar_excluded` convention); stop-first tie-break; gap-through honoured.
cost    ONLY on the exit, on the MEASURED Alpaca SIP NBBO at the exit instant:
        target 0.0 half-spreads (a resting limit) · stop 1.875 · horizon 1.412
```

**Construction check:** the `r_pct = 0` (no-stop) arm reproduces PASSIVE's b=0/touch gross to three
decimals — **+0.3187 R vs PASSIVE's +0.319 on TRAIN**, fill rate **94.00%** vs 94.0%. Same book;
only the cost treatment and the levels changed.

Measured exit NBBO on charged exits, % of price: TRAIN mean 1.99 / median **1.54** / p95 5.45 ·
VAL 1.99 / **1.41** / 5.97 · TEST 2.11 / **1.50** / 5.70. New quote pulls: 6,550 exit instants +
1,105 long-book entry instants, Alpaca SIP, **$0** (no Databento byte).

---

## 3. Lever by lever — what each buys, TRAIN, one lever at a time

Base = short · b0 · R 2% · +5m · no gate = **−1.222 R** (gross −0.519: the bracket itself is the
first problem). `pt` = per-trade measured charge · `w95` = winsorised at the split p95 · `med` = the
split's median spread charged to every trade.

| lever | n | tr/wk | net R (pt) | t | w95 | med | gross | what it buys |
|---|---|---|---|---|---|---|---|---|
| **BASE** short b0 R2% +5m | 768 | 14.8 | **−1.222** | −9.78 | −1.160 | −0.939 | −0.519 | — |
| L1 spread gate ≤ 1.25% | 245 | 5.0 | −0.601 | −3.18 | −0.584 | −0.666 | −0.265 | **+0.62 R**, at 1/3 the trades |
| L1 gate ≤ 0.75% | 159 | 3.5 | −0.463 | −2.11 | −0.463 | −0.566 | −0.180 | +0.76 R |
| L1 gate ≤ 0.50% | 104 | 2.5 | −0.461 | −1.65 | −0.461 | −0.556 | −0.176 | +0.76 R, below 3/wk |
| L1 gate ≤ 0.25% | 44 | 1.4 | **+0.221** | +0.68 | +0.221 | +0.098 | +0.426 | the only positive cut in the study — 44 trades, t 0.68 |
| **L2 NO-STOP** (= PASSIVE) | 768 | 14.8 | −0.410 | −2.28 | −0.375 | −0.224 | **+0.319** | **+0.81 R — the stop is the single most expensive choice** |
| L2 R = 4% | 768 | 14.8 | −0.370 | −4.89 | −0.343 | −0.245 | −0.019 | +0.85 R |
| L2 R = 6% | 768 | 14.8 | −0.226 | −4.08 | −0.211 | −0.142 | +0.036 | +1.00 R |
| L2 R = 8% | 768 | 14.8 | −0.151 | −3.44 | −0.132 | −0.083 | +0.059 | +1.07 R — monotone, never crosses zero |
| L3 exit all-marketable | 768 | 14.8 | −1.373 | −11.46 | | | −0.519 | the passive target leg is worth **+0.152 R** |
| L4 horizon +30m | 768 | 14.8 | −1.281 | −10.52 | −1.217 | −0.995 | −0.578 | −0.06 R |
| L4 horizon EOD | 768 | 14.8 | −1.278 | −10.49 | −1.214 | −0.992 | −0.576 | −0.06 R |
| L5 price ≥ $10 | 338 | 7.0 | −1.127 | −6.23 | −1.053 | −0.861 | −0.439 | +0.09 R |
| L5 price ≥ $20 | 174 | 4.6 | −1.386 | −5.29 | −1.325 | −1.062 | −0.636 | **−0.16 R (hurts)** |
| L5 ADV20 ≥ 500K | 491 | 9.4 | −0.897 | −6.31 | −0.867 | −0.751 | −0.339 | +0.33 R |
| L5 ADV20 ≥ 1M | 354 | 7.1 | −0.822 | −5.03 | −0.803 | −0.712 | −0.293 | +0.40 R |
| **L6 LONG** rung d = 0 | 673 | 12.7 | −0.856 | −6.05 | −0.813 | −0.591 | −0.164 | +0.37 R vs the short base |
| L6 LONG rung −0.5% | 588 | 11.1 | −1.303 | −8.79 | −1.252 | −0.992 | −0.526 | **−0.45 R** |
| L6 LONG rung −1% | 554 | 10.7 | −1.355 | −9.01 | −1.302 | −1.047 | −0.566 | **−0.50 R** |
| L6 LONG rung −2% | 478 | 9.2 | −1.436 | −9.41 | −1.382 | −1.127 | −0.650 | **−0.58 R — worst arm in the study** |

Every lever moves the book in the direction the owner's intuition said it would — the spread gate is
worth +0.6 to +0.8 R, a wider R up to +1.07 R, a liquidity floor +0.40 R, the passive target leg
+0.15 R — **and the sum of all of them still does not reach zero.** Best of the 24 gated cells:
−0.011 R.

---

## 4. The long side, and the mechanism that kills it (the answer to "control the buy order")

The hypothesis: a buy limit **below** the reopening print only fills on a pullback, which screens out
the worst entries. The measurement says it screens out the **best** ones. TRAIN, +5m, R = 2%, per
rung, against the counterfactual of entering every signal at the print:

| rung | fill rate | filled net R | print-entry, **the fills** | print-entry, **the NON-fills we missed** | n missed |
|---|---|---|---|---|---|
| d = 0 (at the print) | 92.6% | −0.856 | −0.858 | **−4.44** | 53 |
| d = −0.5% | 81.8% | −1.303 | −1.301 | −0.345 | 138 |
| d = −1% | 77.5% | −1.355 | −1.487 | **+0.065** | 172 |
| d = −2% | 68.6% | −1.436 | **−2.027** | **+0.631** | 248 |

Read the last two columns. At d = −2%, the 248 events the limit never caught would have returned
**+0.63 R** entered at the print, while the ones it did catch returned **−2.03 R** at that same print
entry. **Requiring a 2% pullback is a filter that keeps the falling knives and discards the runners.**
Genuine adverse selection, measured, not assumed — and the opposite sign to the hypothesis. (At
d = 0 the 53 missed are −4.44 R, i.e. right at the print the non-fills are the disasters and the
limit is mildly protective; the protection reverses as soon as the limit is moved down.)

**The short side has the symmetric structure**: PASSIVE found net R falling monotonically in the
offset `b` on all three splits, because a sell limit above the print only fills on events that kept
running up. Both sides, one sentence: **moving the limit away from the print buys fills you do not
want.**

---

## 5. The 24 pre-declared cells — TRAIN and VAL (Bar A). TEST not read.

`pt` charge. MDE = 2.8 × SE. Full table incl. w95/med, ex-top-5% and +3R cap: `cells.csv`.

| # | side | rung | gate | R | hz | TRAIN n | tr/wk | TRAIN mean | t | MDE | VAL mean | t | VAL wks grn | G1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | short | b0 | — | 2% | +5m | 768 | 14.8 | −1.221 | −9.78 | 0.350 | −0.559 | −3.17 | 18% | ✗ |
| 1 | short | b0 | — | 2% | EOD | 768 | 14.8 | −1.278 | −10.49 | 0.341 | −0.683 | −4.18 | 18% | ✗ |
| 2 | short | b0 | — | 6% | +5m | 768 | 14.8 | −0.226 | −4.08 | 0.155 | **+0.067** | 0.81 | 59% | ✗ |
| 3 | short | b0 | — | 6% | EOD | 768 | 14.8 | −0.179 | −2.80 | 0.179 | **+0.042** | 0.45 | 46% | ✗ |
| 4 | short | b0 | 1.25% | 2% | +5m | 245 | 5.0 | −0.601 | −3.18 | 0.530 | −0.296 | −1.12 | 53% | ✗ |
| 5 | short | b0 | 1.25% | 2% | EOD | 245 | 5.0 | −0.641 | −3.42 | 0.524 | −0.327 | −1.26 | 47% | ✗ |
| 6 | short | b0 | 1.25% | 6% | +5m | 245 | 5.0 | −0.039 | −0.44 | 0.247 | **+0.113** | 0.86 | 58% | ✗ |
| **7** | **short** | **b0** | **1.25%** | **6%** | **EOD** | 245 | 5.0 | **−0.011** | −0.10 | 0.296 | **+0.059** | 0.38 | 53% | ✗ |
| 8 | long | d0 | — | 2% | +5m | 673 | 12.7 | −0.856 | −6.05 | 0.396 | −1.262 | −6.55 | 14% | ✗ |
| 9 | long | d0 | — | 2% | EOD | 673 | 12.7 | −1.084 | −8.81 | 0.344 | −1.332 | −7.21 | 9% | ✗ |
| 10 | long | d0 | — | 6% | +5m | 673 | 12.7 | −0.163 | −2.60 | 0.176 | −0.221 | −2.40 | 23% | ✗ |
| **11** | **long** | **d0** | — | **6%** | **EOD** | 673 | 12.7 | **−0.188** | −2.79 | 0.189 | −0.172 | −1.66 | 46% | ✗ |
| 12 | long | d0 | 1.25% | 2% | +5m | 233 | 4.7 | −0.963 | −4.55 | 0.592 | −1.503 | −5.29 | 10% | ✗ |
| 13 | long | d0 | 1.25% | 2% | EOD | 233 | 4.7 | −1.097 | −5.87 | 0.523 | −1.499 | −5.23 | 10% | ✗ |
| 14 | long | d0 | 1.25% | 6% | +5m | 233 | 4.7 | −0.246 | −2.65 | 0.260 | −0.329 | −2.30 | 25% | ✗ |
| 15 | long | d0 | 1.25% | 6% | EOD | 233 | 4.7 | −0.288 | −2.83 | 0.285 | −0.242 | −1.49 | 40% | ✗ |
| 16 | long | −1% | — | 2% | +5m | 554 | 10.7 | −1.355 | −9.01 | 0.421 | −1.594 | −7.67 | 5% | ✗ |
| 17 | long | −1% | — | 2% | EOD | 554 | 10.7 | −1.512 | −11.23 | 0.377 | −1.609 | −7.83 | 5% | ✗ |
| 18 | long | −1% | — | 6% | +5m | 554 | 10.7 | −0.312 | −4.69 | 0.186 | −0.301 | −3.09 | 14% | ✗ |
| 19 | long | −1% | — | 6% | EOD | 554 | 10.7 | −0.359 | −4.99 | 0.202 | −0.237 | −2.11 | 36% | ✗ |
| 20 | long | −2% | — | 2% | +5m | 478 | 9.2 | −1.435 | −9.41 | 0.427 | −1.525 | −6.97 | 5% | ✗ |
| 21 | long | −2% | — | 2% | EOD | 478 | 9.2 | −1.516 | −10.52 | 0.404 | −1.572 | −7.18 | 5% | ✗ |
| 22 | long | −2% | — | 6% | +5m | 478 | 9.2 | −0.418 | −6.25 | 0.187 | −0.198 | −1.94 | 27% | ✗ |
| 23 | long | −2% | — | 6% | EOD | 478 | 9.2 | −0.454 | −5.99 | 0.212 | −0.090 | −0.75 | 55% | ✗ |

**Zero cells clear G1.** Four (2, 3, 6, 7) are positive on VAL and negative on TRAIN; none is positive
on both. Under the `w95` and `med` charges **no decision flips** — cell 7 moves to −0.005 and −0.045
on TRAIN, and the best cell stays negative under all three charges. The mean-vs-median sensitivity
that reopened this book changes the *size* of the loss, never its sign.

Tail tests on the two least-bad cells: #7 TRAIN ex-top-5% −0.123 / cap+3R −0.011; #11 TRAIN −0.305 /
−0.188. Every cell is negative ex-tail on TRAIN.

---

## 6. Multiplicity

* **Within this stage**: search-adjusted permutation on TRAIN, symmetric sign-flip null, B = 2,000,
  max |t| over the 24 cells: **observed 11.23, p = 0.0005** — and, exactly as in PASSIVE, that
  significance belongs to a **losing** cell (#17, long −1% · R 2% · EOD, t −11.2). It says the levers
  move the book reliably, not that an edge exists.
* The honest statistic for a revival is the **best POSITIVE t across the 24 cells: −0.10** — there is
  no positive TRAIN result at all. Permutation p on the best positive t = **1.0000**.
* **Cumulative looks on this population**: 12 (O_halt) + 6 (PASSIVE) + **59 here** (24 gated cells +
  23 TRAIN lever views + 12 sensitivity-arm views) = **77**. Bonferroni at α = 0.05 over 77 needs
  p ≤ 0.00065; BH over the same set needs a positive p-value to rank. Neither is reachable because
  there is no positive TRAIN cell to adjust. In the required terms: **nothing clears after
  adjustment.**

---

## 7. L4 — the EOD swing, EXPLAINED (not just reported)

PASSIVE's unexplained swing (`netR_eod` +0.129 TRAIN vs +2.33 TEST while +5m was −0.84) is **a tail
of five names**, and two of the five are the *same symbol-day counted three times* (three separate
LULD halts on one name — not independent observations):

| no-stop arm | TRAIN | VAL | TEST |
|---|---|---|---|
| +5m mean / ex-top-5% | −0.410 / **−0.921** | +0.431 / **−0.063** | +0.402 / **−0.080** |
| +30m mean / ex-top-5% | −0.266 / **−1.270** | +0.414 / **−0.539** | +0.313 / **−0.576** |
| EOD mean / ex-top-5% | +0.332 / **−1.378** | −0.686 / **−2.441** | +2.189 / **+0.931** |

Top-5 contributors to the EOD arm: TRAIN **+202 of +255 R (80%)** — WOK 2025-12-10, OCG 2025-12-11,
WFF 2025-06-27, SPHL 2025-03-12, SKBL 2025-07-07. VAL **+209 R against a split total of −231 R** —
TDIC 2026-05-14 ×3 (+46.7/+44.3/+43.0, one symbol-day, three halts) and THH 2026-01-16 ×2. TEST
**+143 of +539 R** — BRNX 2026-08-27 ×2, NVVE 2026-07-23 ×2, KXIN 2026-08-27.

So the EOD arm is not a longer-horizon edge; it is **a lottery ticket with an unstable sign** (+0.33 /
−0.69 / +2.19), negative ex-top-5% on two of three splits, and **−3.39 / −4.76 / −1.80 with winners
capped at +3R**. That is the shape this owner has already rejected once. It is also why the +5m-vs-EOD
choice swung so hard between PASSIVE's splits: five names, not a horizon effect.

*(These are the only TEST figures in this report; they restate a column PASSIVE already published and
promote nothing — see `FREEZE.md`.)*

---

## 8. Bar B — the live-exploration bar (owner 9/18: "we can always run if there's an edge")

| condition | verdict |
|---|---|
| **1. positive point estimate** on TRAIN and VAL under the measured cost/fill model | **FAIL — no cell.** Four short cells are VAL-positive and TRAIN-negative; taking them is taking the positive split |
| **2. stated mechanism; ex ante or grid max?** | the four VAL-positive cells are a **grid maximum**, not an ex-ante choice. The mechanism (the reopening print overshoots) is the one PASSIVE already showed lives entirely in the gross, and §7 shows lives in five names |
| **3. bounded downside** | available and written (`PREREG.md` §9c: $100 risk, −$300 daily, −$600 weekly, demote on −$600 in a week or 6 consecutive losers, kill on −$1,000 cumulative or a live-vs-spec parity break) — but there is nothing to bound |
| **4. resolution within one quarter** | **FAIL, independently.** At their **borrow-constrained** frequency the four cells run at 0.93–1.23 trades/week and need **170 / 415 / 870 / 1,369 weeks** to separate their own point estimate from zero at 1 SE. Three to twenty-six **years**, against a 13-week bar |

**Exploration tier: NO, for all 24 cells.** No cell clears Bar B but not Bar A; nothing clears either.
There is no live spec to hand over, because condition 1 fails before 3 and 4 are even reached.

---

## 9. Long vs short capacity — stated plainly, as required

* **A short-side survivor is capacity-dead, whatever the cost treatment.** Alpaca `shortable AND
  easy_to_borrow` (today's flag — the declared survivorship caveat) holds on **8.1%** of the short
  book's TRAIN trades, 15.1% under the 1.25% spread gate. A 15-trades-per-week book becomes
  **0.93–1.23 trades/week** before a single spread is paid. Median capacity at 1% of the
  resume-minute volume is **568 shares / $4,870 notional**; at R = 2% of price, $100 of risk already
  needs $5,000, so even the surviving trades sit at the capacity limit and the owner's $375 is ~3.9×
  beyond it.
* **A long-side survivor would have been instantly 100% tradeable** — no borrow, no locate, no Reg
  SHO — at 12.7 trades/week and median capacity **494 shares / $4,177 notional** (at R = 6% of price
  $100 of risk needs $1,667, so the long book would have supported ~$250/trade before capacity bound).
  **That is exactly why the long side got equal billing here — and it is where the answer is most
  decisive:** every long cell is negative on TRAIN, every long cell is negative on VAL, and the
  pullback rungs that were supposed to be the mechanism are the worst arms in the entire study (§4).

---

## 10. Verdict (PLAN §1 phrasing)

> **No edge was detectable in THIS universe** (Nasdaq-listed LULD halt-resumes, prev close ≥ $5,
> ADV20 ≥ 100K — 49.5% of our tradeable names, zero NYSE/BATS coverage), **on EITHER side** (1,445
> short-book and 1,290 long-book events), **at THESE horizons** (+5m / +30m / EOD), **at THESE
> levels** (R = 2/4/6/8% of price with a 1R stop and a 2R target, and with no stop at all), **at
> THESE limit placements** (short at the print; long at the print and 0.5 / 1 / 2% below it),
> **under THIS selection** (spread gates 1.25 / 0.75 / 0.50 / 0.25%, price ≥ $10 / $20,
> ADV20 ≥ 500K / 1M), **at THIS book size** (every event, no slot competition), **over THIS window**
> (2025-01 → 2026-09), **at THIS cost** (zero on the entry limit; the measured per-trade Alpaca SIP
> NBBO on the exit, reported additionally winsorised at p95 and at the split median).
> **0 of 24 pre-declared cells has a positive TRAIN mean; the best is −0.011 R.** The smallest
> per-trade effect TRAIN could resolve at 80% power is **0.155–0.19 R** on the full-frequency cells
> and **0.25–0.59 R** on the spread-gated ones — so the wide-R full-frequency cells are a **powered**
> rejection of anything at or above ~0.16 R, while the gated cells (n 44–245) stay underpowered and
> their nulls are weak. The verdict does not flip under any of the three spread charges.

**What survives as a true statement:** the LULD reopening print does overshoot — gross +0.32 / +1.16
/ +1.07 R on the no-stop +5m arm. The overshoot is **smaller than the spread you must cross to
collect it**, it is **concentrated in a handful of names** (ex-top-5% negative on TRAIN and ~zero on
VAL/TEST; capped at +3R negative on all three), and **any stop at 2–8% of price costs more than it
saves**, because post-halt one-minute volatility is larger than the move being harvested. Those three
facts are the book, and they do not combine into a trade.

**What the owner was right about:** the closure's grounds were wrong (§1). Both defects flattered the
*negative* case, and both were fixed in the book's favour here. The revival is refused on evidence
that is now stronger and better measured, not on the evidence that was challenged.

---

## 11. Nothing was built, and what a future revival would still need

No config, order, service, cron or cache was touched; nothing is enabled or proposed. The engine gaps
from `O_halt/REPORT.md` §7 were re-verified today and all still hold: **`grep` finds zero occurrences
of `subscribe_trading_statuses` and zero trading-status ingestion anywhere in the repo**, there is no
short-sell entry path in `trading_engine.py`, and **the off-hours Alpaca `statuses` websocket check is
STILL UNRUN** (one market-data websocket per key; the live `onemil-trader` holds it through the
session, so a second connection would knock the live stream off — it must be run off-hours or on a
separate key). Since no cell survives, no engine build list is issued and no independent rebuild is
commissioned; the standing requirement (a second implementation from prose, compared trade by trade
on (day, symbol)) is recorded as still owed if this line is ever reopened on new evidence.

**Data spend: $0.** 6,550 exit-instant and 1,105 entry-instant Alpaca SIP quote pulls on the existing
subscription; no Databento byte.

Artefacts: `PREREG.md`, `FREEZE.md`, `cells.csv`, `levers.csv`, `adverse_selection.csv`,
`sim_rows.parquet`, `entry_nbbo_long.csv`, `exit_nbbo.csv`, `excluded_no_quote.csv`,
`score_summary.json`; scripts `fetch_entry_nbbo_long.py`, `sim2.py`, `fetch_exit_nbbo.py`,
`score3.py` (`score2.py` is the superseded first scorer, kept for the audit trail).
