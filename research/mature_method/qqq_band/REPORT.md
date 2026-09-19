# QQQ noise band (M6) through the mature method — REPORT (2026-09-19)

Candidate #2 of `research/mature_method/RUNBOOK.md`, all ten steps, cells as declared in `PREREG.md`
(committed `f67e74a` **before any cell was scored**). Artifacts: `score.py` → `score.log` + `cells.csv`
+ `nulls.csv`, `cost_nbbo.py` → `cost_nbbo.csv`, `supp.py` → `supp.log`, `supp2.py` → `supp2.log`.
One python process at a time, `nice -n 10`, `ulimit -v 3000000`; `research/lit_review_2026/etf_1min.db`
opened **read-only**. No config, `orb.yaml`, systemd unit, cron, order or cache was written.

---

## VERDICT — **NOT DECIDABLE, AND NO PURCHASABLE DATA DECIDES IT → STAY DEAD IN PRACTICE**

*In THIS universe — QQQ, one ETF, no survivorship exposure at all — at THIS horizon (12 semi-hourly
decisions, VWAP/band stop, flat at the close), at THIS book size (1x $60,000 and 2x Reg-T $120,000),
over 2016-01-04 → 2026-09-15, at THIS cost (**measured** Alpaca SIP NBBO, 0.236 bp/leg on 600 of the
book's own leg instants, 100 % coverage), **the sleeve is gross-POSITIVE in every split and the cost
is not what threatens it — the cost was over-charged 2.1x by our own prior reports.** What kills it
is arithmetic: the smallest effect the 2024-25 window can see is **12.6 bps per traded day against an
observed 5.8**, and resolving the sleeve's own point estimate at 80 % power takes **2,388 sessions =
9.5 years**. No data on disk, and none for sale, shortens that — only forward time does.*

**This is the opposite finding to candidate #1.** HOD-break was gross-flat and paid 0.215 R to trade:
correcting the cost could never save it. Here the gross is real (+7.99 / +6.50 / +5.66 bps per traded
day, TRAIN/VAL/TEST) and the cost is ~nothing (0.236 bp/leg against a 2.01 bp/leg breakeven — **8.5x
headroom**). Separating those two is the whole point of the pass, and for this candidate the answer is
**"the gross is fine and the cost was wrong in our favour, and it still does not matter."**

**And it is not additive.** Week-level correlation with the live ORB+BF path is **+0.004** (B0) /
**-0.134** (H2) — genuinely uncorrelated, as an ETF book at different hours should be. But stacking it
on ORB+BF moves green weeks 60.0 % → 58.9 % (B0 1x) or 60.0 % → 61.1 % (H2 1x, +1 week in 90) while the
worst combined week goes **-$960 → -$1,423 (B0) / -$2,053 (H2 1x) / -$3,535 (H2 2x)** and the combined
drawdown doubles to trebles. Uncorrelated is not the same as additive: this sleeve adds dollars and
variance in roughly equal measure, which is the case the owner named on 9/19 as *not* what he asked for.

Not SHIP-TO-DRY: a dry run for this book would measure almost nothing. The two things it exists to
measure are already measured here — the fill (§4) and the cost (§3) — and the third, market impact, is
bounded below the tick (§3c). Not STAY DEAD as a *claim about the effect*: the in-sample anchor is
real and all three splits are positive; "no edge exists" is not supported and is not claimed.

---

## 1. Reproduction gate — EXACT, on both references

`supp.log` §"REPRODUCTION GATE":

| reference | published | reproduced |
|---|---|---|
| `RESULTS.md` row 1b, QQQ IS 2016-23, paper costs, dyn | 12.8 bps/traded day, t 3.7, 44 % hit | **12.77, t 3.73** |
| `RESULTS.md` row 1b, QQQ OOS 2024→, paper costs, dyn | 10.1 bps/day, t 1.6, SR 0.99 (1x), 8.4 %/yr | **10.14, t 1.59, SR 0.987, 8.35 %** |
| `Q/REPORT.md` §3 scenario C, QQQ IS (live fill, 0.5 bp/leg) | 6.48 bps, t 3.32, SR 1.18, ann 9.85 %, MDD 9.9 % | **6.48, 3.32, 1.178, 9.85, 9.9** |
| `Q/REPORT.md` §3 scenario C, QQQ OOS | 4.57 bps, t 1.27, SR 0.77, ann 6.40 %, MDD 10.6 % | **4.57, 1.27, 0.772, 6.40, 10.6** |

**Independent blind rebuild — agrees to three decimals.** A second agent, forbidden to open `Q/zsim.py`,
`H/QQQ/`, this directory, `test_zarattini_spy.py`, `zarattini_spy.md` or `RESULTS.md`, wrote its own
simulator from the PREREG §1 prose alone:

| | blind rebuild | this study |
|---|---|---|
| gross, IS | 7.992 bps/traded day (t 4.11) | **7.992** |
| at 0.5 bp/leg, IS / OOS | 6.501 / 4.570 | **6.501** / 4.570 (= `Q` exactly) |
| at the measured 0.2364 bp/leg, IS | 7.287 | **7.287** |
| "no stop" variant (cell `H2`), IS | 6.132 | **6.132** |
| `H2` green weeks IS / 2024-25 / 2026 | 54.83 % / 55.24 % / 52.63 % | **54.831 / 55.238 / 52.632** |
| traded sessions IS / OOS | 1,191 / 386 | **1,191 / 285 + 101** |

The only divergence is 2 round trips (2,360 vs 2,362) — both fall on sessions the `sigma`/`sig14`
validity mask excludes, so no statistic moves. This discharges CLAUDE.md's "independent
reimplementation" rule, which `Q/REPORT.md` §6 correctly recorded as still owed. It catches coding
errors and **cannot** catch specification errors; §4 and §12 are where those are hunted.

My own `sim3` (the scorer, which adds four knobs to the published loop) was diffed trade-by-trade
against `zsim.simulate` at the default knobs: **2,362 vs 2,362 trades, identical = True**.

## 2. Gross before net — the book is gross-POSITIVE, and that is the whole difference from #1

No cost at all, per traded day (bps), after the validity mask:

| population | TRAIN (IS 2016-23) | VAL (2024-25) | TEST (2026) |
|---|---|---|---|
| **B0 gross** | **+7.99** (n 1,191) | **+6.50** (n 285) | +5.66 (n 101, the H2 row; B0's TEST stays sealed) |
| B0 net @ measured 0.236 bp/leg | +7.29 (t 3.74) | +5.78 (t 1.28) | — |
| B0 net @ the 0.5 bp ASSUMPTION of `Q`/`H` | +6.50 | +4.97 | — |
| MDE @ 80 % power (traded days) | **5.46** | **12.59** | 19.69 |

Runbook step 2's stop condition (*gross ≤ 0 with the MDE below the effect the book needs*) is **not**
met — the gross is positive in every split and TRAIN is above its own detection floor. The pass
therefore continues, and everything below is the reason it still ends where it does: **VAL's MDE is
2.2x its observed effect.**

## 3. Measured cost — the band was never the right instrument here, and 0.5 bp/leg was 2.1x too much

Alpaca SIP NBBO fetched at the **exact instant of the book's own legs** — 600 legs (300 TRAIN,
300 VAL), first quote at or after the fill second, **100 % coverage** (`cost_nbbo.csv`). TEST legs were
not sampled (sealed).

| | TRAIN | VAL | all |
|---|---|---|---|
| NBBO half-spread, mean | 0.286 bp | 0.186 bp | **0.236 bp** |
| NBBO half-spread, median / p90 | 0.281 | 0.123 | 0.203 / 0.424 |
| **direct**: what a marketable order pays ON TOP of the modelled bar-open fill, mean | +0.150 bp | +0.063 bp | **+0.107 bp** |
| same, median / p90 | +0.062 | +0.164 | +0.162 / +0.935 |

Charged cost in every cell = the **half-spread, 0.236 bp/leg** — the conservative of the two measures
(the direct one is 2.2x smaller). `Q` and `H/QQQ` charged **0.5 bp/leg**, i.e. **2.1x the measured
half-spread and 4.7x the direct measurement**. Correcting it is worth **+0.79 bps/traded day on TRAIN
and +0.81 on VAL** ≈ **+$100/month at $60 K** — real, and nowhere near enough.

**(a) Auction or intraday?** Every one of the 12 decisions is an intraday market order at 10:00…15:30
filling at the next bar's open; **only the final flat could ever be an auction print**. So the
"auction fills pay no quoted spread" carve-out of runbook step 3 applies to at most one leg of one
trade a day, and cell `H4` prices even that at ~zero (§6).

**(b) Order types — DAY is the only TIF in the repo, and it does NOT block this book.**
`data_sources/alpaca_client.py` uses `TimeInForce.DAY` at all **7** submit sites and never `CLS`/`OPG`
(a repo-wide grep finds `TimeInForce.DAY` 7x and nothing else). There is also **no plain market-order
helper at all** — the six submit functions are bracket, stop-bracket, stop-limit, stop-sell, limit-buy,
limit-sell. Consequences, stated plainly: the published construction's *entries and stops* need only a
marketable DAY limit, which `submit_limit_buy_order` / `submit_limit_sell_order` already are; the
*closing* leg would ideally be an MOC (`TimeInForce.CLS`), which this repo cannot place — and cell `H4`
replaces it with a market order at 15:58 filling at the 15:59 open and costs **nothing**
(TRAIN 7.29 → 7.01, VAL 5.78 → **5.88**, i.e. VAL is marginally *better*). **The missing MOC does not
block the book.** What a dry run would have to build is a market-order path and a 12-times-a-day timer,
not a new order type.

**(c) Market impact, measured, not asserted.** QQQ displayed top of book at 40 random VAL leg instants:
median ask depth **$365,151**, p10 **$83,655**, median price $517. A $60 K order is 16 % of the median
displayed top of book; a $120 K (2x Reg-T) order is 33 % and sits inside it on ~90 % of instants. One
QQQ tick is $0.01 = **0.19 bp**, so even a full one-level walk costs less than the half-spread. Against
`Q`'s measured **breakeven of 2.01 bp/leg OOS**, the book has ~8.5x headroom on the measured cost and
~10x on a one-tick walk. **Cost and impact are settled questions for this candidate.**

## 4. The engine's real fill model, and obtainability

The fill is the live convention established by `Q`: decide on a CLOSED bar, market order, fill at the
**next bar's open**. There is no limit-at-our-price population here and therefore no unfilled
counterfactual to classify — **every signal fills; non-fill is not a state this book has**. The
relevant realism tests instead are:

- **Obtainability**: every fill is the open print of a bar, i.e. a price the market actually traded
  (`Q` §3 verified `low <= fill <= high` on 2,362/2,362 entries). The direct NBBO measurement in §3 is
  the stronger version of the same test and says the ask at that instant was on average **+0.107 bp**
  away from the modelled fill.
- **The paper's own fill (the check bar's close) is a <= 60-second look-ahead**, and replacing it with
  the live convention costs **0.01 bps/traded day** (`Q` §3: 5.84 → 5.83 OOS). At a 30-minute cadence
  the fill convention is irrelevant; the 82 % of legs where the two prices differ differ by a median of
  0.000 bp.
- **A resting limit AT the band is not available.** It would fill on every crossing minute, i.e. at all
  the minutes the rule deliberately does not trade, which converts the book into the every-minute
  variant that is dead (`T3`/`C3`, §6). The cheap fill and the semi-hourly cadence are the same choice.

## 5. Gate-separation map — there is no cascade, and saying so is the finding

This candidate has **no gate cascade**. Nothing is filtered, nothing is ranked, no slot is consumed;
the "gates" are four *shape* knobs — the band multiplier, the band anchor, the decision cadence /
time-of-day window, and the hold rule. A leave-one-out kept-minus-rejected table is not defined for
them, because turning a knob does not partition one population into kept and rejected: it produces a
different book with different trades on the same days. The runbook's step 5 therefore collapses into
step 6, and the ladder in §6 **is** the gate map. Reported in its place, since the sizer cannot hide
anything here either: sizing is 1x notional on every trade, identical in every cell, so there is no
position-size channel for a knob to act through.

## 6 / 7. Frequency frontier, ranked on % GREEN WEEKS (dollars at $60 K unlevered and 2x Reg-T)

18 declared cells. A week with no traded day counts **FLAT** and stays in the denominator (TRAIN 414
weeks, VAL 105, TEST 38). Net at the measured 0.236 bp/leg. `$/wk` = split total / split weeks.

| cell | knob | tr/wk | TRAIN green % | VAL green % | TRAIN net bps (t) | VAL net bps (t) | TRAIN $ tot | VAL $ tot | VAL $/wk 1x | VAL $/wk 2x | VAL worst wk 1x | VAL MDD 1x |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **H2** | **no stop, hold to flat** | 2.88 | 54.8 | **55.2** | +6.13 (2.37) | +6.11 (1.04) | 43,817 | 10,453 | **+100** | **+199** | -2,068 | 4,698 |
| T2 | cadence 60 min | 3.21 | 53.6 | 53.3 | +8.15 (**3.94**) | +7.39 (1.51) | 53,403 | **11,476** | +109 | +219 | -1,561 | 4,228 |
| H4 | flat at the 15:59 OPEN | 4.29 | 56.0 | 52.4 | +7.01 (3.61) | +5.88 (1.31) | 50,072 | 10,054 | +96 | +192 | -1,613 | 4,302 |
| T1 | cadence 15 min | 6.37 | 53.9 | 52.4 | +6.08 (3.17) | +3.37 (0.65) | 48,782 | 6,296 | +60 | +120 | -1,990 | 6,064 |
| H3 | flat at 15:30 | 4.08 | 55.8 | 51.4 | +6.52 (3.56) | +5.50 (1.31) | 45,746 | 9,309 | +89 | +177 | -1,387 | 4,268 |
| V2 | VM 1.2 | 3.60 | 53.1 | 48.6 | +6.35 (3.19) | **+7.74 (1.61)** | 38,563 | 11,241 | +107 | +214 | **-983** | 4,089 |
| **B0 shipped** | — | 4.29 | **56.5** | 47.6 | +7.29 (3.74) | +5.78 (1.28) | 52,073 | 9,877 | +94 | +188 | -1,690 | 4,547 |
| V1 | VM 0.8 | 5.22 | 56.5 | 47.6 | +6.70 (3.49) | +2.37 (0.55) | 55,817 | 4,657 | +44 | +89 | -2,025 | 6,691 |
| A1 | anchor = open only | 6.10 | 54.1 | 47.6 | +4.78 (2.54) | +3.51 (0.91) | 44,785 | 8,229 | +78 | +157 | -1,751 | 5,268 |
| W2 | entries <= 13:30 | 3.48 | 51.7 | 47.6 | +5.51 (2.79) | +7.59 (1.56) | 36,240 | 11,470 | +109 | +219 | -1,536 | 3,120 |
| C3 | **ceiling**: 1-min + VM 0.8 | **33.20** | 47.8 | 47.6 | +2.15 (1.26) | +0.97 (0.24) | 20,461 | 2,232 | +21 | +43 | -1,234 | 6,083 |
| C2 | VM 1.2 + 15:59 open | 3.60 | 52.9 | 46.7 | +6.16 (3.12) | +7.83 (1.63) | 37,437 | 11,375 | +108 | +217 | -977 | 4,097 |
| H1 | stop = opposite band only | 4.04 | 56.5 | 46.7 | +7.61 (3.85) | +5.74 (1.28) | 54,355 | 9,807 | +93 | +187 | -1,690 | 4,253 |
| W1 | entries <= 12:00 | 2.80 | 50.0 | 45.7 | +5.36 (2.66) | +3.61 (0.90) | 31,688 | 4,811 | +46 | +92 | -1,299 | 2,934 |
| T3 | cadence 1 min | 26.34 | 48.3 | 45.7 | +2.12 (1.25) | +0.66 (0.15) | 18,159 | 1,324 | +13 | +25 | -1,477 | 6,890 |
| C1 | VM 1.2 + entries <= 12:00 | 2.30 | 44.7 | 45.7 | +3.86 (1.90) | +3.86 (0.90) | 19,312 | 4,448 | +42 | +85 | -1,663 | 2,933 |
| W3 | first decision 11:00 | 3.47 | 54.3 | 43.8 | +7.12 (3.58) | +6.66 (1.29) | 43,975 | 10,103 | +96 | +193 | -847 | 3,151 |
| V3 | VM 1.5 | 2.67 | 46.1 | 39.0 | +6.79 (2.99) | +5.64 (1.25) | 31,346 | 6,430 | +61 | +122 | -1,186 | 3,018 |

**The structural ceiling**, as the runbook requires: every-minute checks with the widest-firing band
(`C3`) gives **33.2 trades a week — 7.7x the shipped rate — and the gross there is +6.23 (TRAIN) /
+5.05 (VAL)**, i.e. the extra 29 trades a week are each worth a fraction of the shipped ones; net falls
to +2.15 / +0.97 and, at the old 0.5 bp assumption, goes **negative (-2.40 / -3.58)**. This reproduces
the July ablation C independently: **the semi-hourly cadence is the load-bearing ingredient, and the
frontier falls monotonically with frequency — but here, unlike HOD-break, it falls on the GROSS, not
on the cost.** More looks at the same bands is not more edge; it is the same edge sliced thinner than
the noise.

**The frontier does not bend anywhere.** Dollars at 1x peak at ~$110/week (VAL) and the green-week
share never exceeds 55.2 % on VAL or 56.5 % on TRAIN. At 2x Reg-T the whole frontier is $25-220/week.
Against the owner's standing target (30-50 %/yr delivered monthly on a $60 K account = $1,500-2,500 a
month) the best cell delivers **$430-870 a month at 2x**, in the best of three windows.

## 8. Week-by-week at live sizing

`B0` and `H2`, `$` at $60 K unlevered (double for 2x). Calendar-2025 (the recent half of VAL):

```
B0  +778  -82 -163 -257 +518  -63 +231 -328   -5 -306 -510 +1170 +1159 +4451    0 +294
    -490  -75 +412 -587 -224 +122  -61 -494  -93  -53 -644 -166 -478 -581 +518  -11
    +337 +137 +119    0 -767 -991 -342 +1844 -1690 +421  -16 +602 +210 -978 +566 -301
H2  +547 +118 -145 -860 +666 +152 +438 +1489 -389 +660 -949 +1225 +1624 +2358    0  +50
   -2068 -510 +412 -407 +201 -739 -176 -506 +367 +149 -606  -20 -527  -55 +746 +177
    +436 +137 +119    0 -208 -279 -656 +1924 -929 +432 +395 +193 +541 -519 +566 -323
```
`B0` 2025: **18 green of 48 (37.5 %), +$3,136, worst -$1,690.** `H2` 2025: **26 of 48 (54.2 %),
+$5,248, worst -$2,068.** One week (2025-04-11, the tariff-pause reversal) is +$4,451 of B0's +$3,136
— **without it calendar-2025 is negative**, which is §10's tail result in the dollar path the runbook
demands. The last two quarters (2026 Q2-Q3, the sealed window — see `FREEZE.md` §2.4) run
`B0` 9 green of 24, +$2,332; `H2` 12 of 24, +$1,510.

## 9. The one authorised TEST read (`FREEZE.md` §1)

G1 (TRAIN net > 0, t >= 2) passes on **15 of 18** cells. G2 (VAL same sign **and** >= 55 % green weeks)
passes on **exactly one — `H2`, 55.24 %**. TEST was therefore opened for `H2` and for nothing else:

| `H2` | TEST 2026-01-01 → 2026-09-15 |
|---|---|
| traded days / weeks | 101 / 38 |
| gross / net bps per traded day | +5.66 / **+5.18** |
| t / MDE @ 80 % | **0.74** / 19.69 |
| **green weeks** | **52.6 %** (below the 55 % bar) |
| longest red streak / worst week | 4 weeks / -$1,482 |
| total / MDD at $60 K 1x | **+$3,141** ($83/wk; $165/wk at 2x) / $3,271 |
| green months | 66.7 % |
| count-matched null | obs 52.6 % vs null 52.9 % [44.7, 60.5] — **inside** |

Read plainly: **the sealed split agrees in SIGN and in dollars and does not confirm the week shape.**
It is the third consecutive positive window for this sleeve and the third that cannot resolve itself.

## 10. Count-matched permutation null (2,000 draws, per-week traded-day count held fixed)

| | TRAIN | VAL |
|---|---|---|
| cells ABOVE their p95 | **11 of 18** | **0 of 18** |
| cells BELOW their p5 | 0 | 0 |
| `B0` observed vs null | 56.5 % vs 51.9 % [49.5, 54.6] — **ABOVE** | 47.6 % vs 47.4 % [42.9, 52.4] — inside |
| `H2` observed vs null | 54.8 % vs 55.6 % [53.1, 58.0] — inside | 55.2 % vs 53.1 % [48.6, 58.1] — inside |
| `T2` observed vs null | 53.6 % vs 51.9 % [49.5, 54.4] — inside | 53.3 % vs 48.6 % [43.8, 53.3] — inside (at p95) |

**The answer to the null question is clean and it is a negative.** In-sample the sleeve's daily P&L is
spread across weeks better than chance in 11 of 18 cells; **out of sample not one cell of eighteen sits
outside its own band, and the cell that passed the claim bar is inside on all three splits.** Green
weeks on this book, out of sample, are bought with trade count — the same answer `orb_gates2` and the
HOD-break pass reached on entirely different books.

**Tail** (`B0`, net, per calendar day — reported as a diagnostic, never a rejection reason):

| | TRAIN | VAL |
|---|---|---|
| full | +4.35 bps (+$548/mo) | +3.28 bps (+$413/mo) |
| ex-top-1 % of days | +1.77 (+$222) | **-0.45 (-$56)** |
| ex-top-5 % | -3.74 (-$471) | -5.15 (-$648) |
| capped at +100 bp/day | +1.46 (+$184) | **+0.21 (+$26)** |
| top-5 days as a share of the split total | **21 %** | **104 %** |

In-sample the edge is broad; out of sample it is five days. This is `Q` §5's finding re-derived on the
narrower VAL window, and it is the shape the owner has already rejected once.

## 11. Additivity — uncorrelated, and still not additive on the owner's metric

90 W-FRI weeks, 2025-01-02 → 2026-09-15. Live = ORB (`orb_gates2/book_G3_meas.csv`, `_sized_pnl`,
stage size) + BF (`bf_frequency/runs/VOL_OFF.csv`, `pnl x 0.075`). Sleeve $ at $60 K 1x and 2x.

| book | green % | total $ | worst wk $ | longest red streak | MDD $ |
|---|---|---|---|---|---|
| ORB alone | 50.0 | +12,592 | -895 | 5 | 2,581 |
| BF alone | 44.4 | +11,323 | -982 | 5 | 1,213 |
| **LIVE = ORB + BF** | **60.0** | **+23,915** | **-960** | **5** | **1,969** |
| B0 sleeve 1x | 37.8 | +5,686 | -1,690 | 8 | 5,886 |
| LIVE + B0 1x | **58.9** | +29,601 | **-1,423** | 7 | 2,520 |
| LIVE + B0 2x | **55.6** | +35,286 | **-2,414** | 7 | 5,752 |
| H2 sleeve 1x | 53.3 | +8,584 | -2,068 | 4 | 4,485 |
| **LIVE + H2 1x** | **61.1** | +32,499 | **-2,053** | **4** | 4,162 |
| LIVE + H2 2x | 60.0 | +41,083 | **-3,535** | 4 | 6,907 |
| LIVE + T2 1x / 2x | 58.9 / 53.3 | +30,199 / +36,483 | -1,723 / -2,875 | 5 | 2,492 / 5,456 |

**Correlation of weekly dollars: B0 ~ LIVE +0.004 · H2 ~ LIVE -0.134 · B0 ~ ORB +0.057 · H2 ~ BF
-0.234** (ORB ~ BF is +0.024 for scale). The sleeve is genuinely uncorrelated with both live books —
different instrument, different hours, exactly as the runbook expected.

**And it fails the pre-committed additivity criterion anyway** (PREREG §6.4: *raise combined green-week
% AND not worsen the worst combined week*):

| | B0 1x | B0 2x | H2 1x | H2 2x |
|---|---|---|---|---|
| live-RED weeks (36) turned green | 9 | 9 | 12 | 14 |
| live-GREEN weeks (54) turned red | 10 | 13 | 11 | 14 |
| **net green weeks** | **-1** | **-4** | **+1** | **0** |
| worst combined week | 1.5x worse | 2.5x worse | 2.1x worse | 3.7x worse |

The mechanism is arithmetic and worth stating because it generalises: a sleeve helps a 60 %-green book
only if its own green-week rate is *above* the host's and its weekly dispersion is *comparable*. This
sleeve is 37.8 % (B0) / 53.3 % (H2) green over the window, with a weekly standard deviation 2-4x the
live books' — so every red week it rescues, it pays for with a green week it drags under. **A sleeve
that merely adds variance is exactly what it is.**

## 12. Both bars, and the adequacy review answered in writing

**Claim bar — passed by one cell of eighteen, and the sealed split did not confirm it.** G1: 15/18.
G2: 1/18 (`H2`). TEST (`H2`): positive sign, +$3,141, t 0.74, **52.6 % green weeks — below the 55 %
bar it had to clear**, inside its null band. On the pre-committed reading the claim is **not
established**.

**Live-exploration bar — fails on (c) and (d).**
(a) positive point estimate on green weeks *and* dollars at live size in both read splits: `H2` passes
(54.8/55.2 % green; +$43,817 / +$10,453). (b) mechanism: **yes, and it is a good one** — `H/QQQ`
§1 measured that 100 % of the sleeve's loss is the stop leg (1,044 stop exits at -19.12 bps against
517 end-of-day exits at +51.21), so deleting the VWAP/band trail is the mechanically indicated change,
not a fitted one. (c) bounded downside with a pre-committed stop: **`H2` has no stop at all** — its
TRAIN MDD is $12,202 on $60 K = **20.3 % of equity at 1x, ~41 % at 2x**, against B0's 9.9 %. There is
no stop to pre-commit. **Fails.** (d) resolution inside a quarter at the book's own frequency:
`supp.log` — to resolve `B0`'s own VAL point estimate (3.28 bps/calendar day, sd 57.2) at 80 % power
needs **2,388 sessions = 9.5 years**; `H2` needs **3,650 sessions = 14.5 years**. A quarter is 63
sessions. **Fails by 38-58x.**

**Adequacy review.**

1. *Did we test what the book actually IS?* Yes. The paper's rule at the live fill convention,
   reproduced to the printed digit on two independent references and rebuilt blind by a second agent
   to three decimals. Availability audit (`score.log`): 2,688 sessions 2016-01-04→2026-09-15, **2**
   dropped for < 150 bars, **0** splits (QQQ never split in sample), missing `sigma[k=30]` 10,
   `sig14` 15, `prevclose` 1, `VWAP[k=30]` 0, `C[k=30]` 0; 2,661 of 2,688 sessions have all 390 bars,
   median 390. There is **no universe and no survivorship exposure** — one symbol, listed throughout —
   which makes this the cleanest population in the whole programme and removes the failure mode that
   killed four other claims this month. Price-scale: one vendor, one tape, no daily-vs-intraday
   crossing anywhere in the rule, and QQQ has no split in the sample (`n_split = 0`), so the hazard
   that `Q` §4 had to handle for TQQQ does not exist here.
2. *Is the cost and fill model right for its venue?* **Now yes, and it was not before.** Per-leg SIP
   NBBO at 600 of the book's own decision instants at 100 % coverage; the 0.5 bp/leg our prior reports
   charged was 2.1x the measured half-spread. Impact is measured against displayed depth and bounded
   below one tick. The fill is the engine's own next-bar-open convention and the paper's <= 60-second
   look-ahead is worth 0.01 bps/day.
3. *Does any caveat in our own report explain the headline?* Yes, and it is §10's: **VAL's top five
   days are 104 % of the split total, and 2025-04-09 alone was 45 % of `Q`'s whole OOS**. The positive
   headline in 2024-26 IS the tail. Every other caveat pushes the same way (the sealed-split read is
   inside its null; the cost correction, the only thing that moved in the book's favour, is worth
   $100/month).
4. *What is the MDE?* Per traded day: **5.46 bps TRAIN, 12.59 VAL, 19.69 TEST** against observed 7.29 /
   5.78 / 5.18. On the green-week share the count-matched band is **+/-3 pp TRAIN, +/-5 pp VAL, +/-8 pp
   TEST** — which is why no two cells in §6 are separable on the primary metric, and why that is itself
   part of the answer.

**Multiplicity.** 18 declared cells x 2 read splits = 36, plus 1 TEST read, plus the null draws and the
descriptive tables. Prior programme count on this sleeve: 36 (`Q`) + 30 (`H/QQQ`) = **66 variants / 89
cell-instances**; this stage takes it to **84 variants / 125 cell-instances**. Sidak-adjusted alpha at
84 variants is 0.0006; the best VAL t anywhere is **1.63** (p ~ 0.10) and the best TRAIN t is 3.94
(p ~ 8e-5, still short of 0.0006 after 84 looks). **Nothing in this sleeve survives its own multiplicity
count**, which is the same conclusion `H/QQQ` §4 reached at 30 variants.

---

## What would change this verdict, and what it would cost

**Nothing on disk, and nothing for sale.** The binding constraint is effect size against daily
variance, and the three things one could buy do not touch it:

| candidate purchase | what it buys | verdict |
|---|---|---|
| Databento QQQ 1-min 2007-2015 (pre-Alpaca history) | ~2,200 more IS sessions at ~$0.0004/symbol-day = **~$1**, plus the EQUS.SUMMARY daily file already on disk | **answers the wrong question.** IS already passes at t 3.7; the open question is post-2024 persistence, and no amount of older data speaks to it |
| Databento MBP-1 / TBBO for QQQ at the decision minutes | a per-leg impact model better than §3c | **redundant.** Cost has 8.5x headroom and impact is bounded below one tick at both sizes |
| a dry run | slippage, decision parity, latency | **near-worthless here.** §3 measured the cost from the real tape and §3c bounds impact; `Q` §8 already computed that 20 sessions carry a standard error of 12 bps/day against an effect of 2.6-4.6. A dry run of this book measures a number we already have and cannot see the one we do not |
| **forward sessions** | the only thing that resolves it | **9.5 years** at 1x on B0, 14.5 on H2 |

The one structural change that could re-open it is a **different instrument mix**: the sleeve's problem
is n, and the rule is symbol-agnostic. Running the same bands on a basket of 8-15 uncorrelated liquid
ETFs (the store already holds SPY, IWM, DIA, SOXL, SQQQ, UVXY, TQQQ) multiplies the daily n without
touching the rule — `H/QQQ` §7 named this as "the only way to answer it rather than re-cutting these
2,124 days". That is a **new pre-registration**, not a continuation of this one, and SPY's own OOS book
is already negative (`Q` §3: -1.86 bps/traded day), so the prior is not good.

**Recommended action: NONE.** No config, no flag, no engine, no dry run. This sleeve stays where the
July programme left it — with its cost claim corrected in our favour, its gross confirmed positive by a
blind rebuild, its week shape shown to be trade count rather than skill out of sample, and its
additivity to the live books measured and found to be variance.
