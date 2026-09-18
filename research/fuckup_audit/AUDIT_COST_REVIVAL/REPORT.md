# AUDIT — which CLOSED books were killed by a COST or FILL MODELLING error, not by the market

**Run 2026-09-18. Read-only audit of existing artifacts.** No backtest was run, no data bought, no config,
cache, order, service or cron touched. `nice -n 15`, markdown only, two heavy agents already on the box.
Every figure below is either quoted from a stage's own `REPORT.md` / `LOG.md` entry, or is arithmetic on
per-cell aggregates those reports already publish — every line of the latter kind is marked **[derived]**.
The one computation performed is a re-read of `A/a0_cells.csv`, the 52-cell table Stage A itself produced
under six cost contracts including a zero-cost one. That is a restatement of published aggregates, not a
re-simulation.

Owner's question (9/18): *"does it revive other strats and cells that you fucking killed?"*

---

## 0. The answer in one page

**No cell is REVIVABLE, and the reason is that the correction was already made — twice, on the two grids
where it mattered, before this audit existed.**

* The largest cost defect in the program — a band table **3.8x too wide** — was found and fixed on
  **2026-09-16 at Stage A**. It was worth **+0.412 R/trade** across the 52-cell `bf_zero2/score4` grid and it
  flipped **9 of 52** TRAIN cells from negative to positive. **0 of 52 then cleared the gate** (best t 1.56
  against a bar of 2.0 and a 52-cell noise ceiling of ~2.7). Every stage from B onward already runs the
  corrected contract (c).
* **E1 on top of that is small and I have now sized it exactly.** Zeroing the entry leg entirely across all
  52 cells — the most generous reading of "a capped limit pays nothing" — takes **12 of 52** cells positive
  and the best t to **1.68**. Still under 2.0, still 1 unit under the noise ceiling. **[derived]**
* **E2 is real, is confined to three stages (K, N2, R_daily), is worth 43-53 bps of round trip — and all
  three stages already computed and published the corrected arm.** `K/build_k.py::cost_rt` carries
  `AUCTION_RT_BPS = 10.0` in the same function as the over-charge. Under it, 4 of 20 R_daily cells turn
  positive on TRAIN and **all four are negative on VAL — gross**, i.e. before a cent of cost.
* **E3 points the wrong way for a revival on our own populations.** Stage P measured the band at 1.92x the
  measured *median* but **1.03x the measured mean**, and a book's mean net R owes the **mean**. Stage P's own
  outstanding fix is therefore to charge the cell MEAN, which makes every past number **more** conservative.
  The one place the band constant manufactured a result rather than killing one is `O_halt`, where 0.40% was
  charged to a cohort quoting 1.9% — the correction **killed** a shipped +0.53/+1.00/+0.90 R headline.

**Classification counts (families / cell-groups, §1 table): REVIVABLE 0 · MARGINAL 9 · STAYS DEAD 21.**

The single honest positive finding: three stages' closures should be restated in the language of §4
("indistinguishable from zero at this power", not "clearly negative"), and one stage — `O_halt/PASSIVE` —
rests its "powered rejection" on a **mean** spread drawn from a distribution with a 474% maximum, where the
**median** arm is a statistical null, not a rejection.

---

## 1. The master table

Ship bars, throughout: **G1** = TRAIN mean net R > 0 AND t >= 2.0 AND >= 5 trades/week. **G2** = VAL mean > 0,
t >= 1.0, >= 55% weeks green. Stages that used a different bar have it named in the row.

Corrected-net estimate convention, stated once so it can be checked:
`corrected_net = net_as_reported + charge x entry_share`, where `charge = gross - net` and
`entry_share = ENTRY_COEF / (ENTRY_COEF + exit_coef)` with contract (c)'s legs
(ENTRY 0.25 next-open / 1.00 resting; stop 0.875, eod 0.412, target 0.875). This is the **maximum** E1 credit
— it assumes a capped limit pays literally nothing at entry. Stage Q measured that it does not
(85.6% marketable at the trigger, filling at `min(ask, cap)`, median 11.9 bps over the level), so every
"corrected net" below is an upper bound, and contract (c)'s 0.25x is approximately the honest number.

| # | stage / cell-group | GROSS | NET as reported | cost model | entry convention assumed | corrected-net estimate | class | gap to that stage's bar |
|---|---|---|---|---|---|---|---|---|
| 1 | **bf_zero2 `score4`, 52 cells** (`REPORT.md` L45-48) | best **+0.18 R** (F1 P=12%) | best **+0.007 R** (F6), 0 of 52 pass | 0.40%/banded 190/120/80/60/50 bps, **ENTRY 1.00 x half**, first-5-min spread applied all day | next bar's open under `level x 1.006`; non-fill = row dropped, no slot penalty | superseded by rows 2-3 | **MARGINAL** | superseded |
| 2 | **Stage A re-score of the same 52 cells, contract (c)** | F6 {} hold **+0.084 (t 2.21)**; F6 {} 2r +0.061 (t 2.37); F1 P=.12 hold +0.344 (t 2.03) | **9 of 52 positive**, best F2 P=.12 +0.427 (t 1.43); F6 {} hold +0.051 (t 1.34) | corrected: measured NBBO per (price x hour) 22-76 bps, ENTRY **0.25** x half | same capped next-open | **12 of 52 positive, best t 1.68** (F1 P=.12 +0.285) **[derived]** | **MARGINAL** | best t 1.68 vs **2.0**; 52-cell noise ceiling ~2.7. Gap **-0.32 t** to G1, **-1.0 t** to noise |
| 2b | ...the only cells that clear t>=2 at ZERO cost: **F6 {} hold / 2r** | TRAIN +0.084 / +0.061 (t 2.21 / 2.37), VAL +0.197 / +0.106 (t 2.99 / 2.60) | +0.051 / +0.028 | as above | as above | zero-cost is the ceiling | **STAYS DEAD** | **TEST gross is -0.043 / -0.008.** The one family that clears the TRAIN+VAL bar at zero cost is gross-**negative** out of sample. No cost model can fix that |
| 3 | **Stage C, 230 gate cells on `candidates4`** | whole grid **-0.016 R**; best `F6 {} next 2R stop-1% >=10:00` **+0.049** | grid **-0.073**; best **+0.031 (t 1.43)**, VAL +0.063 (t 1.93) | contract (c); ENTRY 0.25 next-open / **1.00 resting** | next-open under the 0.6% cap (ref) vs resting stop-limit at the level | best cell **+0.036** **[derived]**; at literal zero cost +0.049 | **MARGINAL** | G1 needs **0.044 R**; corrected +0.036 = gap **-0.008 R**. At zero cost it clears the mean but t ~2.26 vs a 107-cell permutation p95 of **3.88** |
| 3b | Stage C **resting-fill cells** — the purest E1 in the tree | F1 rest hold **+0.281** TRAIN / +0.051 VAL | +0.184 / **-0.054** | **ENTRY 1.00 x half on a capped stop-limit** | resting limit, obtainability 1.000 | +0.245 / **+0.012** **[derived]** | **MARGINAL** | Stage C prices the E1 error itself at **~0.032 R/trade** (45% of the -0.058 paired gap). But the resting fill is also **gross-worse by -0.026 R paired (t -26.5)**, and its queue check passes for 1 of 9 families vs a pre-registered 95% |
| 4 | **Stage D (D0), 18 cells** | **no gross column exists** | best TRAIN +0.054 (t 1.16); 0 of 18 | contract (c) | next-open under the cap | not computable | **STAYS DEAD** | killed by the **reversed-tape gate** (the winning model is profitable on an inverted tape: +0.187 R, t 4.39) and by TEST negative on all five. Not a cost story |
| 5 | **Stage D1, 48 cells** | selected book gross +0.442 vs all-candidate +0.085 | contaminated VAL +0.415 (t 5.10); clean G1 **0 of 48** | contract (c); charge measured at **0.017-0.033 R, flat across selections** | next-open under the cap | unchanged | **STAYS DEAD** | the report's own sentence: *"the cost term is NOT the leak"* — 0.02 R against MDEs of 0.12-0.26 R. Killed by an **availability look-ahead** (`pm_bars.db` keyed to another stage's key set) |
| 6 | **Stage E, 54 cells, causal universe** | F6 {} **rest**/hold **+0.139**; F6 {} next/hold +0.1005; F8 N=15 next/hold +0.070 | **-0.0207 / +0.0177 / +0.0008**; best cell +0.018 at **t 0.15** | contract (c) | next-open under 0.6% cap AND resting stop-limit | rest/hold **+0.079**; next/hold **+0.042** **[derived]** | **MARGINAL** | E's own MDE is **0.083-0.154 R**, so the corrected +0.079 sits **at or under its own detection floor**. Permutation: observed max t 0.15 vs null p95 **3.81**, p = 1.000 |
| 7 | **Stage G, 24 short cells** | S5 UB 1030 **+0.108**; S1 UB hold **+0.057**; S2 +0.032 | +0.0444 (2.6 tr/wk) / +0.0261 (t 0.65) | contract (c) mirrored + borrow | next bar's open, gap through the cap = non-fill | S5 **+0.063**; S1 **+0.035** **[derived]** | **MARGINAL** | S5 fails the **>= 5 trades/week** bar structurally (2.6/wk) regardless of cost. S1 corrected +0.035 vs an MDE of **0.056**. Permutation max t 0.65 vs null p95 3.53 |
| 8 | **Stage H / F5 (the HOD-break book)** | stacked book at **zero cost +0.105 R**; 4 of 6 populations gross-positive on a split | booked **+0.0102**; cost charged **0.092-0.095 R** | contract (c) | next-open under the cap | **+0.038** **[derived]** | **STAYS DEAD** | VAL hold stack +0.0032 (t 0.03) fails the stage's own rule at ANY cost; and `bf_zero/REPORT.md` §6b **voids the population** (the cache = days a scanner had already flagged as movers) |
| 9 | **Stage H / F6 red-to-green** | Q + frozen stack **+0.1685** | **+0.1420** (charge ~0.030 R) | contract (c) | next-open under the cap | +0.150 **[derived]** | **STAYS DEAD** | killed at VAL (stack +0.088 vs unfiltered +0.209; the vetoed cohort **earns +0.2285 R on VAL**). Cost is 0.03 R against an MDE of 0.175-0.209 R |
| 10 | **H / F6_rebuild + F6_reconcile** | rebuild TEST 2R **+0.0131**, partial +0.0192 | TEST -0.022 / -0.016 (charge 0.030-0.033) | contract (c) | next printed bar's open under the cap; **28.5% of signals dropped to the cap, uncharged** — correct E1 handling | TEST ~+0.010 **[derived]** | **STAYS DEAD** | three **non-cost** defects settle it: `ZVZZT` the NASDAQ test ticker (+46.76 R on one synthetic day, +50.80 R of a +39.00 R TEST), `BFZ_SLIP=0` (level at the prior close, not x1.003; 308 of 339 TEST-only candidates, +56.05 R), and the scan rule (first-break vs the engine's keep-scanning, worth +0.09/+0.23 R = the whole positive result). Under the engine's own rule: **-0.027 / -0.012 / -0.102 R** |
| 11 | **H / F14_F8_F11, 83 TRAIN book cells** | **no gross column printed** | best stack F14 +0.1398 (t 2.41) | contract (c) | next-open under the cap | not computable | **STAYS DEAD** | permutation: max \|t\| **2.44 over 83 cells vs a null MEAN of 3.04** and p95 3.85, p = 0.932 — the best cell is below what noise produces. Cost-independent |
| 12 | **H / QQQ + Stage Q (the noise-band sleeve)** | OOS **6.08 bps/day** zero-cost | 4.57 bps/day at 0.5 bp/leg | flat **0.5 bp/leg**, ~6x QQQ's ~0.09 bp quoted half spread. Contains an **E2** MOC leg (`zsim.py:18`, "proxy for the closing auction print") charged 0.5 bp | decide on the closed bar, market at the next bar's open; median next-open deviation **0.0 bp** | zero-cost 6.08 | **STAYS DEAD** | the stage's MDE is **5.75 bps/day** — the ZERO-cost result is inside it. And top 5 days = **102% of OOS return**; capping daily return at +1% gives +0.03 bps/day. Cost is ~6x conservative and irrelevant |
| 13 | **Stage I (stacking + the frozen F6 book)** | S0 hold TRAIN/VAL/TEST **+0.106 / +0.281 / +0.138** | +0.078 / +0.251 / +0.104 (charge 0.028-0.034) | contract (c); a **x1.5 spread sensitivity moves no sign** ($200-320/month) | next-open, obtainability 100.00% | +0.086 / +0.259 / +0.114 **[derived]** | **STAYS DEAD** | killed by **capacity**, not spread: at a 1% participation cap and $300 risk the book goes +$1,274 / +$1,877 / **-$3,229** per month. *"The liquidity cap is not a haircut on this book — it is a different book."* |
| 14 | **Stage J, 144 cells, liquid universe U3** | not printed; **[derived]** best cell gross ~**+0.219** from the published 0.143 R median round trip | best `F14 N=15 hold PDR>=8` **+0.0762 (t 1.71)** | half-spread contract, ENTRY 0.25; **median round trip 0.143 R** (the stage's own falsified premise: median R is 1.24% of price, not 2%) | next **printed** bar's open under the 0.6% cap; 97.1% fill; non-fills uncharged | **+0.113 (t ~2.5)** **[derived]** | **MARGINAL** | with 144 cells the significance bar is **t > 4.5**, not 2.0 (permutation null p95 4.52, p = 1.000). And the cell is **-0.1255 ex-top-5%**, and **76.2% of the population is untradeable at $300 risk** under the 1% participation cap |
| 15 | **Stage K, 20 daily cells** | **17 of 20 gross-negative**. Best `K3_h3_n10` **+62.9 bps**, `K2_h10_n20` +24.6, `K3_h3_n20` +15.4 | +17.4 (t 0.51) / -23.0 / -30.4 | **E2**: `cost_rt` = half the liquidity band + **5 bps per side** = 29.3 / 63.9 / 89.3 bps RT | **market-on-open** entry, market-on-close exit — a single-price auction | **auction arm already published**: K3_h3_n10 **+52.9 bps, t 0.51 -> 1.57** | **MARGINAL** | G1 needs **t >= 2.0**; the corrected t is **1.57**, gap -0.43 t. And +17.4 goes to **-26.9 ex-top-1%**. The D1 random control is charged the same constant, so the excess-over-control table is cost-invariant and shows no family beating a coin flip on both splits |
| 16 | **Stage L, 44 transfer-filter cells** | n/a — the statistic is a **difference of two net books at identical cost** | best improvement +0.247 | inherited per book | 1-min bar opens, `low <= fill <= high` on 100.000% | **E1/E2/E3 cancel in the diff** | **STAYS DEAD** | permutation: observed max improvement 0.247 vs a 500-draw null p95 of **0.550, p = 0.794** |
| 17 | **Stage M, 20 ORB exit cells** | n/a — identical picks, identical entries in all 6 shapes | X0 (shipped) $14,429 wins; 0 of 5 challengers clear | shipped ORB physics: 30 bps entry **buffer** (a limit price, not a debit) + 10 bps exit slip. **No quoted spread** | resting stop-limit capped at `range_high x 1.003` | cost cancels across cells | **STAYS DEAD** | Stage P measured the shipped model as **$792 pessimistic over 21 months**, an order of magnitude below the $2-4K per-split gaps the gate judges. The time stop loses on mechanism: 27 wins +$2,091 vs 20 losses -$6,253 |
| 18 | **Stage N1, 12 ORB order-flow cells** | not printed | baseline +0.486/+0.862/+0.172 R; best increment **+$300 over 21 months** | inherited ORB 30/10 | ORB capped stop-limit | cost never varied; increments unaffected | **STAYS DEAD** | **MDE 0.424 R/pick = 87% of the book's entire existing edge**; and EQUS.MINI shows <20 prints in the whole opening range on 54.4% of the picks |
| 19 | **Stage N2, 4 cells (K2 with a real 52w lookback)** | `K2_h10_n10` **+37.1 TRAIN / +67.3 VAL**; `K2_h10_n20` **+30.4 / +202.2** | -11.6 / +19.2 and -18.4 / +154.3 | **E2**, the same `cost_rt`, ~48 bps RT over-charge | **market-on-open** | **auction arm: +27.1 / +57.3 and +20.4 / +192.2 — positive on BOTH splits** | **MARGINAL** | **the single clearest E2 sign-flip in the tree.** But G1 needs t >= 2.0 and the arms run **t +0.12 to +1.17**, against an **MDE of 80-205 bps — 2 to 4x the entire cost error**. And `K2_h10_n20` is **-164.7 bps ex-top-5%** |
| 20 | **Stage R_daily v2, 20 cells (5 years of TRAIN)** | **16 of 20 gross-negative**. Best `K5_h5_n10` **+24.7**, `K5_h5_n20` +15.9, `K3_h3_n20` +15.1, `K3_h3_n10` +3.8 bps | -28.4 / -36.7 / -37.8 / -49.4 | **E2**, 50-60 bps RT charged | **market-on-open** | **auction arm already published**: +14.7 / +5.9 / +5.1 / -6.2 | **MARGINAL then dead** | the three that turn positive on TRAIN are **gross-negative on VAL (-9.0, -37.8, -10.3)** — *"There is no gross edge for the cost model to be blamed for."* The powered cells (K3 h1, K4, K5 h2, 3.4K-25K trades) are gross-negative at **t -4 to -25** — cost-invariant |
| 21 | **Stage N3 (the published stocks-in-play ORB, replayed)** | **+0.173 R/trade, t 6.4, positive in 5 of 5 years**, hit ratio 48.4% vs the paper's published 48.4% | **-0.619 R** under our banded table; -0.435 at a flat 40 bps | **E3-wide, the flagship case**: median R is **0.403% of price**, so 40 bps RT = **0.99 R** and the band's 50-190 bps = **1.2 to 4.7 R** | 09:35 stop order at the 5-min high; the resting arm charged **1.00 x half** | **+0.021 R at a flat 10 bps** | **MARGINAL** | the stage's bar was *"same sign and order of magnitude as published"* and **it passed at zero cost — the simulator is vindicated, the cost contract is condemned.** But the corrected book is **break-even**, and ex-top-5% the gross itself is **-0.549 R**. A 0.79 R swing that lands on zero |
| 22 | **`O_halt` (LULD halt-resume), 12 cells** | pooled **+0.802 R**; up/fade+5m +0.655/+1.010/+0.970 | published **+0.53 / +1.00 / +0.90 R** | **E3-NARROW**: 0.40% band charged to a cohort whose measured NBBO is a **median 1.9%** (mean 4.6%, p95 13%) | capped next-open at the reopen, rejects not booked, entry at **0.25 x half** — E1 already handled | measured cost: **-0.671**; + honest cover **-0.840** | **STAYS DEAD** | **the correction KILLS. This is the only place in the tree where the band constant manufactured a result.** The headline is SUPERSEDED. Also 76% of booked trades are SSR-restricted and 3-8% are borrowable |
| 23 | **`O_halt/PASSIVE`, 6 cells** | passive **+0.517** vs marketable +0.487 TRAIN; VAL/TEST +1.08 / +0.97 | best cell **-0.840 (t -3.31)** under the mean-spread arm | **measured per-trade SIP NBBO**; 0 half-spreads on the passive entry, **1.00 on the marketable cover** | **resting sell limit — the correct E1 convention, and it is worth +0.03 R MORE than crossing** | under the **MEDIAN** spread: **-0.160 (t -0.90)** TRAIN, **+0.662 VAL, +0.714 TEST** | **MARGINAL** | the stage's "powered rejection" holds **only under the mean arm**, on a distribution with a **474% maximum and 16.7% of quotes >30 s stale**. Under the median arm TRAIN is a null (MDE 0.711), not a rejection. Borrow kills it anyway: 8.1/6.6/3.3% shortable+ETB, ~1 trade/week |
| 24 | **`bf_zero` HOD-break, causal-filter baseline, 12 cells** | **TRAIN -0.043 / VAL -0.002 R, booked, ZERO cost** | best cell **-0.126 (t -3.89)** | band arm 43-115 bps AND measured per-trade NBBO (34 median / 51 mean bps) | capped limit `min(ask, cap)` at `level x 1.006`, **ask > cap => no fill, row dropped** — correct E1 | cannot exceed the gross | **STAYS DEAD** | ship bar **+0.15 R**. Gross ceiling is **-0.043 R**. Even crediting the permutation's genuine **+0.086 R** of information leaves ~+0.04 R, a quarter of the bar. Forward check: 31 live dry-run trades, **-0.454 R/trade** |
| 25 | **`bf_zero` §8 spread study / the 15% gate / the $20 floor** | flat across spread quintiles (+0.17 to +0.45, no order) | monotone **by construction** | **the worst E1 in the tree**: `spread_score.py:10` debits **one FULL quoted spread** to every trade (~39 bps flat = ~0.18 R at median R) | next-open under the 0.6% cap | the gate would have been chosen differently | **STAYS DEAD** | the E1 over-charge **did** drive a selection (the `max_spread_frac_r: 0.15` gate and the $20 price floor were chosen on the inflated after-cost column). But the population it selected on is **voided by §6b** as a look-ahead. Correcting E1 changes which gate looks best, not the sign: the SIP re-sim is **-0.051 / -0.071 / -0.103 R at zero cost** |
| 26 | **`orb_multiwindow`, 6+5 cells** | not split out | every ADDED set negative: W15 -0.220 (t -2.75), 5+15 -0.254, 5+30 -0.240 | **30 bps = the limit price, not a debit**; 10 bps exit slip; **no quoted spread anywhere** | **already correct**: capped stop-limit, `ask > cap => no fill`, entered-inclusive $0 no-fills | n/a | **STAYS DEAD (correct model)** | bar is +0.30 R/added pick; MDEs 0.148-0.272 R are **below** it, i.e. powered. The report tests the strict arm explicitly: rescuing 5+15 would need the deleted fill to have been **-5.2 R** against a per-trade floor of about -1 R |
| 27 | **`orb_veto_study`, 7 cells** | dollars only | baseline $6,085 -> $6,627 | shipped 30/10, no spread charged | **already correct** (same capped stop-limit) | n/a | **SHIPPED, not closed** | this one passed its bar and went live 2026-09-07 |
| 28 | **`orb_anchor_dedup`, 4 cells** | dollars only | N=8 $14,429 -> $12,710 | shipped 30/10; **both arms priced identically** | **already correct** | cost-neutral by construction | **STAYS DEAD** | fails on tail shape (worst month -148 -> -221 at N=8), not on cost |
| 29 | **`D1_orb` (the live ORB book)** | n/a | $342/mo at 3 slots, $687/mo at 8 | shipped ORB physics | capped stop-limit | Stage P: shipped model is **$792 pessimistic**; Stage Q: the honest fill model is **-$2.2K to -$4.1K**, carried by **six picks** | **SURVIVED (small)** | the only book in this audit that is alive. Restated $14,429 -> **$12.2K** by Stage Q |
| 30 | **`lit_review_2026` M6..M41 (~25 rows)** | see §3 | see §3 | **E2 present on 10 rows** (MOC->MOO auction pairs charged a 6-40 bps band round trip) | close-to-open / open-to-close auctions | largest over-charge ~13 bps | **STAYS DEAD** | every affected row fails on **gross sign** or on a **read-once TEST sign flip of 17-570 bps**, or at t < 1 in the gross. The correction was already run on the flagship (`overnight_auction.md`, 2/5/10 bps) and the verdict held: VAL -22.2, TEST -23.1 bps **gross** |

---

## 2. REVIVABLE shortlist

**There are none.** No cell in the tree has a gross result whose gross-to-net gap is a mis-modelled cost of a
size that a corrected charge would carry past that stage's own ship bar. That is the finding, and it is a
valid one.

What follows is the honest substitute: the **five candidates with the most headroom**, ranked, each with the
re-run that would settle it, its compute cost, and whether the population is still on disk. None of them is
recommended without its own pre-registration; two of them are recommended against outright.

| rank | candidate | what would settle it | compute | population on disk? |
|---|---|---|---|---|
| **1** | **`O_halt/PASSIVE` — the spread STATISTIC, and the passive COVER** | The stage rejects on the **mean** minute spread (TRAIN 4.63%, max 474%, 16.7% of quotes >30 s stale). Under the **median** (1.92%) the same cell is **-0.160, t -0.90, MDE 0.711** — a null, with VAL +0.662 and TEST +0.714. Two questions, one PREREG: (i) which statistic the cover leg owes — this is the same mean-vs-median argument Stage P settled the other way for ORB, and it has never been settled here; (ii) the **passive cover** (a resting buy limit below the NBO), named in §9 as untested, where 100% of the remaining cost sits. It needs its own fill-rate measurement and its own unfilled-cover handling (an unfilled cover is an open short, not a zero) | the entry/cover NBBO pulls are **already done and on disk** (`O_halt/` 127 MB). A re-score is **minutes**; a passive-cover sim needs a new fill-rate walk, ~1-2 h of Alpaca SIP at $0 | **YES** — `O_halt/` intact, 127 MB |
| **2** | **Stage N2 / K2 at hold 10** | The only cells in the tree that are **gross-positive on both splits AND auction-positive on both splits** (+27.1/+57.3 and +20.4/+192.2 bps). They fail on **power** (MDE 80-205 bps vs a 48 bps cost error) and on the tail (`K2_h10_n20` is -164.7 bps ex-top-5%). What would settle it is **more TRAIN, not a cost patch** — and R_daily already bought five years of it and found K2 **gross-negative at -24 to -47 bps on the clean, split-adjusted panel**. **Recommendation: do not re-open.** It is already answered in the direction that closes it | `R_daily/` 368 MB + `K/trades/*.csv` on disk; a re-read is **minutes** | **YES** |
| **3** | **Stage C's closest miss — `F6 {} next / 2R stop-1% / entries >= 10:00`** | gross **+0.049** vs a G1 requirement of **0.044 R**; the E1-corrected net is **+0.036**, gap **-0.008 R**. It clears the *mean* bar only at literally zero cost, where its t is ~2.26 against a 107-cell permutation p95 of **3.88**. Settling it means a **per-trade measured NBBO** on this cell's booked trades (PLAN §3 H7 Stage B, never run) — but Stage P already published the ceiling: the whole band charge on this population is **0.057 R**, and a +0.057 R move cannot carry a t from 1.93 past 3.88 | `C/pop_c.csv` and `B/candidates4.csv` were **DELETED 9/18**. Rebuild = **B ~2.5 h + 2 GB**, then C ~20 min. An NBBO pull on the booked trades only is ~1 h at $0 | **NO** — must rebuild (B), then re-score |
| **4** | **Stage E's resting-fill cells (`F6 {} rest/hold`)** | gross **+0.139**, net **-0.0207**, charged **ENTRY 1.00 x half** on a capped stop-limit. Max E1 credit is **+0.099**, corrected net **+0.079** — which sits **at or below E's own MDE of 0.083-0.154 R**. Settling it means measuring the actual ask at the resting fill instant, exactly as Stage Q did for ORB (which found the honest charge is ~0.25x, not 0). Expected outcome: the corrected number lands between +0.02 and +0.05 R, inside the MDE | `E/` is 19 MB of keys and scores; the **candidates file was deleted 9/18**. The causal builder was never run to completion — **4-6 h + ~4 GB** to rebuild | **NO** — expensive rebuild, never finished once |
| **5** | **Stage J's `F14 N=15 / hold / PDR>=8`** | net **+0.0762 (t 1.71)**; the stage's published median round trip is **0.143 R**, so max E1 credit is ~+0.037 and the corrected cell is ~**+0.113, t ~2.5**. Against a **144-cell significance bar of t > 4.5**. And it is **-0.1255 ex-top-5%**, and 76.2% of its population cannot be traded at $300 of risk under a 1% participation cap. **Recommendation: do not re-open** — it fails three independent tests, only one of which is cost | **`J/pop_j_TRAIN.parquet` (177 MB) IS on disk** — a cost-contract re-score is **~10 min** | **YES** (TRAIN only; VAL/TEST were never built) |

**The single highest-value re-run** is rank 1's first half, and it costs essentially nothing: re-score
`O_halt/PASSIVE`'s six cells under the **median** cover spread alongside the mean, from data already on disk,
and state both. It is the one place in this tree where a cost *statistic* — not a cost *error* — decides
whether a stage reports "a powered rejection" or "a null". Stage P already made exactly this argument in the
opposite direction for ORB (the mean is what a book's mean net R owes); consistency demands the same question
be asked here, in writing, before `O_halt`'s closure is treated as settled. **It is not a revival candidate —
borrow availability caps the book at ~1 trade/week regardless — but the closure's stated grounds are wrong
as written, and that is worth an hour.**

---

## 3. Stays dead, and why — on the record as sound

These closures do not depend on any cost or fill assumption. If every spread in this program were set to
zero tomorrow, none of them would move.

**Gross <= 0 — no cost correction can help:**

1. **`bf_zero` HOD-break causal filter** — booked baseline gross **TRAIN -0.043 / VAL -0.002 R** against a
   **+0.15 R** ship bar. The live dry run agrees: 31 booked trades, **-0.454 R/trade**.
2. **`bf_zero` HOD-break on the honest SIP tape (§6a)** — **-0.030 / -0.006 / +0.016 R at ZERO cost.**
3. **`bf_zero2` F6 {} across splits** — the one family that clears TRAIN and VAL at zero cost is
   **gross-negative on TEST (-0.043 R)**.
4. **Stage K: 17 of 20 cells gross-negative. Stage R_daily v2: 16 of 20 gross-negative**, the powered ones at
   **t -4 to -25** on 3,400-25,000 trades.
5. **Stage C's grid as a whole**: gross **-0.016 R** over 107 cells.
6. **`lit_review` M8** (stocks-in-play ORB OOS): **-0.12 / -0.25 / -0.17 R gross**. **M10** (QQQ VWAP flip):
   -20.9 / -28.5 bps/day gross. **M36/M37** large-loser reversal: **-86 to -156 bps gross** against a 12-25
   bps charge. **M16**: 0 of 27 gap cells reach +50 bps. **M1/M2/M3/M12** last-30-min timing: gross already
   negative.
7. **Stage C's F12 retest (-0.209/-0.233 R) and F13 sweep-and-reclaim (-0.19 to -0.35 R)** — gross-negative
   too; a consolidation break taken in either direction is where the stops are (Stage G found the mirror on
   S3/S4).

**Killed by a defect that is not a cost:**

8. **H / F6 + F6_rebuild + F6_reconcile (red-to-green)** — `ZVZZT`, a NASDAQ test ticker with zero rows in
   `daily_bars`, carried +46.76 R on one synthetic day and +50.80 R of a +39.00 R TEST; `BFZ_SLIP=0` put the
   break level at the prior close instead of x1.003 (308 of 339 TEST-only candidates, +56.05 R); and the
   engine's own keep-scanning rule versus the studies' first-break rule is worth +0.09/+0.23 R — the whole
   positive result. Under the engine's rule the book is **-0.027 / -0.012 / -0.102 R**.
9. **Stage D1** — an **availability look-ahead**: `pm_bars.db`'s coverage was keyed to a set of symbol-days
   that signalled *later*. The report measures the cost term at **0.017-0.033 R and calls it flat across
   every selection**: *"the cost term is NOT the leak."*
10. **Stage D (D0)** — the winning model is **profitable on a reversed tape** (+0.187 R, t 4.39). No cost
    model produces that.
11. **H / F14_F8_F11** — max \|t\| **2.44 over 83 cells against a null MEAN of 3.04**. The best cell in the
    sub-stage is below the average of what pure noise produces.
12. **Stage J** — permutation p = 1.000 at a 144-cell bar of t > 4.5; and **capacity fails before edge does**
    (76.2% of the population untradeable at $300 risk).
13. **Stage L** — the decision statistic is a difference of two net books at identical per-trade cost, so all
    three errors cancel to first order. p = 0.794.
14. **Stage I** — killed by the participation cap, and a **x1.5 spread sensitivity moves no sign**.
15. **H / QQQ and Stage Q** — cost is ~6x conservative on a mega-cap ETF; the **zero-cost** result (6.08
    bps/day) is inside the stage's own MDE (5.75), and the top 5 days are **102% of the OOS return**.
16. **Stage N1** — MDE 0.424 R/pick is **87% of the book's entire existing edge**; the feature is computable
    on half the picks.
17. **Stage M** — identical picks and identical entries across all six exit shapes; cost is additive and
    cancels. The shipped exit wins on mechanism.
18. **`orb_multiwindow`** — already on the correct capped-limit convention, and it tests the strict arm
    explicitly: rescuing the best added-pick cell would require a deleted fill of **-5.2 R** against a
    per-trade floor near -1 R.
19. **`orb_anchor_dedup`** — both arms priced identically; fails on tail shape.
20. **`O_halt`** — the cost correction is what **killed** it. Charging 0.40% to a cohort quoting a median
    1.9% manufactured the +0.53/+1.00/+0.90 R headline. Reg SHO (76% restricted) and borrow (3-8%) were
    already binding.
21. **`lit_review` M29 / M41 / M30 / M22** — E2 is genuinely present (6-40 bps band round-trips charged on
    auction pairs), but every one fails on a **read-once TEST sign flip of 17 to 570 bps** or at t < 1 in the
    gross. A 13 bps correction against a -183 bps TEST is nothing.

---

## 4. Honest restatements for the MARGINAL cells

These sentences replace the ones currently in circulation. Each obeys the phrasing rule: the population, the
horizon, the book size, the window, the cost and the power are all named.

1. **`bf_zero2/score4`, 52 cells.** Not *"0 of 52 cells are positive; the best is -0.106 R"*. The correct
   sentence is: **under a measured rather than a banded spread, 9 of 52 cells are net-positive and 12 of 52
   are positive if a capped-limit entry is charged nothing at all; the best of them reaches t 1.68 against a
   requirement of 2.0 and a 52-cell noise ceiling near 2.7. In this >=5%-range-day population, at a 1-minute
   horizon, in a 12-a-day 4-concurrent book, over 2025-01 to 2026-09, the grid is indistinguishable from zero
   — not clearly negative. The smallest per-trade effect the headline tests could have seen at 80% power is
   0.066 R (F8 N=30), 0.106 R (F6), 0.476 R (F1 P=0.12).** The one family that is gross-significant on TRAIN
   and VAL (F6 {}) is gross-negative on TEST, so this restatement is about power, not about a hidden book.

2. **Stage C, 230 cells.** Not *"nothing works"*. **The grid's gross is -0.016 R and its best cell is
   +0.049 R gross / +0.031 R net against a gate that needs 0.044 R. Setting the spread to exactly zero is
   worth +0.057 R on this population — that is the entire ceiling on any cost correction — and it cannot
   carry the best cell's t from 1.93 past a 107-cell permutation p95 of 3.88. The result is a powered null on
   the mean and an underpowered null on the search-adjusted t.**

3. **Stage E, 54 cells.** **Seven of the top ten cells are gross-POSITIVE and net-negative; the charge is
   0.10-0.15 R at these R sizes. Correcting the resting-fill entry leg to zero takes the best cell to
   +0.079 R, which sits at or under the stage's own MDE of 0.083-0.154 R.** In this causal universe, at this
   horizon and book, no effect larger than ~0.08 R was detectable; effects smaller than that are not
   excluded.

4. **Stage G, 24 short cells.** **The short side is gross ~flat-positive in 5 of 6 families (+0.108 to
   -0.007 R) and the cost contract charges 0.052-0.120 R per trade. The best-gross family (S5, +0.108) fires
   2.6 times a week and fails the >= 5/week bar structurally at any cost. The best family that meets the
   frequency bar (S1, gross +0.057) corrects to +0.035 R against an MDE of 0.056 R.** This is the same
   near-zero-gross / negative-net shape Stages C and E found long, sign-flipped.

5. **Stage J, 144 cells.** **The best cell is +0.076 R net and roughly +0.11 R with the entry leg zeroed —
   t ~2.5 against a 144-cell significance bar of t > 4.5. No edge was detectable in the liquid universe U3,
   at a 1-minute horizon, in a 12-a-day 4-concurrent book, over 2025-01-17 to 2025-12-31, at a measured
   0.143 R median round trip; the smallest per-trade mean this test could have called significant at t >= 2
   is 0.050-0.099 R.** Capacity, not edge, is the binding constraint.

6. **Stage K / R_daily v2, 40 daily cells.** Not *"the multi-day direction is closed"* without this
   qualifier: **the cost model over-charged these market-on-open fills by 43-53 bps of round trip, because an
   opening auction is a single-price cross with no quoted spread to pay. Under the auction-honest 10 bps
   round trip, 4 of 20 R_daily cells and K3_h3_n10 turn positive on TRAIN — and every one of them is
   gross-negative on VAL, and the powered cells (3,400-25,000 trades) are gross-negative at t -4 to -25.
   There is no gross edge for the cost model to be blamed for.**

7. **Stage N2, 4 cells.** **K2 at hold 10 is the only place in this program where the auction cost error
   flips the reported sign on both splits: gross +37.1/+67.3 and +30.4/+202.2 bps, net -11.6/+19.2 and
   -18.4/+154.3, auction-honest +27.1/+57.3 and +20.4/+192.2. It is nonetheless not a book: t runs +0.12 to
   +1.17 against an MDE of 80-205 bps — two to four times the entire cost error — and K2_h10_n20 is
   -164.7 bps with the top 5% removed. Read this as a null with no power, not as a cost kill; and R_daily's
   five-year clean panel finds the same family gross-negative.**

8. **Stage N3.** **Our simulator reproduces a peer-reviewed 8-year intraday book in sign and size at zero
   cost (+0.173 R/trade, positive in 5 of 5 years, daily hit ratio 48.4% against the paper's published
   48.4%). The banded spread table takes the same trades to -0.619 R. The arithmetic is the whole story:
   median R on that book is 0.403% of price, so one 40 bps round trip is 0.99 R of cost. At a flat 10 bps the
   book is +0.021 R — break-even. A published intraday edge is worth ~+0.17 R gross and ~0.00 R at any
   spread we can actually pay; and ex-top-5% its gross is -0.549 R.** The correct conclusion is not "the
   effect is fake" and not "the effect is tradable"; it is "the effect is real and smaller than its own
   execution cost at our size".

9. **`O_halt/PASSIVE`, 6 cells.** **The passive limit is not what fails — it fills (94.0% at b=0) and it is
   worth +0.03 R MORE than crossing. What fails is the cover, charged a full half-spread as a marketable buy,
   and the statistic used for it. Under the MEAN minute spread (TRAIN 4.63%, on a distribution with a 474%
   maximum and 16.7% of quotes over 30 seconds stale) the cell is -0.840, t -3.31, a powered rejection
   against an MDE of 0.711. Under the MEDIAN (1.92%) the same cell is -0.160, t -0.90 — a null — with VAL
   +0.662 and TEST +0.714. The rejection is real only under the mean arm, and which statistic the cover leg
   owes has never been argued in writing.** Independent of all of it, borrow availability caps the book at
   roughly 1 trade a week.

---

## 5. What is actually owed, and what is not

**Three contract fixes were already written down and are still outstanding** (Stage P §6d, `LOG.md` L186).
None of them revives anything; two of them make past numbers worse. They are recorded here so they are not
lost again:

1. **`acore.corrected_spread_table()` takes `.median()` where it owes the `.mean()`** — one line. The
   distribution's mean is ~1.9x its median in every cell, and a book's mean net R is a linear functional of
   the spread. **This makes every past number in A/B/C/D/E/G/J/H MORE conservative, not less.**
2. **The band table has no cell below $5.** 17.6% of ORB's fills live there. Any future population that
   allows sub-$5 names is unpriced today.
3. **Stop charging a capped limit the whole spread distribution.** The honest contract is the pair
   (`entry at min(ask, cap)`, `ask > cap => no fill`). Stage Q measured both halves: the "no fill" half is
   right only 8% of the time, the rest is a delay of a median 20.8 s, and the fill lands **at the cap**. This
   is E1 stated correctly, and it says contract (c)'s 0.25x is approximately right — **not zero**.

**And one standing rule earned at `O_halt` that has NOT been applied retroactively** (`LOG.md` L211): *a
band-table cost constant is a hypothesis, not a cost — any cohort outside the band table's own population
(halt reopenings, first minutes, microcaps) must have its spread MEASURED before a cell is reported.* Stages
E, G, H/F5 and J all book sub-$10 microcaps in the 09:30-10:00 window, i.e. exactly that out-of-population
cohort. The measurements that exist point in **both** directions (ORB: band 1.92x the median but 1.03x the
mean; halted microcaps: band 4.8x too NARROW), so the retroactive application is not a free upgrade and
should not be sold as one.

---

## 6. Cells looked at, and where this audit could be wrong

**Cells.** This audit ran **0 new cells**. It re-read **30 stage/sub-stage reports** and restated
**52 published cells** (`A/a0_cells.csv`, TRAIN/VAL/TEST x 6 contracts) arithmetically. Program-wide cell
counts as published by the stages themselves: A 192 · C 1,305 · D1 1,353 · E ~748 · G 200 · H/F6 ~875 ·
H/F14 ~540 · I 84 · J 144 · K 53 · L 54 · M 20 · N1 12 · N2 4 · N3 1 · O_halt 12 · O_halt/PASSIVE 6 ·
P_cost 48 descriptive + 40 config x split · Q_fill 64 + 16 · Q/QQQ 36 · R_daily 40 · bf_zero 439 cumulative ·
bf_zero2 275 + 108 + 52 · lit_review ~25 rows + 63 queue cells · orb_multiwindow 11 · orb_veto 7 ·
orb_anchor_dedup 4.

**Where this could be wrong:**

1. **The `corrected_net` column is arithmetic on published aggregates, not a re-simulation.** It assumes the
   cost charge is additive and that the per-cell exit mix is the one published. Stage P and Stage Q both
   verified additivity on the ORB pipeline (selection never reads P&L); it is assumed, not verified, for
   B/C/E/G/J.
2. **It assumes the maximum E1 credit** (a capped limit pays nothing at entry). Stage Q measured that this is
   false: 85.6% of capped orders are marketable at the trigger and fill at `min(ask, cap)`, median 11.9 bps
   over the level. Every corrected figure here is therefore an **upper bound**, and the true correction is
   roughly one quarter of it — which is exactly what contract (c) already charges.
3. **Stages D, H/F14_F8_F11, J, M, N1 and L publish no gross column**, so their cost exposure is inferred
   from the stage's own per-trade charge, not measured. For D and D1 the reports state the charge directly
   (0.017-0.033 R) and that is enough; for H/F14 it is not, and the only reason that stage is filed under
   STAYS DEAD is its permutation result, which is cost-independent.
4. **Several populations no longer exist** (`B/candidates4.csv`, `C/pop_c.csv`, `E`'s causal candidates,
   `J`'s U3 tape, deleted 2026-09-18 with owner GO). Their reports remain and carry the numbers used here,
   but no figure in the MARGINAL rows above can be re-derived from disk without a rebuild, and the rebuild
   hours are stated per row in §2.
5. **`lit_review`'s M-rows were read as a table, not re-scored.** The E2 classification of each row rests on
   the fill convention the row's own script declares.
6. **This audit reads reports.** Where a report is wrong about itself, this audit is wrong with it. The one
   place that is known to have happened in this tree is R_daily v1, whose verdict survived a contaminated
   panel by luck; only v2 is used above.

**Phrasing.** Nothing here says a cost correction can never matter. It says that **in THIS program**, across
**THESE 30 closed stages**, at **THESE book sizes** (12/day 4-concurrent for the intraday grids, 3-8 slots at
$10K stage for ORB, 10-20 slots for the daily panels), over **THESE windows** (2025-01 to 2026-09 intraday,
2018-05 to 2026-09 daily), the two errors named in the brief are worth **+0.412 R/trade where they were
largest (already applied, 9/16), a further ~+0.02 R at most (E1's residual), and 43-53 bps of round trip on
three daily stages (already published as a corrected arm)** — and the smallest per-trade effect the affected
tests could resolve at 80% power runs **0.056 to 0.476 R intraday and 6.3 to 205 bps daily**, which is in
every case larger than the correction. **The corrections move the level. They move no verdict, except
`O_halt`'s, which they reverse in the direction of killing it.**
