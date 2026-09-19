# Red-to-green (F6-PDR) through the mature method — REPORT (2026-09-19)

Candidate #3 of `research/mature_method/RUNBOOK.md`, all ten steps, cells as declared in `PREREG.md`
(committed `1aea9d5` **before any cell was scored**). Artifacts: `prep_pop.py` → `pop.csv` (616,883
symbol-days), `pass_r2g.py` → `cands.csv` (471,924 signal rows, both scan rules), `repro.py`,
`cost_nbbo.py` + `supp_cost.py` → `nbbo.csv` / `cost_curve_measured.csv`, `score.py` → `score.log`,
`cells.csv`, `gatemap.csv`, `nulls.csv`, `bars.csv`, `supp.py` → `supp.log`, `cost_models.csv`,
`test_read.csv`. One `nice -n 10` python process at a time, `ulimit -v 3000000`; `data/cache.db`,
`research/bf_zero/bars_sip.db` opened **read-only**. No config, `orb.yaml`, systemd unit, cron, order
or cache was written. `red_to_green.enabled` is still `false` and was not touched.

---

## VERDICT — **STAY DEAD**

*In THIS universe — the whole `research/bf_zero` point-in-time day list minus NASDAQ test tickers and
non-`daily_bars` names (616,883 symbol-days, 7,583 symbols), screened by the live engine's own
universe rule (ADV20 ≥ 100 K, prior close ≥ $5, prior-day range ≥ 8 %) — at THIS horizon (stop at the
pre-entry low, +2R bracket, flat 15:55), at THIS book size (12/day, 4 concurrent, $100 risk), over
2025-01-03 → 2026-09-04, at THIS cost (per-signal measured Alpaca SIP NBBO at the fill instant):*

* ***the shipped engine's scan rule (S2, keep scanning past a floor-failing break) is gross-flat and
  dead*** — booked gross **+0.027 R TRAIN (t 0.97) / +0.001 R VAL (t 0.03)** against a measured cost
  of 0.08–0.14 R. Not one of its 26 declared cells passes the claim bar on either split.
* ***the studies' scan rule (S1, first break only) has a REAL gross — +0.108 R a trade on both open
  splits, t 3.61 and 2.51 — and it is still not a book.*** The measured entry leg costs **1.00 ×
  half-spread, four times the 0.25 × half the program's cost contract charges**; correcting it takes
  the net from +0.054 / +0.051 R to **+0.011 / +0.006 R**, i.e. **$103 and $71 a month at the live
  $100 risk**. And the two cells that cleared both claim gates were opened on the sealed split and
  **both are gross-ZERO there** (−0.002 and −0.018 R, 35.7 % and 28.6 % green weeks, −$2,217 and
  −$2,768).

**This is neither of the two worked examples — it is both at once.** HOD-break was gross-flat and
could not be saved by any cost correction. QQQ's gross was real and the cost was over-charged, and it
still did not matter. Red-to-green's **shipped rule** is the first case and its **unshipped rule** is
the second, with the cost error running the *other* way: our band table was **too NARROW** here, as
pre-registered, and the entry leg was wrong by 4x in the book's favour.

Not SHIP-TO-DRY: a dry run of the engine as it stands measures S2, which is gross-flat; a dry run of
S1 would need a code change to a rule whose sealed split is already negative, and whose honest
throughput at full capacity is $400–500/month (`H/F6_sizing`). Not NOT-DECIDABLE: the TEST read was
pre-authorised, was taken, and answered.

---

## 1. Reproduction gate — EXACT on both scan rules, against two independent implementations

`repro.py` and `supp.log` §(a). Neither comparison is an aggregate: it is candidate by candidate on
(day, symbol).

| this pass | vs | shared | identical `sig_m` / `entry_m` / `entry` / `stop` | max \|Δ gross R\| | unmatched |
|---|---|---|---|---|---|
| S2 at E's gates | `H/F6_reconcile` **implementation E** (`e_cands.csv`, 13,541) | **13,065** | **13,065 / 13,065** | **5e-6** (CSV rounding) | E-only 476, mine-only 7 |
| S1 at L2's gates | `H/F6_reconcile` **ladder step L2** (`ladder_L2_engine_cut.csv`, 7,755) | **7,489** | **7,489 / 7,489** | **5e-6** | L2-only 266, mine-only 4 |

**Every one of the 476 E-only rows is a symbol absent from `daily_bars`** (22 of them also test
tickers) — the two standing membership cuts account for the whole difference, with nothing left over.
The 7 and 4 mine-only rows are 0.05 % bar-gap edges at the `no_next` boundary. E and L2 were written
by a different agent from a different code path against a different prior-day source; with
implementations A and B behind them, **four independent implementations now agree on this book**, and
`pipeline.py`'s L0 already reproduced B to the third decimal. CLAUDE.md's independent-rebuild rule is
discharged; it catches coding errors and not specification errors, which is what §3 and §4 are for.

## 2. Gross before net — the scan rule decides whether there is anything here at all

No cost at all, exit = the engine's 2R bracket, after the membership cuts and the live universe screen:

| scan | population | split | n | /wk | **gross R** | ± SE | t | **MDE80** | WR |
|---|---|---|---|---|---|---|---|---|---|
| **S1** first break only | every live-config signal | TRAIN | 3,581 | 67.6 | **+0.0555** | 0.0136 | **+4.09** | 0.038 | 48.3 % |
| **S1** | every live-config signal | VAL | 2,153 | 97.9 | **+0.0484** | 0.0178 | **+2.71** | 0.050 | 47.0 % |
| **S1** | the shipped 12/4 book | TRAIN | 1,125 | 21.2 | **+0.1081** | 0.0300 | **+3.61** | 0.084 | 48.6 % |
| **S1** | the shipped 12/4 book | VAL | 604 | 27.5 | **+0.1076** | 0.0428 | **+2.51** | 0.120 | 49.5 % |
| **S2** keep scanning (**shipped**) | every live-config signal | TRAIN | 6,162 | 116.3 | +0.0648 | 0.0118 | +5.47 | 0.033 | 48.1 % |
| **S2** | every live-config signal | VAL | 3,911 | 177.8 | **+0.0191** | 0.0148 | +1.29 | 0.041 | 45.0 % |
| **S2** | the shipped 12/4 book | TRAIN | 1,597 | 30.1 | **+0.0269** | 0.0277 | **+0.97** | 0.078 | 43.5 % |
| **S2** | the shipped 12/4 book | VAL | 762 | 34.6 | **+0.0013** | 0.0415 | **+0.03** | 0.116 | 43.3 % |

Runbook step 2's stop condition (*gross ≤ 0 with the MDE below the effect the book needs*) is met for
**S2's book** — +0.027 / +0.001 R against a measured cost of 0.080–0.137 R, with 0.078–0.116 R of
resolution. It is **not** met for S1, whose gross is 3–4 standard errors clear of zero on both splits,
so the pass continues on S1 and everything below is why it still ends where it does.

**S1 is a strict subset of S2.** Where the first level break clears the 5 % floor, S1's and S2's
signal bars are identical; S1 simply kills the day when it does not. So the scan rule is a **gate**,
and §5 measures it as one. The engine keeps 4,339 extra signals that S1 discards, and they are where
the gross goes.

## 3. Measured cost — the band was too NARROW, as pre-registered, and the ENTRY LEG was wrong by 4x

Alpaca SIP NBBO at the modelled fill instant (the open of the bar after the signal), 2,963 signals
sampled across 30 price × hour strata, TRAIN and VAL only (TEST legs never sampled):

| | median | mean | p90 |
|---|---|---|---|
| **measured** quoted spread, all sampled signals | **45.2 bps** | 92.1 | 216.8 |
| **measured**, restricted to B0-eligible signals | **51.1 bps** | 99.9 | 231.1 |
| `cost_curve.csv` **band** on the same signals | 39.2 / 42.2 bps | 44.4 | — |

**The band under-charges this population by 1.15–1.21x on the median and 2.1x on the mean** — the
direction pre-registered in `PREREG.md` §5 and the opposite of HOD-break's $20+ names, which the band
*over*-charged 1.35–1.96x. Coverage is **82.2 %**; the 17.8 % with no SIP quote inside a 20-second
window are the thinnest names, so the measured curve is if anything still too kind. **8.7 % of
B0-eligible sampled signals quote wider than the live `max_spread_bps: 300` gate** and would be
skipped live — not modelled here, and the direction is favourable to the reported book.

**The finding that decides this candidate is not the curve, it is the leg weights.** The program's
cost contract charges the entry at `0.25 × half-spread` on the theory that a next-bar-open fill under
a cap does not cross the spread. Measured: the ask at that instant is **+23.2 bps above the modelled
open fill**, on a 59.3 bps spread — **1.00 × half-spread, four times the charge.** Five cost models,
`cost_models.csv`, cell `B0`, exit = 2R:

| cost model | S1 TRAIN net R | S1 VAL net R | S1 TRAIN $ | S1 VAL $ | S1 TRAIN green | S1 VAL green | S2 TRAIN | S2 VAL |
|---|---|---|---|---|---|---|---|---|
| `M1band` band + contract (c) legs | +0.0775 (t 2.59) | +0.0740 | +8,713 | +4,472 | 60.4 % | 86.4 % | −0.0198 | −0.0479 |
| `M2meas` measured curve + contract (c) legs | +0.0543 (t 1.81) | +0.0510 | +6,107 | +3,083 | 58.5 % | 77.3 % | −0.0553 | −0.0826 |
| **`M3entry` measured curve + MEASURED entry leg** | **+0.0109 (t 0.36)** | **+0.0059** | **+1,231** | **+354** | 47.2 % | 54.5 % | −0.1174 | −0.1462 |
| `M4live` M3 with venue legs (TP = a resting limit) | +0.0176 | +0.0137 | +1,976 | +829 | 50.9 % | 54.5 % | −0.1074 | −0.1372 |
| `M5cons` conservative: full half-spread both legs | −0.0074 | −0.0129 | −835 | −782 | 43.4 % | 54.5 % | −0.1387 | −0.1681 |

`M2meas` is the headline used elsewhere in this report, for comparability with the other two
candidates. **`M3entry` is the honest one**, and under it S1's +0.108 R of gross buys +0.011 R of net:
the measured cost is **0.097 R a trade against a 0.108 R gross**. At $100 risk that is **$1,231 over
53 TRAIN weeks and $354 over 22 VAL weeks — $103 and $71 a month.** `H/F6_sizing` already measured
this book's capacity ceiling at $400–500/month at $100 risk and a plateau of $1.4–2.0 K/month at any
larger risk, so scaling does not rescue the number either.

## 4. The engine's real fill model, and the unfilled counterfactual — opposite answers per scan rule

Capped limit at `level × 1.006`, filled at the next printed bar's open iff that open is at or under
the cap; an over-cap signal is not a trade.

| scan | signals | filled | fill rate | over-cap overshoot (med / p90) | fills on the next CLOCK minute |
|---|---|---|---|---|---|
| S1 | 7,867 | 5,734 | **72.9 %** | +67 / +296 bps | 86.1 % |
| S2 | 24,258 | 10,073 | **41.5 %** | +162 / +382 bps | 82.0 % |

| scan | split | FILLED gross | UNFILLED counterfactual | gap | t |
|---|---|---|---|---|---|
| **S1** | TRAIN | +0.0555 (3,581) | **−0.0122** (1,325) | **+0.068** | **+2.55** |
| S1 | VAL | +0.0484 (2,153) | +0.0507 (808) | −0.002 | −0.06 |
| **S2** | TRAIN | +0.0648 (6,162) | **+0.1192** (8,689) | **−0.054** | **−3.49** |
| S2 | VAL | +0.0191 (3,911) | +0.0232 (5,496) | −0.004 | −0.22 |

**Classification, S1: a CHASE GUARD on TRAIN (t +2.55), neutral on VAL.** The cap declines trades that
were genuinely worse — the runbook's ORB case. **Classification, S2: mildly ADVERSE on TRAIN
(t −3.49).** Under the shipped scan rule the cap turns away the better half of a population that is
58 % unfillable to begin with: the engine spends its day on breaks that gap through its own limit.
That is a second, independent reason the shipped rule is not the book the studies measured.

Obtainability: every modelled fill is a bar's OPEN, a price the tape printed. 14 % (S1) / 18 % (S2) of
fills land on a later print than the next clock minute — the engine's order lives ~20 s, so those are
optimistic; cell `G1c` restricts to the clock-minute fills and is scored throughout.

## 5. Gate-separation map — leave-one-out on the whole live stack (`gatemap.csv`)

Net Δ at the measured cost; `notl` = median position notional at $100 risk (`shares = 100 / R`).

| scan | gate | grossΔ | netΔ | t | n kept | n rej | TRAIN | VAL | notl kept | notl rej |
|---|---|---|---|---|---|---|---|---|---|---|
| — | **Gs first break clears the floor (S1 vs the extra S2 signals)** | **+0.014** | **+0.049** | **+2.52** | 5,734 | 4,339 | +0.014 (t 0.56) | **+0.100 (t 3.27)** | **1,481** | **2,310** |
| S1 | **Gp pdr ≥ 8** | +0.078 | +0.073 | +5.37 | 5,734 | 4,430 | **+0.120** | **−0.013** | 1,481 | 1,706 |
| S1 | Gp pdr ≥ 10 / ≥ 12 | +0.087 / +0.082 | +0.082 / +0.076 | +5.38 / +4.40 | 4,349 / 3,275 | 5,815 / 6,889 | +0.136 / +0.146 | −0.017 / −0.043 | — | — |
| S1 | **Gt signal ≤ 14:00** | +0.087 | **+0.068** | **+4.32** | 5,734 | 719 | **+0.090** | **+0.025** | 1,481 | 1,391 |
| S1 | Gt signal ≤ 13:00 (tighter) | −0.033 | **−0.054** | **−3.05** | 5,109 | 1,344 | −0.120 | +0.071 | — | — |
| S1 | Ga adv20 ≥ 100 K | +0.181 | +0.194 | +1.97 | 5,734 | **107** | +0.256 | −0.014 | 1,481 | 1,400 |
| S1 | Ga adv20 ≥ 500 K | −0.006 | +0.001 | +0.04 | 4,071 | 1,770 | +0.048 | −0.075 | — | — |
| S1 | **Gx level ≥ $10 / ≥ $20** | −0.024 / −0.043 | **−0.009 / −0.028** | −0.38 / −1.31 | 3,771 / 2,248 | 1,963 / 3,486 | +0.037 / +0.013 | −0.087 / −0.096 | 1,468 | 1,505 |
| S1 | Gr r ≥ 1 % | +1.418 | +2.821 | +3.54 | 5,734 | **33** | +1.892 | +5.300 | 1,481 | **17,333** |
| S1 | Gr r ≥ 2 % | +0.381 | +1.095 | +3.01 | 5,680 | 87 | +0.740 | +1.976 | 1,475 | 8,385 |
| S1 | Gc the cap (`level × 1.006`) | +0.041 | +0.041 | +1.87 | 5,734 | 2,133 | +0.066 | +0.000 | 1,481 | 1,155 |
| S1 | **Gv rv in [1,5)** | +0.003 | **+0.009** | **+0.40** | 3,622 | 2,112 | +0.002 | +0.020 | 1,460 | 1,516 |
| S1 | **Gk next clock minute** | −0.004 | **+0.008** | **+0.23** | 4,939 | 795 | +0.039 | −0.040 | 1,462 | 1,580 |
| S1 | Gf floor ≥ 5 · Gx level ≥ $5 · Gu prev close ≥ $5 | — | — | — | **0–8 rejects — INERT** | | | | | |
| S2 | Gp pdr ≥ 8 | +0.049 | +0.038 | +3.34 | 10,073 | 9,301 | +0.095 | **−0.053** | 1,842 | 2,010 |
| S2 | **Gc the cap** | **−0.035** | **−0.041** | **−3.38** | 10,073 | 14,185 | −0.061 | −0.010 | 1,842 | 1,948 |
| S2 | Gv rv in [1,5) | −0.027 | −0.017 | −0.87 | 5,917 | 4,156 | −0.030 | +0.005 | 1,763 | 1,961 |

What the map says, in order:

1. **The scan rule is the biggest real gate in the book, and it is on the wrong side of the shipped
   code.** Keeping only the days whose first break clears the floor is **+0.049 R net (t 2.52)** —
   +0.100 on VAL (t 3.27), +0.014 on TRAIN. The rejected side also carries **1.6x the position
   notional** ($2,310 vs $1,481 median), so the sizer is *amplifying* the worse half: exactly the
   wrong-side gate that step 5's notional column exists to find.
2. **The book's own name — PDR ≥ 8 — is a TRAIN-only effect.** +0.120 R on TRAIN, **−0.013 on VAL**,
   and the same sign flip at 10 and 12. The single rule that Stage H reported as "the one that
   replicated on VAL for every book" does not replicate on VAL *for this book* once the honest
   population and the measured cost are used. That is the headline caveat on the whole F6 program.
3. **The only era-consistent selection gate is the clock.** `signal ≤ 14:00` is +0.090 TRAIN /
   +0.025 VAL (t 4.32) and tightening it to 13:00 costs −0.054 (t −3.05). Late-morning-to-early-
   afternoon is where this pattern lives; nothing else survives both years.
4. **Four shipped knobs are dead or wrong-side.** `range_floor_pct`, `min_price` on the level and
   `universe_min_prev_close` reject **0, 0 and 8** signals at their cascade position — byte-inert.
   `min_adv20` rejects 107. The price ladder is *negative* (−0.009 at $10, −0.028 at $20), the
   opposite of HOD-break, where the price floor was the one gate that worked.
5. **`min_r_pct` is a notional gate wearing a selection gate's clothes.** +2.82 R of separation on 33
   rejects whose median notional is **$17,333** against $1,481 kept — it removes positions the
   $10,500 notional cap would have removed anyway. Its gross separation is large only because the R
   denominator is tiny; it buys no selection.
6. **The rv band the spec computes and never uses is correctly unused**: +0.009 R, t 0.40, flat in
   both years. The r2g spec's decision not to gate on it is right, and can now be said to be right.

## 6 / 7. Frequency frontier, ranked on % GREEN WEEKS (dollars at the live $100 risk)

52 declared cell-scores (26 cells × 2 scan rules), `cells.csv`, cost `M2meas`, no-trade week = FLAT
and in the denominator (TRAIN 53 weeks, VAL 22).

| cell | scan | split | n | /wk | gross R | net R | t | **green %** | red streak | worst wk $ | **total $** | MDD $ | green mo % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **B0 shipped** | **S1** | TRAIN | 1,125 | 21.2 | +0.108 | +0.054 | 1.81 | **58.5** | 3 | −1,046 | **+6,107** | −2,015 | 66.7 |
| **B0 shipped** | **S1** | VAL | 604 | 27.5 | +0.108 | +0.051 | 1.20 | **77.3** | 3 | −1,503 | **+3,083** | −2,838 | 80.0 |
| P1 pdr ≥ 6 | S1 | TRAIN | 1,217 | 23.0 | +0.090 | +0.035 | 1.21 | **62.3** | 3 | −1,351 | +4,223 | −1,606 | 50.0 |
| L2 level ×1.000 | S1 | TRAIN / VAL | 1,122 / 603 | 21.2 / 27.4 | +0.107 / +0.139 | +0.054 / +0.082 | 1.81 / 1.90 | 60.4 / 68.2 | 4 / 2 | −925 / −1,295 | +6,100 / +4,932 | — | — |
| G1c obtainable | S1 | TRAIN / VAL | 1,014 / 567 | 19.1 / 25.8 | +0.115 / +0.096 | +0.066 / +0.044 | **2.15** / 1.01 | 56.6 / 68.2 | 3 / 2 | −989 / −1,157 | +6,732 / +2,501 | −1,589 | 75.0 / 80.0 |
| P4 pdr ≥ 12 | S1 | TRAIN / VAL | 932 / 544 | 17.6 / 24.7 | +0.119 / +0.082 | +0.070 / +0.030 | **2.09** / 0.68 | 56.6 / 63.6 | 4 / 4 | −944 / −1,134 | +6,485 / +1,648 | −1,744 | 66.7 / 60.0 |
| A3 adv20 off | S1 | VAL | 609 | 27.7 | +0.116 | +0.059 | 1.39 | **81.8** | 2 | −1,396 | +3,594 | −2,437 | 80.0 |
| X3 level ≥ $20 | S1 | TRAIN | 585 | 11.0 | +0.024 | −0.007 | −0.20 | 45.3 | 4 | −958 | −419 | −1,500 | 58.3 |
| **C4 ceiling A** | S1 | TRAIN | 2,415 | **45.6** | **+0.032** | −0.015 | −0.86 | 43.4 | 5 | −1,857 | −3,666 | −6,345 | 58.3 |
| **C5 ceiling B** | S1 | TRAIN | 3,086 | **58.2** | **+0.027** | −0.044 | −2.49 | **30.2** | 9 | −1,867 | **−13,557** | −13,580 | 25.0 |
| **B0 shipped** | **S2** | TRAIN | 1,597 | 30.1 | +0.027 | **−0.055** | −1.98 | **39.6** | 4 | −1,445 | **−8,833** | −9,883 | 16.7 |
| **B0 shipped** | **S2** | VAL | 762 | 34.6 | +0.001 | **−0.083** | −1.98 | 54.5 | 3 | −2,684 | **−6,295** | −7,314 | 20.0 |
| **C5 ceiling B** | S2 | TRAIN | 3,901 | **73.6** | +0.017 | −0.086 | −4.77 | **24.5** | 11 | −2,754 | **−33,465** | −33,299 | 8.3 |

(The full 52-row table with every declared cell is `cells.csv` and `score.log` §6/7.)

**The structural ceiling, as the runbook requires**: every optional gate off — no PDR screen, no ADV
screen, 15:30 last entry, 20/8 slots, price floor down to $1, the 5 % causal floor kept because the
universe file requires it — gives **58.2 trades a week on S1, and the raw detector's gross R there is
+0.027 (TRAIN)** against the shipped book's +0.108 at 21.2/week. On S2 the ceiling is 73.6/week at
+0.017 gross. **The frontier falls monotonically as frequency rises, on the GROSS, in both scan
rules** — the same shape QQQ's cadence ladder had and the opposite of a book with unexploited volume.
The edge such as it is lives entirely in the selection.

No cell is top-ranked on green weeks in **both** splits. S1's TRAIN leader is `P1` (62.3 %) and VAL's
is `A3` (81.8 %); `B0` is 4th and 3rd. Nothing in the frontier beats the shipped configuration in a
way that holds across the two open splits.

## 8. Week by week at the live $100 risk (`supp.log` §(c))

**S1 `B0`, VAL (2026-01 → 2026-05), 22 W-FRI weeks, `$ (trades)`:**

```
+33(3) -9(31) +299(24) +958(24) -1503(36) -1325(31) -9(31) +456(21) +1001(28) +400(34) +128(29)
+404(27) -1259(29) +1161(21) +201(29) +955(28) +167(24) +95(30) +50(29) +71(37) +80(34) +723(24)
```
**17 green / 22 (77.3 %), total +$3,082, worst −$1,503, longest red streak 3.** Monthly: −220 / +122 /
+222 / +1,773 / +1,185 — 4 of 5 green, **+$616 a month**.

**S1 `B0`, TRAIN**: 31 green / 53 (58.5 %), total +$6,106, worst −$1,046, 8 of 12 months green,
**+$508 a month**.

**Under `M3entry` (the measured entry leg) the same two paths are +$1,231 and +$354 — $103 and $71 a
month.** That is the whole finding in the dollar path the runbook demands: the ratio (green weeks) is
healthy and stable, and the dollars behind it are inside the cost uncertainty. F7's lesson inverted.

For comparison, S2 `B0` VAL: 12 green / 22, **−$6,295**, worst −$2,684. S1 `C5` (ceiling) TRAIN:
**−$13,557** at 58 trades a week.

## 9. Count-matched permutation null (2,000 draws, per-week pick count held fixed, `nulls.csv`)

| cell | scan | split | observed green % | null mean | [p5, p95] | outside? |
|---|---|---|---|---|---|---|
| B0 | S1 | TRAIN | 58.5 | 58.1 | [50.9, 64.2] | **inside** |
| B0 | S1 | VAL | **77.3** | 59.5 | [50.0, 72.7] | **ABOVE** |
| P4 | S1 | TRAIN / VAL | 56.6 / 63.6 | 59.4 / 55.0 | [52.8, 66.0] / [45.5, 63.6] | inside / inside |
| R2 | S1 | VAL | 72.7 | 58.3 | [50.0, 68.2] | ABOVE |
| C4 | S1 | TRAIN / VAL | 43.4 / 59.1 | 44.7 / 55.9 | [37.7, 50.9] / [45.5, 68.2] | inside / inside |
| B0 | S2 | TRAIN / VAL | 39.6 / 54.5 | 37.9 / 33.0 | [32.1, 45.3] / [22.7, 45.5] | inside / ABOVE* |

**S1 `B0`'s TRAIN week shape is exactly pick count** (58.5 vs a null mean of 58.1). Its VAL week shape
is genuinely above the band — 77.3 % against [50.0, 72.7] — which is the one place in this study where
a week-level result is not explained by how many trades were taken. Of 32 cell × split nulls, 2 S1
cells are favourably above on VAL and **none is above on both splits**.

*The S2 "ABOVE" rows are an artefact worth naming: the permutation preserves the cell's own negative
mean, so the null draws mostly-red weeks and any observed green share sits above the band. For a
negative-mean cell an above-band green share means the losses are concentrated, not that the timing is
skilful.

## 10. Both bars, the authorised TEST read, and the adequacy review

**Claim bar — 2 of 52 cell-scores pass G1** (TRAIN net R > 0, t ≥ 2, ≥ 5 trades/week), both on scan
rule **S1**: `G1c` (t 2.15) and `P4` (t 2.09). Both also pass **G2** (VAL same sign, ≥ 55 % green
weeks: 68.2 % and 63.6 %). Per `FREEZE.md` §1 TEST was therefore opened **for those two cells and
nothing else**:

| cell | split | n | /wk | **gross R** | net R | t | **green %** | worst wk $ | **total $** | MDD $ |
|---|---|---|---|---|---|---|---|---|---|---|
| **P4** | TRAIN | 932 | 17.6 | +0.1190 | +0.0696 | +2.09 | 56.6 | −944 | +6,485 | −1,744 |
| **P4** | VAL | 544 | 24.7 | +0.0819 | +0.0303 | +0.68 | 63.6 | −1,134 | +1,648 | −1,584 |
| **P4** | **TEST** | 388 | 27.7 | **−0.0015** | **−0.0571** | −1.06 | **35.7** | −912 | **−2,217** | −2,527 |
| **G1c** | TRAIN | 1,014 | 19.1 | +0.1146 | +0.0664 | +2.15 | 56.6 | −989 | +6,732 | −1,589 |
| **G1c** | VAL | 567 | 25.8 | +0.0964 | +0.0441 | +1.01 | 68.2 | −1,157 | +2,501 | −1,561 |
| **G1c** | **TEST** | 388 | 27.7 | **−0.0178** | **−0.0713** | −1.40 | **28.6** | −1,065 | **−2,768** | −3,219 |

**Both fail, and they fail on the GROSS, not on the cost.** TEST gross is 0.00 ± 0.05 R against a
TRAIN gross of +0.119 — about 2.4 standard errors apart. This agrees with the prior art it was
pre-committed to be tested against: the reconciliation's L2 (the same first-break rule, an independent
implementation) read **−0.046 R hold / −0.024 R 2R on TEST**, and implementation B read −0.042 / −0.022.
Three implementations, one answer.

**Live-exploration bar — not met.** It needs a positive point estimate on green weeks *and* on dollars
at live size in both open splits, with resolution inside a quarter. S1 `B0` satisfies the first half
under `M2meas` (58.5 / 77.3 % green, +$6,107 / +$3,083) and fails it under the honest `M3entry`
(47.2 % green on TRAIN); at $103/month the dollar resolution inside a quarter is ~$300 against a
per-trade SD of 1 R — nothing resolves. And the sealed split is already negative.

**Adequacy review, answered in writing:**

1. *Did we test what the book actually IS?* **Yes, and the answer is that the engine is not the book.**
   `trading/red_to_green.py` at the `config.yaml red_to_green` block, including the two live universe
   screens (`min_adv20`, `universe_min_prev_close`) that no previous study modelled, on the whole
   point-in-time day list minus test tickers and non-`daily_bars` names. Both scan rules were run as
   declared cells, side by side, at every step. The shipped `detect` is S2; every positive number in
   this report and in the two prior studies belongs to S1.
2. *Is the cost and fill model right for its venue?* **Better than it has ever been, and it moved the
   verdict.** Per-signal SIP NBBO at the fill instant on 82 % of a 2,963-signal stratified sample; the
   band was 1.15–1.21x too NARROW (pre-registered direction, confirmed); the **entry leg was 4x too
   cheap**, worth 0.043 R a trade — the difference between "+$508 a month" and "+$103 a month". The
   fill is the engine's own capped limit and its counterfactual is classified per scan rule (§4). The
   residual optimisms — 17.8 % of signals with no quote at all, 8.7 % above the live 300 bps gate,
   14–18 % of fills on a later print — all push the same way: the reported net is **too kind**.
3. *Does any caveat in our own report explain the headline?* **Yes, one, and it is disclosed as the
   headline**: the whole positive result belongs to a scan rule the engine does not implement, and it
   does not survive the sealed split. The previously known artefacts (ZVZZT, the `BFZ_SLIP=0` level)
   are removed by construction here and are not what is left.
4. *What is the MDE?* Per trade, S1 `B0`: **0.084 R (TRAIN), 0.119 R (VAL)**, ~0.142 R (TEST, n 388).
   On the primary metric: **±19.2 pp of green-week share over 53 TRAIN weeks and ±29.8 pp over 22 VAL
   weeks**, which is why no cell in §6/7 is resolvable against `B0` on green weeks. The test could see
   S1's +0.108 R gross (it did, at t 3.6) and **cannot** resolve the +0.011 R net that survives the
   measured cost — that residual is 8x below the TRAIN MDE. The honest phrasing is therefore: *no net
   edge was detectable in THIS universe, at THIS horizon, at THIS book size, over THIS window, at THIS
   cost; the gross was detectable on the open splits and was absent on the sealed one.*

**Multiplicity.** 26 declared cells × 2 scan rules × 2 open splits = **104 declared cell-scores**, plus
36 gate-map rows, 40 cost-model rows, 32 permutation nulls, 2 reproduction tables and 4 authorised
TEST rows. Expected largest |t| under a pure null over ~104 cells is **3.0–3.2**; the best TRAIN t in
the entire study is **2.15**. Neither gate-passing cell is beyond what multiplicity alone explains —
which is exactly why TEST was the arbiter and not the TRAIN t.

**Tail dependence** (diagnostic, never a rejection reason). S1 `B0` TRAIN net +0.054 → ex-top-1 %
+0.035 → **ex-top-5 % −0.045**; VAL +0.051 → +0.032 → **−0.048**. Removing the best 5 % of trades flips
both open splits negative. S2 `B0` is negative before the tail is touched and gets worse. This book's
open-split gross lives in its right tail.

---

## What would change this verdict, and what to do with the code

Nothing inside this gate set. Two things could, and neither is cheap:

1. **An entry that does not pay a full half-spread.** The measurement in §3 is the binding constraint:
   at 51–59 bps of quoted spread on a book whose R is a few percent of price, the round trip is
   ~0.10 R. A resting limit **at or below** the level (rather than a marketable capped limit above it)
   would change the sign of that term — but §4 shows the fill population is already selected by where
   price goes, and a resting limit at the level is the dip-buy variant the runbook warns about. A new
   pre-registration, not a knob.
2. **A gate with era-consistent GROSS separation.** The only one in the map is `signal ≤ 14:00`
   (+0.090 / +0.025). The book's own PDR ≥ 8 rule is TRAIN-only here (+0.120 / −0.013), which is a
   finding that reaches past this candidate: **program H's "the one rule that replicated on VAL for
   every book" does not replicate for the book it named.**

**Recommended action: NONE.** `red_to_green.enabled: false` stays as the owner set it on 9/17, and the
`--r2g` flag stays inert. Specifically **do not** change `detect` to the first-break rule: it is the
better of the two, its gross is real, and it is still gross-zero on the sealed split at a measured
cost — and the change is not the "one line" it looks like, because the engine calls `detect` once per
closed bar with `start_idx = that bar`, so a `return None` inside `detect` does not retire the
symbol-day. Implementing S1 needs the first level break to set `cand.rejected_reason` across the
`detect`/engine boundary. There is no reason to write it.
