# frames12 — F39 THE MULTI-DAY HIGH · F38 THE LAST HOUR · F37 THE 5-/15-MINUTE BAR — REPORT (2026-09-20)

Pass 12 of the frame programme, and the first in which all three objects are genuinely NEW: F34
showed the bare long bracket is <= 0 in every cell of THIS universe, THIS clock and THIS bar, so a
new object had to change at least one of the three. F39 changes the **population**, F38 the
**clock**, F37 the **bar size**.

`PREREG.md` committed (`3444266`) with the prediction and the falsifier for each object **before any
cell was scored**. TEST never opened (`FREEZE.md`; the one declared exception — the next session's
daily `open` for the 8 overnight trades of the last two VAL sessions — is counted in §1.4).

Artifacts: `c12.py` (the shared exits, cost and reporting) · `w39.py` -> `w39.csv` (50,276 rows),
`p39.py` -> `sig39.csv`, `ctrl39.csv`, `wc.py` -> `w39c.csv` (81,726), `s39.py` -> `s39.log`,
`cells39.csv`, `nulls39.csv`, `sup39.py` -> `sup39.log`, `paired39.csv` · `p38.py` -> `sig38.csv`,
`ctrl38.csv`, `floor38.csv`, `orbpop38.csv`, `w38.py` -> `w38.csv` (427,314), `s38.py` -> `s38.log`,
`cells38.csv`, `floor_cells38.csv`, `nulls38.csv`, `tail5m.log` · `w37.py` -> `w37.csv`
(1,225,431), `s37.py` -> `s37.log`, `cells37.csv`, `nulls37.csv` · `scale12.log` (the price-scale
audit).

One python process at a time, `nice -n 10`, `ulimit -v 3000000`, every walk checkpointed per
session; `cache.db`, `bars_sip.db`, the Databento stores and `daily_bars` opened **read-only**.
Nothing was written outside `frames12/`. No config, `orb.yaml`, systemd unit, cron or order was
touched. `hod_break` stays `enabled: true, dry_run: true`; Monday's 12:30 UTC boot is unchanged.

---

## 0. THE THREE SENTENCES

1. **F39 — the multi-day high is a genuinely different population, and it produces the programme's
   FIFTH both-splits-positive object; but the thing that makes it positive is the OVERNIGHT SESSION,
   which a matched non-signal name captures just as well.** On the PIT panel (prev close >= $17,
   ADV20 >= 100K — **not** conditioned on having moved today), the first 1-minute close above the
   prior 20-session high, held to the **next session's opening auction**, books **+0.079 R / +$5,733
   TRAIN and +0.064 R / +$2,947 VAL at 13.8 / 19.9 trades a week**, 56.6 / 47.8 % green weeks, both
   TRAIN halves same-signed positive. Every intraday exit of the same signal is **negative net on
   VAL**. The paired placebo kills it: against its own matched non-signal control **under the same
   overnight exit** the detector is worth **+0.049 R (day-clustered t +0.71) TRAIN and +0.024 R
   (t +0.28) VAL against an MDE of 0.162 / 0.184** — the control itself books +0.044 / +0.037 R.
   The exit-ordering clause (overnight beats every intraday exit) is **confirmed on both levels and
   both splits**; the magnitude clause is met only because R is wide (per unit of risk the detector
   adds ~+0.03 R).
2. **F38 — the clock gradient does NOT continue into the close, and the frame's own falsifier fires.**
   The floor in **14:00-15:30**, walked first on 116,010 / 76,853 detector-free controls, is
   **-0.416 % of price TRAIN / -0.252 % VAL** at a 2 % stop — indistinguishable from F34's
   13:00-14:01 cell (-0.402 % / -0.255 %), and TRAIN is *worse*. The **closing auction** does what
   RUNBOOK step 3 says: the MOC floor is **+0.017 % / -0.143 %**, and **0.064 / 0.068 pp of that is
   measured cost the auction leg does not pay** — but the rest of the TRAIN gain is an **H1-2025
   outlier tail with a median of -0.006 %** (`tail5m.log`). The detector itself almost vanishes at
   this clock: **2.3 / 3.3 trades a week**, a quarter of the frequency floor.
3. **F37 — a coarser bar changes the frequency and the exit mix exactly as predicted, and changes
   the edge not at all.** The stop share falls monotonically with bar size (**1-min 52 % -> 5-min
   32 % -> 15-min 15 %**) and the frequency floor never binds (16-20 booked trades a week at 15
   minutes, against a predicted failure) — but **all nine coarse cells are negative net on BOTH
   splits**, because the coarser consolidation gives a **tighter stop** (r 2.9-3.1 % vs the
   1-minute book's 3.7-4.1 %) so the cost per unit of risk **rises** (0.068 -> 0.080-0.090 R) while
   the cost in % of price stays flat at 0.22-0.25 %. F34's arithmetic, through a new door.

**Verdict: STAY-DRY on all three. 0 of 31 cells clears the bar.** The engine stays dry; no config,
no flag, no rollback.

---

## 0b. Reproduction gates — asserted in code, raising, before any cell was read

| id | gate | result |
|---|---|---|
| **G-B2** | the reference book: 1,622 / **-$17,346** TRAIN, 706 / **+$893** VAL | **MATCH** (`c12.repro_gate`, raises) |
| **G-WALK** | `common6.walk_from` re-prices 120 booked trades from their own bar and own stop | **max abs(d rr) 2.22e-16** |
| **G-SCALE** | Databento daily close vs Alpaca intraday last 1-min close, 50,276 rows | **median ratio 1.000000 in every one of 17 months**, mean within +/-1.5 bps; 0.74 % of rows off by > 1 % and **dropped** |
| **G-AVAIL** | bar-join coverage on the F39 candidate panel | 40,072 candidate symbol-days -> **50,276 walked level-rows**; 84 % of asked symbols have bars (above the 80 % rail) |
| **G-FREQ** | F37's frequency check, run and printed BEFORE any F37 P&L | see §3.1 |

---

# F39 — THE MULTI-DAY HIGH (11 cells)

## 1.1 The population, and why it is new

Every one of the programme's 1,130 prior cells lived on a population pre-screened to
`day_high >= open x 1.05` and broke **today's** high. F39's population is the whole PIT daily panel
— `prev_close >= $17`, `ADV20 >= 100,000`, test tickers and non-`daily_bars` names out — with **no
same-day move condition at all**. Of 160,802 eligible symbol-days, **40,072 (115 a session)** take
out their prior 5-session high at some point in the day; 19,027 take out the prior 20-session high.

The level is built from daily bars **strictly before** the signal day. The 52-week level is
availability-bound: the PIT panel starts 2025-01-02, so 252 prior sessions do not exist until
2026-01 — **`H252` is VAL-only with 87 signals (3.6/wk)** and is a diagnostic, exactly what
PREREG §2.4 said to expect.

The cascade is the shipped one, unchanged: first close above the level, entry at the **next bar's
open under the 0.6 % cap**, `rv_profile >= 1`, `next_open >= $20`, spread <= 100 bps **and**
<= 15 % of R, obtainable, entry 09:37-14:01, book 12/day 4-concurrent. **50,276 level-rows ->
4,368 (H5) / 2,454 (H20) pre-book signals.**

## 1.2 The eleven cells ($ at $100 risk a trade, the dry-run size)

| # | cell | split | n | /wk | gross R | cost R | **net R** | gross % | **net %** | tc | green (null p95) | worst wk | **total $** | MDD $ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | H5.S20.+2R | TRAIN | 1044 | 19.7 | +0.079 | 0.078 | +0.001 | +0.34 | -0.00 | +0.03 | 47.2 (54.7) | -1405 | **+121** | -4229 |
| | | VAL | 500 | 21.7 | +0.012 | 0.080 | -0.069 | +0.06 | -0.33 | -1.33 | 30.4 (43.5) | -1146 | -3431 | -4681 |
| 2 | H5.S20.lock | TRAIN | 1013 | 19.1 | +0.095 | 0.081 | +0.014 | +0.41 | +0.06 | +0.34 | 43.4 (54.7) | -1276 | +1417 | -4500 |
| | | VAL | 484 | 21.0 | +0.039 | 0.082 | -0.043 | +0.19 | -0.20 | -0.77 | 30.4 (47.8) | -1041 | -2091 | -4220 |
| 3 | H5.S20.bare | TRAIN | 1005 | 19.0 | +0.098 | 0.080 | +0.019 | +0.43 | +0.08 | +0.43 | 45.3 (56.6) | -1367 | +1863 | -4531 |
| | | VAL | 482 | 21.0 | +0.038 | 0.082 | -0.044 | +0.18 | -0.21 | -0.77 | 34.8 (47.8) | -1041 | -2131 | -4335 |
| **4** | **H5.S20.next open** | TRAIN | 1005 | 19.0 | +0.140 | 0.065 | **+0.074** | +0.61 | +0.32 | +1.22 | 49.1 (60.4) | -2177 | **+7459** | -5818 |
| | | VAL | 482 | 21.0 | +0.083 | 0.067 | **+0.016** | +0.40 | +0.08 | +0.19 | 56.5 (56.5) | -1774 | **+773** | -6392 |
| 5 | H20.S20.+2R | TRAIN | 750 | 14.2 | +0.034 | 0.076 | -0.042 | +0.14 | -0.18 | -1.03 | 37.7 (45.3) | -914 | -3185 | -4567 |
| | | VAL | 471 | 20.5 | +0.047 | 0.078 | -0.031 | +0.22 | -0.15 | -0.58 | 43.5 (52.2) | -913 | -1473 | -3219 |
| 6 | H20.S20.lock | TRAIN | 732 | 13.8 | +0.049 | 0.078 | -0.029 | +0.21 | -0.12 | -0.65 | 45.3 (47.2) | -984 | -2155 | -3999 |
| | | VAL | 458 | 19.9 | +0.063 | 0.080 | -0.017 | +0.30 | -0.08 | -0.28 | 43.5 (52.2) | -1003 | -756 | -3194 |
| 7 | H20.S20.bare | TRAIN | 729 | 13.8 | +0.053 | 0.078 | -0.025 | +0.23 | -0.11 | -0.52 | 47.2 (47.2) | -1045 | -1795 | -3428 |
| | | VAL | 457 | 19.9 | +0.058 | 0.080 | -0.021 | +0.27 | -0.10 | -0.36 | 39.1 (52.2) | -1003 | -966 | -3520 |
| **8** | **H20.S20.next open** | TRAIN | 729 | 13.8 | +0.144 | 0.065 | **+0.079** | +0.61 | **+0.34** | +1.15 | 56.6 (58.5) | -1262 | **+5733** | -2393 |
| | | VAL | 457 | 19.9 | +0.129 | 0.065 | **+0.064** | +0.61 | **+0.30** | +0.80 | 47.8 (65.2) | -860 | **+2947** | -3269 |
| 9 | H252.S20.+2R | VAL | 83 | 3.6 | -0.050 | 0.069 | -0.120 | -0.24 | -0.57 | -1.03 | 21.7 (34.8) | -489 | -993 | -1296 |
| 10 | H5.**prev-day low**.+2R | TRAIN | 987 | 18.6 | -0.000 | 0.049 | -0.050 | -0.00 | -0.44 | -2.16 | 39.6 (39.6) | -970 | -4910 | -4910 |
| | | VAL | 424 | 18.4 | -0.019 | 0.052 | -0.071 | -0.16 | -0.60 | -2.16 | 43.5 (39.1) | -746 | -3027 | -3886 |
| 11 | H20.prev-day low.+2R | TRAIN | 811 | 15.3 | -0.014 | 0.044 | -0.057 | -0.14 | -0.55 | -2.32 | 37.7 (35.8) | -1063 | -4648 | -4959 |
| | | VAL | 424 | 18.4 | -0.009 | 0.050 | -0.058 | -0.08 | -0.52 | -1.89 | 30.4 (43.5) | -987 | -2477 | -2676 |

Green-week nulls: **every book cell sits INSIDE its own 2,000-draw count-matched band** except cells
10 and 11, which sit *above* it on one split while being net-negative — pick count, not skill.

## 1.3 The finding: the overnight exit is the book, and the detector is not in it

**Cell 8 is the FIFTH object in 1,161 cells positive-net on both splits at >= 10 trades a week**
(the prior four: `hod_fresh` C1, pass-9 SUPP A, F32-L, F35-A3). It is the first of the five whose
green weeks, worst week and MDD all look respectable at once.

The four-row decomposition says why it must not be believed:

| cell 8 (H20 . next open) | universe bound (F34, intraday) | matched non-signal, SAME minute, SAME overnight exit | same name-day, later minute | **the signal** |
|---|---|---|---|---|
| TRAIN | -0.389 % | **+0.044 R / +0.19 %** (n 6,223) | +0.026 R / +0.11 % (n 6,295) | +0.079 R / +0.34 % |
| VAL | -0.360 % | **+0.037 R / +0.16 %** (n 3,759) | +0.045 R / +0.20 % (n 3,826) | +0.064 R / +0.30 % |

**The matched non-signal control is POSITIVE under the same exit.** F34's -0.389 % is the intraday
floor and is the **wrong bound for an overnight cell**; the right bound is the control, and against
it the detector's paired, day-clustered contribution is:

| cell | split | signal net R | **paired CB diff (tc)** | paired CA diff (tc) | MDE |
|---|---|---|---|---|---|
| 4 H5 . next open | TRAIN | +0.074 | **+0.013 (+0.19)** | +0.036 (+0.87) | 0.145 |
| | VAL | +0.016 | **+0.006 (+0.08)** | +0.011 (+0.22) | 0.195 |
| 8 H20 . next open | TRAIN | +0.079 | **+0.049 (+0.71)** | +0.059 (+1.28) | 0.162 |
| | VAL | +0.064 | **+0.024 (+0.28)** | +0.041 (+0.92) | 0.184 |
| 1 H5 . +2R (intraday) | TRAIN | +0.001 | **+0.090 (+2.42)** | +0.031 (+1.10) | 0.087 |
| 3 H5 . bare (intraday) | TRAIN | +0.019 | **+0.112 (+2.63)** | +0.048 (+1.49) | 0.098 |
| 3 H5 . bare (intraday) | VAL | -0.044 | +0.053 (+0.96) | +0.013 (+0.33) | 0.130 |

**The detector's selection value is real and INTRADAY (+0.09 .. +0.11 R at t +2.4 .. +2.6 on TRAIN,
+0.05 on VAL) and is not big enough to lift its own book above zero; the overnight exit lifts the
book above zero and the detector contributes nothing detectable to it.** Two separate objects, and
the one that pays is not the one the frame was about.

**ex-top-5 %** (uncapped exits only, F35's rule): cell 8 **-0.141 / -0.132**, cell 4
**-0.166 / -0.210** — even the overnight book is tail-carried. On the capped cells the honest form
is `net > 0.05 x 2 / 0.95 = +0.105 R`; none is within reach.

## 1.4 Exit mix, cost and the overnight leg

| cell | stop share | force-close / auction share | cost R | cost % of price | r % |
|---|---|---|---|---|---|
| 1 H5 +2R | 0.30 / 0.28 | eod 0.61 / 0.66, target 0.09 / 0.06 | 0.078 / 0.080 | 0.34 / 0.38 | 4.32 / 4.73 |
| 8 H20 next open | 0.33 / 0.28 | **nextopen 0.67 / 0.72** | **0.065 / 0.065** | 0.28 / 0.31 | 4.27 / 4.72 |
| 10 H5 prev-day low | 0.07 / 0.06 | eod 0.92 / 0.93 | 0.049 / 0.052 | 0.43 / 0.44 | 8.72 / 8.42 |

The overnight leg: **813 of 1,186 cell-8 trades (69 %) are carried**, mean +0.677 R on that leg,
range -8.98 R .. +8.54 R. **8 of them** sit on the last two VAL sessions and read the next session's
daily `open` from a TEST-side date — the single `FREEZE.md` exception, 0.7 % of carried trades.

The prev-day-low stop (cells 10, 11) is **cheaper** per unit of risk (0.044-0.052 R vs 0.078)
because its R is twice as wide — and it is the worst cell in the frame on net, at t -2.16 / -2.32.
A wider stop is not a free option here either: F34 §3.4 point 2, confirmed on a new population.

## 1.5 Prediction verdict (PREREG §2.3 / §2.4)

| clause | stated before | measured | verdict |
|---|---|---|---|
| gross positive in % of price where F34's cell is -0.09..-0.14 % | positive | **+0.14 .. +0.61 % gross, both splits, both levels** | **CONFIRMED** |
| `H20` gross >= **+0.10 % of price** both splits | >= +0.10 | +0.14 % TR / +0.22 % VAL at +2R; +0.61 % at the overnight exit | **CONFIRMED on the level — but on the 4.3 %-stop scale; per unit of risk it is +0.03 R** |
| the **overnight** cell is the BEST of the four exits | best | **best on both levels and both splits** | **CONFIRMED** |
| falsifier: gross <= 0 on either split for both H5 and H20 | — | no | not refuted |
| falsifier: overnight not better than its sibling | — | no | not refuted |
| falsifier: frequency floor fails on both levels | — | no (13.8-21.7/wk) | not refuted |
| `H252` expected to fail the frequency floor | fails | **3.6/wk, VAL only**, availability-bound | **CONFIRMED** |

**The frame is not refuted and no cell clears the bar.** Cell 8 fails on green weeks (VAL 47.8 %),
on day-clustered t (+1.15 / +0.80 against +2), and on the paired placebo — the one that matters.

---

# F38 — THE LAST HOUR (10 cells)

## 2.1 The floor, walked FIRST as the frame required

250,171 detector-free keys (every eligible PIT-panel non-signal name of every session at 2 random
minutes in 14:00-15:30), priced at a fixed 2 % and 3 % stop, plus the MOC variant.

| cell | split | n | gross % | cost % | **net %** | t | tc | H1 % | H2 % | MDE % |
|---|---|---|---|---|---|---|---|---|---|---|
| **U1** floor 2 % bracket | TRAIN | 116,010 | -0.132 | 0.284 | **-0.416** | -107.9 | -7.30 | -0.449 | -0.383 | 0.011 |
| | VAL | 76,853 | +0.033 | 0.285 | **-0.252** | -53.0 | -4.83 | | | 0.013 |
| U2 floor 3 % bracket | TRAIN | 116,010 | -0.121 | 0.277 | -0.397 | -98.3 | -7.01 | -0.419 | -0.377 | 0.011 |
| | VAL | 76,853 | +0.044 | 0.280 | -0.236 | -47.1 | -4.36 | | | 0.014 |
| **U3** floor 2 % **MOC** | TRAIN | 116,010 | +0.237 | 0.220 | **+0.017** | +0.30 | +0.17 | **+0.338** | **-0.298** | 0.160 |
| | VAL | 76,853 | +0.074 | 0.217 | **-0.143** | -27.0 | -2.53 | | | 0.015 |
| *F34's 13:00-14:01, 2 % stop* | TR / VAL | 44,553 | -0.118 | 0.284 | **-0.402 / -0.255** | | | | | |

**The clock gradient stops at 14:00.** U1 is not an improvement on F34's 13:00-14:01 row; TRAIN is
worse. The frame's own falsifier — *"refuted if U1/U2 are not less negative than F34's 13:00-14:01
row on both splits"* — **FIRES**.

**The auction leg is real, and smaller than it looks.** U3 - U1 is +0.433 pp TRAIN and +0.109 pp
VAL, clearing the pre-registered +0.10 pp on both splits. But only **0.064 / 0.068 pp is the
measured cost the closing auction does not pay** (0.284 -> 0.220 % of price, the `eod` leg's 0.412
ratio removed). The remainder is the 15:55 -> closing-auction leg itself, and it is a tail:

```
THE 15:55 -> CLOSING-AUCTION LEG, floor rows that exited at neither a stop nor a target
  TRAIN: n=95,508  mean +0.460 % of price  MEDIAN -0.006 %   t +6.7
  VAL:   n=66,056  mean +0.048 % of price  MEDIAN +0.036 %   t +29.4
  TRAIN H1: mean +0.905 %  median -0.007 %    TRAIN H2: mean +0.035 %  median -0.006 %
```

Median ~ 0 on TRAIN with a mean of +0.46 % living entirely in H1-2025; U3's own halves are
**+0.338 / -0.298**, opposite-signed. **The honest MOC statement: the closing auction is worth the
0.065 pp of spread it does not pay, plus ~ +0.04 % of genuine close drift on VAL — the TRAIN
headline is an outlier tail and must not be quoted.** The price-scale audit (§0b) rules out the
obvious data explanation: daily close and intraday tape agree to a median ratio of 1.000000 in all
17 months.

## 2.2 The detector at this clock — it almost ceases to exist

The shipped detector with only the entry window moved to 14:00-15:30: **26,582 raw last-hour breaks
-> 344 after the shipped cascade** (263 TRAIN = **2.3/wk**, 81 VAL = **3.3/wk**), on 129 of 344
sessions. A quarter of the frequency floor, and it is a fact about the instrument, not the filter:
a late-day 5-bar low is tight, so `r_pct` falls under the 1 % floor and the 15 %-of-R spread gate
bites.

| cell | split | n | /wk | gross R | cost R | net R | gross % | net % | tc | green (null p95) | total $ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L1 last hour +2R | TRAIN | 121 | 2.3 | +0.133 | 0.082 | +0.051 | +0.54 | +0.27 | +0.41 | 28.3 (43.4) | +612 |
| | VAL | 76 | 3.3 | -0.025 | 0.089 | -0.114 | +0.34 | +0.07 | -0.81 | 39.1 (43.5) | -869 |
| L2 bare | TR/VAL | 121/75 | 2.3/3.3 | +0.127/+0.013 | 0.086/0.091 | +0.040/-0.078 | +0.46/+0.59 | +0.18/+0.31 | +0.28/-0.52 | 26.4/34.8 | +489/-586 |
| **L3 next open** | TRAIN | 121 | 2.3 | +0.216 | 0.071 | **+0.144** | +0.87 | +0.63 | +0.53 | 35.8 (41.5) | **+1745** |
| | VAL | 75 | 3.3 | +0.133 | 0.077 | **+0.057** | +0.28 | +0.04 | +0.23 | 34.8 (47.8) | **+424** |
| L4 **MOC** | TR/VAL | 121/75 | 2.3/3.3 | +0.071/+0.042 | 0.071/0.077 | -0.001/-0.035 | +0.27/+0.63 | +0.04/+0.40 | -0.01/-0.21 | 30.2/34.8 | -9/-263 |
| L5 lock | TR/VAL | 121/75 | 2.3/3.3 | +0.159/+0.009 | 0.088/0.092 | +0.071/-0.083 | +0.58/+0.58 | +0.29/+0.29 | +0.52/-0.56 | 28.3/34.8 | +863/-626 |
| O1 ORB 14:00-14:30 range, +2R | TRAIN | 585 | **11.0** | +0.084 | 0.084 | -0.001 | +0.36 | +0.09 | -0.02 | 49.1 (54.7) | -38 |
| | VAL | 308 | **13.4** | +0.053 | 0.083 | -0.029 | +0.14 | -0.14 | -0.51 | 43.5 (52.2) | -903 |
| **O2 ORB range, MOC** | TRAIN | 582 | 11.0 | +0.118 | 0.067 | **+0.051** | +0.50 | +0.29 | +1.01 | 50.9 (58.5) | **+2991** |
| | VAL | 307 | 13.3 | +0.050 | 0.068 | **-0.017** | +0.11 | -0.11 | -0.29 | 47.8 (52.2) | -536 |

The ORB-style object is the only one at this clock clearing the frequency floor (11.0 / 13.4 a week)
and it is **net-negative on VAL** in both exits. Its gross halves are same-signed positive
(H1 +0.118 / H2 +0.118 / VAL +0.050 on O2) — the one era-stable-looking thing here — but the whole
of its TRAIN net is the MOC leg whose median is zero (§2.1), and the green weeks sit inside their
own null on every cell.

Decomposition, L-cells (TRAIN): floor -0.416 %, matched non-signal -0.156 R / -0.489 %, same
name-day later minute -0.052 R / -0.204 %, the signal +0.051 R / +0.272 %. The detector **does**
separate from its control here too — on 121 trades that is an MDE-sized observation, not a finding.

## 2.3 Prediction verdict (PREREG §3.3 / §3.4)

| clause | stated before | measured | verdict |
|---|---|---|---|
| U1 less negative than F34's 13:00-14:01 | yes | **-0.416 vs -0.402 TRAIN (worse), -0.252 vs -0.255 VAL** | **FALSIFIED — the frame's declared falsifier** |
| U3 (MOC) beats U1 by >= 0.10 pp | >= 0.10 | **+0.433 TRAIN / +0.109 VAL** | **CONFIRMED** (only 0.065 pp is cost; the rest is an H1 tail with a zero median) |
| the MOC exit is the BEST of L1-L5 | MOC best | **L3 (next open) is best on both splits; L4 is 4th on TRAIN** | **FALSIFIED** |
| if any last-hour cell is net-positive on both splits it is L4 | L4 | **L3 is, L4 is not** | **FALSIFIED** |

**F38 is refuted on its own terms.** The last hour is not a different buyer in this instrument's
prices; it is the same floor with a quarter of the signals.

---

# F37 — THE 5-MINUTE AND 15-MINUTE BAR (10 cells)

## 3.1 Frequency FIRST, as the frame required

| bar | k | raw TRAIN | /wk | raw VAL | /wk | booked /wk TRAIN | VAL |
|---|---|---|---|---|---|---|---|
| 5 m | 3 | 23,804 | 449 | 14,615 | 635 | 24.5 | 26.3 |
| 5 m | 5 | 19,857 | 375 | 12,444 | 541 | 23.4 | 24.2 |
| 15 m | 3 | 8,874 | 167 | 5,755 | 250 | 18.3 | 19.6 |
| 15 m | 5 | 5,879 | 111 | 3,853 | 168 | 16.2 | 17.6 |

**The prediction that the 15-minute cells would fail the 10-trades-a-week floor is FALSIFIED** —
they book 16-20 a week. Signals per week do fall monotonically with bar size, as predicted.

## 3.2 The nine cells against the 1-minute baseline

| cell | split | n | /wk | gross R | cost R | **net R** | gross % | net % | tc | green (null p95) | total $ | stop share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **B0** 1 m k5 +2R (B2) | TRAIN | 1622 | 30.6 | -0.039 | 0.068 | -0.107 | -0.13 | -0.37 | -2.99 | 32.1 (37.7) | -17,346 | **0.52** |
| | VAL | 706 | 30.7 | +0.083 | 0.070 | +0.013 | +0.34 | +0.07 | +0.26 | 43.5 (60.9) | +893 | 0.46 |
| M1 5 m k3 +2R | TRAIN | 1296 | 24.5 | -0.056 | 0.087 | -0.143 | -0.16 | -0.40 | -4.13 | 26.4 (32.1) | -18,493 | 0.42 |
| | VAL | 606 | 26.3 | +0.015 | 0.086 | -0.071 | +0.04 | -0.22 | -1.25 | 39.1 (43.5) | -4,295 | 0.41 |
| M2 5 m k5 +2R | TRAIN | 1239 | 23.4 | +0.033 | 0.080 | -0.047 | +0.09 | -0.15 | -1.46 | 43.4 (47.2) | -5,807 | **0.32** |
| | VAL | 557 | 24.2 | +0.006 | 0.080 | -0.073 | -0.01 | -0.24 | -1.62 | 39.1 (43.5) | -4,087 | 0.33 |
| M3 15 m k3 +2R | TRAIN | 972 | 18.3 | -0.017 | 0.080 | -0.097 | -0.06 | -0.29 | -2.73 | 32.1 (37.7) | -9,460 | 0.26 |
| | VAL | 450 | 19.6 | +0.035 | 0.080 | -0.044 | +0.13 | -0.10 | -0.93 | 34.8 (47.8) | -2,001 | 0.22 |
| M4 15 m k5 +2R | TRAIN | 859 | 16.2 | +0.045 | 0.075 | -0.029 | +0.14 | -0.08 | -0.87 | 37.7 (49.1) | -2,522 | **0.15** |
| | VAL | 405 | 17.6 | +0.002 | 0.073 | -0.071 | +0.02 | -0.19 | -1.67 | 39.1 (39.1) | -2,875 | 0.15 |
| M5 5 m k5 bare | TR/VAL | 1187/532 | 22.4/23.1 | +0.034/-0.008 | 0.082 | -0.048/-0.090 | +0.08/-0.03 | -0.16/-0.27 | -1.25/-1.82 | 34.0/26.1 | -5,756/-4,771 | 0.33 |
| M6 15 m k3 bare | TR/VAL | 944/440 | 17.8/19.1 | -0.003/+0.038 | 0.082 | -0.085/-0.043 | -0.03/+0.14 | -0.26/-0.10 | -1.94/-0.85 | 37.7/39.1 | -8,019/-1,887 | 0.26 |
| M7 5 m k5 lock | TR/VAL | 1196/534 | 22.6/23.2 | +0.034/-0.012 | 0.083 | -0.049/-0.094 | +0.08/-0.04 | -0.16/-0.29 | -1.31/-2.00 | 35.8/26.1 | -5,827/-5,040 | 0.32 |
| M8 15 m k3 lock | TR/VAL | 947/443 | 17.9/19.3 | -0.008/+0.022 | 0.082 | -0.090/-0.059 | -0.04/+0.09 | -0.28/-0.14 | -2.10/-1.29 | 37.7/39.1 | -8,520/-2,632 | 0.26 |
| M9 5 m k3 bare | TR/VAL | 1214/554 | 22.9/24.1 | -0.026/+0.032 | 0.090 | -0.116/-0.057 | -0.07/+0.10 | -0.32/-0.16 | -2.65/-0.89 | 34.0/47.8 | -14,075/-3,172 | 0.33 |

**All nine are negative net on BOTH splits.** Every green-week figure sits inside its own null band.

## 3.3 The mechanism the frame got right, and the one it got wrong

Right: **the coarse bar really does stop signalling the noise breaks.** The stop share falls
monotonically and hard — 1-min **52 %** -> 5-min k5 **32 %** -> 15-min k5 **15 %** — and the
force-close share rises to match (0.28 -> 0.59 -> 0.81). The object *is* different.

Wrong: that this buys an edge. The consolidation low of a 5- or 15-minute bar sits **closer** to the
break, so `r_pct` falls from **3.65 / 4.06 %** (B0) to **2.9-3.1 %** in every coarse cell. Cost in %
of price is flat at 0.22-0.25 % — F34's finding again — so **cost per unit of risk RISES**
(0.068 -> 0.080-0.090 R) and eats the whole gross improvement. Gross in % of price improves with bar
size on TRAIN (-0.13 -> +0.09 -> +0.14) and **degrades on VAL** (+0.34 -> -0.01 -> +0.02), so the
monotonicity clause fails on the split that matters.

Decomposition (M4, the best coarse cell): universe bound -0.389 %, matched non-signal -0.136 R /
-0.306 %, later minute -0.124 R / -0.255 %, the signal -0.029 R / -0.083 %. The coarse detector
separates from its control by the same ~0.10 R the 1-minute one does, on the same negative pond.

## 3.4 Prediction verdict (PREREG §4.3 / §4.4)

| clause | stated before | measured | verdict |
|---|---|---|---|
| signals/wk fall monotonically with bar size | yes | 449 -> 375 -> 167 -> 111 raw | **CONFIRMED** |
| the 15-minute cells FAIL the 10 tr/wk floor | fail | **16.2-19.6 booked a week** | **FALSIFIED** |
| the stop share falls with bar size | yes | **0.52 -> 0.32 -> 0.15** | **CONFIRMED** |
| gross % improves monotonically with bar size on BOTH splits | yes | TRAIN yes, **VAL no** | **FALSIFIED** |
| falsifier: coarse cells negative like the 1-minute parent | — | **negative net in 9 of 9 on both splits** | **REFUTED — the bar size changes the frequency and the exit mix, not the edge** |

---

## 4. The adequacy review (RUNBOOK step 10)

* **Did we test what the objects actually ARE?** F39: the whole PIT panel, not a mover screen — the
  first population in the programme not conditioned on a same-day move; and the horizon was tested
  *with* the object (four exits including a real overnight hold with the gap charged in full and no
  overnight stop), not assumed. F38: the floor was walked **before** the detector, on 250,171
  detector-free keys, because the frame said its first deliverable was the floor. F37: the detector
  was rebuilt on coarse bars with the decision coarse and the exits on the 1-minute tape — the
  engine's actual geometry, not a coarse-exit approximation.
* **Is the cost and fill model right for these venues?** Two new legs were declared before use and
  both matter: an auction fill pays **no quoted spread**. The intraday legs use the programme's own
  measured-NBBO imputation table, and the imputed share is **97-100 % on every new cell** — stated,
  and load-bearing only for the *level* of a cell; the rankings and the paired placebo differences
  are cost-invariant by construction (the control carries the same imputed spread). Every fill is
  the next bar's open under the 0.6 % cap; a signal that gaps through the cap is a non-fill.
* **Does any caveat in our own report explain the headline?** **YES, twice, and both were found by
  pre-committed checks.** (i) F39's only positive book is the overnight session, and its matched
  non-signal control is positive too — the paired difference is +0.049 / +0.024 R at t +0.71 / +0.28
  against an MDE of 0.16-0.18. (ii) F38's MOC headline is an H1-2025 tail with a **median of
  -0.006 %**; the reliable part is the 0.065 pp of spread the auction does not pay.
* **What is the MDE?** F39 0.087-0.195 R per cell (the paired effects are 3-8x smaller); F38 floor
  **0.011-0.160 % of price** (the floor's negativity is measured to a hundredth of a percent — not a
  power failure), F38 detector cells ~0.5 R at n=121; F37 0.09-0.13 R.
* **Multiplicity.** 31 cells declared in `PREREG.md` before scoring, 31 scored and printed, none
  selected after the fact. Rails rather than cells: the paired placebo table (F39), the
  15:55->auction leg decomposition (F38), the price-scale audit. **Programme cell count
  1,130 + 31 = 1,161.**

## 5. What this pass settles for the programme

1. **An overnight hold is a different instrument from everything the programme has measured, and its
   floor is POSITIVE where the intraday floor is -0.39 %.** A matched non-signal name, bought at a
   random intraday minute and sold into the next opening auction, books +0.037 .. +0.044 R net on
   both splits — the first positive unconditional number in 1,161 cells. It is **not** an edge we
   can claim (it is the overnight risk premium, and this book has no overnight risk budget), but it
   means **every future "does the signal help?" question with an overnight exit must be scored
   against an overnight control, never against F34's intraday floor.**
2. **The closing auction is worth about 0.065 % of price, and no more than that reliably.** Booked
   as a ratio-0 exit leg it removes two-thirds of the exit cost. Anything beyond that in a MOC
   number is a 15:55->16:00 tail whose median is zero: check the median before quoting the mean.
3. **A coarser bar is a frequency and exit-mix instrument, not an edge instrument.** It cuts the
   stop rate by two-thirds and tightens R by the same proportion, so the cost per unit of risk rises
   and the two cancel. Any future "try it on 5-minute bars" must state the `r_pct` it will pay first.

---

## 6. VERDICT

**STAY-DRY on all three objects. 0 of 31 cells clears the bar.** No engine diff is proposed, no flag
is added, nothing is rolled back. `trading/hod_break.py` and `HodBreakEngine` are untouched;
`config.yaml`, `orb.yaml`, the systemd unit, the crons and every order are exactly as the owner left
them.

**The MDE per object**, so the null is quotable honestly:

| object | the smallest per-trade effect this pass could have seen | the effect it read |
|---|---|---|
| **F39** multi-day high | **0.087 - 0.195 R** (0.38 - 0.92 % of price) per cell; **0.16 - 0.18 R** on the overnight paired difference | detector, paired: **+0.049 / +0.024 R** |
| **F38** last hour | **0.011 - 0.016 % of price** on the floor (n = 116,010 / 76,853); ~**0.50 R** on the detector cells (n = 121 / 76) | floor **-0.416 / -0.252 %**; detector cells inside their MDE |
| **F37** 5-/15-minute bar | **0.09 - 0.13 R** per cell | **-0.03 .. -0.14 R**, negative in 9 of 9 |

Phrased as CLAUDE.md requires: **no edge was detectable in THIS universe (the PIT panel, prev close
>= $17, ADV20 >= 100K), at THESE horizons (intraday bracket / static lock / bare stop / closing
auction / next opening auction), at THESE bar sizes (1, 5, 15 minutes), over 2025-01 -> 2026-05, at
a measured 0.22 - 0.44 % of price of cost, at the smallest detectable effects in the table above.**
The one positive object found — the overnight session — is not the detector's, and it is measured to
be the control's as well.
