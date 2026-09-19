# hod_frames — three NEW frames on HOD-break — REPORT (2026-09-19)

Owner 9/19: *"continue and iterate and look at it from scratch every time, with fresh eyes, using
different ways to extract $$$ from this. You won't stop till this happens."* and, mid-pass,
*"short is a valid move if that helps."*

The first 831 cells of this programme were **one frame**: filter the signal minute, shape the exit.
This pass opens three structurally different ones — change the **side** (Frame 1), change the
**admission** (Frame 2), change the **size** (Frame 3). Cells exactly as declared in `PREREG.md`,
**committed `ecf0682` before any cell was scored**. `FRAMES.md` is the standing ledger and now
carries the F5–F8 queue.

Artifacts: `walk.py` → `short.csv` (23,424 short rows over 344 TRAIN+VAL sessions) + `range.csv`
(37,465 symbol-days) · `score.py` → `score.log`, `cells.csv`, `nulls.csv` · `supp.py` → `supp.log` ·
`supp2.log`, `supp3.log`. One python process at a time, `nice -n 15`, `ulimit -v 1500000`/`2500000`;
`cache.db`, `bars_sip.db`, the SIP side store opened **read-only**. No config, `orb.yaml`, systemd
unit, cron, order or cache was written. The dry run was not touched. **TEST was never opened**
(`FREEZE.md`).

---

## VERDICT — **STAY DRY on all three frames**, and the pass closes the last open structure

**0 of 39 declared cells clear the claim bar. 0 clear the live-exploration bar. 0 sit above their
null band on both splits.**

*But the pass is not empty, because Frame 2 was built to test the one structure five passes had
pointed at and never tested — `bf_zero` §6b's **+0.43 R on >=10 %-range days vs −0.55 R on the
rest, in every split** — and the answer is that **the cohort is substantially the trade's own
outcome.** On this book the oracle split is even bigger than §6b's (TRAIN +0.266 vs −0.569 net,
green weeks **71.7 % vs 3.8 %**; VAL +0.293 vs −0.543, **82.6 % vs 0.0 %**) — it would clear every
bar in the programme twice over. A **causal** proxy reaches that cohort with **97.3 % (TRAIN) /
98.8 % (VAL) accuracy** — and captures **+0.102 / −0.076 R** of it, against the oracle's
**+0.777 / +0.273** on the identical population. Membership is nearly perfect; the return is gone.
§2.3 shows why in one line: inside the oracle cohort, the trades whose day was **already** wide by
11:00 read **+0.118 R at 40.9 % WR**, and the trades whose range **arrived after** the signal read
**+0.882 R at 82.0 % WR**. `corr(gross R, range added after the mark) = +0.448 / +0.181`. **A
winning HOD break is what makes the day a >=10 %-range day.** The two-cohort finding is real as a
description and is not an admission rule — and that was the last unexploited structure on the
ledger.*

*Frame 1 (short) is negative in every form: the pure short book reads **−0.080 … +0.032 gross** and
**−0.164 … −0.040 net** at 7.7–26.0 trades a week, with **no** cell same-signed positive across
H1 / H2 / VAL; the mirror-of-D2 SPY-down arm does not save it; the borrowable subset is **worse**
(VAL −0.131 … −0.151 gross). Borrow is NOT the binding constraint here — **62.3 %** of short
signals are `shortable AND easy_to_borrow` today, against halt-resume's 8.1 % — the edge is.*

*Frame 3 (size, frequency preserved) is the most interesting near-miss in the pass and still fails:
all four sizers lift VAL green weeks **43.5 → 65.2 %** and VAL dollars **+$893 → +$3,886 / +$5,605 /
+$6,878 / +$12,821**, and all four leave TRAIN **negative** (−$6,118 … −$13,261). A sizing rule
cannot create an edge; it re-weights one. It re-weights a losing TRAIN year into a smaller losing
TRAIN year.*

---

## 0. Reproduction gate — EXACT

| ref | this pass | verdict |
|---|---|---|
| `B2` shipped, TRAIN | 1,622 · 30.6/wk · −0.039 gross · −0.107 net · 32.1 % green · **−$17,346** | **MATCH** (Δ$ 0) |
| `B2` shipped, VAL | 706 · 30.7 · +0.083 · +0.013 · 43.5 % · **+$893** | **MATCH** |

Cost model unchanged (measured NBBO on 10.5 % of break rows, the declared price-band × hour-band
imputation elsewhere). The break-even used throughout is the **booked-set** cost of `hod_fresh` §2 —
**0.061 TRAIN / 0.065 VAL** — not the retired 0.2151 constant.

---

# FRAME 1 — SHORT the failed break  (26 cells)

**Mechanism.** `hod_losers` §4: the losing long goes +0.45 R by minute 6 and −1.06 R by minute 28,
and the long book cannot monetise that path (the bleed is booked by minute 10; cutting it is worth
+0.012 R). A **short** entered at the confirmation is a different instrument on the same information.

**Construction.** Triggers on closed bars after the long's fill: `a01`/`a00`/`am1` = the long ARMED
(MFE >= +0.4 R) and then closed back at +0.1 R / 0 / −0.1 R within 15 minutes; `fb` = the first
close back below the HOD. Short fills at the NEXT bar's open, obtainable iff that open is
>= trigger close × 0.994 (the mirror of the long's no-chase cap) — **97.0 %** are. Stops: the MFE
high, `level + 0.5 R_long`, `level + 1.0 R_long`. Targets: −1 R_short, −2 R_short, the long's own
stop level. Exit walk is the sign-flipped twin of the long walk (eod → stop → target), same slip.

## 1.1 The books

| cell | TRAIN n | /wk | gross | net | grn % | **$** | VAL n | /wk | gross | net | grn % | **$** | H1/H2/VAL | same-signed + |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **(a) `a01`/mfe/tls** | 94 | 1.8 | +0.196 | +0.126 | 41.5 | **+1,185** | 43 | 1.9 | +0.027 | −0.047 | 39.1 | −202 | +0.102/+0.318/+0.027 | **YES** |
| (a) `a01`/mfe/t2 | 94 | 1.8 | **+0.340** | +0.264 | 35.8 | +2,485 | 43 | 1.9 | −0.146 | −0.226 | 30.4 | −971 | +0.188/+0.536/−0.146 | no |
| (a) `a00`/mfe/tls | 102 | 1.9 | +0.161 | +0.089 | 39.6 | +907 | 37 | 1.6 | +0.009 | −0.059 | 39.1 | −220 | +0.209/+0.107/+0.009 | **YES** |
| (a) `a00`/mfe/t2 | 102 | 1.9 | +0.271 | +0.193 | 35.8 | +1,965 | 37 | 1.6 | −0.017 | −0.092 | 34.8 | −342 | +0.283/+0.256/−0.017 | no |
| (a) `am1`/mfe/tls | 102 | 1.9 | +0.047 | −0.018 | 34.0 | −188 | 36 | 1.6 | −0.012 | −0.079 | 39.1 | −285 | — | no |
| (a) `am1`/mfe/t2 | 102 | 1.9 | +0.205 | +0.132 | 35.8 | +1,347 | 36 | 1.6 | −0.025 | −0.099 | 34.8 | −357 | — | no |
| **(b) `fb`/mfe/tls** | 211 | 4.0 | +0.057 | −0.003 | **45.3** | −59 | 75 | 3.3 | +0.012 | −0.046 | 43.5 | −345 | +0.114/+0.005/+0.012 | **YES** |
| (b) `fb`/mfe/t2 | 211 | 4.0 | +0.140 | +0.074 | **50.9** | +1,556 | 75 | 3.3 | −0.039 | −0.105 | 43.5 | −785 | — | no |
| (c) `fb`/mfe/t1 | 417 | 7.9 | −0.025 | −0.092 | 39.6 | −3,852 | 258 | 11.2 | −0.044 | −0.108 | 39.1 | −2,798 | — | no |
| (c) `fb`/mfe/t2 | 410 | 7.7 | +0.032 | −0.040 | 43.4 | −1,653 | 253 | 11.0 | −0.080 | −0.151 | 30.4 | −3,818 | — | no |
| (c) `fb`/mfe/tls | 420 | 7.9 | −0.015 | −0.080 | 35.8 | −3,362 | 258 | 11.2 | −0.029 | −0.092 | 39.1 | −2,375 | — | no |
| (c) `fb`/h05/t1 | 696 | 13.1 | −0.049 | −0.130 | 37.7 | −9,053 | 483 | 21.0 | −0.019 | −0.098 | 34.8 | −4,755 | — | no |
| (c) `fb`/h05/t2 | 680 | 12.8 | −0.029 | −0.117 | 35.8 | −7,961 | 440 | 19.1 | −0.015 | −0.101 | 34.8 | −4,453 | — | no |
| (c) `fb`/h05/tls | 687 | 13.0 | −0.080 | −0.164 | 32.1 | −11,248 | 475 | 20.7 | −0.009 | −0.090 | 34.8 | −4,298 | — | no |
| (c) `fb`/h10/t1 | 1,319 | 24.9 | +0.015 | −0.052 | 43.4 | −6,924 | 660 | 28.7 | −0.086 | −0.155 | **13.0** | −10,224 | — | no |
| (c) `fb`/h10/t2 | 1,176 | 22.2 | +0.006 | −0.067 | 41.5 | −7,882 | 574 | 25.0 | −0.093 | −0.167 | 21.7 | −9,558 | — | no |
| (c) `fb`/h10/tls | 1,378 | 26.0 | −0.015 | −0.081 | 37.7 | −11,132 | 744 | 32.3 | −0.082 | −0.148 | 17.4 | −11,047 | — | no |
| (d) `fb`/mfe/t2 SPY-dn | 195 | 3.7 | +0.106 | +0.037 | 43.4 | **+722** | 133 | 5.8 | −0.034 | −0.105 | 39.1 | −1,395 | +0.257/+0.006/−0.034 | no |
| (d) `fb`/h05/tls SPY-dn | 328 | 6.2 | −0.114 | −0.199 | 34.0 | −6,517 | 242 | 10.5 | +0.029 | −0.053 | **52.2** | −1,276 | −0.105/−0.121/+0.029 | no |
| (d) `fb`/h10/t1 SPY-dn | 626 | 11.8 | +0.052 | −0.015 | 47.2 | −915 | 314 | 13.7 | −0.031 | −0.097 | 34.8 | −3,060 | — | no |
| (c-borrowable) `fb`/mfe/t2 | 216 | 4.1 | +0.013 | −0.060 | 39.6 | −1,290 | 132 | 5.7 | **−0.151** | −0.225 | 26.1 | −2,969 | — | no |

*(the full 26 × 2 rows are in `cells.csv` / `score.log`; the nine (d) rows and the nine borrowable
rows are all negative-net on at least one split and none is same-signed positive across the halves)*

## 1.2 Borrow, obtainability, cost — the three mechanics the brief asked for

* **Borrow**: of 1,788 distinct short symbols, 1,753 are in today's Alpaca asset list and **1,397
  are `shortable AND easy_to_borrow`**; signal-weighted tradeable share **62.3 %** (names absent
  from today's list — 1.3 % — counted NOT borrowable). This is **7.7×** halt-resume's 8.1 %, exactly
  as the brief predicted for >=$17 / ADV>=100K names. **Borrow is not the binding constraint.** The
  borrowable subset is scored separately and is **worse** than the full set on VAL (gross −0.131 …
  −0.151 vs −0.029 … −0.080) — so the tradeable half of the population is the worse half.
* **Obtainability**: 97.0 % of short triggers fill under the cap. The unfilled counterfactual is
  reported per the runbook's step 4: obtainable rows read **−0.168 (TRAIN) / −0.666 (VAL)** gross
  and un-obtainable rows **−0.023 / −0.019** on `fb`/mfe/tls — i.e. the cap is **protective**, the
  fills we get are the worse population (a gap-down through the trigger is a fill we would rather
  not have). That is the halt-resume signature, not the ORB one.
* **Cost**: the short leg's R is **1.80–1.89 % of price** against the long book's 2.71–2.99 % — the
  short's R is **0.56–0.57×** the long's R, so the same spread is roughly **1.8× more expensive per
  R**. Measured NBBO coverage at the short entry minute is effectively **nil** (`imp` 99–100 %):
  `nbbo.csv` was measured at the LONG signal minutes, and a short triggers 1–30 minutes later.
  **Every Frame-1 net number is therefore an imputed-cost number** and is flagged as such. It does
  not change the verdict — the cells are negative on **gross** on VAL in every (c)/(d) design — but
  a cell that had cleared the bar would have required a dedicated NBBO fetch before anything was
  recommended. None did.

## 1.3 Verdict on Frame 1 — **STAY DRY. The engine build is NOT triggered.**

The only cells that are same-signed positive across H1 / H2 / VAL are `a01`/mfe/tls, `a00`/mfe/tls
and `fb`/mfe/tls, all on the **booked-long** population, all at **1.8–4.0 trades a week** — below
the frequency floor by a factor of three — and all with **negative VAL dollars**. Their TRAIN
green-week share sits **below its own count-matched null mean** (`a01`/mfe/t2: 35.8 vs null 46.3).
The pure short book at a tradeable frequency (13–26/wk) is negative on both splits and its VAL green
weeks collapse to **13–22 %** on the wide-stop designs.

**For the record, had a cell cleared**: the engine has no short path and this is a BUILD, not a
config flip — (1) a `side` field on `HodBreakParams` and a second book in `HodBreakEngine`;
(2) `submit_bracket_order` with `side=sell` and the OCO legs inverted (stop ABOVE, limit BELOW) —
`data_sources/alpaca_client.py` has no short bracket path today; (3) a borrow check at admission
(`TradingClient.get_asset(sym).shortable and .easy_to_borrow`, refreshed daily, fail-closed) —
today's snapshot is not the borrow state on the trade date, so live would need the check at order
time; (4) Reg SHO: the short-sale price test (Rule 201) arms for the rest of the day plus the next
after a −10 % intraday move, and a HOD-break *failure* is exactly the tape that trips it — a
short-sale order would then be **limited to above the NBB**, which is the opposite of our "sell at
the next open under a floor" fill model, so the fill assumption itself would need re-measuring on
the circuit-breaker days before the number could be believed. None of this was built.

---

# FRAME 2 — the noon conditional-mover book  (9 cells + the decider table)

## 2.1 The conditional hit rate — the frame's own pre-committed decider

**P(EOD RTH range >= 10 % | session range >= X % at T)**, on the signal population:

| T | X | signals | **P(EOD >= 10 %)** | base rate | median EOD range |
|---|---|---|---|---|---|
| 11:00 | 6 % | 630 | **80.6 %** | 62.4 % | 13.9 % |
| 11:00 | 7 % | 447 | **89.5 %** | 62.4 % | 15.3 % |
| **11:00** | **8 %** | 339 | **96.2 %** | 62.4 % | 17.1 % |
| 12:00 | 6 % | 330 | 80.6 % | 65.2 % | 14.2 % |
| 12:00 | 7 % | 238 | 88.7 % | 65.2 % | 14.9 % |
| 12:00 | 8 % | 185 | **93.5 %** | 65.2 % | 16.2 % |
| 13:00 | 6 % | 169 | 79.3 % | 68.1 % | 14.0 % |
| 13:00 | 7 % | 115 | 86.1 % | 68.1 % | 15.0 % |
| 13:00 | 8 % | 85 | **89.4 %** | 68.1 % | 15.3 % |

**The pre-committed frame verdict is met on the hit rate and failed on the frequency**: the rule
was *">= 70 % conditional hit rate AND >= 10 trades/week on both splits"*. Every one of the nine
cells clears 70 % (79.3–96.2 %); only `T 11:00 / X 6 %` reaches 10 trades a week on either split
(TRAIN 6.6, VAL 10.0) and none reaches it on both. **The proxy works. The book does not.**

## 2.2 The nine cells

| cell | TRAIN n | /wk | gross | net | grn % | **$** | VAL n | /wk | gross | net | grn % | **$** | H1/H2/VAL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T11:00 X6 % | 350 | 6.6 | −0.038 | −0.102 | 30.2 | −3,566 | 231 | 10.0 | −0.055 | −0.124 | 34.8 | −2,858 | −0.062/−0.015/−0.055 |
| **T11:00 X7 %** | 251 | 4.7 | +0.066 | +0.002 | 41.5 | **+56** | 179 | 7.8 | +0.006 | −0.065 | 30.4 | −1,172 | **+0.039/+0.094/+0.006** |
| **T11:00 X8 %** | 192 | 3.6 | **+0.141** | +0.078 | 47.2 | **+1,496** | 142 | 6.2 | +0.025 | −0.047 | 30.4 | −670 | **+0.104/+0.178/+0.025** |
| T12:00 X6 % | 176 | 3.3 | +0.075 | +0.014 | 32.1 | +254 | 118 | 5.1 | −0.084 | −0.156 | 30.4 | −1,838 | +0.314/−0.170/−0.084 |
| T12:00 X7 % | 134 | 2.5 | +0.125 | +0.062 | 32.1 | +830 | 90 | 3.9 | −0.084 | −0.153 | 30.4 | −1,381 | +0.386/−0.151/−0.084 |
| T12:00 X8 % | 107 | 2.0 | +0.184 | +0.120 | 37.7 | +1,286 | 75 | 3.3 | −0.196 | −0.267 | 26.1 | −2,004 | +0.487/−0.136/−0.196 |
| T13:00 X6 % | 90 | 1.7 | −0.254 | −0.316 | 26.4 | −2,847 | 52 | 2.3 | −0.024 | −0.093 | 43.5 | −481 | — |
| T13:00 X7 % | 65 | 1.2 | −0.145 | −0.209 | 22.6 | −1,360 | 40 | 1.7 | +0.072 | +0.004 | 43.5 | +14 | — |
| T13:00 X8 % | 47 | 0.9 | −0.112 | −0.179 | 22.6 | −841 | 37 | 1.6 | +0.113 | +0.045 | 43.5 | +168 | — |

Two cells (`T11:00 X7 %`, `X8 %`) are same-signed positive in H1, H2 and VAL — the second and third
such cells in 870 — and both sit **inside** their own count-matched null band on both splits
(X8 %: TRAIN 47.2 vs null 45.6 [39.6, 52.8]; VAL 30.4 vs 38.5 [30.4, 47.8]), both are **below
4 trades a week on TRAIN**, and both have **negative VAL dollars**. The frequency cost is not the
range filter, it is the clock: the B2 book has only **23.9 / 17.4 (TRAIN/VAL) signals a week after
11:00**, 16.3 / 7.8 after 12:00 and 12.9 / **3.0** after 13:00 (`supp.log` S2).

## 2.3 **The load-bearing result of the pass: the two-cohort split is the trade's own outcome**

The oracle version of the cohort — the same `>= 10 % EOD range` split `bf_zero` §6b used, which is
**unobservable at the signal** — is even stronger on this book than §6b reported:

| split | cohort | n | gross | net | green % | total $ |
|---|---|---|---|---|---|---|
| TRAIN | EOD range >= 10 % | 1,030 | **+0.266** | **+0.200** | **71.7 %** | **+$20,585** |
| TRAIN | EOD range < 10 % | 592 | −0.569 | −0.641 | **3.8 %** | −$37,931 |
| VAL | EOD range >= 10 % | 469 | **+0.360** | **+0.293** | **82.6 %** | **+$13,754** |
| VAL | EOD range < 10 % | 237 | −0.466 | −0.543 | **0.0 %** | −$12,861 |

That book clears every bar in this programme twice over. So the only question that matters is
whether the cohort can be joined causally — and it can, almost perfectly, **for no return**
(`supp2.log` S7, on the `entry_m > 11:00` population where both selectors are available):

| split | selector | n | gross | net | **P(EOD >= 10 %)** |
|---|---|---|---|---|---|
| TRAIN | none | 1,265 | +0.428 | +0.371 | 63.5 % |
| TRAIN | **ORACLE** `rng_day >= 10 %` | 803 | **+0.777** | +0.722 | 100 % |
| TRAIN | proxy `rng@11:00 >= 8 %` | 113 | **+0.102** | +0.041 | **97.3 %** |
| VAL | none | 400 | −0.089 | −0.152 | 50.5 % |
| VAL | **ORACLE** `rng_day >= 10 %` | 202 | **+0.273** | +0.210 | 100 % |
| VAL | proxy `rng@11:00 >= 8 %` | 81 | **−0.076** | −0.148 | **98.8 %** |

**97–99 % of the membership and 13 % / −28 % of the return.** The mechanism is in `supp3.log` S9 —
split the oracle cohort by *when* the range arrived:

| split | inside the >=10 % cohort | n | gross | **WR** | median range at 11:00 | median EOD range |
|---|---|---|---|---|---|---|
| TRAIN | range **already** >= 8 % at 11:00 | 110 | **+0.118** | **40.9 %** | 10.5 % | 18.3 % |
| TRAIN | range **arrived after** the signal | 693 | **+0.882** | **82.0 %** | 4.0 % | 12.6 % |
| VAL | range already >= 8 % at 11:00 | 80 | −0.063 | 37.5 % | 9.5 % | 15.3 % |
| VAL | range arrived after the signal | 122 | **+0.493** | **58.2 %** | 5.7 % | 12.2 % |

and the correlation says it without a table: `corr(gross R, range added after 11:00)` =
**+0.448 (TRAIN) / +0.181 (VAL)**; the median range added after the mark is **8.66 pp on winners
vs 4.00 pp on losers**. **A winning HOD break is what makes the day a >=10 %-range day.** The
cohort is not a class of days you can join; it is a description of what happened, and most of it
happened *because of the trade*. `bf_zero` §6b's +0.43 / −0.55 stands as a description of the two
populations and **is not an admission rule**, and the programme should stop citing it as the
unexploited structure — after this pass it is exploited, measured, and empty.

*(This does not retract §6b's provenance point. The cache-population look-ahead was real and the
book built on it was void. What is settled here is only the narrower claim that the `>= 10 %-range
day` marker could be proxied into money.)*

---

# FRAME 3 — size by regime, frequency preserved  (4 cells)

Pick set **identical to the shipped book in every cell** — the gross R/trade check holds exactly
(−0.0390 TRAIN / +0.0827 VAL in all four, equal to the base), so nothing here is a selection.

| cell | split | n | /wk | weekly $ | **green %** | worst $ | total $ | MDD $ | red streak |
|---|---|---|---|---|---|---|---|---|---|
| BASE 1.0× | TRAIN | 1,622 | 30.6 | −327 | 32.1 | −2,231 | **−17,346** | −18,929 | 5 |
| BASE 1.0× | VAL | 706 | 30.7 | +39 | 43.5 | −957 | **+893** | −2,940 | 4 |
| S1 1.5up/0.5dn | TRAIN | 1,622 | 30.6 | −221 | 34.0 | −2,294 | −11,732 | −16,322 | 8 |
| S1 1.5up/0.5dn | VAL | 706 | 30.7 | +169 | **65.2** | −1,859 | **+3,886** | −3,173 | 3 |
| S2 2.0up/0.5dn | TRAIN | 1,622 | 30.6 | −250 | 34.0 | −2,883 | −13,261 | −20,301 | 8 |
| S2 2.0up/0.5dn | VAL | 706 | 30.7 | +244 | **65.2** | −2,549 | **+5,605** | −4,082 | 3 |
| **S3 2.0up/0.0dn [ctrl]** | TRAIN | 1,622 | 30.6 | −115 | **43.4** | −2,378 | **−6,118** | −15,999 | 4 |
| **S3 2.0up/0.0dn [ctrl]** | VAL | 706 | 30.7 | +299 | **65.2** | −2,761 | **+6,878** | −3,771 | 3 |
| **S4 rng@12:00>=7 % 2.0/0.5** | TRAIN | 1,622 | 30.6 | −243 | 39.6 | −3,648 | −12,861 | −18,780 | 3 |
| **S4 rng@12:00>=7 % 2.0/0.5** | VAL | 706 | 30.7 | +557 | **65.2** | −1,670 | **+12,821** | −2,568 | 3 |

**Every sizer lifts VAL green weeks from 43.5 % to 65.2 % and multiplies VAL dollars up to 14×.
Every sizer leaves TRAIN negative.** The live-exploration bar requires positive dollars on BOTH
splits; none is close on TRAIN, whose year the book simply loses.

**Nulls** (the sizing key shuffled, pick set fixed, 2,000 draws):

| cell | split | total $ obs | null mean [p5, p95] | green % obs | null [p5, p95] |
|---|---|---|---|---|---|
| S1 | VAL | +3,886 | +869 [−2,108, +3,776] | **65.2** | 50.1 [39.1, 60.9] — **above** |
| S2 | VAL | +5,605 | +1,105 [−3,049, +5,321] | **65.2** | 50.3 [39.1, 60.9] — **above** |
| S3 | VAL | +6,878 | +1,064 [−4,309, +6,393] | **65.2** | 47.6 [34.8, 60.9] — **above** |
| S3 | TRAIN | −6,118 | −17,629 [−27,530, −8,299] | 43.4 | 35.6 [28.3, 43.4] — at p95 |
| **S4** | TRAIN | −12,861 | −29,623 [−34,493, −24,692]¹ | 39.6 | 32.7 [28.3, 37.7] — **above** |
| **S4** | VAL | +12,821 | +1,636 [−1,509, +4,713]¹ | **65.2** | 47.0 [39.1, 56.5] — **above** |

¹ the PREREG declared a **day-level** shuffle and S4's key is **trade-level**; the day-level figure
is in `score.log` and is mis-specified. The row above is the corrected trade-level shuffle
(`supp.log` S3). Reported both ways; the conclusion is identical.

S4 sits above its null band on both splits on dollars and green weeks — and its key is
`rng@12:00 >= 7 %`, i.e. **Frame 2's proxy for the leaking cohort**. §2.3 is the reason it looks
good and the reason it is not a finding: a sizer keyed on "the day is already wide" is keyed on a
variable that correlates with the trade's own outcome through the range it is about to add. S4 is
the same illusion as Frame 2, measured in dollars instead of R — and it still cannot make TRAIN
positive.

---

## BOTH BARS

**Claim bar G1 — 0 of 45 scored cell-rows** (39 declared + the 6 borrowable-subset rows scored as a
robustness arm). No cell has TRAIN net R > 0 with iid **and** clustered t >= 2 at >= 10 trades/week.
The best TRAIN net in the pass at >= 10 trades/week is **−0.015** (F1d `fb`/h10/t1 SPY-down,
clustered t −0.34). G2 was never evaluated; **TEST was never opened** (`FREEZE.md`).

**Live-exploration bar — 0 cells.** It needs positive dollars on both splits at >= 10 trades/week;
the four cells with positive TRAIN dollars and same-signed halves (F1a `a01`/mfe/tls, F1a
`a00`/mfe/tls, F2 `T11:00 X7 %`, F2 `T11:00 X8 %`) run at **1.8–4.7 trades a week** and lose money
on VAL.

**MDE — this is a powered rejection.** Per trade, 80 % power, on net R: **0.052 R (TRAIN) /
0.072 R (VAL)** on the long population (n 4,575 / 2,452), **0.090 / 0.119 R** on the Frame-1 short
population (n 1,802 / 951), 0.208–0.506 R on the Frame-2 subsets (they are small by construction —
that limit IS the frame's answer). Against the true break-even of **0.061–0.065 R**, Frames 1 and 3
are decisive and Frame 2's cells are individually underpowered — which is why Frame 2's verdict
rests on §2.3's **oracle-vs-proxy contrast at n 803 / 202**, not on its nine books.

**Multiplicity.** 39 declared decision cells × 2 splits, plus 6 robustness rows, 1 declared
descriptive table (9 entries) and 10 supplementary diagnostics (S1–S10, no decision attached).
**Programme cumulative: 831 (through `hod_bleed`) + 39 = 870.** (`PREREG.md` §5 wrote "799 → 838";
the `hod_bleed` pass landed its 32 cells in `LOG.md` between the PREREG commit and this report —
the corrected count is 870.) Expected largest |t| under a pure null over 39 × 2 ≈ 2.9–3.1; the
largest positive TRAIN t in the pass is **+1.91** (F1a `a01`/mfe/t2, at 1.8 trades a week).

---

## WHAT THIS PASS ADDS THAT THE 831 CELLS DID NOT

1. **The `bf_zero` §6b two-cohort structure is closed.** It is 97–99 % reachable causally and worth
   **+0.102 / −0.076 R** when reached, against an oracle **+0.777 / +0.273** on the same population.
   Inside the cohort, the trades whose range arrived *after* the signal carry 82 % WR and +0.882 R;
   the ones that were already wide carry 40.9 % WR and +0.118 R. **The cohort marker is downstream
   of the trade's outcome.** Every future study that uses an end-of-day aggregate (range, volume,
   dollar volume, high) as a cohort label must run this same already-there / arrived-after split
   before the label is allowed into a rule.
2. **The short side is measured on HOD for the first time, and the borrow objection does not apply
   here** — 62.3 % of short signals are `shortable AND easy_to_borrow` (halt-resume: 8.1 %). The
   short book is negative anyway, on gross, on VAL, in every design, and the **borrowable subset is
   the worse half**. Short-side capacity was never the reason this book has no edge.
3. **A sizing rule is not a source of edge and the null proves it cleanly.** All four sizers lift
   VAL green weeks to 65.2 % and sit above their null bands on VAL — and all four leave TRAIN
   negative, because re-weighting a losing year re-weights it. This is the cleanest demonstration in
   the programme that the owner's primary metric can be moved without any edge being present, and
   it is why green weeks are always reported against a count-matched **and** key-matched null.
4. **The no-chase cap is protective on the short side, unlike on the long side.** Obtainable short
   fills read −0.168 / −0.666 R and un-obtainable ones −0.023 / −0.019 — the fills we get are the
   worse population. That is the halt-resume dip-buy signature appearing in a different book.

## WHAT WOULD CHANGE THE VERDICT

The three frames failed for *different* reasons, and only one of them is informative about where to
look next. Frame 1 failed on edge. Frame 3 failed on arithmetic. **Frame 2 failed on timing**: the
causal proxy for a real +0.8 R separation only becomes available at 11:00, by which time the book
has 23.9 / 17.4 signals a week left and the separation it can still reach is +0.102 / −0.076. A
field carrying the same information **at the signal minute** — available on every signal at every
hour — is the only direction this pass produced. `dist_open_pct`, the one causal field we already
have that ranks cohort membership (Q1 40.5 % → Q5 85.0 % of days ending >= 10 %), is **flat on gross
in both years** (TRAIN −0.052 … +0.042; VAL −0.041 … +0.109, `supp2.log` S8) — so the obvious
version of that field is already known to be empty, and the next frame has to find a different one.

## VERDICT — **STAY DRY.** No `HodBreakParams` change, no sizing change, no short path built.

`config.yaml hod_break` stays exactly as the owner set it (`enabled: true, dry_run: true`);
`trading.enabled` and `orb.yaml` untouched. There is no SHIP-TO-DRY diff to write.

**The next three frames are F5, F6 and F9 in `FRAMES.md`** (F7 and F8 stay queued behind them):

* **F5 — the RETEST book** (owner-queued). Admission = the SECOND HOD break of the day only, after a
  first break that failed. Mechanism: the first break's failure flushes the stops above the level
  and the impatient longs, and the retest is the confirmation. Ranked first because it is the only
  queued frame that changes what a *signal* is rather than what is known about one, and
  `hod_losers` §6 already measured re-breaks as the better side on three of four base × split
  (+0.113 / +0.236).
* **F6 — ABSORPTION at the level** (owner-queued). Volume traded within ±0.5 % of the HOD in the
  bars that formed it, as a share of ADV. Mechanism: a heavy shelf = supply cleared before the
  break. This is the plausible causal content of `hod_fresh`'s `consol_bars >= 20` control — the
  only admission in 870 cells same-signed positive in H1, H2 and VAL — never measured directly.
* **F9 — the signal-minute cohort field** (NEW, from §2.3). Frame 2 proves a +0.8 R separation
  exists and is reachable only too late. The frame: find a field computable **at the break bar**
  that ranks "this day will keep expanding" without being downstream of the trade — candidates are
  cumulative $ volume vs ADV$ at the signal (F8, which should run *inside* F9 rather than
  separately), the count of prior 5-minute range expansions in the session, and the ratio of range
  added in the last 30 minutes to range added in the first 30. Every candidate must pass the
  already-there / arrived-after split of §2.3 before it is scored as a rule.

**Recommended action: NONE.**
