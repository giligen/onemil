# frames11 — F36 THE POOLED INSTRUMENT FED + THE BF BAND'S R BASIS · F35 THE TAIL IS THE BOOK · F34 PRICE THE UNIVERSE — REPORT (2026-09-20)

Pass 11 of the frame programme. `PREREG.md` committed (`620f36d`) before any F35/F34 cell was
scored; the F35 amendment (the admission-cell spec and the circularity check) was committed before
either admission cell was read. TEST was never opened (`FREEZE.md`).

Artifacts: `trading/ramp_bt_band.py` + `trading/ramp_pool.py` + `scripts/hod_break_eod_check.py` +
`scripts/bf_ramp_check.py` + `docs/scaling_plan_2026.md` + `tests/test_bf_band_r_basis.py` +
`tests/test_ramp_pool.py` (F36) · `s35.py` → `cells35.csv` · `s35b.py` → `cells35b.csv`,
`admit35.csv` · `s35c.py` → `cells35c.csv` · `s35d.py` → `admit35c.csv` · `w34.py` → `w34.csv`,
`w34.log` · `s34.py` → `cells34.csv`, `s34.log`.

One python process at a time, `nice -n 10`, `ulimit -v 3000000`, the walk checkpointed per session;
`cache.db`, `bars_sip.db`, the Databento stores, `daily_bars` and `trades.db` opened **read-only**.
The only file written outside `frames11/` is `data/hod_dry_pool.csv` (F36's producer, gitignored).
No config, `orb.yaml`, systemd unit, cron or order was written.

---

## 0. THE THREE SENTENCES

1. **F36 — the instrument feeds itself, and the BF gate was measuring the wrong thing.** One call in
   the EOD check the daily cron already runs now appends the session's dry book to
   `data/hod_dry_pool.csv`, idempotent on (day, symbol, entry minute); backfilled over the four dry
   sessions it reproduces F33's replay exactly — **31 trades, pooled z −0.099 ± 0.300 on n=31, band
   [p5 −0.34, p10 −0.28, p90 +0.18] → IN-BAND**. And BF's BT band, which compared a
   **share-multiplied** backtest (`pnl/$2,000`) with a **flat-risk** live book, now divides by each
   BT trade's own risk: the reference mean R falls **+1.242 → +0.701** (TRAIN **+1.651 → +0.912**,
   F31's O6 row to the third decimal), and at a completed stage's n=30 the bar moves from
   `[p5 +0.29, p10 +0.50, p90 +1.99]` to `[p5 +0.23, p10 +0.32, p90 +1.07]` — a live book earning
   +0.40 R at flat risk reads IN-BAND where it read BELOW-p10.
2. **F35 — on the book we actually ship there is no tail to identify, and on the uncapped book there
   is exactly one weak finder.** B2 is **+2 R capped**: its "top 5 %" is 81 of the 330 trades that
   hit the cap, **every one at rr = +2.000 exactly**, so the ex-top-5 % clause on a capped book is
   not a tail test — it is a mechanical **−0.104 R toll** (0.05 × 1.98 ÷ 0.95), which reproduces
   B2's ex-top-5 % of −0.217 from its net of −0.107 to the third decimal. Both fields that
   "separated" that pseudo-tail collapse under the pre-committed circularity check: **`spread/R`
   goes from t −10.38 / −7.50 on the net label to t −0.11 / +0.39 on the gross label** — it was the
   cost model ranking inside a point mass. On the **uncapped** G3 book (a real tail: mean +5.16 R,
   up to +17.2 R) **one field of eleven separates it on BOTH labels and both splits — `rv_profile`,
   and the direction is LOWER relative volume** (−3.63 TRAIN t −4.66; −3.92 VAL t −2.43). Its
   admission cell is the **fourth** object in the programme positive-net on both splits at
   ≥ 10 tr/wk (+0.027 / +0.038 R, +$3,449 / +$2,324, both TRAIN halves positive) and it fails
   everything else: clustered t **+0.38 / +0.47** against an **MDE of 0.19 / 0.24**, green weeks
   **below its own count-matched null on both splits**, ex-top-5 % still **−0.306 / −0.220**.
3. **F34 — the floor under every admission rule this programme can write is NEGATIVE EVERYWHERE, and
   the cost is why.** The unconditional long bracket on the PIT HOD universe, at a fixed 2/3/4 %
   stop, priced in % of entry price on **285,485 detector-free controls: 0 of 39 scored rows is
   positive** — not one cell, not one stop width, not one era, not even on GROSS. Best gross cell:
   ADV$ ≥ $150M at a 2 % stop, **−0.051 % of price**; best net cell: 11:30–13:00 at a 2 % stop,
   **−0.297 % TRAIN / −0.331 % VAL**. The **measured cost is 0.24–0.31 % of price and barely moves
   with the stop width** (F31's arithmetic in reverse), so the instrument would need a drift five
   times its best observed value merely to break even. The map's *ordering* is real and not
   multiplicity (later clock > early, ADV$ ≥ $150M > small, common > wrapper; the best cell sits
   outside the 13-cell permutation p95 at all three stops) — but every level is below zero.

**Verdict: F36 SHIPPED (code + tests, no config touched) · F35 NO SHIP, the sentence is "the
monsters are not identifiable at entry on this book" · F34 NO SHIP, the floor is ≤ 0 in every cell
of the map.** `hod_break` stays `enabled: true, dry_run: true`; `config.yaml trading.enabled` and
`orb.yaml` are exactly as the owner set them; Monday's 12:30 UTC boot is unchanged.

---

## 0b. Reproduction gates — asserted in code, raising, before any number was read

| id | gate | result |
|---|---|---|
| **G-B2** | `common6.base_book()` — 1,622 TRAIN / **−$17,346**; 706 VAL / **+$893** | **MATCH** (`s35.repro_gate`) |
| **G-WALK** | `common6.walk_from` handed the booked stop must reproduce `book6.rr` | **max \|diff\| 2.22e-16** on 120 booked trades (`w34.repro_gate`) |
| **G-G3** | the F25 `G3` walk joins the admitted signal set | **7,027 / 7,027 = 100 %** |
| **G-POOL** | the backfilled dry pool must reproduce F33's replay | 11 / 7 / 7 / 6 on 9/14, 9/16, 9/17, 9/18; **31 total, z −0.099 vs F33's −0.100** |
| **G-BF** | `bf_frequency/runs/P1.csv` 56 trades, TRAIN n=34 | **MATCH**; the fixed basis reproduces F31's +0.9118 / +0.5556 |
| **G-AVAIL** | arm-d controls with a usable bar join | **285,485 / 288,174 = 99.1 %** |

---

# F36 — THE POOLED INSTRUMENT, FED, AND THE BF BAND'S R BASIS FIXED (code; 0 cells)

## 1.1 The producer — one call, and the pool feeds itself from Monday

`scripts/hod_break_eod_check.py` appends the session's **DRY-RUN EXECUTABLE book** — the engine's
own logged `[HOD DRY] WOULD BUY` signals, filled at the next open under the logged limit, walked
forward on the day's bars, cut by `trading.hod_break.run_book(12, 4)` — through
`trading.ramp_pool.append_dry_trades`.

**It is the engine's book, not the spec's, and that was a correction made during the build.** The
first wiring hooked the *spec-on-REST-bars* book; on 2026-09-18 that book has **zero** trades while
the engine's has six (+9.4 R), so the pool would have been silently empty on the best dry session so
far. F33's replay parsed the engine's book, and now so does the producer — verified session by
session:

| | 9/14 | 9/16 | 9/17 | 9/18 | total |
|---|---|---|---|---|---|
| F33's replay (`r33.log`) | 11 tr −4.6R | 7 tr −4.2R | 7 tr −4.5R | 6 tr +9.4R | 31 |
| the producer, backfilled | **11** | **7** | **7** | **6** | **31** |

**Idempotence** is on `(day, symbol, entry_minute)` and is proven live, not only in tests: re-running
the 9/18 check appends `0 of 6` and leaves the file byte-identical. Schema
(`day, symbol, entry_minute, r`) is documented in `ramp_pool`'s module docstring; the file is
gitignored; legacy 3-column files still append and read (at WARNING, key degraded to day+symbol).

Reading it back through the real checker:

```
POOLED z = -0.099 +/- 0.300 on n=31 (hod_dry 31; dry 31 of 31, 4 sessions) vs pooled BT band
[p5 -0.34, p10 -0.28, p90 +0.18] [2328 BT trades, 2000 draws] -> IN-BAND (ADVISORY, ...)
```

Guards kept and tested: LIVE mode writes nothing (a live HOD book enters through `trades.db`); the
red-to-green book is not a pooled book; a failing pool write returns a message instead of breaking
the EOD check the owner's cron runs.

## 1.2 The BF band's R basis — the defect frames10 F31 §1.4.2 found, fixed

**What was wrong.** The BF BT band divided by a constant $2,000. That is not a per-trade R: the
reference book's `shares` already carry the conviction, MACD-zone and regime multipliers, so the
average BT trade risks ~1.8× its nominal base. The live side divides by a **flat**
`trading.risk_per_trade` ($150 at L0). The gate was asking a flat-risk live book to clear a bar set
by a multiplied one.

**Column audit, as pre-committed.** `bf_trade_risk_usd` prefers `risk_per_share × shares`, then
`shares × |planned_entry − stop_loss|`, then `shares × |entry_price − stop_loss|`. On `P1.csv`:
`risk_per_share` is **ABSENT**; `planned_entry` exists as a column but is **EMPTY on all 56 rows**;
`shares` / `entry_price` / `stop_loss` are present on all 56 — **so the file resolves to the third
form**, and a test asserts each of those three facts so the band shifts loudly if `planned_entry`
is ever backfilled.

**Old vs new, side by side** (printed once per run by `scripts/bf_ramp_check.py` until 2026-09-27):

| basis | BT mean R (56 tr) | TRAIN (34) | VAL (22) | SD | band at n=30 | a +0.40 R flat-risk live book |
|---|---|---|---|---|---|---|
| **LEGACY** `pnl/$2,000` | **+1.242** | **+1.651** | +0.610 | 3.151 | [p5 +0.29, p10 +0.50, p90 +1.99] | **BELOW-p10** |
| **FIXED** `pnl/(shares × \|entry−stop\|)` | **+0.701** | **+0.912** | +0.376 | 1.574 | [p5 +0.23, p10 +0.32, p90 +1.07] | **IN-BAND** |

The fixed basis reproduces F31's price-consistent +0.9118 / +0.5556 to the fourth decimal and equals
`pnl_pct / stop_pct` row by row to the CSV's own 2-dp precision (a test asserts that identity on all
56 rows).

**Honest caveat, pre-committed and reported.** At small n the LEGACY band is so wide (SD 3.151) that
it classifies almost anything as IN-BAND — its failure at n=8 is that it *cannot reject*, not that
it rejects wrongly. The classification flip is therefore demonstrated at n=30, the n of a completed
stage, and the test docstring says so rather than quietly picking the flattering n.
`BOOK_SD_FALLBACK['bf']` moved 3.151 → 1.574 for the same reason (a band and an SD from different
distributions is the defect class this house keeps shipping).

**Rollback.** `BF_BAND_R_BASIS=notional` reproduces every band printed before today, shouting at
WARNING each time; it is removed after 2026-09-27. `docs/scaling_plan_2026.md` Gate 2 item 4 carries
the whole story.

---

# F35 — THE TAIL IS THE BOOK (26 scored cells)

## 2.1 The population and the label

`hod_frames6/book6.csv` — the B2 booked book, 2,328 trades (TRAIN 1,622 / VAL 706). TAIL = the top
5 % by `net` within its own split, rank-trimmed exactly as the programme's clause is: **81 TRAIN,
35 VAL**.

| split | tail mean net | rest | book | the tail carries |
|---|---|---|---|---|
| TRAIN | **+1.982 R** | −0.217 | −0.107 | **+$16,055 of −$17,346** |
| VAL | **+1.980 R** | −0.090 | +0.013 | **+$6,929 of +$893** |

## 2.2 The finding that reframes the question: **there is no tail on the shipped book**

| split | tail exit mix | share at rr = +2.000 | book's target share | tail net range | best non-tail net |
|---|---|---|---|---|---|
| TRAIN | **target 81 of 81** | **100 %** | 20.3 % | +1.973 … +1.993 | +1.972 |
| VAL | **target 35 of 35** | **100 %** | 21.2 % | +1.970 … +1.996 | +1.969 |

B2 is **+2 R capped**. Its top 5 % is not a tail — it is 81 of the 330 trades that hit the cap, all
with the identical market outcome, and the net label ranks *inside that point mass by COST*. Three
consequences, all arithmetic:

1. **The ex-top-5 % clause on a capped book is a fixed toll, not a test.** Removing 5 % of trades
   each worth +1.98 R moves the mean by `0.05 × 1.98 ÷ 0.95 = 0.104 R` however skilful the book is.
   B2: net −0.107 → ex-top-5 % **−0.217** (predicted −0.2165). **Any capped book netting less than
   ≈ +0.10 R fails the clause by construction** — `hod_fresh` C1 (+0.033 / +0.050) and pass-9 SUPP A
   could never have passed it, whatever their merit.
2. **The clause was still right to kill them**, but for a reason nobody had written down: their
   whole net edge is *smaller than a twentieth of their own cap*.
3. **F32-L is the exception and is unaffected** — it lives on the **uncapped** G3 geometry, where
   the trim removes a real right tail, so its −0.301 / −0.211 is a genuine tail reading.

## 2.3 The eleven cells on the capped book, and the circularity check that voids both passes

| cell | field | TRAIN diff (t) | VAL diff (t) | null p TR/VAL | **on the GROSS label** | verdict |
|---|---|---|---|---|---|---|
| **C1** | `rv_profile` | **−2.564 (−2.39)** | **−4.700 (−3.97)** | 0.034 / 0.002 | −1.650 (**−1.34**) / −3.143 (−2.01) | passes on net, **fails on gross** |
| **C3** | `spread_over_r` | **−0.054 (−10.38)** | **−0.054 (−7.50)** | 0.000 / 0.000 | **−0.002 (−0.11) / +0.008 (+0.39)** | **pure cost artefact** |
| C2 | `dollar_frac` | +3.564 (+0.45) | −17.305 (−0.96) | 0.388 / 0.643 | −6.771 (−1.01) / −14.266 (−0.64) | no |
| C4 | `dist_open_pct` | −0.175 (−0.18) | −0.604 (−0.58) | 0.830 / 0.103 | −0.481 (−0.45) / −0.975 (−0.79) | no |
| C5 | `gap_pct` | +4.765 (+0.90) | +0.781 (+0.49) | 0.099 / 0.645 | −3.490 (−2.24) / −1.639 (−0.71) | no, and it flips sign |
| C6 | wrapper | +0.169 (+1.57) | +0.040 (+0.38) | 0.100 / 0.696 | +0.169 (+1.22) / +0.010 (+0.07) | no |
| C7 | log price | −0.099 (−0.48) | −0.015 (−0.05) | 0.006 / 0.839 | +0.039 (+0.13) / +0.065 (+0.17) | no |
| C8 | entry minute | +30.6 (+0.35) | +20.6 (+0.21) | 0.001 / 0.059 | +10.0 (+0.09) / +7.7 (+0.06) | no |
| C9 | `spy_r5_pct` | −0.028 (−1.43) | −0.012 (−0.67) | 1.000 / 1.000 | −0.011 (−0.32) / −0.026 (−1.06) | no |
| C10 | `r_pct` | −0.852 (−1.75) | −1.127 (−1.29) | 0.005 / 0.017 | −0.671 (−1.20) / −1.062 (−1.47) | no |
| C11 | log ADV$ | +0.468 (+0.42) | +0.635 (+0.48) | 0.000 / 0.000 | +0.016 (+0.01) / +0.299 (+0.17) | no |

The net-label and gross-label tails overlap on only **22 %** of trades — itself the point mass
speaking: among 330 identical +2 R outcomes the net ranking IS the cost ranking.

Banded diagnostics (the tail's share per level; 5.0 % expected): the only visible structure is the
clock (TRAIN 3.2 % in 09:37–10:30 vs 9.7 % / 9.3 % midday; VAL 3.6 % vs 12.0 %) and ADV$
(TRAIN 1.4 % / 3.9 % / 8.2 % across the three bands; VAL 2.2 % / 2.0 % / 9.6 %) — neither clears the
decision rule as a continuous field.

**The two admission cells, scored anyway because the PREREG said to:**

| cell | split | n | /wk | net | $ | green (null p95) | t | ex-top-5 % |
|---|---|---|---|---|---|---|---|---|
| B2 reference | TRAIN | 1,622 | 30.6 | −0.107 | −$17,346 | 32.1 (76.9) | −2.99 | −0.217 |
| | VAL | 706 | 30.7 | +0.013 | +$893 | 43.5 (60.9) | +0.26 | −0.090 |
| **A1** `rv ≤ 3.75` | TRAIN | 1,384 | 26.1 | −0.089 | −$12,282 | 41.5 (84.6) | −2.13 | −0.198 |
| | VAL | 707 | 30.7 | **−0.058** | −$4,101 | 39.1 (47.8) | −1.05 | −0.164 |
| **A2** `spread/R ≤ med` | TRAIN | 1,307 | 24.7 | −0.118 | −$15,444 | 32.1 (76.9) | −2.82 | −0.228 |
| | VAL | 626 | 27.2 | +0.016 | +$998 | 56.5 (60.9) | +0.28 | −0.087 |

Neither is a book: A2's VAL +0.016 is inside the reference's +0.013, A1 is negative on both splits.

## 2.4 The supplement: the same question on the UNCAPPED book, where a real tail exists

`G3` = the bare stop ridden to 15:55, no target — the geometry the tail-killed objects live on.
Walk joined on **7,027 / 7,027** admitted signals; the geometry's own exit minutes decide slots.

| split | book n | gross | net | rr max | share at +2R | tail n | tail mean net | ex-top-5 % |
|---|---|---|---|---|---|---|---|---|
| TRAIN | 1,434 | +0.014 | −0.056 | **+17.22** | 10.0 % | 72 | **+5.157 R** (+3.07 … +17.20) | −0.331 |
| VAL | 609 | +0.094 | +0.022 | **+7.67** | 10.3 % | 30 | **+4.260 R** (+2.79 … +7.63) | −0.198 |

Net-label and gross-label tails now overlap **100 %** — a genuine tail, and both labels find the
same trades.

**Of the same eleven fields, exactly ONE separates it, on both labels and both splits:**

| cell | field | TRAIN diff (t) | VAL diff (t) | null p TRAIN | verdict |
|---|---|---|---|---|---|
| **C1G** | **`rv_profile`** | **−3.625 (−4.66)** | **−3.917 (−2.43)** | **0.000** | **SEPARATES (net and gross)** |
| C10G | `r_pct` | −1.174 (−2.43) | −0.683 (−0.88) | 0.000 | no (VAL \|t\| < 1) |
| C6G | wrapper | +0.191 (+1.66) | +0.141 (+1.04) | 0.110 | no |
| C5G | `gap_pct` | −1.757 (−1.66) | −1.744 (−1.17) | 0.228 | no |
| C3G | `spread_over_r` | −0.000 (−0.02) | +0.004 (+0.20) | 0.955 | no — the capped book's "winner" is gone |
| C2G · C4G · C7G · C8G · C9G · C11G | — | \|t\| ≤ 1.6 | \|t\| ≤ 1.6 | — | no |

**The direction is the surprise, and it is against the engine's own gate**: the monsters have LOWER
relative volume — the tail averages ~3.6–3.9 units of `rv_profile` below the rest, on a book whose
admission band is `rv ∈ [1, 5)`. A quiet break runs further than a crowded one.

**The earned admission cell** (parameter-free median cut on the TRAIN book, re-booked with
`run_book(12, 4)`; the tercile is one declared robustness read, not a search):

| cell | split | n | /wk | gross | net | $ | green (null p95) | t | ex-top-5 % | MDE |
|---|---|---|---|---|---|---|---|---|---|---|
| G3 baseline | TRAIN | 1,434 | 27.1 | +0.014 | −0.056 | −$7,969 | 41.5 (92.3) | −0.97 | −0.331 | 0.155 |
| | VAL | 609 | 26.5 | +0.094 | +0.022 | +$1,337 | 56.5 (60.9) | +0.32 | −0.198 | 0.207 |
| **A3 `rv ≤ 3.75`** | TRAIN | 1,295 | 24.4 | +0.088 | **+0.027** | **+$3,449** | 41.5 (111.5) | +0.38 | −0.306 | 0.189 |
| | VAL | 609 | 26.5 | +0.105 | **+0.038** | **+$2,324** | 56.5 (60.9) | +0.47 | −0.220 | 0.235 |
| A4 `rv ≤ 2.61` | TRAIN | 1,060 | 20.0 | +0.079 | +0.016 | +$1,748 | 37.7 (107.7) | +0.20 | −0.312 | 0.200 |
| | VAL | 569 | 24.7 | +0.037 | **−0.030** | −$1,710 | 43.5 (52.2) | −0.38 | −0.274 | 0.231 |

TRAIN halves for A3: **H1 +0.022 / +$1,361, H2 +0.031 / +$2,088** — same-signed, which none of the
programme's earlier both-splits objects managed.

**A3 is the FOURTH object in 1,130 cells positive-net on both splits at ≥ 10 trades a week.** It
fails the claim bar on every other axis: clustered t **+0.38 / +0.47** (needs ≥ +2) against an
**MDE of 0.189 / 0.235** — roughly **six times less resolution than the effect it reads**; green
weeks **41.5 / 56.5 against their own count-matched null p95 of 111.5 / 60.9**, below the null on
both splits (the week shape is pick count, not skill); ex-top-5 % **−0.306 / −0.220**, so the
admission rule moved the tail but did not remove the dependence on it; and the weekly dollar path is
a coin flip (TRAIN last 8 weeks: −887, +57, +262, +1418, −520, −160, +809, −886). The tercile
robustness read goes **negative on VAL**, so even the cut is not stable.

## 2.5 THE SENTENCE

> **The monsters are not identifiable at entry on this book.** On the book we actually ship they do
> not exist — the "top 5 %" is the +2 R cap's point mass, and the clause that killed three objects
> is a mechanical −0.10 R toll that any book netting less than a twentieth of its own cap must fail.
> On the uncapped book the tail is real and exactly one causal field of eleven points at it —
> relative volume, LOWER, against the engine's own gate — and the admission rule it buys is
> +0.027 / +0.038 R at a clustered t of +0.4 with an MDE of 0.19–0.24, green weeks below its own
> null on both splits, and an ex-top-5 % still at −0.31 / −0.22. **Every tail-carried object stays
> dead**, and the ex-top-5 % clause is recorded as **protecting against luck, not killing real
> books** — with the caveat, now on the record, that on a CAPPED book it is not a tail test at all
> and should be replaced there by `net > 0.05 × cap ÷ 0.95`, which is the same number said honestly.

## 2.6 Rails (F35)

G-B2 asserted in code and raising · the GROSS-label circularity check on all eleven cells
(pre-committed in the amendment, and it voided both of the capped book's passes) · both TRAIN halves
+ VAL on every admission cell · day-clustered t throughout · 2,000-draw count-matched null on the
tail LABEL, preserving per-day tail counts · green-week nulls on every book cell · availability
audit per field (only `dollar_frac` below 100 %, at 82 % TRAIN / 96 % VAL — above the 80 % rail,
stated) · MDE on every cell · **26 scored cells** (11 capped + 2 admission + 11 uncapped + 2
admission), all printed, none selected after the fact · TEST never opened.

---

# F34 — PRICE THE UNIVERSE, NOT THE SIGNAL: THE FLOOR MAP (13 cells × 3 stop widths = 39 rows)

## 3.1 What was walked

The unconditional long bracket — entered at the OPEN of the control's own bar, **stop at a fixed
s % of the entry price** (s ∈ {2, 3, 4}), target +2R = +2s %, flat 15:55, `common6.walk_from`'s exit
convention verbatim — on pass 6's **arm-d detector-free controls**, a matched NON-signal name at a
random eligible minute on the PIT HOD universe (prev close ≥ $17, ADV20 ≥ 100K). **285,485 of
288,174 keys walked = 99.1 %.** No detector anywhere in the construction. TEST cut off.

This is NOT F22/F25's object: those walked the control at the **booked trade's own `r_pct`**, so the
stop width carried the signal's information. Here the stop is a pure function of price, so each cell
is a property of the UNIVERSE and the clock alone. Unit: **% of entry price** (F31's unit). Cost:
the programme's own model, `0.5 × spread% × (1 + ratio(exit))`, from the same imputation table the
book uses — and note that in % of price **it does not depend on the stop width at all**, which is
F31's arithmetic running in reverse.

## 3.2 The map — every number is % of entry price, TRAIN n / per week beside it

| cell | name | s% | n TRAIN | /wk | gross TR | cost TR | **net TR** | t | **net VAL** | t | H1 | H2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **U0** | unconditional | 2 | 198,060 | 3,737 | −0.136 | 0.253 | **−0.389** | −7.17 | **−0.360** | −6.51 | −0.392 | −0.386 |
| U0 | | 3 | | | −0.229 | 0.248 | −0.476 | −6.72 | −0.406 | −5.85 | −0.493 | −0.461 |
| U0 | | 4 | | | −0.261 | 0.241 | −0.502 | −6.22 | −0.415 | −5.33 | −0.520 | −0.485 |
| U1 | 09:37–10:30 | 2 | 41,202 | 777 | −0.216 | 0.312 | −0.528 | −7.93 | −0.509 | −6.01 | −0.512 | −0.542 |
| U2 | 10:30–11:30 | 2 | 46,274 | 873 | −0.144 | 0.241 | −0.385 | −5.55 | −0.367 | −4.87 | −0.401 | −0.370 |
| **U3** | **11:30–13:00** | **2** | 66,031 | 1,246 | **−0.093** | 0.204 | **−0.297** | −4.70 | **−0.331** | −5.27 | −0.304 | −0.291 |
| U4 | 13:00–14:01 | 2 | 44,553 | 841 | −0.118 | 0.284 | −0.402 | −7.69 | −0.255 | −3.82 | −0.406 | −0.397 |
| U5 | price $17–30 | 2 | 96,978 | 1,830 | −0.176 | 0.247 | −0.422 | −8.09 | −0.419 | −6.91 | −0.438 | −0.406 |
| U6 | price $30–100 | 2 | 87,285 | 1,647 | −0.101 | 0.253 | −0.355 | −5.68 | −0.338 | −6.08 | −0.346 | −0.362 |
| U7 | price ≥ $100 | 2 | 13,797 | 260 | −0.083 | 0.295 | −0.378 | −4.39 | −0.246 | −2.85 | −0.337 | −0.412 |
| U8 | ADV$ < $25M | 2 | 54,794 | 1,034 | −0.191 | 0.250 | −0.441 | −6.82 | −0.522 | −9.52 | −0.384 | −0.477 |
| U9 | ADV$ $25–150M | 2 | 87,254 | 1,646 | −0.157 | 0.253 | −0.410 | −7.20 | −0.396 | −6.97 | −0.418 | −0.403 |
| **U10** | **ADV$ ≥ $150M** | **2** | 56,012 | 1,057 | **−0.051** | 0.256 | −0.306 | −4.47 | **−0.194** | −2.41 | −0.362 | −0.241 |
| U11 | wrapper | 2 | 76,016 | 1,434 | −0.262 | 0.257 | −0.519 | −7.76 | −0.487 | −9.52 | −0.601 | −0.446 |
| U12 | common | 2 | 121,139 | 2,286 | −0.058 | 0.250 | −0.309 | −5.32 | −0.279 | −4.12 | −0.267 | −0.349 |

(The 3 % and 4 % rows are in `cells34.csv` and `s34.log`; every one is MORE negative than its 2 %
row, on gross and on net, in both eras — a wider stop on a 2:1 bracket buys more force-closes on a
population that drifts down.)

## 3.3 The answer to the question the frame was set

> **NO CELL OF THE BARE INSTRUMENT IS POSITIVE. 0 of 39 scored rows clears the pre-committed bar**
> — and 0 of 39 is even positive on GROSS, before a cent of cost, on TRAIN, on VAL, in H1 and in H2.

* **Best gross cell**: `ADV$ ≥ $150M` at a 2 % stop, **−0.051 % of price** (VAL is where it looks
  best; TRAIN H1 −0.362 / H2 −0.241 net).
* **Best net cell**: `11:30–13:00` at a 2 % stop, **−0.297 % TRAIN / −0.331 % VAL**.
* **The cost is the whole story and it is not a modelling artefact**: 0.20–0.31 % of price in every
  cell, essentially flat across the stop widths, against a gross drift that never exceeds −0.05 %.
  The instrument would need to find **five to six times its best observed drift, with the sign
  reversed**, merely to break even.
* **The ordering is real, the level is not**: the best cell sits **OUTSIDE** the 13-cell
  within-day permutation p95 at all three stop widths (2 %: −0.297 vs p95 −0.358; 3 %: −0.342 vs
  −0.440; 4 %: −0.343 vs −0.463), so the map's structure — **later clock > early, ADV$ ≥ $150M >
  small, common > wrapper, price > $30 > cheap** — survives its own multiplicity. It simply orders
  a set of negatives. The diagnostic cross-tab says the same in two dimensions (net % of price at
  the 3 % stop, TRAIN): 09:37–10:30 × < $25M is **−0.845**, 11:30–13:00 × ≥ $150M is **−0.283**.

**The pre-committed sentence:** *the floor under every admission rule on this universe is ≤ 0 in
every cell of the map, so an admission rule can only pick the least-negative region* — which is
exactly what 1,104 cells found. Honestly phrased: **no positive unconditional drift was detectable
in THIS universe (PIT HOD, prev close ≥ $17, ADV20 ≥ 100K), at THIS horizon (entry-bar open to a
±2:1 bracket, flat 15:55), at THIS stop geometry (2/3/4 % of price), over 2025-01 → 2026-05, at a
measured 0.20–0.31 % of price of cost, with a smallest detectable effect of 0.33–0.63 % of price
per trade** — and the observed drift is the wrong sign by five times its own best value, which is
not a power problem.

## 3.4 What this settles for the programme

1. **HOD-break's negative book is the universe, not the rule.** The detector's name-day selection is
   real (+0.183 / +0.245 R, pass 8) and the pond it fishes in has an unconditional drift of
   −0.30 to −0.50 % of price after cost. No filter on the signal minute can pay that.
2. **A wider stop is not a free option here.** Every cell is worse at 3 % and 4 % than at 2 %,
   because the cost in % of price is flat while the horizon truncation is not — the opposite of the
   intuition that a wider stop "gives the trade room".
3. **The one direction with a mechanism is the clock and the liquidity band**, and it is worth about
   +0.23 % of price from the worst cell to the best — real, ordered, era-stable, and still negative.

## 3.5 Rails (F34)

The walker reproduces `book6.rr` to **2.22e-16** on 120 booked trades, asserted before any control
was priced · **99.1 % bar-join availability**, above the 80 % rail · both TRAIN halves + VAL on every
row · day-clustered t on every cell · cost booked per cell from its own exit mix and its own imputed
spread · **2,000-draw permutation across all 13 cells at each stop width**, so the multiplicity of
the whole map is paid once · MDE per cell (0.33–0.63 % of price) · 39 scored rows over 13 declared
cells, all printed · duplicate walk rows removed on the natural key before any statistic · every
store read-only · TEST never opened.

---

## 4. The adequacy review (RUNBOOK step 10)

* **Did we test what the books actually ARE?** F36: the producer was run against the REAL EOD check
  on the four real dry sessions and the pooled line was read from the REAL checker; the BF band is
  read from the exact CSV the gate uses. F35: the shipped B2 book at the shipped cascade, plus the
  uncapped G3 geometry that the tail-killed objects actually live on — the second was added
  precisely because the first turned out to be the wrong object for the question. F34: pass 6's own
  control population, the programme's own walker, the programme's own cost model.
* **Is the cost and fill model right?** F34's cost (0.20–0.31 % of price) comes from the same
  measured-NBBO imputation table the book uses; it is 1.0 % measured on the full break population,
  which is the weakest link and is stated. But the conclusion does not rest on it: **the GROSS is
  negative in all 39 rows**, so zeroing the cost entirely leaves no positive cell.
* **Does any caveat in our own report explain the headline?** F35: YES, and it was pre-committed and
  it fired — the capped book's two "separating" fields are cost-model artefacts under the gross
  label, and the report leads with that rather than the headline it would have made. F34: the
  imputed-spread share, named above and not load-bearing. F36: the legacy band's width at small n,
  named in §1.2.
* **What is the MDE?** F35: 0.110 / 0.167 R on the capped book, 0.189 / 0.235 R on A3 — six times
  the effect A3 reads, which is why A3 is reported as undetectable rather than as an edge. F34:
  0.33–0.63 % of price per cell, against an observed drift of the wrong sign at five times that
  size, so this is not a power failure.
* **Multiplicity**: 24 declared + 2 pre-committed-amendment admission cells + the 11-cell uncapped
  supplement + its 2 admission cells + the 11-cell gross-label re-read as a rail. Scored and
  printed: **26 (F35) + 39 rows over 13 cells (F34) = 26 + 13 declared cells.**
  **Programme cell count 1,104 + 26 = 1,130.**

---

## 5. Test and suite status

* **`tests/test_bf_band_r_basis.py` — 21 new tests**: the column audit (including the assertion that
  `planned_entry` is empty on all 56 rows, so the band shifts loudly if it is ever backfilled), the
  `pnl_pct / stop_pct` identity row by row, the preference order, the band move +1.651 → +0.912 on
  TRAIN, the classification flip at n=30, the env flag and its WARNING, the side-by-side line, the
  ADV-off reference, and rows without a computable risk dropped at ERROR (never counted as zero).
* **`tests/test_ramp_pool.py` — 12 new tests on top of F33's 64 = 76**: a session re-run appends
  nothing and leaves the file byte-identical, the three-part key, in-call duplicate collapse, the
  pooled band reading the appended rows, the legacy 3-column schema, 2-tuples, non-finite R, a
  missing file; plus four on the EOD-check producer itself (dry appends once, live writes nothing,
  red-to-green is not pooled, a broken write never breaks the check).
* Two pre-existing tests encoded the retired BF basis
  (`tests/test_ramp_bt_band.py::TestLoaders`) and were updated to the fixed one, with the legacy
  path kept explicit in the same test.
* Full suite: see the commit message.
