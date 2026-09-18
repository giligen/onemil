# REPORT — F4 (weekly industry-adjusted reversal, 1 long-only cell) + F3 (momentum, 6 cells)

Run 2026-09-18 on the panel in `DATA.md`. Pre-registration: `PREREG_F4_F3.md`, written **before any
return was computed**. Code: `build_panel_f3f4.py` (the panel + the ADV fix), `run_f4_f3.py` (the 7
cells), `pit_rerun.py` (the survivorship re-run). Artifacts: `out_f4f3/` (primary),
`out_f4f3_{adv10m,flat5}/` (arms), `out_pit/`, `data/adv_fix_check.json`.

**Verdict in one line:** *no edge was detectable in the US-common-stock universe above $5 and $1M
ADV20$, at a one-week industry-residual reversal horizon or at 1-month-held 12-1 / 6-1 / residual
momentum, at a $66K / 20-slot book, over 2016-01 -> 2023-12, at close-to-close auction cost — and the
smallest monthly effect this test could have seen is **$1,012–$1,996/month at book size**, against a
Chen–Velikov prior of **$25–65/month**. **0 of 7 cells clear G1** (best TRAIN t = 1.45); **0 of 7
clear G2**. Removing the top 1% of trades takes six of the seven to zero or below, and all seven are
significantly negative ex-top-5%.*

**TEST (2024-01 -> 2026-09) was NOT opened. No `FREEZE.md` was written.**

Two things in this stage are findings rather than nulls, and both are negative for us:
1. **F4's profit is a lottery-name artefact, exactly as AMENDMENT 3(a) pre-committed.** On TRAIN the
   top MAX quintile earns 122 bps/trade and carries **50.5% of the P&L**; the bottom quintile earns
   21.6. On VAL that same top quintile is **-50.9 bps, t -2.65**. Sub-$10 names are 18% of TRAIN
   trades and 25% of its P&L, and on VAL they are **107% of the (negative) P&L**. Per the
   pre-registration this is a **NEGATIVE result for this account**, not a positive one.
2. **Residualising momentum halves the crash, as Blitz–Huij–Martens predict — and it is the only
   prediction in this stage that replicated.** Max drawdown 2016-2023: 12-1 L-S **-37.3%**, 6-1 L-S
   -30.6%, residual L-S **-20.5%**; long-only 12-1 -15.5% vs residual **-11.0%**. The mechanism is
   real; the return premium it is supposed to protect is not detectable here.

---

## 0. The ADV fix — done first, and it is bigger than F2/A1 estimated

`REPORT_F2_A1.md` §9 defect 6: `ADV20$` was built as `vwap x volume` on the **split+dividend-adjusted**
pull. Dollar volume is split-invariant, but the dividend factor scales price down without touching
volume, and that factor is a function of every dividend paid between the bar and today — so the $1M
liquidity gate was **mildly forward-looking**. Rebuilt on the **RAW** panel (`build_panel_f3f4.py`,
`trailing_adv`), verified on 200 random symbol-sessions (`data/adv_fix_check.json`):

| | |
|---|---|
| keys where the buggy series **understated** true dollar volume | **112 of 200** |
| keys exact (no dividend in the window) | 82 of 200 |
| keys overstated (float32 noise) | 6 of 200 |
| median relative error of the buggy series | **-1.01%** |
| mean / p01 / worst | -9.04% / **-54.05%** / **-100.00%** (BINI 2017-10-19: buggy ADV literally 0) |

Worst five: BINI -100%, VATE -85.7%, ARLP -53.7%, ETD -53.6%, BHK -51.1% — MLPs, BDCs and high-payout
names, i.e. the bug was systematically deleting **dividend payers**, not a random 1%.

**Effect on universe membership** (the whole grid, 9.16M symbol-sessions with both series finite):

| gate | cells passing, RAW (fixed) | cells passing, adjusted (buggy) | delta | admitted only by the fix | admitted only by the bug |
|---|---|---|---|---|---|
| ADV20$ >= $1M | 6,729,292 | 6,619,488 | **+109,804 (+1.66%)** | 109,807 | 3 |
| ADV20$ >= $10M | 4,382,433 | 4,258,644 | **+123,789 (+2.91%)** | 123,790 | 1 |

**825 of 4,871 symbols (16.9%)** change $1M membership on at least one session. The direction is
one-sided: the bug **excluded** high-yield names near the threshold and admitted essentially nothing.
Every family from here uses the RAW gate; F2/A1's cells were run on the buggy gate and their
`out_f2a1*` numbers carry that caveat (their $10M arm reached the same verdict, so the F2/A1
conclusion does not move).

---

## 1. What was run

Execution is **close-to-close, `cls` both legs, one-session skip** between the signal close and the
entry close (AMENDMENT 2(a); Goyal–Jegadeesh–Wu JFQA 2026: *"Opening auctions are illiquid"*). No open
is touched. No quoted spread on an auction cross. Costs: impact `10 bps x (order$ / 1% of ADV20$)` both
sides **with order$ scaled to the real position count** (F2/A1 defect 3 — a 200-name decile takes $330
positions, not $3,300, and the impact charge falls accordingly), + 0.4 bps SEC/TAF on the sell, +
0.3%/yr borrow on short legs gated on `easy_to_borrow`. **Costs are charged to the book only; the
benchmark series is gross** (defect 4). The long-only benchmark **excludes the book's own decile**
(defect 5 — the clean contrast). **Sealed splits**: a trade whose exit leaves the split leaves the
split (§8b); VAL is reported both ways and the two never differ by more than 36 bps.

Reference population for every decile sort, stated explicitly: **the eligible universe of that
rebalance** — `kind=='common'`, RAW close >= $5, RAW ADV20$ >= $1M, formation prices present. F4 adds
"has a 2-digit SIC and its SIC group has >= 5 eligible members that week" (DATA.md §6: the 7.9% of
common stocks without a SIC are DROPPED, never bucketed as "unknown"). Cross-section size: F4 ~2,200
names/week over 556 weeks (1.23M signal rows); F3 ~2,200/month over 115 (12-1), 121 (6-1) and 103
(residual) months.

**Declared deviation, F3 residual cells.** Blitz–Huij–Martens residualise on Fama–French **three**
factors. This panel has no book-to-market and no shares outstanding, so SMB and HML cannot be built
from it; the residual is a **market-model (CAPM) residual** from a rolling 36-month regression ending
at the decision month (>= 24 observations required), standardised by the residual standard deviation
over the 11 formation months. That is a weaker purge of the dynamic beta Daniel–Moskowitz blame for
the crashes — so §8's crash result is the *weak* form of the claim, and it still holds. The residual
cells also start later (first signal 2018-02, 103 months) because of the 36-month window.

---

## 2. The 7-cell table, per split

`bps` = monthly benchmark-adjusted excess. `t` = ordinary t on the monthly series; `NW t` = Newey–West
at lag = the hold length in months (1 for every cell here). `MDE` = 2 x SE of the monthly mean.
**ex-1% / ex-5%** = the portfolio **rebuilt** with the top 1% / 5% of trade contributions removed
(F2/A1 defect 2 — mandatory, not a robustness afterthought). VAL rows are the **sealed** convention.

| cell | split | n trades | raw bps | **excess bps** | **t** | NW t | %mo+ | **ex-top-1%** | **ex-top-5%** | ex-Jan | **MDE** | tr/wk |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **F4-LO** | TRAIN | 61,552 | +206.4 | **+56.1** | **1.34** | 1.24 | 55.6 | **-81.7** | **-348.2** | +46.7 | 84 | **19.7** |
| **F4-LO** | VAL | 24,099 | -15.0 | **-14.4** | -0.18 | -0.19 | 54.2 | -174.8 | -451.3 | +33.1 | 159 | **19.8** |
| F3-LO-12-1 | TRAIN | 11,283 | +189.4 | **+70.2** | **1.45** | 1.55 | 60.3 | **+4.3** | **-127.7** | +34.0 | 97 | 3.6 |
| F3-LO-12-1 | VAL | 5,348 | -6.0 | **+17.2** | 0.25 | 0.28 | 62.5 | -29.8 | -132.3 | +68.3 | 139 | 4.4 |
| F3-LS-12-1 | TRAIN | 21,805 | +13.7 | +13.7 | 0.18 | 0.20 | 46.6 | -103.3 | -355.4 | -21.3 | 154 | 3.6 |
| F3-LS-12-1 | VAL | 10,246 | +100.3 | +100.3 | 0.79 | 0.78 | 70.8 | -3.9 | -248.9 | +157.5 | 255 | 4.4 |
| F3-LO-6-1 | TRAIN | 12,639 | +192.5 | +41.6 | 0.90 | 0.95 | 59.4 | -26.4 | -159.3 | +44.0 | 92 | 4.0 |
| F3-LO-6-1 | VAL | 5,457 | -46.5 | -21.7 | -0.31 | -0.33 | 62.5 | -75.6 | -190.9 | +25.7 | 140 | 4.4 |
| F3-LS-6-1 | TRAIN | 24,330 | +48.6 | +48.6 | 0.82 | 0.98 | 46.9 | -66.8 | -315.1 | +42.4 | 119 | 4.0 |
| F3-LS-6-1 | VAL | 10,461 | +13.1 | +13.1 | 0.12 | 0.11 | 58.3 | -89.3 | -347.4 | +59.1 | 223 | 4.4 |
| F3-LO-RES | TRAIN | 8,719 | +167.4 | +35.0 | 0.79 | 0.89 | 50.0 | -23.9 | -133.9 | +7.0 | 88 | 2.9 |
| F3-LO-RES | VAL | 5,030 | +45.5 | +60.1 | 1.21 | 1.25 | 66.7 | **+19.6** | -64.2 | +81.7 | 99 | 4.4 |
| F3-LS-RES | TRAIN | 17,025 | +5.8 | +5.8 | 0.09 | 0.10 | 37.0 | -93.8 | -298.7 | -32.2 | 133 | 2.9 |
| F3-LS-RES | VAL | 9,740 | +149.1 | +149.1 | 1.46 | 1.44 | 66.7 | +56.3 | -151.6 | +178.8 | 204 | 4.4 |

**The ex-top-1% column is the report.** Six of seven cells go to zero or negative on TRAIN when ~1% of
trades is removed; F4-LO falls **+56 -> -82 bps** and F3-LO-12-1 **+70 -> +4**. Every cell is
**significantly negative ex-top-5%** (-64 to -451 bps). There is no cell here whose point estimate is
not a right-tail artefact — the same signature F2/A1 found, on a different family, a different horizon
and a 3x larger universe. The owner has already rejected one lottery ticket; these are seven more.

**Gross / raw column.** The `raw bps` column (+167 to +206 bps/month on TRAIN for the long-only cells)
is **not P&L and must not be quoted**: it is the panel's beta plus survivorship (§6). Only the
benchmark-differenced `excess` column is scored.

### The mandatory columns, read out
1. **Long-leg share.** Only one cell has a materially positive L-S spread and a defined share:
   **F3-LS-6-1 TRAIN, LLS = 87%**. The others' spreads are within noise of zero, where "share of a
   spread" is not a number (F3-LS-12-1 TRAIN reads 3.7, F3-LS-RES TRAIN 43.3 — both are division by
   ~zero and are reported as **undefined**). 87% is *better* than Israel–Moskowitz's published ~50%
   and is the structurally good answer for a long-only, no-margin account. It is also moot: the cell
   is t = 0.82 and dies on VAL and on the tails.
2. **Break-even cost.** The honest auction round trip on this book is **0.47–0.50 bps** long-only
   (impact at $330–$3,300 orders against >= $1M ADV is near-nil; almost the whole charge is the
   0.4 bps SEC/TAF) and **1.66–1.69 bps** for an L-S cell (the short leg's borrow). Break-evens:
   - **F4-LO, at its ~100%/week turnover** (4.33 round trips/month, ~52/yr): TRAIN gross excess
     58.3 bps/month -> break-even **13.5 bps per round trip = 26.9x the honest auction cost**, i.e.
     ~700 bps/yr of cost budget. **On VAL the break-even is -2.8 bps: no cost reduction whatsoever
     makes F4 positive.** 13.5 bps/RT is also thin in absolute terms against
     Detzel–Novy-Marx–Velikov (JF 78(3) 1743-1775, 2023), whose point is precisely that ~100%/week
     strategies look profitable only when this charge is neglected — our 0.5 bps is defensible
     **only** because every fill is a closing-auction cross; any non-auction execution of the same
     book (a marketable order in an $8 name) eats the entire 13.5 bps and more.
   - **F3, at monthly turnover** (1 round trip/month): break-even = the monthly excess itself, so
     F3-LO-12-1 TRAIN 70.2 bps = **146x the honest cost**, VAL 17.2 bps = 36x. F3 is nowhere near
     cost-constrained; it is signal-constrained.
3. **Ex-January.** Nothing here is a January effect. Ex-Jan tracks all-months within ~35 bps
   everywhere and is usually *larger* where the cell is positive. No cell is rescued or killed by it.
   (F5, not run here, is the family the literature flags for this.)
4. **Trades/week at $50–66K, and additivity.** **F4-LO is the only cell in the whole multi-day program
   so far that clears the owner's >= 10/week column on its own: 19.7–19.8/week**, because 20 slots
   divided by a 5-session hold is 20 entries a week. F3 is structurally slow at **2.9–4.6/week** —
   20 slots over a 21-session hold. Additivity: §7.

---

## 3. The executable $66K / 20-slot book — the money object, and its power

Alpha basis (the 20-slot book's own equal-weighted series, net of cost, **minus the gross benchmark** —
a raw long-only book in 2016-2021 earns the market plus this panel's survivorship and is not a money
estimate). The book is **long-only in every row**, including the L-S cells: this account has no margin
and no reliable borrow, so the short leg is a return estimate only and the two L-S rows share their
long cousin's book by construction.

| cell | split | book trades | tr/wk | **$/month (alpha)** | $/month (raw) | t | **MDE $/mo** | %mo+ | worst month | book MDD |
|---|---|---|---|---|---|---|---|---|---|---|
| F4-LO | TRAIN | 6,180 | 19.7 | **+850** | +1,855 | 1.3 | **1,280** | 52.8 | -10,202 | -23,762 |
| F4-LO | VAL | 2,060 | 19.8 | **-1,012** | -1,011 | -1.0 | **1,943** | 58.3 | -8,839 | -28,381 |
| F3-LO-12-1 | TRAIN | 1,140 | 3.6 | +717 | +1,502 | 0.9 | 1,596 | 51.7 | -15,168 | -18,192 |
| F3-LO-12-1 | VAL | 460 | 4.4 | +654 | +472 | 0.7 | 1,929 | 58.3 | -8,734 | -12,298 |
| F3-LO-6-1 | TRAIN | 1,260 | 4.0 | +263 | +1,241 | 0.3 | 1,565 | 48.4 | -15,081 | -26,995 |
| F3-LO-6-1 | VAL | 460 | 4.4 | -212 | -414 | -0.2 | 1,869 | 37.5 | -10,716 | -11,351 |
| F3-LO-RES | TRAIN | 900 | 2.9 | +523 | +1,394 | 1.0 | 1,012 | 52.2 | -9,431 | -12,854 |
| F3-LO-RES | VAL | 460 | 4.4 | +411 | +287 | 0.7 | 1,165 | 62.5 | -6,354 | -8,441 |

**This table is the report's second half.** Every |t| <= 1.3. The MDE column says the book could not
have detected anything smaller than **$1,012–$1,996/month**, while the pre-committed calibration prior
(Chen & Velikov: 204 published anomalies net ~4 bps/month) puts the expected effect at **$25–65/month
on $66K**. The experiment is **16–80x too blunt for the thing it is looking for** — the same
conclusion F2/A1 reached at 10–120x, and it is a property of the account size, not of these families.

**Bounded downside, the one thing well measured.** A $66K long-only auction sleeve at 20 x $3,300, no
leverage, no short, no stop: worst month **-$6.4K to -$15.2K**, book drawdown **-$8.4K to -$28.4K**,
i.e. **13–43% of the sleeve**. That is the honest risk of holding an equal-weighted basket of weekly
reversal losers or momentum winners through 2016–2023.

---

## 4. Gates, tails, multiplicity

**G1 (TRAIN t >= 2 on the benchmark-adjusted monthly series): 0 of 7.** Best is F3-LO-12-1 at
t = 1.45 (NW 1.55); F4-LO is 1.34.
**G2 (VAL same sign AND >= 55% of months positive): 0 of 7.** F4-LO's VAL excess flips sign. The three
cells that keep their sign on VAL (F3-LO-12-1, F3-LO-RES, F3-LS-RES) never cleared G1 to begin with.
**TEST stays sealed.**

**Robustness arms — the verdict is identical in all three** (`out_f4f3_*`):

| arm | F4-LO TRAIN t | F4-LO VAL | F3-LO-12-1 TRAIN t | F3-LO-12-1 VAL | best t anywhere |
|---|---|---|---|---|---|
| primary ($1M ADV, auction cost) | 1.34 | -14 bps | 1.45 | +17 bps | 1.45 |
| ADV20$ >= $10M | 1.58 | +3 bps | 1.37 | +5 bps | 1.58 |
| flat 5 bps/side | 0.37 | -55 bps | 1.25 | +8 bps | 1.46 |

The $10M arm is the interesting one: raising the liquidity floor *raises* F4's TRAIN t to 1.58 and
takes its VAL from -14 to +3 bps — consistent with §5's finding that F4's damage lives in the cheap,
lottery-shaped names — and it still does not reach G1, and G2 still fails on the months-positive half.

**Multiplicity.** 7 scored cells this stage, 3 execution/liquidity arms = **21 distinct looks**.
Block-bootstrap p (block 3 months, 5,000 resamples) on the TRAIN excess, Sidak-adjusted across 7 cells:

| cell | raw p | Sidak-adjusted |
|---|---|---|
| F3-LO-12-1 | 0.110 | 0.56 |
| F4-LO | 0.238 | 0.88 |
| F3-LS-6-1 | 0.306 | 0.93 |
| F3-LO-6-1 | 0.339 | 0.95 |
| F3-LO-RES | 0.358 | 0.96 |
| F3-LS-12-1 | 0.829 | >0.99 |
| F3-LS-RES | 0.912 | >0.99 |

Nothing is close, before or after adjustment.
**Cumulative multi-day scored-cell count: K 20 + N2 4 + R_daily 20 + F2/A1 8 + 7 here = 59.**

---

## 5. F4's three mandatory splits — AMENDMENT 3(a), and the sign was pre-committed

Chen–Cohen–Liang–Sun (JEF 82, 101608, 2025) find weekly US reversal **only** among high-MAX
lottery-like names (1.66%/wk vs 0.65%/wk low-MAX) and only in the top retail-order-imbalance periods;
SEF 2023 reports the unconditional US effect dead since 2000. AMENDMENT 3(a) pre-committed that
**profit concentrated in the top MAX quintile is a NEGATIVE result for this account**. It is.

**(i) MAX quintile** (MAX = the largest single-session return in the 21 sessions ending at the signal
close; quintiles of the book's own entries; per-trade net bps):

| MAX quintile | TRAIN n | TRAIN net bps | t | **TRAIN P&L share** | VAL n | VAL net bps | t | VAL P&L share |
|---|---|---|---|---|---|---|---|---|
| Q1 (lowest MAX) | 12,245 | +21.6 | 4.33 | 8.9% | 4,820 | -7.1 | -0.80 | 19.8% |
| Q2 | 12,244 | +13.7 | 2.33 | 5.6% | 4,820 | -5.2 | -0.51 | 14.5% |
| Q3 | 12,244 | +23.1 | 3.23 | 9.6% | 4,819 | +6.4 | 0.51 | -18.1% |
| Q4 | 12,244 | +68.0 | 7.81 | 28.1% | 4,820 | +21.1 | 1.22 | -59.2% |
| **Q5 (highest MAX)** | 12,244 | **+122.2** | 9.87 | **50.5%** | 4,820 | **-50.9** | **-2.65** | **142.9%** |

Half of F4's TRAIN profit is the top MAX quintile, and on VAL that same quintile is the single
significantly negative bucket and carries 143% of the loss. This is the JEF 2025 result reproduced —
and then reversed out of sample. **It is the bucket our floors are supposed to exclude.**

**(ii) The $10 price floor.** TRAIN: names below $10 are **18.1% of trades and 24.9% of P&L**
(+66.1 bps vs +44.1 above $10). VAL: **23.4% of trades and 106.8% of the P&L** — at -32.5 bps vs
**+0.6 bps** above $10. On VAL, *the entire F4 loss is the sub-$10 cohort and the rest is exactly
zero.* A $10 floor does not rescue F4; it reduces it to nothing, which is the same answer.

**(iii) Break-even at ~100%/week turnover**: §2 column 2 — 13.5 bps per round trip on TRAIN (26.9x the
0.50 bps auction cost, ~700 bps/yr of budget), **-2.8 bps on VAL**. Detzel–Novy-Marx–Velikov is the
anchor and their point stands: this is a strategy whose survival depends entirely on paying auction
prices, and whose TRAIN edge is in the names where an auction print is least reliable.

---

## 6. Survivorship — measured, not argued, and it is still a blocker for F3

**The panel is 100% survivors.** F2/A1 established it: 3,806 of 3,807 scored symbols still quoted in
2026; `delisted_names.parquet` (9,126 names) has **zero ticker overlap** with `universe.parquet` and
those names were never priced; and the "missing price -> 0% return" convention lets an in-sample
delisting break even instead of going to zero. This biases long-only UP, and per DATA.md §7 gap 1 it
hits **F3 hardest** (a 12-month formation window whose D1 decile IS the delisting cohort) and **F4
next**. The `raw bps` column of §2 is therefore withdrawn as a quotable number for every cell.

**What was done about it.** `pit_rerun.py` runs ONE code path over ONE delisting-inclusive tape twice —
the XNAS.ITCH daily tape on disk (`N_databento/N3/xnas_daily.parquet` + `R_daily/xnas_daily_2024H1`),
2018-05 -> 2023-12, truncated at the VAL end so TEST stays sealed:
**PIT = 16,353 symbols; SURVIVOR = the 3,729 (22.8%) of them Alpaca still lists today.** The
difference is the bias.

**F4 — the re-run succeeded, because its windows are five sessions long.** The tape is unadjusted, but
a 1-week formation and a 1-week hold are almost never split-straddling, so **no split adjustment is
used at all**: any symbol-week with a >90% single-session move in either window is screened out
(285 of 52,802 trades in the PIT arm, 179 of 46,473 in the SURVIVOR arm — 0.5% and 0.4%). The screen
looks at the holding window and so is **not causal**; it is a measurement correction applied
*identically to both arms*, so it cannot manufacture a survivorship difference.

| F4-LO on the Nasdaq tape | trades | TRAIN bps | t | VAL bps | t |
|---|---|---|---|---|---|
| **PIT** (delisted included) | 52,517 | **+62.9** | 1.07 | **-95.2** | -0.82 |
| **SURVIVOR** (today's names only) | 46,294 | **+78.2** | 1.24 | -110.7 | -0.93 |
| PIT with a -100% terminal haircut | 52,517 | +62.9 | 1.07 | -95.2 | -0.82 |

**Survivorship inflates F4's TRAIN long-only excess by +15.3 bps/month, i.e. ~20% of the point
estimate** — a real bias, and not large enough to be the whole story: F4 is a null before and after
it. The -100% haircut arm is *identical* to the PIT arm, which settles the second half of the
objection: at a 5-session hold, **zero** positions have their symbol stop trading mid-hold, so the
"frozen position breaks even" convention is not load-bearing for F4. (On VAL survivorship works the
other way, -15.5 bps, because in 2022 the absent cohort would have dragged the benchmark more than the
book; both arms are negative either way.)

**F3 — the re-run FAILED, and the failure is reported rather than the number.** A 12-month formation
return on an unadjusted tape needs a split adjustment. A volume-confirmed split detector was built and
**validated against the truth** (Alpaca's own daily corporate-action factor
`(adj_t/adj_{t-1})/(raw_t/raw_{t-1})`, on the 3,613 symbols in both panels): after four iterations it
reaches **precision 38.2% / recall 19.8% (TP 113, FP 183, FN 458)**. The three failed iterations and
why each failed are recorded in the code comments; the decisive one is that 3:2 and 5:3 splits sit on
top of an ordinary +-50% biotech session, and dropping them from the ratio set is what took precision
from 8.6% to 38.2% at the cost of recall. **That is not good enough to trust a 12-month return**, and
the symptom is visible: the F3 PIT arm reads -584 bps/month on VAL with 58% of months positive — one
contaminated month. Both F3 PIT rows are written to `out_pit/pit.json` tagged `*_unreliable` and are
**not used anywhere in this report**.

**Consequence, stated plainly as F2/A1 did:** for F3 the **GROSS per-trade column is NOT quotable**,
and no F3 result may be believed as a level. What survives is the *differenced* column: every scored
number in §2 is a difference against a benchmark drawn from the same surviving population, in which
the long bias very largely cancels — which is what makes these nulls credible rather than
survivorship-driven, because the bias would have *inflated* the cells, not deflated them. Pricing the
delisted names properly for a 12-month-formation family needs a split-adjusted delisting-inclusive
daily file we do not own (DATA.md §7 gap 8); that is the concrete purchase this program would need
before any momentum result could be stated as a level.

**Other rails.** Price scale: cleared at the data stage (0 of 200 keys fail at 0.01%). Causality:
every field is computable at or before its decision close — the 12-1 and 6-1 formation windows end one
month *before* the decision month, the residual regression window ends *at* it, F4's `r1w` ends at the
signal close, the decile is cut against that rebalance's eligible set only, and both entries are one
session *after* the signal. Obtainability: every fill is an official consolidated closing-auction
print decided one session in advance on both legs; no gap-through, no intrabar stop, no double-counted
slip, no touch fill. Frozen positions: `ends_mid_hold = 0` on every one of the 7 cells.

---

## 7. Additivity vs the live ORB book

**Resource overlap: none on ORB's binding constraint.** ORB's binding resources are the 4 concurrent
intraday slots, 09:35 buying power and the 09:35 attention window; every ORB position is flat by
15:45. F4 and F3 both open in the **16:00 closing auction** and hold 5 or 21 sessions. They never
contend for an ORB slot, never need capital at 09:35 on the day they enter, and never touch the ORB
engine, StopMonitor or the intraday tape. **Structurally additive on all three.**

**The one real interaction** is the balance sheet, unchanged from F2/A1: $66K of equity held overnight
reduces the next morning's day-trading buying power by that notional in a margin account. ORB's stage
budget is $10K, so the two fit today, but this is an account-level check to run before any multi-day
sleeve is armed, not an assumption. Stage I's "stacking hurts" is **not** imported (that was same-day
intraday books fighting for the same slots).

**Return correlation with ORB: NOT COMPUTED, and the reason is the seal.** The live ORB book
(`research/fuckup_audit/D1_orb/book_n8_q1on.csv`) runs 2025-01-07 -> 2026-09-15, entirely inside the
sealed TEST window; computing the correlation means opening TEST. No cell earned that. There are zero
overlapping months on TRAIN/VAL, so there is no partial answer.

**Frequency, at the portfolio level (the softened AMENDMENT's actual question).** F4-LO at 19.7/week
is the first multi-day cell that would resolve inside a quarter *on frequency* — and its MDE is
$1,280–$1,943/month, so a quarter of live trading would resolve nothing anyway. F3 at 2.9–4.6/week
fails both halves. Frequency without a measurable estimate is the HOD-break failure mode, and F4 is
its multi-day twin.

---

## 8. Momentum crashes — the mandatory F3 column, and the one prediction that replicated

Daniel–Moskowitz (JFE 122(2) 221-247, 2016): momentum crashes in panic states, and for a compounding
account **the drawdown IS the finding**. Monthly compounded series, 2016-01 -> 2023-12,
benchmark-adjusted for the long-only rows and self-benchmarked for the L-S rows:

| cell | **max drawdown** | trough | **Mar–Jun 2020** | **2022 (full year)** | worst month | best month |
|---|---|---|---|---|---|---|
| 12-1 long-only | -15.5% | 2022-01 | +12.5% | +10.6% | -8.4% (2019-09) | +15.3% |
| **12-1 L-S** | **-37.3%** | 2021-04 | -3.8% | +38.5% | **-15.7% (2020-11)** | +15.0% |
| 6-1 long-only | -17.2% | 2017-06 | +7.9% | -5.4% | -11.0% (2017-01) | +12.2% |
| 6-1 L-S | -30.6% | 2023-12 | -9.0% | +8.1% | -11.3% (2019-09) | +15.1% |
| **residual long-only** | **-11.0%** | 2020-12 | +8.1% | +13.8% | **-5.7% (2020-08)** | +13.6% |
| **residual L-S** | **-20.5%** | 2020-12 | +4.9% | +40.4% | -9.6% (2023-01) | +16.1% |

Three things to read here:
1. **The crash is real and it is the reason a compounding account cannot run published momentum.**
   12-1 L-S loses **15.7% in November 2020 alone** (the vaccine rotation — the textbook panic-state
   reversal) and runs a **-37.3% drawdown** into April 2021, on a strategy whose whole-period mean is
   +13.7 bps/month. No G1, and a -37% hole: the worst possible combination.
2. **Residualising halves it**, exactly as Blitz–Huij–Martens argue and for the reason
   Daniel–Moskowitz give: -37.3% -> -20.5% L-S, -15.5% -> -11.0% long-only, worst month -15.7% ->
   -9.6%, and the drawdown trough moves out of the rotation. **This is the only prediction in this
   stage, and one of only two in the whole multi-day program, that replicated.** It replicates in the
   *weak* (CAPM-residual) form, without SMB/HML, which makes it a lower bound on the published claim.
3. **It buys nothing tradable.** The residual cells' return premium is t = 0.79 TRAIN / 1.21 VAL
   long-only, dies ex-top-5%, and runs at 2.9 trades/week. A halved crash on an undetectable edge is
   a better-shaped null, not a book.

---

## 9. Independent rebuild (CLAUDE.md "No research claim ships without an independent check")

A second implementation was written from a **prose specification only**, by an agent explicitly
forbidden to read `run_f4_f3.py`, `build_panel_f3f4.py`, `PREREG_F4_F3.md`, `REPORT_F2_A1.md` or any
output directory of this stage (`indep_f4f3.py`, `out_indep_f4f3/`). It rebuilt the two strongest
cells -- `F4-LO` and `F3-LO-12-1` -- from the year-partitioned parquets: its own session calendar, its
own RAW-panel ADV20, its own SIC screen, its own deciles, its own portfolio aggregation and its own
tail rebuilds.

**(a) The numbers agree.**

| cell | split | mine: trades / bps / t | independent: trades / bps / t | mine ex-1% | indep ex-1% | mean cost bps |
|---|---|---|---|---|---|---|
| F4-LO | TRAIN | 61,552 / +56.1 / 1.342 | 61,717 / **+58.2** / **1.335** | -81.7 (t -2.37) | **-80.3 (t -2.20)** | 0.501 / 0.501 |
| F4-LO | VAL | 24,099 / -14.4 / -0.181 | 24,099 / **-15.3** / **-0.197** | -174.8 (t -2.53) | **-177.7 (t -2.69)** | 0.482 / 0.482 |
| F3-LO-12-1 | TRAIN | 11,283 / +70.2 / 1.448 | 11,507 / **+66.8** / **1.349** | +4.3 (t 0.10) | **-2.1 (t -0.05)** | 0.480 / 0.480 |
| F3-LO-12-1 | VAL | 5,348 / +17.2 / 0.248 | 5,367 / **+23.6** / **0.340** | -29.8 (t -0.46) | **-24.5 (t -0.38)** | 0.472 / 0.472 |

Trade counts differ by **0.27%** (F4) and **2.0%** (F3, from the `ceil(n/10)` decile boundary and a
different handling of the eligibility check at the entry vs the decision session); mean cost agrees to
three decimals on all four rows; every point estimate agrees within 6.4 bps and every t within 0.10.
The MAX-quintile table reproduces essentially exactly (TRAIN Q1 21.5 / Q2 13.8 / Q3 23.1 / Q4 67.9 /
**Q5 122.2** vs my 21.6 / 13.7 / 23.1 / 68.0 / **122.2**; VAL Q5 **-50.9** on both) and the sub-$10
P&L share reads 25.2% TRAIN / 106.8% VAL against my 24.9% / 106.8%. **A coding error is ruled out for
the universe, the ADV gate, the industry residual, the decile, the hold, the return, the cost model
and the tail rebuild.** Per the standing rail this cannot rule out a *specification* error.

**(b) Where it disagrees, and the disagreement is itself the point.** F3-LO-12-1's ex-top-1% TRAIN
reads **+4.3 bps on mine and -2.1 bps on theirs** -- the cell's entire surviving point estimate sits
on *either side of zero* depending on which 113 of 11,283 trades a `ceil` lands on. And F4-LO's VAL
months-positive reads **54.2% mine / 45.8% theirs**: two of twenty-four months flip sign under the
monthly-aggregation convention. Both are the F2/A1 §8c finding again -- **a real effect does not move
across zero on a bookkeeping convention** -- and both make the verdict more null, not less.

**(c) A specification defect the rebuild found in the PANEL, not in the code.** It reported that the
adjusted panel carries **post-reorganisation / ticker-recycling jumps**: GPOR 2021-05-18 **+52,648%**,
LINE +44,787%, CBL +33,461%, AQB +15,819%, ASTI +15,059%. These are companies emerging from Chapter 11
onto a recycled ticker, whose adjusted series Alpaca chains straight through the reorganisation. I
quantified it: **96 sessions on 89 symbols, 0.00104% of the 9.19M finite daily returns.** Load-bearing?
No, and measurably so:

| cell | book trades | trades whose HOLD window contains a >500% session | trades whose FORMATION window does | their mean gross vs the book's |
|---|---|---|---|---|
| F4-LO | 85,909 | **0** | **0** | -- |
| F3-LO-12-1 | 16,868 | **0** | 124 (0.74%) | +1.3% vs +1.27% |
| F3-LO-6-1 | 18,346 | **0** | 73 (0.40%) | -0.3% vs +1.19% |
| F3-LO-RES | 13,970 | **0** | 4 (0.03%) | +9.4% vs +1.34% |

The $5 raw-close and $1M ADV gates at the decision close exclude the reorg names on the day they
matter, so no scored trade ever holds one through its jump, and the 0.03-0.74% whose *formation*
window contains one earn the book average. **The defect is real and it is disclosed; it is not what
produces any number in this report.** It is, however, a named blocker for any future family that
relaxes the price or liquidity floor, and it belongs in `DATA.md` as gap 10.

**(d) Conventions the rebuild chose differently, all disclosed by it and none material**: cost charged
per entering/exiting cohort rather than per open position; no return credited on the entry day (same as
mine); the <100-symbol skip applied after the SIC filter (no week or month was skipped under either
reading); a missing ADV20$ at the *entry* session charged the full 20.4 bps cap (F3's max cost 1.31 bps,
so this binds only on a handful of F4 rows).

---

## 10. The two bars

**Claim bar (G1 + G2): FAILED, 0 of 7.** Nothing from this stage may be stated as a finding, and TEST
stays sealed. The two *negative* findings in §5 and §8 are reported as such.

**Live-exploration bar** (positive point estimate + mechanism + bounded downside + resolution inside a
quarter at its own trade frequency):

| cell | point est. TRAIN/VAL | mechanism | bounded downside | resolves in a quarter | verdict |
|---|---|---|---|---|---|
| **F4-LO** | +56 / **-14** bps | published (DLS 2014), but the live form is high-MAX only | yes (-$28K MDD) | **frequency yes, 19.7/wk** — MDE $1.3–1.9K/mo no | **FAIL**: sign flip, top-MAX artefact, $10-floor artefact |
| F3-LO-12-1 | +70 / +17 bps | published (JT 1993, IM 2013, JKP 2023) | yes (-$18K MDD) | no (3.6/wk) | **FAIL on noise** (ex-1% -> +4 bps) |
| F3-LS-12-1 | +14 / +100 bps | published | **no** — -37% crash | no | **FAIL** |
| F3-LO-6-1 | +42 / -22 bps | published | yes | no | **FAIL** (sign flip) |
| F3-LS-6-1 | +49 / +13 bps | published | no — -31% crash | no | **FAIL** |
| F3-LO-RES | +35 / +60 bps | published (BHM 2011, BHV 2020), **crash halved** | yes (-$8.4K MDD, best shape here) | no (2.9/wk) | **FAIL on noise** |
| F3-LS-RES | +6 / +149 bps | published | -20.5% | no | **FAIL** |

**F3-LO-RES is the only cell that keeps its sign on VAL, survives ex-top-1% on VAL (+19.6 bps), has
the smallest drawdown and the smallest MDE.** It is still t = 0.79 on TRAIN, 2.9 trades/week and
-134 bps ex-top-5%. It is the best-shaped null of the seven, not a candidate.

---

## 11. What I would do next, and what I would not

- **Do not re-propose F4.** Its live published form is conditional on exactly the cohort our floors
  delete (AMENDMENT 3(e) pre-committed this and it came true to the quintile), its TRAIN edge is 50%
  top-MAX and 25% sub-$10, its VAL sign flips, and it dies ex-top-1%. The one thing it has —
  19.7 trades/week, the first multi-day cell to clear the owner's frequency column — is worth
  remembering as a *structure* (a weekly closing-auction rebalance at 20 slots is how a multi-day
  sleeve gets to 20/week) and not as a *signal*.
- **Do not re-propose F3 L-S in any form.** -37%, -31% and -20.5% drawdowns on t = 0.1–0.8 means that
  even if the premium were real this account could not carry it, and the short leg is unreachable
  without margin and reliable borrow. Israel–Moskowitz's ~50% long-leg share is the reason the
  long-only variants are the only ones worth measuring, and they are the ones measured here.
- **The binding problem is power, not signal, and it is now measured twice.** F2/A1: MDE 10–120x the
  literature's expected effect. Here: 16–80x. That is a property of $66K and 20 slots and no further
  searching on this panel changes it. Every subsequent multi-day cell should state its MDE in $/month
  *before* it is run; a cell whose MDE exceeds ~$100/month can refute something large and confirm
  nothing.
- **The one purchase that would change the answer** is a split-adjusted, delisting-inclusive US daily
  file (DATA.md §7 gap 8). It would make F3's and F5's level column quotable and would let every
  survivor be PIT-re-run rather than argued about. Everything else on this line is cheap and already
  on disk.
- **Newest verified evidence age.** F4 — JEF 2025 (conditional on high MAX and retail order imbalance)
  and SEF 2023 (unconditional effect dead since 2000): both are *against* us and both were confirmed
  by this run. F3 — Jensen–Kelly–Pedersen JF 2023 replicates momentum across 93 countries, but no
  post-2022 study specific to US 12-1 decay was found, and the residual-momentum cells rest on 2020
  evidence (Blitz–Hanauer–Vidojevic, IREF 69), as the lit review required us to say.
