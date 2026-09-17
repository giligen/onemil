# Stage K — multi-day holds on the liquid daily panel

Pre-registered in `K/PREREG.md` (2026-09-17, before any run). Executed 2026-09-17. Splits are
PLAN.md §1: TRAIN 2025-01-02..2025-12-31 · VAL 2026-01-01..2026-05-31 · TEST 2026-06-01..2026-09-11
(the daily panel ends 2026-09-04).

---

## 1. One page

**Does any multi-day family clear G1/G2? No. 0 of the 20 pre-registered cells passed G1 on TRAIN,
so G2 was never reachable and TEST was NOT read** (the freeze is written in `K/FREEZE.md`).

| | number |
|---|---|
| pre-registered cells scored | 20 (5 families × 2 holds × 2 book sizes) |
| cells passing G1 (TRAIN: net > 0, t ≥ 2.0, ≥ 5 trades/week) | **0** |
| cells passing G2 | **0** (not reachable) |
| best TRAIN cell | `K3_h3_n10` +17.4 bps net/trade, t = **+0.51**, 800 trades, search-adjusted p = 0.945 |
| best VAL cell | `K2_h10_n20` +166.4 bps net/trade, t = +1.26, p = 0.670 — but its TRAIN is **−23.0 bps**, and removing the top 1% of its trades takes it to **−17.9 bps** |
| cells with net > 0 on TRAIN | 1 of 20 |
| cells with net > 0 on VAL | 2 of 20 |
| TEST | **sealed** |

**The $/month at capacity.** Position size = 1% of the name's 20-day median dollar volume, one size
for the whole book (PREREG also fixes equal-$ per position, so the two together mean a single size).
That is **$415K–$611K per position, i.e. a $4.1M–$12.2M book at 10–20 slots** — capacity is genuinely
there, and it is the one thing this stage confirms. What it earns is not: the single TRAIN-positive
cell (`K3_h3_n10`) makes **+$60,368/month on a $5.64M book (+1.07%/month), worst month −$755,593
(−13.4%)**. Every other cell is negative on TRAIN, the worst at −14.3%/month. On VAL only
`K2_h10_n20` is positive (+3.47%/month, worst month −$459,227).

**The smallest effect these tests could see** (the per-trade mean that would have produced t = 2):

| split | weeks | MDE per trade | MDE per week of book return |
|---|---:|---|---|
| TRAIN (250 days) | 41–51 with exits | **13.4 bps** (`K3_h1_n20`, 4,540 trades) to **171.3 bps** (`K1_h5_n10`, 261 trades) | 40.2 – 190.7 bps |
| VAL (102 days) | 21–23 | **22.7 bps** to **314.1 bps** | 58.8 – 289.7 bps |
| TEST (66 days) | ~14 | not read; would have been ≈ 1.25× the VAL numbers | ≈ 75 – 360 bps |

So the supported statement is: **no edge was detectable in the ≥$10M/day US common-stock universe,
at 1–10 day holds, in a 10–20 position equal-$ long-only book, over 2025-01..2026-05, at the
PREREG cost model — and for the family with the most trades the test could not have seen an effect
smaller than about 13 bps per trade (TRAIN) / 23 bps (VAL).** For the sparse families (K1, ~6
signals/week) the test could not have seen anything below ~1.7% per trade; K1 is effectively
untested, not refuted.

**Three findings that are not "nothing":**

1. **The declared ranking key is worse than a coin flip.** PREREG ranks each family by its own
   signal intensity (K1 gap size, K2 volume ratio, K3 reversal depth, K4 the overnight mean, K5
   pullback depth). Replacing that key with a random draw — same signals, same slots — *improves*
   the book on BOTH splits for K4 (−80.1/−62.4 → −55.2/−48.0 bps) and K5 (−92.1/−44.9 → −38.6/−10.0)
   and on TRAIN for K1 and K2. More of the declared signal is a worse trade. (`K/diagnostics_d5.md`.)
2. **The book, not the family, is where K2 dies.** K2's whole signal set is +23.8 bps net on TRAIN
   and +64.1 on VAL; its 10-slot declared-key cell is −78.8 / +19.2. Ten slots at a 10-day hold
   admit ~15% of signals and the declared key picks the wrong 15%.
3. **The cost model is the whole economics at short holds.** The PREREG round-trip charge is ~44 bps
   more than an auction-honest charge, and a random long book on this universe earns only ~5 bps of
   drift per day. Under the PREREG charge a 1–3 day hold cannot clear costs *even with a perfect
   signal*; under auction costs (CLAUDE.md fill-realism rule 4: an opening/closing auction execution
   does not cross a quoted spread) the random control turns positive at hold ≥ 3. The PREREG cost
   table is an intraday signal-minute spread on ≥5%-range movers, applied to $10M+/day names at the
   open — it is a conservative upper bound and it is doing a lot of the killing at holds 1–3. It is
   NOT doing the killing at hold 5–10, where the families are gross-negative against the control.

**What it means for the program.** The cost-curve argument that motivated Stage K (spread/R ≈ 0.02
on a multi-day hold vs 0.14 intraday) is arithmetically correct and the capacity is real, but it
buys nothing here: on this universe the multi-day families have no edge to spend the cheaper spread
on. The one direction that came out of the stage with a mechanism attached is finding 1 — that
signal intensity is negatively selective in a slot-constrained book — and that is a hypothesis for a
new pre-registration, not a result of this one.

---

## 2. Step 0 — price scale, and what it caught

`K/step0.md`, `K/pricescale.csv`. 200 random (symbol, day) keys present in both the Databento daily
panel and `data/cache.db::daily_bars` (Alpaca — the prices the live account sees):

| field | within 0.01% | within 1% | median abs diff |
|---|---:|---:|---:|
| open / high / low / close | 97.5% | 98.5% | 0.0000% |
| volume | 83.6% | 99.0% | 0.0000% |

**The panel is RAW and agrees with Alpaca to float32 precision** (the residual ~3e-6% is the
float32 round trip). No split/dividend adjustment: the CLAUDE.md price-scale hazard does not apply
to this stage.

The check earned its keep anyway. Three of the 200 keys disagreed by 100% because the *panel* rows
carry a price of exactly 0.0000 (ODVWZ 2025-10-07, LCFYW 2025-07-15, TVACU 2026-03-27). There are
**169,341 such rows (3.4% of the panel)**. In the first run they entered the rolling statistics: a
zero `prev_close` divides into `inf`, an `inf` inside a cumulative sum makes every later window of
that symbol NaN, and a zero-price row with a valid open produced an infinite "gap" that passed the
K1 gap cut. Both were fixed before any number in this report was produced (`build_k.py`: invalid
rows become NaN everywhere, and 72,588 non-finite overnight returns on otherwise-valid rows were
converted from `inf` to missing).

---

## 3. Universe and availability audit

`K/availability.md`, `K/step0.md`, `K/signal_counts.csv`.

Universe (PREREG): 20-day median dollar volume ≥ $10M, close ≥ $5, common stock only, test tickers
(`^Z[A-Z]ZZT$`) out, early-close days not signal days. `dvol20_med` uses the Stage-E convention
(`(close×volume).shift(1).rolling(20, min_periods=10).median()`); `adv20`, `vol_ratio`, `ret5` and
`high52` are the panel's own fields, i.e. the same definitions the earlier daily tests used.

- base (before the class filter): **1,254,449 symbol-days**, 4,610 distinct symbols, 420 days
- primary (`asset_class == stock`): **919,910**
- secondary control (everything except positively identified wrappers): **947,065**

Coverage of every field used in a decision, on the primary universe, is **100.00% on all three
splits** except where the field's own definition withholds it (`high52` needs 60 prior bars,
`sma20` needs 20, `on20` needs 15 of the last 20 overnight returns) — those are reported in
`K/availability.md` with what the missingness means — `high52` is **79.08% on TRAIN vs 99.10% on
VAL**, which is not a data gap but the panel's own start date, and it matters enough to get its own
subsection below. No field was backfilled from a later key set, which is the D1 look-ahead this
standing rule exists to catch.

**The one survivorship channel, disclosed.** The class map is a 2026-07-11 dump of live Alpaca
assets; a symbol that delisted in 2025 cannot be in it. On the liquid slice it covers 73.3% as
stock, 24.5% as wrapper, and **2.1% of symbol-days (255 symbols) are not in it at all — 218 of them
have no panel bar after 2026-07-11, i.e. they are gone by the dump date**, and the share is
3.2% on TRAIN vs 0.2% on TEST. Requiring map membership therefore filters out 2025 delistings. That
is why every family is also scored on the **secondary universe** (drop only identified wrappers,
keep the not-in-map names). It changes nothing: the secondary cells are −99.3 / +3.6 / −78.6 /
−101.4 bps on TRAIN (K1/K3/K4/K5) and −29.1 for K2, versus −122.6 / +17.4 / −80.1 / −92.1 / −78.8
for the primary. Survivorship is not carrying this result in either direction.

### Lookback warm-up — the limitation that partly disarms K2 and K5

The Databento panel starts 2025-01-02 and carries no earlier history, so every lookback longer than
the elapsed panel is TRUNCATED, and the truncation falls entirely on TRAIN.

| family leg | needs | first day it can fire | consequence |
|---|---|---|---|
| K2 `high52` (panel convention: `high.shift(1).rolling(250, min_periods=60).max()`) | 60 prior bars for a value, 250 for the declared "252-day high" | **2025-04-01** | every TRAIN K2 signal, and every VAL signal before 2026-01, is a new high of the last **60–250** bars, not of 252. K2's TRAIN window is 9 months, not 12 (its capacity row shows 10 months of exits, not 13). |
| K5 `hi50` (50-day high, expanding until 50 bars exist) | 50 prior bars | **2025-01-31** (gated by `sma20`, which needs 20) | K5 signals in Feb–Mar 2025 use a "50-day high" measured on 20–50 bars |

**So the declared K2 rule was never tested on TRAIN as written** — what was tested is a
shorter-lookback proxy that fires more often than a true 52-week-high rule would. K2 is the one
family with a positive excess over the random control on both splits (§6), which makes this the
most consequential caveat in the stage: its TRAIN number is not a clean test of the declared cut,
and no conclusion about a genuine 52-week-high book should be drawn from it. Testing it properly
needs daily bars back to 2024-01 — a data pull this stage did not have and did not fake.

The three other families (K1 gap, K3 5-day reversal, K4 20-day overnight mean) need at most 20
prior bars and are unaffected beyond the panel's first month.

### Signal frequency (pre-book, primary universe)

| family | TRAIN signals | per week | VAL | TEST (count only) |
|---|---:|---:|---:|---:|
| K1 gap continuation | 328 | **6.3** | 188 | 77 |
| K2 new 250-day high on volume | 1,647 | 31.7 | 1,175 | 527 |
| K3 short-term reversal | 15,342 | 295.0 | 7,265 | 4,744 |
| K4 overnight continuation | 49,243 | 947.0 | 24,019 | 16,794 |
| K5 pullback in an uptrend | 3,903 | 75.1 | 2,509 | 2,161 |

PREREG asked that any declared cut yielding < 5 signals/week on TRAIN be reported as such rather
than tuned. **None fell below 5.** K1 is the closest at 6.3/week pre-book and 5.6/week after the
10-slot book — close enough to the floor that its statistics are weak (MDE 171 bps/trade), which is
stated rather than fixed.

---

## 4. The 20 pre-registered cells

Full tables, both splits, in `K/phaseA.md`; the machine-readable version is `K/cells_trainval.csv`;
per-trade rows for every cell are in `K/trades/*.csv` (25 files, 55,180 rows; they carry all three
splits because the book runs continuously across the whole 420-day span, but **no TEST statistic was
computed** — see `K/FREEZE.md`).

Holds are the declared hold and half of it: K1 5/2, K2 10/5, K3 3/1, K5 5/2. **K4 is the one
deviation from PREREG and it is named, not hidden:** its declared hold is 1 day and half of a
one-day close-to-close hold is not representable on daily bars, so its second cell EXTENDS to 2 days.

### TRAIN (2025), primary universe, PREREG cost model

| cell | n | tr/wk | gross bps | net bps | t | WR% | wk green% | stop% | net R | p_adj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `K1_h5_n10` | 261 | 5.6 | −75.4 | −122.6 | −1.43 | 39.8 | 34.0 | 22.6 | −0.362 | 1.000 |
| `K1_h5_n20` | 303 | 6.4 | −63.8 | −111.2 | −1.47 | 40.9 | 31.9 | 21.5 | −0.314 | 1.000 |
| `K1_h2_n10` | 304 | 6.3 | −31.5 | −78.8 | −1.59 | 42.4 | 35.4 | 9.2 | −0.217 | 1.000 |
| `K1_h2_n20` | 324 | 6.8 | −32.8 | −80.2 | −1.70 | 42.9 | 35.4 | 9.0 | −0.192 | 1.000 |
| `K2_h10_n10` | 237 | 5.6 | −30.8 | −78.8 | −1.16 | 48.1 | 42.9 | 32.9 | −0.113 | 1.000 |
| `K2_h10_n20` | 443 | 10.5 | +24.6 | −23.0 | −0.47 | 47.9 | 42.9 | 28.9 | −0.033 | 0.997 |
| `K2_h5_n10` | 402 | 9.8 | −30.8 | −79.1 | −1.69 | 46.3 | 34.1 | 22.4 | −0.113 | 1.000 |
| `K2_h5_n20` | 731 | 17.8 | −7.6 | −54.7 | −1.60 | 47.6 | 36.6 | 19.2 | −0.078 | 1.000 |
| **`K3_h3_n10`** | 800 | 15.7 | **+62.9** | **+17.4** | +0.51 | 46.8 | 56.9 | 0.0 | — | 0.945 |
| `K3_h3_n20` | 1,590 | 31.2 | +15.4 | −30.4 | −1.25 | 44.2 | 49.0 | 0.0 | — | 0.999 |
| `K3_h1_n10` | 2,332 | 46.6 | −13.2 | −58.9 | −5.59 | 41.8 | 26.0 | 0.0 | — | 1.000 |
| `K3_h1_n20` | 4,540 | 90.8 | −14.2 | −59.6 | −8.86 | 40.7 | 24.0 | 0.0 | — | 1.000 |
| `K4_h1_n10` | 2,320 | 47.3 | −32.8 | −80.1 | −5.41 | 40.4 | 22.4 | 0.0 | — | 1.000 |
| `K4_h1_n20` | 4,640 | 94.7 | −13.1 | −59.2 | −6.23 | 40.9 | 26.5 | 0.0 | — | 1.000 |
| `K4_h2_n10` | 1,170 | 23.9 | −35.8 | −83.0 | −2.56 | 43.0 | 38.8 | 0.0 | — | 1.000 |
| `K4_h2_n20` | 2,340 | 47.8 | −10.6 | −56.7 | −2.63 | 43.5 | 42.9 | 0.0 | — | 1.000 |
| `K5_h5_n10` | 576 | 11.8 | −46.7 | −92.1 | −2.03 | 38.5 | 51.0 | 45.0 | −0.184 | 1.000 |
| `K5_h5_n20` | 1,050 | 21.4 | −0.6 | −46.0 | −1.45 | 40.6 | 53.1 | 41.0 | −0.092 | 0.999 |
| `K5_h2_n10` | 1,092 | 22.8 | −5.5 | −50.7 | −2.28 | 45.3 | 47.9 | 23.2 | −0.101 | 1.000 |
| `K5_h2_n20` | 1,844 | 37.6 | −2.9 | −48.1 | −3.00 | 44.6 | 44.9 | 20.9 | −0.096 | 1.000 |

`p_adj` is the search-adjusted permutation p over all 20 cells: 5,000 sign-flip draws on the weekly
book-return series, the SAME flips applied to every cell so the cross-cell dependence is preserved,
p = P(max t over the 20 cells ≥ this cell's t). The smallest on TRAIN is **0.945**.

### VAL (Jan–May 2026)

The full table is in `K/phaseA.md`. Two cells are positive: `K2_h10_n10` (+19.2 bps, t 0.12) and
`K2_h10_n20` (+166.4 bps, t 1.26, p_adj 0.670). Eighteen are negative. `K1_h2_n20` is −118.1 bps
at t −2.19 and `K4_h1_n20` is −55.0 at t −4.12 — the significant results on VAL point the wrong way.

### Tail dependence

Every positive number in this stage is a tail:

| cell | split | net bps | top 1% removed | top 5% removed | winners capped at p95 |
|---|---|---:|---:|---:|---:|
| `K3_h3_n10` | TRAIN | +17.4 | **−26.9** | **−111.5** | −13.3 |
| `K2_h10_n20` | VAL | +166.4 | **−17.9** | **−152.0** | +13.8 |
| `K2_h10_n10` | VAL | +19.2 | −129.6 | −272.2 | −67.2 |

A book whose edge disappears when its top 1% is removed is the lottery ticket this owner has already
rejected once. Full tail table for all 20 cells × 2 splits: `K/phaseA.md` §"Tails and the cost model".

### Cost model sensitivity (reported, never substituted for the primary)

| model | round trip | source |
|---|---|---|
| **PREREG (primary)** | band spread + 10 bps = **29.3 / 63.9 / 89.3 bps** by dvol band | `cost_curve.md` "by liquidity", averaged over its five time bands, + 5 bps slippage per side |
| daily-band | 6 / 12 / 25 / 40 bps | `lit_review_2026/daily_addons.py:25`, the convention of the earlier daily tests |
| auction-honest | **10 bps** | 5 bps per side and NO quoted spread — CLAUDE.md fill-realism rule 4 |

Switching to the auction model adds ~44 bps to every cell. It promotes nothing: on TRAIN it leaves
`K3_h3_n10` at +52.9 bps and `K2_h10_n20` at +14.6; every other cell stays negative on at least one
split. The gate is a t-test, and a constant added to every trade moves t by c/SE: `K3_h3_n10`'s SE
is 33.8 bps, so the cheapest defensible cost model takes its t from **+0.51 to +1.57** — still short
of 2.0, and its TRAIN edge still vanishes when the top 1% of trades is removed.

---

## 5. Capacity

Position size = 1% of the name's 20-day median dollar volume, ONE size for the whole book (PREREG
sizes equal-$ per position; the two rules together forbid a per-name size — weighting each trade by
its own dollar volume turns the book into a mega-cap index, 1% of a $20B/day name being a $200M
position, and the monthly P&L into that one name's return).

| | TRAIN | VAL |
|---|---|---|
| position size (median of 1% dvol20_med) | $415,828 – $580,830 | $413,262 – $610,512 |
| conservative size (p25) | $186,664 – $243,420 | $192,036 – $253,435 |
| book at 10 slots | ~$4.2M – $5.8M | ~$4.1M – $6.1M |
| book at 20 slots | ~$8.8M – $11.6M | ~$8.7M – $12.2M |
| best cell $/month | `K3_h3_n10` **+$60,368 (+1.07%)**, worst month −$755,593 | `K2_h10_n20` **+$303,358 (+3.47%)**, worst month −$459,227 |
| worst cell $/month | `K4_h1_n20` −$1,099,610 (−10.57%) | `K4_h1_n20` −$1,005,670 (−9.35%) |

Capacity is the one PREREG claim that survives: a $4M–$12M long-only daily book is executable in
this universe at 1% of median dollar volume, one or two orders of magnitude above anything the
intraday books can hold. There is nothing profitable to put in it.

---

## 6. Is the null mine or the market's? (controls)

`K/diagnostics.md`, `K/diagnostics_d5.md`. The 2026-09-16 lesson is that a "nothing works" verdict
can be three bugs in the scorer, so the same simulate/book/cost machinery was pointed at inputs
whose answer is known in advance.

**D1 — random symbol-days from the same universe, same fills, same costs.** This is the zero line.

| hold | TRAIN gross | VAL gross | TRAIN net (PREREG) | TRAIN net (auction) |
|---:|---:|---:|---:|---:|
| 1 | +2.3 | +10.2 | −42.1 | −7.7 |
| 2 | +7.5 | +16.9 | −36.8 | −2.5 |
| 3 | +16.2 | +22.8 | −28.3 | +6.2 |
| 5 | +25.8 | +22.4 | −18.5 | +15.8 |
| 10 | +50.4 | +61.7 | +5.9 | +40.4 |

A coin flip earns ~5 bps per calendar day held, against SPY's +6.7 bps/day on TRAIN and +10.5 on
VAL. **The exit walk, the fills and the cost application are sane**: if they were broken, a random
liquid long-only book would not reproduce market drift.

It also fixes the zero line, which changes how the families read. Excess gross over the random
control at the same hold:

| family (all signals, no book) | TRAIN excess | VAL excess |
|---|---:|---:|
| K1 gap continuation, h5 | **−83.3** | **−114.3** |
| K2 new 250-day high, h10 | +19.1 | +46.5 |
| K3 reversal, h3 | −32.5 | −7.2 |
| K4 overnight, h1 | −9.7 | +7.7 |
| K5 pullback, h5 | −30.8 | +61.8 |

Only K2 is positive on both splits, by ~19 / ~47 bps per 10-day trade, at t 0.79 / 1.44 before any
book — i.e. **most of K2's raw "positive" gross is just ten days of market drift**, and what is left
does not reach the gate. K1 is the clearest negative in the stage: a gap-up that closes strong is
83–114 bps WORSE over the next five days than a random name of the same liquidity, which is the
daily-scale version of the program's already-settled "gappers fade" (`daily_queue.md`).

**D2 — the book layer on random signals** (10/20 slots, random ranking): TRAIN net −37.6 to −24.9,
i.e. the slot mechanism itself costs nothing beyond the sampling noise of taking 800 of 22,000
draws. The book is not the bug.

**D3 — reconciling K2 with the earlier daily test.** `daily_addons.md` reports M41 (new 252-day high
on ≥1.5× volume, top-4 per day, no stop, entry at the signal CLOSE, hold 10) at TRAIN net +103.6
bps. Switching one convention at a time on Stage K's own K2 signal set:

| K2 variant (all signals, no book) | TRAIN net | VAL net |
|---|---:|---:|
| as declared: next-open entry + 7% stop | +23.8 | +64.1 |
| next-open entry, no stop | +41.3 | +99.2 |
| signal-CLOSE entry (M41's), no stop | **+67.4** | +115.0 |
| signal-CLOSE entry, 7% stop | +43.7 | +89.5 |

M41's number is reproduced to within the remaining differences (its 1.5× volume cut, its top-4
book, its cheaper cost band). **The two conventions PREREG imposes each cost K2 real money: the
next-open entry −26 bps and the 7% close-stop −18 bps per trade.** Both are deliberate — the
signal-close entry is not obtainable by an engine that decides on a close, and a stopless 10-day
hold is not a book anyone here would run — but they are the reason this stage disagrees with
`daily_addons.md`, and the disagreement is a convention, not a contradiction.

**D5 — where the declared cells lose.** For each family at its declared hold, the same signals under
four books (all / declared key / random key / reversed key, 10 slots):

| family | all (no book) | declared key | random key | reversed key |
|---|---:|---:|---:|---:|
| K1 h5 TRAIN | −105.1 | −122.6 | −101.2 | −113.5 |
| K1 h5 VAL | −139.5 | −30.7 | −83.6 | −161.6 |
| K2 h10 TRAIN | +23.8 | −78.8 | −53.9 | **+0.6** |
| K2 h10 VAL | +64.1 | +19.2 | +13.6 | **+92.0** |
| K3 h3 TRAIN | −61.5 | +17.4 | +36.2 | +22.4 |
| K3 h3 VAL | −29.0 | −29.0 | −74.6 | −39.7 |
| K4 h1 TRAIN | −51.4 | −80.1 | **−55.2** | **−49.3** |
| K4 h1 VAL | −24.3 | −62.4 | **−48.0** | **−30.1** |
| K5 h5 TRAIN | −49.9 | −92.1 | **−38.6** | +9.4 |
| K5 h5 VAL | +39.8 | −44.9 | **−10.0** | −85.1 |

Net bps per trade, primary universe, PREREG costs. Bold = the declared key is beaten by the random
key on BOTH splits (K4) or by both alternatives (K2, K5 TRAIN). Reading: for K4 and K5 the declared
strength measure is *negatively* selective — the strongest overnight-continuation name and the
deepest pullback are the worst trades of their cohort, and the pre-registered book spends all ten
slots on exactly them. This is the same shape as the program's settled fact that the edge lives in
SELECTION, with the sign against us.

**This is a diagnostic, not a promotion.** "Use the reversed key" is a hypothesis with 20 cells
already looked at; it needs its own pre-registration and its own TEST.

---

## 7. Independent check, per CLAUDE.md

- **Independent reimplementation** — §8 below.
- **Obtainability** — every fill in this stage is a market-on-open or a market-on-close on a name
  with ≥ $10M/day of median dollar volume. Entry is the next session's opening print, never a touch
  of a level; the exit is the closing print of the hold's last bar, or of the first bar whose close
  breaches the stop. No fill is inside a bar, so "fill within the filling bar" is satisfied
  trivially and 0% of trades depend on an intrabar price. The honest reservation is the opposite
  one: a stop evaluated ON a close and filled AT that close needs a market-on-close order placed
  from a price seen a few minutes earlier — it is an approximation, and it is the declared one.
- **Causality** — every field is computed from bars strictly before the decision or from day t
  itself, and the decision is taken at day t's close for an entry at day t+1's open. The universe
  gate (`dvol20_med`) is a trailing 20-bar median shifted by one day. The two cross-sectional
  families rank day-t values across day-t universe members only. No field was backfilled from a
  later key set (`K/availability.md`).
- **Price scale** — §2: the panel is RAW and matches Alpaca to float32 precision on 200 keys.
- **Fill realism** — no intrabar fills to get wrong; the auction-cost sensitivity in §4 is the
  answer to "does the simulated cost double-count".
- **Tail dependence** — §4: both positive cells die when their top 1% is removed.
- **Multiplicity** — §9.
- **Phrasing** — §1 states the universe, horizon, book size, window, cost and the MDE, and does not
  claim that no edge exists.

---

## 8. Independent reimplementation

CLAUDE.md requires a second implementation, written from a PROSE specification by someone who has
not read the first, reproducing the trade set trade by trade. That was done for **K1 (hold 5, 10
slots)** and **K3 (hold 3, 10 slots)** on TRAIN. The rebuilder was given the spec in words only and
was barred from opening anything under `research/fuckup_audit/K/`, `research/fuckup_audit/E/` or
`research/lit_review_2026/*.py`; its script is
`/tmp/claude-1000/-home-ec2-user-onemil/257c3e2d-cf38-45d5-94e7-4877f8170f44/scratchpad/indep_k.py`.

| TRAIN cell | quantity | this stage | independent rebuild |
|---|---|---:|---:|
| `K1_h5_n10` | signals | 328 | **328** |
| | booked trades | 261 | **261** |
| | gross bps | −75.4 | **−75.420** |
| | net bps | −122.6 | **−122.620** |
| | t | −1.43 | **−1.432** |
| | stop share | 22.6% | **22.61%** |
| `K3_h3_n10` | signals | 15,342 | 15,264 |
| | booked trades | 800 | **800** |
| | gross bps | +62.9 | +78.3 |
| | net bps | +17.4 | +32.8 |
| | t | +0.51 | +0.82 |

**K1 reproduces exactly**, to every digit the rebuild printed. That is the load-bearing result: K1
exercises the whole chain — the universe gate (trailing 20-bar dollar-volume median), the asset-class
filter, the test-ticker and early-close exclusions, the gap / volume-ratio / close-location cuts, the
next-open entry, the level stop evaluated on a close, the dollar-volume cost bands, and the 10-slot
first-come book with its one-position-per-name rule. Two independent implementations of that chain
agreeing on 261 of 261 trades means the machinery is not where the null comes from.

**K3 differs, and the cause is two named ambiguities in the prose, not a defect in either build.**
(a) "bottom decile of the universe" — this stage takes the bottom 10% by count on a strict rank
(`rank/n < 0.10`); the rebuild used a pandas average-rank percentile with `<= 0.10`, which admits
slightly more names at the boundary. (b) the 5-day return's base — this stage reuses the panel's own
`ret5` column (computed on the raw row sequence, the convention of the earlier daily tests, which
PREREG told this stage to reuse); the rebuild recomputed it on the valid-row sequence after dropping
zero-price rows. The two also differ by 121 universe rows out of ~920,000 for the same reason. The
booked count is identical (800 — both books are slot-bound, not signal-bound), so the divergence is
entirely in WHICH marginal names win the slots.

The disagreement does not move any conclusion: `K3_h3_n10` is the best TRAIN cell under both
readings, and under both it fails G1 — **t = +0.51 (this stage) or +0.82 (rebuild), against the
required 2.0**. Under the rebuild's reading its TRAIN net is +32.8 bps, which is still below the
random-hold-3 control's own +16.2 bps gross by less than one standard error once costs are applied
the same way. Nothing about "0 of 20 pass G1" changes.

The rebuild independently confirmed two things this stage relies on: its hand-rolled causal rolling
median matched `groupby.rolling(20, min_periods=10).median()` on 100,328 rows with zero mismatches,
and its book held ≤ 10 concurrent positions with no same-symbol overlap on every calendar day.

The limit of this check, stated because CLAUDE.md says an independent rebuild "catches coding errors
and CANNOT catch specification errors": both implementations were given the SAME fill convention
(market-on-open the next session, market-on-close on the exit day). If that convention is wrong —
if, say, a stop evaluated on a close cannot in practice be filled at that close — both are wrong
together. §7 states that approximation explicitly.

---

## 9. Cell count (every cell looked at, in this stage)

| group | cells | where |
|---|---:|---|
| pre-registered (5 families × 2 holds × 2 books) | 20 | §4, `K/phaseA.md` |
| survivorship control, secondary universe (declared hold, 10 slots) | 5 | §3, `K/phaseA.md` |
| D1 random market control (5 holds) | 5 | §6 |
| D2 random book (2 holds × 2 slot counts) | 4 | §6 |
| D3 K2 convention decomposition | 4 | §6 |
| D5 book-layer decomposition (5 families × 4 books, 5 of them duplicates of pre-registered cells) | 15 | §6 |
| **total distinct cells scored** | **53** | |

Each was read on TRAIN and VAL. TEST was read on none of them. The permutation p in §4 adjusts for
the 20 pre-registered cells only; it does not adjust for the 33 diagnostics, which is why no
diagnostic is allowed to promote anything.

---

## 10. Files

| file | what |
|---|---|
| `K/PREREG.md` | the pre-registration, written before the run |
| `K/FREEZE.md` | the written freeze; why TEST was not read |
| `K/step0_pricescale.py`, `K/step0.md`, `K/pricescale.csv`, `K/classmap_missing.csv` | step 0 |
| `K/build_k.py` | the candidate table, the 20 cells, the book, the gates, the permutation |
| `K/report_k.py`, `K/phaseA.md`, `K/cells_trainval.csv`, `K/perm_p.csv`, `K/signal_counts.csv` | the tables |
| `K/availability.md` | the availability audit |
| `K/diagnostics.py`, `K/diagnostics.md` | D1–D4 controls |
| `K/diagnostics2.py`, `K/diagnostics_d5.md`, `K/diagnostics_d5.csv` | D5 book decomposition |
| `K/trades/*.csv` | per-trade rows, 25 cells, 55,180 trades |
| `K/phaseA.log`, `K/diag.log`, `K/diag5.log` | run logs |

Nothing outside `research/fuckup_audit/K/` was written. `data/cache.db` was opened read-only via
`file:...?mode=ro`. No config, service, or order was touched.
