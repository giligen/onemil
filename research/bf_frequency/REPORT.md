# The bull-flag frequency-vs-edge frontier

2026-09-19. Owner ask, after approving the BF resume: *"deep dive and increase the
frequency significantly."* Then, mid-stage, two refinements of the **objective
function** — folded in below and both honoured:

1. *"We prefer frequency with stable consistent winners over catching money monsters
   if this contradicts each other."*
2. *"We must score on green weeks yes. Monster is ok if greens weeks dominate even if
   near breakeven."* — **this one supersedes (1)** and is the ranking used here.

Pre-registration: `PREREG.md` (written before any cell was scored). TEST seal:
`FREEZE.md`. Nothing ships from this stage. A survivor needs its own pre-registration
and the owner's word.

**Instrument**: the shipped `batch_backtest.py` Stage-2 on the honest regen-7 cache
`data/bull_flag_cache_causal_full_20260905.csv`, **regen-7's own exits** (never a
resim — BT_STATUS §2a measures the resim path at −$10.3K / 7.4% unfaithful, so the
+2R profit partial is OFF in every cell here). Knobs changed only through scratch
copies of `config.yaml`. Production config, caches, orders, services and crons were
never written. Baseline verified byte-equal to BT_STATUS **run A: 56 trades /
$139,113.67**. Normalization `--capital 50000 --risk 2000 --max-shares 10000`,
daily-loss rail **−$10,000 = −5u** (the live ramp's own proportion).

Splits: **TRAIN 2025** · **VAL 2026-01..05** · **TEST 2026-06..08 (sealed, §12)**.

---

## 0. The structural ceiling — state this before anything else

The cache holds **886 in-window detections over 20 months ≈ 44/month raw**, and the
raw detector is **edgeless**: −0.010R in 2025 (n=463), −0.077R in 2026 (n=423)
(`bf_decay/REPORT.md` row L0). The shipped P1 book takes **2.8/month**.

**10 trades/week = 43/month = essentially every detection in the cache.** That book
exists in this grid — it is cell **F6** — and on VAL it is **−0.073R/pick, −6.0R
total, 40.9% green weeks, 50% red weeks**. It is not reachable, it is not desirable,
and the frontier bends down well before it.

The real ceiling is tighter than 44/month. Keeping the four gates that are worth
keeping (§1), the cache's cascade over the full 20 months is:

```
886 raw -> 630 live universe name rule -> 603 universe table -> 565 price <= $20
    -> 378 pole >= 5% -> 308 VWAP gate -> 249 two-tier -> 245 intraday-change
    -> 242 after the day/slot/rail loop            = 12.1 trades/month
```

**~12 trades/month ~ 2.8/week is the maximum honest frequency of this cache for a
book that keeps price <= $20 and pole >= 5%.** Everything above that requires
relaxing one of those two gates, and §8 shows that is exactly where consistency dies.

---

## 1. The separation map — every gate, including the one never measured

`separation.py` -> `separation.csv`. Kept-R minus rejected-R at each gate's **own
position in the live chain**, per year and pooled. This extends
`bf_decay/REPORT.md` §1b, which applied the ADV20 gate as layer L2 but never
measured its separation.

| gate | n kept / rej | sep 2025 | sep 2026 | **sep pooled** | t | verdict |
|---|---|---|---|---|---|---|
| L1 live universe (name rule) | 630 / 256 | +0.341 | −0.058 | +0.109 | 1.07 | weak, but it is a *live* rule, not a choice |
| **L2 ADV20 >= 200K** *(never measured before)* | **371 / 259** | **−0.031** | **−0.017** | **−0.033** | **−0.29** | **WRONG SIDE in both years. Cuts 41% of the field for nothing.** |
| L3 entry price <= $20 | 335 / 36 | +0.282 | +0.121 | +0.213 | 0.89 | weak separator — but see §8, it is a **loss-size bound** |
| L4 pole gain >= 5% | 206 / 129 | +0.587 | +0.268 | **+0.455** | **3.12** | real, era-consistent |
| L5 conviction >= 1.8 | 91 / 115 | +0.089 | +0.332 | +0.205 | 0.96 | weak — cuts **57% of survivors** |
| L6 pole_bars <= 3 | 91 / 0 | — | — | — | — | inert (0 rejected) |
| L7 VWAP gate | 78 / 13 | +0.567 | +0.815 | **+0.704** | 1.83 | real, better in 2026 |
| L8a two-tier: MACD surgical-drop leg | 72 / 6 | +0.245 | −0.299 | +0.144 | 0.20 | near-worthless, wrong-signed in 2026 |
| **L8b two-tier: composite leg** | **68 / 10** | **+1.479** | **+0.931** | **+1.308** | **4.36** | **the strongest gate in the book** |
| L8 two-tier (both legs) | 62 / 16 | +1.094 | +0.644 | +0.959 | 2.62 | |
| L9 MACD-zone >= 1.5 (sizing, in-book) | 50 / 12 | −0.340 | +0.072 | −0.116 | −0.26 | sizing layer, wrong-signed in 2025 |
| **WHOLE P1 STACK picked vs rejected** | 62 / 568 | +0.673 | +0.629 | **+0.659** | **3.13** | intact |

**The headline: the ADV20 >= 200K gate is the largest cut in the whole cascade (−250
of 603, 41% of the field) and it is the only gate with a NEGATIVE pooled separation.**
It removes marginally *better* trades than it keeps, in both years. It had never been
measured because `bf_decay` treated it as plumbing.

Mechanism for why nobody noticed: the sizer already handles illiquid names. Median
dollar risk of the trades the gate KEEPS is **$2,801**; of the trades it REJECTS,
**$431**. The gate is removing trades the book was going to put one-sixth of the
money on anyway.

| ADV20 bucket | n | mean R | median $ risk | sum P&L |
|---|---|---|---|---|
| < 50K | 83 | +0.17 | $147 | +$201 |
| 50–100K | 69 | −0.09 | $347 | −$11,237 |
| 100–200K | 107 | −0.06 | $922 | +$4,521 |
| 200–500K | 130 | −0.06 | $2,403 | +$35,721 |
| > 500K | 241 | −0.01 | $3,022 | −$26,330 |

No monotonicity in R across liquidity — only in position size. This table is also
**why R/pick and total $ disagree** across the frontier, and why both are reported.

---

## 2. The cascade being moved (today's Stage-2 log, 886 in-window)

```
886 -> 630 universe name rule -> 603 universe table
    -> 353  volume >= 200K     (-250, 41% of the field)   <- negative separation
    -> 323  price <= $20       (-30)                      <- loss-size bound (§8)
    -> 197  pole >= 5%         (-126)                     <- real, t 3.12
    ->  85  conviction >= 1.8  (-112, 57% of survivors)   <- weak, t 0.96
    ->  72  VWAP gate          (-13)                      <- real
    ->  58  two-tier           (-14)                      <- composite leg t 4.36
    ->  56  book constraints   (-2)
```

---

## 3. TIER A2 — the single-gate ladders (everything else at P1)

Full table in `grid.csv`. Frequency and the week-level headline:

| rung | TRAIN tr/mo | TRAIN green wk % | TRAIN $ | VAL tr/mo | VAL green wk % | VAL $ | VAL R/pick |
|---|---|---|---|---|---|---|---|
| **P1 (baseline)** | 2.83 | **30.2%** | 112,281 | 3.00 | **40.9%** | 26,781 | +0.56 |
| conviction >= 1.5 | 3.92 | 34.0% | 151,133 | 3.60 | 36.4% | 15,208 | +0.37 |
| conviction >= 1.2 | 4.58 | 35.9% | 158,569 | 4.60 | 40.9% | 29,327 | +0.44 |
| conviction >= 1.0 | 5.08 | 39.6% | 164,851 | 5.00 | 45.5% | 27,843 | +0.42 |
| **conviction OFF** | **6.17** | **39.6%** | **165,162** | **5.80** | **54.5%** | **32,194** | +0.43 |
| ADV20 >= 100K | 4.17 | 41.5% | 118,012 | 4.00 | 45.5% | 32,305 | +0.65 |
| ADV20 >= 50K | 4.83 | 41.5% | 118,274 | 4.20 | 50.0% | 34,074 | +0.86 |
| **ADV20 OFF** | **5.75** | **45.3%** | 116,835 | **4.60** | **54.5%** | **34,313** | **+0.85** |
| price <= $25 | 3.00 | 32.1% | 139,734 | 3.00 | 40.9% | 26,781 | +0.56 |
| price <= $30 (off) | 3.00 | 32.1% | 139,734 | 3.20 | 36.4% | 20,178 | +0.45 |
| pole >= 4% | 3.08 | 28.3% | 115,450 | 3.00 | 40.9% | 26,781 | +0.56 |
| pole >= 3% (off) | 3.33 | 30.2% | 133,403 | 3.20 | 40.9% | 26,717 | +0.52 |
| MACD gate off | 3.17 | 30.2% | 121,987 | 3.00 | 40.9% | 26,781 | +0.56 |

Read it in one line: **only two ladders move frequency at all — volume and
conviction — and both raise the green-week share on both splits.** Price, pole and
the MACD leg are near-inert on frequency (+0 to +6 trades over 17 months) because
the conviction gate downstream was already removing whatever they would have added.

---

## 4. TIER A3 — book constraints, and buying power

`max_concurrent {3,5,8} x max_trades_per_day {5,8}`, on the P1 gate set:
**all six cells are byte-identical to P1** (34 TRAIN / 15 VAL trades, $112,280.52 /
$26,780.96). At 2.8 trades/month the rails never bind — there is nothing to win here.

Because A3 came back inert, it was extended (declared as A3b, counted in §9's
multiplicity) to **frontier density**: `F3_L / F4_L / F7_L / F8_L / F6_L` re-run at the
**LIVE rails (3 concurrent / 5 per day)**. Every one is **identical to its 8x8 twin**.
**The live slot rails do not bind even at 18 trades/month** — bull-flag entries are
spread across the session and across days, not stacked.

Buying power, at the live ramp L0 ($150 risk = x0.075 of the $2K normalization),
against the ~$66K account:

| cell | trades (TRAIN+VAL) | median notional @L0 | max notional @L0 | **peak concurrent notional @L0** | $200K per-position BP bind rate |
|---|---|---|---|---|---|
| P1 | 49 | $7,020 | $15,000 | **$19,359** (2025-03-20) | 0.0% |
| **F7** | 200 | $2,957 | $15,000 | **$19,359** (2025-03-20) | 0.0% |
| F8 | 208 | $3,350 | $15,000 | $27,845 (2025-02-27) | 0.0% |

**Frequency here is free in capital.** F7 takes 4x the trades of P1 at the *same*
peak concurrent notional ($19.4K on a $66K account, 29% utilisation) because the
added trades are the small-risk, low-ADV ones. At the $2K normalization the $200K
per-position ceiling clamps 4/58 rows for P1 and 8/249 for F7 — 3.2%, never binding
at L0.

---

## 5. TIER A4 — the frontier, ranked on GREEN WEEKS

Owner ranking: **% green weeks** (primary) · longest red-week streak · worst week ·
% green months · MDD · then total P&L. Weeks are counted over **every market week in
the split** (from SPY's calendar in `cache.db`), so a week the book did not trade is
a **FLAT** week — which is the whole point: P1's problem is that it does nothing in
most weeks.

### TRAIN (2025, 53 market weeks)

| cell | tr/mo | **green wk %** | flat wk % | red wk % | red streak | worst week | green mo % | MDD | total $ | top-5 share | WR | ex-top-5% R |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **P1 (shipped)** | 2.83 | **30.2%** | 56.6% | 13.2% | 2 | −12,857 | 81.8% | −14,237 | 112,281 | 71.1% | 64.7% | +0.69 |
| price <= $25 | 3.00 | 32.1% | 54.7% | 13.2% | 2 | −7,356 | 83.3% | −11,148 | 139,734 | 63.1% | 66.7% | +0.79 |
| ADV20 >= 100K (F1) | 4.17 | 41.5% | 43.4% | 15.1% | 2 | −12,857 | 75.0% | −15,772 | 118,012 | 67.7% | 62.0% | +0.50 |
| ADV20 OFF | 5.75 | 45.3% | 35.8% | 18.9% | 2 | −13,098 | 75.0% | −16,175 | 116,835 | 68.4% | 56.5% | +0.30 |
| conviction OFF | 6.17 | 39.6% | 30.2% | 30.2% | 3 | −12,857 | 90.9% | −17,116 | 165,162 | 51.8% | 58.1% | +0.37 |
| F3 (vol 50K+conv 1.5+px 25) | 6.83 | 45.3% | 30.2% | 24.5% | 3 | −7,607 | 91.7% | −15,226 | 177,186 | 50.6% | 58.5% | +0.36 |
| F4 (vol off+conv 1.5+px 30) | 8.00 | 49.1% | 24.5% | 26.4% | 3 | −7,243 | 91.7% | −14,722 | 179,805 | 49.9% | 58.3% | +0.28 |
| F4b (F4 + pole 4) | 9.08 | 47.2% | 18.9% | 34.0% | 3 | −14,425 | 91.7% | −30,644 | 147,200 | 61.0% | 53.2% | +0.17 |
| F4c (F4 + conv 1.2) | 9.83 | 49.1% | 15.1% | 35.8% | 3 | −12,784 | 83.3% | −27,883 | 170,782 | 52.5% | 53.4% | +0.16 |
| F5 (+pole 4, conv 1.2) | 11.50 | 45.3% | 11.3% | 43.4% | **6** | −18,445 | 66.7% | **−49,352** | 122,983 | 73.0% | 48.6% | +0.02 |
| **F7 (vol off + conv off)** | **12.42** | **52.8%** | **9.4%** | 37.7% | 3 | −13,098 | 75.0% | −19,091 | 169,954 | 50.4% | 53.0% | +0.16 |
| F8 (F7 + price 30) | 13.00 | **52.8%** | 9.4% | 37.7% | 3 | −12,784 | 83.3% | −22,737 | 182,210 | 49.2% | 52.6% | +0.14 |
| F6 (all gates off) | 18.42 | 50.9% | 5.7% | 43.4% | 3 | −16,200 | 66.7% | **−62,964** | 119,398 | 75.1% | 48.0% | −0.02 |

### VAL (2026-01..05, 22 market weeks)

| cell | tr/mo | **green wk %** | flat wk % | red wk % | red streak | worst week | green mo % | MDD | total $ | top-5 share | WR | ex-top-5% R |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **P1 (shipped)** | 3.00 | **40.9%** | 45.5% | 13.6% | 2 | −11,671 | 80.0% | −15,153 | 26,781 | 145.9% | 66.7% | +0.38 |
| ADV20 >= 100K (F1) | 4.00 | 45.5% | 40.9% | 13.6% | 2 | −11,671 | 80.0% | −15,902 | 32,305 | 124.0% | 65.0% | +0.52 |
| ADV20 OFF | 4.60 | 54.5% | 31.8% | 13.6% | 2 | −11,671 | 80.0% | −15,902 | 34,313 | 116.8% | 69.6% | +0.65 |
| conviction OFF | 5.80 | 54.5% | 22.7% | 22.7% | 2 | −11,335 | 60.0% | −14,817 | 32,194 | 133.5% | 65.5% | +0.33 |
| F3 | 5.20 | 40.9% | 36.4% | 22.7% | 2 | −11,335 | 80.0% | −15,566 | 22,240 | 180.1% | 61.5% | +0.44 |
| F4 | 6.00 | 45.5% | 31.8% | 22.7% | 2 | −11,335 | 60.0% | −18,945 | 15,769 | 254.1% | 60.0% | +0.25 |
| F4c | 8.40 | 50.0% | 27.3% | 22.7% | 2 | −9,702 | 80.0% | −18,540 | 31,568 | 136.1% | 59.5% | +0.20 |
| F5 | 10.20 | 45.5% | 22.7% | 31.8% | 2 | −10,429 | 60.0% | −18,540 | 17,096 | 251.4% | 51.0% | −0.04 |
| **F7 (vol off + conv off)** | **10.20** | **54.5%** | **13.6%** | 31.8% | **2** | **−9,702** | **100.0%** | **−15,446** | **38,967** | 110.3% | 56.9% | +0.09 |
| F8 (F7 + price 30) | 10.40 | 50.0% | 13.6% | 36.4% | 3 | −9,702 | 80.0% | −16,299 | 32,364 | 132.8% | 55.8% | +0.06 |
| F6 (all gates off) | 16.40 | 40.9% | 9.1% | **50.0%** | **4** | −10,429 | 40.0% | −19,962 | 4,044 | 1062.7% | 42.7% | −0.25 |

### The shape

**F7 is the only cell that is top-ranked on green weeks in BOTH splits** — 52.8%
TRAIN (tied best with F8) and 54.5% VAL (tied best with its own two component
rungs). Its two components each get part of the way (ADV20-off: 45.3 / 54.5;
conviction-off: 39.6 / 54.5); **the combination is what lifts TRAIN**.

Beyond F7 the curve turns over on every secondary metric at once: F8 loses 4.5pp of
VAL green weeks and gains a red-week streak; F5 doubles TRAIN's red-week streak to 6
and triples MDD; F6 puts half of VAL's weeks red.

---

## 6. What F7 actually buys, in the owner's own terms

| | P1 shipped | **F7** | change |
|---|---|---|---|
| trades / month (TRAIN / VAL) | 2.83 / 3.00 | **12.42 / 10.20** | **4.2x / 3.4x** |
| **green weeks** TRAIN / VAL | 30.2% / 40.9% | **52.8% / 54.5%** | **+22.6pp / +13.6pp** |
| flat weeks TRAIN / VAL | 56.6% / 45.5% | **9.4% / 13.6%** | −47.2pp / −31.8pp |
| red weeks TRAIN / VAL | 13.2% / 13.6% | 37.7% / 31.8% | **+24.5pp / +18.2pp** |
| longest red-week streak | 2 / 2 | 3 / **2** | +1 / 0 |
| worst week | −12,857 / −11,671 | −13,098 / **−9,702** | ~ / **better** |
| green months | 81.8% / 80.0% | 75.0% / **100.0%** | −6.8pp / **+20pp** |
| worst month | −12,857 / −4,757 | **−4,107 / +159** | **3.1x better / no red month** |
| MDD | −14,237 / −15,153 | −19,091 / **−15,446** | −34% / ~flat |
| total $ | 112,281 / 26,781 | **169,954 / 38,967** | **+51% / +46%** |
| top-5-trade P&L share | 71.1% / 145.9% | **50.4% / 110.3%** | less concentrated |
| win rate | 64.7% / 66.7% | 53.0% / 56.9% | **−11.7pp / −9.8pp** |
| R/pick | +0.91 / +0.56 | +0.39 / +0.30 | **−0.52 / −0.26** |
| ex-top-5% R/pick | +0.69 / +0.38 | +0.16 / +0.09 | near breakeven ex-tail |
| median R / IQR | 1.19 / 3.28 | 0.55 / 2.33 | smaller, tighter |

**The honest trade, in one sentence:** F7 converts about 47pp (TRAIN) / 32pp (VAL) of
FLAT weeks into roughly equal parts green and red, and the green half is bigger — so
the green-week share rises 14–23pp, the worst week does not get worse, the worst
month improves 3x, and total P&L rises ~50%, while R/pick halves and the win rate
falls ~10pp. The added trades are individually near-breakeven ex-tail (+0.09 to
+0.16R). **Under the owner's stated ranking that is a win; under an R/pick ranking it
is not.** Both are reported; the ranking used is the owner's.

Monster-dependence, reported as a diagnostic and **not** used to reject:
F7's top-5 trades are 50.4% (TRAIN) / 110.3% (VAL) of net P&L; top-1 is 11.7% / 26.7%;
top-10 is 83.4% / 158.3%. Every cell in this grid exceeds 50% on VAL — VAL is 5
months — but **F7 is the least concentrated frontier point on both splits**, and the
shipped P1 is *more* monster-dependent than F7 (71.1% / 145.9%).

F7's monthly series ($2K book; L0 = x0.075), **17 months, 14 green, every month
traded** (P1 has zero trades in Jan-2025):

```
25-01 -3,109(2)   25-02 +16,492(13) 25-03 +38,489(11) 25-04 +42,399(10) 25-05 +21,172(19)
25-06   -601(13)  25-07 +13,906(14) 25-08    +890(10) 25-09  +4,921(12) 25-10 +13,772(20)
25-11 +25,728(13) 25-12  -4,107(12) 26-01  +5,296(6)  26-02    +159(10) 26-03 +16,114(9)
26-04 +13,303(11) 26-05  +4,095(15)
```

### Contrast with the shape the owner wants to move away from
ORB at 8 slots over 2026-Q2/Q3: **23 weeks, ~11 flat (47.8%), two weeks carrying
+$1,596 while the other 21 netted −$543.** The shipped bull-flag P1 is the same
shape — **45–57% flat weeks** — and that is precisely what F7 fixes (**9–14% flat**).
The exposure being bought is week-level presence, not P&L per trade.

---

## 7. TIER A5 — the point-in-time universe defect (`batch_backtest.py:3409`)

Stage-2 filters the cache by `db.get_active_universe()` — **today's** 5,973-symbol
table — so delisted 2025 names silently vanish from the 2025 book. Bracketed by
re-running every frontier point with `--full-market` (universe snapshot off; the live
BF *name* rule still applies). Lower bracket = today's snapshot; upper = no deletion;
true PIT is between.

| cell | TRAIN $ (snapshot -> full) | TRAIN green wk % | VAL $ (snapshot -> full) | VAL green wk % |
|---|---|---|---|---|
| P1 | 112,281 -> **96,534 (−14.0%)** | 30.2 -> 30.2 | 26,781 -> 26,781 (0%) | 40.9 -> 40.9 |
| F3 | 177,186 -> 168,063 (−5.1%) | 45.3 -> 47.2 | 22,240 -> 22,094 (−0.7%) | 40.9 -> 40.9 |
| F4 | 179,805 -> 166,154 (−7.6%) | 49.1 -> 49.1 | 15,769 -> 15,623 (−0.9%) | 45.5 -> 45.5 |
| **F7** | 169,954 -> **160,832 (−5.4%)** | 52.8 -> **54.7** | 38,967 -> 38,821 (−0.4%) | 54.5 -> **54.5** |
| F6 | 119,398 -> 98,480 (−17.5%) | 50.9 -> 52.8 | 4,044 -> 3,898 (−3.6%) | 40.9 -> 40.9 |

**The defect inflates the SHIPPED book most: P1 loses 14.0% of TRAIN P&L when the
snapshot is removed, F7 only 5.4%** (the deleted names are 2025 losers, and the
low-frequency book concentrates in them). VAL is essentially unaffected (<=1%).
**Every conclusion in §5–§6 survives the bracket** — F7 stays the top green-week cell
on both splits under both universes, and its advantage over P1 widens.

I did not patch `get_active_universe()` to point-in-time in my copy: the Databento PIT
universe that exists for ORB has no BF-side loader, and building one is a separate
piece of work. It stays logged as a **measurement defect** (already `bf_decay` §5a
pre-registration queue item 2), now with a frontier-wide bracket around it.

---

## 8. Where consistency breaks — and it is NOT frequency

The single most useful result in the stage:

**Frequency itself does not break consistency. Relaxing the PRICE CAP or the POLE
GATE does — at any frequency.**

| move | tr/mo TRAIN | worst month | MDD | red-week streak | verdict |
|---|---|---|---|---|---|
| P1 -> F4 (drop vol, loosen conv, price 30) | 2.8 -> 8.0 | −12,857 -> −7,637 | −14,237 -> −14,722 | 2 -> 3 | fine |
| **F4 -> F4b (pole 5 -> 4)** | 8.0 -> 9.1 | **−7,637 -> −23,559** | **−14,722 -> −30,644** | 3 -> 3 | **breaks** |
| **F4 -> F4c (conv 1.5 -> 1.2, price already 30)** | 8.0 -> 9.8 | **−7,637 -> −18,629** | **−14,722 -> −27,883** | 3 -> 3 | **breaks** |
| F4b/F4c -> F5 (both) | -> 11.5 | −34,551 | **−49,352** | 3 -> **6** | destroyed |
| **F7 -> F8 (price 20 -> 30)** | 12.4 -> 13.0 | **−4,107 -> −13,484** | **−19,091 -> −22,737** | 3 -> 3 | **degrades** |
| P1 -> **F7 (drop vol + conv; price 20, pole 5 kept)** | 2.8 -> **12.4** | **−12,857 -> −4,107** | −14,237 -> −19,091 | 2 -> 3 | **holds** |

Mechanism, and it is simple: **the $20 price cap and the 5% pole floor are LOSS-SIZE
bounds, not edge filters.** Their separation is weak (+0.21 / +0.46R) but a $28 stock
at the same share risk carries several times the dollar risk of a $6 one, and a 3%
pole gives a wider stop for the same setup. Removing them does not move the hit rate
much — it enlarges the left tail, which is exactly what worst-month, MDD and
red-week-streak measure. The volume and conviction gates bound nothing (the sizer
already shrinks illiquid names to ~$430 of risk, §1), so removing them adds trades
without adding tail.

**So the frequency at which consistency starts to break is not a number of trades —
it is the moment either loss-size bound is relaxed.** Inside the bounded family the
cache's ceiling is ~12/month (§0), and consistency is still *improving* at that
ceiling. Outside it, consistency is already broken at 9/month (F4b).

---

## 9. Cells, multiplicity and power

**43 Stage-2 runs** (13 ladder + 6 book-constraint + 8 frontier + 5 live-rail +
2 bend-diagnostic + 10 full-market brackets, sharing baselines), scored on
**2 splits = 86 decision cells**, plus ~33 descriptive separation cells. The declared
budget in PREREG was <=32 runs / <=64+33 cells; A3b (live rails, 5 runs) and the two
bend diagnostics were added after A3 returned inert, and are counted here.

With 43 cells the expected largest |t| under a pure null is ~2.8–3.0. **Treat any
single point as a maximum over 43 cells, not a discovery.** What protects F7 from
being a maximum-picking artefact is not its t-statistic:

- it is the **top green-week cell in BOTH splits independently**, and its two
  component rungs each rank top-3 on VAL on their own;
- its mechanism was named by an **independent measurement** (§1's separation map,
  computed before the frontier was built) — the two gates it removes are exactly
  the two with no separation, one of them wrong-signed in both years;
- it **survives the survivorship bracket** (§7) with its advantage widening.

Power, per split, for F7 (R/pick vs 0): TRAIN t = 2.89, MDE80 = 0.38R;
VAL t = 1.49, MDE80 = 0.56R. P1: TRAIN t = 3.15, MDE80 = 0.81R; VAL t = 1.50,
MDE80 = 1.04R. **F7's instrument is roughly twice as sharp as P1's** over the same
window, purely from n.

**Claim bar** (PLAN §1): G1 = TRAIN t >= 2 -> **F7 passes (2.89)**. G2 = VAL same sign
and >= 55% of weeks green -> F7's VAL sign is positive and **63.2% of *traded* weeks
are green**; on the all-market-weeks denominator it is **54.5%**, i.e. right at the
bar. Both numbers are given because the bar predates the all-weeks convention; on the
stricter reading G2 is a marginal miss by 0.5pp.

**Live-exploration bar**: positive point estimate (TRAIN +0.39R, VAL +0.30R);
named mechanism (§1); bounded downside (L0 $150 risk, rails −750/−1050/−1200,
`daily_loss_limit` −750 unchanged, peak concurrent notional $19.4K on $66K, §4);
**resolution inside a quarter — partial**: at 10.2 trades/month a quarter is ~31
trades, MDE80 ~0.72R against a point estimate of 0.30R, so a quarter still cannot
confirm the *edge*. It **can** move the week-level measurement: ~13 traded weeks per
quarter instead of P1's ~6. **This is exploration to learn the week shape, not to
confirm an edge — say it to the owner in those words.**

**Phrasing rule.** No edge was *established* here. What was established is that two
gates in the shipped cascade have no measurable separation in either year (one of
them wrong-signed) while cutting 41% and 57% of the field, and that removing them
raises the green-week share on both splits — in THIS cache (886 detections), over
THIS window (2025-01 -> 2026-05 scored), at THIS book size (<=3 concurrent, L0 risk),
under regen-7's own exits, at the shipped costs. The smallest per-pick effect this
instrument could have seen is 0.38R (TRAIN) / 0.56R (VAL).

---

## 10. TIER B — verdict: NOT TRIGGERED, and costed

PREREG §4's trigger was: *run Tier B only if the highest-frequency surviving cell is
also the highest-frequency cell in the whole grid — i.e. the frontier is still rising
at its high end.*

**It is not.** The grid's high-frequency end (F5 at 11.5, F6 at 18.4 TRAIN / 16.4
VAL) is where every consistency metric collapses (§5, §8), and the survivor sits at
12.4 — below the grid's maximum. The frontier has **bent down inside the existing
cache**, so a rebuild that enlarges the raw field cannot help: a bigger field is a
*lower-quality* field by construction, and the binding constraint is the loss-size
bounds, not the detector's supply.

**Costing (measured, not estimated).** A `BF_MIN_POLE_CANDLES=2` Stage-1 rebuild was
timed on a single month (2025-03) writing to a scratch path via
`BT_CACHE_PATH_OVERRIDE` (**never** `--build-cache` into `data/`): it **had not
completed after 600 s** and was killed. That is **> 10 min/month -> > 3.5 wall-hours
for the 20-month window**, single-process, `nice -n 10`, with 1-min bars already warm
in `cache.db`. Not run. CLAUDE.md's `min_pole_candles` 3->2 record (2025 +18.3%,
2026 Jan–Apr −$16,007) remains an **AS-IS-stack** number and is **not quoted either
way for P1** — it has not been re-tested under P1, exactly as PREREG required.

---

## 11. RECOMMENDATION — one point

> ### **F7 — drop the ADV20 >= 200K gate and the conviction >= 1.8 gate. Keep everything else at P1.**
>
> `scanner.min_daily_volume: 200000 -> 0` and
> `trading.conviction_scoring.min_threshold: 1.8 -> 0` (or `enabled: false`).
> **Nothing else changes** — price <= $20, pole >= 5%, VWAP gate, two-tier filter,
> profit partial, regime sizing off, risk L0 $150, rails −750/−1050/−1200,
> `max_positions 3`, `max_trades_per_day 5` all stay exactly as shipped (§4 proves the
> live rails never bind, so they need not move).

**Frequency: 12.4 trades/month on TRAIN, 10.2 on VAL — 4.2x / 3.4x the shipped book,
~2.4–2.9 trades/week.** Not 10/week: §0 shows 10/week is the whole edgeless cache and
is cell F6, which is VAL-negative.

**Evidence** — the primary metric, both splits, both universes:

| | TRAIN | VAL |
|---|---|---|
| green weeks, P1 -> F7 | 30.2% -> **52.8%** | 40.9% -> **54.5%** |
| under `--full-market` (no survivorship snapshot) | 30.2% -> **54.7%** | 40.9% -> **54.5%** |
| flat weeks | 56.6% -> 9.4% | 45.5% -> 13.6% |
| longest red-week streak | 2 -> 3 | 2 -> **2** |
| worst week | −12,857 -> −13,098 | −11,671 -> **−9,702** |

**versus shipped P1 on worst month and MDD** (the two the owner named):

| | P1 | F7 | |
|---|---|---|---|
| worst month TRAIN | −$12,857 | **−$4,107** | **3.1x better** |
| worst month VAL | −$4,757 | **+$159** | **no red month in VAL at all** |
| green months | 81.8% / 80.0% | 75.0% / **100.0%** | TRAIN −6.8pp, VAL +20pp |
| MDD TRAIN | −$14,237 | −$19,091 | **−34% worse — the one real cost** |
| MDD VAL | −$15,153 | −$15,446 | flat |

At the live ramp **L0 (x0.075)**: F7 TRAIN ~$1,062/month, VAL ~$585/month,
MDD ~−$1,432, worst month ~−$308. Relative tool, never a forecast.

**The cost, stated plainly:** win rate falls 64.7% -> 53.0% (TRAIN) and 66.7% ->
56.9% (VAL); R/pick halves; TRAIN MDD worsens 34%; red weeks rise 18–25pp. The
owner's refinement explicitly accepts near-breakeven added trades when green weeks
dominate — that is exactly this trade, and it is taken for that stated reason.

**Why these two gates and not others (mechanism, not fitting):** the ADV20 gate is the
*only* gate in the cascade with a **negative pooled separation** (−0.033, negative in
both years separately) while cutting 41% of the field; the conviction gate separates
+0.205R at t = 0.96 while cutting 57% of survivors. The four gates kept — price <= $20
and pole >= 5% as loss-size bounds, VWAP (+0.70R) and the two-tier composite leg
(+1.31R, t 4.36) as the real separators — are untouched. §8 shows relaxing either
bound is what actually breaks consistency.

**Interaction with the profit partial, unmeasured and disclosed:** every cell here
runs regen-7's own exits, so the shipped **+2R / 50% partial is OFF** in all of them.
BT_STATUS §2b measured the partial as −$3.9K net / +$12.4K on 2026 at unchanged WR
and MDD — pointing the same way as F7 (more consistency, less tail). **F7 and the
partial have never been measured together**, and cannot be until a regen-8 builds a
cache with the partial in the Stage-1 walk. Do not add their effects.

**Nothing ships from this stage.** F7 needs its own pre-registration (a declared split
plan, an independent rebuild of its trade set from prose, a causality trace on
`avg_volume_20d` and `conviction_mult` at the decision bar, and a TEST plan) plus the
owner's word, per PLAN §1 and `feedback_independent_check_before_claims`.

---

## 12. TEST reveal

Written after §11 was committed. See `FREEZE.md`.

**Revealed 2026-09-19, once, for two cells only (P1 and F7), after §11 was committed
as `173c88f`.** TEST = 2026-06-01 -> 2026-08-31, **14 market weeks**. Nothing below
was re-ranked and §11 stands exactly as written.

| | P1 shipped | **F7 (recommended)** |
|---|---|---|
| trades (per month) | 7 (2.33) | 42 (**14.0**) |
| **green weeks** | **14.3%** | **35.7%** |
| flat weeks | 64.3% | 7.1% |
| red weeks | 21.4% | 57.1% |
| longest red-week streak | 1 | **5** |
| worst week | −$5,210 | **−$13,086** |
| green months | 33.3% (1 of 3) | 33.3% (1 of 3) |
| total $ | **+$52** | **−$28,284** |
| total R | −0.06 | **+0.16** |
| R/pick | −0.01 | **+0.004** |
| win rate | 42.9% | 35.7% |
| MDD | −$7,066 | −$45,010 |

Monthly: P1 `Jun +$6,181 / Jul −$1,856 / Aug −$4,272` (4/1/2 trades);
F7 `Jun +$2,409 (+1.7R, 12 tr) / Jul −$25,625 (−4.5R, 13 tr) / Aug −$5,068
(**+3.0R**, 17 tr)`.

**Read it honestly, in three parts.**

1. **The primary metric direction HOLDS.** F7's green-week share is 35.7% vs P1's
   14.3% — +21.4pp, the same direction and roughly the same magnitude as TRAIN
   (+22.6pp) and VAL (+13.6pp). Flat weeks fall 64.3% -> 7.1%. On the metric the
   owner named, F7 beats P1 on all three splits.
2. **Everything else on TEST is worse, and the red-week streak is the real damage.**
   F7 runs **5 consecutive red weeks** (P1's longest is 1, only because P1 is flat in
   9 of 14 weeks), its worst week is 2.5x P1's, and it loses **$28.3K** ($2.1K at L0)
   where P1 is flat. Red weeks are 57.1% — the first split where F7's red share
   exceeds its green share. That is a genuine failure of the secondary criteria.
3. **Neither book had any edge in this quarter, and that is the fairest reading.**
   R/pick is **−0.01 (P1) and +0.004 (F7)** — both exactly zero. F7 did not select
   worse than P1; it selected the same nothing, 6x more often. This is the same
   quarter `bf_decay` §6 already flagged (2026H2 = 3 P1 picks, mean −0.588R,
   t −1.06); F7 turns it into a 42-trade sample and confirms it is flat, not merely
   noisy. The dollar gap between the two comes from **sizing, not selection**:
   F7's Aug-26 is **+3.0R and −$5,068** — its small-risk added trades won while its
   full-size trades lost, the same R-vs-$ divergence §1 documents.

**What this does to the recommendation.** §11 stands as written (FREEZE.md forbids
re-ranking after a reveal), but it must be put to the owner with this attached:
**F7's week-shape claim survived the sealed quarter; its P&L and drawdown claims did
not.** TEST is 3 months and 14 weeks — MDE80 on R/pick is ~0.62R against a point
estimate of 0.004R, so it cannot refute the edge either. The defensible statement is:
*F7 reliably converts flat weeks into traded weeks, with more green weeks than P1 on
every split including the sealed one; whether those weeks are net profitable is
undecided, and in the one quarter where the underlying book earned nothing, F7
amplified the dollar loss and produced a 5-week red streak.*

**Therefore the live-exploration framing in §9 is the only honest one, and it
tightens:** if F7 is explored live, it should be at L0 ($150) with the existing
rails, explicitly to measure the week shape, and a **5-consecutive-red-week
observation is a pre-committed stop** — that is what TEST says the downside looks
like, and it is already inside the ramp's demotion rules (5 losers in a row / <= −6u).
