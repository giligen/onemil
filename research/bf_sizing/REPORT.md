# Is the bull-flag conviction score a sizer at all?

2026-09-19. Owner ask: *"ultrathink about sizing being data driven. Potentially our
confidence score is fucked up entirely."*

Pre-registration: `PREREG.md`, committed as `aba637c` **before any cell was scored**.
TEST seal: `FREEZE.md`. Nothing ships from this stage.

**Instrument**: the honest regen-7 cache `data/bull_flag_cache_causal_full_20260905.csv`
with **regen-7's own exits** (never a resim — BT_STATUS §2a measures the resim path at
−$10.3K / 7.4% unfaithful), and the pick sets produced by the shipped
`batch_backtest.py` Stage-2 in `research/bf_frequency/runs/` — **P1 (56 trades, the
shipped book)** and **F7 (242 trades, the frequency stage's recommendation)**. F7 is the
declared robustness arm and is the only arm with statistical power: P1 holds 34 TRAIN /
15 VAL / 7 TEST trades. Selection is **held fixed in every cell**; only size varies.

Splits: **TRAIN 2025** · **VAL 2026-01..05** · **TEST 2026-06..08 (sealed, §7)**.

---

## 0. The answer, in four lines

1. **Conviction does not predict R.** Spearman rho(conviction, R) is **−0.022 (2025) /
   −0.063 (2026), pooled −0.035, CI [−0.185, +0.115]** on `P_gate` — the exact
   population the gate and the multiplier act on. Negative in **both** years, and
   indistinguishable from zero in all of them.
2. **Position size does not predict R either.** rho($ risk, R) is **+0.026 (2025) /
   +0.000 (2026)** on the traded universe; the only nominally significant reading in the
   whole study (F7 2025, +0.198, p 0.016) **flips to −0.084 in 2026**.
3. **The sizer is not anti-predictive — it is inert, with a tail cost.** Under a
   permutation null that destroys the conviction ordering, the shipped sizer's
   green-week share sits at **p 0.87 (TRAIN) / 0.53 (VAL)** — i.e. *on the wrong side of
   the null median*. Its dollar advantage is **TRAIN-only** (p 0.059) and **does not
   replicate on VAL** (p 0.541).
4. **And it is not even the sizer.** The conviction × MACD multiplier explains **20–26%
   of the variance in position size**. The rest is the **ADV-participation cap and the
   stop distance**, and the covariance between the two is **negative** — the multiplier
   is systematically largest on the names liquidity has already shrunk.

**Recommendation (b): the score is NOISE as a sizer.** Details and the gate verdict in §6.

---

## 1. What the score actually does to size — measured, not assumed

Read out of the code (`backtest.py:3092` → `trade_planner.py:199`):
`combined = min(3.0, risk_tier × conviction)` → `shares = min(floor(2000 × combined / rps),
max_shares × combined)` → **ADV participation cap (2% of ADV20, hard)** → **then**
`macd_zone_mult` multiplies shares again and can re-exceed the ADV cap. The cache's
`shares` therefore already contain conviction × macd_zone (`batch_backtest.py:547`).

Decomposing `log(dollar risk)` on the shipped books:

| | P1 (56) | F7 (242) |
|---|---|---|
| shipped $ risk: min → max | $900 → $8,085 (**9x**) | $28 → $8,085 (**285x**) |
| base $ risk (multipliers divided out) | $167 → $2,556 | $15 → $2,968 |
| conviction × macd multiplier | 1.80 → 5.40 (median 3.96) | 0.40 → 5.40 (median 2.40) |
| var(log shipped risk) | 0.302 | 1.579 |
| — of which var(log **multiplier**) | 0.077 = **26%** | 0.308 = **20%** |
| — of which var(log **base**: stop distance + ADV cap) | 0.419 = **139%** | 1.591 = **101%** |
| — 2·cov(base, multiplier) | **−0.097** | **−0.270** |

**Two findings that nobody had written down.**

- **The conviction multiplier is a minority shareholder in position size.** Four fifths of
  the dispersion in how much money a bull-flag trade gets is the stop distance and the 2%
  ADV cap. When the owner asks whether sizing is data-driven, the honest answer is that it
  is *liquidity*-driven, and the conviction score rides on top of it.
- **The covariance term is negative in both books.** The multiplier is *largest* exactly
  where the ADV cap has already cut the position — the sizer believes it is putting 3x on
  its best setups while the liquidity cap is taking it back. That is not a designed
  interaction; it is two rules that have never been measured together.

---

## 2. PART 1 — does conviction predict anything?

Full output: `part1_out.txt`; CSVs `d2_conviction_rho.csv`, `d3_size_rho.csv`,
`d3b_cov_reconciliation.csv`, `d4_components.csv`, `d5_august2026_F7.csv`.

Populations (each a stage of the live cascade): **P_all 886** · **P_univ 630** (post live
name rule) · **P_gate 206** (the population entering the conviction gate: post ADV20,
price <= $20, pole >= 5%) · **P_pick(P1) 56** · **P_pickF7 242**.

### 2a. rho(conviction, R) — the score itself

| population | 2025 | 2026 | pooled | p | 95% CI |
|---|---|---|---|---|---|
| P_all (886) | −0.025 | −0.003 | −0.015 | 0.652 | [−0.083, +0.053] |
| P_univ (630) | +0.021 | −0.005 | +0.011 | 0.792 | [−0.071, +0.091] |
| **P_gate (206)** | **−0.022** | **−0.063** | **−0.035** | 0.621 | [−0.185, +0.115] |
| P_pick P1 (56) | +0.233 | — | +0.143 | 0.293 | [−0.116, +0.385] |
| P_pickF7 (242) | +0.039 | +0.111 | +0.068 | 0.289 | [−0.060, +0.199] |

**Zero everywhere, and NEGATIVE in both years on the one population that matters** — the
206 detections the conviction gate is applied to. The positive readings live on the
post-selection pick sets, where the score has already been conditioned on.

This is compatible with `bf_frequency` §1's **+0.205R mean separation at t 0.96** and
explains it: the gate's apparent value is a **tail effect** (a handful of large winners
happen to sit above 1.8), not a **rank** effect. The decile table on P_gate is not
monotone in any year — 2026's third quintile (conviction 1.50–1.70) is **−0.74R at an
11.8% win rate**, and the top quintile is **+0.19R**.

### 2b. rho(position size, R) — the headline the owner asked for

| population | 2025 | 2026 | pooled | p | 95% CI |
|---|---|---|---|---|---|
| P_all (886) | −0.006 | −0.007 | −0.009 | 0.781 | [−0.076, +0.055] |
| **P_univ (630)** | **+0.026** | **+0.000** | **+0.017** | 0.679 | [−0.062, +0.093] |
| P_gate (206) | +0.120 | −0.026 | +0.064 | 0.364 | [−0.074, +0.193] |
| P_pick P1 (56) | +0.013 | +0.077 | +0.022 | 0.873 | [−0.249, +0.288] |
| P_pickF7 (242) | **+0.198 (p 0.016)** | **−0.084** | +0.088 | 0.172 | [−0.035, +0.208] |

**The size-R correlation is not negative — it is zero, and sign-inconsistent across
years.** The single p < 0.05 reading in the entire study (F7 2025) reverses sign in 2026.
`rho(size, sign(R))` — does the book put more money on trades that merely *win* — is
negative in 11 of 15 population×window cells and significant in none.

**The dollar reconciliation** (`sum pnl = n·Rbar·riskbar + sum (R_i−Rbar)(risk_i−riskbar)`;
the covariance term IS the sizer's contribution, in dollars):

| population | 2025 cov term | 2026 cov term | verdict |
|---|---|---|---|
| P_all | −$48,092 | −$49,351 | hurts, both years |
| **P_univ** | **+$35,663** | **−$23,283** | **helped in 2025, hurt in 2026** |
| P_gate | +$9,793 | −$16,417 | same flip |
| P_pick (P1) | −$2,030 | −$671 | negative, but **−0.09% of the $139K book** |
| P_pickF7 | +$45,404 | −$21,364 | same flip |

On the **traded** population the sizer contributed **+$36K in 2025 and −$23K in 2026**.
That is not skill decaying; that is a coin landing heads then tails.

### 2c. Components — where a bolted-on bonus would show

rho(component, R), 2025 / 2026 / pooled. "ANTI-PREDICTIVE" = negative in **both** years.

| component | P_gate 25 / 26 | P_univ 25 / 26 (pooled p) | verdict |
|---|---|---|---|
| `conv_vwap_dist` | +0.104 / +0.039 | +0.106 / +0.090 (**p 0.011**) | **the only era-consistent PREDICTIVE component** |
| `conv_gap_fading` | +0.002 / −0.046 | +0.023 / +0.031 (0.538) | weakly positive, not significant |
| `conv_pole_gain` | −0.204 / +0.017 | −0.017 / +0.063 (0.786) | noise |
| `conv_spy_regime` | −0.080 / +0.053 | −0.017 / +0.018 (0.880) | noise |
| **`conv_flag_tightness`** | **−0.061 / −0.230** | **−0.069 / −0.142 (p 0.018)** | **ANTI-PREDICTIVE in both years on both populations** |
| `conv_retracement` | −0.074 / −0.051 | −0.071 / −0.027 (0.150) | anti-predictive, both years, both populations |
| `conv_vol_ratio` | −0.085 / −0.042 | −0.026 / −0.038 (0.437) | anti-predictive, both years, both populations |
| **`macd_zone_mult`** (sizing) | **−0.165 / −0.101** (pooled −0.120, p 0.085) | +0.059 / +0.100 (0.047) | **anti-predictive on the traded population, positive on the untraded one** |
| `conv_raw_score` (composite) | −0.019 / −0.063 | +0.022 / −0.005 (0.790) | noise |

**Named anti-predictive components** (negative in both years on both the gated and the
universe population): **`conv_flag_tightness`** (the strongest of them: p 0.018 pooled on
630 trades), **`conv_retracement`**, **`conv_vol_ratio`**. The V-reversal bonus is not a
separate column and could not be isolated.

**The sharpest single table in the study** — `macd_zone_mult` deciles on P_gate, the
206 trades the book selects from:

| macd_zone_mult | n | mean R | WR% | mean $ risk | sum $ |
|---|---|---|---|---|---|
| **1.00** (no boost) | 42 | **+0.71** | **71.4** | **$2,143** | +$62,643 |
| 1.00–1.80 | 41 | +0.31 | 39.0 | $3,019 | +$45,624 |
| 1.80 | 41 | +0.19 | 43.9 | $3,671 | +$43,531 |
| 1.80–2.00 | 41 | **−0.34** | 31.7 | **$3,988** | −$42,154 |
| 2.00 (max boost) | 41 | −0.02 | 48.8 | **$4,147** | +$5,132 |

**The MACD-zone multiplier gives the least money to the best bucket and the most money to
the worst one, on the traded population, in both years.** That is the single most
defect-shaped result in this study. It is p 0.085 on 206 trades — not significant — but
it is era-consistent and it is *monotone in dollars*.

### 2d. The +3.0R / −$5,068 August-2026 case, decomposed

F7's August 2026: **17 trades, +2.97R, −$5,068**. Per trade (`d5_august2026_F7.csv`):

```
LOSERS  (10, all ~ -1R)   TJGC $4,840  DUOT $3,260  AKAN $2,842  OESX $1,895
                          VCIG $1,800  SGLY $1,632  INCR $873  TMCR $637
                          SLE $115  SPHL $93
WINNERS (7)               TNON +5.01R $1,690 -> +$8,469     DBGI +1.04R $3,120
                          BCAB +2.29R $333   PRHI +2.25R $74   CLNN +1.84R $251
                          RDAC +0.52R $1,800  OPAD +0.74R $345
```

mean R **+0.175**, mean $ risk **$1,506**; `n·Rbar·riskbar = +$4,472`; **covariance term
= −$9,540**. rho(size, R) that month = **−0.346**. Mean risk on winners **$1,088** vs
losers **$1,799** — the book put **1.65x more money on the losers**.

**But the mechanism is not conviction.** Mean conviction on the winners was **1.47**, on
the losers **1.36** — the score actually leaned the *right* way. The inversion came from
the **base**: the four biggest R winners (PRHI $74, CLNN $251, BCAB $333, OPAD $345) were
thin names the **2% ADV cap** had already shrunk to a few hundred dollars of risk, while
the four biggest losers were the liquid names that got $2.8–4.8K. **August 2026 is the
liquidity cap inverting the book, not the confidence score.** `bf_frequency` §1 documents
the same structure across the whole cache: no monotonicity in R across ADV20 buckets, only
in position size.

---

## 3. PART 2 — seven sizing cells, selection held FIXED

Construction (PREREG §1a): `base_shares = cache_shares / (conviction × macd_zone)`;
`shares = min(base_shares × m', floor(200000/entry))`; `pnl = R × rps × shares`; each cell
renormalised by ONE constant so its **TRAIN mean dollar risk equals S0's**. Every fitted
constant is TRAIN-only. Green-week share is invariant to that constant; MDD, worst week
and total $ are not, which is why it is fixed on TRAIN.

**Disclosed approximation (PREREG §7, checked first):** for **30.4%** of cache rows a
share cap bound *after* the multiplier, so dividing out over-states the base for those
rows. It biases every cell in the same direction and cannot flip the sign of a
correlation. ATR20 was unavailable for 4/56 (P1) and 20/242 (F7) picks — those are sized
flat in S3, disclosed.

### 3a. P1 (the shipped book) — **sizing cannot move the week shape at all**

| cell | TRAIN green wk | VAL green wk | TRAIN worst wk | VAL worst wk | TRAIN MDD | VAL MDD | TRAIN $ | VAL $ |
|---|---|---|---|---|---|---|---|---|
| **S0 shipped** | **30.2%** | **40.9%** | −12,857 | −11,671 | −14,237 | −15,153 | 112,279 | 26,781 |
| S1 flat | 30.2% | 40.9% | −12,382 | −11,121 | −13,589 | −14,771 | 99,473 | 35,621 |
| S2 inverse | 30.2% | 40.9% | −15,008 | −9,721 | −16,270 | −14,176 | 91,675 | 36,151 |
| **S3 vol-norm** | 30.2% | 40.9% | −15,166 | **−6,779** | −15,166 | **−7,829** | **117,641** | **58,542** |
| S4 binary | 30.2% | 40.9% | **−10,326** | −14,412 | **−10,326** | −16,778 | 124,634 | 33,642 |
| S5 refit | 30.2% | 40.9% | −12,968 | −31,495 | −14,232 | −35,669 | 113,585 | **−2,519** |
| S6 regime | 28.3% | 40.9% | −14,706 | −11,900 | −16,152 | −11,900 | 95,285 | 39,558 |
| *S1b equal-$* | 30.2% | 40.9% | −3,935 | −3,521 | −5,079 | −4,688 | 33,659 | 9,044 |

**Every cell has an identical green-week share.** At 2.8 trades/month a traded week
usually holds one trade, and no sizing scheme can change that trade's sign. Only S6 moves
it, and only by *removing* trades (C2 days size to zero).

> **This is the most consequential single result for the owner's consistency objective:
> at P1's frequency, sizing is not a lever on week shape. Only frequency is.** It joins
> `bf_frequency`'s finding from the other side — and note that S0 is **last or
> second-to-last on VAL total $** of the seven.

### 3b. F7 (242 picks — the arm with power)

| cell | TRAIN green wk | VAL green wk | red streak T/V | worst wk T/V | green mo T/V | MDD T/V | $ T/V |
|---|---|---|---|---|---|---|---|
| **S0 shipped** | 52.8% | 54.5% | 3 / 2 | **−13,098** / **−9,702** | 75.0 / 100.0 | **−19,091** / −14,502 | 169,953 / 38,967 |
| S1 flat | 56.6% | 54.5% | 3 / 2 | −9,831 / −6,447 | 75.0 / 100.0 | −17,717 / −11,242 | 145,719 / 39,279 |
| S2 inverse | 56.6% | 50.0% | 3 / 2 | −12,072 / −8,463 | 75.0 / 100.0 | −16,808 / −13,964 | 121,776 / 35,138 |
| **S3 vol-norm** | 52.8% | **59.1%** | 3 / 2 | **−9,036** / **−3,771** | 66.7 / 100.0 | **−14,259** / **−5,384** | 141,013 / **55,813** |
| **S4 binary** | **58.5%** | **59.1%** | 3 / 2 | −12,179 / −9,295 | **91.7** / 80.0 | −15,981 / −13,625 | **175,584** / 48,061 |
| S5 refit | 47.2% | 50.0% | **5** / 3 | −18,887 / −8,332 | 58.3 / 40.0 | **−43,360** / −26,155 | 72,766 / 9,422 |
| S6 regime | 52.8% | 54.5% | 3 / 2 | −10,079 / −6,478 | 75.0 / 80.0 | −17,804 / −11,297 | 147,676 / 40,606 |
| *S1b equal-$* | 58.5% | 54.5% | 2 / 2 | −4,260 / −1,991 | 75.0 / 100.0 | −6,579 / −4,456 | 54,618 / 14,452 |

**S0 is last or tied-last on green weeks on TRAIN, and it has the worst worst-week and
the worst MDD in three of the four arm x split cells.** Its only wins are total P&L on
TRAIN (ranked TERTIARY by the owner, and see §4c).

**Noise band, stated before reading the table as a ranking**: at p ~ 0.55 the binomial SE
on the green-week share is **6.8pp on 53 TRAIN weeks and 10.6pp on 22 VAL weeks**. S4's
1.9pp TRAIN / 4.6pp VAL advantage over S1 is **one week in each split.** Every cell except
S5 is inside one standard error of every other cell on the primary metric.

### 3c. S5 — the whipsaw, exactly as pre-committed

PREREG §4 wrote down, before running it, that ORB's `adaptive_mults` refit whipsawed and
that S5 would be checked the same way. **BF behaves the same, and worse:**

| arm / split | S5 − S1 | largest single contributor | ex-that-trade |
|---|---|---|---|
| P1 TRAIN | **+$14,111** | KZIA 2025-12-23 +$4,931 | +$9,181 |
| P1 VAL | **−$38,140** | ANPA 2026-02-27 −$13,326 | −$24,814 |
| F7 TRAIN | **−$72,953** | EPSM 2025-10-03 −$16,980 | −$55,973 |
| F7 VAL | **−$29,857** | ALMU 2026-03-02 +$7,286 | −$37,142 |

It is not one trade this time — F7/TRAIN is −$56K even with the largest contributor
removed. S5 also posts the worst green-week share (47.2 / 50.0), the only 5-week red
streak, and a −$43K MDD. **Refitting the conviction weights walk-forward is strictly worse
than throwing them away.** CLAUDE.md's ORB rule transfers to BF: do not refit the sizing
map.

---

## 4. The decisive test — is any cell's edge the CONVICTION ORDERING?

Declared **post-hoc** and counted in §5's budget. `perm.py` -> `perm_null.csv`. Null: the
(conviction, macd_zone) pair carries no information about R, realised by shuffling the
multiplier across picks **within the split**, 2,000 times, and re-scoring the cell. (S0 is
rebuilt through the same `apply_cell` machinery as the null draws, so its dollars here
differ from §3b by ~1.6–4%; the comparison is internally consistent.)

| arm / split | cell | green wk | null mean +- sd | **perm p** | $ | null mean +- sd | **perm p** |
|---|---|---|---|---|---|---|---|
| F7 TRAIN | **S0** | 52.8% | 54.8 +- 2.6 | **0.872** | 172,645 | 139,140 +- 21,404 | **0.059** |
| F7 VAL | **S0** | 54.5% | 52.5 +- 4.4 | **0.532** | 41,459 | 42,745 +- 11,849 | **0.541** |
| F7 TRAIN | S4 | 58.5% | 55.4 +- 2.1 | 0.143 | 175,584 | 145,283 +- 17,037 | **0.036** |
| F7 VAL | S4 | 59.1% | 53.0 +- 3.7 | 0.131 | 48,061 | 42,176 +- 8,800 | 0.252 |
| F7 TRAIN | S2 | 56.6% | 55.1 +- 2.3 | 0.421 | 121,776 | 145,718 +- 20,654 | 0.870 |
| F7 VAL | S2 | 50.0% | 53.0 +- 3.7 | 0.923 | 35,138 | 35,239 +- 7,607 | 0.503 |
| P1 TRAIN | S4 | 30.2% | 30.0 +- 0.5 | 0.911 | 124,634 | 100,188 +- 11,778 | **0.018** |
| P1 VAL | S4 | 40.9% | 40.9 +- 0.0 | 1.000 | 33,642 | 35,527 +- 6,845 | 0.593 |

**Read it in three lines.**

1. **On the primary metric, the shipped sizer is on the WRONG side of a random
   reweighting** — p 0.872 on TRAIN and 0.532 on VAL. A coin-flip multiplier would have
   produced a better green-week share than conviction × MACD does, most of the time.
2. **Every dollar advantage in the study is TRAIN-only and fails to replicate on VAL.**
   S0: p 0.059 -> 0.541. S4: p 0.036/0.018 -> 0.252/0.593. This is the signature of a score
   whose weights were hand-tuned on 2025.
3. **S4 is not evidence for conviction.** Its edge over flat is one week per split and
   its permutation p on the primary metric is 0.143 / 0.131 — and it is the maximum over
   the cells that were looked at.

---

## 5. Cells, multiplicity, power

**7 cells x 2 pick sets x 2 open splits = 28 decision cells** as declared, plus the
S1b diagnostic (4), the post-hoc permutation block (12), and ~40 descriptive Part-1
cells. With 28 decision cells the expected largest |t| under a pure null is ~2.7;
**no cell in this study reaches it.**

**Power.** R/pick is identical across all seven cells by construction (sizing does not
change R), so the usual MDE on R/pick is not the right statistic here — it is the *week*
metric that discriminates. MDE80 on the green-week share: **+-9.5pp on 53 TRAIN weeks,
+-14.9pp on 22 VAL weeks** (two-proportion, p0 = 0.55). The largest observed gap between
any two non-S5 cells is **5.7pp**. **The instrument cannot resolve the sizing cells on the
primary metric at all**, and that is itself the answer to "is our sizing data-driven":
whatever the score is doing, it is smaller than what 75 weeks of this book can see.

**Claim bar (PLAN §1).** G1 = TRAIN t >= 2 on the effect: no cell clears it (best
permutation p 0.036 on a TERTIARY metric, on TRAIN only). G2 = VAL sign agreement and
>= 55% green weeks: S3 (59.1%) and S4 (59.1%) clear the 55% VAL threshold, **neither
clears G1**. **No cell in this study passes the claim bar.**
**Live-exploration bar**: S1 (flat) clears it trivially — it is the *removal* of a rule,
its downside is bounded by the same rails, and it resolves as fast as the book trades.

---

## 6. PART 3 — the recommendation

The rule was written in PREREG §6 before any number existed. Applying it literally:

- **(a) keep S0** — requires S0 top on green weeks on both splits of the primary pick set
  **and** rho(conviction, R) positive in both years. S0 is only *tied* top on P1 because
  **no** cell can move P1's week shape (§3a), and rho(conviction, R) is **negative in both
  years on P_gate**. -> **does not fire.**
- **(c) ANTI-PREDICTIVE** — requires S2 above S0 on green weeks on **both** splits (S2
  wins TRAIN 56.6 vs 52.8 but **loses VAL 50.0 vs 54.5**), or rho($ risk, R) negative in
  both years on P_univ (+0.026 / +0.000 — it is not), or the D3b covariance term negative
  in both years on P_univ (+$35,663 / −$23,283 — it is not). -> **does not fire.**
  *Disclosure*: the covariance term IS negative in both years on `P_all` (a population the
  book never trades) and on `P_pick(P1)` (−$2,030 / −$671, i.e. **−0.09% of a $139K
  book**). The clause's population is `P_univ`, named in the sentence before it. On any
  reading the effect on the traded population is sign-inconsistent across years, so
  "anti-predictive" is not a supportable claim.
- **(b) the score is NOISE as a sizer** — requires S1 or S3 at least as good as S0 on
  green weeks on both splits. **S1: 56.6 >= 52.8 and 54.5 >= 54.5. S3: 52.8 >= 52.8 and
  59.1 >= 54.5. Both qualify. -> (b) FIRES.**

> ### **(b). The conviction score is NOISE as a sizer. Take the conviction and MACD-zone multipliers out of sizing.**
>
> **Ship S1 (flat multiplier) as the immediate change** — it is the *removal* of a rule,
> needs no new data feed, and is the null the shipped sizer fails to beat on the primary
> metric under a permutation null (p 0.87 TRAIN / 0.53 VAL). Live equivalent: set the
> conviction and `macd_zones` **sizing** multipliers to 1.0 and raise `risk_per_trade` by
> **3.40x (P1 book) / 2.28x (F7 book)** to hold average exposure constant — *the rails,
> the ramp stage and the BP ceiling must move with it, per `docs/bf_p1_ramp.md`.*
> **S3 (volatility-normalised: equal account volatility per trade, size proportional to
> stop distance / ATR20) is the pre-registered next candidate**, not this one: it is the
> best cell on worst week and MDD on **both** splits of the powered arm
> (−$9,036/−$3,771 and −$14,259/−$5,384 vs S0's −$13,098/−$9,702 and −$19,091/−$14,502)
> and best on VAL green weeks — but its advantage is inside the noise band, it needs a
> causal ATR feed the live engine does not have, and it was missing ATR on 8% of picks.

**What (b) does to the book's week shape — stated plainly.** On P1: **nothing.** Green
weeks stay 30.2% / 40.9% whatever the sizer does; what changes is the depth of the weeks
(VAL total $ 26,781 -> 35,621, VAL MDD −15,153 -> −14,771, VAL worst week −11,671 ->
−11,121). On F7: green weeks **52.8% -> 56.6% (TRAIN)** and unchanged at 54.5% (VAL), worst
week **−13,098 -> −9,831** and **−9,702 -> −6,447**, MDD **−19,091 -> −17,717** and
**−14,502 -> −11,242**, at the cost of **−$24K of TRAIN P&L** (which §4 shows does not
replicate on VAL: S1 39,279 vs S0 38,967). **The trade is: give up a TRAIN-only dollar
advantage that fails its own permutation test, and buy a materially smaller left tail.**

### 6a. Two defects named, separately from the recommendation

1. **`macd_zone_mult` is the anti-predictive layer** (§2c): on the 206 traded-population
   detections its rho with R is −0.165 (2025) / −0.101 (2026), and its decile table pays
   the *least* money to the +0.71R / 71.4%-WR no-boost bucket and the *most* to the
   −0.34R bucket. It is not significant (p 0.085) and it is not the whole sizer, so this
   is **a named suspect, not a proven defect** — but if one layer is removed first, it is
   this one, and it is the layer CLAUDE.md records as shipped **without a feature flag**.
2. **The multiplier and the 2% ADV cap fight each other** (§1): cov(log base, log mult) is
   negative in both books. Whatever is done with conviction, that interaction should be
   made deliberate — it is exactly the "accidental rule" the owner has already ruled
   unacceptable (`feedback_deliberate_rules_no_accidents`).

### 6b. The GATE verdict — separate from the sizing verdict

`bf_frequency` §1 already put the conviction **gate** at **+0.205R separation, t 0.96,
cutting 57% of survivors**. This study adds the rank evidence on the same population:
**rho(conviction, R) = −0.022 (2025) / −0.063 (2026) on P_gate**, and a non-monotone decile
table in both years. The gate's mean separation is a **tail artefact**, not a ranking.

> **The gate should go, and that is the same change `bf_frequency` already recommended as
> F7** (`min_threshold: 1.8 -> 0` together with `scanner.min_daily_volume: 200000 -> 0`).
> This study supplies the mechanism the frequency study could only assert: the score it
> gates on has no rank content on the population it gates. **The gate decision belongs to
> the F7 pre-registration and the owner's word — nothing changes here.**

### 6c. What would make me wrong

- The instrument cannot resolve +-9.5pp of green weeks on TRAIN. A real sizing edge of
  3–5pp would be invisible here. **"Conviction is noise as a sizer" means "no effect
  larger than +-9.5pp of green weeks / +-0.19 of rank correlation was detectable in THIS
  cache (886 detections), over THIS window (2025-01 -> 2026-05 scored), at THIS book size
  (<= 3 concurrent, the $2K normalisation), under regen-7's own exits, at the shipped
  costs."** It does not mean no edge exists.
- R is taken from the cache row, so the exits are identical in all seven cells and cancel
  in every comparison — but they do **not** cancel in §2's absolute correlations.
- 30.4% of rows had a share cap bind after the multiplier, so the base is over-stated for
  them; it biases all cells the same way and cannot flip a correlation sign.
- P1's TEST arm is 7 trades and is vacuous by construction; only F7's 42 carries anything.

---

## 7. TEST reveal

Written after §6 was committed as `5cf2f84`. See `FREEZE.md`. TEST =
**2026-06-01 → 2026-08-31, 14 market weeks**. Revealed once, with
`python3 research/bf_sizing/part2.py --reveal-test`. **Nothing above was re-ranked and
§0–§6 stand exactly as written.**

### P1 arm — 7 trades (vacuous by construction, printed for completeness)

| cell | green wk | red streak | worst wk | MDD | total $ | mean $ risk |
|---|---|---|---|---|---|---|
| **S0 shipped** | **14.3%** | 1 | −5,210 | −7,066 | **+52** | 3,395 |
| S1 flat | **14.3%** | 1 | −8,043 | −9,794 | −7,442 | 3,945 |
| S2 inverse | **14.3%** | 1 | −8,030 | −9,953 | −7,196 | 3,918 |
| S3 vol-norm | **14.3%** | 1 | −10,955 | −11,913 | −8,061 | 3,303 |
| S4 binary | **14.3%** | 1 | −5,212 | −6,346 | −4,485 | 3,070 |
| S5 refit | 21.4% | 1 | −16,744 | −20,616 | −16,796 | 3,886 |
| S6 regime | **14.3%** | 1 | −8,033 | −9,782 | −7,433 | 3,940 |
| *S1b equal-$* | 21.4% | 1 | −1,312 | −2,481 | −70 | 1,086 |

### F7 arm — 42 trades (the only TEST reading with any power)

| cell | green wk | flat | red | red streak | worst wk | worst mo | MDD | total $ | mean $ risk |
|---|---|---|---|---|---|---|---|---|---|
| **S0 shipped** | **35.7%** | 7.1 | 57.1 | 5 | −13,086 | −25,625 | −45,009 | −28,283 | 2,061 |
| S1 flat | **35.7%** | 7.1 | 57.1 | 5 | −14,182 | −25,777 | −49,173 | −30,105 | 2,370 |
| S2 inverse | **35.7%** | 7.1 | 57.1 | 5 | −14,656 | −28,459 | −52,586 | −31,517 | 2,347 |
| S3 vol-norm | 28.6% | 7.1 | 64.3 | 5 | −13,998 | −21,318 | −47,538 | −35,844 | 1,699 |
| S4 binary | **35.7%** | 7.1 | 57.1 | 5 | −20,482 | −19,049 | −52,870 | −39,639 | 2,394 |
| S5 refit | **35.7%** | 7.1 | 57.1 | 5 | −22,329 | −20,625 | −47,766 | −20,839 | 2,800 |
| S6 regime | **35.7%** | 21.4 | **42.9** | **2** | −14,251 | **−12,004** | **−35,515** | **−16,353** | 2,058 |
| *S1b equal-$* | **35.7%** | 7.1 | 57.1 | 4 | **−4,192** | **−4,260** | **−10,657** | **+152** | 946 |

### Read it honestly, in four parts

1. **The study's central claim is CONFIRMED, and more starkly than on the open splits.**
   **Six of the eight cells post an identical green-week share** — 14.3% on P1 and 35.7%
   on F7. Whatever the sizer does, it does not move the week shape. The exceptions are
   S3 (28.6%, *worse*) and the two cells that change the *number* of live trades (S5 and
   S6 on flat weeks). **On the owner's primary metric, sizing is inert on TEST.** This is
   the same result as §3a, now on unseen data and on the powered arm.
2. **The recommendation's DOLLAR claim is NOT confirmed.** In this quarter S0 beats S1 on
   both arms: +$52 vs −$7,442 (P1) and −$28,283 vs −$30,105 (F7). Part of that is scale —
   S1 carries 16% (P1) / 15% (F7) more mean dollar risk in this quarter because the TEST
   picks' conviction ran *below* the TRAIN mean, and in a quarter where the book loses,
   whichever cell happens to be smaller wins. Scale-adjusting S1 to S0's mean risk still
   leaves S0 ahead on P1 (−$6,404) and roughly level on F7 (−$26,180 vs −$28,283). With
   **R/pick of −0.01 (P1) and +0.004 (F7)** — the book earned nothing in this quarter,
   exactly as `bf_frequency` §12 found — these dollar gaps are noise around zero, and
   §6's own permutation evidence says a TRAIN-only dollar advantage is what a hand-tuned
   score produces.
3. **S3 (volatility-normalised) FAILS its TEST.** It is the only cell to *lose*
   green weeks (35.7% → 28.6%), it posts 0% green months, and it is $7.6K worse than S0
   on dollars. §6 named it "the pre-registered next candidate"; TEST says that
   candidacy is weaker than the open splits suggested, and any future pre-registration
   must carry this reveal.
4. **The one thing that did work in this quarter is the one thing Part 1 pointed at, and
   it is not conviction.** `S1b` — equal DOLLAR risk on every trade — is the only cell
   that is **positive (+$152)**, with an MDD of **−$10,657 against S0's −$45,009** and a
   worst week of **−$4,192 against −$13,086**. Scaled up to S0's mean risk it is still
   +$347 with an MDD of −$24,298. That is the August-2026 mechanism (§2d) generalising:
   **the dispersion in the BASE size — the 2% ADV cap and the stop distance, which §1
   shows is 80% of all sizing variance — is where this book loses its money, not the
   confidence score.**
   **S1b is not obtainable** (you cannot put $2,000 of risk on a name whose 2% ADV cap
   allows $74), so it is not a recommendation and FREEZE.md forbids promoting it here.
   The obtainable form of the same idea is to **clamp the LARGE end** — which is exactly
   `trading.risk_cap` (`trading/bf_risk_cap.py`, built, default OFF, and CLAUDE.md's P2
   profile already measured a 2× variant at a better MDD). **That is the next
   pre-registration this program should write**, and it must be pre-registered, not
   inferred from this paragraph.

**What TEST does to the recommendation.** §6 stands as written. It must be put to the
owner with this attached: **the claim that sizing is inert on week shape survived the
sealed quarter and got stronger; the claim that flat sizing buys a smaller left tail did
NOT** — on TEST the shipped sizer's tail was the *second best* of the seven cells. TEST
is 14 weeks and a quarter in which the book had no edge at all, so it can neither confirm
nor refute a sizing effect of the size this study can see (MDE₈₀ ±14.9pp of green weeks
on 14 weeks is ±18.8pp). The defensible statement is:

> *The conviction score does not predict outcomes and does not move the shape of the
> book's weeks — on TRAIN, on VAL, and on the sealed quarter. Removing it from sizing is
> therefore free in week shape and unproven in dollars. The sizing variable that DOES
> move dollars is the base size dispersion created by the ADV cap, and clamping its large
> end is the next thing to pre-register.*
