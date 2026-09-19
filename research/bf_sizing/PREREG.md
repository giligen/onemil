# PREREG — is the bull-flag conviction score a sizer at all?

Written **before any cell was scored**. 2026-09-19. Owner ask (9/19):
*"ultrathink about sizing being data driven. Potentially our confidence score is fucked
up entirely."*

Nothing here ships. A survivor needs the owner's word.

---

## 0. The suspicion, stated as a testable claim

The conviction score is used four ways: a **gate** (`conviction_scoring.min_threshold`
1.8), a **sizing multiplier** (`combined_mult = min(3.0, risk_tier × conviction)` inside
`TradePlanner.create_plan`), the **risk tiers** (which multiply it), and it feeds the
two-tier filter's population split. Three prior facts motivate the audit:

1. `bf_frequency/REPORT.md` §1: the conviction GATE separates **+0.205R at t = 0.96**
   while cutting **57% of survivors**.
2. `bf_frequency/REPORT.md` §12: on the sealed quarter, F7's **Aug-2026 is +3.0R and
   −$5,068**. Positive in risk units, negative in dollars ⇒ size was large on losers and
   small on winners over that month.
3. The score is a hand-built 7-rule stack with bolted-on bonuses (V-reversal 0.4 → 1.0,
   per-tier MACD multipliers, final clamp [0.25, 3.0]). Its composite has never been
   tested for predictive content.

**H0 (the null being tested): conviction, and every component of it, has zero rank
correlation with realized R, and position size has zero rank correlation with realized
R.** The alternative of interest is not only "positive" — a *negative* size↔R
correlation is the defect the owner suspects and is reported as the headline either way.

---

## 1. Instrument (frozen before running)

- **Data**: `data/bull_flag_cache_causal_full_20260905.csv` (regen-7, 896 rows, 886 in
  2025-01-01 → 2026-08-31), read-only, **regen-7's own exits** — never `--resim-exits`
  (BT_STATUS §2a: the resim path is −$10.3K / 7.4% unfaithful). The +2R profit partial is
  therefore OFF in every cell here, exactly as in `bf_frequency`.
- **Pick sets**, taken exactly as produced by the shipped `batch_backtest.py` Stage-2 in
  `bf_frequency/runs/`: **P1 = `runs/P1.csv` (56 trades)** is the primary pick set;
  **F7 = `runs/F7.csv` (242 trades)** is a declared robustness arm, used because P1 holds
  only 7 TEST trades and 56 in total — too few to decide a sizing question. Selection is
  **held fixed** in every cell: no gate, no rail and no slot rule is re-run. The
  day-level daily-loss rail would in reality re-select under a different sizer; that is a
  disclosed limitation, not a modelled effect.
- **R is defined once**, identically to `bf_frequency/PREREG` §1:
  `R = cache_pnl / (cache_shares × (entry_price − stop_loss))`, taken from the **cache
  row**, so R is invariant to every sizing multiplier. Cross-checked to 1.3e-4 against
  the Stage-2 rows.
- **Read-only**: `cache.db` and `trades.db` opened `mode=ro`; `config.yaml`, `orb.yaml`,
  production caches, orders, services and crons are never written. Everything written
  goes under `research/bf_sizing/`.
- **Node**: one python process, `nice -n 10`, `ulimit -v 1500000`.

### 1a. The sizing model — how a cell re-sizes a fixed pick set

Stage-1 sizing, read out of the code (`backtest.py:3092` + `trade_planner.py:199`):
`combined = min(3.0, risk_tier × conviction)` → `shares = min(⌊risk_per_trade × combined
/ rps⌋, max_shares × combined)` → ADV participation cap → **then** `macd_zone_mult`
multiplies shares again (and can re-exceed the ADV cap). So the cache's `shares`
**already contain conviction × macd_zone**, as `batch_backtest.py:547` states.

The multipliers are therefore **divided out** rather than the sizer re-implemented:

```
rps_i        = entry_price_i − stop_loss_i
m_i          = conviction_mult_i × macd_zone_mult_i         (the shipped sizing pair)
base_shares_i = cache_shares_i / m_i                        (every cap of the build kept)
base_risk_i   = base_shares_i × rps_i
```

A cell supplies a multiplier `m'_i`; then

```
shares_i  = min( base_shares_i × m'_i , ⌊200000 / entry_price_i⌋ )   # shipped BP ceiling
pnl_i     = R_i × rps_i × shares_i
```

**Disclosed approximation**: where a share cap bound *after* the multiplier in the
original build, dividing out under-states `base_shares`. This makes every cell
conservative in the same direction and cannot create a sign flip in a correlation.

**Normalisation**: each cell is multiplied by ONE constant `k` so that its **mean dollar
risk over the TRAIN picks equals S0's**; the same `k` is applied to VAL and TEST. The
primary metric (% green weeks) is invariant to `k`; MDD, worst week and total $ are not,
which is why it is fixed on TRAIN only.

---

## 2. Splits — TEST IS SEALED

| split | window | role |
|---|---|---|
| TRAIN | 2025-01-01 → 2025-12-31 | diagnosis, every fitted constant |
| VAL | 2026-01-01 → 2026-05-31 | confirmation |
| **TEST** | **2026-06-01 → 2026-08-31** | **SEALED — `FREEZE.md`** |

TEST is scored **once**, after the recommendation is written into `REPORT.md` and
committed, and `FREEZE.md` records that commit and names the surviving cells. Every
constant any cell fits (S2's `C`, S3's median ratio, S4's threshold, S5's windows, every
normalisation `k`) is fitted on **TRAIN only**.

Weeks are ISO weeks over **every market week in the split** (SPY's calendar in
`cache.db`); a week the book did not trade is **FLAT**. This is `bf_frequency/score.py`'s
convention, and that module is reused verbatim for the week/month/MDD block.

---

## 3. PART 1 — declared diagnostic cells (descriptive; no rule is drawn from them)

Populations, each one a stage of the live cascade (`bf_frequency/REPORT.md` §2):
**P_all** (886) · **P_univ** (post live name rule, 630) · **P_gate** (the population
actually entering the conviction gate, 197) · **P_pick** (P1's 56) · **P_pickF7** (242).

- **D1** conviction decile table → n, mean R, median R, win rate, mean $ risk, sum $ —
  per year and pooled, on P_gate and P_univ.
- **D2** Spearman ρ(conviction, R) with two-sided p and a bootstrap 95% CI (10,000
  resamples, seed 20260919), on all five populations, per year and pooled.
- **D3 — the headline.** Spearman ρ(dollar risk, R) and ρ(dollar risk, sign(R)) on the
  same populations, plus the **$/R reconciliation**:
  `Σ pnl − mean(R)·Σ risk = Σ (R_i − R̄)(risk_i − risk̄)` — the covariance term IS the
  sizer's contribution in dollars, and it is reported as a dollar number per split.
- **D4** the same ρ and decile treatment for each of the seven `conv_*` components and
  for `macd_zone_mult`. A component whose ρ is negative in **both** years is named
  **anti-predictive**; one negative in a single year is reported as noise.
- **D5** the August-2026 case, trade by trade: R, dollars, size, conviction, on F7's
  17 August picks (the source of the +3.0R / −$5,068 pair), with the D3 covariance
  decomposition for that month.

**Budget: ~40 descriptive cells.** No cell in Part 1 selects a cell in Part 2 — Part 2's
seven cells are fully declared below, before Part 1 is run.

---

## 4. PART 2 — the seven sizing cells, selection held FIXED

Every cell supplies `m'_i` for the SAME pick set. All constants fitted on TRAIN picks.

| cell | `m'_i` | what it tests |
|---|---|---|
| **S0** shipped | the P1/F7 Stage-2 `shares` exactly as run (conviction × macd_zone × risk tiers × BP ceiling) | the baseline |
| **S1** flat | `1.0` for every trade — the shipped base sizer with both multipliers OFF | **the null, and the one to beat** |
| **S2** inverse | `clip(C / conviction_i, 0.25, 3.0)`, `C = median(conviction over TRAIN picks)²` | **diagnostic only.** If S2 beats S0, the score is anti-predictive |
| **S3** vol-normalised | `clip( (rps_i/entry_i) / atr20pct_i ÷ med_TRAIN, 0.25, 3.0 )` — equal account-volatility per trade; ATR20 from `cache.db daily_bars` over the 20 sessions **strictly before** the entry date | is the stop distance, not the score, the thing worth sizing on? |
| **S4** binary | `1.4` if `conviction_i ≥ thr` else `0.7`; `thr = median(conviction over TRAIN picks)` | a crude, robust version of S0 |
| **S5** refit | rolling walk-forward: OLS of R on the seven standardised `conv_*` components, fitted on trades with `exit_date ≤ t − 5 trading days` (**embargo**) inside a **rolling 252-trading-day window**, minimum 30 trades or `m'=1.0`; `m'_i = clip(1 + (pred_i − μ_fit)/σ_fit, 0.25, 3.0)` | are hand-set weights the problem, or the features? |
| **S6** regime | `get_regime_multiplier(regime(date), …)` from `trading/regime_helpers` (A 1.25 · B 1.00 · C1 1.50 · C2 0.00), on a flat base | `regime_sizing` is OFF in P1 — turn it on as a sizing arm only |

Plus one **reported diagnostic, not a cell**: **S1b equal-dollar** — identical dollar risk
on every trade regardless of liquidity. It is *not obtainable* on the thin names (the ADV
participation cap exists for a reason) and is reported only to separate "the multiplier is
noise" from "the liquidity cap is the real sizer".

**Pre-committed expectation on S5** (CLAUDE.md, ORB `adaptive_mults`, 2026-09-08): a
walk-forward refit of a sizing map whipsawed there, and its apparent gain was a single
fill sized 3×. **S5 is therefore reported with the top-1 trade removed as well as whole**,
and if its gain over S1 disappears when the single largest contributor is dropped, it is
recorded as the same whipsaw and NOT recommended. This is written down before S5 is run.

**7 cells × 2 pick sets × 2 open splits = 28 decision cells** (+14 on TEST, revealed once).
Multiplicity is reported in the REPORT; with 28 cells the expected largest |t| under a
pure null is ≈ 2.7.

---

## 5. Scoring — the owner's metric (set 2026-09-19)

Ranked in this order, both pick sets, TRAIN and VAL:

1. **% GREEN WEEKS** over every market week (no-trade = flat) — PRIMARY
2. longest red-week streak
3. worst week
4. % green months
5. MDD
6. **total P&L — TERTIARY, reported, never ranked on**

**Monster concentration** (share of net P&L from the top 1 / 5 / 10 trades) is a
**reported diagnostic and never a rejection reason**. MDE₈₀ on R/pick is reported per
split. Both bars are reported: the **claim bar** (G1 = TRAIN t ≥ 2; G2 = VAL same sign
and ≥ 55% green weeks) and the **live-exploration bar** (positive point estimate, named
mechanism, bounded downside, resolution inside a quarter).

---

## 6. PART 3 — the recommendation rule, pre-committed

Written before any number exists, so the answer cannot be chosen after the fact.

- **(a) keep S0** iff S0 is the top cell on green weeks on **both** TRAIN and VAL for the
  primary pick set, **and** ρ(conviction, R) is positive in both years.
- **(c) ANTI-PREDICTIVE — fix immediately** iff **S2 (inverse) ranks above S0 on green
  weeks on both TRAIN and VAL**, or ρ(dollar risk, R) is negative in both years on
  P_univ, or the D3 covariance term is negative in both years. The component named is the
  one with the most negative two-year-consistent ρ in D4.
- **(b) NOISE — ship flat or vol-normalised** in every other case where S1 or S3 is at
  least as good as S0 on green weeks on both splits. Between S1 and S3, the tie-break is
  the declared ranking order in §5, top-down.
- If none of the three fires cleanly, the report says so and recommends nothing.

**The GATE verdict is reported separately from the sizing verdict**, because
`bf_frequency` already put the gate at t = 0.96 and the two uses of the score can fail
independently. The gate verdict is stated on the D2/D3 evidence at `P_gate` plus the
already-published F7 result; no new gate cell is run here.

---

## 7. What would make me wrong

- If `base_shares = cache_shares / m` is badly wrong because caps bound after the
  multiplier, every cell's dollars are distorted. **Check, run first**: the share of picks
  whose `base_shares` exceeds `⌊risk_per_trade/rps⌋` at the build's base risk, reported
  before any cell is scored.
- If R itself is contaminated by the exits, R↔size correlations measure the exit, not the
  sizer. R is taken from the cache row and the exits are identical across all seven cells,
  so this cancels in every comparison — but it does NOT cancel in Part 1's absolute ρ, and
  that is stated where it applies.
- TEST holds 7 P1 trades. Any TEST statement on the P1 arm is vacuous and will be
  labelled so; the F7 arm's 42 is the only TEST reading with any power.
