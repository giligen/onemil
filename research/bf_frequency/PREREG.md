# PREREG — bull-flag frequency-vs-edge frontier

Written **before any cell was scored**. 2026-09-19. Owner ask (9/19, after approving
the BF resume): *"deep dive and increase the frequency significantly."*

Nothing here ships. A survivor needs its own pre-registration and the owner's word.

---

## 0. The structural ceiling (stated before measuring, not after)

The honest regen-7 cache `data/bull_flag_cache_causal_full_20260905.csv` holds
**886 in-window detections** over 2025-01-01 → 2026-08-31 (20 months) ≈ **44/month raw**.
The shipped P1 book takes **56 trades ≈ 2.8/month**.

**10 trades/week = 43/month = essentially every detection in the cache**, and the RAW
detector is edgeless: −0.010R (2025, n=463) / −0.077R (2026, n=423) — `bf_decay/REPORT.md`
row L0. So R/pick must fall monotonically-ish as frequency rises toward 44/month, and
total R is the product of a rising count and a falling mean. **The frontier necessarily
bends down. The deliverable is WHERE it bends, not a promise to reach 10/week on this
cache.** Any claim of "10/week at P1's edge" would be arithmetically impossible here.

---

## 1. Instrument (frozen before running)

- **Cache**: `data/bull_flag_cache_causal_full_20260905.csv`, read-only, **regen-7's own
  exits**. No `--resim-exits` anywhere (BT_STATUS §2a: the resim path is −$10.3K / 7.4%
  unfaithful). Therefore the **+2R profit partial is OFF in every cell** — it only exists
  in the resim. Every number here is the "run A" exit spec, and the baseline is run A.
- **Engine**: the shipped `batch_backtest.py` Stage-2 (not a private re-implementation),
  so slot rails, daily-loss rail, risk tiers, BP ceiling and the MarketRegimeFilter are
  the shipped ones. Knobs are changed only through a **scratch copy of `config.yaml`**
  (`_ONEMIL_CFG`). Production config/caches/orders/services/crons are never written.
- **Normalization**: `--capital 50000 --risk 2000 --max-shares 10000`, daily-loss rail
  **−$10,000 = −5u**, the live ramp's proportion (BT_STATUS §3(i): the −$5,000 research
  rail is −2.5u and is the reason CLAUDE.md's worst month reads −$7.6K instead of −$12.7K).
- **Baseline, verified before pre-registering**: 56 trades / **$139,113.67** — byte-equal
  to BT_STATUS run A.

### R is defined once
`R = cache_pnl / (cache_shares × (entry_price − stop_loss))` — taken from the **cache row**,
never from the Stage-2 sized row, so R is invariant to risk tiers / regime mult / BP clamp.
This is the same R as `bf_decay/REPORT.md` (independently cross-checked there to 1e-6).

### Two books are reported for every cell
- **$ book** — Stage-2's own sized P&L at the $2K normalization (MDD, worst month, red months).
- **R book** — equal-weight sum of R (sizing-agnostic; the frontier's y-axis).

---

## 2. Splits — TEST IS SEALED

| split | window | role |
|---|---|---|
| TRAIN | 2025-01-01 → 2025-12-31 | ladders, frontier construction |
| VAL | 2026-01-01 → 2026-05-31 | confirmation, the recommendation rule |
| **TEST** | **2026-06-01 → 2026-08-31** | **SEALED — see FREEZE.md** |

TEST is scored **once**, for exactly **two** cells (shipped-P1 baseline and the single
recommended frontier point), and only **after** the recommendation is written into
REPORT.md. `score.py` refuses to print TEST without `--reveal-test`, and FREEZE.md
records the commit of the sealed recommendation. TEST holds ~3 months and ~3 P1 picks;
it is a sanity check, not a decision instrument, and will be labelled as such.

Running the full window in one pass and partitioning afterwards is identical to running
each split separately: the day loop, the daily-loss rail and the concurrency rail are all
**per-day**, so no cross-split state exists.

---

## 3. TIER A — declared cells (free, existing cache, no rebuild)

### A1 — separation map for EVERY gate, including volume
Kept-R minus rejected-R, evaluated **at that gate's own position in the live chain**
(i.e. the population entering the gate), for 2025 / 2026 / pooled, with n kept and
n rejected on each side, plus se and t on the pooled separation.

Gates: live universe name rule · **ADV20 ≥ 200K (never measured before — measured first)** ·
price ≤ $20 · pole ≥ 5% · conviction ≥ 1.8 · pole_bars ≤ 3 · VWAP gate · two-tier filter
(and its two legs) · MACD-zone ≥ 1.5 (sizing) · whole stack.
= **8 gate rows (+3 sub-rows) × 3 populations = ~33 descriptive cells.** No rule is drawn
from A1; it is the map that chooses which ladders are worth running (they are already
declared below, so A1 cannot steer the grid).

### A2 — single-gate ladders, one at a time, everything else at P1
| ladder | rungs | config knob |
|---|---|---|
| conviction | 1.8 (P1) · 1.5 · 1.2 · 1.0 · off | `trading.conviction_scoring.min_threshold` |
| volume | 200K (P1) · 100K · 50K · off | `scanner.min_daily_volume` |
| price cap | $20 (P1) · $25 · $30 (= off, the cache band) | `trading.bull_flag.max_entry_price` |
| pole gain | 5.0% (P1) · 4.0% · 3.0% (= off, the cache floor) | `trading.bull_flag.min_pole_gain_pct` |
| MACD gate | on (P1) · off | `…two_tier_filter.drop_extras_macd_below` 1.25 → 0 |

The "MACD gate" is the two-tier filter's `extras_macd_surgical_drop` leg — the only
MACD **gate** in the chain (`macd_zones` is sizing). The composite leg stays on.
**17 rungs = 13 distinct runs** (the P1 rung is shared by all five ladders).

Reported per rung, per split: **trades/month, R/pick, total R, total $, WR, MDD ($ and R),
worst month ($ and R), red months, ex-top-1% R/pick, ex-top-5% R/pick, t, MDE₈₀.**
The ex-top-1% and ex-top-5% columns are **mandatory** (four prior stages had ~1% of trades
carrying a whole effect).

### A3 — book constraints on the P1 gate set
`max_concurrent {3, 5, 8} × max_trades_per_day {5, 8}` = **6 cells (5 new runs)**.
Plus, at each: the **buying-power bind rate** — the share of trades the $200K per-position
BP ceiling clamps, and the **peak concurrent notional** rescaled to the live L0 risk
($150/trade) against the live account (~$66K), so "8 concurrent" is checked for
affordability, not assumed.

### A4 — the frontier
**5–7 combined points**, each one declared as *"rung X from ladder L1 + rung Y from ladder
L2 + …"*, spanning ~3/month to ~20/month. Each point's provenance is written next to it so
the curve is a **curve, not a search**. Plot / table: trades/month vs R/pick, total R,
total $, MDD, worst month — TRAIN and VAL side by side.

### A5 — the point-in-time universe defect (`batch_backtest.py:3409`)
Stage-2 filters the cache by `db.get_active_universe()` — **today's** 5,973-symbol table,
which deletes delisted 2025 names (bf_decay §5a: 10.4% of 2025 detections at mean −0.357R,
15.8% of 2026 at −0.036R). Quantified at **every frontier point** by re-running it with
`--full-market` (the same Stage-2, universe snapshot off; the live BF *name* rule still
applies unconditionally). Both numbers are reported. `--full-market` is the **upper**
bracket (no deletion at all), today's-universe is the **lower**; true PIT is between them.
A clean PIT fix is out of scope for this stage and is logged as a defect, not patched.

### Cell count and multiplicity (declared before running)
| block | runs | scored (run × split) |
|---|---|---|
| A1 separation map | 0 (descriptive) | ~33 |
| A2 ladders | 13 | 26 |
| A3 constraints | 5 | 10 |
| A4 frontier | ≤ 7 | ≤ 14 |
| A5 PIT bracket | ≤ 7 | ≤ 14 |
| **total** | **≤ 32 runs** | **≤ 64 + 33 descriptive** |

With ~32 cells on two splits, the expected largest |t| under a pure null is ≈ 2.7–2.9.
**Any single "best" point is reported as a maximum over 32 cells, not a discovery**, and
the report will say so in those words. No per-cell p-value is treated as evidence on its own;
only the pre-committed rule in §5 selects.

---

## 4. TIER B — trigger, declared in advance

Tier B (a **new cache build** at `BF_MIN_POLE_CANDLES=2`, raising the raw field above the
44/month detector ceiling) runs **only if** at the end of Tier A:

> the highest-frequency cell that satisfies the §5 survival rule is also the **highest-frequency
> cell in the whole Tier A grid** — i.e. the frontier is still rising (total R per month not
> yet falling) at the grid's high-frequency end.

If the frontier has already bent down inside the cache, Tier B cannot rescue it (a bigger raw
field is a *lower*-quality field, by construction) and the verdict is **costed but not run**.

CLAUDE.md's `min_pole_candles` 3→2 record (2025 +18.3%, 2026 Jan–Apr −$16,007) was measured on
the **AS-IS** stack, not P1 — that is why it is back on the table and why it may not be quoted
either way without a re-test under P1.

**Hard rule**: any Tier B build writes to `research/bf_frequency/cache_pole2.csv` via
`BT_CACHE_PATH_OVERRIDE`. **Never `--build-cache` into `data/bull_flag_cache_*`.** Wall-hours
are costed and stated **before** the build starts.

---

## 5. The pre-committed decision rule (written before any cell was scored)

A frontier point **survives** iff all of:

1. **R/pick > 0 on TRAIN and on VAL** (both splits, point estimate).
2. **ex-top-5% R/pick > 0 on TRAIN and on VAL** — the edge must not be one trade.
3. **Total R per month ≥ shipped-P1's total R per month, on TRAIN and on VAL** — more
   frequency must not cost total R. (This is the whole point of the exercise: frequency is
   only worth having if it buys more R, not just more trades.)
4. **Worst month (in R) no worse than 1.5 × shipped-P1's worst month (in R)**, on TRAIN
   and on VAL — the consistency bar of `feedback_consistency_over_pnl`, in R so it is
   rail- and sizing-independent.

**Among survivors, the recommendation is the one with the highest trades/month.**
Ties (within 0.3 trades/month) break on higher pooled TRAIN+VAL total R.
**If no point survives, the recommendation is "stay at shipped P1"** and that is a valid,
pre-committed outcome.

Reported **beside** the recommendation, never merged with it:
- **Claim bar** (PLAN §1): G1 = t ≥ 2 on TRAIN; G2 = VAL same sign **and ≥ 55% of weeks green**.
- **Live-exploration bar**: positive point estimate + a named mechanism + bounded downside
  (the L0 rails) + **resolution inside a quarter at that cell's own frequency** — i.e. does
  the cell produce enough live trades in 3 months to learn anything at its own MDE.
- **MDE₈₀ per split per cell**, always. A cell that clears the rule at MDE 1.2R is reported
  as clearing a rule, not as evidence of an effect.

---

## 6. What this stage will NOT do

- Not ship, not flip a flag, not touch `config.yaml` / `orb.yaml` / caches / orders / crons.
- Not quote a resim level (regen-7's own exits only).
- Not claim "no edge exists" — only "no edge was detectable in THIS cache, at THIS
  frequency, over THIS window, at THIS cost", with the MDE stated.
- Not quote a Tier A cell's P&L as a forecast. Stage-2 is a relative tool.

---

## 7. Artifacts

`PREREG.md` (this file) · `FREEZE.md` (the TEST seal) · `run_grid.py` (cell runner) ·
`score.py` (metrics, TEST-sealed) · `separation.py` (A1) · `grid.csv` · `separation.csv` ·
`frontier.csv` · `REPORT.md`.
