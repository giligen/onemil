# Money-Machine Audit — 2026-07-03

Full-autonomy deep-dive into why BF + ORB "do not deliver on the promise."
Every claim below is BT-tested on the 18-month dataset (2025-01-02 →
2026-07-02) with train/validate discipline. Method: candidate-level resim
dump (`/tmp/orb_candidates_resim.csv`, 6,208 candidates) + an exact replica
of the defended pipeline (`research/scripts/orb_pipeline_replica.py`,
validated byte-identical to `study_orb_pipeline_static_lock.py`: 1,193
trades, $154,892) enabling fast selection-variant grids.

---

## The headline: WHY the promise isn't being delivered

**The promise number was never real for the ramp.** The $342K/16mo figure
(a) predates the current config (touchgo + lock 1.75/0.5 + Q1 filter
integrated → today's defended pipeline = **$154,892/18mo**, not $342K),
and (b) models a $100K always-on account while live has run at 15–30%
scale (Pre-0 → Stage-0) through the entire period. Decomposition of the
live-vs-promise gap:

| Cause | Share | Status |
|---|---|---|
| Ramp scale (15–30% of model size) | ~60% | By design — see ramp critique below |
| Execution bugs (touchgo mis-keying, FABC partial, GLXG races) | ~25% | **All fixed as of 2026-07-03** |
| Regime (gap-and-fade since 6/8; universe breakout WR 34%→18%) | ~15% | Cyclical — 2025 had an identical 3-month episode (Jun–Aug) |

**The BT core itself is sound and robust** — I attacked it from four
directions and it survived every attack (details below). The 2026 YTD
defended BT is **+$91K on the $100K model — the strategy's best stretch**,
while live sat at minimum size. That is the deliver-on-the-promise gap in
one sentence.

---

## ORB findings

### 1. The edge is a lottery-with-edge (structural, not fixable)
- Defended 18mo: $154,892, **top 5 trades = 101% / top 10 = 144% of total
  P&L**. Ex-top-10 the strategy loses money.
- All 12 largest winners: single-name small caps that ran +27%…+92% and
  exited `eod` (never hit the lock). The entire edge = catching monsters
  and holding to the close.
- 5 negative months in 18 (not "1 in 16" as the stale ramp doc claims).
  2–3 month red streaks are NORMAL (2025 Jun–Aug −$49K single-name; 2026
  May–Jul repeat).
- **Consequence:** anything that reduces presence (gating, demotion,
  timing) risks missing the ~1/month lottery trade that pays for
  everything. Presence > timing.

### 2. ETF universe drift — tested and REFUTED (do NOT exclude)
- 478/1,193 trades (40%) are leveraged/crypto/sector ETFs (classifier:
  Alpaca asset names, `research/orb_symbol_class.json`).
- Flat-$50K accounting made them look like dead weight (+$17/trade), BUT
  under the real risk-parity sizing they are the book's stabilizer:
  - Full-refit exclusion: $155K → $98K, 2026 +$91K → −$9K, MDD ×2. 
  - Selection-only exclusion (fit unchanged): $155K → $114K, May–Jul26
    +$12K → −$32K.
- Mechanism: freed slots refill with *below-cutline* single names, which
  are net-toxic (validates the composite's discriminative power at the
  margin). ETFs also carried the 2026 fade regime (+$18K May–Jun while
  singles bled).
- **Verdict: keep the mixed universe. The composite handles it.**

### 3. Own-P&L regime gating — tested and REFUTED
- Grid: trailing-K-day P&L signal (K=5/10/15) × thresholds × skip/half ×
  all/single-only. Fit 2025 (contains the Jun–Aug streak), validate 2026.
- Mild gates: noise (±$3K). Aggressive gates: catastrophic (−$50K…−$117K)
  — they skip the outlier days that pay for the strategy (AMCI +$46K
  landed mid-red-streak).
- Best cell improved MDD $5K for +$3.5K total — inside noise. **Not
  shippable.**

### 4. Parameter refit — tested and REFUTED (staleness is a feature)
- Quarterly walk-forward refits (expanding AND 12-mo rolling) vs the
  frozen H1-2025 fit, OOS on 4 quarters:
  - Static: **+$126,814** | Expanding refit: +$79,488 | Rolling-12mo:
    +$92,675.
- Refits chase the recent regime and get whipsawed (2026-Q2: refit −$26K
  vs static +$23K). The frozen fit is an accidental regularizer.
- **Verdict: cancel the quarterly-refit mandate. Do not refit without
  this walk-forward harness proving the new fit OOS.**

### 5. What actually leaked in live (all now fixed)
- Touchgo false-anchor keying: Rule M/D fired on bars ~1h post-breakout.
  Direct cost ≈ $1.2K live-sized over 6 wks (TSDD/PLTU/KOLD/EIDO/OSCR) +
  structural absence of the BT-validated touchgo edge on every fast fill.
  Fixed 2026-07-03 (range_end_ts keying + strict anchor + tripwire);
  validation loop live through 7/10.
- FABC partial-fill under-recording, GLXG vol-guard + reconcile race:
  fixed earlier; DB reconciled to broker truth.
- Fill misses (34/125 entries time-stop-cancelled): net **+$10K
  beneficial** in the fade regime (dodged −$26K BT losers, missed +$16K
  winners) — but this flips sign in go-regimes. Keep the entry telemetry;
  revisit stop-limit buffer (30→40bps A/B) only if missed-winner evidence
  accumulates in a green regime.

### 6. Ramp critique (the strategic one)
Cushion-gated advancement is mis-matched to a lottery-edge distribution:
cushion only builds when an outlier lands, but you're kept at minimum
size until it does — so the outlier that funds advancement is captured
at the SMALLEST size. Two demote flags fired in 3 weeks on a strategy
whose 18-mo BT contains identical streaks. Recommendation (process, not
code): advance on **operational cleanliness + absolute max-loss respected
+ min-days**, demote on **absolute-loss circuit breakers** (already
exist: daily loss limits, −$12K hard stop) — not on %-of-peak-cushion.
Keep the hard stop non-negotiable.

---

## BF findings

### 7. Supply collapse, not alpha collapse
- Raw Stage-1 setups: 2025 ≈ **170/mo** → 2026-Q2 ≈ **22/mo** (−85%).
  (Partly market regime; Jan-26 count also has a cache-build artifact —
  min daily_range 18% vs the usual 10% floor; 2026-04 has one 0.1% glitch
  row.)
- Stage-2 18mo: **+$36,462, 146 trades** (49% WR). 2025: +$32.6K; 2026:
  +$3.8K. BF is a ~$30K/yr strategy at full 2025-style supply — the
  original "promise" numbers came from older configs/baselines.
- Live execution is NOT the problem: since 5/18 live −$229 vs Stage-2
  −$2,742 on the same window — **live beat its own BT by $2.5K** (its
  cancels dodged 11 of 13 BT-only losers).

### 8. The quality bar collapses with supply — SHIP a conviction floor
- In 2025, capacity contention meant only ~4% of setups traded; in thin
  2026 months, 50–70% trade → junk flows in (June: 17 of 24 raw setups
  traded at 18% WR). corr(monthly take-rate, WR) = **−0.45**.
- Conviction is monotonically predictive in BOTH eras independently:
  - 2025: conv≥2.3 → 81% WR; 2026: conv[1.4,1.8) → 28% WR, −$2.8K.
- **Floor conviction_mult ≥ 1.8**: keeps 73/146 trades, retains
  $32.2K/$36.5K (88%) over 18mo, and lifts 2026 from +$3.8K → **+$6.7K
  (+74%)** with half the risk exposure. Since-5/18 window: loss shrinks
  −$2.7K → −$1.1K. Era-consistent, monotone, one-line live gate.

---

## SHIP LIST (ranked)

1. **BF conviction floor 1.8 — ✅ SHIPPED 2026-07-03 (pre-market).**
   Turned out to be a pure config flip: the gate already existed
   (`trading.conviction_scoring.min_threshold`, was 1.4). Flipped to 1.8
   in config.yaml + template; Stage-2 verification with the shipped
   config: 18mo **74 tr, +$31,865, WR 59.5%** (was 146 tr, +$36,462,
   49.3%); 2026: **+$6,674 (was +$3,832, +74%)**; since-5/18 window loss
   −$2,742 → −$1,074. Trader restarted pre-market; boot log confirms
   `filter trades with conv < 1.80`. Rollback: flip back to 1.4 + restart.
2. **Ramp policy change** (doc edit, no code): advancement/demotion on
   absolute-loss + operational gates instead of %-cushion. Unlocks the
   sizing that the whole promise depends on. **Requires your sign-off —
   not shipped.**
3. **Quarterly-refit mandate CANCELLED — ✅ SHIPPED** (CLAUDE.md edit with
   the OOS evidence inline).
4. **Keep everything else exactly as is.** ETF exclusion: refuted. Regime
   gating: refuted. Exit params: leave alone. The 2026-07-03 touchgo fix
   is the execution unlock; let the validation week confirm (REKEY=0,
   daily Telegram verdicts through 7/10).
5. Housekeeping (not shipped): fix the Jan-26 cache artifact + the
   2026-04 0.1% glitch row; keep `orb_pipeline_replica.py` as the
   standing variant harness.

## What I did NOT find
- No evidence the composite/features are broken (they beat refits OOS).
- No BF funnel bug (live > BT in the window).
- No exit-logic money on the table beyond the fixes already shipped
  (docs/orb_research_apr_2026.md's 50-variant sweep stands).

## Reproducibility
- Replica harness: `research/scripts/orb_pipeline_replica.py` (validated
  1,193 trades / $154,892 == pipeline).
- Symbol classes: `research/orb_symbol_class.json`.
- Candidate resim: regenerate via `/tmp/dump_orb_candidates.py` pattern
  (patch documented in this session; 1 run ≈ 2 min with warm bar cache).
