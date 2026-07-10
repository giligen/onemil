# ORB news-catalyst study (2026-07-10)

**Question (owner)**: do the monsters have pre-market news catalysts? Can we use
news the way `../stupid-money` does? Top-down evidence → hypothesis → TRAIN/OOS.

**Data**: Alpaca News API (Benzinga) backfill for all 1,359 post-veto qualified
candidates Jan'25–Jul'26. Window: prev day 15:00 ET → trade day 09:35 ET.
Artifact: `data/research/orb_news_catalyst_20260710.csv`. Base rate: 18.7% of
candidates had ≥1 article. Zero lookahead: every newsy candidate's earliest
article precedes 09:30 ET (live fetch at ~09:31 sees identical information).

## Findings

1. **News alone is NOT the signal.** It enriches both tails (monsters ≥$5K: 31%
   newsy; big losers ≤−$1.5K: 33%; mid-tier ~16-19%) — volatility amplifier,
   not direction. News WITHOUT premarket volume is negative in all 3 eras
   (−$166/−$349/−$186 per trade): a headline nobody trades is a dud.

2. **The signal is the INTERACTION: news × high PM$** (PM$ ≥ $5,816,688, the
   already-shipped TRAIN cut). Combo-cell per-trade mean, full universe:
   **TRAIN +$1,580 (n=32) / 25H2 +$1,569 (n=26) / 2026 +$935 (n=39)** — the
   only feature besides PM$ itself to pass era-consistency, and ~10× stronger.
   Survives the continuous control: within the top PM$ tercile, news lift is
   +$1,335/+$1,294/+$806 vs +$105/−$27/+$104 per era.

3. **Mirror finding — the shipped PM mult is boosting a flat bucket.** pm_hi
   WITHOUT news: +$145/−$101/+$36 per era (n=76/100/117). 74% of PM-boosted
   trades carry no news and ~zero EV; the PM mult's BT lift lives in the
   news subset.

4. **Monster RATE is era-stable in the combo cell**: 28.1%/15.4%/12.8% hit
   rate on ≥$2K vs 5.6%/7.1%/7.7% for everything else. Rate-based, not
   5-lucky-draws — but see risk note below.

5. **LLM/keyword "real catalyst" classification adds NOTHING for longs.**
   Recap-only news (Benzinga "20 stocks moving premarket") performs equal to
   real catalysts in the combo cell (mean $1,461 vs $1,285) and holds AMCI
   +$23K + BNAI +$13.6K. A recap exists BECAUSE the stock is a recognized
   premarket mover — that's the confirmation we want. (Opposite of
   stupid-money's short-divergence use case, where catalysts are a blocking
   risk. Do not port the catalyst-quality filter.)

6. **Slot promotion is dead**: benched 2026 combo trades net −$3K. The lever
   is sizing only — selection untouched, zero slot-mechanics risk.

## Pipeline-integrated variants (shipped-config selected book, baseline $250,276 / MDD −$18,815 verified)

| Variant | TOT | Δ25H1 | Δ25H2 | Δ2026 | MDD |
|---|---|---|---|---|---|
| A: gate PM 1.5 on news | $244,618 | −7.9K | +3.4K | −1.1K | −$17,974 |
| **A2: combo-only 2.0** | **$301,518** | **+7.6K** | **+15.4K** | **+28.3K** | **−$18,174** |
| B: pm 1.5 + combo 2.0 | $307,176 | +8.9K | +12.0K | +29.4K | −$19,015 |

A2 passes the owner rule (profit up all 3 eras AND MDD better). Magnitude grid
is monotone 1.0→3.0 (TRAIN alone would pick the cap — classic unbounded-mult
smell; 2.0 = 1.5 (PM) × 1.33, or read as two stacked ≤1.5 mults à la
qmult×pm_mult precedent).

## Risk notes (honest)

- **Lift is monster-concentrated**: drop top-5 boosted trades → net lift goes
  negative (−$2.5K). Structural for a lottery amplifier (same shape as the
  Q2/Q3 mult correction), justified by the stable monster RATE, but any
  14-trade window will look like bleed.
- **Combo big-loser rate is RISING**: 3.1% → 7.7% → 10.3% per era (2026: 6× the
  rest). The boost doubles those losers too. Worst historical lift month:
  2025-05 at −$7.7K.
- Only 47 boosted trades in 18 months (~2.6/mo). Slow to validate forward.
- Stacks on the Monday 7/13 quintile-mult correction (also not yet
  live-validated). Two simultaneous unvalidated sizing changes muddy
  attribution.

## Recommendation

1. **Ship now (zero-risk): live news capture** — one batched news call at the
   PM prefetch (~09:31), log `has_news`/`n_articles`/headline per candidate,
   persist in pattern_data. No trading behavior change; builds the forward
   parity dataset. Fail-open (fetch failure → no news recorded → WARNING).
2. **A2 sizing ship = owner decision** (passes his rule; risks above). If
   approved, implement as news-gate inside `trading/orb_pm_mult.py` with
   `high_mult_news: 2.0` / `high_mult: 1.0`, env kill switch, parity tests.
   Rollback = 2 yaml lines.
3. Do NOT port the LLM catalyst classifier (finding 5).

Artifacts: `data/research/orb_news_catalyst_20260710.csv`,
`/tmp/news_backfill.py` (regen script), this doc.
