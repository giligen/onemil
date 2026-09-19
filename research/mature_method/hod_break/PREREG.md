# HOD-break — pre-registration (mature_method candidate #1)

Written 2026-09-19 **before any cell was scored**. Method: `research/mature_method/RUNBOOK.md`,
ten steps, in order. Prior reports inherited and NOT redone: `research/bf_zero/REPORT.md` §6a/§6b
(the honest SIP population and its two cohorts), `research/bf_zero/CAUSAL_FILTER_REPORT.md`
(0/12 cells, the anatomy, the measured NBBO), `research/green_weeks/REPORT.md` (the exit cells),
the 9/14–9/18 dry-run EOD checks.

## 0. Population, splits, membership

Population = the live-config HOD-break signals on the whole point-in-time market, Alpaca SIP tape:
`research/bf_zero/spec_trades.csv` (60,461 signals of `trading/hod_break.py` after the 9/15
re-fetch) for the gates measurable on it, and a fresh unfiltered bar pass
(`pass_breaks.py` → `breaks.csv`) for the gates whose REJECTED side `detect()` never records.

Membership cuts, applied identically everywhere (CAUSAL_FILTER §1):
early-close sessions 2025-07-03 / 2025-11-28 / 2025-12-24 out; NASDAQ test tickers
(`research/scripts/pit_listings.is_test_ticker`) out; names absent from `daily_bars` out.

Splits as the book already uses: **TRAIN** 2025-01-02→2025-12-31 (53 W-FRI weeks), **VAL**
2026-01-01→2026-05-31 (23), **TEST** 2026-06-01→2026-09-11 (15). **TEST is sealed**: no TEST
number is read until `FREEZE.md` exists in this directory carrying the recommendation.

The bar pass is seeded from symbol-days whose day high reaches **both** `open × 1.05` and **$19**
— the causal superset of every signal the live config can produce (level ≥ 5% above the open,
price floor $20, streamed-universe screen prev close ≥ 17). Relaxing the +5% floor DOWNWARD is
NOT superset-safe on this universe file (range ≥ 5% is an end-of-day membership), so the
`min_dist_open_pct` gate is laddered **upward only**; the downward direction is quoted from
REPORT §7's unbiased sample and is not re-measured.

## 1. Cost model (step 3)

Measured per-trade NBBO at the signal minute, Alpaca SIP, from
`research/bf_zero/causal_filter/nbbo.csv` (mean ask−bid over the signal minute; 99.0% quoted).
Charged with the score4 contract used by the causal-filter study:

    half = 0.5 × spread / R ;  net = rr − half − half × {stop 0.875, eod 0.412, target 0.0}[why]

plus the obtainability rail: an NBBO ask above the capped limit `level × 1.006` is **no fill** and
the row is dropped before the book. Signals outside the measured set (cells that open a gate) get
an **imputed** spread = the measured sample's median bps in their (price band × hour band) cell;
the imputed share is reported per cell. The §8 band constant is reported beside it as the
comparison arm only — the measured arm decides.

## 2. Declared cells

Baseline **B0** = the shipped live config: `consol_bars 5, consol_pct 0.04, min_dist_open_pct 5,
rv [1,5), min_price 20, min_r_pct 1, cap 0.006, target_r 2, last_entry_minute 840,
max_per_day 12, max_concurrent 4`, spread gates `max_spread_bps 100` and `max_spread_frac_r 0.15`,
book by `trading.hod_break.run_book`.

### Step 5 — gate-separation map (10 gates, each at its cascade position)
Kept-minus-rejected mean net R, n each side, t, per year and pooled; median position risk
(dollars at $100 risk) kept vs rejected.

| id | gate | rejected side measured as |
|---|---|---|
| Ga | `min_dist_open_pct ≥ 5` | ladder upward 5 / 7 / 10 / 15 (downward not superset-safe) |
| Gb | `rv_profile ∈ [1,5)` | rv < 1 and rv ≥ 5 break bars from the pass |
| Gc | consolidation K=5 within 4% | break bars valid at K=3/X=8% but NOT at K=5/X=4%, walked on the loose stop |
| Gd | `min_price ≥ $20` | fills under $20 (spec_trades) |
| Ge | `min_r_pct ≥ 1%` | breaks whose stop distance is < 1% of entry |
| Gf | `max_spread_frac_r ≤ 0.15` | measured spread > 15% of R |
| Gg | `max_spread_bps ≤ 100` | measured spread > 100 bps |
| Gh | `last_entry_minute 840` | fills 14:01→15:31 |
| Gi | `max_per_day 12` | the 13th+ first-come signal of a day |
| Gj | `max_concurrent 4` | signals refused a slot |

### Step 6 — frequency frontier (declared points)
Single-gate REMOVALS: **F-a** dist floor 5 (as-is, reference) · **F-b** rv band off ·
**F-c** consolidation loose (K=3/X=8%) · **F-d** price floor $5 · **F-e** r_min off ·
**F-f** 15%-of-R spread cap off · **F-g** 100 bps ceiling off · **F-h** last entry 15:30 ·
**F-i** 20/day · **F-j** 8 concurrent.
TIGHTENINGS: **T1** dist ≥ 10% · **T2** rv ∈ [1,3) · **T3** price ≥ $50 · **T4** r_min ≥ 2% ·
**T5** spread ≤ 8% of R.
COMBINED: **C1** both spread gates off · **C2** rv off + r_min off · **C3** price $5 + rv off ·
**C4** slots 20/8 + last entry 15:30 · **C5** the structural ceiling (every optional gate off:
rv off, r_min off, price $5, both spread gates off, last entry 15:30, slots 20/8, loose
consolidation) · **C6** T1+T2 (the tight point) · **C7** B0 + the declared decile veto below.

### The causal-filter study's dropped lead — ONE cell
**V-D5** — veto every signal whose fill minute `entry_m` falls in the **5th decile of `entry_m`
on TRAIN**, i.e. **[615, 624) minutes since midnight ET = 10:15–10:23 ET inclusive**. Edges
computed on the TRAIN live-config population of `causal_filter/features.csv` and printed here
BEFORE scoring: deciles 577 / 588 / 596 / 605 / **615 / 624** / 643 / 676 / 736 / 807 / 841;
that bucket is 784 TRAIN signals at −0.547 R gross (the anatomy's −0.541 on the pre-membership
set). Applied PRE-book, as a live filter would be (a post-book veto would leave slots empty).
This is the decile-level veto `CAUSAL_FILTER_REPORT` §3 named as its lead and then forbade itself.

**Cell count declared: 10 gate-map rows + 10 removals + 5 tightenings + 7 combined + 1 veto = 33
decision cells**, each scored on TRAIN and VAL. Everything else in the report is descriptive.

## 3. Ranking and the two bars (steps 7 and 10)

PRIMARY = **% green weeks** over EVERY market week in the split (a no-trade week counts FLAT and
is in the denominator). Then longest red streak, worst week, % green months, max drawdown. Total
P&L is TERTIARY. Dollars printed beside every ratio at the live `risk_usd: 100`.
Ex-top-1% / ex-top-5% are reported diagnostics, never rejection reasons.

- **Claim bar**: G1 = TRAIN mean net R > 0 with t ≥ 2.0 and ≥ 5 trades/week; G2 = VAL same sign
  AND ≥ 55% green weeks. Only a cell clearing G1+G2 opens TEST, once, after FREEZE.md.
- **Live-exploration bar**: positive point estimate on BOTH green weeks and dollars at $100 risk,
  a stated mechanism, bounded downside with a pre-committed stop, and resolution inside one
  quarter at the cell's own frequency.

## 4. Null (step 9)

For every reported cell: 2,000 draws permuting that cell's own per-trade P&L across its own
weeks with the per-week pick count held fixed; report observed green% vs the null mean and
[p5, p95]. A green% inside its own band is pick COUNT, not skill, and is reported as such.

## 5. Verdicts available

The dry run is already LIVE (`hod_break.enabled: true, dry_run: true`), so **SHIP-TO-DRY is the
current state, not a verdict**. The three available verdicts are
**STAY-DRY-AS-INSTRUMENT** / **SHIP-TO-LIVE-SMALL** (with the pre-committed stop and the exact
live config diff) / **STAY DEAD**.

## 6. Rails

Reproduction gate on §6a's live-config book before anything is quoted. One python process,
`nice -n 10`, `ulimit -v 3000000`. `bars_sip.db`, `data/cache.db`, `data/trades.db` opened
read-only. No config, `orb.yaml`, systemd unit, cron, order or cache is written. Availability
audit on every scored field. Independent rebuild: the bar pass reproduces `spec_trades.csv` on
the shared population before any cell is scored.
