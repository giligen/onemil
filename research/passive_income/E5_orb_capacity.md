# E5 — ORB Capacity: how far does the one edge scale?

Date: 2026-09-25. Data: `research/thermo/book_2025_26.csv` (live-config ORB book, 2025-01→2026-09,
`entered==1` = fills, read via `trading/orb_csv.read_orb_csv`) cross-joined on (symbol,date) against
`analysis_results/orb_features_20260923_2049.csv`; live executions from `data/trades.db` (`strategy='orb'`).
Script: computation reproduced in this session, not committed (scratchpad only).

## Bottom line

The ORB edge is **real but small (+0.105R/fill, +0.086R out-of-sample) and it is already close to its
capacity ceiling.** Under a standard square-root impact proxy, charged on entry and exit against the
09:30–09:35 opening-range volume, the modeled edge is at or below zero somewhere between **$375 and
$750 risk/trade** at the base impact coefficient, and honest live latency (not modeled by the impact
term at all) already costs ~70 bps per fill on top. Passive income from this alone is **$150–900/month**
on this account, not "multi-$1000s." Scaling risk-per-trade to get to $3–6K income directly destroys the
edge — this is a frequency problem (more qualifying setups), not a size problem, and today's setup
frequency (473 fills / 21 months = ~1/trading day) caps it there.

## Method and validation (independent check first)

`shares = R / stop_distance_$`, `stop_distance_$ = entry_price × range_size_pct/100` (book carries no
explicit stop column, so the task's fallback was used for every fill — same formula for BT and this
study). `participation = shares / range_total_volume` (the **5-minute** opening-range volume — a strict,
worst-case liquidity denominator, not ADV; see caveat below). `impact_bps_per_leg = k × sqrt(participation)
× range_size_pct`, charged on entry **and** exit (`2×`), `k ∈ {50,100,200}`.

- **Join check**: `analysis_results/orb_features_20260923_2049.csv` matched 100% of the 473 book fills on
  (symbol,date); `entry_price` and `range_total_volume` agreed to 0.00% median relative difference — the
  book's own fields are already sourced from this features file, so the join adds no new information here,
  only confirms no divergence between the two files the task named.
- **Reproduction check**: recomputing R from the book's own live-sizing column (`_sized_pnl/375`) gives
  mean **+0.1052R/fill, n=473** — matches the committed number in `git log` (`7d75aef`: "+0.105R/fill, t
  3.31, n 473") to 4 decimal places. OOS (excluding 2025-01→06) = **+0.0862R/fill, n=389**. This confirms
  the fill population and file reads are correct before any new claim is built on them.
- **Sizing-formula caveat (important, read before trusting the $375 row below)**: the task's fallback
  `shares = R/(entry_price×range_size_pct%)` does **not** reproduce live production sizing. At R=$375 it
  gives **2.72× more shares** (median ratio) than the book's actual `_sized_pnl` implies — live sizing uses
  the quintile/adaptive-mult scheme (`_quintile`, Q5-capped, `docs/CLAUDE_HISTORY.md`), which this study was
  told never to touch or reverse-engineer. That means **the impact-cost numbers below are likely pessimistic
  for today's actual size** (fewer real shares → less real participation → less real impact than modeled),
  but the *shape* of the degradation as R grows is still informative because shares scale linearly with R in
  both the proxy and (presumably) the real scheme.
- **Proxy caveat**: `range_total_volume` is 5 minutes of volume, not a full trading day. Using it as the
  impact denominator is conservative if size can be worked over the day, and understates true difficulty if
  it must clear in the opening range (which is when ORB actually enters). Report both readings; do not treat
  either as a venue simulation. Empirical square-root-law coefficients in the literature (large-cap, using
  **daily** volume) run Y≈0.3–1.0 (Tóth et al.; arXiv:2606.24019 confirms the law holds out-of-sample on a
  US large-cap); our k=50/100/200 bracket that range once converted for a 5-minute denominator, but there is
  no small-cap-intraday calibration in the literature to pin the true k — that is exactly why three k values
  are reported instead of one number.

## Capacity table (k=100, base case)

| R | n fills | mean gross R | mean net R (ALL) | mean net R (OOS) | $/month (ALL) | $/month (OOS) | worst month $ | fills >5% particip. | >10% | >25% |
|---|---|---|---|---|---|---|---|---|---|---|
| 375 | 473 | 0.286 | **+0.071** | **+0.019** | 596 | 187 | −3,118 | 7.8% | 2.5% | 0.2% |
| 750 | 473 | 0.286 | −0.019 | −0.072 | −314 | −1,393 | −8,021 | 19.2% | 7.8% | 1.5% |
| 1,500 | 473 | 0.286 | −0.145 | −0.200 | −4,888 | −7,784 | −21,094 | 37.4% | 19.2% | 5.9% |
| 3,000 | 473 | 0.286 | −0.323 | −0.382 | −21,822 | −29,706 | −56,472 | 54.3% | 37.4% | 14.8% |
| 6,000 | 473 | 0.286 | −0.575 | −0.639 | −77,716 | −99,400 | −172,088 | 70.2% | 54.3% | 31.7% |

(`mean gross R` uses the proxy sizing formula, not live sizing — see caveat; it is shown to isolate the
impact-cost effect from the sizing-formula offset. `median` net R is more negative than the mean at every
R — the book is right-skewed: a handful of big winners carry the average, the typical fill loses.)

**Sensitivity (k=50, optimistic / k=200, pessimistic), $/month, ALL fills:**

| R | k=50 | k=100 (base) | k=200 |
|---|---|---|---|
| 375 | 1,505 | 596 | −1,222 |
| 750 | 2,256 | −314 | −5,455 |
| 1,500 | 2,383 | −4,888 | −19,429 |
| 3,000 | −1,258 | −21,822 | −62,951 |
| 6,000 | −19,551 | −77,716 | −194,045 |

**Where the edge dies, by coefficient:** k=50 (optimistic): edge survives through **R≈1,500**, gone by
R=3,000. k=100 (base): edge is already marginal at **R=375** (OOS mean is only +0.019R — not
distinguishable from zero at n=389) and clearly gone by R=750. k=200 (pessimistic): edge is gone
**at today's size**. There is no k in this range at which R=3,000+ is sane — and separately, R=3,000–6,000
is 4.6–9.2% of this $65,083 account's equity risked on a single trade, which is a risk-of-ruin problem
independent of market impact.

## Live realized slippage (`data/trades.db`, strategy='orb', n=123 fills with both fields)

- Modeled `entry_price` vs actual `fill_price`: mean **−14.0 bps** (fills executed *better* than the
  reference — favorable, consistent with resting/limit-style entries).
- `drift_bar_to_fill_bps` (signal bar close → fill): mean **+15.9 bps** adverse.
- `drift_ask_to_fill_bps` (quoted ask at decision time → fill): mean **+72.0 bps** adverse — this is
  latency cost (loop → quote → order → fill), separate from and additive to the size-dependent impact
  modeled above. It already burdens every live fill at today's size and does not shrink if R shrinks; it
  will not go away by staying small. n=123 is early-live and directional, not final — flag, don't ship,
  until the live ledger has more fills (`orb_ramp_check.py` already gates ramp advance on 40 live fills).

## Monthly income, honest ranges

Basis: OOS mean +0.086R/fill (own computation above, live sizing), ~1 fill/trading day (473/21mo), current
$375 risk/trade, before any scale-up.

- **Calm** (upper end of the OOS distribution, good regime like 2025 per `research/orb_2023/REPORT.md`
  pooled +0.055–0.27R band): **+$600 to +$900/month**.
- **Normal** (pooled out-of-regime estimate, `+0.055 ± 0.046R`, `research/orb_2023/REPORT.md`): **+$300 to
  +$600/month**.
- **Stressed** (2024H2 out-of-regime cell, −0.007R/fill, `research/orb_regime_2024` project note; or a
  single bad month from the table above): **−$1,000 to −$4,000** in a bad month — the −$3,118 worst month
  at R=375/k=100 in this own computation is the concrete worst-single-month number on this exact book.

This is **not** "multi-$1000s/month passive" at current size, and pushing size to get there (R≥$1,500)
converts a thin positive edge into a reliably negative one under every impact coefficient tested, on top of
already being an imprudent fraction of a $65K account.

## Skew, tail, and passivity — the skeptic's read

- **Negative skew inside a positive mean**: median net R is *below* mean net R at every size in the table
  above — most fills lose a little, a few large winners carry the book (matches the standing house rule
  "always show the contribution distribution").
- **Not passive**: ORB requires the systemd scanner + entry-drain thread live at the open every day,
  catalyst gates, weekly `orb_weekly_refit.py`, and a human watching corpse-gate / spread-gate alarms
  (`project_orb_corpse_gate_defect_sep2026`). It is automated, not unattended — a real defect (like the
  corpse-gate bug) silently drops thousands of candidates/day for two months before anyone notices.
- **This is beta to "risk-on small-cap momentum,"** not an uncorrelated income stream: the +0.32R quarter
  that reversed to −0.13R on 2024H2 (`project_risk_on_tape_winners_sep2026`) shows the same population
  flips sign with the regime. Diversifying income by adding size to *this* book concentrates the account
  further into that one regime dependency rather than diversifying it.

## First experiment (small, falsifiable, cheap)

Do **not** raise R. Instead measure the frequency lever, which this study did not touch: pull
`data/research/databento` point-in-time universe for the **add-on pools** already coded
(`universe.addon_pools`, gap 4–5% $3–30 and gap 3–5% $30–50) — they are dry-run only pending owner GO per
`research/orb_seed_wide/PREREG_LIVE_UNION.md`. Cost: $0 (data already bought), 1 day: run the existing
`scripts/orb_weekly_refit.py` selection chain on the two pools' dry-run ledger and report fills/week and R/fill
against the same cadence bar (`docs/cadence_bar.md`) used for the core book. If both pools clear the bar,
more qualifying setups/week is the only lever in this study that raises $/month without raising R into the
impact wall documented above.

## Sources

- Own computation: `research/thermo/book_2025_26.csv` × `analysis_results/orb_features_20260923_2049.csv`
  join, 2026-09-25 session (473 fills, 2025-01→2026-09); `data/trades.db` query (`strategy='orb'`, n=123).
- `git log 7d75aef` (repo, 2026-09-25): "+0.105R/fill, t 3.31, n 473" — reproduced exactly in this study.
- `research/orb_2023/REPORT.md` (cited in CLAUDE.md ORB row): out-of-regime pooled +0.055±0.046R vs +0.27
  in 2025.
- Square-root market impact law: Tóth et al., "Anomalous price impact and the critical nature of liquidity
  in financial markets"; arXiv:2606.24019 "Empirical Confirmation of the Square-Root Law of Market Impact in
  a U.S. Large-Cap Equity" (2026) — coefficient range Y≈0.3–1.0 on daily-volume denominators, cited for
  context only; no small-cap intraday calibration exists in the literature, hence the k=50/100/200 bracket
  rather than a single claimed value.
- Account state ($65,083 equity, shorting 4x/2x, options not approved): computed-task context, as relayed
  by the harness, 2026-09-25.
