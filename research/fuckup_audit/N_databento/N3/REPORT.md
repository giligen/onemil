# Stage N3 — simulator calibration on the published stocks-in-play ORB (2019-2023)

**Pre-registered** in `research/fuckup_audit/N_databento/PREREG.md` §N3, written before any pull:

> XNAS.ITCH daily → the paper's universe rule (top-20 relative volume at the open, price ≥ $5) → 1-min
> bars for those ~20 names/day. Run OUR simulator with OUR cost contract on THEIR rule in THEIR period.
> **Pass = same sign and same order of magnitude of R/trade per year as published. Fail = the simulator
> or the cost contract is wrong and every summer null is suspect.** One cell, no gate.

## The paper

Zarattini, Barbon & Aziz (2024), *A Profitable Day Trading Strategy For The U.S. Equity Market*,
SSRN 4729284 / SFI WP 24-98. The rule and the published numbers below are quoted from
`research/lit_review_2026/B_stocks_in_play.md` §1.1, which was written from the paper's full text on
2026-09-16 (this session's web-search budget was already exhausted and SSRN returns 403 to a fetch,
so the paper was not re-read here).

Rule, unchanged in our run: universe = NYSE+Nasdaq, opening price > $5, 14-day ADV ≥ 1,000,000
shares, 14-day ATR > $0.50. At 09:35, `RelativeVolume = ORVolume(t) / mean of the previous 14
ORVolumes`, `ORVolume` = 09:30–09:35 volume; keep RV ≥ 1.0 and take the **top 20** by RV. Direction =
the colour of the first 5-minute candle (up → long, down → short, doji → no order). Entry = a stop
order at the 5-min high (long) / low (short) placed at 09:35. **Stop = 0.10 × ATR14 from the entry —
that is the whole risk unit R.** No target; otherwise exit at 16:00. Costs: $0.0035/share, no spread.

Published results (2016–2023, their Table 2 / Figure 4): ORB+RelVol top-20 total return **1,637%**,
IRR 41.6%, Sharpe 2.81, **daily portfolio hit ratio 48.4%**, MDD 12%. Per-trade by RV bucket, net of
commission: RV < 1 → **−0.02R**, RV > 1 → **+0.08R**, RV > 30× → **+0.38R**. No per-year table is
published, so the comparison anchor is the per-trade R (+0.08R at RV ≥ 1; ≈ +0.14R implied for the
top-20 book from IRR 41.6% under their 20-position, 4×-leverage arithmetic) and the daily hit ratio.

## What was run

`fetch_daily.py` (XNAS.ITCH `ohlcv-1d` ALL_SYMBOLS 2018-05-01→2024-01-01, **priced before the first
byte: $19.3759 against the $30 stop**) → `fix_symbol_map.py` → `xnas_daily.parquet` (12,284,593 rows,
18,575 symbols, 1,426 sessions). `calibrate_venue_share.py` (one extra month, $0.2954) →
`build_pool.py` → `fetch_1m.py window` ($11.98, ALL_SYMBOLS 09:30–09:35 per session, 1,288 sessions)
→ `rank_top20.py` → `fetch_1m.py tape` ($5.480, the 20 selected names 09:30–16:00, 1,258 sessions) →
`sim.py`. **Total data spend $37.13** (PREREG allowed ≈$23 daily + ≤$60 for the 1-min).

25,135 picks over 1,258 sessions (median exactly 20/day), 3,764 distinct symbols, 12,304 long /
12,446 short / 385 doji. 4,091 picks never traded through the level; 20,659 fills under the paper's
convention, 20,651 under ours.

**Independent check (CLAUDE.md rule).** A second agent, given only a prose specification and forbidden
to read `sim.py`, rebuilt the zero-cost arm from scratch (`indep_sim.py`). Trade sets identical
(20,651 / 20,651, zero keys on either side only), `max |entry diff| = 0`, `max |rr diff| = 1.4e-14`,
pooled mean rr 0.173001 both. See `indep_check.md`.

## Result — per year, R per trade

| arm | 2019 | 2020 | 2021 | 2022 | 2023 | ALL | WR % | t (ALL) |
|---|---|---|---|---|---|---|---|---|
| **published anchor** (RV ≥ 1 bucket, their fill, their cost) | — | — | — | — | — | **+0.08** | 17–24 | — |
| A · paper fill (at the level) + their $0.0035/share | −0.439 | −0.332 | −0.292 | −0.117 | −0.242 | **−0.284** | 13.1 | −10.6 |
| A0 · paper fill, zero cost | −0.375 | −0.275 | −0.236 | −0.064 | −0.184 | −0.227 | 13.1 | −8.4 |
| **B0 · OUR fill (next-bar open), zero cost** | **+0.176** | **+0.021** | **+0.213** | **+0.240** | **+0.215** | **+0.173** | 14.9 | **+6.4** |
| B1 · our fill + 10 bps spread, our coefficients | −0.020 | −0.112 | +0.079 | +0.111 | +0.046 | +0.021 | 14.8 | +0.8 |
| B2 · our fill + 40 bps spread, our coefficients | −0.608 | −0.508 | −0.324 | −0.277 | −0.463 | −0.435 | 14.5 | −15.7 |
| **B3 · our fill + OUR banded cost contract** | −0.855 | −0.650 | −0.485 | −0.427 | −0.685 | **−0.619** | 14.4 | −22.1 |
| A3 · paper fill + our banded cost contract | −2.150 | −1.425 | −1.438 | −1.218 | −1.734 | −1.590 | 12.0 | −55.9 |

Cost coefficients are the Stage-A0 contract: entry 0.25 × half-spread for a next-bar-open fill and
1.00 × for a resting fill executed on arrival, stop 0.875 ×, close 0.412 ×. "Banded" = the median
NBBO spread of OUR 2025-26 mover population by price band × hour (`lit_review_2026/cost_curve.csv`).
Equal dollar risk per trade, so `total R` = net dollars per dollar risked: B0 **+3,573 R** over five
years (+715 / +85 / +896 / +994 / +883), B3 **−12,779 R**.

**Daily portfolio hit ratio** (share of the 1,258 sessions whose total R is positive) — the paper
publishes **48.4%**: A 31.2% · A0 33.4% · **B0 48.4%** · B1 43.1% · B2 28.7% · B3 24.2% · A3 8.4%.

## Verdict on the pre-registered criterion — SPLIT, and the split is the finding

**The simulator PASSES.** Under our own fill and exit machinery at zero cost the published book is
reproduced in sign and magnitude: **+0.173 R/trade against a published +0.08R**, positive in **5 of 5
years**, and a daily hit ratio of **48.4%** against the paper's published 48.4%. Nothing in our
entry/exit/stop/EOD code is manufacturing losses.

**The cost contract FAILS.** The same trades, same fills, charged our banded spread table, read
**−0.619 R/trade** — a 0.79R swing that turns a replicated published winner into a catastrophe. The
arithmetic is not subtle: **the median R on this book is 0.403% of price (40 bps)**, so one 40 bps
round trip is **0.99 R of cost** and the band table's 50–190 bps is **1.2 R to 4.7 R**. A spread
measured on $5–20 illiquid movers cannot be charged to KR, LEN, HLT and AAPL. This is PLAN §3 H7,
now demonstrated on a book nobody in this program designed.

**A third finding, not pre-registered, that matters as much.** The paper's own fill assumption fails
badly under our simulator: −0.284R, wrong sign. The entire A↔B gap is **whether the trigger bar's own
low can stop you out**. **34.5%** of paper-fill trades die inside their own trigger minute (ours:
21.3%), because R ≈ 40 bps is smaller than a first-hour 1-minute bar's range. The published +0.08R is
therefore only obtainable if the entry bar cannot stop the trade — i.e. the paper's number is a
sub-minute claim that 1-minute bars cannot settle. Measured (`fill_convention.md`):

| R as % of price | trades | A paper fill | B our fill | gap | A stopped in its own bar | 40 bps in R |
|---|---|---|---|---|---|---|
| < 0.25% | 4,444 | −0.248 | +0.209 | +0.457 | 31.7% | 2.04 |
| 0.25–0.5% | 8,447 | −0.228 | +0.175 | +0.403 | 35.2% | 1.14 |
| 0.5–1% | 5,920 | −0.243 | +0.183 | +0.426 | 36.4% | 0.61 |
| 1–2% | 1,551 | −0.214 | −0.035 | +0.178 | 34.1% | 0.32 |
| > 2% | 289 | +0.419 | +0.476 | +0.058 | 17.6% | 0.15 |

## Supporting tables

**R/trade by relative volume** (the paper's Figure 4 shape: −0.02R below 1×, +0.08R above, +0.38R
above 30×). Ours is hump-shaped, not monotone, and the > 30× bucket is +0.086R, not +0.38R:

| RV bucket | trades | A paper fill + comm | B0 our fill, zero cost | B1 our fill + 10 bps |
|---|---|---|---|---|
| 1–2× | 35 | −0.275 | −0.481 | −0.672 |
| 2–3× | 120 | −0.038 | +0.034 | −0.142 |
| 3–5× | 2,661 | −0.002 | +0.156 | −0.013 |
| 5–10× | 10,736 | −0.166 | +0.216 | +0.064 |
| 10–30× | 5,863 | −0.517 | +0.127 | −0.019 |
| > 30× | 1,244 | −0.835 | +0.086 | −0.053 |

**Tail dependence** (PLAN §1 item 5). The book IS its right tail — the paper says so itself ("most
trades lose ~1R and a few make 5–15R"), and we reproduce that shape:

| variant | A | B0 | B1 |
|---|---|---|---|
| all | −0.284 | +0.173 | +0.021 |
| ex top 1% | −0.521 | −0.075 | −0.227 |
| ex top 5% | −0.954 | −0.549 | −0.704 |
| winners capped at +5R | −0.667 | −0.309 | −0.453 |
| winners capped at +10R | −0.452 | −0.040 | −0.189 |

**Side split** (B0): long +0.156R on 10,274 fills, short +0.190R on 10,385. Both sides carry it, as
published. Borrow cost and shortability are not modelled — the same omission as the paper.

## Approximations, each one a place this could be wrong

1. **Venue, not listing.** XNAS.ITCH is the Nasdaq exchange feed. It carries NYSE-listed names traded
   on Nasdaq (LEN, KR, JPM, XOM all appear with correct prices), so the caveat PREREG expected
   ("Nasdaq-listed only") is wrong in our favour — but the **volume** is a venue subset. Measured on
   2024-09 against the consolidated EQUS.SUMMARY tape (`venue_share.md`): median XNAS share **0.118**
   on names with ≥ 1M consolidated shares (p25 0.084, p75 0.238 — Nasdaq-listed names sit near 0.24,
   NYSE-listed near 0.09). Relative volume is a ratio of same-venue volumes, so the share cancels;
   the **ADV filter had to be scaled** and that is the load-bearing approximation: at 1M × 0.118 =
   118,424 XNAS shares the pool is ~1,538 names/day, and re-ranking at 200K / 300K keeps ~20 picks a
   day but only **60.4% / 37.3%** of the same picks (`rank_sensitivity.md`). A tighter threshold
   drops small high-RV names, so the pick set — not the count — is threshold-sensitive, and our
   universe tilts Nasdaq-listed relative to the paper's.
2. **Period.** 2019-01-02 → 2023-12-29, five of the paper's eight years; XNAS.ITCH begins 2018-05.
3. **Bars are venue bars.** The 5-min candle and the intraday tape are Nasdaq-venue OHLCV, so both
   the breakout level and the stop touch marginally less often than on the consolidated tape.
4. **Price scale.** XNAS close vs consolidated close, 188,793 symbol-days: median ratio 1.00000.
   Within the traded set, 158 of 25,135 picks (0.6%) show a > 50% jump from the daily-file prior
   close to the 09:30 open (splits); they read −1.690R under arm A vs −0.275R for the rest and
   −0.189R vs +0.175R under B0. Excluding them changes no conclusion.
5. **ATR basis.** XNAS `ohlcv-1d` spans the feed day (04:00–20:00), so its range is **5.2% wider**
   than the 09:30–15:59 range on the traded names (`atr_scale.md`); R is therefore ~5% too large,
   a small conservative bias. `sim.py --atr-scale` exists for the correction; the headline uses 1.0.
6. **Survivorship: none.** ALL_SYMBOLS pulls carry delisted names (18,575 symbols over the window).
7. **Doji rows** (385) spend a top-20 slot with no order, as in the paper.
8. **Costs are modelled, never measured, on THIS population** — there is no 2019-2023 NBBO in this
   stage. B1/B2 are flat-spread sensitivities, not measurements.

## What it implies for the summer nulls

1. **The fill/exit machinery is exonerated.** It reproduces a published positive book, in sign, in
   magnitude, and in daily hit ratio, at zero cost. "Our simulator loses money on everything" is not
   what the summer nulls are.
2. **The spread constant is the null.** On a book whose R is 40 bps the banded table charges up to
   4.7R; on the summer population (R ≈ 1–3% of price) it charges **0.2–0.5R**, still the same order
   as every effect those stages were trying to detect, and A0 already showed the corrected contract is
   worth +0.412R per trade across the 52 cells. **The band table must not be applied to liquid names
   at all, and on any population it is a placeholder until Stage B measures per-trade NBBO.** The next
   honest step for `bf_zero2`/`score4` is not another entry family; it is the per-trade spread.
3. **The 1-minute entry-bar convention is a real, quantified lever** — worth +0.40R here and
   **+0.06 to +0.18R at the 1–3% stop widths our own books use**, which is larger than several of the
   effects the summer stages called noise. Our convention (fill at the next bar's open, exits checked
   from the fill bar inclusive) is the FAVOURABLE one, so the nulls are not an artefact of a
   pessimistic fill — but any future book with a sub-1% stop cannot be scored on 1-minute bars at all.
4. **A published, peer-reviewed, 8-year intraday edge is worth ≈ +0.17R gross and ≈ 0.00R at 10 bps**
   of quoted spread. That is the honest scale of this whole problem class, and it is the reason the
   program's "+0.1R at best" honest edges have never survived costs.

Cells: 1 pre-registered (the replication), reported under 7 fill × cost arms plus diagnostics. No
parameter was fitted, no threshold searched. Data spend $37.13.

## Files

`fetch_daily.py` · `fix_symbol_map.py` · `calibrate_venue_share.py` → `venue_share.md` ·
`build_pool.py` · `price_1m.py` · `fetch_1m.py` · `rank_top20.py` → `top20.csv.gz`,
`rank_sensitivity.md` · `atr_scale_check.py` → `atr_scale.md` · `sim.py` → `trades.csv.gz`,
`tables.md`, `year_tables.csv`, `fill_convention.md` · `indep_sim.py` → `indep_check.md`.
`xnas_daily.parquet`, `pool.parquet`, `tape.db`, `raw_daily/` are gitignored (bulk data, regenerable
from the scripts).
