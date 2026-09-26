# RESULT — cells 1,550–1,551: the overnight new-high leg, extended (PREREG_1550)

Builder run, independent rebuild not yet done. Spec: `research/overnight_high/PREREG_1550.md`.
Code: `fetch_alpaca_daily.py` (step 1), `build_panel.py` (step 2), `cell_1550.py` (step 3),
`test_cell_1550.py` (13/13 pass, step 4). Raw stats: `cell_1550_stats.csv`; one row per name-night:
`cell_1550_nights.csv` (35,592 rows).

## VERDICT: FAIL — every quantitative pass-bar criterion fails except the count-matched null.
Per the PREREG's own instruction ("FAIL → the overnight family is closed with the extension
numbers on record"), this leg is **not a live candidate**. The point estimates are frequently
positive but never survive a day-clustered t-test, the edge is entirely tail-carried (ex-top-5%
goes strongly negative in every cell/period), and only 2 of the 5 live calendar years in the
EXTENSION sample are net-positive. This replicates and sharpens the owner's own read ("regime, not
edge... not it").

## 1. Data build
* **Universe**: `alpaca_assets_all_20260905.csv` (33,211 real symbols after dropping 397 CUSIP/CVR/
  ESC/CNT-suffix non-ticker placeholders, all status=inactive) UNION every symbol carrying a
  definition record in any of the 27 PIT-feed months (16,353 symbols) = 36,288 union, test tickers
  (`^Z[A-Z]ZZT$`, `^ZZ`) already excluded.
* **Fetch**: Alpaca daily bars, 2019-01-02..2024-06-28, batches of 10, `feed=SIP, adjustment=raw`.
  **Completeness gate PASSED: LOST = 0 (0.00%)** of 36,288 requested. 21,635 symbols returned zero
  bars in the window after an individual solo-retry confirmed "no data" (expected: post-2024 IPOs,
  OTC shells with no 2019-2024 history, etc.) — these are excluded from the LOST count by
  construction (a solo retry that raises "invalid symbol" or repeats empty is not a batch-drop).
  13,775 symbols returned real bars (12,904,836 rows). Full log: `fetch_run.log`;
  `_fetch_state/lost_symbols.csv` lists every excluded/invalid symbol with its reason.
* **PANEL rebuild**: `equs_daily_2024H2.parquet` + `equs_daily_2025_2026.parquet` concatenated
  (3,888 overlapping (symbol,bar_date) rows resolved keep-last, i.e. the 2025_2026 file wins) →
  6,464,692 rows, 16,263 symbols, 2024-07-01..2026-09-04.

## 2. Convention (stated per the PREREG's explicit requirement)
* `adv20` / `dvol20` = mean(volume) / mean(close×volume) over the **prior** 20 sessions
  (`shift(1).rolling(20, min_periods=10)`) — matches the existing `build_daily_panel.py` convention.
* `vol_ratio = volume_t / adv20_t` (today's volume against yesterday-and-before ADV).
* **`high252` = MAX of the prior 252 sessions' CLOSE** (`shift(1).rolling(252, min_periods=252)`)
  — close, not high; 252 sessions, not 250; **strictly prior**, and **min_periods set to the full
  252** (not the looser 60 used for `high52` in the legacy `build_daily_panel.py`) because the
  PREREG's own refuter re-checks this exact field for look-ahead and a "252-day high" computed off
  a 40-day warm-up is not one.
* `ret_on_next = next_open / close − 1`, raw close and raw next open (`adjustment=raw` on the
  Alpaca fetch; Databento EQUS.SUMMARY is unadjusted by construction).

**Consequence disclosed up front**: the strict min_periods=252 choice, combined with the PANEL
parquets starting fresh at 2024-07-01 (no earlier history in the two purchased files), means
`high252` is null for every row until ~253 trading sessions in — i.e. **PANEL TRAIN's usable signal
only starts in 2025-07**, not 2024-07. The disclosed prior run (`overnight_auction.md`) used the
legacy `high52` field (HIGH, 250d, min_periods=60), which had signal from ~Oct 2024 — that is why
its "TRAIN 25 names/day" run had ~350+ signal-days where this rebuild's PANEL TRAIN period has 126.
Likewise EXTENSION's own 2019 is void (high252 needs 252 prior EXTENSION sessions too), leaving 5
live years (2020–2024H1), not 5.5. This is a direct, disclosed effect of taking the PREREG's rule
text literally rather than re-using the legacy field — not a bug, but it means PANEL TRAIN/VAL here
is really "2025H2 + 2026 through May" and should be read as such.

## 3. Universe gate (both samples)
close ≥ $5, 20-day dollar volume ≥ $10M, `|ret_on_next| ≤ 0.5` (already excludes any night beyond
±50%; a further ±30% flag is reported per PREREG's refuter list), no test tickers. Signal: `close_t
≥ high252_t` and `vol_ratio_t ≥ 1.5`; ranked by `vol_ratio` descending; top N ∈ {10, 25}.

## 4. Results — primary read (net @ 5 bps unless noted; bps/night)

| sample | period | cell(N) | n_nights | names/day | gross | net@5bp | t(dayclust) | ex-top5% | ex-top1% | capped+5% | universe(placebo) | placebo margin | placebo t | null pctile | ±30% nights |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EXTENSION | FULL (2019–2024H1) | 1550 (10) | 1,109 | 8.44 | +9.75 | +4.75 | **1.29** | **−47.77** | −18.53 | −21.49 | +4.09 | +5.66 | 1.29 | 100.0 | 25 |
| EXTENSION | FULL | 1551 (25) | 1,109 | 15.38 | +10.94 | +5.94 | **1.19** | **−37.07** | −13.36 | −12.53 | +4.09 | +6.85 | 1.28 | 100.0 | 27 |
| EXTENSION | 2020 | 1550 | 247 | 8.89 | +51.43 | +46.43 | 3.18 | −28.17 | +19.58 | −4.21 | +8.79 | +42.64 | 2.59 | 100.0 | 7 |
| EXTENSION | 2020 | 1551 | 247 | 17.46 | +38.06 | +33.06 | 2.96 | −25.67 | +10.80 | +0.64 | +8.79 | +29.27 | 2.36 | 100.0 | 7 |
| EXTENSION | 2021 | 1550 | 252 | 9.42 | −3.11 | −8.11 | −0.59 | −69.32 | −36.10 | −41.87 | +9.42 | −12.53 | −0.97 | 0.0 | 10 |
| EXTENSION | 2021 | 1551 | 252 | 19.42 | +6.38 | +1.38 | −0.09 | −45.68 | −20.59 | −20.44 | +9.42 | −3.04 | −0.60 | 5.1 | 11 |
| EXTENSION | 2022 | 1550 | 239 | 6.62 | −13.07 | −18.07 | −0.40 | −51.80 | −33.93 | −29.65 | −3.16 | −9.92 | 0.02 | 0.0 | 3 |
| EXTENSION | 2022 | 1551 | 239 | 8.87 | −13.79 | −18.79 | −0.50 | −48.37 | −32.33 | −27.43 | −3.16 | −10.63 | −0.08 | 0.0 | 3 |
| EXTENSION | 2023 | 1550 | 249 | 8.22 | −1.68 | −6.68 | −0.09 | −34.78 | −19.30 | −14.74 | +1.62 | −3.30 | 0.33 | 11.8 | 1 |
| EXTENSION | 2023 | 1551 | 249 | 13.31 | −4.03 | −9.03 | −0.36 | −33.61 | −20.16 | −15.10 | +1.62 | −5.66 | 0.09 | 0.0 | 2 |
| EXTENSION | 2024H1 | 1550 | 122 | 9.54 | +8.48 | +3.48 | 0.27 | −35.65 | −16.59 | −13.30 | +2.87 | +5.61 | 0.53 | 92.3 | 4 |
| EXTENSION | 2024H1 | 1551 | 122 | 19.81 | +14.00 | +9.00 | 0.78 | −25.10 | −6.96 | −3.42 | +2.87 | +11.13 | 1.40 | 100.0 | 4 |
| PANEL | TRAIN (2025H2-eff.) | 1550 | 126 | 9.75 | +14.85 | +9.85 | 0.59 | −44.42 | −10.61 | −17.50 | +9.22 | +5.62 | 0.33 | 88.4 | 3 |
| PANEL | TRAIN | 1551 | 126 | 21.90 | +19.58 | +14.58 | 0.96 | −29.75 | −2.96 | −4.11 | +9.22 | +10.35 | 0.69 | 99.9 | 3 |
| PANEL | VAL (2026 Jan–May) | 1550 | 102 | 9.83 | +18.66 | +13.66 | 0.71 | −48.03 | −9.03 | −20.96 | +0.92 | +17.74 | 0.91 | 99.8 | 4 |
| PANEL | VAL | 1551 | 102 | 22.25 | +13.26 | +8.26 | 0.29 | −44.33 | −13.40 | −17.36 | +0.92 | +12.34 | 0.58 | 99.9 | 7 |
| **PANEL** | **TRAIN+VAL pooled** | **1550** | **228** | 9.79 | +16.56 | **+11.56** | **0.92** | −46.48 | — | — | — | +11.21 | 0.88 | 99.6 | — |
| **PANEL** | **TRAIN+VAL pooled** | **1551** | **228** | 22.07 | +16.72 | **+11.72** | **0.91** | −36.33 | — | — | — | +11.37 | 0.90 | 100.0 | — |
| PANEL | **TEST (spent)** | 1550 | 67 | 9.64 | +2.99 | −2.01 | −0.05 | −55.60 | −30.29 | −31.05 | +7.83 | −4.83 | −0.21 | 28.3 | 6 |
| PANEL | **TEST (spent)** | 1551 | 67 | 18.87 | −4.23 | −9.23 | −0.44 | −47.42 | −29.86 | −25.37 | +7.83 | −12.05 | −0.77 | 1.3 | 6 |

Full per-year and per-N detail (24 rows) in `cell_1550_stats.csv`; every column in the PREREG's
list is a column there (gross/net at 2/5/10bp, both ex-top cuts, capped+5%, weekly P10/green-share/
strong-week-gap, placebo, null, names/day).

## 5. Pass-bar checklist (frozen bar from PREREG_1550.md)

**EXTENSION 2019–2024H1** (bar: net ≥ +8bp @5bp AND t≥2.5 AND ex-top5%>0 AND ≥4/5.5 yrs positive
AND placebo margin ≥+5bp with t≥2 AND null pctile ≥99):

| criterion | cell 1550 | cell 1551 | pass? |
|---|---|---|---|
| net@5bp ≥ +8bp | +4.75 | +5.94 | **FAIL** (both) |
| t(dayclust) ≥ 2.5 | 1.29 | 1.19 | **FAIL** (both) |
| ex-top-5% > 0 | −47.77 | −37.07 | **FAIL** (both, badly) |
| ≥4 of 5 live years net-positive | 2/5 (2020,2024H1) | 2/5 (2020,2024H1) | **FAIL** (both) |
| placebo margin ≥+5bp, t≥2 | margin +5.66 OK, t 1.29 | margin +6.85 OK, t 1.28 | **FAIL** (t, both) |
| null percentile ≥99 | 100.0 | 100.0 | PASS (both) |

**1 of 6 criteria passes. EXTENSION fails the pass bar decisively for both N=10 and N=25.**

**PANEL TRAIN+VAL pooled** (bar: net ≥ +8bp, t≥2): net +11.56/+11.72bp clears the point estimate,
**t = 0.92 / 0.91 fails the significance bar by more than half**.

**Combined rule ("AND")**: since EXTENSION already fails, the PANEL read is moot for the pass/fail
call, but is reported in full per the PREREG's own instruction that both are always read.

## 6. Tail dependence (why this is a lottery ticket, not an edge)
In every sample/period/cell, **ex-top-5% of nights is deeply negative** (−28 to −70 bps) while
gross is 0 to +51 bps: the entire positive point estimate is carried by roughly one in twenty
nights. Winner-capping at +5% (`capped5_bps`) also flips most cells negative or near-zero. This is
the exact failure mode CLAUDE.md's tail-dependence check exists to catch — "edge that dies under a
cap is a lottery ticket." Combined with 25–27 nights per EXTENSION cell (~2.3%) beyond ±30% in a
single overnight session, the mechanism is a handful of explosive gap nights (biotech/small-cap
news, some plausibly reverse-split artifacts — see caveats) rather than a repeatable premium.

## 7. Placebo and null
Placebo = the whole eligible (price/dollar-volume/no-test-ticker) universe's own MOC→MOO return on
the *same nights* the book traded. The rule beats the placebo by +5 to +18 bps in most periods
(margin mostly positive) but the day-clustered t on that margin never reaches 2 except in 2020
(t 2.59/2.36) — i.e. the "rule vs. universe on the same tape" difference is not reliably real
outside one volatile year. The count-matched null (1,000 draws of N random eligible names/night,
seed 1550) is beaten on point estimate almost everywhere (percentile 88–100), which is consistent
with the rule's picks simply being higher-volatility, higher-beta names than a random draw of the
same liquid universe — exactly the kind of thing a volume-shock/new-high filter selects for by
construction, and not on its own evidence of an executable edge once cost and clustering are
accounted for.

## 8. The 15:49-order variant — NOT COMPUTABLE with this run's inputs
The PREREG asks for a variant where the signal is computed from the **15:45 price and
volume-to-date scaled by 390/375** (an MOC-imbalance-safe order timing check). This run fetched
**daily** bars only (per the task's step-1 instruction); no intraday/minute bars were pulled for
either sample. Approximating the 15:45 price from the daily close would be fabricating a number
that was never observed, which CLAUDE.md's "no research claim ships" rule forbids — this is
reported as **VOID (data not fetched)**, not skipped silently. Building it for real needs minute
bars for the ~1,109/126/102 signal-night symbol-days above only (a small, targeted fetch, not a
repeat of the full universe pull) — flagged as the one clean follow-up if the owner still wants the
15:49 variant despite the FAIL verdict above.

## 9. Caveats (read as an adversary)
1. **Survivorship, EXTENSION**: names delisted *before* 2024-07 are missing from the union universe
   (the PIT feed only covers 2024-07 onward); for a long-only rule on ≥$10M-ADV names this biases
   the EXTENSION numbers **upward**, not downward — the FAIL verdict is if anything conservative.
2. **PANEL TRAIN truncation**: see §2 — PANEL's "TRAIN" period only carries usable signal from
   ~2025-07 onward, not 2024-07, because of the panel's short pre-history and the strict 252-session
   convention. The pooled TRAIN+VAL numbers above are the honest read of what's computable from the
   two purchased Databento files, not a like-for-like match to the disclosed prior run's PANEL TRAIN.
3. **Raw (unadjusted) prices**: both fetches used raw/unadjusted OHLC by design (avoids adjusted-vs-
   intraday mismatches), which means a reverse split landing between the MOC and the next MOO would
   show up as a large fake return. The existing ±50% hard exclusion and the ±30% flag bound this,
   but 25–27 EXTENSION nights per cell are flagged and have **not** been individually checked against
   a corporate-actions calendar in this run — do that before trusting any single top-decile night.
4. **TEST is spent** (PANEL 2026-06 on, 67 nights): reported for completeness only, per the PREREG;
   it is negative-to-flat in both cells and changes nothing about the FAIL verdict.
5. **Cadence-bar ($3K/name) reporting**: `scripts/cadence_bar.py`'s weekly-P10 / strong-week-gap
   library functions were called directly (R := net fractional return, since position size = risk
   at $3K/name with no stop). At this book's return scale (tens of bps/night on $3K), the default
   `strong_r=5.0` (5R/week) threshold is essentially never reached — 0 strong weeks in every
   period — so the strong-week gap is **not computable** at this notional; weekly P10 and green-week
   share are in `cell_1550_stats.csv` (`weekly_p10`, `weekly_green_share` columns) but are reported,
   not gated, since the cadence bar's own thresholds were built for a stop-based day-trading book.
6. **Independent reimplementation and causality trace**: NOT yet done (this file is the BUILDER's
   output only, per the workflow's role split) — required before this result goes in front of the
   owner as a claim, per CLAUDE.md's "no research claim ships without an independent check."

## Judge (main session, 2026-09-26 20:20 UTC) — FAIL both cells; a lottery ticket, not a book

* EXTENSION 2019–2024H1 (the fresh out-of-sample): builder net +4.8 / +5.9 bps per night at 5 bps cost (t 1.3 / 1.2);
  rebuild +17 / +14 bps (t 2.4 / 1.9). The gap is the builder's outcome filter (|overnight return| ≤ 50 %, an
  undisclosed look-ahead that removes the squeeze nights GME +113 %, VIRX +215 %, CODX, SPRT) and 228K zero-OHLCV
  placeholder rows in the Databento panel (cash-merger nights booked as −100 %). Under every correction the verdict is
  the same: ex-top-5 % −29 to −47 bps, winner-capped at +5 % negative, 2–3 of 5 live years positive (2019 is warm-up),
  placebo margin t < 2, day-clustered t < 2.5. The edge is a handful of squeeze nights; it fails the tail rule.
* PANEL 2025H2–2026H1: +12 / +12 bps (builder) to +23–33 (rebuild) with t 0.9–1.7; TEST (spent) −2 / −9 bps.
* Spec note: the disclosed 9/26 code used the 250-day HIGH with a 60-day warm-up; this PREREG's prose said 252 closes —
  a divergence I introduced when writing the PREREG; both definitions were run by the two sides and both fail.
* Builder vs rebuild: nightly membership Jaccard 0.99; returns bit-identical on shared keys; the differing rows are the
  extreme nights above. Renamed tickers double-counted in 4.2 % of extension fills (EMBJ/ERJ, FB/META, NBIS/YNDX).
Consequence per PREREG: the overnight new-high family is closed with the extension numbers on record. Programme count
1,551. Data kept: alpaca_daily_2019_2024H1.parquet (36,288 symbols, 0 lost) for any later daily-horizon study.
