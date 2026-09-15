# Bull-flag from zero — REPORT (2026-09-13/14)

Owner (9/13): "BF is profitable only because of the 6312, so this is BS! so few trades is also BS!
start from scratch, read the literature, remove the filters, the quartiles, everything, re-build in
the BT to generate 5+ trades per week and 1:2 R:R" → "start from scratch, no gaps, nothing, clean
sheet" → "you continue till you have a strategy at hand fully ready".

Pre-registration: `DESIGN.md` (written before any scan). Pipeline: `build_candidates.py` →
`pass2.py` → `score.py`; then `sensitivity.py`, `book_sim.py`, `spec_sim.py`. Tables:
`score_tables.md`, `step1_*.csv`, `results.csv`, `sensitivity_rows.csv`, `hodbreak_book_cap60.csv`,
`spec_book.csv`.

## 0. Verdict

- **Rule as pre-registered: nothing passes** (0 of 84 family × exit combinations). Part of that is a
  defect in my rule: "worst week ≥ −10R" was written for a 5-a-week book and is meaningless at 300+
  trades a week. Reported as written, not rewritten.
- **A look-ahead in my own universe rule was found and removed before any number was reported**
  (§2). It inflated the high-of-day-break family from +0.085R to +0.38R.
- **One book survives every honest test that could be run tonight**: the **HOD-break** —
  a stock ≥ 5% above its 09:30 open, relative volume 1–5× its normal pace, ≥ 5 one-minute bars
  within 4% of the high-of-day, then a break of the HOD. Capped limit entry (60 bps, no chase), stop
  at the consolidation low, fixed +2R target, first-come 8/day, 4 concurrent, flat 15:55.
  Study book (§5): TRAIN +0.09R / VAL +0.15R / TEST +0.16R at 33–35 trades a week. **Parity
  simulation with the exact live fill rule (§6): TRAIN +0.21R / VAL +0.21R / TEST +0.23R at 35–37 a
  week, TEST 12/15 weeks green, worst week −2.9R; with the $5 price floor adopted for live:
  +0.30 / +0.27 / +0.33R at 19–27 a week, TEST 13/14 weeks green, worst −3.1R.**
- **Built tonight, one spec for BT and live** (`trading/hod_break.py`, `trading/hod_break_engine.py`,
  wired, 3,667 tests green, boot rehearsal on the real account, service restarted). **Shipped
  enabled + dry_run for Mon 9/14** (zero orders). Go-live only after a POSITIVE dry-run day and the
  owner's six gates (CLAUDE.md "Strategy 4"). Unbiased-sample confirmation running (§7).

## 1. What was built

- Universe: whole market, point-in-time (Databento EQUS daily incl. delisted): range ≥ 5%, price
  ≥ $1, ADV20 ≥ 100K → **647,796 symbol-days**; bars for 289,575 symbol-days fetched (~$116). No
  gap rule, no mover threshold, no scanner list, no price cap.
- 8 entry families × 21 configs × 4 exits (E1 fixed +2R/−1R with trade-through targets; E2 partial
  50%@2R + trail; E3 +3R; E4 hold) → **2,780,556 candidate rows**. Every old filter came back only as
  a hypothesis split. Same-clock volume checkpoints written for 559K symbol-days (the owner's
  "own volume DB" — `volume_profile.csv`; live: cache.db `hod_volume_profile`, nightly).
- Splits fixed: TRAIN 2025 / VAL Jan–May 26 / TEST Jun–Sep 26.

## 2. The look-ahead in the universe rule (caught 9/13 evening, before reporting)

"Day range ≥ 5%" is implied by a 5% pole (F1–F4) but NOT by a HOD break, a red-to-green cross, a
PM-high break or an opening-range break (F5–F8). For those, keeping only days that ended with a
5% range keeps days where the break was FOLLOWED by a move — the ignition error in a new coat.
Measured on F5 K5 X0.04, fixed 1:2 exit, by distance from the open at entry:

| entry vs open | n | mean R | WR |
|---|---|---|---|
| 0–2% (the rule doing the work) | 191,858 | +0.377 | 52% |
| 2–5% | 115,777 | +0.257 | 49% |
| 5–10% (causal) | 23,723 | +0.085 | 42% |
| 10–20% | 4,747 | +0.098 | 41% |

Fix: require the entry level ≥ 5% above the open (a causal condition that guarantees the day is in
the universe). Applied to F5–F8 before scoring; 2.03M of 2.36M F5–F8 rows dropped.

## 3. Every family, fixed 1:2 exit, causal floor (mean R per trade)

| family | TRAIN | VAL | TEST | trades/wk | reading |
|---|---|---|---|---|---|
| F1 bull flag P≥5% | +0.048 | −0.026 | +0.019 | 880–1,090 | no edge |
| F2 micro pullback | +0.020 | −0.049 | +0.022 | 960–1,180 | no edge |
| F3 opening-drive pullback | −0.003 | +0.034 | +0.051 | 520–790 | flat |
| F4 VWAP bounce | −0.117 | −0.064 | −0.120 | 530–640 | negative |
| **F5 HOD break K5 X4%** | **+0.108** | **+0.087** | **+0.050** | 280–440 | least-bad, positive all splits |
| F5 K10 X4% | +0.113 | +0.080 | +0.010 | 360–580 | fades in TEST |
| F6 red-to-green | +0.013 | +0.101 | +0.009 | 120–160 | flat |
| F8 OR-30 break | +0.071 | +0.058 | +0.012 | 560–820 | fades in TEST |

Hold-to-close (E4) shows larger means with 15–23% win rates — tail-only, the shape the owner
rejected. Partial-and-trail (E2) sits between. The Cameron flag, on the whole market with no
filters, has no edge at any exit: the owner's instinct about the old book was right for the
wrong reason — it was never the detector, it was selection plus one trade.

## 4. Sensitivities (population F5 K5 X4%, causal floor; `sensitivity.py`)

| fill model | TRAIN | VAL | TEST | fill rate |
|---|---|---|---|---|
| base (slip 0.3%, target trade-through) | +0.108 | +0.087 | +0.050 | 100% |
| 60 bps cap, no chase | +0.103 | +0.081 | +0.011 | 62% |
| target only on bar CLOSE (no wick fills) | +0.094 | +0.081 | +0.038 | 100% |
| both | +0.085 | +0.066 | +0.002 | 62% |

The population edge is fragile in TEST under harsh fills. The book (next section) is not, because
the relative-volume band and first-come selection concentrate it.

## 5. The executable HOD-break book (`book_sim.py`; 60 bps cap at the break bar, close-fill
## target, first-come 8/day, 4 concurrent, rv 1–5×, stop ≥ 1% of price)

| split | trades | /wk | mean R | WR | PF | weekly R mean / sd | weeks green | worst week |
|---|---|---|---|---|---|---|---|---|
| TRAIN 2025 | 1,748 | 33 | +0.091 | 41% | 1.15 | +3.0 / 8.7 | 31/53 | −12.5 |
| VAL Jan–May 26 | 767 | 35 | +0.146 | 44% | 1.25 | +5.1 / 8.7 | 17/22 | −15.4 |
| TEST Jun–Sep 26 | 491 | 35 | +0.162 | 44% | 1.28 | +5.7 / 9.6 | 11/14 | −8.3 |

TEST week by week (R): +12.4, +6.8, −5.0, +4.6, +9.5, +5.4, −8.3, +7.4, +2.5, +32.6, +4.2, −3.2,
+6.9, +3.8. Exits: 51% stop, 31% target, 15% flat at the close.

- **Relative volume was the selection, and 2025 alone picks the same band**: TRAIN-only mean R by
  rv 0–0.5×: +0.045 · 0.5–1: +0.102 · **1–2: +0.138 · 2–5: +0.131** · >5: +0.055. The band was first
  seen pooled (disclosed); this TRAIN-only read is the honest confirmation.
- **Stop distance**: TRAIN 1–2% of price +0.163, 2–4% +0.096, >4% +0.002 — the spec keeps ≥ 1%.
- **Price**: $1–5 +0.06, $5–10 +0.09, **$10–20 +0.18, $20–50 +0.18, >$50 +0.41 (n=144)**. The owner's
  point stands: the $20 cap was mine and it was wrong for this book. No cap in the spec.
- Wrappers +0.19 (n=613) vs common +0.10 — kept, not selected on.
- Random-8 control: +0.15 vs first-come +0.16–0.28 in VAL/TEST — first-come adds a little (early
  entries score better); the filter does the real work.

Money at this shape: ~5.7R/week on TEST at 35 trades. At $100 risk ≈ $2.3K/month; at $300 ≈ $7K
with ~$2.9K weekly noise. Not a forecast — a relative tool with a live tape still at zero.

## 6. Parity simulation (`spec_sim.py`, the exact live spec: fill at the NEXT bar's open if ≤ cap) — DONE 21:10 UTC

`trading.hod_break.simulate` run over 189,830 symbol-days (every day with a break and a high ≥ 5% above the
open); 38,953 spec signals. THE numbers for the live build:

| book | split | trades | /wk | mean R | WR | PF | weekly R mean / sd | weeks green | worst week |
|---|---|---|---|---|---|---|---|---|---|
| population (no caps) | TRAIN | 20,872 | 394 | +0.231 | 46% | 1.42 | +91 / 149 | 45/53 | −37 |
| population | VAL | 11,085 | 482 | +0.228 | 46% | 1.41 | +110 / 89 | 20/23 | −47 |
| population | TEST | 6,996 | 466 | +0.214 | 45% | 1.38 | +100 / 128 | 13/15 | −23 |
| **8/day, 4 concurrent** | TRAIN | 1,949 | 37 | +0.212 | 44% | 1.37 | +7.8 / 10.0 | 36/53 | −14.3 |
| **8/day, 4 concurrent** | VAL | 813 | 35 | +0.206 | 45% | 1.35 | +7.3 / 10.6 | 17/23 | −12.9 |
| **8/day, 4 concurrent** | TEST | 538 | 36 | +0.230 | 47% | 1.42 | +8.2 / 9.7 | 12/15 | −2.9 |
| 5/day, 3 concurrent | TEST | 339 | 23 | +0.195 | 46% | 1.35 | +4.4 / 6.9 | 11/15 | −6.0 |
| 12/day, 4 concurrent | TEST | 778 | 52 | +0.192 | 46% | 1.34 | +9.9 / 9.1 | 12/15 | −5.1 |

TEST week by week (8/4 book, R): +2.5, +11.9, +10.7, +4.9, +0.1, +5.4, +5.5, −2.8, +17.5, +29.3, −2.9, +3.4,
+24.0, +14.0 (the 15th week has no data). Stronger and steadier than the study's book (§5) because the spec
takes the first QUALIFYING break (a later break can qualify when the first fails the filters) and fills at
the next open instead of paying the cap.

Scrutiny: half the fills land below the break level (median entry = the level, IQR −42 to +21 bps) and the
edge is NOT on one side (below/above: TRAIN +0.30/+0.13, VAL +0.13/+0.29, TEST +0.20/+0.25). Entry hour is
flat (+0.21 to +0.23 from 9:30 to 14:00; the 16 trades after 14:00 are −0.21 → `last_entry_minute` 930 is
generous, 840 would be tighter). Wrappers +0.13 vs common +0.24.

**Price (cost rule adopted for live, `min_price: 5`)**: $1–2 +0.12 · $2–5 +0.08 · $5–10 +0.27 · $10–20
+0.16 · $20–50 +0.38 · >$50 +0.57. The sim cannot see spreads; under $5 the thin edge would not survive
them. Book ≥ $5: **TRAIN +0.303 (19/wk, 39/53) / VAL +0.266 (23/wk, 19/22) / TEST +0.331 (27/wk, 13/14,
worst −3.1R)**, WR 48–50%. Chosen after seeing all splits — disclosed; it is a tradability rule, and the live
100 bps spread gate would have removed most of the same names causally.

What the sim still assumes optimistically: a fill AT the next bar's open (a marketable limit fills at the
ask, half a spread worse); stops at 10 bps through. Both are smallest above $5.

## 7. Unbiased sample (random 10% of tradable symbols, ALL their days, no range gate)

Done 23:04 UTC (`sample_analysis.py`, `sample_analysis.log`). 959 symbols (10% of the 9,595 tradable ones), ALL their
days: 346,721 symbol-days requested, bars for 66% (the missing third is vendor-absent or renamed tickers — reported,
not hidden), 666,999 candidate rows. HOD-break F5 K5 X4%, fixed 1:2 exit:

| population | TRAIN | VAL | TEST | trades/wk (10% sample) |
|---|---|---|---|---|
| A. raw, no floor — includes days the old universe never saw | +0.039 | +0.044 | **−0.014** | 830–960 |
| B. causal floor: entry ≥ 5% above the open | +0.211 | +0.140 | +0.049 | 45–57 |
| C. floor + rv 1–5× + stop ≥ 1% (the live filters) | **+0.296** | **+0.299** | **+0.255** | 15–20 |
| C5. C + price ≥ $5 (the live cost rule) | +0.229 | +0.301 | **+0.316** (11/14 wks) | 6–14 |

By distance from the open on this unbiased population: 0–2% +0.008 · 2–5% +0.068 · 5–10% +0.160 · 10–20% +0.162 ·
20%+ +0.214. The days with a range under 5% (never in the main universe) hold 35,022 F5 rows at **−0.316R** — the
exact population the biased universe rule had been excluding, and the causal floor excludes causally (6 rows survive
the floor on those days, as arithmetic requires). Verdict: the main study's population numbers are reproduced on an
unbiased sample; the relative-volume band is where the edge lives; the $5 floor holds. Purpose: confirm
F5 on the full population without the 5% day-range universe rule (the causal floor already makes
the population superset-exact; this is the independent check).

## 8. What shipped tonight (all DISABLED + dry_run; zero orders)

- `trading/hod_break.py` — ONE spec (detect, capped next-open fill, close-fill target, flat 15:55,
  sizing). 17 tests.
- `trading/hod_break_engine.py` — live engine: scanner mover hook on the TRUE 09:30 open, bar
  stream handler id 'hod_break', capped limit BUY with a broker bracket (stop = consolidation low,
  target +2R) — the legs ARE the exits, polled by the engine; no-chase (ask above cap → skip);
  spread gate; DB-derived per-day cap; concurrency cap; kill rails (fail closed); 75s unfilled →
  cancel; 15:55 flat; restart sync from the trades DB. 22 tests. Parity tests lock the constants.
- Wiring: `main.py --hod` (deployed unit updated, owner-approved sudo), scanner tick that outlives
  the 15:45 latch, mover hook, bar drains. `config.yaml hod_break` block (+ template).
- Nightly `scripts/build_hod_volume_profile.py` → cache.db `hod_volume_profile` (crontab 23:00 UTC).

## 9. Known limits, stated

- The live fill (limit at the cap right after the break bar closes) and the parity sim's fill (next
  open if ≤ cap) are the same rule; the study's book sim (§5) used "break-bar high reached the cap"
  which is slightly more optimistic — §6 is the number that counts.
- Bracket TP/SL are set from the LIMIT price at submission; a fill below the limit makes the real
  target slightly under 2R and the stop slightly under 1R. Conservative for the target, harmless.
- Stops fill at the broker as market orders; modeled at 10 bps through or the gap-through open.
- `StopMonitor` is not used for exits (its symbol-keyed watch rejects shared symbols); the shared
  bar stream is. A symbol another book holds can still be entered — the cross-strategy conflict
  rule is an open question for the owner.
- 34 trades/week × $100 risk is the proposed start; the ramp on positive realized P&L (like BF P1)
  is the path, never the backtest.

## 8. Spread study (pre-registered 9/14, `spread_study.py` → `spread_study.md`; scoring `spread_score.py`)

6,847 of 7,000 sampled spec signals (price ≥ $5, 2,266 / 2,278 / 2,303 per split) with the real NBBO at the
signal minute (Alpaca SIP quotes). Spread as a fraction of R, quintiles cut on TRAIN.

**The spread is a cost, not a signal**: raw mean R is flat across spread quintiles (+0.17 to +0.45, no order).
Net of cost it is monotone in every split. Realistic cost = half a spread on entry (ask vs mid) + half a spread
on the way out when the exit is not the target:

| book | TRAIN | VAL | TEST | share of signals |
|---|---|---|---|---|
| no gate | +0.049 | −0.003 | +0.010 | 100% |
| spread ≤ 15% of R | **+0.283** | **+0.195** | **+0.260** | 42% |
| ≤ 15% and price ≥ $20 | +0.341 | +0.350 | +0.444 | 16% |

By price band, net, with the gate: $5–10 +0.28 / +0.17 / +0.13 · $10–20 +0.20 / +0.04 / +0.20 · $20–50
+0.34 / +0.33 / +0.38 · $50+ +0.35 / +0.40 / +0.59. Spread in bps is FLAT across price (~39 bps everywhere);
the price gradient is signal quality, not cost. The old $20 CAP was wrong in the strongest possible way: the
best trades are above it.

**Adopted**: `max_spread_frac_r: 0.15` (the pre-registered rule: excluded buckets negative on TRAIN and VAL,
TEST agrees). The $5 floor stays for now (the $5–20 bands are positive with the gate, VAL $10–20 marginal);
the dry week logs price per signal and the $20 question is re-read on that. Today's dry book with the gate:
7 trades +1.7R vs −4.2R ungated. Exit variants under real spreads (target +spread / both widened): §9.

### 8a. Price floor decided on the history (9/14 19:40 UTC, owner: "why score tomorrow and not use historical data")
The price split meets the same pre-registered rule as the spread gate → **`min_price: 20`** (config + template), live from
the 9/15 boot. Capacity/weekly check on the spec signals with price ≥ $20 and the break bar ≤ 14:00 (8,606 of 38,953),
the gate modeled as a cost filter INDEPENDENT of the outcome (each signal passes with p = 0.42, the study's pass rate;
a passing signal is charged a flat 0.08R, the study's median spread/R of passing signals; failing signals are not traded),
12/day, 4 concurrent, the shared book rule `trading.hod_break.run_book` (causal slot freeing), **20 random seeds —
mean [min..max]**. Producer: `research/bf_zero/capacity_8a.py` (the 9/14 table had no script; its "14/14 green, worst
+0.4" TEST row was one favourable draw — replaced 9/15 by the parity review).

| book | split | trades/wk | net R/trade | weekly net R | weeks green | worst week |
|---|---|---|---|---|---|---|
| $20, no gate (raw, reference) | TRAIN | 37.1 | +0.342 | +12.7 | 43/53 | −10.6 |
| $20, no gate (raw, reference) | VAL | 41.3 | +0.365 | +15.1 | 21/23 | −5.5 |
| $20, no gate (raw, reference) | TEST | 39.3 | +0.421 | +16.5 | 13/15 | −0.9 |
| **$20 + gate (42% pass, −0.08R)** | TRAIN | 22.2 [21.4..23.1] | +0.263 [+0.198..+0.330] | +5.8 [+4.4..+7.3] | 39.5/53 [36..44] | −11.0 [−15.9..−5.7] |
| **$20 + gate (42% pass, −0.08R)** | VAL | 31.7 [29.7..33.7] | +0.303 [+0.234..+0.359] | +9.6 [+7.5..+11.4] | 20.1/23 [18..21] | −4.2 [−9.8..−0.2] |
| **$20 + gate (42% pass, −0.08R)** | TEST | 29.0 [28.0..30.5] | +0.360 [+0.252..+0.500] | +10.4 [+7.3..+14.2] | 12.8/15 [10..14] | −3.1 [−13.7..+5.1] |

Gated fills are 4.6–6.6 per day (22–32/wk), not 5–8. 4 concurrent is the binding constraint, not 12/day (seed-0 rejections:
concurrency 1,002 vs day-cap 298; without the concurrency cap the ungated weekly R rises +12.7→+14.1 / +15.1→+20.1 /
+16.5→+22.5). `run_book` causal (exit_m < entry_m) vs the old hindsight rule (exit_m > entry_m), gated seed means:
TRAIN 22.2/wk +0.263R +5.8R/wk vs 22.3 +0.262 +5.8; VAL 31.7 +0.303 +9.6 vs 31.9 +0.305 +9.7; TEST 29.0 +0.360 +10.4 vs
29.1 +0.366 +10.6 — immaterial at this book size. The EOD check scores the dry book at both floors daily as the plumbing
check, not the decision.

## 9. Exit variants under real spreads (`spread_exit_variants.py`, 1,908 quoted signals re-walked from bars; costs charged identically)

| book | V0 baseline (target +2R) | V1 target +2R+spread | V2 stop −spread, target +2(R+spread) |
|---|---|---|---|
| all ≥ $5, $ per $100 risk T/V/T | 11.9 / 7.6 / 6.9 | 13.2 / 7.4 / 7.3 | 19.2 / 13.7 / 13.6 |
| spread ≤ 15% of R | 31.8 / 22.2 / 28.4 | 32.2 / 22.5 / 28.8 | 35.3 / 23.2 / 28.9 |
| **gate + price ≥ $20 (the live book)** | **35.5 / 36.1 / 44.9** | 35.5 / 35.7 / 44.6 | 37.3 / 33.4 / 43.7 |

V1 is a wash everywhere (fewer targets, bigger payoff). V2 helps a lot on the ungated population (stop-outs 50% → 43%)
but NOT inside the gated $20 book (up in TRAIN, down in VAL and TEST): the widening only pays where the spread is wide
relative to R, which the gate already excludes. **Exits unchanged.** The +0.35 to +0.45R here is the per-SIGNAL
population (1,170 quoted signals, no book caps); the executable book — 12/day, 4 concurrent, gate as a cost filter — nets
**+0.26 / +0.30 / +0.36R per trade** (§8a seed means), ~0.1R lower: first-come on the $20 book LOWERS per-trade R vs the
population on TRAIN/VAL (+0.342/+0.365 vs +0.422/+0.424; TEST the exception). The book number is the one to hold live to.

## 10. Live vs spec — every place the engine is NOT the backtest (9/15 review, owner: "1000% identical or tell me how the P&L is imaginary")

The spec (`trading/hod_break.py::simulate`) and the engine share `detect` byte-for-byte; the differences are in the
WORLD the detector sees and in how fills happen. Each row says which side is favoured; nothing here was in the study's
P&L except where marked "modeled".

| # | spec (backtest) | engine (live) | status | who it favours |
|---|---|---|---|---|
| 1 | every universe symbol-day's bars from 09:30 | **streamed universe** (prev close ≥ $17, ADV20 ≥ 100K) from 09:30; scan hook admits the rest at +3.5% | fixed 9/15 (was: admission after the break = CRWL, 2 of 3 dry days) | — |
| 2 | first break only per symbol-day | first break only; a break seen late (restart, outage) marks the symbol `stale_break`, never a later break | fixed 9/15 (was: engine could take a second break the spec never traded) | — |
| 3 | acts at the bar close | dedicated drain thread evaluates at bar arrival (~1-3 s after the close); before 9/15 the scan cycle could delay by up to ~30 s | fixed 9/15 | — |
| 4 | fill = next bar's OPEN if ≤ level×1.006, else no trade | limit at level×1.006 sent ~2 s into the next minute, only if the ask ≤ limit; fills at the ask ≈ the open; canceled after 20 s (was 75 s: a pullback fill a minute later is a trade the spec never took) | fixed 9/15 | live pays the ask (half-spread) — **modeled** as the entry cost in §8 |
| 5 | R, size, target from the actual fill | R/size from the ask at submission; target re-anchored to the real fill via a take-profit replace after the fill | fixed 9/15 (was: target fixed from the estimate) | — |
| 6 | target fills when a bar CLOSES ≥ target, at the target price | broker take-profit LIMIT leg: fills on any trade through the price (wicks included) | structural — cannot be identical | **live** (a wick touch that reverses is +2R live, a stop or less in the spec); never harms live |
| 7 | stop fills at min(stop, open) × 0.999 when a bar's low ≤ stop | broker stop-market leg triggered by a trade ≤ stop; fills at the market | structural | slippage beyond 10 bps on thin names hurts live — measured per trade (`EXIT … (R)` lines) |
| 8 | 15:55 flat at the bar's open | 15:55 marketable limit at bid × 0.99 | structural, minor | live pays the half-spread — modeled in §8 for non-target exits |
| 9 | ADV20 = mean volume of the 20 prior sessions | daily_bars' latest 20 rows (≥ 10 rows) | same definition, nightly-refreshed table | — |
| 10 | volume/HOD from SIP cached bars | SIP websocket bars (same feed); a reconnect after the open now re-backfills every candidate before any evaluation | fixed 9/15 (was: an outage silently shrank the day) | — |
| 11 | no notional cap, size = risk / R | `max_notional_usd` 10,500: never binds at risk $100 / min_r 1%; binding is logged as a WARNING | fixed 9/15 (was 5,000: every trade with R < 2% was under-sized vs the backtest) | — |
| 12 | no kill rails, no once-per-symbol rule (implicit), 12/day first-come, 4 concurrent | daily/weekly kill rails, once-per-symbol, 12/day, 4 concurrent (working orders count; a no-fill frees the slot) | rails are live-only risk controls | — |

Verdict on the study's P&L: the only fill assumptions the live account cannot reproduce are rows 6-8. Row 6 is
conservative in the study's favour (the study under-counts target exits relative to a real limit leg); rows 7-8 are
costs the §8 cost model charges at half a spread — stop slippage beyond that is the residual the EOD parity line
measures trade by trade. The numbers that were imaginary were not the P&L but the INPUT: the engine was not seeing the
spec's world (rows 1-3, 10) and would have traded a different, smaller book. Parity is now tested end-to-end through the
live seams (`tests/test_hod_break_replay.py`) and measured daily by `scripts/hod_break_miss_audit.py` (spec over the
whole universe vs the engine's journal — the miss rate is THE number).

## 11. Selection integrity (9/15 parity review of §8/§8a; probes in `research/bf_zero/parity_review/`)

- **Both cuts were chosen with TEST on screen.** `spread_score.py:14-19` prints TRAIN/VAL/TEST in one table; commit
  `628516b` (18:49 UTC) adopted `max_spread_frac_r: 0.15` with §8 already showing TEST; `2fa1f51` (18:55, six minutes
  later) set `min_price: 20` from the price-band × split table with TEST visible. §6 already discloses that the $5 floor
  was chosen after seeing all splits. There is no TEST-blind decision anywhere in the chain; "TEST read once" means
  "TEST displayed in the same run as the decision".
- **The pre-registered rule fails at 0.15 on VAL.** The rule (`spread_study.py:13-16`) was on RAW mean R: excluded
  buckets worse than kept on TRAIN AND VAL. At ≤ 15% VAL kept +0.262 vs dropped +0.270 (`spread_study.md:31`). The
  adoption switched to the after-cost criterion, which is monotone in the cost by construction. As a cost filter the
  gate is legitimate; as a "pre-registered rule met" it is not.
- **16 cells were visible**: 4 cuts (0.10/0.15/0.20/0.30) × 4 floors ($5/10/20/50). The chosen (0.15, $20) is the
  maximum in no split — (0.10, $20) nets +0.294/+0.426/+0.571, (0.15, $50) +0.350/+0.395/+0.586. Per-split SE of the
  gated $20 book is 0.067–0.078R (sd 1.41, n 331–437). Choosing a floor among ~4 with the scoring split visible costs
  about one SE: **haircut −0.05 to −0.08R** off the pooled +0.38.
- Inside the $20 band the gate is a cost filter, not a quality selector: raw R pass/fail TRAIN +0.402/+0.475,
  VAL +0.418/+0.402, TEST +0.512/+0.384; the excluded 57% are still net ≈ +0.2R (median spread/R 0.306). The gate
  trades frequency for per-trade quality; it is not what makes the sign.
- **Costs live pays that §8 did not charge** (none flips the sign; the $20 gated book has median spread 16.6 bps and
  median R 2.2% of price, so 10 bps = 0.045R): the ask moving in the 2–3 s between the bar close and the order (each
  +10 bps on the fill = −0.05R/trade; historically the last ask of the signal minute is only +5.9 bps median above the
  next open, so the modeled half-spread is about right IF the fill is at that ask); stop-market slippage beyond the
  modeled 10 bps + half spread (each +10 bps on stops = −0.02R/trade; +50 bps still leaves +0.23/+0.26/+0.34); partial
  or missed fills of the 20 s limit (unmeasured); kill rails (−6R day rail hits 1.6% of days, truncates ≈ +9R over
  ~2,300 trades — negligible). The target moved to ask + 2(ask − stop) (1.5 spreads above the spec's) is already in
  §9's V0 re-walk: ≤ 0.017R.

**Honest expectation at $100 risk.** Net **+0.20 to +0.30R per trade** (point +0.25; pessimistic +0.10 with entry
+20 bps, stop +30 bps and the selection haircut; optimistic +0.40), **25–30 trades a week → +5 to +8R ≈ $500–$800 a
week**, weekly sd ≈ $1,000, 25–30% red weeks, worst week −$1,000 to −$1,500. Range across assumptions $250–$1,100 a
week. The sign rests on the ($20, ≤ 15%) cell chosen with TEST visible; it has to be earned by the live measurables
below (rows 5, 9, 11 first), never by the backtest.

## 12. Live measurables (the EOD check's spec — each must match for the §8a claim to transfer)

| # | measurable | spec expectation / band | live source |
|---|---|---|---|
| 1 | gate pass rate, $20+ signals | 42.6% (48/39/40 by split); band 30–55% over ≥ 50 signals | count `[HOD]…of R … > 15% — skip` vs `WOULD BUY`/`ENTRY SUBMITTED` + `no_chase`/`spread` skips (`cand.rejected_reason`) |
| 2 | signals/day ($20, ≤14:00, pre-gate) | 11–25 median; 0-signal days ≈ 0 | `_try_enter` reached, any reason ≠ price/day_cap/conc |
| 3 | fills/day | 4.6–6.6 (22–32/wk); day-cap 12 rarely binds, concurrency binds ~30% of gated signals | `trades` rows `strategy='hod_break'` with `fill_price`; `concurrency cap — skip` count |
| 4 | fill rate of submitted orders | ~100% of ask ≤ cap (spec); alert < 85% | `order_status` filled vs `time_stop_canceled`; `filled_qty`/`shares` for partials |
| 5 | entry fill vs next-minute open | median ≤ +8 bps (half spread), mean ≤ +15; each +10 bps = −0.05R | `fill_price` (DB) vs REST 1-min bar open of the fill minute (EOD check already loads bars); `FILLED … slip … bps vs level` is vs level, not vs open — add the open |
| 6 | ask at decision vs next open | median +6 bps | `pattern_data.quote_ask` vs REST open |
| 7 | TP-fill rate | 33–39% of trades (V0 gated $20); live ≥ spec (wick fills, §10 row 6) | `exit_reason='target'` share |
| 8 | stop rate / eod rate | 41–45% / 20–22% | `exit_reason` shares |
| 9 | stop fill vs stop price | modeled −10 bps −half spread ≈ −18 bps; alert if mean worse than −40 bps (−0.09R/stop) | `exit_price` vs `stop_loss_price` on `exit_reason='stop'` |
| 10 | eod fill vs 15:55 open | −half spread | `exit_price` vs REST 15:55 open |
| 11 | mean R/trade (net, realized) | +0.25–0.35 book; SE 0.07 at n 350 → no verdict before ~150 trades (SE 0.12) | `EXIT … (±x.xxR)` / `pnl / (fill−stop)·shares` |
| 12 | WR | 48–53% | `pnl > 0` share |
| 13 | weekly R | +6 to +10R mean, sd 8–10; red weeks ~25%; worst −8 to −12 | weekly `pnl` sum / risk_usd |
| 14 | miss rate vs spec | 0 | `scripts/hod_break_miss_audit.py` |
| 15 | spread at decision, $20+ passing | median 17 bps, sfr median 0.082 | `WOULD BUY`/`ENTRY SUBMITTED … spread N bps`, `pattern_data.quote_bid/ask` |
| 16 | `r_min` reject rate ($20+ signals) | a few % of signals; the live gate is on r = ask − stop, which is LARGER than level − stop when the ask is above the level, so it is looser than the spec's next-open basis, not tighter — the 7 rejections on 9/15 were stops genuinely within 1% that `simulate` rejects too (r/entry < 1% on the next open ≈ the ask) | `[HOD] … stop … within 1.0% of the ask … — skip` |

Dry-run tape so far (journal 9/14–9/15): 9/14 gate OFF, $5 floor — 42 `WOULD BUY`, 11 `NO CHASE`, 10 `spread > 100 bps`;
9/15 gate ON, $20 floor, engine restarted 16:49 UTC mid-session — 0 `WOULD BUY`, 7 `r_min`, 3 gate, 3 `NO CHASE`,
1 `spread > 100 bps`. Zero live fills: every number above is still model-vs-model.

### 6a. Book re-run 2026-09-15 evening (causal slot rule, early-close days excluded, live config)
`spec_sim.py` now (a) frees a concurrency slot only for an exit on a bar STRICTLY BEFORE the entry bar (the old rule
freed it in hindsight for an exit during the entry bar — 5% of the study's trades were admitted that way, review F),
(b) breaks same-minute ties by symbol (the live engine's order), (c) excludes the early-close days 2025-07-03,
2025-11-28, 2025-12-24 (the cache carried after-hours prints; live never runs on them), and (d) reports the LIVE-CONFIG
book: last entry 14:00, price ≥ $20, 12/day, 4 concurrent — the same signals, the engine's knobs:

| book | TRAIN | VAL | TEST |
|---|---|---|---|
| executable 8/4 (study defaults, all prices) | 36.8/wk, +0.236R, weekly +8.7, green 42/53, worst −18.5 | 35.4/wk, +0.208R, +7.4, 16/23, −13.5 | 35.8/wk, +0.213R, +7.6, 12/15, −4.9 |
| LIVE-CONFIG, spread cost NOT charged | 37.1/wk, +0.342R, +12.7, 43/53, −10.6 | 41.3/wk, +0.365R, +15.1, 21/23, −5.5 | 39.3/wk, +0.421R, +16.5, 13/15, −0.9 |
| LIVE-CONFIG + the 15% gate as a 42% cost filter (§8a, 20 seeds, mean) | 22.2/wk, +0.263R, +5.8, 39.5/53, −11.0 | 31.7/wk, +0.303R, +9.6, 20.1/23, −4.2 | 29.0/wk, +0.360R, +10.4, 12.8/15, −3.1 |

**Open on the study side (reviews A and B, 9/15)**: 45% of the study's symbol-days (`bars.db`, `pit_bars_1min.db`)
were fetched from Databento EQUS.MINI — ONE publisher, 1–10% of consolidated volume, highs differ on 95% of
symbol-days. Those days could never pass the rv ≥ 1 gate, so the book was scored on the other 55% (Alpaca SIP, byte-
identical to live); the 61 book trades served by the pit store do not reproduce on SIP bars. A re-fetch from
consolidated sources and a full re-simulation is running; until it lands, every number above is a book over the
SIP-served half of the universe, and the population it excludes (~1.6% of the ≥$19 causal superset) is one the live
engine WILL trade. The study population also excludes touch-only breaks (`h == level` without a 0.3% trade-through:
2.1% of live signals on two sampled days, all −1R) — decision after the re-sim by the pre-registered rule: a
`break_through` knob in `detect` on both sides, or none.
