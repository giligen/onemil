# ORB Bar-1 Fill Study — capture policy × placement latency (2026-08-14)

Trigger: IREX 2026-07-30. The 9:35 bar opened AT the range high (10.9445) and ran
+2.8% in 60s; the live stop-limit was created 73s after 9:35:00, born with the
market 2.8% above its limit, never filled (`time_stop_canceled`), and missed the
BT's only ≥$3K monster in 4 months (+$4,361). Question: which capture policy
(band width / order type / latency) retains the most of the BT book?

**Read section 1 first. The study surfaced a BT data bug that is bigger than the
fill question and reframes the book this study was asked to defend.**

---

## 1. CRITICAL SIDE-FINDING: DST session-open bug contaminates 40/236 BT trades (62% of book P&L)

`study_orb.py::_session_open_timestamp` picks the first cached bar whose UTC
timestamp has `minute == 30 and hour in {13, 14}`:

- In **EST season (early Nov → early Mar)**, 9:30 ET = **14:30** UTC, but 13:30
  UTC = **8:30 ET premarket**. Any winter symbol-day whose cache contains an
  8:30 ET premarket bar gets its "opening range" computed from **8:30–8:34
  premarket**, and its entry trigger = premarket-range-high × 1.003.
- Variant: a day with **no 9:30 bar at all** (thin open) matches 14:30 UTC in
  EDT season = **10:30 ET** → range computed from 10:30–10:34 (JMIA 2026-05-07,
  AVEX 2026-06-26).

Verified example: ACON 2025-03-03 — CSV entry 6.3892 = 6.3701 × 1.003, where
6.3701 is the **8:31 ET premarket** bar high (cache.db `2025-03-03T13:31Z`).
The true 9:30–9:34 range high was 8.13; the CSV trigger is below the entire
regular-session opening range.

Scope on the current book (`analysis_results/orb_static_lock_trades.csv`,
236 trades Jan'25→Aug'26, total `_sized_pnl` $251,647):

| | trades | sized P&L | monsters (≥$3K) |
|---|---|---|---|
| Clean (SIP-reconstructed trigger within 10bps of CSV) | 196 | $96,048 | 14 ($143,865) |
| **Bugged** (38 winter-DST + 2 missing-9:30-bar) | **40** | **$155,599 (62%)** | **10 ($148,868)** |

The bugged rows include the two biggest trades in the entire book — CRNC
2025-01-03 (+$40,866) and AMCI 2025-12-15 (+$45,954) — plus LAES, SANA, VNCE,
TNXP, QCLS, NAMM, PHGE, CDIO. Era totals flip: the clean book's 2025H1 is
**−$14,166** (published book: +$60,016).

Implications:

- The BT's winter-season entries fire off premarket ranges the live engine
  (which builds its range from live 9:30+ bars) would never compute →
  **BT↔live parity is broken by construction for EST-season days**, despite the
  shared-helper parity architecture. Live has only run May–Aug 2026 (EDT), so
  no live day has been affected *yet* — the divergence goes live ~Nov 1 2026.
- These triggers are also frequently *untradeable*: by 9:35 the market is far
  above a premarket-derived trigger, which is exactly why 7 of the 8 monsters
  the CURRENT policy "misses" in the full-book simulation below are bugged rows
  — they are artifacts, not latency victims.
- Every downstream decision validated on this book (selection filters, veto
  budgets, sizing, the $342K/Calmar headline lineage) inherits some
  contamination. **This bug must be fixed and the features/book regenerated
  before any further BT-based ship decision.** Note the bug does NOT mean
  winter is unprofitable — it means the current winter evidence is fiction in
  both directions (a corrected re-run may find different, possibly still-good,
  winter trades).

This study therefore reports two parallel analyses: the **CLEAN-196 subset**
(primary evidence for the fill-policy question) and the FULL-236 book (as
specified, using CSV triggers verbatim).

---

## 2. Measured live placement latency (data/trades.db, strategy='orb', n=159, 2026-05-18→2026-08-12, all EDT)

Latency = `created_at` − 13:35:00 UTC of trade date:

- **All orders: p10 = 2.1s, p50 = 3.9s, p90 = 73.5s** (min 1.1s, max 661s; the
  661s outlier is GLL 2026-06-24).
- **Regime break at 2026-07-07.** First-order-of-day latency by era:
  - Before 2026-07-07: ~1.7–3s (p50 ≈ 2.6s).
  - **2026-07-07 onward: every single day's first order is 32–73s late**
    (07-29: 62s, 07-30: 73s, 07-31: 62s, 08-11: 55s, 08-12: 66s). Trending up.
- Later orders within a day: p50 4.1s, p90 121s.

So IREX's 73s was not tail bad luck — it is the **current first-order norm**.
The regression date coincides exactly with the PM-mult/news prefetch added to
`orb_engine.check_entries` (code comment dated 2026-07-06: "warm the premarket
dollar-volume cache on early ticks (>=9:31 ET) so the 9:35 burst never
blocks"). The measured data says the 9:35 burst **does** still block 40–73s —
the prefetch chain (news batch fetch w/ 8s timeout, asset-class resolution,
anchor pre-warm, REST range sweep) sits synchronously inside `check_entries`
ahead of order submission. Exact blocking call not root-caused here (read-only
study); that is follow-up work.

Latency grid used below: **L ∈ {4s (all-order p50), 30s, 74s (current
first-order norm / p90), 120s}**. Note L=4 and L=30 both fall inside the 9:35
bar, so under bar-level logic they produce near-identical results by
construction (see approximations).

---

## 3. Methodology

Data: 236 BT trades from `analysis_results/orb_static_lock_trades.csv`; 1-min
**SIP** bars per symbol-day fetched from Alpaca (`/v2/stocks/bars`, feed=sip),
cached in the session scratchpad (`bars_cache/`, 236/236 fetched, 0 failures).

**Trigger reconstruction.** T_recon = max(high of 9:30–9:34 ET bars) × 1.003.
Match rate vs CSV `entry_price`: **196/236 (83.1%) within 10bps**. All 40
material disagreements are the bug rows of section 1 (CSV trigger far BELOW
reconstruction). Per spec, the simulation uses the CSV's implied trigger
(T = `entry_price`) for all rows; the clean/full split isolates the bug.

**Sizing reconstruction** (from `study_orb_pipeline_static_lock.py`): CSV `pnl`
is at a $50K nominal position; `_rp_position` = min($25K, $3K risk /
range_size_pct) — **capped at $25K for all 236 rows** (every selected range
< 12%), so BT position dollars are entry-price-independent; `_sized_pnl` =
`pnl` × (25K/50K) × quintile mult × `_pm_mult`. I therefore recover each
trade's effective sized dollars as **P = `_sized_pnl` / (pnl_pct/100)** and
re-simulate `new_pnl = P × (exit − fill)/fill`. This mirrors the BT's own
sizing model exactly. (Live risk-parity uses (fill − range_low), which would
shrink size slightly on higher fills — cushioning wide-band losses AND gains;
ignoring it is roughly symmetric, slightly pessimistic for wide-band losers.)

**Exit recovery.** exit = `entry_price` × (1 + pnl_pct/100), held fixed across
policies. Exact for `stop` (range_low), `eod`, `tag_bb` exits (entry-
independent levels/decisions; 193/236 trades). Approximate for `lock` (34) and
`tag_b1` (1): the +1R lock stop would sit higher for a higher fill, so holding
the BT exit **understates** wide-band/stop-market P&L slightly (conservative
against the alternatives that end up losing anyway — bias does not change the
ranking direction).

**Fill simulation** (order lives 60 min from placement; +10bps slip on every
fill; bar containing 9:35:00+L is the placement bar):

- Stop-limit (band b): stop T, limit T×(1+b). Trigger = first bar ≥ placement
  with high ≥ T. Fill on trigger bar at min(limit, max(T, open)) if open ≤
  limit, else at limit if low ≤ limit; on later bars at min(limit, open) if
  open ≤ limit else limit when low ≤ limit. **Placement-bar conservative
  rule**: if the placement bar's OPEN > limit, the order cannot fill in that
  bar (models being born above-limit mid-bar, i.e. IREX).
- Stop-market: first bar with high ≥ T; fill = max(T, open); on the placement
  bar with open already > T, fill = max(open, close) (conservative: mid-bar
  market entry at the worse of open/close).
- No fill within 60 min → missed, P&L $0.

**Approximations and bias directions** (beyond the above):

1. 60s bar granularity: within-bar trade sequencing is invisible. For L=30 the
   order is credited with the full 9:35 bar (optimistic); sub-minute
   gap-throughs inside the trigger bar are invisible (optimistic for narrow
   bands). The placement-bar open>limit rule claws back the worst of this
   (conservative), verified to reproduce the real IREX miss at L≥74.
2. Flat 10bps entry slip regardless of speed/spread — real slip on fast tape
   exceeds this (optimistic everywhere, most for stop-market).
3. No slot/budget/concurrency re-interaction; each trade re-simmed in
   isolation.
4. SIP bars vs the live/BT bar feed: raw cache.db bars were spot-checked
   identical to SIP for matched windows. 4 clean trades (ALMS 7/25/25, CCUP
   9/15/25, BTQ 10/27/25, MSTX 3/25/26) never reach their CSV trigger in SIP
   even at L=4 — all four were BT losers totaling −$2,139, so reality is
   slightly *better* than the BT on them; they count as misses in every cell.

---

## 4. Results — CLEAN-196 (primary): BT baseline $96,048; 14 monsters $143,865; eras H1 −$14,166 / H2 +$19,390 / 2026 +$90,824

| L | policy | cap | miss | total P&L | %BT | mon cap | mon P&L | mon% | avg degr | H1 / H2 / 2026 |
|---|---|---|---|---|---|---|---|---|---|---|
| 4s | **limit30 (CURRENT)** | 192 | 4 | **$92,864** | **96.7%** | 14/14 | $143,322 | 99.6% | 9bp | −15.5K / 19.6K / 88.8K |
| 4s | limit100 | 192 | 4 | $91,684 | 95.5% | 14/14 | $143,322 | 99.6% | 12bp | −15.5K / 19.4K / 87.8K |
| 4s | limit200/300/stopmkt | 192 | 4 | $91,671 | 95.4% | 14/14 | $143,322 | 99.6% | 12bp | −15.5K / 19.4K / 87.8K |
| 74s | **limit30 (CURRENT)** | 185 | 11 | **$82,145** | **85.5%** | 13/14 | $139,109 | 96.7% | 12bp | −17.1K / 15.6K / 83.6K |
| 74s | limit100 | 187 | 9 | $78,543 | 81.8% | 13/14 | $138,294 | 96.1% | 22bp | −18.1K / 15.6K / 81.0K |
| 74s | limit200 | 190 | 6 | $77,815 | 81.0% | 13/14 | $138,294 | 96.1% | 30bp | −18.3K / 17.1K / 79.0K |
| 74s | limit300 | 191 | 5 | $78,608 | 81.8% | 14/14 | $141,827 | 98.6% | 35bp | −19.9K / 16.9K / 81.7K |
| 74s | stopmkt | 191 | 5 | $71,472 | 74.4% | 14/14 | $141,639 | 98.5% | 47bp | −21.9K / 15.8K / 77.6K |
| 120s | limit30 | 179 | 17 | $58,861 | 61.3% | 12/14 | $120,480 | 83.7% | 14bp | −21.5K / 14.7K / 65.6K |
| 120s | limit100 | 182 | 14 | $73,755 | 76.8% | 13/14 | $138,056 | 96.0% | 24bp | −20.6K / 13.6K / 80.7K |
| 120s | limit200 | 186 | 10 | $73,587 | 76.6% | 13/14 | $137,533 | 95.6% | 37bp | −20.0K / 15.8K / 77.8K |
| 120s | limit300 | 188 | 8 | $73,518 | 76.5% | 14/14 | $141,010 | 98.0% | 46bp | −22.1K / 15.6K / 79.9K |
| 120s | stopmkt | 188 | 8 | $60,585 | 63.1% | 14/14 | $138,279 | 96.1% | 68bp | −25.7K / 12.2K / 74.0K |

(L=30 rows identical to L=4 by bar-construction; omitted. "degr" = mean fill
price above trigger in bps, incl. 10bp slip.)

Monster misses (clean): L≤30: none. L=74 limit30: **IREX only** (−$4,361).
L=120 limit30: IREX + ANTX 2026-03-09 (−$18,280). limit300/stopmkt at any L:
zero monster misses. Only 3 clean trades in 19 months had a 9:35 bar that
OPENED above the 30bps limit (combined BT value **−$1,580**) — true bar-1
gap-throughs are rare and, historically, not even profitable as a group.

### FULL-236 (per spec, CSV triggers verbatim; interpret with section 1 in mind)

BT baseline $251,647; 24 monsters $292,733.

| L | policy | cap | miss | total P&L | %BT | mon cap | mon P&L | mon% |
|---|---|---|---|---|---|---|---|---|
| 4s | limit30 | 213 | 23 | $123,990 | 49.3% | 17 | $162,474 | 55.5% |
| 4s | limit100 | 215 | 21 | $123,558 | 49.1% | 18 | $164,199 | 56.1% |
| 4s | limit200 | 216 | 20 | $115,175 | 45.8% | 19 | $164,990 | 56.4% |
| 4s | limit300 | 217 | 19 | $147,904 | 58.8% | 20 | $201,042 | 68.7% |
| 4s | stopmkt | 222 | 14 | $151,131 | 60.1% | 24 | $218,973 | 74.8% |
| 74s | limit30 | 205 | 31 | $106,376 | 42.3% | 16 | $158,261 | 54.1% |
| 74s | limit100 | 208 | 28 | $100,455 | 39.9% | 16 | $156,672 | 53.5% |
| 74s | limit200 | 213 | 23 | $98,614 | 39.2% | 18 | $159,962 | 54.6% |
| 74s | limit300 | 215 | 21 | $131,880 | 52.4% | 20 | $199,547 | 68.2% |
| 74s | stopmkt | 220 | 16 | $132,678 | 52.7% | 24 | $221,877 | 75.8% |
| 120s | limit30 | 198 | 38 | $83,071 | 33.0% | 15 | $139,632 | 47.7% |
| 120s | limit100 | 203 | 33 | $95,190 | 37.8% | 16 | $156,434 | 53.4% |
| 120s | limit200 | 209 | 27 | $93,882 | 37.3% | 18 | $159,201 | 54.4% |
| 120s | limit300 | 212 | 24 | $126,133 | 50.1% | 20 | $198,730 | 67.9% |
| 120s | stopmkt | 217 | 19 | $121,498 | 48.3% | 24 | $219,163 | 74.9% |

Even at L=4 with an infinite-band proxy (stop-market) the full book captures
only 60% of BT P&L, and 7 of the 8 monsters "missed" by CURRENT at L=4 are bug
rows (CRNC, VNCE, TNXP, QCLS, AMCI, NAMM, PHGE) — the market was already far
above their fictional premarket triggers at 9:35. The full-book table measures
the bug, not the fill policy. The apparent case for stop-market here ("+24
monsters at every latency!") is an artifact of chasing untradeable triggers
and vanishes on the clean subset.

---

## 5. Answers to the three key questions (clean subset)

**(i) How much of the BT book does CURRENT capture at measured latency?**
At the all-order p50 (3.9s): **96.7% of P&L, 100% of monsters** — the BT's
L≈0/band-hit fill assumption is nearly honest *when orders go out fast*. At
the current post-2026-07-07 first-order norm (~74s): **85.5% of P&L**, and it
starts dropping monsters (IREX). The structural BT-live fill gap is therefore
~3pp at healthy latency and ~15pp at today's degraded latency.

**(ii) Which policy dominates at realistic latency?** **CURRENT (30bps) wins
on total P&L at every latency ≤74s.** At L=74: limit30 $82.1K > limit300
$78.6K > limit100/200 ≈ $78K > stopmkt $71.5K. Wider bands do recover IREX
(+$4.4K monster P&L) but pay ~23–35bps of extra entry degradation across ~190
ordinary trades (≈$8–11K) — a net loss. Stop-market is strictly worst at
every latency. Era check: limit30 beats limit300 in 2 of 3 eras (limit300 is
+$1.3K better in 2025H2 only) — no era where band-widening decisively wins.
The one winner-flip: **if latency degrades to ~120s, limit30 collapses (61.3%)
and limit100–300 (~76.6%) become clearly better** — band width is a hedge
against latency you should not need.

**(iii) What does cutting 74s→5s buy without changing order type?**
**+$10.7K over 19 months on the clean book (85.5% → 96.7% capture, +7
trades, +IREX-class monsters)** — more than any band/order-type change is
worth at any latency, with zero adverse-selection cost.

---

## 6. Ranked recommendation

1. **Fix the `_session_open_timestamp` DST/missing-bar bug** (require the bar
   matching the date's true 9:30 ET UTC instant; fall back to first bar ≥9:30
   ET, never a minute==30 pattern match), regenerate `orb_features_*.csv` and
   the trade book, and re-validate downstream decisions made on the
   contaminated book. Until then, treat all EST-season BT results — and the
   book's headline totals — as unreliable. (Not a fill-policy item, but it
   dominates everything else found here: $155.6K of the $251.6K book.)
2. **Fix the order-placement latency regression** (first-order 2–3s before
   2026-07-07, 40–73s since; the prefetch chain in
   `orb_engine.check_entries` is the prime suspect — make the news/PM/anchor
   warm-up genuinely complete before 9:35 or move it off the entry path, and
   alert if first-order latency exceeds ~10s). Target ≤5s. Worth ~+11pp of
   book capture; recovers the IREX class.
3. **Keep the CURRENT 30bps stop-limit. Do not widen the band; do not switch
   to stop-market** at fixed-latency parity they lose $3.5–21K vs CURRENT on
   the clean book, and true bar-1 gap-throughs (3 in 19 months, −$1.6K BT
   value) are not worth chasing.
4. **Conditional fallback only**: if latency cannot be held under ~2 minutes,
   a 100bps band beats 30bps (L=120: $73.8K vs $58.9K). This is an argument
   for the latency alert in (2), not for shipping a wider band now.

## 7. Caveats

- 60s bar granularity: sub-minute sequencing invisible; L=5s vs 30s
  indistinguishable by construction; narrow-band results slightly optimistic
  inside trigger bars, partially offset by the conservative placement-bar rule.
- Fixed BT exit path per trade (exact for 82% of trades; slightly pessimistic
  for lock exits under higher fills) and fixed sized dollars (BT's own model —
  all 236 trades cap at $25K).
- Flat 10bps slip; real fast-tape slippage is worse, which further penalizes
  stop-market relative to the table.
- No slot/concurrency/daily-loss re-simulation; trades re-simmed independently.
- The clean-book totals ($96K/19mo) are NOT a P&L forecast — they are the
  uncontaminated slice of a book whose selection layer itself needs a re-run
  after the bug fix.
- Latency sample is 51 live days, one season (EDT); the 2026-07-07 regime break
  is measured fact, its root cause is attributed but not proven here.

*Artifacts: scratchpad `sim_fill.py`, `classify_bug.py`, `fetch_bars.py`,
`row_classification.json`, `sim_results.json`, `bars_cache/` (236 symbol-day
JSONs) under the session scratchpad dir. Report generated 2026-08-14.*
