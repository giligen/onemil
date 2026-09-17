# D1 / ORB — the retro-veto audit of the live trades, and the selection dose-response

Run 2026-09-17, one process at a time, `nice -n 10`, detached with a log for the
long walk. Read-only outside `research/fuckup_audit/D1_orb/`; `data/cache.db`
opened read-only; `data/trades.db` never opened (the source is the existing dump
`research/fuckup_audit/live_trades_dump.csv`). No config, service, order, cron or
production artefact written. **No veto threshold, z-param, quintile cutoff or
mult was re-tuned** — every rule is evaluated through the shipped helper
(`trading/orb_{pdr,g1,range_size,catalyst}_veto.py`, `study_orb_filter`,
`study_orb_sizing`) with the values that are in `orb.yaml` right now.

Parity anchor: the base run reproduces the nightly production book to the cent —
**$7,186.64 / 88 picks / 70 fills / 21 months**, identical to
`analysis_results/orb_monthly_static_lock.csv` (cum_pnl 7186.637916695402) and
`orb_bplus_book.csv` (88 rows). Everything below is measured off that same
candidate dump.

---

## PAGE ONE

### 1. What B+ would have done with the 115 pre-B+ live trades

Live ORB, 2026-05-19 → 2026-08-27: **117 trades, −$4,803, −10.31R, WR 38.5%**.
115 of them ran under the retired pre-B+ config; only 2 are B+ (+$113).

Retro-applying the **shipped** B+ stack to those 117 trades:

* **94 of 117 are blocked** (−$2,428 of realized P&L).
* **69.8% of live stop dollars sit in blocked trades** — 41 of the 55 stops,
  −$8,525 of −$12,214. The review's pre-registered bar was ≥ 60%. **It passes.**
* The 23 survivors are still **−$2,375**… but **−$1,959 of that is two
  `stop_loss_market_fallback` exits** (EHGO 6/25, IRE 6/29) — execution
  pathology, not selection. **Survivors ex-fallback: 21 trades, −$416, −2.42R,
  WR 47.6%** — a book that is flat inside its own noise.
* Tighter still: only **8 of the 117 live trades are both inside B+'s top-3
  ranking for their day and clear every veto**. Those 8 realized **+$363**.
  The other 109 trades (−$5,166) are trades the shipped machine would not own.

**Verdict on the "116 real ORB losers":** they are a post-mortem of a retired
config. The live loss decomposes as −$2,428 config-era selection + −$1,959
execution pathology + −$416 residual. Nothing in the 115 trades is evidence
against the B+ book, and nothing in them is evidence *for* it either (n=8).

### 2. Does frequency add $/month, or only variance?

It adds money, and the selection keeps working down to about **rank 8**, then
stops. At the same per-position sizing (`per_pos_cap = account/N = $3,333` for
every N, so each trade is sized identically and only the slot count changes):

| slots | picks | fills | P&L 21 mo | $/month | R / pick | MDD | worst month | red months |
|---|---|---|---|---|---|---|---|---|
| 3 (shipped) | 88 | 70 | $7,187 | **$342** | +0.515 | −$714 | −$223 | 6 |
| 4 | 118 | 93 | $10,214 | **$486** | +0.568 | −$813 | −$250 | 2 |
| 6 | 165 | 131 | $12,543 | **$597** | +0.536 | −$782 | −$148 | 3 |
| 8 | 215 | 162 | $14,429 | **$687** | +0.480 | **−$576** | −$148 | 2 |
| 12 | 267 | 198 | $15,563 | **$741** | +0.395 | −$944 | −$411 | 4 |

Marginal picks (the books nest — greedy rank order, per-pick vetoes):

| added slots | picks | fills | P&L | R / added pick | 2025 | 2026-01..05 | 2026-06+ |
|---|---|---|---|---|---|---|---|
| 4 | +30 | +23 | +$3,027 | **+0.725** | +1,156 | +1,943 | −71 |
| 5–6 | +47 | +38 | +$2,329 | **+0.456** | +1,166 | +1,230 | −66 |
| 7–8 | +50 | +31 | +$1,885 | **+0.293** | +689 | +533 | +663 |
| 9–12 | +52 | +36 | +$1,134 | **+0.042** | +434 | +246 | +454 |

So: **yes below rank 4, weakening monotonically, gone by rank 9.** Ranks 9–12
are +0.042R per pick — noise at this n. Going 3 → 8 slots roughly doubles the
dollars *and* shrinks the drawdown (−$714 → −$576) and the red-month count
(6 → 2), because the extra picks diversify the day. Going past 8 buys $54/month
for a 64% worse drawdown and a worst month that goes from −$148 to −$411.

Widening the *candidate pool* instead of the slot count does **not** work — and
cannot be pushed to 3×: with Q1 dropped the composite threshold is inert (Q1 is
`composite < 0.1059`, far above the threshold `0.0121`), so the only pool lever
is Q1-on, worth 1.25× at the frozen threshold and 2.26× at the whole candidate
universe. 3× does not exist. At N=4 the 2×/full-pool book is **$9,533** vs
$10,214 (worse). At N=12 it is $16,415 vs $15,563 (+$852) for **MDD −$2,388 vs
−$944 and a worst month of −$1,193 vs −$411**. Pool widening fails the shape test.

### 3. The honest $/month, and what 3× sizing would look like

Two numbers matter and one of them is not `risk_per_trade`. **The $3,333
per-position cap binds on 100% of picks** — the effective risk per trade is
**~$143, not the configured $375** (ORB stops are 4–8% of price, so
`375 / stop%` always exceeds the cap). At this stage the book is
cap-constrained; `sizing.risk_per_trade_usd` is an inert knob. "3× sizing"
therefore means "3× the per-position cap" and, because every position is
cap-bound, the P&L scales **exactly** 3× before impact.

The declared account budget (`account/N`) is also not the capital actually
needed: the post-veto daily pool is small, so the slot cap rarely binds.

| slots | max picks in one day | capital really needed (max-day × pos) | $/month at stage | $/month at 3× |
|---|---|---|---|---|
| 3 | 2 | $6,666 | $342 | $1,027 |
| 4 | 3 | $9,999 | $486 | $1,459 |
| 6 | 4 | $13,332 | $597 | $1,792 |
| 8 | 4 | $13,332 | **$687** | **$2,061** |
| 12 | 5 | $16,665 | $741 | $2,223 |

Participation (position $ ÷ the 5-minute opening-range dollar volume, from
`range_total_volume × entry_price`, N=4 book, 118 picks):

| sizing | median | p75 | p90 | picks > 5% of the 5-min tape |
|---|---|---|---|---|
| stage ($3.3K position) | 0.32% | 0.67% | 1.68% | 2 / 118 |
| **3× ($10K position)** | **0.96%** | 2.01% | 5.04% | **13 / 118** |
| 5× ($16.7K position) | 1.59% | 3.35% | 8.40% | 24 / 118 |

3× is obtainable for the median name and marginal for the top decile: 11% of
picks would be taking more than 5% of the opening range's whole dollar volume,
and those are exactly the illiquid names that carry the right tail (the same
trap Stage I found on the intraday sleeve). **The honest sentence: ORB B+ at 8
slots and 3× the per-position cap is a ~$2,000/month book needing ~$40K of
buying power, with 11% of its picks at questionable participation and a 21-month
drawdown of −$1,700 at that scale. At the shipped 3 slots and stage sizing it is
$342/month.** Neither is $10K/month; nothing here changes that.

---

## TASK A — detail

Source: 117 ORB rows of `live_trades_dump.csv` + their `pattern_data` JSON
(`composite_score`, `quintile`, `range_high/low`, `adaptive_mult`, and — on 24
rows only — `pm_dollar_vol`/`has_news`/`anchor`). Features row found for
**113 / 117** (`analysis_results/orb_features_20260916_2053.csv`); the other 4
(IREG 5/21, SOFX 5/27, SSRM 6/16, MUU 7/21) were reconstructed from
`cache.db daily_bars` — prev-day range and 20-day return volatility computable,
**`range_size_pct` NOT reconstructible** (the 09:30 open is not in the dump), so
the range-size veto fails open on those 4 by construction.

### Which rule blocks what (all 117; the pre-B+ 115 in parentheses)

| rule (value from `orb.yaml`) | blocked | P&L | R | mean R | WR |
|---|---|---|---|---|---|
| composite < 0.012081536791 | 0 | — | — | — | — |
| Q1 filter | 1 | −$51 | −0.15 | −0.150 | 0% |
| PDR ≤ 11.0 | 64 | −$2,287 | −7.98 | −0.125 | 37.5% |
| G1 fingerprint (rv20 ≥ 7.106 ∧ pdr ≥ 9.226) | 59 | −$3,606 | −9.88 | −0.167 | 32.2% |
| range-size ≤ 2.221 | 28 | +$12 | +1.43 | +0.051 | 35.7% |
| catalyst required (news ∨ cohort ≥ 2) | 71 | −$3,309 | −5.38 | −0.076 | 36.6% |
| *G1 short-history — OFF since 9/13* | *2* | *+$443* | *+1.83* | *+0.916* | *100%* |
| **ANY shipped veto → BLOCKED** | **94** (94) | **−$2,428** (−$2,428) | −5.35 | −0.057 | 37.2% |
| **SURVIVES → KEPT** | **23** (21) | **−$2,375** (−$2,488) | −4.96 | −0.216 | 43.5% |

The rules overlap heavily (PDR and G1 share the `prev_day_range_pct` leg):

| block reason combination | n | P&L |
|---|---|---|
| (none — kept) | 23 | −$2,375 |
| catalyst only | 23 | −$801 |
| pdr+g1+rs+cat | 19 | −$746 |
| pdr+g1+cat | 18 | −$2,706 |
| pdr+g1 | 12 | +$24 |
| pdr only | 4 | +$790 |
| pdr+g1+rs | 4 | −$80 |
| pdr+cat | 4 | +$400 |
| everything else (6 combos) | 10 | +$514 |

Leave-one-out on the live sample (what each veto is individually worth here):

| veto turned OFF (others on) | kept | P&L | Δ vs all-on |
|---|---|---|---|
| — (all on) | 23 | −$2,375 | — |
| PDR off | 27 | −$1,585 | **+$790** (the PDR veto COSTS money on this sample) |
| G1 off | 25 | −$2,177 | +$198 |
| range-size off | 23 | −$2,375 | $0 |
| catalyst off | 46 | −$3,176 | **−$801** (the catalyst veto is the only one that pays here) |
| Q1 off | 23 | −$2,375 | $0 |

n is 117 and these deltas are 1–3 trades wide; they are reported because they
were asked for, not because they rank the vetoes.

### Stops only

| | n | P&L | R | mean R |
|---|---|---|---|---|
| all live stops (`stop_loss*`) | 55 | −$12,214 | −35.66 | −0.648 |
| blocked by B+ | 41 | −$8,525 | −27.64 | −0.674 |
| kept by B+ | 14 | −$3,689 | −8.02 | −0.573 |

**69.8% of stop dollars blocked.**

### The surviving book, by month (live sizing, live slots)

| month | live n | live P&L | kept n | kept P&L | kept R | kept WR |
|---|---|---|---|---|---|---|
| 2026-05 | 24 | +$1,539 | 2 | +$759 | +2.17 | 100% |
| 2026-06 | 55 | −$3,187 | 5 | −$1,858 | −3.17 | 40% |
| 2026-07 | 32 | −$2,570 | 10 | −$691 | −2.09 | 50% |
| 2026-08 | 6 | −$585 | 6 | −$585 | −1.86 | 17% |

At B+'s own $375 risk the 23 survivors would be **−$1,859**; the whole 117 would
be −$3,867.

### The caveat that limits all of the above

Live picked with the **old** ranking. The B+ frozen fit scores those same trades
differently — mean composite drift **−0.019**, |drift| > 0.05 on 12 trades, and
**26 of 113 land in a different quintile**. Recomputing each live trade's B+ rank
within its day (post threshold + Q1 + family/super-group dedup, pre no-refill
vetoes):

| B+ rank | 1 | 2 | 3 | 4 | 5 | 6 | 7–8 | 10–16 | not in pool |
|---|---|---|---|---|---|---|---|---|---|
| live trades | 16 | 23 | 25 | 17 | 12 | 6 | 5 | 8 | 5 |

| B+ slots | live trades inside top-N | of those, surviving all vetoes | their realized P&L |
|---|---|---|---|
| 3 (shipped) | 64 / 117 | **8** | **+$363** |
| 4 | 81 / 117 | 11 | +$687 |
| 6 | 99 / 117 | 14 | −$225 |
| 8 | 104 / 117 | 17 | −$1,713 |
| 12 | 107 / 117 | 18 | −$1,985 |

This is a **subtractive** counterfactual only: it can say which live trades B+
would have refused, never which trades B+ would have taken instead (those were
never ordered, so they have no realized fill). Post-veto daily pool, over the
423 days in the features file: mean 12.9 deduped candidates, median 8, p90 24.

---

## TASK B — detail

14 pipeline runs, all off ONE candidate dump (`candidates_dump.csv`, 13,033
rows, produced by a single full bar-walk with the shipped exit physics: static
lock 1.75R→0.5R, ATR stop-floor k=0.25 ON, 40%@+3R scale-out ON, touchgo M/D,
15:45 force close). `ORB_BT_RESIM_CACHE` replays the selector only; the N=3
Q1-on cell reproduces the base walk byte-for-byte ($7,187 / 88 / 70), which is
the resim-cache parity check.

Knobs moved: `ORB_BT_N` (slots), `ORB_BT_ACCOUNT` (= 3,333.33 × N so
`per_pos_cap` and therefore every position's dollar size is invariant),
`ORB_SKIP_Q1`, and for the 4 pool cells `ORB_BT_THRESHOLD`. `ORB_BT_RISK` fixed
at the shipped $375. **Every veto stayed at its `orb.yaml` value in all 14 runs.**

Population: 13,033 entered-inclusive candidates = 7,402 fills + 5,631 modeled
non-fills, 427 trading days, 2025-01-02 → 2026-09-16 (21 months; ORB has been
paused since 9/14, so 2026-09 is a partial month of BT-only picks).

### Whole window and per split

Splits: 2025 (12 mo), 2026-01..05 (5 mo), 2026-06+ (4 mo).

| book | picks | fills | fill% | P&L | WR(fills) | $/fill | $/pick | MDD | worst mo | red mo | $/mo |
|---|---|---|---|---|---|---|---|---|---|---|---|
| N=3 Q1-on | 88 | 70 | 79.5 | 7,187 | 42.9 | 103 | 82 | −714 | −223 | 6 | 342 |
| · 2025 | 47 | 38 | 80.9 | 3,650 | 42.1 | 96 | 78 | −618 | −203 | 3 | 304 |
| · 2026-01..05 | 19 | 15 | 78.9 | 2,681 | 33.3 | 179 | 141 | −409 | −223 | 2 | 536 |
| · 2026-06+ | 22 | 17 | 77.3 | 855 | 52.9 | 50 | 39 | −193 | −122 | 1 | 214 |
| N=4 Q1-on | 118 | 93 | 78.8 | 10,214 | 43.0 | 110 | 87 | −813 | −250 | 2 | 486 |
| · 2025 | 66 | 52 | 78.8 | 4,807 | 42.3 | 92 | 73 | −813 | −250 | 1 | 401 |
| · 2026-01..05 | 25 | 21 | 84.0 | 4,624 | 42.9 | 220 | 185 | −409 | +210 | 0 | 925 |
| · 2026-06+ | 27 | 20 | 74.1 | 783 | 45.0 | 39 | 29 | −219 | −122 | 1 | 196 |
| N=6 Q1-on | 165 | 131 | 79.4 | 12,543 | 40.5 | 96 | 76 | −782 | −148 | 3 | 597 |
| · 2025 | 85 | 68 | 80.0 | 5,972 | 42.6 | 88 | 70 | −782 | −134 | 2 | 498 |
| · 2026-01..05 | 40 | 33 | 82.5 | 5,854 | 39.4 | 177 | 146 | −659 | +209 | 0 | 1,171 |
| · 2026-06+ | 40 | 30 | 75.0 | 717 | 36.7 | 24 | 18 | −352 | −148 | 1 | 179 |
| N=8 Q1-on | 215 | 162 | 75.3 | 14,429 | 41.4 | 89 | 67 | **−576** | −148 | 2 | 687 |
| · 2025 | 105 | 84 | 80.0 | 6,662 | 44.0 | 79 | 63 | −528 | −25 | 1 | 555 |
| · 2026-01..05 | 53 | 40 | 75.5 | 6,386 | 37.5 | 160 | 121 | −489 | +159 | 0 | 1,277 |
| · 2026-06+ | 57 | 38 | 66.7 | 1,381 | 39.5 | 36 | 24 | −516 | −148 | 1 | 345 |
| N=12 Q1-on | 267 | 198 | 74.2 | 15,563 | 40.4 | 79 | 58 | −944 | −411 | 4 | 741 |
| · 2025 | 129 | 101 | 78.3 | 7,096 | 41.6 | 70 | 55 | −944 | −411 | 3 | 591 |
| · 2026-01..05 | 62 | 47 | 75.8 | 6,632 | 38.3 | 141 | 107 | −489 | +159 | 0 | 1,326 |
| · 2026-06+ | 76 | 50 | 65.8 | 1,835 | 40.0 | 37 | 24 | −689 | −232 | 1 | 459 |

Q1-off twins (the same slot counts with the bottom quintile allowed to rank):

| book | picks | fills | P&L | MDD | worst mo | red mo |
|---|---|---|---|---|---|---|
| N=3 Q1-off | 94 | 72 | 7,041 | −714 | −223 | 7 |
| N=4 Q1-off | 126 | 96 | 10,058 | −813 | −250 | 3 |
| N=6 Q1-off | 186 | 143 | 12,385 | −782 | −148 | 4 |
| N=8 Q1-off | 251 | 182 | **18,300** | −600 | −182 | 3 |
| N=12 Q1-off | 341 | 242 | 18,538 | −983 | −357 | 5 |

Q1-off is worse at 3/4/6 slots, better at 8 and 12 — and the N=8 Q1-off gain is
**one cell**: 2026-01..05 goes $6,386 → $10,693 (+$4,307) while 2025 goes
$6,662 → $6,301 (−$361) and 2026-06+ −$75. A +$4.3K jump in one 5-month split
from letting the bottom quintile rank is exactly the shape the Q1 filter was
shipped to suppress. Not a recommendation; a single fat trade.

Wider candidate pool (Q1-on + threshold dropped; vetoes untouched):

| book | pool | picks | P&L | MDD | worst mo | red mo |
|---|---|---|---|---|---|---|
| N=4 Q1-on (1.00×) | 5,771 | 118 | 10,214 | −813 | −250 | 2 |
| N=4 Q1-off (1.25×) | 7,225 | 126 | 10,058 | −813 | −250 | 3 |
| N=4 thr −0.5 (2.03×) | 11,699 | 135 | 9,533 | −813 | −250 | 4 |
| N=4 thr −99 (2.26×, everything) | 13,033 | 135 | 9,533 | −813 | −250 | 4 |
| N=12 Q1-on | 5,771 | 267 | 15,563 | −944 | −411 | 4 |
| N=12 thr −0.5 (2.03×) | 11,699 | 444 | 16,415 | **−2,388** | **−1,193** | 5 |
| N=12 thr −99 (2.26×) | 13,033 | 497 | 14,553 | **−2,732** | **−1,537** | 5 |

At N=4 the 2.03× and 2.26× cells are identical (135 picks): nothing below the
composite's bottom quintile ever wins a top-4 slot. **3× the pool is impossible
— the whole candidate universe is 2.26× the B+ pool.**

### Per month, Q1-on, slot count across the top (P&L / picks)

| month | N=3 | N=4 | N=6 | N=8 | N=12 |
|---|---|---|---|---|---|
| 2025-01 | 228 / 4 | 565 / 5 | 565 / 7 | 565 / 7 | 388 / 8 |
| 2025-02 | 634 / 4 | 526 / 6 | 365 / 8 | 501 / 9 | 501 / 9 |
| 2025-03 | 82 / 2 | 82 / 2 | 82 / 2 | 567 / 5 | 504 / 6 |
| 2025-04 | 179 / 1 | 179 / 2 | 555 / 3 | 555 / 3 | 555 / 3 |
| 2025-05 | 895 / 2 | 895 / 2 | 1,295 / 4 | 1,347 / 6 | 1,509 / 9 |
| 2025-06 | −203 / 4 | 349 / 8 | 307 / 9 | 209 / 11 | 479 / 15 |
| 2025-07 | −13 / 2 | 81 / 4 | 154 / 5 | 54 / 6 | −104 / 9 |
| 2025-08 | 158 / 2 | 158 / 2 | 158 / 2 | 63 / 3 | 5 / 4 |
| 2025-09 | 524 / 9 | 470 / 11 | 416 / 13 | 436 / 16 | 1,561 / 18 |
| 2025-10 | 1,115 / 7 | 1,676 / 8 | 2,236 / 9 | 2,341 / 10 | 2,249 / 12 |
| 2025-11 | 64 / 3 | 76 / 5 | −134 / 9 | 50 / 14 | −411 / 18 |
| 2025-12 | −12 / 7 | −250 / 11 | −25 / 14 | −25 / 15 | −139 / 18 |
| 2026-01 | −223 / 3 | 1,070 / 5 | 808 / 9 | 750 / 12 | 823 / 14 |
| 2026-02 | 707 / 6 | 707 / 6 | 588 / 9 | 1,229 / 13 | 1,624 / 17 |
| 2026-03 | 2,038 / 3 | 2,361 / 4 | 4,040 / 8 | 4,040 / 9 | 3,840 / 11 |
| 2026-04 | 293 / 6 | 277 / 7 | 208 / 11 | 208 / 13 | 187 / 14 |
| 2026-05 | −133 / 1 | 210 / 3 | 210 / 3 | 159 / 6 | 159 / 6 |
| 2026-06 | 390 / 7 | 345 / 8 | 531 / 12 | 292 / 16 | 258 / 19 |
| 2026-07 | 419 / 7 | 392 / 9 | 266 / 14 | 589 / 20 | 1,359 / 25 |
| 2026-08 | 168 / 6 | 168 / 8 | 68 / 11 | 648 / 17 | 450 / 25 |
| 2026-09 | −122 / 2 | −122 / 2 | −148 / 3 | −148 / 4 | −232 / 7 |

Frequency is still tiny in absolute terms: even at 12 slots the book takes 267
picks in 21 months — **0.63 picks per trading day**, because the no-refill
vetoes empty most slots. The slot cap is almost never the binding constraint
(max 5 picks on the busiest day at N=12).

### Per-trade risk — the inert knob

| slots | mean realized risk / trade | cap binds |
|---|---|---|
| 3 | $142 | 100% of picks |
| 4 | $143 | 100% |
| 6 | $148 | 100% |
| 8 | $149 | 100% |
| 12 | $151 | 100% |

`sizing.risk_per_trade_usd: 375` never binds at the $10K stage. The live sizing
lever is `account_budget_usd / max_concurrent`.

---

## Caveats

1. **Task A is subtractive only.** It cannot price the trades B+ would have
   taken instead of the ones it refuses. n = 8 for the "B+ would have owned it"
   set; n = 2 for trades actually taken under B+.
2. **4 of 117 live trades have no features row**; their `range_size_pct` is not
   reconstructible from the dump, so the range-size veto fails open on them by
   construction (it blocks none of the 4 anyway).
3. News in Task A comes from the **backfill** CSVs, not from what the live
   engine saw at 9:31 (live recorded news on only 24 of 117 rows, because B+
   disables the PM mult). Unknown news fails open, as in production.
4. The BT book is a **relative tool at stage sizing, never a forecast** — the
   entered-inclusive rebuild removed the selection lookahead but the fill model
   is still a simulator, and the 3× column assumes linear scaling of a
   cap-bound position with zero impact.
5. `orb.yaml` was last written 2026-09-14; the weekly selection refit
   (`scripts/orb_weekly_refit.py`, Sunday cron) has not rewritten it since, so
   the frozen fit used here is the one live would trade today.
6. 2026-09 is a partial month and ORB is paused (`strategy.enabled: false`) —
   its picks are BT-only.
7. The N=8 Q1-off result rests on one 5-month split; the 2026-01..05 split
   contains the single largest month in every configuration (2026-03).

## Cell count

* Task A: **1** rule-set applied to **1** population, zero parameters fitted.
  Reported: 7 rules × 3 eras = 21, + 5 leave-one-out, + 5 solo-veto, + 14
  overlap combinations, + 5 rank thresholds, + 1 stop-dollar share, + 1
  exit-reason decomposition = **52 descriptive cells, 0 search cells.**
* Task B: **14 pipeline runs** (5 slot counts × 2 Q1 states + 4 pool cells),
  each read on 1 whole window + 3 splits = **56 config × split cells**, plus 8
  marginal-band rows, 3 participation multiples, and a 21 × 5 monthly display.
  The two pool thresholds were picked from a 12-row pool-SIZE table before any
  P&L was computed.
* **Stage total: 14 runs, 108 reported cells, 0 thresholds tuned.**

## Artefacts

```
research/fuckup_audit/D1_orb/
  REPORT.md                      this file
  taskA_retro_veto.py            Task A (shared helpers, orb.yaml values)
  taskA_rank.py                  B+ rank of each live trade
  taskA_live_retro_veto.csv      117 rows: features, veto flags, B+ rank
  run_base.sh / base_n3.log      the full bar-walk (reproduces the nightly book)
  candidates_dump.csv            13,033 resimmed candidates (shipped exit physics)
  run_grid.sh / grid.log         the 14 selector runs
  log_*.txt, book_*.csv, monthly_*.csv
  analyze_grid.py / taskB_summary.csv / taskB_monthly.csv
  pool_vs_threshold.py           pool size vs threshold (declared before P&L)
  capacity.py                    capital actually deployed + participation
```
