# PREREG — cells 1,684–1,687: ORB frequency and the HOD transplants — 62 ideas, 18 untested, run one at a time (FROZEN 2026-10-01 01:10 UTC)

Owner 10/1: "How can we increase ORB frequency and add some of the HOD stuff to make it more profitable? Look at all
the HOD stuff and build 30–50 ideas." Inventory of what exists: `research/orb_freq/INVENTORY_20261001.md` (46 levers).
ORB's edge is per fill (+0.105 R at the live config 2025-01..2026-09, n 473, ex-top-5 % ≈ 0 — the top 5 % of fills
carry the book); its constraint is frequency: ≈ 6 fills/week in regime, 1.4–2.3 out of regime. Frequency = gappers
that qualify × share that break the range × share that pass the gates. The gates were tested (removal 0/20 vs the
null) and the slots do not bind (8 slots, ~1.2 fills/day); the throttle is the ADMISSION rule (gap ≥ 5 % at 09:30,
$3–30, volume ≥ 500K). Every untested idea below widens admission with a mechanism or transplants a HOD lesson.

## The idea map (T = tested, verdict in the inventory; N = new, with its cell)
A. Admission by a different measure of "in play"
 1 N  gap + 5-min range ≥ 5 % by 09:35 (prior close → range high), admitted at the same decision point — a 3 % gapper
      that runs to +6 % in the first five minutes is the move the gap gate misses.                      → 1,684
 2 N  gap measured vs the prior day's HIGH, floor 3 % (a gap above yesterday's high is a true breakout gap). → 1,684
 3 N  pre-market dollar volume ≥ $2M by 09:30 with gap 2–5 % (participation instead of gap).           → 1,685
 4 N  the HOD scanner's relative-volume universe at 09:35 (cumulative volume ≥ 5 × the prior-20-day same-minute
      average) as an ORB pool for names that are NOT ≥ 5 % gappers — the HOD "stuff" transplanted as a pool. → 1,685
 5 T  gap 4–5 % $3–30 and gap 3–5 % $30–50 add-on pools — DRY-ONLY, p30 negative out of regime (cell 1,328).
 6 N  volume floor 300K instead of 500K with the spread gate as the quality guard.                        → 1,685
 7 N  price band $1–3 with the spread gate (prior: cost-dominated; R must exceed the spread).             → 1,685
 8 T  2× leveraged wrappers in the universe — SHIPPED 9/5.
 9 T  wrapper-news mapping — REFUTED, never.
B. Admission by a different population
 10 N gap-DOWN reversal ORB: names gapping ≤ −5 % that break their 5-min range HIGH (short covering).     → 1,684
 11 N day-2 continuation ORB: yesterday's ≥ 10 % gapper, today's 5-min range break, any gap today.        → 1,684
 12 N late-opener ORB: first print after 09:35 (halt or late open) → the range = the first 5 minutes after it. → 1,685
 13 N news-time ORB: a news item released 09:30–11:00 (the recorded news stream / Alpaca news archive) → the 5-min
      range after the first post-news bar, same gates.                                                     → 1,686
 14 T index ORB (QQQ/SPY) — 0/18 cells, closed.      15 T stocks-in-play replication — FAIL.
 16 T ORB short side — FAIL.                          17 T prior-day-high level (HOD cell 1,441) — FAIL.
 18 T HOD breaks on the gapper universe (cell 1,428) — FAIL.
C. Time and structure
 19 T multi-window ORB (later windows) — FAIL.        20 T 10/15-minute ranges — FAIL (F2-10/F2-15).
 21 T slot recycling / re-entry after a stop — FAIL.  22 T refill after a post-ranking veto — NEVER.
 23 T retest-bid entry — FAIL.                        24 T anchor dedup re-litigation — closed.
 25 N second-chance range: a name whose 5-min break stopped out may enter ONCE on the break of its 30-min range high
      at 10:00 — distinct structure, not a refill (prior: 21 says FAIL for recycling; this is one pre-declared read). → 1,687
D. Gates and selection (frequency × quality)
 26 T gate removal (every gate, 0/20 vs the null) — closed.   27 T spread gate below 150 bps — NEVER.
 28 T skip_q1 removal — never without the April doc.        29 T Q5 1.5× cap removal — NEVER.
 30 T catalyst veto — ON then OFF (the live book is veto-OFF).  31 T LLM catalyst filter — REFUTED.
 32 T weekly selection refit (26 w) — SHIPPED; adaptive_mults — NEVER refit.
 33 N earnings-gapper vs non-earnings split of the book (EDGAR desk on disk): if one cohort is negative, dropping it
      is a profitability move; if both positive, nothing changes.                                         → 1,687
E. Entry mechanics (HOD transplants)
 34 T pre-placement at 09:35:00 + 5 s (cell 1,655) — PAPER.   35 T target resting limit — PAPER (unexercised so far).
 36 T chase cap / post-then-cross / spread-in-R gate — tested (cells 214, 1,426: only 21 % of BT fills match the
      30-bps entry model; the tape edge at zero latency +0.028 R).
 37 N the measured ORB cost model: re-read the whole book at the cost the PAPER fills will measure (entry vs trigger,
      exit vs stop) — the HOD lesson that every verdict carried a 3× cost; runs after ≥ 30 paper fills.   → later
F. Exits (from the 58-rule HOD sweep; ORB's own 1,679/1,680 tested six)
 38 T 50 % at +1 R — +0.02 R, consistency up, strong weeks down (1,680) — watch on paper.
 39 T locks/trails/time stop (1,679) — negative.   40 T HOD-trained exit models on ORB — negative on real ledgers.
 41 N no exit at 2 R, half at +3 R, trail the rest MFE − 1 R (the one robust HOD exit, +0.11 R on the sealed quarter)
      on the ORB book with real times.                                                                    → 1,687
 42 N power-hour hold: from 15:00 ET replace the target by the 15:45 close for fills ≥ +1 R.              → 1,687
 43 N add one unit at +1 R with the original stop (ORB-specific read; negative on HOD).                   → 1,687
G. Sizing and cost (HOD's floor lesson)
 44 N range floor: R ≥ 0.75 % of price, else size down to keep cost ≤ 0.1 R (keeps frequency, caps the cost tax). → 1,687
 45 T breakout thermometer sizing — FAIL.   46 T regime sizing (Feb 2026 counterexample) — rejected.
 47 T ramp: advance only above water, 40 live fills — rule (no research).
H. Portfolio
 48 N the union book: production + every passing pool, with the shared tail (a gap-down morning hits every pool) and
      the cadence bar on the union.                                                                       → synthesis
 49 T BF add-to-winners — parked.   50 T ignition, red-to-green, MACD — closed.
I. Machinery transplants that are engineering, not cells (do, no read needed)
 51 N arm-time telemetry per ORB candidate (the HOD ledger columns) — the forward read of every pool for free.
 52 N the HOD fill-persistence and account-stamped state (ported 9/30).    53 N no per-trade Telegram (ported 10/1).
 54 N the 12-fills/day cap logic as ORB's slot policy if frequency ever binds.
J. Already closed on HOD, do not transplant
 55–62 T relative volume, candle shapes, patterns, failure cuts, adds after +R with locks, model take-profits,
      the failure short, pre-entry classifiers — all closed on HOD at the measured cost (cells 1,660–1,683).

## Amendment 1 (owner 10/1 ~08:00 UTC, before 1,685 runs): SUB-POOLS, each with its own rules
"We do not need a single rule against all of them; create sub-pools — 4 % + relative volume, 3 % + $5M, VWAP, close
to the 52-week high — each with its own rules." Cell 1,685 becomes the sub-pool grid:
* Admission = gap band × ONE feature (18 sub-pools): gap bands {2–3 %, 3–4 %, 4–5 %} (names NOT in the production
  ≥ 5 % universe) × features {F1 relative volume at 09:35 ≥ 3× (cumulative volume to 09:35 ÷ ADV20 × the TRAIN
  cross-sectional 09:35 profile — the 1,665 definition B), F2 pre-market dollar volume ≥ $5M, F3 price above the
  day's VWAP at 09:35 AND the 5-min range closes in its top half, F4 within 5 % of the 52-week high (daily bars),
  F5 prior-day range ≥ 1.5 × ATR14 (a volatile setup), F6 day-2 of a ≥ 10 % gapper}. Price $3–30, volume ≥ 300K,
  the spread gate and the never-rules unchanged.
* Each sub-pool gets its OWN selection chain (the per-pool 26-week refit as in cell 1,328) and its OWN exit, chosen
  on TRAIN only from the fixed menu {the live rule; no exit at 2 R → half at +3 R → trail MFE − 1 R; 50 % at +1 R}
  by TRAIN mean R, then read on VAL and out of regime with that exit frozen.
* Sub-pools 19–20 (owner 10/1 ~08:10 UTC: "5 % + 2M is dropping stuff because 2M is not relative to the stock's
  regular volume"): the production floor is `min_prev_volume: 500000` — YESTERDAY's volume, absolute. Recovery
  sub-pools: gap ≥ 5 %, $3–30, prior-day volume between 100K and 500K (the names the floor drops) × {F1 relative
  volume at 09:35 ≥ 3×, F2 pre-market dollar volume ≥ $5M}. Also reported: how many ≥ 5 % gappers per day the
  500K floor removes, and the production book's own fills split by prior-day volume tercile (does the absolute floor
  track quality at all?).
* Stage 2 (pre-declared): pairs of features (gap band × F_i × F_j) ONLY for sub-pools that passed stage 1, read the
  same way. No stage 3.
* The bar per sub-pool is unchanged; the union adds every passing sub-pool; multiplicity stated: 18 + ≤ 15 pair reads
  × 3 windows. Overlap between sub-pools is reported and de-duplicated in the union (a name-day counts once).

## Method (every N cell)
Point-in-time universes from `data/cache.db daily_bars` and Databento EQUS.SUMMARY (delisted included), minute bars
from the completed store (append missing symbol-days through the designed appender only), the production pipeline
`study_orb_pipeline_static_lock.py` with the LIVE config (veto OFF, 8 slots, spread gate, Q1 skip, 15:45 close), the
per-pool selection chain as in cell 1,328, costs as the pipeline charges them today (noted as pre-measurement).
Windows: in-regime 2025-01..2026-09 (halves by year) AND out of regime 2023-01..2024-12. Reads per pool: n, fills/week,
mean R, iid and day-clustered t, ex-top-5 %, MDE, weekly P10, worst week; then the UNION with the production book:
frequency gain, union mean R, union cadence bar (strong week ≥ +5 R, median gap, P90 gap, green weeks vs the null),
the shared tail (worst day with both on). Exit/sizing cells (1,687) run on the production book's real-time ledger
(1,679's walker, R-floored) with the same reads.

## Pass bar (per pool) and the union rule
A pool is ADDED to the union only if its own mean R ≥ +0.05 with day-clustered t ≥ 2.0 in regime, ≥ 0 out of regime
(not p30-negative), ex-top-5 % > 0, and the union's weekly P10 and strong-week gap are not worse than production's.
An exit/sizing change ships only under 1,679's bar (both years + same sign live). Any pass → independent rebuild
from prose → paper as one mechanics change. Programme count: ORB line + 18 cells.

## Run order (one Sonnet agent per cell, sequential; Fable judges each before the next)
1,684 daily-bar pools (ideas 1, 2, 10, 11) → 1,685 minute-bar pools (3, 4, 6, 7, 12) → 1,687 exits/sizing/split on
the real-time book (25, 33, 41, 42, 43, 44) → 1,686 news-time pool (13; needs the news archive) → union synthesis.

## Not allowed
Any gate change from the never-list; refilling slots; reading a pool's number before its out-of-regime window is
built; dropping the R floor; pooled-only numbers.

## Output
`research/orb_freq/RESULT_1684.md` … `RESULT_1687.md`, `UNION_1688.md`; per cell `*_pool_book.csv`, `*_reads.csv`,
the scripts and logs. Each agent returns ≤ 150 words.
