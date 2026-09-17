# Stage G — the SHORT side (PLAN §3 H12 / §4 row G)

Written 2026-09-17. **RESEARCH ONLY** — the owner's rule is that shorts go live only after a long
book works, and nothing in this stage touches a config, a service, an order or a live file.

Part I is `G/PREREG.md` reproduced verbatim and unmodified (it was written before any G script ran;
the only edits it ever received were made BEFORE the first run and are marked in it as such).
Part II is what the run says.

---

# PART I — PRE-REGISTRATION (verbatim)

# Stage G — PRE-REGISTRATION (the SHORT side)

Written **2026-09-17, before any G script was run**. Nothing below this heading was informed by a
Stage-G number. `research/fuckup_audit/PLAN.md` §3 H12 / §4 row G.

**RESEARCH ONLY.** The owner's standing rule is that shorts go live only after a long book works. No
G artefact touches a config, a service, an order or a live file. Everything is written under
`research/fuckup_audit/G/`; every other path is read-only (`data/cache.db` via `file:...?mode=ro`).

---

## 0. Why this stage exists

Two long-side facts from this program point short and have never been traded as shorts:

- `research/lit_review_2026/daily_queue.md` (M16): gap-ups fade. `+10..+20%` gaps average
  **−166 / −91 / −152 bps** open→close in 2025 by dollar-volume band, `>+20%` **−246 / −355 / −341**;
  **0 of 27 gap × $-volume cells have mean open→close ≥ +50 bps in both years**.
- `research/lit_review_2026/open_fade.md` (M18): prior-day attention names lose **−60 / −26 / −20 bps**
  open→10:30 on TRAIN / VAL / TEST (attention minus control −46.1 bps, t −4.92 on TRAIN).

Both are *daily-bar* measurements of a *drift*. Neither has ever been run through this program's
honest machinery: a 1-minute entry rule, an obtainable fill, a stop, the corrected cost contract, a
12/day 4-concurrent book, and PLAN §1's gate. That is what Stage G does, on the SHORT side, with the
long-side family definitions mirrored.

---

## 1. Universe

Base = **U1 ∪ U2** of Stage E (`E/members.csv` rows with `u1 | u2`; `E/u1_keys.csv ∪ E/u2_keys.csv`),
i.e. causal at 09:30: U1 `open_t/close_{t−1} − 1 ≥ +3%`, U2 `(high_{t−1}−low_{t−1})/low_{t−1} ≥ 8%`,
both with `open ≥ $5`, `adv20 ≥ 100K` shares. 198,318 symbol-days, 410 days, bars already on disk
(`E/bars_causal/day=*/bars.parquet` ∪ `research/bf_zero/bars_sip.db`, one tape: Alpaca SIP raw).

**Borrowable-looking restriction** (a short needs shares to borrow; this program does not model
locate availability, so it stays in the band where hard-to-borrow is rare):

| rule | field | source | when known |
|---|---|---|---|
| 20-day median dollar volume ≥ **$10M** | `dvol20_med` | Databento daily panel, `(close×volume).shift(1).rolling(20, min_periods=10).median()` | 09:30 of day t |
| price ≥ **$10** | `open` ≥ 10 at 09:30 as the universe pre-filter; **and** the scored `entry` fill ≥ 10 at scoring | panel open / the minute tape | 09:30 / signal |
| **common stock** | `asset_class == 'stock'` | `trading/orb_asset_class.classify_asset` + the 2026-07-11 offline map (`build_candidates4.asset_class`) | symbol-level |

Declared limitation: `asset_class` is a **symbol-level, not point-in-time** attribute — a name that
became a leveraged wrapper later is tagged `wrapper` for the whole window. Wrappers are excluded, so
the error direction is "a few stock-days dropped", never "a wrapper shorted".

**S5's population (written before any run).** S5 is defined by a cross-sectional daily ranking, not by
a pattern, so its natural population is the attention list itself. Restricting it to U1∪U2 would add
a gap/prior-range filter the M18 result never had. S5 therefore runs on **the attention list ∩ the
borrowable filter**, with bars from the union of the three stores of the SAME tape (Alpaca SIP,
`adjustment=raw`, 1-minute): `E/bars_causal/` ∪ `research/bf_zero/bars_sip.db` ∪
`research/lit_review_2026/attention.db` (the M18 fetch — same feed, same adjustment, RTH only, which
is all S5 needs). The overlap of the attention list with U1∪U2 is reported, and **S5 restricted to
U1∪U2 is reported as a variant**. S1–S4 use U1∪U2 only.

**Two universes are scored (the 2 in the 24-cell count):**

- **UA** — the borrowable U1∪U2 as above, **no range floor**.
- **UB** — UA restricted to `range_so_far_pct ≥ 5` (the day's high−low over bars **strictly before**
  the signal bar, as % of the 09:30 open). Declared sensitivity, because Stage E §8.4 found this
  floor is the single most effective causal filter on the long side and the rows it removes are the
  losing rows; whether that holds with the sign flipped is a question, not an assumption.

---

## 2. Families — all SHORT, each mirrored from a long definition

Every field is computed on bars at or before the signal bar. `l`, `h`, `c`, `o` are the RTH 1-minute
arrays (09:30 ≤ m < 16:00), index 0 = the 09:30 bar. A short's `stop` is **above** the entry; a
candidate with `stop <= level` is rejected. The long-side functions in
`research/fuckup_audit/B/build_candidates4.py` and `research/bf_zero/build_candidates.py` are
**imported** for everything shared (bars loader, cost curve, asset class, the day loop); the short
detectors and the short walk are written as mirrors, not copies of the long bodies.

| id | mirror of | rule | level | stop |
|---|---|---|---|---|
| **S1** gap-fade at the open | F8 N=5 (the ORB entry) | require `gap_pct ≥ +5.0` (09:30 minute-tape open vs `prev_close`); signal = the first bar `i ≥ 5` with `low[i] ≤ level` | `min(low[0:5])` (the 09:30–09:34 range low) | `max(high[0:5])` (the range high) |
| **S2a** ORB breakdown N=15 | F8 N=15 | signal = first `i ≥ 15` with `low[i] ≤ level`, no gap requirement | `min(low[0:15])` | `max(high[0:15])` |
| **S2b** ORB breakdown N=30 | F8 N=30 | as S2a with N=30 | `min(low[0:30])` | `max(high[0:30])` |
| **S3** green-to-red | F6 red-to-green | require `prev_close` known and `open[0] > prev_close`; signal = first `i ≥ 1` with `low[i] ≤ level` | `prev_close × 0.997` | `max(high[0:i])` — the highest high strictly before the signal bar |
| **S4** HOD-rejection | F5 HOD-break, entered the other way (H9's F13 mechanism as a SHORT) | with K=5, X=0.04: bar `j=i−1` must satisfy `rollmin(low,K)[j] ≥ runmax(high)[j]×(1−X)` and `rollmin(low,K)[j] < runmax(high)[j]` (F5's own consolidation test); signal = the first bar `i > K` whose **CLOSE** is below `rollmin(low,K)[i−1]` | `rollmin(low,K)[i−1]` (the consolidation low) | `runmax(high)[i−1]` (the HOD) |
| **S5** first-hour attention fade | `lit_review_2026/test_open_fade.py` M18 | prior-day top-20 attention name (definition below); **short at the 09:35 bar's OPEN regardless of pattern** | none (entry is the 09:35 open) | `max(high[0:5])` — declared addition, see below |

**S5's attention list, reproduced exactly from the M18 script** (`test_open_fade.py`), from
`research/lit_review_2026/daily_panel.parquet`: on day t−1 keep rows with `close ≥ 5`,
`dvol20 ≥ $2M`, `|ret_cc| ≤ 0.5`, `ret_cc` and `vol_ratio` present and a symbol matching
`^[A-Z]{1,5}(\.[A-Z]{1,2})?$`; score `attn = |ret_cc| × vol_ratio`; rank descending per day
(`method='first'`); **top 20**; trade on the next trading day t. The G borrowable filter (§1) is then
applied on top. The M18 *control* group (ranks 100–120) is **not** traded here; it is reported as the
same-machinery reference book so "the attention names specifically" is separable from "shorting
anything at 09:35".

**S5's stop is a declared addition, not the paper's spec.** The M18 result is a drift measurement
with no stop; the cost contract, the book and the gate of this program are all in R units, which
needs an R. S5 therefore carries a touch stop at `max(high[0:5])` (the same structural level S1
uses) and is gated on that basis. The **pure M18 spec — no stop, cover at 10:30 / at the close — is
reported as a sensitivity in bps net of the same cost contract**, outside the gate, because a
stopless position has no R and cannot be booked in R.

---

## 3. Fill, exits, obtainability

**Entry fill — ONE model, the engine's**: `entry = open[i+1]`, accepted only if
`open[i+1] ≥ level × (1 − 0.006)` — the mirror of the long side's "no chase above the cap" is **no
chase below**. A signal whose next open gaps through the cap is a NON-fill and is recorded as such
(the Stage-B lesson: a table that only holds cheap fills cannot answer a fill question).
The resting stop-limit fill is **not** built: Stage C rejected it on the paired test
(−0.058 R net, t −61, on the 89,826 signals both models fill) and it is not re-litigated here.
S5's entry is the 09:35 bar's open with no cap (there is no level).

**Exits.** Walk starts the bar AFTER the fill bar (fill bar = i+1, so the walk starts at i+2).

| exit | rule | fill |
|---|---|---|
| **hold** (PRIMARY) | hold to 15:55; stop only | eod: that bar's open |
| **2R** (secondary) | as hold plus a target: the first bar whose **CLOSE ≤ entry − 2R** | at the target price |
| stop (both) | **touch**: the first bar with `high[k] ≥ stop` | `max(stop, open[k]) × 1.001` |
| S5 only | **1030** (PRIMARY for S5): cover at the 10:30 bar's open; **eod** (secondary): 15:55 | that bar's open |

`R = stop − entry` (positive). `rr = (entry − exit_px) / R`. Within-bar priority, mirroring
`build_candidates4.walk_2r` exactly: **eod beats stop beats target**.

**Obtainability (PLAN §1).** Every simulated fill must satisfy `low ≤ fill ≤ high` of the bar that
fills it, and be reachable by an order the engine would have had resting or would have sent on the
bar's close. The share of fills failing the bar-containment check is reported per family; a family
below 100% is reported, never silently repaired.

---

## 4. Costs — contract (c) mirrored, plus borrow

`half = 0.5 × (spread_cc_bps / 100) / max(r_pct, 0.05)` in R units, with `spread_cc_bps` the
`cost_curve.csv` median NBBO spread for the (price band × time-of-day band) of the signal
(`build_candidates4.cc_bps`).

| leg | multiple of `half` |
|---|---|
| entry (next-open fill) | **0.25** |
| exit = stop | 0.875 |
| exit = eod / timed cover | 0.412 |
| exit = target | 0.875 |

**Borrow. PRIMARY = 0**, declared: the G universe is ≥ $10 and ≥ $10M a day in common stock, the
easy-to-borrow band, and these are intraday positions (no overnight carry). **SENSITIVITY**, reported
for every gate cell: **5 bps locate + 1% annualised / 365 for the one day held (0.274 bps) = 5.274 bps
of notional**, charged as `(5.274 / 100) / r_pct` R. *Borrow availability itself is NOT modelled* —
see §8.

---

## 5. Short-sale restriction (SSR)

Reg SHO Rule 201: a −10% intraday decline from the prior close triggers the restriction for the rest
of that day **and all of the next day**; while it is on, a short sale may only execute on an uptick
(above the current NBB). This program cannot model an uptick requirement at 1-minute resolution.

**Definition.** Day t is an SSR day for symbol s iff `low_{t−1} ≤ 0.90 × close_{t−2}` (the trigger
fired on t−1), computed from the Databento daily panel with the same causal shift convention as
`E/universes.py`. A second marker, `prev_ret_cc ≤ −10%` (the prior day's close-to-close), is computed
and reported alongside it.

**PRIMARY: SSR days are EXCLUDED — no short entry.** This is the conservative choice (an uptick rule
can only make a fill worse or absent). **SENSITIVITY: the same cells with SSR days included.** The
count and share of excluded signals is reported per family and per split.

---

## 6. Population, book, splits, gate

- Population: `entry ≥ $10`, `entry_m ≤ 841` (14:01), `r_pct ≥ 1.0` of the scored variant, plus the
  universe of the cell (UA or UB). S5 is 09:35 by construction.
- Book: `trading.hod_break.run_book(rows, 12, 4)` — 12 a day, 4 concurrent, first-come, alphabetical
  tie-break, causal slot freeing (identical object to every other stage).
- Splits: TRAIN 2025-01-02..2025-12-31 · VAL 2026-01-01..2026-05-31 · TEST 2026-06-01..2026-09-11.
- **Gate (PLAN §1, verbatim)**: G1 TRAIN mean net R > 0, t ≥ 2.0, ≥ 5 trades/week. G2 VAL mean net
  R > 0, t ≥ 1.0, ≥ 55% of weeks green, bar raised by 1 SE of weekly R per 10 cells passing G1.
  G3 **TEST is read ONCE, for G2 survivors only**, after the selection is frozen in writing.
- Every G2 survivor additionally gets: tail removal (top 1% / 5%), a +3R winner cap, a per-month
  table, a day-level sign-flip permutation p **search-adjusted over all G cells**, the per-trade CSV,
  the prose rule for an independent rebuild, and the live-constraint section.

---

## 7. Cells — declared before the run

**24 gate cells** = 6 family-configs (S1, S2a, S2b, S3, S4, S5) × 2 exits (S1–S4: `hold`, `2R`;
S5: `1030`, `eod`) × 2 universes (UA, UB).

Reported but **not** gated, and counted in the REPORT's multiplicity total:

| block | count |
|---|---|
| time bands 09:30–10:00 / 10:00–11:00 / 11:00+ on the 24 | 72 |
| SSR-included re-score of the 24 | 24 |
| borrow-charged re-score of the 24 | 24 |
| S5 pure-M18 no-stop bps sensitivity (10:30 and close, attention and control) | 4 |
| S5 control-group (ranks 100–120) book, both exits × 2 universes | 4 |
| S5 restricted to U1∪U2 (the strict-universe variant), both exits × 2 universes | 4 |
| **declared total** | **156** |

Anything looked at beyond this list is counted and named in the REPORT's cell count.

---

## 8. What Stage G cannot say, declared in advance

1. **Borrow availability is NOT modelled.** No locate feed, no borrow-rate history, no fail-to-deliver
   or threshold-list data is on this node. Every number assumes shares were available at the size the
   book implies. The $10 / $10M / common-stock band makes that plausible and not certain; a
   gap-up small cap is exactly the name that goes hard-to-borrow on the day it gaps. Any positive
   result must be read as an upper bound and would need a broker locate audit before belief.
2. **The uptick rule is not modelled** — SSR days are excluded outright (§5).
3. **`asset_class` is symbol-level, not point-in-time** (§1).
4. Short interest, days-to-cover, options-market and ETF-creation channels are not in the data.
5. The reversed-tape gate of PLAN §3 H4 does not apply (no model is fit in G); the tail tests and the
   permutation p do apply.
6. Phrasing rule (PLAN §1): a null here means "no edge detectable in THIS universe, at THIS horizon,
   at THIS book size, over THIS window, at THIS cost", reported with the smallest effect the test
   could have seen.

---

## 9. Decision rule (the only thing that promotes a cell)

A cell becomes a Stage-G candidate **only if** it clears G1 on TRAIN **and** G2 on VAL under the
PRIMARY contract (borrow 0, SSR excluded, next-open fill, touch stop). For such a cell and only such
a cell, TEST is then read once and reported whatever it says, together with the tail tests, the +3R
cap, the per-month table and the permutation p. A cell that clears the gate only with borrow 0 but
dies under the 5.274 bps borrow sensitivity is reported as a failure. A cell that needs SSR days
included is reported as a failure.

---

# PART II — WHAT THE RUN SAYS

## 1. One page

**The short side of the first hour does not pay net of costs on this universe, at this book size,
over this window.** 0 of the 24 pre-registered cells clear G1; therefore 0 clear G2; therefore
**TEST was not read**. The best TRAIN cell is `S5 UB 1030` at **+0.0444 R** (t 0.22) and it trades
2.6 times a week, so it fails the ≥ 5 trades/week bar on its own; the best cell that meets the
frequency bar is `S1 UB hold` at **+0.0261 R** (t 0.65, 9.4 tr/wk; VAL +0.0235, t 0.37, 64% of weeks
green). A 500-draw day-label sign-flip null over all 24 cells puts the **observed max TRAIN t at
0.65 against a null 95th percentile of 3.53 — p = 1.000**. The grid contains no signal of the size
it can detect.

**Which family, if forced to name one:** `S1` — the gap-fade (gap ≥ +5%, break of the 09:30–09:34
range low) — on the range-floor universe UB, held to 15:55. It is the only family positive on TRAIN *and* VAL under the
primary contract that also trades often enough to be gated (`S5 UB 1030` is positive on both at 2.6
trades/week), it keeps its sign under the borrow charge (+0.0172 / +0.0148)
and under the fund-name exclusion (+0.0261 / +0.0333), and it survives the +3R winner cap
(+0.0191). It dies with the top 5% of trades removed (−0.083). At 9.4 trades/week and +0.026 R it is
**0.24 R/week at 4 slots ≈ $24/week at $100 risk**, against a per-trade MDE of 0.112 R (0.9–3.3 R/week across the gated cells) — i.e. the
number is indistinguishable from zero by its own test.

**Gross, the short side is roughly flat, and the cost contract is the whole deficit.** Gross mean R
on the booked TRAIN trades runs **+0.108 (S5 UB) / +0.057 (S1 UB) / +0.032 (S4 UB) / +0.019 (S3 UB)
/ +0.005 (S2 N15 UB) / −0.007 (S2 N30 UB)**; the cost contract charges **0.052–0.120 R** per trade at
a median cost-curve spread of 39–52 bps against a median R of 1.9–3.7% of price. This is the same
shape Stages C and E found on the long side — near-zero gross, negative net — with the sign flipped.

**How much is SSR-blocked**: the Rule 201 exclusion removes **6.7% (S1) to 14.8% (S3)** of the
pattern families' signals and **41.0% of S5's** — the attention names are, by construction, names
that just moved 10%+, so two in five of them are legally uptick-only on the day the book wants to
short them. Including them changes no verdict (0 of 24 clear G1 either way).

**How much is borrow-dependent**: the universe rule (price ≥ $10, 20-day median dollar volume ≥
$10M, common stock) throws away **64.3% of U1∪U2** before anything is measured — of which the
common-stock rule alone accounts for 27,361 symbol-days (27.9% of what survives the price and
liquidity gates). The explicit borrow charge (5 bps locate + 1%/365) costs **≈ 0.018 R** at the
median R and moves no cell across a gate. **Borrow AVAILABILITY is not modelled at all** (§8 of the
pre-registration) — everything above assumes the shares were there.

**And the one substantive finding about the literature that motivated this stage:** the M18 open
fade **does not survive into the borrowable band at a 09:35 entry**. Measured on exactly M18's spec
(no stop, no R filter) over the attention names that pass the borrowable rule, the 09:35→10:30
return of the SHORT is **−4.6 bps on TRAIN (i.e. the stock ROSE) and +32.9 on VAL**, against a
control group at −5.8 / −3.7 and an MDE of 38 / 53 bps. M18's TRAIN fade lives (a) in the first five
minutes — its own table puts −37 of the −69 bps at $10–50M before 09:35 — and (b) in the $2–10M
dollar-volume band, which the borrow rule deletes. What is left for a 09:35 short in nameable,
borrowable stock is inside the noise.

**Nothing in Stage G is a ship candidate, and nothing here changes a live setting.**

## 2. Verification, before any number

| check | result |
|---|---|
| universe | `G/g0_universe.py` → `members_g.csv` **72,793 symbol-days / 410 days / 2,690 symbols** (TRAIN 36,034 / VAL 21,782 / TEST 14,977), `EXIT=0` |
| provenance | E/members.csv vs `lit_review_2026/daily_panel.parquet` on the 70,713 shared keys (both derived from the SAME Databento parquet): `open` median abs diff **0.000002%**, p99 0.000005%, max 0.0005%; `prev_close` the same — **no price-scale break** (PLAN §1) |
| build | `build_short.log` ends `DONE rows 187,773` + **`EXIT=0`**; `candidates_short.csv` has **187,774 lines = 1 header + 187,773** exactly; `build_short_state.json` holds **410** days, 2025-01-17..2026-09-04 |
| header | **60 columns, byte-identical and in order to the builder's `COLS`** |
| tape coverage | **1 symbol-day of 72,793 has no usable tape** (VSECU 2026-03-12) = **0.0014%**; `G/coverage_short_missing.csv` |
| independent rebuild | `G/verify_rows.py` — a second implementation with its own bar loader, plain-Python loops and no shared code — **42 of 42 rows MATCH** across all six family-configs and four seeds (18 on the 3-day smoke, 24 on the finished file), on `sig_m`, `level`, `stop`, `entry`, `entry_m`, `r_pct`, `rr_hold`, `why_hold`, `exit_m_hold`, `rr_2r`, `why_2r`, `rr_1030` |
| a bug found and fixed mid-stage | `na_values=['']` turns an empty `attn_grp` into NaN and `not NaN` is False, so the S5 branch was emitted for every member (245,187 rows instead of 187,773). The scorer filtered on `attn_grp == 'attention'` and was never affected: **the 24-cell table is byte-identical before and after the fix** (`score_short_tables.prebugfix.md` kept for the diff). Fixed at source and the file rebuilt. |

## 3. Obtainability and availability audit (PLAN §1 standing rule)

| family | entry fill inside its bar | exit fills inside the bar | outside, WORSE for the short (conservative) | **outside, FAVOURABLE (the only defect)** | queue-OK | spread known |
|---|---|---|---|---|---|---|
| S1 | 100.00% | 86.8 / 85.9% | 13.2 / 14.1% | **0.00%** | 96.7% | 100.00% |
| S2 N=15 | 100.00% | 85.7 / 84.8% | 14.3 / 15.2% | **0.00%** | 91.2% | 100.00% |
| S2 N=30 | 100.00% | 89.9 / 89.4% | 10.1 / 10.6% | **0.00%** | 92.6% | 100.00% |
| S3 | 100.00% | 86.2 / 85.6% | 13.8 / 14.4% | **0.00%** | 90.7% | 100.00% |
| S4 | 100.00% | 75.9 / 74.1% | 24.1 / 25.9% | **0.00%** | 87.6% | 100.00% |
| S5 | 100.00% | 86.2 (10:30) / 78.3 (eod) | 13.8 / 21.7% | **0.00%** | 93.5% | 100.00% |

Every entry fill is a price inside the bar that fills it. The exit fills that fall outside their bar
are the 0.1% stop slip and the cover-at-target, and **in every family, every exit and every split
they fall on the side that costs the short** — a cover above the bar's high. Not one fill in
142,561 scoreable rows is on the favourable-impossible side. No feature used in a decision has
partial coverage: `spread_cc_bps` 100%, `ssr` 100% (0 rows with an unknown t−2 close in the whole G
key set), `prev_day_range_pct`/`adv20`/`gap_pct` from the same causal panel. `pm_dollar_vol` and the
news columns are **not used in Stage G at all**, so D1's leak has no surface here.

`queue_ok` (the signal bar's volume ≥ 5× the shares a $100-risk position needs) holds for 87.6–96.7%
of rows; it is reported, not applied — at $100 risk and $10+ prices the position is a few hundred
shares.

## 4. The 24 pre-registered cells — PRIMARY contract (borrow 0, SSR excluded, next-open fill, touch stop)

Full table `G/score_short_tables.md`; per-cell CSV `G/score_short_results.csv`. Ranked by TRAIN mean
net R (the top and bottom; `grossR` beside `meanR`, MDE = the smallest per-trade effect the cell's
own TRAIN book could detect at 80% power):

| key | uni | exit | TRAIN n | tr/wk | TRAIN net R | TRAIN gross | t | MDE | WR | stop% | wk R | green | VAL n | VAL net R | VAL t | ex5 | cap3 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S5 | UB | 1030 | 134 | 2.6 | **+0.0444** | +0.108 | 0.22 | 0.553 | 39.6 | 36.6 | 0.12 | 0.29 | 98 | +0.245 | 1.36 | −0.294 | −0.158 |
| S1 | UB | hold | 478 | 9.4 | **+0.0261** | +0.057 | 0.65 | 0.112 | 50.6 | 21.8 | 0.24 | 0.49 | 258 | +0.024 | 0.37 | −0.083 | +0.019 |
| S1 | UB | 2R | 484 | 9.5 | +0.0231 | +0.055 | 0.61 | 0.106 | 50.6 | 21.5 | 0.22 | 0.49 | 260 | +0.008 | 0.15 | −0.077 | +0.023 |
| S3 | UB | hold | 660 | 12.9 | −0.0036 | +0.019 | −0.14 | 0.072 | 49.7 | 11.4 | −0.05 | 0.53 | 363 | +0.002 | 0.04 | −0.092 | −0.005 |
| S2 N15 | UB | hold | 968 | 19.0 | −0.0214 | +0.005 | −0.89 | 0.067 | 47.8 | 17.3 | −0.41 | 0.41 | 463 | −0.091 | −2.69 | −0.115 | −0.024 |
| S2 N30 | UB | hold | 960 | 18.8 | −0.0296 | −0.007 | −1.43 | 0.058 | 47.8 | 12.4 | −0.56 | 0.41 | 448 | −0.107 | −3.46 | −0.109 | −0.031 |
| S1 | UA | hold | 903 | 17.7 | −0.0546 | +0.016 | −1.56 | 0.098 | 45.1 | 31.5 | −0.97 | 0.39 | 438 | −0.170 | −3.62 | −0.197 | −0.067 |
| S5 | UA | 1030 | 519 | 10.2 | −0.0639 | +0.056 | −0.89 | 0.201 | 41.8 | 39.7 | −0.65 | 0.35 | 241 | +0.018 | 0.19 | −0.288 | −0.142 |
| S4 | UB | hold | 1439 | 28.2 | −0.0659 | +0.032 | −1.56 | 0.119 | 36.5 | 53.3 | −1.86 | 0.35 | 704 | −0.221 | −4.07 | −0.308 | −0.146 |
| S4 | UA | 2R | 1515 | 29.7 | −0.1326 | −0.013 | −4.33 | 0.086 | 41.9 | 45.8 | −3.94 | 0.31 | 697 | −0.157 | −3.45 | −0.239 | −0.133 |
| S3 | UA | 2R | 1578 | 30.9 | −0.1545 | −0.047 | −5.54 | 0.078 | 40.7 | 39.5 | −4.78 | 0.24 | 778 | −0.131 | −3.22 | −0.262 | −0.155 |
| S3 | UA | hold | 1393 | 27.3 | −0.1574 | −0.053 | −4.03 | 0.109 | 36.5 | 44.4 | −4.30 | 0.24 | 666 | −0.068 | −1.02 | −0.391 | −0.225 |

**G1 (TRAIN mean net R > 0, t ≥ 2.0, ≥ 5 trades/week): 0 of 24. G2: 0. TEST: not read.**

Three structural readings of the full table:

1. **The range floor helps the short side too.** UB (`range_so_far_pct ≥ 5` on bars strictly before
   the signal) beats UA in **12 of 12** family × exit pairs, by +0.05 to +0.15 R per trade. Stage E
   found the same thing on the long side and concluded the floor is a filter, not a bias; the short
   side reproduces it with the sign flipped. Whatever the floor is selecting for, it is not
   direction.
2. **Hold-to-close beats the −2R target in 8 of the 10 S1–S4 pairs** (the two exceptions are S1 UA
   and S3 UA, both deeply negative either way), as on the long side (Stage C, Stage E). For S5 the
   pair is a horizon, not a target: the 10:30 cover beats holding to 15:55 on both universes.
3. **S4 (HOD-rejection) and S3 (green-to-red) are the destructive entries**, −0.10 to −0.16 R with
   stop rates of 39–53% — the mirror of Stage C's finding that F13 sweep-and-reclaim and F12 retest
   were the worst families ever scored there. A consolidation break, taken in either direction, is
   where the stops are.

## 5. Time bands (reported, not gated)

Full table in `G/score_short_tables.md`. The pattern that survives both splits: on UB, **S1 is
positive in 09:30–10:00 (+0.038 TRAIN / +0.024 VAL) and 11:00+ (+0.079 / +0.026) and negative in
10:00–11:00**; S3 UB is positive early (+0.017 / +0.034) and negative after 11:00; S2 and S4 are
negative in every band on VAL. There is no band in which the short side is reliably positive across
families, and the 09:30–10:00 window — the one the gap and attention literature points at — is not
systematically the best one. (Note the multiplicity: 72 band cells, all of them selected maxima.)

## 6. The declared sensitivities

| sensitivity | what it does to the verdict |
|---|---|
| **SSR days INCLUDED** (the uptick rule un-modelled) | mixed by family (S1 UB hold +0.026 → +0.011, S2 N15 UB −0.021 → −0.003); **0 of 24 clear G1** either way. The SSR days are not systematically the good ones or the bad ones. |
| **Borrow charged 5.274 bps** (5 bps locate + 1%/365 for the day) | costs ≈ **0.018 R** at the median R; S1 UB hold +0.0261 → +0.0172, S5 UB 1030 +0.0444 → +0.0294. **0 of 24 clear G1.** |
| **S5 on the CONTROL group** (attention ranks 101–120) | UA: −0.237 TRAIN / −0.185 VAL vs attention's −0.064 / +0.018 → **attention minus control = +0.173 R TRAIN, +0.204 VAL**. The attention names DO fade relative to ordinary names, exactly as M18 says; the level is still negative and the book still fails. |
| **S5 restricted to U1∪U2** (the strict-universe variant) | 83.0% of the borrowable attention names are already inside U1∪U2, and the restricted cells move by ≤ 0.01 R. The universe choice is not load-bearing for S5. |
| **fund/ETF names removed** (post-hoc, §8) | 0.5% of rows; **G1 still 0 of 24**. |

## 7. The pure M18 spec, corrected

`score_short.py` first reported the no-stop M18 sensitivity on the SCOREABLE population, which
carries `r_pct ≥ 1.0` — for S5 that means "the 09:30–09:34 high is at least 1% above the 09:35
open", i.e. the name had **already** dropped in the first five minutes. M18 has no such filter; it
keeps 51.2% of the rows and it selects on the outcome's first leg. `G/g_m18.py` recomputes the block
on the right population and keeps the contaminated version beside it (`G/m18_check.md`). **This is
recorded as a defect found and fixed, not as a result.**

| population | group | 09:35→10:30 | split | n | gross bps (short) | net bps | t | net + borrow | MDE bps |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| correct (no R filter) | attention | | TRAIN | 998 | **−4.6** | −22.5 | −1.65 | −27.8 | 38.1 |
| correct | attention | | VAL | 429 | **+32.9** | +15.1 | 0.81 | +9.9 | 52.6 |
| correct | control | | TRAIN | 1,532 | −5.8 | −23.9 | −3.94 | −29.2 | 17.0 |
| correct | control | | VAL | 646 | −3.7 | −21.8 | −2.03 | −27.1 | 30.0 |
| contaminated (r_pct ≥ 1) | attention | | TRAIN | 565 | −19.3 | −36.9 | −1.70 | −42.2 | 60.6 |
| contaminated | attention | | VAL | 272 | +56.2 | +38.4 | 1.62 | +33.2 | 66.3 |

Positive = the short made money = the stock fell. By M18's dollar-volume bands, 09:35→10:30, the
stock's own move on the attention names is **+15.9 bps (TRAIN) / −21.0 (VAL) at $10–50M** and
**−6.2 / −45.6 at >$50M**; the control names move +4.3 / +1.2 and +7.0 / +5.2. Against M18's own
TRAIN table (attention −69 bps at $10–50M and −33 at >$50M, open→10:30, of which −37 and −14 are
already printed by 09:35) the arithmetic is consistent and the conclusion is narrow and specific:
**the fade M18 measured is a first-five-minutes, small-dollar-volume effect, and neither half of
that is available to a book that must borrow the shares and enters at 09:35.**

## 8. What this stage cannot say (the pre-registration's §8, now with numbers)

1. **Borrow availability is still not modelled.** No locate feed, no borrow-rate history, no
   threshold-list data. The 5.274 bps charge is a *price* assumption, not an *availability*
   assumption. A gap-up small cap is precisely the name that goes hard-to-borrow on the morning the
   book wants it. Since nothing cleared the gate, this does not change a verdict — but **any future
   positive short result must be re-run against a real locate history before it is believed.**
2. **The uptick rule is not modelled**; SSR days are excluded outright and their share is reported
   (6.7–41.0% of signals by family).
3. `asset_class` is symbol-level, not point-in-time, and it excludes **leveraged** wrappers, not
   plain funds: 38 of 2,610 G symbols (0.5% of rows) carry a fund/ETF token in the Alpaca asset
   name and were **left in** the primary grid; removing them changes nothing (§6). 40 symbols have
   no Alpaca name row and could not be audited.
4. Short interest, days-to-cover, options and ETF-creation channels are not in the data.
5. TRAIN is 2025 only and VAL+TEST is 2026 to 09-04; the base rate of everything in this program
   drifts across those splits.
6. **Phrasing (PLAN §1).** No edge was detectable **in this universe** (borrowable U1∪U2 ∪ the M18
   attention list: price ≥ $10, 20-day median dollar volume ≥ $10M, common stock), **at this
   horizon** (1-minute entries 09:30–14:01, exits by 15:55), **at this book size** (12/day,
   4 concurrent), **over this window** (2025-01-17 → 2026-09-04, TEST unread), **at this cost**
   (contract (c) + 0–5.274 bps borrow). The smallest per-trade effect the headline cells could have
   seen at 80% power is **0.056–0.616 R per trade**, i.e. **0.9–3.3 R/week at 4 slots** for the 22
   cells that trade often enough to be gated. Effects below that are not excluded by this test.

## 9. Cells looked at

| block | cells | declared in PREREG §7? |
|---|---:|---|
| the 24 gate cells, primary contract | 24 | yes |
| SSR-included re-score | 24 | yes |
| borrow-charged re-score | 24 | yes |
| S5 control group | 4 | yes |
| S5 restricted to U1∪U2 | 4 | yes |
| time bands (24 × 3) | 72 | yes |
| M18 no-stop bps rows (8 correct + 8 contaminated + 8 by dollar-volume band) | 24 | 4 declared, 20 beyond |
| the 24 primary cells re-scored without fund names | 24 | no — post-hoc, prompted by the ETF leak in §8.3 |
| **total looked at** | **200** | **156 declared** |

The pre-bugfix scoring run (§2) re-scored the same 24+24+24+4+4+72 cells and produced a
byte-identical 24-cell table; it is a repeat of declared cells, not 152 new ones.

## 10. Files

| path | what |
|---|---|
| `G/PREREG.md` | the pre-registration, unmodified (Part I above is its verbatim copy) |
| `G/g0_universe.py` → `G/members_g.csv`, `G/attention_list.csv`, `G/g0_universe.md` | the borrowable universe, the SSR flags, the M18 attention/control selection |
| `G/build_candidates_short.py` → `G/candidates_short.csv` (187,773 rows), `G/coverage_short_missing.csv`, `G/build_short.log` | pass 1: one row per short signal, six mirrored families, one fill model, three exits |
| `G/verify_rows.py` | the independent row-level rebuild (42/42) |
| `G/score_short.py` → `G/score_short_tables.md`, `G/score_short_results.csv`, `G/score.log` | the 24 cells + sensitivities + bands + the permutation |
| `G/g_extras.py` → `G/extras.md` | fills/SSR counts, the cost decomposition, the ETF audit, attention-minus-control |
| `G/g_m18.py` → `G/m18_check.md` | the corrected pure-M18 block |
| `G/score_short_tables.prebugfix.md` | the pre-bugfix scorer output, kept for the byte-identity diff in §2 (its 141 MB candidate CSV was deleted after the diff — the tables are the evidence) |

## 11. What to do with this

Nothing live. The short side was queued (PLAN §3 H12) as "the data's sign points there in every
first-hour cell"; measured with the same machinery as the long side, **the sign is there and the
size is not** — the attention names really do underperform their controls by +0.17 to +0.20 R, the
gap-fade really is the best short family, and none of it clears a spread. Two things are worth
carrying forward:

- **The `range_so_far_pct ≥ 5` floor is direction-agnostic**: it improved 12 of 12 short pairs here
  and 8 of 10 long comparisons in Stage E. It is a live-computable filter and the most reliable
  single thing this program has found. It is a *filter*, not a book.
- **The M18 veto stands, and its scope is now bounded**: prior-day attention names are worth
  avoiding on the long side (and they fade relative to controls on the short side), but the tradable
  part of that fade is in the first five minutes and in the $2–10M dollar-volume band — outside a
  borrowable short book. Record it in the rulebook as a bound on M18, not as a book.

If the short side is ever revisited, the cheapest un-run question is the one Stage E left open for
the long side: **U3, the unbiased liquid slice** (1.51M symbol-days, ~4.6 GB, ~7.5 h) — every G
family here was measured on a population pre-selected by a gap or a prior-day range, and that is a
conditioning choice, not a law.
