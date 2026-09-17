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
