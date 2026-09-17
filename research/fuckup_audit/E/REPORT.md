# Stage E — the causal universe (H6): definition, tape, and the pass-1 builder

`research/fuckup_audit/PLAN.md` §3 H6 / §4 row E. Written 2026-09-16. **No result of any kind is
reported here** — this stage builds a universe, a tape and a builder, and measures that they are
what they claim to be. The scan itself has not been run.

---

## 0. PRE-REGISTRATION (written before any candidate table was scored; nothing below it was run)

### The universes (causal at 09:30 ET on day t, no range gate, no hindsight)

| id | rule | built by |
|---|---|---|
| **U1 gap** | `open_t / close_{t-1} - 1 >= +3%`, `open >= $5`, `adv20 >= 100K` shares | `E/universes.py` -> `E/u1_keys.csv` |
| **U2 prior-day range** | `(high_{t-1} - low_{t-1}) / low_{t-1} >= 8%`, `open >= $5`, `adv20 >= 100K` | `E/u2_keys.csv` |
| **U3 liquid slice** | `median_20d(close x volume) >= $5M`, `open >= $5`, no other gate | `E/u3_keys.csv` (**not fetched in this stage**) |
| **U4 premarket** | `pm_dollar_vol(04:00-09:29) >= $500K` — not computable from a daily panel; it is a COLUMN of the Stage-E candidate table (`pm_dollar_vol`), so U4 is applied at scoring time, not at fetch time | `E/build_candidates_causal.py` |

`adv20 = volume.shift(1).rolling(20, min_periods=10).mean()`;
`dvol20_med = (close x volume).shift(1).rolling(20, min_periods=10).median()` — the convention of
`research/lit_review_2026/build_daily_panel.py:17`, i.e. the one behind `research/bf_zero/universe.csv`.
Every input except the day's own open comes from days strictly before t; the open is known at 09:30
and every entry is 09:35 or later.

### The cells Stage C-E will look at: **60**

**5 families** (`F8 N=5` — the ORB entry at 09:35 — `F8 N=15`, `F8 N=30`, `F6`, `F13`)
x **2 fills** (`entry_next` = the engine's next-bar open under the +0.6% cap; `entry_rest` = an
ORB-style resting stop-limit at the level)
x **2 exits** (`hold` to 15:55, `2r` = +2R on a bar close)
x **3 populations** (`all`; the ORB two-leg rule `has_news AND pm_dollar_vol > $5,816,688`;
`pm_dollar_vol > $5,816,688` alone)
= **60 cells**, declared here, to be counted against multiplicity in whatever stage scores them.
`F13` is close-triggered, so its `entry_rest` cells are structurally empty (6 of the 60) and will be
reported as such rather than silently dropped.

### The gates (PLAN.md §1, unchanged)

G1 TRAIN mean net R > 0, t >= 2.0, >= 5 trades/week. G2 VAL mean net R > 0, t >= 1.0, >= 55% weeks
green, bar raised by 1 SE of weekly R per 10 cells that passed G1. G3 **TEST is read ONCE**, after
the selection is frozen in writing, and reported whatever it says. Every G2 survivor also gets the
search-adjusted permutation p over all 60 cells, tail removal (top 1% / 5%), a +3R winner cap and a
per-month table. Cost contract: the corrected one adopted in Stage A (`A/REPORT.md` contract (c)),
not `score4`'s band table. Book: `trading.hod_break.run_book(rows, 12, 4)`.

### The one thing this universe is FOR

The >=5%-range universe plus its causal floor (`range_so_far_pct >= 5` before the signal) cannot
express an entry before a 5% range has already printed. On the Stage-E smoke sample **46.5% of
signals fall in 09:30-10:00 and only 24.8% of all signals would have passed that floor** — i.e. the
old design discarded roughly three quarters of the early-window population by construction. Stage
D0b's finding (the ORB two-leg news x PM$ rule is NEGATIVE after 10:00 on the >=5%-range universe)
is the reason the 09:35 window on a gap-up universe had to be built before that rule can be called
dead.

---

## 1. Universe sizes (PART 1)

Full table, per split, with overlap and tape coverage: **`E/universes.md`**. Summary:

| universe | symbol-days | distinct symbols | in `bf_zero/universe.csv` (the >=5%-range set) | already in `bars_sip.db` |
|---|---:|---:|---:|---:|
| U1 gap | 62,038 | 6,050 | 43,308 (69.8%) | 17,681 (28.5%) |
| U2 prior-day range | 160,619 | 5,976 | 134,049 (83.5%) | 62,739 (39.1%) |
| U3 liquid slice | 1,512,031 | 5,571 | 280,828 (18.6%) | 205,401 (13.6%) |
| **U1 u U2** | **198,318** | — | — | 73,732 (37.2%) |

Per split (TRAIN 2025 / VAL 2026-01..05 / TEST 2026-06..09-04): U1 30,871 / 17,839 / 13,328;
U2 81,168 / 45,772 / 33,679; U3 842,396 / 396,575 / 273,060. The calendar is the 420-day set of
`bf_zero/universe.csv` (early closes already excluded there), so the splits line up with every
earlier stage. `bars_sip.db` holds 305,547 symbol-days on that calendar.

Two things the overlap column says: **30% of U1 and 17% of U2 are symbol-days the old universe never
contained at all** (they did not end the day with a 5% range), and conversely U3 — the unbiased
liquid slice — is only 18.6% inside the old universe, which is the size of the selection the whole
program has been running on.

A `data/cache.db::intraday_bars_1min` probe on 4,000 random U1uU2 keys hits 7.3%. It is reported as
information only: cache.db is a different fetch provenance (and RTH-only), and this stage keeps ONE
tape (Alpaca SIP, `adjustment=raw`) so the new store and `bars_sip.db` are the same series.

Two panel hygiene items, fixed and recorded: the Databento panel carries rows with a NULL ticker
(1,913 of them fell inside U1uU2 and are dropped), and no (symbol, day) is duplicated once those are
removed. Membership is necessarily empty for the first ~10 trading days of 2025 (adv20 / dvol20_med
need 10 prior observations and the panel starts 2025-01-01).

### Price-scale check (PLAN.md §1)

Databento daily vs the Alpaca raw minute tape, 200 random keys present in both, `panel open` vs the
**09:30 ET minute bar's open**:

| within 0.01% | within 0.1% | within 0.5% | within 2% | median abs diff | p95 abs diff |
|---:|---:|---:|---:|---:|---:|
| 89.9% | 95.5% | 99.0% | 100.0% | 0.0000% | 0.0865% |

198 of 200 keys had a 09:30 minute bar. Rows: `E/pricescale.csv`. **No split/dividend scale break**:
the panel's open is the RTH open of the same raw series to well inside a tick for 99% of keys.

A second, per-row version of the same check is built into the candidate table: `gap_pct` is computed
from the MINUTE tape's 09:30 open and `gap_pct_daily` from the panel. On the smoke sample the two
agree to a **median 0.0000 pp, p95 0.033 pp, and 0.66% of rows differ by more than 0.5 pp**.

---

## 2. The fetch (PART 2) — COMPLETE

**Keys** = (U1 u U2) minus what `bars_sip.db` already holds = **124,586**. PLAN's Stage-E
instruction caps the fetch at 120,000 and, if larger, takes U1uU2 in full only from 2025-07 onward.
It is larger, so:

- **fetched set: 93,715 keys, 2025-07-01..2026-09-04** (`E/fetch_keys.csv`);
- **overflow: 30,871 keys, 2025-01-02..2025-06-30, all TRAIN** (`E/fetch_keys_overflow.csv`) — NOT
  fetched by default, recorded rather than dropped, obtainable with one command (§5).
  **Consequence to carry into any Stage-E scan: for keys not already in `bars_sip.db`, TRAIN covers
  only 2025-07 onward.** TRAIN is the split G1 is evaluated on; a scan run before the overflow pass
  is a half-TRAIN scan and must say so.

The running fetch was launched from the pre-cleanup key file (95,177 rows / 94,004 unique), which
additionally contained the NULL-ticker keys; those are recorded as `src='badsym'`. The current
`fetch_keys.csv` (93,715) is the clean set; a resume uses it.

**Window** 04:00:00-15:59:59 ET, DST-exact per day via `zoneinfo`. Premarket is fetched because
`pm_dollar_vol` (the ORB rule's missing leg, and U4) is computed from it.

**Store** `E/bars_causal/day=YYYY-MM-DD/bars.parquet`, pyarrow 23.0.1 + **zstd level 5**, columns
`symbol` (dictionary), `t` **int16 ET minute of day** (570 = 09:30), `o,h,l,c,v` float32. pyarrow is
installed, so the parquet path is the one used — no CSV fallback was needed.
`ARROW_DEFAULT_MEMORY_POOL=system` is set in both scripts: pyarrow's default jemalloc arenas blow
`ulimit -v 1300000` on this node.

**Status: COMPLETE — `DONE` + `EXIT=0` at 22:17 UTC, 32.7 min, 94,004 keys, 31,500,546 bars,
339 MB, median 350 bars per served symbol-day. Measured cost 3.69 KB per symbol-day** (target
<= 3 KB, hard cap 6 GB — 0.06 of the cap). ETA printed from the first 500 keys was 33 min.

| src | meaning | keys | share |
|---|---|---:|---:|
| `alpaca` | >= 1 bar returned | 93,119 | **99.06%** |
| `none` | Alpaca served nothing — survivorship residual | 46 | 0.05% |
| `badsym` | ticker rejected by Alpaca / NULL in the panel — survivorship residual | 839 | 0.89% |
| `error` | failed after the retry ladder | 0 | 0.00% |

Index: `E/bars_causal_index.csv`, one row per key, 0 duplicates. **Nothing is silently dropped**: an
unserved key is written with `n_bars=0`. PLAN's Stage-E stop rule ("if the fetch cannot serve >= 90%
of the symbol-days, report the survivorship residual and stop") is **passed: 99.06% served,
0.94% residual**.

### U3 is NOT fetched — the disk it would cost

U3 is 1,512,031 symbol-days, **1,306,630 of them not in `bars_sip.db`**. At the measured 3.69 KB
that is **~4.6 GB** — inside the 16 GB free and inside the 6 GB cap, but it is a later decision, not
this stage's, and it would take ~7.5 h at the measured rate. (For scale: the same keys in
`bars_sip.db`'s SQLite-with-ISO-timestamps format would be ~84 KB each = ~105 GB, which is the
"~100 GB, beyond this node" verdict `bf_zero2/REPORT.md` recorded. That verdict was a storage-format
artefact, exactly as `probe_design.md` item 6 argued: parquet is **23x** smaller here.)

---

## 3. The pass-1 builder (PART 3)

`E/build_candidates_causal.py`. Four differences from Stage B, and only these:

1. the universe is causal (`E/members.csv`) instead of the >=5%-range day list;
2. the bars come from `E/bars_causal/` first and `research/bf_zero/bars_sip.db` for keys it already
   held — one tape, SIP, raw. `data/cache.db` is deliberately NOT a source;
3. **no range-so-far floor** — membership is causal by construction, so a 09:35 entry is legitimate.
   `range_so_far_pct` is still emitted so both universes can be scored under the same scorer;
4. five family-configs: `F8 N=5`, `F8 N=15`, `F8 N=30`, `F6`, `F13`.

Everything else is Stage B's code, **imported, not copied**: the family detectors, both fill models,
the exit walks, MAE/MFE, the H1 stop variants and every feature come from `B/build_candidates4.py`
via `B4.build_day`. `B4.FAMS` is reassigned and `B.load_bars` is replaced with the causal loader; no
line of the family or walk code is duplicated.

**The thin import.** `build_candidates4` imports `build_candidates`, which at module scope loads the
whole 5M-row Databento panel and the >=5%-range universe (Stage B ran at `ulimit -v 3500000`, which
this node cannot spare). The builder therefore imports `build_candidates` FIRST with
`pd.read_parquet` / `pd.read_csv` stubbed to one-row frames, then imports `build_candidates4`, whose
own `import build_candidates` is a `sys.modules` cache hit. Nothing used is affected: the
panel-derived fields (`prev_close`, `prev_high`, `prev_low`, `adv20`) are supplied per row from
`E/members.csv`, computed from the SAME parquet panel with the same causal convention, and
`B.uni` / `B.daily` / `B.spy` are never read. This is the "thin loader" the stage brief allows, and
the parity number below is the evidence that it changed nothing.

**Columns** = `B4.COLS` (unchanged, in order) + `u1, u2, u3, split, open_daily, gap_pct_daily,
prev_day_range_pct_daily, dvol20_med, news_key, has_news, n_news_prev15`. `pm_dollar_vol` and
`prev_day_range_pct` are already in `B4.COLS` and are computed by `B4.build_day` from this store's
04:00-09:29 bars and from the per-row panel fields respectively.

### Smoke test — 3 fetched days (2025-07-02, 2025-07-08, 2025-07-15)

`E/candidates_causal_smoke.csv`: 3,326 signal rows over 1,070 symbol-days, 8 keys with no usable
tape. Per family: F8 N=5 801, F8 N=15 719, F8 N=30 639, F6 240, F13 927.

| entry-minute band | rows | share |
|---|---:|---:|
| 09:30-09:35 | 237 | 7.1% |
| 09:35-10:00 | 1,308 | 39.3% |
| 10:00-11:00 | 1,098 | 33.0% |
| 11:00-14:01 | 510 | 15.3% |
| 14:01+ (outside the scored window) | 173 | 5.2% |

`entry_next` fills 79.9% of signals, `entry_rest` 68.4% (F13 is close-triggered, so it has no
resting fill by construction — that is the whole of the gap). `pm_dollar_vol` is computable on
76.2% of rows (median $221K; **30.7% clear the $500K U4 cut**). Only **24.8% of rows would have
passed the old `range_so_far_pct >= 5` floor** — the measurement this stage exists to make.
`news_key` (a row in `D/news_presence.csv`) is present on **19.2%** of rows and `has_news` is 1 on
15.2% of those; **80.8% of rows have NO news value** and must be scored as such, or `D/d0_news.py`
must be re-pointed at this key set first (§5). Membership mix: u1 19.4%, u2 88.8%, both 8.1%.

### Parity vs `B/candidates4.csv` — `E/parity_causal.py`

Stage B's loader reads `cache.db` first, so only keys where cache.db holds nothing and
`bars_sip.db` holds something are tape-identical. On the 3 smoke days: 3,493 symbol-days, 1,240 in
cache.db, 1,966 in `bars_sip.db`, **1,966 tape-identical**, of which **1,231 signal rows appear in
both files**.

| column | n compared | max abs diff | n over 1e-6 |
|---|---:|---:|---:|
| `sig_m`, `level`, `stop`, `range_so_far_pct` | 1,231 | 0 | 0 |
| `next_entry`, `next_entry_m`, `next_rr_2r` | 992 | 0 | 0 |
| `rest_entry`, `rest_entry_m`, `rest_rr_2r` | 840 | 0 | 0 |

**Exact: max abs diff 0 on every column, 0 rows over tolerance** (the task's bar was 1e-6 on
`entry_next` / `rr_2r`). 1,083 rows exist only in the causal file and 8,089 only in Stage B's — the
two universes are different by design and that difference is the point of the stage, not an error.
Full table: `E/parity_causal_smoke.md`.

---

## 4. What this stage does and does not support

It supports: (a) a universe whose membership is knowable at 09:30 exists and is 198,318 symbol-days
for U1uU2 and 1.51M for U3; (b) the daily panel and the minute tape are on the same price scale
(99% within 0.5%, per-row median 0.0000 pp); (c) parquet makes the "infeasible" fetch feasible —
3.69 KB/symbol-day, 23x smaller than the SQLite store, 99.06% served; (d) the Stage-B family and
walk code, run on the new universe and the new tape, reproduces Stage B EXACTLY on every key the two
share with the same tape.

It does not support any claim about whether a book exists in the early window. **No cell has been
scored.** The 60 cells above are declared, not run. The honest phrasing for whatever comes next must
carry the TRAIN truncation of §2 and the 80.8% news-null share of §3.

---

## 5. Exact instructions

**The fetch is finished** (`E/fetch.log` ends `DONE` + `EXIT=0`). To resume it after any future
interruption — it is resumable at day granularity, `E/fetch_state.json` plus the per-day parquet
files are the state — re-run the same command; finished days are skipped:

```bash
setsid nohup bash -c "cd /home/ec2-user/onemil; ulimit -v 1300000; nice -n 10 \
  python3 research/fuckup_audit/E/fetch_causal.py > research/fuckup_audit/E/fetch.log 2>&1; \
  echo EXIT=\$? >> research/fuckup_audit/E/fetch.log" >/dev/null 2>&1 </dev/null &
```

**Fetch the 2025-01..06 overflow** (30,871 TRAIN keys, ~111 MB, ~11 min) — do this before any TRAIN
number is quoted:

```bash
setsid nohup bash -c "cd /home/ec2-user/onemil; ulimit -v 1300000; nice -n 10 \
  python3 research/fuckup_audit/E/fetch_causal.py --overflow \
  > research/fuckup_audit/E/fetch_overflow.log 2>&1; \
  echo EXIT=\$? >> research/fuckup_audit/E/fetch_overflow.log" >/dev/null 2>&1 </dev/null &
```

The overflow pass shares `E/fetch_state.json`; its day set is disjoint from the main pass, so the
two never collide, and a finished day is always skipped. It appends to the same
`E/bars_causal_index.csv`.

**Run the candidate build** — ONLY when `research/fuckup_audit/B/build4.log` has ended (the Stage-B
rebuild was still running at hand-off; one heavy python process at a time, PLAN.md §1). ~410 days at
~13 s/day on the smoke sample -> roughly 1.5 h; resumable per day via `E/build_causal_state.json`:

```bash
setsid nohup bash -c "cd /home/ec2-user/onemil; ulimit -v 1300000; nice -n 10 \
  python3 research/fuckup_audit/E/build_candidates_causal.py \
  > research/fuckup_audit/E/build_causal.log 2>&1; \
  echo EXIT=\$? >> research/fuckup_audit/E/build_causal.log" >/dev/null 2>&1 </dev/null &
```

Outputs `E/candidates_causal.csv` and `E/coverage_causal_missing.csv` (keys with no usable tape —
join it to `E/bars_causal_index.csv` to separate "Alpaca served nothing" from "fewer than 10 RTH
bars"). Then re-run the parity check on the finished file:

```bash
ulimit -v 1300000; nice -n 10 python3 research/fuckup_audit/E/parity_causal.py
```

(no `E_TAG`), which compares against `B/candidates4.csv` on every shared, tape-identical key of the
whole window rather than the 3 smoke days. A smoke re-run is
`E_TAG=_smoke E_DAYS=3 E_DAY_LIST=2025-07-02,2025-07-08,2025-07-15 python3 .../build_candidates_causal.py`.

**Before the scan**, re-point the news pull at this key set — the Stage-D file covers only 19.2% of
these rows: `D/d0_news.py` with `KEYS` = the (day, symbol) pairs of `E/candidates_causal.csv`. It is
already causal (prev-day 15:00 ET -> 09:30) and throttled; ~200K keys, expect ~30 min.

**Never** run two heavy python processes on this node at once, always `nice -n 10`, always
`ulimit -v 1300000`, never `| tail` a long job (PLAN.md §1).

---

## 6. Files

| path | what |
|---|---|
| `E/universes.py` | Part 1 — `panel` phase (daily panel -> `members.csv`) and `report` phase (coverage, price scale, key files). Two processes because the 5M-row panel and the coverage work do not coexist in 1.3 GB. |
| `E/members.csv` | 1,570,771 rows = U1uU2uU3 with `open, prev_close, prev_high, prev_low, gap_pct, prev_day_range_pct, adv20, dvol20_med, u1, u2, u3, split` |
| `E/u1_keys.csv`, `E/u2_keys.csv`, `E/u3_keys.csv` | the three universes |
| `E/universes.md` | Part 1's full tables |
| `E/pricescale.csv` | the 200-key price-scale rows |
| `E/fetch_keys.csv`, `E/fetch_keys_overflow.csv` | the fetched set and the truncated remainder |
| `E/fetch_causal.py`, `E/fetch.log`, `E/fetch_state.json` | Part 2 |
| `E/bars_causal/day=*/bars.parquet`, `E/bars_causal_index.csv` | the tape (339 MB, 31.5M bars) and its per-key index |
| `E/build_candidates_causal.py` | Part 3, importing `B/build_candidates4.py` |
| `E/parity_causal.py`, `E/parity_causal_smoke.md` | the parity evidence |
| `E/candidates_causal_smoke.csv` | the 3-day smoke output |

---

## 7. FREEZE (written 2026-09-17 00:45 UTC, BEFORE TEST was read)

The 60 pre-registered cells produced **0 G1 passes**, so the pre-registered selection is empty and
TEST is not owed to it. One cell in the stage clears the G1 and G2 arithmetic. It is **post-hoc** —
it came out of follow-up A, the H6 diagnostic that splits the causal population by the old floor —
and it is frozen here, exactly and in full, before `E_READ_TEST=1` is run:

> **The frozen cell.** Family `F6 {}` (red-to-green: the first 1-min bar whose close crosses back
> above the prior day's close, as implemented in `B/build_candidates4.py`), fill `entry_next`
> (the next bar's open under the +0.6% cap), exit **hold to 15:55**, universe **U1 u U2** (gap >= +3%
> at the 09:30 open, or prior-day range >= 8%; open >= $5, adv20 >= 100K shares — all knowable at
> 09:30), population **all** AND **`range_so_far_pct >= 5`** (the day's high-low range on bars
> STRICTLY BEFORE the signal bar, i.e. causal and live-computable), entry window 09:30-14:01,
> price >= $5 at the fill, R >= 1% of price, cost contract (c), book `run_book(12, 4)`.
> No news leg, no premarket-dollar leg, no other filter.

Its TRAIN and VAL numbers, its tail tests and its permutation p are already recorded in
`E/score_e_f6floor.md` and are repeated in §8 below; **they were computed before this freeze and
are not changed by what TEST says.** Pre-committed reading of the TEST result: this cell is
reported as a candidate ONLY if TEST is positive AND it survives the top-5% removal on TEST — it
already FAILS top-5% removal on both TRAIN (-0.139) and VAL (-0.062), so the pre-committed
expectation is that it is reported as a failure regardless of the TEST sign, and TEST is read for
the record, not for selection.

---

# 8. SCORING RESULT (2026-09-17)

_Everything above §7 is the pre-registration and was written before any cell was scored. This
section is the result. Scripts: `E/e_avail.py` (step 0), `E/e_score.py` (the 60 cells + declared
sensitivities), `E/e_follow.py` (the H6 diagnostic, `news_only` booked, the monotonicity and
Simpson's checks, the Stage-C diff), `E/e_f6floor.py` (the one post-hoc gate pass, TEST), `E/e_headlines.py`.
Tables: `score_e_availability.md`, `score_e_tables.md`, `score_e_follow.md`, `score_e_f6floor*.md`,
`score_e_headlines.md`; per-cell CSVs `score_e_results*.csv`, `score_e_buckets.csv`,
`score_e_newsonly.csv`._

## 8.0 One page

**The causal universe does not change the picture; it makes it worse, and it does so by removing
the very rows the old design's floor was throwing away.** On the causal universe (U1 gap ≥ +3% ∪
U2 prior-day range ≥ 8%, membership knowable at 09:30), **0 of the 60 pre-registered cells clear
G1** — the best TRAIN cell is `F6 {}` next/hold/combo at **+0.018 R, t 0.15, 7.1 trades/week**, and
a 500-draw day-label sign-flip null over the 54 scoreable cells puts the **observed max TRAIN t at
0.15 against a null 95th percentile of 3.81, p = 1.000**. On the 18 family × fill × exit cells that
Stage C also scored on the ≥5%-range universe, Stage E is **−0.066 R per trade worse on TRAIN**.

**The H6 premise is refuted in the direction opposite to the hypothesis.** H6 said the ≥5%-range
design "forces late entries" and discards the early window by construction. It does discard it —
60–80% of the causal population never reaches a 5% pre-signal range — but the discarded rows are
the LOSING rows, in every family and on both TRAIN and VAL. `F8 N=5`: floor-passing −0.005 vs
below-floor −0.144 (TRAIN); `F6 {}`: +0.090 vs −0.114 (TRAIN), +0.209 vs −0.046 (VAL). The pre-signal
range floor is not a bias to be removed — it is, on this tape, the single most effective causal
filter in the stage.

**Early window vs later.** Booked per band, the 09:30–10:00 window is the WORST band for every
family (`F6 {}` 09:35–10:00 −0.130 R TRAIN at a 59% stop rate; `F8 N=5` −0.088; `F13` −0.273), and
the later bands are flat to positive (`F8 N=15` 11:00–13:00 +0.007 TRAIN / −0.023 VAL). Stage A's and
Stage H5's direction survives; the causal universe gives the early window a fair test and it fails it.

**Fill and exit.** Next-open beats resting in all five families (by +0.013 to +0.063 R), and
hold-to-close beats the +2R close-target in all five (by +0.005 to +0.072 R) — Stage C's two
conclusions reproduce on a different universe and a different tape slice.

**The ORB two-leg rule, in the 09:35 window on a gap-up universe that was built to give it its
home, is negative.** All-day combo-minus-rest is negative on TRAIN in 5 of 5 families and on TEST
in 5 of 5; in the 09:35–10:00 band it is negative on TRAIN in every family that has one. D0b's ≥10:00 finding and D1's
all-day finding now hold in the one window where the rule had not yet been tested. **The rule is
dead on this population, not merely untested.**

**`news_only`** (news present AND premarket $ ≤ $5.82M) passes the pre-registered sign-agreement
rule in **1 of 5 families** — `F8 N=5`, TRAIN +0.061 R (t 2.54), VAL +0.026 (sign agrees), TEST
+0.073 (t 2.15). It survives the pm-unknown ablation (+0.057 / +0.013 / +0.074) and is not a
Simpson's artefact of one price or time slice. But it is **anti-monotone in PM$ inside the news
bucket** (TRAIN decile rank correlation **−0.78**, VAL +0.09), the bucket itself is **negative in
absolute R in every split** (−0.027 / −0.037 / −0.092), and **as a book it clears G1 in 0 of 9
cells**. It is a weak relative veto, not an edge.

**One cell cleared the G1/G2 arithmetic** — post-hoc, from the H6 diagnostic: `F6 {}` next/hold on
the causal universe RESTRICTED to `range_so_far_pct ≥ 5`. TRAIN +0.090 R (t 2.03, 20.3 tr/wk, 55%
green), VAL +0.209 (t 2.80, 77% green). It was frozen in writing in §7 and **TEST was then read
once: +0.096 R, t 0.67, 43% of weeks green, 2 of 4 months negative**. It dies with the top 5%
removed on **all three** splits (−0.139 / −0.062 / −0.227) and under a +3R winner cap on TEST
(−0.071); its permutation p over this stage's cells is **0.936**. Reported as a failure, exactly as
pre-committed.

**Nothing in Stage E is a ship candidate.**

## 8.1 Step 0 — verification (before any number)

| check | result |
|---|---|
| build | `build_causal2.log` ends `DONE rows 506,195` + **`EXIT=0`**; the first run had written 57 days (77,574 rows) before its OOM, and 77,574 + 506,195 = **583,769 = the file's data rows exactly** |
| days | `build_causal_state.json` 410 done, min 2025-01-17 max 2026-09-04; **410 distinct days in the file**; one header line, no mid-file header from the resume |
| header | 88 columns, **byte-identical and in order** to `B4.COLS` (77) + the 11 declared extras |
| overflow | `fetch_overflow.log` ends `DONE … EXIT=0`, 30,871 TRAIN keys of 2025-01..06 fetched — **TRAIN is NOT truncated**; §2's warning is discharged |
| parity | `parity_causal.py` over the **full 410 days**: 542,876 symbol-days, 289,225 tape-identical with Stage B, **217,576 signal rows in both files**; `sig_m` / `stop` / `range_so_far_pct` / `next_entry` / `next_entry_m` / `next_rr_2r` **max abs diff 0.0, 0 rows over 1e-6** (the task's bar). Residual float32↔float64 noise on 13 of 217,576 `level` values (max 5e-4) and 3 `rest_entry` (3e-4, `rest_rr_2r` 3.1e-5) — 0.006% of rows, from the parquet store's float32 bar prices; it moves no decision. |
| survivorship | 1,510 of 198,318 U1∪U2 keys have no usable tape = **0.76%**; of the fetched index, 1,067 of 124,875 = 0.85% returned zero bars (702 `badsym`, 661 `alpaca`-served-but-too-thin, 76 `none`). PLAN's ≥90% bar is passed by a wide margin. |

## 8.2 Step 0 — the AVAILABILITY AUDIT (PLAN §1 standing rule, earned in D1)

Coverage on the 314,288-row scoreable `next` population: `has_news` **100.00%**,
`prev_day_range_pct` / `adv20` / `gap_pct` / `spread_cc_bps` / `range_so_far_pct` **100.00%**
(8–10 rows of 314,288 missing), `pm_dollar_vol` **89.01%**.

`has_news` carries **no missingness at all** — the whole point of `E/e_news.py` re-pointing the
causal news pull at this key set — so the D1 failure mode cannot recur for the news leg.

`pm_dollar_vol` is the one column with structure, and it is reported in full before it is used:

| missing rate | 09:30–09:35 | 09:35–10:00 | 10:00–11:00 | 11:00–13:00 | 13:00–14:01 |
|---|---:|---:|---:|---:|---:|
| TRAIN | 8.7% | 8.9% | 13.3% | 18.0% | 19.4% |
| VAL | 7.1% | 6.9% | 11.1% | 15.4% | 13.8% |
| TEST | 5.0% | 5.8% | 9.6% | 13.5% | 17.1% |

Outcome correlation of the missingness (mean net R, hold): missing −0.013 vs present −0.041 on
TRAIN, −0.089 vs −0.040 on VAL, −0.166 vs −0.114 on TEST — **the sign flips across splits and the
magnitude is 0.03–0.05 R.** That is NOT D1's signature (there the missing bucket was −0.28..−0.46 R
at t −6..−16 in every family and every split, an availability indicator worth ~0.8 R). Provenance
was checked as well: pm-missing runs 11.8% on keys served by the Stage-E parquet store (fetched
04:00–15:59, so a missing value is a genuine "no premarket trades") and 13.6% on keys served from
`bars_sip.db` — **close enough that pm-availability is not a proxy for "this key was in the old
≥5%-range fetch"**, which is exactly the leak D1 found. `pm_dollar_vol` is therefore admitted, with
two disclosures: (a) NaN is treated as BELOW the cut, so the `combo` and `pm_only` buckets contain
only positively-established values while the complement mixes known-low with unknown; (b) every
`news_only` result below is re-run with the 11% unknown rows dropped and does not change (§8.7).

Bucket shares on the scoreable population (TRAIN / VAL / TEST): `neither` 72.8 / 73.6 / 74.6%,
`pm_only` 11.0 / 12.0 / 12.7%, `news_only` 11.3 / 9.6 / 8.1%, `combo` 5.0 / 4.8 / 4.7%.

## 8.3 The 60 pre-registered cells

54 scoreable (6 structurally empty: F13 × `rest`, close-triggered by construction). Full table
`score_e_tables.md`; per-cell CSV `score_e_results.csv`.

**G1 (TRAIN mean net R > 0, t ≥ 2.0, ≥ 5 trades/week): 0 of 54. G2: 0. TEST not read for the grid.**

Top of the ranking (TRAIN), with the gross column beside the net:

| key | fill | exit | pop | TR n | tr/wk | **net R** | gross R | t | WR | stop% | VAL net R | VAL t | VAL green |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F6 {} | next | hold | combo | 362 | 7.1 | **+0.0177** | +0.1005 | 0.15 | 37.3 | 43.6 | +0.0775 | 0.73 | 0.59 |
| F8 N=15 | next | hold | pm_only | 1311 | 25.7 | **+0.0008** | +0.0700 | 0.02 | 41.9 | 37.9 | −0.0279 | −0.60 | 0.45 |
| F6 {} | rest | hold | combo | 440 | 8.6 | −0.0207 | +0.1390 | −0.21 | 34.3 | 44.8 | +0.0892 | 0.76 | 0.64 |
| F8 N=15 | next | hold | all | 1368 | 26.8 | −0.0260 | +0.0352 | −0.76 | 41.9 | 35.4 | −0.0288 | −0.58 | 0.55 |
| F8 N=30 | next | hold | all | 1200 | 23.5 | −0.0355 | +0.0144 | −1.20 | 43.8 | 26.2 | −0.0269 | −0.67 | 0.55 |
| F8 N=5 | next | hold | all | 1656 | 32.5 | −0.0899 | −0.0052 | −2.35 | 37.4 | 48.6 | −0.1206 | −2.12 | 0.41 |
| F13 | next | hold | all | 2247 | 44.1 | −0.2686 | −0.1023 | −4.91 | 21.2 | 76.1 | −0.1286 | −1.54 | 0.32 |

Worst cell of the 54: `F13` rest-impossible, `next`/2r/combo at **−0.436 R**. The exit mix is the
familiar one: hold-to-close books 43–81% `eod` exits and 19–76% stops with 0% targets by
construction; the 2R variant converts 6–29% of trades into targets and is worse in every family.

**Permutation, search-adjusted over all 54 cells (500 day-label sign-flip draws): observed max
TRAIN t = 0.15, null 95th pct = 3.81, p = 1.000.** The grid contains no signal of the size it can
detect, and the best cell in it is indistinguishable from a coin.

**Declared sensitivities** (counted, none changes the verdict): universe **U1 only** 42 scoreable
cells, G1 0 — best `F8 N=30` next/hold/all −0.018 R; universe **U2 only** 54 cells, G1 0; the
**resting fill restricted to `rest_queue_ok == 1`** 24 cells, best −0.018 R, G1 0 (Stage C's
finding that the queue check passes on too few rows to matter carries over).

## 8.4 H6 answered: the old floor was keeping the better rows

`score_e_follow.md` §A. Same table, same tape, same contract; each half booked on its own.

| family | split | passed floor (net R, t) | below floor (net R, t) | % of family below floor |
|---|---|---|---|---:|
| F8 N=5 | TRAIN | **−0.005** (−0.13) | **−0.144** (−3.84) | 79.7% |
| F8 N=5 | VAL | +0.033 (0.54) | −0.132 (−2.20) | |
| F8 N=15 | TRAIN | −0.004 (−0.13) | −0.071 (−2.04) | 69.3% |
| F8 N=15 | VAL | −0.074 (−1.92) | −0.034 (−0.67) | |
| F8 N=30 | TRAIN | −0.032 (−1.29) | −0.073 (−2.38) | 60.4% |
| F8 N=30 | VAL | +0.062 (1.71) | −0.047 (−1.01) | |
| F6 {} | TRAIN | **+0.090** (2.03) | **−0.114** (−2.05) | 75.2% |
| F6 {} | VAL | **+0.209** (2.80) | −0.046 (−0.61) | |
| F13 | TRAIN | −0.225 (−3.44) | −0.227 (−3.98) | 60.7% |
| F13 | VAL | −0.195 (−2.18) | −0.126 (−1.38) | |

8 of 10 family × split comparisons favour the floor-passing half (7 of 10 by the booked mean and an 8th, F13 TRAIN, by a hair), and the two that do not (F8 N=15
VAL, F13 VAL) are within one SE. The stop rate tells the mechanism: `F6 {}` stops on 28.2% of
floor-passing trades and **58.9%** of below-floor trades; `F8 N=5` 39.6% vs 52.6%. A breakout on a
name that has not yet moved 5% is a breakout with nothing behind it.

## 8.5 Time bands (the H5/H6 crossing point)

`score_e_follow.md` §A2, each band booked on its own, `all` population, next/hold.

| family | 09:30–09:35 | 09:35–10:00 | 10:00–11:00 | 11:00–13:00 | 13:00–14:01 |
|---|---:|---:|---:|---:|---:|
| F8 N=5 TRAIN | — | −0.088 | −0.090 | −0.067 | −0.104 |
| F8 N=15 TRAIN | — | −0.033 | −0.040 | **+0.007** | −0.042 |
| F8 N=30 TRAIN | — | — | −0.041 | −0.025 | −0.016 |
| F6 {} TRAIN | −0.100 | −0.130 | −0.014 | **+0.019** | **+0.035** |
| F6 {} VAL | +0.021 | +0.005 | +0.121 | +0.052 | −0.006 |
| F13 TRAIN | — | −0.273 | −0.250 | −0.317 | −0.241 |

The early window is where the stop rate lives (F6 70.4% at 09:30–09:35, 58.9% at 09:35–10:00,
against 4.3% at 13:00–14:01) and the cost per R is worst there. On the universe built specifically
to let the early window compete, it does not.

## 8.6 The ORB two-leg rule in its own window

Combo (`has_news` AND `pm_dollar_vol > $5,816,688`) minus the rest, per trade, next/hold:

| family | TRAIN | VAL | TEST | TRAIN 09:35–10:00 |
|---|---:|---:|---:|---:|
| F8 N=5 | −0.054 | −0.015 | −0.025 | −0.024 |
| F8 N=15 | −0.087 | −0.039 | −0.032 | −0.056 |
| F8 N=30 | −0.098 | +0.002 | −0.071 | — |
| F6 {} | −0.019 | +0.091 | −0.109 | −0.168 |
| F13 | −0.164 | −0.038 | −0.017 | −0.164 |

Negative on TRAIN in 5 of 5, on TEST in 5 of 5, and negative on TRAIN in the 09:35–10:00 band in
every family that has one. The booked form is the same: the 18 scoreable `combo` cells of the grid are all
G1 failures and the best of them (`F6 {}` next/hold) runs 7.1 trades/week — under the gate's own
5/week floor only because F6 is thin. **This closes the question D0b and D1 left open.** The
`pm_only` leg alone is negative on TRAIN in 5 of 5 families as well (−0.003 to −0.161).

## 8.7 `news_only` — the pre-registered bucket

Per-trade, all-day, next/hold (`score_e_buckets.csv`):

| family | TRAIN diff (t) | VAL diff (t) | TEST diff (t) | sign-agreement rule |
|---|---|---|---|---|
| F8 N=5 | **+0.061 (2.54)** | **+0.026 (0.78)** | +0.073 (2.15) | **PASS** |
| F8 N=15 | +0.054 (2.99) | −0.021 (−1.01) | +0.008 (0.31) | fail (VAL sign) |
| F8 N=30 | +0.041 (2.59) | −0.004 (−0.25) | +0.011 (0.50) | fail (< +0.05 R) |
| F6 {} | +0.110 (1.91) | −0.017 (−0.23) | +0.183 (2.24) | fail (t < 2, VAL sign) |
| F13 | −0.039 (−0.47) | −0.079 (−0.72) | +0.077 (0.57) | fail (sign) |

For the one survivor, `F8 N=5`, the three required follow-ups:

- **Monotone in PM$?** No, and the sign is the opposite of the shipped ORB gate's. Inside the
  news-carrying rows, decile of `pm_dollar_vol` vs mean net R: **TRAIN rank correlation −0.78**
  (decile 0 +0.025 → decile 9 −0.193), VAL +0.09. The premarket-unknown rows are the best decile on
  both splits (+0.007 TRAIN, +0.037 VAL). Read plainly: within newsy names, MORE premarket dollars
  is WORSE, which is why `news_only` beats `combo` — and it says the lift is a crowding penalty on
  the high-PM$ tail, not a catalyst premium.
- **Simpson's check.** The difference is positive in 5 of 5 price bands and 4 of 4 time bands on
  TRAIN, so the all-day number is not one slice; but on VAL it is positive in 3 of 5 price bands and
  2 of 4 time bands and negative in the $100+ band on both VAL and TEST. No single slice carries it;
  no slice replicates it either.
- **pm-unknown ablation.** Dropping the 11% of rows with no premarket value: TRAIN +0.057 (t 2.26),
  VAL +0.013 (t 0.36), TEST +0.074 (t 2.11) — the effect is not an availability artefact.
- **As a book: 0 of 9 G1 passes** (`score_e_newsonly.csv`), best `F8 N=30` next/hold at +0.0006 R
  (t 0.02). The bucket's own level is negative on every split for `F8 N=5` (−0.027 / −0.037 / −0.092);
  what is positive is only the *difference* to a worse remainder.
- **Recaps vs company events** (30 random TRAIN trades of the bucket, 54 premarket articles, keyword
  classifier fixed before the pull, `score_e_headlines.md`): 14 event-worded, 6 recap/list-worded, 3
  both, 31 neither. Mean net R of the sampled trades by dominant class: **event-dominant −0.82
  (n 9), recap-dominant +0.54 (n 3), neither +0.90 (n 17)**. Directionally the same as the ORB rule
  book ("recaps performed EQUAL to real catalysts for longs" — here, if anything, better), but n = 30
  symbol-days is descriptive and the classifier leaves 57% unclassified; this is a sanity check, not
  evidence.

## 8.8 The one post-hoc cell that cleared the arithmetic — frozen in §7, TEST read once

`F6 {}` × next × hold × `range_so_far_pct ≥ 5` on the causal universe (`score_e_f6floor*.md`):

| split | n | tr/wk | net R | gross R | t | WR | weeks green | top-1% off | top-5% off | +3R cap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TRAIN | 1034 | 20.3 | +0.0896 | +0.1182 | 2.03 | 45.1 | 0.55 | +0.006 | **−0.139** | +0.012 |
| VAL | 512 | 23.3 | +0.2088 | +0.2398 | 2.80 | 46.7 | 0.77 | +0.098 | **−0.062** | +0.086 |
| **TEST** | 372 | 26.6 | **+0.0958** | +0.1291 | **0.67** | 40.6 | **0.43** | **−0.065** | **−0.227** | **−0.071** |

Per month on TEST: 2026-06 **−31.1 R**, 2026-07 −8.9 R, 2026-08 **+72.2 R**, 2026-09 +3.4 R — one
month is the whole book. Permutation p over this stage's cells **0.936**. It fails the tail test on
all three splits and the winner cap on TEST, its TEST t is 0.67 and fewer than half its TEST weeks
are green. **Not a candidate.** The honest reading is that this is the same `F6 {}` shape Stage C
already recorded as its closest miss (TRAIN +0.051 t 1.34, VAL +0.163 on the ≥5%-range universe),
re-selected on a slightly different universe, with its t pushed over 2.0 by the extra gap-up and
prior-day-range symbol-days — and it dies to the same tail test that killed it in Stage C.

## 8.9 What it means, and the smallest effect each test could see

1. **H6 is settled and it is settled against itself.** A universe knowable at 09:30 exists, it is
   198,318 symbol-days, the tape for it is 99.06% served, the builder reproduces Stage B exactly on
   shared keys — and the entry families are WORSE on it (−0.066 R/trade on TRAIN over 18 matched
   cells) because the rows it adds are rows on which nothing has happened yet. The pre-signal range
   floor stays, and it stays as a causal, live-computable FILTER rather than as a universe artefact.
2. **The 09:35 gap-up window is not a hidden home for anything here** — not for the raw families,
   not for the ORB two-leg rule, not for `news_only`. It is the highest-stop, highest-cost band on
   a causal universe just as it was on the ≥5%-range one.
3. **The news leg keeps producing a weak, wrong-signed relative effect.** `news_only` beat the rest
   in D1 (not pre-registered) and beats the rest again here on one family of five, with the PM$
   deciles running the wrong way for the shipped ORB logic. It is worth exactly one sentence in the
   ORB rulebook — *within newsy names, high premarket dollars mark crowding* — and it is not worth a
   book: 0 of 9 booked cells clear G1.
4. **The cost constant is still the whole deficit.** Seven of the top ten cells are gross-POSITIVE
   and net-negative; `F6 {}` next/hold/combo is +0.101 gross and +0.018 net, `F6 {}` rest/hold/combo
   +0.139 gross and −0.021 net. Stage A's H7 correction is already in contract (c); what remains is
   that at these R sizes a half-spread is still 0.10–0.15 R and no family clears it.

**Smallest per-trade effect each headline cell could have seen** (2.8 × SE on the TRAIN book,
≈ 80% power at 5% two-sided):

| headline cell (next / hold / all) | TRAIN SE | MDE per trade | tr/wk | MDE in weekly R at 4 slots |
|---|---:|---:|---:|---:|
| F8 N=30 | 0.0296 | **0.083 R** | 23.5 | 1.95 R/wk |
| F8 N=15 | 0.0344 | 0.096 R | 26.8 | 2.58 R/wk |
| F8 N=5 | 0.0383 | 0.107 R | 32.5 | 3.49 R/wk |
| F13 | 0.0548 | 0.153 R | 44.1 | 6.77 R/wk |
| F6 {} | 0.0551 | 0.154 R | 36.2 | 5.58 R/wk |
| F6 {} floor-passing (post-hoc) | 0.0442 | 0.124 R | 20.3 | 2.52 R/wk |
| `news_only` difference, F8 N=5 | 0.0239 | 0.067 R | — | — |

**Phrasing, per PLAN §1:** no edge was detectable **in this universe** (causal U1 ∪ U2 at 09:30),
**at this horizon** (entries 09:30–14:01, exits hold-to-close or +2R on a close), **at this book
size** (12 candidates/day, 4 concurrent), **over this window** (2025-01-17 → 2026-09-04, 410 days),
**at this cost** (contract (c)); the smallest per-trade effect the headline tests could have seen is
**0.083–0.154 R**, i.e. **2.0–6.8 R per week at 4 slots**. Effects smaller than that — including the
+0.02..+0.06 R that several cells actually show — are excluded by nothing here.

## 8.10 Cells looked at in this stage

| what | declared | scoreable / reported |
|---|---:|---:|
| the 60 pre-registered cells (5 families × 2 fills × 2 exits × 3 populations) | 60 | 54 |
| sensitivity: universe U1 only | 60 | 42 |
| sensitivity: universe U2 only | 60 | 54 |
| sensitivity: resting fill restricted to `rest_queue_ok == 1` | 30 | 24 |
| `news_only` booked (PLAN §1 pre-registration, hold exit) | 10 | 9 |
| the D0b 2×2 per family × 6 scopes × 3 splits × 4 buckets (per-trade) | 360 | 296 |
| H6 floor split, booked (5 families × 2 splits × 2 subsets) | 20 | 20 |
| time bands, booked (5 families × 2 splits × 5 bands) | 50 | 40 |
| PM$ decile monotonicity inside the news bucket (F8 N=5, 2 splits × 11 deciles) | 22 | 22 |
| Simpson's slices (price band and time band × 3 splits) | 27 | 27 |
| pm-unknown ablation | 3 | 3 |
| Stage-C matched comparison | 18 | 18 |
| the post-hoc `F6` floor cell (TRAIN / VAL / TEST) | 3 | 3 |
| availability-audit diagnostic cells (missingness, provenance, bucket shares) | ~24 | ~24 |
| headline classification sample (30 symbol-days, 54 articles) | 1 | 1 |
| **total cell-instances** | **~748** | **~637** |

Plus two 500-draw day-label sign-flip permutation nulls (over the 54 primary cells, and over those
54 plus the post-hoc cell).

## 8.11 Live implementability — moot, recorded anyway

No candidate survives, so no engine delta is proposed. For the record, the two things Stage E
established that ARE live-implementable and causal at the decision minute — `range_so_far_pct`
computed on bars strictly before the signal, and a 09:30 universe membership from the previous
day's daily bar plus the day's open — are both already computable in `trading/hod_break_engine.py`
from the streamed bars and `daily_bars`; neither needs new data. They are filters, not a book.
