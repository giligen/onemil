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
