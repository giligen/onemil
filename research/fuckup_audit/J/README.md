# Stage J — the liquid universe (U3): how to run it, and how to resume it

Pre-registration: `J/PREREG.md` (written 2026-09-17 20:10 UTC, before any bar of U3 was scanned) and
its **ADDENDUM 1-3** (2026-09-17 14:40 UTC, also before the scan — test-ticker / `daily_bars` universe
hygiene, every family's level buffer stated rather than inherited, and BOTH scan rules as twin cells;
the 19 days built under the un-amended spec were deleted and the scan restarted from day 1).
Read both first. `J/REPORT.md` is the deliverable. Nothing here is enabled, proposed or live.

Everything in this directory writes ONLY under `research/fuckup_audit/J/`. `data/cache.db` and
`data/trades.db` are never opened (the tape is the three SIP stores); no config, service or order
is touched.

---

## The chain, in order

| # | what | script | output | wall time |
|---|---|---|---|---|
| 0 | the U3 tape (a separate stage) | `U3/fetch_u3.py` | `U3/bars_u3/day=*/bars.parquet` | ~7.1 h |
| 1 | the daily-context table + ADDENDUM 1 hygiene | `J/make_members_u3.py` | `J/members_u3.parquet`, `J/universe_exclusions.csv` | 1 min |
| 2 | signals + fills + exits + capacity | `J/build_candidates_u3.py` | `J/candidates_u3.csv` | ~29 s/day, 410 days |
| 3 | the independent check | `J/verify_rows.py` | `J/verify.md` | 2 min |
| 4 | the 144 declared cells | `J/score_u3.py` | `J/score_u3_tables.md`, `_results.csv`, `_capacity.csv` | ~15 min |
| 5 | the deliverable | — | `J/REPORT.md` + 3 lines in `LOG.md` | — |

**State at 2026-09-17 18:40 UTC**: steps 0-2 running (fetch day ~252/410, builder at 2026-01-15);
**TRAIN is complete, verified (`verify_TRAIN.md`, exit 0) and scored (`score_u3_tables_TRAIN.md`):
0 of 144 cells clear G1**. VAL and TEST are not built. TEST is unread and stays unread until a
`J/FREEZE.md` naming G2 survivors exists — and G1 selected none.

Step 2 **chases** step 0: it only reads days that `U3/fetch_state.json` lists as finished (a day is
finished only after its parquet is closed, its index rows appended and the state saved — `U3/README.md`),
and it is resumable per day via `J/build_u3_state.json`. It is ~2x faster than the fetch, so it idles.

---

## Resume commands (all idempotent — safe to re-run at any point)

**0. The U3 fetch** (if `pgrep -af fetch_u3.py` comes back empty and
`U3/fetch_state.json` has fewer than 410 days):

```bash
setsid nohup bash -c "cd /home/ec2-user/onemil; ulimit -v 1300000; nice -n 10 \
  python3 research/fuckup_audit/U3/fetch_u3.py >> research/fuckup_audit/U3/fetch.log 2>&1; \
  echo EXIT=\$? >> research/fuckup_audit/U3/fetch.log" >/dev/null 2>&1 </dev/null &
```
Check: `tail -2 research/fuckup_audit/U3/fetch.log` · `pgrep -af fetch_u3.py` ·
`python3 -c "import json;print(len(json.load(open('research/fuckup_audit/U3/fetch_state.json'))['days_done']))"`

**1. The daily-context table** (only if `J/members_u3.parquet` is missing or predates ADDENDUM 1 — the
builder refuses to start without the `univ_flag` column):

```bash
nice -n 10 python3 research/fuckup_audit/J/make_members_u3.py
```
Expect exactly: `test tickers dropped: ['ZAZZT', 'ZJZZT', 'ZVZZT'] -> 116 symbol-days` ·
`no_daily_bars tagged: 309 symbols -> 29775 symbol-days` ·
`U3 before 1,512,031 rows / 5,571 symbols -> after 1,511,915 rows / 5,568 symbols`.

**2. The builder — CHASE MODE** (keeps polling `U3/fetch_state.json` for newly finished days; exits
when it has been idle for `J_WAIT_MAX_S` seconds). This is the command to leave running — note the
`export`: `nice` treats a leading `VAR=x` as its command name and dies with `No such file or directory`.

```bash
setsid nohup bash -c "cd /home/ec2-user/onemil; ulimit -v 2600000; \
  export J_WAIT=1 J_WAIT_MAX_S=28800; nice -n 10 \
  python3 research/fuckup_audit/J/build_candidates_u3.py \
  >> research/fuckup_audit/J/build_u3.log 2>&1; \
  echo EXIT=\$? >> research/fuckup_audit/J/build_u3.log" >/dev/null 2>&1 </dev/null &
```
Without `J_WAIT=1` it processes whatever is ready and exits. It NEVER re-does a finished day
(`J/build_u3_state.json`) and the CSV is appended one whole day at a time, so a kill can leave neither
a duplicate nor a half-day. Check:
`tail -2 research/fuckup_audit/J/build_u3.log` · `pgrep -af build_candidates_u3`.
Done when the log's `days left` reaches 0 / the state holds 410 days.

**3. The independent check** (run it on whatever the builder has produced; it does not need all 410 days):

```bash
( ulimit -v 1500000; nice -n 10 python3 research/fuckup_audit/J/verify_rows.py --n 5 ) \
  > research/fuckup_audit/J/verify.md 2>&1; echo EXIT=$?
```
Exit 0 = every drawn row reproduced by the second implementation, every fill obtainable, and the
Stage-E parity clean. Exit 1 prints the failures at the end of `verify.md`.

**3b. Scoring ONE split before the build finishes** (this is how the TRAIN-only result in `REPORT.md`
was produced on 2026-09-17 while the scan was still running). The builder appends one whole day at a
time in day order, so filtering by date gives a file that ends at a complete day even mid-append:

```bash
cd /home/ec2-user/onemil/research/fuckup_audit/J
nice -n 15 awk -F, 'NR==1 || $1 < "2026-01-01"' candidates_u3.csv > candidates_u3_TRAIN.csv
( ulimit -v 1500000; J_SRC=research/fuckup_audit/J/candidates_u3_TRAIN.csv \
  nice -n 15 python3 verify_rows.py --n 8 > verify_TRAIN.md 2>&1 ); echo EXIT=$?
( ulimit -v 2500000; export J_TAG=_TRAIN J_SRC=research/fuckup_audit/J/candidates_u3_TRAIN.csv; \
  nice -n 15 python3 score_u3.py --perm 300 > score_TRAIN.log 2>&1 ); echo EXIT=$?
rm -f candidates_u3_TRAIN.csv          # 1.8 GB; `pop_j_TRAIN.parquet` is the scored population
```
Re-runs of the scorer on the same split take seconds with `J_REUSE_POP=1` (it skips the CSV pass and
the availability audit and reads `pop_j_<TAG>.parquet`). The scorer needs `ulimit -v 2500000` on a
full split: at 1.5 GB pyarrow aborts with *"cannot allocate memory for thread-local data"*.

**4. The scorer** (needs the WHOLE build — the gates are split statistics):

```bash
setsid nohup bash -c "cd /home/ec2-user/onemil; ulimit -v 1500000; nice -n 10 \
  python3 research/fuckup_audit/J/score_u3.py --perm 500 \
  > research/fuckup_audit/J/score.log 2>&1; \
  echo EXIT=\$? >> research/fuckup_audit/J/score.log" >/dev/null 2>&1 &
```
TEST stays UNREAD. Only after a freeze is written (`J/FREEZE.md`, naming the G2 survivors), re-run with
`J_READ_TEST=1`.

---

## Files

| file | what |
|---|---|
| `PREREG.md` | the pre-registration — universe, families, fills, exits, costs, book, gates, cells |
| `make_members_u3.py` | builds the members table and applies ADDENDUM 1's universe hygiene |
| `members_u3.parquet` | 1,511,915 U3 symbol-days with their causal daily context (prior close/high/low, gap, adv20, dvol20_med) plus `univ_flag` |
| `universe_exclusions.csv` | the 312 excluded symbols and why — 3 test tickers dropped outright, 309 `no_daily_bars` tagged and excluded from every scored cell |
| `build_candidates_u3.py` | pass 1 — one row per signal, long AND short, both stores + sqlite read as one tape |
| `candidates_u3.csv` | the signals (~19K rows/day, ~98% filled, ~7 MB/day); `scan` = first \| keep, `univ_flag` = ok \| no_daily_bars |
| `pop_j.parquet` | the scorer's compact scoreable population (written by `score_u3.py`; `J_REUSE_POP=1` re-uses it instead of re-reading the CSV) |
| `build_u3_state.json` | the resume state (finished days) |
| `coverage_u3_missing.csv` | every U3 key with no usable tape — nothing is silently dropped |
| `verify_rows.py` / `verify.md` | the independent check (PLAN §1) |
| `score_u3.py` | the 126 declared cells, the gates, capacity, tails, permutation |
| `score_u3_tables.md` / `_results.csv` / `_capacity.csv` | the scorer's output |
| `REPORT.md` | the deliverable |

## Node rules that apply to every command above

ONE heavy python process at a time, always `nice -n 10`, always `ulimit -v`. A live trader runs on this
node and it has frozen twice under parallel jobs. Never pipe a long job through `| tail/head/grep`.
Every CSV read in this tree uses `keep_default_na=False, na_values=['']` (the ticker `NA`).
