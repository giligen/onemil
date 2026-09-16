# Stage B — the pass-1 rebuild: both fills, the level's history, MAE/MFE, the H1 stop matrix, the H9 families

Executed 2026-09-16 per `research/fuckup_audit/PLAN.md` §4 row B (H2 resting fill · H1 stop placement · H9 new
families · H7's corrected cost inputs · H4's feature carry). Stage B **builds and pre-registers**; it claims no book.
Everything written is under `research/fuckup_audit/B/`; everything outside it was read only (`data/cache.db` via
`file:...?mode=ro`). No config, service, cache or order was touched.

Output: **`research/fuckup_audit/B/candidates4.csv`** — one row per (day, symbol, family, config) SIGNAL, with the
fill columns of a model that does not fill left EMPTY. The scorer contract is stated in full at the top of
`build_candidates4.py` and is repeated in code in `score5.py`; if the two ever disagree, the builder's header wins.

---

## 0. Why the file had to be rebuilt at all

`research/bf_zero2/candidates3.csv` contains **only signals whose next-bar open came back under `level × 1.006`**.
It is conditioned to hold cheap fills, so it cannot answer H2 (does a resting stop-limit at the level do better?) —
`probe_stops.md` §3 measured that on 45 re-detected days: the resting model books **+30% more trades** and the whole
gain is the 992 signals the next-open convention silently threw away. It also carries five features and no level
history, no MAE/MFE and one stop geometry, so H1, H4 and H9 are all unanswerable on it.

`candidates4.csv` fixes exactly that and nothing else: same universe, same tape, same causality rules, same detector
code (`research/bf_zero/build_candidates.py` is imported, not re-implemented).

---

## 1. PRE-REGISTRATION — the cells Stage C will look at (written before any Stage C number exists)

**Families — 14 configs.** Eight carried from earlier stages, six new (H9 / PLAN §3 H9). Each is defined causally:
every field a decision reads is computed on bars at or before the signal bar.

| # | family | config | level | stop | signal |
|---|---|---|---|---|---|
| 1 | F1 bull flag | `{"P":0.12}` | flag high | flag low | high reaches the level |
| 2 | F5 HOD break | `{"K":5,"X":0.04}` | running HOD | consolidation low | high reaches the level — **reference only** (gross-negative at ZERO cost, t -7.9, `probe_stops.md` §6) |
| 3 | F6 red-to-green | `{}` | prior close | day low so far | high reaches the level |
| 4-6 | F8 opening-range break | `{"N":5}` `{"N":15}` `{"N":30}` | max high of the first N bars | min low of the first N bars | high reaches the level |
| 7 | F9 gap-and-go | `{"G":0.05}` | 5-bar range high (gap >= 5%) | 5-bar low | high reaches the level |
| 8 | F10 VWAP reclaim | `{}` | the reclaim bar's high | low of the below-VWAP stretch | high reaches the level |
| 9-10 | **F11 close confirmation** | `{"base":"F8","N":15}` · `{"base":"F6"}` | the base level | the base stop (F6: running low) | the break bar must **CLOSE** at/above the level |
| 11-12 | **F12 retest** | `{"base":"F8","N":15}` · `{"base":"F6"}` | the base level | the retest bar's low | after the base break, a bar's low comes within 0.3% of the level within 30 min and the NEXT bar's low holds at/above it — that hold bar is the signal |
| 13 | **F13 sweep-and-reclaim** | `{"K":5,"X":0.04}` | the F5 consolidation low | the sweep low | the consolidation low is pierced by <= 1%, then a bar CLOSES back above it within 5 bars |
| 14 | **F14 second break** | `{"N":15}` | the F8-15 level | lowest low from the stop bar through the signal bar | the first break filled and STOPPED OUT under the 2R exit; the next break of the same level that day |

**Fills — 2.**
`entry_next = o[i+1]` iff `<= level*1.006` (the shipped HOD engine, which reacts after the bar closes) ·
`entry_rest = max(o[i], level)` iff `<= level*1.006` (an ORB-style resting stop-limit at the level).
F11/F12/F13 are **close-triggered** — the decision needs the bar's close, so no resting order can express them and
`entry_rest` is empty for those three **by construction, not by failure**. That is 5 of the 14 configs x 5 outcomes
= 25 impossible cells.

**Stop variants x exits — 5 outcome columns per fill** (the H1 cross, computed inside the walk):

| outcome | stop | exit |
|---|---|---|
| `2R close-fill` | structural, on a TOUCH | +2R on a bar CLOSE, flat 15:55 |
| `hold-to-close` | structural, on a TOUCH | flat 15:55, -1R stop |
| `lock 1.75/0.5` | structural, then the ORB static lock (a closed bar's high at +1.75R arms it from the NEXT bar, stop -> +0.5R) | hold to close, no target |
| `2R close-stop` | H1 **S1**: a bar CLOSE at/below the level, filled at the NEXT bar's open x0.999 | +2R close-fill |
| `2R stop-1%` | H1 **S4**: structural - 1% of price, **R redefined** (`r_pct_m1`), so the comparison is at constant $ risk | +2R close-fill |

**CELL COUNT: 14 x 2 x 5 = 140, minus 25 impossible = 115 cells.** A second, separately counted pass restricted to
`entry_m >= 600` (Stage A's 10:00 recommendation) is `score5.py --min-entry-m 600` and adds **115 more**, for a Stage
B+C declared total of **230**. Two cost sensitivities (`--free-target`, `--legacy-spread`) are re-scores of the same
cells and are reported as sensitivities, never as additional gate candidates.

**Population, cost, book and gate** are `build_candidates4.py`'s header contract verbatim (= Stage A's adopted
contract (c)): price >= $5 on the FILL, entry <= 14:01, R >= 1% of price **of the variant being scored**,
`range_so_far_pct >= 5` for every family except F1-F4; `half_cc = 0.5*(spread_cc_bps/100)/max(r_pct,0.05)`, entry
0.25x (next-open) or 1.00x (resting), exit stop/lock 0.875x · eod 0.412x · target 0.875x (0 only under
`--free-target`); `run_book(rows, 12, 4)`; G1 TRAIN mean net R > 0, t >= 2.0, >= 5 trades/week; G2 VAL mean > 0,
t >= 1.0, >= 55% weeks green, bar raised by 1 SE of weekly R per 10 G1 survivors; **TEST is computed only with
`SCORE5_READ_TEST=1`**, after the selection is frozen in writing.

**Decision rules carried from the PLAN.**
- H2: the resting fill replaces next-open as the reference fill only if >= 95% of its fills pass obtainability and the
  queue check, and the improvement holds on TRAIN **and** VAL.
- H1: a stop variant is adopted only if the paired difference vs the touch stop is >= +0.05R with t >= 2 on TRAIN, the
  sign agrees on VAL, the stop rate falls, and the new stop's fill is obtainable.
- Anything clearing G2 additionally gets the search-adjusted permutation p over all 115 cells, the tail test
  (top 1% / top 5% removed, winners capped at +3R) and a per-month table.

---

## 2. Smoke test and parity — PASSED, exactly

`parity_smoke.py` compares, on the smoke days, every `candidates3.csv` row of the five families the two files share
(`F1 {"P":0.12}`, `F6 {}`, `F8 {"N":5|15|30}`) against `candidates4`'s next-open fill.

| window | candidates3 rows | rows with no candidates4 fill | max abs diff `entry` vs `entry_next` | max abs diff `rr_2r` vs `rr_2r_next` | `entry_m` / `exit_m` / `why` mismatches |
|---|---|---|---|---|---|
| 2025-01-02, 01-03, 01-06 (`_smoke`) | 7,052 | 0 | **0.0** | **0.0** | 0 / 0 / 0 |
| 2025-10-21, 10-22, 10-23 (`_smoke2`) | 6,178 | 0 | **0.0** | **0.0** | 0 / 0 / 0 |

Both windows are exact, not within 1e-6. On the same days `candidates4` carries **2,721 / 2,216 additional signal
rows** with no `candidates3` counterpart (86% / 69% of them have no next-open fill at all) — that is precisely the
selection `candidates3` could not see, and the reason H2 was unanswerable on it.

**One bug the parity check caught and the report must record.** The first draft precomputed the 15:55 bar index once
per symbol-day and used it only when it was at or after the walk's start bar; `build_candidates3.walk` instead scans
forward from the start bar, so a signal that is *itself* past 15:55 exits on its own first walked bar. 20 of 7,052
rows (all with `sig_m >= 951`, all outside the scoring population's 14:01 cut) disagreed. Fixed by taking
`max(eod_idx, k0)` — minutes are sorted, so that is the first bar at/after 15:55 in the walk range. Both windows
then matched exactly. Cheap to find, and it is the same class of defect (a precomputed index used outside its
window) as the CSV column-misalignment bug `probe_stops.md` §9 documents.

**Two structural facts the smoke already fixes in writing, so Stage C cannot be surprised by them:**

1. **F12's base stop is structurally < 1% of price.** The retest bar's low is within 0.3% of the level by
   construction, so `r_pct` for F12 has a hard ceiling near 0.9% (observed max 0.858%, median 0.36% over 992 fills on
   the 2025-10 window). Under PLAN §1's `R >= 1%` floor F12 is therefore scoreable **only through the `2R stop-1%`
   variant** (`r_pct_m1 = r_pct + 1%`), which is exactly H1's lever. This is a property of the PLAN's own F12
   specification, not a defect; it is recorded here so that "F12 has no trades" is read correctly in Stage C.
2. **The resting fill's queue check bites.** `rest_obtain` is **1.000** on every resting fill (the level is inside the
   signal bar by construction, as `probe_stops.md` §3 measured), but `rest_queue_ok` (signal-bar volume >= 5x the
   shares $100 of risk buys) is only **0.56-0.88 depending on the family** (F5 0.563, F6 0.566, F10 0.599, F1 0.629,
   F8 0.688, F14 0.712, F9 0.880 on the 2025-10 window). H2's decision rule requires >= 95% of fills to pass
   obtainability **and** the queue check; on this evidence the queue check alone will fail that bar for most
   families, so Stage C must report the resting cells **both** unrestricted and restricted to `rest_queue_ok == 1`,
   and the honest H2 claim is about the restricted set.

---

## 3. What is in a row (the columns Stage D will select on)

Keys and context (all fill-independent and known at the signal bar): `day, symbol, fam, cfg, sig_m,
minutes_since_open, level, stop, price, dist_open_pct, range_so_far_pct, rv_adv, gap_pct, adv20, spread_pct,
spread_cc_bps`.

> **`price` is the LEVEL, not a fill.** `candidates3.csv`'s `price` was the next-open fill; use `entry_next` for
> that. Making `price` fill-independent is what lets a single row carry two fill models and a no-fill state.

Signal bar: `sig_o, sig_h, sig_l, sig_c, sig_v`.
Level / consolidation history: `n_touches` (bars before the signal whose high came within 0.2% of the level without
exceeding it), `consol_bars` (bars since the level was set, per family), `consol_vol_ratio` (signal-bar volume / mean
volume of the consolidation bars), `cum_dollar_vol` (open->signal), `vwap_dist_pct` (level vs session VWAP at the bar
*before* the signal), `close_confirm`, `pm_dollar_vol` (04:00-09:29 where the tape has it), `prev_day_range_pct`,
`prev_close`, `asset_class`.
Fills: `rest_obtain`, `rest_queue_ok`, and for each of `next_`/`rest_`: `entry, entry_m, r_pct, r_pct_m1` plus the
five outcome triples (`rr_`, `why_`, `exit_m_`) and `mae_pct`, `mfe_r`, `pnl100_2r_stopm1`.

**Known limitations of three columns, stated now rather than discovered later:**
- `asset_class` (stock / wrapper / unknown) comes from the 2026-07-11 offline ORB asset dump plus the leveraged-family
  sets (`trading/orb_asset_class.py`). It is a **symbol-level, not point-in-time** attribute. Over the finished file it
  splits **82.4% stock / 13.6% wrapper / 4.0% unknown**. This is the "wrapper flag from the pass-2 outputs" the brief
  asked for; `research/bf_zero2/pass2b.py`'s own flag was built the same way.
- `pm_dollar_vol` is **empty on 52.9% of rows** — `bars_sip.db` simply has no pre-market bars for those symbol-days.
  Any Stage D feature built on it must treat "missing" as its own bucket, never as zero.
- `adv20`, `rv_adv`, `gap_pct`, `prev_close`, `prev_day_range_pct` are empty on the **first five trading days of
  2025** (the universe's own ADV20 warm-up and the daily panel's first row), and on ~0.2-0.8% of rows thereafter.
  The smoke-test window 2025-01-02..06 shows 100% empty `adv20` for that reason; the mid-window check
  (2025-10-21..23) shows 0.8%. Not a defect — a warm-up.

---

## 4. Run status

| item | value |
|---|---|
| launched | 2026-09-16, detached: `setsid nohup ... ulimit -v 3500000; nice -n 10 python3 research/fuckup_audit/B/build_candidates4.py > research/fuckup_audit/B/build4.log` |
| days | 420 (2025-01-02 -> 2026-09-04), the `research/bf_zero/universe.csv` point-in-time >=5%-range list |
| rate | 19.8 s/day -> 138.5 min total; 7,380 signal rows/day -> **3,099,499 rows, 1.99 GB** |
| resumable | `build4_state.json` holds the finished days; the CSV is appended with the PINNED `COLS` list |
| memory | `build_candidates.py`'s import of the 5.05M-row Databento daily panel needed > 3.5 GB. It is now read with the five columns it uses and filtered to the universe's own symbols (2.98M x 5), and the panel is dropped after the merges. **No family function, no fill and no merged value changes** — the parity table in §2 is the proof. |

**COMPLETE — `DONE` + `EXIT=0` at 2026-09-16 22:37 UTC.** 420 of 420 days (2025-01-02 -> 2026-09-04),
**3,099,499 signal rows**, 138.5 min, 19.8 s/day, 1.99 GB. Verification below. No Stage C number exists yet: the
scorer has been run only as a machinery smoke test on the first days.

**How to check it:** `tail -2 research/fuckup_audit/B/build4.log`. It finishes with `DONE` then `EXIT=0`.

**How to resume it** if it died (`EXIT` non-zero, or no new log line for 10+ minutes with no process in
`ps aux | grep build_candidates4`): fix whatever the traceback says and relaunch the SAME command — the state file
makes it resume at the first unfinished day and the CSV is appended, not rewritten. Do **not** delete
`candidates4.csv` without also deleting `build4_state.json`, or the file will be missing the days the state claims
are done. `B4_TAG=_x BFZ_DAYS=N` runs a throwaway N-day build into its own files.

### 4a. Verification of the finished file — all four checks passed

| check | result |
|---|---|
| days | `build4_state.json` holds **420** days, 2025-01-02 -> 2026-09-04 — the whole universe |
| rows | `wc -l` = 3,099,500 = 3,099,499 data rows + 1 header, matching the log's own running total exactly |
| column alignment | the file's header is **identical** to the builder's pinned `COLS` list, 77 columns (this is the `probe_stops.md` §9 misalignment trap; it is closed) |
| **parity, FULL 420 days** (not the 3-day smoke) | **913,985** `candidates3` rows of the five shared families, **0** with no `candidates4` fill, `entry` vs `entry_next` and `rr_2r` vs `rr_2r_next` both **max abs diff 0.0**, and 0 mismatches on `sig_m`, `entry_m`, `exit_m_2r` and `why_2r`. `candidates4` adds **290,719** signal rows candidates3 could never contain (77.5% of them with no next-open fill at all). |
| hand recomputation | `verify_rows.py 5`: five random rows (F5, F13, F8 N=15, F6), bars reloaded from the stores, every field recomputed with a plain Python loop instead of the builder's vectorised helpers — **ALL MATCH**, including a correctly-empty `next_entry` on a signal the cap refused |
| coverage | `coverage4_missing.csv` is **byte-identical** to `bf_zero2/coverage3_missing.csv` — the same symbol-days were unserved by the tape, so the two files see exactly the same universe |

(The 2025-10 `_smoke2` intermediate CSV was deleted once the full-file parity superseded it; its result is the one
recorded in §2 and it was produced by the same code path.)

### 4b. Whole-file build diagnostics (descriptive; NOT scored cells)

| family-config | signals | next fill rate | rest fill rate | rest obtainable | rest queue-OK |
|---|---|---|---|---|---|
| F1 `{"P":0.12}` | 11,517 | 0.531 | 0.846 | **1.0000** | 0.609 |
| F5 `{"K":5,"X":0.04}` | 366,864 | 0.815 | 0.948 | **1.0000** | 0.574 |
| F6 `{}` | 180,084 | 0.680 | 0.820 | **1.0000** | 0.602 |
| F8 `{"N":5}` | 391,664 | 0.802 | 0.937 | **1.0000** | 0.650 |
| F8 `{"N":15}` | 333,543 | 0.853 | 0.966 | **1.0000** | 0.725 |
| F8 `{"N":30}` | 287,896 | 0.877 | 0.978 | **1.0000** | 0.774 |
| F9 `{"G":0.05}` | 22,543 | 0.729 | 0.954 | **1.0000** | 0.878 |
| F10 `{}` | 286,385 | 0.895 | 0.961 | **1.0000** | 0.568 |
| F11 F8-15 / F6 | 323,753 / 174,263 | 0.810 / 0.653 | — close-triggered | — | — |
| F12 F8-15 / F6 | 156,937 / 67,566 | 0.675 / 0.601 | — close-triggered | — | — |
| F13 `{"K":5,"X":0.04}` | 477,516 | 0.706 | — close-triggered | — | — |
| F14 `{"N":15}` | 18,968 | 0.880 | 0.971 | **1.0000** | 0.754 |

**The H2 obtainability half of the decision rule is settled on the full 420 days: the resting fill is obtainable
inside the signal bar on 100.000% of fills, on every family.** It also books far more signals than the engine's
next-open convention — +0.10 to +0.32 of fill rate depending on family (F1 0.531 -> 0.846, F6 0.680 -> 0.820,
F5 0.815 -> 0.948) — which is the +30% `probe_stops.md` §3 measured on 45 days, confirmed on all 420.
**The queue half is not settled in H2's favour**: the 5x-volume check passes on only **0.57-0.88** of resting fills
(F10 0.568, F5 0.574, F6 0.602, F1 0.609, F8 0.650/0.725/0.774, F14 0.754, F9 0.878), far below the >= 95% the PLAN
demands. Stage C must therefore report every resting cell twice — unrestricted and restricted to
`rest_queue_ok == 1` — and the honest H2 claim belongs to the restricted set.

Column coverage over the whole file: `spread_cc_bps` **0.00%** empty, `adv20` **1.81%**, `pm_dollar_vol` **52.9%**
(no pre-market bars in `bars_sip.db` for those symbol-days — a missing bucket, never a zero).
`asset_class`: **82.4% stock / 13.6% wrapper / 4.0% unknown**.

---

## 5. Stage C — how to run it

```bash
ulimit -v 1500000; nice -n 15 python3 research/fuckup_audit/B/score5.py            # the 115 pre-registered cells
ulimit -v 1500000; nice -n 15 python3 research/fuckup_audit/B/score5.py --min-entry-m 600   # +115, Stage A's 10:00
ulimit -v 1500000; nice -n 15 python3 research/fuckup_audit/B/score5.py --free-target       # sensitivity (c')
ulimit -v 1500000; nice -n 15 python3 research/fuckup_audit/B/score5.py --legacy-spread     # score4's band table
# ONLY after the selection is frozen in writing:
SCORE5_READ_TEST=1 ulimit -v 1500000; nice -n 15 python3 research/fuckup_audit/B/score5.py --perm 200
```

Outputs `score5_tables.md` + `score5_results.csv` (suffixed per variant). The scorer was smoke-tested end to end on
the first 8 days of the partial build (115 cells declared, 107 with a book, G1 0 — meaningless numbers on 8 days,
run only to prove the machinery). It has **not** been run on the partial file for any result, and TEST has not been
read.
