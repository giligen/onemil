# RESULT -- cells 1,599-1,606 (v3, Amendment 3: Databento OPRA cbbo-1m NBBO)
# Updated 2026-09-30: `_OPEN_CREDIT` defect fix + `no_partner` VOID diagnosis (see "Defects" below).
# Rebuild/refuter re-run NOT completed this pass -- see "Not completed" at the end.

## TRAIN (all 8 cells)

|   cell |   delta | mgmt   |   gate |   n_cycles |   void_share |   mean_monthly_ret_on_B |   monthly_sharpe |   green_month_share |   worst_month_usd |   max_dd_usd |   win_rate |
|-------:|--------:|:-------|-------:|-----------:|-------------:|------------------------:|-----------------:|--------------------:|------------------:|-------------:|-----------:|
|   1599 |     0.2 | A      |      0 |         57 |     0.109375 |              0.0306968  |         1.65005  |            0.823529 |              -854 |         1707 |   0.929825 |
|   1601 |     0.2 | B      |      0 |         57 |     0.109375 |              0.0306968  |         1.65005  |            0.823529 |              -854 |         1707 |   0.929825 |
|   1600 |     0.2 | A      |      1 |         23 |     0.08     |              0.00970136 |         0.738847 |            0.470588 |              -853 |          862 |   0.869565 |
|   1602 |     0.2 | B      |      1 |         23 |     0.08     |              0.00970136 |         0.738847 |            0.470588 |              -853 |          862 |   0.869565 |
|   1603 |     0.3 | A      |      0 |         60 |     0.104478 |              0.0251222  |         0.632458 |            0.764706 |             -2505 |         4019 |   0.85     |
|   1605 |     0.3 | B      |      0 |         60 |     0.104478 |              0.0251222  |         0.632458 |            0.764706 |             -2505 |         4019 |   0.85     |
|   1604 |     0.3 | A      |      1 |         24 |     0.172414 |              0.00487783 |         0.20417  |            0.411765 |             -1514 |         2328 |   0.875    |
|   1606 |     0.3 | B      |      1 |         24 |     0.172414 |              0.00487783 |         0.20417  |            0.411765 |             -1514 |         2328 |   0.875    |

## Selected: cell 1599 (delta=0.2, mgmt=A, gate=0)

WARNING/fallback/VOID counters (TRAIN, post-fix): {'mark_missing': 6338, 'exit_quote_missing': 646, 'void_no_strike': 292, 'void_no_quote': 0, 'void_no_settlement': 6, 'open_credit_missing': 0}

## Defects found and fixed/diagnosed (2026-09-30)

### 1. `_OPEN_CREDIT` key-miss defect -- FIXED, but numerically INERT on this cache
`run_cycle` (cell_1599.py) read `_OPEN_CREDIT.get((short_sym,long_sym,entry_d), mark+1)` /
`(..., mark-1)` for the stop/profit checks. On a genuine key miss that default made both checks
compare `mark` to a value derived from `mark` itself -- structurally unsatisfiable for ordinary
marks -- so the cycle fell through to `expiry_no_trigger`, priced by `run_cell` through
`intrinsic_settlement` IDENTICALLY to how Management B prices `expiry_intrinsic`. **Fixed**: a
sentinel (`_NO_CREDIT`) now detects a real miss; on a miss the cycle is VOID for management
purposes (`void_reason='open_credit_missing'`, logged as ERROR, counted in `warn_counter`), never
silently defaulted. Three new unit tests added in `research/options_vrp/tests/test_open_credit_void.py`
(24/24 tests pass, including the full existing `test_cell_1599.py` suite): a synthetic cycle
reproduces the collapse mechanism (same mark/credit numbers that trigger `profit_target` when the
key IS present) and asserts the fix VOIDs instead of collapsing to B; a hit-path regression guard;
a static source guard against the `mark +/- 1` pattern reappearing.

**Re-run result: `open_credit_missing = 0`** across the full TRAIN run -- this key never actually
misses under the current single-threaded `run_cell` control flow (the dict is written immediately
before `run_cycle` is called with the identical tuple). Cell 1599's TRAIN table, cycle-by-cycle
`exit_reason` distribution, and every downstream stat are **byte-identical before and after the
fix** (verified). The fix is a correct, now-tested defensive fix (and forecloses the failure mode
for any future caller of `run_cycle` that does not go through `run_cell`'s credit-setting step,
e.g. `rebuild_1599.py` if it is ever refactored to call it) -- but it is **not** the cause of the
observed Management-A-collapses-to-B pattern in the current data. That pattern's real driver is
`mark_missing` = 6,338: refuter_1.md defect #1 (`fetch_dbn.py` drops `ts_recv`, keying the 15:59
mark lookup on the staler `ts_event` instead), which starves the stop/profit/dte_21 checks of a
usable mark on most sessions and is a **separate, out-of-scope defect this task did not fix**.

### 2. `no_partner` VOID cause -- 100% plan gap, 0% key/naming defect
Replayed `build_ladder`/`select_strikes` directly against the cached `mondays.parquet` (read-only,
no fetch) for every TRAIN+VAL entry Monday at width=$10:

| Split | Delta | Mondays | never_pulled (plan gap) | pulled-but-unmatched (key/naming) |
|---|---|---|---|---|
| TRAIN | 0.20 | 73 | 8 (11.0%) | 0 |
| VAL   | 0.20 | 59 | 11 (18.6%) | 0 |
| TRAIN | 0.30 | 73 | 8 (11.0%) | 0 |
| VAL   | 0.30 | 59 | 3 (5.1%) | 0 |

Every `no_partner` VOID is a **plan gap**: the K-10 partner strike has no row at all in
`mondays.parquet` for that Monday (0 cases had a cached-but-unusable row: bad mid or non-convergent
IV, and 0 cases were a genuine lookup/key bug -- `np.isclose` on exact half-dollar/dollar strikes
matches correctly whenever the row exists). `select_strikes`'s lookup code is correct as written;
**no code fix applied** (none needed). The void stays. Closing the 20-delta gap needs exactly 19
new single-leg full-life pulls (8 TRAIN + 11 VAL, one missing partner strike per affected Monday;
dates/strikes enumerated in `/tmp/.../scratchpad/diagnose_no_partner.py` output). Per-leg full-life
cost from `opt_cache/dbn/spend.json`'s 4,006 single-leg `cbbo-1m` purchases: median **$0.00205**,
mean $0.00956 (mean pulled up by a handful of long-duration outliers, max $0.185). 19 legs at the
median $\approx$ **$0.04 total**. **No purchase was made** -- this is the owner's decision.
EXTENSION's much larger void share (below) is dominated by a different, already-documented cause
(SPY spot gaps 2013-2015, refuter_1.md defect #4) and was not re-diagnosed here (out of this task's
20-delta-grid scope).

## Per-split headline, cell 1599 (delta=0.2, mgmt=A, gate=0) -- post-fix, PREREG's bar

| Split | n_cycles | void_share | mean %/mo on B | monthly Sharpe | t (mean/SE, n_months) | green months |
|---|---|---|---|---|---|---|
| TRAIN | 57 | **10.94%** | +3.07% | 1.65 | 1.96 (n=17) | 14/17 (82.4%) |
| VAL | 43 | **20.37%** | +3.57% | 2.32 | 2.50 (n=14) | 12/14 (85.7%) |
| EXTENSION | 293 | **42.66%** | +0.89% | 0.61 | 2.01 (n=129) | 72/129 (55.8%) |

MDE not separately computed: Amendment 3's own rule VOIDs a cell on void_share > 10% before any
power/effect-size question is reached, and all three splits fail that rail (TRAIN only marginally,
VAL and EXTENSION by a wide margin) -- these numbers are unchanged from the pre-fix run (confirmed
identical) because neither fix altered any scored cycle: the `_OPEN_CREDIT` fix never fires on this
cache, and the `no_partner` void has no available code fix.

## Rebuild agreement / refuter re-run -- NOT completed this pass
Given the task's ~50 tool-call budget, this pass completed the code fix + unit tests (item 1), the
no_partner diagnosis (item 2), and one full TRAIN+VAL+EXTENSION re-run of `cell_1599.py` (item 3
first half, confirming the fix is inert on this cache). It did **not** complete: re-running
`rebuild_1599.py` with the newly-added `fetch_leg_life` cache-fallback active (the fallback and its
`--allow-fetch` spend guard were added and verified by inspection to funnel every Databento call
through one gated choke point (`guarded_range`, defaults `ALLOW_FETCH=False`), but the rebuild
itself was not re-run), the two pre-registered refuters, or a trade-by-trade builder-vs-rebuild
comparison. `REBUILD_1599.md` and `review/1599_refuter_1.md`/`1599_refuter_2.md` in this directory
are therefore **still the pre-fix versions** and should not be read as reflecting the fixes above.
This is flagged rather than papered over per this project's independent-check rule -- the next pass
should run `python3 rebuild_1599.py --stage report` (no `--allow-fetch`) and the two refuters before
any number from this RESULT is relayed further.

## Verdict
**Still VOID** by Amendment 3's own void-share rail in every split (TRAIN 10.94%, VAL 20.37%,
EXTENSION 42.66%, all > 10%) -- unchanged by either fix, because neither fix altered a scored
cycle. The exploration-tier line does **not** apply: it requires the rail to pass, and it does not,
in any split, including EXTENSION where the +0.89%/mo point estimate lives. The builder defect
(`_OPEN_CREDIT`) is real, now fixed and unit-tested, but was never the cause of this cell's result.
The `no_partner` void is a genuine, currently-unfixable data gap (plan, not code), closeable for
~$0.04 (20-delta, 19 legs) pending owner go-ahead -- closing it would lower void_share by roughly
11 points at 20-delta but EXTENSION's 42.66% has a separate, larger, undiagnosed-here cause, so
closing only the 20-delta TRAIN/VAL gap would not by itself pass the rail everywhere.
