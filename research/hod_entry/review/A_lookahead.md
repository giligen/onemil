# LENS A — Look-ahead audit of cell 1,427's E1 resting-stop-limit claim

Adversarial trace of every quantity the fill decision and population depend on. Verdict: **one confirmed,
quantifiable look-ahead** (the rv admission filter), already named by the claim's own judge in PREREG_1438.
Everything else traced causal. The leak does not reverse the sign of the edge in a first-pass check, but it
does mean the reported population (n=3,580 fills, +0.285/+0.238/+0.330 R) is NOT what a live resting order would
have produced, and the one study designed to answer that (cell 1,438) has not finished.

## 1. CONFIRMED LEAK — the rv admission filter (mechanism + quantified)

`trading/hod_break.py::detect()` (pre-fix version, still what built the population under review):
```
cumv = np.cumsum(v)
...
level = float(hod[i - 1])          # causal: HOD through bar i-1
if h[i] < level: continue          # break bar i: price crosses INSIDE this bar
stop = consolidation_low(l, h, i - 1, p)   # causal: through i-1
...
rv = rv_profile(float(cumv[i]), adv20, int(m[i]))   # cumv[i] = FULL volume of bar i, the break bar itself
if not (p.rv_lo <= rv < p.rv_hi): continue
return HodBreakSignal(bar_idx=i, ...)
```
`cumv[i]` is cumulative volume THROUGH bar `i` **inclusive** — the break bar's own complete volume. The claim's
own fill mechanism (`research/hod_entry/PREREG_1427.md`, `sip_rebuild.py::simulate_entry`) fills INSIDE that same
bar `i` at the first tape print ≥ trigger (`bb = trades[(trades.ts >= S_ns - 60e9) & (trades.ts < S_ns)]`, window =
bar i). So the admission gate for a signal is evaluated with volume data — including trades that occur AFTER the
fill instant, within the same bar — that a live resting order cannot have when it decides to arm/hold. This is
exactly the gap `PREREG_1438.md` names ("`cumv[i]` ... is not known when a resting order fills mid-bar at the
first cross") and it is real: confirmed by direct code read, not inference.

Level, consolidation/stop, and distance-from-open all use bars strictly before `i` (`hod_level(bars, break_m)`
in `sip_rebuild.py` filters `bars.m < break_m`; `consolidation_low(l, h, j=i-1, ...)`) — causal. **rv is the only
look-ahead quantity in the admission decision.**

### Quantification (this audit, not reused from any prior run)
Script: recomputed `rv_used = rv_profile(cumv[i], adv20, m[i])` (what 1,427's population was actually filtered
on) vs `rv_causal = rv_profile(cumv[i-1], adv20, m[i-1])` (PREREG_1438's arm-at-close-of-prior-bar rule) for
every "fill" row in `research/hod_entry/sip_rebuild_val.csv` + `sip_rebuild_test.csv`, using `data/cache.db`
1-min bars (read-only) and `research/bf_zero/universe.csv` adv20. `rv_lo=1.0, rv_hi=5.0` from
`HodBreakParams` defaults (matches `causal_arming.log`'s params line).

| | n | mean net_R |
|---|---|---|
| all 1,427 fills (VAL+TEST pooled, scoreable) | 3,366 | +0.314 |
| **mis-armed** (rv_used in-band, rv_causal NOT in-band — should never have been admitted for a resting order) | **365 (10.8%)** | **+0.154** |
| still causally armed (rv_causal also in-band) | 3,001 (89.2%) | +0.334 |

By part: TEST 907 fills, 114 mis-armed (12.6%, mean +0.198) vs 793 still-armed (mean +0.391); VAL(+TRAIN-H2)
2,459 fills, 251 mis-armed (10.2%, mean +0.135) vs 2,208 still-armed (mean +0.313). (214 of 3,580 total fills
dropped for missing cache.db minute bars at i or i-1; 0 dropped for missing adv20.)

**Reading, stated adversarially**: the leak is real and touches ~11% of the reported fills. It does NOT explain
the edge — the mis-armed subset is still net positive (+0.154R), and purging it would *raise* the surviving
mean (+0.334 vs the pooled +0.314), not collapse it. So on this one-bar test, the rv look-ahead looks like noise
dilution, not the source of the +0.238/+0.285/+0.330 result. **This is not the full answer** — see limitations.

### Limitations of this check (adversarial on my own method)
1. This only re-scores signals that ALREADY exist in `b0_trades.csv`/1,427's population, i.e. signals `detect()`
   already selected using the look-ahead rv gate on their own break bar. `detect()` returns the FIRST bar meeting
   all conditions and stops (`for i in range(...): ... return`). If the look-ahead rv filter wrongly rejected an
   earlier bar (a causal arm would have taken it) or wrongly accepted a later bar over an earlier causal-armable
   one, that reshapes WHICH signal is "the" signal per symbol-day — a population-identity change my per-row
   re-score cannot see. Only a from-scratch causal rebuild over the full candidate superset (which is what
   `research/hod_entry/causal_arming.py` — cell 1,438 — is doing) answers this fully.
2. I did not re-verify universe membership causality (prev close ≥ 15, ADV20 trailing window) independently in
   this session — `research/bf_zero/parity_review/step_universe_*.py` already exists for that; not re-run here
   under the 40-call budget. Lower priority: these are slow-moving daily quantities, low risk of intraday leak,
   but NOT independently confirmed by this audit.
3. `m[i]`/`m[i-1]` minute-of-day granularity vs `VP_CHECKPOINTS` step function — immaterial at 1-bar resolution.

## 2. Traced clean (causal, no leak found)

* **HOD level**: `hod_level(bars, break_m)` in `sip_rebuild.py` uses `bars.m < break_m` — strictly before the
  break bar. `detect()`'s `hod[i-1]` — same. Causal.
* **Consolidation test / stop**: `consolidation_low(l, h, j=i-1, p)` — bars `i-K..i-1` only. Causal.
* **min_dist**: `(level/o0 - 1)*100`, both quantities causal (level as above, `o0` = day open, known at 09:30).
* **last_entry_minute**: static clock cutoff on `m[i]`, no data dependency. Causal.
* **Break bar identity**: `h[i] >= level` is a real-time tape event (price actually trades through the level);
  not itself a look-ahead — the leak is specifically that ADMISSION (rv) needs the bar's own full volume, not
  that the crossing condition is unknowable.
* **First-12/day, 4-concurrent slot rule**: first-come by fill-minute order (`run_consol.simulate_slots` per
  REPORT_1427.md footnote) — inherently causal (chronological), no future info in slot assignment. Confirmed the
  12/4 figures match live `config.yaml` (`hod_break: max_per_day: 12 / max_concurrent: 4`, owner-set 9/14), not a
  research-only override — no research/live parameter mismatch.
* **Stop / 2R target**: stop = consolidation low (causal, above); target = fill + 2×(fill−stop), purely a
  downstream arithmetic quantity of already-causal entry/stop. No leak.
* **Exit walk start bar** (`research/hod_exit_lab` path, `simulate_book`'s `p[p.m >= r.entry_m]`): the walk for
  target/EOD starts at `entry_m = break_m + 1`, i.e. the bar AFTER the fill bar — it does NOT reuse the break
  bar's post-fill bar-level OHLC for target evaluation. The ONLY thing checked inside the break bar after the
  fill is a genuine tick-level stop check (`stopped_bb`, real SIP prints after the fill instant) — that is MORE
  precise than a look-ahead, not a leak. Net effect if anything: a 2R touch in the remainder of the break bar,
  before the walk starts, is never credited — this can only UNDERSTATE net R, not inflate it. Flag as a
  completeness gap, not a look-ahead threat to the claim.

## 3. What has already been done about the leak (status, not yet closing the loop)

`PREREG_1438.md` (frozen ~19:40 UTC today) pre-registers exactly this fix: arm at the close of bar `j=i-1` using
`rv_profile(cumv[j], adv20, m[j])`, pass bar on VAL (mean net R ≥ +0.10, t ≥ 2, ex-top-5% > 0, coverage ≥ 80%,
gap ≤ 5pp, ≥3 fills/wk), TEST read once only if VAL passes.

The codebase's working tree (uncommitted) already contains the causal primitives, wired dry-only:
* `trading/hod_break.py` — new `arm_state()` (uses `np.sum(v[:j+1])`, i.e. through bar `j` only, never `j+1` —
  correctly causal) and `resting_entry_fill()`. Diff cites `docs/hod_resting_entry_spec_20260925.md` and a
  parity test `tests/test_hod_resting_entry.py` against `research/hod_entry/causal_arming.py`'s own `arm_state`.
* `trading/hod_break_engine.py` — new `entry_mode='resting_stop_limit'` path, explicitly **DRY-ONLY: "never
  submits an order regardless of `self.dry_run`"** (hardcoded, not a config flag), logs every cross to
  `logs/hod_dry_entry_ledger.csv`, and is honest in its own comments that the live engine lacks a tick stream so
  it resolves crosses with bar-high + a live quote fetched at bar-close ("not tape-accurate", logged WARNING
  each time) rather than the true first-print. No live capital is at risk from this code today.
* `research/hod_entry/causal_arming.py` (cell 1,438, PID 3431964, started 16:15 UTC, still running at the time
  of this audit — 60/230 TRAIN-H2+VAL days done as of 16:26, ~42 min ETA for VAL alone) is walking the FULL
  causal superset (not just re-scoring 1,427's population) exactly to answer limitation #1 above. **No
  `REPORT_1438.md` exists yet. VAL has not passed. TEST has not been opened for this cell.**

## Bottom line

The look-ahead the claim's own judge flagged is real, mechanistically confirmed by code, and quantified here at
~11% of reported fills (mean +0.154R, i.e. still-positive, and excluding them raises the surviving mean to
+0.334R) — on the population-identity caveat above, which this one-bar check cannot fully resolve. Every other
traced quantity (level, consolidation/stop, min_dist, last_entry_minute, slots, target, exit-walk start) is
causal. The engineering fix for the leak already exists in the working tree and is dry-only / no live orders.
**What has not happened yet is the validation**: cell 1,438's VAL pass bar, pre-committed before any of its
numbers were seen, has not been cleared, and TEST for 1,438 is sealed. Per this repo's own testing protocol
(CLAUDE.md §"No research claim ships without an independent check"), the claim under review — cell 1,427's
+0.285/+0.238/+0.330 R — is not the number a causal live order would earn; cell 1,438 is that number, and it
is roughly 30-40 minutes from a VAL verdict as of this write-up.
