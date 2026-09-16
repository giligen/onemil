# Stage C — score5 on `candidates4.csv`: 14 families x 2 fills x 5 stop/exit outcomes, contract (c), PLAN §1 gate

Executed 2026-09-16 per `research/fuckup_audit/PLAN.md` §4 row C. Everything written is under
`research/fuckup_audit/C/`; everything outside it was read only. No config, service, cache or order was touched.
**TEST was NOT read** (no cell reached G2; `SCORE5_READ_TEST` was never set).

---

## 0. One page

**Nothing clears G1. Nothing clears G2. In all seven runs, 0 of 107 scoreable cells.** The best TRAIN t over the
whole pre-registered grid is **1.93**, and a 500-draw day-label sign-flip permutation over the same 107 cells puts
the 95th percentile of the null max-t at **3.88** (**p = 0.998**). The grid's best cell is not merely
non-significant — it is more than two units of t *below* what pure noise produces when you look at 107 cells.

| run (115 cells declared, 107 with a book) | TRAIN cells > 0 | gross > 0 | best mean R | best t | **G1** | **G2** | TRAIN&VAL both > 0 |
|---|---|---|---|---|---|---|---|
| **>=10:00, contract (c) — PRIMARY** | 19 | 43 | +0.536 | 1.93 | **0** | **0** | 12 |
| all-day, contract (c) — companion | 21 | 54 | +0.266 | 1.56 | **0** | **0** | 12 |
| >=10:00, resting restricted to queue-OK (sens Q) | 21 | 44 | +0.536 | 1.93 | 0 | 0 | 14 |
| all-day, queue-OK (sens Q) | 23 | 60 | +0.266 | 1.56 | 0 | 0 | 13 |
| >=10:00, free target — contract (c') (sens T0) | 20 | 43 | +0.536 | 1.93 | 0 | 0 | 13 |
| >=10:00, score4 legacy spread (sens L) | 2 | 43 | +0.282 | 1.01 | 0 | 0 | 0 |
| all-day, score4 legacy spread (sens L) | 1 | 54 | +0.026 | 0.15 | 0 | 0 | 0 |

**The five things Stage C settles, each with its number.**

1. **H2 — the resting fill loses, and it loses on price, not only on the cost convention.** On the 89,826 TRAIN
   signals *both* models fill (>=10:00, 2R close-fill), the resting fill is **-0.058 R paired (t -61.5)** net and
   **-0.026 R (t -26.5) GROSS**; VAL agrees (-0.057 / -0.023). Roughly 45% of the net gap is the honest cost
   asymmetry (1.00x vs 0.25x the entry half-spread) and **55% is a genuinely worse price**. The resting-ONLY
   signals — the bursts the next-open cap throws away — are indeed positive (**+0.049 R net TRAIN, +0.053 VAL**,
   gross +0.124/+0.132; queue-OK restricted +0.101/+0.108), exactly Stage B's finding, but they do not pay for the
   loss on the shared majority: at BOOK level the resting book is **-0.083 vs -0.042 R** on TRAIN over 45 matched
   family x outcome pairs. The decision rule also fails on its own terms: obtainability is **1.000** but the queue
   check passes on only **0.72-0.96** of resting fills (F10 0.719 ... F9 0.962), i.e. **8 of 9 families below the
   pre-registered 95%**. -> **Next-open stays the reference fill.** The resting columns stay in the file as
   evidence, not as the engine's convention.
2. **H1 — the stop variants are real and an order of magnitude too small.** Pooled over 148,139 TRAIN signals,
   paired vs the touch stop: `stop - 1%` **+0.0299 R (t 22.9)** TRAIN and **+0.0277 (t 17.0)** VAL, stop rate
   28.9% -> 21.3%; `close-stop` **+0.0050 (t 6.9)** / **+0.0037 (t 4.4)**, stop rate -> 27.3%. Both have the right
   sign on both splits and both lower the stop rate — and **both are below the pre-registered +0.05 R adoption
   bar**, so neither is adopted. The wide-stop lever is concentrated exactly where H7 says it should be (F13
   +0.166, F1 +0.138, F5 +0.077, F10 +0.051 — the families whose structural stop is tightest), and it is ~+0.003
   to +0.02 on F6/F8, where the stop is already wide.
3. **H8 — the +2R target is the worst exit in the grid; the ORB lock is worse than holding.** Averaged over the
   family-configs (PRIMARY, next-open): `hold-to-close` **-0.019 R net / +0.027 gross**, `lock 1.75/0.5` -0.045 /
   +0.002, `2R stop-1%` -0.078 / -0.026, `2R close-stop` -0.081 / -0.032, `2R close-fill` **-0.100 / -0.051**.
   Paired, the lock costs **-0.0110 R (t -7.8)** TRAIN and **-0.0098 (t -6.2)** VAL against hold. Cutting winners
   at +2R on a close costs about 0.08 R a trade here.
4. **H9 — the four new families get their first honest numbers, and two of them are disasters.** `F14` second
   break is the best-behaved newcomer (TRAIN **+0.056 R**, t 1.32, WR 45.2%, 14.5 trades/wk; VAL +0.050, t 0.91 —
   both positive, neither significant). `F11` close-confirmation on the F6 base is ~0 on TRAIN (+0.006, t 0.29) and
   +0.065 on VAL (t 2.00) — it improves every F11/F8 cell versus not confirming, but not enough. `F12` retest and
   `F13` sweep-and-reclaim are **large, significant losers on both splits and gross-negative too**: F12 -0.209 /
   -0.233 R (t -7.7 / -6.5), F13 -0.19 to -0.35 R (t -6.4 to -10.7), VAL the same sign and size. **The owner's
   stop-sweep mechanism is real as a signature (Stage A) and destructive as an ENTRY.**
5. **The >=10:00 window does not generalise.** Stage A adopted it from an F6/F8 cut on `candidates3`; on the full
   14-family grid it is **worse**: the same 107 cells average -0.064 -> **-0.073 R** net and -0.002 -> **-0.016 R**
   gross, with trades/week 27.9 -> 24.6, and only **30 of 107** cells improve. It was declared PRIMARY before the
   run and is reported as PRIMARY; the honest reading is that it is not a lever.

**The closest miss, named.** `F6 {}` (red-to-green) with the next-open fill, `2R stop-1%` exit, entries >= 10:00:
TRAIN **+0.031 R** (t 1.43, 19.2 trades/wk, WR 46.9%), VAL **+0.063 R** (t 1.93, 55% weeks green), gross +0.049 /
+0.082, and it survives the +3R winner cap (+0.031) but not the top-5%-removed test (-0.067). Its MDE is
**0.0615 R**, about double the point estimate. To clear G1 it would have to earn 0.044 R/trade; it earns 0.031.
The only larger mean in the grid is `F1 {"P":0.12}` hold-to-close (+0.536 R, t 1.93) — WR 20.7%, 77% stops,
VAL -0.311: a lottery ticket, reported, never proposed.

**What Stage C does NOT say.** It does not say no edge exists. In THIS universe (the point-in-time >=5%-range day
population with the causal `range_so_far_pct >= 5` floor), at THIS 1-minute horizon, over 2025-01-02 -> 2026-09-11,
under THIS book (12/day, 4 concurrent), THIS cost contract and these 14 raw family shapes, **no cell showed an edge
that survives TRAIN significance**, and the smallest per-trade effect the headline cells could have seen at 80%
power is **0.060-0.076 R** for the F6/F8/F11 shapes (~1.2-1.9 R/week at 4 slots), **0.12 R** for F14 and
**0.78 R** for F1. Effects below those sizes are invisible here and are not excluded — and the +0.2..+0.4 R that
the BF and ORB *selection* stacks historically carried is a different question, which is Stage D's (PLAN §4 row C
stop rule: "if nothing clears G1 ... go to D anyway — selection can carry a raw-negative family").

---

## 1. PRE-REGISTRATION (verbatim from `C/PREREG.md`, written before any scoring run existed)

> Written 2026-09-16, before any Stage C scoring run existed (the only Stage C python that had run when it was
> written was verification: `C/parity_full.log`, `C/verify_rows.log`, `C/verify30.log`, none of which computes a net
> R, a book or a gate).

**Cells.** The **115** pre-registered in `B/REPORT.md` §1: 14 family-configs x 2 fills x 5 outcomes = 140, minus
**25 impossible resting cells** (F11 x2, F12 x2, F13 x1 are CLOSE-triggered — a resting order cannot express them).

**Windows.** **PRIMARY = `--min-entry-m 600` (entries >= 10:00)**, per Stage A §A4 item 3 and PLAN §3 H5, applied
uniformly to all 14 configs so the primary pass is one rule, not a per-family choice. **COMPANION = all-day**
(09:30-14:01), its own separately counted 115 cells. **Gate cells: 115 + 115 = 230.**

**Contract (Stage A's adopted contract (c), `A/acore.py`):**
`half_cc = 0.5 x (spread_cc_bps/100) / max(r_pct_of_the_variant, 0.05)`;
`net = rr - ENTRY x half_cc - EXIT[why] x half_cc`; `ENTRY` = 0.25 (next-open) / 1.00 (resting);
`EXIT` = stop 0.875 · lock 0.875 · eod 0.412 · target 0.875 (0 only under the (c') sensitivity).
Population: fill >= $5 of the model being scored, `entry_m <= 841`, `r_pct >= 1.0 of the variant`,
`range_so_far_pct >= 5` except F1-F4; for the resting fill the $5 and R >= 1% floors are applied to the **resting**
entry and its own R. Book `run_book(rows, 12, 4)`. Splits TRAIN 2025 · VAL 2026-01..05 · TEST 2026-06-01..09-11.
**GROSS reported next to net everywhere.**

**Gates.** G1 TRAIN mean > 0, t >= 2.0, >= 5 trades/wk. G2 VAL mean > 0, t >= 1.0, >= 55% weeks green, weekly R >=
(G1 passes / 10) x SE(weekly R). G3 TEST read ONCE, only for G2 survivors, only after the selection is frozen in
writing. Every G2 survivor first gets the tail test (top 1%/5% removed, cap +3R), a per-month table and the
search-adjusted permutation p (500 day-label sign-flip draws, max TRAIN t over all cells).

**Declared sensitivities** (re-scores, never gate candidates): **Q** resting restricted to `rest_queue_ok == 1`
(the honest H2 claim); **T0** `--free-target` = contract (c'); **L** `--legacy-spread` = score4's band table.

**Declared paired comparisons** (deltas on matched signals, not gate cells): **P1** resting vs next-open, on the
signals both models fill and on the resting-only signals; **P2** each H1 stop variant vs the touch stop, in net R
and at constant $100 risk, with the stop rate; **P3** the lock exit vs hold.

**Adoption rules carried from the PLAN.** H2: the resting fill replaces next-open only if >= 95% of its fills pass
obtainability AND the queue check, and the improvement holds on TRAIN and VAL. H1: a stop variant is adopted only
if the paired difference vs the touch stop is >= +0.05 R with t >= 2 on TRAIN, the sign agrees on VAL, the stop rate
falls, and the new stop's fill is obtainable.

---

## 2. Verification of the input — all four checks PASSED (run before any cell was scored)

| # | check | result |
|---|---|---|
| 1 | row count | `wc -l` **3,099,500** = 1 header + **3,099,499** rows = the builder log's final `total` exactly; `build4_state.json` holds **420** days; `build4.log` ends `DONE` / `EXIT=0` |
| 2 | header alignment | `head -1` vs `build_candidates4.py::COLS` — **77 vs 77, IDENTICAL element by element** |
| 3a | `B/parity_smoke.py` on the FINISHED file (all 420 days, not the 3-day smoke) | **913,985** `candidates3` rows of the 5 shared family-configs, **0** with no `candidates4` fill, `entry` vs `entry_next` **max abs diff = 0.000e+00**, `rr_2r` vs `rr_2r_next` **0.000e+00**, `sig_m`/`entry_m`/`exit_m_2r` **0 mismatches**, `why_2r` **0 mismatches** -> `PARITY OK`. `candidates4` additionally carries **290,719** signal rows `candidates3` could never hold (225,321 of them with no next-open fill) — the selection that made H2 unanswerable |
| 3b | `B/verify_rows.py 5` | 5 random rows recomputed by a plain python loop straight from the bars (`sig_o/h/l/c/v`, `range_so_far_pct`, `cum_dollar_vol`, `n_touches`, `close_confirm`, both fills, `r_pct`, `rr_2r`, `rr_hold`, `rr_2r_stopm1`, `exit_m`, `why`, `rest_queue_ok`) — **ALL MATCH** |
| 4 | 30-row sanity, 3 random days (`C/verify30.py`, days 2025-01-21 / 2025-04-22 / 2026-06-05) | 6,862 `candidates4` vs 5,272 `candidates3` shared-family rows, 0 missing, **30/30 EXACT** on entry / rr_2r / why / exit_m; whole-3-day max abs d entry = **0.0**, max abs d rr_2r = **0.0** |

Logs: `C/parity_full.log`, `C/verify_rows.log`, `C/verify30.log`.

**One extra check, not required but load-bearing for everything below.** To avoid re-parsing 2.0 GB seven times,
`C/c0_extract.py` wrote a lossless row-subset (`C/pop_c.csv`, **447,306 of 3,099,499 rows = 14.4%**, 311 MB, every
column kept) using the union over fills of score5's own per-fill filter. The identity was **not assumed**: the
all-day pass was run on the extract AND on the full 2.0 GB file, and the two result tables are identical —
`shape (107, 28)` both, keys identical, **max abs diff over all numeric cells = 0.0** (`C/run_fullcheck.log`,
`C/score5_results.fullcheck.csv`).

---

## 3. The scorer — what the Stage C copy changed (and what it must not)

`B/score5.py` was copied to `C/score5c.py`. The diff is **reporting only**:
`--queue-ok` (restrict the resting fill to `rest_queue_ok == 1`), `--tag`, `--perm-always` (report the permutation
null even with no G2 survivor), a **gross** column, per-trade **SE** and **MDE = 2.8 x SE**, a **stopP** column
(the share of booked trades exiting on the stop — H1's adoption rule needs it), a per-month table and a booked-trade
dump for G2 survivors, and output paths under `C/`. `rest_queue_ok` was added to the loaded columns.
**Unchanged, line for line:** the population filter, `ENTRY_MULT`, `EXIT_RATIO`, `net_r`, the `run_book(rows, 12, 4)`
call, `FAM_KEYS`, `OUTCOMES`, `CLOSE_TRIGGERED`, the split boundaries, the G1/G2 thresholds and the G2 SE bar, the
tail test and the permutation. The builder's header (`B/build_candidates4.py`) remains the specification.

Scoreable rows, PRIMARY window: **next 299,780 · rest 230,895** (all-day: 397,116 · 306,970; queue-OK: 181,305 rest).
Weeks: TRAIN 53 · VAL 22 · TEST 14. All 14 pre-registered family keys produced scoreable rows (no
`pre-registered keys with no scoreable row` warning in any run). 107 of the 115 cells have a book; the 8 without are **exactly F12's
eight non-`stop-1%` cells** (2 configs x 4 outcomes) — F12's structural stop is < 1% of price by construction, so
the PLAN's R >= 1% floor leaves it scoreable only through the `2R stop-1%` variant. `B/REPORT.md` §2.1 pinned this
in writing before the run; it is a property of the PLAN's own F12 specification, not a defect.

---

## 4. The tables

### 4.1 PRIMARY (>=10:00) — every cell with a positive TRAIN mean

Full table: `C/score5_tables.m600.md` (all 107 cells) · `C/score5_results.m600.csv`.

| key | fill | outcome | n | tpw | TRAIN meanR | TRAIN **gross** | SE | MDE | t | WR | VAL meanR | VAL gross | VAL t | ex5 | cap3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F1 {"P":0.12} | next | hold | 454 | 8.6 | **+0.536** | +0.600 | 0.278 | 0.778 | 1.93 | 20.7 | -0.311 | -0.243 | -1.53 | -0.532 | -0.511 |
| F1 {"P":0.12} | next | lock | 456 | 8.6 | +0.322 | +0.389 | 0.246 | 0.688 | 1.31 | 34.0 | -0.274 | -0.204 | -1.67 | -0.592 | -0.441 |
| F1 {"P":0.12} | rest | hold | 907 | 17.1 | +0.184 | +0.281 | 0.181 | 0.508 | 1.01 | 19.6 | -0.054 | +0.051 | -0.26 | -0.670 | -0.580 |
| F14 {"N":15} | next | hold | 768 | 14.5 | +0.056 | +0.084 | 0.043 | 0.120 | 1.32 | 45.2 | +0.050 | +0.077 | 0.91 | -0.119 | +0.013 |
| F1 {"P":0.12} | rest | lock | 911 | 17.2 | +0.055 | +0.155 | 0.162 | 0.452 | 0.34 | 36.0 | -0.094 | +0.013 | -0.59 | -0.631 | -0.456 |
| F14 {"N":15} | next | lock | 771 | 14.5 | +0.053 | +0.080 | 0.042 | 0.117 | 1.25 | 45.8 | +0.038 | +0.065 | 0.71 | -0.118 | +0.011 |
| **F6 {}** | next | hold | 1018 | 19.2 | **+0.036** | +0.055 | 0.028 | 0.077 | 1.30 | 45.3 | **+0.101** | +0.122 | 1.83 | -0.096 | +0.022 |
| F14 {"N":15} | next | 2R close-stop | 785 | 14.8 | +0.035 | +0.064 | 0.033 | 0.092 | 1.08 | 46.4 | +0.029 | +0.057 | 0.63 | -0.070 | +0.033 |
| **F6 {}** | next | **2R stop-1%** | 1017 | 19.2 | **+0.031** | +0.049 | 0.022 | **0.062** | **1.43** | 46.9 | **+0.063** | +0.082 | **1.93** | -0.067 | +0.031 |
| F6 {} | next | lock | 1029 | 19.4 | +0.028 | +0.048 | 0.026 | 0.073 | 1.07 | 46.0 | +0.097 | +0.119 | 1.82 | -0.097 | +0.016 |
| F6 {} | next | 2R close-stop | 1037 | 19.6 | +0.026 | +0.046 | 0.024 | 0.067 | 1.09 | 46.2 | +0.072 | +0.094 | 2.05 | -0.075 | +0.026 |
| F6 {} | next | 2R close-fill | 1044 | 19.7 | +0.026 | +0.046 | 0.024 | 0.066 | 1.10 | 46.1 | +0.067 | +0.089 | 1.93 | -0.076 | +0.026 |
| F14 {"N":15} | next | 2R close-fill | 789 | 14.9 | +0.025 | +0.054 | 0.032 | 0.089 | 0.79 | 46.0 | +0.031 | +0.059 | 0.67 | -0.076 | +0.025 |
| F14 {"N":15} | next | 2R stop-1% | 772 | 14.6 | +0.019 | +0.040 | 0.028 | 0.079 | 0.68 | 46.2 | +0.043 | +0.064 | 1.06 | -0.082 | +0.019 |
| F8 {"N":30} | next | hold | 1124 | 21.2 | +0.008 | +0.032 | 0.024 | 0.066 | 0.34 | 48.4 | -0.011 | +0.014 | -0.32 | -0.105 | +0.001 |
| F11 {base F6} | next | 2R stop-1% | 1011 | 19.1 | +0.006 | +0.024 | 0.021 | 0.060 | 0.29 | 45.8 | +0.065 | +0.084 | 2.00 | -0.090 | +0.006 |
| F8 {"N":30} | next | lock | 1136 | 21.4 | +0.005 | +0.029 | 0.023 | 0.064 | 0.22 | 48.6 | -0.018 | +0.008 | -0.53 | -0.103 | -0.002 |
| F11 {base F6} | next | 2R close-stop | 1036 | 19.5 | +0.003 | +0.024 | 0.023 | 0.065 | 0.11 | 45.5 | +0.064 | +0.087 | 1.81 | -0.100 | +0.003 |
| F8 {"N":30} | next | 2R close-fill | 1148 | 21.7 | +0.001 | +0.025 | 0.022 | 0.060 | 0.05 | 48.8 | -0.010 | +0.015 | -0.31 | -0.096 | +0.001 |

That is all 19 of the 107 cells with a positive TRAIN mean. The all-day companion's 21 positives are the same
shapes with F6 at the top (+0.051 hold, t 1.34, VAL +0.163 t 2.46) — `C/score5_tables.md`.

### 4.2 Closest miss per family-config, PRIMARY window, with its power

`mde = 2.8 x SE` is the smallest per-trade effect that cell could have seen at 80% power. Best cell per key by
TRAIN t among the positive-mean cells, else best mean. Source `C/c3_summary.md` §2.

| key | fill | outcome | n | tpw | TRAIN meanR | gross | SE | **MDE** | t | VAL meanR | VAL t | ex5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F1 {"P":0.12} | next | hold | 454 | 8.6 | +0.536 | +0.600 | 0.278 | 0.778 | 1.93 | -0.311 | -1.53 | -0.532 |
| F6 {} | next | 2R stop-1% | 1017 | 19.2 | +0.031 | +0.049 | 0.022 | 0.062 | 1.43 | +0.063 | 1.93 | -0.067 |
| F14 {"N":15} | next | hold | 768 | 14.5 | +0.056 | +0.084 | 0.043 | 0.120 | 1.32 | +0.050 | 0.91 | -0.119 |
| F8 {"N":30} | next | hold | 1124 | 21.2 | +0.008 | +0.032 | 0.024 | 0.066 | 0.34 | -0.011 | -0.32 | -0.105 |
| F11 {base F6} | next | 2R stop-1% | 1011 | 19.1 | +0.006 | +0.024 | 0.021 | 0.060 | 0.29 | +0.065 | 2.00 | -0.090 |
| F8 {"N":15} | next | 2R stop-1% | 1204 | 22.7 | -0.008 | +0.013 | 0.023 | 0.064 | -0.34 | +0.052 | 1.59 | -0.112 |
| F8 {"N":5} | next | hold | 1366 | 25.8 | -0.029 | +0.006 | 0.044 | 0.122 | -0.66 | +0.090 | 1.54 | -0.273 |
| F11 {N 15, base F8} | next | 2R close-stop | 1256 | 23.7 | -0.028 | -0.003 | 0.023 | 0.065 | -1.20 | +0.037 | 1.04 | -0.132 |
| F9 {"G":0.05} | next | hold | 649 | 12.2 | -0.085 | -0.057 | 0.055 | 0.154 | -1.54 | -0.023 | -0.37 | -0.303 |
| F10 {} | next | 2R stop-1% | 1636 | 30.9 | -0.109 | -0.061 | 0.026 | 0.071 | -4.26 | -0.054 | -1.36 | -0.216 |
| F5 {"K":5,"X":0.04} | next | 2R stop-1% | 2098 | 39.6 | -0.130 | -0.067 | 0.027 | 0.076 | -4.79 | -0.051 | -1.17 | -0.239 |
| F12 {base F6} | next | 2R stop-1% | 1420 | 26.8 | -0.233 | -0.106 | 0.036 | 0.100 | -6.49 | -0.232 | -4.60 | -0.345 |
| F13 {"K":5,"X":0.04} | next | 2R stop-1% | 2617 | 49.4 | -0.193 | -0.068 | 0.027 | 0.075 | -7.20 | -0.231 | -5.61 | -0.303 |
| F12 {N 15, base F8} | next | 2R stop-1% | 2659 | 50.2 | -0.209 | -0.076 | 0.027 | 0.076 | -7.71 | -0.253 | -6.16 | -0.320 |

F5 stays what Stage A said it was (-0.130 R at its *best* cell, t -4.8). F10 (VWAP reclaim) is its equal.

### 4.3 The H9 families — F11-F14's first honest numbers (PRIMARY, all cells with a book)

Full table `C/c3_summary.md` §3.

| key | fill | outcome | n | tpw | TRAIN meanR | gross | MDE | t | WR | stopP | VAL meanR | VAL gross | VAL t |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **F14** second break | next | hold | 768 | 14.5 | **+0.056** | +0.084 | 0.120 | 1.32 | 45.2 | 21.2 | **+0.050** | +0.077 | 0.91 |
| F14 | next | lock | 771 | 14.5 | +0.053 | +0.080 | 0.117 | 1.25 | 45.8 | 23.3 | +0.038 | +0.065 | 0.71 |
| F14 | next | 2R close-stop | 785 | 14.8 | +0.035 | +0.064 | 0.092 | 1.08 | 46.4 | 19.1 | +0.029 | +0.057 | 0.63 |
| F14 | next | 2R close-fill | 789 | 14.9 | +0.025 | +0.054 | 0.089 | 0.79 | 46.0 | 20.7 | +0.031 | +0.059 | 0.67 |
| F14 | next | 2R stop-1% | 772 | 14.6 | +0.019 | +0.040 | 0.079 | 0.68 | 46.2 | 15.7 | +0.043 | +0.064 | 1.06 |
| **F11** close-confirm (F6) | next | 2R stop-1% | 1011 | 19.1 | +0.006 | +0.024 | 0.060 | 0.29 | 45.8 | 10.7 | **+0.065** | +0.084 | **2.00** |
| F11 (F6) | next | 2R close-stop | 1036 | 19.5 | +0.003 | +0.024 | 0.065 | 0.11 | 45.5 | 13.6 | +0.064 | +0.087 | 1.81 |
| F11 (F6) | next | 2R close-fill | 1044 | 19.7 | -0.001 | +0.020 | 0.065 | -0.04 | 45.3 | 14.2 | +0.056 | +0.079 | 1.61 |
| F11 (F6) | next | hold | 1018 | 19.2 | -0.005 | +0.016 | 0.070 | -0.18 | 44.7 | 14.7 | +0.085 | +0.107 | 1.59 |
| F11 (F6) | next | lock | 1027 | 19.4 | -0.016 | +0.005 | 0.066 | -0.67 | 45.4 | 16.7 | +0.084 | +0.106 | 1.60 |
| F14 | rest | 5 cells | 802-828 | 15.1-15.6 | -0.002 .. -0.046 | -0.007..+0.048 | 0.075-0.125 | -0.04..-1.72 | 42-43 | 17-24 | +0.008 .. -0.017 | +0.033..+0.057 | 0.18..-0.38 |
| F11 (N 15, base F8) | next | 5 cells | 1188-1280 | 22.4-24.2 | -0.028 .. -0.054 | -0.003..-0.030 | 0.061-0.073 | -1.20..-2.21 | 42-45 | 16-25 | +0.037 .. +0.059 | +0.062..+0.083 | 1.04..1.75 |
| **F13** sweep-and-reclaim | next | 2R stop-1% | 2617 | 49.4 | **-0.193** | -0.068 | 0.075 | **-7.20** | 36.1 | 58.0 | -0.231 | -0.099 | -5.61 |
| F13 | next | 2R close-stop | 1998 | 37.7 | -0.279 | -0.142 | 0.095 | -8.25 | 35.3 | 62.9 | -0.294 | -0.148 | -5.95 |
| F13 | next | lock | 1910 | 36.0 | -0.327 | -0.195 | 0.130 | -7.04 | 32.6 | 83.5 | -0.268 | -0.127 | -4.00 |
| F13 | next | 2R close-fill | 2052 | 38.7 | -0.335 | -0.197 | 0.087 | -10.73 | 31.7 | 66.1 | -0.309 | -0.162 | -6.81 |
| F13 | next | hold | 1847 | 34.8 | -0.351 | -0.222 | 0.153 | -6.41 | 20.6 | 76.4 | -0.158 | -0.022 | -1.76 |
| **F12** retest (F8 N15) | next | 2R stop-1% | 2659 | 50.2 | **-0.209** | -0.076 | 0.076 | **-7.71** | 35.4 | 60.7 | -0.253 | -0.112 | -6.16 |
| F12 retest (F6) | next | 2R stop-1% | 1420 | 26.8 | -0.233 | -0.106 | 0.100 | -6.49 | 36.1 | 57.6 | -0.232 | -0.097 | -4.60 |

Readings. (a) **F11 is the cheapest real improvement in the grid**: requiring the break bar to CLOSE above the level
lifts every F6-based cell's VAL by ~+0.01 and cuts the stop rate (14.2% -> 10.7% under stop-1%), consistent with
the live "touch-only breaks were -1R" observation — it just is not worth +0.05 R. (b) **F12 and F13 are the two
worst families ever scored in this program**, gross-negative by -0.07 to -0.22 R with t below -6 on both splits, at
27-50 trades/week. Both are "second-chance" entries after a failure, and both buy the trades the first entry
correctly avoided. (c) **F14 is the only new family positive on both splits**; its resting-only signals are
spectacular in isolation (+0.51 R net on 199 TRAIN signals, +0.28 on 91 VAL) and its resting *book* is negative —
a tail, not an edge.

### 4.4 Outcome shape (H8) at book level, PRIMARY, averaged over the family-configs

| fill | outcome | cells | TRAIN net | TRAIN **gross** | stop % | VAL net | tpw |
|---|---|---|---|---|---|---|---|
| next | hold-to-close | 12 | **-0.019** | **+0.027** | 37.9 | -0.013 | 23.1 |
| next | lock 1.75/0.5 | 12 | -0.045 | +0.002 | 42.2 | -0.036 | 23.9 |
| next | 2R stop-1% | 14 | -0.078 | -0.026 | 32.7 | -0.049 | 26.1 |
| next | 2R close-stop | 12 | -0.081 | -0.032 | 32.7 | -0.031 | 24.7 |
| next | 2R close-fill | 12 | -0.100 | -0.051 | 34.8 | -0.032 | 25.2 |
| rest | hold-to-close | 9 | -0.047 | +0.023 | 37.9 | -0.003 | 23.5 |
| rest | lock 1.75/0.5 | 9 | -0.072 | -0.002 | 42.3 | -0.012 | 24.5 |
| rest | 2R stop-1% | 9 | -0.082 | -0.027 | 28.4 | -0.015 | 23.0 |
| rest | 2R close-stop | 9 | -0.104 | -0.032 | 32.8 | -0.035 | 25.4 |
| rest | 2R close-fill | 9 | -0.112 | -0.040 | 34.8 | -0.037 | 25.8 |

---

## 5. P1 — H2: the resting fill vs the engine's next-open fill (`C/c1_paired.md`, `C/c1_paired_P1.csv`)

PRIMARY window, all non-close-triggered families pooled. `paired_d` = mean(net_rest - net_next) on the signals
**both** models fill; `_GROSS` the same on `rr`; `restonly` = the mean of the signals only the resting model fills.

| outcome | split | n both | paired d net | t | paired d **GROSS** | t | n rest-only | rest-only net | t | rest-only **gross** | queue-OK rest-only net | n next-only | next-only net |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2R close-fill | TRAIN | 89,826 | **-0.0582** | -61.5 | **-0.0255** | -26.5 | 18,233 | +0.0485 | 5.9 | +0.124 | +0.1005 | 3,723 | -0.104 |
| 2R close-fill | VAL | 54,513 | -0.0571 | -49.2 | -0.0231 | -19.5 | 9,642 | +0.0531 | 4.8 | +0.132 | +0.1083 | 2,148 | -0.066 |
| hold-to-close | TRAIN | 89,826 | -0.0619 | -62.9 | -0.0290 | -29.2 | 18,233 | +0.1121 | 7.4 | +0.184 | +0.1681 | 3,723 | -0.082 |
| hold-to-close | VAL | 54,513 | -0.0624 | -43.5 | -0.0282 | -19.6 | 9,642 | +0.0837 | 4.7 | +0.159 | +0.1395 | 2,148 | -0.040 |
| lock 1.75/0.5 | TRAIN | 89,826 | -0.0582 | -48.6 | -0.0254 | -21.2 | 18,233 | +0.0904 | 6.4 | +0.164 | +0.1512 | 3,723 | -0.091 |
| lock 1.75/0.5 | VAL | 54,513 | -0.0595 | -37.0 | -0.0254 | -15.7 | 9,642 | +0.0652 | 3.9 | +0.142 | +0.1173 | 2,148 | -0.065 |
| 2R close-stop | TRAIN | 89,826 | -0.0591 | -60.8 | -0.0264 | -26.7 | 18,233 | +0.0613 | 7.3 | +0.137 | +0.1034 | 3,723 | -0.080 |
| 2R close-stop | VAL | 54,513 | -0.0579 | -48.6 | -0.0239 | -19.7 | 9,642 | +0.0612 | 5.5 | +0.140 | +0.1114 | 2,148 | -0.034 |
| 2R stop-1% | TRAIN | 97,169 | -0.0518 | -74.6 | -0.0252 | -35.8 | 16,700 | +0.1185 | 15.4 | +0.165 | +0.1379 | 3,111 | -0.005 |
| 2R stop-1% | VAL | 59,206 | -0.0504 | -62.6 | -0.0225 | -27.5 | 8,765 | +0.1203 | 11.9 | +0.168 | +0.1505 | 1,765 | +0.031 |

All-day is the same to within 0.005 R on every row (`C/c1_paired.md`).

**Obtainability and the queue check** (scored population, resting fills):

| key | n | obtainable | queue-OK |
|---|---|---|---|
| F9 {"G":0.05} | 6,154 | 1.000 | **0.962** |
| F6 {} | 14,381 | 1.000 | 0.904 |
| F8 {"N":30} | 62,298 | 1.000 | 0.899 |
| F8 {"N":15} | 53,075 | 1.000 | 0.890 |
| F14 {"N":15} | 2,806 | 1.000 | 0.886 |
| F8 {"N":5} | 36,862 | 1.000 | 0.874 |
| F1 {"P":0.12} | 3,124 | 1.000 | 0.826 |
| F5 {"K":5,"X":0.04} | 33,638 | 1.000 | 0.779 |
| F10 {} | 81,776 | 1.000 | 0.719 |

**Verdict on H2 (the pre-registered rule, both legs):** obtainability 1.000 everywhere — the level is inside the
signal bar by construction — but the queue check clears 95% for **1 of 9** families, and the improvement does **not**
hold on TRAIN or VAL: it is negative, paired, on the shared signals, gross as well as net, and the book is worse
(TRAIN -0.083 rest vs -0.042 next over 45 matched pairs; gross -0.015 vs -0.000). **The resting fill is NOT adopted
as the reference fill.** What Stage B's probe got right stands: the next-open convention *does* discard a positive
cohort (+0.05..+0.12 R net, +0.12..+0.18 gross, ~20% more signals). What it got wrong is the conclusion that
capturing them is worth it — it is not, because the same order pays a worse price on the other 80%.

## 6. P2 / P3 — H1's stops and H8's lock (`C/c1_paired.md` §P2, `C/c1_paired_P2.csv`)

All-day, next-open fill, pooled over the 14 configs (148,139 TRAIN / 91,772 VAL matched signals). Deltas are in net
R of each variant's own R, which at a constant $100 of risk is the $ delta / 100.

| split | base (touch) meanR | base stop % | d close-stop | t | stop % | d **stop-1%** | t | stop % | d lock vs hold | t |
|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN | -0.0488 | 28.9 | +0.0050 | 6.91 | 27.3 | **+0.0299** | **22.86** | **21.3** | **-0.0110** | -7.79 |
| VAL | -0.0392 | 27.1 | +0.0037 | 4.42 | 25.7 | **+0.0277** | **17.04** | 19.8 | -0.0098 | -6.23 |

Per family (TRAIN, d stop-1%): F13 **+0.166**, F1 +0.138, F5 +0.077, F10 +0.051, F8 N=5 +0.020, F11(F6) +0.015,
F6 +0.008, F8 N=15 +0.006, F8 N=30 +0.004, F14 -0.019. The lever is entirely the H7 mechanism — it pays where the
structural stop is tight relative to the spread, and it is worth nothing where the stop is already wide.

**Verdict on H1:** both variants have the right sign on both splits and both lower the stop rate, and **neither
reaches the pre-registered +0.05 R bar** (stop-1% reaches it only inside F13/F1/F5/F10, all of which are
-0.13 to -0.35 R books before the help). **No stop variant is adopted.** The touch stop stays the reference.
**Verdict on H8/P3:** the ORB static lock is worse than simply holding, by -0.011 R with t ~ -7.8 on TRAIN and -6.2
on VAL, and it is negative on TRAIN for every one of the 12 configs that has a touch-stop base population (F12 has none). **The lock is rejected for this population.** Holding
to 15:55 is the best exit tested and the +2R close-target is the worst, a spread of ~0.08 R per trade.

## 7. Sensitivities (declared, reported, never gate candidates)

| sensitivity | effect |
|---|---|
| **Q** resting restricted to `rest_queue_ok == 1` | the resting cells improve slightly (14 vs 12 cells positive on both splits; the pooled paired d moves from -0.058 to -0.051) and **still nothing clears G1**. Best cell and best t unchanged (they are next-open cells). |
| **T0** free target, contract (c') | +1 positive TRAIN cell (20 vs 19). The +2R target fires on 12-20% of trades in the surviving cells, so the whole convention is worth ~0.005 R. It changes no verdict. |
| **L** score4's legacy band table | **2 of 107** cells positive on TRAIN (vs 19), 0 positive on both splits (vs 12), best t 1.01. This reproduces score4's "0 of 52" verdict on the new grid and restates Stage A's finding: the corrected spread table is worth ~+0.41 R/trade, and it is the difference between "everything is deeply negative" and "a fifth of the grid is marginally positive and none of it is significant". |

## 8. What it means

**The gate.** 0 of 230 declared gate cells clear G1; therefore 0 clear G2; therefore **TEST was not read**, in
either window, under any sensitivity. PLAN §4's stop rule for Stage C applies: the closest miss per family is
in §4.2 with its power, and the program proceeds to Stage D, because selection can carry a raw-negative family —
that is exactly the BF and ORB history (`bf_consistency/README.md` §6, `orb_veto_study/REPORT.md`).

**The permutation, run even without a survivor.** 500 day-label sign-flip draws over all 107 PRIMARY cells:
observed max TRAIN t **1.93**, null 95th percentile **3.88**, **p = 0.998** (`C/score5_tables.m600.perm.md`). A grid
this wide manufactures t ~ 3.9 out of nothing; the best real cell is at half that. There is no multiple-testing
argument to be had here in either direction — the grid simply contains no signal of the size it could detect.

**Smallest visible effect, per headline cell** (MDE = 2.8 x SE per trade, and the same at 4 slots):

| cell | trades/wk | MDE per trade | MDE per week |
|---|---|---|---|
| F6 {} next 2R stop-1% | 19.2 | 0.062 R | 1.2 R |
| F11 {base F6} next 2R stop-1% | 19.1 | 0.060 R | 1.1 R |
| F8 {"N":30} next hold | 21.2 | 0.066 R | 1.4 R |
| F8 {"N":15} next 2R stop-1% | 22.7 | 0.064 R | 1.4 R |
| F14 {"N":15} next hold | 14.5 | 0.120 R | 1.7 R |
| F1 {"P":0.12} next hold | 8.6 | 0.778 R | 6.7 R |

The owner's scale target is 10 R/week at 4 slots ~ 0.43 R/trade. **That size would have been seen with overwhelming
significance in any of these cells** — it is excluded. What is NOT excluded is anything under ~1.2 R/week, which is
$120/week at $100 risk: real money at scale, invisible at this n.

**The phrasing, per PLAN §1.** No edge was detectable in THIS universe (point-in-time >=5%-range days with the
causal range floor), at THIS horizon (1-minute bars, entries 09:30-14:01 or 10:00-14:01), at THIS book size (12/day,
4 concurrent), over THIS window (2025-01-02 -> 2026-05-31 for selection), at THIS cost (contract (c)), for any of 14
raw family shapes x 2 fills x 5 stop/exit combinations. The smallest effect the test could have seen is in the
table above. Three structural facts were nailed down that are *not* null results: the resting fill is worse on
price (H2), the wide stop is worth +0.03 R and only where the stop is tight (H1/H7), and holding beats both the
+2R target and the ORB lock (H8).

**One thing Stage C could not test, stated so it is not forgotten.** Stage A adopted "entries >= 10:00" from an
F6/F8 cut; on this grid it costs 0.009 R/trade and 3.3 trades/week and helps only 30 of 107 cells. The PLAN's H5
mechanism (the overnight return predicts the first half-hour negatively) is about *gappers at the open*; this
universe's causal floor forces most signals past 10:00 anyway (`E/REPORT.md` measured that only 24.8% of causal-
universe signals pass the >=5%-range floor). H5 is properly tested on Stage E's causal universe, not here.

## 9. Cell count

| what | count |
|---|---|
| **Gate cells, declared in advance** — PRIMARY >=10:00 (14 configs x 2 fills x 5 outcomes - 25 impossible resting) | **115** |
| **Gate cells** — companion all-day, same 115 | **115** |
| re-scores of the same cells under the 3 declared sensitivities (Q x2 windows, T0 x1, L x2) | 575 |
| P1 paired cells (2 windows x 5 outcomes x (9 families + pooled) x 2 splits) | 200 |
| P2/P3 paired statistics (2 windows x 2 fills x (keys + ALL) x 2 splits x 3 deltas) | 300 |
| the extract-vs-full-file identity re-run (same cells, not new) | 0 |
| **Stage C total numbers looked at** | **1,305** |
| Stage A (52 x 6 contracts + 104 state buckets), Stage B (0 scored), Stage D0/D0b (18 + 42) are counted in their own reports | — |

## 10. Files

| path | what |
|---|---|
| `C/PREREG.md` | the pre-registration, written before any scoring run |
| `C/c0_extract.py` -> `C/pop_c.csv` | lossless 14.4% row-subset of `B/candidates4.csv` (identity verified) |
| `C/score5c.py` | `B/score5.py` + reporting only (§3) |
| `C/score5_tables{,.m600,.queueok,.queueok.m600,.freetgt.m600,.legacy,.legacy.m600,.m600.perm}.md` + `.csv` | the seven runs + the permutation |
| `C/c1_paired.py` -> `C/c1_paired.md`, `_P1.csv`, `_P2.csv` | P1 / P2 / P3 |
| `C/c3_summary.py` -> `C/c3_summary.md` | the derived tables of §4.1-4.4 |
| `C/verify30.py`, `C/parity_full.log`, `C/verify_rows.log`, `C/verify30.log`, `C/run_fullcheck.log` | §2 |
