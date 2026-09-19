# hod_losers — PRE-REGISTRATION (Part 2)

Written **after Part 1 (the anatomy) and BEFORE any Part 2 cell was scored.** Part 1 is descriptive
only — no cell, no gate, no selection was evaluated in it. This file is committed before `cells.py`
is run for the first time. Owner brief 2026-09-19: *"think deep, be creative, what else can separate
here? Also look at the shape of the losing days — are these DAYS we don't want to trade? WEEKS? or
specific TRADES we want to avoid."*

## 0. What Part 1 found, in the order it decides Part 2

| anatomy line | what it rules in / out |
|---|---|
| Worst 10 % of DAYS carry 36–47 % of loss $; best 10 % carry 43–55 % of gain $ — day dispersion is **symmetric** | a day gate is not nominated by concentration alone |
| Week SIGNS pass a runs test in all four base × split (z −0.13…+0.93); weekly-$ lag-1 autocorr −0.43…+0.10 | **red weeks are noise, not a regime — no week gate** |
| Within-day win-rate ICC +0.052…+0.074 (positive) — outcomes do cluster inside a day | an IN-DAY, causal kill switch is nominated |
| Losers: MFE **+0.45 R at 6 min**, MAE −1.06 R, stop at 28 min. Winners: MFE +2.10 R at 65 min | the loss lives in the **first 10 minutes of the trade** |
| `MFE@10min < 0.25 R` → gross −0.524 (TRAIN) / −0.521 (VAL) vs +0.248 / +0.364 kept — same sign, both bases | post-fill time stop nominated (primary) |
| fill-bar close in the bottom half of its own range → −0.166 / −0.193 vs +0.112 / +0.227 | fill-bar shape exit nominated |
| fill-bar low ≥ 0.25 R below entry → −0.297 / −0.302 vs +0.056 / +0.099 | Rule D nominated |
| **breakout-bar** close position (ORB's Rule M input) REVERSES: bottom half is BETTER (+0.042 / +0.080) | Rule M nominated as a **falsification** cell, not as a lever |
| `touch_n ≥ 5` (level tested ≥ 5× before the break) worse in all four base × split (sep −0.065…−0.175) | touches veto nominated |
| `consol_bars ≥ 8` worse in all four (sep −0.052…−0.226) — a FRESH HOD beats an old one | duration veto nominated, in the direction OPPOSITE to the brief's (d) |
| `dist_open / ATR14` quintiles non-monotone, VAL Q4 −0.149 vs TRAIN +0.077 | **(c) ATR-normalised admission NOT RUN** |
| `coh_by_t` coverage 10.6 %, 94 % of covered rows = 0 | **(f) family/sector confirmation NOT RUN** — the field is 1-in-10 covered and constant where covered |
| the losing-day signature is `spy_ret_0930_1000` (−0.043 % vs +0.223 % median) and `breadth_by_1000` | **(g) day gate NOT RUN HERE** — non-causal for the 43 % of signals that fire before 10:00 (`hod_filter_stack` §7 already scored and refuted C1/C2), and the causal day-gate family is the concurrent `hod_preopen_regime` pass's 162 declared cells. Not duplicated. |

**The Part 1 sentence: the losses live in the first ten minutes of the trade, not in a day, a week or
a symbol.**

## 1. Population, book, cost — unchanged from the two prior passes

`hod_filter_stack/pop.csv` → `score2.sig_set` at the declared base knobs; `trading.hod_break.run_book`
at 12/day, 4 concurrent; `$100` risk; the measured per-trade SIP NBBO cost model of
`hod_break/score.py::attach_cost` with the `score4` exit-side ratio contract
`{stop 0.875, eod 0.412, target 0.0}`. Membership cuts (early closes, test tickers, non-`daily_bars`
names) as before. Splits TRAIN 2025, VAL 2026-01→05.

**Bases**: `B0` = the shipped population (the thing that is in `config.yaml` today).
`B2` = the corrected base of `hod_filter_stack` §2 (consolidation proximity test off, rv upper cut
off). Both are reported for every cell that is scored on both; the base is part of the cell.

**Cost of a NEW exit.** Every exit this pass introduces (time stop, Rule D, fill-bar exit, breakeven
stop) is charged the **marketable** exit ratio `0.875`, the same as a stop — the conservative choice,
declared here so it is not chosen after seeing the numbers. Rule D's exit at `E − 0.5 R` is a limit
in ORB; it is charged as marketable here anyway.

**Fill-bar convention.** The engine buys the open of the bar AFTER the break bar (`e`), and the spec's
`walk_exit` starts checking exits at `e+1` — the fill bar's own low is NOT checked against the stop.
Every rule below that reads bar `e` therefore reads a bar whose outcome the shipped book ignores, and
acts at or after its close. Causality trace for each cell is in §4.

## 2. Declared cells — 20, no more

Primary metric, as in every pass of this programme: **% green weeks**, on BOTH splits, with weekly $
at $100 risk beside it; trades/week ≥ 10 to be a candidate; count-matched permutation null
(2,000 draws, per-week pick count held fixed) on every cell that is top-ranked on either split.

### A. Post-fill exits (need the bar walk; `path.csv` + `walk2.py`)

| # | cell | rule |
|---|---|---|
| P1 | **T10/0.25, B0** | at the close of the bar `e+9` (fill + 10 minutes), if the max high since the fill is `< E + 0.25 R`, sell at that bar's close. Thresholds are ORB's `LIVE_LOSERS.md` lever verbatim (10 min, +0.25 R) — **not retuned** |
| P2 | T10/0.25, B2 | same on the corrected base |
| P3 | T5/0.25, B0 | same at fill + 5 minutes (the other rung ORB's validation declared) |
| P4 | **Rule D, B0** | ORB `trading/orb_touchgo_filter` shipped thresholds verbatim: at the close of the fill bar, if its low ≤ `E − 0.75 R`, exit at `E − 0.5 R` |
| P5 | Rule D, B2 | same |
| P6 | **fill-bar shape exit, B0** | at the close of the fill bar, if it closed in the **bottom half of its own high-low range**, sell at the next bar's open. (ORB's Rule M geometry, moved one bar later to where HOD's fill actually is) |
| P7 | fill-bar shape exit, B2 | same |
| P8 | **breakeven at +0.5 R, B0** | once the running max high reaches `E + 0.5 R`, the stop moves to `E`; the +2R target and the 15:55 flat are unchanged |
| P9 | P1 + P8, B0 | the time stop and the breakeven move together |

### B. Entry / level gates (no walk — fields already in `pop.csv` / `path.csv`)

| # | cell | rule |
|---|---|---|
| P10 | **Rule M as an entry veto, B0** | skip the signal if the BREAKOUT bar closed in the bottom half of its own range (`range_pos < 0.5`). Declared as a **falsification** cell: Part 1 says the sign is reversed, and this is the cell that says so on the book |
| P11 | touches veto, B0 | skip if the level was touched (a bar high within 0.5 %) **≥ 5 times** before the break bar |
| P12 | touches veto, B2 | same |
| P13 | duration veto, B0 | skip if `consol_bars ≥ 8` — the count of consecutive bars before the break whose low held within 4 % of the level. The brief's candidate (d) asked for `≥ N`; Part 1 says the sign is the other way, so the declared form is `< 8` |
| P14 | 20-day-high gate, B0 | keep only signals whose level is at or above the prior 20-day high (`dist_20d_high_pct ≥ 0`) — the one bar-free gate with the same sign on both splits in Part 1 |
| P15 | first-break-only, B0 | keep only `n_break == 0` (the day's first qualifying break). Part 1 says TRAIN prefers the RE-break and VAL is flat, so this is declared as the **frequency-safe** side and reported with its complement |

### C. In-day kill switch (causal, day-level, computed on the booked sequence)

| # | cell | rule |
|---|---|---|
| P16 | **kill switch K = 3, B0** | after the 3rd consecutive booked trade of the session exits at a stop, take no further entries that day. Applied INSIDE the book: entries are taken first-come, a trade's exit is known before a later entry is decided iff `exit_m ≤ entry_m` of the later signal — only stops already CLOSED at the candidate's entry minute count |
| P17 | kill switch K = 2, B0 | same at 2 |

### D. The stack

| # | cell | rule |
|---|---|---|
| P18 | **best stack, B0** | the single best cell of P1–P17 on the pre-committed selector (§3) combined with the shipped book; if two rules are selected, both |
| P19 | best stack, B2 | same on the corrected base |
| P20 | count-matched null for P18/P19 | 2,000 draws, per-week pick count fixed |

### Declared and NOT run, with the anatomy line that ruled each out

* **(c) ATR-normalised admission** — `dist_open/ATR14` quintiles are non-monotone and VAL's top quintile is −0.149 against TRAIN's +0.077 (Part 1 §5/6 table). The fixed +5 % floor is not the problem.
* **(f) family / sector confirmation** — `coh_by_t` is 10.6 % covered and 94 % of the covered rows are 0. There is no cohort to confirm against in this population.
* **(g) day gate on the losing-day signature** — the signature (`spy_ret_0930_1000`, `breadth_by_1000`) is not known before 10:00 and 43 % of signals fire earlier; `hod_filter_stack` §7 already scored the two causal forms (C1 +0.028 R TRAIN, C2 sign-flipping). The causal day-gate family is the concurrent `hod_preopen_regime` pass's 162 cells. Not duplicated here.
* **week gate** — ruled out by the runs test in Part 1 §3 before any cell was declared.

## 3. Bars and the selector — fixed before scoring

**Selector for P18/P19**: among P1–P17 with ≥ 10 trades/week on both splits, rank by
`min(green % TRAIN, green % VAL)`; ties broken by `min(total $ TRAIN, total $ VAL)`. No other
criterion may promote a cell into the stack.

**Claim bar (G1)**, identical to `hod_filter_stack`: TRAIN mean net R > 0 with t ≥ 2.0, ≥ 10
trades/week, AND selected-subset TRAIN **gross** ≥ +0.25 R (the measured cost is +0.2151 R/trade).
G2 (the VAL confirmation) is evaluated only on cells that pass G1.

**Live-exploration bar**: a positive point estimate on green weeks AND on dollars at $100 risk on
**both** splits, at ≥ 10 trades/week.

**TEST (2026-06-01 → 2026-09-11) is sealed.** No TEST number is computed unless every cell above is
scored, `FREEZE.md` is written, and `--test` is passed. If no cell passes G1, TEST is never opened.

## 4. Causality trace — every field a cell reads

| field | constructed from | known at |
|---|---|---|
| `range_pos` (P10) | the break bar's own o/h/l/c | the break bar's close, one minute BEFORE the fill |
| `e_low`, `e_close`, `e_high` (P4, P6) | the fill bar's own o/h/l/c | the fill bar's close, 1 min after the fill; P6 acts at the NEXT bar's open, P4 at that close |
| `mfe{k}` (P1, P3, P8, P9) | running max high from the fill bar to bar `e+k−1` | the close of bar `e+k−1`; the cell acts at that same close |
| `touch_n` (P11, P12) | bar highs strictly BEFORE the break bar, vs the level | the break bar's close |
| `consol_bars` (P13) | bar lows strictly BEFORE the break bar, vs the level | the break bar's close |
| `dist_20d_high_pct` (P14) | `high20` from `daily_bars` strictly before `day` | the prior close |
| `n_break` (P15) | count of this symbol-day's earlier qualifying break bars | the break bar's close |
| kill-switch state (P16, P17) | exits of THIS session's own booked trades with `exit_m ≤` the candidate's `entry_m` | the candidate's entry minute |

No cell reads a field dated after the minute it acts on. `pdh_ratio` / `h5_ratio` (Part 1 only, not a
cell) come from `daily_bars` rows strictly before `day`.

## 5. Multiplicity

Declared decision cells **20**. Programme cumulative: **580** (through `hod_filter_stack`) **+ 162**
(`hod_preopen_regime`, its own PREREG §3) **+ 20 = 762.** Part 1's descriptive tables are not cells
and are not counted; they are reported in full so the reader can see everything that was looked at.
Expected largest |t| under a pure null over 20 cells × 2 splits ≈ 2.6–2.8.
