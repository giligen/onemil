# HOD-break pre-open / causal day-regime — pre-registration

Written 2026-09-19 **before any cell was scored**. Data acquisition (`fetch_idx.py`,
`fetch_imbalance.py`, `dayfields.py`) ran first and its availability/causality trace is reproduced in
§2; **no cell result was computed before this file was committed.**

## 0. The lead this pass tests

`research/mature_method/hod_filter_stack/REPORT.md` §7: a day gate on **SPY's 09:30→10:00 direction**
gave VAL 73.9–78.3 % green weeks and +$4,840–5,112 with gross ≈ net, separation +0.156 R TRAIN
(t 2.49) / +0.264 R VAL (t 2.81) — then **failed its causality trace**, because 43.3 % of the
population fires before 10:00 and the gate retro-applied a 10:00 fact to them. Its causal form C1
was +0.028 R on TRAIN (t 0.42).

That report reduced the whole question to ONE R number. The owner's three corrections (9/19) are the
shape of this pass:

1. *"if we have a 10am signal then we trade the stuff that happens after it"* — the 09:30→10:00 gate
   is **fully causal for the 56.7 % of signals firing at or after 10:00**, and that arm was never
   scored on the owner's metric (green weeks, red streak, worst week, weekly $), never laddered.
2. *"maybe at 09:40 there's also signal"* — make it a **curve**, not four points: gate times
   09:35 / 09:40 / 09:45 / 09:50 / 10:00 / 10:15 / 10:30, reporting separation **and** trades-kept per
   rung so the trade-off is visible.
3. *"the SPY previous 2 days is a signal for the first 30 min"* — a multi-day SPY context family,
   known at 09:29, which is the only family that can reach the early signals the intraday ladder
   cannot.

Plus the brief's original question: does the SAME mechanism exist in a day-regime field known
**before 09:35 ET**?

## 1. Population, splits, book, cost — inherited unchanged, nothing re-derived

Populations are the exact objects of `hod_filter_stack`: `pop.csv` (147,196 rows) → `score2.sig_set`.

| id | definition |
|---|---|
| **B0** | the shipped live config (consolidation K=5/X=4 %, `rv ∈ [1,5)`, dist ≥ 5 %, price ≥ $20, `r_min` 1 %, cap 0.6 %, spread ≤ 100 bps and ≤ 15 % of R, 12/day, 4 concurrent, last entry 14:00, +2R close-fill target, 15:55 flat) |
| **B2** | B0 with the consolidation proximity test OFF and the `rv < 5` upper cut OFF — the corrected base that was VAL-positive (+$893, gross +0.083) |
| **B3** | B2 + spread ≤ 8 % of R — used ONLY inside interaction F2 |

**Reproduction gate**: B0 and B2 are re-derived here and must match `hod_filter_stack/REPORT.md` §2
to the dollar (B0 TRAIN 1,688 tr / −$14,835 / 41.5 % green; B0 VAL 820 / −$4,128 / 43.5 %;
B2 TRAIN 1,622 / −$17,346; B2 VAL 706 / +$893) **before any cell below is read**.

Splits: **TRAIN** 2025-01-02→2025-12-31 · **VAL** 2026-01-01→2026-05-31 · **TEST**
2026-06-01→2026-09-11, **SEALED** behind `FREEZE.md`. Cost: the measured per-trade Alpaca SIP NBBO
with the declared (price band × hour band) imputation; break-even **+0.2151 R/trade**. Risk $100.
Membership cuts identical (early closes, test tickers, names absent from `daily_bars`).

**The decision instant.** The earliest `entry_m` in the population is **577** (fill at 09:37); its
break bar is 576, closing at 09:37:00 ET. So a fact known at 09:35:00 precedes **every** decision in
the book, and the 09:35 bound in the brief is satisfied with two minutes to spare. Asserted in code.

## 2. Declared fields, each with its causality trace (built and audited BEFORE this file)

Trace format: field → source → construction → the raw timestamp at which it is fully known.

| # | field | source | construction | KNOWN AT | cov TRAIN/VAL |
|---|---|---|---|---|---|
| D1 | `spy_gap_pct` | Alpaca SIP 1-min SPY | prev session 15:59 bar close → 09:30 bar OPEN | 09:30:00 | 100 / 100 % |
| D2 | `spy_r5_pct` | Alpaca SIP 1-min SPY | 09:30 bar open → 09:34 bar close | **09:35:00** | 100 / 100 % |
| D2b | `spy_pm_ret_pct` | Alpaca SIP 1-min SPY | prev close → last premarket bar close (`m_et ≤ 569`) | ≤ 09:29:59 | 100 / 100 % |
| D3 | `qqq_gap_pct`, `qqq_r5_pct`, `qqq_pm_ret_pct` | Alpaca SIP 1-min QQQ | as D1/D2/D2b | 09:30 / 09:35 / ≤09:29:59 | 100 / 100 % |
| D4 | `qqq_imb_ratio`, `qqq_imb_side` | **Databento XNAS.ITCH `imbalance`** | last opening-cross (type `O`) message strictly before 09:30 (observed max `m_et` 569, sec 59); signed by `side` (B=+, A=−, N=0), normalised by `paired_qty` | ≤ 09:29:59 | 100 / 100 % |
| D5 | premarket breadth | — | **DECLARED AND DROPPED, see §2a** | — | — |
| D6 | `spy_prev_c2c_pct`, `spy_vol20`, `qqq_prev_c2c_pct`, `qqq_vol20` | `cache.db daily_bars` + the 1-min closes | prior-day close-to-close; 20-day realised vol ending T−1 | T−1 16:00 | 100 / 100 % (`qqq_vol20` 92.3 / 100 % — `daily_bars` QQQ starts 2024-12-30, so the first 20 sessions of 2025 have no vol20; those days FAIL OPEN, i.e. are never excluded by a `qqq_vol20` gate, and the count is reported) |
| D7 | `dow` | calendar | the date | T−1 | 100 / 100 % |
| M | `spy_ret_{1,2,3,5}d`, `spy_path2`, `spy_prev_close_pos`, `spy_inside_day`, `spy_outside_day` (+ QQQ twins) | the 1-min 15:59 closes / RTH high-low | strictly prior sessions only | T−1 16:00 | 100 / 100 % |
| T | `spy_ret_0930_g` for g ∈ {09:35, 09:40, 09:45, 09:50, 10:00, 10:15, 10:30} (+ QQQ twins) | Alpaca SIP 1-min | 09:30 bar open → close of the bar ending at `g` | **`g`** | 100 / 100 % |
| R | `spy_ret_open_sig`, `spy_dist_hod_pct` (from `pop.csv`) + QQQ twins built here | causal_filter `spy_1min` / this pass's QQQ | 09:30 → the signal minute; SPY/QQQ distance below its own running intraday high at the signal minute | the signal minute | 100 / 100 % |
| — | `spy_ret_0930_1000` applied to ALL signals | — | **THE FAILED FIELD** — carried only as the look-ahead reference row, never as a claim | 10:00:00 | — |

**The T-family causality rule, asserted in code**: a gate computed from bars closing at or before
`g` is applied ONLY to signals whose decision instant is strictly after `g`, i.e. `entry_m ≥ g + 1`.
Signals with `entry_m < g + 1` are **taken ungated** (they are not silently dropped), and the
pre-gate arm's own book is printed separately per rung.

**The M-family causality rule**: every M field uses only sessions strictly before T, so it gates
every signal including the early ones.

### 2a. D5 premarket breadth — DECLARED AND DROPPED, with the reason

The brief's D5 is "count of universe names up ≥ 3 % at 09:29 on premarket prints". The causal pool
is the live engine's own streamed universe (every `daily_bars` name with prev close ≥ $17 and
ADV20 ≥ 100 K at T−1, ≈ 3,600 names a session). No premarket source on this node covers it:

- `data/cache.db::intraday_bars_1min` is **RTH-only** (first bar 09:30) — the same finding as
  `hod_filter_stack` §3 (this was D1's leak on ORB; it is not available and will not be imputed).
- `research/bf_zero/bars_sip.db` **does** carry premarket (307,343 symbol-days, median 681 names a
  session, earliest prints 04:00 ET) — but its membership is the **end-of-day-selected** HOD
  candidate pool (symbol-days whose day high reached open × 1.05). A breadth count over that pool
  inherits exactly the look-ahead this pass exists to remove: days that ended with many movers have
  more names in the denominator. **Rejected on causality, not on coverage.**
- `data/research/orb_premarket_dollar_vol_nightly.csv` covers a median of **11 names a session** —
  ORB's ranked gap-up candidates, not breadth.
- The Databento breadth form (§2b) is over the cost cap.

D5 is therefore **not tested**. It is the one declared field this pass cannot reach, and it is
recorded as such rather than approximated.

### 2b. Databento cost, priced before any pull (`fetch_imbalance.py`, printed to stdout)

| form | dataset / schema | price | action |
|---|---|---|---|
| **QQQ only, 2025-01-02 → 2026-09-12** | XNAS.ITCH `imbalance` | **$0.3740** | **PULLED** — 224,107 messages, 424 sessions, 58,512 of them opening-cross messages strictly before 09:30 |
| breadth (`ALL_SYMBOLS`, same range) | XNAS.ITCH `imbalance` | **$4,180.24** | **NOT PULLED — over the $40 cap** |
| breadth, windowed to 09:28–09:30 only | XNAS.ITCH `imbalance` | $2.589 per session × 415 = **$1,074.31** (a request minimum applies, so windowing does not reduce it) | NOT PULLED |
| breadth, a symbol subset | XNAS.ITCH `imbalance` | $14.55 / 50 names, $30.28 / 100 names full-range (≈ $0.30 a name) — the ~3,600-name universe ≈ $1,080 | NOT PULLED |

**SPY is NYSE-Arca listed and is NOT disseminated on XNAS.ITCH**, so there is no SPY opening-auction
imbalance on this feed at any price; QQQ (Nasdaq-listed) is the index proxy. Stated, not imputed.

## 3. Declared cells

All gates are **"trade today / do not"** (or, for the T and R families, "take this signal / do not"),
applied to the pre-book signal set; the book (12/day, 4 concurrent) is then run on what survives, so
a removed pick never silently frees a slot for a later one. Terciles/medians are cut on **TRAIN**.
Every cell is scored on the owner's metric: **% green weeks (PRIMARY, every market week in the
denominator, a no-trade week is FLAT)**, longest red streak, worst week $, weekly $ at $100 risk,
total $, MDD, % green months, and **trades/week kept**.

### A. Pre-09:35 day gates (14 cells, on B0 **and** B2 = 28)
A1 `spy_gap_pct > 0` · A2 `|spy_gap_pct| ≤ TRAIN median` · A3 `spy_r5_pct > 0` ·
A4 `spy_r5_pct` TOP TRAIN tercile · A5 `qqq_gap_pct > 0` · A6 `qqq_r5_pct > 0` ·
A7 `qqq_imb_side ≠ A` (not sell-side into the cross) · A8 `qqq_imb_side = B` ·
A9 `spy_pm_ret_pct > 0` · A10 `spy_vol20` BOTTOM TRAIN tercile · A11 `qqq_vol20` BOTTOM tercile ·
A12 skip the worst TRAIN weekday (edges printed before scoring) · A13 `spy_r5 > 0 AND qqq_r5 > 0` ·
A14 `spy_gap > 0 AND spy_r5 > 0`.

### B. First-5-minute SPY signals, known 09:35 (8 cells on B2 + the 2 sign cells on B0 = 10)
F5M-1 first (09:30) bar return > 0 · F5M-2/3 first-bar range ÷ SPY 20-day ATR, TOP / BOTTOM TRAIN
tercile · F5M-4/5 first-bar volume ÷ its own 20-day 09:30-bar average, TOP / BOTTOM tercile ·
F5M-6 first-bar close position in its range ≥ 0.5 · F5M-7 **gap-and-go** (`sign(gap) = sign(r5)`) ·
F5M-8 **gap-fade** (`sign(gap) ≠ sign(r5)`).

### C. The T-ladder — time-segmented index gates (the owner's curve)
Gate times g ∈ **{09:35, 09:40, 09:45, 09:50, 10:00, 10:15, 10:30}**; index ∈ {SPY, QQQ};
threshold ∈ {sign > 0, ≥ +0.2 %, ≥ +0.4 %, ≥ +0.6 %}. Signals with `entry_m < g+1` are taken
ungated. = 7 × 2 × 4 = **56 cells on B2**, plus the 7 × 2 sign-only rungs on B0 = **14** → **70**.
Reported per rung: separation (kept-minus-rejected, gross and net, t), **trades/week kept**, and the
pre-gate arm's own book — the two axes of the trade-off. The rung that maximises weekly $ at
≥ 10 trades/week is named.

### D. Rolling causal form — one rule, every signal (16 cells on B2 + 4 sign cells on B0 = 20)
ROLL-S: SPY return 09:30 → the **signal minute**, ladder {> 0, ≥ +0.2 %, ≥ +0.4 %, ≥ +0.6 %} ·
ROLL-Q: QQQ twin · HOD-S: SPY within X % of its own running intraday high at the signal minute,
ladder {TRAIN top tercile (the prior pass's R3), ≤ 0.10 %, ≤ 0.20 %, ≤ 0.30 %} · HOD-Q: QQQ twin.

### E. Multi-day SPY context, known 09:29 (21 cells on B2 + 4 on B0 = 25)
M1 prior-**2**-day close-to-close: {> 0, ≥ +0.5 %, ≥ +1.0 %, ≥ +1.5 %, ≤ −0.5 %, ≤ −1.0 %, ≤ −1.5 %}
= 7 · M2 prior-1-day > 0, prior-3-day > 0, prior-5-day > 0 = 3 · M3 prior-2-day path: two-up /
two-down / mixed = 3 · M4 prior-day close position in its range ≥ 0.5, ≤ 0.5, inside day, outside
day = 4 · QQQ twins: M1 sign + M3's three = 4.

### F. Interactions — exactly these, no more (9 cells, B2 unless stated)
F1 T2 (`SPY 09:30→10:00 > 0` gating `entry_m ≥ 601`) × B2 — already inside C, cross-referenced ·
F2 T2 × the spread ≤ 8 %-of-R gate (i.e. on **B3**) · F3 T2 × `rv_profile ≥ 5` (the strongest
positive gate in the gate map) · **F4 = the owner's M5**: M1, M3 and M4's best rule each scored
separately on signals firing **09:35–10:00** vs **10:00–14:00**, to test his specific claim that
2-day context predicts the FIRST 30 minutes (6 readings) · **F5 = M6**: the best M1 rung ANDed with
the best T rung, against each alone on the same restricted population.

### Cell count and multiplicity
**Declared decision cells: 28 (A) + 10 (B) + 70 (C) + 20 (D) + 25 (E) + 9 (F) = 162**, each scored
on TRAIN and VAL = **324 cell×splits**. Screening/diagnostic rows (the pre-gate arms, the separation
curves, the look-ahead reference) are reported and counted separately.
**Programme cumulative: 580 (through `hod_filter_stack`) + 162 = 742.**
Expected largest |t| under a pure null over 324 cell×splits ≈ **3.4**, so §5's claim bar carries a
multiplicity-adjusted threshold as well as the standing t ≥ 2.0.

## 4. Null

Count-matched permutation on **every** cell: 2,000 draws permuting that cell's own per-trade P&L
across its own weeks with the per-week pick count held fixed; observed green % against the null mean
and [p5, p95]. A green % inside its own band is pick COUNT, not skill, and is reported as such.

## 5. The two bars, and the multiplicity adjustment

- **Claim bar — G1**: TRAIN mean net R > 0 with **t ≥ 2.0**, **≥ 10 trades/week**, and selected-subset
  TRAIN **gross ≥ +0.25 R** (the +0.2151 R measured cost plus a margin). **G2**: VAL same sign,
  **≥ 55 % green weeks**, VAL gross ≥ +0.25 R. **G-mult**: because 324 cell×splits are scored, a cell
  that clears G1+G2 must additionally carry |t| ≥ **3.4** on TRAIN to be called a finding rather than
  the expected maximum of a null. Only a cell clearing G1+G2 opens TEST, **once**, after `FREEZE.md`.
- **Live-exploration bar**: a positive point estimate on **both** green weeks and dollars at $100 on
  **both** splits, ≥ 10 trades/week kept, a stated mechanism, a causality trace that holds, and
  resolution inside one quarter at the cell's own frequency.
- **MDE, stated before scoring**: VAL is **22 W-FRI weeks** and TRAIN **53**. At those n the
  green-week share has a 95 % half-width of roughly ±21 pp (VAL) and ±13.5 pp (TRAIN) around 50 %
  (exact numbers recomputed and printed). **A VAL sign flip smaller than that is noise, not a
  verdict** — and equally, a VAL green-week share must beat ~71 % before it is separable from 50 %
  on 22 weeks alone. Per-trade MDE is recomputed per cell from its own n.

## 6. Verdicts available

Exactly one of:
- **SHIP-TO-DRY** — the exact `HodBreakParams` / day-gate diff for a dry-run variant, with the
  pre-committed stop.
- **STAY-DRY** / **STAY DEAD** — with the adequacy review answered and the MDE that states what this
  test could not have seen.

## 7. Rails

Reproduction gate on B0 (and B2) before anything is quoted. One python process, `nice -n 10`,
`ulimit -v 3000000`, checkpointed. `bars_sip.db`, `data/cache.db`, `data/trades.db` opened
**read-only**. No `config.yaml`, `orb.yaml`, systemd unit, cron, order or cache is written;
`hod_break` stays `enabled: true, dry_run: true`. TEST is not read until `FREEZE.md` exists.
