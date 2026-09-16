# Stage C — PRE-REGISTRATION

Written 2026-09-16, **before any Stage C scoring run existed** (the only Stage C python that had run when this file
was written was verification: `C/parity_full.log`, `C/verify_rows.log`, `C/verify30.log`, none of which computes a
net R, a book or a gate). It is reproduced verbatim as §1 of `C/REPORT.md`; if the two ever differ, this file wins.

---

## 1. Input and its verification (must all pass before a single cell is scored)

`research/fuckup_audit/B/candidates4.csv` — 3,099,499 rows, 77 columns, built by `B/build_candidates4.py` over the
420 point-in-time days of `research/bf_zero/universe.csv`, finished with `DONE` + `EXIT=0`.

| # | check | pass condition |
|---|---|---|
| 1 | row count | CSV lines − 1 == the builder log's final `total` AND `build4_state.json` holds 420 days |
| 2 | header alignment | `head -1` == `build_candidates4.py::COLS`, element by element |
| 3a | `B/parity_smoke.py` on the FINISHED file | every `candidates3` row of the 5 shared family-configs matches `entry_next` / `rr_2r_next` / `entry_m` / `exit_m` / `why` with 0 missing |
| 3b | `B/verify_rows.py 5` | 5 random rows recomputed by a plain python loop straight from the bars, all fields match |
| 4 | 30-row sanity on 3 random days (`C/verify30.py`) | `entry` / `rr_2r` EXACT (not 1e-6) against `candidates3` |

**If any fails, Stage C stops and reports. Nothing is scored.**

## 2. The cells

The **115** pre-registered in `B/REPORT.md` §1: **14 family-configs × 2 fills × 5 outcomes = 140, minus 25 impossible
resting cells** (F11 ×2, F12 ×2, F13 ×1 are CLOSE-triggered — a resting order cannot express them, by construction).

Families: `F1 {"P":0.12}` · `F5 {"K":5,"X":0.04}` (reference only) · `F6 {}` · `F8 {"N":5|15|30}` ·
`F9 {"G":0.05}` · `F10 {}` · `F11 {base F8,N 15}` · `F11 {base F6}` · `F12 {base F8,N 15}` · `F12 {base F6}` ·
`F13 {"K":5,"X":0.04}` · `F14 {"N":15}`.

Fills: `next` = `o[i+1]` iff `≤ level×1.006` (the shipped HOD engine) · `rest` = `max(o[i], level)` iff
`≤ level×1.006` (an ORB-style resting stop-limit at the level).

Outcomes (the H1 stop cross and H8 exits, all computed inside the builder's walk):
`2R close-fill` (touch stop, +2R on a bar close) · `hold-to-close` (touch stop, flat 15:55) ·
`lock 1.75/0.5` (ORB static lock, hold) · `2R close-stop` (**H1 S1**: stop on a bar CLOSE at/below the level, filled
at the next bar's open ×0.999) · `2R stop-1%` (**H1 S4**: stop = structural − 1% of price, **R redefined**, so the
comparison is at constant $ risk).

## 3. Two windows — and which one is PRIMARY (declared now, before any number)

- **PRIMARY: `--min-entry-m 600`, entries ≥ 10:00.** Stage A adopted this window (`A/REPORT.md` §A4 item 3) for the
  base families, and PLAN §3 H5's mechanism (overnight return predicts the first half-hour NEGATIVELY; spreads are
  widest at the open) applies to every family here. It is applied **uniformly to all 14 configs**, so the primary
  pass is one rule, not a per-family choice.
- **COMPANION: all-day (entries 09:30–14:01).** Run and reported in full, as its own separately counted 115 cells.
  It is the reference the earlier stages used; it is not the primary.

**Gate cells: 115 + 115 = 230.** G1/G2 are applied inside each pass with that pass's own G1 count setting the G2 bar;
the pooled bar over all 230 is also reported, and it is the conservative reading.

## 4. The contract (Stage A's adopted contract (c), `A/acore.py`, verbatim)

```
half_cc = 0.5 * (spread_cc_bps / 100) / max(r_pct_of_the_variant, 0.05)          # R units
net     = rr − ENTRY × half_cc − EXIT[why] × half_cc
ENTRY   = 0.25  (next-open fill: an already-printed, ask-side price)
          1.00  (resting fill: an arrival execution)
EXIT    = stop 0.875 · lock 0.875 · eod 0.412 · target 0.875     ( target 0.0 ONLY under the (c') sensitivity )
```
Population: fill price ≥ $5 **of the model being scored**, `entry_m ≤ 841`, `r_pct ≥ 1.0 of the variant being
scored`, `range_so_far_pct ≥ 5` for every family except F1–F4. For the resting fill the $5 floor and the R ≥ 1%
floor are applied to the **resting** entry and its own R. Book: `trading.hod_break.run_book(rows, 12, 4)`.
Splits: TRAIN 2025-01-02..2025-12-31 · VAL 2026-01-01..2026-05-31 · TEST 2026-06-01..2026-09-11.
**The GROSS mean R is reported next to the net in every table.**

## 5. Gates (PLAN §1 H10)

- **G1 (TRAIN):** mean net R > 0, t ≥ 2.0, ≥ 5 trades/week.
- **G2 (VAL):** mean net R > 0, t ≥ 1.0, ≥ 55 % of weeks green, and weekly R ≥ (G1 passes ÷ 10, floored) × SE(weekly R).
- **G3 (TEST):** read ONCE, only for G2 survivors, only after the selection is frozen in writing in this report,
  via `SCORE5_READ_TEST=1`. Reported week by week whatever it says.
- Every G2 survivor additionally gets, **before** TEST is opened: the tail test (top 1 % and top 5 % of trades
  removed; winners capped at +3R), a per-month table, and a search-adjusted permutation p (500 day-label sign-flip
  draws, max TRAIN t over ALL cells of the pass).

## 6. Declared sensitivities (re-scores of the same cells — reported, never gate candidates)

| tag | what |
|---|---|
| **Q** `--queue-ok` | resting cells restricted to `rest_queue_ok == 1` (signal-bar volume ≥ 5 × the shares $100 of risk buys). `B/REPORT.md` §2.2 measured the unrestricted queue-OK rate at only 0.56–0.88, so **the honest H2 claim is the restricted one** and the H2 decision rule (≥ 95 % obtainable AND queue-OK) is judged here. |
| **T0** `--free-target` | contract (c'): the target exit pays 0 (only legitimate for an engine that RESTS its take-profit leg — the HOD bracket does). |
| **L** `--legacy-spread` | score4's 1.90/1.20/0.80/0.60/0.50 % band table, for the before/after. |

## 7. Declared paired comparisons (deltas on matched signals — not gate cells)

| id | comparison | how |
|---|---|---|
| **P1** | H2: resting vs next-open | on the signals **both** models fill (paired t on the same (day, symbol, fam, cfg, outcome)), and separately the mean of the **resting-only** signals the next-open cap threw away. `B/REPORT.md` §2.2's finding is that these two populations differ in sign. |
| **P2** | H1: each stop variant vs the touch stop | paired per signal, `2R close-stop` and `2R stop-1%` vs `2R close-fill`, in net R **and** in $ at a constant $100 of risk; the stop RATE must fall for adoption. |
| **P3** | H8: the lock exit vs hold | paired per signal, `lock 1.75/0.5` vs `hold-to-close`. |

**Adoption rules, carried verbatim from the PLAN.** H2: the resting fill replaces next-open as the reference only if
≥ 95 % of its fills pass obtainability AND the queue check, and the improvement holds on TRAIN and VAL. H1: a stop
variant is adopted only if the paired difference vs the touch stop is ≥ +0.05 R with t ≥ 2 on TRAIN, the sign agrees
on VAL, the stop rate falls, and the new stop's fill is obtainable.

## 8. What would make Stage C say "there is something here"

A cell that clears G1 **and** G2, whose gross is also positive, whose tail test (top 5 % removed) keeps the mean
positive, whose permutation p is ≤ 0.05 after the search adjustment over all cells of its pass — and only then is
TEST read. Anything short of that is reported as a closest miss with the smallest effect the test could have seen
(2.8 × SE per headline cell), per PLAN §1's phrasing rule.

## 9. The scorer

`B/score5.py` (pre-written in Stage B, smoke-tested only) is copied to `C/score5c.py`. The copy adds **reporting**
only — `--queue-ok`, the gross column, the per-trade SE and MDE, a per-month table and a dump of the booked trades
of G2 survivors — and changes **no** population rule, cost coefficient, book call or gate. The diff is listed in
`C/REPORT.md` §2. Population/cost/book/gate lines that differ from `B/score5.py` would be a contract break and are
not permitted.

Runs (all `nice -n 10`, `ulimit -v 1800000`, ONE at a time, detached with a log when > 2 min):
`C/c0_extract.py` (a lossless row-subset of candidates4 so later passes do not re-parse 2.0 GB; its identity with
the full file is re-run and checked, not assumed) → `C/score5c.py` primary (≥10:00) and companion (all-day) →
the three sensitivities → `C/c1_paired.py` (P1) and `C/c2_stops.py` (P2, P3) → permutation/tail/month for any G2
survivor → TEST, once, last.
