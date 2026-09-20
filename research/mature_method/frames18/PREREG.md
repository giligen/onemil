# frames18 — PREREG (written and frozen BEFORE any scoring)

**Cell F55 — the passive mirror short: is the F52 lead a NEIGHBOURHOOD or a lone cell?**

frames17 F52 walked a resting SELL limit at `ref x (1+k)` on the frames16 arm3 MIR2 mirror-short
signal, 5-minute order life, fill AT the limit on the first bar whose high touches it, entry charged
ZERO, exit charged the measured NBBO half-spread. `k=0.3 %` DIED on its own deciding table (adverse
selection on TRAIN). `k=0.6 %` cleared the deciding table (TRAIN margin **+0.002 R** — noise — and
VAL **+0.285 R**) at net **+0.129 / +0.320 R**, but its ex-top-5 % was **NEGATIVE on TRAIN
(−0.023)** — the same tail kill that has killed every mirror-short cell in this programme. F52 called
it a LEAD and named the two things it could not answer: (i) whether the surviving cell is a smooth
region of `(k, w)` or a single lucky point, and (ii) two rails it assumed away.

**F55 asks exactly that**: is there a neighbourhood of (offset `k`, window `w`) in which the passive
mirror short's net edge survives **ex-top-5 % on BOTH splits**, with **no adverse selection**, and is
that neighbourhood **smooth** rather than a lone cell — with the two assumed-away rails now applied?

Programme cell count: **1,254 → 1,266** (12 cells, all declared here, all scored). TEST sealed.

## 1. Population — unchanged object, verbatim re-use

Exactly frames17 F52's population, no re-derivation: `frames16/sw_*.csv` rows with
`f_mir2 & gate5 & price >= $5 & ex-wrapper & day < 2026-06-01`, restricted to Alpaca-ETB names
(`research/fuckup_audit/O_halt/PASSIVE/borrow_flags.csv`, `shortable AND easy_to_borrow`, absent =
not shortable). Splits unchanged: TRAIN `day < 2026-01-01`, VAL `2026-01-01 <= day < 2026-06-01`.
TEST (`day >= 2026-06-01`) is **never opened**; the scorer asserts `max(day) < 2026-06-01`.

No second signal family, no second exit family, no TEST. Anything outside the 12 declared cells is
not run.

## 2. The grid — 12 cells, declared before scoring

`k in {0.4 %, 0.6 %, 0.8 %, 1.0 %}` x `w in {3, 5, 10}` minutes of order life = **12 cells**.
`k=0.6 %, w=5` is F52's surviving cell and is re-scored here under the new rails (its numbers WILL
move; the F52 values are not carried forward). `k=0.3 %` is not re-run — F52 refuted it on its own
deciding table and a refuted cell is not resurrected by a window sweep.

Entry mechanic unchanged from F52: a resting SELL limit at `ref x (1+k)`, placed at the close of the
signal-hour cut bar `e`, live for `w` bars (`e .. e+w-1`), filling AT THE LIMIT on the first bar in
that window whose HIGH >= limit. No touch in the window = UNFILLED, 0 P&L, never a loss. Entry
charged **zero** (a resting limit that is touched is PAID the spread).

Exit unchanged from frames16 arm3 spec A and F52: `stop = entry x 1.02`, `R = stop - entry`, **bare**
exit (EOD at `m >= 955` -> stop -> nothing else), walked from fill bar + 1 by
`frames16/short_walk.py::swalk` **imported verbatim**. Exit charged `0.5 x measured_spread(exit day,
symbol, exit_m) / rpct` in R, measured NBBO only (`frames16/nbbo16.csv` + `frames17/nbbo17.csv` +
`frames18/nbbo18.csv` for legs neither covers, same Alpaca SIP method as `frames16/nbbo.py`).
Exit legs with no measurable quote fall back to the **population mean cost**, exactly as F52 did, and
the coverage % is reported per cell.

## 3. The two rails F52 assumed away — now APPLIED to every cell

### (a) Reg SHO 201 (SSR)
F52 declared, without measuring, that a resting limit above the market is not what Rule 201
restricts. F55 applies the rail instead of arguing it.

* **SSR active** for a (day, symbol) from the first bar at which the session's running minimum low
  is `<= prev_close x 0.90`, and for every bar after it that day. `prev_close` = the previous
  trading day's `close` for that symbol from `cache.db::daily_bars` (Alpaca, the same vendor as the
  1-minute tape — no cross-vendor price-scale hazard). A symbol-day with no prior daily bar has
  **SSR undetermined** and is counted in the availability rail, NOT silently passed.
* When SSR is active at the fill bar, the fill is **valid only if the limit is STRICTLY above the
  NBB at that minute**. `NBB(day, symbol, m) = mid_med - sp_med / 2` from the measured quote table
  (the same SIP quote measurement used for the exit leg). If the NBB for that minute is
  **unavailable**, the fill is **VOID** (dropped from the filled book, moved to the unfilled side at
  0 P&L) and counted in the availability rail.
* Reported per cell and overall: **SSR-active share**, **SSR-voided share** (fills removed), and the
  **undetermined share**.

### (b) Borrow
The Alpaca ETB flag is already the rail (§1) and is already applied. Two things it does NOT cover,
stated once here as assumptions, unchanged from frames16/frames17: the flag is **today's snapshot**,
not the trade-date state (a name easy to borrow today may not have been in 2025), and the **borrow
fee is assumed 0** for an intraday position closed by 15:55. No hard-to-borrow rate is charged. This
is a declared assumption, not a measurement, and is repeated in the REPORT regardless of verdict.

## 4. Pre-committed pass bar — per cell, ALL of these

1. **Net R >= +0.10 R on TRAIN AND on VAL**, with **day-clustered t >= 2.0 on BOTH splits**.
2. **Ex-top-5 % (drop the best 5 % of `rr` by value) >= 0 on BOTH splits.** This is the kill F52's
   surviving cell failed and the primary question of this frame.
3. **The deciding table, net vs net, passes on BOTH splits with margin >= 0.02 R on each.** Unfilled
   rows are scored at frames16 arm3's own reacting fill NET of its own measured entry+exit
   half-spreads (`score17.py`'s corrected contract, re-used verbatim). `filled_net - unfilled_net >=
   +0.02 R` on TRAIN and on VAL. F52's `+0.002 R` TRAIN margin would FAIL this bar — deliberately:
   a noise-width margin is not a pass.
4. **TRAIN halves (`day < 2025-07-01` / `>= 2025-07-01`) same-signed** on net.
5. **Count-matched green-week share above the permutation null p50** (`common15.null_green`,
   unchanged), reported on both splits.

A cell failing ANY of 1-5 is not a pass. Bars 1-5 are read only after the rails in §3 are applied.

## 5. THE NEIGHBOURHOOD RULE (the reason this frame exists)

A cell that passes §4 **counts as a finding only if every grid-adjacent cell has the same net sign on
both splits**. Adjacency = one step in `k` or one step in `w` (von Neumann, not diagonal); cells on
the grid edge are judged on the neighbours they have. A cell that passes §4 while sitting next to a
sign flip is reported as an **ISOLATED CELL** — i.e. a coincidence of the grid — and is explicitly
NOT a finding, however good its own numbers are.

If no cell passes §4, the neighbourhood rule is moot and the frame reports the grid's shape anyway
(the sign map is the deliverable either way).

## 6. Reported for every cell

n (signal rows) · n filled · fill rate · SSR-active / SSR-voided / undetermined shares · gross R ·
net R with **both iid and day-clustered SE and t** · ex-top-5 % · deciding-table margin · TRAIN
halves · green % vs null p50 · exit-leg measured coverage %.

## 7. What this pass will not do

No TEST. No touch to `trading/`, `config.yaml`, `orb.yaml`, any engine, systemd, cron, or any order.
No re-fit of any threshold. No new signal family, no new exit family. A positive verdict here is a
recommendation to pre-register a TEST read, never a change made in this pass.

## 8. Mid-run changes

Any change to anything in this file after the freeze is recorded verbatim in the REPORT, per the
process note frames17 was corrected with.
