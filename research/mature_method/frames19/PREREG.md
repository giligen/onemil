# frames19 — PREREG (written and frozen BEFORE any scoring)

**Cell F56 — placebo decomposition of the passive mirror short.**

frames18 F55 found the first two cells in 1,266 to clear a full pre-committed bar: the PASSIVE mirror
short at offset `k = 1.0 %`, order life `w = 5` and `w = 10` minutes — net **+0.211 / +0.372 R** (w=5,
TRAIN/VAL) and **+0.246 / +0.329 R** (w=10), ex-top-5 % positive on both splits, fill rate ~15 % / 25 %.
F55's own REPORT named the caveat that could explain the whole headline on its own: **the filled rows
are precisely the rows that popped 1 % against the signal within `w` minutes**, so the book may be a
generic "fade a 1 % intraday pop in a small cap" and the volume signal may contribute nothing.

**F56 asks exactly that, and nothing else**: does the MIR2 volume signal add anything beyond the
touch-and-fade mechanics? Method = frames15's placebo decomposition (D1 / D3), applied to the passive
form instead of the reacting form.

Programme cell count: **1,266 → 1,268** (2 cells: w=5 and w=10, both at k=1.0 %). The two placebo
populations are **controls, not cells**. TEST (`day >= 2026-06-01`) is never opened; the walker and
the scorer both assert `max(day) < 2026-06-01`.

## 1. The three populations

All three are walked with an **identical entry, exit, cost and rail stack**. The ONLY thing that
differs is which (day, symbol, hour) the resting order is placed on.

* **S — the signal.** Verbatim frames18 `grid18.csv` rows at `k = 0.010`: frames16 `sw_*.csv` rows
  with `f_mir2 & gate5 & price >= $5 & ex-wrapper & ETB & day < 2026-06-01`. n = 2,294 signal rows
  over 333 sessions. Nothing is re-derived.
* **P3 — the same name-day at another hour, with no signal** (frames15 D3). For every (day, symbol)
  that appears in S, the SAME session is walked at **every** hour boundary `hour in 9..14`, and every
  hour that carries a frames16 candidate-signal row for that name-day (any of `f_mir2 / f_up2 /
  f_abs`) is **removed**. Same clock-hour rules as S: `ref` = the close of the last bar before the cut
  `(hour+1)*60`, order placed at the first bar at or within 5 minutes after the cut. The same causal
  `gate5` rail (session high up to that hour's close >= session open x 1.05) is required, as are
  `price >= $5`, ex-wrapper and ETB.
* **P1 — a matched NON-signal name** (frames15 D1/D2 in its matched form). For every S row, up to
  **3** symbols are drawn (numpy `default_rng(19)`, per-day, without replacement) from the same
  session's SIP tape pool (`bars_sip.db::fetch_log`, `n_bars > 0`), excluding every symbol that
  carries any frames16 `sw` row that day, excluding wrappers, non-ETB names and the `^Z[A-Z]ZZT$`
  test tickers. A candidate matches when, **on the previous trading day** (`cache.db::daily_bars`,
  causal):
  * `|log(prev_close_cand / prev_close_sig)| <= log(1.25)` — same price band, and
  * `|log2(adv20_cand / adv20_sig)| <= 1` — same ADV bucket (within a factor of 2), `adv20` = mean
    daily share volume over the 20 trading days ending the previous session.

  The draw is placed at the **same clock minute** as its S row (same hour cut), and must pass the same
  `gate5`, `price >= $5` rails. S rows with no qualifying match are reported as match attrition, not
  dropped from S.

## 2. Entry / exit / cost — identical for all three

* **Entry**: a resting SELL limit at `ref x 1.01`, placed at the close of the cut bar, live for `w`
  bars (`e .. e+w-1`), filling **at the limit** on the first bar in that window whose HIGH >= limit.
  No touch in the window = UNFILLED, 0 P&L, never a loss. Entry charged **zero**.
* **Exit**: `stop = entry x 1.02`, `R = stop - entry` (so `rpct = 0.02` exactly for every row in every
  population), bare exit (EOD at `m >= 955` -> stop -> nothing else), walked from the fill bar + 1 by
  `frames16/short_walk.py::swalk` imported verbatim.
* **Cost** (PRIMARY): the exit leg is charged `0.5 x sp_hat / entry / rpct` in R, where `sp_hat` is
  the **measured** NBBO spread at `(exit day, symbol, exit_m)` from the pooled quote table
  (`frames16/nbbo16.csv` + `frames17/nbbo17.csv` + `frames18/nbbo18.csv`) when that leg was measured,
  and otherwise `entry x` the **measured price-decile median of `sp/mid`** from that same pooled table
  (deciles cut on `mid_med`, fitted once, applied identically to S, P3 and P1). Coverage % is reported
  per population. No entry cost anywhere.
* **Cost (SENSITIVITY)**: all three populations charged one common constant — the S-population mean
  measured cost in R. Reported alongside; the differences under it are purely gross.

## 3. Rails and the one declared asymmetry

* `gate5`, `price >= $5`, ex-wrapper and ETB are applied to all three populations (§1).
* **Reg SHO 201 (SSR)**: S keeps F55's rail exactly (an SSR-active fill is void unless the limit is
  strictly above the measured NBB at that minute; NBB unavailable or SSR undetermined -> VOID). P3 and
  P1 have **no measured NBB at their fill minutes**, so the rail is NOT applied to them; their
  SSR-active share is measured and reported instead. This is a declared ASYMMETRY. It is bounded:
  in F55 the rail removed **5 of 357** touches at k=1.0 %, w=5 (1.4 %), so it cannot move a 0.10 R
  difference. S is reported **both** with and without the rail so the asymmetry is auditable.
* Borrow: the ETB flag is today's snapshot, not the trade-date state, and the borrow fee is assumed 0
  for an intraday position. Declared assumption, unchanged from frames16/17/18, repeated regardless of
  verdict.

## 4. Pre-committed bar (the verdict rule, per window w in {5, 10})

1. `net(S) - net(P3) >= +0.10 R` on **BOTH** splits, each difference with **day-clustered t >= 2.0**.
2. `net(S) - net(P1) >= +0.10 R` on **BOTH** splits, each difference with **day-clustered t >= 2.0**.

Differences are means of filled rows; the day-clustered t of a difference is computed on the pooled
two-sample book with a day cluster (a day contributes one residual across both populations —
`common15.clust_t1` applied to the demeaned stack, two-sample form).

**Fill-rate comparability gate.** If `|fill_rate(P) - fill_rate(S)| > 10 pp` for a placebo P, the
comparison against that placebo is **VOID** for that window, and BOTH a fill-conditioned version (as
above) and a **touch-conditioned** version (every row scored, unfilled = 0 P&L, so the mean is over
all signal/placebo rows) are reported. A VOID comparison cannot produce a pass.

Both differences must clear for a window to be a finding. **Anything less than both is a REFUTATION of
the volume signal's contribution**, not a partial pass.

## 5. Reported for every population and window

n rows · n filled · fill rate · SSR-active share · gross R · net R with **iid and day-clustered SE and
t** · **ex-top-5 %** (drop the best 5 % of `rr` by value) · exit-leg measured coverage % · the two
differences with day-clustered t · the touch-conditioned versions · P1 match attrition · the
hour-stratified P3 mean (weights = S's hour distribution) as well as the raw P3 mean, because P3's
hour mix is mechanically different from S's and the hour determines how long the position can run to
the 15:55 EOD.

## 6. What this pass will not do

No TEST. No touch to `trading/`, `config.yaml`, `orb.yaml`, any engine, systemd, cron, or any order.
No re-fit of any threshold, no new signal family, no new exit family, no new grid. A positive verdict
here is a recommendation to pre-register a TEST read, never a change made in this pass.

## 7. Mid-run changes

Any change to anything in this file after the freeze is recorded verbatim in the REPORT.
