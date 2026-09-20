# frames20 — PREREG (written and frozen BEFORE any scoring)

**Cell F57 — the POWERED placebo test of the passive mirror short.**

frames19 F56 asked whether the MIR2 volume signal contributes anything beyond the touch-and-fade
mechanics of a resting sell limit 1 % above `ref`. The matched-non-signal control (P1) read
**+0.122 / −0.011 R** alone and the difference **S − P1 = +0.088 (t 0.52) / +0.382 (t 1.23)** at
`w=5`; MDE 0.35–0.87 R. That is **underpowered, not a null** — VAL had only 13 filled control rows.

**F57 re-runs the SAME test with more control rows**: **FIVE** matched non-signal names per signal
row instead of frames19's three. Nothing else about the construction changes.

> **Correction to the task framing, recorded before scoring.** The brief describes frames19's P1 as
> "ONE matched non-signal name per signal". frames19's `walk19.py` actually used `NMATCH = 3`
> (PREREG frames19 §1 says "up to **3** symbols are drawn"). F57's increase is therefore **3 → 5**,
> not 1 → 5. The expected power gain is correspondingly smaller (~√(5/3) ≈ 1.29× on the control
> leg), and the MDE printed below is the honest arbiter.

Programme cell count: **1,268 → 1,270** (2 cells: `w = 5` and `w = 10`, both at `k = 1.0 %`). The
secondary read (P1 alone as a book) is a **diagnostic on a control population, not a cell** — it
proposes nothing and no ship language may be attached to it. TEST (`day >= 2026-06-01`) is never
opened; walker and scorer both assert `max(day) < 2026-06-01`.

## 1. Populations

* **S — the signal.** Verbatim `frames18/grid18.csv` rows at `k = 0.010` (frames16 `sw_*.csv` rows
  with `f_mir2 & gate5 & price >= $5 & ex-wrapper & ETB & day < 2026-06-01`). **Not re-walked.**
* **P1×5 — matched non-signal names.** For every S row, **up to 5** symbols are drawn
  (`numpy.random.default_rng(57)`, per-day, **without replacement**) from that session's SIP tape
  pool (`bars_sip.db::fetch_log`, `n_bars > 0`), excluding every symbol carrying any frames16 `sw`
  row that day, wrappers, non-ETB names and `^Z[A-Z]ZZT$` test tickers. A candidate matches when, on
  the **previous** trading day (`cache.db::daily_bars`, causal):
  * `|log(prev_close_cand / prev_close_sig)| <= log(1.25)` (same price band), and
  * `|log2(adv20_cand / adv20_sig)| <= 1` (ADV within a factor of 2).

  The draw is placed at the **same clock minute** as its S row and must pass the same `gate5`,
  `price >= $5` rails. **If a bucket holds fewer than 5 candidates, all of them are taken and the
  shortfall is recorded** (reported as a distribution of bucket sizes and a total shortfall count).
  S rows with no qualifying match are reported as match attrition, never dropped from S.

  P3 (same name-day, other hours) is **not** walked in F57 — F56 already answered it and it is not
  the underpowered leg.

## 2. Entry / exit / cost — identical for S and P1 (unchanged from frames18/19)

* **Entry**: resting SELL limit at `ref × 1.01`, placed at the close of the hour cut bar, live for
  `w` bars (`e .. e+w−1`), filling **at the limit** on the first bar in the window whose HIGH ≥ limit.
  No touch = UNFILLED, 0 P&L. Entry charged **zero**.
* **Exit**: `stop = entry × 1.02` (`rpct = 0.02` for every row in both populations), bare exit,
  walked from fill bar + 1 by `frames16/short_walk.py::swalk`, imported verbatim.
* **Cost**: exit leg charged `0.5 × sp_hat / entry / rpct` in R — `sp_hat` = the **measured** NBBO
  spread at `(exit day, symbol, exit_m)` from the pooled quote table (`frames16/nbbo16.csv` +
  `frames17/nbbo17.csv` + `frames18/nbbo18.csv`) when available, else `entry ×` the measured
  price-decile median of `sp/mid` from that same pooled table. Fitted ONCE on the pooled table and
  applied to both populations identically.
* **Reg SHO 201 rail**: applied to S exactly as in frames18/19 (SSR-active fill valid only if the
  limit is strictly above the measured NBB; NBB unavailable → VOID). P1 rows carry `ssr_void = False`
  (frames19 convention), so the rail is, if anything, charged against S.
* `gate5`, `price >= $5`, ex-wrapper, ETB apply to both.

## 3. Statistics

* Day-clustered t on every mean (`frames15/common15.py::clust_t1`) **and** the iid t alongside it.
* **S − P1** = the two-sample difference of means with a day-cluster-robust SE (a day contributes one
  residual across both populations) — `diff_clust`, verbatim from `frames19/score19.py`.
* **80 %-power MDE** on each difference = `2.8 × |diff / t|` (the day-clustered SE × 2.8), printed
  for every comparison whether it passes or fails.
* **ex-top-5 % of the difference** (pre-committed definition, fixed here before scoring): within each
  split, drop the top 5 % of rows by **gross** `rr` **separately in S and in P1**, then recompute the
  difference of means and its day-clustered t on the trimmed samples.

## 4. PRIMARY decision bar (pre-committed; cells = `w ∈ {5, 10}`)

A window **PASSES** only if ALL of:
1. `S − P1 >= +0.10 R` on **BOTH** TRAIN and VAL, and
2. day-clustered `t >= 2.0` on **BOTH** splits, and
3. **ex-top-5 % of the difference `>= 0`** on **BOTH** splits, and
4. the P1 fill rate is within **10 pp** of the S fill rate on both splits.

If (4) fails the comparison is marked **VOID** and the fill-conditioned **and** touch-conditioned
versions are reported instead (every row scored, unfilled = 0), with the VOID label carried into the
verdict. The MDE is printed regardless. Anything short of all four is a FAIL; no rescue, no
post-hoc re-slicing of `k`, `w`, the stop width, the match tolerances or the draw count.

## 5. SECONDARY (report-only diagnostic, no bar, no ship language)

**Is "fade a 1 % pop" a book by itself on the matched universe?** For P1×5 alone, per window:
net R and day-clustered t, ex-top-5 %, the two TRAIN halves (`day < 2025-07-01` vs `>=`),
green-week share vs the count-matched permutation null median (`common15::null_green`, 2,000 draws),
and fills per week (`common15::week_shape`). This is a **control population**; a positive reading
here is evidence AGAINST the signal's contribution, never a proposal.

## 6. What would change my mind / what this cannot settle

* A PASS at both windows says the volume signal adds ≥ +0.10 R over a matched name that popped the
  same 1 %. It does **not** rescue F55's named mechanism caveat (entering 1 % above ref against a
  2 % stop is a mechanical head start) — that is a **stop-width** question and stays un-swept.
* A FAIL with MDE above the effect size is **"not detectable at this power"**, never "no edge".
  The MDE is reported in the verdict sentence itself.

## 7. Files

`walk20.py` (P1×5 walk → `p120.csv`), `score20.py` (→ `pops20.csv`, `diffs20.csv`), `REPORT.md`.
Nothing under `trading/`, `config.yaml`, `orb.yaml`, systemd or cron is touched.
