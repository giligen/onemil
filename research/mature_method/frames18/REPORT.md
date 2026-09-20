# frames18 — F55: THE PASSIVE MIRROR SHORT OVER A (k, w) GRID (2026-09-20)

Programme cell count **1,254 → 1,266** (12 declared in `PREREG.md`, 12 scored). TEST never opened
(`max day scored 2026-05-29`). No Databento spend; 584 new Alpaca SIP quote-minutes measured.

## Verdict

**Two cells pass the full pre-committed bar — `k=1.0 %, w=5` and `k=1.0 %, w=10` — and BOTH are
SMOOTH. All 12 cells are net-positive on both splits; net, ex-top-5 % and the deciding margin rise
monotonically with `k`. These are the FIRST cells in this programme to survive ex-top-5 % on both
splits.**

**And the headline is almost certainly arithmetic, not signal.** A post-hoc diagnostic (caveat #1)
shows the resting limit selects **adversely** in the signal at *every* cell, monotonically worse with
`k`: at the best cell the rows it fills would have returned **−0.319 R TRAIN / −0.178 R VAL** under
frames16's reacting entry against **+0.252 / +0.193 R** for the rows it skips. The book is positive
only because entering 1 % higher against a 2 %-of-entry stop is a mechanical ≈ **+0.5 R** head start,
which the pre-committed deciding table cannot see (its two sides use different entry prices).
**A LEAD with a named mechanism, not a pass to ship.**

## The 12-cell grid (MIR2 signal, ETB-only, gate5, price ≥ $5, ex-wrapper; 2,473 signal rows)

Net = gross − measured NBBO exit half-spread (entry charged zero); t = day-clustered; margin = filled
net − unfilled net (unfilled at frames16's reacting fill, net of its own measured entry+exit halves);
fill % is after the Reg SHO 201 rail.

| k | w | fill % | n filled | net TRAIN (t) | net VAL (t) | ex5 TRAIN | ex5 VAL | marg TR | marg VA | pass |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.4 % | 3 | 39.0 | 894 | +0.113 (+1.93) | +0.259 (+2.89) | −0.037 | +0.019 | +0.043 | +0.306 | no |
| 0.4 % | 5 | 47.4 | 1,087 | +0.139 (+2.63) | +0.209 (+2.80) | −0.005 | −0.012 | +0.019 | +0.185 | no |
| 0.4 % | 10 | 56.6 | 1,299 | +0.121 (+2.63) | +0.204 (+3.05) | −0.022 | −0.011 | −0.089 | +0.136 | no |
| 0.6 % | 3 | 24.8 | 568 | +0.114 (+1.59) | +0.252 (+2.17) | −0.042 | +0.001 | +0.054 | +0.259 | no |
| 0.6 % | 5 | 32.9 | 754 | +0.121 (+1.99) | +0.308 (+2.30) | −0.033 | +0.017 | −0.005 | +0.268 | no |
| 0.6 % | 10 | 43.5 | 997 | +0.133 (+2.56) | +0.285 (+2.73) | −0.015 | +0.031 | −0.065 | +0.172 | no |
| 0.8 % | 3 | 16.2 | 372 | +0.173 (+1.70) | +0.336 (+2.15) | +0.015 | +0.039 | +0.143 | +0.357 | no |
| 0.8 % | 5 | 22.3 | 511 | +0.173 (+2.25) | +0.348 (+2.74) | +0.014 | +0.078 | +0.082 | +0.348 | no |
| 0.8 % | 10 | 32.5 | 746 | +0.213 (+3.50) | +0.325 (+3.46) | +0.065 | +0.096 | +0.058 | +0.242 | no |
| 1.0 % | 3 | 9.9 | 228 | +0.208 (+1.53) | +0.395 (+1.96) | +0.026 | +0.066 | +0.193 | +0.426 | no |
| **1.0 %** | **5** | **15.3** | **352** | **+0.211 (+2.35)** | **+0.372 (+2.36)** | **+0.050** | **+0.081** | **+0.138** | **+0.378** | **YES** |
| **1.0 %** | **10** | **24.8** | **568** | **+0.246 (+3.47)** | **+0.329 (+3.04)** | **+0.090** | **+0.078** | **+0.111** | **+0.249** | **YES** |

Non-pass reasons by how often they bind: `ex5 < 0` (all six k ≤ 0.6 % cells, on TRAIN); `margin <
+0.02 R` (0.4/10, 0.6/5, 0.6/10 — negative = **adverse selection** on the pre-committed table);
green-week share ≤ null p50 (0.8/5 and 0.8/10 on VAL, 1.0/3 on TRAIN); `t < 2` (all w=3 at k ≥ 0.6 %).

## SE, fill rate, n (all cells in `cells18.csv`)

`k=1.0 %, w=5`: TRAIN n=219, clust SE 0.090 (t +2.35), **iid t +2.43**; VAL n=133, clust SE 0.158
(t +2.36), **iid t +2.17**. `k=1.0 %, w=10`: TRAIN n=343 (+3.47/+3.73), VAL n=225 (+3.04/+2.84). iid
and clustered t agree within ~0.2 everywhere — day clustering is not doing the work. Fill rate
9.9 %→56.6 %; exit-leg coverage 99.6–100.0 %; best cell 69 % EOD / 31 % stop, WR 52–56 %.

## Neighbourhood verdict

Sign map: **all 12 cells are `++`** (TRAIN net +, VAL net +) — every k × every w. Both pass cells are **SMOOTH** — neither is an isolated grid coincidence, and the `k` gradient is
monotone in net, ex-top-5 % and (bar the 0.4→0.6 dip at w=10) the deciding margin. What the rule does
NOT certify: a monotone gradient in a parameter that also changes the entry PRICE is exactly what a
mechanical effect looks like.

## Reg SHO 201 rail (PREREG §3a, applied to every cell)

3,668 touches. **307 (8.4 %) landed on an SSR-active bar** (session running-min low ≤ prev close ×
0.90; prev close from `cache.db::daily_bars`, same vendor as the 1-min tape). The limit was above the
measured NBB (`mid_med − sp_med/2`) in most: **SSR-voided share 1.0–2.2 % of touches per cell**
(5–27 fills; max k=0.4 %/w=5, min k=0.6 %/w=10). 11 rows had no prior daily bar (undetermined) and
were voided. The rail moves no verdict — F52's *declared* assumption that a resting limit above the
market is largely untouched by Rule 201 is **measured here and holds at the ~1.5 % level**.

## Caveats, read as an adversary would

1. **THE ONE THAT ALONE EXPLAINS THE HEADLINE — the head start, not the signal.** On the SAME metric
   (frames16's reacting P&L for the identical day/symbol/hour), the rows the resting order FILLS are
   far worse than the rows it SKIPS, at every cell, monotonically worse with k: Δ = −0.286 R TRAIN at
   k=0.4 %/w=5 → **−0.571 R TRAIN / −0.371 R VAL at k=1.0 %/w=5** → −0.632 R TRAIN at k=1.0 %/w=10.
   The book is positive because entering at `ref × 1.01` with a stop at `entry × 1.02` is a ≈ +0.5 R
   price head start that more than pays for the adverse draw. Obtainable (the bar's high reached the
   limit; the exit walk is unchanged) — but arithmetic about the stop width, not evidence the signal
   picks better shorts, and a bet ON that width: halve the stop and the head start halves while the
   adverse draw does not. **Stop width was NOT swept — do not sweep it post-hoc to rescue the cell.**
2. **The pre-committed deciding table is structurally blind to #1**: it compares a filled book at
   `ref × (1+k)` with an unfilled counterfactual near `ref`, so its margin grows with k for the same
   mechanical reason. The +0.138/+0.378 margins must NOT be read as "no adverse selection" — in F52
   a +0.002 R version of exactly this number was.
3. **Thin.** 15.3 % fill: 219 TRAIN + 133 VAL trades / 17 months = **$82/wk TRAIN, $214/wk VAL at
   $100 risk** ($148/$340 at w=10) — far below the $10K/month north star before scaling capital.
4. **Both pass cells sit at the grid EDGE (k=1.0 %)**, gradient pointing off the grid; a k=1.2–1.5 %
   extension is the obvious next ask — pre-register it, never just take it.
5. **Tail: genuinely not the story this time.** Median net +0.198 TRAIN / +0.175 VAL at the best
   cell, ex-top-1 % +0.157/+0.219 — the edge sits mid-distribution, unlike every prior mirror-short
   cell. This is the bar F52 failed and F55 clears honestly.
6. **Carried unchanged from frames16/17**: ETB flag is TODAY's snapshot, not the trade-date state
   (7.2 % excluded); **borrow fee assumed 0** intraday (PREREG §3b — declared, never measured);
   halts/LULD not modelled; uncovered exit legs fall back to population-mean cost (≤ 0.4 % of fills).
   **No independent rebuild of the grid walk exists** — not owner-ready until one reproduces the
   trade set day-by-day.

## Mid-run changes to pre-committed items

One: PREREG §3a said an SSR-**undetermined** row (no prior daily bar) is "counted in the availability
rail" without naming an action; `score18.py` **voids** such fills (conservative) — 11 rows of 3,668,
no verdict turns on it. Nothing else in PREREG.md changed after freeze.

Studies: `frames18/{grid_walk,nbbo18,score18}.py`, `frames18/{grid18,cells18,nbbo18}.csv`.
