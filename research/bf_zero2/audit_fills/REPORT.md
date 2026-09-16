# Adversarial fill audit — the F6 "red-to-green" +2R book (2026-09-16)

**Claim under audit** (`research/bf_zero2/f6_2r_book.csv`): long the first 1-min bar whose HIGH reaches
`prev_close x 1.003`, filled AT `prev_close x 1.003`; stop = the lowest low before the entry bar, filled at
`min(stop, bar open) x 0.999` on the first later bar whose LOW <= stop; target = `entry + 2R` filled AT the
target on the first later bar whose CLOSE is through it; otherwise flat at the open of the first bar at/after
15:55; a 20 bps half-spread (of price, converted to R) charged on every exit except `target`; book = 4
concurrent / 4 per day, first-come. Claimed **+0.130 / +0.322 / +0.350 R per trade** on TRAIN 2025 /
VAL Jan-May 26 / TEST Jun-Sep 26 (996 / 408 / 272 trades).

**Method.** The full F6 candidate POOL behind the book was rebuilt from `candidates_full.csv` with the study's
own filters (fam F6, price >= 5, entry <= 14:01, r_pct >= 1, range-so-far >= 5%) = **25,876 candidates**, and
every candidate was re-walked bar by bar on the same tape the study used (`research/bf_zero/bars_sip.db`, then
`data/cache.db`). Each correction is applied to the POOL and the book rule (`trading.hod_break.run_book`, 4/4,
first-come, causal freeing) is re-run, so a correction that moves an exit minute also moves which trades win a
slot. Nothing is patched post-hoc.

**Parity.** The rebuilt baseline is 1,641 of 1,676 trades identical to the published book, same trade count per
split, **+0.124 / +0.320 / +0.353 R** vs the published +0.130 / +0.322 / +0.350 (the 35 differences are
pool-edge rows: one is the ticker `NA` read as NaN by the study's loader, the rest are 4th-slot ties). The bar
re-walk reproduces the study's stored `rr_e1c` to **4.4e-6** on all 25,876 rows; the only exit-reason
differences are the 675 rows where I relabel the "tape ended before 15:55" fallback (item 7).

---

## The table — mean net R per trade, same pool, book re-run for every row

| # | correction | TRAIN | VAL | TEST | green weeks (T/V/E) |
|---|---|---|---|---|---|
| — | **BASE — the study as published** | **+0.124** | **+0.320** | **+0.353** | 34/53 · 17/22 · 12/14 |
| 1 | entry = max(level x 1.003, entry-bar OPEN) | **−0.150** | **+0.057** | **+0.086** | 14/53 · 14/22 · 8/14 |
| 2 | stop charged inside the entry bar (pessimistic) | +0.121 | +0.324 | +0.358 | 35/53 · 17/22 · 12/14 |
| 3a | stop fill 25 bps through (study: 10) | +0.103 | +0.299 | +0.335 | 33/53 · 17/22 · 11/14 |
| 3b | stop fill 50 bps through | +0.069 | +0.264 | +0.305 | 30/53 · 16/22 · 10/14 |
| 3c | stop fill 100 bps through | −0.000 | +0.194 | +0.245 | 28/53 · 16/22 · 10/14 |
| 4 | target fill | *no correction — the rule is conservative* | | | |
| 5 | 15:55 exit 10 bps below that bar's open | +0.118 | +0.315 | +0.347 | 34/53 · 17/22 · 12/14 |
| 6 | drop positions > 1% of the 5-min $ volume | +0.146 | +0.494 | +0.322 | 33/53 · 18/22 · 13/14 |
| 6b | drop positions > 5% of the 5-min $ volume | +0.107 | +0.389 | +0.346 | 32/53 · 19/22 · 11/14 |
| 7 | tape-ends-early names booked at −1R | +0.106 | +0.312 | +0.353 | 34/53 · 17/22 · 12/14 |
| 7b | tape-ends-early names excluded | +0.121 | +0.319 | +0.353 | 34/53 · 17/22 · 12/14 |
| 8 | exit cost = MEASURED median spread (actively quoted) | +0.073 | +0.272 | +0.309 | 30/53 · 16/22 · 11/14 |
| 8b | exit cost = MEASURED mean spread (actively quoted) | +0.041 | +0.240 | +0.280 | 30/53 · 16/22 · 10/14 |
| 8c | exit cost = MEASURED median spread (whole sample) | +0.001 | +0.210 | +0.252 | 28/53 · 16/22 · 10/14 |
| 8d | entry slip 50 bps instead of 30 | +0.064 | +0.273 | +0.319 | 30/53 · 17/22 · 12/14 |
| 8e | entry slip 75 bps (half the measured median spread) | −0.003 | +0.200 | +0.262 | 25/53 · 17/22 · 11/14 |
| 8f | entry slip 150 bps (half the measured mean spread) | −0.174 | +0.030 | +0.074 | 15/53 · 13/22 · 9/14 |
| — | ALL corrections, study's own 40 bps cost | −0.029 | +0.266 | +0.087 | 22/53 · 16/22 · 7/14 |
| — | **ALL corrections + measured median spread** | **−0.066** | **+0.233** | **+0.050** | 18/53 · 15/22 · 5/14 |
| — | **ALL + 75 bps entry slip + measured median spread** | **−0.117** | **+0.168** | **+0.012** | 16/53 · 15/22 · 4/14 |
| — | **ALL + 75 bps entry slip + measured MEAN spread (worst case)** | **−0.143** | **+0.143** | **−0.014** | 14/53 · 15/22 · 4/14 |
| — | MID (only correction 1 + the measured median spread) | −0.196 | +0.014 | +0.046 | 12/53 · 12/22 · 7/14 |
| — | *informational*: real stop TRIGGER (a print at the level, not at level x 1.003) | −0.080 | +0.129 | +0.304 | |

t-stats: BASE 2.78 / 4.29 / 3.87 → ALL+measured median −1.74 / 3.46 / 0.63 → worst case −3.84 / 2.17 / −0.18.
Trade counts are unchanged (996 / 408 / 272) except where the liquidity filter drops 7 TRAIN trades.
Row-level detail incl. R totals and win rates: `variants.csv`; per-item counts: `diagnostics.md`.

## Verdict

**The book is NOT positive on all three splits under the all-corrections version.** TRAIN goes negative
(−0.066R, −65R over 53 weeks, 18/53 green weeks vs 34/53 published) and TEST falls from +0.353R to +0.050R
with t = 0.63 — indistinguishable from zero; in the worst-case cost version TEST is −0.014R. Only VAL survives,
and its survival is driven by the liquidity filter (row 6, +0.494R) rather than by the entry rule.

**The single correction that hurts most is #1 — the entry fill when the bar gaps through the level**
(−0.27 / −0.26 / −0.27 R per trade on its own; it removes ~75% of the claimed edge). Second is the cost model
(items 8/8c, −0.12R on TRAIN), third the stop slip (3b, −0.06R). Corrections 2, 5 and 7 are immaterial, and
item 4 runs in the book's favour.

---

## Item by item

### 1. Entry fill when the bar gaps through the level — FATAL
A stop-buy cannot fill below the bar's open. The study fills every entry at `prev_close x 1.003` even when the
entry bar OPENED above it.

* **55.1% of book trades (924 of 1,676)** have `open > fill price` (33.4% of the pool). Median gap-through
  **1.59% of the entry price**, p90 4.45%, max 16.7%.
* In the trade's own risk unit that is a median **0.58R**, mean **0.98R** of free entry edge per gapped trade.
* Those trades ARE the book: they contribute **93% (TRAIN), 73% (VAL), 85% (TEST)** of its total R. Mean net R
  gapped vs clean: TRAIN +0.221 vs +0.017 · VAL +0.398 vs +0.208 · TEST +0.498 vs +0.132.
* At POOL level the non-gapped candidates are flat-to-negative in two of three splits (net R −0.016 TRAIN,
  +0.050 VAL, −0.038 TEST). The "edge" is the fill assumption, not the pattern.
* Correcting only this: **−0.150 / +0.057 / +0.086**.

Why it is so prevalent: 81% of the book's trades enter in minute 571–572 (09:31–09:32); the red-to-green cross
usually happens inside those minutes, so the bar that first trades through `prev_close x 1.003` very often
opens already through it.

*Related (informational)*: the study's trigger is also stricter than a real stop order — it waits for a bar
high at `level x 1.003` and then fills there, whereas a live stop triggers on any print at the level. Using the
real trigger (still filling at 1.003) the book is **−0.080 / +0.129 / +0.304**; the trigger bar differs on 40%
of the pool.

### 2. Stop-out inside the entry bar — immaterial
The simulator only looks for stops from the bar AFTER entry. The entry bar's low is at/below the stop on
**1.2% of pool candidates and 7.0% of book trades (118)**. Optimistic (the study's assumption — the low
preceded our fill): as published. Pessimistic (charge the stop on the entry bar): **+0.121 / +0.324 / +0.358**,
i.e. very slightly BETTER, because 96 of those 118 stopped out anyway and the 13 that reached the target are
offset by the slot being released earlier. The intrabar path is unknowable; either way this is not a source of
the claimed edge.

### 3. Stop slippage — a real, steady tax
The stop fills at `min(stop, bar open) x 0.999`. Gap-downs are handled correctly: the mean gross R on stop
exits is **−1.155**, so the `min(stop, open)` rule already books 15.5% of R beyond −1R on average — gap risk is
not hidden. Only the 10 bps "through the stop" is optimistic for a market stop in a $5–20 name. At 25 / 50 /
100 bps: TRAIN +0.103 / +0.069 / −0.000, VAL +0.299 / +0.264 / +0.194, TEST +0.335 / +0.305 / +0.245. 41% of
book trades exit on the stop.

### 4. Target fill — CONSERVATIVE, no correction
The rule requires a bar CLOSE at/above `entry + 2R` and fills AT the target, so the bar necessarily traded
through the fill price: **zero impossible fills** (2,625 target exits in the pool checked). It errs the other
way — a resting +2R limit fills on a touch: **35.3% of book trades saw their HIGH reach +2R before the booked
exit**, and 38 of them went on to exit elsewhere (mean −0.707R instead of +2.0R), so the close-fill rule
*costs* the study about **+0.06R per book trade**. This is the only material correction favouring the book and
it is an order of magnitude smaller than #1.

### 5. The 15:55 exit — cost IS charged; the fill is mildly optimistic
Verified: `score3.py` line 40 and `verify_f6.py` line 23 charge the half-spread on every exit whose reason is
not `target`, `eod` included. The fill is the *open* of the first bar at/after 15:55 — a market order sent at
15:55 pays the bid, not the print. At 10 bps worse: **+0.118 / +0.315 / +0.347**. 25% of book trades exit this
way; the effect is small because R is wide (median r_pct 3.8%).

### 6. Liquidity — not a P&L leak at $100 of risk, but a hard ceiling above it
At $100 of risk the median position is **$2,626** against a median **$280,633** of dollar volume in the 5
minutes after entry (median 0.98% of it). Dropping trades above 1% of that volume *raises* the book
(+0.146 / +0.494 / +0.322) because the illiquid names are the poor performers (mean net R +0.059 vs +0.357).
**The caveat is size**: 49.7% of book trades already exceed 1% at $100 of risk; at a realistic **$2,000 of risk
per trade, 88% of the book would exceed 1%** of the 5-min dollar volume. This book cannot be traded at the
simulated prices at any size that matters.

### 7. Halts / tape ending before 15:55 — negligible here
675 pool candidates (2.6%) have no bar at/after 15:55 and are exited at the last available bar; only **15 reach
the book** (0.9%), contributing +4.6R of +349R. Booking them at −1R: +0.106 / +0.312 / +0.353; excluding them:
+0.121 / +0.319 / +0.353. Wider tape holes: 11.9% of book trades have a >= 15-minute gap between entry and exit
(a plausible halt) and they book −0.028R mean / −5.5R total — the study is not living off halted names. 6.9% of
book trades are in symbols with no bar at or after 15:55 at all.

### 8. Costs — the 40 bps spread is 2–4x too cheap, and the 30 bps entry slip does NOT double-count
Fresh SIP NBBO was pulled for a stratified sample of **872 of the book's own trades in the book's own fill
minute** (`fetch_spreads.py` → `book_spreads.csv`; 837 returned quotes). The existing spread study could not
answer this: its earliest sampled signal is 09:36 and 81% of this book fills at 09:31–09:32.

| sample | median full spread | mean |
|---|---|---|
| all 837 sampled book trades | **1.52% of price** | 2.93% |
| actively quoted only (>= 50 quotes in the minute, mid within 2% of the print) | **0.86%** | 1.23% |
| the study's assumption | 0.40% | 0.40% |

Per non-target exit that is a median cost of **0.21R** (whole sample) or **0.10R** (actively quoted) versus the
**0.05R** the study charges. Re-scored with per-price-band measured spreads: **+0.073 / +0.272 / +0.309**
(actively-quoted median), **+0.001 / +0.210 / +0.252** (whole-sample median).

*Double-count question — no.* The 0.3% inside the entry price is the ENTRY's half-spread-plus-slip; the 20 bps
charged at exit is the EXIT's. They are opposite sides of the trade, and the earlier double-count that score3
fixed was charging the entry twice, not this. If anything the entry is now **under**-charged: half the measured
spread exceeds 30 bps on **84%** of sampled trades (median half-spread 76 bps whole sample, 43 bps
actively-quoted). Re-walking with a bigger entry slip: 50 bps → +0.064 / +0.273 / +0.319; 75 bps →
−0.003 / +0.200 / +0.262; 150 bps → −0.174 / +0.030 / +0.074.

---

## The structural finding behind all of it

The F6 pool as a whole is worth nothing — **+0.027 / +0.113 / +0.048 R gross per candidate** over 25,876 rows.
The book's +0.13 / +0.32 / +0.35 comes from the first-come 4/day rule concentrating it into the 09:31–09:32
bucket, which is exactly where (a) 55% of fills are through the open, (b) spreads are at their widest of the
session (measured median 1.5% of price), and (c) the tape is fastest. Net R per trade by entry minute (pool,
study's own cost model):

| entry minute | TRAIN | VAL | TEST |
|---|---|---|---|
| 09:31–09:32 | +0.097 | +0.298 | +0.467 |
| 09:33–09:35 | +0.037 | +0.098 | −0.084 |
| 09:36–09:40 | −0.074 | +0.102 | +0.187 |
| 09:41–10:00 | +0.010 | +0.111 | −0.105 |
| 10:00–11:00 | −0.127 | +0.042 | −0.075 |
| 11:00–14:01 | +0.122 | −0.003 | −0.077 |

The claimed edge is one two-minute window whose fills are the least realistic in the whole session.

## Files
`extract_pool.py` (pool + parity) → `pool.csv` · `rewalk.py` (bar re-walk: corrections 1, 2, 3, 5, 7 + liquidity
and tape diagnostics) → `pool_rewalk.csv` · `rewalk2.py` (entry-price corrections + real-trigger check) →
`pool_rewalk2.csv` · `fetch_spreads.py` (real SIP NBBO at the fill minute) → `book_spreads.csv` ·
`score_variants.py` → `variants.csv`, `book_all_corrections.csv` · `diagnostics.py` → `diagnostics.md`.
Read-only throughout: no service touched, nothing written outside `research/bf_zero2/audit_fills/`.
