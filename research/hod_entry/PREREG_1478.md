# PREREG — cells 1,478–1,480: PREDICT the big day at the arm bar (multi-feature, causal), and two trades that let the day reveal itself

FROZEN 2026-09-26 before any number. Owner: "How can you predict the lookahead you had last night that led to the 0.3 R.
Be creative." Programme count on the HOD line: 1,477 → 1,480.

## The prize, restated correctly
Cell 1,457's ceiling (full-day range ≥ 10 %) is +0.17 R on VAL — but the look-ahead cohort that produced the +0.3 R was
the STRICTER mover-day cohort (`review/1445_sipzero_builders.md`: MACD-builder rule = full-day range ≥ 10 % ∧ close ≥ $10 ∧
prior-day volume ≥ 1M): +0.41 R inside the cache-only fills and +0.30 R outside, under the 1,445 cost. With the verified
stop-limit exit (+0.14 R per stop ≈ +0.08 R per fill) the ceilings are ≈ +0.25 R (range) and ≈ +0.4 R (mover day) at
20–45 fills/week. Perfect foresight of the mover day ≈ $11K/month at $375 risk; a predictor that captures a third of it
clears the owner's bar. Rounds 1–2 tested single features with single cuts; this round asks the actual question.

## Base book, cost, exits
The 9,911 fills of cell 1,438 (correct levels), corrected cost per `PREREG_1457.md` (half-spread once, measured stop slip,
EOD fallback fix) AND the verified 20 bps stop-limit exit slip on stop exits (cell 1,463 holdout means 2.9 / 3.2 bps on
filled stops; the 12 % no-fill tail at its measured mean) — this is now the standard exit for every HOD number. TRAIN-H2 /
VAL as before; TEST sealed, read once for the single best passing cell.

## Cell 1,478 — supervised big-day predictor (the "lookahead" as a label)
* Labels (two, each its own model): L1 = full-day range ≥ 10 % (PIT daily bar); L2 = the mover-day rule (L1 ∧ close ≥ $10
  ∧ prior-day volume ≥ 1M). Base rates reported per holdout.
* Features at the close of arm bar j — ALL causal, each with a one-line timestamp proof in the feature file:
  1. from rounds 1–2 (already built, `cell_1445_features.csv` / `cell_1457_features.csv`): distance from open, rv at j,
     ask distance, spread at fill, time of day, R %, range to j, arm index, bar density, $ volume to j, PM $ volume, ATR %,
     prior-day range, float (snapshot, disclosed), gap vs prior close, level vs prior-day high and 20-session high;
  2. NEW — the stock's own state: prior-day volume and prior-day relative volume (volume / ADV20: "in play yesterday");
     consolidation volume shape (OLS slope of the last K bar volumes; mean last-K volume / mean bar volume of the day);
     count of higher lows in the last K bars; number of touches of the level (bars within 0.2 % of it) before the break;
     pullback depth from the running high before the consolidation (%); VWAP distance and VWAP slope over the last 15
     bars; a halt proxy (any RTH gap ≥ 5 minutes without a bar before j); day of week;
  3. NEW — the crowd: universe breadth at bar j (count and share of HOD-universe names ≥ 5 % above their open at that
     minute, from bars_sip.db + cache.db across the universe), the same breadth's 5-session average (the regime: big days
     cluster), the count of SIC-2 sector peers (multiday panel `sic2`) ≥ 5 % above the open at bar j, SPY return from the
     open to bar j and SPY's 5-session return;
  4. NEW — the tape: trigger-print size class (odd lot / round lot; cell 1,429 reported +0.33 vs +0.07 R report-only on
     void levels), odd-lot share of prints and mean trade size in the last 3 bars before j (from the 1,438 tick windows
     `sip_cache/`, which cover bars j and j+1 — where the window lacks bar j−2..j the feature is NaN and reported);
  5. NEW — symbol persistence: over the prior 60 sessions, the share of sessions where the symbol was ≥ 5 % above its
     open at 11:00 ET and closed above that 11:00 price (from bars_sip.db / cache.db; NaN if < 10 such sessions).
* Model: sklearn HistGradientBoostingClassifier, seed 1478, hyper-parameters chosen by 5-fold CV INSIDE TRAIN-H2 over a
  fixed grid (max_depth {3, 5}, learning_rate {0.03, 0.1}, max_iter {200, 600}, min_samples_leaf {50, 200}); NaN handled
  natively; monotonic constraints none. Fit once on all of TRAIN-H2; applied ONCE to VAL. Also a logistic regression on
  the same features (the simple model) — reported beside.
* Selection rule (frozen): keep fills with predicted probability ≥ the threshold that keeps the top TERCILE of TRAIN-H2
  fills; report the deciles too (report-only).
* Pass bar on the kept VAL fills: mean net R ≥ +0.15, day-clustered t ≥ 2.5, ex-top-5 % > 0, ≥ 3 fills/week at 12/4,
  dropped mean < kept mean on both holdouts, AND the model's VAL AUC ≥ 0.60 for the label it was trained on, AND a
  label-shuffled placebo model (seed 1478) gives VAL AUC ≤ 0.53 and a kept-set mean within 0.05 R of the whole book.
  Feature importances and per-feature causal proofs are part of the result.

## Cell 1,479 — PYRAMID (the day reveals itself after entry)
Paired re-walk on the 1,438 fill paths (minute bars, `sip_rebuild.walk_path` semantics): 1/3 of the risk at the break;
if a bar's high ≥ fill + 1 R before the stop, add 2/3 at fill + 1 R (a second ask fill: charge the measured spread again)
and move the whole position's stop to the fill price (breakeven for the first third, −1 R on the added two thirds → the
combined risk stays ≤ 1 R); target 2 R from the ORIGINAL fill for the whole position; 15:55 exit; stop-limit slip on
stops. Report the book in R of the ORIGINAL risk unit. Pass: ΔR vs the base ≥ +0.05 on both holdouts, VAL t ≥ 2.5,
kept mean ≥ +0.10 on VAL (a lift on a negative book is not a book).

## Cell 1,480 — FAILED-BREAK FLIP SHORT
After a fill, if a print ≤ level − $0.01 occurs within 15 minutes of the fill and before the target: exit the long at
that print (this is the trade's new stop, replacing the consolidation-low stop for this cell — report the long side's ΔR
too) and enter SHORT at the NBBO bid at that print (tick cache; NaN if the window ends first — reported); short stop =
the break bar's high + $0.01; short target = 2 × (short stop − short entry) below the entry; cover at 15:55; shortable
flag from the asset dump required (share reported); SSR flag (prior close × 0.9 breached that day — rare on an up-day
population) reported and those trades excluded from the primary book. Costs: the short's entry at the bid, exit at the
ask (measured spread), stop-limit slip on the short's stop (35 bps fallback where unmeasured). Pass: short-leg mean net
R ≥ +0.15 on VAL with t ≥ 2.5, ≥ 3 shorts/week, ex-top-5 % > 0, TRAIN-H2 same sign.

## Independent check and consequences
1,478: a second agent rebuilds the FEATURE MATRIX from this prose (never reading the builder's code), refits the same
model with the same grid and seed, and compares the kept VAL set (Jaccard ≥ 0.95) and its mean (within 0.03 R); the
refuters' first lens is the timestamp proof of every feature with non-zero importance. 1,479/1,480: paired trade-level
rebuild ≥ 99 % within 0.01 R. PASS on any cell → dry run 5 sessions with the rule logged per arm (1,478: the model's
probability written to the ledger), then $50 real orders under the 9/25 fixes; the stop-limit exit ships with it.
FAIL on all → the "predict the big day" question is answered with a measured AUC and the owner report says so.

## Not allowed
Any feature using data after bar j (the labels are the ONLY look-ahead, and only as training targets); tuning outside
the fixed grid; choosing the threshold on VAL; reading TEST for more than one cell; dropping NaN rows selectively.

## Amendment 2026-09-26 (before any model is fit; the first execution attempt built only feature sets B and C)
1. **Single-source bar store (closes a leak the adequacy critic named).** In cell 1,438 each fill's bars came from
   whichever store held more bars, and store identity IS the look-ahead cohort (cache.db-complete ⇒ builder-selected ⇒ big
   day). Every bar-derived feature is therefore recomputed from ONE fresh store, `research/hod_entry/bars_fills_1478.db`:
   Alpaca SIP 1-minute bars (04:00–20:00 ET) fetched for all 9,911 fill symbol-days with a completeness gate (LOST count,
   ERROR if > 2 % lost); no cache.db or bars_sip.db bar enters any feature. Items 1's bar-derived columns (rv at j, range
   to j, bar density, $ volume to j, PM $ volume, arm index) are recomputed from it; the tick features (C) and the crowd
   features (B, a property of the day) stand. A METADATA-ONLY decoy model (features: store served in 1,438, its RTH bar
   count, the tick-window coverage flags) is fit first: if its VAL AUC > 0.55 the pipeline leaks the label and the cell is
   VOID; the real model excludes those columns and its kept set's cache-only share must be within 5 pp of the base 19.5 %.
2. **1,480 short R floor:** short stop = max(break-bar high + $0.01, entry × 1.01) (a doji break bar made R ≈ 0).
3. **Numbers seen before this amendment (disclosed):** an independent rebuilder ran its own sample of 1,479 (ΔR −0.05 /
   −0.02 R, under its reading of the add-leg cost) and of 1,480 (500-fetch sample, median short −1.79 R, mean undefined
   by the R ≈ 0 defect). No model number exists. The bars and thresholds above are unchanged.
