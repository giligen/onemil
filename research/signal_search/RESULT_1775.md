# RESULT 1,775 — headline type vs the squeeze tail of the gap-fade short (executed exactly per PREREG_1775 + 3 amendments)

Coverage 6,908/6,908 symbol-days queried (0 failures; winner/loser missingness gap 0). 7,515 headlines labelled by Haiku 4.5 (selftest 20/20; 1 re-ask, 0 NONE-fallbacks). No headline in window: 61 % of symbol-days.
Categories (strongest headline per symbol-day): NONE 4,191; PR_OTHER 917; EARNINGS 904; CONTRACT 313; ANALYST 248; MA 123; FDA 109; DILUTION 103.

PRIMARY (worst-5 % share odds ratio, day-clustered 95 % CI). TRAIN: only PR_OTHER > 2 (OR 2.20, 1.46–2.88, 44 of the worst-5 %); MA 1.80, FDA 1.86, CONTRACT 1.60, NONE 0.61. VAL: PR_OTHER 1.53 (0.94–2.25) — CI includes 1; strength >= 4 OR 1.19 (0.51–1.91). DILUTION is OR 4.99 on VAL (1.88–8.79) but 1.11 on TRAIN, so not a pre-declared veto and sign-unstable (one of 8 categories x 2 splits).

VAL book (n 3,540, net +79.5 bp, t 1.61): veto PR_OTHER drops 430 (88 % retained): net +53.1 (t 1.00), worst-5 % improves +138 bp (pass bar 500 bp; placebo p95 +61), mean change -26.5 vs placebo p5 -15.7 → FAIL. Veto strength >= 4 drops 564 (84 %): net +69.9, worst-5 % +6 bp (placebo p95 +69) → FAIL.
Read (c), 1774 10:30-cover book (present at scoring; net = r_ask1030 − 13 bp): VAL n 988, PR_OTHER OR 1.7 (1.0–3.2), worst-5 % +170 bp vs placebo p95 +118, mean +93 → +53; strength >= 4 OR 0.6. Not TRAIN-supported (OR 1.2, 0.6–2.0) → FAIL.
Cuts (d): n_headlines and recency terciles all OR 0.9–1.9 with CIs spanning 1 except VAL recency T2 (1.90, 1.18–2.93), not replicated in TRAIN.

Verdict: FAIL as a veto. Pre-committed null on the mean holds; the tail is not separable by headline type at the 500 bp bar. Caveats: categories are Haiku labels (one model, one prompt); window starts the prior business day 16:00 (holidays ignored); Haldane-corrected OR; worst-5 % is ~177 trades so CIs are wide (MDE large for rare categories, e.g. MA/FDA/DILUTION n 41–63 per split). Files: 1775_{fetch,classify,score,readc}.py, 1775_events.parquet, 1775_summary.csv, 1775_placebo.csv, 1775_readc.csv.

CHILDREN: (1) DILUTION-headline short only (VAL OR 5, mean −356 bp) as a pre-registered standalone cell on a fresh 2026H2 window, since TRAIN did not show it; (2) a no-headline-only book (61 % of days, lowest tail share) with 09:25 stream parity.
