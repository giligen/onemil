# RESULT 1,755 — short the gap at the open (MOO/MOC, hard stop): KILL

Caches 1748/1746; rank ≤ 2,000 by 20-day dollar volume (borrow proxy), raw px ≥ $5 (shortable proxy), attention as 1,754. TEST never loaded. Cost 2 bp/leg, borrow 0–100 bp/day; stop charged at open×(1+s)×1.005 on daily bars.

Names/wk at g 15 %: 3.7 TRAIN / 4.6 VAL all prices; primary $5–20 only 2.0 / 2.3.

Primary (g 15 %, $5–20, s 15 %, borrow 50 bp), net bp (day-t):
- TRAIN 16–18: −10 (−0.13); TRAIN 19–20: +19 (0.22); VAL: +86 (1.12).

VAL net at borrow 0 / 50 / 100 bp: +136 (t 1.78) / +86 (1.12) / +36 (0.47). Ex-top-5 % of winners −51; capped +20 % +55; excess vs shorting the universe +98 (t 1.37); MDE 215. Years (net 50): 16 +152, 17 −34, 18 −152, 19 −29, 20 +28, 21 +225, **22 −126**, 23 +29. Half-spread sensitivity −8.

Tail: 21.6 % of VAL trades hit the stop; 5 %-worst mean −16.1 %, breaching the −8 % rule. No-stop book: VAL +124 (t 1.27) but worst 1 % −81 %, 5 %-worst mean −39.8 %. The stop costs 38 bp VAL / 65 bp TRAIN and does not fix the sign.

Kill rules hit: VAL t < 2, halves opposite, ex-top-5 % ≤ 0, 2022 negative, < 3 names/wk, tail. The 1,754 long fade is real, but net of borrow and squeeze the short is indistinguishable from zero.

Winners vs losers (causal terciles, TRAIN → VAL net bp): prior range < 7.1 % −202 → +11, ≥ 14 % +112 → +165; prior-day return > 4.6 % +123 → +132; price tercile flips sign. Intraday-high terciles are non-causal.

CHILDREN:
1. Short only g ≥ 15 %, prior range ≥ 7 %, prior-day return ≥ 0, $5–20; kill: TRAIN halves opposite, VAL net at 100 bp ≤ 0, ex-top-5 % ≤ 0.
2. Minute-bar entry: short at 10:00 only if price < open, stop at the morning high; kill: stop-hit ≥ 15 % or VAL net at 100 bp ≤ half-spread + 40 bp.
