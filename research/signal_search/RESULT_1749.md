# RESULT 1,749 — volume shock without a price move (GKM 2001, candidate H1) — KILL (Stage 1, TEST sealed)

Source: Gervais–Kaniel–Mingelgrin, JF 56(3) 2001 (onlinelibrary.wiley.com/doi/10.1111/0022-1082.00349). Cell 1,749.

**Build.** `1749_score.py` on the 1,748 bar cache (7,292 symbols, adjusted OHLCV, 2016-01..2026-11; survivor-only, rejection-only). Shock = volume_t highest or 2nd highest of own 50 bars; npm = |ret_t| <= 1.5 x 49-day median |ret|; entry open t+1, exit close t+5 (and t+20); excess = leg minus tier universe on the same day. PIT tiers from t-1. Windows spanning a >10 % adjusted/raw jump dropped. Signals begin 2016-03 (50-bar warm-up). TEST not opened.

**Signals/week (liquid / $10–100 M / rank 1k–3k), HIGH-npm:** 33–50 / 92–115 / 116–159 (kill line 3: passed).

**Primary (liquid, HIGH-npm, 5 d), gross excess bp:** TRAIN −8 (NW t −0.9; halves −16 / +2), VAL +10 (NW t 0.7, MDE 42 bp), ex-top-5 % VAL −111, cap +10 % −28, green months 58 %. Net of cost (round trip 2 x 13.8 + 4 = 32 bp) VAL −21. Years: 2016 −37, 17 −14, 18 −1, 19 +25, 20 −21, 21 −12, **22 −18**, 23 +62.
HIGH − LOW (liquid, VAL): +43 (t 1.9) with the LOW leg −26 (t −1.8); TRAIN −11. HIGH-all VAL +21 (t 1.8), the extreme-return HIGH leg +29 (t 2.0): the premium, if any, sits where the paper says it should not.
**Mechanism check (npm − extreme-return HIGH):** TRAIN +4 (t 0.3), VAL −15 (t −0.7): fails on VAL.
Other tiers: mid HIGH-npm VAL −11, rank 1k–3k −19 (t −2.1) and net ≈ −110 bp (1,748 half-spreads, loser-day biased). 20 d primary: VAL +26 (t 0.9), net −5.

**KILL list hit (6 of 9):** VAL gross <= 15; |t| < 2; TRAIN halves opposite sign; ex-top-5 % <= 0; 2022 negative; VAL net <= 0; npm not above extreme on VAL. Passed: npm > extreme on TRAIN, >= 3/wk.

**Caveats.** Row AQB 2017-01-09 (mid tier) shows r5 = +169x, an unadjusted corporate action the 10 % ratio guard missed; it touches the mid tier only (the liquid primary has max r5 +252 %). Mid/rank cost uses the 1,748 loser-day half-spreads (pessimistic). Daily resolution, survivor-only bars.
Files: `1749_score.py`, `1749_signals.parquet`, `1749_summary.csv`, `1749_yearly.csv`, `1749_run.log`.
