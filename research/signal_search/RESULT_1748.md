# RESULT 1,748 — no-news extreme-loser reversal, ranks 1,001–3,000 (D4 + D5) — KILL (Stage 1, TEST sealed)

Sources (as PREREG): Chan JFE 70(2) 2003 (ideas.repec.org/a/eee/jfinec/v70y2003i2p223-260.html); Tetlock RFS 2011 (Columbia pdf in CANDIDATES D4); Nagel RFS 25(7) 2012 (nber w17653).

**Data.** Alpaca news depth: earliest item 2015-01-01/02 for AAPL, F, AMD, PLUG, GPRO (SNAP from 2017-02, its IPO) — history adequate for 2016+. 57,338 events (day-t close-to-close ≤ −8 %, PIT rank 1,001–3,000 by 20d median $ volume, px ≥ $3, dv20 ≥ $2 M); 2,667 event days, one news query per day-chunk, all cached. Share with ≥ 1 news item in [16:00 ET t−1, 16:00 t]: 40.7 %. Dropped: 23 missing next-day bars, 42 raw-vs-adjusted scale mismatches.

**VIX.** `cache.db` has no VIX before 2024-12 (VIXY from 2024-12 only), so the stress proxy is SPY 20-day realised vol, terciles cut on TRAIN (15.0 / 42.5). Deviation from the PREREG, stated.

**Result (gross, open t+1 → close t+1, bp).**
| Split | NO-NEWS (n, per wk) | t (day-clustered) | NEWS | NO-NEWS − NEWS |
|---|---|---|---|---|
| TRAIN | +19 (11,259; 43) | 0.40 | +44 | −24 |
| TRAIN H1 16–18 / H2 19–20 | −21 / +34 | | | |
| VAL | **−30** (10,787; 69) | **−1.43** | −18 | −12 |

VAL: ex-top-5 % −106 bp, winners capped at +10 % −51 bp, placebo percentile 0 (the same names on later days did better), 42 % of events gap further down at the open, median 14-day range-ATR 8.6 % of price (R is not the constraint). The close t+3 exit is worse (VAL −32, TRAIN −71 bp). SPY-residual ≤ −8 % version: VAL −31 bp.

**Cost** (measured NBBO, 400 VAL no-news events): half-spread 78 bp at 09:30–09:35 on t+1 (median 53), 16 bp at the close, plus 4 bp → round trip 98 bp. VAL net −128 bp.

**VIX pattern.** TRAIN high tercile +131 bp (t 0.9), low −15, mid −40: the positive cell is March 2020 alone, and the VAL window has zero events in the high tercile (SPY vol never exceeded 42.5), so the mechanism cannot be shown on VAL.

**MDE** at 80 % power on VAL no-news: 59.5 bp, against the +40 bp kill line.

**KILL list hit:** VAL gross ≤ +40 and |t| < 2; NO-NEWS not above NEWS on VAL; high-VIX not the largest on VAL; TRAIN halves opposite sign; ex-top-5 % ≤ 0; net ≤ 0. Passed: ≥ 3 events/week.

**Caveats.** Only 7,292 of 23,207 name candidates have SIP bars (delisted names absent), a survivorship bias that favours a reversal, and the cell still fails. TEST 2024–26 not opened. Files: `1748_fetch.py`, `1748_score.py`, `1748_events.parquet`, `1748_summary.csv`, `1748_cost.csv`.
