# RESULT 1,776 — 1,774 book by PRE-OPEN liquidity (L1/L2/L3): FAIL

Bar-absent diagnosis (1,066 of 2,269 rows lack a 10:30 bar): 1,020 (95.7 %) are symbol-days already in bars_sip.db (presence.in_db, n_rth 298-390 — RGTZ, DMAC, AEHL checked); 1762_cache/bars holds ONLY symbols that were absent from bars_sip.db, so "no bar" = "found elsewhere", not "no trading". 46 (4.3 %) have n_rth = 0 (no RTH minute bars anywhere). So bar-present is a build-route artifact, not a liquidity flag; the +366/−155 split is not a bar effect, and L1 reproduces part of it.

TRAIN cuts (book rows, coverage 100 % premarket files): L1 2.4 M / 17.7 M, L2 1.1 M / 4.5 M, L3 14.7 M / 54.9 M. VAL top terciles (net of 41.8 bp): L1 T3 n 342, +159 bp, day-t 2.87, ex-top-5 % +17, worst-5 % −1,723 bp (−17 %), weekly P10 −0.51 R, 3 names/day, bar-present 51 %; TRAIN +312 (halves +445 / +180, same sign). L2 T3 +32 (t 0.6; the LOW tercile T1 is the TRAIN winner +293, VAL +104, t 1.6 — sign reverses vs L1). L3 T3 +97 (t 1.8; TRAIN +82). L1 T1 and T2 are VAL −41 / +70, non-monotone.

Placebos, measured ask, net of cost: (i) same days, same L1 T3, non-book population names (pool 255 over 76 days, covers 82 % of events): mean −59, p95 −41 — beaten, but the pool is the lower-gap names beyond the top-10 book, so it is a weak comparator. (ii) same name +5d (n 39, thin): mean +173, p95 +357 — NOT beaten (any L1-liquid gapper fades by 10:30).

Joint gap-T3 × L1-T3 (TRAIN gap cut 17.6 %): VAL n 200, +266 bp, t 3.4, ex-top-5 % +128, worst-5 % −1,770, weekly P10 −0.56 R, 2 names/day (fails ≥ 3); TRAIN +526. Gap-T3 alone VAL +205.

Verdict FAIL: worst-5 % −17 % vs bar −12 %, placebo (ii) not beaten, joint fails names/day. Passed: VAL ≥ +50, t ≥ 2, halves, ex-top-5 %, P10. Pre-open L1 is predictive (VAL +159 vs T1 −41) but the tail and same-name fade remain. Independent rebuild not done.

CHILDREN: (1) tail control on L1-T3 / gap-T3: hard cap on cover loss (stop-buy at +X %) priced at the measured ask, with its own PREREG; (2) same-name +5d placebo with full pool (all population days, ≥ 100 n) to resolve the thin placebo.

## Independent rebuild (prose-only, `1776R_build.py`, 2026-11-14) — NOT CONFIRMED on the book size, CONFIRMED on pricing and sign
Per-trade pricing identical (342 matched trades, mean |net diff| 0.00 bp). The rebuild's L1 top tercile holds 496 VAL trades
(first build 342: a tercile-boundary/population difference, not pricing): VAL +144.6 bp net, day-t 2.73, ex-top-5 % +0.2 bp,
worst-5 % −17.0 %, weekly P10 −0.66 R, median 5 names/day; TRAIN +188 (t 3.5; Jul–Sep +282 / Oct–Dec +88). Placebos at the
measured ask, properly sized: same-name +5d n 456 mean +3 bp p95 +51 — CLEARED; same-tercile non-book n 573 mean −74 p95 −57 —
CLEARED. The reportable line is the rebuild's. Still failing the bar: worst-5 % (−17 % vs −12 %), ex-top-5 % ≈ 0.
