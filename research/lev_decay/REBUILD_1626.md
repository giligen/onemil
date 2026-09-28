# REBUILD_1626 -- independent rebuild of cells 1,626-1,627

Generated 2026-09-28T18:06:53.066912+00:00. Built from PREREG_1626.md prose only; cell_1626.py / cell_1626_days.csv / RESULT_1626.md were never opened.

Pairs matched: 93 (own parser, `build_pairs()` in this file). Pairs with usable bars and scored: 81. Skipped (missing/short bars): 11.

Caveat: 30/35 underlyings have >1 matched pair (multiple issuers on the same underlying+factor, cross-joined long x short) -- these pairs share a leg and are NOT independent draws; day-clustered t (clustering on day, across all pairs) partially controls for this but 'share of pairs positive' is inflated by near-duplicate pairs. Not deduplicated further given the step budget.


## Hand-check sample (20 pairs, `random_state=1626` on the full pair list)

| Underlying | Lev | Long | Long name | Short | Short name |
|---|---|---|---|---|---|
| AAOI | 2.00x | AAOX | Tradr 2X Long AAOI Daily ETF | AAOZ | Tradr 2X Short AAOI Daily ETF |
| AAOI | 2.00x | AAOG | Leverage Shares 2X Long AAOI Daily ETF | AAOZ | Tradr 2X Short AAOI Daily ETF |
| AMD | 2.00x | AMUU | Direxion Shares ETF Trust Direxion Daily AMD Bull 2X ET | DAMD | Defiance Daily Target 2X Short AMD ETF |
| ASTS | 2.00x | ASTX | Tradr 2X Long ASTS Daily ETF | ASTN | Defiance Daily Target 2X Short ASTS ETF |
| AXTI | 2.00x | AXTX | Tradr 2X Long AXTI Daily ETF | AXTQ | Tradr 2X Short AXTI Daily ETF |
| AXTI | 2.00x | AXTL | Leverage Shares 2X Long AXTI Daily ETF | AXTQ | Tradr 2X Short AXTI Daily ETF |
| COHR | 2.00x | COHH | Leverage Shares 2X Long COHR Daily ETF | COHQ | Tradr 2X Short COHR Daily ETF |
| COHR | 2.00x | COHX | Tradr 2X Long COHR Daily ETF | COHQ | Tradr 2X Short COHR Daily ETF |
| CRCL | 2.00x | CCUP | T-REX 2X Long CRCL Daily Target ETF | CRCD | T-REX 2X Inverse CRCL Daily Target ETF |
| NVDA | 2.00x | NVDU | Direxion Shares ETF Trust Direxion Daily NVDA Bull 2X E | NVD | GraniteShares ETF Trust GraniteShares 2x Short NVDA Dai |
| ORCL | 2.00x | ORCX | Tidal Trust II Defiance Daily Target 2X Long ORCL ETF | ORCZ | Tradr 2X Short ORCL Daily ETF |
| ORCL | 2.00x | ORCU | Direxion Shares ETF Trust Direxion Daily ORCL Bull 2X E | ORCZ | Tradr 2X Short ORCL Daily ETF |
| SMCI | 2.00x | SMCL | GraniteShares ETF Trust GraniteShares 2x Long SMCI Dail | SMCZ | Tidal Trust II Defiance Daily Target 2X Short SMCI ETF |
| SNDK | 2.00x | SNDG | Leverage Shares 2X Long SNDK Daily ETF | SNDQ | Tradr 2X Short SNDK Daily ETF |
| SNDK | 2.00x | SNXX | Tradr 2X Long SNDK Daily ETF | SNDQ | Tradr 2X Short SNDK Daily ETF |
| SPCX | 2.00x | SPCH | Leverage Shares 2X Long SPCX Daily ETF | SPCQ | Defiance Daily Target 2X Short SPCX ETF |
| TSLA | 2.00x | TSLR | GraniteShares ETF Trust GraniteShares 2x Long TSLA Dail | TSLQ | Investment Managers Series Trust II Tradr 2X Short TSLA |
| TSLA | 2.00x | TSLL | Direxion Shares ETF Trust Direxion Daily TSLA Bull 2X E | TSDD | GraniteShares ETF Trust GraniteShares 2x Short TSLA Dai |
| TSLA | 2.00x | TSLG | Themes ETF Trust Leverage Shares 2X Long TSLA Daily ETF | TSLQ | Investment Managers Series Trust II Tradr 2X Short TSLA |
| TSLA | 2.00x | TSLG | Themes ETF Trust Leverage Shares 2X Long TSLA Daily ETF | TSDD | GraniteShares ETF Trust GraniteShares 2x Short TSLA Dai |

## Cell 1,626 -- PAIR-SHORT static

| Split | n pair-days | n pairs | mean bps/day | day-clustered t | share pairs positive |
|---|---|---|---|---|---|
| TRAIN (rail5) | 3242 | 14 | +7.45 | 1.27 | 71.4% |
| TRAIN (rail15) | 3242 | 14 | +3.49 | 0.59 | 71.4% |
| TRAIN (rail30) | 3242 | 14 | -2.45 | -0.42 | 35.7% |
| TRAIN (gross) | 3242 | 14 | +10.43 | 1.77 | 78.6% |
| VAL (rail5) | 15356 | 81 | +11.00 | 1.97 | 76.5% |
| VAL (rail15) | 15356 | 81 | +7.04 | 1.26 | 67.9% |
| VAL (rail30) | 15356 | 81 | +1.10 | 0.20 | 46.9% |
| VAL (gross) | 15356 | 81 | +13.97 | 2.50 | 84.0% |

## Cell 1,627 -- PAIR-SHORT vol-gated (active days only)

| Split | n pair-days | n pairs | mean bps/day | day-clustered t | share pairs positive |
|---|---|---|---|---|---|
| TRAIN (rail15 (only while active)) | 3242 | 14 | -2.45 | -0.42 | 28.6% |
| VAL (rail15 (only while active)) | 15356 | 81 | +2.34 | 0.47 | 55.6% |

Share of pair-days active under the 1627 vol gate (>=60% enter / <40% exit, annualised): TRAIN 67.1%, VAL 73.0%


## Vol-tercile calibration (theory vs realised, gross bps/day, VAL split)

| Tercile | n | mean sigma_daily | mean gross bps/day | mean theory bps/day | realised/theory |
|---|---|---|---|---|---|
| Q1_low | 5120 | 0.0265 | +7.10 | +7.27 | 0.98 |
| Q2_mid | 5118 | 0.0442 | +8.31 | +19.84 | 0.42 |
| Q3_high | 5118 | 0.0709 | +26.50 | +51.92 | 0.51 |

## Borrow-rail sensitivity, VAL, cell 1,626

| Rail | mean bps/day | day-clustered t |
|---|---|---|
| 5%/yr | +11.00 | 1.97 |
| 15%/yr | +7.04 | 1.26 |
| 30%/yr | +1.10 | 0.20 |

## Price-scale check (CLAUDE.md caveat 3)

Split-adjusted bars (`adjustment=split`) still contained 4 distinct symbol-day events with |daily return| > 90% (pre-fix, raw bars had 50 events up to +2,589% in one day -- unadjusted reverse splits). Every such event is treated as an uncorrected corporate-action artifact and NEUTRALISED to a 0% return for that symbol-day (both the P&L and the notional carried flat through it) rather than reported as real P&L. This is a disclosed limitation, not a resolution: a full corporate-actions reconciliation was out of scope for this rebuild's step budget. Sample (up to 15):

- IONX 2026-03-19 r_long=+1.90
- RGTX 2026-03-19 r_long=+2.85
- STSM 2026-03-23 r_short=+1.89
- STSM 2026-09-14 r_short=+0.95
