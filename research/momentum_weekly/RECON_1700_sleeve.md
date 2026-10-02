# RECON_1700_sleeve: build A (1700g_vol.py V2_N20) vs build B (REBUILD_1700_sleeve.py)
Method: logic untouched; copies in `recon/` (A_dump.py = A cut before its grid + holdings dump; B_dump.py = B + holdings dump + env
toggles AEXCL/NODOTS/ACOST/CLEAN/EQW, all OFF = shipped B, reproduces 1700_rebuild_weekly.csv exactly). Builder `recon/mk.py`,
compare `recon/compare.py`. Outputs: A_/B_*_holdings.csv, weekly_cmp_{base,all}.csv, top15_weeks.csv, exclusion_diff_held.csv,
summary_variants.txt. Same 503 weekly rebalance dates, first 2017-02-06, identical calendars.

## Result in one line
Two rule differences explain everything: (1) B's name regex is an unanchored SUBSTRING match, (2) B lets weights drift between
rebalances while A re-equalises to 1/20 every Monday. B + A's conventions reproduces A within 1.3 pts/yr, DD -41.6 %.

## Weekly agreement (A vs shipped B)
Mean Jaccard 0.905 (median 0.905, 47 % of weeks identical). By year: 2017 .83, 18 .91, 19 .99, 20 .75, 21 .81, 22 .99, 23 .94,
24 .99, 25 .88, 26 .97. Weekly return diff (B-A) sd 1.16 %, sum -0.8 %. B + all A conventions: Jaccard 0.999 (99.4 % weeks
identical), diff sd 0.03 %. Max-DD peak/trough identical in both builds (2021-02-08 -> 2022-07-05).
Top-15 |diff| weeks: 11 of 15 are Jan-May 2021 / Oct-Nov 2020 (Jaccard 0.54-0.74); the one-sided names are A = Chinese ADRs (NIO,
FUTU, DQ, SE, PDD, BILI, JD, GOTU) vs B = US growth/speculative (MSTR, MRNA, PTON, WKHS -51 %, MVIS -19 %, FCEL, CRWD...). The 4
others (2025-11-10, 2026-06-08/22, 2026-09-14) have Jaccard 1.0: same names, different weights (drift) = the drift effect.

## Variants (B toggled toward A; by year 2017..2026, CAGR, max DD)
| variant | 2017..2026 | CAGR | DD |
|---|---|---|---|
| A V2_N20 | 19.9 -6.1 14.8 80.7 16.5 -1.9 15.2 50.5 36.6 64.7 | 27.1 | -41.6 |
| B shipped | 17.0 -5.7 15.2 74.8 2.5 -0.4 15.7 56.5 31.7 58.8 | 24.9 | -49.9 |
| B + equal-weight Monday rebalance (EQW) | 17.7 -6.3 14.7 65.9 17.2 -2.1 17.5 50.0 38.6 61.1 | 26.2 | -39.5 |
| B + A exclusions, dotted tickers kept | 18.5 -6.2 15.0 86.6 -1.5 -0.8 13.3 56.3 27.9 61.7 | 24.8 | -51.5 |
| B + A cost cap (spread proxy <= 20 bps) | 17.3 -5.3 15.6 75.8 3.3 0.3 16.2 57.4 32.5 59.8 | 25.6 | -49.3 |
| B + A row cleaning (dups, non-positive OHLC) | identical to shipped (no rows affected) | 24.9 | -49.9 |
| B + A excl + EQW | 19.6 -6.5 14.4 78.2 15.6 -2.5 14.7 49.6 35.8 63.7 | 26.6 | -42.2 |
| B + all A conventions | 19.8 -6.0 14.9 79.4 16.5 -1.9 15.2 50.5 36.6 64.7 | 27.3 | -41.6 |
Effects interact (the single-cause deltas do not add: 2021 EQW +14.7, exclusions -4.0, cost +0.8, together +14.0).

## Cause 1: name exclusion (explains the different holdings, Jaccard 0.905)
A regex: `\bETFs?\b|\bETNs?\b|\bFUNDs?\b|\bTRUSTs?\b|\bWARRANTS?\b|\bUNITS?\b|\bPREFERRED\b|\bRIGHTS?\b` (9,829 symbols incl. test tickers)
B regex: `ETF|ETN|Fund|Trust|Index|Warrant|Unit|Preferred|Depositary|Right|Notes|Bond|Portfolio` unanchored + any symbol with `.` or `/`
(12,096 symbols). A excludes nothing B keeps; B excludes 2,267 more. Among names ever held, 32 differ, all B-only exclusions,
none a fund: substring hits NFLX ("netflix" contains ETF; 63 A-weeks), UNH, UAL, URI, UTHR, UMC, BTSG ("Unit"/"Right"), and
every ADR via "Depositary" (SE 86 wks, NIO 47, JD 42, PDD 41, FUTU 27, BABA 26, GOTU, BILI, DQ, BNTX, NTES...), plus RDS.A.
546 of A's 10,060 holding-weeks are names B drops. Mean weekly return of A-only holdings +1.09 % vs B-only +0.87 %.
Size: with drift on, A's list moves 2021 from +2.5 to -1.5 (the ADR names fell in Feb-May 2021) and 2020 from 74.8 to 86.6 (NIO).
CORRECT for live: A's word-boundary rule on the fund-type words. B's substring match is a bug (NFLX/UNH excluded by accident),
violating the no-accidental-behaviour rule. ADRs and class-share tickers (BRK.B) are tradable on Alpaca at the open, so keeping
them is right; if the owner wants ADRs out that must be an explicit rule. A's list does not catch "Index/Notes/Bond/Portfolio"
words, but none of the 32 held differing names is such an instrument, so no held fund slipped through.

## Cause 2: weight convention (explains 2021 gap +14.7 pts and 10 pts of DD)
A: weekly gross = simple mean of 20 names' open-to-open returns, i.e. a free rebalance to 1/20 every Monday; cost only on names
bought/sold. B: entrants sized 1/20, kept names drift (winners grow, 2021 high-flyers crashed at 2x weight). Shipped B DD -49.9 ->
-39.5 with EQW alone; 2021 +2.5 -> +17.2; 2020 falls 74.8 -> 65.9 (drift helped in the run-up).
CORRECT for live: equal-weight rebalance, a Monday order for the delta shares is executable (limit/MOO on ~20 names, kept names
trade a few %). A's missing cost on kept-name deltas is ~0.2-0.5 %/yr (rough: |dw| ~ 0.2 % per name per week x half-spread);
add it explicitly in the paper sleeve. Drift is also executable but is the unintended behaviour of a 'top 20 equal weight' spec.

## Smaller causes
Cost cap: A clips the spread proxy at 20 bps (total <= 15 bps); B caps TOTAL cost at 20 bps; A also charges 15 bps when the
proxy is missing, B 0. Effect +0.0 to +1.0 pt/yr, CAGR +0.7 (B + A cap). B's cap is the more conservative; both use the
(high-low)/close*0.1 BAND, not measured NBBO, so neither is a cost estimate by the project rule.
History gate: A needs close[t-273] (274 bars), B hist_count >= 273: one day, immaterial. Vol window 252 returns and the 12-1
offsets (21/252 along each symbol's own bar sequence), prior-day signal, Monday-open to Monday-open returns: IDENTICAL in both.
Missing bars: A fills a missing forward return with 0 (flat) and keeps the name; B carries the last price, 3 write-off events;
a symbol with no prior-day bar is out of the pool in both. Row cleaning: no effect. Open-to-open with a Monday-open fill is the
right live convention (signal Friday close, order in the open auction); real fills will differ from the print.
Residual after all conventions: <= 1.3 pts/yr (2020: 79.4 vs 80.7; 2024 identical), mean Jaccard 0.999: unexplained but small
(candidates: A's 15 bps cost for missing proxies, cost base on drifted values, float32 near-ties).

## Verdict for the paper sleeve
Reconciled book = B with A's exclusions + equal-weight Monday rebalance (+ B's stricter cost cap): CAGR ~26.6 %, DD -42 %, by
year 19.6 -6.5 14.4 78.2 15.6 -2.5 14.7 49.6 35.8 63.7 (the "B + A excl + EQW" row; B's cap adds ~0 net here). The 2021 gap is
the weight convention, and the drawdown gap is the weight convention. Caveats: DD ~ -42 % is real and survives either build;
2020 (+78 %) carries the book; cost is band-based, so the DD/CAGR are before measured cost; holdings-level agreement of the two
builds does not validate the shared universe/panel (survivorship, adjusted prices), which neither build tests here.
