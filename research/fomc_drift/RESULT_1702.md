# RESULT 1,702 -- pre-FOMC drift (PREREG_1702.md, FROZEN; 10 cells; net of 2 bps round trip; bps)

Coverage (scheduled meetings/yr, Fed pages): 1994:8 1995:8 1996:8 1997:8 1998:8 1999:8 2000:8 2001:8 2002:8 2003:8 2004:8 2005:8 2006:8 2007:8 2008:8 2009:8 2010:8 2011:8 2012:8 2013:8 2014:8 2015:8 2016:8 2017:8 2018:8 2019:8 2020:7 2021:8 2022:8 2023:8 2024:8 2025:8 2026:8 2027:8
Years != 8: [2020]; events analysed (<= 2026-10-02): 261; future-dated rows in fomc_dates.csv: 10
Excluded (listed in fomc_dates.csv): 1994-02-28 conference call; 1994-03-24 conference call; 1994-04-18 conference call; 1994-07-20 conference call; 1994-12-30 conference call; 1995-01-13 conference call; 1995-03-10 conference call; 1995-04-28 conference call; 1998-09-21 conference call; 1998-10-15 conference call; 2001-01-03 conference call; 2001-04-11 conference call; 2001-04-18 conference call; 2001-09-13 conference call; 2001-09-17 conference call; 2003-03-25 conference call; 2003-04-01 conference call; 2003-04-08 conference call; 2003-04-16 conference call; 2003-09-15 pre-meeting panel of the next-day meeting (no announcement); 2007-08-10 conference call; 2007-08-16 conference call; 2007-12-06 conference call; 2008-01-09 conference call; 2008-01-21 conference call; 2008-03-10 conference call; 2008-07-24 conference call; 2008-09-29 conference call; 2008-10-07 conference call; 2009-01-16 conference call; 2009-02-07 confe
Bars completeness (missing/sessions): SPY.o930 0/2702, SPY.o1355 1/2702, SPY.o1400 3/2702, QQQ.o930 0/2702, QQQ.o1355 8/2702, QQQ.o1400 9/2702, IWM.o930 0/2702, IWM.o1355 13/2702, IWM.o1400 9/2702; LOST chunks []
Prior close = official daily close (Alpaca daily, adj=ALL); vs last 15:59 minute close: n 85, mean 3.85 bps, mean|.| 13.65, max|.| 82.23. Alpaca vs yfinance event-day close-to-close: n 85, max|diff| 9.75 bps.

| cell | n | mean | t | %pos | H1 | H2 | ex-top5 | ex-worst5 | worst | plc | diff (t) | diff same-wd (t) | MDE | PASS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SPY W1 | 85 | 14.6 | 2.17 | 52 | 16.6 (16-20) | 12.9 (21-26) | 3.8 | 20.7 | -153 | 3.6 | 11.1 (1.58) | 3.7 (0.54) | 18.8 | fail |
| SPY W2 | 85 | 16.3 | 2.42 | 55 | 13.9 (16-20) | 18.3 (21-26) | 6.7 | 23.6 | -174 | 3.9 | 12.4 (1.76) | 5.6 (0.81) | 18.9 | fail |
| SPY W3 | 85 | -2.4 | -0.77 | 39 | -2.0 (16-20) | -2.8 (21-26) | -6.9 | 0.6 | -62 | 0.0 | -2.4 (-0.71) | -7.7 (-2.43) | 8.8 | fail |
| QQQ W1 | 84 | 24.5 | 2.49 | 62 | 22.5 (16-20) | 26.1 (21-26) | 8.8 | 33.7 | -148 | 5.1 | 19.4 (1.92) | 12.0 (1.21) | 27.5 | PASS |
| QQQ W2 | 84 | 26.0 | 2.58 | 58 | 19.7 (16-20) | 31.2 (21-26) | 11.2 | 36.4 | -159 | 5.4 | 20.6 (1.98) | 12.8 (1.26) | 28.2 | PASS |
| QQQ W3 | 84 | -3.1 | -0.67 | 40 | -7.0 (16-20) | 0.1 (21-26) | -9.4 | 1.7 | -100 | 0.6 | -3.7 (-0.75) | -8.4 (-1.81) | 12.9 | fail |
| IWM W1 | 85 | 21.3 | 1.89 | 48 | 14.3 (16-20) | 27.2 (21-26) | 4.5 | 30.9 | -200 | 3.4 | 17.9 (1.55) | 13.1 (1.17) | 31.4 | fail |
| IWM W2 | 85 | 19.4 | 1.70 | 48 | 10.9 (16-20) | 26.6 (21-26) | 2.7 | 30.3 | -261 | 2.8 | 16.7 (1.42) | 12.0 (1.05) | 31.9 | fail |
| IWM W3 | 85 | 3.4 | 0.44 | 52 | -6.2 (16-20) | 11.4 (21-26) | -5.7 | 11.8 | -199 | -2.3 | 5.6 (0.71) | 2.6 (0.34) | 21.5 | fail |
| SPY W4 | 261 | 21.6 | 3.00 | 54 | 32.2 (94-11) | 8.6 (12-26) | 6.1 | 35.7 | -300 | 2.2 | 19.4 (2.65) | 18.7 (2.60) | 20.2 | fail |

W4 SPY close-to-close by era (net bps): 
- 1994-2011: n 144, mean 32.2, t 3.11, %pos 60, placebo 0.7, diff 31.4 (t 2.99), same-weekday diff 31.0 (t 3.00), MDE 28.9
- 2012-2026: n 117, mean 8.6, t 0.88, %pos 47, placebo 4.0, diff 4.6 (t 0.46), same-weekday diff 1.5 (t 0.16), MDE 27.3

SPY W1 by year (mean bps, n): 2016 -7 (8); 2017 -0 (8); 2018 16 (8); 2019 -3 (8); 2020 87 (7); 2021 -28 (8); 2022 79 (8); 2023 3 (8); 2024 31 (8); 2025 6 (8); 2026 -22 (6)

SPY W2 by year (mean bps, n): 2016 5 (8); 2017 5 (8); 2018 23 (8); 2019 -8 (8); 2020 50 (7); 2021 -18 (8); 2022 88 (8); 2023 22 (8); 2024 31 (8); 2025 4 (8); 2026 -28 (6)

Press conference vs not, events < 2019 (mean bps, n):
- SPY W1: PC 17.3 (n 12), no-PC -11.9 (n 12)
- SPY W2: PC 18.1 (n 12), no-PC 3.7 (n 12)
- SPY W3: PC 9.6 (n 12), no-PC -14.3 (n 12)
- QQQ W1: PC 18.3 (n 12), no-PC -24.4 (n 11)
- QQQ W2: PC 22.4 (n 12), no-PC -9.8 (n 11)
- QQQ W3: PC 7.2 (n 12), no-PC -30.2 (n 11)
- IWM W1: PC 22.6 (n 12), no-PC -33.7 (n 12)
- IWM W2: PC 17.5 (n 12), no-PC -9.7 (n 12)
- IWM W3: PC 15.2 (n 12), no-PC -35.8 (n 12)

Cells passing: 2 of 10. SPY passes W1 or W2: False. W4 2012-26 mean>0 and t>=1.5: False.
VERDICT: NOT RECOMMENDED
Best SPY point estimate 16.3 bps, sd 62.2: years to resolve at 2.8 SE with 8 events/yr = 14

Pass-rule reading: "event-minus-placebo >= 10" was applied to BOTH placebos (all non-FOMC sessions and same-weekday), the stricter reading -> 2 of 10 pass (QQQ W1, W2). On the all-sessions placebo alone SPY W2 also passes (diff 12.4) -> 3 of 10. The verdict is the same either way: the sleeve rule needs W4 2012-26 t >= 1.5 and it is 0.88.

## Adversary caveats
- Replication of the published window: SPY W4 1994-2011 +32 bps net (t 3.1, diff vs placebo +31, t 3.0, same-weekday +31) matches Lucca-Moench in size. It is NOT present 2012-2026 (+8.6, t 0.9, diff +4.6 t 0.5, 47 % positive; MDE 27 bps, so half the published size is unresolvable there).
- Intraday 2016-2026 SPY W1/W2 +15/+16 bps look like the drift but sit against a Wednesday-conditioned background (same-weekday placebo +11 bps): FOMC days are mostly Wednesdays. Edge over same-weekday = +4/+6 bps (t 0.5-0.8).
- Tail dependence: SPY W1 ex-top-5 % falls 14.6 -> 3.8; W2 16.3 -> 6.7. 2020 (+87/+50) and 2022 (+79/+88) carry the book; 2021 and 2026 are negative. %positive 52-55 %.
- Windows overlap with a 1-day pre-announcement hold; the 2 bps cost is stated, not measured NBBO (the auction leg has no spread, so for W1 it is generous to the strategy).
- Press-conference split (pre-2019, n 12 vs 12, tiny) shows PC +17 vs no-PC -12 bps on SPY W1; do not read as a finding (2 cells of noise, uncounted in the 10).
- Prior close: official daily close used; differs from the last 15:59 minute close by mean +3.9 bps, mean |.| 13.6, max 82 bps (closing-auction prints), so W1 depends on that choice. Alpaca vs yfinance event-day returns agree (max |diff| 9.8 bps, mean 1.4).
- Adjustment=ALL (total return) on Alpaca; yfinance auto_adjust. Ex-dividend days inside windows are treated consistently in both event and placebo.
- Dates: parsed from the Fed's own pages only (no second source; the pages' statement-link dates agree with the last-day rule for 2021+). 2020 has 7 (Mar 17-18 cancelled, replaced by the unscheduled Mar 15). 2003-09-15 is an agenda-only panel of the Sep 16 meeting (excluded). 2027 and Oct/Dec 2026 rows are future-dated and unused. Release time 14:15 (< 2013) / 14:00 is an era assumption, irrelevant to 2016+ windows.
- Minute-bar gaps: SPY 1-3, QQQ 8-9, IWM 9-13 of 2,702 sessions missing at the used minutes (QQQ lost 1 event); no LOST chunks.
- The 3-instrument cells are one correlated bet (SPY/QQQ/IWM); QQQ's pass is not independent evidence.
- Resolution: SPY W2 at its +16.3 bps point estimate needs about 14 years at 8 events/yr to reach 2.8 SE. Positive, not resolvable in under ~14 years; no paper build. Independent rebuild not done (nothing recommended).
