# PREREG — cell 1,702: the pre-FOMC drift as a stacking sleeve (FROZEN 2026-10-02 18:10 UTC, before any number)

Owner 10/2: "no literature on spy direction?" The one documented directional effect not yet tested here.
Mechanism (Lucca & Moench 2015): the S&P earned ≈ +49 bps in the 24 hours BEFORE scheduled FOMC announcements
1994–2011 (a premium for holding risk into the decision), nothing comparable on other days. Post-publication evidence
is mixed (weaker after 2011, concentrated in meetings with a press conference). Already here: turn-of-month PASS
(1,651, on paper), overnight FAIL (1,649/1,650). 8 events a year: this is a stacking sleeve judged on its annual
contribution next to turn-of-month — the cadence bar (≥ 3 fills/week) does not apply and is not claimed.

## Data (free only)
* Scheduled FOMC announcement dates and times 1994–2026 from the Federal Reserve's calendar pages (unscheduled /
  emergency meetings EXCLUDED and listed; announcement time per era: 14:15 ET to 2013-01, 14:00 ET after; verify).
  Coverage line: meetings found per year (expect 8); a year with ≠ 8 is explained or the cell is VOID.
* SPY, QQQ, IWM 1-minute bars for each event day and the session before, 2016–2026, fetched from Alpaca (SIP, free)
  into `research/fomc_drift/minute_bars.parquet` (LOST count; never written to bars_sip.db or cache.db).
* SPY daily 1994–2026 via yfinance for the long-history daily read; overlap check vs Alpaca daily closes on event days.

## Cells (10)
For SPY, QQQ, IWM (2016–2026, n ≈ 86 events): W1 prior close → 13:55 ET on the day (buy market-on-close the day
before, sell at the 13:55 bar's open) · W2 14:00 ET the day before → 13:55 ET (the paper's window; entry at the
14:00 bar's open) · W3 the day's open → 13:55 ET. Plus W4: SPY prior close → close, daily, 1994–2026 (halves
1994–2011 published window / 2012–2026 after it).
Cost 1 bp per side (stated, conservative for these three ETFs; the auction leg has no spread).

## Reads per cell
n, mean net bps, t, share of positive events, both halves (2016–2020 / 2021–2026), ex-top-5 % and ex-worst-5 %,
worst event, by-year table, press-conference vs not (pre-2019), and the PLACEBO: the same window on every non-FOMC
session and on same-weekday non-FOMC sessions (mean, and the event-minus-placebo difference with t).
MDE beside every mean (2.8 × SE; at n 86 and ~75 bps window sd ≈ 23 bps — an effect half the published size is NOT
resolvable on 2016–2026 alone; that is what W4's 2012–2026 half, n ≈ 120, is for).

## Pass rule
A window passes on an instrument if mean net ≥ +15 bps AND t ≥ 2.0 AND both halves > 0 AND ex-top-5 % > 0 AND the
event-minus-placebo difference ≥ +10 bps AND ≥ 55 % of events positive. The sleeve is recommended only if SPY passes
on W1 or W2 AND W4's 2012–2026 half is positive with t ≥ 1.5 (the effect survived publication). If the point estimate
is positive but unresolved: "positive, not resolvable in under ~N years" with N computed — no paper build. Then an
independent rebuild before any paper flag. Nothing outside this list after numbers. Cells on this line: 10.

## Output
`1702_fomc.py`, `fomc_dates.csv` (date, time ET, scheduled flag, press conference flag, source URL),
`1702_events.csv` (one row per event × instrument × window), `RESULT_1702.md` (≤ 70 lines, adversary caveats).
ONE process through `bash scripts/research_run.sh -m 2000M`. The agent returns ≤ 150 words.
