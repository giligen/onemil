# 1701_calendar completeness (cell 1,701 step 1)
events_raw.csv rows read: 4,410,667 -> 40,564 matched 8-K(/A)+item2.02+filing_date>=2024-06-01
Drop reasons (sequential, on rows remaining after prior drops):
no_symbol 0 | out_of_window(session outside 2024-07-01..2026-09-30) 392 | dup(same symbol+session,
later acceptance) 196 | not_common(incl. ^Z[A-Z]ZZT$ test tickers, non-'C' security_type) 6,179 |
not_listed(no Databento bar that day) 83 | price(prior_close<$5 or unknown) 7,197 |
adv(adv20_dollar<$5M or unknown) 7,318 -> **final candidates: 19,199**
**DATA GAP (material): the Databento panel ends 2026-09-04**, 26 days short of the 2026-09-30
window end (confirmed max bar_date). Zero candidates have session > 2026-09-04. Raw item-2.02
8-K(/A) filings with filing_date >= 2026-08-15 = 419; most map to sessions already covered, but
2026-09-05..09-30 (~19 trading days) is entirely unrepresented for lack of panel bars, not lack
of events -- do not read late-Aug/Sep-2026 frequency as real until the panel is extended.
Candidates per half: A (2024-07..2025-06) = 7,887 | B (2025-07..2026-09) = 11,312
Per week (full calendar incl. zero weeks): mean 162.70, median 55.0, p90 561.8, max 886,
share of weeks >=10 fills = 89.83% (depressed by the data-gap tail above).
UTC->ET check (5 sample rows):
  DVN    session 2025-08-06  acceptance_et 2025-08-05T16:14:45-04:00
  THRM   session 2025-10-23  acceptance_et 2025-10-23T06:10:33-04:00
  MNKD   session 2025-11-05  acceptance_et 2025-11-05T08:03:50-05:00
  PZZA   session 2025-11-06  acceptance_et 2025-11-06T06:59:34-05:00
  ED     session 2026-02-20  acceptance_et 2026-02-19T16:40:52-05:00
Symbol-session count step 2 must backfill: 19,199
Caveats: early closes not flagged (session rule uses the 09:30 open only); EDGAR symbol matched
as-is against Databento raw_symbol, no class-share normalisation (mismatches drop under
not_common/not_listed).
