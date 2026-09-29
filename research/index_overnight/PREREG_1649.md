# PREREG — cells 1,649–1,651: INDEX OVERNIGHT PREMIUM, conditioned on the close's volume shape, and TURN-OF-MONTH

FROZEN 2026-09-29 06:20 UTC before any number. Programme count: 1,648 → 1,651. Free queue #2 and #3 (`research/
ideas_web/RANKED_20260928.md`; ideas 4–5 of `research/IDEAS_20260928.md`). No paid data: the minute bars cached under
`research/lev_flow/data/` (Alpaca, SPY/QQQ/IWM and six sector ETFs, 2016-01-04..2026-09-04) and daily bars.
Honest scale up front: these are index premia of a few bps a night; at this account they are hundreds of dollars a
month at full capital use, not thousands. They are tested because they are free and because a conditional version
that doubles the unconditional premium is a real allocation improvement for the idle cash.

## Mechanisms (documented)
* Overnight premium (Lou, Polk & Skouras 2019; Kelly & Clark 2011): the equity premium accrues close-to-open; intraday
  ≈ 0. Conditional version (the practitioner scan's claim): nights after a heavy-close session — institutional and
  index rebalancing flow concentrated in the last half hour — carry a larger overnight return.
* Turn-of-month (Lakonishok & Smidt 1988; McConnell & Xu 2008): the last session of the month through the third of
  the next carries most of the month's return, attributed to pension/payroll flows and month-end rebalancing.

## Signals and trades (SPY, QQQ, IWM)
* 1,649 report-only benchmark: unconditional overnight (buy MOC, sell MOO every night) and intraday (MOO → MOC).
* 1,650 CONDITIONAL OVERNIGHT: s_t = the share of the session's 09:30–16:00 volume printed 15:30–16:00; heavy close =
  s_t ≥ 1.25 × the ETF's trailing 60-session median of s_t (known at 16:00). Buy MOC on heavy-close days, sell MOO next
  session. The mechanism table: next-overnight return by s_t tercile (must rise with s_t on both halves).
* 1,651 TURN-OF-MONTH: buy MOC on the last session of the month, sell MOC on the third session of the next month
  (four nights). Per-year sign table.
Costs: 0.5 bp per auction leg plus 0.5 bp half-spread (1 bp per round trip is generous for these three).
Splits: 1,650 TRAIN 2016–2019, VAL 2020–2023; 1,651 judged on 2016–2023 pooled (a calendar effect has 12 events a year
per ETF; a 4-year VAL cannot resolve it — stated, with the per-year table as the consistency check); TEST 2024-01..
2026-09 sealed for both (one read for the single best passing cell).
MDE printed beside each verdict: 1,650 ≈ 190 heavy-close nights in VAL × 3 ETFs clustered by night, overnight SD ≈ 60 bps
→ SE ≈ 4.4 bps → MDE ≈ 11 bps against an 8-bps bar (the t item is informational if the realised MDE exceeds the bar;
the mechanism table and the excess over the unconditional premium decide); 1,651 ≈ 96 events per ETF, SD ≈ 1.5 % →
SE ≈ 22 bps clustered by month → MDE ≈ 55 bps against a 30-bps bar (same rule).

## Reporting
Per cell and split: n, mean net bps per event, night/month-clustered t, ex-top-5 % / ex-worst-5 %, per-ETF, per-year,
the excess of 1,650 over the unconditional overnight premium on the same nights' complement, the tercile table, the
worst night/event, the MDE line, and the annualised net return on fully deployed capital beside the drawdown.

## Pass bar (frozen)
1,650 (VAL): mean net ≥ +8 bps per night, clustered t ≥ 2.5 (or, if MDE > 8 bps, the excess over the unconditional
premium ≥ +4 bps with t ≥ 2 AND the tercile table monotone on both halves), ex-top-5 % > 0, TRAIN same sign t ≥ 1,
positive in all three ETFs. 1,651 (2016–2023): mean net ≥ +30 bps per event, clustered t ≥ 2.5 (same MDE rule with
≥ 6 of 8 years positive and all three ETFs positive as the alternative), ex-top-5 % > 0.

## Independent check and consequences
Rebuild from the prose (night/event set Jaccard ≥ 0.98, bps within 1); refuters: the volume share uses only the
session's own bars to 16:00 (nothing after), the MOC fill at the official close vs the 16:00 bar, holidays and early
closes (excluded), the month boundary on holidays, dividends (adjusted bars), the overlap of 1,651 nights with 1,650
nights (report the overlap share), tails. PASS → paper MOC/MOO legs for 4 weeks, then the idle cash sleeve at $20K per
night on the owner's word. FAIL → closed; no re-run without a different conditioning variable.

## Not allowed
Tuning the 1.25 multiplier, the 60-session window, the 15:30 boundary or the four-night TOM window after a number.
