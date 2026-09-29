# RESULT — cell 1,672: pre-holiday index sleeve

FROZEN spec: `PREREG_1672.md`. Code: `1672_preholiday.py`. Events: `1672_events.csv` (134 rows,
date/etf/variant/leg/entry_date/exit_date/entry_close/exit_price/gross_bps/net_bps/year/half).
Full stats incl. ex-top-5%: `1672_stats.csv`.

## Data-range mismatch (report, not patch)
PREREG names 2016–2026. On this node `data/cache.db daily_bars` and the on-disk databento
EQUS.SUMMARY files cover only **SPY 2024-06-03..2026-09-28 (n=582), QQQ 2024-12-30..2026-09-28
(n=437)** — an 8–9x shorter window. "Odd vs even years" is therefore lopsided: odd = 2025 only
(full year), even = 2024-H2 (from Jun 3) + 2026-YTD (to Sep 28), not a real independent book.
Reported as required; not substituted with a different split.

## Holiday count verification
SPY: 24 weekday-gaps detected vs 23 hand-verified NYSE holidays in range; QQQ: 19 vs 18. Zero
missing either symbol (every known holiday shows up as a gap). The one "extra" both symbols
share: **2025-01-09**, the National Day of Mourning for President Carter — a genuine one-off NYSE
closure, correctly included as an event, not a data defect. Per-year (SPY): 2024(from 6/3)=5,
2025=11 (10 recurring + the mourning day), 2026(to 9/28)=8 — matches the ~9–10/yr expectation.

## Reads (net bps per event; t = iid = day-clustered, ≤1 event/day; null = 1,000-draw count-matched)
| Var | ETF | Leg | Half | n | mean bps | t | hit | worst bps | MDE bps | null %ile |
|---|---|---|---|---|---|---|---|---|---|---|
| A | SPY | pre | odd | 11 | −23.3 | −0.64 | 36% | −240.0 | 101.6 | 17.3 |
| A | SPY | pre | even | 13 | −14.9 | −0.56 | 54% | −207.8 | 73.8 | 26.9 |
| A | SPY | post | odd | 11 | +42.4 | 1.53 | 64% | −59.9 | 77.8 | 91.7 |
| A | SPY | post | even | 13 | −13.2 | −0.68 | 46% | −147.2 | 54.1 | 28.2 |
| B | SPY | pre | odd | 11 | −5.3 | −0.25 | 64% | −120.0 | 59.4 | 37.5 |
| B | SPY | pre | even | 13 | −7.2 | −0.47 | 54% | −150.0 | 42.7 | 31.9 |
| B | SPY | post | odd | 11 | +3.8 | 0.22 | 64% | −84.3 | 49.2 | 60.0 |
| B | SPY | post | even | 13 | −21.3 | −1.56 | 46% | −145.1 | 38.3 | 11.5 |
| C | QQQ | pre | odd | 11 | −22.6 | −0.59 | 36% | −249.5 | 107.4 | 23.7 |
| C | QQQ | pre | even | 8 | +9.8 | 0.23 | 38% | −214.5 | 119.3 | 54.7 |
| C | QQQ | post | odd | 11 | +50.2 | 1.64 | 64% | −50.4 | 85.7 | 88.1 |
| C | QQQ | post | even | 8 | −35.0 | −0.64 | 38% | −331.3 | 152.1 | 20.0 |

Ex-top-5% (drops the single best event, n<20): moves every negative-mean row further negative
(e.g. A/pre/odd −23.3→−46.2 bps) — no hidden positive tail being masked by a bad average.

## $/month (primary = variant A, pre-holiday, pooled both halves)
n=24, mean net **−18.7 bps/event**. Capital window: one overnight per holiday (close-to-close,
spans the weekend when adjacent), measured frequency 0.863/mo (23 SPY holidays / 27.9 mo)
vs the PREREG's ~9.5/yr = 0.792/mo baseline — freq matches. At $60K notional: **≈ −$97/mo**
measured freq, ≈ −$89/mo at the baseline freq. Negative either way.

## Pass bar (mean ≥+8bps & t≥2.0 & null %ile ≥95 & worst >−3%, BOTH halves, primary=A/pre)
**FAILS**, every clause, both halves: mean net is negative in both halves (−23.3 / −14.9 bps),
t negative in both, null percentile 17–27 (need ≥95). Only the worst-event clause clears
(−240/−208 bps > −300). No variant (B, C) or leg (post-holiday mirror) clears the full bar either;
the closest is A's odd-half mirror (+42.4 bps, t 1.53, pctile 91.7) but it's inconsistent across
halves (even-half mirror is −13.2 bps) and mirror was never the gating read.

## Verdict
No independent rebuild, no paper, no TOM-cron join. This is a null on the population actually on
disk (SPY/QQQ, 2024-06..2026-09, n=11–13/half): point estimates are small and negative, MDE
≈74–152 bps swamps the ±8bps bar at this n, so the test is underpowered for a small true effect —
but nothing here points positive, so there's no reason to spend more data on this construction.
Sleeve stacking day (holidays, distinct from TOM/ORB/HOD trigger days) is moot since it fails
standalone.
