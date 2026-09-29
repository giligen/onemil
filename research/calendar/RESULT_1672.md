# RESULT — cell 1,672: pre-holiday index sleeve (full-span rebuild)

FROZEN spec: `PREREG_1672.md`. Code: `1672_preholiday.py`. Fetch: `fetch_spy_qqq_2016_2026.py`.
Full-span source CSV: `spy_qqq_daily_2016_2026.csv` (Alpaca SIP, `get_daily_bars_range`, read-only —
never wrote `data/cache.db`). Events: `1672_events.csv` (612 rows). Full stats:
`1672_stats_fullspan.csv`, `1672_summary_fullspan.txt`.

Prior run judged 2024-06..2026-09 only (n=11–13/half) — an underpowered null (MDE ≈74–152bps at
that n), not a verdict, per the coordinator. This run fetches SPY/QQQ 2016-01-04..2026-09-29 from
Alpaca as the PREREG's Data section actually asks for.

## Fetch + cross-check
SPY n=2,700, QQQ n=2,700, both 2016-01-04..2026-09-29 (~252/yr over 10.74yr — the "~2,450"
estimate undershot; 2,700 is arithmetically right for the full span, not a truncation artifact).
Cross-checked against `data/cache.db` (read-only) on the overlap window: **SPY 582/582 dates,
QQQ 437/437 dates, 0 mismatches >1¢, max diff $0.0000** — the CSV and the live production cache
agree exactly where both exist.

## Holiday count verification
Rule-based NYSE calendar (explicit code: nth-weekday/last-weekday rules + `dateutil.easter` for
Good Friday + observed-date shift for fixed dates), not a hand table. **First pass caught a real
rule bug**: it flagged 2021-12-31 as a missing New Year's-2022 observance (Jan 1, 2022 is a
Saturday) — but SPY/QQQ both traded that day. Fix: NYSE does not roll New Year's Day back to the
preceding Friday when it falls on a Saturday (unlike Independence Day/Juneteenth/Christmas, which
do — confirmed by 2026-07-03 correctly showing as a gap). After the fix: **102 detected = 102
known, 0 missing, 0 extra**, both symbols. Per-year: 2016=8 (Jan-1-2016 predates the fetch
window's first session, 2016-01-04 — boundary artifact, not a defect), 2017–2022=9, 2018=10 &
2025=11 (each +1 for a one-off National Day of Mourning, Bush/Carter), 2023–2024=10, 2026=8
(partial, cutoff 9/29).

## Reads, full span (net bps/event; t = iid = day-clustered; MDE printed beside t; null =
## 1,000-draw count-matched, pool n≈2,041 ordinary pairs/symbol)
| Var | ETF | Leg | Half | n | mean bps | t | MDE bps | hit | worst bps | null %ile |
|---|---|---|---|---|---|---|---|---|---|---|
| A | SPY | pre | odd | 48 | −17.2 | −1.35 | 35.5 | 42% | −240.0 | 11.5 |
| A | SPY | pre | even | 54 | −3.6 | −0.23 | 44.8 | 46% | −267.2 | 34.9 |
| A | SPY | post | odd | 48 | +3.6 | 0.31 | 32.4 | 48% | −245.0 | 51.3 |
| A | SPY | post | even | 54 | +3.1 | 0.22 | 40.4 | 46% | −228.5 | 53.7 |
| B | SPY | pre | odd | 48 | −11.1 | −1.33 | 23.3 | 48% | −166.9 | 13.4 |
| B | SPY | pre | even | 54 | −3.6 | −0.35 | 28.4 | 50% | −168.3 | 33.7 |
| B | SPY | post | odd | 48 | −0.6 | −0.10 | 18.1 | 54% | −86.7 | 46.5 |
| B | SPY | post | even | 54 | −9.1 | −1.01 | 25.3 | 46% | −164.1 | 18.1 |
| C | QQQ | pre | odd | 48 | −8.2 | −0.56 | 41.0 | 48% | −249.5 | 25.3 |
| C | QQQ | pre | even | 54 | +8.1 | 0.37 | 61.6 | 50% | −482.6 | 53.3 |
| C | QQQ | post | odd | 48 | +2.3 | 0.15 | 41.0 | 46% | −328.7 | 44.6 |
| C | QQQ | post | even | 54 | +7.8 | 0.40 | 54.5 | 48% | −332.7 | 60.4 |

Odd = {2017,2019,2021,2023,2025}, even = {2016,2018,2020,2022,2024,2026(partial)} — the real
odd/even-years split now (the short-span run's "odd" was 2025 alone).

## $/month (primary = variant A, pre-holiday, pooled both halves)
n=102, mean net **−10.0 bps/event**. Measured frequency 0.792/mo (102 events / 128.9 mo) —
matches the PREREG's ~9.5/yr baseline almost exactly. At $60K notional: **≈ −$48/mo** either way
(measured or baseline freq).

## Pass bar (mean≥+8bps & t≥2.0 & null%ile≥95 & worst>−3%, BOTH halves, primary=A/pre)
**FAILS**, every clause, both halves, now on a well-powered sample: mean net negative both halves
(−17.2 / −3.6 bps), t negative both (−1.35 / −0.23), null percentile 11.5 / 34.9 (need ≥95). MDE
at this n (35.5 / 44.8 bps) is still above the 8bps bar, so this can't rule out a small true edge,
but the point estimate, sign, and null-percentile all point the same (wrong) way — this is a real
null, not just an underpowered one. The short-span run's positive post-holiday blip (odd +42.4bps,
t 1.53) regresses to +3.6bps/t 0.31 with the full sample — it was noise. Variants B and C fail
identically; no half of any variant/leg clears the bar.

## Verdict
No independent rebuild, no paper, no TOM-cron join. Confirmed on the full 2016–2026 SPY/QQQ span
(n=48–54/half, ~5x the prior sample), cross-checked against production data, holiday calendar
verified by explicit rule (one rule bug caught and fixed pre-scoring). Point estimates are small
and negative or flat across every variant, leg and half — the pre-holiday effect this population
would need does not show up here.

## Appendix — superseded short-span run (data/cache.db only, 2024-06-03..2026-09-28, n=11–13/half)
Kept for the record; do not cite as the verdict (underpowered, MDE 74–152bps, lopsided halves:
odd=2025 only, even=2024H2+2026YTD). Variant A pre-holiday: odd n=11 mean −23.3bps t=−0.64
pctile 17.3; even n=13 mean −14.9bps t=−0.56 pctile 26.9. Post mirror: odd +42.4bps t=1.53
pctile 91.7; even −13.2bps t=−0.68 pctile 28.2. $/mo ≈ −$89 to −$97. Same direction as the
full-span result throughout — the rebuild did not reverse anything, it added power.
