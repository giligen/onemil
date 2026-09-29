# PREREG — cell 1,672: pre-holiday index sleeve (FROZEN 2026-09-29 20:32 UTC)

Act-as-owner pick from `research/ideas_web/UNTESTED_20260929.md` (idea rank 10, data fully on disk). Sister of the
turn-of-month sleeve (cell 1,651, PASS → paper). Stacking rule: fires on different days from TOM, ORB and HOD; judged on
its own bar, then summed.

## Mechanism
Index returns on the last session before a US market holiday have historically been positive (the pre-holiday effect;
publication decay expected). Sleeve: buy SPY at the close of the session BEFORE a market holiday (MOC, 1 bp), sell at
the close of the next session (MOC, 1 bp). Variant B: sell at the next open (+ open-auction 2 bps). Variant C: QQQ.

## Data
`data/cache.db daily_bars` for SPY/QQQ (and `data/research/databento` EQUS.SUMMARY parquet as the cross-check for
2016–2026 closes). Holiday calendar: sessions absent from the daily series on a weekday = holiday (verify against the
NYSE list for 2016–2026 by hand for the count; report any mismatch).

## Reads (halves = odd vs even years, as 1,651; all with iid t, day-clustered = trade-clustered, ex-top-5 %, MDE)
Mean net return per event (bps), hit rate, n per half, worst event, the count-matched random-day null (1,000 draws of
the same number of ordinary sessions per half → percentile of the sleeve mean), the same for the session AFTER the
holiday (mirror read: decay check), and the $/month at $60K notional (the TOM size) with the capital window (one
overnight per holiday, ~9/year).

## Pass bar
Mean net ≥ +8 bps per event on BOTH halves, t ≥ 2.0 both, null percentile ≥ 95 both, worst event > −3 %. A pass →
independent rebuild from prose (a second agent) before paper; then it joins the TOM cron as a second calendar leg.

## Multiplicity
3 variants × 2 halves × 2 (pre/post) = 12 reads. Not allowed: choosing the holiday subset, the ETF or the exit after
seeing numbers.

## Output
`research/calendar/RESULT_1672.md` (≤ 80 lines), `1672_events.csv`, `1672_preholiday.py`. The agent returns ≤ 120 words.
