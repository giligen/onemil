# PREREG — cells 1,633–1,636: POST-EARNINGS DRIFT ON THE REACTION (the 8-K 2.02 cache)

FROZEN 2026-09-28 18:40 UTC before any number. Programme count: 1,632 → 1,636. Idea 6 of `research/IDEAS_20260928.md`.

## Mechanism (documented: Bernard & Thomas 1989; Brandt, Kishore, Santa-Clara & Venkatachalam 2008 — the announcement
return as the surprise; decayed since the 2000s, persistent in small caps)
After an earnings announcement the price continues to drift in the direction of the initial reaction for weeks. The
reaction itself is the observable surprise (no estimates needed). The desk's cell 1,552 read the 2.02 class as a
whole with no conditioning and found nothing; this conditions on the reaction.

## Data (on disk)
`research/edgar_desk/events_raw.csv` (4.4 M filings; form 8-K with item 2.02 = 116 k earnings releases; acceptance
datetime UTC — the 1,552 refuter found it was read as ET: convert properly); daily bars `research/overnight_high/
alpaca_daily_2019_2024H1.parquet` and `panel_2024_2026.parquet` (zero-OHLCV dropped); universe price ≥ $3 and 20-day
dollar volume ≥ $1M on the prior session; splits: TRAIN 2019–2022, VAL 2023–2024H1, TEST 2024H2–2026-09 sealed.
Reaction R0 = the return from the close before the announcement to the close of the first full session after it
(after-close filing → next session; pre-market filing → that session; intraday filing → that session's close vs the
prior close).

## Cells
* 1,633 LONG top decile: R0 in the TRAIN top decile → buy at the NEXT open after the reaction session (MOO), sell at
  the close of session +10 (MOC); 1,634 the same with +20; 1,635 SHORT bottom decile (mirror; shortable/SSR
  excluded, borrow 3 %/yr); 1,636 report-only: the full decile table of the 10- and 20-session post-reaction return,
  the small-cap (≤ $1B) vs larger split, and the abnormal version (minus SPY over the same window).
Costs: 5 bps per auction leg. Report per cell and split: n, events/week, mean net bps, day-clustered t (entry session),
ex-top-5 % / ex-top-1 %, winner-capped +30 %, the decile monotonicity (the mechanism), per-year, the SPY-adjusted
version beside the raw.

## Pass bar (frozen; VAL, per cell)
Mean net ≥ +50 bps per event over the hold (≈ 5 bps/day), day-clustered t ≥ 2.5, ex-top-5 % > 0, ≥ 5 events/week in
season (report the seasonality), TRAIN same sign t ≥ 1, the decile table monotone on both halves, SPY-adjusted ≥ +30 bps.
TEST once for the single best passing cell.

## Independent check and consequences
Rebuild from the prose (event set Jaccard ≥ 0.98, bps within 2); refuters: the acceptance-time → reaction-session
mapping (UTC/ET; pre-market vs after-close), raw price scale (splits inside the hold), survivorship for 2019–2024H1,
duplicate 8-Ks (amendments, multiple 2.02 per quarter), delistings inside the hold (kept at −100 %), tails and month
concentration. PASS → a daily MOO/MOC leg on the paper account (the EOD-mode order types exist) for 6 weeks, then
live at $3K per event on the owner's word. FAIL → PEAD closes on this population with the decile table on record.

## Not allowed
Choosing the decile or the hold on VAL; more than one TEST read; conditioning on anything after the reaction session.
