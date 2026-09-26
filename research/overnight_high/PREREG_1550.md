# PREREG — cells 1,550–1,551: the OVERNIGHT NEW-HIGH leg, extended out of regime

FROZEN 2026-09-26 18:50 UTC before any new number. Programme count: 1,549 → 1,551. Owner 9/26: "both sound interesting,
maybe we should dive more into both" (the overnight 52-week-high leg).

## What was seen (disclosed)
`research/lit_review_2026/overnight_auction.md` (test `test_overnight_auction.py`, 9/26): rule H = the close is a new
252-day high on volume ≥ 1.5 × ADV20, universe close ≥ $5 and 20-day dollar volume ≥ $10M, top-N by volume ratio, buy
MOC, sell MOO next morning, costs 2 / 5 / 10 bps round trip (auction fills, no quoted spread). Gross per night:
TRAIN (< 2026-01-01, from the panel's start) top-25 +26.6 bps (t 4.6), top-50 +27.0 (t 5.6); VAL (2026-01..05) +17.3 (t 1.9)
/ +15.5 (t 2.3); TEST (2026-06 on, 14 weeks) −11.6 (t −1.0) / −13.2 (t −1.3). Availability 20–36 names/day. That TEST
read is SPENT: the frozen out-of-sample of this PREREG is the pre-panel history. Mechanism on record: overnight returns
carry the momentum premium (Lou, Polk & Skouras 2019) and the 52-week-high anchor (George & Hwang 2004); the retail
edge, if any, is that the trade is an auction-to-auction hold with near-zero spread. The daily panel behind the note
was deleted in the 9/26 disk clean-up and must be rebuilt (`build_daily_panel.py`, same definitions).

## Rule (unchanged from the note; nothing re-tuned)
Signal at the 15:59 close of day t: close_t ≥ the highest close of the prior 252 sessions, volume_t ≥ 1.5 × ADV20,
close ≥ $5, 20-day dollar volume ≥ $10M, no test tickers (`^Z[A-Z]ZZT$`, `^ZZ`); rank by volume ratio, take the top N.
Buy at the day-t closing auction (MOC), sell at the day-t+1 opening auction (MOO). Return = open_{t+1} / close_t − 1.
Cells: 1,550 = N 10; 1,551 = N 25. Costs: 5 bps round trip primary (2 and 10 reported); an impact line for a $3K
order in a ≥ $10M-ADV name is reported as 0 with the reason.

## Samples
* EXTENSION (the decisive read, never seen): 2019-01-02 … 2024-06-28 from Alpaca daily bars for every symbol in the
  current listing plus every symbol in the Databento point-in-time listing feed (`data/research/databento/pit_definition`,
  `research/scripts/pit_listings.py`) so delisted names since 2024-07 are in; SURVIVORSHIP CAVEAT on record for
  2019–2024H1 (names delisted before 2024-07 are missing; for a long-only overnight rule on ≥ $10M-ADV names the bias
  is upward, bounded by the delisting rate of that universe — report the count of names present per year).
* PANEL (rebuilt): 2024-07 … 2026-09 from the Databento EQUS.SUMMARY daily parquet (delisted included), the same splits
  as the note (TRAIN < 2026-01-01, VAL 2026-01..05, TEST 2026-06 on — TEST reported as already spent).
Report per sample and per calendar year: n nights, gross and net bps/night, day-clustered t, ex-top-5 % and ex-top-1 %
(nights are the cluster), winner-capped at +5 %, share of green weeks, the weekly P10 and the strong-week gap (cadence
bar `docs/cadence_bar.md` at $3K per name), the universe's own overnight return on the same nights (the placebo: the
whole eligible universe MOC→MOO), and a count-matched null (N random eligible names per night, 1,000 draws, seed 1550).

## Pass bar (frozen)
EXTENSION 2019–2024H1: net ≥ +8 bps/night at 5 bps, day-clustered t ≥ 2.5, ex-top-5 % > 0, ≥ 4 of the 5.5 calendar years
positive net, placebo margin (rule minus the universe on the same nights) ≥ +5 bps with t ≥ 2, null percentile ≥ 99;
AND PANEL TRAIN + VAL pooled net ≥ +8 bps with t ≥ 2. Both cells are read; the better one on the extension is named
before the panel is re-read. The already-spent TEST stays disclosed beside the result, whatever its sign.

## Independent check and consequences
Rebuild from this prose on the same bars (nightly book membership Jaccard ≥ 0.99, net bps within 0.5); refuters: the
price-scale check (splits and dividends: the return must use the raw close and the raw next open of the same share
class; flag any night beyond ±30 %), survivorship (the extension's missing delistings), look-ahead (the 252-day high
uses closes through t only; the volume ratio uses ADV through t−1), the MOC/MOO fill assumption (auction participation,
imbalance publication at 15:50 — an MOC order placed after 15:50 may be rejected; the rule must be executable by a
15:49 order on a signal computed from 15:45 data, and that variant is reported beside the 15:59 one). PASS → a dry
MOC/MOO ledger (Alpaca `cls`/`opg` orders) for 10 sessions, then $3K per name at N 10. FAIL → the overnight family is
closed with the extension numbers on record.

## Not allowed
Re-tuning the 1.5 × volume filter, the $5 / $10M universe or N; selecting N on the panel; treating TEST as unread.
