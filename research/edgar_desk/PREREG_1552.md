# PREREG — cells 1,552–1,561: the EDGAR EVENT DESK — filing classes with a fixed direction, auction-to-auction

FROZEN 2026-09-26 20:40 UTC before any number. Programme count: 1,551 → 1,561 (ten classes, each one cell). Leg B of
`research/passive_income/AI_NATIVE_PLAN_20260926.md`; owner 9/26: "so you will look for correlation between edgar
filings and stock movement?" — no: a short catalogue of filing events with a known mechanism and a direction fixed here.

## Mechanism and what was seen (disclosed)
Dilution and distress filings add supply or remove trust and drift down after the first print (documented in the
small-cap literature; the published post-8-K drift in large caps is small). Contract and activist filings are positive
news where the open question is whether anything remains after the first obtainable price. The HOD-restricted version
(cells 1,483–1,485) was void: Benzinga covers 17 % of those names; EDGAR covers 100 % of filers. No number of this
desk has been seen.

## Events (class → direction; the item codes come from the SEC submissions API `items` field, forms from the same file)
| cell | class | definition | direction |
|---|---|---|---|
| 1,552 | OFFERING | form 424B1–424B5 or 8-K item 3.02 (unregistered sale of equity) or 8-K 1.01 whose text is skipped (structured only) | SHORT |
| 1,553 | SHELF | S-3, S-3ASR, S-1 (initial or resale registration) | SHORT |
| 1,554 | REVERSE_SPLIT | 8-K item 5.03 (amendments to articles) | SHORT |
| 1,555 | AUDITOR | 8-K item 4.01 (change in certifying accountant) | SHORT |
| 1,556 | NON_RELIANCE | 8-K item 4.02 (restatement) | SHORT |
| 1,557 | LATE_FILING | NT 10-K, NT 10-Q | SHORT |
| 1,558 | OFFICER_EXIT | 8-K item 5.02 (departure of officers) | SHORT |
| 1,559 | CONTRACT | 8-K item 1.01 (entry into a material agreement) WITHOUT items 3.02 / 2.03 on the same filing | LONG |
| 1,560 | ACTIVIST | SC 13D (initial) | LONG |
| 1,561 | BUYBACK_OR_INSIDER | 8-K item 8.01 with a Form 4 purchase by an officer within 2 sessions (structured join) | LONG |
Report-only: 8-K 2.02 (earnings), 7.01 (Reg FD), 8.01 alone, 10-Q/10-K — no direction, shown for the record.

## Population, timing and prices
Every company with a CIK in `data/research/databento/alpaca_assets_all_20260905.csv` ∪ the PIT listing feed, common stock
only, price ≥ $1 and 20-day dollar volume ≥ $1M on the prior session; test tickers excluded. Filing time = the
acceptance datetime (ET). Entry = the first auction after acceptance: same-day MOO if accepted before 09:00 ET, next
session's MOO if accepted after 09:00 (intraday acceptances are entered at the NEXT open — conservative; the intraday
reaction is report-only from minute bars later, not here). Exits: E1 = the same session's MOC (day-1 return: open →
close); E2 = MOC on session +5 (drift). Prices: Alpaca daily bars 2019-01..2024-06 (`research/overnight_high/
alpaca_daily_2019_2024H1.parquet`, raw, survivorship caveat as PREREG_1550) and the Databento EQUS.SUMMARY panel
2024-07..2026-09 (delisted included; zero-OHLCV placeholder rows dropped — the 1,550 defect). Splits: TRAIN 2019–2022,
VAL 2023–2024H1, TEST 2024H2–2026-09 sealed (one read for the best passing cell). Costs: 5 bps per auction leg
(MOO + MOC = 10 bps round trip); SHORT cells add borrow at 3 %/yr pro rata and are EXCLUDED on SSR days (prior close
−10 %) and when the prior session's close < $5 (locate reality); a −100 % night (delisting) stays in the book.

## Report per cell, per holdout
n events, events/week, mean net return in bps for E1 and E5 with day-clustered t (the session is the cluster), ex-top-5 %
and ex-top-1 % (in the direction of the trade), winner-capped at ±20 %, median, share of events in the trade direction,
the universe's same-session return (placebo: all eligible names on the same session, same leg), a count-matched null
(the same number of random eligible names per session, 1,000 draws, seed 1552), per-year table, and the report-only
classes. Multiplicity: ten directional cells, each pre-directed; no selection among them except TEST for the best.

## Pass bar (frozen; VAL, per cell, in the fixed direction)
E1 or E5 mean net ≥ +15 bps with day-clustered t ≥ 2.5 (the leg named on TRAIN before VAL is read), ex-top-5 % > 0,
winner-capped positive, ≥ 3 events/week, placebo margin ≥ +10 bps with t ≥ 2, null ≥ 99, TRAIN same sign t ≥ 1,
≥ 3 of 4 TRAIN years positive. SHORT cells also: positive after borrow and with SSR/price exclusions applied.

## Independent check and consequences
Rebuild from this prose on the same filings and bars (event-set Jaccard ≥ 0.99, net bps within 1). Refuters: timing
(acceptance datetime vs the entry auction — no same-day open for a 09:31 acceptance), price scale (raw close/open of
the same share class; ±50 % events inspected, never filtered), survivorship (2019–2024H1), the item-code mapping
(sample 30 filings per class by hand: does the item mean what the class says), SSR/borrow realism, tails. PASS → a dry
MOO/MOC ledger for 10 sessions on Alpaca `opg`/`cls` orders (long cells first; short cells only after the owner's
borrow decision), then $3K per event. FAIL → leg B closes with the class table on record; leg A (sentiment) stays
low priority per the owner.

## Not allowed
Adding classes after seeing TRAIN; text classification in this pass (structured codes only; the LLM pass is a later
PREREG on the CONTRACT / 8.01 texts if this one passes); selecting E1 vs E5 on VAL; more than one TEST read.
