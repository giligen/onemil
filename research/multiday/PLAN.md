# Multi-day program — pre-registered 2026-09-18 21:40 UTC (owner: "Go deep on multiday. Get what data you need. Make it work.")

## Why the previous multi-day tests do not settle the question
Stage K / N2 / R_daily tested FIVE home-made long-only rules at 1–10 day holds on Nasdaq-listed names, charged a quoted
half-spread + 5 bps per side on auction fills, and had no earnings calendar. The published multi-day effects are none of
those things: they are (a) event-anchored (earnings), (b) long-SHORT cross-sectional portfolios, (c) held 20–250 sessions,
(d) executed in the opening/closing auctions where a resting order pays no spread (PLAN §1 item 4 of the audit program:
"auction-executed trades must NOT be charged a quoted spread" — the K family was over-charged on every trade). This
program tests the published constructions faithfully FIRST, then the executable long-only / small-account variants as
separate cells, and reports the gap between them. "Does the anomaly exist" and "can this account harvest it" are two
different questions and get two different answers.

## Data (all free; purchases only if the plan below cannot be met, priced first)
- **Prices**: Alpaca SIP daily bars 2016-01-04 → now, `adjustment='all'` (split+dividend adjusted; verified 9/18: AAPL
  2016-01-04 raw 105.35 / adjusted 23.69), for every currently listed US common stock and ETF (`get_all_tradeable_assets`).
  Store raw AND adjusted; features on adjusted, fills and share counts on raw. Opens/closes are the official consolidated
  session prints (the auction fill the engine would get with `opg`/`cls`).
- **Survivorship** (the known hole: Alpaca lists only current names): quantify it on two overlaps — Databento
  `pit_definition` 2024-07→now (all exchanges, `research/scripts/pit_listings.py`) and XNAS.ITCH daily 2018→2024
  (Nasdaq, delisted included, on disk under N3/R_daily). Report the delisting rate per year and re-run every survivor cell
  on the point-in-time Nasdaq panel as the survivorship check. A long-only result that does not survive the PIT re-run is
  survivorship.
- **Earnings calendar**: SEC EDGAR submissions API — 8-K filings with Item 2.02 (Results of Operations) = the earnings
  release, timestamped by `acceptanceDateTime` (point-in-time; verified 9/18: AAPL 45 of 103 8-Ks). Event day = the first
  session whose close is after the acceptance time.
- **Earnings surprise without a consensus vendor**: XBRL `companyconcept` `EarningsPerShareDiluted` quarterly facts with
  their `filed` dates (verified: AAPL 174 facts 2008→2026). SUE per Bernard–Thomas 1989 / Foster–Olsen–Shevlin 1984:
  seasonal random walk with drift, (EPS_q − EPS_{q−4} − drift) / σ of the last 8 seasonal differences, computable ONLY from
  facts filed before the event (availability audit on `filed`). Secondary surprise measure that needs no accounting data:
  the 2-day announcement return (Chan–Jegadeesh–Lakonishok 1996) — the "earnings announcement return" drift.
- **Industry** for the industry-adjusted reversal: SIC code from the EDGAR submissions record (2-digit).
- Test tickers excluded; names absent from `daily_bars`/Alpaca assets excluded; the raw close ≥ $5 gate on RAW prices.

## Families (the published constructions; each with its citation, exactly as published, then the executable variant)
- **F1 PEAD** (Bernard–Thomas 1989; Livnat–Mendenhall 2006 for the modern decay): sort into SUE deciles at each event, long
  D10 short D1, enter at the NEXT open after the 8-K acceptance, hold 60 sessions (also 20, 40). Executable variant: long
  D10 only, 20 names max, MOO entry, MOC exit.
- **F2 Earnings-announcement-return drift** (CJL 1996): sort on the 2-day announcement return, same holds, L-S and long-only.
- **F3 Cross-sectional momentum 12-1** (Jegadeesh–Titman 1993; Asness–Moskowitz–Pedersen 2013 for the modern universe):
  monthly rebalance, skip the last month, deciles, L-S; executable: long D10 only, 20 names, and the 6-1 variant. Report
  the 2016–2026 momentum crashes explicitly (Daniel–Moskowitz 2016) — the drawdown IS the finding for a compounding account.
- **F4 Short-term reversal, industry-adjusted** (Da–Liu–Schaumburg 2014 — the part that survives costs): weekly, L-S on
  the residual of the 1-week return vs the 2-digit-SIC industry mean; executable long-only leg reported separately.
- **F5 52-week-high** (George–Hwang 2004): monthly, L-S on nearness to the 52-week high; long-only leg.
- **F6 Overnight-vs-intraday cross-section** (Lou–Polk–Skouras 2019): long past overnight winners / short past intraday
  winners, monthly; executable: hold overnight only (MOC buy, MOO sell) on the long leg.
Cells: 6 families × {published L-S, executable long-only} × the declared holds (F1/F2: 20/40/60; F3: 12-1 and 6-1; F4: 1 wk;
F5: 1 mo; F6: 1 mo) = **26 cells**, declared here; nothing added later.

## Splits, costs, gates
- **Splits**: TRAIN 2016-01→2021-12 (6 yr), VAL 2022-01→2023-12 (2 yr), TEST 2024-01→2026-09 (sealed; opened once behind
  `FREEZE.md` for G2 survivors). Power statement per split BEFORE returns are looked at (MDE in bps/month of L-S spread).
- **Costs, auction-faithful**: entries/exits at the official open/close via `opg`/`cls` → no quoted spread; charge SEC fee
  + FINRA TAF on sells (~0.4 bps), a market-impact allowance = 10 bps × (order $ / 1% of ADV$) capped at 1% ADV, and
  borrow cost on the short leg = the general-collateral 0.3%/yr plus a flag: a short is allowed only if today's Alpaca
  asset record is `easy_to_borrow` (survivorship caveat stated). Secondary cost arm: 5 bps/side flat.
- **Gates** (PLAN §1 of the audit program): G1 t ≥ 2 on TRAIN; G2 VAL same sign and ≥ 55% of months positive; TEST once.
  Tail tests (ex-top-5%, winners capped at +3× the cell's median win); permutation p across the 26 cells; availability
  audit on every field (`filed`/`acceptanceDateTime` ≤ decision time); the PIT Nasdaq re-run for every survivor; price-scale
  check raw vs adjusted on 200 keys; cell count reported (26 here; 70 cumulative on the multi-day line incl. K/N2/R_daily).
- **Executability** is scored, not assumed: Alpaca `opg`/`cls` TIF exist in the SDK and are UNUSED in this repo (zero
  `TimeInForce.OPG/CLS`); shorting needs margin + ETB. A survivor needs a daily-order engine — a build of ~2 weeks, listed
  in the report with its parts.

## Deliverables
`research/multiday/DATA.md` (panel sizes, survivorship rates, EDGAR coverage share of the universe, EPS-fact coverage,
the availability audit), `research/multiday/REPORT.md` (one page: the 26-cell table per split with L-S and long-only side
by side, MDE, drawdowns and worst month for every long-only survivor at a $50K book, the PIT re-run, the verdict in the
phrasing rule, and for any survivor the exact rule in prose for an independent rebuild + the engine build list), 3 lines in
`research/fuckup_audit/LOG.md`. Order: data → F1/F2 (event-anchored, highest prior) → F3 → F4/F5/F6.
