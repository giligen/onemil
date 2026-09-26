# PREREG — information events: Form 4 insider open-market purchases and Schedule 13D filings (cells I1–I4), FROZEN 2026-09-26 before any number

Owner 9/26: "Find a different population or something else. Make it work." Desk ledger cells 1,474–1,477.

## Why this population (and why it is not a re-run of `research/multiday/`)
The 26-cell multi-day programme refuted every BEHAVIOURAL anomaly at this account (effects of a few bps/month, 7–121× below
its MDE). Insider open-market purchases are INFORMATION: Lakonishok & Lee (2001), Jeng, Metrick & Zeckhauser (2003) and
Cohen, Malloy & Pomorski (2012, "opportunistic" trades ≈ 0.8 %/month alpha; routine trades ≈ 0) report effects an order of
magnitude larger than the price anomalies, concentrated in smaller firms and in clustered/officer buying, persistent
post-publication because the signal is private information disclosed with a lag, not a mispricing that arbitrage removes.
Activist 13D filings carry a post-filing drift (Brav, Jiang, Partnoy & Thomas 2008). Holding periods are weeks, fills are
closing-auction prints, R is the multi-week move (≈ 8–10 % of price), so the cost structure that killed every intraday
book (spread + stop slip ≈ 0.25 R) does not apply. Nothing in this repo has ever scored an insider or 13D signal.

## Data (all point-in-time)
* Insider transactions: SEC structured "Insider Transactions Data Sets" quarterly zips 2016Q1–2026Q2
  (`https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/<yyyy>q<n>_form345.zip`, verified reachable:
  2024q1 = 13.9 MB, NONDERIV_TRANS.tsv with TRANS_CODE, TRANS_SHARES, TRANS_PRICEPERSHARE, TRANS_DATE, TRANS_ACQUIRED_DISP_CD;
  SUBMISSION.tsv with ACCESSION_NUMBER, FILING_DATE, DOCUMENT_TYPE, ISSUERCIK, ISSUERTRADINGSYMBOL, AFF10B5ONE;
  REPORTINGOWNER.tsv with the owner's relationship flags and title). Downloaded once to `research/insider_events/data/`
  (gitignored), User-Agent "onemil research giligen@gmail.com", ≤ 1 request/second.
* Timestamp rule (conservative, no JSON fetch needed): a filing is knowable at the CLOSE of the first session strictly
  AFTER its FILING_DATE (Form 4s accepted after 17:30 ET are disseminated the next morning; using the next session's close
  everywhere is always point-in-time). Report-only refinement I1-t: acceptanceDateTime from the issuer's EDGAR submissions
  JSON (`research/multiday/data/edgar_common.py`, the F2 fetcher) → same-day close when accepted before 15:30 ET.
* 13D: initial Schedule 13D filings (form "SC 13D", not 13D/A, not 13G) with their acceptanceDateTime from the EDGAR
  full-text search API (`efts.sec.gov/LATEST/search-index?forms=SC%2013D&dateRange=custom&startdt=…&enddt=…`, paginated),
  mapped to a ticker through the filer's subject-company CIK → the panel's symbol map; unmapped filings counted.
* Prices, universe, costs, book: `research/multiday/data/panel_final.npz` (close_adj / close_raw / adv_raw / elig / taint,
  2016-01-04 → 2026-09-18) through the `Panel` class of `research/multiday/run_final.py`; PIT common stocks (raw close ≥ $5,
  ADV20 ≥ $1M at the signal session, `elig`), test tickers excluded; the multiday cost model (impact 10 bps × order$/1 %
  ADV$ capped at 10 bps per leg + fees; `run_f2_a1.py:94-109`); `book_sim()` $66K / 20 slots / $3,300 per position, first
  come, no refill. Survivorship: the panel carries delisted names' prices from 2018 (XNAS tape) but none after 2024-06
  (DATA.md gap 8) — every cell is scored ALSO on the 2018-01 → 2023-12 sub-window with delisted names included vs
  excluded; if the effect lives only in the survivors the cell is VOID. `taint` (unreverted reverse splits) excluded.

## Signal definitions (a "purchase" = NONDERIV_TRANS row with TRANS_CODE = 'P', TRANS_ACQUIRED_DISP_CD = 'A', DOCUMENT_TYPE
'4' (amendments '4/A' excluded), AFF10B5ONE ≠ 1 (planned 10b5-1 trades carry no information), TRANS_PRICEPERSHARE > 0)
| cell | signal at the close of the first session after FILING_DATE | hold |
|---|---|---|
| I1 CLUSTER (primary) | ≥ 2 distinct reporting owners of the issuer filed purchases within the 10 sessions ending on this filing date, total value (Σ shares × price) ≥ $100K | 20 sessions (report-only 5 and 60) |
| I2 OFFICER | a single purchase ≥ $50K by a reporting owner flagged Officer or Director with a title containing CEO, CFO, President, Chair or Director | 20 sessions |
| I3 OPPORTUNISTIC | any purchase ≥ $25K by an owner who has NO purchase of that issuer in the prior 12 months (Cohen-Malloy-Pomorski non-routine) | 20 sessions |
| I4 13D | initial SC 13D filing on a PIT-eligible name | 20 sessions (report-only 60) |
One position per (symbol, signal session); overlapping signals on the same name within a hold extend nothing (the first
signal owns the slot). Long only.

## Pass bar (frozen)
TRAIN 2016-01 → 2021-12: mean net return per trade ≥ +0.8 %, day-clustered t ≥ 2.5, ≥ 55 % of months positive.
VAL 2022-01 → 2023-12: same sign, mean net ≥ +0.5 % per trade, day-clustered t ≥ 2.0, ex-top-5 % > 0, ≥ 3 signals/week,
and the 20-slot book ≥ +1.0 %/month average with max drawdown ≤ 15 %; count-matched null (1,000 draws of random
eligible symbol-sessions on the same dates, same n) percentile ≥ 99. Survivorship arm (2018-23 with delisted names) must
keep ≥ half the effect. TEST 2024-01 → 2026-09 SEALED; read ONCE for the single best passing cell.

## Independent check and consequences
Rebuild of every signal row and every trade from this prose by an agent that has not read the builder's code (the
`independent_check.py` pattern of the multiday programme): signal-set agreement ≥ 99 %, trade returns within 0.05 %.
Refuters on any passing cell: look-ahead (FILING_DATE vs TRANS_DATE, late filings, the next-session rule, amendments),
data (survivorship, ticker mapping, price adjustment, 10b5-1 flag coverage), statistics (concentration in symbols/months,
winner-capped, drop the best 2 months, multiplicity across 4 cells + 3 hold variants). PASS → an owner decision on a new
book: paper-trade 8 weeks with the exact orders (MOC via Alpaca `cls` time-in-force — an engine the repo does not yet
have, ≈ 2 weeks of build), then a $10K allocation. FAIL → closed; the data build stays for any future information cell.

## Not allowed
Moving a threshold or hold; adding a cell after a number exists; reading TEST for more than one cell; using TRANS_DATE
as the signal date; scoring without the survivorship arm.
