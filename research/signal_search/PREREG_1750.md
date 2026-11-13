# PREREG 1,750 — opportunistic insider cluster buys on filing (candidate G1) — Stage 1, rejection-only

FROZEN 2026-11-13 16:10 UTC before any number. Programme cell count: 1,749 → 1,750. Rules as PREREG_1746 (write
only `research/signal_search/1750_*`, `RESULT_1750.md`, cache `1750_cache/`; no git / service / orders; budget ≤ 60
calls; `df` ≥ 5 GB; SEC fair-access: a descriptive User-Agent with a contact e-mail from `.env` if present, else
"onemil research", ≤ 8 requests/s). This task IS the owner's request; do not pivot.

## Claim under test (Cohen–Malloy–Pomorski JF 2012 "Decoding inside information"; microcap update arXiv
2602.06198 — URLs in the G1 block of CANDIDATES_20261113.md)
Open-market insider PURCHASES that break the insider's routine calendar (opportunistic) predict +82 bps/month
abnormal; routine trades ≈ 0; clustered buys (≥ 2 insiders in a window) carry more. Entry is on the filing, which
Form 4 makes public within 2 business days — the signal is causal at the filing timestamp.

## Data (free)
1. SEC "Insider Transactions Data Sets" (Form 3/4/5 structured bulk, quarterly zips:
   https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets) 2016Q1 → 2026Q3: tables
   SUBMISSION (ACCESSION_NUMBER, FILING_DATE, ISSUERTRADINGSYMBOL), NONDERIV_TRANS (TRANS_CODE 'P' = open-market
   purchase, TRANS_DATE, TRANS_SHARES, TRANS_PRICEPERSHARE, TRANS_ACQUIRED_DISP_CD 'A'), REPORTINGOWNER (RPTOWNERNAME,
   RPTOWNER_RELATIONSHIP). Probe ONE quarter first and report the row counts; a quarter missing → LOST count.
2. Daily bars: on disk from 1,746/1,748 for ranks 1–3,000; fetch the missing symbols the same way (1746_fetch
   pattern) if the signal names fall outside — report the share outside.
3. Universe at the filing date: common stocks ranks 1–3,000 by 20-day dollar volume, price ≥ $3.

## Signal
Event = a Form 4 with ≥ 1 open-market purchase ('P', acquired) by an officer or director, total value ≥ $25K,
filed on day f (FILING_DATE). Opportunistic = the same owner did NOT file a purchase in the same calendar month of
each of the prior 2 years (the paper's routine definition needs 3 years; use 2 and say so). Cluster = ≥ 2 distinct
owners with qualifying purchases in the same name within 5 trading days (count the second filing as the event day).
Entry at the OPEN of f+1 (the filing is often after 16:00), exits at the close of f+5 (primary) and f+20 (secondary).
Reads: ALL purchases; OPPORTUNISTIC; CLUSTER; ROUTINE (the paper's zero — a placebo by construction); by tier.

## Splits
TRAIN 2016-01 → 2020-12 (halves 2016–18 / 2019–20); VAL 2021-01 → 2023-12; TEST 2024-01 → 2026-11 sealed.

## Numbers (per split, per read)
n events/wk, mean gross excess vs the universe mean over 5 d and 20 d, day-clustered t, ex-top-5 %, winner-capped
+20 %, monthly green share, year-by-year sign, the ROUTINE read beside OPPORTUNISTIC, cost from the measured
half-spread tables (+ 2 bp per leg), net, MDE at 80 % power.

## Kill (pre-committed — any one kills the primary = OPPORTUNISTIC, 5 d)
VAL gross excess ≤ +30 bp or |t| < 2; TRAIN halves opposite sign; ex-top-5 % ≤ 0 on VAL; 2022 negative;
OPPORTUNISTIC not above ROUTINE on TRAIN and VAL (mechanism fails); net of cost ≤ 0 on VAL; < 3 events/week.
Survive → TEST once → independent rebuild → paper-sleeve spec (long on filing, 5-day hold, 8 names, $150 R).

## Output
`1750_fetch.py`, `1750_events.parquet`, `1750_score.py`, `1750_summary.csv`, `RESULT_1750.md` ≤ 350 words.
Return ≤ 120 words: data coverage, events/wk by read, VAL opportunistic gross / t / net vs routine, year signs,
KILL/SURVIVE.
