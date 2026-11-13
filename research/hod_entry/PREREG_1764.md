# PREREG 1,764 — EVENTS the population hides (book E1 dilution filter, E2 earnings-day breaks): losers anatomy on the honest gapper bars with EDGAR

FROZEN 2026-11-13 20:45 UTC before any number. Programme cell count: 1,763 → 1,764. Rules: write only
`research/hod_entry/1764_*`, `1764_cache/`, `RESULT_1764.md`; no git/service/orders; network ONLY to SEC EDGAR
(`https://www.sec.gov/files/company_tickers.json` once; `https://data.sec.gov/submissions/CIK##########.json`
one per CIK; descriptive User-Agent with a contact e-mail from `.env` if present else "onemil research
admin@onemil.local"; ≤ 8 requests/s; checkpoint per CIK into `1764_cache/submissions/`); `ulimit -v 2500000;
nice -n 19`, peak RSS < 1 GB; budget ≤ 60 calls. TEST sealed.

## Claim (losers anatomy — the bottom-up rule: find what the −1 R fast losers share)
E1 Serial diluters: a gapper with a live shelf / registration (S-1, S-3, F-1, F-3, 424B2/B3/B4/B5, or an 8-K with
   an at-the-market / securities-purchase agreement in the prior 90 days) has sellers with inventory above the
   level; its breaks fail faster. The filter is causal (filings are public before the day).
E2 Earnings-day breaks (8-K item 2.02 or 7.01 filed on day t−1 after 16:00 or on day t before 09:30) carry a
   PEAD-type follow-through; non-event gappers do not.

## Data
Population and outcomes: the honest gapper symbol-days (`cell_1428_population.csv`) with the regular HOD-break
rule's net R per symbol-day from `1763_score.py`'s HOD comparator (import it; same costs, same R floor). Ticker →
CIK from company_tickers.json (report the unmapped share; ≥ 80 % mapped or VOID). Per CIK the submissions JSON
(form, filingDate, items for 8-K, primaryDocument). Features per symbol-day: dil_90 (any of the E1 forms in
[t−90, t−1]), dil_30, n_dil_filings_365, earn_day (E2 definition), any_8k_2d (any 8-K in [t−2, t]).

## Reads (TRAIN halves, TRAIN, VAL once)
Bucket tables: HOD-break net R by dil_90 (yes/no), by earn_day (yes/no); n, mean, day-clustered t of the
difference, ex-top-5 %, share of −1 R-fast losers (stop hit within 15 min) per bucket. Filter read: the book
without dil_90 = yes vs with, paired on days (ΔR/wk, fills/wk kept, green-week share). Placebo: the same split by
a random 50 % of CIKs.

## Kill (per read; adopt only if ALL hold)
E1: flagged bucket below unflagged on BOTH TRAIN halves and on VAL with day-t ≥ 2; filtered ΔR/wk > 0 on VAL with
≥ 3 fills/wk kept; placebo shows no effect. E2: earn_day bucket above the rest on both halves and VAL with
day-t ≥ 2; ≥ 3 fills/wk. A survivor → TEST once → independent rebuild → dry-run filter spec (`hod_break` config
key, parity test) / owner-list column.

## Output
`1764_fetch_edgar.py`, `1764_score.py`, `1764_features.parquet`, `1764_buckets.csv`, `RESULT_1764.md` ≤ 350 words
ending with "CHILDREN:" (≥ 2). Return ≤ 120 words: mapped share, flag prevalence (dil_90 %, earn_day %), per read
the VAL bucket means / day-t / fast-loser shares, filtered ΔR/wk, placebo, ADOPT or REJECT per read, children.
