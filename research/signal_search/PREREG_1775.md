# PREREG 1,775 — does the pre-market HEADLINE TYPE predict the squeeze tail of the gap-fade short? (news veto for 1,773/1,774)

FROZEN 2026-11-14 before any number. Cell 1,775. Rules: write only `research/signal_search/1775_*`, `RESULT_1775.md`; read-only DB
URIs; `ulimit -v 2500000; nice -n 19`; budget ≤ 60 calls. Network: ONLY Alpaca news REST history
(`GET https://data.alpaca.markets/v1beta1/news?symbols=&start=&end=&limit=50&include_content=false`, HOD paper keys from .env,
never printed), ≤ 200 req/min, cached per day in `research/signal_search/1775_cache/news/<day>.parquet` (atomic, resumable).
Classification = Claude Haiku 4.5 (owner's order 11/14: 'no regex'), run through this session's agent harness in batches of
≤ 300 headlines (the cell's own python has no API key — ANTHROPIC_API_KEY pending from the owner); the rubric below is the
prompt, fixed before any number; labels cached in `1775_labels.parquet` (headline id → category, strength 1–5, one-line why).
No regex anywhere in this cell (owner 11/14).

## Why
1,773 (MOO short → MOC) fails on its tail: worst-5 % −21 %, ex-top-5 % of winners < 0. The question is whether the squeeze days
are the REAL-catalyst gappers (FDA / M&A / contract) and the bleeders are the fluff-PR / offering gappers. If the headline type
separates the tail from the mean, a pre-market veto exists that the live Benzinga stream (`data_sources/news_stream.py`, p50
0.17 s) can apply at 09:25. Prior: 1,764 (EDGAR dilution/earnings) REJECT; ORB LLM catalyst filter REFUTED on LONG fills; the
2026 "buy the rumor" paper: news direction adds nothing to shorting covered small caps. So the pre-committed expectation is a
null on the MEAN; the read that matters is the TAIL.

## Data
`1773_events.parquet` (symbol-day, open, close, r_moc, tiers, split) + news for each symbol-day in [prev close 16:00 ET, 09:30 ET
of day D] from Alpaca news (Benzinga, history back to 2015). Coverage rail: ≥ 80 % of symbol-days queried successfully;
"no headline" is a valid category, not missing.

## Rubric (Haiku prompt categories; the word lists are DESCRIPTIONS for the prompt, not patterns)
Haiku also returns `strength` 1–5: 5 = binding, company-changing, verified (FDA approval, signed M&A, large contract with a
named counterparty); 1 = promotional / vague / recycled. Output strict JSON per headline; unparseable → re-ask once, else NONE.
1 MA: acqui|merger|buyout|takeover|to be acquired|definitive agreement
2 DILUTION: offering|pricing of|registered direct|private placement|warrant|at-the-market|ATM|dilut|reverse split|
  shelf|convertible
3 FDA: FDA|phase [123]|clinical|trial results|approval|breakthrough|PDUFA|topline
4 CONTRACT: contract|partnership|collaboration|agreement with|awarded|order from|deal with|license
5 EARNINGS: earnings|quarterly results|Q[1-4]|fiscal|guidance|revenue
6 ANALYST: upgrade|downgrade|initiat|price target
7 PR_OTHER: any other headline
8 NONE: no headline in the window
Also: n_headlines in the window, minutes from the last headline to 09:30, source (benzinga vs press release wire if the field
distinguishes it).

## Reads
PRIMARY (tail): on VAL, for the 1,773 primary book (equal-$, no stop): the share of each category in the worst-5 % trades
(the squeezes, i.e. the long's top-5 %) vs its share in the book; odds ratio with a 95 % CI (day-clustered bootstrap, 500
draws). Pre-committed veto candidates (two, declared): (i) the category with odds ratio > 2 on TRAIN; (ii) strength ≥ 4. Each tested once on VAL.
Secondary: (a) mean net (1,773 cost lines) by category, TRAIN then VAL, n per category; (b) the 1,773 book with the TRAIN-chosen
veto applied on VAL: net, t, ex-top-5 %, worst-5 %, weekly P10 — against the count-matched placebo p95 of dropping the same
NUMBER of random trades (binding lesson 11/13); (c) the same for the 1,774 book if its events parquet exists by then (10:30
cover), else say so; (d) n_headlines tercile and recency tercile as two extra declared cuts.
## Pass bar (for a VETO rule)
VAL: the vetoed category's worst-5 % share odds ratio > 2 with CI excluding 1; the vetoed book's worst-5 % improves by ≥ 5 pp
AND its net mean does not fall by more than the placebo p95 drop; ≥ 60 % of the book retained. PASS → the veto becomes a line in
the paper instrument spec (stream-based at 09:25, same rubric); FAIL → FACT, CHILDREN ≥ 2.

## Output
`1775_fetch.py`, `1775_classify.py` (rubric as code + 20 unit asserts on sample headlines inside the file), `1775_score.py`,
`1775_events.parquet`, `1775_summary.csv`, `1775_placebo.csv`, `RESULT_1775.md` ≤ 350 words ending "CHILDREN:". Return ≤ 150
words: coverage, category counts, the tail odds ratios (TRAIN, VAL), the veto's VAL effect vs placebo, PASS/FAIL, children.
