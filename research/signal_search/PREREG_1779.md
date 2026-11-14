# PREREG 1,779 — GAPSHORT frequency frontier (owner 2026-11-14: "can we increase the frequency of shorts?")

## Facts on disk that bound the answer (scoping counts only, no edge read)
* Population (cell 1,428 ∪ tiers, `1773_events.parquet`, gap_ok, gap ≤ 200 %): 132 (TRAIN) / 174 (VAL) symbol-days
  per week. The live rule takes top-10 by gap (49–50/wk) then L1 top tercile (≈ 22–25/wk, median 5 names/day).
* Gross open→close fade by gap rank, VAL: rank 1–5 +364 bp, 6–10 +220, 11–15 +24, 16+ +26 — the fade's money sits in
  the top-10; rank 11+ is below the ≈ 60 bp cost stack on the mean. L1 terciles in 1,776 first build: T3 +159,
  T2 +70 (t 0.6), T1 −41 (VAL, net).
* Extended survivors at 10:30 (`p1030 ≥ 1.05·open`, 1,774 rows with a 10:30 quote): 7 % of rows, 3.8/wk VAL;
  as a LONG they were the worst 1,778 cut (TRAIN −212 bp).

## Question
Which expansions of the live GAPSHORT rule add names per week whose MARGINAL book (the added rows only) clears the
1776R bar at the same cost stack? Frequency without marginal edge is dilution, not frequency.

## Frozen ladder (each rung read on its MARGINAL rows; the base book is the 1776R rule and is not re-read)
R1  L1 gate: T3 → T2 ∪ T3 (TRAIN cutoff = 1776R's median L1 among top-10 rows). Marginal = top-10 rows in T2.
R2  L1 gate removed. Marginal = top-10 rows in T1.
R3  TOP_K 10 → 15 at L1 T3. Marginal = rank 11–15 rows in T3.
R4  TOP_K 10 → 15 at L1 T2 ∪ T3. Marginal = rank 11–15 rows in T2.
R5  Second slot 10:30: SHORT at 10:30 the names with `p1030 ≥ 1.05·open` that are NOT already in the base book that
    day (base book names are covered at 10:30 — no re-short), entry at the measured `bid1030` − 10 bp, cover MOC at
    `close` (auction, no spread), fees 4 + locate 9 + borrow 27.8. Marginal = those rows. Secondary read: cover at
    the measured `ask1100` (rows that have it).
R6  Second slot 10:30 restricted to L1 T2 ∪ T3.
Pricing for R1–R4: short MOO at `open`, cover at the measured 10:30 ask + nothing (1776R convention: `open/ask − 1`,
cost FEES 4 + LOCATE 9 + BORROW 27.8 bp). Quotes: `1776R_cache/quotes.jsonl` (k = day|symbol, ap/bp at 10:30) and
`1774_events.parquet` (ask1030/bid1030/ask1100); rows still missing a quote are FETCHED with the resumable class in
`1776R_build.py` (free Alpaca SIP quotes, HOD keys from .env, ≤ 190/min, append to the same jsonl). Report quote
coverage per rung; a rung is VOID below 80 % or with a winner/loser missingness gap > 5 pp.

## Reads per rung (TRAIN halves first, one VAL read)
n/wk added (median, p10), names/day after the expansion, mean net bp, day-clustered t, ex-top-5 %, worst-5 %, weekly
P10 in R ($300 notional, R = $150 as 1776R), TRAIN H1/H2 sign; the COMBINED book (base + rung) weekly P10 and mean.

## Placebos (binding 11/13): for every rung whose VAL mean > 0
(i) count-matched draws from the population rows of the same days NOT in the base book and not in the rung, same
    pricing (quotes needed → draw only from quoted rows, say so); (ii) same symbol ±5 days, own quote. 200 draws, p95.

## Pass bar (per rung, VAL, marginal rows)
net ≥ +50 bp, day-t ≥ 2, ex-top-5 % > 0, worst-5 % ≥ −12 %, both TRAIN halves same sign, beats both p95s, adds
≥ 3 names/wk median. A rung that passes becomes a candidate for the live rule ONLY after an independent rebuild;
a rung that fails is reported with its n/wk so the owner sees what frequency costs.

## Output
`1779_build.py`, `1779_marginal.parquet`, `1779_summary.csv`, `1779_placebo.csv`, `RESULT_1779.md` ≤ 120 lines:
table FIRST = rung × (names/wk added, VAL net, t, ex-top-5 %, worst-5 %, pass/fail), then details, coverage,
caveats, "CHILDREN:" ≥ 2.

## Rules for the agent
Sonnet, ≤ 50 calls, docstrings everywhere, verbose progress, `os.nice(19)`; writes only `research/signal_search/1779_*`,
`RESULT_1779.md` and appends to `1776R_cache/quotes.jsonl` (atomic append per row, resumable). Read-only sqlite
(`?mode=ro`). No git, no services, no orders; GET-only toward Alpaca data (quotes). Return ≤ 150 words.
