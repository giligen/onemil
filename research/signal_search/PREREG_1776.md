# PREREG 1,776 — the 1,774 book split by PRE-OPEN liquidity, with a same-convention placebo (resolves the 1,774 placebo defect)

FROZEN 2026-11-14 before any number. Cell 1,776. Rules as 1,774 (write only `1776_*`, `RESULT_1776.md`; read-only DB; nice 19;
budget ≤ 50 calls; network = Alpaca sip quotes only, HOD paper keys, cached under `1774_cache/quotes/` which already holds the
1,774 windows — new fetches only for placebo events, resumable).

## What 1,774 left open (facts from its own events parquet, checked in-session)
* VAL primary +65 bp net (t 1.7). Rows WITH a 10:30 bar in `1762_cache` (n 495): gross +366 bp; rows WITHOUT (n 493): −155 bp.
  Same in TRAIN (+406 / −117). Bar-present rows have a median 09:30-bar $vol of $1.2 M vs $0.17 M. Not a price-scale artifact
  (|open/mid − 1| > 0.5 in 0.0 % / 0.4 % of rows). Five sampled bar-absent rows have ZERO rows for the symbol in that day's
  cache file (RGTZ, DMAC, VIR, AEHL, VRDN) — find out why (not in the 1,428 population day list? wrapper? build gap?) and state it.
* The 1,774 placebo (i) drew only bar-present population events (it needed p1030) and priced them bar-open − median half-spread,
  so it compared a +366 subset against the whole book — VOID by construction.
## Question
Is the split predictable BEFORE the open with a causal liquidity variable, and does the liquid book beat a same-subset,
same-convention placebo? Declared causal variables (knowable at 09:25 ET): (L1) pre-market $ volume 04:00–09:25 from
`1762_cache/premarket/<day>.parquet` (present for 98.8 % of population days; absent → NONE bucket, reported); (L2) adv20 from
the population file; (L3) prior-day $ volume from daily_bars. Terciles computed on TRAIN, applied to VAL.
## Reads
1. Bar-absent diagnosis (above) — one paragraph, with the share of absent rows explained by each cause.
2. For L1, L2, L3 terciles: n, net mean (1,774 cost lines: ask cover, fees 4, locate 9, borrow 100), day-t, ex-top-5 %, worst-5 %,
   weekly P10, bar-present share — TRAIN then VAL. PRIMARY = L1 top tercile on VAL (declared).
3. Placebo, same convention and same subset: for the primary tercile's VAL events, 200 count-matched draws of OTHER population
   symbol-days from the SAME days and the SAME L1 tercile, priced at the MEASURED ask at 10:30 (fetch their quotes — up to
   ~1,000 extra symbol-days; cache) → placebo mean p95 on the identical price convention. Also the same-name +5d placebo at the
   measured ask (fetch), n reported; if n < 100 say "thin".
4. Gap T3 × L1 top tercile (declared joint cell, one read, multiplicity 1).
## Pass bar (primary)
VAL net ≥ +50 bp, day-t ≥ 2, TRAIN halves same sign, ex-top-5 % > 0, worst-5 % ≥ −12 %, weekly P10 ≥ −2 R, beats the
same-convention placebo p95, ≥ 3 names/day median. PASS → paper instrument spec (locate 09:20, MOO short, 10:30 cover at ask,
L1 filter) and an INDEPENDENT REBUILD from this prose before the owner sees the number. FAIL → FACT, CHILDREN ≥ 2.
## Output
`1776_score.py`, `1776_events.parquet`, `1776_summary.csv`, `1776_placebo.csv`, `RESULT_1776.md` ≤ 350 words ending "CHILDREN:".
Return ≤ 150 words: bar-absent cause, L1/L2/L3 top-tercile VAL lines, placebo p95 (same convention), joint read, PASS/FAIL.
