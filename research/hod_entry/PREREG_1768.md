# PREREG 1,768 — J7: short the ≥ 10 % gapper EARLY (09:35–09:45) with the opening-range high as the stop — the fade is a first-30-minute event (child of 1,767)

FROZEN 2026-11-13 21:08 UTC before any number. Programme cell count: 1,767 → 1,768. Rules as PREREG_1767 (write
only `research/hod_entry/1768_*`, `RESULT_1768.md`; data on disk; `ulimit -v 2500000; nice -n 19`; budget ≤ 40
calls; TEST sealed). Parent facts: 62–68 % of the fade is done by 10:00, the high is set by 09:45 in ~60 % of
names; the 10:00 short is too late and its tail is −18 % on the worst 5 % of days. The squeeze tail is the
question — reads are built to measure it, not to hide it.

## Entries (short, price ≥ $5 at the open, gap ≥ 10 %; secondary gap 5–10 %)
E1 09:35 close (after the first 5-minute range), stop = the 09:30–09:35 high + 50 bp slip.
E2 09:45 close, stop = the 09:30–09:45 high + 50 bp.
E3 09:45 close ONLY IF close_09:45 < open_09:30 (fade declared early), stop as E2.
E4 the first 1-minute close below the 09:30–09:35 low between 09:36 and 10:00 (range-low break = the ORB short
   mirror), stop = the range high + 50 bp.
Cover at 10:30 (primary — where the fade is done) and at 15:55 (secondary). Cost: 2 bp per leg + symbol-median
half-spread (say so) + borrow 50 bp/day (100 bp sensitivity). Stop fills at the stop price + slip (minute bars
cannot time intrabar; say so).

## Numbers (TRAIN halves, TRAIN, VAL once; per entry × cover)
n/wk, mean net bp, day-clustered t, hit rate, ex-top-5 % of winners, worst-1 % and worst-5 % day means, max
single-trade loss, stop-hit share, mean R-multiple with R = stop distance, weekly P10, year signs, MDE.

## Kill (primary = E3 × 10:30 cover; any one kills)
VAL net ≤ +50 bp or day-t < 2; TRAIN halves opposite; ex-top-5 % of winners ≤ 0; worst-5 % day mean < −8 %;
< 3 names/wk; net at borrow 100 bp ≤ 0. Survive → TEST once → independent rebuild → owner short-list line
(report-only) — never an engine order.

## Output
`1768_score.py`, `1768_grid.csv`, `RESULT_1768.md` ≤ 350 words ending with "CHILDREN:" (≥ 2). Return ≤ 120 words:
per entry (E1–E4, 10:30 cover) VAL net / t / n-wk / stop-hit / worst-5 %; the primary's TRAIN halves; KILL/SURVIVE;
children.
