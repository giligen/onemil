# PREREG 1,767 — J2: WHERE IN THE DAY does the gapper fade live? (minute-bar anatomy of 1,754's −185 bp, on the honest bars) — a read for the owner's discretionary short list

FROZEN 2026-11-13 21:05 UTC before any number. Programme cell count: 1,766 → 1,767. Rules: write only
`research/hod_entry/1767_*`, `RESULT_1767.md`; no git/service/orders/network; data on disk; `ulimit -v 2500000;
nice -n 19`, peak RSS < 1 GB; budget ≤ 40 calls. TEST sealed. Parent: 1,754 (gap ≥ 10 % $1–20 names fade −185 bp
open-to-close, 8/8 years) and 1,755 (the engine cannot short them; the owner CAN, name by name, where borrow
exists). This cell produces no engine trade: it produces the fade's clock and the filter for a list.

## Data
`1762_cache/bars/<day>.parquet` (honest PIT gapper symbol-days; raw SIP minute bars), `1762_cache/daily.parquet`
(prev close), `cell_1428_population.csv`. Gap = open_09:30 / prev_close − 1. Sub-population: gap ≥ 10 % (and
the 5–10 % and ≥ 20 % buckets beside it); price tiers $1–5 / $5–20 / > $20.

## Reads (TRAIN halves, TRAIN, VAL once; all in bp of the 09:30 open, equal weight, day-clustered t)
1. The clock: cumulative mean return from the open at 09:35, 09:45, 10:00, 10:30, 11:00, 12:00, 14:00, 15:00,
   15:55 — the path of the fade; the share of the open-to-close move done by 10:00 and by 10:30.
2. The 10:00 rule: among gappers with close_10:00 < open (fade declared), the 10:00 → 15:55 return; among those
   with close_10:00 > open (holding), the same. Day-t of the difference.
3. The high clock: the minute of the day's high (distribution: share set by 09:35 / 09:45 / 10:00 / later) and the
   mean drawdown from the first 15-minute high to the close.
4. The owner's short rule, scored as a SHORT with a hard stop: enter short at the 10:00 close if close_10:00 < open
   AND price ≥ $5 (borrowable proxy), stop at the morning high (09:30–10:00) + 50 bp slip, cover at 15:55; cost
   2 bp per leg + half_spread (symbol-median, say so) + borrow 50 bp/day. Numbers: n/wk, mean net bp, day-t,
   ex-top-5 % of winners, 5 %-worst tail mean, stop-hit share, year signs, TRAIN halves, VAL once, MDE.

## Kill (read 4 only — reads 1–3 are facts, reported whatever they say)
VAL net ≤ +50 bp or day-t < 2; TRAIN halves opposite; ex-top-5 % of winners ≤ 0; 5 %-worst tail mean < −8 %;
< 3 names/wk at price ≥ $5. Survive → TEST once → independent rebuild → an "owner short list" line in
`scripts/hod_owner_list.py` (report-only; the owner decides and borrows).

## Output
`1767_score.py`, `1767_clock.csv`, `1767_short.csv`, `RESULT_1767.md` ≤ 350 words ending with "CHILDREN:" (≥ 2).
Return ≤ 120 words: the clock (share of the fade done by 10:00 / 10:30), read 2's difference and t, the high-minute
distribution, read 4's VAL net / t / n-wk / tail / stop-hit share, KILL/SURVIVE, children.
