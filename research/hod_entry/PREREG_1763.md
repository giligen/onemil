# PREREG 1,763 — LEVELS & CLOCKS on the honest gapper bars (book B1, B2, B4, B6, B7, B8): six reads, one table, each its own kill

FROZEN 2026-11-13 20:35 UTC before any number. Programme cell count: 1,762 → 1,763. Rules: write only
`research/hod_entry/1763_*`, `RESULT_1763.md`; no git/service/orders; network ONLY for Alpaca extended-hours
minute bars (B1; free; ≤ 5 req/s; checkpoint into `1762_cache/premarket/<day>.parquet` — that directory only);
`ulimit -v 2500000; nice -n 19`, peak RSS < 1.5 GB; budget ≤ 70 calls. TEST sealed.

## Data (on disk from 1,762)
`research/hod_entry/1762_cache/bars/<day>.parquet` — full-day minute bars (SIP, raw, with vwap/volume) for the
7,487 PIT gapper symbol-days of `cell_1428_population.csv` (TRAIN/VAL, split, wk); `1762_cache/daily.parquet`;
costs as 1,762 (symbol-median half-spread — say so); stop slip as PREREG_1740. Shared exit for every read:
stop as defined per read, target +2 R, time-out 15:55. R floor 1.0 % of price (1,762 rule). Entry at the bar
close after the trigger + half_spread + 2 bp.

## Reads (all causal at the trigger bar)
B1 PRE-MARKET HIGH: level = max high 04:00–09:29 (fetched; coverage ≥ 90 % or VOID); first 1-min close above it
   from 09:31; stop = min low of the last 3 bars. Comparator: the regular HOD break on the same days.
B2 5-MINUTE CLOSE: level = running HOD (set ≥ 10 min earlier); trigger = a 5-minute bar (09:30-aligned) CLOSING
   above the level vs the 1-minute wick break; stop = the 5-min bar's low. Paired: the same symbol-days, 1-min rule
   vs 5-min rule, ΔR.
B4 ROUND NUMBERS: level = the nearest $1 (price < $10) / $5 ($10–50) / $10 (> $50) above the open that is ≥ 1 %
   away; first close above it after 09:45; stop = min low of the last 3 bars. Comparator: the HOD break.
B6 HOD-AGE day typing: at 10:30, age = minutes since the day's high was set; buckets ≤ 15 / 15–45 / > 45 min;
   outcome = the regular HOD-break rule's net R on that day AND the 10:30 → close return. No trade — a filter read
   on the break and a report line for the owner list.
B7 VWAP FIRST PULLBACK: after a new HOD by 10:00, first bar whose low touches vwap and closes above it; entry at
   that close; stop = that bar's low − 0.2 %; target +2 R.
B8 HALT RESUMPTION: a gap of ≥ 5 consecutive missing minutes between 09:35 and 15:30 with the first bar after it
   ≥ 5 % away from the last bar before it (LULD proxy — say so); entry = the first close above the resumption bar's
   high; stop = the resumption bar's low. Report n/yr.

## Numbers (per read; TRAIN halves, TRAIN, VAL once)
n/wk, net mean R, day-clustered t, ex-top-5 %, winner-capped +3 R, weekly P10, green-week share vs count-matched
null, year signs, MDE at 80 % power; for B2 the paired Δ vs the 1-min rule; for B6 the bucket table.

## Kill (per read; a read SURVIVES only if none fires)
VAL net ≤ 0 or day-t < 2; TRAIN halves opposite; VAL ex-top-5 % ≤ 0; < 3 fills/wk (B8: < 1/wk); B2: paired Δ ≤ 0
on VAL; B6: no bucket differs with day-t ≥ 2 on both TRAIN halves and VAL. Any survivor → TEST once → independent
rebuild → dry-run spec.

## Output
`1763_fetch_premarket.py`, `1763_score.py`, `1763_reads.csv`, `1763_val.csv`, `RESULT_1763.md` ≤ 400 words
(six reads) ending with "CHILDREN:" (≥ 2). Return ≤ 150 words: per read n/wk, VAL net, day-t, ex-top-5 %,
KILL/SURVIVE; B6 bucket line; children.
