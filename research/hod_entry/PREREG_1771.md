# PREREG 1,771 — A4: does STOP PLACEMENT relative to round numbers change the measured stop slip? (tape of cell 1,443, on disk)

FROZEN 2026-11-13 22:05 UTC before any number. Programme cell count: 1,770 → 1,771. Rules: write only
`research/hod_entry/1771_*`, `RESULT_1771.md`; no git/service/orders/network; data on disk; `ulimit -v 2500000;
nice -n 19`, peak RSS < 1 GB; budget ≤ 40 calls. Parent: 1,443 (stop slip 35 bp mean / 82 p90 on 4,122 measured
stops; 41 % of stops slip > 30 bp). Execution cost is the wall every cell hit today; this is one lever on it.

## Data
`research/hod_entry/sip_cache_stopslip/<day>.pkl.gz` (230 days, the per-fill measured slip and, if stored, the
tape window — inspect one file first and state exactly what is in it), `causal_arming_causal.csv` (the 1,438 fills:
stop price, exit, why, split), `cell_1443.py` (loader; import `load_cache`, `load_base_fills`, `collect`).
Only TRAIN-H2 and VAL (as 1,443); TEST untouched.

## Features (per measured stop)
d_dollar = distance of the stop price to the nearest whole dollar (cents, signed: stop just ABOVE a round number
= the cluster is below it); d_half = to the nearest $0.50; d_dime = to the nearest $0.10; price tier ($1–5 /
5–20 / > 20); stop distance in bp; minute of day; spread tercile (from signals half_spread if joinable, else
VOID that split).

## Reads
1. Slip (bp) by d_dollar bucket: within 2 ¢ above a round number / 2–10 ¢ above / > 10 ¢ / within 2 ¢ below /
   2–10 ¢ below; same for d_half and d_dime; n, mean, median, p90, day-clustered t of "within 2 ¢ above" vs
   "> 10 ¢". Hypothesis: stops sitting 0–2 ¢ ABOVE a round number slip most (the cluster beneath is where the
   prints jump to).
2. Counterfactual placement (only if the tape windows are stored; else VOID this read and say so): for each stop
   in the "0–10 ¢ above a round number" bucket, move the stop 1 tick BELOW the round number; from the tape find
   the first print ≤ the new stop and its fill (next print), then the slip; compare the mean slip + the extra
   stop distance (bp) against the original slip — paired, day-clustered t; net effect in R using the fill's R.
3. Multiplicity declared: 3 grids (dollar/half/dime) × 5 buckets; the dollar/"0–2 ¢ above" vs "> 10 ¢" contrast
   is the primary.

## Pass bar for a RULE (else a fact only)
Primary contrast same sign on TRAIN-H2 and VAL with day-t ≥ 2 on VAL; read 2's paired net improvement > 0 on VAL
with t ≥ 2 and ≥ 20 % of stops affected. A pass → engine spec (`trading/stop_placement.py` helper shared by BT
and live, parity test), owner's word before live.

## Output
`1771_score.py`, `1771_buckets.csv`, `1771_counterfactual.csv`, `RESULT_1771.md` ≤ 350 words ending with
"CHILDREN:" (≥ 2). Return ≤ 120 words: what the cache holds, n measured, the primary contrast (means, t) on both
holdouts, the dime/half grids in one line, read 2's paired net (or VOID), RULE or FACT, children.
