# PREREG — cell 1,658: HOD-break P&L by STOP DISTANCE (R as % of price) under the MEASURED cost — a read, not a rule

FROZEN 2026-09-29 15:40 UTC before any number (owner: "is the real cost a mix, or should we prefer the 0.04 R cases…
where is the win-rate / P&L better, split into buckets?"). Programme count: 1,657 → 1,658.

## Why
Live 9/29 measured the HOD cost per fill at ≈ 7 bps entry (resting stop-limit vs trigger, 10 fills) and ≈ 3–6 bps
exit (broker-leg stop, 2 fills). In R that is 0.04 R on a 3.4 % stop (TWST) and 0.29 R on a 0.66 % stop (TTAN, paper).
The population's verdict (cell 1,438, −0.21 R net) charged a flat tape-replay cost. The question is whether the gross
edge and the net edge differ by stop distance — a cost-fraction mechanism, not a new signal.

## Population and cost
The cell 1,438 fill set (the live rule with correct levels, bars_sip.db, 9,911 fills 2025-01..2026-09; the file named
in `REPORT_1438.md`). No re-selection, no new filter: every fill keeps its outcome; only the COST is re-expressed per
fill as (7 bps + 6 bps) × price / R_dollars, i.e. 13 bps of price per round trip converted into R with the fill's own
stop distance; EOD exits pay the same 13 bps. Buckets by R% = stop distance / entry price: < 0.75 %, 0.75–1.5 %,
1.5–3 %, ≥ 3 % (pre-declared; the 0.6 % cap in config makes the first bucket the capped one — the agent states what
`params.cap` does from the code).

## Report (per bucket, TRAIN/VAL halves of the 1,438 split and pooled)
n, share of fills, fills/week, win rate, mean gross R, mean net R at the measured cost, day-clustered t of net, ex-top-5 %
net, mean $ at $50 risk, the MDE per bucket, and the same table for the live-fill 9/25–9/29 sample (13 fills, its own
line, no verdict). Also the win rate and net R by TARGET distance (2R in % of price) since the two move together.

## Reading rule (frozen)
This cell produces a table, not a rule. A stop-distance filter becomes a PREREG'd rule only if a bucket shows net
≥ +0.05 R with t ≥ 2 on BOTH halves and ex-top-5 % > 0 AND the bucket keeps ≥ 3 fills/week; then it goes to the
paper session first. Selecting buckets after reading VAL is not allowed (both halves must agree).
