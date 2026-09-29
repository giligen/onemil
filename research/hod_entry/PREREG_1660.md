# PREREG — cells 1,660–1,661: the HOD exit lab and the entry limit RE-READ at the MEASURED cost

FROZEN 2026-09-29 17:25 UTC before any number. Programme count: 1,659 → 1,661. Owner: "apply the minor improvements you
already saw to pull the −0.025 R upwards — think hard." The honest levers are cost levers; every filter on this
population was mined 1,636 times under a cost model that today's live fills showed to be ≈ 3× too high.

## 1,660 — exit lab re-read
`research/hod_exit_lab/` scored 35 exit/stop/hold variants against the live rule B0 under the tape-replay cost
(−0.22 / −0.30 R net, "every variant within ±0.04 R"). Re-score the SAME cells and the same fills with the measured
cost: entry 7 bps of price, stop exit 6 bps, target exit 0 (a resting limit at the broker fills at the touch), EOD exit
11 bps (bid) AND a second line at 1 bp (MOC), each converted to R with the fill's own stop distance; the base and each
variant on both halves, with the 1.5 % and 3 % stop-distance floors applied as separate columns. Report per cell: mean
net R, day-clustered t, ex-top-5 %, fills/week, and the paired ΔR vs the re-read base with ex-top-5 % of ΔR.
Reading rule: a variant is a candidate only if ΔR ≥ +0.05 R with t ≥ 2 on BOTH halves and ex-top-5 % of ΔR > 0 and
the fill rate ≥ 3/week; candidates go to the paper session one at a time, never two in one session.

## 1,661 — entry limit width on the dry cross records
The dry ledger since 9/26 logs every arm's cross: the print that crossed the trigger, the ask at that moment, and
whether a limit of trigger + 0.15 % would have filled ("FILL"/"NO FILL (ask > limit)"). Re-score with limits of
+0.05 %, +0.10 %, +0.15 % (today), +0.25 %: fill rate, mean entry slip vs trigger, and — using the counterfactual
outcomes of the same arms — the net R of the FILLED set under each limit (the unfilled arms are dropped, no chase).
Reading rule: a narrower limit becomes the paper setting only if its filled set's net R is ≥ the +0.15 % set's by
+0.03 R with the fill count ≥ 70 % of today's; a wider limit never.

## Not allowed
Adding variants beyond the 35 of the lab; picking a variant on one half; changing the cost constants after a number.
