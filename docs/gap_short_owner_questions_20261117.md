# GAPSHORT paper instrument: owner questions after day 2 (2026-11-17)

Running paper tally since 2026-11-16: 6 picks, 0 entries filled, $0. BT line for the same days: 11/16 four names mean +60 bp
(CLRO +249, CORD +109, MSTZ -72, TU -46), 11/17 two names mean +160 bp (LITX +327, SPAI -8). Reference: +145 bp net, ~5 names/day.

## (a) The paper account cannot short any hard-to-borrow name
Verified read-only 11/17: LITX, MVLL, MSTZ, CORD, CLRO all carry `shortable=false` on the MOM paper account (and on the live
account); only the ETB name TU is `shortable=true`. The paper account has no locate flow, so 5 of the 6 picks so far were
unexecutable there. The thesis (cell 1,773: HTB locate names) cannot be exercised on paper. Options, your call:
1. keep the paper book as an ETB-only forward instrument (expect few names; the HTB leg stays untested), or
2. bring the live-keys + locate-dollars GO forward (earliest live was 11/30 after >= 2 clean paper weeks; those weeks cannot be
   clean in the sense intended, because the paper account rejects the names the rule picks).

## (b) The 09:19 live gates vs the BT's 09:30 open
SPAI 11/17: pre-market last 2.97 at 09:19 (< the $3 floor), official open 3.01 -> BT pick (+160 bp line contribution), live
miss. Frozen rule, untouched; the parity now labels every such miss (`price_gate_live`, `gap_gate_live`, `l1_gate_live`,
`rank_displaced_live`) with the measured numbers, so after ~10 trading days the tally says how much of the BT line the 09:19
timing forfeits. No PREREG until that count exists.

## (c) Threshold-list skips
MVLL, CLRO (2 of 6) were threshold-list names: by rule, skipped. Same tally applies.
