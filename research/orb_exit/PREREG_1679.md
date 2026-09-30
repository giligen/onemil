# PREREG — cell 1,679: the ORB give-back — exit rules on REAL ledgers, R-floored (FROZEN 2026-09-30 16:10 UTC)

Why: 1,678 Part 3 applied the HOD-trained post-entry exit models to ORB's book and every cell "passed" (+0.5–1.7 R,
t 4–16) — flagged NOT a claim: the book CSV has no entry/exit times (stops and exits were reconstructed), and ORB's R
(entry − range low) can be tiny, so R-multiples explode (own-R tail 11.6 R). One number in that read is robust and is
the lead: 21 % of ORB fills reach +1 R and then close ≤ 0, giving back 1.88 R on average. ORB is the positive book; its
give-back is its known loss (AXTL 9/29: +$170 → −$96). This cell reads exit rules on ledgers with real times.

## Ledgers
L1 LIVE: `data/trades.db` strategy = 'orb', account live (NULL), filled and exited, 2026-05-19..09-23, n 123 — exact
   entry/exit times and prices, actual shares. L2 BT: `analysis_results/orb_bplus_book.csv` (2025-01..2026-09, n 483)
   with the entry MINUTE reconstructed by the BT's own rule from the minute bars (the first bar after the 5-minute
   opening range whose high crosses the range high; entry = the book's entry_price; stop = the book's range low) —
   the reconstruction is validated on L1's overlap (same symbol-days) and its agreement reported. Minute bars: the
   symbol-days appended to `research/bf_zero/bars_sip.db` through the designed appender; coverage line first.
R floor (memory: "R must exceed the spread"): fills with R < 0.5 % of the entry price are reported separately and
excluded from R-unit reads; every read is ALSO given in dollars at the live sizing (L1: actual shares; L2: $375 risk).

## Reads (both ledgers; L2 halves = 2025 vs 2026; L1 whole, with the L2 sign as the agreement line)
1. Give-back anatomy: share of fills that reach +0.5 / +1 / +1.5 R and close ≤ 0; R given back; minutes from the
   peak to the exit; how many were closed by the force-close vs the stop vs the target.
2. The HOD-trained exit models from 1,678 (both halves' models, unchanged): rule X(c = 0.20) and X+ from minute 15 —
   paired ΔR and $ vs the ledger's actual exit, day-clustered t, ex-top-5 %, MDE.
3. Pre-declared mechanical alternatives, each paired vs the actual exit: (a) lock the stop at +0.5 R once +1 R is
   reached; (b) lock at breakeven once +1 R; (c) trail MFE − 1 R; (d) 50 % out at +1 R, rest as the live rule;
   (e) the live lock (arm at +1.75 R, stop to +0.5 R) as the reference (it is what the ledger already did);
   (f) time stop at 60 min if < +0.5 R.
4. The exit-lab caveat applies: any lock or trail is a tighter stop in disguise; report the give-back saved vs the
   continuation forgone for every rule (the 1,669 decomposition) so the trade-off is visible.

## Pass bar (for changing ORB's live exit rule)
Paired ΔR ≥ +0.05 R with day-clustered t ≥ 2.5 and ex-top-5 % > 0 on L2 in BOTH years, AND same-signed on L1, AND
the $ effect at $375 risk positive on both; R-floored. A pass → independent rebuild → ORB PAPER with the new exit as
the one mechanics change of that session → forward read at 100 fills. No rule enters `orb.yaml` without that path
(research/orb_machine_rules.md).

## Multiplicity
9 rules × 2 ledgers × (2 halves + whole) ≈ 54 paired reads. Not allowed: tuning lock levels, trail widths or c; using
bars after the decision bar; dropping the R floor; pooled-only numbers.

## Output
`research/orb_exit/RESULT_1679.md` (≤ 140 lines: coverage and reconstruction agreement first, the anatomy table,
then the rule table with the decomposition), `1679_reads.csv`, `1679_per_fill.csv`, `1679_orb_exit.py`,
`1679_orb_exit.log`. The agent returns ≤ 150 words.
