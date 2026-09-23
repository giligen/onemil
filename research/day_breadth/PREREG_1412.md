# PREREG — cell 1,412: base entry on RISK-ON TAPE only, scored on the sealed quarter (TEST). Frozen before TEST

Source (`BREADTH.md`, report-only tercile table, seen on TRAIN-H1 / TRAIN-H2 / VAL): on the base-entry C1 book, the
top tercile of BR — the share of the day's gapper universe trading above its 09:30 open at the signal minute — is
positive and above the rest in every period: H1 +0.071 vs −0.174 (n 629 / 1,258), H2 +0.084 vs −0.060 (419 / 1,652),
VAL +0.245 vs +0.034 (839 / 1,450). Every pre-registered median cut failed (cells 1,406–1,408). Because the tercile
pattern was observed on all three periods, only TEST (2026-06-01 .. 2026-09-18) can test it.

## Rule (frozen)
Base entry exactly as cell 1,400 (signal, entry next open, stop = base low, target +2 R, flat 15:55, stop-at-open at
−cost, proxy cost 15 bps half-spread both legs + 2 bps/side, fill walk from the bar after the entry bar). **Keep a
trade iff BR(signal_m) ≥ 0.6115** — the TRAIN-H1 upper-tercile edge of BR over the book's TRAIN-H1 trades. BR is
built for TEST days with the identical `breadth.symbol_minute_flags` (causal: closes ≤ signal_m).

## TEST disclosure
DRIFT.md printed the TEST aggregate of the C1 density-filtered primary arm (net +0.090 R, n 1,467) by reused code.
The BR-conditional split on TEST has never been computed. TEST is used once, here.

## Pass (all, on TEST)
Kept net ≥ +0.10 R; kept − rest ≥ +0.10 R; kept trade-weighted day-clustered t ≥ 2 (MDE stated); kept book after the
slot rule (first 12/day, 4 concurrent) > 0; slotted kept fills ≥ 3/week. If the sign and the separation hold but
t < 2 → "consistent, not proven": an exploration-tier candidate with its MDE, never called an edge.

Programme count 1,412.
