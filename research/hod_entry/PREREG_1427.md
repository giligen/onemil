# PREREG — cell 1,427: the resting-stop FILL as the book, confirmed on the sealed TEST with a consolidated tape

Frozen 2026-09-25 ~16:30 UTC, before any TEST row is read and before any consolidated tick exists. Programme count
1,426 → 1,427. Cell 1,423 (E1) FAILED its frozen bar (fill rate 30–34 % < 70 %; paired ΔR +0.05/+0.09 < +0.10) and
its availability rail (coverage 75 % on a single-venue tape; TRAIN-H2 missingness gap 7.1 pp). That verdict stands.

## The finding that motivates this cell (from TRAIN-H2 + VAL only; TEST untouched)
E1 = a buy-stop-limit resting at level + $0.01 with limit = level × 1.0015, filled at the prevailing ask at the first
print ≥ trigger inside the break bar, else no position. Its FILLS (895 / 1,059) earned +0.262 / +0.384 R net of the
measured half-spread on TRAIN-H2 / VAL, ex-top-5 % +0.30 on VAL, 25–32 fills/week after slots — while the same
population at the next-open entry earns −0.29 / −0.33. The paired entry gain is small (+0.05 / +0.09): the value is
the FILL CONDITION, i.e. the order only fills when the break's first cross meets an ask within 15 bps of the level;
breaks that gap through or trade with a wide ask are never entered. That is a causal, tape-based filter the live
engine can implement literally (it IS the order). Hypothesis: the E1 book is a book.

## Cell 1,427 (one cell, two parts, one pass bar)
* **Independent rebuild, consolidated tape.** A second implementer who has NOT read `research/hod_entry/entry_replay.py`
  rebuilds E1 from this prose using Alpaca SIP historical trades and quotes (`feed='sip'`, the NBBO the live engine
  sees) for every signal's break bar [S−60 s, S) (S = entry_m × 60 ET seconds; entry_m = the entry bar; the break bar
  is the bar before it); level = the running max of closed 1-minute highs before the break bar (the HOD-break spec's
  definition; reproduce from the same minute bars); trigger = level + 0.01; limit = level × 1.0015; fill = the
  prevailing NBBO ask at the first consolidated print ≥ trigger inside the break bar if ask ≤ limit, else no fill; a
  print ≤ stop after the fill inside the break bar = stopped at the stop; then the B0 path rules from
  `research/hod_exit_lab/paths.parquet`; stop, 2 R target from the new entry, 15:55 exit; cost = half the NBBO spread at
  the fill instant on entry, B0's exit cost rule. Agreement with cell 1,423's VAL numbers: E1 mean net R within 0.05 R
  and fill rate within 10 pp — else the discrepancy is investigated before TEST is opened.
* **Confirmation on TEST** (2026-06-01..09-04, 3,521 signals, sealed until now, read ONCE): the same rule, the same
  consolidated tape, no constant changed.

## Pass bar (frozen)
TEST: E1-fill mean net R ≥ +0.10, day-clustered t ≥ 2, ex-top-5 % > 0, coverage ≥ 80 % of TEST signals with a usable
break-bar tape and winner/loser missingness gap ≤ 5 pp, ≥ 3 fills/week at first-12/day 4-concurrent. AND the
consolidated rebuild on VAL within the agreement bands above. Report-only: the same table with limits 5 / 30 bps and
with the ask + 1 tick; the fill rate; the no-fill cohort's B0 outcome (it must be the losing side).

## Consequences (pre-committed)
PASS → the HOD dry run (zero orders) switches its entry to the resting stop-limit (trigger level + 0.01, limit 15 bps)
and logs fills/no-fills for 10 sessions; if the dry book tracks TEST's fill rate and mean R, an exploration-tier live
proposal at $100 risk with the live guardrails on. FAIL → record; the fill-condition idea is closed on this population.

## Not allowed
Reading TEST before the VAL agreement bands are met; changing trigger, limit, cost, stop, target or the bar after any
TEST number exists; excluding signals for any reason other than the tape availability rail.
