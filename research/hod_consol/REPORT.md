# Base-under-the-high (cells 1,400–1,402) — VERDICT: FAIL. The owner's question, answered

Run 2026-09-23 19:10–19:41 UTC under the owner's override of the market-hours blackout (nice 19 + ionice idle;
0 engine tick timeouts during the run). Universe 23,767 causal symbol-days, RTH bars 99.2 %, 8,076 signals
(TRAIN 3,974 / VAL 2,301). Cost = 15 bps half-spread both legs + 2 bps/side (proxy, as pre-registered).
Artifacts: `DECOMP.md`, `DRIFT.md`, `CELLS.md`, `cells.json`, `trades/`, `adversarial_read.py`.

## The owner's question — "why not enter an R earlier?" (DECOMP.md, descriptive)
| | 2025 | 2026 |
|---|---|---|
| base entries whose stock later breaks the high | 76 % | 82 % |
| C1 net R, bases that break | +0.25 | +0.39 |
| C1 net R, bases that never break | −1.06 | −1.13 |
| C1 net R, all bases | −0.06 | +0.11 |
| same breakers: base entry vs break entry | +0.25 vs −0.00 | +0.38 vs +0.07 |

Entering at the base beats entering at the break by +0.25 / +0.32 R on the stocks that go on to break. The one in
five bases that never break costs ~1.1 R each and takes it back.

## Cells (trade-weighted mean; t = day-clustered, trade-weighted; SLOTTED = first 12/day, 4 concurrent = the live book)
| cell | TRAIN all | TRAIN slotted | VAL all (t) | VAL slotted (t) | D1 / D3 on VAL | verdict |
|---|---|---|---|---|---|---|
| 1,400 C1 +2 R target | −0.060 | −0.020 | +0.111 (1.27) | +0.045 (0.85) | +0.080 / +0.015 | FAIL |
| 1,401 C2 no target | −0.015 | −0.002 | +0.096 (0.81) | +0.071 (0.89) | +0.080 / +0.046 | FAIL |
| 1,402 C3 breakeven lock | −0.037 | −0.014 | +0.074 (0.68) | +0.051 (0.77) | +0.075 / +0.021 | FAIL |

Every cell fails: TRAIN below zero, VAL t below 1.3, and the D1 margin is ≤ +0.03 R — the base condition adds almost
nothing over a long at a random midday minute on the same stock. VAL's positive trade-weighted mean is carried by
its busiest ~10 days (top 10 % of days = 168–284 % of VAL R); its day-weighted mean is ≈ 0.

## Adequacy
Cluster SE ≈ 0.05 R (TRAIN) / 0.09 R (VAL): a +0.15 R net edge would have shown on TRAIN, which is negative.
Excluded: a ≥ +0.15 R net edge for THIS base definition with these three exits on 2025.

## Disclosures (read as an adversary)
1. **TEST was printed** in DRIFT.md's primary-arm table (C1, net +0.090 R, n 1,467) by reused builder code,
   against the PREREG. TEST is consumed for that descriptive C1 arm; it stays sealed for every new cell.
2. The scorer's `val_t` weights days equally; the trade-weighted cluster-robust t above is from `adversarial_read.py`.
3. DRIFT's 18.7 pp winner/loser missingness flag is mechanical (its conservative arm books a missing minute as a
   stop); the cells walk every signal. The 15 bps proxy likely understates cost on the 16 % thin-bar names, so
   reality is no better than these numbers.
4. Placebos were computed on VAL only (the PREREG asked for both splits).
5. The fill walk starts at the bar AFTER the entry bar (the convention of every HOD study), so a stop hit inside the
   entry minute is not seen — mildly optimistic, equal across cells and placebos.

## What 1 + 1 gives
HOD lab: after a break, the path is a random walk — the exit cannot add information. DECOMP: the base entry makes
+0.3 R on breakers and loses 1.1 R on non-breakers. So the one lever left on this entry is to recognise the
non-breakers by the clock and leave before the stop: exit if no close above the high within N minutes.
Pre-registered as cells 1,403–1,405 (`PREREG_TIMESTOP.md`). Programme count 1,402 → 1,405.

## Cells 1,403–1,405 — no-break time stop: FAIL, worse than no time stop (TIMESTOP.md)
| N | TRAIN slotted | VAL slotted | vs C1 slotted (−0.020 / +0.045) |
|---|---|---|---|
| 15 min | −0.055 | −0.052 | worse both |
| 30 min | −0.069 | +0.015 | worse both |
| 60 min | −0.049 | +0.048 | worse TRAIN, flat VAL |
Why: "no break yet" does not identify the failures. 78 % of bases break eventually, only 56 % of those within 15 min;
of the bases with no break after 15 min, 61 % still break later (52 % after 30 min). Exiting them early gives up
later breakers and pays the spread. Programme count 1,405.

**Synthesis across ORB, HOD-break and this study:** entry timing and exit rules do not separate winners from losers
in these gappers; the result concentrates on a few busy, market-wide days (top 10 % of days ≥ 100 % of the R). The
remaining lever is DAY selection with a breadth measure knowable at the entry minute.
