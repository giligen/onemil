# frames17 — F52: THE MIRROR SHORT ON A RESTING LIMIT (2026-09-20)

Programme cell count **1,252 → 1,254** (2 declared, 2 scored). No Databento spend. TEST never opened
(`day <= 2026-05-29` scored, guard printed by `score17.py`).

## Verdict

**Neither cell ships. k=0.3% is REFUTED by its own deciding table; k=0.6% clears the deciding table
and the net/halves/green bars but fails the same top-5% concentration kill that killed every cell in
frames16 — reported as a LEAD, not a pass.**

## The two cells (MIR2 signal, `frames16/sw_*.csv`, ETB-only, gate5, price>=\$5, ex-wrapper)

Entry = resting sell limit at `ref x (1+k)`, live 5 min, fill AT THE LIMIT on the first bar whose HIGH
touches it; entry charged **zero**. Exit unchanged from arm3 spec A (stop 2%, bare, `swalk` imported
verbatim), exit charged the measured NBBO half-spread (`frames16/nbbo16.csv` + 512 new legs measured
here, `frames17/nbbo17.csv`, 99.2% quote coverage).

| k | split | n filled | fill % | gross R (t_clust) | net R (t_clust, t_iid) | ex-top-5% | TRAIN halves |
|---|---|---|---|---|---|---|---|
| 0.3% | TRAIN | 782 | 57.2% | +0.193 (+4.22) | **+0.114** (+2.45, +2.77) | **-0.026** | +0.190 / +0.046 |
| 0.3% | VAL   | 513 | 55.3% | +0.225 (+3.54) | **+0.138** (+2.14, +1.95) | **-0.071** | — |
| 0.6% | TRAIN | 470 | 34.4% | +0.205 (+3.46) | **+0.129** (+2.13, +2.35) | **-0.023** | +0.169 / +0.084 |
| 0.6% | VAL   | 295 | 31.8% | +0.412 (+3.17) | **+0.320** (+2.45, +2.70) | **+0.032** | — |

## THE DECIDING TABLE (pre-committed, checked first)

Unfilled counterfactual = frames16 arm3's OWN reacting fill, **NET of its own measured half+half
cost** (entry-leg + exit-leg spread at the reacting entry/exit minutes, score3b.py's S1 contract —
a like-for-like NET comparison, not gross vs net).

| k | split | unfilled n | unfilled NET (t) | unfilled GROSS | filled NET | result |
|---|---|---|---|---|---|---|
| 0.3% | TRAIN | 584 | **+0.144** (+2.99) | +0.315 | +0.114 | **ADVERSE SELECTION** |
| 0.3% | VAL   | 415 | +0.077 (+1.09) | +0.267 | **+0.138** | no adverse selection |
| 0.6% | TRAIN | 896 | +0.127 (+3.06) | +0.307 | **+0.129** | no adv. sel. (margin +0.002) |
| 0.6% | VAL   | 633 | +0.035 (+0.57) | +0.225 | **+0.320** | no adverse selection |

**k=0.3% fails on TRAIN alone → adverse selection confirmed, dead per PREREG §5**, regardless of its
net numbers otherwise clearing +0.10 R. **k=0.6% clears both splits** — its skipped signals are,
net-for-net, no better than its filled ones; the "passive selects the better half" reading from
frames16 §3.4 (a different object — a marketable CAP order below the market) does not reproduce here
at k=0.3%, and only barely (0.002 R, noise) survives at k=0.6%.

## Against the pre-committed pass bar (PREREG §6)

* **Net >= +0.10 R both splits**: both k pass.
* **TRAIN halves same sign**: both k pass.
* **Ex-top-5% positive**: **fails for BOTH k on TRAIN** (-0.026, -0.023) — frames16's kill #3,
  unchanged: the edge concentrates in the same tail that killed every prior cell in this programme.
* **Green% above null p50, >=1 split**: both k pass both splits (0.3%: 64.2/60.4, 73.9/65.2%;
  0.6%: 56.6/54.7, 78.3/69.6%).
* **§5 deciding table**: k=0.3% fails, k=0.6% passes.

k=0.3% is REFUTED on two independent grounds. k=0.6% passes the deciding table this frame exists to
run, but still fails the tail kill — a LEAD to narrow further (does a wider k or a different live
window decouple the edge from its top 5%?), not a ship candidate.

## Fill rate

31.8%–57.2% across k and split — most of the signal population is never touched by a resting order
this close to the market; k=0.6% VAL (n=295 filled) is thin.

## Caveats, written as an adversary would

1. **k=0.6% TRAIN's "no adverse selection" is a 0.002 R margin** — noise, not a finding; VAL's
   +0.285 R gap is the only clean separation, and it is the smaller-n split.
2. **Halts/LULD not modelled** (standing caveat) — irrelevant pre-fill (no position held while the
   order rests) but live once filled, on a name already running.
3. **The reacting counterfactual's own entry-leg NBBO is 97.0-97.6% covered**; the gap falls back to
   the population mean cost (minor optimistic bias, shared with frames16's own method).
4. **The 5-min window and the two k values were fixed by this frame's pre-registration, not swept
   after seeing results** — but also not derived from first principles; window sensitivity untested.
5. **SSR's fill-blocking mechanic is deliberately NOT applied** (a resting limit above market is not
   what Reg SHO 201 restricts) — a reasoned, unverified assumption, not a regulatory citation.
6. **frames16 arm3's short-only caveats carry over unchanged**: ETB-only (7.2% excluded), borrow fee
   assumed 0 intraday, today's borrow-flag snapshot not the trade-date state.

Studies: `frames17/passive_walk.py`, `frames17/nbbo17.py`, `frames17/score17.py`,
`frames17/passive17.csv`, `frames17/cells17.csv`.
