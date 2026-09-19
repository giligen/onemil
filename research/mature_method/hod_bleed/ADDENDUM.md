# hod_bleed — PREREG ADDENDUM (instantiation), committed BEFORE any Part 2 cell was scored

Part 1 (`part1.py` → `part1.txt`, `as_matrix.csv`) is complete and is descriptive only. This file
instantiates the cells PREREG §4 left as rules, by those rules and by nothing else. `cells.py` reads
`addendum.json`; it had not been run when this was committed.

## 1. E1 — the 12 declared (a, s) cells

PREREG §4's E1 selection rule: rank the 24 (a, s) pairs by the **TRAIN** trade-off
`dR = [Σ R saved − Σ R given up] / n` on the shipped B0 book; the top 12 are the decision cells.
All 24 were computed; **all 12 below are positive on TRAIN**, so all 12 slots are used.

| # | a | s | TRAIN dR | (hurt n / R) | (helped n / R) |
|---|---|---|---|---|---|
| 1 | 0.4 | −0.2 | **+0.0224** | 249 / −384.7 | 555 / +422.5 |
| 2 | 0.4 | +0.0 | +0.0205 | 328 / −500.4 | 573 / +535.0 |
| 3 | 0.4 | +0.1 | +0.0203 | 367 / −555.8 | 586 / +590.0 |
| 4 | 0.4 | −0.1 | +0.0183 | 289 / −447.7 | 563 / +478.7 |
| 5 | 0.3 | −0.2 | +0.0176 | 286 / −453.9 | 635 / +483.6 |
| 6 | 0.3 | −0.1 | +0.0143 | 330 / −523.5 | 643 / +547.6 |
| 7 | 0.5 | −0.2 | +0.0141 | 213 / −333.1 | 470 / +356.9 |
| 8 | 0.6 | −0.2 | +0.0114 | 184 / −282.8 | 397 / +302.1 |
| 9 | 0.3 | +0.1 | +0.0113 | 420 / −651.6 | 666 / +670.6 |
| 10 | 0.3 | +0.0 | +0.0111 | 380 / −591.8 | 653 / +610.6 |
| 11 | 0.3 | −0.3 | +0.0109 | 256 / −402.5 | 623 / +420.9 |
| 12 | 0.5 | −0.1 | +0.0096 | 252 / −388.2 | 476 / +404.5 |

**`(a*, s*) = (0.4, −0.2)`** — rank 1 of the TRAIN ranking. This is the arm used by E2 (its two rungs
are ranks 1 and 2), E3, E4 and E5.

## 2. The three selections PREREG §4 left open, fixed here

* **E2's two rungs** = ranks 1 and 2 of the E1 ranking above: `(0.4, −0.2)` and `(0.4, +0.0)`.
* **"the best E4 trigger"** (used by E5) = the E4 cell with the highest **TRAIN fixed-cohort ΔnetR**,
  ties broken by VAL ΔnetR.
* **"the best exit of E1–E6"** (used by E7) = the cell with the highest **TRAIN fixed-cohort ΔnetR**
  over E1–E6, ties broken by VAL ΔnetR.

No other criterion may promote a cell. Cell count is unchanged: 12 + 4 + 4 + 6 + 2 + 3 + 1 = **32**.
