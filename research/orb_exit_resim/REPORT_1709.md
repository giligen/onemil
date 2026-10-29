# REPORT 1,709 — ORB exits re-simulated at the live fill price (2026-10-29). Verdict: ORB SHELVED.

Build `RESULT_1709.md` (resim_1709.py) and independent rebuild `REBUILD_1709.md` (rebuild_1709.py, prose-only), compared
row by row on (date, symbol), delay 0 s, 246 scorable replay fills + 121/123 live fills.

## Agreement
* 1,709a stop shares: ledger 39.0 / 38.0 %; BT-entry re-sim 34.1 / 37.2 %; live-fill re-sim 35.0 / 33.9 %. Exit-class
  agreement with the ledger 81 % in both. Build and rebuild differ by ≤ 3.3 pp on share (bar 2 pp — marginal, same sign).
* 1,709b per-fill R by the PREREG definition (pnl / shares×(entry−stop)): build +0.390, rebuild +0.425, corr 0.97, 90 % of
  rows within 0.05 R, exit reason identical on 95 % (234/246). **Agreement bar met on the PREREG's R.** The build's
  headline "+0.14 R" was pnl/$375 (the BT dollar-risk convention) — a definitional difference, not a disagreement;
  under that dollar convention the rebuild gives +0.354. Both conventions are reported below.

## Numbers (both builds; the dollar convention is the money view)
| | per-fill R (PREREG) | pnl/$375 | ex-top-5 % | fills/wk |
|---|---|---|---|---|
| in-regime 2025-01..26-09 | +0.39 / +0.43 [+0.09, +0.73] | +0.14 / +0.35 | −0.01 / −0.02 | 2.4 |
| 2024H2 (34 fills) | — / +0.58 [−0.20, +1.67] | +0.13 | −0.07 / −0.03 | 1.5 |
| 2023-01..24H1 | **VOID** — no 1-min bars on disk, 0 % covered | | | |

## Pass bar (frozen) — line by line
H1 (mechanism): live-fill re-sim explains 8 % (build) / −34 % (rebuild) of the stop excess — **FAIL** (bar ≥ 50 %);
exit-reason agreement 81 % — pass. ⇒ The 10–13 pp live stop excess is NOT the entry price. Neither re-sim reproduces it
from bars, so it lives in the live engine's tick-level exit behaviour (not modelled here) or in noise (n 123, SE ≈ 4 pp).
H2 (money): out-of-regime CI lower bound > 0 — **VOID/FAIL** (only 2024H2 has bars, CI spans −0.20..+1.67);
same sign in TRAIN halves — pass (+0.26/+0.11 build; +0.66/+0.34 rebuild); ex-top-5 % > 0 — **FAIL in both builds
under both conventions**; ≥ 3 replay-filled fills/week — **FAIL** (2.4 in regime, 1.5 out).

## Decision (pre-committed in PREREG_1709.md; executed today rather than 11/13 — the evidence is complete)
ORB is shelved: `orb.yaml strategy.enabled: false` from the 2026-10-30 boot, code frozen, no further ORB cells. Reasons
in one line: the only positive numbers are tail-carried (ex-top-5 % ≤ 0 in every build and convention), the book fires
under 3 times a week, the untouched out-of-sample window cannot be tested with the data on disk, the live record is
−$5,281 on 123 fills, and the engine produced four execution defects in ten days. Cell count: 1,709 (+ 2 builds, 3 conventions).

## Caveats
Add-on pools and the 40 % @ 3R scale-out not simulated; ATR floor applied only where stored (live stops equal range_low
on 96 % of rows — the floor rarely binds); force_close at 15:45 ET per the ledger (PREREG said 15:55; sensitivity ≤ 0.02 R);
intrabar order unknown (stop assumed before target).
