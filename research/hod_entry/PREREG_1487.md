# PREREG — cells 1,487–1,488: CONFIRMATION ENTRY — buy only the breaks that hold 15 minutes without withdrawing

FROZEN 2026-09-26 before any number. Programme count on the HOD line: 1,486 → 1,488.

## What was seen (disclosed) and why
On the 9,911 fills of cell 1,438, measured on `bars_fills_1478.db`: 87 % dip back to ≤ level − $0.01 within 15 minutes
after the fill bar (before their exit); those earn −0.35 R; the 13 % that never come back earn +0.67 R (n 1,304). Among
target-hitters 72 % still withdrew first. The extension model's top tercile withdraws 85 % of the time, so the
non-withdrawers cannot be selected at the break. They can be selected 15 minutes later. The cost of waiting is the part
of the move already gone (fast runners that hit +2 R inside 15 minutes are exited, not available), and a worse entry
price than the break. This cell prices that trade honestly.

## Rule (cell 1,487)
For every base fill still OPEN at the end of the 15th RTH minute after the fill bar (its base exit minute > fill_min + 15),
if NO bar in (fill_min, fill_min + 15] has low ≤ level − $0.01: enter LONG at the ask at the open of minute fill_min + 16
(ask = that bar's open + the fill's measured half-spread from `cell_1445_features.csv`, i.e. the spread is paid), stop =
level − $0.01 (the level held 15 minutes and is the support), R″ = entry − stop, target = entry + 2 R″, 15:55 exit.
Path on minute bars with `sip_rebuild.walk_path` semantics (stop-first on a bar touching both, gap-through at the open).
Costs: entry half-spread (paid), stop-limit exit standard (2.9 / 3.2 bps filled + 12 % tail), target = limit, EOD at the
bid (1,443's EOD means). Fills whose R″ < 0.5 % of price are reported but excluded from the primary book (R must exceed
the spread — the desk's own rule). Report: n, the share of base fills eligible, R″ as % of price (median), the share of
runners lost to the 15-minute wait (base target-hitters with exit_m ≤ fill_min + 15), mean net R″, day-clustered t,
ex-top-5 %, fills/week at 12/4, count-matched null (draws from the base fills on the same days, seed 1487), the base
book's net R on the SAME fills (paired), and the confirmation cohort's base net R (the +0.67 R must reproduce on the
eligible subset before the late entry is priced — the calibration line).

## Rule (cell 1,488 — the pyramid keyed on no-withdrawal)
Base entry at 1/3 risk; at the end of minute fill_min + 15, if no dip below level − $0.01 has occurred and the position is
open, add 2/3 at the ask of minute fill_min + 16, move the whole position's stop to level − $0.01; target 2 R from the
ORIGINAL fill for the whole position; book in original-R units, paired vs the base (same fills), stop-limit standard.

## Pass bar (frozen; VAL)
1,487: mean net R″ ≥ +0.15, day-clustered t ≥ 2.5, ex-top-5 % > 0, ≥ 3 fills/week, null percentile ≥ 99, TRAIN-H2 same
sign with t ≥ 1, kept cache-only share within 5 pp of 19.5 %. 1,488: paired ΔR ≥ +0.05 on both holdouts, VAL t ≥ 2.5,
and the book's VAL mean ≥ +0.10 (a lift on a negative book is not a book). TEST read once for the better cell if it passes.

## Independent check and consequences
Rebuild from this prose on `bars_fills_1478.db` (no tape needed): per-fill agreement ≥ 99 % within 0.01 R. Refuters:
look-ahead (the 15-minute window must be measured from the fill bar, the ask must not use a later bar), obtainability
(an active buy at the ask at minute 16 is obtainable; the half-spread used must be the fill instant's, disclosed as a
proxy), statistics (tail dependence, concentration, drop the best 2 days, the runners-lost table). PASS → the live engine
gets a `confirm_minutes` entry mode (rest nothing at the break; buy at the ask after N minutes without a dip) — dry run
5 sessions, then $50 real orders under the 9/25 fixes. FAIL → the long side of this population is closed at the break,
the retest and the confirmation; the report says so with all three numbers.

## Not allowed
Moving the 15 minutes or the $0.01; using the tape to improve the entry; reading TEST for more than one cell; selecting
the cohort on anything after minute fill_min + 15.
