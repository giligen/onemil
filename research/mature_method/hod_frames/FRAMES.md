# FRAMES.md — the standing frame ledger for HOD-break

One line per FRAME ever tried. A *frame* is a way of extracting money from the book — not a cell, not
a knob. 799 cells were spent inside frame F1 before anyone wrote this file down; the point of the
ledger is that the next pass can see what has already been framed and what has not.

Owner 2026-09-19: *"continue and iterate and look at it from scratch every time, with fresh eyes,
using different ways to extract $$$ from this. You won't stop till this happens."*

| # | frame | mechanism | cells | date | result |
|---|---|---|---|---|---|
| **F1** | **Filter the signal minute, shape the exit** — the whole programme through 2026-09-19 (`hod_break` 12 causal cells + `hod_filter_stack` 580 + `hod_preopen_regime` 162 + `hod_losers` 20 + `hod_fresh` 15 + the entry-cost audit) | a causal feature known at the break bar separates good breaks from bad ones; or a better exit keeps more of the good ones | **799** | 2026-09-13 → 09-19 | **DEAD.** 0 of 799 clear the claim bar; 0 clear the live-exploration bar. Best ever: `hod_fresh` C1 (old-HOD × last-5-bar stop × SPY-up) TRAIN +0.100 gross / +0.033 net / +$2,391, VAL +0.123 / +0.050 / +$1,827 at 13.8–16.0 tr/wk, same-signed positive in H1/H2/VAL — the only such cell in 799 — but clustered t +0.54 / +0.66, green weeks BELOW its own null mean on both splits, one week carrying 85 % of the TRAIN year, and net negative ex-top-5 %. Best gross separator in the programme is a NON-causal one (`MFE@10min`, 0.77–0.88 R) worth **+0.012 R** as a rule. Standing findings: ORB's Rule M is REVERSED here, Rule D is structurally inert, the breakeven move is the worst exit in the set, `touch_n >= 5` and `consol_bars >= 8` are the WRONG side as first-break filters, the booked-set cost is **0.061–0.065 R** (not 0.2151), and the +0.43 R / −0.55 R two-cohort split of `bf_zero` §6b is the one unexploited structure F1 never addressed. |
| **F2** | **SHORT the failed break** | the loser's path (+0.45 R at minute 6 → −1.06 R at minute 28) IS a trade on the other side; the long book books the bleed by minute 10 and cannot monetise it, a short entered at the confirmation can | 26 | 2026-09-19 | see `REPORT.md` §2 |
| **F3** | **The noon conditional-mover book** | `bf_zero` §6b: EOD range >= 10 % days are +0.43 R and the rest −0.55 R in EVERY split; EOD range is unobservable at 10:00, but range is monotone within the day, so "range so far >= X % at T" is a causal proxy for the cohort | 9 | 2026-09-19 | see `REPORT.md` §3 |
| **F4** | **Size by regime, frequency preserved** | every gate in 799 cells threw picks away and lost week shape to arithmetic; a sizing rule keeps all ~30 picks/week and moves only dollars. `spy_r5_pct` (SPY 09:30→09:35, known 09:35:00) separates the book +0.388 / +0.189 R | 4 | 2026-09-19 | see `REPORT.md` §4 |

*(F2/F3/F4 above are this pass's Frames 1/2/3; they take the next free ledger numbers.)*

---

## THE QUEUE — frames declared and not yet run (owner 2026-09-19, appended before this pass scored)

Run one heavy job at a time. Each is a FRAME, not a cell: it changes what the book *is*, not a knob
inside it.

| # | frame | mechanism |
|---|---|---|
| **F5** | **The RETEST book** | admission = the **SECOND** HOD break of the day only, after a first break that failed or retraced. The first break's failure IS the filter (it flushes the impatient longs and the stops above the level), and the retest confirms the level held. A new admission rule, not a feature on the existing one — which is why 799 feature cells could never have found it. |
| **F6** | **ABSORPTION at the level** | volume traded within ±0.5 % of the HOD in the bars that FORMED it, as a share of ADV. A heavy shelf at the high = sellers absorbed = supply cleared before the break. This is the plausible mechanism behind the one admission that held on all three eras (`hod_fresh`'s `consol_bars >= 20` control) and it has never been measured directly. Ladder it. |
| **F7** | **entry-minute × SPY-state 2D map** | the two strongest single features in the programme (entry minute, +0.518 separation; D2 `spy_r5_pct`, +0.220) have **never been crossed**. A decile × {SPY up, SPY down} heat map of R, then the best 2–3 cells promoted to rules with the multiplicity of the whole map accounted for. |
| **F8** | **INSTITUTIONAL FOOTPRINT** | cumulative $ volume from 09:30 to the signal minute as a share of 20-day average daily $ volume — distinct from `bar_vol_x`, which was the breakout bar alone, and from `rv_profile`, which is share-volume against a clock. >= 50 % of ADV$ done by 10:00 = the name is being bought by size, not by the tape. Ladder it. |
