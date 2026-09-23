# Find the winners — cells 1,406–1,412 + post-hoc diagnostics (2026-09-23)

## Where the winners are
The base entry (buy inside the base just under the high of day) on **risk-on tape**: at the signal minute, at least
61.15 % of that day's gapper universe (gap ≥ 3 %, $3–50, prior volume ≥ 500K) trades above its 09:30 open.
Median cuts failed (1,406–1,408); the top-tercile rule was frozen from 2025-H1 and scored once on the sealed
quarter (1,412, `TEST_1412.md`). Diagnostics: `KEPT_DIAG.md` (post-hoc, disclosed, no rule selected).

| period | kept trades | kept mean R | other minutes | random long, same stock, risk-on minutes | setup − random |
|---|---|---|---|---|---|
| 2025 | 1,048 | +0.076 | −0.109 | −0.038 | **+0.115** |
| 2026 Jan–May | 839 | +0.245 | +0.034 | +0.143 | **+0.102** |
| 2026 Jun–Sep (sealed until tonight) | 557 | +0.321 (t 3.1) | −0.080 | +0.168 | **+0.152** |

The setup's own increment over a random long on the same tape is +0.10 to +0.15 R and has the same sign and size in
all three periods, including the one never touched before tonight. That is the winner-selection signal.

## Why it is not a book yet (cell 1,412 fails its frozen bar)
* **Most of the R is the tape, and the tape comes in bunches.** Random longs on risk-on minutes swing −0.04 → +0.17 R
  by period. The kept book is a basket of correlated longs on a few days: 2025 worst day −83 R, worst week −76 R;
  2026 Jan–May best day +174 R = 85 % of that period's R. Green weeks 43–53 %. The cadence bar fails everywhere.
* **The live-style cap picks the wrong ones.** First 4 at a time / 12 a day takes each day's EARLIEST signals: on
  Jun–Sep they lost (−0.03 R, n 107) while the later ones won (+0.40 R, n 450); Jan–May +0.13 vs +0.27; 2025 flat
  (+0.06 vs +0.08). The slotted TEST book is −0.04 R.
* The 15 bps cost is a proxy; measured NBBO is required before any claim.

## What stands, and what is next
* Every historical split of this population has now been used. Any refinement can only be validated FORWARD.
* **Forward ledger** (`FORWARD_SPEC.md`): the frozen rule 1,412 applied each morning to the previous session with the
  identical code, all signals + the capped variants + the risk-on placebo + MEASURED NBBO cost, appended to a ledger.
  Clean out-of-sample evidence from day one, zero live risk, no engine change.
* Exploration only, no claim: hedging the tape component (IWM) to isolate the stable +0.1–0.15 R increment.
Programme count 1,412.

## Exploration (no claim): an IWM hedge does not isolate the increment (`iwm_hedge.py`)
Trade R vs IWM R over the same holding window: correlation +0.14 / +0.36 / +0.19 (TRAIN / VAL / TEST), TRAIN beta 0.42.
Hedged: day std 13.1→12.0 / 26.7→23.9 / 17.1→15.3 R, worst day −82.6→−69.7 R, green weeks unchanged. The tape these
gappers ride is gapper-specific (the speculative small-cap corner), not the index; no liquid instrument hedges it,
and a short-gapper hedge would cost more than the +0.1–0.15 R increment it isolates. The forward ledger keeps
measuring; this population is not a book on bars alone.
