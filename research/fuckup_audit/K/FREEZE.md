# Stage K — the written freeze (2026-09-17, before any TEST read)

PREREG.md and PLAN.md §1 both say TEST is read ONCE per stage, after the stage's selection is frozen
in writing. This is that writing.

## What was run, and what it decided

The 20 pre-registered cells (5 families × 2 holds × 2 book sizes) were scored on TRAIN
(2025-01-02..2025-12-31) and VAL (2026-01-01..2026-05-31) only. `build_k.py A` never computes a
TEST statistic: `per_trade_stats` is called for `('TRAIN', 'VAL')` in phase A, and the permutation
runs on those two splits. The TEST column in `K/signal_counts.csv` is a count of SIGNALS, not a
return — no TEST price was used in any decision.

Gate results (`K/phaseA.md`):

- **G1 (TRAIN: mean net > 0, t >= 2.0, >= 5 trades/week): 0 of 20 cells pass.**
  The best TRAIN cell is `K3_h3_n10` at +17.4 bps net per trade, t = +0.51 — a quarter of the
  required t, on 800 trades.
- **G2 (VAL): 0 of 20**, and it is not reachable: G2 is only applied to G1 survivors, of which there
  are none. Scored on VAL alone, the best cell (`K2_h10_n20`, +166.4 bps, t = 1.26) fails the 55%
  green-week test at 56.5% only by clearing it while failing t, and its TRAIN value is −23.0 bps.

## Decision, frozen

**No cell is promoted. TEST (2026-06-01..2026-09-04) is NOT read for Stage K.** Reading it would
be a third look at the same 20 cells with nothing to confirm; PLAN §1 spends the TEST window on
selections that have already earned it, and Stage K produced none.

This freeze also forecloses the follow-ons that would otherwise be tempting on these tables:

1. `K2_h10_n20`'s VAL number is not carried forward. Its TRAIN sign is negative, 100% of its VAL
   mean is in the top 1% of trades (`+166.4` bps -> `−17.9` with the top 1% removed), and its
   search-adjusted p on VAL is 0.670.
2. The diagnostic finding that the declared ranking keys underperform a coin flip (`K/diagnostics_d5.md`)
   is NOT converted into a "use the reverse key" book here. It is a hypothesis for a new
   pre-registration with its own TRAIN/VAL/TEST, not a result of this one.
3. No family is re-cut, no threshold is moved, no hold is added. PREREG said "if a declared cut
   yields < 5 signals/week on TRAIN, report it as such — do not tune", and nothing was tuned.

Signed off before the TEST split was touched in any form.
