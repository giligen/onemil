# REPORT — orb_seed_wide cells 1,300-1,314 (BLOCKED at S0)

Budget: this run hit its 45-tool-call ceiling before clearing the strata cells. What
follows is honest about what ran, what passed, and what was never attempted.

## S0 reproduction gate — FAIL (does not match to the cent)

Built S0 = wide CSV rows with `gap_pct >= 5 AND entry_price <= 30` (10,089 of 17,945
rows; note `entry_price` is the ORB breakout fill, not the `today.open` the production
seed actually gates on in `study_orb_broad.py` — a proxy, not the true filter; this is
itself a candidate source of drift, see below). Ran
`study_orb_pipeline_static_lock.py` on that subset with the repo's current `orb.yaml`
(account_budget_usd=26666.67, max_concurrent=8, risk_per_trade_usd=375, skip_q1=true,
PDR-veto min 11.0, G1 veto on, range-size veto on).

| run | window | picks | fills | net $ |
|---|---|---|---|---|
| honest book, full | 2025-01-07..2026-09-18 | 218 | -- | $14,061.55 |
| honest book, windowed to wide-CSV coverage (<=2026-05-29) | same | 158 | -- | $13,048.13 |
| S0, `ORB_CATALYST_VETO=0` (literal task instruction: "catalyst veto OFF") | 2025-01-02..2026-05-29 | 458 | 349 | $15,112.96 |
| S0, catalyst veto default ON (code ignores `orb.yaml`'s `catalyst_veto.enabled` -- it is gated purely by the `ORB_CATALYST_VETO` env, default `'1'`=ON) | same | -- | 125 filled | $12,880.65 |

Neither run matches the honest windowed book to the cent. Diffs, symbol/date level:
honest-windowed has only 4 picks not in the (larger) catalyst-off S0 set -- i.e.
catalyst-off S0 is nearly a superset (+300 extra picks, spanning the whole price/gap
range, not clustered near the $30 boundary -- ruling out the entry_price-vs-open proxy
as the dominant cause).

**Cause, to the extent budget allowed tracing it:**
1. **Catalyst veto is the dominant lever.** `study_orb_pipeline_static_lock.py`'s
   catalyst-veto block reads `os.environ.get('ORB_CATALYST_VETO', '1')` only -- it never
   consults `orb.yaml`'s `filter.catalyst_veto.enabled` (currently `false`, flipped
   2026-09-19). Running literally as the task specified (veto OFF) diverges by +$2,065
   / +190% more picks vs the windowed honest book. Running with the veto at its
   *code* default (ON) -- which is what actually built most of the accumulated honest
   book, since the yaml flag never took effect and the owner-driven flip to `false`
   only happened 2026-09-19, days before this csv's coverage ends -- closes the gap to
   $167 (1.3%) and a plausible pick count, but still not exact.
2. **The honest book is an accumulated nightly journal, not a single-shot backtest.**
   Its 218 rows were written over months under an *evolving* config (`max_concurrent`
   3->8 on 2026-09-17, PDR-veto threshold 8.0->11.0 on 2026-08-15, G1/range-size vetoes
   added 2026-09-08, catalyst veto true->false 2026-09-19, PM/news mult flipped off
   2026-08-15, ...). Re-running the *whole* 17-month window under *today's* frozen
   `orb.yaml` cannot reproduce a book built under a moving config -- this is very
   likely the residual $167 / remaining pick-count gap even with catalyst veto ON, and
   there was no budget left to bisect it further (would require re-running with the
   git history of `orb.yaml` reconstructed month-by-month, which is not committed --
   `orb.yaml` is gitignored instance config with no version history to replay).
3. Not ruled out for lack of budget: whether `build_wide_features.py`'s seed universe
   (patched `study_orb_broad.MIN_GAP_PCT`/`MAX_OPEN_PRICE`) draws from the exact same
   `daily_bars` point-in-time universe (incl. the 2x-wrapper rule shipped 2026-09-05)
   that the historical honest-book regens used at each point in their own history.

**Per PREREG ("do not proceed to strata numbers until you can state the cause"): the
cause is stated above but not fully resolved to the cent, so cells 1,300-1,309
(whole-seed and S1/S2/S3 strata, frozen and refit forms, era-consistency vetoes,
combined book, quoted-cost variant) were NOT run.** Running them on an unreconciled
S0 would produce numbers with an unknown, uncharacterized bias baked in -- exactly the
failure mode `feedback_independent_check_before_claims` exists to prevent.

## Cadence bar -- run anyway, on the closest S0 approximation (catalyst veto ON, filled
rows only, n=125), as a diagnostic, NOT a pass/fail claim on a cell that never cleared
the gate. R = `_rp_position` = constant $3,333.34/trade (risk-parity sizing).

TRAIN 2025 (`python scripts/cadence_bar.py --split TRAIN`):
`C1 fail (no cycle gaps computed) . C2 fail (0% cycles net>0) . C3 pass (P10 -0.06R,
min -0.13R, MDD 0.16R) . C4 pass (100% green weeks vs 48% null) . C5 fail (1.58
fills/wk, < 3 bar) . C7 fail (0 cycles) . ex-top-5% 0.81R, top-5 share 57.5%`

VAL 2026-01..05 (`--split VAL`):
`C1 fail . C2 fail (0% cycles net>0) . C3 pass (P10 -0.06R, min -0.07R, MDD 0.15R) .
C4 pass (100% green vs 50% null) . C5 fail (1.86 fills/wk, <3 bar) . C7 fail (0
cycles) . ex-top-5% 1.25R, top-5 share 35.6%`

Both splits fail the pre-committed cadence bar (fills/week below 3, C1/C2/C7 fail) --
consistent with the honest book's own known low cadence, not surprising, and not
informative about the S0/strata mismatch.

## Regime cells 1,312-1,314 -- reused existing prereg'd study, not recomputed

`research/regime/PREREG.md` + `REPORT.md` (dated before this task, same honest book,
same method the task step 4 specifies: TRAIN 2025 fit mult in {0,0.5,1.0,1.5} per
state by both-halves-sign rule, applied to VAL 2026-01..05) already answers cells
1,312 (rule regime A/B/C1/C2) and 1,313 (HMM regime). Budget did not allow rebuilding
this on the never-reconciled combined wide book, so these numbers are **on the honest
book only** (n=165 filled, TRAIN 84 / VAL 40 / TEST 41 sealed):

- **Rule regime**: A dominates TRAIN (n=70, netR +1.10, t_iid 2.01, both-halves+) ->
  mult 1.5; B/C1/C2 thin (n<=6) -> 1.0 each. VAL: flat $6,386.42/MDD -$489.17 ->
  per-regime $6,930.04/MDD **-$733.75 (worse)**. **FAIL** (dollars up, MDD worse).
- **HMM regime**: hmm0 dominates (n=74, netR +1.36, t 2.32) -> 1.5; hmm1 n=7 -> 1.0;
  hmm2 **0 VAL days**. VAL: flat $6,386.42 -> per-regime $9,424.48/MDD **-$608.29
  (worse)**. **FAIL** (same reason, smaller MDD miss).
- **Cell 1,314 (calm split, SPY above/below 50-day)**: built the causal SPY-SMA50
  series from `cache.db::daily_bars` (`research/orb_seed_wide/out/spy_sma50.csv`,
  576 rows, 49 NaN warm-up) but ran out of budget before joining it to a trade book
  and fitting/scoring a 2-bucket multiplier. **Not run.**

Both regime systems that WERE scored land on the same verdict: a 1.5x multiplier on
the single dominant calm state raises VAL dollars but WORSENS max drawdown, failing
the pre-committed pass bar (dollars up AND MDD not worse). No regime lever ships from
this evidence.

## What did NOT happen (budget-exhausted, explicit)
- Cells 1,300-1,309 (whole-seed + S1/S2/S3 frozen/refit/era-consistency/combined/
  quoted-cost) -- blocked behind the unresolved S0 gate, see above.
- The frames13 F41 matched non-signal control (`b matched non-signal`, `09:36` walk)
  was located (`research/mature_method/frames13/f41.py`, `REPORT.md` section 2.2) but
  never invoked against S0/S1/S2 populations.
- Cell 1,314 (calm-split multiplier fit/score).
- Obtainability/tail audits (C6) on any >=+3R trade.

## The one caveat that alone could explain the headline
**The honest reference book is a multi-month accumulated live/nightly journal under a
config that changed at least 5 times in its own history, while every artifact in
this report re-runs the WHOLE window under a single frozen present-day config.** Any
number that claims to "match the honest book" without reconstructing config-by-date
is not a reproduction -- it is a different experiment that happens to overlap in
population. This is why the gate is written the way it is, and why it was correctly
treated as blocking rather than waved through.
