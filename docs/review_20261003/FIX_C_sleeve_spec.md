# Fix spec — momentum sleeve, from the independent review (docs/review_20261003/C_sleeve.md), 2026-10-03

Scope: `scripts/momentum_sleeve.py` and `tests/test_momentum_sleeve.py` only. No change to selection, ranking, guard, gate
percentile, sizing or costs (parity with the BT is enforced by the recon fixtures — all existing tests must stay green).

## C1 — completeness gate must REFUSE, not log (`completeness_line`, ~:212; the --submit path ~:600–665)
Today the 11:50 prefetch logs `COMPLETENESS … LOST n` (ERROR only under 50 %) and the 13:45 `--submit --skip-fetch` run
reads the cached panel without knowing what was lost. Required:
1. The prefetch persists the completeness result beside the panel cache: `cache_file(asof)` + `.completeness.json` with
   `{asof, requested, with_asof_bar, ratio, lost: [...]}`.
2. `--submit` (with or without --skip-fetch) reads that file (or the fresh fetch's result) and REFUSES to place orders —
   exit code 2, one ERROR line `MOM REFUSED: completeness …` and the `[MOM]` Telegram line carrying `REFUSED` — when
   (a) ratio < 0.90, or (b) any symbol currently HELD or in the new top-20 is in `lost`, or (c) the file is missing/for
   another asof. Refusal leaves positions as they are (no sells either): a stale book for a week beats a book built on a
   half universe. `--force` does NOT override (a)/(b); it only overrides the "already rebalanced today" guard, as now.
3. The threshold constant `COMPLETENESS_MIN = 0.90` at module top; the docstring and the code agree.

## C2 — persist `last_rebalance` immediately after `execute()` (~:661–677)
The 14:45 UTC cron line is a retry. If the 13:45 run crashes between `execute()` and `state['last_rebalance'] = …`
(ledger/opens/marking code), the retry re-plans against a state that still shows the pre-trade positions. Required: set
`state['last_rebalance']` and save the state file in a `finally`-style step right after `execute()` returns (before the
ledger loop), then continue with sync/ledger/marking. Add a `state['last_rebalance_run']` with the run timestamp.
Test: a monkeypatched `official_opens` that raises → the state on disk still shows today's `last_rebalance` and a second
invocation the same day exits with the "already rebalanced" guard.

## C3 — non-fractionable names (`~:119–122` universe filter; order building ~:388–404)
Selection must NOT change (the BT has no fractionable filter). Instead, at order time: if the asset is not `fractionable`,
submit a whole-share qty = floor(target_notional / price) (skip with one WARNING if that is 0) and record the residual in
the plan line. Test with a fake client whose asset for one symbol has `fractionable=False`.

## C4 — gate file missing/stale (low): keep the fail-open-to-full-size behaviour (it is the plain BT book) but make the
`[MOM]` line say `gate n/a (full size)` / `gate STALE` explicitly and emit ONE WARNING, not silent. Verify it does.

## Done means
`python3 -m pytest -q tests/test_momentum_sleeve.py -x` green (≥ 55 + the new tests), then a dry plan run
`cd /home/ec2-user/onemil && bash scripts/research_run.sh -m 1500M python3 scripts/momentum_sleeve.py --skip-fetch`
(no --submit; prints the plan or the REFUSED line against Friday's cache) with its output pasted into
`docs/review_20261003/FIX_C_result.md` (≤ 40 lines: what changed per item, test count, the dry-run lines).
Never run with --submit. Never edit crontab, .env, config. No git.
