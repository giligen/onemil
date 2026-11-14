# ORB pick parity: `slot_displaced` reason (2026-11-14)

## The error
`scripts/orb_pick_parity.py` logged ERROR on 2026-11-13 (and 2026-11-11):
```
ORB pick parity 2026-11-13: matched 7/8 | diffs 2 (unexplained 1, universe_live_only 1) | unexplained 1 | NAMES ['BTCZ']
  BTCZ   bt_only   unexplained         ranged live, not in the live final top-K, composite not comparable
  STRO   live_only universe_live_only  no BT feature row for the day (builder/gate dropped it)
```
Facts (journal 2026-11-13 14:35:06 UTC, `es.bt_ranked('2026-11-13')`):
* BT ranked top-8: TNDM SKDD EQPT CABO DAMD KOLD WSE BTCZ (BTCZ comp 0.4325 Q5, 8th).
* Live final top-8 of 11 ranked: TNDM SKDD **STRO** EQPT CABO DAMD KOLD WSE. STRO is a GRACE stand-in ranked from
  its 09:34:57 provisional state (`ORB: GRACE stand-in STRO ranks from its 09:34:57 provisional state`), no real
  range, no BT feature row. Ranking order Q4 → Q5 → Q3 → Q2 → Q1, composite DESC within quintile
  (`self.ranking_order`), so with STRO in the field BTCZ is 9th. Live composites match the BT's to 1e-4.
* Both displaced names were PDR-vetoed (STRO live, BTCZ in the BT) — traded set identical (CABO, DAMD).

The difference is fully determined by the `universe_live_only` diff: one live-only name inside the live top-K
displaces exactly one BT pick to rank K+1. The classifier has no reason for it → `unexplained` → ERROR to the owner.

## The fix (classifier only; engine untouched — the stand-in burning a slot is the documented rule,
`docs/orb_grace_waitlist_fix_RESULT.md`)
1. Add `"slot_displaced"` to `REASONS` in `scripts/orb_pick_parity.py`.
2. In `compare()` (not `_bt_only_reason` alone): after the `live_only` diffs are classified, let
   `n_live_only_in_topk` = number of live-final names that carry a `universe_live_only` reason. Then for each
   `bt_only` name that `_bt_only_reason` would call `unexplained` with the "ranged live, not in the live final
   top-K" detail: compute its LIVE rank in the live scored field using the same key as the engine
   (quintile order `['Q4','Q5','Q3','Q2','Q1']`, then composite DESC) over `live["scored"]` ∪ the live final set;
   if `K < rank ≤ K + n_live_only_in_topk` → reason `slot_displaced`, detail
   `"live rank {rank} > K={K}: displaced by live-only {names} (no BT feature row)"`. Otherwise keep `unexplained`.
   `K` is the live final size (`len(live['final'])`, 8). Do not relax any other branch.
3. `is_error()` must stay ERROR for a real `unexplained`; `slot_displaced` is INFO (explained). Keep the summary
   line format; the reason tally shows `slot_displaced 1`.
4. Tests (TDD, `tests/test_orb_pick_parity.py` — extend; if absent, create with the same fixtures style as the other
   `tests/test_orb_*` files): (a) the 2026-11-13 shape above (live scored dict with the 11 comps/quintiles from the
   journal, live final 8 with STRO, BT ranked 8 with BTCZ) → BTCZ `slot_displaced`, STRO `universe_live_only`,
   `n_unexplained == 0`, `is_error() is False`; (b) a bt_only name ranked 10th live with ONE live-only name →
   stays `unexplained`; (c) two live-only names, bt_only names at ranks 9 and 10 → both `slot_displaced`;
   (d) `log_result` on (a) emits no ERROR. Then re-run the real day: `python3 scripts/orb_pick_parity.py --date
   2026-11-13 --dry-run` if a dry flag exists, else run it and confirm the log line is INFO (the script appends to
   `logs/orb_pick_parity.log` — acceptable). Also run `--date 2026-11-11` and report its reasons.
5. Run `python3 -m pytest tests/test_orb_pick_parity.py tests/test_eod_sections.py -q --ignore=tests/integration`
   (whichever of those exist) — all green.

## Rules
Sonnet, ≤ 40 calls. No git commit, no service restart, no orders, no writes outside `scripts/orb_pick_parity.py`,
`tests/test_orb_pick_parity.py`. Docstrings on every new function. Return ≤ 120 words: the diff summary, test
counts, the re-run lines for 11/13 and 11/11.
