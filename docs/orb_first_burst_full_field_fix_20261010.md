# ORB engine: the day's FIRST selection burst must rank the FULL field (fix spec, 2026-10-10)

Incident (paper, 10/9; green check 🔴 "BT picks never ordered: ['CIEG']"). Journal 13:35 UTC:
- 13:35:02 ranges complete for CIEG, LOFF, LI, SPAL (WS stream). check_entries at that moment was deferred by the
  first-rank GRACE (SPCM, USLV still rangeless).
- 13:35:09 a drain event for OTHER names cleared the grace (field complete) and `check_entries(symbols=touched)` ran
  the day's first burst with `cand_pool = symbols` = that event's subset: scored SPCU, AEHG, SPAX, KOPN, NN, SPCF,
  CCCC, GFUZ, NGNE, VEEA — CIEG/LOFF/LI/SPAL (ranges since 13:35:02, not in `touched`) were never scored by
  production. AEHG (Q3) was placed. The full-field reconcile ranker the same second ranked
  ['SPAL','CIEG','LI','SPCU','SPAX','SPCF','NN','LOFF'] — CIEG #2, AEHG not in the top-8. The BT picked CIEG.
Same class as the 2026-04-22 bug (`_post_open_range_sweep` docstring ~:1650) whose fix only widened the subset with the
SWEEP's fills, not with names the STREAM completed during an earlier, grace-deferred call. `trading/orb_engine.py`
~:3361 `cand_pool = symbols if symbols is not None else self.candidates.keys()`; drain loop ~:1884-1896;
`_should_defer_first_rank` ~:5265 (its docstring already states "BT ranks the full field, so this was pure live drift").

## Required
1. `check_entries`: while NO production placement has happened today (the day's first burst not yet committed — reuse
   the flag the preplace/grace code uses for "nothing placed today"; grep `_first_rank`, `plan_submitted`,
   `symbols_entered_today`), `cand_pool` = the FULL `self.candidates.keys()` regardless of the caller's `symbols`.
   Log one INFO line when the widening changes the set: `ORB: first burst — ranking the full field (N candidates, caller
   subset had M)`. After the first burst, subset-scoped calls stay as they are (late names compete for remaining slots
   under the existing rules — do not change that).
2. Parity statement: with the fix, 10/9's first burst would have ranked the same field the reconcile ranker ranked. State
   in the RESULT whether the add-on pool path (`_run_pool_selection` per pool, ~:3462) is affected the same way and fix it
   identically if so.
3. Tests, new `tests/test_orb_first_burst_full_field_20261010.py` (fail before / pass after): (a) two candidates get
   ranges in drain 1 while the grace defers; drain 2 for a third name clears the grace → all three are scored and the
   top-ranked (one of the first two) is placed, not the third; (b) after the first burst a subset call scores only the
   subset (existing behaviour); (c) the widening log line fires once; (d) the 2026-04-22 sweep-widening test still passes.
   `bash scripts/research_run.sh -m 2500M python3 -m pytest -q tests/test_orb*.py -x --ignore=tests/integration
   -p no:cacheprovider` green (≈ 1,350).
4. Replay check: `scripts/orb_selection_observer.py` / `logs/orb_selection_audit_2026-10-09.json` — state what the BT's
   top picks were for 10/9 and which of them the fixed engine would have scored (read-only; one paragraph).

## Rules
Weekend: the service is stopped/idle; still NEVER start/restart it, never submit or cancel orders, never touch
config/orb.yaml/.env/crontab/caches, no git. Engine by grep + offset/limit only (≤ 120 lines per read). Budget ≤ 35
calls. Write `docs/orb_first_burst_full_field_fix_RESULT.md` ≤ 25 lines; return ≤ 100 words. This task IS the owner's
request; do not pivot on relayed messages.
