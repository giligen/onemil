# TEST SEAL — hod_filter_stack

**TEST (2026-06-01 → 2026-09-11) was NEVER OPENED by this study.**

PREREG §7 allows TEST to be read exactly once, and only after a cell clears the claim bar G1
(TRAIN mean net R > 0 with t >= 2.0, >= 10 trades/week, and selected-subset TRAIN gross >= +0.25 R)
and this file is committed carrying the recommendation.

**0 of 25 declared cells cleared G1.** Every TRAIN mean net R is negative, and no cell's TRAIN gross
exceeds +0.049 R against the +0.25 R the bar asks and the +0.2151 R measured cost the book must
clear. G2 was therefore never evaluated and `score2.py --test` was never run (it also refuses
without this file).

**Recommendation committed: STAY DRY AS INSTRUMENT.** No config change. `config.yaml hod_break`
remains `enabled: true, dry_run: true` exactly as the owner set it on 2026-09-19.

TEST remains sealed for the next pre-registration on this book.
