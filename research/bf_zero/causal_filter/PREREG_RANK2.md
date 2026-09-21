# PREREG — HOD-break attention rank, pass 2 (cells 1,353–1,354). Adequacy fix of PREREG_RANK.md, same day

Pass 1 (`RANK_REPORT.md`) was an inadequate test, not a refutation: (a) `rank_cand` ranked a signal among ALL
`candidates_full.csv` rows that day (thousands of unfired setups), so "≤ 10" kept 22 TRAIN signals in 12,135 —
the wrong population; (b) `rank_mkt` was VOID on 100 % of days because `bars_sip.db` holds only the candidate
symbols, never the day's full universe. Both are spec errors by the author (main session), disclosed here before
any re-scoring. The two cells below replace them; thresholds unchanged (≤ 10), features still causal.

* **1,353 rank_sig** — rank of the signal by `dist_open_pct` among the FIRED signals of `features.csv` on the same
  `day` with `entry_m` ≤ the signal's `entry_m` (what has actually broken out so far today). ~35 signals/day, so
  ≤ 10 is roughly the top third.
* **1,354 rank_active** — rank of the signal's move-from-open at its minute among all symbols present in
  `bars_sip.db` on that day (today's active setup universe; coverage is 100 % by construction), using bars up to
  and including the signal minute. This is the "top gainers among today's movers" list, not the whole market.

Pass bar, diagnostics (≤ 5, ≤ 20, decile table), TRAIN halves, dropped-cohort sign, C4/C5 and the forward check:
exactly as in `PREREG_RANK.md`. Implementation: `rank_cells.py` with the two population changes only, output
`RANK2_REPORT.md`. Programme count 1,354. No further rank definitions after this pass.
