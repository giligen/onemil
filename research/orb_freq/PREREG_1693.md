# PREREG — cell 1,693: every ORB sub-pool × its OWN exit — the test the owner's design calls for (FROZEN 2026-10-01 13:20 UTC)

Owner 10/1 13:15 UTC: "we need increased frequency on additional sub-pools, each with its criteria/filter and its own
exit strategy." Admitted gap: cells 1,684/1,685/1,689a read the pools with the PRODUCTION exit only (the exit menu was
dropped for budget and I accepted it); the own-exit half was run on P1 alone (1,690/1,692), where a different exit
changed +0.05 R into +0.27 R on 2026. This cell runs criteria × exit for every pool, both directions.

## Pools (existing per-fill books; the fills' paths from the completed bar store)
1,684: idea1 (= P1), idea2, idea10, idea11; 1,685: the 11 scored sub-pools (A+F6, B+F1/F3/F4/F5/F6, C+F1/F3/F4/F5/F6
— as named in 1685_pool_books.csv); 1,689a: slice 19', 19, 20; plus production as the reference. Also build the two
missing seeds and score them with the same grid: the 2–3 % gap band × F1/F3/F4/F5/F6 (minute bars through the
appender; `df -h /` ≥ 5 GB first, report), and F2 pre-market dollar volume for bands B/C from the appended 04:00–09:30
bars (coverage reported; VOID under the rail).

## Exit menu (fixed, 12; every exit executes on the stored bars with the 1,679 walker, R-floored at 0.5 % of price)
 E1 production (target 2 R, lock at +1.75 R → +0.5 R, 15:45 close)   E2 target 1 R   E3 target 1.5 R   E4 target 3 R
 E5 50 % at +1 R, rest E1   E6 50 % at +1.5 R, rest E1   E7 breakeven lock at +1 R   E8 lock +0.5 R at +1 R
 E9 trail MFE − 1 R once ≥ +1 R   E10 trail = the trailing 10-min candle low once ≥ +1 R   E11 time stop 60 min if < +0.5 R
 E12 power-hour hold: from 15:00 ET replace the target by the 15:45 close for fills ≥ +1 R

## Protocol
For each pool: the 12 exits read on TRAIN (2025) and VAL (2026-01..09-18) and 2024H2; direction A: select on 2025
(mean R ≥ +0.05, day-clustered t ≥ 1.5), test on 2026; direction B: select on 2026, test on 2025. Classify each
(pool, exit): robust / regime-specific / fails. Reads: n, fills/week, mean R, iid and day-clustered t, ex-top-5 %,
MDE, weekly P10, worst week. Then the UNION: production + every robust (pool, exit) pair and, separately, + the
regime-specific pairs; union mean R, frequency, weekly P10 in R and per fill, strong-week median gap via
scripts/cadence_bar.py, green weeks vs the count-matched null, the shared worst day. Overlaps de-duplicated.

## Pass bar
A (pool, exit) pair joins the union only if it passes at least direction A (robust or regime-specific), ex-top-5 % > 0
on its test window, ≥ 1 fill/week, and the union's weekly P10 per fill and strong-week gap do not worsen. Robust pairs
→ independent rebuild → paper as pools with their exits (each pool's exit via the per-pool exit overrides shipped
10/1). Regime-specific pairs → paper at minimum size, forward read 40 fills.

## Multiplicity
≈ 21 pools × 12 exits × 2 directions ≈ 500 reads, stated; the both-directions classification and the tail clause
are the protection; expected false "robust" ≈ 1 — every robust pair is rebuilt before the owner sees it as a claim.

## Output
`research/orb_freq/RESULT_1693.md` (≤ 200 lines: per pool the best exit per direction with both reads, the robust
and regime-specific lists, the union tables), `1693_reads.csv`, `1693_union.csv`, `1693_pool_exits.py`,
`1693_pool_exits.log`. The agent returns ≤ 200 words.
