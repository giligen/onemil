# RESULT — cell 1,683: the most optimized profitable exit across all 58 rules, week by week on the sealed quarter

PREREG: `research/hod_entry/PREREG_1683.md` (FROZEN). Owner ask: "Get me the most optimized strat across all
rules that is profitable, ignore the long tail. Show me week by week of the past quarter." Code:
`research/hod_entry/1683_optimize.py`, log `1683_optimize.log`. Search on the pooled 2025-07..2026-05 book
(1663 join, r_pct>=1.5%, n=5,506); sealed read on `forward_2026q3/causal_arming_causal.csv` status==fill,
r_pct>=1.5% floored here (2,872 -> 2,859), 2026-06-01..09-04, UNCHANGED rules.

## Week-by-week table FIRST (base vs the optimized candidate = H49, uncapped; $ at $150/fill risk)

| ISO week | fills | base ΣR | opt ΣR | base $ | opt $ | base green | opt green | worst day (opt) |
|---|---|---|---|---|---|---|---|---|
| 2026-W23 | 318 | -7.88 | **+42.57** | -1,182 | **+6,385** | N | **Y** | -12.92 |
| 2026-W24 | 348 | -160.17 | -140.76 | -24,026 | -21,114 | N | N | -96.86 |
| 2026-W25 | 168 | -37.11 | -15.76 | -5,566 | -2,364 | N | N | -32.71 |
| 2026-W26 | 301 | -120.68 | -93.36 | -18,103 | -14,004 | N | N | -35.27 |
| 2026-W27 | 242 | -13.34 | **+12.59** | -2,000 | **+1,888** | N | **Y** | -45.73 |
| 2026-W28 | 191 | -77.42 | -73.60 | -11,613 | -11,039 | N | N | -45.68 |
| 2026-W29 | 165 | -54.08 | -33.21 | -8,113 | -4,982 | N | N | -18.69 |
| 2026-W30 | 201 | -47.08 | -32.77 | -7,061 | -4,916 | N | N | -46.29 |
| 2026-W31 | 152 | -39.99 | -17.07 | -5,998 | -2,560 | N | N | -16.92 |
| 2026-W32 | 313 | +53.21 | **+114.91** | +7,982 | **+17,237** | **Y** | **Y** | -17.24 |
| 2026-W33 | 172 | -17.51 | -0.87 | -2,626 | -130 | N | N | -17.60 |
| 2026-W34 | 101 | -39.28 | -32.80 | -5,892 | -4,920 | N | N | -9.81 |
| 2026-W35 | 96 | -18.98 | -3.84 | -2,848 | -576 | N | N | -9.35 |
| 2026-W36 | 91 | +5.98 | **+15.77** | +897 | **+2,366** | **Y** | **Y** | -7.47 |
| **Total** | **2,859** | **-574.3** | **-258.2** | **-86,149** | **-38,729** | **2/14** | **4/14** | worst week W24 |

Base green weeks 2/14 (14%); optimized (H49) green weeks 4/14 (29%). H49 beats the base rule in **every one of the
14 weeks** (paired ΔR > 0 all 14, see totals) but the underlying book is net-negative in both columns — H49 cuts
the quarter's loss by 55% ($86.1K -> $38.7K), it does not flip it profitable. Full table: `1683_weeks.csv`.

## Step 2: in-sample search (pooled 2025-07..2026-05, n=5,506; objective = mean ΔR, no tail gate)

**Engine-type tally** (mechanical, from the `fn=` wrapper each hid registers under): TRIGGER=23, RESHAPE=12,
PARTIAL=21, BOOK=2 (=58). Joint combination is only mechanically composable for TRIGGER/RESHAPE hids (a shared
per-bar atom can be extracted and combined with a generic precedence walker, `run_joint`, verified against
1681_per_fill.csv on a sample before use — see selfcheck() in the log); PARTIAL/BOOK hids (H17-22,H33,H35,H36,
H39,H40,H42,H46,H47,H49,H51,H52,H54,H55,H57,H58,H48,H56) are each a bespoke multi-leg or cross-fill walk and are
scored **standalone only**. Forward-scoreable pool additionally excludes every model=True hid (H13-17,20,24,32,
34,36,41,43,45,58): the P(+1Rnext15) model needs `g7_k{k}_*` features that do not exist for the forward population
(only F11-F15+ATR14 do) and the P(stop) "model" was never persisted to disk at all (`1670_timing_map.py` has no
`joblib.dump`) — both genuinely unreproducible on new fill_ids, disclosed, not worked around.

**(a) Best single, all 58** (and, tied, best single restricted to the 41 forward-eligible non-model/non-book
hids): **H49** ("half out at +3R (no fixed +2R exit), rest trails MFE-1R") — mean ΔR=+0.0182, iid t=1.74, **day
t=0.47 (NOT significant in-sample)**, ex-top-5%=**-0.073** (in-sample the mean IS tail-driven), 114.7 fills/wk.

**(b) Best joint, ≤3 compatible (TRIGGER/RESHAPE, model=False) members, greedy**: stopped at 2 members —
**H10 + H8** ("climax bar, vol>=3x mean & CLV<=0.5, at any profit" then "5m close below day VWAP while
>=+0.5R"), precedence = that order. mean ΔR=+0.0030, day t=**-0.20 (zero)**, ex5=-0.087. Step 3 found no third
compatible candidate that improved on this (mirrors 1681's own 2-member ceiling).

**(c) One-step threshold grid** on H10+H8: **no hand-built parametrisation exists for H10 or H8** in
`threshold_variants()` (only H1/H4/H7/H25/H26/H50 were hand-built, on the chance one of those was selected) —
logged as ERROR, disclosed; tuned joint = untuned joint unchanged (mean ΔR=+0.0030, identical).

**(d) Under the live cap (first 12/day by fill_min)**: same winners both rounds — single=H49 (mean ΔR=+0.0102,
day t=0.59), joint=H10+H8 (mean ΔR=+0.0036, day t=0.06). Composability pool, full candidate table: `1683_search.csv`
(260 rows: every single + every greedy-step trial, capped and uncapped).

## Step 3/4: sealed quarter (2026-06-01..09-04, 2,859 fills, UNCHANGED rules) — base vs each candidate

Two different numbers matter and are both reported: **mean R** (the strategy's own absolute level — "is it
profitable") and **mean ΔR** (paired vs base — the PREREG decision-rule quantity, "did optimizing help").

| candidate | cap | n | mean R | day t (R) | mean ΔR | day t (ΔR) | ex5 (ΔR) | green wk | $ @ $150 |
|---|---|---|---|---|---|---|---|---|---|
| base | uncapped | 2,859 | -0.2009 | -5.09 | - | - | - | 2/14 | -86,149 |
| base | capped(12/d) | 802 | -0.1084 | -1.76 | - | - | - | 4/14 | -13,037 |
| **H49 (best single)** | uncapped | 2,859 | -0.0903 | -2.64 | **+0.1106** | **+4.67** | **+0.0236** | 4/14 | -38,729 |
| **H49 (best single)** | capped | 802 | +0.0135 | -0.03 | **+0.1218** | **+3.42** | **+0.0371** | 7/14 | +1,619 |
| H10+H8 (best/tuned joint) | uncapped | 2,859 | -0.1324 | -3.84 | +0.0685 | +5.05 | **-0.0309** | 3/14 | -56,786 |
| H10+H8 (best/tuned joint) | capped | 802 | -0.0347 | -0.72 | +0.0737 | +3.18 | **-0.0226** | 6/14 | -4,177 |

MDE (n=2,859, uncapped): ~0.070R. Max drawdown (base) -607R / (H49) -404R over the quarter.

## Adversarial read (CLAUDE.md #7 / "read the report's own red flags")

1. **The book itself stays closed.** Even under the single best exit found across all 58 rules, the HOD-break
   population is still net-negative uncapped (-0.09R, $-38.7K). This does not reopen `docs/CLAUDE_HISTORY.md`'s
   9/26 closure of this population (`dry_run` stays true) — it answers a narrower question (given these entries,
   which exit loses least) not the closed one (does this population have an entry-side edge).
2. **In-sample -> forward divergence is large and should not be over-trusted.** H49's in-sample day t was 0.47
   (not significant) and its in-sample ex5 was *negative* (-0.073, i.e. in-sample the mean was tail-driven); the
   forward day t of 4.67 with *positive* ex5 is a better result than the in-sample search actually predicted.
   H10+H8's in-sample day t was ~0 (0.06 to -0.20, indistinguishable from noise) yet forward it reads day t=+5.05
   — a rule with ~zero in-sample signal should not show a strong forward signal from genuine timing skill; the
   likelier mechanism is that ΔR=rule_R-base_R rewards ANY early exit once the base rule's own outcomes in this
   specific quarter are unusually bad (base mean R=-0.20, matching the already-reported live -0.21R net, cell
   1,438) — exiting sooner than a badly-performing base mechanically produces a positive ΔR without necessarily
   reflecting skillful timing. Treat the forward ΔR magnitude as regime-flattered, not as a validated, stable edge.
3. **H10+H8's forward lift fails the owner's own "ignore the long tail" instruction on inspection**: ex5 is
   *negative* both capped and uncapped — the joint's improvement over base is tail-redistributed, the opposite of
   robust (matches the project's own "Paired-lift tail check" precedent). H49 does NOT have this problem (ex5
   positive both ways) and is simpler (1 rule, not 2) — **H49 alone, not the joint, is the more defensible of the
   two candidates that cleared the decision bar.**
4. **The cap does a lot of the work.** Capped-base ($-13K) already looks much better than uncapped-base ($-86K)
   before any exit optimization — the 12/day cap disproportionately drops the worse, later-in-day fills. H49's
   capped absolute mean (+0.0135R, day t=-0.03) is statistically indistinguishable from zero, i.e. "breakeven,"
   not "profitable" — one quarter's +$1,619 at that t-stat is noise.

## Decision-rule verdict (PREREG, pre-committed)

Rule: "if the optimized strategy's sealed-quarter mean ΔR >= +0.05R with day-clustered t >= 2.0 and >= 3
fills/week under the cap -> exploration-tier PAPER candidate; otherwise the table stands as the answer."

- **H49 (best single) mechanically CLEARS the bar** on both cap variants (ΔR +0.11/+0.12R, day t 4.67/3.42,
  57-204 fills/wk) and, unusually for this programme, ex5 stays positive — the strongest-surviving candidate.
- **H10+H8 (joint/tuned) also mechanically clears the ΔR/t/frequency bar** but fails the ex-top-5% tail check
  the owner explicitly asked to apply ("ignore the long tail") — a lottery-ticket lift, not recommended.
- Per the decision rule's letter, H49's *exit* qualifies for an exploration-tier PAPER instrument -- **applied
  only to the existing dry-run/paper HOD-break ledger, never as a re-opening of the live population** (which
  stays CLOSED/`dry_run: true` per the 9/26 revival verdict; no config, service, or trading/*.py file was touched
  by this cell). Per the owner's plain-language bar ("that is profitable"), nothing in this table is: every row
  is net-negative uncapped, and the one arguably-breakeven cell (H49, capped) carries a t-stat of essentially
  zero. **Both readings are reported; this is an owner call, not a unilateral one** given point 2 above (the
  forward/in-sample divergence means H49's edge is not yet a well-validated, stable number).

## Files
`1683_optimize.py` (engine + search + forward orchestration, additions to `1681_hypotheses.py`:
`load_population_forward()`, parameterised `build_paths()/_persist_paths()`), `1683_optimize.log`,
`1683_search.csv` (260 rows), `1683_forward_per_fill.csv` (11,436 rows: base/best_single/best_joint/tuned_joint x
2,859 fills), `1683_weeks.csv` (14 ISO weeks), `1683_forward_paths.parquet` (forward per-fill bar cache, built
read-only off `bars_sip.db`).
