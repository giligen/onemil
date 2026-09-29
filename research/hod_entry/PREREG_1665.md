# PREREG — cell 1,665: relative volume to the arm minute vs the prior N sessions (FROZEN 2026-09-29 17:46 UTC)

Owner's cut, verbatim: "how come 1663 doesn't look at average volume till that point compared to previous X days?" and
"relative volume till specific TOD sounds to me like a strong feature". Cell 1,663 pre-declared seven cuts on the
cost-structure axis (ATR %, stop/ATR, time of day, target distance, exit type, price, interaction) and left volume out;
that was my choice, not a finding. Relative volume was never tested in this causal, time-of-day-matched form on the
1,438 population; the 9/18 causal-filter cells and flags 1,445–1,455 used same-day bar-volume ratios under the stale
35-bps cost model. This cell is frozen BEFORE any number is read.

## Population
* `research/hod_entry/fills_1658.csv` (9,911 fills) joined 1:1 to `causal_arming_causal.csv` (status == fill) exactly as
  cell 1,663 did (`1663_features.csv` already carries the join, the ATR and the stop bucket — reuse it, do not re-derive).
* Primary book: stop ≥ 1.5 % (n 5,506). The unfloored 9,911 are reported beside it, never instead of it.
* Halves: the `split` column (TRAIN / VAL) as in 1,658. Cost: the `net_R` column (measured 7 bps entry, 6 bps stop).

## Feature — two definitions, both pre-declared, A is primary
Let CV(d, m) = Σ v of the 1-min bars of session d with 09:30 ≤ bar start and bar END ≤ m (bars that CLOSED before the
instant m; the fractional-minute rule — never the bar that contains m). Source: `research/bf_zero/bars_sip.db`, table
`bars(symbol, day, t, o, h, l, c, v)`, PK (symbol, day, t).

* Reference minute m_arm = the START of the bar that set the day's high-of-day level (`level` in the join: the first bar
  of the day whose high equals `level` at the fill's arm; if the store's bar highs do not reproduce `level` within 1 ct,
  the fill is MISSING on this definition and counted in the coverage line). The engine arms on the bar AFTER the level
  bar, so everything through the level bar is known at the arm. The fill bar is never used.
* **A (primary): RVOL_A(N) = CV(d, m_arm) / mean over the prior N sessions k of CV(d−k, m_arm)**, sessions taken from
  the store's own days for that symbol (a prior session with no bars before m_arm counts as missing, not zero). N = 20
  primary, N = 5 secondary. A symbol with fewer than 10 (N=20) or 3 (N=5) prior sessions in the store is MISSING.
* **B (fallback, only if A fails the availability rail): RVOL_B = CV(d, m_arm) / (ADV20(d−1) × P(m_arm))**, ADV20 from
  the daily panel cell 1,663 used for ATR (`research/overnight_high/panel_2024_2026.parquet`, prior sessions only) and
  P(m) = the cross-sectional median of CV(d, m)/day-volume(d) by minute, estimated on TRAIN days only.
* Availability rail (CLAUDE.md §7): coverage ≥ 80 % of the floored book and winner/loser missingness gap ≤ 5 pp, else
  that definition is VOID and the other is reported; if both are VOID the cell is VOID and says why.

## Reads (all pre-declared; every t carries the MDE and a day-clustered SE beside iid)
1. Terciles and quintiles of RVOL_A(20) on the floored book: n, mean net R, t, ex-top-5 %, fills/week, per half.
2. Top tercile vs the rest, paired as a cut of the same book: ΔR, t, ex-top-5 % ΔR, per half.
3. Interaction: stop bucket (1.5–3 %, ≥ 3 %) × RVOL_A(20) tercile, per half.
4. Monotonicity: Spearman ρ of RVOL_A(20) vs net R, per half.
5. Reads 1–4 repeated for RVOL_A(5); read 1 for B only if A is VOID.
6. Coverage line: fills with the feature / floored book, winner vs loser missingness, symbols with < N prior sessions.

## Pass bar (identical to PREREG_1662 §1,664)
A cut ships to paper only if net ≥ +0.05 R AND t ≥ 2.5 in BOTH halves AND ex-top-5 % > 0 in both AND ≥ 3 fills/week at
the live config. A read that passes is not reported to the owner before an independent rebuild of the feature from this
prose (a second agent that has not seen the first script) agrees row-level on ≥ 95 % of fills. A null is a claim about
this test: the RESULT states the MDE at n and the coverage before any "no lift" sentence.

## Multiplicity
2 definitions × 2 N × (3 + 5 + 2 + 6 + 1) ≈ 68 reads. Programme count on the HOD line after 1,663: ≥ 1,700 cells.

## Not allowed
Choosing N or the definition after seeing numbers; any conditioning on exit type; the fill bar's volume in the primary;
same-day ratios (bar volume / same-day mean) — that is the already-tested feature, not this one; refitting the stop floor.

## Output
`research/hod_entry/RESULT_1665.md`, `1665_features.csv` (fill_id, day, symbol, split, r_pct, rvol_a20, rvol_a5,
rvol_b, missing flags), `1665_reads.csv` (one row per read, both halves), `1665_rvol.py`. The agent returns ≤ 150 words.
