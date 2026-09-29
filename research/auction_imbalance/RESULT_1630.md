# RESULT — cells 1,630–1,632 (closing-auction imbalance): SUSPENDED after the free mechanism read — the pre-committed bar to buy the rest was NOT met

Judged 2026-09-29 05:10 UTC by the main session from `MECHANISM_READ_1630.md` (Sonnet, no purchases; rows in
`mechanism_rows_1630.csv`). Data on disk: 46 usable sessions 2024-07-16..2024-09-18 (all TRAIN), 13,790 symbol-sessions
with a first publication and a reference mid (1.8 % lacked a mid; the 2024-07-03 half-day was empty). Pull stopped at
$20.96 on the owner's spend decision (Amendment 3).

| read (top decile of |I| minus bottom, signed by the imbalance side) | pooled | first 28 sessions (19 with data) | last 28 |
|---|---|---|---|
| closing move vs the pre-publication mid | +5.3 bps, clustered t 2.63, decile means monotone (ρ 0.87) | t 1.46 | t 3.61 |
| next-open reversal (negative = reverts) | −13.9 bps, t −1.60 (ρ −0.61) | t −0.94 | t −1.28 |

## Reading
* The mechanism exists but is small: the close moves with the imbalance by ≈ 5 bps top-vs-bottom decile — the size of
  one auction leg's cost. The fade-INTO-close leg (1,631) has nothing to earn.
* The tradeable part would be the overnight reversal (1,630): ≈ 14 bps gross, t −1.6 on 46 sessions (SE ≈ 9 bps). The
  full 548-session sample would resolve it (SE ≈ 2.5 bps), which is what the remaining ≈ $80 would buy.
* Economics if it held at 14 bps gross: ≈ 6–9 bps net after the auction leg and the touch rule, ≈ 10 events/day at
  $5K → ≈ $30–45/day ≈ $700/month on ≈ $50K deployed overnight across ten names, minus a live imbalance feed (≈ $200/mo).
  About 1 %/month on the deployed capital with overnight gap risk.
* The pre-committed ask bar (Amendment 3: monotone with |t| ≥ 3 on both halves) is not met on either read. The first-
  half shortfall is coverage (19 of 28 sessions), not a sign flip, but the rule is the rule.

## Decision
Not asking the owner for the remaining pull. Cells 1,630–1,632 stay SUSPENDED with the read on record; the sister
1,637–1,639 (opening auction) stays suspended by implication. Re-open only if the closing data arrives for another reason.
