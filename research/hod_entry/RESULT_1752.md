# RESULT 1,752: approach entry before the HOD break, strict stop. KILL (TEST sealed)

Rule frozen 20:16 UTC, before VAL. 12,393 grid cells; 0 net > 0 in both TRAIN halves with t ≥ 2.

**Population.** 19,393 symbol-days: 1735 arms (break-conditioned) plus 1428 PIT gappers (2025-07 on). Bars exist for 77.5 % (gappers 40 %: bars_sip.db keeps range ≥ 5 % days). P(break) is an upper bound.

## Winners vs losers (TRAIN, min3, K20)
- **Volume:** weak. vol5/day-avg AUC 0.53 (t 3.7). cumv/ADV 0.46 (t −0.4); P(break) 0.77 → 0.63 across deciles.
- **Time:** earlier minute raises P(break) (AUC 0.375, t −2.8), not wins (0.43). Minutes since HOD 0.56 (t 0.3). Attempts = 0 on first approaches.
- **Daily MAs:** AUC 0.51–0.52. Coverage 64 %, so VOID.
- **Geometry separates:** distance from the level, BREAK 0.64 (t 9), RETEST 0.72 (t 13).
- **The approach gross is an artefact.** Arm days (break known to come) +0.62 / +0.66 R at R ≥ 1 %; gapper days −0.04 / +0.04 R, net −0.28 / −0.17. Post-freeze TRAIN diagnostic.

## Calibration
P(break) under the rule: TRAIN 0.806, VAL 0.865 (unfiltered 0.78).

## Rule and VAL
APPR2, d 0.5 %, stop min3, K 30, cumv/ADV ≤ 0.125. TRAIN −0.04 R (t −0.1); H1 −0.34, H2 +0.40.

| VAL (22 wk) | Value |
|---|---|
| Trades | 193 (8.8/wk) |
| Net | +0.107 R |
| Day-t | 0.85 |
| Ex-top-5 % | +0.004 |
| Weekly P10 | −2.76 R |
| Green weeks | 59 % (null 32 %) |
| Placebo | −0.63 R |
| Cost ×1.5 | −0.22 R |
| MDE 80 % | 0.35 R |

**KILL:** day-t < 2, TRAIN halves opposite. 93 % of VAL rows are arm days (break-conditioned).

CHILDREN:
1. Fetch the missing gapper bars; score the approach on PIT gappers only, R ≥ 1 % (TRAIN ≈ 0).
2. Audit cells that use arm-day bars before `entry_m`; those days are break-conditioned.
3. Floor R ≥ 1 % on strict-stop designs; below 0.5 % net is −0.6 to −1.9 R.
