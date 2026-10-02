# RESULT_1701 -- earnings-session ORB pool E1 (walk)

Candidates: 19,199; filled (ok, any side): 4,068. FULL run, source=sip. SIP coverage: 13160/19199 sessions walkable (68.5%) -- gate NOT met -- VOID per Amendment 2.

## Pass bar per side (>=3 fills/wk test half, mean R>=+0.10 both halves day_t>=2.0 pooled, ex-top5%>=0, obtainable>=80%, cadence weekly P10>=-2R)
- long select=A (X3_R) test=B: n=1185 fpw=18.2 meanR=-0.091 day_t=-3.0 ex5=-0.219 P10=-6.70R -- FAIL
- long select=B (X3_R) test=A: n=842 fpw=16.2 meanR=-0.021 day_t=0.5 ex5=-0.149 P10=-6.87R -- FAIL
- short select=A (X3_R) test=B: n=1180 fpw=18.1 meanR=0.037 day_t=2.0 ex5=-0.087 P10=-6.44R -- FAIL
- short select=B (X3_R) test=A: n=861 fpw=16.6 meanR=0.037 day_t=0.3 ex5=-0.097 P10=-5.66R -- FAIL

## Exits x windows (net R, after measured entry half-spread; exit slip is ADV-conditional 35/10bps applied uniformly per fill -- see module docstring caveat)
side | exit | window | n | fills/wk | mean_R | day_t | iid_t | ex_top5 | runner% | P10(R) | worst_wk(R) | maxDD(R)
long | X1_R | A | 842 | 16.19 | -0.022 | 0.54 | -0.66 | -0.127 | 4.5 | -6.96 | -14.75 | 32.98
long | X1_R | B | 1185 | 18.19 | -0.098 | -2.93 | -3.41 | -0.206 | 4.3 | -7.23 | -20.45 | 122.86
long | X1_R | whole | 2027 | 17.28 | -0.067 | -1.76 | -3.03 | -0.172 | 4.4 | -7.23 | -20.45 | 149.54
long | X2_R | A | 842 | 16.19 | -0.027 | 0.06 | -0.73 | -0.184 | 4.5 | -6.86 | -14.78 | 35.23
long | X2_R | B | 1185 | 18.19 | -0.093 | -3.13 | -2.93 | -0.248 | 4.3 | -7.75 | -20.45 | 127.43
long | X2_R | whole | 2027 | 17.28 | -0.066 | -2.29 | -2.72 | -0.221 | 4.4 | -7.66 | -20.45 | 151.54
long | X3_R | A | 842 | 16.19 | -0.021 | 0.49 | -0.61 | -0.149 | 4.5 | -6.87 | -14.77 | 32.16
long | X3_R | B | 1185 | 18.19 | -0.091 | -3.00 | -3.08 | -0.219 | 4.3 | -6.70 | -20.45 | 119.58
long | X3_R | whole | 2027 | 17.28 | -0.062 | -1.86 | -2.75 | -0.189 | 4.4 | -6.94 | -20.45 | 143.49
short | X1_R | A | 861 | 16.56 | 0.034 | 0.38 | 1.01 | -0.072 | 4.3 | -5.45 | -10.74 | 27.04
short | X1_R | B | 1180 | 18.11 | 0.034 | 1.87 | 1.14 | -0.071 | 5.6 | -6.60 | -13.45 | 29.11
short | X1_R | whole | 2041 | 17.40 | 0.034 | 1.66 | 1.53 | -0.071 | 5.0 | -6.16 | -13.45 | 29.11
short | X2_R | A | 861 | 16.56 | 0.035 | 0.13 | 0.91 | -0.130 | 4.3 | -5.86 | -12.24 | 24.42
short | X2_R | B | 1180 | 18.11 | 0.032 | 1.77 | 1.00 | -0.117 | 5.6 | -6.21 | -14.95 | 35.51
short | X2_R | whole | 2041 | 17.40 | 0.033 | 1.47 | 1.35 | -0.122 | 5.0 | -6.17 | -14.95 | 35.51
short | X3_R | A | 861 | 16.56 | 0.037 | 0.28 | 1.04 | -0.097 | 4.3 | -5.66 | -11.49 | 25.73
short | X3_R | B | 1180 | 18.11 | 0.037 | 1.95 | 1.23 | -0.087 | 5.6 | -6.44 | -14.20 | 30.61
short | X3_R | whole | 2041 | 17.40 | 0.037 | 1.68 | 1.61 | -0.091 | 5.0 | -6.16 | -14.20 | 30.61

## Frequency: in-season (6wk after quarter-end) vs off-season (fills/week)
long X1_R: in-season 23.55/wk, off-season 11.91/wk
long X2_R: in-season 23.55/wk, off-season 11.91/wk
long X3_R: in-season 23.55/wk, off-season 11.91/wk
short X1_R: in-season 23.37/wk, off-season 12.29/wk
short X2_R: in-season 23.37/wk, off-season 12.29/wk
short X3_R: in-season 23.37/wk, off-season 12.29/wk

## Gap-band decomposition (feature, not a filter)
long:
  <=-5%: n=0 mean_R=nan
  -5..0%: n=36 mean_R=-0.413
  0..5%: n=1060 mean_R=-0.068
  >=5%: n=931 mean_R=-0.042
short:
  <=-5%: n=967 mean_R=0.063
  -5..0%: n=1074 mean_R=0.014
  0..5%: n=0 mean_R=nan
  >=5%: n=0 mean_R=nan

## Union with the production book (production first, dedup by day+symbol)
long: production=483 +E1=2027 fills/wk=21.40 union=$243,452 P10=-12.80R worst_wk=$-14,037
short: production=483 +E1=2041 fills/wk=21.52 union=$319,091 P10=-11.97R worst_wk=$-12,772

## Source parity (Databento vs bars_sip.db, read-only cross-check)
checked=194 within_0.1%=8.8% max_dev=4.786%

## Caveats
- Exit slip (35/10bps ADV-conditional) is applied to EVERY exit type (stop/lock/target/EOD), not stop-only -- the reused walkers return no exit-reason tag; conservative (overstates cost).
- Selection rule: highest mean_R among X1/X2/X3 on the selection half (n>=10 else max-n exit) -- stated, not hidden; PREREG did not fully specify a tie-break.
- Half B frequency for the last ~19 trading days (panel ends 2026-09-04) is not claimed (Amendment 1).
- "whole" window pools both halves; reading it as a verdict on its own is not allowed per PREREG multiplicity rules.
