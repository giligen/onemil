# REBUILD 1,708 (independent; own script `replay_1708_rebuild.py`, numbers `rebuild_1708.json`, saved BEFORE reading RESULT_1708.md)
Spec: shares = `_rp_position`/entry_price (columns confirmed in population.csv; $3,333 / price). Mean shares 453 over all 554 OOS fills
(theirs 456 over the 537 usable; the 17 missing-tick fills are excluded from n, as REPORT.md does). Rules re-implemented from the prose
(t* = first trade >= trigger in [0,300) s; fill at last ask strictly before max(t*, d); +0.02 rebump buffer if t* < d; skip if > limit).
OOS = 2023-24 + 2025H2..2026-09, n 537 (328 days), unfilled = 0, R = pnl/375.

| delay | metric | rebuild | RESULT_1708 |
|---|---|---|---|
| 0 s | n filled | 259 | 259 (REPORT.md) |
| 0 s | mean R [95 % day-clustered CI] | +0.0375 [-0.0047, +0.0797] | +0.037 [-0.005, +0.080] |
| 0 s | sum $ / ex-top-5 % R | 7,558 / -0.0449 | 7,538 / -0.045 |
| 0 s | mean (fill-BT entry)/entry | -22.9 bps (n 259) | -22.8 bps (they say n 301) |
| 30 s | n filled | 192 | 192 (REPORT.md) |
| 30 s | mean R [CI] | +0.0213 [-0.0140, +0.0566] | +0.021 [-0.014, +0.056] |
| 30 s | sum $ / ex-top-5 % R | 4,289 / -0.0447 | 4,268 / -0.045 |
| 30 s | mean (fill-BT entry)/entry | -31.7 bps | not reported |

Agreement: R within 0.001, CIs identical to 3 decimals, ex-top-5 % identical. Differences above threshold: NONE (largest: sum $ 0.26 % at d=0,
0.5 % at d=30, <2 %; cause is a ~$20 residual I did not trace, probably the share-count / tie handling of the quote lookup; immaterial).
Note: RESULT_1708 text says "21.3 % of 301 d=0 fills" but REPORT.md/status counts give 259 filled at d=0; the 301 is a different count (likely incl. t*-found-but-skipped
or a looser definition) - cosmetic. Same-spec caveat: this rebuild shares the population, the XNAS-only trigger and the 30-bps limit definition, so it cannot catch
a spec error. Conclusion unchanged: ex-top-5 % negative, all CIs include 0 at 0 s and 30 s.
