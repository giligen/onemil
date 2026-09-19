# FREEZE — bf_sizing TEST seal

TEST = **2026-06-01 → 2026-08-31**. Per `PREREG.md` §2 it is scored **once**, after the
recommendation in `REPORT.md` §6 is committed, and only for the cells named here.

## Sealed recommendation

Committed in `REPORT.md` §6 at commit **<RECOMMIT>** (`research/bf_sizing/REPORT.md`,
sections 0–6 complete, section 7 empty).

**Verdict: (b) — the conviction score is NOISE as a sizer.**

**Cells named for the TEST reveal** (both pick sets, P1 and F7):

| cell | why it is revealed |
|---|---|
| **S0** | the shipped baseline — must be revealed for any comparison to mean anything |
| **S1** | the recommendation (flat multiplier; conviction + MACD-zone sizing off) |
| **S3** | the named next candidate (volatility-normalised) |
| **S2** | the anti-predictive diagnostic — revealed so the (c) branch can be checked once on unseen data |

S4, S5, S6 and S1b are **not** part of the recommendation and are revealed in the same
pass only because `part2.py --reveal-test` prints the whole declared grid; **they are not
re-ranked and nothing in §6 may be rewritten after the reveal.**

## Rules of the reveal

1. Run once: `python3 research/bf_sizing/part2.py --reveal-test`.
2. `REPORT.md` §7 records the result verbatim, including the parts that contradict §6.
3. §0–§6 are **not** edited after the reveal. If TEST disagrees, that disagreement is
   reported in §7 and carried to the owner as-is.
4. TEST holds **7 P1 trades and 42 F7 trades**. The P1 arm is vacuous by construction and
   is labelled so. `bf_frequency` §12 already revealed the same quarter for P1 and F7
   selection (P1 +$52 / 7 trades; F7 −$28,284 / 42 trades, R/pick +0.004) — so the
   *selection* in this quarter is already public knowledge and the only new information
   here is what **sizing** does to it.
