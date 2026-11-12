# RESULT 1,736 — short the failed touch of the HOD: **FAIL** (rebuilt; verdict identical under both bar conventions)

Population: 8,245 arms (`replay_signals.csv`, 2025-07..2026-05). **Premise correction (rebuild):** `entry_m` is the
BREAK minute, not the bar before it — the level was first breached a median 16 min earlier — so the event scored is
the RE-TOUCH of the HOD after the break that then closes back under it (a failed retest), not a failed first touch.
Event rate 18.5 % (647 TRAIN / 873 VAL); 936 trades after rails (571 dropped for R < 1 % of price; SSR proxy 12).

| E1 (2 R target), ETB-now | n | mean net R | day-t | ex-top-5 % | stop share | green wk | placebo pct |
|---|---|---|---|---|---|---|---|
| TRAIN 2025H2 | 193 | −0.717 | −7.6 | −0.86 | 77 % | 11 % | 4 |
| VAL 2026 | 299 | −0.778 | −9.3 | −0.92 | 81 % | 5 % | 0 |

E2 (VWAP reclaim) −0.53 R. All-arms and shortable-now universes agree. Placebo short (random later minute, same
geometry) ≈ −0.6 R: ANY short in this population at this geometry loses; the observed is worse than placebo.
Rebuild (`REBUILD_1736.md`): 0 mismatches on every field with the next-bar fallback for a missing F+1 bar; the
strict reading (no trade when F+1 is missing) moves the means to −0.66 / −0.75 on 177 / 280 trades — same verdict.
Stop-leg cost read as half_spread + 13 bp by both builds.

**Reading.** After a failed retest these names re-take the touch high four times in five. The mirror (buy the
failed retest) is read as the post-hoc secondary M3 in PREREG_1737 Amendment 1 — against a geometry-matched placebo,
never as a pass of this cell. Caveats: shortability flags as of today, borrow cost 0, SSR proxy, spread fallback
for unmatched arms, 2025H2/2026H1 only (few regime episodes).
