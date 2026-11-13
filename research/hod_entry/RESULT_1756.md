# RESULT 1,756 — gap >= 10 % as a filter on HOD / ORB longs: REJECT

**Data.** HOD = 1735 B0 net R per signal (live-rule outcome, whole signal population, not slot-capped), TRAIN 2025, VAL 2026-01..05; 3,521 TEST rows dropped, 125 NaN-R dropped. Raw gap from 1748+1746 caches: **coverage 65.4 % — fails the 80 % rail**; no-gap arms are their own bucket (VAL −0.20 R vs −0.36 R at <5 %, so not random). Computed gap vs ORB's gap_pct column: corr 0.9996. ORB = thermo book, 473 fills.

**VAL buckets (mean net R, n):** <5 % −0.360 (2,879); 5–10 −0.339 (132); 10–20 −0.412 (23); >=20 none; no gap −0.204 (1,661). Rest minus gap >= 10: +0.053 R, day-t 0.18 (TRAIN +0.24, t 1.0; halves +0.36 / +0.11, t 0.9 / 0.4). Right sign, never significant. Placebo (prior-day range, same top 0.5 % share): TRAIN +0.22 (t 0.7), VAL −0.51 (t −1.1) — as large as the gap effect, no specificity.

**Break hour:** gap >= 10 cells n 1–13; TRAIN 10–11h −1.28 (n 8), 11–13h −0.93 (n 13), 13h+ +1.44 (n 4); VAL 11–13h +0.31. No pattern, no power.

**Filter, VAL:** drops 1.05 of 213 signals/wk; R/wk +0.43 (t 1.67; TRAIN +0.39, t 2.2) on a population at −0.30 R/signal, so removing any negative slice "helps". Green-week share unchanged. VAL day-t >= 2 fails.

**ORB by half-year (rest minus gap >= 10, R/fill):** 2025H1 −0.12, 2025H2 +0.14, 2026H1 +0.07, 2026H2 −0.34; pooled −0.04 (t −0.45). ORB gap >= 10 fills earn +0.04 (10–20, n 76) and +0.52 (>=20, n 18); the filter costs −0.15 R/wk (t −1.8).

**Verdict: REJECT** both books. The gap fade is an open-to-close fact; it does not show in breakout outcomes, and HOD is −0.2 to −0.3 R in every bucket.

CHILDREN: (1) fill the missing-gap bucket from cache.db daily bars to clear the 80 % rail (n ~ 30 per split, MDE ~ 0.5 R, likely unresolvable); (2) fade as a SHORT entered only after a failed break on gap >= 10 names; (3) ORB gap >= 20 (+0.52 R, n 18) as a size-up bucket, pooled with the 2023–24 book for n.
