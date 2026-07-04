# BF deep-optimization pass — state (2026-07-04, IN PROGRESS)

Owner mandate: "flag detection is mediocre at best... make BF great."

## Hypothesis under test (the ORB-spread-gate pattern, applied to BF)
Strict detection (pole>=3, retr<=50, green<=2, vol>=1.5, pullback<=5) was
doing crude quality-filtering that the MODERN stack (conviction floor 1.8
+ TTF + per-tier MACD + regime-in-Stage-2, all post-May) now does better
with more information. If true, strict detection is just starving the
funnel (raw supply 171/mo 2025 -> 39/mo 2026; 3,234 'pole too short'
rejections/week live). May's knob sweep rejected loosenings against the
OLD stack — the interaction with the new stack is UNTESTED.

## In flight
- Loose-envelope Stage-1 cache building to /tmp/bf_cache_loose.csv
  (pullback10/pole2/gain2/retr70/green3/vol1.0; FVRR strict unchanged).
  Build log /tmp/bf_loose_build2.log ('LOOSE BUILD DONE' sentinel).
  Rebuild: /tmp/bf_loose_build.sh. NEVER touches prod cache.

## ⚠ METHODOLOGY TRAP (cost one 2.5h build, 2026-07-04)
`--build-cache` WITHOUT `--no-cache` runs the auto-build path that
applies PRODUCTION FILTERS (regime, max_trades/day=5, circuit breakers)
DURING cache construction → produced a 201-trade 'raw' cache. A raw
build MUST mirror scripts/nightly_bt_update.sh exactly:
`--build-cache --no-cache --cache-file X --capital 5000 --risk 60
--max-shares 15000`. First (invalid) comparison discarded.
- Comparison harness READY: /tmp/bf_loose_compare.sh — strict vs loose
  through full Stage-2, eras 2025 / 2026 / recent-3mo / full.

## Decision rules (pre-declared)
- Ship-consider ONLY if loose beats strict in 2026 AND recent-3mo (the
  May rejection was recency-negative — that lesson binds).
- Then per-knob attribution: pole via qf_pole_bars row-filter on the
  loose cache; other knobs via conv-component proxies or targeted
  rebuild. Ship the MINIMAL loosening that captures the gain.
- Monsters-kept + monthly consistency as usual.

## Queued next (after detection verdict)
1. Conviction re-weighting: cache stores per-rule components (conv_*);
   walk-forward re-weight THROUGH Stage-2 (per feedback memory: lever
   isolation on full stack, never per-rule sign analysis).
2. Funnel instrumentation: per-filter kill table from Stage-2 verbose.
3. BF trail/exit params never swept (trail activation/distance) — after
   the above.
