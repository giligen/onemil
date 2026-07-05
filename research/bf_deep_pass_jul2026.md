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

## 🚨 FOUNDATIONAL FINDING (2026-07-04): the Stage-1 cache is IRREPRODUCIBLE
Full rebuilds CANNOT reproduce the prod cache (March-25: 16 vs 83; June-26:
7 vs 24) because `find_big_movers` applies `float_max=10M` using the
CURRENT universe table's float_shares — floats mutate (dilution), so a
rebuild time-travels today's floats onto history (survivorship drift).
Implications:
- The prod cache (nightly same-day appends) is POINT-IN-TIME TRUTH and
  must NEVER be rebuilt — the no-overwrite rule is about correctness,
  not just cost. The 171→39/mo supply collapse is REAL (market), since
  the prod cache is point-in-time.
- ANY study comparing a full rebuild to the prod cache is invalid.
  Valid comparisons: rebuild-vs-rebuild TWINS (identical drift both arms).
- Long-term fix candidate: snapshot float_shares per date (the cache rows
  already capture point-in-time implicitly; a universe history table
  would let rebuilds be honest).

## Loose-detection experiment, redesigned (twin builds)
- /tmp/bf_cache_loose.csv (loose envelope) DONE: 201 trades.
- /tmp/bf_cache_strict_twin.csv building (no env overrides, same flags).
- PREVIEW (loose vs strict-control months): March-25 16 = 16 (loosening
  adds NOTHING in 2025-style months); June-26 22 vs 7 (**3x more setups
  in the starved regime**). Era asymmetry hypothesis: the strict detector
  was implicitly shaped for 2025 flags; 2026 flags are structurally
  different — rigidity binds exactly where supply died.
- VERDICT (2026-07-04 22:00): **REFUTED, decisively.** Stage-2 twins:
  STRICT wins 2025 (+$27.3K vs +$16.1K), 2026 (+$14.2K vs +$8.3K), full
  (+$41.4K/64.5%WR vs +$24.4K/52.6%). The 100 incremental setups are
  +$21/trade raw, NEGATIVE in 2026 (−$1.3K/45tr), and in Stage-2 they
  DISPLACE winners via concurrency/daily-loss sequencing (the ORB-B1
  displacement lesson). The June 3x-setups preview was supply, not edge.
  May's rejection re-confirmed on clean twin methodology. The strict
  detector is load-bearing — do not loosen. Third consecutive refutation
  tonight (knobs, conviction re-weights [hand weights beat ridge/ensemble
  OOS at every pass rate], detection loosening): BF per-setup selection
  is at its information ceiling. Remaining levers: EXITS (never swept)
  and capital allocation (structural, post-ramp).

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

## 🏆 GOLDEN FIND — trail activation 1.5R → 2.0R SHIPPED (2026-07-05)
The one never-swept BF domain (exits) delivered. Replay harness on the
67 Stage-2 defended trades (validated: 58/67 exit prices within 2% of
cache, median diff 0.05%):
- A=2.0/T=1.0: 13 trades changed, **13 improved, 0 hurt**, 19/19 months
  ≥0, top gain only 26% of delta. ≈ **+$12.3K on the $35.6K book (+35%)**.
- Monotone 1.5→1.75→2.0 in BOTH eras (2025 +105→+133 sum-ret%, 2026
  +50→+75); plateau at 2.25.
- REVERSES 2026-04-13 (2.0→1.5, +$9.3K then): that verdict was on the
  pre-floor-1.8/pre-TTF population where OPTX-class (peak 1.5-2R →
  full reverse) was common; in the modern selected book that class
  occurred ZERO times in 18mo. Upstream filters moved the exit optimum —
  the recurring lesson of this audit.
- Caveats: replay skips vol-guard + exhaustion-partial interplay (the
  9/67 validation outliers); revert trigger documented in config.yaml.
- Harness: research/scripts/bf_exit_sweep.py.

## 🔍 THE MISSED GATE (2026-07-05, owner challenge "what are we missing")
Owner: "BF is nothing — what are we missing?" Answer found in three steps:
1. BF's per-trade edge is FINE (58-65% WR, ~$600/trade w/ trail 2.0);
   the problem is THROUGHPUT: 4.3 trades/mo vs ORB's ~33.
2. Universe gates are NOT the choke: the eligible mover field GREW in
   2026 (2,068 → 2,702/mo passing float+price gates). Detection RATE
   collapsed 8% → 1.4% of eligible movers.
3. Sampled kill-table (250 eligible movers/year through the live
   detector with debug capture): **"Pullback too long (max 5 candles)"
   is the terminal blocker for 53% (2025) → 66% (2026) of rejections.**
   2026 movers still flag — they consolidate LONGER than 5 one-minute
   candles. The pattern slowed; the cap can't see it.
   AND: max_pullback_candles was NEVER tested alone — May's sweep had no
   pullback-only cell; the loose envelope bundled it with junk-admitting
   knobs whose earlier intraday matches SHADOW pullback-admitted setups
   (detector is first-match-wins). Twice missed by bundling.
IN FLIGHT: single-knob twin BF_MAX_PULLBACK_CANDLES=10
(/tmp/bf_cache_pullback10.csv) vs strict twin through Stage-2.
Kill-table script: /tmp/bf_rejection_histogram.py (reusable).

## Queued next (after detection verdict)
1. Conviction re-weighting: cache stores per-rule components (conv_*);
   walk-forward re-weight THROUGH Stage-2 (per feedback memory: lever
   isolation on full stack, never per-rule sign analysis).
2. Funnel instrumentation: per-filter kill table from Stage-2 verbose.
3. BF trail/exit params never swept (trail activation/distance) — after
   the above.
