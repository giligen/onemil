# Weekend money-machine program — report (2026-07-04)

Owner mandate: "treat this as your own money machine... be creative."
Everything below is BT-proven or explicitly labeled proposal. All ships
land at Monday 07/06 12:30 UTC auto-start. Full test suite: **2,622
passed / 0 failed** (including 5 pre-existing failures fixed).

## SHIPPED

### 1. ★ PDR veto (ORB) — the weekend's headline (commit d52e80b)
Veto selected picks whose **previous day's range ≤ 8%**. Mechanism: ORB
monetizes continuation — day-2 of the fireworks, not day-1 fresh pops.

| | base | veto 8.0 |
|---|---|---|
| Cum P&L 18mo | $154,892 | **$209,734 (+35%)** |
| Max DD | −$29,297 | **−$20,129 (−31%)** |
| WR | 35.8% | 40.2% |
| Trades/day | 3.30 | 1.57 |
| 2026 YTD | +$91K | **+$106K** |

- All 3 eras positive, monotone thresholds 6–10, ALL top-10 giants kept.
- **NO-REFILL form only** — refill tested toxic (25H2→$0, MDD −$50K).
- Verified dollar-exact in the integrated production BT pipeline.
- Gross losses nearly halve (−$344K→−$178K): this IS the "identify
  losers upfront" thrust — one boring feature, not a fancy model.
- Monitor Monday: `journalctl -u onemil-trader | grep "PDR VETO"`.
- Expect **fewer entries/day (~1.6 avg)** — sizing/cushion cadence slows
  but per-trade quality jumps. See ramp-policy proposal below.

### 2. Code-review fixes (commit aeefedf) — W5, 2 independent deep reviews
The flagship selection-race fix **largely did not work as shipped**:
- Grace gate checked rangeless-ness over the caller's subset; the racing
  path (WS drain) passes only ready names → gate never fired there. Now
  checks the full pool.
- Deferral never re-fetched stragglers (sweep one-shot, WS lacks the 9:30
  anchor) → was pure delay. Defer now re-arms the sweep.
- **DST regression**: month-granularity ET offset would have silently
  produced ZERO ORB entries for ~2 weeks/year (first bite: early Nov
  2026). Now ZoneInfo-accurate.
- Orphan reconciler close_position always failed on broker-held qty
  (open emergency-stop orders) → cancel-first + retry. This was the
  naked-overnight-gap scenario the module exists to prevent.
- Partial-fill escape hatches (stall/terminal/time-stop cancels with
  filled shares) now confirm instead of dropping tracking.
- Exit-reason catalog: orphan_recovered_force_close + exhaust+ composite
  now classify correctly (no false taxonomy-leak alarms).

## REJECTED (evidence recorded so we don't re-litigate)
- **W1 slot/sizing**: N=3/5 slots, rank-weighted sizing, dedup loosening —
  all noise or era-inconsistent (rank-weighting: 2025 +$28K / 2026 −$5K,
  monsters live at ALL ranks). Structure stays 4 slots as-is.
- **W2 stump search**: 418 veto rules tested; only PDR survived. The
  `price_vs_20d_high` family is the cautionary tale: TRAIN +$32K, OOS
  −$70K.
- **W3 BF threshold**: intraday scanner threshold is a DEAD KNOB
  (byte-identical Stage-2 at 10/12/13/15% — the TTF+conviction stack
  rejects that whole band). BF supply cannot be bought there.
- **W4 exits**: post-veto loser anatomy = 182 full stops (−$814 avg)
  vs 132 EOD monsters (+$2,584 avg); April's V0 Pareto stands.

## PROPOSALS (need your sign-off)
1. **Ramp policy** — `docs/ramp_policy_proposal_jul2026.md`: advance on
   operational-green + loss-floor gates instead of %-cushion. Rationale:
   cushion punishes BT-consistent variance (both June overrides), and
   the PDR veto's halved trade count slows cushion accrual exactly while
   quality doubles.
2. **Delete gap_and_go WIP** (3 untracked files): assessed as the
   pre-ORB precursor (dated 2 days before ORB shipped), never run,
   latent trigger-window bug, near-total ORB overlap. Nothing worth
   salvaging that isn't already an ORB exit-variant question.
3. **$30–60 price band ORB** (W8): supply confirmed — 10.4 gap-up
   candidates/day (avg 10.9% intraday range) vs 36.9/day in the current
   $3–30. Institutional-grade spreads. Needs a band-scoped features+
   resim build (~1,300 symbol-days of 1-min bars) + its own TRAIN/OOS
   fit → ~1 day of work + API time. Would run as a separate Pre-Stage-0
   instance. $1–3 band also has supply (13.2/day, 25% ranges) but
   spread economics (~50bps/1¢) likely eat the edge — lower priority.
4. **Gap 4% vs 5%**: untestable on existing data (capture was gated at
   snapshot-gap ≥5) — rides the same features-rebuild as #3.

## Measured, no action needed
- **BT universe look-ahead** (bull-flag-cache-derived = requires 10%+
  EOD range): 28% of live order *placements* since 5/1 were on
  symbol-days outside the BT universe, but only 3 ever filled+closed
  (+$655 total) — quiet candidates rarely break out, so the entry
  mechanism self-limits the bias. Documented; revisit only if the
  filled-outside-universe count grows.

## Monday validation (automatic)
- Observer cron 9:26 ET — expect 0 drops; grace-gate/sweep-retry lines
  on straggler days; PDR VETO lines on quiet-prev-day picks.
- Touchgo debug cron 21:00 UTC — expect REKEY=0.
- Selection audit JSONL on every burst.
