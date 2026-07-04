# Spread-gate fine-tune — SHIPPED: 150 → 300bps (2026-07-04)

Owner ask: can we fine-tune the spread gate — skip vs enter-regardless vs
enter-with-adapted-limit vs delay? First-ever validation of this
live-only filter: historical NBBO at 9:35 ET fetched for the 570-trade
post-veto defended book (94% coverage). Script:
research/scripts/orb_spread_gate_study.py; data:
/tmp/orb_spread_gate_quotes.csv.

## Key findings
1. **The 150bps gate was a monster-killer.** BKKT +$20,762 quoted 153bps
   at 9:35 (SKIPPED live); XNDU +$11,876 at 267bps (SKIPPED). Wide-cohort
   monster rate (2/62) EXCEEDS the book average. Spread at the open is a
   heat signal, not a defect signal.
2. **100-150bps is the richest per-trade bucket**: +$63.7K on 49 trades
   ($1,299 avg). Tightening to 100 would cost $63.7K (kills ANNA 124bps,
   BNAI 121bps). Never tighten this gate.
3. **Threshold sweep** (exit-cost honest: extra spread/2 on wide exits;
   full-spread sensitivity): 300bps = +$24.7K/18mo vs 150 (both penalty
   scenarios). Gateless = worse than 300 (the >300 tail nets negative
   after costs + is where unmodelable fill-adverse-selection is worst).
4. **Delay variant DOMINATED**: only 24/54 wide names compress under
   150bps by 9:40 (21/43 by 9:45), and late placement forfeits the
   early breakouts. Killed.
5. **Downsizing unnecessary**: risk-parity already sizes wide-range
   (=wide-spread) names smaller.
6. **"Enter with a limit that works" is already the mechanism** — the
   stop-limit caps entry at range_high+30bps regardless of spread. The
   only unmodelable risk is fill probability when the ask sits above the
   limit on wide names. Ship keeps the 30bps limit UNCHANGED; the
   entry_quote_spread telemetry monitors wide-name fill rates live.
   Revisit an adaptive limit (e.g. +max(30, spread/2) bps) ONLY if 2-4
   weeks of live data show wide names systematically failing to fill.

## Fragility disclosure
The wide cohort's +$28.5K is 2 monsters (BKKT+XNDU $32.7K; other 60
trades net ~-$4K). Same asymmetry that rejected the $4 price floor, in
reverse: a cost filter must never sit where monsters live — capped
savings vs uncapped forfeits.

## Ship
- orb.yaml + template: entry.max_spread_bps 150 → 300 (code defaults too).
- BT stays gateless (≈ gate-300 within $6K — documented parity gap, small).
- Rollback: flip yaml to 150 + restart.
- Monitor: weekly report edge-capture section + entry_quote_spread on
  fills 150-300bps (expect a handful/month).
