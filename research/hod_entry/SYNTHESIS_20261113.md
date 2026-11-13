# HOD-break line — synthesis of every finding and the strategies they imply (2026-11-13, owner: "go deep, deduce new strats")

## The facts that survived every rebuild (cells 1,351 → 1,743)
1. **Gross at the level is small and real, the stop pays for it.** Live book +0.075 R gross at its own levels; cost
   ≈ 0.15 R (entry chase 8 bps, stop slip 13–35 bps on a 41 % stop-out rate, spread). BT book −0.22 / −0.30 R net.
2. **The post-break path is mean-reverting** (1,743): VR(10) 0.87 / 0.83, VR(30) 0.74 / 0.77, every CI < 1, every
   tercile; stop-first 53–55 % vs target-first 25–26 % at −1 R/+2 R (the driftless coin); minutes 1–15 drift
   NEGATIVE (−0.9 / −0.2 bps/min), 16–60 slightly positive, entry→15:55 **+0.17 / +0.14 R gross** (t 0.8 / 1.4).
3. **The retest is information, one-directional** (1,740/1,742): arms that come back to the level after the break
   have B0 −0.31 / −0.42 R; arms that never do have +0.68 / +0.73 R. Nothing observable before the touch separates
   them (hold length nothing; depth known only after the fill). Every retest entry (at, under, after a hold, after a
   reclaim) loses −0.09 … −0.40 R.
4. **The long tail carries everything**: ex-top-5 % negative in every cell; a design that loses the runners loses.
5. **Shorts lose structurally**: placebo short −0.6 R (1,736) — the body mean-reverts but the tail is UP.
6. **Cost sorts outcomes, the signal does not**: half-spread tercile −0.27 R; optionable-with-print names −0.19 vs
   −0.34 (TRAIN), −0.26 vs −0.51 (VAL) (1,741) — every bucket still negative under the stop design; OFI "lift" was
   cost (1,393–1,395); 2,400+ causal filters, regime, attention rank, candles, time-of-day, options flow: nothing.
7. **Frequency is the asset**: ~93 arms/day, ~50 fills/wk, the only book on the machine with that count.

## What the facts rule out (no more cells here)
Any stop-based long design (2); any retest/pullback entry (3); any short (5); any per-fill filter for direction (6);
options as the instrument (1,738 void, 1,741 fail); confirmation entries (1,487). The engine as built cannot be made
net positive by a knob — that is a measured statement with an MDE of ~0.05 R on 12k signals, not an opinion.

## What the facts leave open — the deductions
**D1. No stop + cheap names.** (2) says holding to the close earns +0.15 R gross; 1,735 V1 (no stop, half size, to
the close) netted −0.09 R on the WHOLE book because the round trip (chase + 15:55 spread) is ≈ 0.2 R on the median
name. (6) says the cheapest tercile pays roughly half of that. The untested combination is geometry-free: the gross
of the close-hold on the names where the round trip is < 0.1 R. Data already on disk (1735_per_signal.csv ×
signals.parquet half_spread × 1741_flow.csv covered). **Cell 1,744**, pass bar below; tail-carried by construction,
so ex-top-5 % and the capped book decide it.
**D2. The population, not the signal.** (1) gives ~+0.1 R gross for a breakout at a level — the same figure the ORB
book shows in and out of regime. On this universe R ≈ 1.5 % of price while the round trip is 0.3–0.4 % (ratio 4–5);
on the liquid top-300 by dollar volume the same geometry at R ≥ 1 % of price costs 0.02–0.05 % (ratio 20–50). If the
gross is a property of breakouts rather than of gappers, the liquid version nets most of it. Free Alpaca SIP bars
(top-300, 2026, ~27 M rows, one night). **Cell 1,745**: the HOD spec unchanged (ONE spec), new universe, cost at
measured NBBO, cadence bar PREREG'd. This is the only idea left that can change the SIGN rather than the cost.
**D3. Sell the information.** 93 arms/day is a stream; the engine cannot monetise it but a discretionary trader
can (owner list, live since 09:40 ET today; Friday scorer decides with n ≥ 5). No cell needed.
**D4. Not deduced, by elimination:** liquidity provision at the level (= the retest bid, adverse), under-flush bids
(= 1 %-under, −0.3/−0.4), EOD split (drift 61–120 ≈ 0, 15:25→15:55 is inside the exit-lab's ±0.04 R band),
queue-imbalance timing (the chase is 8 bps of a 15–40 bps problem — worth measuring once sizes land, not a strategy).

## Order and rules
1,744 tonight on disk data (no fetch). 1,745 fetch tonight if `df` ≥ 5 GB, score tomorrow night in the DB window.
Each: PREREG frozen → score → independent rebuild → Telegram → commit. A pass is a PAPER mode, never live from a cell.
If both fail, the HOD line is closed at the engine level and the frequency asset stays as D3 only — and that is a
conclusion about this universe's breakouts, stated with its MDE, not "no edge exists".

## Pass bars (frozen here, copied into the PREREGs)
1,744: cheapest-spread tercile, no-stop half-size to the close: own book mean net R ≥ +0.10, day-t ≥ 2.0 on TRAIN
and VAL, both TRAIN halves same sign, ex-top-5 % ≥ 0, weekly P10 ≥ −2 R, ≥ 3 fills/wk. 1,745: the cadence bar
(`docs/cadence_bar.md`) on the liquid book, cost from the measured minute-of-day half-spread of each name.
