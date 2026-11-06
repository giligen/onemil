# PREREG — cells 1,723–1,725: ORB loser anatomy → stop cost and early-exit triggers (FROZEN 2026-11-06)

Read `CONTEXT_BRIEF.md` first. Budget ≤ 40 tool calls. Read-only on every project file except this directory and the
scratchpad. Write `RESULT_1723.md` (≤ 400 words + tables) and `per_fill_1723.csv` (one row per fill: ledger fields,
MFE/MAE R, minutes to MFE/MAE, stop slip bps, cluster id, ΔR per candidate rule). Return ≤ 150 words.

## Cell 1,723 — anatomy (no rule yet; TRAIN + VAL both described, nothing chosen on VAL)
For every fill in L1 ∪ L2 (≈ 154), from 1-min bars between fill and exit:
1. Path: MFE and MAE in R, minute of each (from fill), minute of the exit, whether the trade ever printed ≥ +0.25 R /
   +0.5 R / +1 R; close vs VWAP at +5/+10/+20 min; bar-1 close relative to entry; volume in the first 5 min after fill
   vs the range's volume.
2. Stop cost decomposition for the 64 stopped fills (stop_loss + fallback): actual slip bps = (stop_loss_price −
   exit_price)/stop_loss_price; minutes from fill to stop; did price touch the stop then recover above entry within
   30 min (a "stop-and-reverse" count)? How much of the stop $ is slip vs the 1 R itself?
3. Cluster the losers (stops + negative tag/force exits) by the path shape: e.g. immediate failure (MAE −0.5 R within
   5 min, never ≥ +0.25 R), slow bleed (≥ 20 min under entry, MFE < +0.25 R), give-back (MFE ≥ +1 R then stopped), late
   stop (after 12:00 ET). Report count, $ and R per cluster, TRAIN and VAL separately. State the cluster rule in prose.
4. Same table for the winners, so each cluster's "clip risk" is visible.

## Cell 1,724 — pre-declared candidate rules (scored paired vs the actual exit; chosen on TRAIN, read once on VAL)
Exactly these five; add none:
- R1 stagnation: MFE < +0.25 R at 20 min from fill → market exit at the next bar open − 35 bps (the Aug BT survivor).
- R2 bar-1 fail: first full 1-min bar after the fill closes below the entry → exit at the next open − 35 bps.
- R3 VWAP loss: close below session VWAP at the 10-min mark (and still below at 11) → exit next open − 35 bps.
- R4 stop-limit band: the stop becomes stop-LIMIT with the limit 20 bps below the stop; fills at the stop − 13 bps when
  the bar's low ≤ limit, else NOT filled → carried to the next bar whose open ≤ limit... if never, exit at the force
  close (report the unfilled count — this is the HOD 1,466 mechanism and its tail).
- R5 confirmation stop: exit only when a 1-min bar CLOSES below range_low (not the touch), at the next open − 35 bps;
  charges the extra adverse move. (The mirror of R4: fewer stops, deeper ones.)
Per rule: n affected, ΔR paired (mean, day-clustered t, iid t), Δ$ at $375, ex-top-5 % of ΔR, give-back saved vs
continuation forgone, winners clipped (count and $), MDE. TRAIN and VAL separately, L2 shown on its own as well.

## Cell 1,725 — cluster-derived rule (at most ONE, written from 1,723 TRAIN only, frozen in RESULT before VAL)
If a TRAIN cluster carries ≥ 40 % of the stop $ AND has a causal trigger readable at the minute it forms, write the
rule in prose, then score it exactly like 1,724. If no such cluster, say so and skip the cell.

## Pass bar (inherited from PREREG_1679; unchanged)
Paired ΔR ≥ +0.05 R, day-clustered t ≥ 2.5, ex-top-5 % of ΔR > 0, on VAL, same-signed on TRAIN, Δ$ > 0 on both.
A pass → independent rebuild from this prose by an agent that has not read `per_fill_1723.csv` → ORB PAPER with the
rule as the one mechanics change of its session → forward read at 100 fills. A fail is a claim about this test:
state n, MDE and which clusters the rule could not reach.

## Multiplicity
Cells 1,723–1,725; ORB exit cells before this: 1,679 (a)–(f), 1,680, 50+ in `docs/orb_research_apr_2026.md`, early-cut
H1–H5 (Aug). Programme count on the ORB line: see `research/orb_exit/PREREG_1679.md` §Multiplicity and add 3.

## Rails
No orders, no git, no service, no writes to data/*.db, bars_sip.db, cache.db, orb.yaml, logs. Never print API keys.
Scratch: /tmp/claude-1000/-home-ec2-user-onemil/257c3e2d-cf38-45d5-94e7-4877f8170f44/scratchpad/. Run python as
script files (buffered print in `-c`). If a bar fetch is needed for missing symbol-days, cap it at 200 symbol-days
and log the LOST count (fetch completeness gate).
