# PREREG — cell 1,444: HOD-break resting entry ONLY on names the live scanner had already qualified before the break

Frozen 2026-09-25 19:58 UTC, BEFORE the adversarial check of cell 1,438 reports on the scanner-tracked hypothesis.
Programme count 1,443 → 1,444.

## Why
Cell 1,438 (the live rule, correct levels) is −0.21 R net over all armed-and-crossed symbol-days, but the cohort that
BOTH 1,427 and 1,438 fill is +0.24 / +0.28 R (n 1,114 / 1,388). What separated the cohorts mechanically was whether
`data/cache.db` held complete 1-minute bars for the symbol-day — and cache.db is written by the live scanner for the
names it is tracking. If "complete in cache.db" ≈ "the stock had already qualified on the intraday scanner before the
break" (a mover with relative volume, on the watchlist), then the positive cohort is a causal, live-computable
condition: the break of the high of day on a stock that was ALREADY a qualified mover — the crowd's stock, not a
random name making a new high.

## Rule (everything else = cell 1,438's causal-arming rule with levels from bars_sip.db)
Arm only if, at the close of bar j, the symbol has a scanner qualification record for that day with detected_at ≤ the
close of bar j (table `scan_results` / the scanner's qualified list in data/trades.db or cache.db — whichever the live
scanner writes, identified in the check). No other change: trigger HOD + $0.01, limit 15 bps, NBBO ask at the first
print ≥ trigger, B0 stop / 2 R / 15:55, measured half-spread + B0 exit cost, 30 bps stop-slip variant beside.

## Pass bar (frozen)
VAL: fill mean net R ≥ +0.15, day-clustered t ≥ 2, ex-top-5 % > 0, ≥ 3 fills/week at first-12/day 4-concurrent; the
same sign on TRAIN-H2. If VAL passes, TEST is read ONCE (a third read of TEST on this population — disclosed; the rule
is a mechanical restatement of what separated the cohorts, not a fitted parameter). Placebo: the same rule with the
qualification record required AFTER the break bar (look-ahead placebo) must NOT show a larger lift than the causal one.

## Consequences
PASS → the live engine arms only scanner-qualified names (the qualified set already exists in the process:
`TradingEngine._qualified_symbols` / the scanner's `_qualified_stock_data`); dry run 5 sessions with the parity ledger,
then real orders at $50 with today's fixes and caps. FAIL → the positive cohort is not causally isolable; HOD-break is
closed at every executable entry and the weekend moves to the other populations.

## Not allowed
Changing any constant; using cache.db completeness itself as the live rule (it is the proxy, not the mechanism);
reading TEST before the VAL bar.
