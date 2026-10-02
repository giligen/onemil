# Momentum sleeve — hygiene guard + shadow term-structure gate (spec, 2026-10-02)

Evidence: `research/momentum_weekly/RESULT_1700t.md` (guard), `RESULT_1700u.md` (gate: closed as a drawdown repair,
kept as a SHADOW flag only). The paper sleeve's next scheduled run is Monday 10/5 11:50 UTC.

## A. Hygiene guard (changes what the sleeve may buy)
ONE rule, same as the backtest (`research/momentum_weekly/1700s_lowvix.py`, guard mode):
a name is ineligible on a signal date if, inside its last 273 trading bars up to and including the signal date, it has
(a) a one-day close-to-close move > +200 % or < −75 %, or (b) more than 10 calendar days between consecutive bars.

* `trading/momentum_sleeve.py`: module constants `GUARD_MOVE_UP = 2.0`, `GUARD_MOVE_DOWN = -0.75`,
  `GUARD_MAX_GAP_DAYS = 10`; a pure function `hygiene_ineligible(panel, asof) -> Dict[str, str]` (symbol → reason
  'move' | 'gap'); `eligible_universe(...)` removes those names BEFORE ranking (keyword `guard: bool = True`).
  Read the BT's guard code first and match its window convention exactly (which bars are inside the lookback, how the
  first bar is treated) — parity is by test, not by reading the prose above.
* Runner `scripts/momentum_sleeve.py`: one INFO log line per run listing the guard-removed names that would otherwise
  have ranked in the top 40 (symbol, reason, date of the event), or "guard: none in the top 40".
* Parity test: dump the BT's guarded top-20 for three Mondays — 2021-02-08 (AMC removed), 2025-12-29 (ABVX removed),
  2026-06-29 (WOLF removed) — to `research/momentum_weekly/recon/G_holdings.csv` using the BT script's guard mode
  (a small dump script in research/momentum_weekly/, run through the cage), and assert the live functions on the same
  panel slice reproduce 20/20 names on each date. The existing A_holdings parity test stays, run with `guard=False`
  if its dates are affected by the guard (say which).
* Unit tests: +200 % move inside the lookback → out; just under the threshold → in; −75 % → out; an 11-day gap → out,
  a 4-day holiday gap → in; the event 274+ bars back → in; empty / single-row symbols do not raise.

## B. Shadow term-structure gate (changes NOTHING about orders)
The 1,700u median cell: gate ON when the trailing 252-day percentile of VIX ÷ VIX3M (Friday close) is below 30 %.
* Pure function `term_structure_gate(vix: pd.Series, vix3m: pd.Series, asof, window=252, pct=0.30)` →
  dict(ratio, percentile, gate_on); percentile = share of the trailing window's ratios ≤ the as-of ratio; data after
  `asof` never used (unit test: appending later rows does not change the answer).
* Runner: fetch the two CBOE history CSVs (cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv and
  VIX3M_History.csv; see `research/momentum_weekly/1700q_fear.py` for the parse). Any fetch/parse failure logs a
  WARNING and the gate is reported "n/a" — it must never block or change the rebalance.
* Append one row per run to `logs/momentum_sleeve_shadow_gate.csv` (run date, asof, vix, vix3m, ratio, percentile,
  gate_on, sleeve equity) and add `gate ON|OFF|n/a (pXX)` to the single `[MOM]` Telegram line.
* No flag reads the gate when building orders. A test asserts the order list is identical with the gate ON and OFF.

## Verification (all through `bash scripts/research_run.sh -m 2500M …`)
1. `python3 -m pytest tests/test_momentum_sleeve.py -q` (all green, new tests included).
2. A real dry run: `python3 scripts/momentum_sleeve.py` (dry-run is the default; no `--submit`, no `--force`) — the log
   shows the guard line and the gate line; the planned orders are reported, none sent.
3. Report: the dry run's guard line and gate line verbatim, the test count, and the diff stat.

## Not allowed
No git commit, no `--submit`, no `--force`, no order of any kind, no edits to .env / crontab / config.yaml / orb.yaml,
no change to the sleeve's sizing, selection N, schedule or account guard. Never print keys.

## C. Half-size gate (owner asked 10/2 for "a recommendation that balances safety with profit")
Recommendation = guard + HALF size in calm weeks: the 1,700u median half-size cell (VIX ÷ VIX3M trailing-252
percentile below 20 % at the signal close → every name at 1/(2N) of sleeve equity, the rest cash; full 1/N otherwise).
Backtest (one build, independent rebuild owed): 30.7 % / −37.1 % / $659K vs guard-only 29.3 % / −38.3 % / $596K;
half size 28 % of weeks. If the gate is noise it behaves like running ~86 % size; that is the bounded downside.

* `trading/momentum_sleeve.py`: `GATE_MODE_OFF = 'off'`, `GATE_MODE_SHADOW = 'shadow'`, `GATE_MODE_HALF = 'half'`;
  `GATE_HALF_PCT = 0.20`; pure `gate_scale(gate_info, mode) -> float` (0.5 only when mode == 'half' AND the gate
  info exists AND its percentile < GATE_HALF_PCT; 1.0 otherwise — a missing gate (n/a) is FULL size with a WARNING,
  never a silent half). `target_dollars(selected, equity, n, scale=1.0)` multiplies every target by `scale`.
  `term_structure_gate(..., pct=…)` keeps reporting the 30 % shadow state; the half decision uses GATE_HALF_PCT on the
  same percentile.
* Runner: `--gate {off,shadow,half}`, default `half` (PAPER account only — the account guard already refuses a live
  key); the plan line, the ledger row and the `[MOM]` Telegram line show `size 100%|50%`; the shadow CSV gains a
  `scale` column (migrate the header if the file exists).
* Parity: dump from the BT (1700u_gate_guarded.py machinery, cell `VIXratio|w252|p20|half`) the gate state and the
  per-name target weight for three Mondays — two gated, one not — to `research/momentum_weekly/recon/H_gate.csv`;
  assert the live `gate_scale` + `target_dollars` reproduce the weights (0.025 vs 0.05 per name) and the 20 names.
* Tests: scale 0.5 only in 'half' mode below 20 %; 1.0 at exactly 20 %; 1.0 with gate n/a (+ WARNING); 'shadow' and
  'off' never change the order list; a gated week's orders sell every name down to half; the following ungated week
  buys back to full; cash accounting (`broker_sleeve_cash`) is unchanged by the mode.
* Verification: the sleeve test file green; a dry run (cached 10/1 bars, `--skip-fetch`, no --submit/--force) prints
  the size line. Same "not allowed" list as above.
