# QQQ noise-area sleeve — DRY-RUN SPEC (research artifact, 2026-09-17)

**Status: specification only. No engine code is changed by this document, no config key exists yet, nothing is
enabled.** It exists so that a dry run, if the owner asks for one, produces a journal the EOD check can
re-simulate bar for bar. Everything below is the rule audited in `Q/REPORT.md`, written as engine behaviour.

Read `Q/REPORT.md` first: the OOS number this spec would be testing is **+4.57 bps per traded day at 1× on QQQ
(t 1.27), ≈ $321/month at $60K, worst month −$2,243, and it disappears when the top 1% of days is removed.**
The dry run is a measurement of execution, not a decision to trade.

---

## 1. Session state, built once at 09:30:00 ET

| field | construction | source | must be known by |
|---|---|---|---|
| `sigma[k]`, k = 30..389 | mean over the **previous 14 sessions** of `abs(close(k)/open_0930 − 1)`; ≥ 10 of the 14 sessions must be present or the day is skipped | 14 × 390 QQQ 1-min bars — `cache.db intraday_bars_1min`, else `AlpacaClient.get_1min_bars_multi(['QQQ'], t−20d, t−1d)` | 09:29 |
| `prev_close` | the previous session's **15:59 regular-hours bar close** (not the auction print, not extended hours) | same store | 09:29 |
| `open_0930` | the open of today's 09:30 bar | first streamed bar | 09:31 |
| `UB[k]` | `max(open_0930, prev_close) × (1 + VM × sigma[k])`, VM = 1.0 | derived | 09:31 |
| `LB[k]` | `min(open_0930, prev_close) × (1 − VM × sigma[k])` | derived | 09:31 |
| `vwap[k]` | cumulative `Σ(vw×v)/Σv` over **regular-hours bars only**, from 09:30 to k | streamed bars | running |
| `sig14` (only if dynamic sizing is used) | sample std of the 14 daily close-to-close returns ending at t−1 | `cache.db daily_bars` | 09:29 |
| `last_bar_index` | from the market calendar (early closes = 12:59 close → k_last = 209) | `AlpacaClient.get_calendar`, fail closed | 09:29 |

If any of these is missing at 09:31, the day is **skipped with an ERROR log** — never with a default.

**Splits.** `etf_1min.db` bars are raw. QQQ has not split in the sample, but the engine must still compare
`open_0930 / prev_close`: outside 0.75–1.33 without a corresponding index move, skip the day and log ERROR.
(TQQQ, if it is ever the instrument, split 5 times in 2016–2026 — see `REPORT.md` §6.)

## 2. The minute loop

Bars arrive through the StopMonitor websocket, light handler, exactly as the HOD engine registers its own:

```
stop_monitor.register_bar_handler('qqq_noise', self._on_bar_close, window=False)   # ONE bar dict per event
stop_monitor.subscribe_bars('QQQ')
```

The handler enqueues only. A dedicated drain thread does the work (the HOD engine's pattern —
`trading/hod_break_engine.py::_on_bar_close` / `process_tick`).

**Decision minutes: k ∈ {30, 60, 90, ..., 360}, i.e. the 10:00, 10:30 … 15:30 bars.** Nothing is evaluated on any
other bar (the every-minute variant of this rule is dead: SR 0.10 IS — `zarattini_spy.md` ablation C). A bar for
minute k is actionable when it CLOSES, i.e. when the bar timestamped k is received (~k+1:00 to k+1:05 ET).

On receipt of the closed bar k ∈ checks, with `c = bar.close`:

1. **Exit first.** If long and `c < max(UB[k], vwap[k])` → SELL to flat. If short and `c > min(LB[k], vwap[k])`
   → BUY to flat. Record `stopped_side`.
2. **Entry second, same event.** If flat and `c > UB[k]` and `stopped_side != +1` → BUY. If flat and `c < LB[k]`
   and `stopped_side != −1` → SELL SHORT. (So a stop and a reverse can happen on the same check; the side just
   stopped can never be re-entered on that same check.)
3. **Order type: plain market, submitted immediately.** The simulated fill is the **open of bar k+1**. Do NOT
   rest a limit at the band: the band is crossed intrabar between checks, a resting order would fill at minutes
   the rule does not trade, and that is exactly the every-minute variant that loses. The 13 bp median overshoot
   of the signal close beyond the band (REPORT §4) is **not harvestable** under this cadence.
4. **Flat at the end.** On the close of the **15:58** bar, if a position is open, send a market order to flat
   (simulated fill = the 15:59 bar's open). An MOC leg is an acceptable alternative and was modelled: the two
   differ by +0.05 bps/traded day OOS (scenario E vs C), i.e. not materially.

Order life: cancel and re-send at most once if unfilled after 20 s; if still unfilled at the close of bar k+1,
**abandon the trade for that check** and log it as a miss (the simulator assumes the k+1 open).

## 3. Sizing

- 1× = `floor(equity / price)` shares, cash-equivalent, no margin. At $60K that is ~100 QQQ shares.
- 3× notional (the levered variant) = `floor(3 × equity / price)`. The position is intraday only, so it consumes
  **day-trading buying power** (4× equity for a PDT account ≥ $25K), not overnight margin, and pays **zero
  margin interest**. It must never survive the close.
- The paper's dynamic 2%-vol-target sizing (`min(4, 0.02/sig14)`) is **not** recommended for the dry run: it made
  2025 worse (−4.2 bps/traded day vs +2.9 at 1×) and multiplies the drawdown. Fixed 1× only.
- Never share a symbol with another running strategy. QQQ is not in the BF/ORB/HOD universes, but the engine must
  still refuse to act if `get_open_positions()` shows a QQQ position it did not open.

## 4. Kill rails (from the OOS distribution, `REPORT.md` §5; at $60K, 1× notional)

| rail | value at 1× on $60K | derivation |
|---|---|---|
| daily loss stop | **−$750** | worst OOS day was −$971 (−1.62%); p05 is −$364. −$750 sits between, so it fires on ~1 day in 200 |
| 5-day loss stop | **−$1,800** | worst OOS 5-day run −$1,715 |
| month stop (pause, owner decides) | **−$2,300** | worst OOS month −$2,243 (2026-01) |
| max drawdown from the dry-run high-water | **−$6,400** | OOS MDD was 10.6% = $6,369 |
| stale data | no closed QQQ bar for 180 s during RTH | the rule is unrunnable without the tape; flatten and stand down |
| decision miss | 2 missed decision minutes in one day | the EOD re-simulation would be meaningless |

At 3× notional every dollar rail is ×3. A rail hit flattens and disables for the session; two in a rolling 10
sessions disables until the owner re-enables.

## 5. Log lines — the contract with the EOD check

One line per decision minute, whether or not it acts, so the whole day is re-simulable from the journal alone.
Prefix `[QQQN]`. Fields are `key=value`, comma-separated, ET timestamps.

```
[QQQN] SESSION date=2026-09-18 open_0930=598.41 prev_close=595.02 vm=1.0 sigma_src=cache days_used=14 sig14=0.0091 last_bar_k=389
[QQQN] CHECK k=30 et=10:00 c=601.22 ub=600.18 lb=592.77 vwap=599.83 pos=0 action=BUY reason=close_above_ub
[QQQN] ORDER id=... side=buy qty=100 type=market submitted=10:01:00.412
[QQQN] FILL   id=... px=601.35 t=10:01:00.930 next_bar_open=601.31 slip_bp=+0.66
[QQQN] CHECK k=60 et=10:30 c=600.02 ub=600.55 lb=592.40 vwap=600.44 pos=+1 action=EXIT reason=below_max(ub,vwap) stopped_side=+1
[QQQN] CHECK k=90 et=11:00 c=599.10 ub=600.71 lb=592.10 vwap=600.20 pos=0 action=NONE
[QQQN] FLAT   k=388 et=15:58 pos=+1 action=EXIT reason=eod
[QQQN] DAY    date=2026-09-18 trades=2 gross_$=118.40 slip_$=-3.10 net_$=115.30 ret_bp=19.2 rails=none misses=0
```

Rules: every `CHECK` line carries `c`, `ub`, `lb`, `vwap`, `pos` — the four numbers the re-simulation needs —
even when `action=NONE`. A skipped day logs `[QQQN] SKIP date=... reason=...`. Any fallback logs WARNING with
the reason (CLAUDE.md standing rule).

## 6. EOD check (what it must recompute and compare)

`scripts/qqq_noise_eod.py` (to be written only if a dry run is approved) rebuilds the day from Alpaca 1-min bars
and asserts:

1. **Decision parity** — for each of the 12 check minutes, the re-simulated `action` equals the journal's.
   Any mismatch is a HARD fail with the offending `k`, and the three inputs that differ.
2. **Band parity** — re-simulated `ub`/`lb`/`vwap` at each check within 1e-6 of the journal's.
3. **Fill realism** — `abs(fill_px − next_bar_open)` in bp, per leg; the running median is the number that
   decides whether the 0.5 bp/leg assumption holds. A median above 1.0 bp/leg kills the sleeve arithmetically
   (breakeven cost is 2.01 bp/leg OOS, REPORT §5).
4. **P&L parity** — journal net $ vs re-simulated net $ at the same fills, and vs the simulator's own
   next-open fills (two numbers: execution drift, and model drift).
5. **Coverage** — bars received vs bars expected (390), decision minutes missed, seconds from bar-close to order.

## 7. Measurables — what the dry run is actually for

After N sessions, report and compare to the OOS reference:

| measurable | OOS reference | fail threshold |
|---|---|---|
| median fill slippage vs next-bar open | assumed 0.5 bp/leg | median > 1.0 bp/leg |
| decision parity | 100% | any mismatch |
| round trips per day | 0.86 | < 0.6 or > 1.2 over 20 sessions |
| traded-day share | 57% | < 40% or > 75% |
| bar-close → order latency | assumed < 60 s | p90 > 20 s |
| bps per traded day | +4.57 (1×) | reported, never gated on — the MDE over 20 sessions is ±34 bps/day |

**The dry run cannot validate the edge.** With a daily sd of 53.5 bps, 20 sessions have a standard error of
12 bps/day; the effect under test is 2.6 bps/calendar day. A dry run of any plausible length measures
EXECUTION (slippage, parity, latency, coverage) and nothing else. Anyone reading a positive dry-run P&L as
confirmation is reading noise.
