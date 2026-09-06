# BF consistency profile P1 — test and launch runbook (2026-09-06)

P1 = above-VWAP gate + pole gain ≥ 5% + entry price ≤ $20 + 50% off at +2R
(breakeven, remainder trails) + regime sizing OFF. Evidence:
`research/bf_consistency/README.md` §6 (honest regen-7 cache, $50K / $2K base):
$131K / 2026 YTD +$38.0K / worst month −$7.6K / MDD −$15.1K / 14 of 20 green.
Every knob is ONE spec for BT Stage-2 and live; all default OFF.

## 0. What is already done
- Unit + integration tests per knob: `tests/test_bf_vwap_gate.py`,
  `tests/test_bf_profit_partial.py` (one-tape BT/live parity),
  `tests/test_bf_risk_cap.py`, `tests/test_batch_backtest.py::TestStage2LiveKnobs`.
- Rule pedigree: the three entry rules are the three worst raw buckets in
  BOTH years and would have been picked on 2025 alone (§6e) → 2026 is
  out-of-sample for them. The +2R partial and regime-off were chosen on
  the full window (in-sample) — the shadow window is their test.

## 1. Pre-flight (weekend)
1. Full suite green: `ulimit -v 3500000; nohup python -m pytest tests -q -p no:cacheprovider > logs/suite.log 2>&1 &` — never as a harness background task (see §5).
2. Config probe (no market needed):
   ```
   python -c "from config import Config; c=Config()._load_yaml_only()['trading']; print(c['bull_flag']['vwap_gate'], c['profit_partial'], c['risk_cap'], c['regime_sizing']['enabled'])"
   ```
3. Boot rehearsal (protocol): `python main.py --scan --trade --verbose` for ~2 min with the shadow config below; expect the init lines
   `VWAP gate: ON`/`Profit partial: ...` (shadow) and ZERO orders; Ctrl-C.

## 2. Shadow window — 10 sessions, log-only (Mon 9/7 → ~9/18)
`config.yaml`:
```yaml
trading:
  bull_flag:
    vwap_gate: {enabled: false, min_dist_pct: 0.0, shadow: true}   # logs gate + pole>=5 + price<=20 would-skips
  profit_partial: {enabled: true, r_multiple: 2.0, fraction: 0.5, move_to_breakeven: true, shadow: true}
  # regime_sizing stays ON; the report recomputes P1 sizing from the REGIME log lines
```
`profit_partial.enabled` must be true for the arming to happen; `shadow: true`
turns the sell into a `PROFIT PARTIAL [SHADOW] would sell` line. Restart:
`sudo systemctl restart onemil-trader`.

Daily, after the close: `python scripts/bf_shadow_report.py --day YYYY-MM-DD`
(the EOD dive cron runs it too). It joins the journal lines to the BF trades
in the DB and prints, per live BF setup/trade: the three rule decisions, the
would-partial, the regime multiplier, and the P1-counterfactual P&L.

**Pre-committed pass criteria for the window (not P&L — 10 sessions ≈ 3 trades):**
1. Coverage: every live BF setup that reached the conviction stage has a
   `VWAP GATE`/`CONSISTENCY RULES` decision line (100%).
2. Parity: Stage-2 on the roll-forward cache for the same days agrees with
   the live rule decisions on every overlapping trade (any disagreement = a
   parity bug to fix before launch, same class as the CWVX trail defect).
3. No ERROR/Traceback from the new code paths (`grep -E "bf_vwap_gate|bf_profit_partial|bf_risk_cap"`).
4. At least one `PROFIT PARTIAL [SHADOW]` line observed if any trade reached
   +2R — the arming/trigger path proven on a real tape (else state it).

## 3. Launch (joint decision after the window)
```yaml
scanner:
  price_max: 20                      # was 30
trading:
  bull_flag:
    min_pole_gain_pct: 5.0           # was 3.0 (detector knob — same rule Stage-2 applies)
    vwap_gate: {enabled: true, min_dist_pct: 0.0, shadow: false}
  profit_partial: {enabled: true, r_multiple: 2.0, fraction: 0.5, move_to_breakeven: true, shadow: false}
  regime_sizing:
    enabled: false
  risk_cap: {enabled: false, max_risk_mult: 2.0}   # P2's dial; keep available for when base risk scales
```
`sudo systemctl restart onemil-trader`, verify the init lines, Telegram the
flip. Rollback = the previous values + restart (zero state to unwind; open
positions keep their StopMonitor watches).

Monitor: `journalctl -u onemil-trader | grep -E "VWAP GATE|PROFIT PARTIAL|RISK CAP|REGIME"`.
EOD: the green check's BT-vs-live comparison must run Stage-2 with the SAME
config (it reads config.yaml), so the reference flips with live.

## 4. After launch — the sizing question
P1's numbers are at $2K base risk; live BF runs at `risk_per_trade: 60`.
Scaling base risk is the owner's capital decision (retirement plan §7). At
$500 base, P1's 2026 pace is ≈ +$9.5K YTD with a worst month ≈ −$1.3K; the
risk cap (2× base) is the month-variance dial when the base rises.

## 5. The memory rule (why the suite got killed on 9/6)
The bulk 1-min preload reads ~50 GB of `cache.db` and fills the page cache;
`MemFree` drops to ~100–400 MB while `MemAvailable` stays > 6 GB, and the
4 GB swap fills because `vm.swappiness=60` prefers swapping idle process
memory over dropping cache. The Claude Code harness kills *background tool
tasks* on low `MemFree` (page cache is not real pressure).

Fixes:
1. **App side (done 9/6)**: `Database.get_intraday_bars_bulk` releases the
   pages it pulled in (`posix_fadvise(POSIX_FADV_DONTNEED)` on `cache.db`
   after the bulk read) — the loader no longer evicts everything else.
2. **OS side (owner runs once — the agent's sudo is blocked for sysctl)**:
   ```
   ! sudo sysctl -w vm.swappiness=10 && echo "vm.swappiness=10" | sudo tee /etc/sysctl.d/90-onemil-swappiness.conf
   ! sudo sync && echo 1 | sudo tee /proc/sys/vm/drop_caches
   ```
3. **Rules**: one bulk job at a time; long jobs run under `nohup` + a
   Monitor, never as harness background tasks; `ulimit -v` on every python;
   the full suite runs only when no preload is in flight.
