# CRITICAL - ALWAYS READ FIRST
Since you have a memory of a chicken, you MUST stop every 10min and re-read CLAUDE.md AND DO THIS before every 10th prompt -- this is a must!!!

**This file is the OPERATING RULES only.** The full per-strategy history (every feature flag's evidence, every
rejected knob, every incident) lives verbatim in `docs/CLAUDE_HISTORY.md`. Read the relevant section of that file
BEFORE touching a strategy's rules, and grep it before re-testing any knob — most have been tested and rejected.

# Project: OneMil - Day Trading System
Real-time scanner + automated trading on Alpaca (Ross Cameron momentum style). Live account, owner trades manually
on the same account (his orders/positions are NEVER touched — report only). North star: $10K/month by compounding.

# Token discipline (owner 2026-09-20)
* One agent at a time, fresh context (never fork), prompt = a file pointer + a hard step budget (≤ 40 calls).
* Agents write results to disk and RETURN ≤ 150 words. Never read an agent's log or transcript.
* Never Read a file > 300 lines in full — grep / offset / head. Mechanical work on Sonnet, study logic on Opus,
  Fable plans. Be short and crisp with the owner.
* **Model cascade (owner 2026-09-22): Haiku first, Sonnet only if Haiku fails or the task needs real code, Opus
  only if Sonnet fails. NEVER Fable for agent work** — Fable writes the PREREG/spec, reviews the diff and talks to
  the owner; every execution step (fetch, walk, score, implement, summarise a long report) goes to the cascade.
  A long report is read by a Haiku agent that returns ≤ 200 words, never by the main session.

# CRITICAL: Running Long Commands
* **NEVER pipe long-running commands through `| tail`, `| head`, `| grep`** — buffers everything, you see nothing.
  Run directly; background tasks use `run_in_background=true` without piping.
* **NEVER overwrite or delete cache files (cache.db, CSV caches) without explicit owner permission.** Never use
  `--build-cache` for experiments — write experiment caches to the scratchpad.
* Python `print()` inside `python3 -c` is buffered — run script files or flush.

# Code Quality
* Linus-style: partitioned, modular, reusable, meaningful names, docstrings on every function, verbose progress on
  long processes. Solve root causes, never work-arounds. Use the main code with flags, not bespoke scripts.
* TDD. Coverage ~90%. Validate with the specific unit test in `tests/` AND a system test. Zero failing tests, ever —
  fix the core issue even if someone else broke it.
* **All fallback code paths MUST log ERROR or WARNING** explaining why they triggered. All errors reported
  (missing API keys break execution). Production code never contains mock logic.
* `MagicMock()` MUST use `spec=` for domain classes; `AsyncMock(spec=...)` for async; external SDK objects may omit.
  Fixtures in `tests/conftest.py`.
* Integration tests are REQUIRED for any multi-component flow (DB save→retrieve, serialization both ways, multi-step,
  API→processing→storage, shared config/state): real instances, data checked at each boundary, edge cases. Three
  levels: unit (mocked) / integration (real deps, paper) / system (real environment). Not done until all three.
* Bug protocol: every bug gets BOTH a unit test and an integration test.
* Keep README.md (latest architecture), MD docs and the dependency file up to date. Push to master when the owner
  confirms things work. Commit messages end with the session attribution block.

# CRITICAL: Testing & Deployment Protocol
Never ship untested to production: research the API → isolated test file → real-API test (mocks are not enough) →
verify success in logs → full system test → grep for ERROR/exception → only then commit. Post-incident: revert
immediately, document, commit the revert, fix slowly. Unit tests are not deploy evidence: a weekend boot rehearsal on
the exact ExecStart + real-API probes + a real report run are required before any live change.

# System-in-dev
DB may be locked by parallel processes. Batch processors are verbose. Assume nothing about a fix until the log shows
it; use verbose/debug flags and find the root cause in the logging.

# ONE spec for backtest and live — every rule
A rule is a mechanism + evidence + explicit code, shared by BT and live through ONE helper module (parity by
construction, enforced by a parity test). Accidental behaviour is unacceptable even when profitable. No refill after
a post-ranking veto (refill was tested toxic on every book). The rulebook: `research/orb_machine_rules.md`.

# Strategies — ONE systemd service `onemil-trader` (`python main.py --scan --trade --verbose` + flags)
```bash
sudo systemctl status|restart|stop onemil-trader ; journalctl -u onemil-trader -f
```
State on 2026-09-20 (details, evidence, monitor greps and rollback for every flag: `docs/CLAUDE_HISTORY.md`):
| Book | Flag / service | State | Reference |
|---|---|---|---|
| Bull flag (P1 profile) | `config.yaml trading.enabled` | **PAUSED 9/14**, P1 config intact, LIVE since 9/21, **risk cap 1.0× since 9/24 (owner GO): every trade risks ≤ $150 after all multipliers** (was $270–900 via conviction + MACD tiers), ramp L0→L3 on positive realized P&L (`docs/bf_p1_ramp.md`, `scripts/bf_ramp_check.py`) | honest book $107K/79 tr under the unified trail; **halves under measured NBBO cost** ($139K→$69K, VAL −$8K) — the ramp band must be rebuilt on the measured-cost book |
| ORB B+ | `orb.yaml strategy.enabled`, `--orb` | LIVE since 9/21 (catalyst-off, 8 slots, entry-drain thread, 50 bps); **add-on pools** `universe.addon_pools` (gap 4–5 % $3–30, gap 3–5 % $30–50; selection chain per pool, production first) **DRY DAY 9/22**, enable only on the owner's word (`research/orb_seed_wide/PREREG_LIVE_UNION.md`, kill rules there) | honest $6,085 / 21 mo at $10K stage; **2024H2 out-of-regime test (survivorship-free, live config): flat, −0.01 R/fill over 59 fills vs +0.27 in 2025 (`research/orb_2024/REPORT.md`) — never scale on backtest numbers, only on realized stage P&L**; union rung BT +40 % / +60 % $ at a third of the per-trade edge (cell 1,328, independently rebuilt); weekly selection refit `scripts/orb_weekly_refit.py` (Sun 20:00 UTC); NEVER refit `adaptive_mults`, never drop the Q5 1.5 cap; corpse gate = bar older than 4 days (9/21 fix) |
| MACD wave | `onemil-macd-wave` service, `macd_wave.yaml` | running | outlier-dependent P&L; filters tuned in-sample |
| HOD-break | `hod_break.enabled: true`, `dry_run: true` (ZERO orders), `--hod` | research **CLOSED 9/18** (0/12 causal-filter cells); **exit lab 9/22: 35 more cells, every exit/filter/entry/slot variant within ±0.04R of the live rule (−0.22/−0.30R net) — the population carries no information** (`research/hod_exit_lab/REPORT.md`); dry run stays as a free forward instrument; status = `scripts/hod_dry_ledger.py` | dry_run MUST stay true; no more variants on this population — next = order-flow (Databento, owner spend) or a NEW signal definition under its own PREREG |
| Red-to-green | `red_to_green.enabled: false`, `--r2g` inert | DISABLED 9/17 (TEST profit was a NASDAQ test ticker) | exclude `^Z[A-Z]ZZT$` and any symbol absent from `daily_bars` from every universe |
| Ignition | flags + crons OFF | OFF 9/13/14 | from-zero study found no edge |

Shared machinery: `StopMonitor` (one websocket, routes exits per strategy), `trades.strategy` column, `[ORB]`/
`[HOD]` Telegram prefixes, `daily_bars` universe (2x wrappers IN since 9/5 for ORB, excluded for BF by name).
`orb.yaml` is gitignored — new node: `cp orb.yaml.template orb.yaml`.

**ORB do-NOTs**: enable with empty `ALPACA_ORB_API_KEY`; refit `adaptive_mults`; remove the Q5 cap; disable
`filter.skip_q1` without `docs/orb_research_apr_2026.md`; tighten the spread gate below 150 bps
(`research/orb_spread_gate_verdict.md`); add an LLM catalyst-quality filter (REFUTED); map wrapper news to
underlyings (REFUTED); re-litigate anchor dedup from the CIFG/CIFU anecdote (tested, fails the bar).

# Running Backtests
## Bull flag — TWO stages, ONLY Stage 2 is reportable
```bash
python batch_backtest.py --start 2026-01-01 --end 2026-03-31 --build-cache   # Stage 1: raw movers → cache (NEVER report)
python batch_backtest.py --start 2026-01-01 --end 2026-03-31                 # Stage 2: production filters (<1s)
python backtest.py PLYX 2026-03-13 --verbose                                 # single symbol
```
Always `--capital 50000 --risk 2000 --max-shares 10000`. Stage 2 is a RELATIVE tool for A/B comparisons, never a
P&L forecast (UD scaling and 6 structural BT/live drifts are unmodeled; regime sizing modeled since 7/4,
`BT_REGIME_SIZING=0` for old comparisons). The only honest forecast is accumulated LIVE data. If the numbers don't
match what the owner expects, question YOUR methodology first. Never measure a sizing cap post-hoc on a Stage-2 CSV.
Every ORB CSV goes through `trading/orb_csv.read_orb_csv` (ticker `NA`). ORB BT: `study_orb_pipeline_static_lock.py`
on the entered-inclusive features CSV (older `orb_features_*.csv::pnl` scripts are NOT production-parity).
MACD: `python macd_wave_backtest.py --start 2025-01-01 --end 2026-03-27`.

## Backtest learnings
Gap threshold on the cache build is irrelevant (the pattern detector is the filter). Entry slippage on thin stocks is
multiples of the model — source numbers from the `trades` DB. Bull-flag and MACD P&L are top-trade dependent: always
show the contribution distribution. More trades ≠ more edge (re-entry, pole=2, quick exits all rejected).

# CRITICAL: No research claim ships without an independent check
Five false conclusions were reported to the owner in one week (2026-09-13 → 09-18), every one caught by the OWNER.
**Before ANY research number is put in front of the owner:**
1. **Independent reimplementation** from a prose spec by an agent that has not read the first implementation; compare
   trade by trade on (day, symbol). Catches coding errors, CANNOT catch spec errors.
1b. **Obtainability.** Every fill is a price the market offered inside the bar that fills it, reachable by an order the
   engine would have had resting: next bar's open under a cap, never the touch of a level. Report the share that differs.
2. **Causality trace** for every decision field AND the universe — membership must be knowable at the signal bar.
3. **Price-scale check** daily (possibly adjusted) vs intraday (raw) on the actual trades; unadjusted corporate actions
   fabricate multi-day books. |t| > 6 on any daily cell, one trade > +500% / < −80%, "flagged, not adjusted" → NOT
   reportable until resolved. Read every report's own caveats as an adversary before relaying its headline.
4. **Fill realism and cost.** Gap-through entries, stops inside the entry bar, stop slippage, no double-charged slip,
   no quoted spread on auction fills. **Cost is charged from measured per-trade NBBO at the entry minute, never a
   band** (the band table turned a published +0.3R book into −0.62R; the minute-of-day half-spread table is valid
   09:37–14:01 only). EQUS.MINI quote schemas are never used for spread, cost or order-flow work.
5. **Tail dependence.** Report ex-top-1%/5% and winner-capped. Edge that dies under a cap is a lottery ticket.
6. **Multiplicity.** Count every cell across the whole programme (1,252 on the HOD line as of frames16).
7. **Statistics.** Day-clustered SE beside iid on any day-clustered book; TRAIN edge same-signed in both TRAIN halves;
   count-matched green-week null; placebo decomposition (universe bound / same name-day other hour / signal);
   availability rail (≥ 80% coverage, ≤ 5pp winner/loser missingness gap, else VOID); TEST sealed until the pass bar
   is cleared on VAL. PREREG + FREEZE before scoring, always.

**Phrasing rule.** "No edge exists" is never a conclusion; state universe, horizon, book size, window, cost, and the
MDE. A null is a claim about MY test first — never report a closure without an adequacy review, never end a turn on
one. A research gate is for claims, not capital: positive point estimate + mechanism + bounded downside + resolves in
a quarter = run it live at minimum size; FREQUENCY, not confidence, is the gating quantity. Short is a valid move.

# Interactive Sessions
The owner answers questions and clarifies logic. Decide testable questions yourself with a pre-committed rule; never
hand the owner a menu. Daily brief across books. Fix money-losing defects immediately or pause the book.

# The cadence bar — the pass bar for every book (owner 2026-09-20)
Weeks and cycles, not trades: strong week ≥ +5R with median gap ≤ 3 wk / P90 ≤ 6 wk, bleed between strong weeks ≥ −4R at P90, weekly P10 ≥ −2R, ≥ 55% green weeks above the null, ≥ 3 fills/week at the LIVE config, every ≥3R winner obtainable, ≥ 10 cycles per split. Fat tails allowed, rare tails not. Live tripwire pauses a book at the BT P90 gap + 2 weeks without a strong week. Spec: `docs/cadence_bar.md`; scorer `scripts/cadence_bar.py`. PREREG it on every study.
