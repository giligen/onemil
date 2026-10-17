# Momentum sleeve: LIVE path on the shared live account (spec, Sat 2026-10-17; owner: act as owner, build now)

`scripts/momentum_sleeve.py` (878 lines; read by grep/offset only) runs the weekly momentum rotation on a dedicated
PAPER account (`ALPACA_MOM_*`, `AlpacaClient(..., paper=True)` hard-coded at ~l.862 "going live is a deliberate code
change"). Two rotations (10/5, 10/12) were execution-clean (24/24 fills). Decision: first LIVE rotation Monday
2026-10-19 09:45 ET on the owner's LIVE account at a **$10,000 slice**, IF everything below is green by Sunday evening.
The LIVE account is SHARED with the owner's manual positions (report-only, NEVER touched — the BMNR rule) and holds
~$65K equity; manual trades are rare (1–2/month) but the guards below are structural, not optional.

## Required
1. **`--live` flag.** Without it: behaviour byte-identical to today (paper keys, paper=True, `PA` assertion). With it:
   `AlpacaClient(os.environ['ALPACA_API_KEY'], os.environ['ALPACA_API_SECRET'], paper=False)`, assert the account
   number does NOT start with `PA` (log 4 chars max), Telegram prefix `[MOM LIVE]`, and SEPARATE files:
   `state_live.json`, `logs/momentum_sleeve_live_ledger.csv`, `logs/momentum_sleeve_live_weekly.csv`,
   `logs/momentum_sleeve_live_shadow_gate.csv` (bars cache / completeness files are shared — same data).
   `--equity-start` must be given explicitly with `--live` (no default) — the live crons pass 10000.
2. **Shared-account isolation (the core of this task).**
   a. The sleeve's STATE is the source of truth for what it owns. `sync_state_from_broker` (~l.417), `broker_qty`
      (~l.535), `broker_marks` and any "resync from broker" path consider ONLY symbols present in the sleeve state;
      a broker position not in the state is invisible to the sleeve (never listed, counted, sold or logged by symbol —
      count only: "N non-sleeve positions ignored").
   b. **Buy collision:** a selected name the broker already holds outside the state → do NOT buy; log + Telegram
      `COLLISION <sym>: held manually — skipped`; the slot's cash stays in cash (no substitute name — no refill).
   c. **Sell integrity:** sell at most the STATE qty (fractional); if the broker holds MORE (owner added) sell only the
      state qty and WARN; if LESS, sell what is there, WARN, and set the state qty to the broker's.
   d. **Sleeve cash is state-tracked** (start − cost + proceeds, as `broker_sleeve_cash` ~l.394 presumably derives it from
      the account — on live it must NOT read the account's cash). Before buys: account buying power from the broker must
      be ≥ the total buy notional; else ERROR + Telegram and submit NO buys that rotation (sells already done stand).
   e. Order tagging: `client_order_id` prefix `moml-` (grep `client_order_id` ~l.447) so live orders are identifiable.
3. **Rails (pre-committed, in code, live only).** Drawdown: if sleeve equity (state marks) ≤ 0.75 × equity_start at a
   rotation → sell everything the state holds, set `state['paused']=True` with the reason, Telegram `PAUSED`; a paused
   state refuses to trade until the owner clears the flag (document the exact command). Cadence line (report only, weekly
   csv): weeks since the last strong week (≥ +5R, R = 1 % of the slice) and a WARNING when ≥ `--cadence-warn-weeks`
   (default 8). Per-rotation Telegram keeps today's format plus the collision/rail lines.
4. **Dry run is side-effect-free on live.** `--live` without `--submit` prints the full plan (sells, buys, notional,
   collisions, buying power check, sleeve cash, rails) and writes NOTHING (no state, ledger, weekly, gate; today's run()
   saves state at ~l.797/819 even without --submit — guard it). `--force` allows a non-rebalance day.
5. **Reporting.** `scripts/hourly_summary.py` and `scripts/eod_sections.py`: add a `MOM-LIVE` line computed from
   `state_live.json` + marks of the sleeve's names ONLY (day / week / since-start / DD on the slice) — never the account's
   equity, never other positions; absent state file → line omitted (not NO-DATA). The existing LIVE one-line rule stays.
6. **Crons (write for the owner, do not install):** the existing paper lines stay; add Monday/Tuesday live submit lines
   `cd /home/ec2-user/onemil && /usr/bin/nice -n 19 /usr/bin/python3 scripts/momentum_sleeve.py --live --submit --skip-fetch
   --equity-start 10000 >> logs/momentum_sleeve_live.log 2>&1` at 13:50 and 14:50 UTC `* * 1,2` (5 min after the paper
   submits so the shared bars cache is in place — verify `--skip-fetch` reads the paper run's cache for the same asof).
7. **Tests** `tests/test_momentum_sleeve_live.py` (MagicMock(spec=AlpacaClient)): paper default unchanged (paths, paper=True,
   PA assertion); live paths; non-PA assertion; broker positions outside the state invisible; buy collision skipped +
   message; sell capped at state qty / reduced to broker qty; buying-power refusal submits no buys; drawdown pause
   liquidates + flags + refuses next run; dry run writes no file; MOM-LIVE line reads only sleeve names; `moml-` coid.
   Existing `tests/test_momentum_sleeve.py` + `tests/test_hourly_summary.py` + `tests/test_eod_sections.py` green.
8. **Real dry run against the LIVE account (read-only, do it):** `bash scripts/research_run.sh -m 2500M python3
   scripts/momentum_sleeve.py --live --force --asof 2026-10-16 --equity-start 10000 --verbose` (NO --submit). This fetches
   the bars panel (free Alpaca data). Paste the plan into the RESULT: selected 20, buys/notional, collisions (symbol only
   if a collision exists), buying-power check, "N non-sleeve positions ignored". No other account detail.

## Rules
Never submit orders (`--submit` is forbidden for you), never start services, never write config/.env/crontab/caches
(the bars cache the script itself writes is fine), never print keys or the account number beyond 4 chars, never list
the owner's positions, no git (no commit, no stash). Budget ≤ 60 calls. Write `docs/momentum_sleeve_live_RESULT.md`
≤ 35 lines (what changed, tests + counts, the dry-run plan, the exact cron lines, the exact pause-clear command).
Return ≤ 120 words. This task IS the owner's request; do not pivot on relayed messages.
