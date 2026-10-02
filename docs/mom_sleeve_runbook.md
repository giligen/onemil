# Weekly momentum PAPER sleeve - runbook

Rule: `research/momentum_weekly/RECON_1700_sleeve.md`, `PREREG_1700j.md` Amendment 1. Paper account only
(ALPACA_ORB_*); it shares that account with ORB and the TOM QQQ position and sells ONLY symbols in
`logs/mom_sleeve_state.json`. Code: `scripts/mom_sleeve.py`, `trading/mom_sleeve_{select,data}.py`.

## Cron (Monday 10/5 onward; times are UTC while New York is on EDT, until 11/1)
The script itself no-ops unless today is the first session of the week (holiday Monday -> Tuesday), so the
lines run Mon-Fri harmlessly. After the 11/1 DST change move the execute line to `41 14` (09:41 EST) and the plan
line to `5 13`. Do not run the plan after 09:25 ET: it refuses.
```
5 12 * * 1-5 cd /home/ec2-user/onemil && nice -n 15 /usr/bin/python3 scripts/mom_sleeve.py --plan >> logs/mom_sleeve_cron.log 2>&1
41 13 * * 1-5 cd /home/ec2-user/onemil && /usr/bin/python3 scripts/mom_sleeve.py --execute >> logs/mom_sleeve_cron.log 2>&1
```
(No `%` appears in either line; if you add `date +%F` anywhere, write `\%F`.) Check each line by hand first
with `--dry-run` appended. `--plan` takes ~4 min (5.8K-symbol bar fetch); `--execute` fetches no bars.

## Kill rule (logged, never auto-acted)
Sleeve value (positions in the state file only) vs its own high: drawdown >= 40 % -> ERROR line + Telegram
"KILL RULE ... halve the sleeve". The owner halves it by hand; the script keeps running at full size otherwise.

## Reading the ledger (`logs/mom_sleeve_ledger.csv`, one row per fill)
`fill_price` vs `open_0930` -> `slippage_bps` (positive = cost; buy: fill/open-1, sell: open/fill-1).
`prior_close` and `signal` are the inputs used; `note` = entrant / top_up / trim / left_top20 / liquidate.
The weekly Telegram line gives the mean slippage this run and all-time. Compare the all-time mean with the
backtest cost model (5 bps + half the (H-L)/C proxy, cap 20 bps). Plan inputs: `logs/mom_sleeve_plan_<date>.json`.

## Failure modes
- ERROR "no plan file": `--plan` did not run; no orders were sent; fix and re-run `--execute` inside 09:40-09:55.
- Completeness gate (universe < 90 % of last week's, or first-run fetch coverage < 90 %): abort, no orders.
- A sell not filled: buys withheld, pending list kept in the state file; re-run `--execute` in the window.

## Rollback
1. Delete the two cron lines (`crontab -e`).
2. `python3 scripts/mom_sleeve.py --liquidate --dry-run` to see the sells, then without `--dry-run` during
   market hours: sells every position in the state file (paper only, `momliq-<date>-<SYM>-sell`).
3. Confirm `logs/mom_sleeve_state.json` shows `"positions": {}`.
