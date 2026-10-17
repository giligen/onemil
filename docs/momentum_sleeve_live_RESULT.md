# Momentum sleeve LIVE path - RESULT (2026-10-17)

## What changed
* `trading/momentum_live.py` (new, pure): paths, sleeve view, collisions, sell cap, state-tracked fills, rails, cadence, MOM-LIVE line.
* `scripts/momentum_sleeve.py`: `--live`, `--cadence-warn-weeks` (8), `--equity-start` (REQUIRED with --live, paper default 20000 kept).
  Paper path untouched (strict `args.live is True`; `execute`/`sync_state_from_broker` unchanged).
  Live: `assert_live_account` (not paper, account not `PA*`, logs 4 chars), `execute_live` (state-driven sells capped at state qty,
  broker-less -> state qty set to broker's; buying power >= buy notional after sells else ERROR+Telegram, no buys; `moml-` coids; fills
  booked into state cash/positions, saved after each phase), `live_guards` (collisions, drawdown rail, "N non-sleeve positions ignored"),
  `submit_live` (ledger/weekly/gate to the separate `*_live_*` files, [MOM LIVE] Telegram, cadence R + weeks-since-strong columns).
* `scripts/hourly_summary.py`, `scripts/eod_sections.py`, `scripts/eod_report.py`: `MOM-LIVE` line from state_live.json + marks of the
  state's names only; state file absent -> omitted.
* Dry run (`--live` without `--submit`) writes no state/ledger/weekly/gate (test-enforced; only the shared bars cache is written).
* Rail limit: completeness refusal still runs first (a data outage delays a liquidation to the next run).

## Tests
`tests/test_momentum_sleeve_live.py` 18 new (unit + run() integration on a fake broker). With `test_momentum_sleeve.py` (69),
`test_hourly_summary.py`, `test_eod_sections.py`, `test_eod_report.py`: 181 passed, 0 failed.

## Real dry run (LIVE account, read-only, no --submit)
asof 2026-10-16, completeness 98% / liquid 789/789, gate OFF (p90) size 100%, sleeve equity $10,000, $500 per name.
Selected 20: AXTI SNDK TXG MU TWST LITE RVMD VLO ASX DELL MPC PSX UMC FRO STX DINO ILMN IOVA TRGP WDC.
Orders: 20 buys, $10,000 notional, 0 sells. Collisions: none. `0 non-sleeve positions ignored`.
Buying power $259,248 vs buy notional $10,000: OK. Sleeve cash (state) $10,000. Rail at $7,500. Weeks since strong week 0. Nothing written.

## Cron lines (NOT installed; paper lines stay). Shared bars cache = `data/momentum_sleeve/daily_<asof>.parquet` (same for both modes)
```
50 13 * * 1,2 cd /home/ec2-user/onemil && /usr/bin/nice -n 19 /usr/bin/python3 scripts/momentum_sleeve.py --live --submit --skip-fetch --equity-start 10000 >> logs/momentum_sleeve_live.log 2>&1
50 14 * * 1,2 cd /home/ec2-user/onemil && /usr/bin/nice -n 19 /usr/bin/python3 scripts/momentum_sleeve.py --live --submit --skip-fetch --equity-start 10000 >> logs/momentum_sleeve_live.log 2>&1
```

## Pause clear (after a drawdown rail: state `paused: true`, run exits 3 and Telegrams PAUSED)
```
cd /home/ec2-user/onemil && python3 -c "import json; p='data/momentum_sleeve/state_live.json'; s=json.load(open(p)); s['paused']=False; s.pop('paused_reason',None); json.dump(s,open(p,'w'),indent=2,sort_keys=True)"
```
