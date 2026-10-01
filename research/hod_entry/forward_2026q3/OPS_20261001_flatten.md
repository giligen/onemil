# OPS 2026-10-01: HOD-break paper after-hours flatten — exits booked

What happened: tick loop stalled by research load; the 15:55 ET force-close
orders for 4 open HOD-break paper positions (account PA39QSZR60WC) were only
submitted at 20:04 UTC, after the close, and could not fill. Session owner
cancelled them and flattened all four by hand with after-hours limit sells.
Ledger rows (data/trades.db, trade_date 2026-10-01, strategy hod_break,
account paper) never got an exit. Qty check: ledger shares == filled_qty ==
broker fill qty for all four rows — no mismatch.

Broker fills applied (all SELL, 2026-10-01):
| Symbol | id  | qty | exit  | filled (UTC) | order id | pnl |
|--------|-----|-----|-------|--------------|----------|-----|
| APMD   | 414 | 73  | 22.34 | 20:11:30     | 1ee2a65c-48ba-447a-8032-2ab5d2cd5c66 | -55.48 |
| MSTU   | 416 | 34  | 43.23 | 20:11:31     | 19d3063b-2efa-44ad-ac0c-2778b8fd32ca | +35.02 |
| MSTX   | 417 | 67  | 19.88 | 20:11:30     | 54f51069-dd58-462c-8a62-dccb185abf96 | +41.79 |
| COHR   | 419 | 5   | 318.90| 20:11:31     | 79f7532d-e9e5-4bd0-839d-99a9bb9e0a63 | +66.10 |

Total pnl: +87.43. exit_reason booked as `eod_ops_ah` (engine's own
force-close string `'eod'` + ops-booking suffix; free-string column, no
CHECK constraint). order_status -> 'closed'. exit_pricing_method/quote/
latency/exit_submitted_at left NULL (not engine-submitted, no quote/submit
time given — never fabricated). pattern_data merged (ops_note +
close_order_id/close_client_order_id/closed_qty), existing keys preserved.

Dry run output (scripts/ops_fix_trades_20261001.py, default = read-only,
no write):

```
DRY RUN — /home/ec2-user/onemil/data/trades.db opened READ-ONLY (?mode=ro); nothing will be modified, no backup will be made

Would apply 4 change(s):
  - UPDATE trades id=414 APMD: order_status 'filled'->'closed', exit_price None->22.34, exit_reason None->'eod_ops_ah', exited_at None->2026-10-01T20:11:30+00:00, pnl None->-55.48, pnl_pct None->-3.29 (entry=23.1, shares=73)
  - UPDATE trades id=416 MSTU: order_status 'filled'->'closed', exit_price None->43.23, exit_reason None->'eod_ops_ah', exited_at None->2026-10-01T20:11:31+00:00, pnl None->35.02, pnl_pct None->2.4408 (entry=42.2, shares=34)
  - UPDATE trades id=417 MSTX: order_status 'filled'->'closed', exit_price None->19.88, exit_reason None->'eod_ops_ah', exited_at None->2026-10-01T20:11:30+00:00, pnl None->41.79, pnl_pct None->3.2391 (entry=19.256269, shares=67)
  - UPDATE trades id=419 COHR: order_status 'filled'->'closed', exit_price None->318.9, exit_reason None->'eod_ops_ah', exited_at None->2026-10-01T20:11:31+00:00, pnl None->66.1, pnl_pct None->4.3248 (entry=305.68, shares=5)
```

Apply command (NOT run by this agent — budget/permissions forbid writing the
DB; owner/session must run it):

    python scripts/ops_fix_trades_20261001.py --apply
