# Hourly summary - RESULT (2026-10-09)

Files: `scripts/hourly_summary.py` (read-only; flags `--dry-run`, `--now-et`, `--once`), `tests/test_hourly_summary.py`.

Tests: 21 passed (unit with `spec=` mocks + a `gather()` integration through mocked clients: one failing account,
non-PA slot refused, LIVE never reads positions/orders, dry-run, Telegram unset, send True/False, line cap, no secrets).

Real dry run (paper accounts + LIVE account object only, 13:09 ET; re-run after the 10/9 ORB-week fix):
```
📊 HOURLY 13:09 ET (Fri 10/9)
ORB  $0 (0.0%) | 0 fills | flat   wk +$164
TOM  $0 (0.0%) | 0 fills | flat   wk $0
HOD  −$76 (−0.5%) | 15 fills | open 4: SOXS +$17 GRAL +$6 IONZ −$4 DDOG +$2 | worst UNHG −$54   wk −$29
MOM  +$234 (+0.4%) | open −$397 on $19.6K, 20 names   wk −$472
LIVE $0 | equity $64,812
```
Real send: `Telegram send_message_sync returned True` (logged 17:08:41 UTC, exit 0).

Week-to-date stamping verified: ORB bars stamped 10/06..10/09 = sessions Mon..Thu (stamp = NEXT UTC date); the last
bar's equity equals the account's `last_equity` (runtime WARNING `check_history_alignment` if that ever breaks).
WTD = history sessions Mon..yesterday + today's `equity - last_equity`. ORB/TOM share an account, so their `wk` comes from the orders API since Monday 00:00 ET (realized per symbol, QQQ -> TOM, rest -> ORB; TOM line shows on any QQQ fill this week). KNOWN GAP: TOM wk shows $0 for the 10/6 QQQ exit because its buy was in the prior week (no matching buy in the window); ORB wk +$164 is correct.

Deviations: day boundary = 00:00 ET (not UTC) so yesterday's after-hours orders are excluded. ORB/TOM split
under-counts a position opened on a previous day and closed today (no buy in today's orders) -> the $5 cross-check
logs a WARNING (it will fire on a TOM exit day). Paper guard also checks `is_paper` + SDK base URL.

Cron (NOT installed; 10:05-16:05 ET on EDT, becomes 09:05-15:05 ET after the Nov 1 shift to EST - use `5 15-21` then):
`5 14-20 * * 1-5 cd /home/ec2-user/onemil && /usr/bin/python3 scripts/hourly_summary.py >> logs/hourly_summary.log 2>&1`
