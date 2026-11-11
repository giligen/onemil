# Independent rebuild 2026-09-30..2026-11-10 (script: scratchpad rb.py, sqlite ro)
DEVIATION: order_status='filled' literally yields ORB n=10, HOD n=0; finished trades carry order_status='closed' (HOD 288, ORB 21) so closed fill = status in ('filled','closed') AND fill_price/exit_price not null. ORB account='paper' for all rows.
| | ORB | HOD |
|---|---|---|
| closed fills / days / symbol-days | 31 / 14 / 31 | 288 / 29 / 287 |
| sum pnl | +8.36 | -150.49 |
| mean R / median R | -0.061 / -0.876 | -0.008 / -0.666 |
| win rate | 32.3% | 38.9% |
| ex best fill | -940.64 | -257.31 |
| ex top 5% (k=2 / k=15) | -1437.33 | -1700.06 |
| stop rate (reason contains stop) | 58.1% (18/31) | 51.7% (149/288) |
ORB ex CRDU 10/27 (+949.00) and BEX 11/9 (+47.85): n 29, pnl -988.49, mean R -0.313, 30 distinct symbol-days since 9/25 excl. those two.
HOD broker equity: 2026-09-30 105023.88 -> 2026-11-10 105019.47, diff -4.41 (ledger -150.49).
exit_reason: ORB stop_loss 16, force_close 10, target_rested 3, lock_stop 2; HOD stop_loss 128, target 68, eod 60, stop 20, squeeze 7, eod_ops_ah 4, stop_loss_unconfirmed 1.
