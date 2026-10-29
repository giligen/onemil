Population: fills_1702 123 rows; bars found 123 (sources {'sip': 123}); uncovered 0 (dates []..., all after the bars_sip.db end 2026-09-28; no fallback).
Covered by account: {'live': 123}. ATR source: {'range_low': 115, 'pattern': 8}. Ledger stop == sim initial stop (ii, 1c): 100%; == range_low alone: 96%.
| book | n | stop share | mean R [day-boot 95 % CI] | lock | tag | force |
|---|---|---|---|---|---|---|
| actual ledger | 123 | 39.0% | -0.109 [-0.306,+0.122] | 9.8% | 30.9% | 20.3% |
| (i) BT entry, main | 123 | 34.1% | -0.133 [-0.368,+0.130] | 9.8% | 30.1% | 26.0% |
| (ii) live fill, main | 123 | 35.0% | -0.110 [-0.357,+0.174] | 9.8% | 29.3% | 26.0% |
| (i) BT entry, ebstop | 123 | 35.0% | -0.145 [-0.378,+0.118] | 8.9% | 30.1% | 26.0% |
| (ii) live fill, ebstop | 123 | 35.8% | -0.122 [-0.371,+0.161] | 8.9% | 29.3% | 26.0% |
| (i) BT entry, fc1555 | 123 | 35.0% | -0.141 [-0.370,+0.114] | 9.8% | 30.1% | 25.2% |
| (ii) live fill, fc1555 | 123 | 35.8% | -0.118 [-0.364,+0.155] | 9.8% | 29.3% | 25.2% |
| (i) BT entry, cost10 | 123 | 34.1% | -0.150 [-0.382,+0.110] | 9.8% | 30.1% | 26.0% |
| (ii) live fill, cost10 | 123 | 35.0% | -0.128 [-0.374,+0.155] | 9.8% | 29.3% | 26.0% |
H1 numbers (main): ledger stop share 39.0% (published BT 28.4 %) -> excess +10.6 pp; (i) 34.1%, (ii) 35.0%; (ii)-(i) = +0.8 pp = 8% of the ledger-vs-published excess; (ii)-ledger = -4.1 pp.
Exit-reason agreement (ii main vs ledger): 81.3% of 123. Confusion (rows=ledger, cols=sim):
`ii_main_reason  force_close  lock  stop  tag_bb / act_cls                                         / force_close              23     0     0       2 / lock                      0    11     0       1 / stop                      1     0    40       7 / tag_bb                    8     1     3      26`
Hand-check (5 fills; ledger vs sim (ii) exit time UTC / price / reason):
* id 131 WAY 2026-05-19 fill 19.880 stop 19.230: ledger tag_bb 13:47:16 @ 19.997; sim force_close @ 19.670 19:44:00 (sim stop 19.230)
* id 134 LMRI 2026-05-19 fill 7.290 stop 6.860: ledger force_close 19:45:01 @ 7.010; sim tag_bb @ 7.355 13:39:00 (sim stop 6.860)
* id 139 YANG 2026-05-21 fill 29.780 stop 29.500: ledger stop_loss 13:54:17 @ 29.493; sim stop @ 29.387 13:54:00 (sim stop 29.500)
* id 148 TRT 2026-05-26 fill 15.530 stop 14.530: ledger lock_stop 18:09:44 @ 16.288; sim lock @ 15.949 18:09:00 (sim stop 14.530)
* id 257 RUN 2026-06-24 fill 16.165 stop 15.290: ledger stop_loss_market_fallback 17:46:41 @ 15.255; sim stop @ 15.236 17:45:00 (sim stop 15.290)
