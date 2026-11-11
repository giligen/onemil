# Hourly summary: percent next to every dollar - RESULT 2026-11-11

Helper: `money_pct` at scripts/hourly_summary.py:108 (bare $ when base missing or <= 0). HOD wk base = bought notional of
the week fills (already fetched by collect_book; a failed non-ORB week fetch now WARNs and leaves wk bare).
MOM-LIVE: `_pct_suffix` in trading/momentum_live.py (day on prior-close slice, week on start-of-week, since start).

Tests: tests/test_hourly_summary.py + test_momentum_sleeve_live.py + test_mom_bt_current: 75 passed, 0 failed
(7 new; 2 old-format assertions updated).

Printed message (`--dry-run`, real accounts, read-only):
```
HOURLY 15:17 ET (Wed 11/11)
ORB (since 11/12, A2) $0 (0.0%) | 0 fills | flat   wk $0 (0.0%)
EXT (micha) +$67 (+0.5%) | 2 fills | open 4: AAPL JPM MRK UNH
HOD −$48 (−0.4%) | 13 fills | open 3: VOGX −$38 (−2.6%) AMDL −$30 (−1.3%) AXSM −$15 (−1.1%) | best BEG +$101 (+7.9%) | worst MRVU −$56 (−3.2%)   wk +$46 (+0.1%)
MOM +$712 (+1.4%) | open −$3,373 (−18.3%) on $15.1K, 20 names   wk +$1,229 (+2.5%)
LIVE +$399 (+0.6%) | equity $63,190
MOM-LIVE: day $+399 (+5.0%) | week $-760 (-8.3%) | since start $-1,586 (-15.9%) (slice $8,414 of $10,000) | DD from peak 15.9 %
```
