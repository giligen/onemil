# book_status skill - RESULT (2026-11-02)

Files: scripts/book_status.py, .claude/skills/status/SKILL.md (+ .gitignore un-ignore for .claude/skills/status/), tests/test_book_status_20261102.py.

Tests: new file 8 passed; suite (`--ignore=tests/integration -p no:randomly`) 5487 passed, 6 skipped, 0 failed (311 s).

Deviations: runtime is ~16 s, not <= 5 s - the journal holds ~2.3M lines/day; `journalctl -g` pre-filter halves it and nothing faster is possible from the journal (logs/onemil.log rotates every ~17 min). `logs/cron_*.log` do not exist, so cron status uses each job's own log (mtime + ERROR count today).

Real output (15 lines):

```
STATUS 2026-11-02 (ET)
service: active | boot 2026-11-02 12:30:13 | ERROR/CRITICAL today 0
  HOD gates: dry_run=False risk=$50 per_day=12 concurrent=4
HOD paper: 5 fills, closed -$47, open 4 (UPBD,FTAI,NXT,WING) | stop 1 tgt 0 sqz 0 eod 0
  WTD -$47 (5) | since 2026-09-29 -$256 (238) | ARM SKIP r_pct 4007 min_price 664 other 0 (137 syms) | cap 83 | SQUEEZE lines 0
ORB paper: 3 fills -$388 today | WTD -$388 (3) | PDR VETO 9 PARITY 0 | ramp not frozen
MOM paper: 2026-11-02 eq $16,416 in 3 out 3 | -17.9% vs $20,000 | pause rail +$1,416 halve rail +$4,416
MOM live: 2026-11-02 eq $9,174 in 3 out 3 wk_r -8.22 since_strong 2 | -8.3% vs $10,000 | pause rail +$1,674 halve rail +$3,174
  parity: 2026-10-26 names 17/20 resid +1.02 pp
  last log: momentum_sleeve_live.log: 14:46:06,870 WARNING data_sources.alpaca_client: get_all_assets: SDK rejected the payload (10 validation errors for list[Asset]; 
paused books: BF trading.enabled=False | R2G red_to_green.enabled=False
cron mom: 50 11 * * 1-3, 45 13 * * 1-3, 45 14 * * 1-3, 46 13 * * 1-3, 46 14 * * 1-3 | momentum_sleeve.log 11-02 14:45 err 0; momentum_sleeve_live.log 11-02 14:46 err 0
cron eod: 4 21 * * 1-5, 57 21 * * 1-5 | eod_report.log 10-30 21:58 err 0
cron refresh: 0 13 * * 6 | mom_panel_refresh.log 10-31 13:00 err 0
cron parity: 20 13 * * 6 | mom_bt_parity.log 10-31 13:20 err 0
```
