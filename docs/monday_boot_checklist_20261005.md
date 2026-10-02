# Monday 2026-10-05 — boot checklist (ORB paper proof day + first scheduled momentum rotation)

All times UTC. Service: cron stop 10:25, start 12:30, exits ~20:05. One research agent at a time during market hours,
every python run through `bash scripts/research_run.sh` (cage). Nothing is restarted by hand unless a line below says so.

## Before the boot (by 12:25)
1. `git status --short | grep -v '^??' | grep -E 'trading/|scanner/|data_sources/|scripts/|main.py'` → empty
   (the service must not boot onto uncommitted engine edits; a half-finished edit is saved as a patch and restored).
2. Pre-boot test cron result on Telegram: green. If red: fix or revert before 12:30.
3. `logs/momentum_sleeve.log` after the 11:50 prefetch: `COMPLETENESS … LOST` line present, a `guard:` line, a
   `gate ON|OFF (pNN) size NN%` line with NO `STALE` warning (Friday's VIX close must be in), plan printed, no orders.

## 12:30–12:40 boot
4. `journalctl -u onemil-trader --since 12:30 | grep -E "ORB RVOL tilt:|add_on|addon_gap35_range5|ERROR"` →
   tilt line (edges 2.898/5.966, mults 1.5/1.0/0.5), add-on enabled at +1 R, P1 pool enabled, zero ERROR.
5. Orphan line: QQQ 26 sh on the ORB paper account is the turn-of-month sleeve (exits today 19:45) — expected.
6. HOD paper account flat at boot (flattened after hours 10/2).

## 13:30–13:40 the ORB open (first live proof of the stall fix and the all-day-rebuild fix)
7. `journalctl -u onemil-trader --since 13:29 | grep -E "open-tick budget|prewarm cache (MISS|STALE)|WORK took|ORB GAP_GATE:|Engine tick TIMEOUT"`
   → no `open-tick budget` ERROR, no `WORK took` above 60 s, at most ONE `ORB GAP_GATE:` WARNING per build (aggregated
   count), zero `Engine tick TIMEOUT`.
8. Decision made by 13:36: ORB picks logged with pool_id and tilt multiplier; entries preplaced.
9. After 14:00 (10:00 ET): exactly one `ORB: past last_entry_submit_time … universe build + gap gate skipped` INFO;
   no further `ORB universe seed` lines; cycles back under 60 s.

## 13:45 / 14:45 momentum sleeve (paper account PA3NDODOGPC2)
10. `[MOM]` Telegram line: fills n/n, `size 100%|50%`, gate state. Expected on Friday's ranking: out GH MRK ROIV TRGP,
    in DELL DINO NUE STT (may differ with Friday's final bars). Ledger `logs/momentum_sleeve_ledger.csv` has today's rows
    with `size_pct`; `logs/momentum_sleeve_shadow_gate.csv` has one row per submit run.
11. Any `REJECTED` line → read it, the run continues by design; state is resynced from the broker at the end.

## 19:45–20:10 close
12. TOM sleeve sells QQQ at 19:45 (third session of October).
13. ORB parity read: picks AND fills vs the nightly BT book (`research/orb_freq/paper_2026-10-05_picks.md`);
    add-on events `[ORB] ADD`; exits with reason and P&L. HOD/ORB paper accounts flat after 15:55 ET.
14. ERROR / TIMEOUT counts for the day; disk ≥ 5 GB.

## Fixed on 10/2 evening — Monday is each fix's first live proof (paper only)
* ORB gap gate input = today's official open (commit 2044a17, `research/orb_freq/gap_input_parity_20261002.md`):
  `grep -E "ORB GAP_GATE:"` after 13:35 → at most one aggregated WARNING per build naming symbols with no valid
  today-open (not admitted); the parity read lists every pick with its gap input source (snapshot today / 09:30 bar).
* HOD kill rail + restart quantity (`docs/hod_killrail_restart_qty_20261002.md`, greps in its last section):
  `KILL RAIL (…): cancelling n resting entry order(s)` appears once when a rail trips; any
  `registry qty … < broker qty … — registry corrected` WARNING is read the same day; the HOD paper account is flat
  after 15:55 ET with no after-hours clean-up needed.
* A second Claude process writing to the tree is checked BEFORE any agent is launched:
  `ps -eo pid,etimes,args | grep -E "claude|2\.1\.[0-9]+ --session-id" | grep -v grep`.
