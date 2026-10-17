# Week review 10/12-10/16 (Sat 10/17, service inactive when read, account ids 4 chars; facts only; HOD parity ledger has 275 rows 10/12-10/16; MOM 10/12 -413.42 vs ORB -413.14 on different accounts = coincidence, not investigated)
Sources: paper accounts via alpaca-py (closed orders `filled_qty`/`filled_avg_price`, window 10/12 00:00 ET onward; 1D portfolio
history, bar stamped D = session D-1), `data/trades.db`, `logs/session_archive/2026-10-1{2..6}.log` (ANSI-stripped, websocket lines
excluded), `logs/daily_green_check.log`, `logs/momentum_sleeve*.{log,csv}`, `logs/eod_report.log`, `logs/promotion_state.json`.
## 1. P&L (fills = realized per symbol-day pair; equity = 1D history delta; flag if |diff| > $20)
| Book (acct) | 10/12 | 10/13 | 10/14 | 10/15 | 10/16 | Week fills | Week equity | Fills n |
|---|---|---|---|---|---|---|---|---|
| ORB (PA3Y) | -412.99 | 0 | 0 | 0 | -280.32 | -693.31 | -693.72 | 10 |
| TOM (same acct, QQQ) | 0 | 0 | 0 | 0 | 0 | 0 | (inside the ORB acct) | 0 |
| HOD (PA39) | -40.50 | -150.42 | +47.15 | +419.15 | 0 (0 fills) | +275.38 | +234.68 (-40.92 -150.86 +46.78 +472.50 -92.82) | 83 |
| MOM (PA3N) | -413.42 | -12.27 | -559.25 | -357.72 | -703.04 | n/a (positions carried) | -2,045.70 | 24 (all 10/12) |
* ORB fills-vs-equity diff $0.41 total: OK. HOD diff $40.70 total: FLAG. 10/15 fills +419.15 vs equity +472.50 (diff +53.35) and the green
  check says +397: three numbers for one day. 10/16 HOD equity -92.82 with zero fills = mark-to-market of the open QCMU 63 sh (section 3).
  Fills-minus-equity over the week = QCMU unrealized -40.70 (63 sh bought at 20.157, never sold).
* MOM since inception (session 10/5 onward): +329.15 -578.60 -233.98 -223.26 +117.10 -413.42 -12.27 -559.25 -357.72 -703.04 = -2,635.29
  (= broker unrealized -2,634.99 on 20 names, MV $17,431). MOM account equity $53,239 (cash $35.8K is outside the $20K sleeve).
* LIVE (1 line): equity 10/9 close $64,812.12 vs 10/16 close $64,812.12 (daily P&L 0.00 every day).
## 2. ORB execution
| Day | First submit (engine log) | PARITY n_preplaced/kept/replaced/cancelled/added | Orders -> fills |
|---|---|---|---|
| 10/12 | 8.6 s (ASTX, ASTY preplaced) | 2/0/1/1/1 | ASTX preplaced, cancelled (not in final top-8 of 11); ASTY re-submitted 13:35:26 (26.4 s), APUS 13:35:27 (27.0 s); 2 fills |
| 10/13 | none (0 orders, 25 scored) | no preplace | 0 |
| 10/14 | 8.7 s (SNDQ) | 1/1/0/0/0 | SNDQ 323 sh limit 15.47, never filled, status time_stop_canceled |
| 10/15 | none (0 submitted, 39 scored lines) | no preplace | 0 |
| 10/16 | 8.5 s (PMI, CORD, IREZ) | 3/1/1/1/3 | 6 submits (IREZ replaced 18.37->18.63; PMI, IREZ, BKYI no fill); 3 fills |
* `ORB: first burst - ranking the full field` fired 0 times on all five days; `first_rank_grace ... waited on` 0 times. Commit 327c06f is dated
  10/10 02:18 UTC, before the 10/12 12:30:07 UTC boot (pid 1579097), but the archive has no version/startup line that names the loaded code,
  so "loaded" is not confirmable from logs. The line only logs when a subset is passed and the full field is larger; no day showed the 10/9 shape.
* Broker `submitted_at` (ET) vs engine stamp for the same filled order: ASTY 09:35:31.2 vs 26.4 s, APUS 31.7 vs 27.0, CORD 15.3 vs 8.1, SMST 19.9 vs 11.0,
  AIXI 24.7 vs 14.6 (broker is 4.8-10.1 s later; cause not traced). First-order latency range: engine 8.5-8.7 s (preplaced), filled orders 8.1-27.0 s
  engine-side / 15.3-31.7 s broker-side.
| Fill | Entry | Entry quote spread (DB) | Exit (DB branch) | P&L |
|---|---|---|---|---|
| ASTY 461 (of 798) 10/12 | 4.17 stop-limit, filled 09:42:27 | 0.03 (~72 bps) | stop_loss, limit sell 3.982 at 09:55 (stop 4.01) | -86.65 |
| APUS 882 10/12 | 5.67 stop-limit, filled 10:02:01 | 0.09 (~160 bps) | stop_loss, limit unfilled 10 s -> market 5.30 at 10:42 (stop 5.35) | -326.34 |
| CORD 1053 10/16 | 4.7306 | 0.01 (~21 bps) | stop_loss sl_leg 4.60 at 10:36 (stop 4.6184) | -137.52 |
| SMST 118 10/16 | 14.25 | 0.11 (~77 bps) | stop_loss, limit 13.86 at 10:17 (stop 13.8734) | -46.02 |
| AIXI 1613 10/16 | 3.05 (limit 3.10) | 0.02 (~63 bps) | stop_loss, market fallback 2.99 at 09:42 (stop 3.00), 1 min after fill | -96.78 |
* No lock stop, no time exit and no chase exit occurred; all 5 round trips are stop_loss (1.07-1.17 R each incl. exit slippage). 10/12 -413: ASTY -86.65
  + APUS -326.34, both stop-outs, scale-out (+3R) armed on both and never reached. No rule misapplication found in the entry/exit path; one logged
  oddity: `ADD APUS: blocked - past the 15:00 ET cutoff` at 14:08:00 UTC = 10:08 ET (section 7). The green check on 10/12 was GREEN (1 BT pick, parity clean).
* 10/16 RED = `composite drift BT vs live (1 of 12): RGC live -0.4672 vs BT -0.4513 (delta +0.0159)`. Composite drift compares the live `ORB SCORED` /
  `below filter threshold` composite with the BT `composite_score` on the same features (`scripts/report_common.py:617-641`). Constants: tolerance 0.005,
  NEAR_THRESHOLD_BAND 0.5, HARD_DRIFT_ABS 0.15. delta 0.0159 is 3.2x the tolerance, same sign, below the 0.15 hard bound, but min|comp| 0.4513 < 0.5 so the
  code classifies it `hard`. Both systems reject RGC (negative composite; the audit also lists RGC under spread_gate_would_skip): no decision changed.
  Verdict: not tolerance noise by the check's own definition, and not a decision defect; it is a band-rule classification. Cause of the 0.016 not traced.
  Result: streak reset to 0 and a RAMP FREEZE on ORB stamped 10/16.
* Audit `orb_selection_audit_*.json` `drop_classification` marks ASTX, ASTY (10/12) and CORD, AEHL (10/16) "REAL DROP", and BEZ (10/14); ASTY and CORD were traded. Audit 10/16 `spread_gate_would_skip` = AEHL, AIXI (snapshot 328 bps), CUK, RGC, yet AIXI was entered (DB entry quote 3.19/3.21).
## 3. HOD (config.yaml: `enabled: true`, `dry_run: false`, `resting_stop_limit`, `risk_usd: 50`)
* Real orders on the HOD paper account (PA39), not dry. Fills/day 20/22/20/21/0. 41 round trips: 18 winners, 23 losers. Worst AVXX 10/14 -62.10; best ASTN 10/14 +104.00
  (also STXL 10/13 +103.74, CRWU 10/15 +100.46). Green check HOD: -41, -150, +47, +397, 10/16 not printed; fills -40.50/-150.42/+47.15/+419.15/0.
* Warnings: 1 `orphan` (MSTX 10/12 14:37, "StopMonitor exit event but no tracked position"), 4 `stop leg replaced ... following it` (MSTX, MAZE, AVNT 10/12 +1 later),
  21 StopMonitor `limit sell UNFILLED after 10s -> market close`, 8 PARTIAL fills, 2 `foreign shares present`, 2 `sync: row holds N sh but broker holds M`.
* 10/15 QCMU: buy 63 sh at 20.157 (09:37 ET), partial fill 58/63 then 60/63; two ERRORs at 13:37:28 UTC: safety-net OCO rejected (available 2, held_for_orders 58) and
  stop leg rejected (wash-trade); no exit in the week, position still open on 10/16 close. 10/16: 1,179 `LIVE order blocked by kill rail (unverified_exit)` warnings
  (first 09:36 ET), 0 HOD fills; ERROR at 12:30:09 UTC `reconcile RBRK (2026-10-15): 0/17 shares accounted ... needs a human look`.
## 4. Sleeve (momentum, paper PA3N)  10/12 rotation 2: BT top-20 = picks 20/20; 11 sells ($4,646 notional; DELL, NUE, STT, TGT full exits) then 13 buys ($4,658; DOCN, FRO, ROIV, UMC at $966 each), market
  orders 09:45:56-09:46:02 ET. Intended 24 / filled 24, 0 rejected, 0 partial, reconcile broker 20 names $19,048 = state (OK each day 10/12-10/16).
* Slip vs official open: ledger/EOD mean -35.2 bp (gate <= 20 bp on the signed mean), max |464.9| (LITE); my recompute from the ledger: signed mean -31.1 bp, mean absolute
  109.5 bp. Gate `promotion_state.json` sleeve 10/5 true, 10/12 true; EOD line `GO LIVE $20K on 2026-10-19 [2 clean rotations ...]`. Execution-clean by the gate: YES.
* Warning 10/12 13:46:11 UTC `state resynced from the broker - positions changed for 24 names; cash 3.82 -> -7.82`. Completeness LOST 311 (2.3 %), liquid 791/791.
* Week P&L -2,045.70; since inception -2,635.29 (sleeve equity 19,316.93 at the 10/12 mark vs 20,000 start, ledger peak 20,102.96).
  EOD "MOM P&L day -786 week -786" on 10/12 vs broker 10/12 day -413.42; EOD says `NO-DATA (no equity mark ...)` on 10/13-10/16.
## 5. TOM: No QQQ order or fill this week; last ledger rows 10/7. `logs/tom_sleeve_cron.log` 10/16 20:58: outside the 15:40-15:55 MOC window, no-op. Confirmed none.
## 6. Hygiene (WARNING / ERROR per archive day, websocket excluded)
| Day | WARNING | ERROR | pid |
|---|---|---|---|
| 10/12 | 32 | 0 | 1579097 |
| 10/13 | 45 | 0 | 1759406 |
| 10/14 | 31 | 0 | 1940836 |
| 10/15 | 32,725 (32,677 HOD `mover queue full`) | 27 (19 `NO BARS INGESTED`, 5 limit-sell rejects, 2 QCMU safety-net, 1 other) | 2122204, 2180377 |
| 10/16 | 1,182 (1,179 kill rail) | 1 (RBRK reconcile) | 2299335 |
* Other distinct messages: HOD `arm-time feature computation failed ("arm has no 'idx'")` x42; counterfactual watch closing EOD x29; `backfill returned no bars for 1 of 200 symbols` x12; parity-ledger header predates counterfactual columns x4; `no live print stream - resolving the cross` x4; ORB `anchor: RTSN unwarmed in submit loop` x1; ORB `PREPLACE ... REPLACED` x2; HOD `already carries a fill before any cancel` x2.
* Telegram-sent ERROR: 0 matching lines. Unscheduled restart: 10/15 17:02:41 UTC `Main process exited, code=exited, status=1/FAILURE`, systemd Started 17:03:11
  (app log ends at DEBUG scanner lines, no traceback; cause not found). Other starts are the 12:30 UTC boots (10/16 12:30:01 `Started`).
* Hourly-summary cron: NOT installed (`crontab -l` has only a comment mentioning hourly, 0 active entries).
## 7. Open defects (evidence only)
1. HOD QCMU 63 sh left without exit/protection since 10/15 (OCO + stop rejected 13:37:28 UTC); still open at 10/16 close, equity -40.70 on it.
2. HOD kill rail `unverified_exit` blocked every live order on 10/16 (1,179 warnings, 0 fills); RBRK reconcile ERROR "needs a human look" 10/16 12:30:09.
3. HOD feed degraded 10/15: 32,677 `mover queue full` drops from 15:46 UTC, 19 `NO BARS INGESTED` ERRORs (continuing after restart), service exit status 1 at 17:02:41 UTC.
4. HOD parity ledger written in the OLD column shape (header warning x4); arm-time feature columns empty x42 (`no 'idx'`).
5. ORB `ADD APUS: blocked - past the 15:00 ET cutoff` logged at 10:08 ET (14:08 UTC), before the cutoff.
6. ORB broker `submitted_at` is 4.8-10.1 s later than the engine's submit stamp for the same order; the engine "first order submit 8.5-8.7 s" understates broker-side latency.
7. Commit 327c06f "first burst" line absent all week and no startup line shows the loaded code version; fix unverified in production logs.
8. ORB audit marks traded names (ASTY, CORD) and preplace-cancelled ASTX as "REAL DROP"; ASTX preplaced then cancelled (not in final top-8 of 11) with no reason line extracted.
9. Green check RED 10/16 is a band-rule `hard` classification of a no-decision-change drift (RGC, delta 0.0159); it reset the streak and froze the ORB ramp.
10. APUS entered with DB entry-quote spread 0.09 (~160 bps) vs the 150 bps spread gate; AIXI entered while the audit snapshot says 328 bps (gate sources disagree, not reconciled).
11. EOD MOM P&L: `NO-DATA` 10/13-10/16 and "day -786" on 10/12 vs broker -413.42; sleeve slip gate uses the signed mean (-35.2) while mean absolute is 109.5 bp.
12. HOD 10/15 day P&L three ways (green check +397, fills +419.15, equity +472.50).
