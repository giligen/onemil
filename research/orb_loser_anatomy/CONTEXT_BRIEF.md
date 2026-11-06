# CONTEXT BRIEF — ORB bottom-up loser anatomy: stop cost and early-exit triggers (2026-11-06)

This task IS the owner's request ("if we decrease the stop cost? or exit early on specific triggers? — bottom-up
research"). Method = the owner's loser-anatomy loop (research/H/METHOD.md): per losing trade, what actually happened on
the tape, cluster the losers, derive triggers from the clusters, then score each trigger paired vs the actual exit
under the frozen bar in PREREG_1723.md. Ranked hypotheses with sizes, not pass/fail only.

## The book (what ORB is)
Opening-range breakout, 09:30–09:35 range, pre-placed stop-limit buy at range_high + 30 bps (60-min cancel), stop at
range_low (= 1 R), static lock (price touches +1.75 R → stop to +0.5 R), time-boxed Rule M/D exits (`tag_bb`), force
close at 15:45 ET. Risk $375/trade, ≤ 8 slots. Rules: `research/orb_machine_rules.md` (L8 = exits). Engine:
`trading/orb_engine.py`; exits run in `trading/stop_monitor.py` (market stop on the 1-s websocket trade feed;
`stop_loss_market_fallback` = the stop-limit leg did not fill). Trading rules are FROZEN (PREREG_1709): this is
research only — nothing enters `orb.yaml` without the path in PREREG_1679's pass bar (rebuild → paper → 100 fills).

## Real ledgers (the population — `data/trades.db`, open READ-ONLY `file:...?mode=ro`)
`strategy='orb'`, `fill_price not null`, `exit_price not null`, `order_status in ('filled','closed')` (force-closed
rows stay `filled` — include them). Columns: trade_date, symbol, entry_price, fill_price, stop_loss_price, shares,
total_risk, pnl, exit_price, exit_reason, filled_at, exited_at (UTC), pattern_data (JSON: features at selection).
- L1 LIVE 2026-05-19..09-23: 123 fills, −$5,281. Exit mix: stop_loss 44 (−$11,523, −0.89 R), stop_loss_market_fallback
  4 (−$2,410, −1.11 R), tag_bb 38 (+$1,325, +0.12 R), lock_stop 12 (+$1,168), force_close 25 (+$6,158, +0.86 R).
- L2 PAPER 2026-09-25..11-06: 31 fills, −$136; stop_loss 16 (−$2,189, −0.84 R), force_close 11 (+$2,061), target_rested 3,
  lock_stop 1. 10/27 alone +$1,859. First-order latency settled at 5.5–6 s from 10/19 (was 7–27 s).
- Stops are 39 % (L1) / 52 % (L2) of fills vs 28 % in the BT; cell 1,709 showed the live fill price explains ≤ 8 % of
  that excess (`research/orb_exit_resim/REPORT_1709.md`). The excess is NOT explained yet — that is the gap.

## Bars and quotes (free, local)
- 1-min bars: `research/bf_zero/bars_sip.db` (SIP, table `bars`: symbol, ts, o,h,l,c,v — check `pragma table_info`)
  and `data/cache.db` (sparse, do NOT use for levels). If a symbol-day is missing, fetch IEX/SIP 1-min bars via
  `data_sources/alpaca_client.AlpacaClient.get_bars` with any key in `.env` (`load_dotenv('/home/ec2-user/onemil/.env')`,
  never print keys) into the scratchpad — never write bars_sip.db or cache.db.
- Measured costs: ORB entry quoted half-spread 15 bps, realized 64 bps (`research/exec_cost/REPORT_REALIZED.md`);
  stop slip measured 35 bps mean / 82 p90 on the HOD stop-market leg (cell 1,443), HOD stop-LIMIT 20 bps below the
  stop cut it to ≈ 13 bps (PREREG_1466) — the one exit mechanism that survived on another book. Live ORB stop fills in
  1,709's row trace slipped 2–35 bps. Compute the ACTUAL slip per ORB stop from the ledger (exit_price vs
  stop_loss_price) before assuming any number.

## Already tested — do NOT re-run these as-is (grep the files before adding a cell)
- `research/orb_exit/RESULT_1679.md` (real ledgers, R-floored): (a) lock +0.5 R at +1 R, (b) breakeven at +1 R,
  (c) trail MFE − 1 R, (d) 50 % out at +1 R, (f) 60-min time stop if < +0.5 R → ALL FAIL, L2 paired ΔR −0.28..−0.75 R,
  several opposite-signed between halves. Any lock/trail is a tighter stop in disguise: continuation forgone > give-back saved.
- `research/orb_exit/RESULT_1680.md`: 50 % partial at +1 R buys some consistency (green weeks) but fails weekly P10 /
  gap in one half — not shipped.
- `research/orb_inflight_early_cut_aug2026.md`: stagnation time-stop (MFE < +0.25 R within 20 min → market exit) PASSED
  on the BT book (drag +$4.8K TRAIN / +$0.5K VAL / +$1.5K OOS 2026, 100 % winner retention) but was never scored on
  the REAL ledgers and never shipped. It IS a candidate here — on the real ledger, paired.
- `docs/orb_research_apr_2026.md`: 50+ exit variants on the BT; the live lock 1.75/0.5 is the BT Pareto point.
- `research/orb_failure/RESULT_1564.md`: shorting the failed break at 10:30 — FAIL.
- Exits on HOD (`research/hod_exit_lab/REPORT.md`): 35 exit cells within ±0.04 R — a population without information
  shows that shape; if ORB losers cluster, this book is different; if they don't, say so.

## Standards (CLAUDE.md "No research claim ships without an independent check")
PREREG + FREEZE before scoring; TRAIN = L1 2026-05-19..08-15, VAL = L1 08-16..09-23 ∪ L2; a trigger is chosen on
TRAIN and read once on VAL. Paired ΔR vs the actual exit, day-clustered t beside iid, ex-top-5 % of ΔR, MDE = 2·SE.
Obtainability: an early exit fills at the NEXT 1-min bar's open minus 35 bps (market) or at the limit if touched
within the bar; a tighter stop is charged the measured slip. Report the continuation forgone (winners clipped) next
to the give-back saved for EVERY rule. Count every cell. Report-only, no `orb.yaml` change.
