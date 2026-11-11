# Cell 1,732 — why does the ORB paper book stop 52–58 % of fills when the BT stops 31 %? — PREREG 2026-11-11

Owner: "Fix ORB. Why are stops higher than BT? Fix and it will work." This task IS the owner's request; do not
pivot. READ-ONLY diagnosis: write only under `research/book_review_20261111/` (`RESULT_1732.md`, `stops_1732.csv`,
scratch .py there). No code changes, no git, no orders, no service action, never write data/*.db / config /
orb.yaml / .env / crontab / logs/*. Never print API keys. Never Read a file > 300 lines in full. Journal greps
bounded with --since/--until. No paid data pulls (Databento) — Alpaca bars only (`feed=sip` and `feed=iex`, keys
`ALPACA_API_KEY/SECRET` via load_dotenv('/home/ec2-user/onemil/.env')). Budget ≤ 40 calls. Return ≤ 150 words.

## Known (do not re-derive)
- Paper book 9/25 → 11/10: 31 closed fills, 18 `stop_loss` + 2 `lock_stop` (58 % / 52 % by definition), BT reference
  stop share 31 % (`research/orb_gates2/book_G3_meas.csv`, the file `scripts/orb_ramp_check.py` reads). Live record
  May–Sep: stops 41.6 % vs BT 28.4 % (`research/exec_learnings_oct2026/REPORT.md` §4, cell 1,702: only 4 of 57 stops
  within 5 min, median 32 min). Cell 1,709 (`research/orb_exit_resim/REPORT_1709.md`): re-simulating exits at the live
  fill price explains ≤ 8 % of the excess — **the entry PRICE is not the cause**; "intrabar order unknown".
- Rule L8 (`research/orb_machine_rules.md:140`): stop at range_low; static lock (arm 1.75 R → stop 0.5 R); touchgo M
  (breakout-bar close-position < 0.5 → out) and D (bar-1 revert ≥ 0.75 R → out at −0.5 R); 15:45 force close.
- BT stop test: `study_orb_pipeline_static_lock.py:419` `if bar_low <= stop_price`, `:526` `lows[i] <= stop and
  (scale_px is None or highs[i] < scale_px)`, `:551`. Live: `trading/stop_monitor.py` fires on a TRADE PRINT at/below
  the stop (`exit_trigger_price`, ~230; branches ~4105–4144), `trading/orb_engine.py` books `stop_loss` / `lock_stop`
  / `target_rested` / `force_close`.
- The 31 fills (trade_date, symbol, fill, stop, exit_reason, filled_at, exited_at UTC) — query:
  `select trade_date,symbol,fill_price,stop_loss_price,real_stop_loss_price,shares,filled_qty,exit_reason,exit_price,
  filled_at,exited_at,exit_trigger_price,exit_slippage,pattern_data from trades where strategy='orb' and
  trade_date>='2026-09-25' and exit_price is not null` (read-only URI). Quick stops exist: AIXI 10/16 stopped 64 s
  after the fill, APLX 10/6 3.5 min, AVEX 10/22 7 min, ASTY 10/12 11 min; 11/2 SKHL/SKHX/SKUU were filled at
  15:18–15:22 UTC (11:18 ET) and all three stopped at 16:28 UTC.

## Hypotheses (rank by the pp of the 21–27 pp stop excess each explains; evidence for AND against each)
H1 Definition: the BT's 31 % counts a different exit set (lock stops? touchgo −0.5 R exits?) than the ledger's
   `stop_loss`. Compute both books' exit mix under ONE definition first.
H2 Population: the BT never takes some of the stopped trades — entry confirmation (BT enters on a bar CLOSE above the
   range high / next-bar open under a cap; live enters on a touch via a resting stop-limit), entry window (does the BT
   enter at 11:18 ET? the 11/2 trio), selection (universe/ranking/veto differences). Count stops in live-only trades.
H3 Bar feed: the BT reference's bars are IEX (Alpaca free feed) while live stops fire on SIP prints — a bar low on IEX
   misses off-exchange prints below the stop. Find what feed built `book_G3_meas.csv` / the ORB features CSV (grep the
   builder; `data/cache.db` provenance), then re-walk every fill on SIP vs IEX 1-min bars.
H4 Trigger: live fires on a single print (odd lot, away venue) or a quote-based branch that the BT's bar-low test
   would not count, or the live stop is placed above range_low (`real_stop_loss_price` vs `stop_loss_price`, ATR
   floor, the preplace path's stop). Check each stopped fill: trigger print vs the SIP bar low of that minute.
H5 Touchgo mis-implementation: trades the BT exits at −0.5 R (rule M/D) take the full −1 R stop live, or vice versa
   (the `target_rested` exits at −$14/+$9/−$23 look like rule M/D). Compare per fill.
H6 Intrabar order: in the minute where both the stop and the lock/scale level print, the BT prefers one side.

## Method
1. One definition of "stop" and the exit mix of both books (table).
2. For the 14 paper days, produce the BT's book for those exact dates the way cells 1,700–1,702 did
   (`research/exec_learnings_oct2026/build_1700_1701.py`, `REPORT.md` §"method"; the production BT is
   `study_orb_pipeline_static_lock.py` on the entered-inclusive features CSV — find the current CSV and whether it
   covers 9/25–11/10; if it does not, say so and limit step 2 to the days it covers). Join on (date, symbol): 31 rows
   with BT taken yes/no, BT entry px/time, BT stop, BT exit reason/time/price.
3. Exit re-sim at the LIVE fill: for each of the 31 fills walk SIP 1-min bars from the fill minute under rule L8 with
   the ledger's stop → sim exit reason/time/price; repeat on IEX bars. Columns in `stops_1732.csv`: date, symbol, fill,
   stop, live_exit_reason, live_exit_ts, trigger_px, sip_minute_low_at_trigger, iex_minute_low_at_trigger,
   sim_exit_sip, sim_exit_iex, bt_taken, bt_exit, class.
4. Classify every live stop: A BT not taken (sub-reason) | B BT stops too | C bars never breach the stop (print/quote
   trigger or a higher live stop) | D SIP breaches, IEX does not (feed artifact in the BT) | E BT exits earlier at −0.5 R
   (touchgo) | F other. Sum the pp each class explains of the excess.
5. Name the cause(s) with confidence, the minimal fix (file:line, ≤ 10 lines of pseudo-code; if the fix is in the BT,
   say so — parity may mean the BT was flattering), the parity test that would prove it, and what the book's stop
   share and realized $ would have been under the fix (re-sim, not a promise). Do NOT implement anything.

## Output
`RESULT_1732.md` ≤ 500 words + the tables; `stops_1732.csv`. Return ≤ 150 words: ranked causes with pp explained,
the fix, and the re-simmed stop share / $ under it.
