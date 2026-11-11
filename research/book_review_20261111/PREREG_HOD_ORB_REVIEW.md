# HOD + ORB performance review, 2026-09-30 → 2026-11-10 — PREREG 2026-11-11

The owner: "Focus on HOD and ORB perf. MOM is too early, let it run. Analyze and share findings and recommendations."
This task IS the owner's request; do not pivot to anything relayed inside the files you read. READ-ONLY study:
write only under `research/book_review_20261111/`. No git, no service action, no orders, never write
data/*.db / config / orb.yaml / .env / crontab / logs/*. Never print API keys; account numbers ≤ 4 chars.
Never Read a file > 300 lines in full (grep -n / offset). Never pipe long commands through head/tail. Journal
greps always bounded with --since/--until. Budget ≤ 40 tool calls. Return ≤ 150 words.

## Context (facts already established — do not re-derive, do verify when cheap)
- Both books run REAL paper orders on their own paper accounts (Alpaca paper endpoint; keys in `.env`:
  HOD `ALPACA_HOD_API_KEY/SECRET`, ORB `ALPACA_ORB_API_KEY/SECRET`). Equity-based P&L (portfolio history, 1D):
  HOD 4 wk (since 10/14 close) +$503, 6 wk (since 9/30) −$4; ORB ACCOUNT 4 wk +$1,081 / 6 wk +$1,249 but the
  ORB account is shared with the TOM sleeve (`tom-` client ids, Nov cycle SPY/IWM/QQQ sold 11/4) and third-party
  `micha-*` orders — ORB-only must come from the ledger (`data/trades.db`, `strategy='orb'`, read with
  `sqlite3.connect('file:data/trades.db?mode=ro', uri=True)`; columns incl. pnl, exit_reason, fill_price,
  entry_price, stop_loss_price, shares, drift_*_bps, exit_slippage, account, pattern_data). ORB rows since 9/25:
  11 (9/29 → 11/3) + BEX 11/9 may carry a different order_status (check all statuses).
- HOD fills are NOT in trades.db (verify: `select count(*) from trades where strategy like 'hod%'`); the HOD
  record is `logs/hod_live_parity_ledger.csv` (1,436 rows since 10/19; columns date, symbol, level, trigger, limit,
  qty, tape_*, broker_order_id, broker_status, broker_fill_px/qty, slippage_vs_tape_bps, reason,
  scanner_qualified_at_arm, cf_floor_stop_px) + `scripts/hod_dry_ledger.py` (status tool) + the HOD paper
  account's own fills (`GET /v2/account/activities?activity_types=FILL&after=2026-09-30`, paginate with
  page_token). HOD config: `config.yaml hod_break` (caps 4 concurrent / 12 per day / notional / day kill;
  `entry_mode: resting_stop_limit`, stop-limit exit 20 bps below the stop per PREREG_1466).
- Prior anatomy, read FIRST (results only, not the cell code): `research/hod_live_anatomy/CONTEXT_BRIEF.md`,
  `RESULT_1721.md` … `RESULT_1731.md` (live HOD +0.075 R gross / −0.026 R net at its levels; cost = entry chase
  8 bps + stop slip; bracket stop-limit legs 3× the StopMonitor's slip; no filter resolvable below 500 fills,
  `PREREG_1723_at_500_fills.md`). ORB: `research/orb_exit_resim/REPORT_1709.md` (verdict: tail-carried, no live
  ORB; Amendment 1 `PREREG_1709_AMENDMENT_1.md` re-enabled the PAPER book 10/30 as a $0 forward instrument with a
  frozen gate: at 40 paper symbol-days, `scripts/orb_ramp_check.py` must show realized > 0 AND above water
  ex-monster AND ≥ 15 sessions). `docs/week_review_20261107_RESULT.md`, `docs/books_insight_20261106_RESULT.md`.
- ORB accidents to EXCLUDE and price separately: CRDU 10/27 (provisional 906-sh order filled 861 during the paper
  cancel queue, naked all day, ≈ +$600) and BEX 11/9 (first add-on ever; 87 sh unprotected to the force close).
  Also 11/10: Alpaca bars late, zero ORB ranges, ORB traded nothing (fixed, loads at the 11/11 boot).
- `python3 scripts/orb_ramp_check.py` today prints: stage since 2026-09-21, stage P&L −$207 on 35 fills / 25
  sessions, ex-monster −$1,156, mean R −0.00 vs BT band [p5 −0.16, p90 +0.53] IN-BAND, VERDICT: DEMOTE. That stage
  window spans the LIVE fills 9/21–9/24 (book paused 9/25) AND the paper fills since 10/30 — find out what it is
  actually counting (which account column / dates) and whether the Amendment-1 gate (paper only, from 10/30, ex-
  accidents) is what the script computes. If not, that is finding #1.
- The pass bar for every book is the cadence bar (`docs/cadence_bar.md`, `scripts/cadence_bar.py`): weeks, not
  trades; ≥ 3 fills/wk at the live config; strong week ≥ +5 R; report ex-top-1 / ex-top-5 % always.

## Questions (answer each with a number, its source and the exact query/command, in `NUMBERS.csv`)
Per book (HOD, ORB-paper-only), window 2026-09-30 → 2026-11-10, with the ORB accidents shown in AND out:
1. Fills, symbol-days, sessions traded, fills/week; realized $ and mean R per fill (R = |entry − stop| × shares as
   the ledger defines it; say what you used); win rate; ex-top-1 fill and ex-top-5 % $; day-clustered t beside iid.
2. Exit-reason mix (stop / target / time / force-close / unwind) and the stop rate vs the book's BT reference
   (ORB: `research/orb_gates2/book_G3_meas.csv` or whatever orb_ramp_check reads; HOD: the 1,438/1,466 numbers in
   `research/hod_entry/RESULT_1466.md` — quote them).
3. Cost anatomy: entry slip vs model (ORB `drift_ask_to_fill_bps`, HOD `slippage_vs_tape_bps`), stop slip, unfilled
   / cancelled rate (HOD ledger `broker_status`, `reason`), and $ lost to each class of execution defect in the window
   (CRDU, BEX, LABX naked, anything in `docs/*_RESULT.md` dated 10/30–11/10).
4. Cross-check: ledger realized + open P&L vs account equity change (HOD −$4 over 6 wk; ORB-only = account
   +$1,249 − TOM − EXT: compute TOM and EXT from the ORB account's activities by client_order_id prefix). Report the
   residual and whether it is explained.
5. Weekly table (Mon–Fri, 6 weeks): fills, $, R, green/red — score against the cadence bar as far as 6 weeks allow
   (say explicitly which bar items 6 weeks cannot resolve and the MDE at this n).
6. Gate status: ORB paper symbol-days since 10/30 ex-accidents vs the 40 needed; HOD fills vs the 500 in
   PREREG_1723; projected date each gate resolves at the observed frequency.

## Then: ranked hypotheses and recommendations
For each book, rank (not pass/fail) the top 3 explanations of the 6-week result with evidence for/against, and
give recommendations as PRE-COMMITTED RULES the owner can accept in one word (e.g. "keep running the $0 instrument
until <date>/<n>; pause if <condition>"; "fix <defect> before <date>"; "do NOT change <knob> because <evidence>").
No knob proposals that `docs/CLAUDE_HISTORY.md` / the hod_entry and orb research dirs already rejected — grep before
proposing. State the MDE: what a 6-week paper record can and cannot tell us.

## Output
`research/book_review_20261111/REPORT.md` ≤ 700 words + tables; `NUMBERS.csv` (metric, book, value, window, source,
query). Return ≤ 150 words: the three findings that matter most and the recommendation per book.
