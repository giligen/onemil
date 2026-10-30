# HOD-break — context brief for digging agents (Fable, 2026-10-30 evening). Read fully before any query.

## What the book is
Long breakouts of the intraday high-of-day on small/mid-cap movers (price mostly $5–$30), entry = resting stop-limit at
the level with a +15 bps limit cap (`entry_mode: resting_stop_limit`), stop = consolidation low (`consol_low`), target
2 R, flat at 15:55 ET. Live paper book since 2026-09-29 at $50 risk per trade (`strategy='hod_break'`, `account='paper'`
in `data/trades.db` — read a COPY: `/tmp/claude-1000/-home-ec2-user-onemil/257c3e2d-cf38-45d5-94e7-4877f8170f44/scratchpad/trades_ro.db`).
~50 fills/week, 39 % winners, exits 44 % stop / 23 % target / 20 % EOD flat.

## What is known (all 2026-10-30 unless noted; files in research/hod_live_anatomy/ and research/hod_entry/)
* **Live fills (231, R-valid):** gross at the intended level/stop/target **+0.075 R/trade**; entry slip −0.055 R (fills
  avg 8 bps above the level = the spread of these names when chasing); exit slip −0.037 R (stops: StopMonitor `stop_bid`
  11.8 bps n 96; broker bracket leg `bracket_stop_limit` 34.5 bps n 18 = fast drops); **net −0.026 R** (−$275).
  Per-fill sd 1.37 R. `RESULT_1721.md`, `fills_1721.csv`.
* **The BT population of the same live rule** (`research/hod_entry/1663_features.csv` + `1676_features.csv`, 5,506 fills
  2025-01 → 2026-05, splits TRAIN-H2 / VAL; minute paths in `research/hod_entry/1681_paths.parquet`; raw 1-min bars for
  307K symbol-days 2025-01 → 2026-09 in `research/bf_zero/bars_sip.db` `bars(symbol, day, t, o, h, l, c, v)`, 27 GB,
  read-only, query per (symbol, day)): **−0.21 R net** (cells 1,438/1,681). The LIVE fills beat this population by
  ≈ +0.2 R (10/23, `project_exec_learnings_oct2026`). **Nobody has explained that gap.** It is the lemon: something about
  which breaks actually FILL a resting +15 bps stop-limit in the live engine (or about the live universe/arming) selects
  better trades than the BT's fill model. Candidate differences: live universe = the scanner's qualified movers at arm
  time (gap/volume/catalyst gates, `scanner_qualified_at_arm`), BT universe gates undocumented (10/23 finding); live fills
  need the ask to be within 15 bps at the trigger (fast gaps through the level do NOT fill live; the BT may fill them);
  live caps 4 concurrent / 12 per day; live skips names another strategy holds.
* **Exits are not the lever**: 35 mid-flight exit rules on the BT population (`research/hod_entry/RESULT_1681.md`) and 10
  time-aware variants on the live fills (`RESULT_1724.md`: 14:00 time stop, take-profit, break-even, 1.5 R target,
  decaying targets) are all within ±0.03 R of hold-to-target/flat; the afternoon runner is detectable at 14:00 (≥ +0.5 R
  and near its high: 62 % finish ≥ +1 R vs 15 %) but has NO drift after 14:00 (live: +13 bps; BT 308 cases: +18 / −15 bps)
  so neither squeezing the rest nor adding to it earns (`RESULT_1725.md`, `RESULT_1726.md`, `RESULT_1727.md`).
* **Filters on the live fills**: 12 pre-listed cuts fail A/B at n 231 (MDE 0.45 R for a 50 % cut) — unresolvable there,
  but the BT population has the power (5,506 fills) and the live fills can then serve as the out-of-sample check.
* **Earlier HOD research (1,400+ cells, `research/hod_entry/`, `docs/CLAUDE_HISTORY.md`)**: the BT population "carries no
  information" for causal filters at arm time (OFI, rank, catalyst, extension model all ≈ 0 after cost); the +0.3 R
  "fill filter" of cell 1,427 was a sparse-bar artefact; stop slip measured 35 bps mean on the tape (cell 1,443).

* **Cell 1,728 (Opus dig, 10/30 late): there is NO live-vs-BT gap.** Measured from the level, the BT gross is +0.072 R
  (TRAIN-H2) / +0.061 R (VAL) vs live +0.075 R (95 % CI −0.09..+0.24); both ≈ −0.026 R net at measured cost. The old
  "+0.2 R gap" was cell 1,438's banded cost (0.20–0.24 R/fill) plus ≈ 0.04 R from the 0.75–1.5 % R bucket that the
  9,911-fill population has and the 5,506 lacks. **Parity defect:** the research twin (`research/hod_entry/causal_arming.py`
  l.81) rejects arms with R_pct = (trigger − stop)/trigger < min_r_pct (1.5 %) or trigger < min_price; the live arm
  (`trading/hod_break.py`) has neither gate → 82 of 231 live fills have R_pct < 1.5 % (down to 0.24 %) and they are the
  only net-positive live cohort (< 0.75 %: n 29 +0.105 R; 0.75–1.5 %: n 53 +0.090 R; 1.5–3 %: n 102 −0.066; ≥ 3 %: n 47
  −0.078). Live fills sit at LOWER ATR % and lower prior-day return than the BT (where the BT earns less, not more); the
  4/12 caps change nothing on the BT. `RESULT_1728.md`.
* **Cell 1,729: the tight-R cohort is NOISE.** On the 9,911-fill population (`causal_arming_causal.csv`, floor 0.75 %)
  R_pct 0.75–1.5 % is −0.088 R (TRAIN-H2, n 2,219) / −0.097 R (VAL, n 2,542), day-clustered t −3.3, ex-top-5 % −0.21,
  worse than the complement (−0.008 / −0.020); as a book 19–27 % green weeks vs a 32–40 % random null. Live +0.064 R on
  82 fills, CI [−0.33, +0.45]. TEST not read. → the twin's gate goes live (`docs/hod_arm_gate_parity_spec_20261030.md`).
  Running tally of HOD cells on the live book tonight: 1,721–1,729 (9 cells, 13 filter cuts, 10 exit variants, 2 tells,
  3 execution levers, 1 add-on, 1 cohort) — every one inside noise of the live rule or negative.

## Data quirks that have already bitten
Alpaca 1-min bars have NO bar for a minute without a trade (thin names: 82 % minute coverage; losers thinner) — treat
no-trade minutes as flat, not missing. The live 2 R target CENSORS any "does it keep rising" measurement — use the raw
path. Entry fills can print 1 cent above the limit (penny rounding), not a defect. `entry_quote_*` columns are NULL on
all live HOD rows (recording defect; exit quotes exist on 157/234). The ledger's `filled_qty` can differ from `shares`
(25 rows) — R unit = `total_risk × filled_qty / shares`. `trade_date` comes back as a date object from the project's
`Database`. Live fills have one row per fill (no provisional rows, unlike ORB).

## Standards (non-negotiable)
PREREG before scoring; halves (alternating trade dates / weeks) must agree in sign; report ex-top-5 % and winner-capped
alongside the mean; day-clustered t; cost = measured (12 bps marketable exit, entry slip as observed) — never a band;
count the cells; a null is a claim about the test first (state MDE). No `--build-cache`, no writes to data/*.db, no
orders, no commits. One python process at a time. Budget ≤ 40 calls; write everything to disk, return ≤ 200 words.
