# PREREG — what the live/paper record since 2026-05-19 says about execution (frozen Fri 2026-10-23, before any number)

Owner's ask (10/23): "go deep and research everything we've learned including latency, spreads, etc." Every cell below
is frozen before scoring. Cost = measured per-fill NBBO at the entry minute where available, never a band. Every number
that reaches the owner gets an independent rebuild from this prose by an agent that has not seen the first build, matched
trade by trade on (date, symbol). Phrasing rule applies: state universe, horizon, n, window, cost, MDE; a null is a claim
about the test first. Cells are numbered from 1,700 (programme count as of 10/23).

## Data (read-only; never write any of it)
* `data/trades.db::trades` — ORB LIVE fills 2026-05-19..09-23 (strategy='orb', real account, ~123 closed) and ORB PAPER
  fills 09-28..10-22 (14); HOD paper fills 09-29..10-22 (`hod_break`, ~190); columns incl. fill_price, exit_price,
  exit_reason, pnl, shares, filled_at, pattern_data (JSON: stop, target, level, order ids). Owner's manual trades are
  NOT in this table; never query the live account's positions.
* Order timestamps: `docs/orb_live_endpoint_gap.md` method (order ids from trades.db → broker `created_at`/`submitted_at`/
  `filled_at`; LIVE endpoint via the live key READ-ONLY, paper via `ALPACA_ORB_*`). Engine clock: `SUBMIT LATENCY` /
  `SUBMIT TIMING` lines in `logs/session_archive/*.log` (timing lines exist only from 10/19).
* Quotes: `research/exec_cost/` (measured NBBO per ORB fill, see its REPORT_RECAL.md for the schema); `bars_sip.db`
  1-min bars (`research/bf_zero/bars_sip.db`) for MFE/MAE; the engine's own spread-gate lines (`spread`/`bps`) in the archives.
* HOD: `logs/hod_live_parity_ledger.csv`, `scripts/hod_dry_ledger.py`.
* MOM: `logs/momentum_sleeve_ledger.csv` (paper, 10/5..), `logs/momentum_sleeve_live_ledger.csv` (10/19..), state files.
* BT: `analysis_results/orb_bplus_book.csv` (nightly book), `analysis_results/orb_features_YYYYMMDD_2055.csv` (nightly
  features; read through `trading.orb_csv.read_orb_csv`), `orb.yaml` filter params via `trading.orb_filter`.

## Cells
**1,700 Latency → outcome (ORB LIVE 5/19–9/23, n ≈ 123 closed fills).** Per fill: L_eng = decision→HTTP (engine
`SUBMIT LATENCY` where present, else first-order offset from 09:35:00), L_brk = created→filled (broker), drift =
(fill − decision ask)/ask in bps when the decision quote exists, else (fill − range_high)/range_high. Outcome = R
(pnl / (shares × (entry − stop))). Pre-committed tests: (a) Spearman(L_brk, R) and (L_brk, drift), day-clustered
bootstrap CI; (b) R by latency tercile; (c) R for fills with drift ≤ 30 bps vs > 30 bps (the BT's entry model). MDE: with
n 123, sd(R) ≈ 1, detectable mean difference ≈ 0.36 R at 80 % power — anything smaller is "not resolvable", not "no effect".
**1,701 Spread → outcome (same fills + 14 paper).** Quoted spread at decision (bps) from the engine's spread-gate lines
or `research/exec_cost` NBBO. Tests: R and drift by spread bucket (< 50, 50–150, 150–300 bps); realized round-trip cost
(entry drift + exit slip vs stop/target) by bucket; share of the book's loss attributable to names ≥ 150 bps. This is
MEASUREMENT — the spread gate is never tightened below 150 bps by this cell (`research/orb_spread_gate_verdict.md`).
**1,702 Exit decomposition (ORB live + paper, n ≈ 137).** By exit_reason (stop_loss / time_stop(force_close) /
lock_stop / target / eod): count, Σ$, mean R, and from 1-min bars MFE/MAE in R from entry to exit. Pre-committed
question: do stop-loss exits occur within 5 minutes of entry more than the BT's stop rate implies (entry-bar stops =
spread/latency, not thesis failure)? Report the counterfactual of a stop at the ATR floor only as a DESCRIPTIVE number —
no knob change follows from this cell.
**1,703 Universe coverage BT vs live (12 ORB paper sessions 10/6–10/22).** Per session: names the engine ranked
(`ORB SCORED`) not in that night's features CSV and vice versa; for each, why (liquidity floor, wrapper rule, listing
age, data gap). The 10/22 case: SSG (range volume 5,971) ranked #2 live, absent from the BT → SMST fell from BT rank 8
to live rank 9 → "BT pick never ordered" → streak reset. Deliverable: the rule that decides which side is per spec
(`research/orb_machine_rules.md` universe section) and the list of sessions whose picks would change.
**1,704 HOD forward instrument (paper fills 9/29–10/22, n ≈ 190, $50 R).** Mean R with iid AND day-clustered SE; by exit
(OCO target / stop / eod / safety-net); weekly series; ex-top-5 % of fills; compare to the research prior −0.21 R net
(cell 1,438) and the trailing-40 +0.55 the guardrail shows — pre-committed read: the forward number is "consistent with
the prior" unless its 95 % CI excludes −0.21.
**1,705 MOM sleeve entry timing (paper 3 rotations, live 1).** Per rotation: Σ(fill − prior close)/prior close weighted
by notional (the gap paid at 09:45/09:50 entry), slippage vs the 09:45 bar's open, turnover $; the BT's assumed entry
price (grep `research/` momentum spec) vs the actual. Pre-committed read: if the mean gap paid > 50 bps per rotation,
flag "entry timing is a cost the BT did not charge" — no rotation-time change without its own PREREG.

## Rules for every agent on this programme
Read-only on every data file; no orders; no services; no git; no config; never print keys or account numbers beyond
4 chars; never list the owner's manual positions. Scratch outputs under `research/exec_learnings_oct2026/` only. One
agent at a time, budget per cell ≤ 40 calls. Each cell writes `RESULT_<cell>.md` ≤ 40 lines: data used (rows, window),
the pre-committed tests' numbers with CIs, the caveats as an adversary would list them, and nothing else. Return ≤ 120 words.

## Added 10/23 after 1,700 (frozen before scoring; 1,700/1,701 rebuilt exactly, REBUILD_1700_1701.md)
**1,706 Marketable-at-submit cohort (ORB live 123 + paper 14; table fills_1700.csv + broker_ts.csv).** chase = broker
created→filled ≤ 5 s (the resting order was already marketable); resting = > 5 s. Per cohort: n, mean R (day-clustered
CI), drift_bps, exit_reason mix, Σ$. Pre-committed: the cohort split is DESCRIPTIVE — the engine already has a 50 bps
chase guard; a rule change (e.g. refuse marketable submits) needs its own PREREG with the unfilled counterfactual
(`project_passive_entry_adverse_selection`: a passive limit can select broken setups). Sensitivity: thresholds 2 s / 10 s.

## Added 10/23 (owner: "shouldn't we avoid chases at all?") — frozen before scoring
**1,707 Marketable-at-submit rule on the BT population (tick replay).** Engine: `research/orb_latency_bt/` (cell 1,426:
every BT fill replayed on XNAS ticks with the engine's chase rule). Populations: in-regime 2025-01..2026-09 (n ≈ 473 at the
live config) and out-of-regime 2023-01..2024-06 (n ≈ 165, live exit rule). Classify each fill: MARKETABLE = tape last ≥
range_high at the submit instant (09:35:05 ET for preplaced names, else the engine's decision time); RESTING otherwise.
Per cohort: n, mean R, day-clustered CI, ex-top-5 %, winner-capped at +3R, TRAIN halves (2025H1/2025H2+2026) same sign,
wrapper (2x/3x ETF) vs stock composition, and fills/week at the live config if RESTING-only. Sensitivity: marketable
defined at +0 / +10 / +30 bps above range_high. PASS = RESTING ≥ +0.10 R same sign in both halves AND out of regime;
MARKETABLE ≤ 0 in both; ex-top-5 % of RESTING > 0; then state fills/week against the cadence bar (≥ 3/wk). On PASS the
rule ships as ONE explicit helper (BT + live, parity test) to PAPER ORB and is read at 40 fills; on FAIL the live tercile
split (1,700) is recorded as selection noise. No knob other than this binary skip is touched; no refill after a skip.
**1,707 amendment (10/23, before scoring, after 1,706 showed ≤ 5 s is only 11/137 live fills):** besides MARKETABLE/RESTING,
report the replay population by time-from-submit-to-fill buckets ≤ 5 s | 5–34 s | 34 s–4 min | > 4 min (the 1,700 tercile
edges), same statistics per bucket. The PASS bar stays on the MARKETABLE/RESTING split; the buckets are descriptive and
decide nothing. 1,702 (not yet rebuilt) adds a mechanism to test here: live stop share 41.6 % vs BT 28.4 % — report the
stop share per cohort/bucket in the replay too.

## Added 10/23 (owner: "HOD is our only profitable strat while you claim it is not") — frozen before scoring
**1,704 amendment — book vs population reconciliation.** Same window 9/25–10/22. (a) Book fills (trades.db hod_break) vs
the all-armed population (`counterfactual_all`/`live_dry` source used by `scripts/hod_dry_ledger.py`) on the SAME days:
mean R each, day-clustered CI, and the difference. (b) R by within-day slot order (1st, 2nd, 3rd+ arm of the day) in the
population — the caps select the earliest arms, so "first breaks of the day are better" is the mechanism that would
reconcile a break-even book with a −0.2 R population; pre-committed read: a mechanism exists only if slot-1/2 R exceeds
slot-3+ R with a CI excluding 0 AND the same sign in both halves of the window. (c) Variance check: the book's last-2-week
+$434 on 77 fills against the book's own per-fill sd (z-score). (d) Ramp rule recorded: $50 → $100 R when realized book
P&L since 9/25 > 0 AND fills ≥ 200; never while under water. No knob changes from this cell.

## Added 10/23 after the rebuild found a defect in the shared replay engine — frozen before scoring
**1,708 Re-score of cell 1,426 (ORB latency replay) with a true share count.** `research/orb_latency_bt/replay.py` l.136 sets
`shares = _sized_pnl / pnl` = 0.0667 for every row (a sizing RATIO, not shares), so the price term `shares × (BT entry −
replay fill)` is ~0 and every "R by delay" in `REPORT.md` is the BT's own P&L of whichever fills the replay obtains — no
entry-price effect was ever modelled. Fix for the re-score ONLY (do not edit replay.py in place; copy to
`research/exec_learnings_oct2026/replay_1708.py`): shares = `_rp_position / entry_price` (confirm the column from
population.csv's header; if absent, shares = stage risk $375 / (entry − stop) — state which). Re-run the exact cells of
`research/orb_latency_bt/REPORT.md`: mean R and Σ$ by delay (0, 3, 5, 10, 20, 30, 60 s or whatever grid REPORT.md used) for
out-of-sample (2023-24 + 2025H2–2026-09), May–Sep 2026, 2023-24 untouched, 2025H1, 2025H2–2026-09; day-clustered CIs;
ex-top-5 %; the share of fills matching the BT's 30 bps entry model. Pre-committed read: the 9/25 closure statement
("+0.028 R/fill at zero latency, ≈ 0 at ≥ 30 s, tape edge ≈ $11/trade") is REPLACED by whatever this cell shows, in both
directions; a larger latency effect does NOT reopen ORB by itself (the untouched out-of-regime edge is a separate test).
Report the old and new numbers side by side. Independent rebuild required before the owner sees a number.
