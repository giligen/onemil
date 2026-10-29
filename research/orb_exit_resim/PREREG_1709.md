# PREREG cell 1,709 — ORB exits re-simulated at the LIVE fill price (frozen 2026-10-29, Fable)

Decision this cell serves (owner 10/29, pre-committed): **ORB continues only if 1,709b passes; otherwise ORB is shelved
on 2026-11-13** — code frozen, paper book stopped (`orb.yaml strategy.enabled: false`), no further ORB cells.

## Question
Cell 1,702: live stop share 41.6 % vs BT 28.4 % on the same window; stops are slow (median 32 min). Cell 1,700/1,708:
live fills sit a median 50 bps above the BT entry model; a fill 50–100 bps high on a ~3 % range puts the stop ⅙–⅓ R
closer. The replay (1,426/1,708) re-simulated ENTRIES only and kept the BT's exits. This cell re-simulates the EXITS.
H1 (mechanism): running the engine's exit rules from the live fill price on the same bars reproduces most of the stop
excess. H2 (money): with exits re-simulated from tape-obtainable entries the out-of-sample edge is ≥ +0.10 R/fill.

## Data (no new purchase)
* Live/paper fills: `research/exec_learnings_oct2026/fills_1702.csv` (137 rows; entry, stop, filled_qty, filled_at,
  exit_reason, exit_price, exited_at). 1-min bars: `research/bf_zero/bars_sip.db` (through 2026-09-28; the 14 paper fills
  after that are EXCLUDED — reported as such, no Alpaca fallback).
* Replay population: `research/exec_learnings_oct2026/fills_1708.csv` / `table_1708.csv` (617 replayed fills, replay
  entry price at delay 0 s and 30 s, true shares) + the same bars (fetch only what the 1,426 manifest already has; a
  symbol-day without bars is "uncovered", counted on the availability rail: ≥ 80 % coverage, ≤ 5 pp winner/loser gap, else VOID).

## Exit rule (the LIVE rule, one implementation, parity-tested against 5 live fills by hand)
From `trading/orb_engine.py` / `research/orb_machine_rules.md`: stop = `range_low` with the ATR-14 floor (as stored in
`pattern_data`; if absent use range_low — report the share), lock arm at `lock_arm_at_r` (1.75 R) → stop to
`lock_stop_r` (0.5 R) above entry, bar-close `tag_bb` exit as coded, force_close at 15:55 ET at the bar close. Bars are
processed in order; within a bar the stop is assumed hit BEFORE the target (conservative); the entry bar itself cannot
stop out unless its low is below the stop at the bar close (the live engine uses ticks — report sensitivity with
"entry bar can stop"). Exit price = stop price minus the measured slip (cell 1,443 style: 35 bps mean) for stops; bar
close for the others. Costs: the decision-time quotes stored on the row (never a band).

## Cells
* **1,709a (mechanism, n ≤ 123 live):** re-simulate from (i) the BT entry = range_high × 1.003, (ii) the live fill
  price. Report stop share and mean R under (i), (ii) and the actual ledger; the share of the 13 pp excess explained by
  (ii)−(i); per-fill agreement of (ii) with the actual exit reason (confusion matrix). Day-clustered CIs.
* **1,709b (money, replay 617):** re-simulate from the replay entry at 0 s and 30 s (true shares). Mean R per fill
  in-regime (2025-01..26-09) split into TRAIN halves 2025H1 / 2025H2+2026, and out of regime 2023-01..24H2. Ex-top-5 %,
  +3 R cap, stop share, fills/week.

## Pass bar (frozen)
H1 passes if (ii)−(i) explains ≥ 50 % of the live stop excess AND (ii) reproduces the actual exit reason on ≥ 70 % of
fills. **H2 passes only if** out-of-regime mean R ≥ +0.10 with the day-clustered CI lower bound > 0, same sign in both
TRAIN halves, ex-top-5 % > 0, ≥ 3 replay-filled fills/week. H2 fail → shelve ORB (above). H2 pass + H1 pass → the next
and only engineering item is the entry-price term (limit at the model price, no chase); H2 pass + H1 fail → report,
no engineering, book stays paper.

## Rebuild
Independent Sonnet rebuild from this prose (not from the build script); agreement bar: stop share within 2 pp, mean R
within 0.02, row-level exit-reason agreement ≥ 95 %. No number goes to the owner before the rebuild.
Cell count after this: 1,709 (+ the a/b split, 2 sensitivity reads).
