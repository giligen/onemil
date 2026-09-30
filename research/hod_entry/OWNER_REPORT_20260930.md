# Owner report — 2026-09-30 morning (written overnight 9/29, final 21:10 UTC)

## 1. Bottom line
* **No new winner came out of tonight's 13 cells (1,660–1,672).** Every one ran under a frozen PREREG at the measured
  cost on both halves; nothing clears the bar (net ≥ +0.05 R, t ≥ 2.5 both halves, tail-clean, ≥ 3 fills/week). Four
  of your hypotheses were answered with real power: relative volume (no), candle shapes and the 61 TA-Lib patterns (no),
  early cuts on predicted failure (no), adds on predicted non-failure and after +R (no, the locked-stop pyramid loses).
* **What is real:** (1) the machine is now correct — seven order/bookkeeping defects found today are fixed with tests,
  arm-time telemetry goes into the ledger from the 12:30 UTC boot; (2) the bar store is complete (every fill day, 20
  prior sessions each, SPY), so these answers are powered, not coverage artifacts; (3) the HOD book at the 1.5 % floor
  is ≈ 0 R (+0.007 / −0.028 per half) instead of −0.11 R on the fills the floor removes — a loss avoided, not a win.
* **Where the money is:** ORB is the only book with a positive in-regime backtest at the live config (+0.105 R/fill
  2025-01..2026-09, n 473, t 3.3, tail-clean) and it lost live because of execution (first order 26–49 s late, adverse
  drift). The fixes (pre-placement + 5-s delay, target resting limit) are on paper; **the 9/30 paper parity read decides
  the Wednesday live GO.** Out of regime its edge is ≈ 0, so the regime (frequency) is the risk, not the mechanism.
* **Money 9/29 (closed 20:00 UTC, all accounts flat):** live −$224 (HOD morning stops −$132, CDNA over-exit cover −$92);
  HOD paper −$51 (CONI +$52, four stops −$103, all armed before the floor, 0.8–1.5 % stops); ORB paper AXTL −$96
  (+$170 at 14:00 ET, never reached 2 R, force-closed 15:45 ET as the rule says).

## 2. The machine at the 9/30 12:30 UTC boot
| book | account | state | changed 9/29 (all committed, full suite 4,681 passed) |
|---|---|---|---|
| HOD-break | paper PA39QSZR60WC | ON, min_r_pct 1.5, real paper orders | fill detection (GET before/after cancel, REST poll, boot adoption as FILLED + DB merge), exits clamped to broker qty, OCO parser, account-stamped state file, R2G sync no longer touches HOD rows (self.STRATEGY_NAME), reconcile checks the broker, leg-less restores get a StopMonitor watch, dry ledger writes paper/live fills + 9 arm-time feature columns |
| ORB B+ | paper PA3YNVRTKFMG | ON, 5-s pre-placement delay, catalyst OFF, **target-resting-limit ON (bar ≤ 5 bps vs the market TP)** | BT parity (catalyst veto read from orb.yaml, book regenerated: 2025–26 n 483, +0.107 R); post-close double-close no longer an ERROR |
| Bull flag | live | PAUSED | — |
| TOM sleeve | paper | cron 19:45 / 20:45 UTC | first entry 9/30 at the close (check `scripts/tom_sleeve.py --status` at 19:52 UTC) |
| Live account | — | FLAT; guardrail acknowledgement cutoff live | trade records corrected (rows 381, 393–400); incident write-up `docs/hod_live_incident_20260929.md` |
Telegram: dedupe 60 s + 20/min cap; every 9/29 error storm has a root-cause fix.

## 3. Everything tested at the measured cost (entry 7 bps, stop 6, target 0, EOD 11; halves TRAIN-H2 / VAL; MDE of the floored book 0.077 / 0.066 R)
| cell | what | result | verdict |
|---|---|---|---|
| 1,658 | P&L by stop distance | < 1.5 %: −0.11 R both halves (t −4); 1.5–3 %: −0.01/−0.07; ≥ 3 %: −0.08/+0.14 | floor 1.5 % → paper |
| 1,660 | exit lab (35 variants) re-read | base −0.014/−0.061; floor +0.007/−0.028; best variant tail-carried | FAIL |
| 1,661 | entry limit width | +0.10 %: +0.047 R on n 31 | watch on the ledger (the 0.15 % session contains it) |
| 1,662 | re-score of 16 cell families | nothing; 1,488 flagged; 1,619 artifact | FAIL |
| 1,663 | 7 cost-axis cuts | 0/24 causal reads (4 numeric passes condition on exit type) | FAIL |
| 1,665 | relative volume to the arm minute vs prior 20/5 sessions | coverage 99 %: low tercile +0.03 R, high −0.07 R both halves, t ≤ 1.3, MDE 0.05; 0/48 | FAIL (powered null) |
| 1,666 | independent rebuild of the 1,488 pyramid | paired −0.043/−0.044 R, t −5, MDE 0.02 | REFUTED |
| 1,667 | every causal arm-time feature (17) | daily and intraday all flat (10 reads t ≥ 2.5, all negative direction); n_cross "pass" = full-day count (leak) | FAIL |
| 1,668 | post-entry failure detection: 9 rules × 4 horizons + classifier | 0/36 rules; AUC 0.63–0.67 oos, cut −0.05..+0.02 R; patterns ≈ 0 importance | FAIL |
| 1,669 | fast failures | 3/12/24 % of stops within 2/5/10 min; predictable at minute 1 (AUC 0.76–0.81) but the cut flips sign across scorings; 0/96 | FAIL |
| 1,670 | feature-timing map + adds | information only in the path; cuts +0.02..+0.04 R t < 1; model-gated add ≈ 0; add after +R with locked stop t −3.3 | FAIL |
| 1,671 | MiniRocket raw-sequence model vs trees | trees win at every minute (0.54→0.68 vs 0.50→0.64); stacking hurts; placebo clean | FAIL → **no TSFM fine-tune** |
| 1,672 | pre-holiday index sleeve 2016–26 (102 events) | SPY −17/−4 bps per event, null percentile 12/35, mirror ≈ 0 | FAIL |
Programme count on the HOD line: > 2,400 cells. Detailed RESULT files under `research/hod_entry/` and `research/calendar/`.

## 4. Winners, honestly flagged
1. **ORB at the live config** — the only positive book; execution fixes under paper test; live on GO after parity.
2. **The measured cost** (entry +6.7 bps, stop −2.6/−6 bps on live fills) — every earlier HOD verdict charged 3× too much;
   reusable across every book; it is why the floor exists.
3. **The 1.5 % stop floor** — removes the −0.11 R fills; HOD becomes a free forward instrument at ≈ 0 R.
4. **Turn-of-month sleeve** — passes its own bar, ≈ $85/month at the tested size, stacks on different days; paper first.
5. **The machine** — a paper session tomorrow with real fills on correct code is the bar for any live day, and the ledger
   now records every arm's features so each session reads every cut forward at once.

## 5. Closed — do not re-test on this population
Every filter on arm-time features (price, ATR, time of day, extension, momentum, liquidity, VWAP, volume, relative volume,
patterns); every exit or early cut on the price path or on predicted failure; adds on predicted success or after +R;
the confirmation entry / no-withdrawal pyramid; OFI; sequence models. The information about a HOD trade's fate is in its
own path after entry, and acting on it costs more than it saves. HOD's value is as a forward instrument at the floor.

## 6. Decisions (act-as-owner, pre-committed)
1. **9/30:** HOD floored paper session with telemetry (first clean session with real fills = the bar); ORB paper with the
   target resting limit (bar ≤ 5 bps); TOM paper entry at the close; daily brief 20:10 UTC.
2. **ORB live:** owner GO on Wednesday only if 9/30 paper parity holds (fills match the BT book, exits ≤ 5 bps).
3. **HOD:** forward read at 100 paper fills, buckets 1.5–3 % and ≥ 3 % separately; the +0.10 % limit read from the
   ledger; NO more filter/exit/add cells on this population. Next HOD frame = a NEW signal definition with gross edge,
   one cell per day from `research/ideas_web/UNTESTED_20260929.md`, own PREREG.
4. **TSFM / GPU fine-tune:** not started (1,671's clause failed: no sequence model beat the trees by ≥ 0.03 AUC).
5. **Options v3** ($32 Databento pull, running): cell 1,599 verdict when it completes (watcher armed).
6. **Research spend:** the day books are execution-limited, not idea-limited. Tomorrow's research hour goes to ORB
   parity and frequency (the only positive mechanism), not to HOD.

## 7. Self-audit (owner 05:10 UTC: "find the errors and oversights, there's money there")
Four candidates examined; two closed by a quick check, two run as cells 1,673–1,674. Result: the errors were real,
the money was not.
* **Fill model** — closed: backtest fills sit 6.0 bps above the level (median 5.2, capped at 15) = the live +6.7 bps;
  fill distance carries no P&L pattern. No hidden cost.
* **My bar's power error** — real: "+0.05 R and t ≥ 2.5 in EACH half" rejects a true +0.05 R lift ~75 % of the time.
  Read correctly (1,674: tercile edges fixed on TRAIN, pooled day-clustered t, sign agreement, joint book on VAL): every
  same-signed cut pools to ≈ 0 (relative volume −0.001 R, dollar volume −0.004, VWAP +0.004, ≥ 3 % stops +0.023 t 0.45,
  midday −0.058); the earlier same-sign pattern came partly from per-half tercile edges. Joint book = the floored base,
  VAL −0.030 R (t −2.1). Ordering the capped day by the widest stop: −0.032 vs −0.038 R. Data-integrity finding: the
  exit lab's per-fill book (1,660) and the stop-distance book (1,658) are different pipelines (74.7 % key match, 374
  sign flips) — the MOC +0.010 R transfers only as an aggregate.
* **Short the predicted failure** (1,673, the geometry I never read): the fast-failure signal at minute 1 with the
  day-high stop earns **+0.065 / +0.062 R per short, hit rate 64–66 %, same sign on both scorings — but t 1.1 / 1.6,
  18.5 % of exits priced optimistically (gapped through), 0 / 45 cells pass, placebo unavailable.** Stop-keyed shorts
  at 2–10 min are negative. Portfolio overlay +0.004 / +0.012 R. Verdict: a watch item, not a winner — the one
  positive, same-signed, mechanism-backed read of the night; it needs short-side order mechanics that do not exist in
  the engine (locate, SSR, short OCO), so it is NOT the next engineering item ahead of ORB parity. Queued as the first
  new-mechanism cell once ORB is live and stable, with a pre-committed paper read at 100 shorts.

## 8. The short on HOD-break failures — sealed forward test (owner 06:30 UTC), 08:20 UTC
Window 2026-06-01..09-04 (the universe file ends 9/4), 2,872 fills never seen by any model; both saved minute-1
failure models applied unchanged; short at bar fill+2 when P ≥ 0.6, target = the long's stop, stop = day's high + $0.01.
| model | shorts | weeks | mean net R | week-t | green weeks | worst week |
|---|---|---|---|---|---|---|
| trained on 2025-H2 | 97 | 14 | +0.014 | 0.5 | 2 of 14 | −5.7 R |
| trained on 2026-H1 | 57 | 12 | −0.577 | −1.0 | 4 of 12 | −24 R |
Verdict **NO-GO** on every clause. The +0.06 R of cell 1,673 does not generalize to the sealed months; the second
model's −24 R week is the squeeze tail: a HOD breakout that keeps going gaps through the day-high stop (halts, prints
far above). The model still selects failures (the flagged longs lose −0.5 to −0.8 R), but the short does not capture
the decline. The order mechanics are committed and OFF (`hod_break.failure_short.enabled: false`); nothing trades
today. Closed: shorting HOD-break failures keyed on the post-entry model, on this population.
