# Owner report — 2026-09-30 morning (written overnight 9/29; sections marked PENDING fill in as cells land)

## 1. Bottom line
* **The only proven positive book is ORB at the live config** (+0.105 R/fill 2025-01..2026-09, n 473, t 3.3, ex-top-5 %
  ≈ 0; 9/29 paper AXTL +$170 with the 5-s pre-placement delay). Its edge is frequency-limited (1.4–7 fills/week by
  regime) and it goes live only after the paper parity read holds (owner GO, earliest Wednesday).
* **HOD-break at the 1.5 % stop floor is a zero-edge book at the measured cost** (+0.007 / −0.028 R TRAIN / VAL,
  t −2 on VAL). The floor turns the −0.11 R sub-1.5 % fills (44 % of the old book) into "not traded" — a loss avoided,
  not a win. Twelve re-read cells (1,660–1,671) at the measured cost found NO filter, exit, entry width, volume, shape,
  pattern or post-entry cut that is positive on both halves. Details in §3.
* **Small sleeves that pass on their own bar:** turn-of-month index sleeve (paper, ≈ $85/month at the tested size,
  fires 4 nights a month; first paper entry 9/30 19:52 UTC). Everything else on the ideas web FAILED or is suspended
  (auction imbalance, VIX carry, leveraged flow, EDGAR run-up, sector momentum).
* **Money 9/29:** live account flat since 14:13 UTC. Live HOD −$132 (TWST/MRNA/NBIL stops −$148, hand-flattens +$16)
  + CDNA over-exit cover −$92 = **−$224 live**. HOD paper +$15 realized (CONI 25 sh at target) + 87 sh CONI open;
  TTAN −$17 paper. ORB paper AXTL open (+$170 mark at 18:00 UTC). PENDING: the 20:02 UTC close.

## 2. Books and the machine (state at boot 9/30 12:30 UTC)
| book | account | state | what changed 9/29 |
|---|---|---|---|
| HOD-break | paper PA39QSZR60WC | ON, min_r_pct 1.5, real paper orders | fill detection (GET before/after cancel, REST poll, boot adoption), exit qty clamped to broker, OCO parser, state file per account, R2G sync bug (self.STRATEGY_NAME), adoption as filled + DB merge, reconcile checks the broker, leg-less restores get a StopMonitor watch, dry ledger writes paper/live fills + arm-time telemetry (PENDING commit) |
| ORB B+ | paper PA3YNVRTKFMG | ON, 5-s pre-placement delay, catalyst OFF, **target-resting-limit ON for 9/30** (bar ≤ 5 bps) | BT parity fix (catalyst veto read from orb.yaml), book regenerated (2025–26 n 483, +0.107 R) |
| Bull flag | live | PAUSED (enabled: false) | — |
| TOM sleeve | paper | cron 19:45/20:45 UTC | first entry 9/30 at the close |
| Live account | — | FLAT; guardrail acknowledgement cutoff live | trade records corrected (rows 381, 393–400), CDNA/PRIM/ASTN/WRBY/TTAN/CONI |
Telegram: dedupe 60 s + 20/min cap; the 9/29 error storms (144 "order not found", OCO parser, CONI "unreconciled") each
have a root-cause fix committed. Incident write-up: `docs/hod_live_incident_20260929.md`.

## 3. Everything tested at the measured cost (entry 7 bps, stop 6, target 0, EOD 11; both halves; bar = net ≥ +0.05 R, t ≥ 2.5 both halves, ex-top-5 % > 0, ≥ 3 fills/wk)
| cell | what | result | verdict |
|---|---|---|---|
| 1,658 | P&L by stop distance | < 1.5 %: −0.11 R both halves (t −4); 1.5–3 %: −0.01/−0.07; ≥ 3 %: −0.08/+0.14 | floor 1.5 % shipped to paper (loss avoided) |
| 1,660 | exit lab (35 variants) | base −0.014/−0.061; best variant tail-carried | FAIL |
| 1,661 | entry limit width | +0.10 %: +0.047 R on n 31 | watch on the ledger (the 0.15 % session contains it) |
| 1,662 | re-score of 16 families | nothing; 1,488 flagged; 1,619 artifact | FAIL |
| 1,663 | 7 cost-axis cuts | 0/24 causal reads | FAIL |
| 1,665 | relative volume to the arm minute (owner) | on the completed store (coverage 99 %): low-RVOL tercile +0.03 R, high −0.07 R on both halves, t ≤ 1.3, MDE 0.05; 0/48 reads | FAIL (adequately powered null) |
| 1,666 | rebuild of the 1,488 pyramid | paired −0.043/−0.044 R, t −5, MDE 0.02 | REFUTED |
| 1,667 | every causal feature (owner) | daily flat; intraday flat (10 reads t ≥ 2.5, all negative); n_cross = leak | FAIL |
| 1,668 | post-entry failure detection (owner) | 0/36 rules; classifier AUC 0.63–0.67 oos, cut −0.05..+0.02 R; patterns ≈ 0 | FAIL |
| 1,669 | fast failures + cut decomposition (owner) | PENDING | PENDING |
| 1,670 | feature-timing map (owner) | PENDING | PENDING |
| 1,671 | raw-sequence model vs trees (owner's TSFM question) | PENDING | PENDING |
| 1,672 | pre-holiday index sleeve (sister of TOM), 2016–2026, 102 events | SPY −17 / −4 bps per event (t −1.4 / −0.2), null percentile 12 / 35, mirror ≈ 0; QQQ and open-exit variants the same | FAIL |
MDE of the floored book: 0.077 / 0.066 R per half — a cut must carry ≥ +0.10 R on a third of the book to be visible.

## 4. Real winners (flagged)
1. **ORB live config** — the book to scale; the ramp advance needs 40 live fills; frequency is the limit, not edge.
2. **The measured cost itself** — every HOD verdict before 9/29 charged 3× the real cost; the measurement (entry +6.7
   bps, stop −2.6/−6 bps on live fills) is reusable across every book and is why the floor exists.
3. **Turn-of-month sleeve** — small, positive, bounded, stacks with the day books (paper first).
4. PENDING: anything from 1,669–1,671 that passes and survives its independent rebuild.

## 5. Refuted or void (do not re-test)
Confirmation entry / no-withdrawal pyramid (1,487/1,488/1,666); OFI filters; same-day volume ratios; relative volume
(pending the final re-run); candle patterns as filters or exits; every price-path exit (time stops, locks, trails,
dips); n_cross (full-day count).

## 6. Next steps (pre-committed, no owner action needed unless marked)
1. 9/30 paper: HOD floored session with telemetry (first clean session with real fills = the bar for any live day);
   ORB paper with the target resting limit (bar ≤ 5 bps); TOM paper entry at the close. Daily brief at 20:10 UTC.
2. ORB live: **owner GO** on Wednesday only if the 9/30 paper parity holds (fills match the BT book, exits ≤ 5 bps).
3. HOD: forward read at 100 paper fills, buckets 1.5–3 % and ≥ 3 % separately; no more filter cells on this
   population; next HOD frame = a NEW signal definition (ranked ideas list), own PREREG.
4. TSFM: decided by 1,671 — a GPU fine-tune opens only if the sequence model beats the trees by ≥ 0.03 AUC on both
   scorings and the achieved precision is within 0.10 of break-even.
5. Options v3 (Databento, $32 pull running): cell 1,599 verdict when the pull completes (watcher armed).
