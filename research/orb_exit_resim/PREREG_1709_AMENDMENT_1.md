# PREREG 1,709 — Amendment 1 (2026-10-29 evening, Fable). Paper book re-enabled; verdict on capital unchanged.

## What is amended
PREREG_1709 froze "fail ⇒ code frozen, paper book stopped, no further ORB cells". The verdict (FAIL, `REPORT_1709.md`)
stands for CAPITAL and RESEARCH: no live ORB orders, no new ORB cells, no code changes. The clause "paper book stopped"
is withdrawn: `orb.yaml strategy.enabled` returns to `true` for the 2026-10-30 boot, ORB paper account (PA3Y) only.

## Why (adequacy gap in the test, found 10/29 20:30 UTC, prompted by the owner's "why broken with this week's P&L")
* The frequency criterion in 1,709 (≥ 3 replay-filled fills/week; measured 2.4 in regime, 1.5 out) was measured on the
  REPLAY fills (BT entry model on XNAS ticks). The paper book since the 9/25 pause fills ~5.4 symbol-days/week
  (27 symbol-days, 56 entry fills in 5 weeks; broker orders, read 10/29). The replay understates the live config's
  frequency; the frequency fail therefore rests on the wrong instrument.
* The paper book is free ($0 capital) and is the only forward instrument that can revise the verdict. Stopping it was a
  research-economy decision with no money rationale.
* Not amended: the per-fill-R and tail criteria. The paper record since 9/25 is itself tail-carried: +$1,632 on 25 days,
  8 green / 8 red, best day +$2,031 (10/27: CRDU/LABX/NEBX/NBIG to the force-close), ex-best-day −$399; weeks
  0, +54, +426, −694, −216, +2,060. One $2K day does not change a 123-fill live record of −$5,281.

## Pre-committed rule (frozen now, before any further number)
Judged by `scripts/orb_ramp_check.py`'s existing above-water rule, applied to the PAPER book from 2026-09-25:
* At 40 paper entry symbol-days (27 now): the checker's existing conditions — realized paper P&L since 9/25 > 0 AND
  above water EX-MONSTER (best single fill removed; today −$399, i.e. currently failing) AND ≥ 15 sessions → live ORB
  resumes at the $375-risk exploration stage under the existing auto-pause tripwire; otherwise → ORB shelved for good,
  paper off, no appeal.
* No re-reading before 40; no filter, exit or sizing change to the paper book in the meantime (code frozen).
* The owner's "live exploration tier" principle (positive point estimate + mechanism + bounded downside + resolves in a
  quarter) is the basis; FREQUENCY is measured on the paper book, not the replay.

## Open item
A 32-share TXN market BUY at 15:31 UTC 2026-10-27 on the ORB paper account with no matching sell is not an ORB-shaped
order (ORB entries are stop-limit/limit at 13:35–13:46). Identify its source on 10/30 (journal grep `TXN`, orders API
client_order_id); if it is an engine order it is a defect to spec.
