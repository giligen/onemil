# Owner report — Friday 2026-10-02 morning (written 10/1 evening, session owner)

## 1. The 10/1 paper session — lost to load, nothing else
| Book | What happened |
|---|---|
| ORB paper (production + P1) | **0 decisions, 0 fills.** The scanner's tick loop ran > 50 s per tick from the open (313 "tick TIMEOUT" errors); the 09:35 decision never ran, so neither production nor the new P1 pool was evaluated. Cause: research jobs on the 2-CPU node (the band-23 build + the gap-gate INFO logging at 700 lines/min, logging fix committed 63743c5). |
| HOD paper ($104.8K) | 8 trades. Closed intraday: BMNU −$50, CNXC −$42, EFOR −$37, LITE +$46 (−$83). Four positions (APMD, MSTU, MSTX, COHR) missed the 15:55 ET flatten (the engine's close orders only went out at 20:04 UTC, after the close, so they could not fill). I cancelled them and flattened all four after hours at 20:11 UTC: APMD −$55, MSTU +$35, MSTX +$42, COHR +$66 = +$87, booked in the ledger (`scripts/ops_fix_trades_20261001.py`, exit reason `eod_ops_ah`). Day: 8 trades, +$5. |
| TOM sleeve | No-op (not a turn-of-month session); QQQ 26 held on the ORB paper account as designed. |
| LIVE ($64.8K) | Flat, untouched. |

**Defect found at shutdown:** the ORB force-close verify loop keeps polling Alpaca and Telegram after the
interpreter starts shutting down (24 ERROR lines 20:04–20:10, "cannot schedule new futures"). Harmless to money,
noise on Telegram. **Fixed tonight** (shutdown-aware verify loop, Telegram logs a WARNING at shutdown, the HOD
force-close stops resubmitting once the regular session is over and warns once, a tilt boot log line; 9 new tests,
full suite 4,898 green; `docs/orb_shutdown_hygiene_20261001.md`). The 6-minute hang itself was the stalled tick
loop draining a universe-scan backlog after the close — the research-load problem, handled by the pause window.

## 2. Research results since the last report (every number independently rebuilt)
**The ORB stack (production selection unchanged + RVOL risk tilt + one add at +1 R), Q3 2026, $375 base risk:**

| Variant | Q3 total | Green wks | Worst wk | Max DD |
|---|---|---|---|---|
| Base (corrected by 1,699: the 06-29 week had 3 losing fills the first table missed) | −$304 | 4/10 | −$2,077 | $2,077 |
| + RVOL tilt (1.5× low / 1.0× mid / 0.5× high tercile) | ≈ +$2.4K | 5/10 | −$2,385 | $2,385 |
| + tilt + add one unit at +1 R (original stop) | **+$4,417** | 5/10 | −$4,739 | $4,739 |

* The tilt is the only layer robust in BOTH directions (EV per unit of risk +20 % select-2025→test-2026, +13 % the
  reverse, slice ordering agrees; engine parity 100 % on the tercile).
* The add at +1 R is same-signed both ways but **tail-carried**: paired ΔR vs base+tilt +0.06 R (t 0.3) in 2025 and
  +0.32 R (t 2.1) in 2026, ex-top-5 % −0.16 / −0.05. It is a runner amplifier — EV positive, concentrated in the top
  5 % of fills, worst week doubles. Judge it on ≥ 40 paper fills, never on a week.
* Cells 1,698 + 1,699 (14 more layers: second add, add-unit stops, day budget, 2-D tilt, SPY-context tilt, base
  partial, add-level sweep, lock sweep for two units, P1 on the stack, gapper-count tilt, compounding): **none joins**.
  P1 on the stack drags Q3 from $4,417 to $2,213 (P1 is regime-negative this quarter); the +0.5 R / +1.5 R adds and
  2 units fail the tail bar.
* Compounding the stack at 0.5 % of equity with the above-water ramp: $65K → $199K over 2025-01..2026-09 (3.06×),
  max drawdown $22K, Q3 2026 +$11.1K. Informational: in-regime 2025–26 only; 2024H2 has 0 fills on this book.

**Momentum side project (every Monday, best trailing-year names):**
* Naive top-N by raw trailing return LOSES on every variant (−84 % to +25 %/yr net, below all 1,000 random draws):
  micro-cap hype names and a warrant.
* The literature's definition (common stocks, price ≥ $10, ADV ≥ $20M, 12-1 return): net +21–33 %/yr but **alpha vs
  SPY negative for every weekly book**; the monthly-rebalanced decile is the only positive-alpha book (+8.8 %/yr,
  Sharpe 1.19, DD 17 %, turnover 30 %/month) on 13 months — alpha −27 % / +9 % by half, so it fails the both-halves
  bar. Phase 2 (free Alpaca history to 2016, monthly decile only) on your word; the weekly cadence is the drag.

## 3. Decisions taken for Friday 10/2 (paper, one mechanics change per book)
1. **ORB production: RVOL tilt ON and the add at +1 R ON** (`sizing.rvol_tilt` + `exit.add_on` in orb.yaml, production
   only; owner 10/1: "why wait on a paper account"). P1 stays plain. Attribution if something misbehaves: the add has
   its own `[ORB] ADD` log tag and `pattern_data.add_on`; the tilt shows in the sizing log per pick.
2. **Research pause in the decision window:** every research job is SIGSTOPped 13:27–13:47 UTC (09:27–09:47 ET) and
   resumed after; outside that window research runs as you asked. If tick timeouts still appear at 09:30, the pause
   extends to the close and I tell you.
3. HOD paper: unchanged (1.5 % floor, real paper orders). TOM: QQQ held.
4. Checks armed: 12:36 UTC boot, 13:42 decision, 20:02 close (flatten any carried position after hours as today),
   22:10 parity read.

## 4. Research running tonight (quiet window, sequential, one writer on the bar store)
1. Cell 1,693b: the 2–3 % gap band seed, its sub-pools × their own exits — bars and features BUILT (22,424 in-band candidates, 99 % coverage); the pools/score stages were killed twice by memory pressure and are queued behind 1,689c, same session freeze.
2. Cell 1,689c: the first real IN-REGIME read of pools 21/24/25/26/27/28/30 (pre-market-high break, pre-market
   turnover, compression, opening drive, …) — 1,689b could only read them out of regime (24: +0.16 R on 20 fills,
   30: +0.26 R on 6). Same pass bar, both halves. It queues behind 1,693b and, because it reads the scanner's own
   cache.db, it is frozen 12:25–20:05 UTC on 10/2 so it cannot stall the session (today's failure mode).

## 5. Your calls
* Momentum phase 2 (free data, monthly decile only): GO / no.
* The tilt to LIVE after one paper session whose picks and fills match the book: your word, not mine.
* Research during market hours: my pre-commit is the 20-minute pause above; say so if you want research stopped for
  the whole session instead.
