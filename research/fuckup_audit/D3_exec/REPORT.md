# D3_exec — execution pathology on the 22 real live exits

Run 2026-09-17. **READ-ONLY.** `data/trades.db` and `data/cache.db` opened with `file:...?mode=ro`; nothing under `trading/`, no config, no service, no order, no cron touched. All output under `research/fuckup_audit/D3_exec/` (`d3_walk.py`, `d3_quant.py`, `events_walk.txt`, `events.csv`, `quant.csv`, `hcai_journal.txt`). One process, `nice -n 10`, `ulimit -v 1200000`.

Journal coverage: `journalctl -u onemil-trader` has been vacuumed back to **2026-09-09**, and `logs/onemil.log*` rotations cover only 2026-09-17. Exactly one of the 22 events (HCAI 2026-09-11) has a live log trace; it is reproduced in §1.11 and it confirms the mechanism the other 21 are read off telemetry.

---

## 0. Headline, in dollars

| | |
|---|---|
| Events | 22 (ORB 4, BF 14, ignition 1, macd_wave 3) |
| Sum of their booked P&L | **−$7,568.11** |
| **Execution slip** = (exit fill − the bid the engine had in hand) × qty, 11 events with real exit telemetry | **−$1,352.40** |
| — of which bull flag | −$809.21 = **45.0% of the BF book's entire net P&L** (+$1,799.08) |
| — of which ORB | −$536.87 = **11.2% of the ORB book's net loss** (−$4,803.36) |
| — of which ignition | −$6.32 (2.4% of −$260.38) |
| Worse than a clean stop fill (`stop × 0.999`), 7 events where the stop is the right benchmark | **−$989.69** |
| Unbooked P&L in the 4 `unknown_exit` rows (booked at exactly $0.00) | **−$1.7K to −$6.8K**, central **−$6.5K** (NPT alone −$4,818) |
| `never_reconciled` rows with no terminal state, $0 booked | 4 (BF, 2026-03-17/18), notional $26.8K–$123.5K each |

**Two prior attributions in this tree are overstated and must be corrected:**

1. `D1_orb/REPORT.md` §1: *"−$1,959 of that is two `stop_loss_market_fallback` exits (EHGO 6/25, IRE 6/29) — execution pathology"*. −$1,959.08 is those two trades' **entire** P&L. Their planned stop risk was −$1,521.39; the **execution excess beyond a clean stop is −$437.69** (EHGO −$301.44, IRE −$136.25), −$463.79 against the bid the engine was quoted. The B+ survivor line should read **−$1,937 / 23 trades** (trades kept, exits repriced to a clean stop), not −$416 / 21 trades (trades deleted). That is not "flat inside its own noise".
2. `REVIEW_FRESH_EYES.md` §1: *"`stop_loss_timeout` 1 trade −$3,738 (28% of all BF stop $)"* and *"BF ≈ −$5.0K on a +$1.8K book"*. EEIQ's **planned** risk was $3,504.66 (9,375 sh × $0.3739, a $75K notional on a $50K account); the execution excess is **−$160.68**. The honest BF pathology number is **−$809.21**, not −$5.0K — still 45% of the book, still the single largest controllable line item, but from a different mechanism than "the exit hung".

---

## 1. Per event, minute by minute

Sign convention: a negative Δ means we did **worse** than the benchmark. `q/bid` = order qty ÷ the displayed bid size the engine recorded at pricing time. `partic` = order qty ÷ the volume of the 1-min bar the exit was submitted in. `lat` = `exit_submitted_at` → `exited_at` (includes poll-confirmation lag, so it is an upper bound on exposure). Full tapes: `events_walk.txt`.

| # | sym | date | book | booked reason | qty | pnl | stop | trigger | bid/ask @ pricing | limit (method) | lat s | fill | Δ vs clean stop | Δ vs bid | R vs bid | q/bid | partic |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 24 | EEIQ | 03-26 | BF | stop_loss_timeout | 9,375 | −3,738.03 | 7.68 | 7.67 | 7.67 / 7.72 | **7.70** (quote_tight) | 38.6 | 7.6552 | −160.68 | **−138.93** | −0.04 | **46.9×** | 2.0% |
| 106 | PN | 05-07 | BF | bracket_sl_race | 610 | −220.94 | 6.6240 | — | — | — | — | 6.6111 | −3.84 | — | — | — | — |
| 218 | FJET | 06-12 | BF | market_fallback | 2,415 | **+122.84** | 6.5207 | 6.75 *(trail)* | 6.75 / 6.77 | 6.74 (stop_bid) | 50.6 | 6.70 | +448.76 | **−120.75** | −0.39 | 2.4× | **30.6%** |
| 237 | ARQQ | 06-17 | BF | market_fallback | 298 | −149.00 | 22.82 | 22.81 | 22.80 / 23.05 *(110 bps)* | 22.73 (stop_bid) | 23.4 | 22.64 | −46.84 | **−47.68** | −0.50 | 0.3× | 3.5% |
| 247 | SMCX | 06-22 | BF | market_fallback | 1,413 | −197.68 | 12.32 | 12.56 *(trail)* | 12.56 / 12.58 | 12.55 (stop_bid) | 52.6 | 12.4501 | +201.24 | **−155.29** | −0.41 | 3.5× | 1.3% |
| 306 | RBNE | 07-16 | BF | market_fallback | 2,841 | −553.45 | 4.685 | 4.68 | 4.68 / 4.74 *(128 bps)* | 4.66 (stop_bid) | **68.6** | 4.5727 | **−305.81** | **−304.91** | **−1.30** | **14.2×** | **46.8%** |
| 347 | TJGC | 08-18 | BF | bracket_sl_race | 833 | −241.57 | 4.7484 | 4.74 | 4.75 / 4.77 | 4.74 (stop_bid) | 55.5 | 4.70 | −36.40 | **−41.65** | −0.21 | 2.8× | 1.3% |
| 257 | RUN | 06-24 | ORB | market_fallback | 115 | −104.65 | 15.29 | 15.29 | 15.29 / 15.30 | 15.28 (stop_bid) | n/a | 15.255 | −2.27 | **−4.02** | −0.04 | 0.2× | 0.1% |
| 263 | EHGO | 06-25 | ORB | market_fallback | 2,962 | −1,291.72 | 4.33 | 4.33 | 4.33 / 4.37 *(92 bps)* | 4.32 (stop_bid) | n/a | 4.2239 | **−301.44** | **−314.27** | −0.32 | 4.2× | 14.3% |
| 269 | RCAT | 06-29 | ORB | market_fallback | 1,384 | −345.86 | 9.71 | **9.11** ⚠ | 9.78 / 9.79 | 9.77 (stop_bid) | n/a | 9.7301 | +41.26 | **−69.06** | −0.18 | 1.7× | 3.5% |
| 271 | IRE | 06-29 | ORB | market_fallback | 712 | −667.36 | 18.64 | 18.64 | 18.64 / 18.68 | 18.63 (stop_bid) | n/a | 18.43 | **−136.25** | **−149.52** | −0.29 | **7.1×** | 3.3% |
| 371 | HCAI | 09-11 | ign | market_fallback *(really `stage_force_flat`)* | 421 | −92.62 | 2.1002 | 2.1002 | 2.17 / 2.21 | 2.16 (stop_bid_rest) | n/a | 2.155 | +23.95 | **−6.32** | −0.05 | 2.1× | **30.2%** |
| 80 | APLX | 04-14 | BF | thin_liquidity_reject | 599 | −58.63 | 17.82 | — | — | — | 0.85 | 18.0009 | +119.03 | — | — | — | — |
| 345 | LUNL | 08-13 | BF | thin_liquidity_reject | 602 | −29.44 | 5.1024 | — | — | — | 1.11 | 5.3711 | +164.83 | — | — | — | — |
| 13 | BDMD | 03-20 | BF | unknown_exit | 10,000 | **0.00** *(fabricated)* | 2.12 | — | — | — | — | 2.189915 *(= entry)* | — | — | — | — | — |
| 33 | NPT | 03-30 | macd | unknown_exit | 9,090 | **0.00** *(fabricated)* | 5.39 | — | — | — | — | 5.49 *(= entry)* | — | — | — | — | — |
| 34 | FBYD | 03-30 | macd | unknown_exit | 4,385 | **0.00** *(fabricated)* | 11.17 | — | 11.19 / 11.49 | 11.19 (macd_bid) | — | 11.39 *(= entry)* | — | — | — | — | — |
| 35 | SVRN | 03-30 | macd | unknown_exit | 6,821 | **0.00** *(fabricated)* | 7.18 | — | — | — | — | 7.32 *(= entry)* | — | — | — | — | — |
| 3 | RFIL | 03-17 | BF | never_reconciled | 10,000 | NULL | 12.1601 | — | — | — | — | NULL | — | — | — | — | — |
| 4 | BGLC | 03-17 | BF | never_reconciled | 10,000 | NULL | 2.582 | — | — | — | — | NULL | — | — | — | — | — |
| 6 | ACXP | 03-18 | BF | never_reconciled | 10,000 | NULL | 4.97 | — | — | — | — | NULL | — | — | — | — | — |
| 7 | LONA | 03-18 | BF | never_reconciled | 7,692 | NULL | 9.90 | — | — | — | — | NULL | — | — | — | — | — |

### 1.11 HCAI 2026-09-11 — the one event with a live log trace

```
19:45:34  force_exit(HCAI, stage_force_flat, limit_price=None) scheduled   # EOD flat, NOT a stop
19:45:34  REST stop-exit pricing — bid=$2.17 ask=$2.21 spread=$0.040 -> limit=$2.16 (stop_bid_rest)
19:45:34  cancelled bracket leg 26acc9d5                                   # <-- broker stop GONE
19:45:35  Limit sell submitted: 421 @ $2.16   id=d12ac930
19:45:45  WARNING limit sell UNFILLED after 10s — cancel + market close    # exactly _STOP_EXIT_FILL_TIMEOUT_S
19:45:45  Position closed: HCAI  id=87a43114
19:45:46  cancelled SL leg 26acc9d5-...
19:46:01  STAGED EXIT HCAI @ 2.155 reason=stop_loss_market_fallback PnL $-93
```

Bar 19:45 was `o 2.2056 / h 2.2056 / l 2.1500 / c 2.1550` on **842 shares total volume**; we were selling 421 (50% of the minute, 2.1× the displayed 200-share bid). A marketable sell limit at $2.16 — one cent under the bid — did not fill in ten seconds because the entire displayed bid was 200 shares. The market order then printed $2.155. **This is the template for all eleven.**

---

## 2. Mechanisms

### M1 — the engine cancels the broker stop and then *becomes* the stop
`trading/stop_monitor.py:3560-3576` bulk-cancels **every** open order for the symbol — including the bracket SL leg — *before* submitting its own limit. The code says so in a comment ("there's NO broker-side stop active during the fill-poll window ... up to ~20s max"). Measured: 23.4 / 38.6 / 50.6 / 52.6 / 55.5 / 68.6 s of submit→booked latency. The budget is `_STOP_EXIT_FILL_TIMEOUT_S = 10` (`:3182`) + cancel + `get_order` + `close_position` with a 2.2 s held-qty retry ladder + `_MARKET_CLOSE_FILL_TIMEOUT_S = 60` (`:3190`) = up to **~72 s unprotected**, on a position that is being liquidated precisely because it is falling.

### M2 — the limit is priced for a book that is not there
`compute_limit_price` (`:2391`) returns `bid − max($0.01, 0.30 × spread)` for the **whole quantity in one order**. In **9 of 11** events the order was larger than the displayed bid: 46.9× (EEIQ), 14.2× (RBNE), 7.1× (IRE), 4.2× (EHGO), 3.5× (SMCX), 2.8× (TJGC), 2.4× (FJET), 2.1× (HCAI), 1.7× (RCAT). A one-cent concession buys the top of book and nothing else. Worse: `_poll_order_fill` (`:3193-3226`) explicitly refuses to count `partially_filled` as success — *"a brief `partially_filled` window won't short-circuit us"* — so the order that cleans out the bid and stalls is judged a **total** failure, cancelled, and the remainder is market-ordered into the hole the first slice just made. `_verify_fill_qty` (`:2515`), which already handles exactly this, is wired only into the **partial-exit** path (`:2290`), never into `_execute_stop_exit`.

Corollary (**accounting**, unverified against broker history): the booked `exit_price` is the market order's `filled_avg_price` while the event's `shares` is the full pre-cancel qty. If the limit partially filled, the better slice is priced out of the book — for EHGO ≈ $67 of pessimism. Settled only by pulling the order history for those `order_id`s.

### M3 — the escalation is an unpriced market order
`_escalate_to_market_close` (`:3273`) calls `client.close_position(symbol)` — a market sell. On **RBNE** we were 14.2× the displayed bid and **46.8% of the minute's volume** (the 14:49 bar traded 6,075 shares total while we sold 2,841). The result is a guaranteed book-walk: −$304.91 vs the bid, **−1.30R** on a trade whose entire planned risk was $234. On **FJET** (30.6% of the minute) and **HCAI** (50.2%) the same. Size, not delay, is the binding constraint in those three; delay is the binding constraint in EHGO / IRE / TJGC / EEIQ, whose exit minutes ran −0.24% / +0.35% / **−3.29%** / **−5.74%** open-to-close.

### M4 — no print validation on the trigger
`_on_trade` (`:3018-3155`) does `price = float(trade.price)` and `if price <= watch.stop_price: exit`, with **no** SIP condition-code filter, no odd-lot/derivative exclusion, and no NBBO sanity check. That same price is written to the DB as `exit_trigger_price` and is the last-resort placeholder when nothing confirms. **RCAT 2026-06-29** recorded `exit_trigger_price = 9.11` while the engine's own quote was **9.78 / 9.79** and the consolidated 13:44 bar low was **9.7203** — the stop (9.71) was never touched on the tape. A single aberrant print liquidated 1,384 shares. (RCAT's lock never armed — `lock_arm_at_r 1.75 × R 0.24` = $10.40, and the bar high was 9.91 — so 9.11 is not a lock level either.) `grep -rn "\.conditions" trading/ data_sources/` returns **zero hits**: no code path anywhere filters trade conditions.

### M5 — March-era midpoint pricing (fixed on the `stop_bid` path, still reachable)
EEIQ's method was `quote_tight`, i.e. `compute_limit_price_from_quote` (`:2454`), whose tight tier sells at the **midpoint**. bid 7.67 / ask 7.72 → limit **$7.70, above the bid**, a non-marketable sell limit on a stock printing an 8.8% range in its own entry minute. It sat 38.5 s. That tier is still in the file and still reachable (`macd_bid` wrote it on FBYD; force-exit overrides run through `compute_limit_price`).

### M6 — the escalation BRANCH overwrites the exit REASON
`:3696-3697` sets `exit_reason = STOP_LOSS_MARKET_FALLBACK` for **any** watch that escalated, discarding what actually fired. Proof: **FJET** trigger $6.75 vs stop $6.5207, P&L **+$122.84** — a trailing-stop exit **on a winner** booked as `stop_loss_market_fallback`; **SMCX** trigger $12.56 vs stop $12.32 — also a trail; **HCAI** — journal says `force_exit(HCAI, stage_force_flat)`, booked as `stop_loss_market_fallback`. The bucket therefore contains a winner, an EOD flat and three real stops. **Every downstream number that groups on `exit_reason` is wrong in both numerator and denominator** — including `D1_orb`'s −$1,959 and `LIVE_LOSERS.md`'s pathology tally.

### M7 — `unknown_exit` books a fabricated flat
Four rows (BDMD BF, NPT / FBYD / SVRN macd_wave) carry `exit_price == fill_price` and `pnl == 0.00` exactly. Tape-implied truth:

| row | booked | tape at the exit minute | implied P&L | hole |
|---|---|---|---|---|
| BDMD 03-20 15:23 | $0.00 | bar 2.1408–2.2000 | −$491 … +$101 | up to −$491 |
| **NPT 03-30 16:53** | $0.00 | 16:52 close **4.96** (entry 5.49, stop 5.39) | **−$4,818** | **−$4,818** |
| FBYD 03-30 16:37 | $0.00 | bar 11.34–11.48; recorded limit **11.19** | −$877 … +$395 | ≈ −$877 |
| SVRN 03-30 16:53 | $0.00 | last print 16:46 **7.23** | −$614 | −$614 |

Total unbooked: **−$1,684 (floor) to −$6,800**, central **−$6,509**, of which −$6,309 is in the retired macd_wave book (already −$16,973 booked) and −$200 in BF. The contract for an unattributed exit already exists — `StopExitEvent(confirmed=False)` → `exit_pending_verification`, no price, no P&L (`tests/test_stop_exit_unconfirmed.py`) — and `unknown_exit` does not use it.

### M8 — rows inserted at submit with no terminal-state sweep
The 4 `never_reconciled` rows have `order_status='closed_unverified'` and `created_at == updated_at` to the microsecond: written once at order submit, never touched again. `pattern_data` holds only the detection. Notional at the 10,000-share cap: RFIL $123.5K, LONA $78.2K, ACXP $50.5K, BGLC $26.8K on a $50K account — almost certainly buying-power rejects, but **nothing in the system ever proved it**, and P&L is NULL rather than 0.

### Not a defect: `thin_liquidity_reject`
APLX (0.85 s post-fill) and LUNL (1.11 s) cost −$58.63 and −$29.44 — the round-trip spread. Both positions would have gone to their stops (APLX tape → 17.59 vs stop 17.82; LUNL → 5.14 vs stop 5.1024), i.e. −$167 and −$191. The reject **saved $270**. Leave it alone; the failure it patches is an entry gate, not an exit.

---

## 3. Share of each book

| book | net P&L | stop-family dollars | exec slip (Δ vs bid) | slip / net | slip / stop $ |
|---|---|---|---|---|---|
| bull_flag | **+$1,799.08** | −$18,504.19 | **−$809.21** | **45.0%** | 4.4% |
| orb | **−$4,803.36** | −$13,382.73 | **−$536.87** | 11.2% | 4.0% |
| ignition | −$260.38 | −$467.48 | −$6.32 | 2.4% | 1.4% |
| macd_wave (retired) | −$16,973.17 | — | — | — | + **−$6,309 unbooked** |

Stop-family = `stop_loss` + `stop_loss_timeout` + `stop_loss_market_fallback` + `stop_loss_bracket_sl_race`.

---

## 4. Fixes, with the counterfactual walked on the same events

### Counterfactual C3 (what the numbers below are)
Rule priced: **hit the bid immediately with a marketable limit at `bid − max($0.01, 0.25 × spread)`**, instead of `bid − 0.30 × spread` held for 10 s and then market-ordered. Obtainability is checked against the submit-minute bar (`bar.high >= price`), per CLAUDE.md §1b.

| event | actual fill | C3 price | recovered $ | obtainable at 1-min? |
|---|---|---|---|---|
| EEIQ | 7.6552 | 7.66 | +45.18 | yes |
| FJET | 6.70 | 6.74 | +96.60 | yes |
| ARQQ | 22.64 | 22.74 | +29.80 | yes |
| SMCX | 12.4501 | 12.55 | +141.16 | yes |
| RBNE | 4.5727 | 4.66 | +248.09 | yes *(upper bound — see caveat)* |
| TJGC | 4.70 | 4.74 | +33.32 | yes |
| RUN | 15.255 | 15.28 | +2.87 | yes |
| EHGO | 4.2239 | 4.32 | +284.65 | **undecidable** |
| RCAT | 9.7301 | 9.77 | +55.22 | yes |
| IRE | 18.43 | 18.63 | +142.40 | yes |
| HCAI | 2.155 | 2.16 | +2.11 | yes |
| **total** | | | **+$1,081.40** | **+$796.75 decidable** |

Per book: BF **+$594.15**, ORB **+$485.14**, ignition +$2.11.

**Caveats, stated because they are load-bearing.** (a) The four ORB rows carry no `exit_submitted_at`, so their "submit minute" is the *booking* minute; EHGO's C3 price is above the 14:00 bar high (it was inside the 13:59 bar, 4.19–4.40) and is marked undecidable rather than counted. (b) C3 assumes the **first slice** clears at the bid. On RBNE (14.2× the bid, 47% of the minute) that is false for the full 2,841 shares — its +$248 is an upper bound, and it is exactly the event that FIX-2's slicing, not FIX-2's pricing, is for. (c) These are 1-minute bars; nothing here is a fill simulation at tick resolution.

### FIX 1 — never cancel the broker stop in order to place our own
**Where:** `trading/stop_monitor.py::_execute_stop_exit` :3560-3576 (the `GetOrdersRequest(OPEN)` bulk-cancel) and :3690-3705.
**Rule:** when the watch has a live `sl_leg_id`, do **not** cancel it. Call `alpaca_client.replace_order_stop_price(sl_leg_id, bid − X bps)` (the method already exists, `data_sources/alpaca_client.py:1727`) and let the broker leg be the exit. Cancel-and-place only when there is no live SL leg (ORB's `submit_bracket_order` at `orb_engine.py:2929` and BF's both provide one). This is the design the HOD engine already ships ("the broker legs ARE the exits", CLAUDE.md).
**Counterfactual $:** not priceable on 1-min bars. What it removes is **23–69 s of unprotected exposure on every one of the 11 events**, and the two `bracket_sl_race` rows (PN, TJGC) stop being races at all.

### FIX 2 — slice to the displayed book; re-price instead of market out
**Where:** `_execute_stop_exit` :3663-3705 and `_escalate_to_market_close` :3273-3422.
**Rule:** `slice = min(remaining, max(bid_size, min_slice))`; price each slice at `bid − max(tick, cross_factor × spread)`; after `reprice_after_s` call `replace_order_limit_price` (exists, `alpaca_client.py:1756`) at the **new** bid − the same offset; escalate to `close_position` only after `max_rounds` or `hard_deadline_s`. New config block `trading.self_managed_stops.exit_ladder.{enabled, slice_to_bid_size, min_slice, cross_factor: 0.25, reprice_after_s: 2, max_rounds: 3, hard_deadline_s: 10}`; promote `_STOP_EXIT_FILL_TIMEOUT_S` / `_MARKET_CLOSE_FILL_TIMEOUT_S` out of class constants into that block.
**Counterfactual $:** the C3 table — **+$1,081** gross, **+$797** decidable, BF +$594 / ORB +$485.

### FIX 3 — a partial fill is a partial success
**Where:** `_poll_order_fill` :3193-3226; `_execute_stop_exit` :3663-3705.
**Rule:** return `(filled_qty, filled_avg_price, status)`; book what filled, work only the remainder, and emit the exit event with the **blended** price of every slice (reuse `_verify_fill_qty` :2515, already used at :2290 — do not rewrite it).
**Counterfactual $:** ≈ +$67 of booking accuracy on EHGO alone; the real value is that FIX 2 is impossible without it.

### FIX 4 — validate the trigger print
**Where:** `_on_trade` :3033.
**Rule:** ignore a print whose SIP `conditions` are not regular-way, or that is more than `max_trigger_dev_bps` (150) below the cached NBBO bid; log WARNING and fall through to the bar path.
**Counterfactual $:** RCAT is not liquidated at 13:44:17. Its stop was hit legitimately in the 13:45 bar (low 9.68 < 9.71), so the P&L delta is ≈ $0 — this fix is tail insurance, not a P&L line. One bad print on a 46.9×-the-bid position is what it prevents.

### FIX 5 — a stop exit never prices off the midpoint
**Where:** `compute_limit_price_from_quote` :2454 (`quote_tight` tier).
**Rule:** the tight tier must not be reachable from any urgent-exit path; gate it behind an explicit `urgent=False`.
**Counterfactual $:** EEIQ's limit becomes $7.66 instead of $7.70 → **+$45.18**, and the order is marketable on submit rather than 3¢ above the bid.

### FIX 6 — separate the exit REASON from the escalation BRANCH
**Where:** :3690-3705 (and the mirror at :3735, :3822).
**Rule:** keep `exit_reason` = what fired (`trail_stop`, `lock_stop`, `stage_force_flat`, `stop_loss`); add `exit_branch` ∈ {`limit`, `market_fallback`, `sl_leg_race`, `last_resort`} as its own DB column. Historic strings stay (they are load-bearing per `trading/exit_reasons.py`).
**Counterfactual $:** $0 — and without it **no execution metric in this repo is computable**, including the one D1_orb quoted.

### FIX 7 — `unknown_exit` must never write a price or a P&L
**Where:** `trading_engine.py::_handle_unknown_exit_*` (:2937-3130), `stop_monitor.py:3744`, `:3840`.
**Rule:** route every `unknown_exit` writer through the existing `confirmed=False` contract → `exit_pending_verification`, no `exit_price`, no `exited_at`, no `pnl`; Telegram alert; a daily sweep that resolves rows older than one session from Alpaca order history (`trading/bull_flag_exit_recovery.py` already does the lookup and classification).
**Counterfactual $:** **−$1,684 to −$6,800** of loss stops being invisible. The books that read `trades.pnl` (`LIVE_LOSERS.md`, `D1_orb`, `I/REPORT.md`, `live_trades_dump.csv`) are currently short by that amount.

### FIX 8 — terminal-state sweep
**Where:** wherever `closed_unverified` is written, plus a post-close job.
**Rule:** a row whose status is non-terminal at session close +15 min is a Telegram alert and a green-check HARD fail, not a silent row.
**Counterfactual $:** 4 rows, $0 booked; the exposure it bounds is a filled position nobody is watching.

---

## 5. Tests that pin each fix

Unit (extend `tests/test_stop_exit_limit_buffer.py`, or new files):

| fix | test | contract |
|---|---|---|
| 1 | `test_sl_leg_replaced_not_cancelled_when_live` | with `sl_leg_id` set, `cancel_order` is **never** called on it before a fill; `replace_order_stop_price` is |
| 1 | `test_no_sl_leg_falls_back_to_own_limit` | no `sl_leg_id` → today's path, unchanged |
| 2 | `tests/test_stop_exit_ladder.py::test_slices_to_displayed_bid_size` | qty 2841 / bid_size 200 → first order is 200 sh, not 2841 |
| 2 | `::test_reprices_after_reprice_after_s` | bid falls 4.68→4.60 during the order's life → `replace_order_limit_price` called with the new bid − offset, `cancel_order` not called |
| 2 | `::test_market_escalation_only_after_hard_deadline` | `close_position` is not called before `hard_deadline_s` |
| 2 | `::test_ladder_disabled_is_byte_identical_to_today` | `enabled: false` → today's call sequence exactly (the rollback contract) |
| 3 | `::test_partially_filled_is_booked_and_remainder_worked` | 700/2962 filled → 700 booked at the limit, next order is 2262 |
| 3 | `::test_exit_event_carries_blended_price` | slices at 4.32/4.28/4.24 → event price is the qty-weighted blend |
| 4 | `tests/test_stop_trigger_validation.py::test_print_150bps_below_bid_ignored` | RCAT tape: print 9.11, bid 9.78 → no exit |
| 4 | `::test_regular_print_below_stop_still_triggers` | print 9.70, bid 9.71 → exit fires |
| 5 | `::test_stop_path_never_prices_above_bid` | EEIQ quote 7.67/7.72 → limit ≤ 7.67, never 7.70 |
| 6 | `tests/test_exit_reasons.py::test_escalation_preserves_trail_reason` | FJET: trail armed + market escalation → `exit_reason='trail_stop'`, `exit_branch='market_fallback'` |
| 6 | `::test_force_exit_reason_survives_escalation` | HCAI: `stage_force_flat` + escalation → reason preserved |
| 7 | `tests/test_stop_exit_unconfirmed.py::test_unknown_exit_writes_no_pnl` | `unknown_exit` → `pnl is None`, `exit_price is None`, status `exit_pending_verification` |

Integration (real `StopMonitor` + real `Database` + `tests/fakes/fake_alpaca_broker.py`, which already exists for the HOD lifecycle tests):

1. `test_stop_exit_ladder_e2e_thin_book` — replay **RBNE 2026-07-16**: bid 4.68 size 200, 2,841 sh, the 14:49 tape. Assert the DB row's blended exit ≥ 4.63 (vs the actual 4.5727) and that `close_position` was never called.
2. `test_stop_exit_e2e_sl_leg_is_the_stop` — bracket live, stop breached; assert the SL leg was replaced not cancelled, the position is flat, and there is no window in the call log where neither an SL leg nor a working sell order exists.
3. `test_unknown_exit_e2e_leaves_row_pending` — replay **NPT 2026-03-30**: reconcile finds a flat position with no attributable order; assert the row is `exit_pending_verification` with NULL pnl, and that the next-day sweep resolves it from order history to a **non-zero** P&L.
4. `test_bad_print_e2e_does_not_liquidate` — replay **RCAT 2026-06-29**: inject the 9.11 print between two 9.78 quotes; assert zero orders submitted.

Every fix also needs its `enabled: false` byte-identical test (FIX 2 row above) — the CLAUDE.md rollback contract.

---

## 6. What to monitor live after the fix

Per exit (one log line, greppable):
```
journalctl -u onemil-trader | grep -E "EXIT LADDER|SL LEG REPLACED|TRIGGER PRINT REJECTED|EXIT BRANCH"
```
1. **`slip_vs_bid_bps`** per exit = `(fill − bid_at_pricing) / bid × 1e4`, logged and stored. Today's 11 events: median **−58 bps**, worst **−229 bps** (RBNE). Target median ≥ −25 bps. This is the single number the fix is judged on.
2. **`naked_window_ms`** = SL-leg-cancel → confirmed flat. Today up to 68,596 ms. After FIX 1 it must be **0** on every bracketed exit; any non-zero value is an alert.
3. **`qty_over_bid_size`** at pricing time. Alert above 3×; it predicted 9 of the 11 events. Feeding it back into sizing is a separate question.
4. **`rounds_to_fill`** and **`market_escalations_per_day`**. Escalation to `close_position` should become rare; a daily count > 1 is a regression.
5. **`exit_branch` histogram** (FIX 6). `market_fallback` must fall; `limit` must dominate. Until FIX 6 ships, `stop_loss_market_fallback` is meaningless.
6. **Zero-P&L exits**: `SELECT count(*) FROM trades WHERE pnl = 0.0 AND exit_price = fill_price` — HARD-fail the daily green check on any new row.
7. **Non-terminal rows at session close +15 min** — HARD-fail (FIX 8).
8. **Trigger prints rejected per day** (FIX 4). A sustained non-zero count means the threshold is too tight, not that the feed is broken — check it against the bar tape before loosening.

---

## 7. What this report does NOT establish

* No fill was simulated at tick resolution. Every counterfactual is bounded by 1-minute OHLCV and is marked obtainable / undecidable accordingly.
* Partial fills of the stranded limits are **inferred** from the code path, not observed. Settling them requires Alpaca order history for the 11 `order_id`s — a read-only call this audit did not make.
* The `unknown_exit` holes are tape estimates, not fills. NPT's −$4,818 is the 16:52 close; the true fill is unknown.
* n = 22 events across four books and three config eras. The mechanisms are established from code + telemetry; the **dollar totals are a census of what happened, not a forecast** of what the fix earns per year.
* The 115 pre-B+ ORB rows and 111 NULL-`exit_reason` BF rows were not examined here; `D1_orb` owns the selection question.

---

