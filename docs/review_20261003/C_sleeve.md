# Track C review - momentum sleeve paper (2026-10-03)

Scope: scripts/momentum_sleeve.py, trading/momentum_sleeve.py, BT 1700s/1700u, crontab. Read-only. Tests not run (no `python` on PATH in this shell; use python3).

Parity checks that PASS (no finding): percentile = strict-less + half ties over the trailing 252 ratios incl. the as-of one (trading/momentum_sleeve.py:158-178 vs 1700u:165-167, min 126 obs identical); 273-bar hygiene lookback, +200%/-75%/10-day gap rule (trading/momentum_sleeve.py:103-132 vs 1700s:55-62); half-size = 1/(2N) per name at pct<20 (:204-213); signal at Friday close; selling uses broker qty floored to 9 dp; coid per (day,symbol,side) makes a same-day retry idempotent; account guard + dedicated MOM keys.

## Findings

1. MEDIUM-HIGH, wrong vs BT / silent waste. scripts/momentum_sleeve.py:212-219 + 664. The completeness "gate" only logs; the run proceeds to submit on a partial panel. Docstring says 90% but code errors below 50%, and the denominator includes every tradable asset (OTC-like), so it cannot discriminate. A lost 200-symbol batch (fetch_panel :186-190) silently drops liquid names and the top-20 is picked from the remainder. With --skip-fetch (the 13:45 cron) `lost=[]` and the check is blind to the 11:50 loss. Trigger: any batch failure/timeouts at 11:50 Monday. Fix: persist `lost` beside the cache; in --submit refuse (rc 1 + Telegram) if any batch failed or the share of eligible-ADV names with an asof bar < 95%.

2. MEDIUM, retry timing. crontab 14:45 UTC line. In EDT (Monday 10/5) 14:45 UTC = 10:45 ET, inside the 09:31-15:30 window. `last_rebalance` is written only at the very end (:677), so any crash/timeout after orders but before save_state lets the 14:45 line trade a second pass an hour late at different prices (coid reuse skips existing orders, but unsubmitted legs go in at 10:45). Not a duplicate-fill risk, a drift-from-BT risk. Fix: write `last_rebalance` right after execute() (before ledger/notify), or gate the 14:45 line on `now_et.hour==9`.

3. LOW-MEDIUM, REJECTED handling. :388-404, :567-577. Universe filters on `tradable` only (:119-122), not `fractionable`; buys are notional, so a non-fractionable name is REJECTED, logged ERROR, no retry, no Telegram, and 1/N stays cash all week (ledger/fills show fewer than orders only in the "fills x/y" count). Fix: filter fractionable in fetch_assets, or fall back to whole-share qty = floor(notional/price).

4. LOW. Gate n/a (CBOE missing or 60 s timeout, :472-485) => FULL size, one WARNING, no retry, no Telegram flag. Direction is BT-neutral (ungated REF) but means a calm week silently trades 2x the intended size. Fix: retry CBOE x3, put "gate n/a -> FULL" in the [MOM] message.

5. LOW. A stale CBOE close (Friday file not yet updated) is accepted with a WARNING; percentile then reflects Thursday. Fix: refuse to half-size on a stale gate or retry once.

6. LOW. poll timeout (180 s) returns qty 0 for a still-open order (:407-433, :569-573): it is omitted from the ledger even if it fills later; state resync fixes positions but ledger/slip stats miss it. Fix: re-poll once after sync_state_from_broker and append late fills.

7. INFO. Live trades 09:45 ET market orders vs BT Monday open (acknowledged). DD kill rule is message-only. state.json cash = -8.0 after the forced 10/2 run (harmless; broker_sleeve_cash resyncs from orders each run).
