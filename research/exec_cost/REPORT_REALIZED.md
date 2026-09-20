# Realized execution cost + touchgo cost-model read (2026-09-20)

Source: `data/trades.db::trades` (`data/cache.db`'s trades table is empty; live trades
live in the separate `trades.db`). Rows `trade_date >= 2026-06-01`, all `side='buy'`.
Quoted half-spread bps = (ask-bid)/2/mid; realized = (fill-mid)/mid buys, (mid-fill)/mid
sells, signed so + = cost to us.

## 1. Entry leg (stop-limit buy)
| strategy | n | med quoted | med realized | diff | P75 realized | %worse | odd-lot |
|---|---|---|---|---|---|---|---|
| bull_flag | 23 | 29.85bps | 68.60bps | +38.75bps | 102.60bps | 52.2% | 23/23 |
| ignition  | 13 | 130.10bps | 14.91bps | -115.19bps | 86.43bps | 0.0% | 13/13 |
| orb       | 96 | 15.20bps | 63.71bps | +48.51bps | 132.61bps | 83.3% | 96/96 |

## 2. Exit leg
| strategy/leg | n | med quoted | med realized | diff | P75 realized | %worse |
|---|---|---|---|---|---|---|
| bull_flag/market-limit | 20 | 30.27bps | 14.75bps | -15.52bps | 88.76bps | 35.0% |
| ignition/force-close | 1 | 97.09bps | 96.80bps | -0.29bps | 96.80bps | 0.0% |
| ignition/market-limit | 8 | 76.89bps | 57.17bps | -19.72bps | 148.15bps | 25.0% |
| orb/market-limit | 53 | 9.77bps | 10.62bps | +0.85bps | 22.29bps | 30.2% |

No force-close row has both quotes for orb/bull_flag (56/138 exits lack a captured exit
quote, mostly force_close/manual-reconcile). Odd-lot: 100% of all fills are odd lots
(risk-sizing means round-100 lots basically never occur) — the odd/round split carries no
signal here.

**Verdict**: ORB and bull_flag entries realize ~2-2.3x the quoted half-spread with a fat
right tail (P75 4-9x median quoted, worse-than-quoted 52-83% of fills) — BT's measured-NBBO
charge is **optimistic** on the entry leg for both live strategies. Exit legs are roughly
fair (ORB, median 10.6 vs 9.8bps) to better-than-quoted (bull_flag, 14.8 vs 30.3bps, n=20,
noisy). Ignition (n=13, paused strategy) realizes far better than quoted — not live-relevant.

## 3. Touchgo under measured cost — INCONCLUSIVE
`research/orb_frequency/walklog_tg_off.txt` (30 lines) and `walklog_tgM_off.txt` (5 lines)
both end in `numpy._core._exceptions._ArrayMemoryError` inside `study_orb_pipeline_static_lock.py`
-> `study_orb.py:_bars_to_df` (`walk_touchgo.sh`'s `ulimit -v 3000000` too tight for the
13,033-pair bar load). Neither produced `walkbook_tg_off.csv`/`walkbook_tgM_off.csv` — no
totals, MDD, trade count, or cost-model tag available. Not re-run per instructions.
Context only (not a touchgo A/B, current honest book): `analysis_results/orb_bplus_book.csv`
(2026-09-18, touchgo ON, banded cost) = 218 candidates, 165 filled, P&L $14,061.55, MDD -$642.39.

## Conclusion
1. BT's measured-NBBO half-spread is optimistic for ORB and bull_flag **entry** fills
   (2-2.3x realized, fat tail) — the biggest live/BT cost gap is entries, not exits.
2. Exit-leg realized cost is in line with or better than quoted for both live strategies.
3. Touchgo on/off cost comparison unanswerable from existing logs — both walks OOM'd
   before writing a book; needs a memory-limit fix and re-run (out of scope here).

## Correction — entry leg vs fill-time quote (2026-09-20)
Original §1 compared fill to the quote at SUBMIT (~09:35, before the stop-limit triggers
at range_high x 1.003) — most of the +49bps/+39bps was intended trigger distance, not
execution cost. `trades` has `entry_fill_quote_bid/ask` (captured at fill confirmation).
Recomputed realized = (fill - mid_at_fill)/mid_at_fill, quoted = half-spread at fill:

| strategy | n | med quoted | med realized | diff | P75 realized | %worse |
|---|---|---|---|---|---|---|
| bull_flag | 23 | 33.92bps | 12.55bps | -21.37bps | 43.54bps | 26.1% |
| orb | 96 | 13.50bps | 3.08bps | -10.42bps | 26.92bps | 31.2% |
| ignition | 13 (6 missing) | 136.99bps | 0.00bps | -136.99bps | 166.96bps | 15.4% |

**Restated verdict**: against the fill-time quote, realized entry cost is BELOW the quoted
half-spread for both live strategies (ORB: 3.1 vs 13.5bps; bull_flag: 12.6 vs 33.9bps) —
BT's measured-NBBO half-spread charge on the entry leg is **pessimistic** (overcharges), not
optimistic. The earlier +49/+39bps "cost" was mostly the stop-limit trigger offset baked
into submit-time quotes, not real slippage.
