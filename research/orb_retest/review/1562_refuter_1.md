# Refuter 1 — cells 1,562/1,563, lens: obtainability and look-ahead

Script: `review/refute_1562_obtain.py`. Rows: `review/refute_1562_obtain_rows.csv`. It re-walks every scored fill from the tape and bars (cache.db opened read-only).

## Checks that pass
- The retest window starts at the trigger print. It uses `ts_event > trigger_ts`, with trigger_ts = 09:35 + t_star, and ends 15 or 30 minutes after the trigger print. It does not start at 09:35 or at the range close. OK.
- A tape fill needs an XNAS.ITCH print strictly below the limit. On one venue with price priority, that print means a resting bid at the limit was fully filled, so queue position is not an issue. 42 % of the fill-triggering prints are odd lots under 100 shares, and 33 % of fills have fewer than 500 shares printing below the limit in the next 60 s. The price-priority argument still holds.
- Halts: 6.4 % of tape fills have a gap between prints longer than 60 s before the fill. The longest is 128 s. There is no 5-minute halt in the fill window.
- Guard-skipped signals are booked as base = 0. The range high is known at 09:35. The stop is range_low and is not moved.

## Defects (all favour the retest leg)
1. **The fill price is the first print below the limit, not the limit.** Median 2 c, mean 0.10 R′, max 2.7 R′ of price improvement on 204 tape fills. That improvement is not obtainable: a resting bid fills at its own price.
2. **Look-ahead on the fill bar.** For bar fills, the fill bar's high can arm the lock (+0.5 R′) even when that high came before the dip. For tape fills, the rest of the fill minute is skipped entirely, so a stop hit there is missed. The BT walker skips the entry bar (`post.iloc[1:]`).
3. **Gap-through stops fill at the stop instead of at the bar open.**
4. **The base leg is not symmetric.** It uses pnl_replay/375, which is the full BT rule (touchgo, SZ1 floor, 15:45 close) in sized-$ units. The retest leg uses the static-lock core in its own R′.

## Effect (paired ΔR, day-clustered t, pooled)
| | 1562 | 1563 |
|---|---|---|
| as published | +0.185 (t 1.40) | +0.180 (t 1.33) |
| after fixing defects 1–3 (limit fill, no fill-bar look-ahead, fill at the open on a gap-through) | +0.132 (t 1.05) | +0.123 (t 0.94) |
| **after fixing 1–3, against the symmetric base** (replay chase fill, same exit code, own R, same costs) | **−0.054 (t −0.64)** | **−0.063 (t −0.70)** |

By split, against the symmetric base: 1562 TRAIN −0.09 / VAL −0.05; 1563 TRAIN +0.005 / VAL −0.08. The retest leg's own VAL mean falls to +0.195 (1562) and +0.152 (1563), and its ex-top-5 % stays negative on VAL.

## Verdict
The FAIL verdict stands, and every correction makes it more of a FAIL. RESULT_1562.md also says "paired ΔR positive on every split, so the HOD learning generalises directionally". That claim is **refuted**. The positive sign comes from comparing the legs under different exit rules and cost units, plus the fill-price improvement. When both legs use the same exit code, the retest bid is flat to negative against the chase.
