# BF consistency program — findings and plan (2026-09-06)

Owner: "shaving the P&L significantly but gaining WR and consistency is the right move."
Goal: 30–50%/yr on capital delivered monthly, not a headline total.

## 1. Diagnosis (regen-7 exact exits, live universe rule, $2K risk)
- Losses are bounded and identical across years (19 losses, −1.33R avg, −$95K each year).
- The win side collapsed: 2025 27 wins +2.29R ($214K, top-5 $85K); 2026 15 wins +1.69R ($83K, top-5 $54K). WR 59% → 44%.
- Raw candidate edge ≈ 0 in every half-year (mean +0.04R, median −1.05R): the tail pays, most trades stop out. The Stage-2 stack keeps 9% of the raw pool; lever isolation shows the fitted layers are NOT the 2026 culprit (TTF off makes 2026 worse; the rest are inert at Stage-2).
- Capacity is not the cap: max positions 10 + no daily-loss cap + $1M buying power adds 3 trades in 20 months. 4 trades/month is the stack's yield.

## 2. Exit-profile grid (faithful resim on 705/745 rows; relative tool)

| variant | trades | WR% | total | 2025 | 2026 | green | worst mo | MDD | top-5 % |
|---|---|---|---|---|---|---|---|---|---|
| V0 as-is | 78 | 57.7 | 98,158 | 110,171 | −12,013 | 13/20 | −11,124 | −27,833 | 87 |
| V1 partial 50% @+1R + BE | 78 | 61.5 | 63,970 | 65,915 | −1,945 | 13/20 | −14,939 | −31,728 | 91 |
| **V2 partial 50% @+1.5R + BE** | 78 | 55.1 | **105,105** | 98,870 | **+6,236** | 13/20 | −14,938 | −25,955 | **65** |
| V4 exhaustion @2R | 78 | 57.7 | 96,964 | 109,671 | −12,707 | 13/20 | −11,124 | −28,527 | 81 |
| V5 no-pop 10 bars | 78 | 55.1 | 73,524 | 85,537 | −12,013 | 13/20 | −11,124 | −27,833 | 106 |
| V7 trail arms @1.5R | 78 | 61.5 | 72,250 | 79,953 | −7,703 | 12/20 | −10,702 | −26,825 | 118 |
| V8 breakeven @1R only | 79 | 41.8 | 25,403 | 65,695 | −40,292 | 11/20 | −20,109 | −51,776 | 325 |

(V1 = V3 = V6: the partial-profit path is a legacy BT-only simulate branch that ignores trail/no-pop knobs — and live has no +NR partial exit at all. Any partial profile must be BUILT on the unified trail spec, BT + live, before it is trusted.)

Pre-committed bar (2026 ≥ 0, ≥ 70% green months, worst month ≥ −$8K, MDD ≥ −$20K): **none pass.**

## 3. What the numbers mean
- **Exits change magnitudes, not the sign of months.** Green months are 13/20 in every variant: with 4 trades a month at ~55% WR the month's sign is a 4-sample coin flip. No exit profile makes that consistent.
- **V2 is the right direction** for the book's shape: 2026 flips positive (+$6.2K), the tail is less load-bearing (top-5 65% vs 87%), total P&L is not sacrificed. Its failure is the worst month (−$14.9K, one trade) and the month count.
- **Breakeven-at-1R alone is poison** (2026 −$40K): the book's winners need room; this is the knife-edge the IONX case showed.
- **Monthly consistency for BF can only come from (a) more independent trades of the same quality, which the current signal does not supply, or (b) the portfolio**: the row-2 mix (ORB $50K + BF $500 + IGN $500) was 15/20 green months because ORB and ignition fill BF's red months (research/portfolio_comovement_row2_20260905.csv).

## 4. Plan (proposal — joint decision)
1. **Build the consistency exit properly (Mon 9/7)**: partial 50% at +1.5R → stop to breakeven → remainder on the unified trail (trading/bf_trail.py), implemented in BT `simulate` (not the legacy branch) AND live (StopMonitor partial + trading_engine partial order), with the same one-tape parity test as the trail unification. Then resim + Stage-2 + walk-forward (fit on 2025, read on 2026; then the reverse). Ship candidate if 2026 ≥ 0 holds and the bar's other legs move toward it.
2. **Frequency study on consistency metrics (Tue 9/8, overnight regens on 2026 only)**: Stage-1 threshold 20% → 15%, and the 2-candle pole (rejected in May on recency) re-scored on green months / worst month rather than total. Only more trades of equal quality can move the month count.
3. **Sizing for month variance**: risk per trade sets the month swing (±$10K at $2K). The retirement plan sizes BF at $500–1K, which caps month swings at ±$2.5–5K while the frequency work runs.
4. **Live**: BF stays at $60 with the trail fix from Monday; the consistency exit shadows for 10 sessions before it trades.
5. **ORB next** with the same lens (it is already 15/21 green; its problem is size, and the signal study says its entries are not the lever).

## 5. Profit partial built on the unified spec (2026-09-06 afternoon) — honest numbers

Built as ONE spec for BT and live (`trading/bf_profit_partial.py`, commit 5a8285f; 241 tests incl. one-tape parity). Re-simulated on regen-7 (rich master), Stage-2 at $2K risk:

| profile (unified spec) | total | 2025 | 2026 | green | worst mo | MDD | top-5 |
|---|---|---|---|---|---|---|---|
| as-is (no partial) | 107,351 | 118,292 | −10,941 | 14/20 | −11,186 | −27,487 | 80% |
| 50% @ +1.5R, BE, trail | 56,101 | 55,433 | +668 | 12/20 | −7,672 | −25,330 | 127% |
| **50% @ +2R, BE, trail** | **95,363** | 90,731 | **+4,631** | 14/20 | −11,305 | −26,234 | 84% |
| 67% @ +2R, BE, trail | 88,185 | 84,010 | +4,175 | 14/20 | −11,463 | −25,923 | 84% |
| 50% @ +2.5R, BE, trail | 94,679 | 96,899 | −2,220 | 14/20 | −11,305 | −25,901 | 89% |
| 50% @ +2R, touch-fill at level, runner +3R limit | 87,204 | 91,526 | −4,322 | 14/20 | −12,416 | −25,513 | 82% |

The legacy-branch sweep's $132K / 2026 +$19K (section 2 of the partial sweep) does NOT survive the unified spec: its extra came from optimistic touch-fills on a fill-R level plus a fixed 2.5R runner target; every honest variant of those is worse than "partial at the bar close, remainder trails".

**Honest verdict on exits**: the +2R partial flips 2026 positive (+$15.6K swing) for −$12K of total (2025 gives up $27K of tail), same green-month count, slightly better MDD. It fails the pre-committed bar on worst month (−$11.3K vs −$8K) and MDD (−$26K vs −$20K). It is a real, shippable improvement in consistency — not the whole answer. The month count is the trade count.
