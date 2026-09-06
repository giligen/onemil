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

## 6. The entry side (2026-09-06 evening): the raw detector has no edge in either year — the selection stack does, and it decayed

Raw regen-7 cache (all 896 detections at the standard exits): **2025 mean −0.01R (463 trades, WR 42%), 2026 mean −0.07R (433, WR 41%)**. The Stage-2 stack picked +0.61R trades in 2025 (46, WR 59%) and +0.06R in 2026 (33, WR 45%). The raw material is the same in both years (9% of detections reach +2R in both); the fitted layers (conviction ≥1.8, TTF composite, MACD tiers, regime) stopped separating in 2026. Re-entry (lab L1, June 2026 Stage-1 with `BT_ALLOW_REENTRY=1`) adds 7 trades on 66 symbol-days (+10%) at −0.02R — not a frequency lever.

Era-consistency scan of every causal raw feature by year (896 rows, both years must agree on sign). Three buckets are the worst in BOTH years; none of the fitted features are:

| raw feature bucket | 2025 meanR / WR (n) | 2026 meanR / WR (n) | share of detections |
|---|---|---|---|
| **breakout at/below VWAP** (`qf_vwap_dist_pct ≤ 0`) | −0.29 / 35% (92) | −0.41 / 31% (94) | 21% |
| pole gain 3–5% | −0.19 / 37% (151) | −0.22 / 37% (184) | 37% |
| entry price > $20 | −0.31 / 35% (55) | −0.18 / 36% (69) | 14% |

(`daily_range_pct` separates strongly but is the full-day range — look-ahead, rejected as in backtest.py Rule 6. Conviction buckets flip sign between years.)

### 6a. The above-VWAP gate — built as ONE spec (`trading/bf_vwap_gate.py`, BT Stage-2 + live, `trading.bull_flag.vwap_gate`, default OFF)
Mechanism: a flag below VWAP is a bounce into supply, not continuation; every BF rule book says long only above VWAP, and we only ever had an upper bound. Full Stage-2 runs with the gate ON ($2K risk, $50K capital):

| profile | n | total | 2025 | 2026 | WR | green | worst mo | MDD |
|---|---|---|---|---|---|---|---|---|
| as-is | 79 | 107,351 | 118,292 | −10,941 | 53% | 14/20 | −11,186 | −27,487 |
| gate | 65 | 196,074 | 180,221 | +15,853 | 60% | 15/20 | −11,186 | −15,928 |
| gate + 50%@+2R partial | 64 | 173,398 | 142,967 | **+30,431** | 66% | **16/20** | −10,339 | −15,798 |
| gate + partial + risk cap $4K (post-hoc — SUPERSEDED by §6f, real knob) | 64 | 153,436 | 118,092 | **+35,344** | 66% | 15/20 | **−8,845** | **−13,297** |
| gate + partial + risk cap $3K (post-hoc — superseded by §6f) | 64 | 130,656 | 98,673 | +31,983 | 66% | 15/20 | −6,146 | −11,966 |

What the gate does: removes 16 of 79 Stage-2 trades — 11 losers / 5 winners, net **−$40.7K** (2025 −$13.9K over 8 trades, 2026 −$26.8K over 8). One knock-on: on 2025-03-20 the below-VWAP loser LSE (−$9K) had taken a slot; without it OM (+$24.5K) and TITN (+$23.5K) enter — that is $48K of the 2025 lift from ONE day and must not be counted as evidence. The evidence is the 16 losers and the raw-cache era consistency.

Risk cap: per-trade risk runs $3.5K median, $8K max (conviction × MACD × regime × tier stack). The multipliers are the fitted layers; in 2026 they amplify losses. Capping at 2× base ($4K) costs 2025 $25K and buys worst month −$8.8K / MDD −$13.3K / 2026 +$35K. Post-hoc scaling is exact up to integer shares; a real knob (`max_total_risk_mult`) is the next build.

### 6b. What 30–50% on capital in 2026 looks like on these numbers
Capital normalization is $50K / $2K base risk. 2026 YTD (8 months): as-is −22%; +2R partial alone +9%; gate + partial + $4K cap **+71%** (+$35.3K), 2026 H1 +$46K, Jul–Aug −$11K (4 trades, 3 stopped — the trade count problem is untouched). None of this is a forecast: the gate was found today on the same cache it is measured on, though its rule is pre-registered by every BF practitioner and era-consistent on 896 raw rows, not fitted on the 79.

### 6c. Frequency: the raw cohort that is positive in all four half-years
Raw detections with pole > 5% AND above VWAP AND price ≤ $20: **405 of 896 (45%), mean +0.16R (2025, n=232) / +0.23R (2026, n=173), all four halves positive (+0.07 / +0.25 / +0.25 / +0.20)**. At a flat $2K risk with NO fitted layer: 15/21 green months, worst −$5.7K, 2026 +$80K — ~19 trades/month, one a day. This is the candidate "every day" book: the entry rule is three causal raw filters, no conviction threshold, no composite. It needs its own Stage-2 run (slots, daily-loss, BP, +2R partial) before it is more than a cohort statistic — queued behind the June lab (one bulk-bar process at a time).

### 6d. The frequency book, run properly through Stage-2 (2026-09-06 18:00) — more trades, more variance, no more 2026 edge
Stage-2 now re-applies `scanner.price_max` and `bull_flag.min_pole_gain_pct` from CONFIG to the broad cache (same knobs live uses; `tests/test_batch_backtest.py::TestStage2LiveKnobs`). Configs: **TRI** = current fitted stack + VWAP gate + pole ≥ 5% + price ≤ $20; **F0** = the three raw rules with the fitted layers OFF (conviction threshold 0, TTF off, risk tiers off); reg1/reg0 = regime sizing on/off. $2K base risk, $50K capital; the June lab's L2 (legacy 1R-partial quick exits) is worse than as-is on June 2026 (−0.22R, 4/19 green days) and is dead.

| profile | trades (per mo) | total | 2025 | 2026 YTD | WR | green | worst mo | MDD |
|---|---|---|---|---|---|---|---|---|
| as-is | 79 (4.0) | 107,351 | 118,292 | −10,941 | 53% | 14/20 | −11,186 | −27,487 |
| gate + pp2R | 64 (3.2) | 173,398 | 142,967 | +30,431 | 65% | 16/20 | −10,339 | −15,798 |
| TRI + pp2R, regime on | 52 (2.6) | 168,417 | 116,681 | +51,736 | 69% | 15/20 | −11,471 (Feb-25) | −15,798 |
| **TRI + pp2R, regime off** | 55 (2.8) | 131,377 | 93,382 | **+37,995** | 67% | 14/20 | **−7,600** | **−15,111** |
| TRI + pp2R, regime on, $4K risk cap (post-hoc — superseded by §6f) | 52 | 149,927 | 98,671 | +51,256 | 69% | 15/20 | −10,087 | −13,296 |
| F0 (raw rules only) + pp2R, regime on | 138 (6.9) | 202,680 | 157,436 | +45,243 | 53% | 12/20 | −15,870 | −34,910 |
| F0 + pp2R, regime on, $3K cap (post-hoc — superseded by §6f) | 138 | 154,832 | 105,641 | +49,191 | 53% | 14/20 | −12,388 | −22,468 |
| F0 as-is exits, regime on | 140 (7.0) | 229,761 | 208,506 | +21,254 | 52% | 15/20 | −16,080 | −36,282 |

Read: the 86 trades F0 adds over TRI are worth +$34K over 21 months (+0.1R each) and **−$6.5K in 2026** — they add month variance (worst −$15.9K, MDD −$35K) and no 2026 edge. The fitted layers still separate *within* the raw-rule cohort; the "every day" book at positive 2026 R does not exist in this detector. Frequency is not the consistency lever; per-trade quality and bounded risk are.

**Candidate consistency profile (proposal, joint decision)**: TRI + 50%@+2R partial + regime sizing OFF — the only profile that clears the pre-committed bar (worst month ≥ −$8K, MDD ≥ −$20K) while 2026 YTD is +$38K on $50K (76% in 8 months; 2026 has 22 trades, 13 winners). Cost: 2025 $118K → $93K. Alternative if we keep regime sizing: add a $4K per-trade risk cap (worst −$10.1K, MDD −$13.3K, 2026 +$51K) — needs the cap built as a real knob first. The Jul–Aug 2026 texture remains: 3 trades, −$5.3K.

Walk-forward honesty: the three raw rules were chosen from the era-consistency scan on the full cache; §6e shows the same three are the three worst causal buckets on 2025 ALONE, so a 2025-only rule-picker would have shipped the same rules and 2026 is out-of-sample for them. The +2R partial and regime-off choices were made on the full window (in-sample). Live shadow (10 sessions, `profit_partial.shadow`, VWAP gate log-only is next) before any flag flips.

### 6e. Would a 2025-only rule-picker have found the same rules? Mostly.
Worst causal raw buckets ranked on **2025 alone** (n ≥ 40), with their 2026 out-of-sample mean R:

| rank | bucket | 2025 meanR (n) | 2026 meanR (n) | holds OOS? |
|---|---|---|---|---|
| 1 | entry price > $20 | −0.31 (55) | −0.18 (69) | yes |
| 2 | breakout at/below VWAP | −0.29 (92) | −0.41 (94) | yes |
| 3 | gap +15–30% | −0.27 (40) | **+0.21** (44) | **no — flips** |
| 4 | pole gain 3–5% | −0.19 (151) | −0.22 (184) | yes |
| 5 | entry 10:00–10:15 | −0.15 (77) | −0.01 (94) | weak |
| 6 | conviction ≤ 1.2 | −0.12 (174) | −0.15 (152) | yes (already gated) |

Three of the top four 2025 buckets hold in 2026; the gap bucket flips sign and is NOT a rule (small n, no mechanism). So the three rules are the 2025-selected rules that survived 2026 — not perfectly pre-registered (the picker also had the gap rule available and would have carried one dud), which is the honest strength of the evidence: 2025-in-sample, 2026-out-of-sample for the rule set minus one flip.

### 6f. The risk cap as a real knob (`trading/bf_risk_cap.py`, `trading.risk_cap`, default OFF) — post-hoc numbers were too kind
Built as ONE clamp: final shares ≤ (max_risk_mult × risk_per_trade) / risk_per_share, after every multiplier and before the BP ceiling, on both sides (`tests/test_bf_risk_cap.py`). Running it through Stage-2 disagrees with the post-hoc scaling above: the Stage-2 CSV's `shares` column is PRE-regime (Stage-2 multiplies pnl only), so the post-hoc cap under-measured risk on C1 days (1.5×) and clamped too little. Real runs (2× base = $4K, $2K base, regime on, +2R partial):

| profile | n | total | 2025 | 2026 YTD | WR | green | worst mo | MDD |
|---|---|---|---|---|---|---|---|---|
| TRI + pp2R, regime on, no cap | 52 | 168,417 | 116,681 | +51,736 | 69% | 15/20 | −11,471 (Feb-25) | −15,798 |
| TRI + pp2R, regime on, **cap 2× real** | 54 | 120,785 | 75,916 | +44,869 | 66% | 14/20 | −11,835 (Feb-25) | −14,251 |
| TRI + pp2R, regime OFF, no cap | 55 | 131,377 | 93,382 | +37,995 | 67% | 14/20 | **−7,600** | −15,111 |
| gate + pp2R, regime on, no cap | 64 | 173,398 | 142,967 | +30,431 | 65% | 16/20 | −10,339 (Aug-26) | −15,798 |
| gate + pp2R, regime on, **cap 2× real** | 66 | 121,402 | 93,618 | +27,783 | 63% | 13/20 | **−6,684** | −13,120 |

Read: the cap does what it says on the gate-only book (worst month −$10.3K → −$6.7K, MDD −$13.1K) at a $52K cost over 21 months, and it does NOT fix TRI's Feb-2025 (those losers are under the cap already; the regime C1 1.5× is what amplifies that month — regime OFF is the cleaner lever there). The two extra trades in each capped run are the daily-loss-limit no longer tripping. Regime-off rows are exact (no post-hoc involved).

**Where this leaves the proposal**: two profiles clear the pre-committed bar (worst ≥ −$8K, MDD ≥ −$20K):
- **P1**: VWAP gate + pole ≥ 5% + price ≤ $20 + 50%@+2R partial + regime sizing OFF → $131K / 2026 +$38K / worst −$7.6K / MDD −$15.1K / 14/20 green / 2.8 trades per month.
- **P2**: VWAP gate + 50%@+2R partial + risk cap 2× (regime on) → $121K / 2026 +$27.8K / worst −$6.7K / MDD −$13.1K / 13/20 green / 3.3 trades per month.
P1 keeps more of 2026 and uses three raw rules with a 2025-selected / 2026-OOS pedigree (§6e); P2 keeps the regime layer and fewer rule changes. My recommendation is P1, with the cap kept available as the month-variance dial when base risk scales. Both flags exist in code today, default OFF; ship = joint decision after a 10-session shadow (`profit_partial.shadow: true`; gate/pole/price are log-only until enabled).
