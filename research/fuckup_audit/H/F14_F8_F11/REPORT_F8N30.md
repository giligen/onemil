# Stage H — book **F8 N=30, opening-range break** (`fam F8`, `cfg {"N": 30}`)

Method `H/METHOD.md`; freeze `FREEZE_F8N30.md` (written before `--val` ran). **TEST was not read.**

## One page
| question | answer |
|---|---|
| profit on TRAIN? | **Yes, and the gain is shape.** +0.0079 -> **+0.0581 R/trade, t 1.62** (no-refill), 8.1 tr/wk, halves +0.068/+0.049, WR 48.4 -> 53.0%, **MDD −40.5 -> −9.0 R**, ex-top-5% −0.105 -> −0.047. It does NOT clear G1 (t 1.62 < 2.0). |
| profit on VAL? | **No.** −0.0107 -> **−0.0340** (t −0.60, 45% weeks green). Two of the three vetoed buckets flip sign: M1 −0.024 -> **+0.033**, G1 −0.041 -> **+0.043**; only W1's holds (−0.061 -> −0.023). |
| filters + mechanisms | W1 `asset_class != 'wrapper'` (a 2x/inverse single-stock ETF carries no company information; its break is a geared echo that decays) · M1 `sig_close_pos >= 0.5` (the break bar must CLOSE in the top half of its own range — ORB's shipped touchgo **Rule M**, same 0.5 threshold) · G1 `gap_pct < 3` (gappers fade; PLAN §2 settled that 0 of 27 gap x $-volume cells deliver a positive open-to-close). |
| verdict | **FAIL at METHOD step 4.** |
| smallest visible effect (VAL) | **0.158 R/trade** = 1.0 R/week at 4 slots. |

## 0. Parity anchor
TRAIN n 1124 / +0.0079 / t 0.34 · VAL n 456 / −0.0107 / t −0.32 — exact against `C/score5_results.csv`
(`dn = 0`, `|dR| = 0.0`; all-day and `>=10:00` are the same book, F8 N=30 cannot signal before 10:00).

## 1. Step 1 — anatomy (`h1_anatomy_F8N30.md`, `h1_buckets_F8N30.csv`)
### 1.0 Availability audit
`news_pre` **1.0000** on all three splits (D's key set was built on F6/F8 signals — this IS that key set), so the
news leg is admissible here and was tested: news-present buckets +0.042 booked vs −0.005 news-absent, **not**
era-consistent (H1 +0.031 / H2 +0.054 present, +0.025 / −0.032 absent) and not a veto candidate.
`pm_dollar_vol` **0.3538** with missing +0.137 / +0.111 / +0.049 vs present −0.182 / −0.154 / −0.181 —
**REJECTED**, D1's leak signature, identical on all three splits. `spy_at_entry`, `spread_cc_bps` 1.000;
`prev_day_range_pct` 0.991; `adv20` 0.969.

### 1.1 Concentration
250 booked days, 126 green / 124 red (50%), total **+8.9 R**. Losing days −178.7 · winning +187.7 ·
worst 5% (12 days) **−44.9** (25% of day-losses) · worst 10% −77.8 (44%) · best 5% +62.8 ·
**both tails removed −9.0 R**. Weeks 57% green, worst −9.7, best +12.7. The 20 worst days are the most
market-directional set in the sub-stage (SPY c-o −0.52%, IWM −0.72% vs +0.04/+0.04 overall), and the booked
trades split −0.108 / −0.089 / +0.015 / **+0.214** by SPY close-open — but again that is the outcome, not a
causal feature: `spy_at_entry` buckets run −0.006 / −0.032 / +0.0 / +0.03.

### 1.2 Winners vs losers
`rv_adv` losers 1.044 vs winners 0.496 is the largest gap and again a Simpson artefact (all four rv buckets sit
between −0.019 and +0.044). The genuine separators are the ones that survive as buckets: wrapper (−0.061 booked,
both halves), `sig_close_pos < 0.5` (−0.024 booked, H1 −0.003 / H2 −0.042), `gap_pct >= 3` (−0.041 booked,
population −0.027/−0.034 both halves).

### 1.3 Path anatomy
eod 954 / stop 170 (**15.1%** — the lowest stop rate of the three books); 144 min to the stop (median 120).
**Wick stops 79 = 46.5% of stops** (the touch stop fires on a bar that closes back above the stop level).
Of the stops only 13.5% had +0.5R first, 5.9% +1R — a breakeven rule has much less to work with here than on F14.
MAE winners 1.90% / losers 5.04%. Holding time 0-15 −1.227 · 15-60 −1.065 · 60-150 −0.864 · 150+ **+0.114**.

### 1.4 Era consistency
137 cells; **26 negative in both halves**. After the availability rejection (4 `pm_dollar_vol` cells) and the
non-causal ones (`spy_co`, `iwm_co`), the cells that pass BOTH legs (population negative in both halves AND the
booked vetoed bucket negative in both halves) are: `spy_gap in [0,0.3)`, `dow == 4`, `spy_vs_sma20 in [0,2)`,
`iwm_at_entry in [-0.5,0)`, `sig_seq_day in [1,2)` — **all five are calendar or day-context buckets for which no
mechanism sentence exists**. The three filters with real mechanisms (wrapper, Rule M, gap) each pass leg 2 (the
booked vetoed bucket is negative in both halves) and fail leg 1 at population level (H1 +0.066 / +0.037 / −0.027).
That split is declared in the freeze and is why the mechanism stack is PRIMARY and the era-passing set is the
COMPANION.

## 2-3. Filters and the stack on TRAIN (11 single cells + 3 prefixes; `h_eval_F8N30.md`)

**The booking convention is a lever, not a detail.** Refilling a vetoed pick's slot flips M1 from **+0.026**
(no-refill) to **−0.031** (refill) — the freed slot goes to a worse candidate. Declared in the freeze BEFORE VAL:
no-refill is PRIMARY (ORB's shipped convention; refill was measured toxic there too).

| cell (no-refill) | n | tr/wk | net R | t | WR% | green | MDD | ex5 | cap3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1124 | 21.2 | +0.0079 | 0.34 | 48.4 | 0.57 | −40.5 | −0.1045 | +0.0006 |
| + W1 | 878 | 16.6 | +0.0272 | 1.10 | 50.1 | 0.58 | −37.5 | −0.0744 | +0.0230 |
| + M1 | 562 | 10.6 | +0.0323 | 1.02 | 50.7 | 0.53 | −18.4 | −0.0752 | +0.0287 |
| **+ G1 = STACK** | 428 | 8.1 | **+0.0581** | **1.62** | 53.0 | 0.55 | **−9.0** | −0.0472 | +0.0549 |
| STACK H1 | 210 | 4.0 | +0.0681 | 1.27 | 52.4 | | | −0.0487 | +0.0622 |
| STACK H2 | 218 | 4.1 | +0.0485 | 1.01 | 53.7 | | | −0.0348 | +0.0479 |
| STACK refill | 1100 | 20.8 | −0.0052 | −0.23 | 46.7 | 0.47 | −35.3 | −0.1075 | −0.0123 |

Singles also looked at: `close_pos>=0.25`, `gap<10`, `dist_open>2.5`, `not Friday`, `spy_gap not [0,0.3)`,
`spy_vs_sma20 not [0,2)`, `iwm_entry not [-0.5,0)`, `adv20 known`.
COMPANION stack (declared non-primary in the freeze): `S1+X1+D1` -> TRAIN +0.0598 (t 1.26), H1 +0.083 / **H2
+0.003**, 30% weeks green — rejected on the H2 collapse and the missing mechanism.

## 4. VAL — read once
| cell | n | tr/wk | net R | t | green | ex5 | MDE |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 456 | 20.7 | −0.0107 | −0.32 | 0.45 | −0.1043 | 0.094 |
| **STACK (no-refill, PRIMARY)** | 144 | 6.5 | **−0.0340** | −0.60 | 0.45 | −0.1306 | 0.158 |
| STACK (refill, sensitivity) | 428 | 19.5 | +0.0672 | 2.00 | 0.68 | −0.0262 | 0.094 |
| COMPANION S1+X1+D1 | 183 | 8.3 | +0.0513 | 0.87 | 0.45 | −0.0597 | 0.164 |

**FAIL**: the primary stack does not improve VAL and is negative. The refill twin — negative on TRAIN (−0.0052),
positive on VAL (+0.0672, t 2.00) — is the cleanest demonstration in the sub-stage that this is noise: the two
booking conventions rank in opposite orders on the two splits. The companion stack is positive on VAL (+0.051)
but fails its own >=55%-green leg at 45%.

Per-filter vetoed bucket, TRAIN booked -> VAL booked (`vetoed_persistence.csv`). **7 of this book's 10
era-passing buckets keep their sign on VAL** — the best persistence of the three books — and the three that flip
are exactly the two mechanism filters in the stack plus M1's looser twin:
W1 −0.061 -> −0.023 (holds) · G1b −0.081 -> −0.013 (holds) · C1 −0.100 -> −0.010 (holds) ·
D1 −0.073 -> −0.092 (holds) · S1 −0.060 -> −0.100 (holds) · X1 −0.031 -> −0.055 (holds) ·
I1 −0.017 -> −0.013 (holds) · **M1 −0.024 -> +0.033 (flips)** · **G1 −0.041 -> +0.043 (flips)** ·
M1b −0.067 -> +0.011 (flips).
That is the awkward shape of this book's result: most of its bad buckets stay bad, but the ones with a stateable
mechanism are the ones that reverse, and the ones that persist are calendar and day-context cells that were
declared non-primary before the VAL read precisely because no mechanism exists for them.

## 5. TEST — **not read** (VAL failed)

## Declared sensitivities
`2R stop-1%`: baseline TRAIN −0.0090 / VAL +0.0287; stack TRAIN +0.0382 (t 1.24, halves +0.037/+0.039) /
VAL +0.0065 — no VAL improvement. No-floor twin: combined REPORT §5. Refill: printed above.

## Phrasing (PLAN §1)
In THIS universe, at THIS horizon, at THIS book size, over THIS window and at THIS cost, the F8 N=30 losers are
**not separable** by the 11 causal cuts tested: the three mechanism filters cut the drawdown 4.5x on TRAIN and two
of their three vetoed buckets reverse on VAL. Smallest per-trade effect the VAL test could have seen:
**0.158 R** (~1.0 R/week at 4 slots).
