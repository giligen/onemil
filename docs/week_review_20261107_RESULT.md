# Week review 2026-11-02..11-06 (written 11/7, read-only)

## 1. Daily P&L per book ($, account equity close-to-close; Fri 10/30 = base)
| Book | Mon | Tue | Wed | Thu (crash) | Fri | Week |
|---|---|---|---|---|---|---|
| MOM live (sleeve ~$9.4K -> $7.7K) | -673 | +239 | -374 | -1,123 | +211 | **-1,721** |
| MOM paper (sleeve ~$16.9K -> $13.8K) | -1,209 | +427 | -670 | -2,010 | +378 | **-3,084** |
| HOD paper | -7 | +25 | +15 | +78 | -134 | **-24** |
| ORB key, total | -768 | -272 | +780 | 0 | 0 | -260 |
| of which TOM (est., IEX closes) | -378 | +225 | +776 | 0 | 0 | +622 (realized +642) |
| **ORB ex-TOM** | -390 | -496 | +4 | 0 | 0 | **-882** |

DB cross-check (trades.db): HOD -21.91 (43 fills) vs account -23.84; ORB -884.85 (8 fills) vs -882. Every day within $50 (largest gap $4).

## 2. Market (IEX close-to-close, %)
| | Mon | Tue | Wed | Thu | Fri | Week |
|---|---|---|---|---|---|---|
| SPY | -0.49 | +0.31 | +1.64 | **-4.34** | +0.27 | -2.69 |
| QQQ | -1.35 | +0.60 | +0.93 | -6.31 | +1.27 | -4.96 |
| IWM | -0.06 | +0.23 | +1.47 | -4.55 | +0.95 | -2.07 |
| MTUM | -3.24 | +1.28 | -2.05 | -8.27 | +1.82 | -10.34 |
| VIXY | +0.19 | -1.90 | -9.45 | +30.32 | -1.54 | +14.20 |
| MOM long leg (20 names, EW) | -6.86 | +2.79 | -4.22 | -13.20 | +2.96 | -17.92 |
| MOM short leg | none (all 20 positions long) | | | | | |

Crash day = Thu 11/5. MOM live -$1,123 and paper -$2,010 (long leg -13.2 %, about 2x QQQ). HOD +$78 (12 fills). ORB flat (0 candidates survived G1/PDR vetoes). TOM already out.

## 3. Anatomy
**MOM.** Sleeve return -18.2 % Fri to Fri, identical live and paper (gap 0). The 11/2 weekly-CSV row is a rotation-time mark, so against it the week reads -15.8 % (paper 16,416 to 13,821; live 9,174 to 7,726). Since start: paper -30.9 % (peak DD -31 %), live -22.7 %. Live sleeve $7,726 is **$226 (2.9 %) above the $7,500 rail**. Paper is past -25 % of $20K and $1.8K above -40 %.
Worst since entry (paper / live): AXTI -625 / SNDK -218, SNDK -582 / LITE -193, LITE -554 / UMC -169. Only MRNA is up (+119 / +67; live RVMD +34).
Parity: Saturday run failed. Quote: "ERROR mom_bt_parity: extension last bar 2026-10-30 is 8 days old (> 6) -- the weekly refresh failed". Cause: "GATE FAILED ... liquid 663/825 with an 2026-11-06 bar (80.4%, min 98%)". Last residual: "BT parity wk 10/26: names 17/20 | BT -2.9 % | paper -1.8 % | live n/a | SPY -0.9 % | resid +1.0 pp".

**HOD.** 43 fills (5/7/6/12/13), 16 wins (37 %), 20 stops (47 %), mean -0.215 R on about $48 risk (reference -0.026 R; n 43 cannot separate them). Best PRCT +$103, HIBS +$106; worst IONZ -$67, ALMR -$66.

**ORB.** 8 fills (3 Mon, 5 Tue), 0 winners, 6 stops, 5 of 8 in SK-Hynix-linked tickers (SKUU, SKHL, SKHX, SK, SKHU). Realized -$882 tripped the $750 weekly kill 11/3 16:34 (flat, no entries 11/4). The kill rail was rescaled to -$1,875 by the 11/5 boot. 11/4, 11/5, 11/6: no entries (kill, then 0 candidates after vetoes).

**TOM.** Nov cycle entered 10/30, exited 11/4: +$641.8 (SPY +281, QQQ +31, IWM +330). Exit limits cancelled at 20:58 UTC, then market fallback.

## Does not reconcile / flags
1. BT parity not run for this week; weekly panel refresh gate failed (80.4 % coverage of 11/6 bars); the extension is stale.
2. ORB DB keeps a stale `pending_new` AAOX row (11/3) beside the filled row; LABX exit_reason `target_rested` on a -$22.94 loss.
3. Two ERRORs: RemoteDisconnected 11/3 16:15 (account info) and 11/5 15:30 (latest trades, scanner cycle at 10:30 ET failed once).
4. Paper -25 % threshold uses an assumed $20K start (state.json has no equity_start).
