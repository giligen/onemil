# exec_cost REPORT.md — MOC exit cell (ORB + BF)

## ORB (analysis_results/orb_bplus_book.csv)
Scope: pure `eod` exits only (n below); `scale_eod` excluded (no partial-shares column to split the leg) — count-only context: 26 scale_eod trades not computed.
- **2025**: n=11 time-exits | old net $22,506 -> new $23,399 | Delta $893 (+81.1/trade, t=0.58 n_days=9) | MDD old $-2,719 -> new $-3,581 | stops-in-window: N/A (no stop/shares data, see caveat)
- **2026H1(Jan-May)**: n=3 time-exits | old net $6,654 -> new $6,216 | Delta $-438 (-146.1/trade, t=-0.57 n_days=3) | MDD old $0 -> new $0 | stops-in-window: N/A (no stop/shares data, see caveat)
- daily-close vs 15:59-bar-close differs >0.5% on 0.0% of ORB time-exit trades (n=14)

## Bull flag (research/bf_stage2_regen7_raw_20260905.csv, regen-7 Stage-2)
- **n=0 time-exits found** (no row has exit_time_et in 15:5x; exit_reason set in this book = ['exhaust+trail_stop', 'post_fill_exit', 'stop', 'trail_stop']). Cell is VACUOUS for BF regen-7 — this honest book has no trades that reach the 15:55 flat; Pass bar trivially holds (Delta=$0, MDD unchanged) but there is nothing to ship.

## Caveat
ORB honest book (analysis_results/orb_bplus_book.csv) carries no entry_time/exit_time/stop_loss/shares columns — it is a features+pnl summary, not a per-minute ledger. shares/exit_price were backed out algebraically from entry_price and pnl_pct; the extra-window stop-walk required by the PREREG could NOT be run for ORB (declared as scope-limit in PREREG before running) — the ORB Delta above is a spread-swap-only estimate (assumes price drifts cleanly to the close with no intervening stop), not the full spec. BF got the full bar-walk.
