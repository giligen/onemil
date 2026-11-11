# Cell 1,732 — ORB paper stop share vs BT — RESULT 2026-11-11

Inputs: ledger `data/trades.db` (ro), 32 closed ORB fills from 9/29 to 11/9 (the PREREG said 31; BEX 11/9 is the 32nd).
Nightly production BT `analysis_results/orb_bplus_book.csv` (built 11/10 22:20, covers 2025-01-02 → 2026-11-10) and the
candidate sim `analysis_results/orb_features_20261110_2203.csv`. Alpaca SIP and IEX 1-min bars. Script: `resim_1732.py`.
Rows: `stops_1732.csv`.

## 1. One definition (H1: rejected, 0 pp)
"Stop" = full protective stop only (live `stop_loss`, BT `stop`). Lock stops are counted separately on both sides.

| book | n | stop | lock | touchgo (tag_bb/tag_b1, `target_rested`) | eod/force |
|---|---|---|---|---|---|
| Live paper 9/29–11/9 | 32 | **16 (50.0 %)** | 2 | 3 | 11 |
| BT, full history (entered) | 499 | 31.5 % | 9.4 % + 3.8 % scale_lock | 28.9 % | 26.4 % |
| BT `book_G3_meas` reference | 462 | 31.0 % | 9.5 % | 29.6 % | 25.8 % |
| **BT, same dates** | 31 | **12 (38.7 %)** | 2 | 6 | 11 |

## 2. Decomposition of the 19 pp excess (50.0 vs 31.0)
| cause | pp | evidence |
|---|---|---|
| **Selection: live trades names the BT never takes** | **8.1** | 11 of 32 fills are live-only and 7 of them stopped (64 %). The 10 BT-only fills stopped 4 of 10. Sub-reasons for the 7 stops: 3 were ranked out by the BT, but the BT's own candidate sim also stops them (CORD 10/16, SKUU, SKHX). 2 are absent from the BT candidate universe (ASTX, ASTY). 1 is from the add-on pool `addon_gap35_range5` (APLX), which has no BT book. 1 was ranked out and the candidate sim gives `eod` (SNXX). |
| **Window/regime** | **7.7** | The BT itself stops 38.7 % on these dates, against 31 % over its history. 5 of the 16 live stops are SK Hynix wrappers on 11/2–11/3 (SKUU/SKHL/SKHX/SK/SKHU), one correlated event. |
| Execution on the 21 matched fills | 3.1 | The 8 BT stops are all live stops too. One extra live stop: AIXI 10/16 (BT `scale_lock`, +$2.1K at BT size). The live fill of 3.05 is *below* the range high of 3.09 and the 3.10 stop-limit, and it stopped 64 s later. On SIP the breakout bar fires touchgo M (exit at the bar close, −$95), but no touchgo line was logged. |
| H3 IEX feed in the BT | 0 | `cache.db` bars are SIP. Volumes match SIP: SMST 1.51M vs SIP 1.49M vs IEX 3K. The ORB BT stops SMST and IREZ, which IEX never breaches. Note: on IEX, 7 of the 16 live stops do not breach, so an IEX-built BT would show 28 %. The current one does not. |
| H4 trigger / higher stop | ≈0 | Live and BT share one helper, `floored_stop` (`trading/orb_winner_stack.py:134`, live `orb_engine.py:6603`, BT `study_orb_pipeline_static_lock.py:489`). 15 of 16 stops show a SIP 1-min breach between fill and exit. SMST is the exception: it triggered at 13.86 with the SIP bar low at 13.90, which looks like an odd-lot print, and the BT stops it later anyway. |
| H5/H6 | ≈0 | The 3 live touchgo exits (OKLL, CORD 10/29, LABX 11/3) match the BT `tag_bb` exactly. |

## 3. Verdict
**The stop mechanics are at parity.** On the fills both books take, live stops 9 of 21 and the BT 8 of 21. The extra one is AIXI, an entry anomaly. The excess is (a) the live book choosing different names from the BT (8 pp) and (b) a bad window that the BT also suffers (8 pp). No change to the stop rule closes it.

**Minimal fix = selection parity, not stops.**
1. Make the live top-8 equal the BT's top-8. 6 of the 11 live-only fills were BT candidates that the BT ranked out. Live ranks a *provisional* range at 09:34:58 and reconciles at 09:35:10 (`orb_engine.py:4304/4495`). The parity test: `scripts/orb_selection_observer.py` asserting live-entered ⊆ BT-entered per day. The 7-week observer verdicts should list these 6.
2. Give the add-on pool a BT book, or run it dry (`orb.yaml universe.addon_pools`; CLAUDE.md says enable only on the owner's word).
3. AIXI: a fill below the range high on a buy stop-limit should be rejected or re-checked (preplace path). One case, so diagnose it separately.

**Re-sim under the fix** (trade only the BT-taken names): stops 9/21 = **42.9 %** (8/21 = 38 % if AIXI exits by touchgo). Realized **+$603** (+$604 with the AIXI touchgo) against the actual −$88. The 11 live-only fills lost −$690. My SIP walk from the live fill minute gives 28.6 % / +$144 on those 21, but it treats the fill minute as the breakout bar, so it over-fires touchgo (AVEX, BEZ); the BT's breakout-bar logic is the reference. **This is not a promise:** the stop share stays near 39 % in this window, because the BT itself stops 39 % here.
