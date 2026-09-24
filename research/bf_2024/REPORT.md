# REPORT — bull-flag P1 on 2024H2 (PREREG.md cell 1,417)

Stage-2 P1 trades: n=27
Mean R: 0.2116
Day-clustered t: 0.587
Total $: 32,705.41
Ex-top-5% mean R (dropped 1): 0.0802

## Monthly $
- 2024-07: 16,106.92
- 2024-08: -8,009.25
- 2024-09: 7,986.14
- 2024-10: 15,999.61
- 2024-11: -6,324.76
- 2024-12: 6,946.75

## P1 2025-26 book (beside it)
P1 2025-26: n=56 (no R/r_multiple column found; columns=['symbol', 'date', 'entry_time_et', 'entry_price', 'stop_loss', 'target', 'shares', 'exit_time_et', 'exit_price', 'exit_reason', 'pnl', 'pnl_pct', 'partial_taken', 'partial_price', 'partial_shares', 'partial_pnl', 'daily_range_pct', 'avg_volume_20d', 'qf_vwap_dist_pct', 'qf_gap_pct', 'qf_gap_fading', 'qf_spy_return_pct', 'qf_pole_bars', 'qf_pole_gain_pct', 'qf_fill_vwap_dist_pct', 'conviction_mult', 'macd_zone_mult', 'conv_pole_gain', 'conv_flag_tightness', 'conv_vol_ratio', 'conv_spy_regime', 'conv_retracement', 'conv_raw_score', 'spy_3d_range', 'conv_vwap_dist', 'conv_gap_fading', 'intraday_change_at_entry', 'bk_ratio_at_fill', 'spy_3d_at_fill', 'planned_entry'])

## PREREG verdict: SURVIVES
(mean_r=0.2116 > 0: True; total_$=32,705.41 > 0: True; ex_top5_mean_r=0.0802 > 0: True)

## Main-session review (2026-09-24) — corrected verdict: NOT SURVIVES (neither SURVIVES nor RED FLAG)
* **Universe leak:** QBTS.WS (a warrant) is the #2 trade (+3.00 R, $8,108). Live bull flag excludes warrants by name
  (`trading/bf_universe_filter.py`), but the 2024 symbol had no name on file and was "kept by symbol-list only" (38 such
  symbols, logged as a WARNING). With the live rule applied: n 26, mean **+0.104 R**, total $24,598, ex-top-5 %
  (drop LPA +3.63 R) **−0.038 R** → fails the frozen SURVIVES bar; not a RED FLAG (mean > −0.10).
* Read: bull flag P1 on 2024H2 is about breakeven and its whole profit is one trade. n 26 → SE ≈ 0.35 R: consistent
  with a modest edge and with none.
* The 2025–26 P1 reference book (56 trades) was checked for the same leak: no warrants, units or rights.
* All six months present (Aug/Sep re-run sequentially after the shared-SQLite failure). Programme count 1,417.
