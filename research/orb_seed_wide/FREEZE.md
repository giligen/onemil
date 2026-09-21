# FREEZE — orb_seed_wide cells 1,300-1,314

- Repo HEAD at freeze: 4d7098c27ecc8371fb7696cd347ef7b60725aa80
- Pipeline file `study_orb_pipeline_static_lock.py` last touched at: 795581cc513fbcfdff61638bf802242253c4c78f
- Features CSV: `research/orb_seed_wide/out/orb_features_20260920_2142.csv`
  - md5: 83fd2e7c76ad9e84ea18a6be8234e87f
  - rows: 17945 (+1 header)
  - date range per PREREG: 2025-01-02..2026-05-29
- Honest reference book: `analysis_results/orb_bplus_book.csv` (219 rows incl header), read via `_sized_pnl`.
- orb.yaml (gitignored, instance config) sizing block at freeze time:
  account_budget_usd=26666.67, max_concurrent=8, risk_per_trade_usd=375,
  skip_q1=true, prev_day_range_veto.min_prev_day_range_pct=11.0,
  g1_veto.{return_volatility_20d_min=7.106, prev_day_range_pct_min=9.226, enabled=true},
  range_size_veto.{enabled=true, min_range_size_pct=2.221},
  catalyst_veto.enabled=false in yaml BUT the pipeline gates catalyst veto purely off
  `ORB_CATALYST_VETO` env (default '1'=ON, yaml flag is NOT read) — run with
  ORB_CATALYST_VETO=0 explicitly to match the honest book / task spec ("catalyst veto OFF").
