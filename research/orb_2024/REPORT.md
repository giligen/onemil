# REPORT — cells 1,415-1,416, 2024H2 ORB holdout (PREREG.md)

## Cell 1415

- Cell 1415 (2024H2): n=59 fills/wk=2.57 net_R/fill=-0.007 t=-0.09 total=$-147 ex-top5%_R=-0.062 no-fill=39.2% best_mo=$353 worst_mo=$-630 -> neither

## Cell 1416

- Cell 1416 (2024H2): n=4 fills/wk=1.33 net_R/fill=+0.455 t=+1.63 total=$683 ex-top5%_R=+0.285 no-fill=63.6% best_mo=$683 worst_mo=$683 -> SURVIVES

## 2025 comparison (runB_true, production seed)

- runB_true 2025 (beside cell 1,415): n=127 fills/wk=2.44 net_R/fill=+0.272 t=+3.19 total=$12,959 ex-top5%_R=+0.118 no-fill=21.1% best_mo=$4,040 worst_mo=$-187 -> SURVIVES


## Main-session review (2026-09-23)
* Robustness: the agent found that `cache.db::daily_bars` holds only SPY for 2024-06, so July's 20-day lookbacks were
  degraded. August–December alone (complete features): n 55, −0.012 R/fill (t −0.15), −$247, ex-top-5 % −0.072.
  Same answer.
* Monthly $ (1,415): Jul +100, Aug −206, Sep −630, Oct +353, Nov +322, Dec −87. No month resembles 2025.
* Adequacy: SE ≈ 0.08 R/fill. The 2025 level (+0.272) is ~3.5 SE above the 2024H2 estimate — rejected. A modest
  edge (+0.10 R) is ~1.4 SE away — NOT excluded. Pre-registered verdict: neither SURVIVES nor RED FLAG.
* Cell 1,416 (addon_p30): n 4 (the PDR veto removes most $30–50 names) — no information.
* **Operating consequence:** the ORB edge measured in 2025–26 is at least partly that regime. Do not scale ORB risk on
  backtest numbers; the live ramp's above-water rule (advance only on realized stage profit) is now the binding
  evidence. A decisive test needs more out-of-regime history (EQUS daily 2023–2024H1, ~$10).
Programme count 1,416.

## RERUN 2026-09-25 under the live exit rule (bars-source defect fixed)
`research/orb_verify/SPEC_RESIM_FIX.md`: same defect as `orb_2023/REPORT.md` — `study_orb_pipeline_static_lock.py`
hardcoded `data/cache.db` (no bars before 2025-01-02), so the 2024H2 books above were silently priced by the legacy
2R-target/range-low-stop/time-stop simulator, not the live static-lock+ATR-floor+scale-out rule. Fixed (env-var bars
source, missing-bars rows excluded + logged, `RESIM:` counts, >2% miss-rate gate; 12/12 unit tests). Re-run with
`ORB_BT_BARS_DB=research/day_breadth/y2024/bars.db`, `ORB_BT_DAILY_SOURCE=data/research/databento/equs_daily_2024H2.parquet`
(same source `build_features_2024.py` uses for ATR14/RV20; confirmed 2024-07-01..2024-12-31 coverage,
bar_date/symbol/open/high/low/close/volume schema), `ORB_CATALYST_VETO=0` → `book_141{5,6}_liveexit.csv` (frozen
scorer `score.py`, unchanged verdict logic, pointed at the new books via a `--book-suffix` argument).

**Parity**: cell 1415 n_resimmed 1388/1388 entered rows (100%), atr14_hits 2237/2433 (**91.9% — below the 95% bar**);
cell 1416 n_resimmed 368/368 (100%), atr14_hits 584/626 (**93.3% — below the 95% bar**). Both clear n_resimmed but
miss the atr14_hits target; the shortfall is `equs_daily_2024H2.parquet`'s own coverage gaps (delisted/thin names
short of 14 trading days of history at entry), not a code defect — the pipeline fails OPEN on a missing ATR14 (stop
stays at range_low, logged WARNING, row still resimmed and kept), so no entered row was dropped for this reason.
`exit_reason` sets: `{tag_bb, lock, scale_eod, tag_b1, stop, eod}` (1415), `{tag_bb, scale_eod, eod}` (1416) — subsets
of the live-rule set, confirming the fix took effect. Fill counts unchanged (59/4).

| Cell | old (legacy exit) R/fill, t | new (live exit) R/fill, t | frozen verdict (unchanged) |
|---|---|---|---|
| 1415 | -0.007, t -0.09 | **+0.031, t +0.38** | neither |
| 1416 | +0.455, t +1.63 | **+0.741, t +2.43** | SURVIVES |

Both cells move toward/through zero-or-better under the live exit rule; 1415 flips sign (small, still not
significant) and 1416's SURVIVES verdict strengthens. Neither frozen verdict category changes. The pooled
2023-01..2024-12 out-of-regime read using this book (see `orb_2023/REPORT.md`'s RERUN section) is +0.020 ± 0.043 R/fill,
t +0.48 — down from the legacy-priced +0.055. Full new report: `research/orb_2024/REPORT_liveexit.md`.
