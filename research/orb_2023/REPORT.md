# REPORT — live ORB on 2023-01 .. 2024-06, point-in-time universe (cells 1,418–1,419)

## Cell 1418 (production seed gap ≥ 5 %, $3–30)

- fills 106 (of 171 picks, no-fill share 38%) over 89 days, 1.4 fills/week (78 weeks)
- net R/fill +0.089, t (day-clustered) +1.55, ex-top-5 % +0.027, total $+3,542
- halves: 2023 +0.124 R/fill, 2024H1 +0.039 R/fill
- months: worst -703 (2024-03), best +871 (2023-09), green months 78% of 18
- **VERDICT (frozen): FLAT**

- **Pooled out-of-regime ORB (2023-01 .. 2024-12, same code): n 165, +0.055 R/fill ± 0.046 (t +1.19)** — vs 2025 +0.272 (n 127)

## Cell 1419 (addon_p30 gap 3–5 %, $30–50)

- fills 10 (of 10 picks, no-fill share 0%) over 10 days, 0.1 fills/week (78 weeks)
- net R/fill -0.210, t (day-clustered) -1.42, ex-top-5 % -0.354, total $-786
- halves: 2023 -0.252 R/fill, 2024H1 -0.205 R/fill
- months: worst -470 (2024-04), best +279 (2024-02), green months 17% of 6
- **VERDICT (frozen): NEGATIVE**


## Main-session review and the pre-committed consequences (2026-09-24)
* Data: XNAS.ITCH point-in-time tickers ($5.51) → consolidated Alpaca daily (12,891 of 12,976 tickers, 0 lost after the
  fetch-loop fix) → 15,612 candidates, 99.4 % with minute bars; same code as the 2024H2 run.
* **1,418 production seed: FLAT** (+0.089 R/fill, t 1.55, n 106; 2023 +0.124, 2024H1 +0.039; 78 % green months).
  Pooled out-of-regime 2023-01..2024-12: **+0.055 ± 0.046 R/fill, n 165** — vs +0.272 in 2025.
  → **Consequence applied:** `scripts/orb_ramp_check.py` ADVANCE now needs 40 live fills at the stage
  (`min_fills_advance`); the BT-band DEMOTE keeps its 8-fill trigger. Tests 33/33.
* **1,419 addon_p30: NEGATIVE** (−0.210 R/fill, n 10). With 2024H2 (+0.455, n 4): out-of-regime ≈ −0.02 R on 14 fills.
  → the $30–50 add-on pool is **no longer order-eligible**; it stays dry-only (PREREG_LIVE_UNION amended).
Programme count 1,419.

## Correction 2026-09-24 ~22:10 UTC (found by `research/thermo`'s pre-registered consistency gate)
The "2025 +0.272 (n 127)" comparison above is `runB_true.csv`, a catalyst-veto-ON book covering 2025-01..2026-05;
every out-of-regime book here is veto OFF, the live config since 9/21. Like for like (veto OFF, same code, rebuild
validated against `runB_true` with the veto ON): **2025 +0.106 R/fill (n 211, t 2.45), 2026-01..09 +0.105 (n 262,
t 2.29)**. The frozen FLAT verdict and its consequence are unchanged (they are absolute thresholds), but the reading
"a coin flip outside 2025-26 vs +0.27 inside" was wrong: per fill ORB is ~+0.1 R wherever measured (2024H2 ≈ 0); the
regime changes how OFTEN it trades.

## RERUN 2026-09-25 under the live exit rule (bars-source defect fixed)
`research/orb_verify/SPEC_RESIM_FIX.md`: the runs above were silently priced by the LEGACY 2R-target / range-low-stop
/ time-stop simulator (old `book_141{8,9}.csv`, kept, exit_reason ∈ {stop, target, eod} only) because
`study_orb_pipeline_static_lock.py` hardcoded `data/cache.db`, which has no bars before 2025-01-02 — every entered row
silently fell back to the features CSV's pre-computed legacy `pnl`. Fixed: `ORB_BT_BARS_DB` / `ORB_BT_DAILY_SOURCE`
env vars, missing-bars rows now EXCLUDED + logged ERROR, `RESIM:` counts printed, exits non-zero above a 2% miss rate
(tests/test_orb_pipeline_bars_source.py 12/12). Re-run with `ORB_BT_BARS_DB=research/orb_2023/bars.db`,
`ORB_BT_DAILY_SOURCE=research/orb_2023/daily_alpaca.parquet`, `ORB_CATALYST_VETO=0` → `book_141{8,9}_liveexit.csv`
(frozen scorer `score_2023.py`, unchanged verdict logic, pointed at the new books via a `--book-suffix` argument).

**Parity**: cell 1418 n_resimmed 2644/2644 entered rows (100%), atr14_hits 4508/4567 (98.7%); cell 1419 n_resimmed
870/870 (100%), atr14_hits 1371/1378 (99.5%) — both clear the ≥98%/≥95% bar. `exit_reason` sets are now
`{tag_bb, lock, scale_lock, scale_eod, tag_b1, stop, eod}` (1418) and `{tag_bb, scale_eod, stop}` (1419) — a subset of
`research/thermo/book_2025_26.csv`'s live-rule set, not the legacy stop/target/eod-only set. Fill counts unchanged
(106/171 and 10/10) since the resim only re-prices the exit, not entry/selection.

| Cell | old (legacy exit) R/fill, t | new (live exit) R/fill, t | frozen verdict (unchanged) |
|---|---|---|---|
| 1418 | +0.089, t +1.55 | **+0.015, t +0.30** | FLAT |
| 1419 | -0.210, t -1.42 | **-0.014, t -0.07** | FLAT (was NEGATIVE) |
| Pooled 2023-01..2024-12 (1418 + 2024H2 book_1415) | +0.055 ± 0.046, t +1.19 | **+0.020 ± 0.043, t +0.48** | — |

Both cells move toward zero under the live exit rule (static lock / ATR floor / scale-out cuts winners short relative
to the legacy 2R target). **1419's verdict changes from NEGATIVE to FLAT** — the $30–50 add-on pool's live-rule
out-of-regime read is now "no information" rather than "actively bad," though it stays dry-only per the 9/24 decision
(n=10 is too small to move that call either way). 1418 stays FLAT; the live ramp's 40-live-fills-before-advance
consequence (`orb_ramp_check.py`) is unaffected — it did not depend on the legacy-priced number. Full new report:
`research/orb_2023/REPORT_liveexit.md`.
