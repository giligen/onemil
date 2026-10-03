# Spec — the nightly ORB BT must produce the P1 add-on pool's book too (2026-10-03, owner: "shouldn't it be part of the BT?")

Principle (CLAUDE.md): ONE spec for backtest and live — every traded rule has a BT book beside it. P1 (`orb.yaml`
`universe.addon_pools[0]`: pool_id P1, `addon_gap35_range5`, gap 3–5 %, price ≥ $3, its own selection chain after
production) has placed paper orders since 10/1 with no nightly book → no parity read, no counterfactual.

## Required
1. The nightly producer (ExecStart of `onemil-orb-backtest.service`; today `orb_backtest.py` universe = production only, :55)
   also builds the P1 book for the same day: candidate universe = the P1 pool's gap/price/volume bounds exactly as the engine
   reads them from `orb.yaml` (import the pool definition through the same loader the engine uses — never a second copy of
   the thresholds), the P1 selection chain as the engine runs it (production first, then P1 on the remaining names; the same
   ranking, skip_q1, caps, PDR/G1/range/dedup vetoes, NO refill), with the pool's slot count. Output
   `analysis_results/orb_bplus_book_P1.csv` (same columns as the production book + `pool_id`), and a per-day marker row
   (`picks=0`) when a day is computed with zero picks — for BOTH books, so "0 picks" is never confused with "not computed".
2. The features CSV keeps a `pool` column (production / P1) or a second P1 features CSV — whichever the pipeline's static-lock
   scorer reads more naturally; the research static-lock pipeline's results must not change (production rows identical).
3. `scripts/eod_sections.py`: the ORB section reads the P1 book and prints `P1 ranked: engine n vs BT n | match …`,
   `P1 picks/fills …`, `P1 P&L …`; P1 does NOT feed the production promotion counter (its own counter `P1 clean sessions n`
   is printed for the 100-fill forward read).
4. Acceptance: a one-day run for 2026-10-01 must rank the engine's P1 set that day (APPS, CRMG, KD, TDAY were P1-scored,
   gaps 4.45–4.94 %) — print the ranked list with comp/quintile beside the engine's SCORED `pool=P1` lines from
   `logs/session_archive/`; any name differing must be explained as a data difference with its size. Also 2026-10-02.
5. Runtime: report the extra minutes on the nightly run (today 35–49 min, timer 16:30 ET / 20:30 UTC). If > 20 min extra, say so.

## Rules
Read-only on config/services/crontab/.env/data caches; the one-day runs go through `bash scripts/research_run.sh -m 3000M`
(service down — weekend). Never run the full multi-month BT. Research results and CSVs under research/ untouched. Tests:
the pool loader shared with the engine (parity test on the thresholds), the marker row, the eod P1 lines; `python3 -m pytest
-q tests/test_orb*.py tests/test_eod_sections.py -x --ignore=tests/integration` green. No git. Write
`research/orb_freq/SPEC_nightly_P1_book_RESULT.md` ≤ 50 lines; return ≤ 150 words.
