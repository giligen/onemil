# Dive 2 — why did the engine's ranked top-8 differ from the BT's on 2026-10-01, and fix it in the shared helper (owner's ask)

Facts (docs/eod_monitor_result_20261003.md, DIVE_bt_zero_picks_RESULT.md): 10/1 engine scored 15 names at 09:44 ET, its
top-8 by comp/quintile vs the BT's ranked top-8 (5 names from 12 features rows): match 4 (WVE, CBRX, LPA, IBX), engine-only
APPS, CRMG, KD, TDAY, BT-only RKLX. The post-fix replay (2044a17) matched the gap-gate ADMISSION 35/35 on 10/1, so the
divergence is DOWNSTREAM of admission: the candidate filters between admission and ranking, or the ranking inputs.

## Work (in order, numbers for each)
1. Features builder chain: find the script the nightly BT uses to build `orb_features_YYYYMMDD_HHMM.csv` (grep the
   `onemil-orb-backtest.service` ExecStart and research/orb_freq/). List its per-symbol filters AFTER the gap gate (price
   range, prev volume, 09:35 RVOL floor, ATR availability, bars-source, wrapper/asset-class map, dedup) with thresholds.
2. Engine chain: `trading/orb_engine.py` from admission to SCORED (candidate features at 09:35, filters, `comp` score,
   quintile, skip_q1, dedup, PDR/G1/range vetoes). List the same filters with thresholds. Diff the two lists.
3. For APPS, CRMG, KD, TDAY (engine-only): which BT stage drops each (or: absent from the BT's input universe — then WHY:
   daily_bars vs the BT's universe file, asset-class map, listing date, bars source)? Print the BT's values at that stage and
   the engine's logged values (the 10/1 SCORED lines in logs/session_archive/ carry comp, quintile, gap, rvol, price).
4. RKLX (BT-only): why did the engine not score it — not admitted, filtered, or ranked below 8? Print both sides.
5. Scores: for the 4 matched names, engine `comp` vs BT `comp` (and quintile). Equal to 3 decimals? If not, which input differs.
6. Root cause(s) named as rule differences, each: BT file:line vs engine file:line, which side is the SPEC
   (`research/orb_machine_rules.md` decides; if the rulebook is silent, the BT book's rule is the spec because the P&L
   evidence was built on it).
7. FIX in the SHARED helper (`trading/orb_*.py`; create one if the rule lives in two places), both sides calling it, plus a
   parity test on the 10/1 features rows + engine inputs that fails before and passes after. Then
   `bash scripts/research_run.sh -m 2500M python3 scripts/orb_open_tick_replay.py …` (read --help) for 9/30, 10/1, 10/2 with
   `--parity` must still match admissions, and a ranked-set comparison for 10/1 must now give match 8/8 or explain every
   remaining difference as a DATA difference (snapshot vs SIP bar) with its size.
   If the fix would change the BT side, do NOT touch the BT producer or any research result — write the BT change as a spec
   paragraph and fix the engine side only.

## Rules
Read-only on data, config, services, crontab, .env. All python via `bash scripts/research_run.sh -m 2500M python3 …`.
Full `python3 -m pytest -q tests/test_orb*.py -x --ignore=tests/integration` must be green after the fix. Never start the
service, never submit orders, never git. Write `research/orb_freq/DIVE_ranked_set_RESULT.md` ≤ 70 lines (the diff table,
the five names, the root cause with file:line, the fix, the replay and ranked-set lines after). Return ≤ 150 words.
