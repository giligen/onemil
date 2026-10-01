"""Cell 1,689c -- quiet-window in-regime re-run of pools 24/25/26/30's cache.db-default pipeline
leg, VOID BY DESIGN in 1689b (research/orb_freq/RESULT_1689b.md: "a re-run that ... waits for a
quiet window on this shared box would give pools 21/24/25/26/27/28/30 their first real in-regime
read"). Pools 21/27/28 already have a real in-regime read (1689b's same-day update, via
1689b_pipeline_2128_only.py) -- reused here UNCHANGED, read-only, from subpools_1689b/, never
recomputed or overwritten.

Pools 24/25/26/30's in-regime FEATURES already exist too (built by 1689b's stage_prep, which only
reads cache.db via the read-only bar_aggregates() call -- that never hung; only the FULL
study_orb_pipeline_static_lock.py backtest subprocess hung, per 1689b_pools.py's stage_pipeline
docstring: a write-lock wait against the live trading service's own cache.db, not a slow read).
The ONLY new computation this cell runs is that backtest subprocess for 24/25/26/30, in-regime,
cache.db-default (ORB_BT_BARS_DB left unset), via 1689b_pools.py's own _run_pipeline_once /
_pipeline_env_for, loaded UNCHANGED by file path (importlib -- 1689b_pools.py's name starts with
a digit so it cannot be `import`ed normally; same pattern 1689b_pipeline_2128_only.py already used
in this cell's own lineage). stage_prep/backfill/build_feats do NOT run again here: there is
nothing left to build for any of the 7 pools, and re-running them would risk rewriting the shared
out_1689b_*/ build directories 1689b already produced -- forbidden by the owner's "never
overwrite the 1689b outputs" instruction.

Safety: this script does not itself check whether the live trading service is up. The caller
(queue_1689c.sh) is responsible for (a) confirming no other research job is running and bars_sip/
cache.db are quiet, and (b) a watchdog that SIGSTOPs/SIGCONTs the subprocess by PID around the
live service's trading window so it never overlaps a running live session.

Usage: python3 research/orb_freq/1689c_pools.py --stage {pipeline,score,all}
"""
import argparse
import importlib.util
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path('/home/ec2-user/onemil')
OUT = ROOT / 'research/orb_freq'
POOLDIR_B = OUT / 'subpools_1689b'          # 1689b's own dir -- READ ONLY, never written here
POOLDIR_C = OUT / 'subpools_1689c'          # this cell's own dir -- the only dir this script writes to
POOLDIR_C.mkdir(exist_ok=True)

NEW_POOLS = (24, 25, 26, 30)                 # cache.db-default leg, VOID in 1689b -- this cell's new read
REUSE_POOLS = (21, 27, 28)                   # already have a real in-regime read in 1689b -- reused as-is
ALL_POOLS = sorted(NEW_POOLS + REUSE_POOLS)  # == [21,24,25,26,27,28,30], the owner's 7 pools


def _load_1689b():
    """Load 1689b_pools.py by file path to reuse _run_pipeline_once / _pipeline_env_for /
    _score1684 / DAILY_SRC unchanged. Loading via spec/exec_module does NOT execute its
    `if __name__ == '__main__'` block (that only runs under `python3 1689b_pools.py`), so no
    1689b stage is triggered as a side effect of this import."""
    spec = importlib.util.spec_from_file_location('mod1689b', OUT / '1689b_pools.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stage_pipeline():
    """The one new read this cell adds: pools 24/25/26/30, in-regime, cache.db-default -- VOID in
    1689b only because the live trading service was up and contending for cache.db's lock.
    Features already exist (1689b's stage_prep built them) at
    POOLDIR_B/f'{pool}_in_regime_features.csv' -- reused unchanged, never rebuilt."""
    mod = _load_1689b()
    for pool_id in NEW_POOLS:
        fp = POOLDIR_B / f'{pool_id}_in_regime_features.csv'
        outp = POOLDIR_C / f'{pool_id}_in_regime_true.csv'
        if not fp.exists():
            mod.log.error('1689c pool %d: expected existing features file %s missing -- cannot '
                           'run (1689b stage_prep should already have built this)', pool_id, fp)
            continue
        extra_env = mod._pipeline_env_for(pool_id, 'in_regime')
        mod.log.info('1689c pool %d/in_regime: QUIET-WINDOW cache.db-default run, env=%s',
                      pool_id, extra_env)
        ok = mod._run_pipeline_once(fp, outp, extra_env, f'1689c pool {pool_id}/in_regime')
        mod.log.info('1689c pool %d/in_regime: pipeline %s', pool_id, 'OK' if ok else 'FAILED')


def stage_score():
    """Combine: NEW true.csv (24/25/26/30, in-regime, from POOLDIR_C) + REUSED true.csv
    (21/27/28 in-regime; all 7 pools out-regime -- from POOLDIR_B, read-only) into one report,
    same metrics/pass-bar as 1689b (see PREREG_1689c.md)."""
    mod = _load_1689b()
    s = mod._score1684()
    IN_LO, IN_HI = date(2025, 1, 1), date(2026, 9, 26)
    OUT_LO, OUT_HI = date(2024, 7, 1), date(2024, 12, 31)
    prod_in = s.load_book(OUT / 'fastpath/prod_true.csv')
    prod_out = s.load_book(ROOT / 'research/orb_2024/book_1415_liveexit.csv')
    print(f"production in_regime  n={0 if prod_in is None else len(prod_in)}")
    print(f"production out_regime n={0 if prod_out is None else len(prod_out)}")

    def true_path(pool_id, window):
        if window == 'in_regime' and pool_id in NEW_POOLS:
            return POOLDIR_C / f'{pool_id}_in_regime_true.csv'
        return POOLDIR_B / f'{pool_id}_{window}_true.csv'

    all_rows, reads = [], []
    for pool_id in ALL_POOLS:
        for window, prod, (lo, hi) in (('in_regime', prod_in, (IN_LO, IN_HI)),
                                        ('out_regime', prod_out, (OUT_LO, OUT_HI))):
            path = true_path(pool_id, window)
            df = s.load_book(path)
            n = 0 if df is None else len(df)
            src = 'NEW(1689c)' if (window == 'in_regime' and pool_id in NEW_POOLS) else 'reused(1689b)'
            print(f"\n--- POOL {pool_id} / {window} (n={n}, {src}, path={path}) ---")
            if df is not None and len(df):
                tag = df.copy()
                tag['window'] = window
                tag['pool'] = pool_id
                all_rows.append(tag[['window', 'pool', 'date', 'symbol', 'entry_price', '_sized_pnl', 'R', '_composite']])
                if window == 'in_regime':
                    for yr in (2025, 2026):
                        print(s.fmt(s.stats(df[df.date.dt.year == yr], f'{pool_id}/{window}/{yr}')))
                st = s.stats(df, f'{pool_id}/{window}/FULL')
                print(s.fmt(st))
                st2 = dict(st)
                st2['pool'] = pool_id
                st2['window'] = window
                reads.append(st2)
                union, addon, raw_overlap = s.union_book(prod, df)
                print(f"  raw_overlap_with_prod={raw_overlap:.1%} added_after_excl={len(addon)} "
                      f"(frequency gain = {len(addon)} fills)")
                print(s.fmt(s.stats(union, f'{pool_id}/{window}/UNION')))
                print(s.cadence_report(union, f'{pool_id}/{window}/UNION', lo, hi))
                print(s.cadence_report(prod, f'{window}/prod-alone', lo, hi))
            else:
                print(f"{pool_id}/{window}: n=0 (no fills)")
                reads.append(dict(label=f'{pool_id}/{window}', n=0, pool=pool_id, window=window))

    if all_rows:
        out_df = pd.concat(all_rows, ignore_index=True)
        out_df.to_csv(OUT / '1689c_pool_books.csv', index=False)
        print(f"\nwrote {OUT / '1689c_pool_books.csv'} rows={len(out_df)}")
    if reads:
        pd.DataFrame(reads).to_csv(OUT / '1689c_reads.csv', index=False)
        print(f"wrote {OUT / '1689c_reads.csv'} rows={len(reads)}")

    print("\n=== CELL 1689c PASS BAR (own meanR>=+0.05 & dc_t>=2.0 in-regime, >=0 out-of-regime, "
          "exTop5>0) -- identical to 1689b's bar, PREREG_1689c.md ===")
    for pool_id in ALL_POOLS:
        din = s.load_book(true_path(pool_id, 'in_regime'))
        dout = s.load_book(true_path(pool_id, 'out_regime'))
        sin = s.stats(din, f'{pool_id}/in')
        sout = s.stats(dout, f'{pool_id}/out')
        ok_in = sin.get('n', 0) > 0 and sin.get('mean_r', -9) >= 0.05 and (sin.get('dc_t') or -9) >= 2.0 \
            and (sin.get('ex_top5') or -9) > 0
        ok_out = sout.get('n', 0) == 0 or sout.get('mean_r', -9) >= 0.0
        verdict = 'PASS' if (ok_in and ok_out) else 'FAIL'
        print(f"{pool_id}: in n={sin.get('n',0)} meanR={sin.get('mean_r', float('nan')):+.3f} "
              f"dc_t={sin.get('dc_t', float('nan')):.2f} exTop5={sin.get('ex_top5', float('nan')):+.3f} | "
              f"out n={sout.get('n',0)} meanR={sout.get('mean_r', float('nan')):+.3f} -> {verdict}")

    print("\n=== PAPER-CANDIDACY RULE (PREREG_1689c.md: meanR>=+0.10 BOTH halves 2025/2026, "
          ">=2 fills/wk, exTop5>=0) ===")
    for pool_id in ALL_POOLS:
        din = s.load_book(true_path(pool_id, 'in_regime'))
        if din is None or not len(din):
            print(f"{pool_id}: n=0 -> NOT a candidate")
            continue
        rows_2025 = din[din.date.dt.year == 2025]
        rows_2026 = din[din.date.dt.year == 2026]
        s25 = s.stats(rows_2025, f'{pool_id}/2025')
        s26 = s.stats(rows_2026, f'{pool_id}/2026')
        wk25 = len(rows_2025) / max(1, rows_2025.date.dt.to_period('W').nunique()) if len(rows_2025) else 0
        wk26 = len(rows_2026) / max(1, rows_2026.date.dt.to_period('W').nunique()) if len(rows_2026) else 0
        candidate = (s25.get('mean_r', -9) >= 0.10 and s26.get('mean_r', -9) >= 0.10 and
                     wk25 >= 2 and wk26 >= 2 and (s25.get('ex_top5') or -9) >= 0 and (s26.get('ex_top5') or -9) >= 0)
        print(f"{pool_id}: 2025 meanR={s25.get('mean_r', float('nan')):+.3f} n={s25.get('n',0)} "
              f"({wk25:.1f}/wk) | 2026 meanR={s26.get('mean_r', float('nan')):+.3f} n={s26.get('n',0)} "
              f"({wk26:.1f}/wk) -> {'CANDIDATE FOR PAPER' if candidate else 'research-only'}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['pipeline', 'score', 'all'])
    a = ap.parse_args()
    if a.stage in ('pipeline', 'all'):
        stage_pipeline()
    if a.stage in ('score', 'all'):
        stage_score()
