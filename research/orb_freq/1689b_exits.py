#!/usr/bin/env python3
"""Cell 1,689b exit menu -- for each pool positive in BOTH windows (own LIVE exit,
mean_R>0 in_regime AND out_regime per 1689b_reads.csv), run PREREG_1693's 12-exit
menu in both selection directions. Reuses research/orb_freq/1693_pool_exits.py's
build_per_fill_table/score_table/classify_pool/pool_summary_line VERBATIM (imported
read-only via importlib, same trick 1689b_pools.py's stage_score() uses for
1684_score.py) -- no reimplementation of exit mechanics, no new fetch.

Usage: nice -n 10 python3 research/orb_freq/1689b_exits.py
"""
import importlib.util
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    old_argv = sys.argv
    sys.argv = [fname]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = old_argv
    return mod


def main():
    e = _load('cell1693_exits', '1693_pool_exits.py')

    reads = pd.read_csv(os.path.join(HERE, '1689b_reads.csv'))
    books = pd.read_csv(os.path.join(HERE, '1689b_pool_books.csv'))

    candidates = sorted(books['pool'].unique().tolist())
    qualifying = []
    for pool_id in candidates:
        r = reads[reads['pool'] == pool_id]
        rin = r[r['window'] == 'in_regime']
        rout = r[r['window'] == 'out_regime']
        if rin.empty or rout.empty or rin.iloc[0]['n'] in (0, None) or rout.iloc[0]['n'] in (0, None):
            print(f"pool {pool_id}: SKIP -- missing a real read in one window")
            continue
        mr_in, mr_out = rin.iloc[0]['mean_r'], rout.iloc[0]['mean_r']
        if pd.isna(mr_in) or pd.isna(mr_out):
            print(f"pool {pool_id}: SKIP -- NaN mean_r in one window")
            continue
        if mr_in > 0 and mr_out > 0:
            qualifying.append(pool_id)
            print(f"pool {pool_id}: QUALIFIES both-window-positive (in={mr_in:+.3f}R out={mr_out:+.3f}R)")
        else:
            print(f"pool {pool_id}: fails both-window-positive (in={mr_in:+.3f}R out={mr_out:+.3f}R)")

    if not qualifying:
        print("\nNo pool is positive in both windows -- 12-exit menu not run (nothing qualifies).")
        with open(os.path.join(HERE, '1689b_exits_summary.txt'), 'w') as fh:
            fh.write("No pool positive in both windows -- 12-exit menu not run.\n")
        return

    store = e.CachedStore(e.f1668.BARS_DB)
    lines = []
    for pool_id in qualifying:
        sub = books[books['pool'] == pool_id]
        rows = list(zip(sub['date'], sub['symbol'], sub['entry_price']))
        pf, recon = e.build_per_fill_table(rows, store, f'1689b/{pool_id}')
        rt = e.score_table(pf, str(pool_id))
        ct = e.classify_pool(rt, str(pool_id))
        line = e.pool_summary_line(ct, str(pool_id))
        print(line)
        lines.append(line)
        rt.to_csv(os.path.join(HERE, f'1689b_exits_{pool_id}_reads.csv'), index=False)
        ct.to_csv(os.path.join(HERE, f'1689b_exits_{pool_id}_class.csv'), index=False)
    store.close()

    with open(os.path.join(HERE, '1689b_exits_summary.txt'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f"\nwrote 1689b_exits_summary.txt ({len(lines)} pools)")


if __name__ == '__main__':
    main()
