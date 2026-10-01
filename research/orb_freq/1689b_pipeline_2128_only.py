#!/usr/bin/env python3
"""Targeted stage_pipeline for cell 1,689b restricted to pools 21/27/28 ONLY (both
windows) -- avoids 1689b_pools.py's own stage_pipeline() re-running the pipeline for
pools 23/24/25/26/30 too (it has no skip-if-exists guard and would needlessly redo
already-scored work). Imports 1689b_pools.py's own _run_pipeline_once/_pipeline_env_for/
BARS_SIP/DAILY_SRC/POOLDIR unchanged -- no reimplementation of pipeline mechanics.

Usage: nice -n 10 python3 research/orb_freq/1689b_pipeline_2128_only.py
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    old_argv = sys.argv
    sys.argv = [fname, '--stage', 'score']  # harmless placeholder, module guards stages behind __main__
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = old_argv
    return mod


def main():
    m = _load('cell1689b_pools', '1689b_pools.py')
    for pool_id in (21, 27, 28):
        for window in m.WINDOWS:
            fp = m.POOLDIR / f'{pool_id}_{window}_features.csv'
            outp = m.POOLDIR / f'{pool_id}_{window}_true.csv'
            ok = m._run_pipeline_once(fp, outp, m._pipeline_env_for(pool_id, window), f'pool {pool_id}/{window}')
            print(f"pool {pool_id}/{window}: ok={ok} -> {outp}")


if __name__ == '__main__':
    main()
