#!/usr/bin/env python3
"""Checksum comparison of the two loaders (mine vs the original script's), one symbol per run.

usage: loader_check.py SYMBOL {mine|orig}
Prints nan-safe checksums of every array the simulator consumes.
"""
import sys
import numpy as np

sys.path.insert(0, '/home/ec2-user/onemil/research/fuckup_audit/Q')
sys.path.insert(0, '/home/ec2-user/onemil/research/lit_review_2026')

sym, which = sys.argv[1], sys.argv[2]
if which == 'mine':
    import zsim as M
    d = M.load_symbol(sym)
else:
    import test_zarattini_spy as M
    d = M.load_symbol(sym)
for k in ('C', 'O', 'VW', 'dopen', 'prevclose', 'sig14', 'sigma', 'first', 'last'):
    a = np.asarray(d[k], dtype=float)
    print(f'{which:5s} {sym} {k:10s} shape={a.shape} sum={np.nansum(a):.6f} nan={int(np.isnan(a).sum())}')
print(f'{which:5s} {sym} ndays={len(d["days"])} d0={str(d["days"][0])[:10]} dN={str(d["days"][-1])[:10]}')
