#!/usr/bin/env python3
"""Stage N2 step 3 — K2 (52-week-high breakout on volume) on the 2024-07 .. 2026-09 panel.

This does NOT reimplement K2. It imports `research/fuckup_audit/K/build_k.py` — the frozen Stage-K
code, unmodified on disk — and patches four module globals:

    PANEL     -> N2/daily_panel_2024H2_2026.parquet   (2024H2 Databento pull + the 2025-26 panel)
    K         -> N2/                                   (outputs land here, K/ is not touched)
    SPLITS    -> TRAIN 2025-07-01..2025-12-31 | VAL 2026-01-01..2026-05-31 | TEST 2026-06-01..
    FAMILIES/HOLDS -> K2 only, holds 10 and 5, books 10 and 20 slots  = the 4 pre-registered cells

and ANDs the universe with `bar_date >= 2025-07-01`: before that date the 250-session high52 window
is not full, and N2/PREREG declares K2 signals valid only from 2025-07.

Everything else — universe gates, the fill (next open), the cost model, the stop, the book, the
gates, the tails, the permutation, the capacity — is Stage-K code executing unchanged.

    python3 run_n2.py A     # TRAIN + VAL only (TEST is never computed)
    python3 run_n2.py B     # adds TEST, for frozen survivors only
"""
from __future__ import annotations

import os
import sys

import numpy as np

os.environ.setdefault('ARROW_DEFAULT_MEMORY_POOL', 'system')
ROOT = '/home/ec2-user/onemil'
sys.path.insert(0, f'{ROOT}/research/fuckup_audit/K')
os.chdir(ROOT)

import build_k as bk        # noqa: E402  (the Stage-K implementation, unmodified)
import report_k as rk       # noqa: E402

N2 = f'{ROOT}/research/fuckup_audit/N_databento/N2'
SIGNAL_START = '2025-07-01'

bk.PANEL = f'{N2}/daily_panel_2024H2_2026.parquet'
bk.K = N2
rk.K = N2
bk.SPLITS = [('TRAIN', '2025-07-01', '2025-12-31'),
             ('VAL', '2026-01-01', '2026-05-31'),
             ('TEST', '2026-06-01', '2026-09-11')]
bk.FAMILIES = {'K2': bk.FAMILIES['K2']}
bk.HOLDS = {'K2': (10, 5)}

_build_universe = bk.build_universe


def build_universe(f):
    """Stage-K universe, restricted to days on which the 250-session lookback is full."""
    u_prim, u_sec, cls_of = _build_universe(f)
    ok = np.array([str(d) >= SIGNAL_START for d in f['days']])[f['day']]
    bk.log(f'signal days restricted to >= {SIGNAL_START}: '
           f'primary {int((u_prim & ok).sum()):,} of {int(u_prim.sum()):,}')
    return u_prim & ok, u_sec & ok, cls_of


bk.build_universe = build_universe


if __name__ == '__main__':
    phase = sys.argv[1] if len(sys.argv) > 1 else 'A'
    bk.main(phase)
    rk.main(phase)
