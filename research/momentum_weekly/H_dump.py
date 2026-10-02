#!/usr/bin/env python3
"""Dump the 1,700u half-size cell `VIXratio|w252|p20|half` gate state + per-name target weight for three Mondays
(two gated, one not) to recon/H_gate.csv.

Parity source of truth for ``trading.momentum_sleeve.gate_scale`` / ``target_dollars``: the BT's own panel load, GREF
repro gate, CBOE percentiles (``prank``) and gate rule (``nan_to_num(pct, 1.0) < 0.20``; scale 0.5 -> every name at
eq*0.5/N) are executed verbatim from 1700u_gate_guarded.py up to its shift test (the cell loop and the BT's log
file are not touched). Dates: the 2nd and 4th-quintile gated Mondays and the median ungated Monday since 2021.
Run: bash scripts/research_run.sh -m 2500M python3 research/momentum_weekly/H_dump.py
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
logging.basicConfig(stream=sys.stdout, level=logging.INFO)     # keeps the BT's own 1700u.log untouched
src = (HERE / '1700u_gate_guarded.py').read_text().splitlines()
stop = next(i for i, ln in enumerate(src) if ln.startswith("say(f'SHIFT TEST"))
ns = {'__name__': 'h_dump_bt', '__file__': str(HERE / '1700u_gate_guarded.py')}
exec(compile('\n'.join(src[:stop + 1]), '1700u_gate_guarded.py[:shift]', 'exec'), ns)
rebal_dates, prior, PCTd, ranked_g = ns['rebal_dates'], ns['prior'], ns['PCTd'], ns['ranked_g']
N_NAMES, SCALE_HALF, THR = 20, 0.5, 0.20
pv = np.array([PCTd[('VIXratio', 252)].loc[prior[d]] for d in rebal_dates])
gated = np.nan_to_num(pv, nan=1.0) < THR
idx = [i for i, d in enumerate(rebal_dates) if d >= pd.Timestamp('2021-01-01')]
g_idx = [i for i in idx if gated[i]]
u_idx = [i for i in idx if not gated[i]]
print(f'{len(g_idx)} gated / {len(u_idx)} ungated Mondays since 2021', flush=True)
pick = [g_idx[len(g_idx) // 5], g_idx[3 * len(g_idx) // 5], u_idx[len(u_idx) // 2]]
rows = []
for i in sorted(pick):
    reb = rebal_dates[i]; sig = prior[reb]
    scale = SCALE_HALF if gated[i] else 1.0
    print(reb.date(), 'signal', sig.date(), 'percentile', round(float(pv[i]), 4), 'gated', bool(gated[i]), flush=True)
    for r, s in enumerate(ranked_g[sig][:N_NAMES], 1):
        rows.append(dict(rebalance_date=str(reb.date()), signal_date=str(sig.date()), percentile=float(pv[i]),
                         gated=bool(gated[i]), rank=r, symbol=s, weight=scale / N_NAMES))
pd.DataFrame(rows).to_csv(HERE / 'recon' / 'H_gate.csv', index=False)
print('wrote', len(rows), 'rows', flush=True)
