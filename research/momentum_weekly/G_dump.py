#!/usr/bin/env python3
"""Dump the 1,700t guarded top-20 (REF+guard) for three Mondays to recon/G_holdings.csv.

Parity source of truth for ``trading.momentum_sleeve.hygiene_ineligible``: the BT's own panel load, signal build and
guard flags (lines of 1700s_lowvix.py up to the GUARD_SET build) are executed verbatim, then the guarded ranking is
written. Run: bash scripts/research_run.sh -m 2500M python3 research/momentum_weekly/G_dump.py
"""
import logging
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATES = ['2021-02-08', '2025-12-29', '2026-06-29']
logging.basicConfig(stream=sys.stdout, level=logging.INFO)     # keeps the BT's own 1700s.log untouched
src = (HERE / '1700s_lowvix.py').read_text().splitlines()
stop = next(i for i, ln in enumerate(src) if ln.startswith("log.info('guard flags"))
ns = {'__name__': 'g_dump_bt', '__file__': str(HERE / '1700s_lowvix.py')}
exec(compile('\n'.join(src[:stop + 1]), '1700s_lowvix.py[:guard]', 'exec'), ns)
sr, ranked_g, ranked_syms, prior = ns['sr'], ns['ranked_g'], ns['ranked_syms'], ns['prior']
rows = []
for d in DATES:
    reb = pd.Timestamp(d)
    sig = prior[reb]
    sg = sr[sr.bar_date == sig].set_index('symbol')['sigV2']
    removed = [s for s in ranked_syms[sig][:20] if s not in ranked_g[sig][:20]]
    print(d, 'signal', sig.date(), 'guard removed from top-20:', removed, flush=True)
    for r, s in enumerate(ranked_g[sig][:20], 1):
        rows.append(dict(rebalance_date=d, signal_date=str(sig.date()), rank=r, symbol=s, signal=float(sg[s])))
pd.DataFrame(rows).to_csv(HERE / 'recon' / 'G_holdings.csv', index=False)
print('wrote', len(rows), 'rows', flush=True)
