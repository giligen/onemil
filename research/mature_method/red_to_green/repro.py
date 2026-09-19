#!/usr/bin/env python3
"""Step 1 - REPRODUCTION GATE. This pass's S2 candidate set vs implementation E of
research/fuckup_audit/H/F6_reconcile (r8_engine_book.py), candidate by candidate on (day, symbol).

E's gates: pdr>=8, floor 5 on the running low, level x1.003, cap x1.006, price >= $5 on the ENTRY,
r >= 1%, signal minute <= 840, next printed bar. E did NOT apply the live universe screens
(adv20 >= 100K, prev close >= $5) and did NOT drop non-daily_bars symbols; this pass does both,
so the difference is decomposed rather than hidden.
"""
import os, sys
import numpy as np, pandas as pd
ROOT='/home/ec2-user/onemil'; os.chdir(ROOT)
OUT='research/mature_method/red_to_green'
RD=lambda p,**k: pd.read_csv(p,dtype={'symbol':str,'day':str},keep_default_na=False,na_values=[''],**k)
log=lambda *a:(print(*a),sys.stdout.flush())

e=RD('research/fuckup_audit/H/F6_reconcile/e_cands.csv')
log('E candidates:', len(e))
d=RD(f'{OUT}/cands.csv')
m=d[d.variants.str.contains('lvl1003_f5_S2',regex=False)]
mine=m[(m.pdr>=8)&(m.floor_val>=5)&(m.entry>=5)&(m.r_pct>=1)&(m.sig_m<=840)&(m.over_cap_bps<=0)]
log('this pass, E gates, honest population:', len(mine))

ke=set(zip(e.day,e.symbol)); km=set(zip(mine.day,mine.symbol))
log('shared:',len(ke&km),'  E-only:',len(ke-km),'  mine-only:',len(km-ke))
eo=e[[ (a,b) in (ke-km) for a,b in zip(e.day,e.symbol)]]
import re
TT=re.compile(r'^Z[A-Z]ZZT')
import sqlite3
c=sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro',uri=True)
db={r[0] for r in c.execute('select distinct symbol from daily_bars')}
log('E-only explained by the two membership cuts: test-ticker %d, non-daily_bars %d, other %d'
    % (int(eo.symbol.map(lambda s: bool(TT.match(s))).sum()),
       int((~eo.symbol.isin(db)).sum()),
       int(((~eo.symbol.map(lambda s: bool(TT.match(s)))) & (eo.symbol.isin(db))).sum())))

j=e.merge(mine,on=['day','symbol'],suffixes=('_e','_m'))
for f in ('sig_m','entry_m','entry','stop','r_pct'):
    dd=(j[f+'_e']-j[f+'_m']).abs()
    log('  |d %-8s|  max %.6g   n differing %d of %d'%(f,dd.max(),int((dd>1e-6).sum()),len(j)))
for mode in ('hold','r2','partial'):
    a=j[f'{mode}_grossR_e'] if f'{mode}_grossR_e' in j else j[f'{mode}_grossR']
    b=j[f'{mode}_grossR_m'] if f'{mode}_grossR_m' in j else None
    if b is None: continue
    dd=(a-b).abs(); log('  |d %-8s gross R| max %.6g  n differing %d'%(mode,dd.max(),int((dd>1e-6).sum())))
