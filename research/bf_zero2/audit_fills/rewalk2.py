#!/usr/bin/env python3
"""Second re-walk pass: the ENTRY-price question (audit item 8).

e35 / e50 : same entry BAR as the study, but the stop-buy fills 35 / 50 bps through the level instead of
            30 bps (the level is prev_close; entry_study = level*1.003). Everything else = the study spec.
trig      : the real stop-order TRIGGER (a print at/above the level) instead of the study's stricter
            "bar high >= level*1.003"; fill still 30 bps through, stop = lowest low before the new entry bar.
            Informational — it makes the study look CONSERVATIVE if it helps.
→ audit_fills/pool_rewalk2.csv
"""
import os, sqlite3, sys, time
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
A = 'research/bf_zero2/audit_fills'
sys.path.insert(0, f'{ROOT}/{A}')
from rewalk import load_bars, sim, EOD_M, OPEN_M            # noqa: E402  (module-level connections are read-only)

p = pd.read_csv(f'{A}/pool.csv', dtype={'day': str, 'symbol': str}, keep_default_na=False, na_values=[''])
print(f'pool {len(p):,}', flush=True)
rows = []; t0 = time.time()
days = sorted(p.day.unique())
for di, day in enumerate(days):
    sub = p[p.day == day]
    B = load_bars(day, sorted(sub.symbol.unique()))
    for r in sub.itertuples():
        gg = B.get(r.symbol)
        if gg is None or len(gg) < 10: continue
        o, h, l, c = (gg[k].values.astype(float) for k in ('o', 'h', 'l', 'c')); m = gg.m.values.astype(int)
        idx = np.flatnonzero(m == int(r.entry_m))
        if not len(idx): continue
        i = int(idx[0]); level = float(r.entry) / 1.003; stop = float(r.stop)
        row = dict(day=r.day, symbol=r.symbol, entry_m=int(r.entry_m))
        for tag, slip in (('e35', 0.0035), ('e50', 0.005), ('e75', 0.0075), ('e150', 0.015)):
            ent = level * (1 + slip)
            rr, why, xm, e_used, rp = sim(o, h, l, c, m, i, ent, stop)
            row[f'rr_{tag}'] = rr; row[f'why_{tag}'] = why; row[f'xm_{tag}'] = xm; row[f'rpct_{tag}'] = rp
        # the fully pessimistic entry: gap-through AND a 75 bps slip (half the measured 9:31 spread), plus the
        # entry-bar stop, a 50 bps stop slip and a 10 bps worse 15:55 fill
        rr, why, xm, e_used, rp = sim(o, h, l, c, m, i, level * 1.0075, stop, stop_bps=0.005, gap_entry=True,
                                      entry_bar_stop=True, eod_bps=0.001)
        row.update(rr_allx=rr, why_allx=why, xm_allx=xm, rpct_allx=rp, entry_allx=e_used)
        # real stop trigger: first bar (t>=1) whose high reaches the level itself
        j = np.flatnonzero((np.arange(len(h)) >= 1) & (h >= level))
        if len(j):
            k = int(j[0]); st2 = float(l[:k].min())
            if st2 < level * 1.003 and k + 1 < len(o) and m[k] <= 841:
                rr, why, xm, e_used, rp = sim(o, h, l, c, m, k, level * 1.003, st2)
                row.update(rr_trig=rr, why_trig=why, xm_trig=xm, rpct_trig=rp, entry_m_trig=int(m[k]), entry_trig=level * 1.003, stop_trig=st2)
        rows.append(row)
    if di % 40 == 0 or di == len(days) - 1:
        print(f'{di + 1}/{len(days)} {day} rows {len(rows):,} | {(time.time() - t0) / 60:.1f} min', flush=True)
R = pd.DataFrame(rows)
R.to_csv(f'{A}/pool_rewalk2.csv', index=False)
print(f'rows {len(R):,} | trig rows {int(R.rr_trig.notna().sum()) if "rr_trig" in R else 0}', flush=True)
print('DONE', flush=True)
