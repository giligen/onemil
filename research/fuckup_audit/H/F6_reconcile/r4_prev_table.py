#!/usr/bin/env python3
"""R4 - one prior-day table carrying BOTH conventions for every universe symbol-day in the window.

  panel  : the immediately preceding row for that symbol in the Databento daily panel  (A's convention --
           bf_zero/build_candidates.py merges daily.groupby(symbol).close.shift(1))
  uni    : the same prior date, but its OHLC taken from universe.csv when that (symbol, prior_date) row exists
           there, else the panel                                                        (B's convention)

Also carries the universe file's own `open` for the day (B's day-level open >= 5 prefilter).
"""
import os, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
UNIV = 'research/bf_zero/universe.csv'
PANEL = 'data/research/databento/equs_daily_2025_2026.parquet'
OUT = 'research/fuckup_audit/H/F6_reconcile/prev_table.csv'
START, END = '2025-01-02', '2026-09-11'
log = lambda *a: (print(*a), sys.stdout.flush())

u = pd.read_csv(UNIV, keep_default_na=False, na_values=[''], dtype={'symbol': str, 'bar_date': str})
u = u[(u.bar_date >= START) & (u.bar_date <= END)]
log('universe rows in window:', len(u))
syms = sorted(u.symbol.unique()); sym_ix = {s: i for i, s in enumerate(syms)}; symset = set(syms)

pf = pq.ParquetFile(PANEL)
sc, dt, hi, lo, cl = [], [], [], [], []
for rg in range(pf.metadata.num_row_groups):
    d = pf.read_row_group(rg, columns=['symbol', 'bar_date', 'high', 'low', 'close']).to_pandas()
    d = d[d.symbol.isin(symset)]
    d['bar_date'] = d.bar_date.astype(str).str[:10]
    d = d[(d.bar_date >= '2024-12-01') & (d.bar_date <= END)]
    sc.append(d.symbol.map(sym_ix).to_numpy(dtype=np.int32)); dt.append(d.bar_date.to_numpy())
    hi.append(d.high.to_numpy(float)); lo.append(d.low.to_numpy(float)); cl.append(d.close.to_numpy(float))
    del d
p_sym = np.concatenate(sc); p_ds = np.concatenate(dt); p_hi = np.concatenate(hi)
p_lo = np.concatenate(lo); p_cl = np.concatenate(cl)
del sc, dt, hi, lo, cl
log('panel rows:', len(p_sym))

dates = np.array(sorted(set(p_ds.tolist()) | set(u.bar_date.unique().tolist())))
date_ix = {d: i for i, d in enumerate(dates)}; ND = len(dates)
p_date = np.array([date_ix[d] for d in p_ds], dtype=np.int32); del p_ds
key = p_sym.astype(np.int64) * ND + p_date
o_ = np.argsort(key, kind='stable')
key, p_sym, p_date, p_hi, p_lo, p_cl = key[o_], p_sym[o_], p_date[o_], p_hi[o_], p_lo[o_], p_cl[o_]
keep = np.ones(len(key), bool); keep[:-1] = key[1:] != key[:-1]
key, p_sym, p_date, p_hi, p_lo, p_cl = key[keep], p_sym[keep], p_date[keep], p_hi[keep], p_lo[keep], p_cl[keep]
log('panel unique keys:', len(key))

c = u.reset_index(drop=True)
c_sym = c.symbol.map(sym_ix).to_numpy(np.int64); c_date = c.bar_date.map(date_ix).to_numpy(np.int64)
pos = np.searchsorted(key, c_sym * ND + c_date)
ok = (pos < len(key)) & (key[np.minimum(pos, len(key) - 1)] == c_sym * ND + c_date)
prev_pos = pos - 1
has_prev = ok & (prev_pos >= 0) & (p_sym[np.maximum(prev_pos, 0)] == c_sym)
log('day in panel:', int(ok.sum()), ' with prior row:', int(has_prev.sum()))
c = c[has_prev].reset_index(drop=True); prev_pos = prev_pos[has_prev]

out = pd.DataFrame(dict(day=c.bar_date.values, symbol=c.symbol.values, day_open=c.open.values,
                        prev_date=dates[p_date[prev_pos]],
                        prev_high_panel=p_hi[prev_pos], prev_low_panel=p_lo[prev_pos],
                        prev_close_panel=p_cl[prev_pos]))
uidx = u.set_index(['symbol', 'bar_date'])[['high', 'low', 'close']]
pu = uidx.reindex(pd.MultiIndex.from_arrays([out.symbol.values, out.prev_date.values]))
m = pu.close.notna().to_numpy()
log('prior day also in universe file:', int(m.sum()), 'of', len(out))
out['prev_high_uni'] = np.where(m, pu.high.to_numpy(), out.prev_high_panel)
out['prev_low_uni'] = np.where(m, pu.low.to_numpy(), out.prev_low_panel)
out['prev_close_uni'] = np.where(m, pu.close.to_numpy(), out.prev_close_panel)
out['prev_src_uni'] = np.where(m, 'universe', 'panel')
for t in ('panel', 'uni'):
    out[f'pdr_{t}'] = np.where(out[f'prev_low_{t}'] > 0,
                               (out[f'prev_high_{t}'] - out[f'prev_low_{t}']) / out[f'prev_low_{t}'] * 100.0, np.nan)
out.to_csv(OUT, index=False)
log('wrote', OUT, len(out))
log('pdr>=8 panel:', int((out.pdr_panel >= 8).sum()), ' uni:', int((out.pdr_uni >= 8).sum()),
    ' disagree:', int(((out.pdr_panel >= 8) != (out.pdr_uni >= 8)).sum()))
log('prev_close differs (>1e-6):', int((out.prev_close_panel - out.prev_close_uni).abs().gt(1e-6).sum()))
