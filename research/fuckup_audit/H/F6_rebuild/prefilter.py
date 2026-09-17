"""Stage 1 of the F6 independent rebuild: daily-level prefilter.

Produces candidate (symbol, day) rows that satisfy the daily-bar preconditions
of the F6 red-to-green book:
  * symbol-day is in the point-in-time universe file, open >= 5
  * the PRIOR trading day of that symbol has range % = (high-low)/low*100 >= 8
  * the day's open is BELOW the prior close

Prior-day OHLC source: the universe file when that (symbol, prior_day) row is
present there, else the Databento daily panel.  The trading-day sequence per
symbol is taken from the Databento panel (the only complete daily source).

Memory-conscious: the panel is held as integer-coded numpy arrays, never as a
wide object-dtype DataFrame.
"""
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

UNIV = 'research/bf_zero/universe.csv'
PANEL = 'data/research/databento/equs_daily_2025_2026.parquet'
OUT = 'research/fuckup_audit/H/F6_rebuild/prefilter.csv'

START, END = '2025-01-02', '2026-09-11'


def log(*a):
    print(*a)
    sys.stdout.flush()


def main():
    u = pd.read_csv(UNIV, keep_default_na=False, na_values=[''])
    u = u[(u.bar_date >= START) & (u.bar_date <= END)]
    log('universe rows in window:', len(u))

    syms = sorted(u.symbol.unique())
    sym_ix = {s: i for i, s in enumerate(syms)}
    symset = set(syms)

    # ---- load the daily panel as compact arrays -------------------------
    pf = pq.ParquetFile(PANEL)
    sc, dt, hi, lo, cl = [], [], [], [], []
    for rg in range(pf.metadata.num_row_groups):
        d = pf.read_row_group(rg, columns=['symbol', 'bar_date', 'high', 'low', 'close']).to_pandas()
        d = d[d.symbol.isin(symset)]
        d = d[(d.bar_date >= '2024-12-01') & (d.bar_date <= END)]
        sc.append(d.symbol.map(sym_ix).to_numpy(dtype=np.int32))
        dt.append(d.bar_date.to_numpy())
        hi.append(d.high.to_numpy(dtype=np.float64))
        lo.append(d.low.to_numpy(dtype=np.float64))
        cl.append(d.close.to_numpy(dtype=np.float64))
        del d
        log('  row group %d kept %d' % (rg, len(sc[-1])))
    p_sym = np.concatenate(sc); del sc
    p_datestr = np.concatenate(dt); del dt
    p_hi = np.concatenate(hi); del hi
    p_lo = np.concatenate(lo); del lo
    p_cl = np.concatenate(cl); del cl
    log('panel rows:', len(p_sym))

    dates = np.array(sorted(set(p_datestr.tolist()) | set(u.bar_date.unique().tolist())))
    date_ix = {d: i for i, d in enumerate(dates)}
    p_date = np.array([date_ix[d] for d in p_datestr], dtype=np.int32)
    del p_datestr
    ND = len(dates)

    key = p_sym.astype(np.int64) * ND + p_date
    order = np.argsort(key, kind='stable')
    key = key[order]
    p_sym = p_sym[order]; p_date = p_date[order]
    p_hi = p_hi[order]; p_lo = p_lo[order]; p_cl = p_cl[order]
    # drop duplicate (symbol, date) keeping the last
    keep = np.ones(len(key), dtype=bool)
    keep[:-1] = key[1:] != key[:-1]
    key, p_sym, p_date = key[keep], p_sym[keep], p_date[keep]
    p_hi, p_lo, p_cl = p_hi[keep], p_lo[keep], p_cl[keep]
    log('panel unique (symbol,date):', len(key))

    # ---- candidates ------------------------------------------------------
    cand = u[u.open >= 5.0].copy().reset_index(drop=True)
    log('after open>=5:', len(cand))

    c_sym = cand.symbol.map(sym_ix).to_numpy(dtype=np.int64)
    c_date = cand.bar_date.map(date_ix).to_numpy(dtype=np.int64)
    c_key = c_sym * ND + c_date
    pos = np.searchsorted(key, c_key)
    ok = (pos < len(key)) & (key[np.minimum(pos, len(key) - 1)] == c_key)
    log('candidate day not found in panel:', int((~ok).sum()))
    prev_pos = pos - 1
    has_prev = ok & (prev_pos >= 0) & (p_sym[np.maximum(prev_pos, 0)] == c_sym)
    log('no prior trading day for symbol in panel:', int((ok & ~has_prev).sum()))

    cand = cand[has_prev].reset_index(drop=True)
    prev_pos = prev_pos[has_prev]
    cand['prev_date'] = dates[p_date[prev_pos]]
    cand['prev_high'] = p_hi[prev_pos]
    cand['prev_low'] = p_lo[prev_pos]
    cand['prev_close'] = p_cl[prev_pos]
    cand['prev_src'] = 'panel'

    # prefer the universe file's own row for the prior day when present
    uidx = u.set_index(['symbol', 'bar_date'])[['high', 'low', 'close']]
    ukeys = pd.MultiIndex.from_arrays([cand.symbol.values, cand.prev_date.values])
    pu = uidx.reindex(ukeys)
    m = pu.close.notna().to_numpy()
    log('prior day also present in universe file:', int(m.sum()), 'of', len(cand))
    cand.loc[m, 'prev_high'] = pu.high.to_numpy()[m]
    cand.loc[m, 'prev_low'] = pu.low.to_numpy()[m]
    cand.loc[m, 'prev_close'] = pu.close.to_numpy()[m]
    cand.loc[m, 'prev_src'] = 'universe'

    cand = cand[(cand.prev_low > 0) & cand.prev_close.notna()]
    cand['prev_range_pct'] = (cand.prev_high - cand.prev_low) / cand.prev_low * 100.0
    n0 = len(cand)
    cand = cand[cand.prev_range_pct >= 8.0]
    log('after prev_range_pct>=8: %d (from %d)' % (len(cand), n0))

    cand['level'] = cand.prev_close * 1.003
    cand = cand.sort_values(['bar_date', 'symbol'])
    cand.to_csv(OUT.replace('prefilter.csv', 'prefilter_all.csv'), index=False)
    log('wrote prefilter_all.csv (red-open test deferred to the 09:30 bar):', len(cand))

    n0 = len(cand)
    cand = cand[cand.open < cand.prev_close]
    log('daily-file open < prev_close (reference only): %d (from %d)' % (len(cand), n0))
    cand.to_csv(OUT, index=False)
    log('wrote', OUT, len(cand), 'rows,', cand.symbol.nunique(), 'symbols,',
        cand.bar_date.nunique(), 'days')


if __name__ == '__main__':
    main()
