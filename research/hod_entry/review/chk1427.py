"""1,427 level disagreements: level used vs true HOD (bars_sip.db / cache.db, bars m < break_m), with the crossing print."""
import gzip, os, pickle, sqlite3, sys
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; HE = os.path.join(ROOT, 'research/hod_entry')
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HE); sys.path.insert(0, ROOT)
import causal_arming as ca, sip_rebuild as sr
c27 = pd.read_csv(os.path.join(HE, 'sip_rebuild_val.csv'), dtype={'day': str, 'symbol': str}, keep_default_na=False, na_values=[''])
c27['break_m'] = c27.entry_m - 1
con = sqlite3.connect(sr.CACHE_DB_URI, uri=True); sip = sqlite3.connect(ca.BARS_SIP_URI, uri=True)
rows = []
for day, g in c27.groupby('day'):
    syms = list(g.symbol)
    t = pd.read_sql(f"select symbol, t, o, h, l, c, v from bars where day=? and symbol in ({','.join('?'*len(syms))})", sip, params=[day]+syms)
    sdb = {s: ca._rth(x, 't') for s, x in t.groupby('symbol')}
    q = (f"select symbol, timestamp as t, open as o, high as h, low as l, close as c, volume as v from intraday_bars_1min "
         f"where bar_date=? and symbol in ({','.join('?'*len(syms))})")
    cdb = {s: ca._rth(x, 't') for s, x in pd.read_sql(q, con, params=[day]+syms).groupby('symbol')}
    tp = os.path.join(sr.CACHE_DIR, f'{day}.pkl.gz')
    tapes = pickle.load(gzip.open(tp, 'rb')) if os.path.exists(tp) else {}
    for r in g.itertuples():
        d = dict(day=day, symbol=r.symbol, split=r.split, status=r.status, level=r.level, ask_at=r.ask_at, break_m=r.break_m, net_R=r.net_R)
        for nm, src in (('sip', sdb), ('cdb', cdb)):
            b = src.get(r.symbol)
            sel = b[b.m < r.break_m] if b is not None else None
            d[f'lvl_{nm}'] = float(sel.h.max()) if sel is not None and len(sel) else np.nan
            d[f'n_{nm}'] = 0 if sel is None else len(sel)
        k = sr.sig_key(r.symbol, r.break_m)
        if k in tapes:
            tr, qu = tapes[k]
            S = sr.et_ns(day, r.entry_m * 60)
            bb = tr[(tr.ts >= S - 60 * 10**9) & (tr.ts < S)].sort_values('ts', kind='stable')
            hit = bb[bb.price >= round(r.level + 0.01, 6) - 1e-9]
            if len(hit):
                d['x_print'] = float(hit.price.iloc[0]); d['x_ts'] = pd.Timestamp(int(hit.ts.iloc[0]), tz='UTC').tz_convert(sr.ET).strftime('%H:%M:%S.%f')[:12]
            d['bb_max'] = float(bb.price.max()) if len(bb) else np.nan
        rows.append(d)
res = pd.DataFrame(rows); res.to_csv(os.path.join(OUT, 'chk_1427.csv'), index=False)
tol = 0.005
for sp in ('TRAIN', 'VAL'):
    for st in ('fill', 'nofill'):
        x = res[(res.split == sp) & (res.status == st)]
        print(sp, st, len(x), 'sip-match', int((abs(x.level - x.lvl_sip) <= tol).sum()), 'below', int((x.level < x.lvl_sip - tol).sum()),
              'above', int((x.level > x.lvl_sip + tol).sum()), 'nan', int(x.lvl_sip.isna().sum()),
              '| cdb-match', int((abs(x.level - x.lvl_cdb) <= tol).sum()), flush=True)
