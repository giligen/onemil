"""Adversarial check of cell 1,438 (read-only). Parts 1 (level), 2 (fill rule), 4 (features) -> pickles/CSVs in scratchpad."""
import gzip, os, pickle, sqlite3, sys, math
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'
HE = os.path.join(ROOT, 'research/hod_entry')
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HE); sys.path.insert(0, ROOT)
import causal_arming as ca, sip_rebuild as sr
from trading.hod_break import rv_profile

def P(*a):
    print(*a, flush=True)

p, floor, min_adv = ca.live_params()
c38 = pd.read_csv(os.path.join(HE, 'causal_arming_causal.csv'), dtype={'day': str, 'symbol': str}, keep_default_na=False, na_values=[''])
fills = c38[c38.status == 'fill'].copy()
P('status counts', c38.groupby(['split', 'status']).size().to_dict())
u = pd.read_csv(ca.UNIVERSE_CSV, dtype={'symbol': str}, keep_default_na=False).rename(columns={'bar_date': 'day'})
u['adv20'] = pd.to_numeric(u.adv20, errors='coerce')
adv = u.drop_duplicates(['day', 'symbol']).set_index(['day', 'symbol']).adv20.to_dict()
con = sqlite3.connect(sr.CACHE_DB_URI, uri=True)
sip = sqlite3.connect(ca.BARS_SIP_URI, uri=True)

def rth(g, col):
    return ca._rth(g, col)

rows = []
for day, fd in fills.groupby('day'):
    syms = list(fd.symbol)
    chosen = ca.load_day_bars(con, day, syms, sip)
    q = (f"select symbol, timestamp as t, open as o, high as h, low as l, close as c, volume as v from intraday_bars_1min "
         f"where bar_date=? and symbol in ({','.join('?' * len(syms))})")
    cdb = {s: rth(g, 't') for s, g in pd.read_sql(q, con, params=[day] + syms).groupby('symbol')}
    t = pd.read_sql(f"select symbol, t, o, h, l, c, v from bars where day=? and symbol in ({','.join('?' * len(syms))})", sip, params=[day] + syms)
    sdb = {s: rth(g, 't') for s, g in t.groupby('symbol')}
    cpath = os.path.join(sr.CACHE_DIR, f'c1438_{day}.pkl.gz')
    cache = pickle.load(gzip.open(cpath, 'rb')) if os.path.exists(cpath) else {}
    for r in fd.itertuples():
        bm = int(math.floor(r.fill_min))
        d = dict(day=day, symbol=r.symbol, split=r.split, level=r.level, fill=r.fill, stop=r.stop, fill_min=r.fill_min, net_R=r.net_R, bm=bm)
        s_b, c_b = sdb.get(r.symbol), cdb.get(r.symbol)
        d['lvl_sip'] = float(s_b[s_b.m < bm].h.max()) if s_b is not None and (s_b.m < bm).any() else np.nan
        d['lvl_cdb'] = float(c_b[c_b.m < bm].h.max()) if c_b is not None and (c_b.m < bm).any() else np.nan
        d['n_sip'] = 0 if s_b is None else len(s_b); d['n_cdb'] = 0 if c_b is None else len(c_b)
        d['nb_sip'] = 0 if s_b is None else int((s_b.m < bm).sum()); d['nb_cdb'] = 0 if c_b is None else int((c_b.m < bm).sum())
        if s_b is not None and len(s_b):
            d['day_range'] = float(s_b.h.max() / s_b.l.min() - 1)
        b = chosen.get(r.symbol)
        a_adv = adv.get((day, r.symbol), np.nan)
        cands = ca.armed_crossing_bars(b, a_adv, p, floor) if b is not None else []
        match = [(i, a) for i, a in enumerate(cands) if a['m_lo'] + 1 <= r.fill_min < a['m_hi'] + 1]
        d['n_cands'] = len(cands)
        if match:
            i, a = match[0]
            d.update(arm_idx=i, m_lo=a['m_lo'], m_hi=a['m_hi'], lvl_re=a['level'], stop_re=a['stop'],
                     open0=float(b.o.iloc[0]), rv=rv_profile(a['cumv_j'], a_adv, a['m_lo']),
                     range_j=float(b.h.values[:a['j'] + 1].max() / b.l.values[:a['j'] + 1].min() - 1), cumv_j=a['cumv_j'], adv20=a_adv,
                     src='sip' if (s_b is not None and len(b) == len(s_b)) else 'cdb')
            # fill-rule re-verification from the cached tape
            key = f"{r.symbol}|{a['m_lo']}|{a['m_hi']}"
            if key in cache:
                tr, qu = cache[key]
                st, en = ca.window_ns(day, a['m_lo'], a['m_hi'])
                w = tr[(tr.ts >= st) & (tr.ts < en)].sort_values('ts', kind='stable')
                hit = w[w.price >= a['trigger'] - 1e-9]
                if len(hit):
                    th = int(hit.ts.iloc[0]); pq = sr.prevailing_quote(qu, th)
                    qv = sr._valid(qu[qu.ts <= th])
                    d.update(v_hit_ts_ok=th >= st, v_print=float(hit.price.iloc[0]), v_size=float(hit['size'].iloc[0]),
                             v_ask=pq[1] if pq else np.nan, v_bid=pq[0] if pq else np.nan,
                             v_qage_s=(th - int(qv.ts.max())) / 1e9 if len(qv) else np.nan,
                             v_limit=a['limit'], v_prev_hit=int(((w.ts < th) & (w.price >= a['trigger'] - 1e-9)).any()),
                             v_tape_max_before=float(tr[tr.ts < st].price.max()) if (tr.ts < st).any() else np.nan)
            d['in_cache'] = key in cache
        rows.append(d)
    P(day, len(fd))
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, 'chk_fills.csv'), index=False)
P('done', len(res))
