"""Refuter-1: leg-level print audit for cell 1574 cycles + VOID diagnosis on crash-window Mondays."""
import sys, os
import pandas as pd, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import cell_1567 as C
cache = C.Cache()
cyc = pd.read_csv(os.path.join(os.path.dirname(HERE), 'cell_1567_cycles.csv'))
x = cyc[cyc.cell == 1574]
spy = cache.spy_minute.copy(); spy['t_et'] = spy['t'].dt.tz_convert('America/New_York')
def legrows(sym, m):
    r = cache._entry_by_symbol_monday.get((sym, m))
    return r
def sym_for(m, exp, k):
    g = cache.grid[(cache.grid.monday == m) & (cache.grid.expiry == exp) & (cache.grid.strike == k)]
    return g.symbol.iloc[0] if len(g) else None
out = []
for _, c in x.iterrows():
    m = c.entry_date
    rec = {'entry': m, 'short': c.short_strike, 'long': c.long_strike, 'credit': c.credit}
    for leg, k in [('s', c.short_strike), ('l', c.long_strike)]:
        r = legrows(sym_for(m, c.expiry, k), m)
        mins = r['t_et'].dt.hour * 60 + r['t_et'].dt.minute
        core = r[(mins >= 600) & (mins < 605)]
        rec[leg + '_nbars'] = len(r); rec[leg + '_core'] = len(core); rec[leg + '_vol'] = r['v'].sum()
        use = core if len(core) else r.loc[[(mins - 600).abs().idxmin()]]
        rec[leg + '_min'] = int((use['t_et'].dt.hour * 60 + use['t_et'].dt.minute).mean())
        rec[leg + '_hl'] = float((r['h'] - r['l']).max())
    # SPY move between the two leg print minutes
    d = spy[spy.day == m]; dm = d['t_et'].dt.hour * 60 + d['t_et'].dt.minute
    ps = d.loc[(dm - rec['s_min']).abs().idxmin(), 'c']; pl = d.loc[(dm - rec['l_min']).abs().idxmin(), 'c']
    rec['spy_move_between_legs'] = ps - pl
    rec['fallback_any'] = int(rec['s_core'] == 0 or rec['l_core'] == 0)
    out.append(rec)
L = pd.DataFrame(out)
pd.set_option('display.width', 250)
print(L.to_string())
print('fallback share (either leg):', L.fallback_any.mean(), ' mean |min gap| :', (L.s_min - L.l_min).abs().mean(),
      ' median long vol', L.l_vol.median(), ' median short vol', L.s_vol.median())
# approx credit error from non-synchronous prints: delta_short~0.15, delta_long~0.08 ⇒ net ~0.07*dS
print('mean |0.07*dSPY| between legs ($/sh):', (0.07 * L.spy_move_between_legs.abs()).mean())
L.to_csv(os.path.join(HERE, 'refuter1_1574_legs.csv'), index=False)
# VOID diagnosis
for m in ['2025-01-13', '2025-02-03', '2025-02-24', '2025-03-03', '2025-03-24', '2025-03-31', '2025-04-07', '2026-02-17']:
    wc = {}
    mkt = C.precompute_monday(cache, m, wc)
    st = mkt['strikes']; spot = mkt['spot']; T = mkt['dte'] / 365
    pr = []
    for _, r in st.iterrows():
        mid = cache.entry_mid(r.symbol, m)
        if mid is None: continue
        iv = C.implied_vol_put(mid, spot, r.strike, T, C.R_RATE, C.Q_RATE)
        dl = C.bs_put_delta(spot, r.strike, T, C.R_RATE, C.Q_RATE, iv) if iv else None
        pr.append((r.strike, round(mid, 2), None if dl is None else round(dl, 3)))
    near = [p for p in pr if p[2] is not None and 0.08 < -p[2] < 0.25]
    print(m, 'spot', spot, 'exp', mkt['expiry'], 'grid strikes', len(st), 'printed', len(pr), 'near-0.15 printed:', near)
