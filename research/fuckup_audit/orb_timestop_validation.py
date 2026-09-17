#!/usr/bin/env python3
"""ORB time-stop validation on the HONEST ORB fills (analysis_results/orb_features_20260916_2053.csv, entered==1,
the entered-inclusive rebuild, Jan-2025..Sep-2026). Declared BEFORE running (2026-09-17 23:05 UTC):
  rule T(m, r): at fill + m minutes, if the trade's progress (that bar's open − entry)/R is below r, exit at that open
  (less 0.3% of R for the crossing); otherwise keep the recorded outcome.
  Cells: m ∈ {5, 10, 15} × r ∈ {0.0, 0.25, 0.5} = 9, reported per split (TRAIN 2025 / VAL 2026-01..05 / TEST 2026-06+)
  and on the B+ book subset (analysis_results/orb_bplus_book.csv). The primary cell is the one seen on the live trades:
  m=10, r=0.25. R per trade = the 5-minute range (09:30–09:34 high − low) from the bars; the fill minute = the first
  bar after 09:34 whose high ≥ range_high × 1.003 (the pre-placed stop-limit's trigger).
Output: research/fuckup_audit/orb_timestop_validation.md"""
import sqlite3, numpy as np, pandas as pd
from zoneinfo import ZoneInfo
ET = ZoneInfo('America/New_York'); pd.set_option('display.width', 250)
cache = sqlite3.connect('file:data/cache.db?mode=ro', uri=True)
F = pd.read_csv('analysis_results/orb_features_20260916_2053.csv', keep_default_na=False, na_values=[''], dtype={'symbol': str})
F = F[F.entered == 1].copy(); print('fills', len(F), flush=True)
B = pd.read_csv('analysis_results/orb_bplus_book.csv', keep_default_na=False, na_values=[''], dtype={'symbol': str})
bkey = set(zip(B.symbol, B[[c for c in B.columns if 'date' in c.lower()][0]].astype(str).str[:10]))
MS = (5, 10, 15); RS = (0.0, 0.25, 0.5)


def bars(sym, day):
    b = pd.read_sql("select timestamp, open, high, low, close from intraday_bars_1min where symbol=? and bar_date=? order by timestamp", cache, params=(sym, day))
    if not len(b): return b
    ts = pd.to_datetime(b.timestamp, utc=True).dt.tz_convert(ET); b['m'] = ts.dt.hour * 60 + ts.dt.minute
    return b[(b.m >= 570) & (b.m < 960)].reset_index(drop=True)


rows = []
for i, r in enumerate(F.itertuples()):
    if i % 500 == 0: print(i, flush=True)
    b = bars(r.symbol, r.date)
    if len(b) < 20: continue
    rng = b[(b.m >= 570) & (b.m < 575)]
    if len(rng) < 3: continue
    hi, lo = float(rng.high.max()), float(rng.low.min()); R = hi - lo
    if R <= 0: continue
    trig = hi * 1.003; after = b[b.m >= 575]
    hit = after[after.high >= trig]
    if not len(hit): continue
    fm = int(hit.iloc[0].m); entry = float(r.entry_price) if r.entry_price == r.entry_price and r.entry_price > 0 else trig
    R_real = float(r.pnl_pct) / 100.0 * entry / R if r.pnl_pct == r.pnl_pct else np.nan
    o = dict(symbol=r.symbol, date=r.date, split='TRAIN' if r.date < '2026-01-01' else 'VAL' if r.date < '2026-06-01' else 'TEST', bplus=(r.symbol, r.date) in bkey, R_real=R_real)
    for m in MS:
        at = b[b.m == fm + m]
        prog = (float(at.iloc[0].open) - entry) / R if len(at) else np.nan
        for rr in RS:
            o[f'R_T{m}_{rr}'] = (prog - 0.003) if (prog == prog and prog < rr) else R_real
    rows.append(o)
V = pd.DataFrame(rows).dropna(subset=['R_real']); V.to_csv('research/fuckup_audit/orb_timestop_validation.csv', index=False)
L = ['# ORB time stop — validation on the honest fills', f'fills with bars {len(V)}', '']
for lab, sub in (('ALL FILLS', V), ('B+ BOOK', V[V.bplus])):
    L.append(f'## {lab}')
    for sp in ('TRAIN', 'VAL', 'TEST'):
        x = sub[sub.split == sp]
        if not len(x): continue
        cells = ' | '.join(f'T{m}/{rr}: {x[f"R_T{m}_{rr}"].mean():+.3f}' for m in MS for rr in RS)
        L.append(f'{sp}: n {len(x)} real {x.R_real.mean():+.3f} (WR {(x.R_real > 0).mean():.0%}) | {cells}')
    L.append('')
L.append('primary cell T10/0.25; 9 cells x 2 populations x 3 splits looked at.')
open('research/fuckup_audit/orb_timestop_validation.md', 'w').write('\n'.join(L)); print('\n'.join(L))
