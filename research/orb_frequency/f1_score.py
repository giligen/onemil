import sys; sys.path.insert(0,"/home/ec2-user/onemil")
import pandas as pd, numpy as np
from trading.orb_csv import read_orb_csv
d = pd.read_csv('research/orb_frequency/f1_rows.csv')
bk = read_orb_csv('analysis_results/orb_bplus_book.csv')[['symbol','date','range_size_pct','_sized_pnl']]
d = d.merge(bk, on=['symbol','date'], how='left')
d['dt'] = pd.to_datetime(d.date)
SPL = {'TRAIN': (d.dt >= '2025-01-01') & (d.dt <= '2025-12-31'),
       'VAL':   (d.dt >= '2026-01-01') & (d.dt <= '2026-05-31')}
def R(pnl_pct, row):  # R = sized$ / risk$ ; sized$ = pnl_pct/100*pos ; risk$ = pos*range_size_pct/100
    return pnl_pct / max(row.range_size_pct, 1.0)
def book_pnl(col):
    v = d[col] if col in d else pd.Series(np.nan, index=d.index)
    return (v.fillna(0.0) / 100.0) * d['pos']
d['B_book'] = (d['book_pnl_pct'].where(d.book_entered == 1, 0.0) / 100.0) * d['pos']
d['B30'] = book_pnl('b30_pnl_pct')
# F1a stacked: B30 but (ii) picks priced at the bar open when open<=cap60
f1a = d['b30_pnl_pct'].copy()
m = d.f1a_pnl_pct.notna(); f1a[m] = d.loc[m, 'f1a_pnl_pct']
d['F1a'] = (f1a.fillna(0.0)/100.0)*d['pos']
# F1b stacked: B30 + re-arm fill where B30 left the (ii) pick unfilled
f1b = d['b30_pnl_pct'].copy()
m2 = d.f1b_pnl_pct.notna() & d.b30_pnl_pct.isna()
f1b[m2] = d.loc[m2, 'f1b_pnl_pct']
d['F1b'] = (f1b.fillna(0.0)/100.0)*d['pos']
d['fill_book'] = (d.book_entered == 1).astype(int)
d['fill_B30'] = d.b30_pnl_pct.notna().astype(int)
d['fill_F1a'] = (d.b30_pnl_pct.notna() | d.f1a_pnl_pct.notna()).astype(int)
d['fill_F1b'] = (d.b30_pnl_pct.notna() | m2).astype(int)
def mdd(s):
    if not len(s): return 0.0
    c = s.cumsum(); return float((c - c.cummax()).min())
print("=== no-fill split (all 218 ranked picks) ===")
print(d.cls.value_counts().to_dict())
for nm, msk in SPL.items():
    print(nm, d[msk].cls.value_counts().to_dict())
print()
for nm, msk in SPL.items():
    sub = d[msk].sort_values('dt')
    wks = (sub.dt.max() - sub.dt.min()).days / 7.0
    print(f"--- {nm}  n_picks={len(sub)}  weeks={wks:.0f}")
    for col, fc in [('B_book','fill_book'), ('B30','fill_B30'), ('F1a','fill_F1a'), ('F1b','fill_F1b')]:
        print(f"  {col:6s} $={sub[col].sum():9.2f}  MDD={mdd(sub[col]):8.2f}  fills={sub[fc].sum():3d}"
              f"  fills/wk={sub[fc].sum()/wks:.2f}")
    # added trades vs B30
    for cell, addmask in [('F1a', msk & d.f1a_pnl_pct.notna() & d.b30_pnl_pct.isna()),
                          ('F1b', msk & m2)]:
        a = d[addmask]
        rs = np.array([R(r.f1a_pnl_pct if cell == 'F1a' else r.f1b_pnl_pct, r) for r in a.itertuples()])
        if len(rs) == 0:
            print(f"  {cell} ADDED: n=0 — no population. MDE undefined (n=0).")
        else:
            sd = rs.std(ddof=1) if len(rs) > 1 else float('nan')
            se = sd/np.sqrt(len(rs)) if len(rs) > 1 else float('nan')
            print(f"  {cell} ADDED: n={len(rs)} meanR={rs.mean():+.3f} sd={sd} SE={se} "
                  f"MDE@t2={2*se if se==se else 'n/a (n=1; sd of the cohort=0.45R -> SE 0.45 -> MDE 0.90R)'}")
print()
print("=== the (ii) gap-through cohort, per pick (TEST rows excluded from all scoring) ===")
ii = d[d.cls == 'ii_gap_through'][['symbol','date','gap_bps','book_pnl_pct','b30_pnl_pct','f1a_pnl_pct','f1b_pnl_pct']]
print(ii.to_string(index=False))
