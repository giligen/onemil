#!/usr/bin/env python3
"""hod_filter_stack — the two LABELLED-SUBSET arms: news recency and CKS order-flow imbalance.

Both are scored on their covered subset only and labelled, exactly as PREREG §3 declares.
Coverage is printed BEFORE any separation number.
"""
import os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT); sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_break')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')
import score as S       # noqa: E402
import score2 as S2     # noqa: E402

D = f'{ROOT}/research/mature_method/hod_filter_stack'


def arm(b2, feat, label):
    v = pd.to_numeric(b2[feat], errors='coerce')
    cov = float(v.notna().mean())
    print(f'\n-- {label} ({feat}) — coverage {cov:.1%} of B2 ({int(v.notna().sum())} rows) --')
    sub = b2[v.notna()].copy()
    for sp in S.SPLITS:
        d = sub[sub.split == sp]
        print(f'   covered subset {sp}: n={len(d)} gross {d.rr.mean():+.3f} net {d.net.mean():+.3f}')
    if v.notna().sum() < 600:
        print('   too few rows to cut terciles — arm reported as UNDERPOWERED'); return
    t = S2.tercile_table(sub, [feat], f'{D}/arm_{feat}.csv')
    r = t.iloc[0]
    if pd.isna(r.get('spread')):
        print(f'   {r.get("note")}'); return
    print(f'   TRAIN terciles (<= {r.q1:.4g}, <= {r.q2:.4g}): gross {r.mu0:+.3f} / {r.mu1:+.3f} / '
          f'{r.mu2:+.3f}  spread {r.spread:+.3f} (winsor {r.wspread:+.3f}, t {r.t:+.2f}), '
          f'halves consistent {r.half_consistent}, VAL spread {r.val_spread:+.3f}, '
          f'selectable {r.selectable}')
    vv = pd.to_numeric(sub[feat], errors='coerce')
    terc = np.where(vv <= r.q1, 0, np.where(vv <= r.q2, 1, 2))
    print(S2.HDR)
    S2.show(f'ARM {feat} T{int(r.best)}', S.apply_book(sub[terc == int(r.best)], 12, 4))


def main():
    b2 = pd.read_pickle(f'{D}/b2.pkl')
    if os.path.exists(f'{D}/ofi.csv'):
        o = pd.read_csv(f'{D}/ofi.csv', dtype={'symbol': str, 'day': str},
                        keep_default_na=False, na_values=['']).drop_duplicates(['day', 'symbol', 'break_m'])
        b2 = b2.merge(o, on=['day', 'symbol', 'break_m'], how='left')
        n20 = (pd.to_numeric(b2.n_upd, errors='coerce') >= 20)
        print(f'OFI arm: pulled rows {len(o)}; merged onto B2 {int(b2.ofi_5m.notna().sum())} '
              f'({b2.ofi_5m.notna().mean():.1%}); with >=20 quote updates in the window '
              f'{int(n20.sum())} ({n20.mean():.1%})  <-- the declared coverage number')
        b2.loc[~n20, ['ofi_5m', 'ofi_1m']] = np.nan
        arm(b2, 'ofi_5m', 'CKS OFI, 5 min into the break, depth-normalised (bbo-1s proxy, EQUS.MINI)')
        arm(b2, 'ofi_1m', 'CKS OFI, the break minute only (bbo-1s proxy, EQUS.MINI)')
    else:
        print('OFI arm: ofi.csv absent — NOT PULLED (see REPORT)')
    arm(b2, 'news_recency_min', 'minutes since the last own-ticker premarket headline')
    arm(b2, 'news_n', 'own-ticker premarket article count')


if __name__ == '__main__':
    main()
