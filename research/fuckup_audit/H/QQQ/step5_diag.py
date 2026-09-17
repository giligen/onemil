#!/usr/bin/env python3
"""H/QQQ step 5 — VAL failed, so TEST is NOT read.  This file quantifies the failure:
  (a) minimum detectable effect on TRAIN and VAL (the power of the test that just failed);
  (b) the numerator/denominator decomposition that explains why the stack raised the MEAN while
      LOWERING the SUM on both TRAIN and VAL (the defect in the pre-registration);
  (c) one POST-HOC diagnostic cell, labelled as such and NOT adopted: the AND-veto (skip only when
      band_w < 0.8 AND |gap|/band < 0.20);
  (d) a permutation p for the frozen stack's TRAIN improvement;
  (e) the day-level CSV of the final book with every flag, for an independent rebuild.
No TEST-window filtered book is evaluated anywhere in this file.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/fuckup_audit/H/QQQ')
sys.path.insert(0, '/home/ec2-user/onemil/research/fuckup_audit/Q')
import zsim as Z                                                    # noqa: E402
import filters as F                                                 # noqa: E402

H = F.H
RNG = np.random.default_rng(20260917)


def mde(sd, n, alpha=0.05, power=0.80):
    """Two-sided MDE of a mean at the given power (z 1.96 / 0.84)."""
    return (1.959964 + 0.841621) * sd / np.sqrt(n)


def main():
    data = Z.load_symbol('QQQ')
    UB, LB = Z.bands(data, 1.0)
    fx = F.extra_features(data, UB, LB)
    f1 = (fx['band_w'] >= 0.8).values
    f2 = (fx['gap_over_band'] >= 0.20).values
    base = F.day_df(data, F.simulate2(data, UB, LB))
    stack = F.day_df(data, F.simulate2(data, UB, LB, day_ok=f1 & f2))
    andveto = F.day_df(data, F.simulate2(data, UB, LB, day_ok=~((~f1) & (~f2))))

    print('== (a) power ==')
    for lab, (lo, hi) in F.SPLITS.items():
        if lab == 'TEST':
            continue
        x = base[(base['date'] >= lo) & (base['date'] <= hi)]
        td = x[x['ntr'] > 0]['bps']
        print(f'  {lab}: traded days={len(td)} sd={td.std(ddof=1):.1f} bps  '
              f'MDE@80% = {mde(td.std(ddof=1), len(td)):.2f} bps/traded day; '
              f'observed book mean {td.mean():.2f}')
        cal = x['bps']
        print(f'        calendar days={len(cal)} sd={cal.std(ddof=1):.1f} '
              f'MDE@80% = {mde(cal.std(ddof=1), len(cal)):.2f} bps/calendar day; '
              f'observed {cal.mean():.2f}')
        # power to see the FILTER effect: paired difference on the dropped set
        drop = td[x[x['ntr'] > 0]['date'].isin(set(fx['date'][~(f1 & f2)])).values]
        print(f'        vetoed bucket n={len(drop)} sd={drop.std(ddof=1):.1f} '
              f'MDE@80% = {mde(drop.std(ddof=1), len(drop)):.2f} bps; observed {drop.mean():+.2f}')

    print('\n== (b) numerator vs denominator ==')
    for lab, (lo, hi) in (('TRAIN', F.SPLITS['TRAIN']), ('VAL', F.SPLITS['VAL'])):
        b = base[(base['date'] >= lo) & (base['date'] <= hi)]
        s = stack[(stack['date'] >= lo) & (stack['date'] <= hi)]
        bt = b[b['ntr'] > 0]; st = s[s['ntr'] > 0]
        print(f'  {lab}: base sum {b["bps"].sum():+8.0f} over {len(bt)} traded days '
              f'({bt["bps"].mean():+.2f}/day)  ->  stack sum {s["bps"].sum():+8.0f} over '
              f'{len(st)} ({st["bps"].mean():+.2f}/day); dropped set sums '
              f'{b["bps"].sum()-s["bps"].sum():+.0f} bps over {len(bt)-len(st)} days '
              f'({(b["bps"].sum()-s["bps"].sum())/max(len(bt)-len(st),1):+.2f}/day)')
    # the three disjoint cells of the 2x2
    print('\n  2x2 of the two filters, TRAIN traded days (mean bps / n / sum):')
    lo, hi = F.SPLITS['TRAIN']
    bt = base[(base['date'] >= lo) & (base['date'] <= hi) & (base['ntr'] > 0)]
    dmap = pd.Series(f1, index=fx['date']).to_dict()
    gmap = pd.Series(f2, index=fx['date']).to_dict()
    bt = bt.copy()
    bt['f1'] = bt['date'].map(dmap); bt['f2'] = bt['date'].map(gmap)
    for lab2, (lo2, hi2) in (('TRAIN', F.SPLITS['TRAIN']), ('VAL', F.SPLITS['VAL'])):
        x = base[(base['date'] >= lo2) & (base['date'] <= hi2) & (base['ntr'] > 0)].copy()
        x['f1'] = x['date'].map(dmap); x['f2'] = x['date'].map(gmap)
        print(f'   -- {lab2} --')
        for a in (True, False):
            for c in (True, False):
                sub = x[(x['f1'] == a) & (x['f2'] == c)]
                print(f'     band_w>=0.8={a!s:5s} gap/band>=0.20={c!s:5s} '
                      f'n={len(sub):4d} mean={sub["bps"].mean():+7.2f} sum={sub["bps"].sum():+8.0f}')

    print('\n== (c) POST-HOC diagnostic (NOT adopted, not validated, 1 extra cell) ==')
    print('   AND-veto: skip the day only when band_w < 0.8 AND |gap|/band < 0.20')
    for lab, (lo, hi) in (('TRAIN', F.SPLITS['TRAIN']), ('TRAIN-H1', F.H1),
                          ('TRAIN-H2', F.H2), ('VAL', F.SPLITS['VAL'])):
        print('   ' + F.line(f'AND-veto {lab}', andveto, lo, hi))
    x = base[(base['date'] >= F.SPLITS['VAL'][0]) & (base['date'] <= F.SPLITS['VAL'][1])
             & (base['ntr'] > 0)]
    dd = x[x['date'].isin(set(fx['date'][(~f1) & (~f2)]))]
    print(f'   AND-veto dropped bucket on VAL: n={len(dd)} mean={dd["bps"].mean():+.2f}')

    print('\n== (d) permutation p for the frozen stack, TRAIN ==')
    lo, hi = F.SPLITS['TRAIN']
    x = base[(base['date'] >= lo) & (base['date'] <= hi) & (base['ntr'] > 0)].copy()
    keep = x['date'].isin(set(fx['date'][f1 & f2])).values
    obs = x['bps'][keep].mean() - x['bps'].mean()
    n_keep = keep.sum()
    null = np.array([RNG.permutation(x['bps'].values)[:n_keep].mean() - x['bps'].mean()
                     for _ in range(5000)])
    p = (np.abs(null) >= abs(obs)).mean()
    print(f'   observed mean lift of the kept set vs all traded days: {obs:+.2f} bps; '
          f'random-subset p = {p:.4f} (5,000 draws, same n)')
    print(f'   cells this stage looked at: 15 step-1 features + 28 step-2 grid cells + 4 step-2b '
          f'stacks + 1 post-hoc = 48;  Sidak-adjusted alpha 0.05 -> {1-(1-0.05)**(1/48):.4f}')

    print('\n== (e) final day-level book ==')
    out = base.copy()
    out = out.merge(fx, on='date', how='left')
    out['bps'] = out['r1x'] * 1e4
    out['f1_band_w_ge_0.8'] = out['date'].map(dmap)
    out['f2_gap_over_band_ge_0.20'] = out['date'].map(gmap)
    out['frozen_stack_keep'] = out['f1_band_w_ge_0.8'] & out['f2_gap_over_band_ge_0.20']
    sp = []
    for d in out['date']:
        lab = 'none'
        for k, (lo, hi) in F.SPLITS.items():
            if pd.Timestamp(lo) <= d <= pd.Timestamp(hi):
                lab = k
        sp.append(lab)
    out['split'] = sp
    out = out[['date', 'split', 'ntr', 'bps', 'r1x', 'band_w', 'gap', 'gap_over_band', 'rv20',
               'absmove5', 'trend5', 'above_ma20', 'above_ma5',
               'f1_band_w_ge_0.8', 'f2_gap_over_band_ge_0.20', 'frozen_stack_keep']]
    out.to_csv(H + 'final_book_days.csv', index=False)
    print(f'   wrote final_book_days.csv: {len(out)} rows '
          f'({(out["split"]=="TRAIN").sum()} TRAIN / {(out["split"]=="VAL").sum()} VAL / '
          f'{(out["split"]=="TEST").sum()} TEST)')


if __name__ == '__main__':
    main()
