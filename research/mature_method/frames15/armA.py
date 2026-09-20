#!/usr/bin/env python3
"""frames15 ARM A — the volume fields as a NAME-DAY SELECTOR on HOD-break's B2 population.

Cells A1..A9 exactly as declared in PREREG §2. For each: the availability audit, the
kept-minus-rejected gross R on the BOOKED trades with a day-clustered two-sample t, the two TRAIN
halves and VAL, the RE-BOOKED book's owner metrics with the count-matched permutation null, the
wrapper-enrichment check and the `hrv_raw` sensitivity (diagnostic D4).
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/mature_method/frames15')
from common15 import (D, S, SPLITS, book_ranked,                           # noqa: E402
                      attach_instrument, daily_for_keys, hourly_for_keys, last_closed_hour,
                      clust_t2, clust_t1, mde_pct, null_green)

REF = {'TRAIN': (1622, -0.039, -17346.0), 'VAL': (706, 0.083, 893.0)}


def attach(s):
    """Join every declared field at its decision time. Returns s with the field columns."""
    dcols = ['symbol', 'date', 'p_interest5', 'p_rvd', 'p_ret1', 'p_clspos', 'p_weak', 'p_ret5',
             'volume', 'adv20', 'adv20d', 'ca_bad']
    want_d = set(zip(s.symbol.astype(str), s.day.astype(str)))
    d = daily_for_keys(want_d, dcols).rename(columns={'date': 'day'})
    s = s.merge(d, on=['symbol', 'day'], how='left', suffixes=('', '_d'))
    del d

    s['hh'] = last_closed_hour(s.break_m.values)
    want_h = set(zip(s.symbol.astype(str), s.day.astype(str),
                     pd.Series(s.hh).fillna(-1).astype(int)))
    hcols = ['symbol', 'day', 'hour', 'hrv', 'hrv_raw', 'hour_ret', 'sus2_2', 'sus2_3',
             'sus3_2', 'sus3_3', 'day_v']
    h = hourly_for_keys(want_h, hcols).rename(columns={'hour': 'hh'})
    s = s.merge(h, on=['symbol', 'day', 'hh'], how='left')
    del h

    # data rail: the SIP tape must carry >= 50 % of the panel's daily volume for that symbol-day
    s['tape_ok'] = (s.day_v / s.volume.replace(0, np.nan)) >= 0.5
    s = attach_instrument(s)
    return s


CELLS = [
    ('A1', 'V1 hrv >= 2 at the signal hour', lambda s: s.hrv >= 2.0, 'h'),
    ('A2', 'V1 hrv >= 3 at the signal hour', lambda s: s.hrv >= 3.0, 'h'),
    ('A3', 'V2 sustained hrv>=2 over 2 hours', lambda s: s.sus2_2.fillna(False), 'h'),
    ('A4', 'V2 sustained hrv>=2 over 3 hours', lambda s: s.sus2_3.fillna(False), 'h'),
    ('A5', 'V3 interest5 >= 3 (prior close)', lambda s: s.p_interest5 >= 3, 'd'),
    ('A6', 'V3 rvd_1 >= 1.5 (prior session)', lambda s: s.p_rvd >= 1.5, 'd'),
    ('A7', 'V4 absorption hrv>=2 & |hour ret| <= 1%', lambda s: (s.hrv >= 2.0) &
     (s.hour_ret.abs() <= 0.01), 'h'),
    ('A8', 'V5 accumulation on weakness', lambda s: s.p_weak.fillna(False), 'd'),
    ('A9', 'V6 sus(2,2) OR interest5 >= 3', lambda s: s.sus2_2.fillna(False) |
     (s.p_interest5 >= 3), 'h'),
]


def main():
    s = pd.read_csv(f'{D}/sig15.csv', dtype={'symbol': str, 'day': str},
                    keep_default_na=False, na_values=[''])
    print(f'== ARM A — {len(s)} pre-book signals, {int(s.booked.sum())} booked '
          f'(B2 gate asserted in stage1.py) ==', flush=True)
    s = attach(s)
    b = s[s.booked.astype(bool)].copy()
    print(f'  booked rows re-joined: {len(b)} (expect 2328)', flush=True)

    # ------------------------------------------------------------------ availability audit
    print('\n== availability audit (on the BOOKED book) ==', flush=True)
    rows = []
    for col in ('hrv', 'hrv_raw', 'hour_ret', 'p_interest5', 'p_rvd', 'p_weak', 'tape_ok'):
        cov = b[col].notna().mean() * 100
        win = b[b.rr > 0][col].isna().mean() * 100
        los = b[b.rr <= 0][col].isna().mean() * 100
        flag = 'VOID' if (cov < 80 or abs(win - los) > 5) else 'ok'
        rows.append(dict(field=col, cov=cov, miss_win=win, miss_los=los, verdict=flag))
        print(f'  {col:14s} coverage {cov:5.1f}%  miss winners {win:5.1f}%  losers {los:5.1f}%  '
              f'{flag}', flush=True)
    pd.DataFrame(rows).to_csv(f'{D}/availA.csv', index=False)

    # hourly fields are undefined before 10:00 -> the rail-forced restricted base
    late = b.break_m >= 600
    print(f'  signals with a CLOSED hour (break_m >= 600): {late.mean()*100:.1f}% '
          f'(winners {b[b.rr>0].break_m.ge(600).mean()*100:.1f}%, '
          f'losers {b[b.rr<=0].break_m.ge(600).mean()*100:.1f}%)', flush=True)

    # ------------------------------------------------------------------ base rates
    print('\n== base rates / fire shares on the BOOKED book ==', flush=True)
    for cid, name, fn, kind in CELLS:
        m = fn(b).fillna(False) if hasattr(fn(b), 'fillna') else fn(b)
        base = b[late] if kind == 'h' else b
        mm = m[base.index] if kind == 'h' else m
        print(f'  {cid} {name:42s} fires on {mm.mean()*100:5.1f}% of the base '
              f'(n={int(mm.sum())})', flush=True)

    # ------------------------------------------------------------------ the cells
    out = []
    print('\n== A cells ==', flush=True)
    for cid, name, fn, kind in CELLS:
        base_b = b[b.break_m >= 600] if kind == 'h' else b
        base_s = s[s.break_m >= 600] if kind == 'h' else s
        keep_b = fn(base_b).fillna(False).values
        for sp in SPLITS:
            d = base_b[base_b.split == sp]
            k = fn(d).fillna(False).values
            delta, t = clust_t2(d.rr.values, k, d.day.values)
            kr = d.rr.values[k]
            rr_k = float(np.nanmean(kr)) if k.sum() else np.nan
            rr_r = float(np.nanmean(d.rr.values[~k])) if (~k).sum() else np.nan
            # halves on TRAIN
            if sp == 'TRAIN':
                h1 = d[(d.day < '2025-07-01')]
                h2 = d[(d.day >= '2025-07-01')]
                hk = [float(x.rr[fn(x).fillna(False).values].mean()) if fn(x).fillna(False).sum()
                      else np.nan for x in (h1, h2)]
            wrap = float((d.asset_class[k] == 'wrapper').mean()) if k.sum() else np.nan
            wrap_base = float((d.asset_class == 'wrapper').mean())
            # hrv_raw sensitivity (D4)
            if kind == 'h' and 'hrv' in name:
                kr2 = (d.hrv_raw >= (2.0 if '2' in name else 3.0)).fillna(False).values
                _, t_raw = clust_t2(d.rr.values, kr2, d.day.values)
                d_raw = float(np.nanmean(d.rr.values[kr2]) - np.nanmean(d.rr.values[~kr2])) \
                    if kr2.sum() and (~kr2).sum() else np.nan
            else:
                d_raw, t_raw = np.nan, np.nan
            # the re-booked book
            sm = fn(base_s).fillna(False).values
            rb = book_ranked(base_s[sm], 12, 4)
            wb = S.week_stats(rb, sp) if len(rb) else {}
            nl = null_green(rb.assign(rr=rb.rr), sp, 'rr') if len(rb) > 5 else (np.nan,) * 3
            out.append(dict(cell=cid, name=name, split=sp, n_keep=int(k.sum()),
                            n_rej=int((~k).sum()), rr_keep=rr_k, rr_rej=rr_r, delta=delta,
                            clust_t=t, mde=mde_pct(d.rr.values, d.day.values),
                            wrap_keep=wrap, wrap_base=wrap_base, d_raw=d_raw, t_raw=t_raw,
                            bk_n=wb.get('n'), bk_wk=wb.get('per_wk'), bk_gross=wb.get('gross'),
                            bk_net=wb.get('net'), bk_green=wb.get('green'),
                            bk_total=wb.get('total'), bk_worst=wb.get('worst'),
                            bk_streak=wb.get('redstreak'), null_p95=nl[2],
                            h1=hk[0] if sp == 'TRAIN' else np.nan,
                            h2=hk[1] if sp == 'TRAIN' else np.nan))
            print(f'  {cid} {sp:5s} keep n={int(k.sum()):4d} R={rr_k:+.3f}  rej n={int((~k).sum()):4d} '
                  f'R={rr_r:+.3f}  delta={delta:+.3f} t={t:+.2f}  book {wb.get("n",0):4d} '
                  f'({wb.get("per_wk",0):4.1f}/wk) green={wb.get("green",np.nan):5.1f}% '
                  f'${wb.get("total",np.nan):+,.0f}', flush=True)
    pd.DataFrame(out).to_csv(f'{D}/cellsA.csv', index=False)
    print(f'\n[armA] wrote cellsA.csv ({len(out)} rows)', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
