#!/usr/bin/env python3
"""Steps 2-4 — live-convention fills, leverage, tails.

Scenarios (all on the paper's rule; only the EXECUTION convention changes):
  A  paper faithful      fill = check-bar close, flat at 15:59 close,  cost $0.0045/share/leg
  B  live fill, paper $  fill = next bar's open, flat at 15:59 close,  cost $0.0045/share/leg
  C  live fill, 0.5 bp   fill = next bar's open +/- 0.5 bp, flat at 15:59 close (MOC proxy)
  D  live fill, 1.0 bp   fill = next bar's open +/- 1.0 bp, flat at 15:59 close (MOC proxy)
  E  C, but flat at the OPEN of the 15:59 bar
  F  D, but flat at the OPEN of the 15:59 bar

Outputs (Q/):
  step2_scenarios.csv   one row per symbol x scenario x period
  step2_fillgap.csv     obtainability / fill-difference diagnostics
  step2_attribution.csv P&L share by clock window
  step3_leverage.csv    QQQ 1x / QQQ 3x notional / TQQQ 1x, $ at 60K equity
  step3_months.csv      monthly $ P&L at $60K, OOS
  step4_years.csv       OOS by year
  step4_tail.csv        top-1%/5% day removal
  step4_vm.csv          band multiplier 0.8 / 1.0 / 1.2
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/fuckup_audit/Q')
import zsim as Z

Q = '/home/ec2-user/onemil/research/fuckup_audit/Q/'
EQUITY = 60_000.0

SCEN = [  # name, fill, eod, slip_bp, cost
    ('A paper faithful',        'close',     'moc',  0.0, 'paper'),
    ('B live fill, paper $',    'next_open', 'moc',  0.0, 'paper'),
    ('C live fill, 0.5bp/leg',  'next_open', 'moc',  0.5, 'gross'),
    ('D live fill, 1.0bp/leg',  'next_open', 'moc',  1.0, 'gross'),
    ('E C + 15:59-open flat',   'next_open', 'open', 0.5, 'gross'),
    ('F D + 15:59-open flat',   'next_open', 'open', 1.0, 'gross'),
]


def mrow(sym, scen, label, df, extra=None):
    m1 = Z.metrics(df, 'r1x'); md = Z.metrics(df, 'rdyn')
    r = dict(symbol=sym, scenario=scen, period=label, days=m1['days'], traded=m1['traded'],
             trades_day=m1['trades_day'], bps_1x=m1['bps'], t_1x=m1['t'], hit=m1['hit'],
             ann_1x=m1['ann'], sr_1x=m1['sharpe'], mdd_1x=m1['mdd'],
             bps_dyn=md['bps'], t_dyn=md['t'], ann_dyn=md['ann'], sr_dyn=md['sharpe'], mdd_dyn=md['mdd'])
    if extra:
        r.update(extra)
    return r


def periods(df):
    is_, oos = Z.split(df)
    return [('IS 2016-2023', is_), ('OOS 2024-2026', oos)]


def fill_gap_diag(data, UB, LB):
    """How often does the live fill differ from the paper fill, and by how much (signed, adverse +)."""
    C, O, last = data['C'], data['O'], data['last']
    rows = []
    n_fallback = 0
    tr_close, _ = Z.simulate(data, UB, LB, fill='close', eod='moc', slip_bp=0.0)
    tr_next, _ = Z.simulate(data, UB, LB, fill='next_open', eod='moc', slip_bp=0.0)
    # signal-level diagnostics: for every check at which a position CHANGED in the close-fill run
    for t in tr_close:
        for k, side in ((t['ek'], t['side']), (t['xk'], -t['side'])):
            if k < 0 or k >= last[t['d']]:
                continue
            c = C[t['d'], k]
            o = O[t['d'], k + 1] if k + 1 < Z.N_MIN else np.nan
            if np.isnan(o):
                n_fallback += 1
                continue
            adverse_bp = side * (o - c) / c * 1e4   # buying: paying more than the close = adverse
            rows.append(dict(d=t['d'], k=k, side=side, close=c, next_open=o, adverse_bp=adverse_bp,
                             differs=abs(o - c) > 1e-9))
    g = pd.DataFrame(rows)
    return g, n_fallback, len(tr_close), len(tr_next)


def attribution(data, trades, posm, days_mask):
    """Share of gross P&L (in return terms) by clock window."""
    m = Z.pnl_by_minute(data, trades, posm) / data['dopen'][:, None]
    m = m[days_mask]
    tot = m.sum()
    out = {}
    for name, lo, hi in (('09:30-10:00', 0, 30), ('10:00-10:30', 30, 60), ('10:30-15:30', 60, 360),
                         ('15:30-16:00', 360, 390)):
        out[name] = m[:, lo:hi].sum()
    return tot, out


def run_symbol(sym, vm=1.0):
    data = Z.load_symbol(sym)
    UB, LB = Z.bands(data, vm)
    out = {}
    for name, fill, eod, slip, cost in SCEN:
        tr, posm = Z.simulate(data, UB, LB, fill=fill, eod=eod, slip_bp=slip)
        df = Z.daily_returns(data, tr, cost)
        out[name] = (tr, posm, df)
    return data, UB, LB, out


def main():
    scen_rows, gap_rows, attr_rows = [], [], []
    lev_rows, month_rows, year_rows, tail_rows, vm_rows = [], [], [], [], []
    daily_store = {}

    for sym in ('SPY', 'QQQ', 'TQQQ'):
        data, UB, LB, out = run_symbol(sym)
        for name, _, _, _, _ in SCEN:
            tr, posm, df = out[name]
            for lab, part in periods(df):
                scen_rows.append(mrow(sym, name, lab, part))
        # obtainability / fill gap
        g, nfb, n_close, n_next = fill_gap_diag(data, UB, LB)
        d_is = data['days'] <= Z.IS_END
        for lab, mask in (('IS 2016-2023', g['d'].map(lambda i: bool(d_is[i]))),
                          ('OOS 2024-2026', g['d'].map(lambda i: not bool(d_is[i])))):
            gg = g[mask]
            gap_rows.append(dict(symbol=sym, period=lab, legs=len(gg),
                                 pct_fill_differs=100.0 * gg['differs'].mean(),
                                 mean_adverse_bp=gg['adverse_bp'].mean(),
                                 median_adverse_bp=gg['adverse_bp'].median(),
                                 p90_adverse_bp=gg['adverse_bp'].quantile(0.9),
                                 n_next_open_missing=nfb,
                                 trades_close_fill=n_close, trades_next_open_fill=n_next))
        # attribution, on the live-convention scenario C
        tr, posm, df = out['C live fill, 0.5bp/leg']
        for lab, mask_days in (('IS 2016-2023', np.asarray(data['days'] <= Z.IS_END)),
                               ('OOS 2024-2026', np.asarray(data['days'] >= Z.OOS_START))):
            tot, buckets = attribution(data, tr, posm, mask_days)
            row = dict(symbol=sym, period=lab, total_return_units=tot)
            for kk, vv in buckets.items():
                row[f'share_{kk}'] = 100.0 * vv / tot if tot else np.nan
                row[f'ret_{kk}'] = vv
            attr_rows.append(row)
        daily_store[sym] = {n: out[n][2].copy() for n, *_ in SCEN}

        # step 4 on QQQ only (the sleeve under audit), scenario C
        if sym == 'QQQ':
            dfc = out['C live fill, 0.5bp/leg'][2]
            oos = dfc[dfc['date'] >= Z.OOS_START]
            for y, gy in oos.groupby(oos['date'].dt.year):
                year_rows.append(mrow(sym, 'C live fill, 0.5bp/leg', str(y), gy))
            # tail removal on OOS
            for lab, frac in (('full', 0.0), ('top 1% days removed', 0.01), ('top 5% days removed', 0.05)):
                o2 = oos.copy()
                if frac > 0:
                    cut = o2['r1x'].quantile(1 - frac)
                    o2 = o2[o2['r1x'] < cut]
                tail_rows.append(mrow(sym, 'C live fill, 0.5bp/leg', lab, o2))
                # and the same for IS
            for lab, frac in (('full', 0.0), ('top 1% days removed', 0.01), ('top 5% days removed', 0.05)):
                i2 = dfc[dfc['date'] <= Z.IS_END].copy()
                if frac > 0:
                    cut = i2['r1x'].quantile(1 - frac)
                    i2 = i2[i2['r1x'] < cut]
                tail_rows.append(mrow(sym, 'C live fill, 0.5bp/leg IS', lab, i2))
            # band multiplier sensitivity, live convention
            for v in (0.8, 1.0, 1.2):
                U2, L2 = Z.bands(data, v)
                t2, _ = Z.simulate(data, U2, L2, fill='next_open', eod='moc', slip_bp=0.5)
                d2 = Z.daily_returns(data, t2, 'gross')
                for lab, part in periods(d2):
                    vm_rows.append(mrow(sym, f'VM={v}', lab, part))
        del data, UB, LB, out

    # ---- step 3: leverage, $ at 60K -------------------------------------------------
    qqq = daily_store['QQQ']['C live fill, 0.5bp/leg'][['date', 'r1x', 'ntr']].copy()
    tqqq05 = daily_store['TQQQ']['C live fill, 0.5bp/leg'][['date', 'r1x', 'ntr']].copy()
    tqqq10 = daily_store['TQQQ']['D live fill, 1.0bp/leg'][['date', 'r1x', 'ntr']].copy()
    books = {
        'QQQ 1x notional': (qqq, 1.0),
        'QQQ 3x notional (intraday, no interest)': (qqq, 3.0),
        'TQQQ 1x notional (0.5bp/leg)': (tqqq05, 1.0),
        'TQQQ 1x notional (1.0bp/leg)': (tqqq10, 1.0),
    }
    for name, (src, mult) in books.items():
        b = pd.DataFrame({'date': src['date'].values, 'r1x': mult * src['r1x'].values,
                          'ntr': src['ntr'].values})
        b['rdyn'] = b['r1x']
        for lab, part in periods(b):
            lev_rows.append(mrow(name, 'C live fill, 0.5bp/leg', lab, part))
        oos = b[b['date'] >= Z.OOS_START].copy()
        oos['ym'] = oos['date'].dt.to_period('M').astype(str)
        gm = oos.groupby('ym').agg(days=('r1x', 'size'), ret=('r1x', 'sum'), ntr=('ntr', 'sum'))
        gm['usd'] = gm['ret'] * EQUITY
        for ym, r in gm.iterrows():
            month_rows.append(dict(book=name, month=ym, days=int(r['days']), trades=int(r['ntr']),
                                   ret_pct=100 * r['ret'], usd=r['usd']))

    pd.DataFrame(scen_rows).to_csv(Q + 'step2_scenarios.csv', index=False)
    pd.DataFrame(gap_rows).to_csv(Q + 'step2_fillgap.csv', index=False)
    pd.DataFrame(attr_rows).to_csv(Q + 'step2_attribution.csv', index=False)
    pd.DataFrame(lev_rows).to_csv(Q + 'step3_leverage.csv', index=False)
    pd.DataFrame(month_rows).to_csv(Q + 'step3_months.csv', index=False)
    pd.DataFrame(year_rows).to_csv(Q + 'step4_years.csv', index=False)
    pd.DataFrame(tail_rows).to_csv(Q + 'step4_tail.csv', index=False)
    pd.DataFrame(vm_rows).to_csv(Q + 'step4_vm.csv', index=False)
    print('done')


if __name__ == '__main__':
    main()
