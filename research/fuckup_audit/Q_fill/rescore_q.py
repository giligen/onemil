#!/usr/bin/env python3
"""Stage Q step 3 — re-score the honest ORB book under three FILL models.

All arms run the SAME pipeline path that produced D1's `book_n8_q1on.csv`
(`study_orb_pipeline_static_lock.py` + `ORB_BT_RESIM_CACHE`), so the picks, the
ranking, the vetoes and the slot mechanics are identical in every arm — only the
dollars move.  `asis` is the parity control and must reproduce D1 to the cent.

Arms
----
asis        every elected stop-limit fills at the cap (range_high x 1.003).
            = D1 / `candidates_dump.csv`.
strict      the ask at the trigger instant is ABOVE the cap -> the order is not
            marketable -> NO fill ever; the pick books $0 and still spends its
            slot (the live `time_stop_canceled` outcome).  Stage P's flag.
measured    the MEASURED quote path (Q_fill/walk_rows.csv): the elected order
            rests as a bid at the cap; it fills at the cap the first time the
            NBBO ask reaches it before the 10:35 time stop, and the trade is
            re-simulated from that later bar (`delayed_resim.csv`, no touchgo --
            see resim_delayed.py).  If the ask never returns, $0.
measured_tg the same, touchgo re-keyed to the fill bar (sensitivity band).
meas_cost   `measured` PLUS Stage P's measured prices on the rows that WERE
            marketable: entry at min(NBBO ask, cap), exit at the NBBO bid at
            the exit instant, the resting +3R scale leg free.

Usage: python3 rescore_q.py build | run | analyse
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from trading.orb_csv import read_orb_csv          # noqa: E402

Q = f'{ROOT}/research/fuckup_audit/Q_fill'
PC = f'{ROOT}/research/fuckup_audit/P_cost'
D1 = f'{ROOT}/research/fuckup_audit/D1_orb'
SCALE_FRAC = 0.40
SCALE_LEVEL_R = 3.0
ARMS = ('asis', 'strict', 'measured_live', 'measured', 'measured_tg', 'meas_cost')


def _old_pos() -> float:
    import yaml
    return float((yaml.safe_load(open(f'{ROOT}/orb.yaml')).get('sizing') or {})
                 .get('old_position_reference_usd', 50_000.0))


OLD_POS = _old_pos()


def build():
    dump = read_orb_csv(f'{D1}/candidates_dump.csv')
    dump['date'] = pd.to_datetime(dump['date']).dt.strftime('%Y-%m-%d')
    dump['key'] = dump.symbol + '|' + dump.date
    shares = np.maximum(1, (OLD_POS / dump.entry_price).astype(int))
    chk = np.abs(dump.pnl - dump.pnl_pct / 100 * dump.entry_price * shares)
    assert float(chk.max()) < 1e-6, f'share base wrong: {chk.max()}'

    sp = pd.read_parquet(f'{PC}/spreads.parquet')
    sp['key'] = sp.symbol + '|' + sp.date
    s = sp.set_index('key')
    walk = pd.read_csv(f'{Q}/walk_rows.csv', keep_default_na=False, na_values=[''],
                       dtype={'symbol': str, 'date': str})
    walk['key'] = walk.symbol + '|' + walk.date
    wf = dict(zip(walk.key, walk.filled_later))
    res = pd.read_csv(f'{Q}/delayed_resim.csv', keep_default_na=False, na_values=[''],
                      dtype={'symbol': str, 'date': str})
    res['key'] = res.symbol + '|' + res.date
    r_new = dict(zip(res.key, res.new_pnl))
    r_tg = dict(zip(res.key, res.tg_pnl))

    asis = dump.pnl.astype(float).values
    strict = asis.copy()
    meas_live = asis.copy()
    meas = asis.copy()
    meas_tg = asis.copy()
    mcost = asis.copy()
    n_flag = n_late = n_never = n_cost = n_live_market = 0

    for i, r in enumerate(dump.itertuples()):
        if r.entered != 1 or r.key not in s.index:
            continue
        q = s.loc[r.key]
        if not q.cov_entry:
            continue
        cap = float(r.entry_price)
        ask = float(q.measured_entry)
        if not np.isfinite(ask):
            continue
        sh = int(shares.iat[i])
        cap_live = round(cap, 2)                          # the price the engine SENDS
        if ask > cap * (1 + 1e-12):                       # NOT marketable (raw cap)
            n_flag += 1
            strict[i] = 0.0
            live_flag = ask > cap_live + 1e-9
            if wf.get(r.key, 0) == 1:
                n_late += 1
                meas[i] = float(r_new[r.key])
                meas_tg[i] = float(r_tg[r.key])
                mcost[i] = float(r_new[r.key])
                meas_live[i] = float(r_new[r.key]) if live_flag else asis[i]
            else:
                n_never += 1
                meas[i] = meas_tg[i] = mcost[i] = 0.0
                meas_live[i] = 0.0 if live_flag else asis[i]
            if not live_flag:
                n_live_market += 1
        else:                                             # marketable: P's prices
            if q.cov_exit and np.isfinite(float(q.measured_exit)):
                n_cost += 1
                e_m, x_m = ask, float(q.measured_exit)
                if str(r.exit_reason).startswith('scale_'):
                    rng = float(q.range_high) - float(q.range_low)
                    spx = e_m + SCALE_LEVEL_R * rng       # resting limit, no spread
                    qty = int(np.floor(SCALE_FRAC * sh))
                    frac = qty / float(sh) if qty >= 1 else 0.0
                    ret = frac * (spx / e_m - 1) + (1 - frac) * (x_m / e_m - 1)
                    mcost[i] = ret * e_m * sh
                else:
                    mcost[i] = (x_m - e_m) * sh

    for arm, pnl in (('asis', asis), ('strict', strict),
                     ('measured_live', meas_live), ('measured', meas),
                     ('measured_tg', meas_tg), ('meas_cost', mcost)):
        out = read_orb_csv(f'{D1}/candidates_dump.csv')
        out['pnl'] = pnl
        out['pnl_pct'] = pnl / (out.entry_price * shares.values) * 100
        out.loc[out.entered != 1, ['pnl', 'pnl_pct']] = 0.0
        out.to_csv(f'{Q}/dump_{arm}.csv', index=False)
    print(f'flagged(raw cap) {n_flag} | filled later {n_late} | never {n_never} | '
          f'marketable at the LIVE (2dp) cap {n_live_market} | '
          f'marketable rows priced at measured {n_cost}', flush=True)
    pd.DataFrame(dict(key=dump.key, entered=dump.entered, pnl_asis=asis,
                      pnl_strict=strict, pnl_measured_live=meas_live,
                      pnl_measured=meas,
                      pnl_measured_tg=meas_tg, pnl_meas_cost=mcost)
                 ).to_csv(f'{Q}/per_trade_arms.csv', index=False)


def run():
    env0 = dict(os.environ)
    env0['ORB_BT_FEATURES_CSV'] = 'analysis_results/orb_features_20260916_2053.csv'
    env0['ORB_BT_RISK'] = '375'
    for arm in ARMS:
        for n in (8, 3):
            e = dict(env0)
            e['ORB_BT_RESIM_CACHE'] = f'{Q}/dump_{arm}.csv'
            e['ORB_BT_N'] = str(n)
            e['ORB_BT_ACCOUNT'] = repr(3333.333333333333 * n)
            e['ORB_SKIP_Q1'] = '1'
            e['ORB_BT_BOOK_OUT'] = f'{Q}/book_{arm}_n{n}.csv'
            e['ORB_BT_MONTHLY_OUT'] = f'{Q}/monthly_{arm}_n{n}.csv'
            with open(f'{Q}/log_{arm}_n{n}.txt', 'w') as fh:
                rc = subprocess.call(['nice', '-n', '10', 'python3', '-u',
                                      'study_orb_pipeline_static_lock.py'],
                                     stdout=fh, stderr=subprocess.STDOUT, env=e)
            print(f'{arm} n={n} rc={rc}', flush=True)


def split_of(day):
    return 'TRAIN' if day < '2026-01-01' else ('VAL' if day < '2026-06-01' else 'TEST')


def _mdd(daily):
    cum = daily.cumsum()
    return float((cum - cum.cummax()).min())


_RNG = None


def range_lookup():
    global _RNG
    if _RNG is None:
        x = pd.read_csv(f'{PC}/exit_times.csv', keep_default_na=False, na_values=[''],
                        dtype={'symbol': str, 'date': str},
                        usecols=['symbol', 'date', 'range_high', 'range_low'])
        _RNG = dict(zip(zip(x.symbol, x.date), x.range_high - x.range_low))
    return _RNG


def book_stats(path):
    b = read_orb_csv(path)
    b['date'] = pd.to_datetime(b['date']).dt.strftime('%Y-%m-%d')
    b['split'] = b.date.map(split_of)
    b['month'] = b.date.str[:7]
    rl = range_lookup()
    rng = np.array([rl.get((s, dt), np.nan) for s, dt in zip(b.symbol, b.date)])
    shares = b['_rp_position'] / b['entry_price']
    b['R'] = np.where(rng > 0, b['_sized_pnl'] / (shares * rng), 0.0)
    out = {}
    for label, g in [('ALL', b)] + list(b.groupby('split')):
        mo = g.groupby('month')._sized_pnl.sum()
        out[label] = dict(picks=len(g), fills=int((g.entered.astype(float) != 0).sum()),
                          pnl=round(float(g._sized_pnl.sum()), 0),
                          r_per_pick=round(float(g.R.mean()), 3),
                          worst_mo=round(float(mo.min()), 0) if len(mo) else np.nan,
                          red_mo=int((mo < 0).sum()),
                          mdd=round(_mdd(g.groupby('date')._sized_pnl.sum()), 0))
    return out


def analyse():
    rows = []
    for arm in ARMS:
        for n in (8, 3):
            p = f'{Q}/book_{arm}_n{n}.csv'
            if not os.path.exists(p):
                continue
            for k, v in book_stats(p).items():
                rows.append(dict(arm=arm, slots=n, split=k, **v))
    t = pd.DataFrame(rows)
    t.to_csv(f'{Q}/rescore_table.csv', index=False)
    print(t.to_string(index=False), flush=True)


if __name__ == '__main__':
    {'build': build, 'run': run, 'analyse': analyse}[sys.argv[1]]()
