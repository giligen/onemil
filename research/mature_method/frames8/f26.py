#!/usr/bin/env python3
"""frames8 / F26 — THE ORB MINUTE PRICED AS A LIVE DECISION.

Ladders `orb.yaml::entry.stop_limit_buffer_bps` over 30 / 50 / 100 / 150 bps on Stage Q's already
walked NBBO order lives and scores ONLY THE CONVERTED SUBSET at each rung — the orders that fail at
the tighter cap and fill at the wider one — with the chase-guard-vs-dip-buy discriminator.

No API call, no Databento spend, `orb.yaml` is NEVER opened for writing.  Inputs, all read-only:
  research/fuckup_audit/P_cost/spreads.parquet   the measured NBBO ask at each trigger instant
  research/fuckup_audit/P_cost/exit_times.csv    range_high/low, atr14, the breakout bar ts
  research/fuckup_audit/Q_fill/per_trade_arms.csv  the MEASURED treatment at the 30-bps cap
  data/cache.db                                  1-minute bars (ro)
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/frames7')
sys.path.insert(0, f'{ROOT}/research/fuckup_audit/P_cost')
import c7                                                              # noqa: E402
import study_orb_pipeline_static_lock as P                             # noqa: E402
from study_orb import _bars_to_df                                      # noqa: E402
from exit_times import instrumented_walk                               # noqa: E402

D8 = f'{ROOT}/research/mature_method/frames8'
PC = f'{ROOT}/research/fuckup_audit/P_cost'
Q = f'{ROOT}/research/fuckup_audit/Q_fill'
SQL = ("SELECT timestamp, open, high, low, close, volume FROM intraday_bars_1min "
       "WHERE symbol=? AND bar_date=? ORDER BY timestamp")
OLD_POS = 50_000.0
RUNGS = (30, 50, 100, 150)


def cap_px(rh, bps):
    """The cap the engine SENDS: range_high x (1 + bps/1e4), rounded to the penny."""
    return np.round(rh * (1.0 + bps / 10000.0), 2)


def main():
    cfg = P.load_bt_config()
    P.MIN_STOP_PCT = cfg['min_stop_pct']; P.OLD_POS = cfg['old_pos_ref']
    P.LOCK_TRIGGER_R = cfg['lock_arm_r']; P.LOCK_STOP_R = cfg['lock_stop_r']
    P.EXIT_SLIP_BPS = cfg['exit_slip_bps']; P.FORCE_CLOSE_ET = cfg['force_close_et']

    sp = pd.read_parquet(f'{PC}/spreads.parquet')
    sp['key'] = sp.symbol + '|' + sp.date
    xt = pd.read_csv(f'{PC}/exit_times.csv', keep_default_na=False, na_values=[''],
                     dtype={'symbol': str, 'date': str},
                     usecols=['symbol', 'date', 'range_high', 'range_low', 'atr14', 'entry_ts'])
    xt['key'] = xt.symbol + '|' + xt.date
    X = xt.drop_duplicates('key').set_index('key')
    arms = pd.read_csv(f'{Q}/per_trade_arms.csv', keep_default_na=False, na_values=[''])
    meas = dict(zip(arms.key, arms.pnl_measured))
    asis = dict(zip(arms.key, arms.pnl_asis))

    d = sp[sp.cov_entry & sp.entry_ask.notna() & sp.key.isin(X.index)].copy()
    d['split'] = [c7.split_of(x) for x in d.date]
    d['half'] = np.where(d.date < '2025-07-01', 'H1', 'H2')
    d['rh'] = X.loc[d.key].range_high.values
    d['rl'] = X.loc[d.key].range_low.values
    d['atr'] = X.loc[d.key].atr14.values
    d['brk'] = X.loc[d.key].entry_ts.values
    d['rsize'] = d.rh - d.rl
    d = d[d.rsize > 0]
    d['shares'] = np.maximum(1, (OLD_POS / d.entry_price).astype(int))
    for b in RUNGS:
        d[f'cap{b}'] = cap_px(d.rh.values, b)
    print(f'honest ORB fills with a measured entry quote: {len(d)}  '
          f'(TEST sealed: {int((d.split == "TEST").sum())} rows dropped)', flush=True)
    d = d[d.split.isin(('TRAIN', 'VAL'))].copy()

    # ---------------------------------------------------------- who is already marketable at 30
    d['mkt30'] = d.entry_ask <= d.cap30 + 1e-9
    assert bool((d.loc[d.mkt30, 'entry_ask'] <= d.loc[d.mkt30, 'cap150'] + 1e-9).all()), \
        'a wider cap must never change an already-marketable fill'
    print(f'already marketable at 30 bps: {int(d.mkt30.sum())} '
          f'({d.mkt30.mean():.1%}) — a wider cap CANNOT touch these (asserted)', flush=True)

    con = sqlite3.connect('file:data/cache.db?mode=ro', uri=True, timeout=120)
    con.row_factory = sqlite3.Row
    rows = []
    for i, r in enumerate(d[~d.mkt30].itertuples()):
        rung = None
        for b in RUNGS[1:]:
            if r.entry_ask <= getattr(r, f'cap{b}') + 1e-9:
                rung = b
                break
        a = float(r.entry_ask)
        bars = _bars_to_df([dict(x) for x in con.execute(SQL, (r.symbol, r.date))])
        brk = pd.Timestamp(r.brk).tz_convert('UTC')
        a14 = float(r.atr) if np.isfinite(r.atr) else None
        px, rsn, _, _, _ = instrumented_walk(bars, a, float(r.rh), float(r.rl), brk,
                                             int(r.shares), a14, cfg)
        rr_conv = np.nan if rsn == 'no_bars' else (px - a) / r.rsize
        mp = meas.get(r.key, np.nan)
        ap = asis.get(r.key, np.nan)
        rows.append(dict(key=r.key, symbol=r.symbol, date=r.date, split=r.split, half=r.half,
                         rung=rung if rung is not None else 9999,
                         ask=a, cap30=r.cap30, rsize=r.rsize, shares=int(r.shares),
                         over_bps=(a / r.rh - 1.0) * 1e4,
                         rr_conv=rr_conv, why=rsn,
                         rr_meas=mp / (r.shares * r.rsize) if np.isfinite(mp) else np.nan,
                         rr_asis=ap / (r.shares * r.rsize) if np.isfinite(ap) else np.nan))
        if i % 100 == 0:
            print(f'  [{i}] {r.symbol} {r.date} ask {a:.2f} cap30 {r.cap30:.2f} rung {rung}',
                  flush=True)
    con.close()
    C = pd.DataFrame(rows)
    C.to_csv(f'{D8}/f26_converted.csv', index=False)

    # -------------------------------------------------------------------- the already-marketable R
    mk = d[d.mkt30].copy()
    mk['rr_asis'] = [asis.get(k, np.nan) for k in mk.key]
    mk['rr_asis'] = mk.rr_asis / (mk.shares * mk.rsize)

    print('\n=== F26 — the CONVERTED subset, per rung (TEST sealed) ===', flush=True)
    out = []
    for b in RUNGS[1:]:
        c = C[C.rung == b]
        for spl in ('TRAIN', 'VAL'):
            s = c[c.split == spl]
            if not len(s):
                print(f'cap{b:4d} {spl:5s} n=0', flush=True)
                continue
            gain = s.rr_conv - s.rr_meas
            r = dict(rung=b, split=spl, n=len(s), rr_conv=float(s.rr_conv.mean()),
                     rr_meas=float(s.rr_meas.mean()), gain=float(gain.mean()),
                     rr_asis=float(s.rr_asis.mean()),
                     dollars=float((s.rr_conv * 375.0).sum()),
                     h1=float(s[s.half == 'H1'].rr_conv.mean()) if (s.half == 'H1').any() else np.nan,
                     h2=float(s[s.half == 'H2'].rr_conv.mean()) if (s.half == 'H2').any() else np.nan,
                     wr=float((s.rr_conv > 0).mean() * 100),
                     med_over_bps=float(s.over_bps.median()),
                     mde=2.80 * float(s.rr_conv.std(ddof=1) / np.sqrt(len(s))) if len(s) > 2 else np.nan)
            out.append(r)
            print(f'cap{b:4d} {spl:5s} n={len(s):4d} convR={r["rr_conv"]:+.3f} '
                  f'measR={r["rr_meas"]:+.3f} gain={r["gain"]:+.3f} asisR={r["rr_asis"]:+.3f} '
                  f'WR={r["wr"]:4.1f}% H1={r["h1"]:+.3f} H2={r["h2"]:+.3f} '
                  f'over={r["med_over_bps"]:5.1f}bps mde={r["mde"]:.3f}', flush=True)
    pd.DataFrame(out).to_csv(f'{D8}/cells26.csv', index=False)

    print('\n--- the discriminator: converted vs already-marketable (same book, same eras) ---',
          flush=True)
    for spl in ('TRAIN', 'VAL'):
        m = mk[mk.split == spl].rr_asis.dropna()
        cv = C[(C.split == spl) & (C.rung < 9999)].rr_conv.dropna()
        st = C[(C.split == spl) & (C.rung == 9999)]
        print(f'{spl:5s} already-marketable n={len(m):5d} R={m.mean():+.3f} | '
              f'converted (any rung) n={len(cv):4d} R={cv.mean():+.3f} | '
              f'still resting past 150 bps n={len(st):3d} '
              f'measR={st.rr_meas.mean():+.3f}', flush=True)
    print('F26 DONE', flush=True)


if __name__ == '__main__':
    main()
