#!/usr/bin/env python3
"""hod_losers — the PATH pass.  For every B0 u B2 pre-book signal on TRAIN+VAL, re-walk the
1-minute tape from the fill and record the path fields the anatomy needs (MFE ladder, the entry
bar's own shape, the stop bar's wick-vs-close, level-quality counts).  TEST days are never read.

Bar loading is byte-parity with `hod_filter_stack/pass2.load_bars`.  Read-only.  Resumable.
"""
import json, os, sqlite3, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT); sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_break')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')
import score as S            # noqa: E402
import score2 as S2          # noqa: E402
from pass2 import load_bars  # noqa: E402

D = f'{ROOT}/research/mature_method/hod_losers'
OUTC, STATE = f'{D}/path.csv', f'{D}/walk_state.json'
EOD_M, SLIP = 955, 0.001
MFE_MARKS = (1, 3, 5, 10, 15, 30)

COLS = ['day', 'symbol', 'entry_m', 'break_m', 'tag',
        'e_low', 'e_high', 'e_close', 'e_open',
        'mfe_r', 'mfe_m', 'mae_r', 'exit_m2', 'why2',
        'stop_bar_close_r', 'stop_wick',
        'touch_n', 'consol_bars', 'hod_bar_vol', 'bar_vol_ratio',
        'pdh_ratio', 'h5_ratio'] + [f'mfe{k}' for k in MFE_MARKS] + [f'mae{k}' for k in MFE_MARKS]


def daily_highs(syms):
    """prev-day high and prior-5-day high per (symbol, day), from daily_bars. Causal: strictly
    before `day`."""
    con = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=180)
    parts = []
    syms = sorted(syms)
    for a in range(0, len(syms), 400):
        ch = syms[a: a + 400]
        parts.append(pd.read_sql(
            "select symbol, bar_date as day, high from daily_bars where bar_date >= '2024-11-01' "
            f"and bar_date < '2026-06-02' and symbol in ({','.join('?' * len(ch))})", con, params=ch))
    con.close()
    d = pd.concat(parts, ignore_index=True)
    d['symbol'] = d.symbol.astype(str)
    d = d.sort_values(['symbol', 'day'], kind='mergesort')
    g = d.groupby('symbol', sort=False).high
    d['pdh'] = g.shift(1)
    d['h5'] = g.shift(1).rolling(5, min_periods=3).max().values
    return d[['symbol', 'day', 'pdh', 'h5']]


def main():
    pop = S2.load_pop()
    S.build_impute(pop)
    keys = []
    for nm, kw in (('B0', S2.BASES['B0']), ('B2', S2.BASES['B2'])):
        sg = S2.sig_set(pop, **kw)
        sg = sg[sg.split.isin(('TRAIN', 'VAL'))]
        keys.append(sg[['day', 'symbol', 'entry_m', 'break_m', 'level', 'next_open',
                        'stop', 'rr', 'bar_vol']].assign(tag=nm))
    K = pd.concat(keys).drop_duplicates(['day', 'symbol', 'entry_m']).reset_index(drop=True)
    del pop, keys, sg
    print(f'signals to walk {len(K)} over {K.day.nunique()} days', flush=True)
    DH = daily_highs(K.symbol.unique()).set_index(['symbol', 'day'])

    state = json.load(open(STATE)) if os.path.exists(STATE) else {'done': []}
    done = set(state['done'])
    days = [d for d in sorted(K.day.unique()) if d not in done]
    for nd, day in enumerate(days):
        sub = K[K.day == day]
        bars = load_bars(day, sorted(sub.symbol.unique()))
        rows = []
        for r in sub.itertuples():
            gg = bars.get(r.symbol)
            if gg is None:
                continue
            rth = gg[(gg.m >= 570) & (gg.m < 960)].reset_index(drop=True)
            if len(rth) < 10:
                continue
            o, h, l, c, v = (rth[k].values.astype(float) for k in ('o', 'h', 'l', 'c', 'v'))
            m = rth.m.values.astype(int)
            ii = np.where(m == int(r.break_m))[0]
            if not len(ii):
                continue
            i = int(ii[0])
            e = i + 1                      # the fill bar (we buy its open)
            if e >= len(o):
                continue
            E, ST = float(r.next_open), float(r.stop)
            Rd = E - ST
            if not (Rd > 0):
                continue
            tgt = E + 2.0 * Rd
            # --- the spec walk from e+1 (the engine excludes the fill bar) -----------------
            k_exit, why = len(o) - 1, 'eod'
            for k in range(e + 1, len(o)):
                if int(m[k]) >= EOD_M:
                    k_exit, why = k, 'eod'; break
                if l[k] <= ST:
                    k_exit, why = k, 'stop'; break
                if c[k] >= tgt:
                    k_exit, why = k, 'target'; break
            else:
                k_exit, why = len(o) - 1, 'eod'
            seg_hi = h[e: k_exit + 1]; seg_lo = l[e: k_exit + 1]
            mfe = (seg_hi.max() - E) / Rd
            mfe_i = e + int(np.argmax(seg_hi))
            mae = (seg_lo.min() - E) / Rd
            row = dict(day=day, symbol=r.symbol, entry_m=int(r.entry_m), break_m=int(r.break_m),
                       tag=r.tag, e_low=float(l[e]), e_high=float(h[e]), e_close=float(c[e]),
                       e_open=float(o[e]), mfe_r=float(mfe), mfe_m=int(m[mfe_i]),
                       mae_r=float(mae), exit_m2=int(m[k_exit]), why2=why,
                       stop_bar_close_r=float((c[k_exit] - E) / Rd),
                       stop_wick=int(why == 'stop' and c[k_exit] > ST))
            for kk in MFE_MARKS:
                j = min(e + kk - 1, k_exit)
                row[f'mfe{kk}'] = float((h[e: j + 1].max() - E) / Rd)
                row[f'mae{kk}'] = float((l[e: j + 1].min() - E) / Rd)
            # --- level quality, all from bars strictly before the break bar ---------------
            lev = float(r.level)
            pre_h = h[:i]
            row['touch_n'] = int((pre_h >= lev * 0.995).sum())
            jj = i - 1; nb = 0
            while jj >= 0 and l[jj] >= lev * 0.96:
                nb += 1; jj -= 1
            row['consol_bars'] = nb
            jh = int(np.argmax(pre_h)) if i > 0 else 0
            row['hod_bar_vol'] = float(v[jh])
            row['bar_vol_ratio'] = float(v[i] / v[jh]) if v[jh] > 0 else np.nan
            try:
                dh = DH.loc[(r.symbol, day)]
                row['pdh_ratio'] = float(lev / dh.pdh) if dh.pdh == dh.pdh and dh.pdh > 0 else np.nan
                row['h5_ratio'] = float(lev / dh.h5) if dh.h5 == dh.h5 and dh.h5 > 0 else np.nan
            except KeyError:
                row['pdh_ratio'] = np.nan; row['h5_ratio'] = np.nan
            rows.append(row)
        if rows:
            pd.DataFrame(rows)[COLS].to_csv(OUTC, mode='a', header=not os.path.exists(OUTC),
                                            index=False)
        state['done'].append(day); json.dump(state, open(STATE, 'w'))
        if nd % 25 == 0:
            print(f'{nd + 1}/{len(days)} {day} +{len(rows)}', flush=True)
    print('WALK DONE', flush=True)


if __name__ == '__main__':
    main()
