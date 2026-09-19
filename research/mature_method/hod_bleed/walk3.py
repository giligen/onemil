#!/usr/bin/env python3
"""hod_bleed — THE bar walk (one pass, PREREG §3/§7).

For every B0 u B2 u C1 pre-book signal on TRAIN+VAL, re-walk the 1-minute tape from the fill and
emit (a) a per-signal meta row with the shipped-exit parity fields, (b) the whole post-fill bar
path in R units plus the per-bar booleans the declared volume / VWAP / MACD triggers need.

Everything Part 1 and Part 2 need is then pure table work on these two artifacts -- no second walk.

TEST days are never read.  cache.db / bars_sip.db opened read-only.  Resumable per day.
"""
import json, os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT); sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_break')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_fresh')
import score as S             # noqa: E402
import score2 as S2           # noqa: E402
from pass2 import load_bars   # noqa: E402

D = f'{ROOT}/research/mature_method/hod_bleed'
SIGC, STATE = f'{D}/sigs.csv', f'{D}/walk3_state.json'
BARD = f'{D}/bars'
EOD_M, SLIP, OPEN_M = 955, 0.001, 570

SIG_COLS = ['sid', 'day', 'symbol', 'entry_m', 'break_m', 'E', 'ST', 'Rd', 'e_over_rd',
            'brk_vol', 'nk', 'rr_base', 'exit_k_base', 'exit_m_base', 'why_base']


def ema(x, span):
    """EMA with adjust=False, the pandas/TA convention, over a 1-D float array."""
    a = 2.0 / (span + 1.0)
    out = np.empty(len(x), dtype=np.float64)
    if not len(x):
        return out
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def load_pop4():
    """hod_fresh/score4.py::main's population build, verbatim (membership cuts + NBBO + SPY r5)."""
    import sqlite3
    from research.scripts.pit_listings import is_test_ticker
    RD = dict(dtype={'symbol': str, 'day': str}, keep_default_na=False, na_values=[''])
    EARLY_CLOSE = {'2025-07-03', '2025-11-28', '2025-12-24'}
    p = pd.read_csv(f'{ROOT}/research/mature_method/hod_fresh/sig3.csv', **RD)
    p = p[~p.day.isin(EARLY_CLOSE)]
    p = p[~p.symbol.map(is_test_ticker)]
    con = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=120)
    dbs = set(pd.read_sql('select distinct symbol from daily_bars', con).symbol.astype(str))
    con.close()
    p = p[p.symbol.isin(dbs)].reset_index(drop=True)
    p['split'] = S.split_of(p.day.values)
    p['wk'] = pd.to_datetime(p.day).dt.to_period('W-FRI').astype(str)
    nb = pd.read_csv(f'{ROOT}/research/bf_zero/causal_filter/nbbo.csv', **RD
                     ).drop_duplicates(['day', 'symbol', 'entry_m'])
    p = p.merge(nb[['day', 'symbol', 'entry_m', 'spread_mean', 'ask_dec', 'bid_dec', 'n_sig']],
                on=['day', 'symbol', 'entry_m'], how='left')
    df = pd.read_csv(f'{ROOT}/research/mature_method/hod_preopen_regime/day_fields.csv',
                     dtype={'day': str}, keep_default_na=False, na_values=[''])
    p = p.merge(df[['day', 'spy_r5_pct']], on='day', how='left')
    return p.sort_values(['day', 'symbol', 'entry_m'], kind='mergesort').reset_index(drop=True)


def keyset():
    """B0 u B2 (score2) u C1 (hod_fresh rung ge20 x n5 stop), TRAIN+VAL, keyed by the fields that
    define the trade: (day, symbol, entry_m, break_m, next_open, stop)."""
    parts = []
    pop = S2.load_pop(); S.build_impute(pop)
    for nm in ('B0', 'B2'):
        sg = S2.sig_set(pop, **S2.BASES[nm])
        sg = sg[sg.split.isin(('TRAIN', 'VAL'))]
        parts.append(sg[['day', 'symbol', 'entry_m', 'break_m', 'next_open', 'stop']].copy())
    del pop
    import score4 as S4        # noqa: E402  (its own pop; loaded after score2's is released)
    p4 = load_pop4()
    c1 = S4.sig_set4(p4, rung='ge20', stop='n5')
    c1 = c1[c1.split.isin(('TRAIN', 'VAL'))]
    parts.append(c1[['day', 'symbol', 'entry_m', 'break_m', 'next_open', 'stop']].copy())
    del p4, c1
    K = pd.concat(parts, ignore_index=True)
    K = K.drop_duplicates(['day', 'symbol', 'entry_m', 'next_open', 'stop']).reset_index(drop=True)
    K['sid'] = np.arange(len(K), dtype=np.int64)
    return K


def main():
    os.makedirs(BARD, exist_ok=True)
    K = keyset()
    K.to_csv(f'{D}/keys.csv', index=False)
    print(f'signals to walk {len(K)} over {K.day.nunique()} days', flush=True)
    state = json.load(open(STATE)) if os.path.exists(STATE) else {'done': []}
    done = set(state['done'])
    days = [d for d in sorted(K.day.unique()) if d not in done]
    for nd, day in enumerate(days):
        sub = K[K.day == day]
        bars = load_bars(day, sorted(sub.symbol.unique()))
        srows, brows = [], []
        for r in sub.itertuples():
            gg = bars.get(r.symbol)
            if gg is None:
                continue
            rth = gg[(gg.m >= OPEN_M) & (gg.m < 960)].reset_index(drop=True)
            if len(rth) < 10:
                continue
            o, h, l, c, v = (rth[k].values.astype(float) for k in ('o', 'h', 'l', 'c', 'v'))
            m = rth.m.values.astype(int)
            ii = np.where(m == int(r.break_m))[0]
            if not len(ii):
                continue
            i = int(ii[0]); e = i + 1
            if e >= len(o):
                continue
            E, ST = float(r.next_open), float(r.stop)
            Rd = E - ST
            if not (Rd > 0):
                continue
            tgt = E + 2.0 * Rd
            # ---- the shipped walk (byte-parity with hod_losers/walk2.py::walk, base variant) ----
            k_exit, px, why = len(o) - 1, float(c[-1]), 'eod'
            for k in range(e + 1, len(o)):
                if int(m[k]) >= EOD_M:
                    k_exit, px, why = k, float(o[k]), 'eod'; break
                if l[k] <= ST:
                    k_exit, px, why = k, float(min(ST, o[k]) * (1.0 - SLIP)), 'stop'; break
                if c[k] >= tgt:
                    k_exit, px, why = k, float(tgt), 'target'; break
            # ---- session-level derived series (causal, cumulative from 09:30) ------------------
            tp = (h + l + c) / 3.0
            cum_pv = np.cumsum(tp * v); cum_v = np.cumsum(v)
            vwap = np.where(cum_v > 0, cum_pv / np.maximum(cum_v, 1e-12), c)
            macd = ema(c, 12) - ema(c, 26)
            sigl = ema(macd, 9)
            hist = macd - sigl
            xb = np.zeros(len(c), dtype=bool)
            xb[1:] = (macd[1:] < sigl[1:]) & (macd[:-1] >= sigl[:-1])
            bv = float(v[i])
            cprev = np.empty(len(c)); cprev[0] = c[0]; cprev[1:] = c[:-1]
            # ---- the path window: the fill bar through the shipped exit bar --------------------
            sl = slice(e, k_exit + 1)
            nk = k_exit - e + 1
            brows.append(pd.DataFrame({
                'sid': np.full(nk, int(r.sid), dtype=np.int32),
                'k': np.arange(nk, dtype=np.int16),
                'm': m[sl].astype(np.int16),
                'op_r': ((o[sl] - E) / Rd).astype(np.float32),
                'hi_r': ((h[sl] - E) / Rd).astype(np.float32),
                'lo_r': ((l[sl] - E) / Rd).astype(np.float32),
                'cl_r': ((c[sl] - E) / Rd).astype(np.float32),
                't_vol': (v[sl] >= 2.0 * bv) if bv > 0 else np.zeros(nk, bool),
                't_down': (c[sl] < cprev[sl]),
                't_fade': (v[sl] < 0.5 * bv) if bv > 0 else np.zeros(nk, bool),
                't_vwap': (c[sl] < vwap[sl]),
                't_hist': (hist[sl] < 0),
                't_cross': xb[sl],
            }))
            srows.append(dict(sid=int(r.sid), day=day, symbol=r.symbol, entry_m=int(r.entry_m),
                              break_m=int(r.break_m), E=E, ST=ST, Rd=Rd, e_over_rd=E / Rd,
                              brk_vol=bv, nk=nk, rr_base=(px - E) / Rd, exit_k_base=k_exit - e,
                              exit_m_base=int(m[k_exit]), why_base=why))
        if srows:
            pd.DataFrame(srows)[SIG_COLS].to_csv(SIGC, mode='a', header=not os.path.exists(SIGC),
                                                 index=False)
            pd.concat(brows, ignore_index=True).to_parquet(
                f'{BARD}/{day}.parquet', index=False, compression='zstd')
        state['done'].append(day); json.dump(state, open(STATE, 'w'))
        if nd % 25 == 0:
            print(f'{nd + 1}/{len(days)} {day} +{len(srows)}', flush=True)
    print('WALK3 DONE', flush=True)


if __name__ == '__main__':
    main()
