#!/usr/bin/env python3
"""mature_method / HOD-break — THE bar pass.

Emits EVERY HOD-break bar on the live population (symbol-days whose day high reaches both
open x 1.05 and $19) with the gate fields UNFILTERED, so every gate cell downstream is a
filter + "first qualifying break per symbol-day" selection (exactly detect()'s semantics).

Per break bar: the consolidation stop at the BASE (K=5, X=4%) and at a LOOSE (K=3, X=8%)
setting, the next-bar open (the engine's fill reference), the capped-limit fill flag, and
the full exit walk under BOTH stops -- so the unfilled counterfactual (step 4) and the
consolidation gate (step 5) are both measurable without a second pass.

Read-only on every DB. Resumable per day.
"""
import json, os, sqlite3, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT); sys.path.insert(0, ROOT)
from trading.hod_break import HodBreakParams, profile_fraction  # noqa: E402

# --- bars: the SAME loader order as build_candidates under BFZ_SIP_STORE (cache.db, then the
# --- Alpaca-SIP re-fetch). Standalone so the 4 GB parquet preamble is not paid. Read-only.
_CACHE = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=120)
_SIDE = sqlite3.connect(f'file:{ROOT}/research/bf_zero/bars_sip.db?mode=ro', uri=True, timeout=120)


def load_bars(day, syms):
    """{symbol: DataFrame(m,o,h,l,c,v)} — byte-parity with build_candidates.load_bars (BFZ_SIP_STORE)."""
    out = {}
    q = ("select symbol, timestamp as t, open as o, high as h, low as l, close as c, volume as v "
         f"from intraday_bars_1min where bar_date=? and symbol in ({','.join('?' * len(syms))})")
    for s, gg in pd.read_sql(q, _CACHE, params=[day] + list(syms)).groupby('symbol'):
        out[s] = gg
    left = [s for s in syms if s not in out]
    if left:
        t = pd.read_sql('select symbol, t, o, h, l, c, v from bars where day=?', _SIDE, params=[day])
        for s, gg in t[t.symbol.isin(left)].groupby('symbol'):
            out[s] = gg
    res = {}
    for s, gg in out.items():
        ts = pd.to_datetime(gg.t, utc=True).dt.tz_convert('America/New_York')
        gg = gg.assign(m=(ts.dt.hour * 60 + ts.dt.minute).values).sort_values('m').drop_duplicates('m')
        res[s] = gg[['m', 'o', 'h', 'l', 'c', 'v']].reset_index(drop=True)
    return res

D = 'research/mature_method/hod_break'
STATE, OUT = f'{D}/pass_state.json', f'{D}/breaks.csv'
P = HodBreakParams()
OPEN_M, EOD_M, LAST_M = 570, 955, 930
# NO price floor at detection: `detect()` has none, and the engine applies `min_price` AFTER the
# break and then RETIRES the symbol-day (hod_break_engine._try_enter -> rejected_reason 'price').
# A sub-floor first break therefore KILLS the day, so every break at any price must be emitted or
# the relaxation cells would silently promote a later break the engine would never take.
MIN_LEVEL = float(os.environ.get('MM_MIN_LEVEL', '1.0'))
K_B, X_B = 5, 0.04        # base consolidation (the shipped rule)
K_L, X_L = 3, 0.08        # loose consolidation (the rejected side of the shape gate)
SLIP = 0.001              # stop fills 10 bps through, as the spec
LIMIT_DAYS = int(os.environ.get('MM_LIMIT_DAYS', '0'))

COLS = ['day', 'symbol', 'entry_m', 'level', 'open_px', 'next_open', 'dist_open_pct', 'rv_profile',
        'adv20', 'cumv', 'bar_vol', 'stop_b', 'stop_l', 'r_pct_b', 'r_pct_l', 'fill_capped',
        'rr_b', 'why_b', 'exit_m_b', 'rr_l', 'why_l', 'exit_m_l', 'n_break']


def consol_low(l, h, j, k, x):
    """min low of bars j-k+1..j if they all hold within x of the running HOD at j, else None."""
    if j + 1 < k + 1:
        return None
    hod = float(np.max(h[: j + 1]))
    lo = float(np.min(l[j + 1 - k: j + 1]))
    return lo if (lo >= hod * (1.0 - x) and lo < hod) else None


def walk(o, h, l, c, m, i0, entry, stop, target):
    """the spec's walk_exit, from the bar AFTER entry bar i0."""
    for k in range(i0 + 1, len(o)):
        if int(m[k]) >= EOD_M:
            return k, float(o[k]), 'eod'
        if l[k] <= stop:
            return k, float(min(stop, o[k]) * (1.0 - SLIP)), 'stop'
        if c[k] >= target:
            return k, float(target), 'target'
    return len(o) - 1, float(c[-1]), 'eod'


def main():
    u = pd.read_csv(f'{ROOT}/research/bf_zero/universe.csv', usecols=['symbol', 'bar_date', 'high', 'open', 'adv20'],
                    dtype={'symbol': str}, keep_default_na=False, na_values=[''])
    u = u.rename(columns={'bar_date': 'day'})
    for k in ('high', 'open', 'adv20'):
        u[k] = pd.to_numeric(u[k], errors='coerce')
    c = u[(u.high >= u.open * (1 + P.min_dist_open_pct / 100)) & (u.high >= MIN_LEVEL) & (u.adv20 >= 100000)]
    c = c.drop_duplicates(['day', 'symbol'])
    print(f'symbol-days to walk {len(c)}', flush=True)
    state = json.load(open(STATE)) if os.path.exists(STATE) else {'done': []}
    done = set(state['done'])
    days = [d for d in sorted(c.day.unique()) if d not in done]
    if LIMIT_DAYS:
        days = days[:LIMIT_DAYS]
    for n, day in enumerate(days):
        sub = c[c.day == day]
        bars = load_bars(day, sub.symbol.tolist())
        rows = []
        for r in sub.itertuples():
            gg = bars.get(r.symbol)
            if gg is None:
                continue
            rth = gg[(gg.m >= OPEN_M) & (gg.m < 960)].reset_index(drop=True)
            if len(rth) < 10:
                continue
            o, h, l, cl, v = (rth[k].values.astype(float) for k in ('o', 'h', 'l', 'c', 'v'))
            m = rth.m.values.astype(int)
            n_b = len(h)
            o0 = float(o[0])
            hod = np.maximum.accumulate(h)
            cumv = np.cumsum(v)
            adv = float(r.adv20) if r.adv20 and r.adv20 > 0 else np.nan
            nb = 0
            for i in range(K_B + 1, n_b):
                if int(m[i]) > LAST_M:
                    break
                level = float(hod[i - 1])
                if h[i] < level or level < o0 * 1.05 or level < MIN_LEVEL or i + 1 >= n_b:
                    continue
                sb = consol_low(l, h, i - 1, K_B, X_B)
                sl = consol_low(l, h, i - 1, K_L, X_L)
                if sb is None and sl is None:
                    continue
                nxt = float(o[i + 1])
                rv = float(cumv[i]) / (adv * profile_fraction(int(m[i]))) if adv == adv else np.nan
                row = dict(day=day, symbol=r.symbol, entry_m=int(m[i + 1]), level=level, open_px=o0,
                           next_open=nxt, dist_open_pct=(level / o0 - 1.0) * 100.0, rv_profile=rv,
                           adv20=adv, cumv=float(cumv[i]), bar_vol=float(v[i]),
                           stop_b=sb if sb is not None else np.nan, stop_l=sl if sl is not None else np.nan,
                           fill_capped=int(nxt <= level * (1.0 + P.cap)), n_break=nb)
                for tag, stop in (('b', sb), ('l', sl)):
                    if stop is None or stop >= nxt:
                        row[f'r_pct_{tag}'] = np.nan; row[f'rr_{tag}'] = np.nan
                        row[f'why_{tag}'] = ''; row[f'exit_m_{tag}'] = -1
                        continue
                    Rd = nxt - stop
                    k, px, why = walk(o, h, l, cl, m, i + 1, nxt, stop, nxt + 2.0 * Rd)
                    row[f'r_pct_{tag}'] = Rd / nxt * 100.0
                    row[f'rr_{tag}'] = (px - nxt) / Rd
                    row[f'why_{tag}'] = why
                    row[f'exit_m_{tag}'] = int(m[k])
                rows.append(row)
                nb += 1
        if rows:
            pd.DataFrame(rows)[COLS].to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
        state['done'].append(day)
        json.dump(state, open(STATE, 'w'))
        if n % 20 == 0:
            print(f'{n + 1}/{len(days)} {day} +{len(rows)}', flush=True)
    print('PASS DONE', flush=True)


if __name__ == '__main__':
    main()
