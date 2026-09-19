#!/usr/bin/env python3
"""hod_fresh — THE bar pass.

Emits one row per (symbol-day x declared ADMISSION rung) — the FIRST break bar whose
`consol_bars` satisfies the rung (KEEP-SCANNING: a failing break does NOT retire the day) —
carrying SIX stop designs walked to their exits, plus the level-quality fields.

Rungs (PREREG §2A):  base (no age condition, = hod_filter_stack's B2) · le3 · le5 · le8 · le12 ·
                     ge20 (control).
Stops (PREREG §2B):  n5 = plain low of the last 5 bars (the B2 stop) · n3 = last 3 bars ·
                     b = shipped consolidation low (K5 / X4 %) · bb = the breakout bar's own low ·
                     a05 / a10 = entry - k x ATR20d (20 prior DAILY sessions).

TRAIN + VAL days only — TEST is sealed (FREEZE.md).  Read-only on every DB.  Resumable per day.
"""
import json, os, sqlite3, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT); sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')
from trading.hod_break import profile_fraction                      # noqa: E402
from pass2 import load_bars, consol_low, plain_low                  # noqa: E402  (byte-parity loader)

D = f'{ROOT}/research/mature_method/hod_fresh'
STATE, SIG = f'{D}/pass_state.json', f'{D}/sig3.csv'
OPEN_M, EOD_M, LAST_M = 570, 955, 930
MIN_LEVEL, SLIP, CAP = 1.0, 0.001, 0.006
K_B, X_B = 5, 0.04
DAY_LO, DAY_HI = '2025-01-02', '2026-05-31'          # TRAIN+VAL — TEST is sealed
STOPS = ('n5', 'n3', 'b', 'bb', 'a05', 'a10')
RUNGS = ('base', 'le3', 'le5', 'le8', 'le12', 'ge20')
LIMIT_DAYS = int(os.environ.get('MM_LIMIT_DAYS', '0'))

FCOLS = (['day', 'symbol', 'entry_m', 'break_m', 'level', 'open_px', 'next_open', 'dist_open_pct',
          'rv_profile', 'adv20', 'cumv', 'bar_vol', 'n_break', 'fill_capped',
          'consol_bars', 'touch_n', 'hod_age_bars', 'atr20d', 'atr20d_pct']
         + [f'{p}_{t}' for t in STOPS for p in ('stop', 'r_pct', 'rr', 'why', 'exit_m')]
         + [f'first_{r}' for r in RUNGS])


def rung_ok(rung, cb):
    if rung == 'base':
        return True
    if rung == 'le3':
        return cb <= 3
    if rung == 'le5':
        return cb <= 5
    if rung == 'le8':
        return cb <= 8
    if rung == 'le12':
        return cb <= 12
    return cb >= 20          # ge20 — the control


def walk(o, h, l, c, m, i0, stop, target):
    """The spec's walk_exit, from the bar AFTER the fill bar i0.  Vectorised twin of
    hod_filter_stack/pass2.walk — same priority order (eod, stop, target) and same fills."""
    n = len(o)
    s = i0 + 1
    if s >= n:
        return n - 1, float(c[-1]), 'eod'
    eod = m[s:] >= EOD_M
    hs = l[s:] <= stop
    ht = c[s:] >= target
    any_ = eod | hs | ht
    if not any_.any():
        return n - 1, float(c[-1]), 'eod'
    j = int(np.argmax(any_)); k = s + j
    if eod[j]:
        return k, float(o[k]), 'eod'
    if hs[j]:
        return k, float(min(stop, o[k]) * (1.0 - SLIP)), 'stop'
    return k, float(target), 'target'


def daily_atr20(syms):
    """ATR20 in DOLLARS from the 20 daily sessions STRICTLY BEFORE `day` (prior days only)."""
    con = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=180)
    parts = []
    syms = sorted(syms)
    for a in range(0, len(syms), 400):
        ch = syms[a: a + 400]
        parts.append(pd.read_sql(
            "select symbol, bar_date as day, high, low, close from daily_bars "
            "where bar_date >= '2024-10-01' and bar_date <= ? "
            f"and symbol in ({','.join('?' * len(ch))})", con, params=[DAY_HI] + ch))
    con.close()
    d = pd.concat(parts, ignore_index=True)
    d['symbol'] = d.symbol.astype(str)
    for k in ('high', 'low', 'close'):
        d[k] = pd.to_numeric(d[k], errors='coerce')
    d = d.sort_values(['symbol', 'day'], kind='mergesort')
    pc = d.groupby('symbol', sort=False).close.shift(1)
    tr = np.maximum(d.high - d.low, np.maximum((d.high - pc).abs(), (d.low - pc).abs()))
    d['tr'] = tr
    # strictly prior 20 sessions: shift(1) BEFORE the rolling mean
    d['atr20d'] = (d.groupby('symbol', sort=False).tr.shift(1)
                   .rolling(20, min_periods=20).mean().values)
    return d[['symbol', 'day', 'atr20d']]


def main():
    u = pd.read_csv(f'{ROOT}/research/bf_zero/universe.csv',
                    usecols=['symbol', 'bar_date', 'high', 'open', 'adv20'],
                    dtype={'symbol': str}, keep_default_na=False, na_values=[''])
    u = u.rename(columns={'bar_date': 'day'})
    for k in ('high', 'open', 'adv20'):
        u[k] = pd.to_numeric(u[k], errors='coerce')
    c = u[(u.high >= u.open * 1.05) & (u.high >= MIN_LEVEL) & (u.adv20 >= 100000)]
    c = c[(c.day >= DAY_LO) & (c.day <= DAY_HI)].drop_duplicates(['day', 'symbol'])
    print(f'symbol-days to walk {len(c)} over {c.day.nunique()} days '
          f'[{c.day.min()} .. {c.day.max()}]  (TEST sealed)', flush=True)
    print('loading daily ATR20 (prior 20 sessions) ...', flush=True)
    A = daily_atr20(c.symbol.unique()).set_index(['symbol', 'day']).atr20d
    print(f'  atr20d rows {len(A)}, non-null {A.notna().mean():.1%}', flush=True)

    state = json.load(open(STATE)) if os.path.exists(STATE) else {'done': []}
    done = set(state['done'])
    days = [d for d in sorted(c.day.unique()) if d not in done]
    if LIMIT_DAYS:
        days = days[:LIMIT_DAYS]
    for nd, day in enumerate(days):
        sub = c[c.day == day]
        bars = load_bars(day, sub.symbol.tolist())
        frows = []
        for r in sub.itertuples():
            gg = bars.get(r.symbol)
            if gg is None:
                continue
            rth = gg[(gg.m >= OPEN_M) & (gg.m < 960)].reset_index(drop=True)
            if len(rth) < 10:
                continue
            o, h, l, cl, v = (rth[k].values.astype(float) for k in ('o', 'h', 'l', 'c', 'v'))
            m = rth.m.values.astype(int)
            nb_bars = len(h)
            o0 = float(o[0])
            if not (o0 > 0):
                continue
            hod = np.maximum.accumulate(h)
            cumv = np.cumsum(v)
            adv = float(r.adv20) if r.adv20 and r.adv20 > 0 else np.nan
            try:
                atr = float(A.loc[(r.symbol, day)])
            except KeyError:
                atr = np.nan
            cand, nb = [], 0
            for i in range(K_B + 1, nb_bars):
                if int(m[i]) > LAST_M:
                    break
                level = float(hod[i - 1])
                if h[i] < level or level < o0 * 1.05 or level < MIN_LEVEL or i + 1 >= nb_bars:
                    continue
                rv = float(cumv[i]) / (adv * profile_fraction(int(m[i]))) if adv == adv else np.nan
                cand.append((i, level, rv, nb))
                nb += 1
            if not cand:
                continue
            # ---- consol_bars (hod_losers/walk.py definition, verbatim) per candidate -----------
            cbars = {}
            for (i, level, rv, _n) in cand:
                jj = i - 1; n_ = 0
                while jj >= 0 and l[jj] >= level * 0.96:
                    n_ += 1; jj -= 1
                cbars[i] = n_
            # ---- first-qualifying index per ADMISSION rung (KEEP-SCANNING) --------------------
            firsts = {}
            for rung in RUNGS:
                for (i, level, rv, _n) in cand:
                    if not (rv == rv and rv >= 1.0) or int(m[i + 1]) > LAST_M + 1:
                        continue
                    if plain_low(l, i - 1, 5) is None:
                        continue
                    if not rung_ok(rung, cbars[i]):
                        continue
                    firsts[rung] = i
                    break
            keep = sorted(set(firsts.values()))
            if not keep:
                continue
            for i in keep:
                level, rv, nbi = next((L, R, N) for (j, L, R, N) in cand if j == i)
                nxt = float(o[i + 1])
                row = dict(day=day, symbol=r.symbol, entry_m=int(m[i + 1]), break_m=int(m[i]),
                           level=level, open_px=o0, next_open=nxt,
                           dist_open_pct=(level / o0 - 1.0) * 100.0, rv_profile=rv, adv20=adv,
                           cumv=float(cumv[i]), bar_vol=float(v[i]), n_break=nbi,
                           fill_capped=int(nxt <= level * (1.0 + CAP)),
                           consol_bars=cbars[i],
                           touch_n=int((h[:i] >= level * 0.995).sum()),
                           hod_age_bars=i - (int(np.argmax(h[:i])) if i > 0 else 0),
                           atr20d=atr, atr20d_pct=(atr / nxt * 100.0) if atr == atr else np.nan)
                sd = {'n5': plain_low(l, i - 1, 5),
                      'n3': plain_low(l, i - 1, 3),
                      'b': consol_low(l, h, i - 1, K_B, X_B),
                      'bb': float(l[i]),
                      'a05': (nxt - 0.5 * atr) if atr == atr else None,
                      'a10': (nxt - 1.0 * atr) if atr == atr else None}
                for tag in STOPS:
                    st = sd[tag]
                    if st is None or not (st < nxt) or not (st > 0):
                        row[f'stop_{tag}'] = st if st is not None else np.nan
                        row[f'r_pct_{tag}'] = np.nan; row[f'rr_{tag}'] = np.nan
                        row[f'why_{tag}'] = ''; row[f'exit_m_{tag}'] = -1
                        continue
                    Rd = nxt - st
                    k, px, why = walk(o, h, l, cl, m, i + 1, st, nxt + 2.0 * Rd)
                    row[f'stop_{tag}'] = st
                    row[f'r_pct_{tag}'] = Rd / nxt * 100.0
                    row[f'rr_{tag}'] = (px - nxt) / Rd
                    row[f'why_{tag}'] = why
                    row[f'exit_m_{tag}'] = int(m[k])
                for rung in RUNGS:
                    row[f'first_{rung}'] = int(firsts.get(rung, -1) == i)
                frows.append(row)
        if frows:
            pd.DataFrame(frows)[FCOLS].to_csv(SIG, mode='a', header=not os.path.exists(SIG),
                                              index=False)
        state['done'].append(day); json.dump(state, open(STATE, 'w'))
        if nd % 20 == 0:
            print(f'{nd + 1}/{len(days)} {day} sig+{len(frows)}', flush=True)
    print('PASS DONE', flush=True)


if __name__ == '__main__':
    main()
