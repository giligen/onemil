#!/usr/bin/env python3
"""hod_filter_stack — THE bar pass (stage 1).

Emits, per symbol-day:
  * `sig2.csv`   — one row per FIRST-QUALIFYING break for each declared (stop-variant x rv-band)
                   combo, carrying the three stop walks (base K5/X4, loose K3/X8, NONE = last-5-bar
                   low with no HOD-proximity test) and every PREREG §3 intraday feature that is
                   computable from bars at or before the break bar's close.
  * `blite.csv`  — (day, symbol, break_m, dist_open_pct, ret_open_sig) for EVERY raw break bar, the
                   stream the breadth / cohort features are built from in stage 3.

Read-only on every DB. Resumable per day.
"""
import json, os, sqlite3, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT); sys.path.insert(0, ROOT)
from trading.hod_break import HodBreakParams, profile_fraction  # noqa: E402

_CACHE = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=120)
_SIDE = sqlite3.connect(f'file:{ROOT}/research/bf_zero/bars_sip.db?mode=ro', uri=True, timeout=120)


def load_bars(day, syms):
    """{symbol: DataFrame(m,o,h,l,c,v)} — byte-parity with hod_break/pass_breaks.load_bars."""
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


D = f'{ROOT}/research/mature_method/hod_filter_stack'
STATE, SIG, BLITE = f'{D}/pass_state.json', f'{D}/sig2.csv', f'{D}/blite.csv'
P = HodBreakParams()
OPEN_M, EOD_M, LAST_M, TEN_M = 570, 955, 930, 600
MIN_LEVEL = 1.0           # detect() has no price floor; min_price is applied after and RETIRES the day
K_B, X_B = 5, 0.04        # shipped consolidation
K_L, X_L = 3, 0.08        # loose consolidation
K_N = 5                   # stop window with NO proximity test  (B1/B2: "the filter off")
SLIP = 0.001
LIMIT_DAYS = int(os.environ.get('MM_LIMIT_DAYS', '0'))
TAGS = ('b', 'l', 'n')
COMBOS = [(t, band) for t in TAGS for band in (1, 0)]   # band=1 -> rv in [1,5); band=0 -> rv >= 1

FCOLS = ['day', 'symbol', 'entry_m', 'break_m', 'level', 'open_px', 'next_open', 'dist_open_pct',
         'rv_profile', 'adv20', 'cumv', 'bar_vol', 'n_break', 'fill_capped',
         'stop_b', 'r_pct_b', 'rr_b', 'why_b', 'exit_m_b',
         'stop_l', 'r_pct_l', 'rr_l', 'why_l', 'exit_m_l',
         'stop_n', 'r_pct_n', 'rr_n', 'why_n', 'exit_m_n',
         'vwap_dist_pct', 'ret_open_1000', 'ret_1000_sig', 'ret_open_sig',
         'slope5', 'slope_prior10', 'slope_accel', 'vol3_over_10', 'range_pos', 'atr14_pct',
         'hod_age_bars'] + [f'first_{t}{b}' for t, b in COMBOS]


def consol_low(l, h, j, k, x):
    """min low of bars j-k+1..j if all hold within x of the running HOD at j, else None."""
    if j + 1 < k + 1:
        return None
    hod = float(np.max(h[: j + 1]))
    lo = float(np.min(l[j + 1 - k: j + 1]))
    return lo if (lo >= hod * (1.0 - x) and lo < hod) else None


def plain_low(l, j, k):
    """min low of bars j-k+1..j -- the SAME stop, with the HOD-proximity TEST removed."""
    if j + 1 < k + 1:
        return None
    return float(np.min(l[j + 1 - k: j + 1]))


def walk(o, h, l, c, m, i0, stop, target):
    """the spec's walk_exit, from the bar AFTER entry bar i0."""
    for k in range(i0 + 1, len(o)):
        if int(m[k]) >= EOD_M:
            return k, float(o[k]), 'eod'
        if l[k] <= stop:
            return k, float(min(stop, o[k]) * (1.0 - SLIP)), 'stop'
        if c[k] >= target:
            return k, float(target), 'target'
    return len(o) - 1, float(c[-1]), 'eod'


def ols_slope(y):
    """OLS slope of y against 0..n-1 (units of y per bar)."""
    n = len(y)
    if n < 2:
        return np.nan
    x = np.arange(n, dtype=float)
    xm = x.mean()
    den = ((x - xm) ** 2).sum()
    return float(((x - xm) * (y - y.mean())).sum() / den) if den else np.nan


def main():
    u = pd.read_csv(f'{ROOT}/research/bf_zero/universe.csv',
                    usecols=['symbol', 'bar_date', 'high', 'open', 'adv20'],
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
    for nd, day in enumerate(days):
        sub = c[c.day == day]
        bars = load_bars(day, sub.symbol.tolist())
        frows, brows = [], []
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
            tp = (h + l + cl) / 3.0
            cum_pv = np.cumsum(tp * v)
            adv = float(r.adv20) if r.adv20 and r.adv20 > 0 else np.nan
            # index of the last 10:00-or-earlier bar (for the acceleration split)
            i1000 = int(np.searchsorted(m, TEN_M, side='right')) - 1
            cand, nb = [], 0
            for i in range(K_B + 1, nb_bars):
                if int(m[i]) > LAST_M:
                    break
                level = float(hod[i - 1])
                if h[i] < level or level < o0 * 1.05 or level < MIN_LEVEL or i + 1 >= nb_bars:
                    continue
                rv = float(cumv[i]) / (adv * profile_fraction(int(m[i]))) if adv == adv else np.nan
                brows.append((day, r.symbol, int(m[i]), (level / o0 - 1.0) * 100.0,
                              float(cl[i]) / o0 - 1.0))
                cand.append((i, level, rv, nb))
                nb += 1
            if not cand:
                continue
            # --- first-qualifying index per declared combo -------------------------------------
            stops = {}
            for (i, level, rv, _n) in cand:
                stops[i] = {'b': consol_low(l, h, i - 1, K_B, X_B),
                            'l': consol_low(l, h, i - 1, K_L, X_L),
                            'n': plain_low(l, i - 1, K_N)}
            firsts = {}
            for tag, band in COMBOS:
                for (i, level, rv, _n) in cand:
                    if stops[i][tag] is None or int(m[i + 1]) > LAST_M + 1:
                        continue
                    if not (rv == rv and rv >= 1.0):
                        continue
                    if band and not (rv < 5.0):
                        continue
                    firsts[(tag, band)] = i
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
                           fill_capped=int(nxt <= level * (1.0 + P.cap)))
                for tag in TAGS:
                    st = stops[i][tag]
                    if st is None or st >= nxt:
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
                # ---- PREREG §3 intraday features, all from bars 0..i (the decision instant) ----
                vwap = float(cum_pv[i] / cumv[i]) if cumv[i] > 0 else np.nan
                row['vwap_dist_pct'] = (level / vwap - 1.0) * 100.0 if vwap == vwap and vwap > 0 else np.nan
                c_i = float(cl[i])
                row['ret_open_sig'] = c_i / o0 - 1.0
                if i1000 >= 0 and i1000 < i:
                    c10 = float(cl[i1000])
                    row['ret_open_1000'] = c10 / o0 - 1.0
                    row['ret_1000_sig'] = c_i / c10 - 1.0 if c10 > 0 else np.nan
                else:
                    row['ret_open_1000'] = np.nan; row['ret_1000_sig'] = np.nan
                s5 = ols_slope(cl[max(0, i - 4): i + 1]) if i >= 4 else np.nan
                s10 = ols_slope(cl[max(0, i - 14): max(0, i - 4)]) if i >= 14 else np.nan
                row['slope5'] = s5 / c_i * 100.0 if s5 == s5 and c_i > 0 else np.nan
                row['slope_prior10'] = s10 / c_i * 100.0 if s10 == s10 and c_i > 0 else np.nan
                row['slope_accel'] = row['slope5'] - row['slope_prior10']
                if i >= 12:
                    den = float(np.mean(v[i - 12: i - 2]))
                    row['vol3_over_10'] = float(np.sum(v[i - 2: i + 1])) / den if den > 0 else np.nan
                else:
                    row['vol3_over_10'] = np.nan
                rng = float(h[i] - l[i])
                row['range_pos'] = (c_i - float(l[i])) / rng if rng > 0 else np.nan
                if i >= 13:
                    row['atr14_pct'] = float(np.mean(h[i - 13: i + 1] - l[i - 13: i + 1])) / c_i * 100.0
                else:
                    row['atr14_pct'] = np.nan
                # bars since the running HOD last made a NEW high (strictly before the break bar)
                jh = int(np.argmax(h[:i])) if i > 0 else 0
                row['hod_age_bars'] = i - jh
                for tag, band in COMBOS:
                    row[f'first_{tag}{band}'] = int(firsts.get((tag, band), -1) == i)
                frows.append(row)
        if frows:
            pd.DataFrame(frows)[FCOLS].to_csv(SIG, mode='a', header=not os.path.exists(SIG), index=False)
        if brows:
            pd.DataFrame(brows, columns=['day', 'symbol', 'break_m', 'dist_open_pct', 'ret_open_sig']
                         ).to_csv(BLITE, mode='a', header=not os.path.exists(BLITE), index=False)
        state['done'].append(day)
        json.dump(state, open(STATE, 'w'))
        if nd % 20 == 0:
            print(f'{nd + 1}/{len(days)} {day} sig+{len(frows)} raw+{len(brows)}', flush=True)
    print('PASS DONE', flush=True)


if __name__ == '__main__':
    main()
