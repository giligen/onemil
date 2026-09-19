#!/usr/bin/env python3
"""PART 2 — seven sizing cells over a FIXED pick set (PREREG.md §4).

Selection is never re-run. Each cell supplies m'_i; shares = base_shares x m',
clamped by the shipped $200K BP ceiling; pnl = R x rps x shares. Every constant
is fitted on TRAIN picks only. Each cell is renormalised by ONE constant so its
TRAIN mean dollar risk equals S0's.

TEST is SEALED: nothing about 2026-06-01..2026-08-31 is computed or printed
without --reveal-test (FREEZE.md).

Usage: python3 research/bf_sizing/part2.py [--reveal-test]
"""
import sys
import math
import sqlite3

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil')
sys.path.insert(0, '/home/ec2-user/onemil/research/bf_frequency')
ROOT = '/home/ec2-user/onemil'
OUT = f'{ROOT}/research/bf_sizing'
CACHE = f'{ROOT}/data/bull_flag_cache_causal_full_20260905.csv'
RUNS = f'{ROOT}/research/bf_frequency/runs'

REVEAL = '--reveal-test' in sys.argv

COMPONENTS = ['conv_pole_gain', 'conv_flag_tightness', 'conv_vol_ratio',
              'conv_spy_regime', 'conv_retracement', 'conv_vwap_dist',
              'conv_gap_fading']
NUM = ('entry_price', 'stop_loss', 'pnl', 'shares', 'conviction_mult',
       'macd_zone_mult', 'avg_volume_20d') + tuple(COMPONENTS)

SPLITS = {'TRAIN': ('2025-01-01', '2025-12-31', 12),
          'VAL': ('2026-01-01', '2026-05-31', 5),
          'TEST': ('2026-06-01', '2026-08-31', 3)}

def _load(path):
    d = pd.read_csv(path, keep_default_na=False, na_values=[''],
                    dtype={'symbol': str})
    for c in NUM:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors='coerce')
    d['rps'] = d['entry_price'] - d['stop_loss']
    return d


cache = _load(CACHE)
cache = cache[(cache['date'] >= '2025-01-01') & (cache['date'] <= '2026-08-31')]
cache = cache.copy()
cache['R'] = cache['pnl'] / (cache['rps'] * cache['shares'])
cache = cache[np.isfinite(cache['R'])].copy()
CKEY = ['symbol', 'date', 'entry_time_et']


# ------------------------------------------------------------------ helpers --
def market_weeks(lo, hi):
    con = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True)
    d = pd.read_sql_query(
        "select bar_date from daily_bars where symbol='SPY' and bar_date>=? "
        "and bar_date<=? order by bar_date", con, params=(lo, hi))
    con.close()
    return sorted(set(pd.to_datetime(d['bar_date']).dt.strftime('%G-W%V')))


WEEKS = {k: market_weeks(v[0], v[1]) for k, v in SPLITS.items()}


def mdd(p):
    if len(p) == 0:
        return 0.0
    eq = np.cumsum(p)
    peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:]
    return float((eq - peak).min())


def longest_red(vals):
    best = cur = 0
    for v in vals:
        cur = cur + 1 if v < 0 else 0
        best = max(best, cur)
    return best


def topshare(p, k):
    tot = p.sum()
    if len(p) < k or tot == 0:
        return float('nan')
    return float(np.sort(p)[-k:].sum() / tot * 100)


def score(s, split):
    lo, hi, months = SPLITS[split]
    s = s[(s['date'] >= lo) & (s['date'] <= hi)].copy()
    allw = WEEKS[split]
    n = len(s)
    if n == 0:
        return None
    wk = pd.to_datetime(s['date']).dt.strftime('%G-W%V')
    w = s.groupby(wk)['pnl'].sum().reindex(allw).fillna(0.0)
    s['month'] = s['date'].str[:7]
    mo = s.groupby('month')['pnl'].sum()
    p, r = s['pnl'].values, s['R'].values
    sd = float(r.std(ddof=1)) if n > 1 else float('nan')
    return dict(
        n=n, tr_mo=n / months,
        green_wk=float((w > 0).mean() * 100),
        flat_wk=float((w == 0).mean() * 100),
        red_wk=float((w < 0).mean() * 100),
        red_streak=longest_red(w.values),
        worst_wk=float(w.min()),
        green_mo=float((mo > 0).mean() * 100),
        worst_mo=float(mo.min()),
        mdd=mdd(p), pnl=float(p.sum()), totR=float(r.sum()),
        Rpick=float(r.mean()), wr=float((p > 0).mean() * 100),
        top1=topshare(p, 1), top5=topshare(p, 5), top10=topshare(p, 10),
        mean_risk=float(s['risk_c'].mean()),
        t=float(r.mean() / (sd / math.sqrt(n))) if n > 1 and sd else float('nan'),
        mde80=float((1.96 + 0.84) * sd / math.sqrt(n)) if n > 1 else float('nan'),
    )


# ---------------------------------------------------------------- ATR20 ------
def atr20_map(symbols_dates):
    """ATR20 as a % of close, from the 20 sessions STRICTLY BEFORE the date."""
    con = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True)
    out = {}
    syms = sorted({s for s, _ in symbols_dates})
    q = ("select bar_date, high, low, close from daily_bars where symbol=? "
         "order by bar_date")
    bars = {}
    for s in syms:
        d = pd.read_sql_query(q, con, params=(s,))
        if len(d):
            d['pc'] = d['close'].shift(1)
            tr = np.maximum(d['high'] - d['low'],
                            np.maximum((d['high'] - d['pc']).abs(),
                                       (d['low'] - d['pc']).abs()))
            d['atr20'] = tr.rolling(20).mean()
            bars[s] = d
    con.close()
    for s, dt in symbols_dates:
        d = bars.get(s)
        if d is None:
            out[(s, dt)] = float('nan')
            continue
        prior = d[d['bar_date'] < dt]
        if len(prior) < 21 or not np.isfinite(prior['atr20'].iloc[-1]):
            out[(s, dt)] = float('nan')
        else:
            c = prior['close'].iloc[-1]
            out[(s, dt)] = float(prior['atr20'].iloc[-1] / c * 100) if c > 0 else float('nan')
    return out


# ---------------------------------------------------------------- regime -----
def regime_map():
    from trading.regime_helpers import build_regime_lookup
    con = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True)
    spy = pd.read_sql_query(
        "SELECT bar_date, close FROM daily_bars WHERE symbol='SPY' "
        "ORDER BY bar_date", con)
    con.close()
    spy['bar_date'] = pd.to_datetime(spy['bar_date'])
    return build_regime_lookup(spy, 22.0, 0.15)


REGIME = regime_map()
REG_MULT = {'A': 1.25, 'B': 1.0, 'C1': 1.5, 'C2': 0.0}


# ------------------------------------------------------------- the cells -----
def build(pickset):
    d = _load(f'{RUNS}/{pickset}.csv')
    m = d.merge(cache[CKEY + ['R']], on=CKEY, how='left')
    for c in COMPONENTS:
        m[c] = pd.to_numeric(m[c], errors='coerce')
    assert m['R'].notna().all(), 'unjoined picks'
    m['conv'] = m['conviction_mult'].fillna(1.0)
    m['macd'] = m['macd_zone_mult'].fillna(1.0)
    m['mult_shipped'] = m['conv'] * m['macd']
    # base = the shipped sizer with BOTH multipliers divided out (PREREG §1a).
    # base_shares comes from the CACHE row (Stage-1 sizing) so Stage-2's
    # risk-tier rescale and BP clamp are not baked into the base.
    key = [tuple(x) for x in m[CKEY].values]
    cs = {tuple(x): s for x, s in zip(cache[CKEY].values, cache['shares'].values)}
    m['cache_shares'] = [cs[k] for k in key]
    m['base_shares'] = m['cache_shares'] / m['mult_shipped']
    m['bp_shares'] = np.floor(200000.0 / m['entry_price'])
    m['atr_pct'] = [ATR.get((s, dt), float('nan'))
                    for s, dt in zip(m['symbol'], m['date'])]
    m['stop_pct'] = m['rps'] / m['entry_price'] * 100
    m['regime'] = [REGIME.get(dt, 'unknown') for dt in m['date']]
    m['split'] = np.where(m['date'] <= '2025-12-31', 'TRAIN',
                          np.where(m['date'] <= '2026-05-31', 'VAL', 'TEST'))
    return m


def apply_cell(m, mult, name, renorm_to=None):
    """shares = base x m', BP-clamped; pnl = R x rps x shares; renormalised."""
    out = m.copy()
    sh = np.minimum(out['base_shares'] * mult, out['bp_shares'])
    out['shares_c'] = sh
    out['risk_c'] = sh * out['rps']
    tr = out['split'] == 'TRAIN'
    k = 1.0
    if renorm_to is not None and tr.sum() and out.loc[tr, 'risk_c'].mean() > 0:
        k = renorm_to / out.loc[tr, 'risk_c'].mean()
    out['shares_c'] *= k
    out['risk_c'] *= k
    out['pnl'] = out['R'] * out['risk_c']
    out['cell'] = name
    return out


def cells_for(m):
    tr = m['split'] == 'TRAIN'
    cells = {}

    # S0 — the shipped book, exactly as Stage-2 ran it
    s0 = m.copy()
    s0['shares_c'] = s0['shares'].astype(float)
    s0['risk_c'] = s0['shares_c'] * s0['rps']
    s0['pnl'] = s0['R'] * s0['risk_c']
    s0['cell'] = 'S0'
    cells['S0'] = s0
    BASE = float(s0.loc[tr, 'risk_c'].mean())

    # S1 — flat: both multipliers OFF
    cells['S1'] = apply_cell(m, 1.0, 'S1', BASE)

    # S2 — inverse conviction (DIAGNOSTIC)
    C = float(np.median(m.loc[tr, 'conv'])) ** 2
    cells['S2'] = apply_cell(m, np.clip(C / m['conv'], 0.25, 3.0), 'S2', BASE)

    # S3 — volatility-normalised: equal account volatility per trade
    ratio = m['stop_pct'] / m['atr_pct']
    med = float(np.nanmedian(ratio[tr]))
    mult3 = np.clip(ratio / med, 0.25, 3.0)
    mult3 = pd.Series(mult3).fillna(1.0).values      # missing ATR -> flat, loudly
    n_missing = int((~np.isfinite(ratio)).sum())
    cells['S3'] = apply_cell(m, mult3, 'S3', BASE)
    cells['S3'].attrs['missing_atr'] = n_missing

    # S4 — binary
    thr = float(np.median(m.loc[tr, 'conv']))
    cells['S4'] = apply_cell(m, np.where(m['conv'] >= thr, 1.4, 0.7), 'S4', BASE)

    # S5 — walk-forward refit of the component weights (purged + embargoed)
    cells['S5'] = apply_cell(m, refit_mult(m), 'S5', BASE)

    # S6 — regime-conditional on a flat base
    cells['S6'] = apply_cell(
        m, np.array([REG_MULT.get(r, 1.0) for r in m['regime']]), 'S6', BASE)

    # S1b — equal DOLLAR risk (reported diagnostic, not a cell)
    eq = float(np.nanmean(m.loc[tr, 'base_shares'] * m.loc[tr, 'rps']))
    s1b = m.copy()
    s1b['shares_c'] = eq / s1b['rps']
    s1b['risk_c'] = eq
    s1b['pnl'] = s1b['R'] * s1b['risk_c']
    s1b['cell'] = 'S1b'
    cells['S1b'] = s1b

    return cells, dict(C=C, thr=thr, med_ratio=med, base_risk=BASE,
                       missing_atr=n_missing)


def refit_mult(m):
    """PREREG §4 S5: rolling 252-trading-day window, 5-trading-day embargo.

    Fit OLS of R on the 7 standardised conv_* components using ONLY trades whose
    date is <= t - 5 trading days and >= t - 257 trading days. < 30 trades -> 1.0.
    """
    d = m.sort_values('date').reset_index()
    dates = pd.to_datetime(d['date'])
    X = d[COMPONENTS].astype(float).fillna(0.0).values
    y = d['R'].values
    mult = np.ones(len(d))
    for i in range(len(d)):
        hi = dates.iloc[i] - pd.Timedelta(days=7)      # ~5 trading days
        lo = dates.iloc[i] - pd.Timedelta(days=372)    # ~252 trading days
        sel = (dates <= hi) & (dates > lo)
        n = int(sel.sum())
        if n < 30:
            continue
        Xf, yf = X[sel.values], y[sel.values]
        mu, sg = Xf.mean(0), Xf.std(0)
        sg = np.where(sg > 0, sg, 1.0)
        Z = np.hstack([np.ones((len(Xf), 1)), (Xf - mu) / sg])
        try:
            beta = np.linalg.lstsq(Z, yf, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        pred_f = Z @ beta
        pm, ps = pred_f.mean(), pred_f.std()
        if ps <= 0:
            continue
        z = np.hstack([[1.0], (X[i] - mu) / sg])
        p = float(z @ beta)
        mult[i] = float(np.clip(1.0 + (p - pm) / ps, 0.25, 3.0))
    out = np.ones(len(m))
    out[d['index'].values] = mult
    return out


# ------------------------------------------------------------------- run -----
picks = {}
for ps in ('P1', 'F7'):
    d = _load(f'{RUNS}/{ps}.csv')
    picks[ps] = list(zip(d['symbol'], d['date']))
ATR = atr20_map(sorted(set(picks['P1'] + picks['F7'])))

# The grid below runs only as a script; perm.py imports this module for its
# build()/apply_cell()/score() helpers and must not re-run it.
if __name__ != '__main__':
    import sys as _sys
    _sys.modules[__name__].__dict__['_LIB_ONLY'] = True

if __name__ == '__main__':
    rows = []
    detail = {}
    for ps in ('P1', 'F7'):
        m = build(ps)
        cells, consts = cells_for(m)
        print(f'\n### pick set {ps} — n={len(m)} '
              f'(TRAIN {(m["split"]=="TRAIN").sum()} / VAL {(m["split"]=="VAL").sum()}'
              f' / TEST {(m["split"]=="TEST").sum()})', flush=True)
        print(f'  TRAIN-fitted constants: C={consts["C"]:.3f} thr={consts["thr"]:.2f} '
              f'med(stop%/ATR%)={consts["med_ratio"]:.3f} '
              f'S0 TRAIN mean $risk={consts["base_risk"]:,.0f} '
              f'ATR missing on {consts["missing_atr"]} picks (sized flat)', flush=True)
        order = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S1b']
        for split in (['TRAIN', 'VAL'] + (['TEST'] if REVEAL else [])):
            print(f'\n-- {ps} / {split} '
                  f'({len(WEEKS[split])} market weeks) --', flush=True)
            print('  cell |  n  | tr/mo | GREEN wk% | flat% | red% | redstrk | '
                  'worst wk | green mo% | worst mo |    MDD   |  total $ | R/pick |'
                  '  WR%  | top5% | mean$risk', flush=True)
            for c in order:
                s = score(cells[c], split)
                if s is None:
                    continue
                rows.append(dict(pickset=ps, cell=c, split=split, **s))
                print(f'  {c:4s} | {s["n"]:3d} | {s["tr_mo"]:5.2f} | '
                      f'{s["green_wk"]:8.1f}% | {s["flat_wk"]:5.1f} | {s["red_wk"]:4.1f} | '
                      f'{s["red_streak"]:7d} | {s["worst_wk"]:9,.0f} | '
                      f'{s["green_mo"]:8.1f}% | {s["worst_mo"]:9,.0f} | '
                      f'{s["mdd"]:9,.0f} | {s["pnl"]:9,.0f} | {s["Rpick"]:+.2f} | '
                      f'{s["wr"]:5.1f} | {s["top5"]:5.0f} | {s["mean_risk"]:8,.0f}',
                      flush=True)
        detail[ps] = cells

        # S5 whipsaw check (PREREG §4, pre-committed)
        for split in ['TRAIN', 'VAL']:
            lo, hi, _ = SPLITS[split]
            a = cells['S5']; b = cells['S1']
            am = a[(a.date >= lo) & (a.date <= hi)]
            bm = b[(b.date >= lo) & (b.date <= hi)]
            d5 = am['pnl'].values - bm['pnl'].values
            if len(d5) == 0:
                continue
            j = int(np.argmax(np.abs(d5)))
            print(f'  [S5 whipsaw check {ps}/{split}] S5-S1 = {d5.sum():+,.0f}; '
                  f'largest single contributor {am.iloc[j]["symbol"]} '
                  f'{am.iloc[j]["date"]} {d5[j]:+,.0f}; '
                  f'ex-that-trade {d5.sum()-d5[j]:+,.0f}', flush=True)

    g = pd.DataFrame(rows)
    g.to_csv(f'{OUT}/part2_cells{"_test" if REVEAL else ""}.csv', index=False)
    print(f'\nwrote research/bf_sizing/part2_cells'
          f'{"_test" if REVEAL else ""}.csv', flush=True)
