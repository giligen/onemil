#!/usr/bin/env python3
"""Stage N1 step 3 — the 12 pre-registered cells, each as an 8-slot ORB book.

Books are produced by the D1 code path (`study_orb_pipeline_static_lock.py` with
ORB_BT_RESIM_CACHE=candidates_dump.csv, N=8, per_pos_cap=account/N=3333.33,
Q1 filter on) — the same command run_grid.sh used for `book_n8_q1on.csv`. The
only additions are two inert-by-default hooks in that pipeline:

  ORB_N1_COMPOSITE_FEATURE / _SIGN   8th signed-z feature in the composite
  ORB_N1_VETO_FEATURE / _SIDE / _Q   bottom-quintile veto, post-ranking, no refill

Cells (pre-registered, 12):
  5 features x {composite, veto} = 10, plus 2 break60 hold rules.
  Declared signs: ofi_range +1, tai_range +1, ofi_break60 +1, tai_break60 +1,
  spread_at_break_bps -1. "Bottom quintile" = the worst quintile under that
  sign (low tail for +1, high tail for -1).

NOTE (causality): the two break60 features are measured in the 60 s AFTER the
breakout trade. The ORB order rests from 09:35, so a break60 value cannot
select a pick. Cells 5-8 (break60 x {composite, veto}) are therefore reported
as DIAGNOSTIC ONLY, never shippable. The break60 HOLD cells are live-usable.

Usage:  python3 cells.py --run      # produce the books (sequential subprocesses)
        python3 cells.py --analyse  # tables -> N1/cells_summary.csv
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil')
os.chdir('/home/ec2-user/onemil')
from trading.orb_csv import read_orb_csv  # noqa: E402

D1 = 'research/fuckup_audit/D1_orb'
N1 = 'research/fuckup_audit/N_databento/N1'
BOOKS = f'{N1}/books'
SIDECAR = f'{N1}/sidecar.csv'

FEATS = [('ofi_range', +1), ('tai_range', +1), ('ofi_break60', +1),
         ('tai_break60', +1), ('spread_at_break_bps', -1)]
DIAGNOSTIC_ONLY = {'ofi_break60', 'tai_break60'}

SPLITS = {'TRAIN': ('2025-01-01', '2025-12-31'),
          'VAL': ('2026-01-01', '2026-05-31'),
          'TEST': ('2026-06-01', '2026-12-31')}


# ---------------------------------------------------------------- run books --
def run_book(tag: str, extra_env: dict, sidecar: str = SIDECAR) -> str:
    out = f'{BOOKS}/book_{tag}.csv'
    if os.path.exists(out):
        return out
    env = dict(os.environ)
    env.update({
        'ORB_BT_FEATURES_CSV': 'analysis_results/orb_features_20260916_2053.csv',
        'ORB_BT_RESIM_CACHE': f'{D1}/candidates_dump.csv',
        'ORB_BT_RISK': '375',
        'ORB_BT_N': '8',
        'ORB_BT_ACCOUNT': repr(3333.333333333333 * 8),
        'ORB_SKIP_Q1': '1',
        'ORB_BT_BOOK_OUT': out,
        'ORB_BT_MONTHLY_OUT': f'{BOOKS}/monthly_{tag}.csv',
    })
    if sidecar:
        env['ORB_BT_SIDECAR_CSV'] = sidecar
    env.update(extra_env)
    with open(f'{BOOKS}/log_{tag}.txt', 'w') as fh:
        rc = subprocess.call(['nice', '-n', '10', 'python3', '-u',
                              'study_orb_pipeline_static_lock.py'],
                             env=env, stdout=fh, stderr=subprocess.STDOUT)
    if rc != 0 or not os.path.exists(out):
        raise SystemExit(f"FATAL: cell {tag} failed (rc={rc}) — see {BOOKS}/log_{tag}.txt")
    return out


def run_all() -> None:
    os.makedirs(BOOKS, exist_ok=True)
    # Parity control: the hook OFF must reproduce book_n8_q1on.csv exactly.
    run_book('baseline', {}, sidecar=SIDECAR)
    base = read_orb_csv(f'{BOOKS}/book_baseline.csv')
    ref = read_orb_csv(f'{D1}/book_n8_q1on.csv')
    same = (len(base) == len(ref)
            and abs(base['_sized_pnl'].sum() - ref['_sized_pnl'].sum()) < 1e-6)
    print(f"PARITY baseline vs D1 book_n8_q1on: rows {len(base)}/{len(ref)} "
          f"P&L {base['_sized_pnl'].sum():,.2f}/{ref['_sized_pnl'].sum():,.2f} "
          f"-> {'OK' if same else 'MISMATCH'}", flush=True)
    if not same:
        raise SystemExit("FATAL: N1 hooks are not inert — refusing to run cells")
    for f, sign in FEATS:
        run_book(f'comp_{f}', {'ORB_N1_COMPOSITE_FEATURE': f,
                               'ORB_N1_COMPOSITE_SIGN': str(sign)})
        print(f"  cell comp_{f} done", flush=True)
        run_book(f'veto_{f}', {'ORB_N1_VETO_FEATURE': f,
                               'ORB_N1_VETO_SIDE': 'low' if sign > 0 else 'high',
                               'ORB_N1_VETO_Q': '0.2'})
        print(f"  cell veto_{f} done", flush=True)


# ------------------------------------------------------------- hold rules ----
def hold_rule_book(feat: str) -> pd.DataFrame:
    """Cell 11/12: exit at fill+10min when the break60 imbalance <= 0 and the
    trade is not yet +0.25R. Exit-only change applied to the baseline book."""
    b = read_orb_csv(f'{BOOKS}/book_baseline.csv')
    b['date'] = pd.to_datetime(b['date']).dt.strftime('%Y-%m-%d')
    f = pd.read_parquet(f'{N1}/features.parquet')
    cols = [feat, 'break_hhmmss', 'range_high', 'range_low']
    b = b.drop(columns=[c for c in cols if c in b.columns])
    b = b.merge(f[['symbol', 'date'] + cols], on=['symbol', 'date'], how='left')
    import sqlite3
    con = sqlite3.connect('file:data/cache.db?mode=ro', uri=True)
    new_pnl = []
    n_fired = 0
    for _, r in b.iterrows():
        pnl = float(r['_sized_pnl'])
        imb = r[feat]
        if (float(r.get('entered', 0)) == 0 or pd.isna(imb) or imb > 0
                or not r['break_hhmmss'] or pd.isna(r['range_high'])
                or pd.isna(r['range_low'])):
            new_pnl.append(pnl)
            continue
        R = float(r['range_high']) - float(r['range_low'])
        if R <= 0:
            new_pnl.append(pnl)
            continue
        entry = float(r['entry_price'])
        t10 = (pd.Timestamp(f"{r['date']} {r['break_hhmmss']}", tz='America/New_York')
               + pd.Timedelta(minutes=10)).floor('min')
        q = ("SELECT close FROM intraday_bars_1min WHERE symbol=? AND bar_date=? "
             "AND timestamp=?")
        row = con.execute(q, (r['symbol'], r['date'],
                              t10.tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%S+00:00')
                              )).fetchone()
        if not row:
            new_pnl.append(pnl)
            continue
        px = float(row[0])
        if (px - entry) / R >= 0.25:          # already +0.25R -> hold, unchanged
            new_pnl.append(pnl)
            continue
        px *= (1 - 10 / 10000.0)               # exit slippage, pipeline's 10 bps
        scale = float(r['_sized_pnl']) / float(r['_rp_pnl']) if float(r['_rp_pnl']) else 1.0
        new = float(r['_rp_position']) * (px - entry) / entry * scale
        new_pnl.append(new)
        n_fired += 1
    con.close()
    b['_sized_pnl'] = new_pnl
    print(f"  hold rule on {feat}: fired on {n_fired} filled picks", flush=True)
    return b


# --------------------------------------------------------------- analysis ----
def enrich(b: pd.DataFrame) -> pd.DataFrame:
    f = pd.read_parquet(f'{N1}/features.parquet')[['symbol', 'date', 'range_high',
                                                   'range_low']]
    b = b.copy()
    b['date'] = pd.to_datetime(b['date']).dt.strftime('%Y-%m-%d')
    b = b.drop(columns=[c for c in ('range_high', 'range_low') if c in b.columns])
    b = b.merge(f, on=['symbol', 'date'], how='left')
    b['month'] = b['date'].str[:7]
    b['week'] = pd.to_datetime(b['date']).dt.strftime('%G-W%V')
    R = (b['range_high'] - b['range_low']).replace(0, np.nan)
    mult = (b['_sized_pnl'] / b['_rp_pnl'].replace(0, np.nan)).fillna(1.0)
    b['R'] = (b['pnl_pct'].astype(float) / 100.0 * b['entry_price'].astype(float)
              / R) * mult
    b.loc[b['entered'].astype(float) == 0, 'R'] = 0.0
    return b


def mdd(daily: pd.Series) -> float:
    cum = daily.cumsum()
    return float((cum - cum.cummax()).min())


def split_stats(b: pd.DataFrame, lo: str, hi: str) -> dict:
    s = b[(b['date'] >= lo) & (b['date'] <= hi)]
    if not len(s):
        return {'picks': 0, 'R_per_pick': np.nan, 't': np.nan, 'pnl': 0.0,
                'weeks_green_pct': np.nan}
    r = s['R'].dropna()
    t = float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))) if len(r) > 2 and r.std(ddof=1) else np.nan
    wk = s.groupby('week')['_sized_pnl'].sum()
    return {'picks': int(len(s)), 'R_per_pick': float(r.mean()) if len(r) else np.nan,
            't': t, 'pnl': float(s['_sized_pnl'].sum()),
            'weeks_green_pct': float(100 * (wk > 0).mean())}


def whole(b: pd.DataFrame) -> dict:
    daily = b.groupby('date')['_sized_pnl'].sum()
    monthly = b.groupby('month')['_sized_pnl'].sum()
    cap = b['_sized_pnl'] * np.where(b['R'] > 3, (3.0 / b['R'].replace(0, np.nan)), 1.0)
    thr = b['_sized_pnl'].quantile(0.95)
    return {'picks': int(len(b)), 'fills': int((b['entered'].astype(float) != 0).sum()),
            'pnl': float(b['_sized_pnl'].sum()),
            'mdd': mdd(daily), 'worst_month': float(monthly.min()),
            'red_months': int((monthly < 0).sum()), 'n_months': int(monthly.size),
            'pnl_ex_top5': float(b.loc[b['_sized_pnl'] <= thr, '_sized_pnl'].sum()),
            'pnl_cap3R': float(np.nansum(cap))}


def analyse() -> None:
    cells = {'baseline': read_orb_csv(f'{BOOKS}/book_baseline.csv')}
    for f, _s in FEATS:
        for kind in ('comp', 'veto'):
            p = f'{BOOKS}/book_{kind}_{f}.csv'
            if os.path.exists(p):
                cells[f'{kind}_{f}'] = read_orb_csv(p)
    for f in ('ofi_break60', 'tai_break60'):
        cells[f'hold_{f}'] = hold_rule_book(f)

    rows = []
    for tag, b in cells.items():
        b = enrich(b)
        rec = {'cell': tag,
               'causal': 'DIAGNOSTIC' if any(d in tag and not tag.startswith('hold')
                                             for d in DIAGNOSTIC_ONLY) else 'yes'}
        for sp, (lo, hi) in SPLITS.items():
            st = split_stats(b, lo, hi)
            rec[f'{sp}_picks'] = st['picks']
            rec[f'{sp}_R'] = st['R_per_pick']
            rec[f'{sp}_t'] = st['t']
            rec[f'{sp}_$'] = st['pnl']
            rec[f'{sp}_wk_green'] = st['weeks_green_pct']
        rec.update(whole(b))
        rows.append(rec)
    tab = pd.DataFrame(rows)
    tab.to_csv(f'{N1}/cells_summary.csv', index=False)
    pd.set_option('display.width', 250)
    cols = ['cell', 'causal', 'TRAIN_R', 'TRAIN_t', 'VAL_R', 'VAL_wk_green',
            'TEST_R', 'picks', 'pnl', 'mdd', 'worst_month', 'red_months',
            'pnl_ex_top5', 'pnl_cap3R']
    print(tab[cols].to_string(index=False, float_format=lambda x: f'{x:,.3f}'))
    print(f"\nWrote {N1}/cells_summary.csv")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--analyse', action='store_true')
    a = ap.parse_args()
    if a.run:
        run_all()
    if a.analyse:
        analyse()
