#!/usr/bin/env python3
"""Stage A shared core — ONE definition of the population, the cost contracts and the book, used by a0/a1/a2/a3.

Contracts (all in R units; half = 0.5 * spread_pct / max(r_pct, 0.05)):
  a  gross   : net = rr
  b  score4  : spread = the candidate's banded spread_pct (1.90/1.20/0.80/0.60/0.50 %)
               net = rr - half - half*{stop:.875, eod:.412, target:0}[why]
  c  corrected: spread = median signal-minute NBBO of THIS population per (price band x time band), cost_curve.csv
               net = rr - 0.25*half_c - half_c*{stop:.875, eod:.412, target:.875}[why]
  c' corrected, resting take-profit leg: as c but target pays 0
  d  = c  restricted to candidates with spread_pct_c / r_pct <= 0.15 (the live liquidity gate), applied BEFORE the book
  d' = c' restricted the same way
Book: trading.hod_break.run_book(rows, 12, 4) — first come, 12/day, 4 concurrent, causal slot freeing.
"""
import os, sys
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'
sys.path.insert(0, ROOT)
from trading.hod_break import run_book                                    # noqa: E402

A = f'{ROOT}/research/fuckup_audit/A'
POP = f'{A}/pop_a.csv'
COST_CURVE = f'{ROOT}/research/lit_review_2026/cost_curve.csv'
PB_EDGES = [5, 10, 20, 50, 200, 1e9]
PB_LAB = ['$5-10', '$10-20', '$20-50', '$50-200', '$200+']
HB_EDGES = [569, 575, 600, 660, 780, 960]
HB_LAB = ['09:30-09:35', '09:35-10:00', '10:00-11:00', '11:00-13:00', '13:00+']
RATIO_B = {'stop': 0.875, 'eod': 0.412, 'target': 0.0}
RATIO_C = {'stop': 0.875, 'eod': 0.412, 'target': 0.875}
ENTRY_COEF_C = 0.25
GATE_SP_OVER_R = 0.15


def corrected_spread_table():
    """Median NBBO spread in bps of price, per (price band x time-of-day band), from the measured cost curve."""
    d = pd.read_csv(COST_CURVE, dtype={'symbol': str, 'day': str}, keep_default_na=False, na_values=[''])
    d = d[d.n_q > 0].copy()
    d['sp_bps'] = d.spread / d.price * 1e4
    g = d.groupby(['pb', 'hb'], observed=True).sp_bps.median()
    return g


def load(keys=None):
    c = pd.read_csv(POP, dtype={'symbol': str, 'day': str, 'fam': str, 'cfg': str},
                    keep_default_na=False, na_values=[''], low_memory=True)
    c['key'] = c.fam + ' ' + c.cfg
    if keys is not None:
        c = c[c.key.isin(keys)].reset_index(drop=True)
    c['split'] = np.where(c.day < '2026-01-01', 'TRAIN', np.where(c.day < '2026-06-01', 'VAL', 'TEST'))
    c['wk'] = pd.to_datetime(c.day).dt.to_period('W-FRI').astype(str)
    c['pb'] = pd.cut(c.price, PB_EDGES, labels=PB_LAB, include_lowest=True).astype(str)  # price==5.0 (453 rows) belongs in $5-10
    c['hb'] = pd.cut(c.entry_m, HB_EDGES, labels=HB_LAB).astype(str)
    tab = corrected_spread_table()
    c['spread_pct_c'] = [tab.get((p, h), np.nan) / 100.0 for p, h in zip(c.pb, c.hb)]
    assert c.spread_pct_c.notna().all(), 'unmapped (pb,hb) cell'
    rr = c.r_pct.clip(lower=0.05)
    half_b = 0.5 * c.spread_pct / rr
    half_c = 0.5 * c.spread_pct_c / rr
    c['sp_over_r_c'] = c.spread_pct_c / c.r_pct
    for tag in ('hold', '2r'):
        why = c[f'why_{tag}']
        rb = why.map(RATIO_B).fillna(0.875)
        rc = why.map(RATIO_C).fillna(0.875)
        c[f'a_{tag}'] = c[f'rr_{tag}']
        c[f'b_{tag}'] = c[f'rr_{tag}'] - half_b - half_b * rb
        c[f'c_{tag}'] = c[f'rr_{tag}'] - ENTRY_COEF_C * half_c - half_c * rc
        c[f'cp_{tag}'] = c[f'rr_{tag}'] - ENTRY_COEF_C * half_c - half_c * rb
    return c


# weeks per split are counted on the WHOLE population (score4's convention), so a cell that misses a week is charged 0
def week_index(c):
    return {s: sorted(c[c.split == s].wk.unique()) for s in ('TRAIN', 'VAL', 'TEST')}


def book_rows(x, tag):
    """Book a candidate subset with run_book(12, 4). Returns the taken rows as a DataFrame with every cost column."""
    if len(x) < 40:
        return None
    rows = [(r.day, int(r.entry_m), int(getattr(r, f'exit_m_{tag}')), r.symbol,
             getattr(r, f'a_{tag}'), getattr(r, f'b_{tag}'), getattr(r, f'c_{tag}'), getattr(r, f'cp_{tag}'),
             getattr(r, f'why_{tag}'), r.wk, r.sp_over_r_c) for r in x.itertuples()]
    t = run_book(rows, 12, 4)
    if not t:
        return None
    return pd.DataFrame(t, columns=['day', 'em', 'xm', 'symbol', 'a', 'b', 'c', 'cp', 'why', 'wk', 'sp_over_r_c'])


def stats(t, col, weeks):
    """Per-cell statistics on a booked table for one cost column."""
    v = t[col].values
    n = len(v)
    se = v.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    w = t.groupby('wk')[col].sum().reindex(weeks).fillna(0)
    return dict(n=n, tpw=round(n / len(weeks), 1), meanR=round(float(v.mean()), 3),
                se=round(float(se), 4), t=round(float(v.mean() / se), 2) if se and se == se else np.nan,
                WR=round(float((v > 0).mean() * 100), 1), wkR=round(float(w.mean()), 2),
                green=round(float((w > 0).mean()), 2), worst=round(float(w.min()), 1))
