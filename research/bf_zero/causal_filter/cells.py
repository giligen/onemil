#!/usr/bin/env python3
"""CAUSAL_FILTER step 5 — the 12 pre-declared cells, run as the live-config book.

Cells (PREREG §3): each of the <=5 survivors alone as a VETO (drop its worst TRAIN tercile) and as a
GATE (keep its best TRAIN tercile) = 10, plus the AND of the top-two gates and the AND of the
top-three gates = 12. Tercile edges are cut on TRAIN and APPLIED unchanged to VAL/TEST.

Book: trading.hod_break.run_book(rows, 12, 4) — the ONE book rule.
Cost, two arms:
  band     — the score4 contract with a (price band x hour band) spread constant fit on TRAIN.
  measured — the same contract with this trade's own measured NBBO spread (MEAN over the signal
             minute), plus the OBTAINABILITY rail: the fill is a capped limit at level x (1+cap);
             if the NBBO ask at the decision instant is above that cap the order does not fill and
             the row is removed BEFORE the book is run.
  net = rr - half - half * {stop: 0.875, eod: 0.412, target: 0.0}[why],  half = 0.5 * spread / R

`--test` reads TEST. It refuses to run unless CAUSAL_FILTER_FREEZE.md exists.
"""
import json, os, sys
import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from trading.hod_break import run_book                          # noqa: E402

D = f'{ROOT}/research/bf_zero/causal_filter'
RATIO = {'stop': 0.875, 'eod': 0.412, 'target': 0.0}
PB_EDGES = [0, 20, 30, 50, 100, 1e9]
PB_LAB = ['<$20', '$20-30', '$30-50', '$50-100', '$100+']
HB_EDGES = [569, 585, 600, 660, 780, 960]
HB_LAB = ['09:30-09:45', '09:45-10:00', '10:00-11:00', '11:00-13:00', '13:00+']
CAP = 0.006
NW = {'TRAIN': 53, 'VAL': 23, 'TEST': 15}
WITH_TEST = '--test' in sys.argv


def load():
    c = pd.read_csv(f'{D}/features.csv', dtype={'symbol': str, 'day': str},
                    keep_default_na=False, na_values=[''])
    if os.path.exists(f'{D}/nbbo.csv'):
        n = pd.read_csv(f'{D}/nbbo.csv', dtype={'symbol': str, 'day': str},
                        keep_default_na=False, na_values=['']).drop_duplicates(['day', 'symbol'])
        c = c.merge(n.drop(columns=['entry_m']), on=['day', 'symbol'], how='left')
    else:                                                        # band arm only, loudly flagged
        print('WARNING: nbbo.csv absent — the measured arm is not computable in this run', flush=True)
        for k in ('spread_mean', 'spread_med', 'ask_dec', 'bid_dec'):
            c[k] = np.nan
    c['pb'] = pd.cut(c.price, PB_EDGES, labels=PB_LAB)
    c['hb'] = pd.cut(c.entry_m, HB_EDGES, labels=HB_LAB)
    c['sp_pct'] = c.spread_mean / c.price * 100
    # BAND arm: the §8 band constant, built from the independent cost-curve sample
    # (research/lit_review_2026/cost_curve.csv, Alpaca SIP NBBO) as the MEAN spread-% per
    # (price band x hour band) — mean, not median (Stage P's correction).
    cc = pd.read_csv(f'{ROOT}/research/lit_review_2026/cost_curve.csv',
                     dtype={'symbol': str, 'day': str}, keep_default_na=False, na_values=[''])
    cc = cc[(cc.n_q > 0) & (cc.price > 0)].copy()
    cc['sp_pct'] = cc.spread / cc.price * 100
    cc['pb'] = pd.cut(cc.price, PB_EDGES, labels=PB_LAB)
    cc['hb'] = pd.cut(cc.entry_m, HB_EDGES, labels=HB_LAB)
    band = cc.groupby(['pb', 'hb'], observed=True).sp_pct.mean()
    glob = float(cc[cc.price >= 20].sp_pct.mean())
    c['sp_band'] = [band.get((p, h), np.nan) for p, h in zip(c.pb, c.hb)]
    c['sp_band'] = c.sp_band.fillna(glob)
    ratio = c.why.map(RATIO).fillna(0.875)
    for tag, sp in (('band', c.sp_band), ('meas', c.sp_pct)):
        half = 0.5 * sp / c.r_pct.clip(lower=0.05)
        c[f'net_{tag}'] = c.rr - half - half * ratio
    # obtainability: the capped limit is level x (1+cap); an ask above it is NO fill
    c['cap_px'] = c.level * (1 + CAP)
    c['obtainable'] = np.where(c.ask_dec.notna(), c.ask_dec <= c.cap_px * (1 + 1e-9), np.nan)
    return c


def book(d, col):
    x = d[d[col].notna()]
    if len(x) < 20:
        return None
    rows = [(r.day, int(r.entry_m), int(r.exit_m), r.symbol, getattr(r, col), r.wk) for r in x.itertuples()]
    t = pd.DataFrame(run_book(rows, 12, 4), columns=['day', 'em', 'xm', 'symbol', 'net', 'wk'])
    return t if len(t) else None


def stats(t, split):
    if t is None:
        return None
    nw = NW[split]
    w = t.groupby('wk').net.sum()
    se = t.net.std() / np.sqrt(len(t))
    cap3 = t.net.clip(upper=3.0)
    q95, q99 = t.net.quantile(0.95), t.net.quantile(0.99)
    return dict(n=len(t), tpw=round(len(t) / nw, 1), meanR=round(float(t.net.mean()), 3),
                t=round(float(t.net.mean() / se), 2) if se > 0 else np.nan,
                WR=round(float((t.net > 0).mean() * 100), 1),
                wkR=round(float(t.net.sum() / nw), 1),
                green=round(float((w > 0).sum() / nw), 2), worst=round(float(w.min()), 1),
                ex5=round(float(t.net[t.net <= q95].mean()), 3),
                ex1=round(float(t.net[t.net <= q99].mean()), 3),
                cap3=round(float(cap3.mean()), 3),
                mde=round(float(2.0 * se), 3))


def build_cells(c, sel):
    """The 12 pre-declared cells as boolean masks over the whole frame (edges cut on TRAIN)."""
    tr = c[c.split == 'TRAIN']
    cells, defs = {}, {}
    grp = {}
    for s in sel:
        f = s['feat']
        vals = tr[f].dropna().unique()
        if len(vals) <= 3:
            g = c[f].astype('object').where(c[f].notna())
        else:
            edges = np.unique(np.nanquantile(tr[f].astype(float), [0, 1 / 3, 2 / 3, 1]))
            edges[0], edges[-1] = -np.inf, np.inf
            g = pd.cut(c[f].astype(float), edges, labels=[f'T{i + 1}' for i in range(len(edges) - 1)])
            g = g.astype('object').where(c[f].notna())
        grp[f] = g
    for s in sel:
        f, best, worst = s['feat'], s['best'], s['worst']
        g = grp[f].astype(str)
        cells[f'veto_{f}'] = g != worst
        defs[f'veto_{f}'] = f'drop {f} tercile {worst}'
        cells[f'gate_{f}'] = g == best
        defs[f'gate_{f}'] = f'keep only {f} tercile {best}'
    if len(sel) >= 2:
        m = (grp[sel[0]['feat']].astype(str) == sel[0]['best']) & (grp[sel[1]['feat']].astype(str) == sel[1]['best'])
        cells['AND2'] = m
        defs['AND2'] = f"gate {sel[0]['feat']}={sel[0]['best']} AND gate {sel[1]['feat']}={sel[1]['best']}"
    if len(sel) >= 3:
        m = cells['AND2'] & (grp[sel[2]['feat']].astype(str) == sel[2]['best'])
        cells['AND3'] = m
        defs['AND3'] = defs['AND2'] + f" AND gate {sel[2]['feat']}={sel[2]['best']}"
    return cells, defs


def main():
    sel = json.load(open(f'{D}/selection.json'))['survivors']
    c = load()
    if not WITH_TEST:
        c = c[c.split != 'TEST']
    elif not os.path.exists(f'{ROOT}/research/bf_zero/CAUSAL_FILTER_FREEZE.md'):
        sys.exit('REFUSED: TEST may only be read after CAUSAL_FILTER_FREEZE.md is written.')
    print(f'rows {len(c)} | NBBO covered {c.spread_mean.notna().mean() * 100:.1f}% | '
          f'obtainable {np.nanmean(c.obtainable.astype(float)) * 100:.1f}%', flush=True)
    cells, defs = build_cells(c, sel)
    cells = {'BASELINE (reference, not one of the 12)': pd.Series(True, index=c.index), **cells}
    splits = ('TRAIN', 'VAL', 'TEST') if WITH_TEST else ('TRAIN', 'VAL')
    out = []
    for name, mask in cells.items():
        m = mask.fillna(False) if hasattr(mask, 'fillna') else mask
        for arm, col in (('band', 'net_band'), ('meas', 'net_meas')):
            d = c[m]
            if arm == 'meas':
                d = d[d.obtainable == True]                       # noqa: E712 — the NO-FILL rail
            row = dict(cell=name, rule=defs.get(name, 'no filter'), arm=arm)
            for sp in splits:
                st = stats(book(d[d.split == sp], col), sp)
                for k, v in (st or {}).items():
                    row[f'{sp}_{k}'] = v
            out.append(row)
    R = pd.DataFrame(out)
    R.to_csv(f'{D}/cells{"_test" if WITH_TEST else ""}.csv', index=False)
    show = ['cell', 'arm'] + [f'{s}_{k}' for s in splits for k in ('n', 'tpw', 'meanR', 't', 'wkR', 'green')]
    print(R[[x for x in show if x in R.columns]].to_string(index=False), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
