#!/usr/bin/env python3
"""R_daily step 1 (REPORT_v2) — how much of each family's negative net comes from split-contaminated
trades, on the EXISTING per-trade outputs.

Two overlap definitions, because the original control used only the first and it is the wrong one:
  HOLD   : a flagged split candidate date lies inside [entry_date, exit_date]  (controls.py control A)
  SIGNAL : a flagged split candidate lies inside [sig_date - lookback, exit_date] — the window the
           family's OWN features are built from (K4's 20-day mean overnight, K2's 250-day high,
           K5's 50-day high / 20-day SMA, K3's 5-day return, K1's 20-day ADV).  An unadjusted split
           inside the feature window fabricates the SELECTION, not the exit.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/fuckup_audit/K')
os.chdir('/home/ec2-user/onemil')
R = 'research/fuckup_audit/R_daily'
from build_k import tstat  # noqa: E402

# calendar-day lookback that covers each family's feature window (sessions -> ~7/5 calendar days)
LOOKBACK_D = {'K1': 30, 'K2': 365, 'K3': 12, 'K4': 30, 'K5': 80}


def main():
    sc = pd.read_csv(f'{R}/split_candidates.csv', keep_default_na=False, na_values=[''])
    by_sym = {}
    for s, d in zip(sc.symbol.astype(str), sc.bar_date.astype(str)):
        by_sym.setdefault(s, []).append(d)
    for k in by_sym:
        by_sym[k].sort()
    print(f'flagged split candidates: {len(sc):,} events, {len(by_sym):,} symbols, '
          f'{sc.bar_date.min()}..{sc.bar_date.max()}\n', flush=True)

    rows, alltr = [], []
    for fn in sorted(os.listdir(f'{R}/trades')):
        if not fn.endswith('.csv') or fn.endswith('_sec.csv'):
            continue
        cell = fn[:-4]
        fam = cell[:2]
        t = pd.read_csv(f'{R}/trades/{fn}', keep_default_na=False, na_values=[''])
        if not len(t):
            continue
        lb = pd.Timedelta(days=LOOKBACK_D[fam])
        sig = pd.to_datetime(t.sig_date)
        lo_sig = (sig - lb).dt.strftime('%Y-%m-%d').to_numpy()
        ent = t.entry_date.astype(str).to_numpy()
        ex = t.exit_date.astype(str).to_numpy()
        sy = t.symbol.astype(str).to_numpy()
        hold_hit = np.zeros(len(t), bool)
        sigw_hit = np.zeros(len(t), bool)
        for i in range(len(t)):
            ds = by_sym.get(sy[i])
            if not ds:
                continue
            hold_hit[i] = any(ent[i] <= d <= ex[i] for d in ds)
            sigw_hit[i] = any(lo_sig[i] <= d <= ex[i] for d in ds)
        t['fam'] = fam
        t['cell'] = cell
        t['hold_hit'] = hold_hit
        t['sigw_hit'] = sigw_hit
        alltr.append(t)
        for sp in ('TRAIN', 'VAL'):
            m = (t.split == sp).to_numpy()
            if not m.sum():
                continue
            x = t[m]
            keep_h = ~x.hold_hit.to_numpy()
            keep_s = ~x.sigw_hit.to_numpy()
            rows.append(dict(
                cell=cell, fam=fam, split=sp, n=int(m.sum()),
                net_bps=x.net.mean() * 1e4, t=tstat(x.net),
                n_hold=int(x.hold_hit.sum()),
                net_bps_ex_hold=x.net[keep_h].mean() * 1e4,
                n_sigw=int(x.sigw_hit.sum()),
                pct_sigw=x.sigw_hit.mean() * 100,
                net_bps_ex_sigw=x.net[keep_s].mean() * 1e4,
                t_ex_sigw=tstat(x.net[keep_s]),
                # share of the cell's total net carried by the contaminated trades
                share_of_total_net=(x.net[x.sigw_hit.to_numpy()].sum() / x.net.sum() * 100
                                    if x.net.sum() != 0 else np.nan)))
    out = pd.DataFrame(rows)
    out.to_csv(f'{R}/contam_step1.csv', index=False, float_format='%.6g')
    pd.set_option('display.width', 250)
    print(out.to_string(index=False, float_format='%.1f'), flush=True)

    tr = pd.concat(alltr, ignore_index=True)
    print('\n--- per FAMILY (booked trades pooled over its 4 cells, TRAIN+VAL) ---', flush=True)
    fam_rows = []
    for fam, g in tr.groupby('fam'):
        g2 = g[g.split.isin(['TRAIN', 'VAL'])]
        k = ~g2.sigw_hit.to_numpy()
        fam_rows.append(dict(fam=fam, n=len(g2), net_bps=g2.net.mean() * 1e4, t=tstat(g2.net),
                             n_contam=int(g2.sigw_hit.sum()), pct=g2.sigw_hit.mean() * 100,
                             net_bps_clean=g2.net[k].mean() * 1e4, t_clean=tstat(g2.net[k]),
                             share_total_net_pct=(g2.net[~k].sum() / g2.net.sum() * 100
                                                  if g2.net.sum() else np.nan)))
    fr = pd.DataFrame(fam_rows)
    fr.to_csv(f'{R}/contam_step1_fam.csv', index=False, float_format='%.6g')
    print(fr.to_string(index=False, float_format='%.1f'), flush=True)

    print('\n--- 20 most extreme single-trade GROSS returns, each tail ---', flush=True)
    u = tr.drop_duplicates(subset=['cell', 'symbol', 'sig_date'])
    u = u.assign(entry_vs_sigclose=(u.entry / u.sig_close - 1.0) * 100)
    cols = ['cell', 'symbol', 'sig_date', 'entry_date', 'exit_date', 'entry', 'exit', 'sig_close',
            'entry_vs_sigclose', 'gross', 'sigw_hit']
    top = u.nlargest(20, 'gross')[cols]
    bot = u.nsmallest(20, 'gross')[cols]
    print(top.to_string(index=False, float_format='%.4g'), flush=True)
    print(flush=True)
    print(bot.to_string(index=False, float_format='%.4g'), flush=True)
    pd.concat([top, bot]).to_csv(f'{R}/contam_extremes.csv', index=False, float_format='%.6g')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
