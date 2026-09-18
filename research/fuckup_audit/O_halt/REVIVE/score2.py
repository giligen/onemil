#!/usr/bin/env python3
"""S1-REVIVE step 3 — cost, levers, the 24 pre-declared cells, gates, multiplicity.

Contract frozen in REVIVE/PREREG.md. Entry spread charge is ZERO everywhere (owner 9/18: we control
the entry price). Exit charge is outcome-conditional:
    target  0.0 half-spreads (a resting limit)
    stop    1.875            (1.0 crossing + the score4 0.875 stop excess)
    horizon 1.412            (1.0 crossing + the score4 0.412 eod excess)
    half = 0.5 * spread_pct / r_pct, spread = MEASURED Alpaca SIP mean NBBO over the resume minute.

The GATE uses `spread` (last quote at or BEFORE entry_t) — observable at the decision instant.
The COST uses `mean_spread` (the resume-minute mean) — the declared measurement.

usage: score2.py [TRAIN,VAL | TRAIN,VAL,TEST]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
P = f'{ROOT}/research/fuckup_audit/O_halt/REVIVE'
PAS = f'{ROOT}/research/fuckup_audit/O_halt/PASSIVE'
COEF = {'target': 0.0, 'stop': 1.875, 'horizon': 1.412, 'nobar': 1.412}

# ---- the 24 pre-declared cells (PREREG §6) ----
CELLS = ([dict(g='A', side='short', rung='b0', gate=g, r=r, hz=h)
          for g in (None, 1.25) for r in (2.0, 6.0) for h in ('h5', 'eod')]
         + [dict(g='B', side='long', rung='d0', gate=g, r=r, hz=h)
            for g in (None, 1.25) for r in (2.0, 6.0) for h in ('h5', 'eod')]
         + [dict(g='C', side='long', rung=k, gate=None, r=r, hz=h)
            for k in ('d10', 'd20') for r in (2.0, 6.0) for h in ('h5', 'eod')])


def stats(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return dict(n=n, mean=np.nan, t=np.nan, mde=np.nan)
    se = float(np.std(x, ddof=1)) / np.sqrt(n)
    return dict(n=n, mean=float(np.mean(x)), t=float(np.mean(x) / se) if se else np.nan,
                mde=float(2.8 * se))


def weeks(df):
    if df.empty:
        return np.nan, 0
    g = df.groupby(pd.to_datetime(df['day']).dt.to_period('W'))['net_R'].sum()
    return float((g > 0).mean()), len(g)


def load():
    s = pd.read_parquet(f'{P}/sim_rows.parquet')
    s = s[s['filled'] == 1].copy()
    s['r_scale'] = np.where(s['r_pct'] > 0, s['r_pct'], 2.0)   # no-stop arm scored at the O_halt 2% R
    s['half'] = 0.5 * (s['mean_spread'] / s['fill_px'] * 100.0) / s['r_scale']
    s['gross_R'] = s['raw_pct'] / s['r_scale']
    s['coef'] = s['why'].map(COEF)
    s['net_R'] = s['gross_R'] - s['coef'] * s['half']
    s['net_R_allmkt'] = s['gross_R'] - np.where(s['why'] == 'target', 1.0, s['coef']) * s['half']
    s['gate_spread_pct'] = s['spread'] / s['reopen'] * 100.0
    s['spread_over_R'] = s['mean_spread'] / s['fill_px'] * 100.0 / s['r_scale']
    bf = pd.read_csv(f'{PAS}/borrow_flags.csv', keep_default_na=False, na_values=[''],
                     dtype={'symbol': str})
    bf['tradeable'] = bf['shortable'].astype(str).str.lower().eq('true') & \
        bf['easy_to_borrow'].astype(str).str.lower().eq('true')
    s = s.merge(bf[['symbol', 'tradeable']], on='symbol', how='left')
    s['tradeable'] = s['tradeable'].fillna(False)
    return s


def sel(s, side, rung, r, hz, gate=None, floors=None):
    d = s[(s.side == side) & (s.rung == rung) & (s.r_pct == r) & (s.hz == hz)]
    if gate is not None:
        d = d[d.gate_spread_pct <= gate]
    if floors:
        for k, v in floors.items():
            d = d[d[k] >= v]
    return d


def row(d, split, col='net_R'):
    ds = d[d.split == split]
    st = stats(ds[col].values)
    wg, nw = weeks(ds)
    x = np.sort(ds[col].values)
    return dict(n=st['n'], trades_wk=round(st['n'] / nw, 2) if nw else np.nan,
                mean=st['mean'], t=st['t'], mde=st['mde'], wks_green=wg,
                ex_top5=float(np.mean(x[:int(len(x) * 0.95)])) if len(x) > 20 else np.nan,
                cap3R=float(np.mean(np.minimum(x, 3.0))) if len(x) > 2 else np.nan,
                gross=float(ds['gross_R'].mean()) if len(ds) else np.nan,
                cost=float((ds['coef'] * ds['half']).mean()) if len(ds) else np.nan,
                sp_R=float(ds['spread_over_R'].median()) if len(ds) else np.nan,
                tgt_share=float((ds['why'] == 'target').mean()) if len(ds) else np.nan,
                stop_share=float((ds['why'] == 'stop').mean()) if len(ds) else np.nan)


def main() -> int:
    splits = (sys.argv[1] if len(sys.argv) > 1 else 'TRAIN,VAL').split(',')
    s = load()
    raw = pd.read_parquet(f'{P}/sim_rows.parquet')

    # ---------- cost-proxy validation vs PASSIVE's measured cover quotes ----------
    cq = pd.read_csv(f'{PAS}/cover_nbbo.csv', keep_default_na=False, na_values=[''],
                     dtype={'symbol': str, 'day': str})
    cq['cov_spread'] = pd.to_numeric(cq['spread'], errors='coerce')
    ov = s[(s.side == 'short') & (s.rung == 'b0') & (s.r_pct == 2.0) & (s.hz == 'h5')].copy()
    ov['ts'] = pd.to_datetime(ov['exit_t'], utc=True).dt.tz_convert('UTC').map(
        lambda x: x.isoformat().replace('+00:00', '+00:00'))
    m = ov.merge(cq.drop_duplicates(['symbol', 'day', 'ts'])[['symbol', 'day', 'ts', 'cov_spread']],
                 on=['symbol', 'day', 'ts'], how='inner')
    if len(m):
        print(f'\ncost proxy check: n={len(m)}  entry-minute mean spread '
              f'{m.mean_spread.mean():.4f} vs measured cover spread {m.cov_spread.mean():.4f} '
              f'(ratio {m.mean_spread.mean()/max(m.cov_spread.mean(),1e-9):.2f}x)')

    # ---------- LEVER TABLE (TRAIN only, one lever at a time from the base) ----------
    base = dict(side='short', rung='b0', r=2.0, hz='h5', gate=None)
    lev = []

    def add(label, **kw):
        cfg = dict(base, **kw)
        floors = cfg.pop('floors', None)
        d = sel(s, cfg['side'], cfg['rung'], cfg['r'], cfg['hz'], cfg['gate'], floors)
        r = row(d, 'TRAIN')
        lev.append(dict(lever=label, **{k: r[k] for k in
                                        ('n', 'trades_wk', 'mean', 't', 'gross', 'cost', 'sp_R',
                                         'tgt_share', 'stop_share')}))

    add('BASE short b0 R2% +5m no gate')
    for g in (1.25, 0.75, 0.50, 0.25):
        add(f'L1 gate spread<={g}%', gate=g)
    for r in (0.0, 2.0, 4.0, 6.0, 8.0):
        add('L2 NO-STOP (PASSIVE construction)' if r==0 else f'L2 R={r}%', r=r)
    add('L3 exit all-marketable', )
    lev[-1]['note'] = 'see net_R_allmkt'
    for hz in ('h5', 'h30', 'eod'):
        add(f'L4 horizon {hz}', hz=hz)
    for lab, fl in (('price>=10', {'reopen': 10}), ('price>=20', {'reopen': 20}),
                    ('adv20>=500K', {'adv20': 500_000}), ('adv20>=1M', {'adv20': 1_000_000})):
        add(f'L5 {lab}', floors=fl)
    for rg in ('print', 'd0', 'd05', 'd10', 'd20'):
        add(f'L6 LONG rung {rg}', side='long', rung=rg)
    lv = pd.DataFrame(lev)
    # L3 as a real number on the base cell
    d0 = sel(s, 'short', 'b0', 2.0, 'h5')
    lv.loc[lv.lever == 'L3 exit all-marketable', 'mean'] = \
        float(d0[d0.split == 'TRAIN']['net_R_allmkt'].mean())
    lv.loc[lv.lever == 'L3 exit all-marketable', 't'] = \
        stats(d0[d0.split == 'TRAIN']['net_R_allmkt'].values)['t']
    lv.to_csv(f'{P}/levers.csv', index=False)
    print('\n===== LEVER TABLE (TRAIN, one lever at a time) =====')
    print(lv.round(4).to_string(index=False))

    # ---------- ADVERSE SELECTION / OPPORTUNITY COST per rung (PREREG §1) ----------
    adv = []
    for side, rungs in (('short', ['b0']), ('long', ['d0', 'd05', 'd10', 'd20'])):
        for rg in rungs:
            for r_pct in (2.0, 6.0):
                f_all = raw[(raw.side == side) & (raw.rung == rg)]
                key = ['day', 'symbol', 'halt_seq']
                filled_keys = set(map(tuple, f_all[f_all.filled == 1][key].drop_duplicates().values))
                allk = set(map(tuple, f_all[key].drop_duplicates().values))
                pr = s[(s.side == side) & (s.rung == 'print') & (s.r_pct == r_pct) & (s.hz == 'h5')]
                pr = pr[pr.split == 'TRAIN']
                prk = pr.set_index(key)
                onf = [k for k in (allk - filled_keys) if k in prk.index]
                fk = [k for k in filled_keys if k in prk.index]
                d = sel(s, side, rg, r_pct, 'h5')
                d = d[d.split == 'TRAIN']
                adv.append(dict(side=side, rung=rg, r_pct=r_pct,
                                n_signals=len(allk), fill_rate=len(filled_keys) / max(len(allk), 1),
                                mean_filled=float(d['net_R'].mean()) if len(d) else np.nan,
                                mean_print_all=float(pr['net_R'].mean()) if len(pr) else np.nan,
                                mean_print_on_fills=float(prk.loc[fk, 'net_R'].mean()) if fk else np.nan,
                                mean_print_on_NONfills=float(prk.loc[onf, 'net_R'].mean()) if onf else np.nan,
                                n_nonfill=len(onf)))
    ad = pd.DataFrame(adv)
    ad.to_csv(f'{P}/adverse_selection.csv', index=False)
    print('\n===== ADVERSE SELECTION / MISSED-WINNER COST (TRAIN, +5m) =====')
    print(ad.round(4).to_string(index=False))

    # ---------- THE 24 CELLS ----------
    out = []
    for i, c in enumerate(CELLS):
        d = sel(s, c['side'], c['rung'], c['r'], c['hz'], c['gate'])
        rec = dict(cell=i, grp=c['g'], side=c['side'], rung=c['rung'],
                   gate=c['gate'] if c['gate'] else 'none', R=c['r'], hz=c['hz'])
        for sp in splits:
            for k, v in row(d, sp).items():
                rec[f'{sp}_{k}'] = v
        rec['G1'] = (rec.get('TRAIN_mean', -9) > 0 and rec.get('TRAIN_t', 0) >= 2.0
                     and rec.get('TRAIN_trades_wk', 0) >= 3 and rec.get('TRAIN_mean', -9) >= 0.15)
        rec['G2'] = bool(rec['G1'] and rec.get('VAL_mean', -9) > 0
                         and rec.get('VAL_wks_green', 0) >= 0.55 and rec.get('VAL_trades_wk', 0) >= 3)
        rec['borrow_share'] = float(d[d.split == 'TRAIN']['tradeable'].mean()) if c['side'] == 'short' and len(d) else np.nan
        out.append(rec)
    cells = pd.DataFrame(out)
    cells.to_csv(f'{P}/cells.csv', index=False)
    cols = ['cell', 'grp', 'side', 'rung', 'gate', 'R', 'hz'] + \
           [f'{sp}_{k}' for sp in splits for k in ('n', 'trades_wk', 'mean', 't', 'mde', 'wks_green',
                                                   'ex_top5', 'cap3R')] + ['G1', 'G2']
    print('\n===== THE 24 PRE-DECLARED CELLS =====')
    with pd.option_context('display.width', 300, 'display.max_columns', 60):
        print(cells[cols].round(3).to_string(index=False))

    # ---------- permutation across MY 24 cells on TRAIN ----------
    arrs = [sel(s, c['side'], c['rung'], c['r'], c['hz'], c['gate']).query("split=='TRAIN'")['net_R'].values
            for c in CELLS]
    obs = max(abs(stats(a)['t']) for a in arrs if len(a) > 2)
    rng = np.random.default_rng(11)
    B, hits = 2000, 0
    for _ in range(B):
        mx = 0.0
        for a in arrs:
            if len(a) < 3:
                continue
            y = a * rng.choice([-1.0, 1.0], size=len(a))
            se = np.std(y, ddof=1) / np.sqrt(len(y))
            mx = max(mx, abs(np.mean(y) / se) if se else 0.0)
        hits += mx >= obs
    perm_p = (hits + 1) / (B + 1)
    print(f'\nTRAIN max|t| over the 24 cells = {obs:.2f}   permutation p = {perm_p:.4f}')

    json.dump(dict(train_max_abs_t=round(float(obs), 4), perm_p_24=round(float(perm_p), 4),
                   n_rows=int(len(s))), open(f'{P}/score_summary.json', 'w'), indent=1)

    # ---------- EOD swing decomposition (L4) ----------
    e = sel(s, 'short', 'b0', 2.0, 'eod')
    e = e.assign(month=e['day'].str.slice(0, 7))
    e.groupby(['split', 'month'])['net_R'].agg(['size', 'sum', 'mean']).round(3).to_csv(f'{P}/eod_monthly.csv')
    print('\n===== EOD arm, top contributors per split (short b0 R2% eod) =====')
    for sp in ('TRAIN', 'VAL', 'TEST'):
        es = e[e.split == sp]
        if es.empty:
            continue
        top = es.nlargest(5, 'net_R')[['day', 'symbol', 'net_R']]
        tot = es['net_R'].sum()
        print(f'{sp}: n={len(es)} total={tot:.1f}R mean={es.net_R.mean():.3f} '
              f'top5={top.net_R.sum():.1f}R ({top.net_R.sum()/tot*100 if tot else 0:.0f}%) '
              f'| {", ".join(f"{r.symbol} {r.day} {r.net_R:+.1f}" for r in top.itertuples())}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
