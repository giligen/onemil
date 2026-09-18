#!/usr/bin/env python3
"""S1-REVIVE step 3 — measured-cost scoring, levers, the 24 pre-declared cells, Bar A and Bar B.

Contract frozen in REVIVE/PREREG.md + the two 9/18 coordinator amendments:

  ENTRY COST = ZERO everywhere. We control the entry price; a non-fill is 0 P&L, never a loss.
  EXIT COST  = outcome-conditional, on the MEASURED Alpaca SIP NBBO at the exit instant:
                 target  0.0   half-spreads (a resting limit — costs nothing to be filled)
                 stop    1.875 (1.0 crossing + the score4 0.875 stop excess)
                 horizon 1.412 (1.0 crossing + the score4 0.412 eod excess)
               half = 0.5 * exit_spread_pct / r_scale
  STALE-QUOTE QUARANTINE (AUDIT_COST_REVIVAL): a quote older than 30 s at the instant it is used,
               or a spread wider than 50% of price, is a BROKEN PRINT, not a cost. Dropped, counted.
  THREE CHARGES reported for every cell: per-trade measured (pt), winsorised at the split's p95
               spread (w95), and the split's MEDIAN spread applied to every trade (med).

usage: score3.py [TRAIN,VAL | TRAIN,VAL,TEST]
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
MAX_AGE_S = 30.0
MAX_SPREAD_PCT = 50.0

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
        return dict(n=n, mean=np.nan, t=np.nan, mde=np.nan, sd=np.nan)
    sd = float(np.std(x, ddof=1))
    se = sd / np.sqrt(n)
    return dict(n=n, mean=float(np.mean(x)), t=float(np.mean(x) / se) if se else np.nan,
                mde=float(2.8 * se), sd=sd)


def weeks(df, col='net_R'):
    if df.empty:
        return np.nan, 0
    g = df.groupby(pd.to_datetime(df['day']).dt.to_period('W'))[col].sum()
    return float((g > 0).mean()), len(g)


def load():
    s = pd.read_parquet(f'{P}/sim_rows.parquet')
    s = s[s['filled'] == 1].copy()

    # ---- entry quote age (the limit price + the L1 gate both use this quote) ----
    eq = pd.concat([pd.read_csv(f'{PAS}/entry_nbbo.csv', keep_default_na=False, na_values=[''],
                                dtype={'symbol': str, 'day': str}),
                    pd.read_csv(f'{P}/entry_nbbo_long.csv', keep_default_na=False, na_values=[''],
                                dtype={'symbol': str, 'day': str})], ignore_index=True)
    eq = eq.drop_duplicates(['symbol', 'day', 'halt_seq'], keep='last')
    eq['ent_qts'] = pd.to_datetime(eq['q_ts'], utc=True, errors='coerce')
    s = s.merge(eq[['symbol', 'day', 'halt_seq', 'ent_qts']], on=['symbol', 'day', 'halt_seq'],
                how='left')
    s['ent_age_s'] = (pd.to_datetime(s['entry_t']).dt.tz_convert('UTC') - s['ent_qts']).dt.total_seconds()

    # ---- measured NBBO at every CHARGED exit instant ----
    xq = [pd.read_csv(f'{PAS}/cover_nbbo.csv', keep_default_na=False, na_values=[''],
                      dtype={'symbol': str, 'day': str, 'ts': str})]
    if os.path.exists(f'{P}/exit_nbbo.csv'):
        xq.append(pd.read_csv(f'{P}/exit_nbbo.csv', keep_default_na=False, na_values=[''],
                              dtype={'symbol': str, 'day': str, 'ts': str}))
    xq = pd.concat(xq, ignore_index=True).drop_duplicates(['symbol', 'day', 'ts'], keep='last')
    xq['ex_spread'] = pd.to_numeric(xq['spread'], errors='coerce')
    xq['ex_qts'] = pd.to_datetime(xq['q_ts'], utc=True, errors='coerce')
    s['ts'] = pd.to_datetime(s['exit_t']).dt.tz_convert('UTC').map(lambda x: x.isoformat())
    s = s.merge(xq[['symbol', 'day', 'ts', 'ex_spread', 'ex_qts']], on=['symbol', 'day', 'ts'],
                how='left')
    s['ex_age_s'] = (pd.to_datetime(s['ts'], utc=True) - s['ex_qts']).dt.total_seconds()

    s['r_scale'] = np.where(s['r_pct'] > 0, s['r_pct'], 2.0)
    s['ex_spread_pct'] = s['ex_spread'] / s['exit_px'] * 100.0
    s['ent_spread_pct'] = s['spread'] / s['reopen'] * 100.0          # the L1 gate quantity
    s['coef'] = s['why'].map(COEF)
    s['free_exit'] = s['why'].eq('target')

    # ---- quarantine ----
    # ENTRY side: a stale/broken quote at the decision instant is knowable BEFORE we act, so the
    # event is DROPPED (we would not have placed the order on a quote we could not trust).
    # EXIT side: whether the exit quote will be stale is NOT knowable at entry — dropping on it
    # would be a look-ahead selection (it correlates with illiquidity AND with the exit type,
    # since target exits need no quote at all). So the TRADE IS KEPT and only the unusable
    # MEASUREMENT is replaced by the split's median charged spread, flagged.
    bad_ent = (s['ent_age_s'] > MAX_AGE_S) | s['ent_age_s'].isna() | \
              (s['ent_spread_pct'] > MAX_SPREAD_PCT)
    s['bad_exit_meas'] = (~s['free_exit']) & ((s['ex_age_s'] > MAX_AGE_S) | s['ex_spread'].isna() |
                                              (s['ex_spread_pct'] > MAX_SPREAD_PCT))
    one = s[(s.r_pct == 0.0) & (s.hz == 'h5')]
    q = pd.DataFrame({
        'events': one.groupby('split').size(),
        'entry_quote_dropped': one.groupby('split')['ent_age_s'].apply(
            lambda x: float((x > MAX_AGE_S).mean())),
    })
    q['exit_meas_replaced'] = s[~s['free_exit']].groupby('split')['bad_exit_meas'].mean()
    print('\n===== QUOTE QUARANTINE (30 s stale / >50% spread) =====')
    print(q.round(4).to_string())
    s = s[~bad_ent].copy()
    return s


def apply_charges(s):
    """pt = per-trade measured; w95 = spread winsorised at the split's p95; med = the split's median."""
    out = []
    for split, g in s.groupby('split'):
        chg = g[(~g.free_exit) & (~g.bad_exit_meas)]['ex_spread_pct']
        p95 = float(np.nanpercentile(chg, 95)) if len(chg) else np.nan
        med = float(np.nanmedian(chg)) if len(chg) else np.nan
        g = g.copy()
        # unusable exit measurement -> the split's median charged spread (never a free pass)
        g['ex_spread_pct'] = np.where(g['bad_exit_meas'], med, g['ex_spread_pct'])
        g['sp_pt'] = np.where(g.free_exit, 0.0, g['ex_spread_pct'].fillna(med))
        g['sp_w95'] = np.minimum(g['sp_pt'], p95)
        g['sp_med'] = np.where(g.free_exit, 0.0, med)
        g['gross_R'] = g['raw_pct'] / g['r_scale']
        for tag in ('pt', 'w95', 'med'):
            half = 0.5 * g[f'sp_{tag}'] / g['r_scale']
            g[f'net_{tag}'] = g['gross_R'] - g['coef'] * half
        g['net_R'] = g['net_pt']
        g['cost_R'] = g['coef'] * (0.5 * g['sp_pt'] / g['r_scale'])
        g['spread_over_R'] = g['sp_pt'] / g['r_scale']
        g['p95_spread_pct'] = p95
        g['med_spread_pct'] = med
        out.append(g)
    return pd.concat(out, ignore_index=True)


def sel(s, side, rung, r, hz, gate=None, floors=None):
    d = s[(s.side == side) & (s.rung == rung) & (s.r_pct == r) & (s.hz == hz)]
    if gate is not None:
        d = d[d.ent_spread_pct <= gate]
    if floors:
        for k, v in floors.items():
            d = d[d[k] >= v]
    return d


def row(d, split):
    ds = d[d.split == split]
    st = stats(ds['net_R'].values)
    wg, nw = weeks(ds)
    x = np.sort(ds['net_R'].values)
    twk = round(st['n'] / nw, 2) if nw else np.nan
    # weeks to resolve the point estimate from zero at ~1 SE
    if st['n'] >= 3 and st['mean'] and st['mean'] > 0 and twk:
        wks_res = (st['sd'] / st['mean']) ** 2 / twk
    else:
        wks_res = np.nan
    return dict(n=st['n'], trades_wk=twk, mean=st['mean'], t=st['t'], mde=st['mde'], sd=st['sd'],
                wks_green=wg, wks_resolve=wks_res,
                ex_top5=float(np.mean(x[:int(len(x) * 0.95)])) if len(x) > 20 else np.nan,
                cap3R=float(np.mean(np.minimum(x, 3.0))) if len(x) > 2 else np.nan,
                gross=float(ds['gross_R'].mean()) if len(ds) else np.nan,
                cost=float(ds['cost_R'].mean()) if len(ds) else np.nan,
                sp_R=float(ds['spread_over_R'].median()) if len(ds) else np.nan,
                mean_w95=float(ds['net_w95'].mean()) if len(ds) else np.nan,
                mean_med=float(ds['net_med'].mean()) if len(ds) else np.nan,
                tgt=float((ds['why'] == 'target').mean()) if len(ds) else np.nan,
                stop=float((ds['why'] == 'stop').mean()) if len(ds) else np.nan)


def main() -> int:
    splits = (sys.argv[1] if len(sys.argv) > 1 else 'TRAIN,VAL').split(',')
    s = apply_charges(load())
    raw = pd.read_parquet(f'{P}/sim_rows.parquet')
    bf = pd.read_csv(f'{PAS}/borrow_flags.csv', keep_default_na=False, na_values=[''],
                     dtype={'symbol': str})
    bf['tradeable'] = bf['shortable'].astype(str).str.lower().eq('true') & \
        bf['easy_to_borrow'].astype(str).str.lower().eq('true')
    s = s.merge(bf[['symbol', 'tradeable']], on='symbol', how='left')
    s['tradeable'] = s['tradeable'].astype('object').fillna(False).astype(bool)

    print('\nmeasured exit spread %% of price, charged exits only:')
    ce = s[(~s.free_exit) & (s.r_pct == 0.0) & (s.hz == 'h5')]
    print(ce.groupby('split')['ex_spread_pct'].describe(
        percentiles=[.5, .75, .9, .95]).round(3).to_string())

    # ---------------- LEVER TABLE (TRAIN only) ----------------
    base = dict(side='short', rung='b0', r=2.0, hz='h5', gate=None)
    lev = []

    def add(label, **kw):
        cfg = dict(base, **kw)
        fl = cfg.pop('floors', None)
        r = row(sel(s, cfg['side'], cfg['rung'], cfg['r'], cfg['hz'], cfg['gate'], fl), 'TRAIN')
        lev.append(dict(lever=label, **{k: r[k] for k in
                                        ('n', 'trades_wk', 'mean', 't', 'mean_w95', 'mean_med',
                                         'gross', 'cost', 'sp_R', 'tgt', 'stop')}))

    add('BASE  short b0 · R2% · +5m · no gate')
    for g in (1.25, 0.75, 0.50, 0.25):
        add(f'L1 spread gate <= {g}%', gate=g)
    for r in (0.0, 2.0, 4.0, 6.0, 8.0):
        add('L2 NO-STOP (= the PASSIVE construction)' if r == 0 else f'L2 R = {r}%', r=r)
    for hz in ('h5', 'h30', 'eod'):
        add(f'L4 horizon {hz}', hz=hz)
    for lab, fl in (('price>=$10', {'reopen': 10}), ('price>=$20', {'reopen': 20}),
                    ('ADV20>=500K', {'adv20': 500_000}), ('ADV20>=1M', {'adv20': 1_000_000})):
        add(f'L5 {lab}', floors=fl)
    for rg in ('print', 'd0', 'd05', 'd10', 'd20'):
        add(f'L6 LONG rung {rg}', side='long', rung=rg)
    lv = pd.DataFrame(lev)
    # L3 (exit construction) measured on the base cell: target leg free vs charged
    d0 = sel(s, 'short', 'b0', 2.0, 'h5')
    d0t = d0[d0.split == 'TRAIN']
    sp_alt = np.where(d0t.free_exit, d0t['med_spread_pct'], d0t['sp_pt'])
    allmkt = (d0t['gross_R'] - np.where(d0t.free_exit, 1.0, d0t['coef']) *
              0.5 * sp_alt / d0t['r_scale'])
    lv = pd.concat([lv, pd.DataFrame([dict(
        lever='L3 exit ALL-marketable (target charged too)', n=len(d0t),
        trades_wk=np.nan, mean=float(allmkt.mean()), t=stats(allmkt.values)['t'],
        mean_w95=np.nan, mean_med=np.nan, gross=float(d0t['gross_R'].mean()),
        cost=np.nan, sp_R=np.nan, tgt=float(d0t.free_exit.mean()), stop=np.nan)])],
        ignore_index=True)
    lv.to_csv(f'{P}/levers.csv', index=False)
    print('\n===== LEVER TABLE (TRAIN, one lever at a time from the base) =====')
    print(lv.round(4).to_string(index=False))

    # ---------------- ADVERSE SELECTION / MISSED-WINNER COST ----------------
    adv = []
    key = ['day', 'symbol', 'halt_seq']
    for side, rungs in (('short', ['b0']), ('long', ['d0', 'd05', 'd10', 'd20'])):
        for rg in rungs:
            f_all = raw[(raw.side == side) & (raw.rung == rg)]
            allk = set(map(tuple, f_all[key].drop_duplicates().values))
            fk = set(map(tuple, f_all[f_all.filled == 1][key].drop_duplicates().values))
            for r_pct in (2.0, 6.0):
                pr = sel(s, side, 'print', r_pct, 'h5')
                pr = pr[pr.split == 'TRAIN'].set_index(key)
                nf = [k for k in (allk - fk) if k in pr.index]
                yf = [k for k in fk if k in pr.index]
                d = sel(s, side, rg, r_pct, 'h5')
                d = d[d.split == 'TRAIN']
                adv.append(dict(side=side, rung=rg, R=r_pct, n_signals=len(allk),
                                fill_rate=round(len(fk) / max(len(allk), 1), 4),
                                filled_netR=float(d['net_R'].mean()) if len(d) else np.nan,
                                print_all_netR=float(pr['net_R'].mean()) if len(pr) else np.nan,
                                print_on_fills=float(pr.loc[yf, 'net_R'].mean()) if yf else np.nan,
                                MISSED_nonfills=float(pr.loc[nf, 'net_R'].mean()) if nf else np.nan,
                                n_nonfill=len(nf)))
    ad = pd.DataFrame(adv)
    ad.to_csv(f'{P}/adverse_selection.csv', index=False)
    print('\n===== ADVERSE SELECTION & MISSED-WINNER COST (TRAIN, +5m) =====')
    print(ad.round(4).to_string(index=False))

    # ---------------- THE 24 CELLS ----------------
    out = []
    for i, c in enumerate(CELLS):
        d = sel(s, c['side'], c['rung'], c['r'], c['hz'], c['gate'])
        rec = dict(cell=i, grp=c['g'], side=c['side'], rung=c['rung'],
                   gate=c['gate'] if c['gate'] else 'none', R=c['r'], hz=c['hz'])
        for sp in splits:
            for k, v in row(d, sp).items():
                rec[f'{sp}_{k}'] = v
        rec['G1'] = bool(rec.get('TRAIN_mean', -9) > 0 and rec.get('TRAIN_t', 0) >= 2.0
                         and rec.get('TRAIN_trades_wk', 0) >= 3
                         and rec.get('TRAIN_mean', -9) >= 0.15)
        rec['G2'] = bool(rec['G1'] and rec.get('VAL_mean', -9) > 0
                         and rec.get('VAL_wks_green', 0) >= 0.55
                         and rec.get('VAL_trades_wk', 0) >= 3)
        # ---- Bar B, the live-exploration bar ----
        b1 = bool(rec.get('TRAIN_mean', -9) > 0 and rec.get('VAL_mean', -9) > 0)
        b4_wk = rec.get('VAL_wks_resolve', np.nan)
        b4 = bool(np.isfinite(b4_wk) and b4_wk <= 13)
        rec['B1_positive'] = b1
        rec['B4_wks_to_resolve'] = b4_wk
        rec['B4_ok'] = b4
        rec['borrow_share'] = (float(d[d.split == 'TRAIN']['tradeable'].mean())
                               if c['side'] == 'short' and len(d) else np.nan)
        # capacity-adjusted frequency for shorts (borrow-constrained)
        rec['eff_trades_wk'] = (rec.get('TRAIN_trades_wk', np.nan) * rec['borrow_share']
                                if c['side'] == 'short' else rec.get('TRAIN_trades_wk', np.nan))
        rec['B3_bounded'] = True            # $100 risk, rails below — same for every cell
        rec['BarB'] = bool(b1 and b4 and rec['eff_trades_wk'] >= 3)
        out.append(rec)
    cells = pd.DataFrame(out)
    cells.to_csv(f'{P}/cells.csv', index=False)
    cols = ['cell', 'grp', 'side', 'rung', 'gate', 'R', 'hz'] + \
        [f'{sp}_{k}' for sp in splits
         for k in ('n', 'trades_wk', 'mean', 't', 'mde', 'wks_green', 'mean_w95', 'mean_med',
                   'ex_top5', 'cap3R')] + \
        ['G1', 'G2', 'B1_positive', 'B4_wks_to_resolve', 'eff_trades_wk', 'BarB']
    print('\n===== THE 24 PRE-DECLARED CELLS =====')
    with pd.option_context('display.width', 400, 'display.max_columns', 80):
        print(cells[cols].round(3).to_string(index=False))

    # ---------------- permutation across the 24 cells ----------------
    arrs = [sel(s, c['side'], c['rung'], c['r'], c['hz'], c['gate'])
            .query("split=='TRAIN'")['net_R'].values for c in CELLS]
    obs = max(abs(stats(a)['t']) for a in arrs if len(a) > 2)
    best_pos = max((stats(a)['t'] for a in arrs if len(a) > 2))
    rng = np.random.default_rng(11)
    B, hits, hits_pos = 2000, 0, 0
    for _ in range(B):
        mx, mxp = 0.0, -9.0
        for a in arrs:
            if len(a) < 3:
                continue
            y = a * rng.choice([-1.0, 1.0], size=len(a))
            se = np.std(y, ddof=1) / np.sqrt(len(y))
            tt = (np.mean(y) / se) if se else 0.0
            mx, mxp = max(mx, abs(tt)), max(mxp, tt)
        hits += mx >= obs
        hits_pos += mxp >= best_pos
    perm_p = (hits + 1) / (B + 1)
    perm_p_pos = (hits_pos + 1) / (B + 1)
    print(f'\nTRAIN max|t| over the 24 cells = {obs:.2f}  perm p = {perm_p:.4f}')
    print(f'TRAIN best POSITIVE t over the 24 cells = {best_pos:.2f}  perm p = {perm_p_pos:.4f}')

    json.dump(dict(train_max_abs_t=round(float(obs), 4), perm_p_24=round(float(perm_p), 4),
                   train_best_pos_t=round(float(best_pos), 4),
                   perm_p_best_positive=round(float(perm_p_pos), 4),
                   n_rows=int(len(s))), open(f'{P}/score_summary.json', 'w'), indent=1)

    # ---------------- L4: explain the EOD/no-stop swing ----------------
    print('\n===== L4 — the NO-STOP arm by horizon and split (the PASSIVE swing) =====')
    for rg, side in (('b0', 'short'), ('d0', 'long')):
        for hz in ('h5', 'h30', 'eod'):
            d = sel(s, side, rg, 0.0, hz)
            line = f'{side:5s} {rg:4s} {hz:4s}'
            for sp in ('TRAIN', 'VAL', 'TEST'):
                ds = d[d.split == sp]
                if ds.empty:
                    continue
                x = np.sort(ds['net_R'].values)
                ex5 = np.mean(x[:int(len(x) * 0.95)]) if len(x) > 20 else np.nan
                line += f' | {sp} n={len(ds)} m={ds.net_R.mean():+.3f} ex5={ex5:+.3f}'
            print(line)
    e = sel(s, 'short', 'b0', 0.0, 'eod')
    print('\nNO-STOP eod, top-5 contributors per split:')
    for sp in ('TRAIN', 'VAL', 'TEST'):
        es = e[e.split == sp]
        if es.empty:
            continue
        top = es.nlargest(5, 'net_R')
        print(f'  {sp}: total={es.net_R.sum():+.1f}R  top5={top.net_R.sum():+.1f}R '
              f'({top.net_R.sum()/es.net_R.sum()*100 if es.net_R.sum() else 0:.0f}%) '
              f'| {", ".join(f"{r.symbol} {r.day} {r.net_R:+.1f}" for r in top.itertuples())}')

    # ---------------- L5 / L3 sensitivity arms on the best-by-TRAIN cell per side ----------------
    print('\n===== SENSITIVITY ARMS on the best-by-TRAIN cell of each side =====')
    for side in ('short', 'long'):
        sub = cells[cells.side == side].sort_values('TRAIN_mean', ascending=False).iloc[0]
        cfg = dict(side=side, rung=sub.rung, r=sub.R, hz=sub.hz,
                   gate=None if sub.gate == 'none' else float(sub.gate))
        print(f'\n  best {side}: rung={sub.rung} gate={sub.gate} R={sub.R} hz={sub.hz} '
              f'TRAIN {sub.TRAIN_mean:+.3f} (t {sub.TRAIN_t:.2f})')
        for lab, fl in (('none', None), ('price>=$10', {'reopen': 10}), ('price>=$20', {'reopen': 20}),
                        ('ADV20>=500K', {'adv20': 500_000}), ('ADV20>=1M', {'adv20': 1_000_000})):
            d = sel(s, cfg['side'], cfg['rung'], cfg['r'], cfg['hz'], cfg['gate'], fl)
            rt, rv = row(d, 'TRAIN'), row(d, 'VAL')
            print(f'    L5 {lab:12s} TRAIN n={rt["n"]:4d} {rt["mean"]:+.3f} (t {rt["t"]:.2f}) '
                  f'| VAL n={rv["n"]:4d} {rv["mean"]:+.3f} (t {rv["t"]:.2f})')
        d = sel(s, cfg['side'], cfg['rung'], cfg['r'], cfg['hz'], cfg['gate'])
        for sp in ('TRAIN', 'VAL'):
            ds = d[d.split == sp]
            if ds.empty:
                continue
            spa = np.where(ds.free_exit, ds['med_spread_pct'], ds['sp_pt'])
            am = (ds['gross_R'] - np.where(ds.free_exit, 1.0, ds['coef']) *
                  0.5 * spa / ds['r_scale'])
            print(f'    L3 {sp}: passive-target {ds.net_R.mean():+.3f} vs all-marketable '
                  f'{am.mean():+.3f}  (target leg = {ds.free_exit.mean()*100:.0f}% of exits)')

    # ---------------- capacity ----------------
    cap = s[(s.r_pct == 6.0) & (s.hz == 'h5')].copy()
    cap['shares'] = 0.01 * cap['entry_vol']
    cap['notional'] = cap['shares'] * cap['fill_px']
    print('\n===== CAPACITY (1% of the resume-minute volume) =====')
    print(cap.groupby(['side', 'rung'])[['shares', 'notional']].median().round(0).to_string())
    return 0


if __name__ == '__main__':
    sys.exit(main())
