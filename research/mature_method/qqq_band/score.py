#!/usr/bin/env python3
"""Candidate #2 — QQQ noise band — the ten steps of research/mature_method/RUNBOOK.md.

Stages (argv):
  legs   : build B0 and dump every leg's (date, minute, side, modelled fill) for the NBBO pull
  score  : the whole study (needs cost_nbbo.csv if present; falls back to 0.5 bp with a WARNING)

Read-only on research/lit_review_2026/etf_1min.db.  Writes only into this directory.
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/fuckup_audit/Q')
import zsim as Z                                                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)) + '/'
N_MIN = Z.N_MIN
CHECKS = Z.CHECKS_SEMI
SPLITS = {'TRAIN': ('2016-01-01', '2023-12-31'),
          'VAL':   ('2024-01-01', '2025-12-31'),
          'TEST':  ('2026-01-01', '2026-12-31')}
AUM = 60_000.0
SEED = 20260919


def log(m):
    print(m, flush=True)


# --------------------------------------------------------------------------------------------
def sim3(data, UB, LB, checks=CHECKS, fill='next_open', eod='moc', slip_bp=0.0,
         last_check=999, first_check=0, stop_mode='vwap_band', flat_k=None):
    """zsim.simulate + (last_check, first_check, stop_mode, flat_k).  Identical otherwise.

    stop_mode: 'vwap_band' = max(UB,VWAP)/min(LB,VWAP) (the published rule)
               'band_only' = UB/LB only (the paper's BASE model)
               'none'      = no intraday stop; the position runs to the flat
    flat_k:    force the position flat at the open of bar flat_k+1 (None = run to the EOD rule)
    """
    C, O, VW, last = data['C'], data['O'], data['VW'], data['last']
    D = C.shape[0]
    s = slip_bp * 1e-4
    trades = []

    def leg(px, side):
        return px * (1.0 + s) if side > 0 else px * (1.0 - s)

    for d in range(D):
        ld = last[d]
        if np.isnan(UB[d, 30]) or ld < 60:
            continue
        pos = 0; entry = np.nan; ek = -1
        for k in checks:
            if k >= ld:
                break
            px = C[d, k]
            ub, lb, vw = UB[d, k], LB[d, k], VW[d, k]
            if np.isnan(px) or np.isnan(ub) or np.isnan(vw):
                continue
            if fill == 'next_open':
                fp = O[d, k + 1]; fk = k + 1
                if np.isnan(fp):
                    fp = px; fk = k
            else:
                fp = px; fk = k
            stopped_side = 0
            if stop_mode != 'none':
                lstop = max(ub, vw) if stop_mode == 'vwap_band' else ub
                sstop = min(lb, vw) if stop_mode == 'vwap_band' else lb
                if pos == 1 and px < lstop:
                    trades.append(dict(d=d, side=1, e=entry, x=leg(fp, -1), ek=ek, xk=fk, why='stop'))
                    pos = 0; stopped_side = 1
                elif pos == -1 and px > sstop:
                    trades.append(dict(d=d, side=-1, e=entry, x=leg(fp, 1), ek=ek, xk=fk, why='stop'))
                    pos = 0; stopped_side = -1
            if pos == 0 and first_check <= k <= last_check:
                if px > ub and stopped_side != 1:
                    pos = 1; entry = leg(fp, 1); ek = fk
                elif px < lb and stopped_side != -1:
                    pos = -1; entry = leg(fp, -1); ek = fk
            if flat_k is not None and k >= flat_k and pos != 0:
                fp2 = O[d, k + 1] if (k + 1) < N_MIN and not np.isnan(O[d, k + 1]) else px
                trades.append(dict(d=d, side=pos, e=entry, x=leg(fp2, -pos), ek=ek,
                                   xk=min(k + 1, ld), why='flat'))
                pos = 0
                break
        if pos != 0:
            if eod == 'open' and ld == N_MIN - 1 and not np.isnan(O[d, ld]):
                xp, xk = O[d, ld], ld
            else:
                xp, xk = C[d, ld], ld
            trades.append(dict(d=d, side=pos, e=entry, x=leg(xp, -pos), ek=ek, xk=xk, why='eod'))
    return trades


def day_frame(data, trades, cost_bp_leg=0.0):
    """Per-day $/share P&L and 1x return.  cost_bp_leg charged on BOTH legs, on top of slippage."""
    D = len(data['days'])
    pnl = np.zeros(D); ntr = np.zeros(D, dtype=int)
    for tr in trades:
        e, x, side = tr['e'], tr['x'], tr['side']
        c = cost_bp_leg * 1e-4 * (e + x)
        pnl[tr['d']] += side * (x - e) - c
        ntr[tr['d']] += 1
    valid = ~np.isnan(data['sigma'][:, 30]) & ~np.isnan(data['sig14'])
    df = pd.DataFrame({'date': data['days'], 'r1x': pnl / data['dopen'], 'ntr': ntr,
                       'valid': valid})
    return df[df['valid']].reset_index(drop=True)


# --------------------------------------------------------------------------------------------
def weekly(df, aum=AUM):
    """Week table over EVERY market week in df (no-trade week = flat, kept in the denominator)."""
    d = df.copy()
    d['wk'] = pd.to_datetime(d['date']).dt.to_period('W-FRI')
    g = d.groupby('wk').agg(r=('r1x', 'sum'), ntr=('ntr', 'sum'), nd=('r1x', 'size'))
    g['usd'] = g['r'] * aum
    return g


def red_streak(g):
    best = cur = 0
    for v in g['usd'].values:
        cur = cur + 1 if v <= 0 else 0
        best = max(best, cur)
    return best


def mdd_usd(df, aum=AUM):
    c = np.cumsum(df['r1x'].values) * aum
    return float((np.maximum.accumulate(np.r_[0.0, c]) - np.r_[0.0, c]).max())


def cell_stats(df, aum=AUM):
    r = df['r1x'].values
    tr = r[df['ntr'].values > 0] * 1e4
    n, nd = len(tr), len(r)
    sd = tr.std(ddof=1) if n > 2 else np.nan
    t = tr.mean() / sd * math.sqrt(n) if n > 2 and sd > 0 else np.nan
    mde = 2.80 * sd / math.sqrt(n) if n > 2 else np.nan          # 80% power, two-sided 5%
    g = weekly(df, aum)
    m = df.copy(); m['mo'] = pd.to_datetime(m['date']).dt.to_period('M')
    mo = m.groupby('mo')['r1x'].sum()
    return dict(days=nd, traded=n, tr_wk=n / max(len(g), 1), bps=tr.mean() if n else np.nan,
                t=t, mde=mde, hit=(tr > 0).mean() * 100 if n else np.nan,
                weeks=len(g), green_wk=100.0 * (g['usd'] > 0).mean(),
                streak=red_streak(g), worst_wk=g['usd'].min(), best_wk=g['usd'].max(),
                total=float(df['r1x'].sum() * aum), mdd=mdd_usd(df, aum),
                green_mo=100.0 * (mo > 0).mean(), months=len(mo))


def null_band(df, draws=2000, aum=AUM, seed=SEED):
    """Shuffle the cell's own daily P&L across its own weeks, per-week TRADED-DAY count fixed."""
    d = df.copy()
    d['wk'] = pd.to_datetime(d['date']).dt.to_period('W-FRI')
    pool = d.loc[d['ntr'] > 0, 'r1x'].values
    counts = d.groupby('wk', sort=True)['ntr'].apply(lambda s: int((s > 0).sum())).values
    if len(pool) == 0:
        return (np.nan, np.nan, np.nan)
    idx = np.cumsum(counts)
    lo = np.r_[0, idx[:-1]]
    rng = np.random.default_rng(seed)
    out = np.empty(draws)
    for i in range(draws):
        p = rng.permutation(pool)
        sums = np.array([p[a:b].sum() for a, b in zip(lo, idx)])
        out[i] = 100.0 * (sums > 0).mean()
    return float(out.mean()), float(np.percentile(out, 5)), float(np.percentile(out, 95))


# --------------------------------------------------------------------------------------------
def cells(data):
    """(id, label, kwargs for sim3, band kwargs)."""
    EVERY15 = list(range(15, 376, 15))
    EVERY60 = list(range(30, 331, 60))
    EVERY1 = list(range(30, 385, 1))
    c = [
        ('B0', 'shipped spec (VM 1.0, semi-hourly 10:00-15:30, VWAP/band stop, flat 15:59 close)',
         {}, {}),
        ('V1', 'VM 0.8', {}, dict(vm=0.8)),
        ('V2', 'VM 1.2', {}, dict(vm=1.2)),
        ('V3', 'VM 1.5', {}, dict(vm=1.5)),
        ('A1', 'anchor = open only', {}, dict(anchor='open')),
        ('T1', 'cadence 15 min', dict(checks=EVERY15), {}),
        ('T2', 'cadence 60 min', dict(checks=EVERY60), {}),
        ('T3', 'cadence 1 min (ceiling)', dict(checks=EVERY1), {}),
        ('W1', 'entries only <= 12:00', dict(last_check=150), {}),
        ('W2', 'entries only <= 13:30', dict(last_check=240), {}),
        ('W3', 'first decision 11:00', dict(first_check=90, checks=[k for k in CHECKS if k >= 90]), {}),
        ('H1', 'stop = opposite band only', dict(stop_mode='band_only'), {}),
        ('H2', 'no stop, hold to flat', dict(stop_mode='none'), {}),
        ('H3', 'flat at 15:30 (no 15:30 entry)', dict(flat_k=360, last_check=330), {}),
        ('H4', 'flat at the 15:59 OPEN', dict(eod='open'), {}),
        ('C1', 'VM 1.2 + entries <= 12:00', dict(last_check=150), dict(vm=1.2)),
        ('C2', 'VM 1.2 + flat at the 15:59 open', dict(eod='open'), dict(vm=1.2)),
        ('C3', 'cadence 1 min + VM 0.8 (max-frequency corner)', dict(checks=EVERY1), dict(vm=0.8)),
    ]
    return c


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else 'score'
    data = Z.load_symbol('QQQ')
    days = pd.to_datetime(data['days'])

    # ---- availability audit -------------------------------------------------------------
    av = dict(days_in_store=int(len(days)), dropped_short_days=int(data['n_dropped']),
              splits_adjusted=int(data['n_split']))
    for f, arr in (('sigma[k=30]', data['sigma'][:, 30]), ('sig14', data['sig14']),
                   ('prevclose', data['prevclose']), ('VWAP[k=30]', data['VW'][:, 30]),
                   ('C[k=30]', data['C'][:, 30])):
        av[f'missing_{f}'] = int(np.isnan(arr).sum())
    bars = (~np.isnan(data['C'])).sum(axis=1)
    av['days_with_390_bars'] = int((bars == 390).sum())
    av['median_bars_per_day'] = int(np.median(bars))
    log('AVAILABILITY ' + json.dumps(av))

    UB0, LB0 = Z.bands(data, vm=1.0)

    # ---- parity of sim3 against zsim.simulate at the default knobs ----------------------
    a = sim3(data, UB0, LB0, slip_bp=0.5)
    b, _ = Z.simulate(data, UB0, LB0, fill='next_open', eod='moc', slip_bp=0.5)
    same = len(a) == len(b) and all(
        x['d'] == y['d'] and x['side'] == y['side'] and abs(x['e'] - y['e']) < 1e-12
        and abs(x['x'] - y['x']) < 1e-12 for x, y in zip(a, b))
    log(f'PARITY sim3 vs zsim.simulate: {len(a)} vs {len(b)} trades, identical={same}')

    if stage == 'legs':
        rows = []
        for tr in sim3(data, UB0, LB0, slip_bp=0.0):
            rows.append(dict(date=str(days[tr['d']])[:10], k=tr['ek'], side=tr['side'],
                             px=tr['e'], leg='entry'))
            rows.append(dict(date=str(days[tr['d']])[:10], k=tr['xk'], side=-tr['side'],
                             px=tr['x'], leg='exit'))
        pd.DataFrame(rows).to_csv(HERE + 'legs.csv', index=False)
        log(f'legs.csv written: {len(rows)} legs')
        return

    # ---- measured cost ------------------------------------------------------------------
    cost_path = HERE + 'cost_nbbo.csv'
    if os.path.exists(cost_path):
        q = pd.read_csv(cost_path)
        q = q[q['cost_bp'].notna()]
        # charged cost = the measured NBBO HALF-SPREAD (pay half the spread from mid, per leg).
        # The direct measurement -- what a marketable order would pay ON TOP of the modelled
        # bar-open fill -- is logged beside it and is smaller; the half-spread is the conservative
        # of the two and is the one charged.
        COST = float(q['half_bp'].mean())
        log(f'MEASURED COST n={len(q)} CHARGED half-spread mean={COST:.4f} bp/leg '
            f'(median {q["half_bp"].median():.4f}, p90 {q["half_bp"].quantile(0.9):.4f}); '
            f'direct ask-minus-fill mean={q["cost_bp"].mean():+.4f} median={q["cost_bp"].median():+.4f} '
            f'p90={q["cost_bp"].quantile(0.9):+.4f}')
    else:
        COST = 0.5
        log('WARNING cost_nbbo.csv missing -- falling back to the 0.5 bp/leg ASSUMPTION of Q/H')

    # ---- score every cell ----------------------------------------------------------------
    out = []
    daybooks = {}
    for cid, label, kw, bkw in cells(data):
        UB, LB = Z.bands(data, vm=bkw.get('vm', 1.0), anchor=bkw.get('anchor', 'open_prevclose'))
        trg = sim3(data, UB, LB, slip_bp=0.0, **kw)
        dg = day_frame(data, trg, 0.0)
        dn = day_frame(data, trg, COST)
        d05 = day_frame(data, trg, 0.5)
        daybooks[cid] = dn
        for sp, (a, b) in SPLITS.items():
            m = (dn['date'] >= a) & (dn['date'] <= b)
            if m.sum() == 0:
                continue
            sg, sn = cell_stats(dg[m.values]), cell_stats(dn[m.values])
            s05 = cell_stats(d05[m.values])
            out.append(dict(cell=cid, label=label, split=sp, days=sn['days'], traded=sn['traded'],
                            tr_wk=sn['tr_wk'], gross_bps=sg['bps'], net_bps=sn['bps'],
                            net05_bps=s05['bps'], t=sn['t'], mde=sn['mde'], hit=sn['hit'],
                            weeks=sn['weeks'], green_wk=sn['green_wk'], streak=sn['streak'],
                            worst_wk=sn['worst_wk'], best_wk=sn['best_wk'], total=sn['total'],
                            mdd=sn['mdd'], green_mo=sn['green_mo'], months=sn['months']))
        log(f'  scored {cid}')
    R = pd.DataFrame(out)
    R.to_csv(HERE + 'cells.csv', index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 40):
        log('\n=== CELLS (TRAIN/VAL; TEST rows present but NOT to be read unless G1&G2 pass) ===')
        log(R[R.split != 'TEST'][['cell', 'split', 'traded', 'tr_wk', 'gross_bps', 'net_bps',
                                  'net05_bps', 't', 'mde', 'green_wk', 'streak', 'worst_wk',
                                  'total', 'mdd', 'green_mo']].round(3).to_string(index=False))

    # ---- null ---------------------------------------------------------------------------
    nl = []
    for cid in daybooks:
        for sp in ('TRAIN', 'VAL'):
            a, b = SPLITS[sp]
            d = daybooks[cid]
            d = d[(d['date'] >= a) & (d['date'] <= b)]
            if len(d) == 0:
                continue
            mu, p5, p95 = null_band(d)
            obs = cell_stats(d)['green_wk']
            nl.append(dict(cell=cid, split=sp, obs=obs, null_mean=mu, p5=p5, p95=p95,
                           outside='ABOVE' if obs > p95 else ('BELOW' if obs < p5 else 'inside')))
    NL = pd.DataFrame(nl)
    NL.to_csv(HERE + 'nulls.csv', index=False)
    log('\n=== NULL ===')
    log(NL.round(2).to_string(index=False))

    # ---- tails on B0 ---------------------------------------------------------------------
    log('\n=== TAIL (B0, net at the measured cost) ===')
    for sp in ('TRAIN', 'VAL'):
        a, b = SPLITS[sp]
        d = daybooks['B0']
        d = d[(d['date'] >= a) & (d['date'] <= b)].copy()
        r = d['r1x'].values * 1e4
        for lab, rr in (('full', r), ('ex-top-1%', np.sort(r)[:int(len(r) * 0.99)]),
                        ('ex-top-5%', np.sort(r)[:int(len(r) * 0.95)]),
                        ('capped +100bp', np.minimum(r, 100.0))):
            log(f'  {sp:5s} {lab:14s} {rr.mean():+7.3f} bps/calendar day  '
                f'${rr.mean() * 1e-4 * AUM * 21:+8.0f}/month')
        top5 = np.sort(r)[-5:].sum() / r.sum() * 100 if r.sum() != 0 else np.nan
        log(f'  {sp:5s} top-5 days = {top5:.0f}% of the split total')

    # ---- additivity ----------------------------------------------------------------------
    log('\n=== ADDITIVITY ===')
    orb = pd.read_csv('/home/ec2-user/onemil/research/orb_gates2/book_G3_meas.csv',
                      usecols=['date', '_sized_pnl'])
    orb['date'] = pd.to_datetime(orb['date'])
    bf = pd.read_csv('/home/ec2-user/onemil/research/bf_frequency/runs/VOL_OFF.csv',
                     usecols=['date', 'pnl'])
    bf['date'] = pd.to_datetime(bf['date']); bf['pnl'] *= 0.075
    lo, hi = pd.Timestamp('2025-01-02'), pd.Timestamp('2026-09-15')
    cal = daybooks['B0'][['date', 'r1x', 'ntr']].copy()
    cal = cal[(cal['date'] >= lo) & (cal['date'] <= hi)].copy()
    cal['wk'] = cal['date'].dt.to_period('W-FRI')
    W = cal.groupby('wk').agg(qqq1x=('r1x', 'sum')).reset_index()
    W['qqq1x'] *= AUM
    W['qqq2x'] = W['qqq1x'] * 2
    for nm, src, col in (('orb', orb, '_sized_pnl'), ('bf', bf, 'pnl')):
        s = src[(src['date'] >= lo) & (src['date'] <= hi)].copy()
        s['wk'] = s['date'].dt.to_period('W-FRI')
        W = W.merge(s.groupby('wk')[col].sum().rename(nm).reset_index(), on='wk', how='left')
    W[['orb', 'bf']] = W[['orb', 'bf']].fillna(0.0)
    W['live'] = W['orb'] + W['bf']
    for lev in ('qqq1x', 'qqq2x'):
        W['comb_' + lev] = W['live'] + W[lev]
    W.to_csv(HERE + 'additivity_weeks.csv', index=False)

    def wk_line(col):
        v = W[col].values
        return (f'n={len(v)} green={100 * (v > 0).mean():.1f}% total=${v.sum():+,.0f} '
                f'worst=${v.min():+,.0f} best=${v.max():+,.0f} '
                f'streak={red_streak(pd.DataFrame({"usd": v}))} '
                f'mdd=${(np.maximum.accumulate(np.r_[0, np.cumsum(v)]) - np.r_[0, np.cumsum(v)]).max():,.0f}')
    for c in ('orb', 'bf', 'live', 'qqq1x', 'qqq2x', 'comb_qqq1x', 'comb_qqq2x'):
        log(f'  {c:12s} {wk_line(c)}')
    log('  corr(week $): ' + json.dumps({
        'qqq1x~orb': round(float(W['qqq1x'].corr(W['orb'])), 3),
        'qqq1x~bf': round(float(W['qqq1x'].corr(W['bf'])), 3),
        'qqq1x~live': round(float(W['qqq1x'].corr(W['live'])), 3),
        'orb~bf': round(float(W['orb'].corr(W['bf'])), 3)}))
    red = W[W['live'] <= 0]
    log(f'  live-red weeks n={len(red)}: turned green by qqq1x {int((red["comb_qqq1x"] > 0).sum())}, '
        f'by qqq2x {int((red["comb_qqq2x"] > 0).sum())}')
    grn = W[W['live'] > 0]
    log(f'  live-green weeks n={len(grn)}: turned red by qqq1x {int((grn["comb_qqq1x"] <= 0).sum())}, '
        f'by qqq2x {int((grn["comb_qqq2x"] <= 0).sum())}')

    # ---- the weekly path, last two quarters ---------------------------------------------
    log('\n=== WEEKLY PATH (B0 net, $ at 1x / 2x on $60K), 2026-04-01 -> end of store ===')
    tail = W[W['wk'].astype(str) >= '2026-03-30']
    log('  ' + ' '.join(f'{v:+.0f}' for v in tail['qqq1x'].values))
    log(f'  {len(tail)} weeks, green {100 * (tail["qqq1x"] > 0).mean():.0f}%, '
        f'total ${tail["qqq1x"].sum():+,.0f} (1x) / ${tail["qqq2x"].sum():+,.0f} (2x)')


if __name__ == '__main__':
    main()
