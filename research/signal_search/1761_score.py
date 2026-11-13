"""Cell 1,761 scorer (PREREG_1761): first close above the prior 252-day (primary) / 20-day high, daily bars.

Universe at t-1: rank <= 3,000 by 20-day median raw dollar volume, raw close_{t-1} >= $3.
Event at close t: adjusted close_t > max adjusted close over t-252..t-1 (20-d: t-20..t-1), no such close in the prior 20 days,
(volume_t >= 1.5 x mean volume t-20..t-1 for the with-volume read).  Entry = open t+1, exit = close t+h (h = 5, 20), long.
Return = adj close_{t+h} / adj open_{t+1} - 1 (raw open/close used for the corporate-action scale check: |adj-raw| > 3 pp dropped).
Excess = return minus the equal-weight same-horizon return of the whole t-1 universe.  Cost 2 bp/leg (auction, 0 spread);
sensitivity = + measured exit half-spread (1748_cost.csv exit_hs_bp).  TEST (>= 2024) is cut from the matrices at load time.
An event is kept only if its exit date lies inside its own split.  TRAIN table is written before VAL is read.
Usage: cd /home/ec2-user/onemil && ulimit -v 2500000 && nice -n 19 timeout 600 python3 research/signal_search/1761_score.py
"""
import os
import sys
import importlib.util
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('s1754', os.path.join(HERE, '1754_score.py'))
S54 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S54)          # loader, roll_cols, cl_t, ex_top5 reused read-only

SEAL = '2024-01-01'
SPLITS = {'T_H1_16_18': ('2016-01-01', '2018-12-31'), 'T_H2_19_20': ('2019-01-01', '2020-12-31'),
          'TRAIN': ('2016-01-01', '2020-12-31'), 'VAL': ('2021-01-01', '2023-12-31')}
SPLIT_END = {'TRAIN': '2020-12-31', 'VAL': '2023-12-31'}
TOPN, MIN_PX, VOLX, LEG_BP = 3000, 3.0, 1.5, 2.0
HORIZONS = (5, 20)
RTIERS = {'all': (1, 3000), 'r1-1000': (1, 1000), 'r1001-3000': (1001, 3000)}
PTIERS = {'p3-10': (3, 10), 'p10-30': (10, 30), 'p>30': (30, 1e9)}
log = S54.log


def load_matrices():
    """Pivot both caches (1748 wins, 1746 fills holes) into D x S float32 matrices for dates < SEAL only (TEST never loaded):
    Oadj, Cadj, Oraw, Craw, Vraw.  Lighter than the 1754 loader (no range matrix, reads only needed columns)."""
    import pyarrow.parquet as pq
    paths = [(S54.C48, 'a'), (S54.C46, 'b')]
    syms, dates = set(), set()
    for c, _ in paths:
        p = pq.ParquetFile(os.path.join(c, 'bars_raw.parquet'))
        for g in range(p.metadata.num_row_groups):
            t = p.read_row_group(g, columns=['symbol', 'date'])
            syms |= set(t.column('symbol').unique().to_pylist()); dates |= set(t.column('date').unique().to_pylist())
    syms, dates = sorted(syms), sorted(d for d in dates if d < SEAL)
    sidx, didx = pd.Index(syms), pd.Index(dates)
    D, S = len(dates), len(syms)
    log(f'matrices: {D} dates ({dates[0]}..{dates[-1]}) x {S} symbols')
    M = {k: np.full((D, S), np.nan, dtype=np.float32) for k in ('Oadj', 'Cadj', 'Oraw', 'Craw', 'Vraw')}
    for c, tag in paths:
        pa_, pr_ = (pq.ParquetFile(os.path.join(c, f)) for f in ('bars_all.parquet', 'bars_raw.parquet'))
        assert pa_.metadata.num_row_groups == pr_.metadata.num_row_groups
        for g in range(pr_.metadata.num_row_groups):
            ra = pa_.read_row_group(g, columns=['symbol', 'date', 'o', 'c', 'v'], use_threads=False).to_pandas()
            rr = pr_.read_row_group(g, columns=['symbol', 'date', 'o', 'c', 'v'], use_threads=False).to_pandas()
            m = ra.merge(rr, on=['symbol', 'date'], suffixes=('', '_r'))
            m = m[m.date < SEAL]
            di, si = didx.get_indexer(m.date), sidx.get_indexer(m.symbol)
            if tag == 'b':
                keep = np.isnan(M['Craw'][di, si]); di, si, m = di[keep], si[keep], m[keep]
            M['Oadj'][di, si] = m.o; M['Cadj'][di, si] = m.c; M['Oraw'][di, si] = m.o_r; M['Craw'][di, si] = m.c_r; M['Vraw'][di, si] = m.v_r
            del ra, rr, m
        log(f'loaded {c}')
    return dates, syms, M


def high_events(C, win, block=600):
    """Bool D x S: close_t > max close over t-win..t-1 AND none of the previous 20 days was such a close."""
    D, S = C.shape
    ev = np.zeros((D, S), dtype=bool)
    mp = 200 if win == 252 else 15
    for j in range(0, S, block):
        c = pd.DataFrame(C[:, j:j + block])
        hi = c.rolling(win, min_periods=mp).max().shift(1).to_numpy(np.float32)
        above = (C[:, j:j + block] > hi)
        prev = pd.DataFrame(above.astype(np.float32)).rolling(20, min_periods=1).sum().shift(1).fillna(1).to_numpy()
        # require the full 20 prior days to exist and to be below
        ev[:, j:j + block] = above & (prev == 0) & (np.arange(D)[:, None] >= 20)
    return ev


def build_events(dates, syms, M):
    """Event table: one row per (event kind, symbol, signal date) with horizon returns, benchmark and causal flags."""
    D, S = M['Craw'].shape
    dates_a, syms_a = np.array(dates), np.array(syms)
    dv20 = S54.roll_cols(M['Craw'] * M['Vraw'], 'median')
    rank = np.zeros((D, S), dtype=np.int16)
    for i in range(D):
        row = dv20[i]; ok_ = np.where(np.isfinite(row))[0]
        o = ok_[np.argsort(-row[ok_], kind='stable')]; rank[i, o] = np.arange(1, len(o) + 1)
    del dv20
    vmean = S54.roll_cols(M['Vraw'], 'mean')
    volc = np.zeros((D, S), dtype=bool)
    volc[1:] = M['Vraw'][1:] >= VOLX * vmean[:-1]          # volume_t vs mean of t-20..t-1 (causal)
    del vmean
    ev = {252: high_events(M['Cadj'], 252), 20: high_events(M['Cadj'], 20)}
    log(f'raw events (no universe filter): 252d {int(ev[252].sum()):,}  20d {int(ev[20].sum()):,}')
    univ = np.zeros((D, S), dtype=bool)                    # universe membership AT t-1, stored on row t
    univ[1:] = (rank[:-1] >= 1) & (rank[:-1] <= TOPN) & (M['Craw'][:-1] >= MIN_PX)
    rank_p1 = np.zeros((D, S), dtype=np.int16); rank_p1[1:] = rank[:-1]
    del rank
    parts, Uday = [], {}
    tot_bad = tot_ok = 0
    for h in HORIZONS:
        U = np.full(D, np.nan)
        for i in range(1, D - h):
            r = M['Cadj'][i + h] / M['Oadj'][i + 1] - 1; rr = M['Craw'][i + h] / M['Oraw'][i + 1] - 1
            ok = univ[i] & np.isfinite(r)
            bad = ok & ((np.abs(r - rr) > 0.03) | ~np.isfinite(rr)); tot_ok += int(ok.sum()); tot_bad += int(bad.sum())
            ok &= ~bad
            if ok.any():
                U[i] = r[ok].mean()
            for k in (252, 20):
                m = ev[k][i] & ok
                si = np.where(m)[0]
                if len(si):
                    parts.append(pd.DataFrame({'date': dates_a[i], 'di': i, 'symbol': syms_a[si], 'kind': k, 'h': h, 'r': r[si],
                                               'vol': volc[i][si], 'rank': rank_p1[i][si].astype(int), 'px': M['Craw'][i][si],
                                               'exit_date': dates_a[i + h]}))
        Uday[h] = U
    E = pd.concat(parts, ignore_index=True)
    E['U'] = [Uday[h][i] for h, i in zip(E.h, E.di)]
    E['ex'] = E.r - E.U
    E['year'] = E.date.str[:4].astype(int); E['month'] = E.date.str[:7]; E['week'] = pd.to_datetime(E.date).dt.strftime('%G-%V')
    E = E[np.isfinite(E.U)]
    E['split'] = np.where(E.date <= SPLIT_END['TRAIN'], 'TRAIN', 'VAL')
    E = E[E.exit_date <= E.split.map(SPLIT_END)]            # exit inside own split
    ndays = {s: int(((dates_a >= a) & (dates_a <= b)).sum()) for s, (a, b) in SPLITS.items()}
    log(f'scale-check (adj vs raw return >3 pp) drops {tot_bad:,} of {tot_ok:,} ({tot_bad / tot_ok:.2%}); events kept {len(E):,}')
    log(f'max r {E.r.max():.2f} min r {E.r.min():.2f}; events with r>+500% or < -80%: {int(((E.r > 5) | (E.r < -0.8)).sum())}')
    return E, ndays


def exit_costs():
    """Measured exit half-spread (bp): mean / median / p75 from 1748_cost.csv (extreme-loser name-days, ranks 1001-3000)."""
    c = pd.read_csv(os.path.join(HERE, '1748_cost.csv')).exit_hs_bp
    out = {'mean': c.mean(), 'median': c.median(), 'p75': c.quantile(.75)}
    log(f'exit half-spread sample n={len(c)}: {out}')
    return out


def read(E, ndays, split, kind, h, vol, rt, pt=None, hs=None):
    """One summary row: events of `split` for (kind, horizon, volume flag, rank tier, price tier)."""
    a, b = SPLITS[split]
    s = E[(E.date >= a) & (E.date <= b) & (E.kind == kind) & (E.h == h)]
    if vol:
        s = s[s.vol]
    lo, hi = RTIERS[rt]; s = s[(s['rank'] >= lo) & (s['rank'] <= hi)]
    if pt:
        s = s[(s.px >= PTIERS[pt][0]) & (s.px < PTIERS[pt][1])]
    row = dict(split=split, kind=kind, h=h, vol=int(vol), rtier=rt, ptier=pt or 'all', n=len(s), n_wk=len(s) / (ndays[split] / 5.0))
    if len(s) < 10:
        return row
    mu, t, ti, se = S54.cl_t(s.ex.values, s.date.values)
    _, tw, _, _ = S54.cl_t(s.ex.values, s.week.values)
    gm, gt, _, _ = S54.cl_t(s.r.values, s.date.values)
    cm, ct, _, _ = S54.cl_t(np.minimum(s.r.values, 0.20) - s.U.values, s.date.values)
    row.update(gross_ret_bp=gm * 1e4, gross_ret_t=gt, ex_bp=mu * 1e4, ex_t=t, ex_t_iid=ti, ex_t_week=tw, ex_top5_bp=S54.ex_top5(s.ex.values) * 1e4,
               ex_capped20_bp=cm * 1e4, ex_capped_t=ct, green_month=(s.groupby('month').ex.mean() > 0).mean(), n_months=s.month.nunique(),
               mde80_bp=2.8 * se * 1e4, win_rate=(s.ex > 0).mean(), net0_bp=mu * 1e4 - 2 * LEG_BP)
    if hs:
        row.update({f'net_hs_{k}_bp': mu * 1e4 - 2 * LEG_BP - v for k, v in hs.items()})
    return row


def table(E, ndays, split, hs):
    """All reads for one split."""
    rows = []
    for kind in (252, 20):
        for h in HORIZONS:
            for vol in (True, False):
                for rt in RTIERS:
                    rows.append(read(E, ndays, split, kind, h, vol, rt, None, hs))
                for pt in PTIERS:
                    rows.append(read(E, ndays, split, kind, h, vol, 'r1001-3000', pt, hs))
    return pd.DataFrame(rows)


def main():
    dates, syms, M = load_matrices()
    E, ndays = build_events(dates, syms, M)
    del M
    assert E.date.max() < SEAL and E.exit_date.max() < SEAL, 'TEST leaked'
    hs = exit_costs()
    tr = pd.concat([table(E, ndays, s, hs) for s in ('T_H1_16_18', 'T_H2_19_20', 'TRAIN')], ignore_index=True)
    tr.to_csv(os.path.join(HERE, '1761_train.csv'), index=False)
    pr = (tr.kind == 252) & (tr.h == 5) & (tr.vol == 1) & (tr.rtier == 'r1001-3000') & (tr.ptier == 'all')
    log('--- TRAIN written; primary + n/wk by read ---')
    log(tr[pr][['split', 'n', 'n_wk', 'ex_bp', 'ex_t', 'ex_top5_bp', 'gross_ret_bp']].to_string())
    log(tr[tr.rtier != 'r1-1000'][tr.ptier == 'all'][['split', 'kind', 'h', 'vol', 'rtier', 'n', 'n_wk', 'ex_bp', 'ex_t']].to_string())
    log('--- reading VAL once ---')
    va = table(E, ndays, 'VAL', hs)
    yr = []
    P = E[(E.kind == 252) & (E.h == 5) & E.vol & (E['rank'] >= 1001)]
    for y, q in P.groupby('year'):
        m, t, _, _ = S54.cl_t(q.ex.values, q.date.values)
        yr.append(dict(split='YEAR', kind=252, h=5, vol=1, rtier='r1001-3000', ptier=str(y), n=len(q), ex_bp=m * 1e4, ex_t=t, gross_ret_bp=q.r.mean() * 1e4))
    out = pd.concat([tr, va, pd.DataFrame(yr)], ignore_index=True)
    out.to_csv(os.path.join(HERE, '1761_summary.csv'), index=False)
    cols = ['split', 'kind', 'h', 'vol', 'rtier', 'ptier', 'n', 'n_wk', 'ex_bp', 'ex_t', 'ex_t_week', 'ex_top5_bp', 'ex_capped20_bp', 'green_month', 'mde80_bp', 'net0_bp', 'net_hs_median_bp', 'net_hs_p75_bp']
    log(va[(va.kind == 252) & (va.h == 5) & (va.vol == 1) & (va.rtier == 'r1001-3000') & (va.ptier == 'all')].T.to_string())
    log(va[va.ptier == 'all'][cols[:9]].to_string())
    log(va[va.ptier != 'all'][cols[:9]].to_string())
    log(pd.DataFrame(yr)[['ptier', 'n', 'ex_bp', 'ex_t', 'gross_ret_bp']].to_string())


if __name__ == '__main__':
    main()
