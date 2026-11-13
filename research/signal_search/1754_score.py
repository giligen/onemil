"""Cell 1,754 scorer (PREREG_1754): auction-to-auction gapper day trade, MOO at open_t, MOC at close_t, long only.

Universe at t-1: rank <= 3,000 by 20-day median raw dollar volume (rank over every symbol in the cache), raw close_{t-1} >= $1.
Gap = adj open_t / adj close_{t-1} - 1 >= g; attention = prior-day volume >= 1.5x its prior-20-day mean OR prior-day range >= 5 %.
Return = adj close_t / adj open_t - 1.  Excess = return minus the equal-weight open-to-close of the whole t-1 universe that day.
Cost: 2 bp per leg, 0 spread (auction) ; sensitivity = + measured half-spread (1748_cost.csv, entry + exit).
TEST (>= 2024-01-01) is dropped at the panel level and never scored.  TRAIN reads + pick file are written BEFORE VAL is touched.
Usage: cd /home/ec2-user/onemil && ulimit -v 2500000 && nice -n 19 timeout 600 python3 research/signal_search/1754_score.py
"""
import os, json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
pa.set_cpu_count(1); pa.set_io_thread_count(1)

HERE = os.path.dirname(os.path.abspath(__file__))
C46, C48 = os.path.join(HERE, '1746_cache'), os.path.join(HERE, '1748_cache')
SEAL = '2024-01-01'
SPLITS = {'T_H1_16_18': ('2016-01-01', '2018-12-31'), 'T_H2_19_20': ('2019-01-01', '2020-12-31'),
          'TRAIN': ('2016-01-01', '2020-12-31'), 'VAL': ('2021-01-01', '2023-12-31')}
LEG_BP = 2.0           # per leg, auction fills pay no quoted spread
TOPN, MIN_PX, VOLX, RNG_MIN = 3000, 1.0, 1.5, 0.05
GAPS = (0.05, 0.10, 0.20)
TIERS = {'1-5': (1, 5), '5-20': (5, 20), '>20': (20, 1e9), '1-20': (1, 20)}
RTIERS = {'r1-1000': (1, 1000), 'r1001-2000': (1001, 2000), 'r2001-3000': (2001, 3000)}


def log(m):
    print(m, flush=True)


def load_matrices():
    """Pivot both caches into D x S float32 matrices: Oadj, Cadj, Rng (prior-day adj range/close), Oraw, Craw, Vraw."""
    paths = [(C48, 'a'), (C46, 'b')]
    syms, dates = set(), set()
    for c, _ in paths:
        p = pq.ParquetFile(os.path.join(c, 'bars_raw.parquet'))
        for g in range(p.metadata.num_row_groups):
            t = p.read_row_group(g, columns=['symbol', 'date'])
            syms |= set(t.column('symbol').unique().to_pylist()); dates |= set(t.column('date').unique().to_pylist())
    syms, dates = sorted(syms), sorted(dates)
    sidx, didx = pd.Index(syms), pd.Index(dates)
    D, S = len(dates), len(syms)
    log(f'matrices: {D} dates x {S} symbols')
    M = {k: np.full((D, S), np.nan, dtype=np.float32) for k in ('Oadj', 'Cadj', 'Rng', 'Oraw', 'Craw', 'Vraw')}
    for c, tag in paths:
        pa_, pr_ = (pq.ParquetFile(os.path.join(c, f)) for f in ('bars_all.parquet', 'bars_raw.parquet'))
        assert pa_.metadata.num_row_groups == pr_.metadata.num_row_groups
        for g in range(pr_.metadata.num_row_groups):
            ra = pa_.read_row_group(g, use_threads=False).to_pandas(); rr = pr_.read_row_group(g, use_threads=False).to_pandas()
            m = ra.merge(rr, on=['symbol', 'date'], suffixes=('', '_r'))
            di, si = didx.get_indexer(m.date), sidx.get_indexer(m.symbol)
            if tag == 'b':   # 1746 only fills holes (1748 wins)
                keep = np.isnan(M['Craw'][di, si]); di, si, m = di[keep], si[keep], m[keep]
            M['Oadj'][di, si] = m.o; M['Cadj'][di, si] = m.c; M['Oraw'][di, si] = m.o_r; M['Craw'][di, si] = m.c_r
            M['Vraw'][di, si] = m.v_r; M['Rng'][di, si] = ((m.h - m.l) / m.c).to_numpy(np.float32)
            del ra, rr, m
        log(f'loaded {c}')
    return dates, syms, M


def roll_cols(a, fn, block=800):
    """Rolling-20 (min 15) median/mean of a D x S float32 matrix, column blocks to bound memory."""
    out = np.empty_like(a)
    for j in range(0, a.shape[1], block):
        r = pd.DataFrame(a[:, j:j + block]).rolling(20, min_periods=15)
        out[:, j:j + block] = (r.median() if fn == 'median' else r.mean()).to_numpy(np.float32)
    return out


def build_events(dates, syms, M):
    """Per-day universe mean open-to-close and the event table (all gaps >= 5 %), with scale checks and causal features.
    Everything dated 'prev' is row d-1 (= t-1); processed in day chunks to keep RSS small."""
    D, S = M['Craw'].shape
    rank = np.zeros((D, S), dtype=np.int16)
    dv20 = roll_cols(M['Craw'] * M['Vraw'], 'median')
    for i in range(D):          # rank 1 = largest 20d $ volume, nan -> 0 (ineligible)
        row = dv20[i]; ok_ = np.where(np.isfinite(row))[0]
        o = ok_[np.argsort(-row[ok_], kind='stable')]; rank[i, o] = np.arange(1, len(o) + 1)
    del dv20
    vmean = roll_cols(M['Vraw'], 'mean')       # vmean[d] = mean of volume d-19..d ; denominator for ratio at t uses vmean[t-2]
    dates_a, syms_a = np.array(dates), np.array(syms)
    parts, Us, nUs, ds = [], [], [], []
    tot_ok = tot_bad = 0
    for lo in range(2, D, 200):
        hi = min(D, lo + 200); sl = slice(lo, hi); p1 = slice(lo - 1, hi - 1); p2 = slice(lo - 2, hi - 2)
        gap = M['Oadj'][sl] / M['Cadj'][p1] - 1
        gap_raw = M['Oraw'][sl] / M['Craw'][p1] - 1
        r = M['Cadj'][sl] / M['Oadj'][sl] - 1
        r_raw = M['Craw'][sl] / M['Oraw'][sl] - 1
        ok = (rank[p1] >= 1) & (rank[p1] <= TOPN) & (M['Craw'][p1] >= MIN_PX) & np.isfinite(gap) & np.isfinite(r)
        bad = ok & ((np.abs(gap - gap_raw) > 0.03) | (np.abs(r - r_raw) > 0.01) | ~np.isfinite(gap_raw) | ~np.isfinite(r_raw))
        tot_ok += int(ok.sum()); tot_bad += int(bad.sum())
        ok &= ~bad
        ok[dates_a[sl] >= SEAL] = False                        # TEST sealed: never scored
        U = np.nanmean(np.where(ok, r, np.nan), axis=1) if ok.any() else np.full(hi - lo, np.nan)
        vr = M['Vraw'][p1] / vmean[p2]
        attn = (vr >= VOLX) | (M['Rng'][p1] >= RNG_MIN)
        ev = ok & attn & (gap >= GAPS[0])
        di, si = np.where(ev)
        parts.append(pd.DataFrame({'date': dates_a[sl][di], 'symbol': syms_a[si], 'gap': gap[di, si], 'r': r[di, si], 'U': U[di],
                                   'px': M['Craw'][p1][di, si], 'rank': rank[p1][di, si].astype(int), 'vol_ratio': vr[di, si],
                                   'rng_prev': M['Rng'][p1][di, si], 'ret_prev': (M['Cadj'][p1] / M['Cadj'][p2] - 1)[di, si]}))
        Us.append(U); nUs.append(ok.sum(axis=1)); ds.append(dates_a[sl])
    log(f'universe name-days {tot_ok:,}; scale-check (adj vs raw gap >3pp or open-close >1pp) drops {tot_bad:,} ({tot_bad/tot_ok:.2%})')
    E = pd.concat(parts, ignore_index=True)
    E['ex'] = E.r - E.U
    E['year'] = E.date.str[:4].astype(int); E['month'] = E.date.str[:7]
    days = pd.DataFrame({'date': np.concatenate(ds), 'U': np.concatenate(Us), 'nU': np.concatenate(nUs)})
    days = days[days.nU > 0]
    log(f'events (gap>=5%, attention) {len(E):,}; max |r| {E.r.abs().max():.2f}; trading days {len(days)}; universe names/day median {days.nU.median():.0f}')
    big = E[(E.r > 5) | (E.r < -0.8)]
    log(f'scale-check on trades: {len(big)} with r > +500% or < -80% (kept, listed): {big[["date", "symbol", "gap", "r"]].to_dict("records")[:10]}')
    return E, days


def cl_t(x, d):
    """(mean, day-clustered t, iid t, clustered SE) of x grouped by day label array d."""
    n = len(x)
    if n < 5:
        return np.nan, np.nan, np.nan, np.nan
    mu = x.mean(); e = pd.Series(x - mu).groupby(np.asarray(d)).sum()
    se = np.sqrt((e ** 2).sum()) / n
    return mu, mu / se if se > 0 else np.nan, mu / (x.std(ddof=1) / np.sqrt(n)), se


def ex_top5(x):
    """Mean after dropping the top 5 % of values."""
    x = np.sort(np.asarray(x)); k = int(np.ceil(0.05 * len(x)))
    return x[:len(x) - k].mean() if len(x) > k else np.nan


def read(E, days, split, g, tier=None, rtier=None, cost_hs=None):
    """One summary row for events of `split` with gap >= g (+ optional price/rank tier)."""
    a, b = SPLITS[split]
    s = E[(E.date >= a) & (E.date <= b) & (E.gap >= g)]
    if tier:
        s = s[(s.px >= TIERS[tier][0]) & (s.px < TIERS[tier][1])]
    if rtier:
        s = s[(s['rank'] >= RTIERS[rtier][0]) & (s['rank'] <= RTIERS[rtier][1])]
    nd = ((days.date >= a) & (days.date <= b)).sum()
    row = dict(split=split, gap=g, tier=tier or 'all', rtier=rtier or 'all', n=len(s), n_wk=len(s) / (nd / 5.0), n_days_with=s.date.nunique())
    if len(s) < 5:
        return row
    mu, t, ti, se = cl_t(s.ex.values, s.date.values)
    gm, gt, _, _ = cl_t(s.r.values, s.date.values)
    cap = np.minimum(s.r.values, 0.20) - s.U.values
    cm, ct, _, _ = cl_t(cap, s.date.values)
    row.update(gross_bp=gm * 1e4, gross_t=gt, ex_bp=mu * 1e4, ex_t=t, ex_t_iid=ti, ex_top5_bp=ex_top5(s.ex.values) * 1e4,
               ex_capped20_bp=cm * 1e4, ex_capped_t=ct, green_month=(s.groupby('month').ex.mean() > 0).mean(), n_months=s.month.nunique(),
               mde80_bp=2.8 * se * 1e4, win_rate=(s.r > 0).mean(), net0_bp=gm * 1e4 - 2 * LEG_BP)
    if cost_hs:
        row.update({f'net_hs_{k}_bp': gm * 1e4 - 2 * LEG_BP - v for k, v in cost_hs.items()})
        row.update({f'netex_hs_{k}_bp': mu * 1e4 - 2 * LEG_BP - v for k, v in cost_hs.items()})
    return row


def half_spread():
    """Measured entry+exit half-spread (bp) from the quote sample: mean / median / p75 of (entry_hs + exit_hs)."""
    c = pd.read_csv(os.path.join(HERE, '1748_cost.csv'))
    tot = (c.entry_hs_bp + c.exit_hs_bp)
    out = {'mean': tot.mean(), 'median': c.entry_hs_bp.median() + c.exit_hs_bp.median(), 'p75': tot.quantile(.75)}
    log(f'half-spread sample n={len(c)} (1748 extreme-loser name-days, ranks 1001-3000): entry+exit bp {out}')
    return out


def feature_table(E, split, cuts=None):
    """Mean excess by tercile of each causal feature, primary events of `split`; TRAIN supplies the cut points."""
    a, b = SPLITS[split]
    s = E[(E.date >= a) & (E.date <= b) & (E.gap >= .10) & (E.px >= 1) & (E.px < 20)]
    feats = ['gap', 'px', 'vol_ratio', 'rng_prev', 'ret_prev', 'rank']
    new = {} if cuts is None else cuts
    rows = []
    for f in feats:
        c = new.get(f) or list(s[f].quantile([1 / 3, 2 / 3]))
        new[f] = c
        for k, (lo, hi) in enumerate([(-np.inf, c[0]), (c[0], c[1]), (c[1], np.inf)]):
            q = s[(s[f] > lo) & (s[f] <= hi)]
            m, t, _, _ = cl_t(q.ex.values, q.date.values) if len(q) > 5 else (np.nan,) * 4
            rows.append(dict(split=split, feature=f, tercile=k + 1, lo=lo, hi=hi, n=len(q), ex_bp=m * 1e4, ex_t=t, win_rate=(q.r > 0).mean() if len(q) else np.nan))
    wl = s.assign(win=s.ex > 0).groupby('win')[feats].median()
    return pd.DataFrame(rows), new, wl


def main():
    dates, syms, M = load_matrices()
    E, days = build_events(dates, syms, M)
    del M
    assert E.date.max() < SEAL, 'TEST leaked'
    hs = half_spread()
    rows = []
    for split in ('T_H1_16_18', 'T_H2_19_20', 'TRAIN'):
        for g in GAPS:
            for tier in (None, '1-5', '5-20', '>20', '1-20'):
                rows.append(read(E, days, split, g, tier, cost_hs=hs))
        for rt in RTIERS:
            rows.append(read(E, days, split, .10, '1-20', rt, cost_hs=hs))
    tr = pd.DataFrame(rows)
    ft_tr, cuts, wl_tr = feature_table(E, 'TRAIN')
    tr.to_csv(os.path.join(HERE, '1754_train.csv'), index=False)
    prim = tr[(tr.split == 'TRAIN') & (tr.gap == .10) & (tr.tier == '1-20') & (tr.rtier == 'all')].iloc[0].to_dict()
    json.dump({'primary': 'g10%, px $1-20, excess open-to-close', 'TRAIN_ex_bp': prim['ex_bp'], 'TRAIN_t': prim['ex_t'], 'frozen': True},
              open(os.path.join(HERE, '1754_train_pick.json'), 'w'))
    log(tr[tr.rtier == 'all'][['split', 'gap', 'tier', 'n', 'n_wk', 'gross_bp', 'ex_bp', 'ex_t', 'ex_top5_bp', 'green_month']].to_string())
    log('--- TRAIN pick written; reading VAL once ---')
    rows = []
    for g in GAPS:
        for tier in (None, '1-5', '5-20', '>20', '1-20'):
            rows.append(read(E, days, 'VAL', g, tier, cost_hs=hs))
    for rt in RTIERS:
        rows.append(read(E, days, 'VAL', .10, '1-20', rt, cost_hs=hs))
    va = pd.DataFrame(rows)
    # year signs of the primary book, TRAIN + VAL
    yr = []
    P = E[(E.gap >= .10) & (E.px >= 1) & (E.px < 20)]
    for y, q in P.groupby('year'):
        m, t, _, _ = cl_t(q.ex.values, q.date.values)
        yr.append(dict(split='YEAR', gap=.10, tier='1-20', rtier=str(y), n=len(q), ex_bp=m * 1e4, ex_t=t, gross_bp=q.r.mean() * 1e4))
    ft_va, _, wl_va = feature_table(E, 'VAL', cuts)
    out = pd.concat([tr, va, pd.DataFrame(yr)], ignore_index=True)
    out.to_csv(os.path.join(HERE, '1754_summary.csv'), index=False)
    pd.concat([ft_tr, ft_va]).to_csv(os.path.join(HERE, '1754_features.csv'), index=False)
    log(va[va.rtier == 'all'].T.to_string()); log(va[va.rtier != 'all'][['rtier', 'n', 'ex_bp', 'ex_t']].to_string())
    log(pd.DataFrame(yr).to_string())
    log('feature terciles (TRAIN then VAL)'); log(pd.concat([ft_tr, ft_va])[['split', 'feature', 'tercile', 'lo', 'hi', 'n', 'ex_bp', 'ex_t']].to_string())
    log('winners vs losers medians TRAIN'); log(wl_tr.to_string()); log('VAL'); log(wl_va.to_string())


if __name__ == '__main__':
    main()
