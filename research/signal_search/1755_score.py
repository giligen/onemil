"""Cell 1,755 scorer (PREREG_1755): SHORT the gapper at the open (MOO), cover at close (MOC), hard stop at open*(1+s)+50bp.
Data loading and event construction copied from 1754_score.py; changes: universe rank<=2000 & raw px>=$5, adj HIGH matrix, short sign, stop/borrow reads.
Original 1754 docstring follows.


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
TOPN, MIN_PX, VOLX, RNG_MIN = 2000, 5.0, 1.5, 0.05
GAPS = (0.10, 0.15, 0.20)
STOPS = (0.10, 0.15, 0.25)
STOP_SLIP = 0.005
BORROW = (0, 25, 50, 100)
TIERS = {'5-20': (5, 20), '>20': (20, 1e9)}
RTIERS = {'r1-1000': (1, 1000), 'r1001-2000': (1001, 2000)}


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
    M = {k: np.full((D, S), np.nan, dtype=np.float16 if k in ('Rng', 'Hrel', 'Oraw') else np.float32) for k in ('Oadj', 'Cadj', 'Hrel', 'Rng', 'Oraw', 'Craw', 'Vraw')}  # float16 for Rng (vs 5 %), Hrel (high/open-1, 0.1 % rel. precision) and Oraw (3 pp scale check) to fit the 2.5 GB VA cap with the extra high matrix
    for c, tag in paths:
        pa_, pr_ = (pq.ParquetFile(os.path.join(c, f)) for f in ('bars_all.parquet', 'bars_raw.parquet'))
        assert pa_.metadata.num_row_groups == pr_.metadata.num_row_groups
        for g in range(pr_.metadata.num_row_groups):
            ra = pa_.read_row_group(g, use_threads=False).to_pandas(); rr = pr_.read_row_group(g, use_threads=False).to_pandas()
            # inner join on (date, symbol) via integer keys + searchsorted (same result as the pandas key merge, a fraction of the RAM)
            ka = didx.get_indexer(ra.date).astype(np.int64) * S + sidx.get_indexer(ra.symbol)
            kr = didx.get_indexer(rr.date).astype(np.int64) * S + sidx.get_indexer(rr.symbol)
            order = np.argsort(kr, kind='stable'); krs = kr[order]
            pos = np.clip(np.searchsorted(krs, ka), 0, len(krs) - 1); hit = krs[pos] == ka
            ra = ra[hit].reset_index(drop=True); rrm = rr.iloc[order[pos[hit]]].reset_index(drop=True)
            m = ra.join(rrm[['o', 'c', 'v']].add_suffix('_r'))
            del rrm, ka, kr, order, krs, pos, hit
            di, si = didx.get_indexer(m.date), sidx.get_indexer(m.symbol)
            if tag == 'b':   # 1746 only fills holes (1748 wins)
                keep = np.isnan(M['Craw'][di, si]); di, si, m = di[keep], si[keep], m[keep]
            M['Oadj'][di, si] = m.o; M['Cadj'][di, si] = m.c; M['Hrel'][di, si] = (m.h / m.o - 1).to_numpy(np.float32); M['Oraw'][di, si] = m.o_r; M['Craw'][di, si] = m.c_r
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
        hi_rel = M['Hrel'][sl].astype(np.float32)
        di, si = np.where(ev)
        parts.append(pd.DataFrame({'date': dates_a[sl][di], 'symbol': syms_a[si], 'gap': gap[di, si], 'r': r[di, si], 'hi': hi_rel[di, si], 'U': U[di],
                                   'px': M['Craw'][p1][di, si], 'rank': rank[p1][di, si].astype(int), 'vol_ratio': vr[di, si],
                                   'rng_prev': M['Rng'][p1][di, si], 'ret_prev': (M['Cadj'][p1] / M['Cadj'][p2] - 1)[di, si]}))
        Us.append(U); nUs.append(ok.sum(axis=1)); ds.append(dates_a[sl])
    log(f'universe name-days {tot_ok:,}; scale-check (adj vs raw gap >3pp or open-close >1pp) drops {tot_bad:,} ({tot_bad/tot_ok:.2%})')
    E = pd.concat(parts, ignore_index=True)
    E['ex'] = E.r - E.U
    E['short_none'] = -E.r
    for s_ in STOPS:      # daily bars cannot time the stop: charged at the level open*(1+s) plus slip, covered there if the high reached it
        E[f'short_s{int(s_*100)}'] = np.where(E.hi >= s_, -((1 + s_) * (1 + STOP_SLIP) - 1), -E.r)
        E[f'hit_s{int(s_*100)}'] = (E.hi >= s_).astype(int)
    E['year'] = E.date.str[:4].astype(int); E['month'] = E.date.str[:7]
    days = pd.DataFrame({'date': np.concatenate(ds), 'U': np.concatenate(Us), 'nU': np.concatenate(nUs)})
    days = days[days.nU > 0]
    log(f'events (gap>=10%, attention) {len(E):,}; max |r| {E.r.abs().max():.2f}; trading days {len(days)}; universe names/day median {days.nU.median():.0f}')
    big = E[(E.r > 5) | (E.r < -0.8) | (E.hi > 5)]
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


def half_spread():
    """Measured entry+exit half-spread (bp) from the quote sample: mean / median / p75 of (entry_hs + exit_hs)."""
    c = pd.read_csv(os.path.join(HERE, '1748_cost.csv'))
    tot = (c.entry_hs_bp + c.exit_hs_bp)
    out = {'mean': tot.mean(), 'median': c.entry_hs_bp.median() + c.exit_hs_bp.median(), 'p75': tot.quantile(.75)}
    log(f'half-spread sample n={len(c)} (1748 extreme-loser name-days, ranks 1001-3000): entry+exit bp {out}')
    return out



def stop_col(s):
    """Return-column name for stop s (None = no-stop book)."""
    return 'short_none' if s is None else f'short_s{int(s * 100)}'


def select(E, split, g, tier=None, rtier=None):
    """Events of `split` with gap >= g (+ optional raw-price tier / rank tier)."""
    a, b = SPLITS[split]
    s = E[(E.date >= a) & (E.date <= b) & (E.gap >= g)]
    if tier:
        s = s[(s.px >= TIERS[tier][0]) & (s.px < TIERS[tier][1])]
    if rtier:
        s = s[(s['rank'] >= RTIERS[rtier][0]) & (s['rank'] <= RTIERS[rtier][1])]
    return s


def read(E, days, split, g, stop, tier=None, rtier=None, hs=None):
    """One summary row: short book, gross (-r or stop-capped) and net of 2x2 bp legs and borrow rows 0/25/50/100 bp, tails, excess vs shorting the universe."""
    a, b = SPLITS[split]
    s = select(E, split, g, tier, rtier)
    nd = ((days.date >= a) & (days.date <= b)).sum()
    row = dict(split=split, gap=g, stop='none' if stop is None else stop, tier=tier or 'all', rtier=rtier or 'all', n=len(s),
               n_wk=len(s) / (nd / 5.0), n_days_with=s.date.nunique())
    if len(s) < 5:
        return row
    gross = s[stop_col(stop)].values; d = s.date.values
    row.update(gross_bp=gross.mean() * 1e4, hit_rate=(s[f'hit_s{int(stop*100)}'].mean() if stop else np.nan), win_rate=(gross > 0).mean())
    for bo in BORROW:
        net = gross - 2 * LEG_BP / 1e4 - bo / 1e4
        mu, t, ti, se = cl_t(net, d)
        row.update({f'net_b{bo}_bp': mu * 1e4, f't_b{bo}': t, f'tiid_b{bo}': ti, f'extop5_b{bo}_bp': ex_top5(net) * 1e4})
        if bo == 50:
            cm, ct, _, _ = cl_t(np.minimum(net, 0.20), d)
            srt = np.sort(net); k5 = max(1, int(np.ceil(.05 * len(net)))); k1 = max(1, int(np.ceil(.01 * len(net))))
            pday = pd.Series(net).groupby(d).mean().sort_values()
            p5 = max(1, int(np.ceil(.05 * len(pday)))); p1 = max(1, int(np.ceil(.01 * len(pday))))
            em, et, _, _ = cl_t(net + s.U.values, d)
            row.update(capped20_b50_bp=cm * 1e4, capped_t=ct, q01_b50=srt[k1 - 1], q05_b50=srt[k5 - 1], tail5_mean_b50=srt[:k5].mean(),
                       tail1_mean_b50=srt[:k1].mean(), pday_worst=pday.iloc[0], pday_tail1_mean=pday.iloc[:p1].mean(), pday_tail5_mean=pday.iloc[:p5].mean(),
                       exU_b50_bp=em * 1e4, exU_t=et, green_month=(pd.Series(net).groupby(s.month.values).mean() > 0).mean(), n_months=s.month.nunique(),
                       mde80_b50_bp=2.8 * se * 1e4)
            if hs:
                row.update({f'net_hs_{k}_b50_bp': (net.mean() - v / 1e4) * 1e4 for k, v in hs.items()})
    return row


def feature_table(E, split, col, cuts=None):
    """Mean net(b50) of the primary book by tercile of each causal feature; TRAIN supplies the cut points; plus winners-vs-losers medians."""
    s = select(E, split, .15, '5-20').copy(); s['net'] = s[col] - 4e-4 - 50e-4
    feats = ['gap', 'px', 'vol_ratio', 'rng_prev', 'ret_prev', 'rank', 'hi']
    new = {} if cuts is None else cuts; rows = []
    for f in feats:
        c = new.get(f) or list(s[f].quantile([1 / 3, 2 / 3])); new[f] = c
        for k, (lo, hi_) in enumerate([(-np.inf, c[0]), (c[0], c[1]), (c[1], np.inf)]):
            q = s[(s[f] > lo) & (s[f] <= hi_)]
            m, t, _, _ = cl_t(q.net.values, q.date.values) if len(q) > 5 else (np.nan,) * 4
            rows.append(dict(split=split, feature=f, tercile=k + 1, lo=lo, hi=hi_, n=len(q), net_b50_bp=m * 1e4, t=t, win_rate=(q.net > 0).mean() if len(q) else np.nan))
    return pd.DataFrame(rows), new, s.assign(win=s.net > 0).groupby('win')[feats].median()


def verdict(tr, va, yr):
    """Apply the PREREG kill list to the primary cell (g 15 %, $5-20, s 15 %, borrow 50 bp)."""
    def pick(df, split, g=.15, stop=.15, tier='5-20', rtier='all'):
        return df[(df.split == split) & (df.gap == g) & (df.stop == stop) & (df.tier == tier) & (df.rtier == rtier)].iloc[0]
    v = pick(va, 'VAL'); h1 = pick(tr, 'T_H1_16_18'); h2 = pick(tr, 'T_H2_19_20')
    y22 = yr[yr.rtier == '2022'].iloc[0]
    k = {'VAL_net50_le_50bp_or_t_lt_2': bool(v.net_b50_bp <= 50 or v.t_b50 < 2),
         'TRAIN_halves_opposite_sign': bool(np.sign(h1.net_b50_bp) != np.sign(h2.net_b50_bp)),
         'VAL_extop5_le_0': bool(v.extop5_b50_bp <= 0), '2022_negative': bool(y22.net_b50_bp < 0),
         'lt_3_names_per_wk': bool(v.n_wk < 3), 'tail5_mean_worse_than_-8pct': bool(v.tail5_mean_b50 < -0.08)}
    for g in (.15, .20):
        r = pick(va, 'VAL', g=g, stop=.15, tier=None) if False else va[(va.split == 'VAL') & (va.gap == g) & (va.stop == .15) & (va.tier == 'all') & (va.rtier == 'all')].iloc[0]
        k[f'net_b100_le_0_g{int(g*100)}'] = bool(r.net_b100_bp <= 0)
    k['KILL'] = any(k.values())
    return k


def main():
    dates, syms, M = load_matrices()
    E, days = build_events(dates, syms, M)
    del M
    assert E.date.max() < SEAL, 'TEST leaked'
    hs = half_spread()
    stops = [None] + list(STOPS)
    def block(split):
        rows = []
        for g in GAPS:
            for st in stops:
                for tier in (None, '5-20', '>20'):
                    rows.append(read(E, days, split, g, st, tier, hs=hs))
                for rt in RTIERS:
                    rows.append(read(E, days, split, g, st, '5-20', rt, hs=hs))
        return pd.DataFrame(rows)
    tr = pd.concat([block(sp) for sp in ('T_H1_16_18', 'T_H2_19_20', 'TRAIN')], ignore_index=True)
    tr.to_csv(os.path.join(HERE, '1755_train.csv'), index=False)
    ptr = tr[(tr.split == 'TRAIN') & (tr.gap == .15) & (tr.stop == .15) & (tr.tier == '5-20') & (tr.rtier == 'all')].iloc[0]
    json.dump({'primary': 'g15%, $5-20, s15%, borrow 50bp, short MOO/MOC', 'TRAIN_net_b50_bp': float(ptr.net_b50_bp), 'TRAIN_t': float(ptr.t_b50), 'frozen': True},
              open(os.path.join(HERE, '1755_train_pick.json'), 'w'))
    cols = ['split', 'gap', 'stop', 'tier', 'n', 'n_wk', 'hit_rate', 'gross_bp', 'net_b0_bp', 'net_b50_bp', 't_b50', 'net_b100_bp', 'extop5_b50_bp', 'tail5_mean_b50']
    log(tr[tr.rtier == 'all'][cols].to_string())
    ft_tr, cuts, wl_tr = feature_table(E, 'TRAIN', 'short_s15')
    log('--- TRAIN pick written; reading VAL once ---')
    va = block('VAL')
    yr = []
    for y, q in E[(E.gap >= .15) & (E.px >= 5) & (E.px < 20)].groupby('year'):
        for st in stops:
            net = q[stop_col(st)].values - 54e-4
            m, t, _, _ = cl_t(net, q.date.values)
            yr.append(dict(split='YEAR', gap=.15, stop='none' if st is None else st, tier='5-20', rtier=str(y), n=len(q), net_b50_bp=m * 1e4, t_b50=t, gross_bp=(net.mean() + 54e-4) * 1e4))
    yr = pd.DataFrame(yr)
    yr15 = yr[yr.stop == .15]
    ft_va, _, wl_va = feature_table(E, 'VAL', 'short_s15', cuts)
    out = pd.concat([tr, va, yr], ignore_index=True)
    out.to_csv(os.path.join(HERE, '1755_summary.csv'), index=False)
    pd.concat([ft_tr, ft_va]).to_csv(os.path.join(HERE, '1755_features.csv'), index=False)
    log(va[va.rtier == 'all'][cols].to_string())
    log(va[va.rtier != 'all'][['gap', 'stop', 'rtier', 'n', 'net_b50_bp', 't_b50']].query('stop == 0.15').to_string())
    log('primary VAL full row'); log(va[(va.gap == .15) & (va.stop == .15) & (va.tier == '5-20') & (va.rtier == 'all')].iloc[0].to_string())
    log('primary VAL no-stop row'); log(va[(va.gap == .15) & (va.stop == 'none') & (va.tier == '5-20') & (va.rtier == 'all')].iloc[0].to_string())
    log('years primary s15, then no-stop'); log(yr15[['rtier', 'n', 'net_b50_bp', 't_b50']].to_string()); log(yr[yr.stop == 'none'][['rtier', 'n', 'net_b50_bp', 't_b50']].to_string())
    k = verdict(tr, va, yr15)
    json.dump(k, open(os.path.join(HERE, '1755_verdict.json'), 'w'), indent=1); log(json.dumps(k, indent=1))
    log('feature terciles'); log(pd.concat([ft_tr, ft_va])[['split', 'feature', 'tercile', 'lo', 'hi', 'n', 'net_b50_bp', 't', 'win_rate']].to_string())
    log('winners vs losers medians TRAIN'); log(wl_tr.to_string()); log('VAL'); log(wl_va.to_string())


if __name__ == '__main__':
    main()
