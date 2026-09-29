#!/usr/bin/env python3
"""Adversarial refuter checks for cell 1,625 (PREREG_1623.md, 'the index as the instrument').

Report-only. Reads the builder's artefacts (cell_1625_signals.csv, index_bars_1625.parquet), the fill
population (causal_arming_causal.csv + features_1478_A.csv arm_m) and data/cache.db READ-ONLY. Checks:
  A. bar timestamps / source splice (UTC vs ET, DST, open/close auction alignment vs daily_bars)
  B. independent recompute of the builder's per-day SPY/IWM returns
  C. signal-time distribution; D. results excluding signals before 10:00
  E. time-of-day-matched placebo (same minute, other days of the same holdout)
  F. the builder's uniform random-minute placebo: draw replication + late-minute defect
  G. signal-minute causality: arms counted at m* whose fill is not yet known at the entry instant;
     a strictly causal re-run (fill-minute floor) and a +1-minute-delay sensitivity
  H. tails and day concentration; I. tau semantics (ties, afternoon zero mass)
Writes review/1623_index_refuter_chk.json.
"""
import json
import math
import os
import sqlite3

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

HERE = '/home/ec2-user/onemil/research/hod_entry'
CACHE_URI = 'file:/home/ec2-user/onemil/data/cache.db?mode=ro'
OUT_JSON = os.path.join(HERE, 'review', '1623_index_refuter_chk.json')
ET = ZoneInfo('America/New_York')
COST = 2.0
M0, M_LAST, CUT = 570, 959, 955
COLS = np.arange(M0, 966)  # 09:30 .. 16:05 (post-market lets an at-or-after lookup near the close resolve)
RES = {}


def say(*a):
    """Print with flush (long-ish script, keep progress visible)."""
    print(*a, flush=True)


def stats(y):
    """n, mean, iid t (== day-clustered t: one signal per day), ex-top-5 % mean (drop ceil(5 % n) best)."""
    y = pd.Series(y, dtype=float).dropna().to_numpy()
    n = len(y)
    if n < 3:
        return dict(n=n, mean=float(np.mean(y)) if n else None, t=None, ex5=None)
    mean = y.mean()
    t = mean / (y.std(ddof=1) / math.sqrt(n))
    k = int(math.ceil(0.05 * n))
    ex5 = np.sort(y)[:-k].mean()
    return dict(n=n, mean=round(float(mean), 2), t=round(float(t), 2), ex5=round(float(ex5), 2))


def load_fills():
    """Builder-identical fill population: status==fill joined to features_1478_A arm_m."""
    c = pd.read_csv(os.path.join(HERE, 'causal_arming_causal.csv'), low_memory=False)
    c = c[c.status == 'fill'][['day', 'symbol', 'split', 'fill_min']]
    a = pd.read_csv(os.path.join(HERE, 'features_1478_A.csv'), usecols=['day', 'symbol', 'fill_min', 'arm_m'])
    m = c.merge(a, on=['day', 'symbol', 'fill_min'], how='inner')
    m['holdout'] = np.where(m.split == 'VAL', 'VAL', 'TRAIN-H2')
    say(f'fills {len(c)} joined {len(m)}')
    return m


def open_matrix(bars, symbol, days):
    """(day x minute) matrix of RTH opens, at-or-after lookup within 5 minutes (builder's obtainability rule)."""
    b = bars[(bars.symbol == symbol) & (bars.m >= M0) & (bars.m <= COLS[-1])]
    piv = b.pivot_table(index='d_et', columns='m', values='o', aggfunc='first')
    piv = piv.reindex(index=days, columns=COLS)
    return piv.bfill(axis=1, limit=5)


def ret_matrix(O, hold):
    """R[d, m] = net bps of buy open(m+1) -> open(min(m+hold, 15:55)); NaN when the hold is empty/backward."""
    R = pd.DataFrame(np.nan, index=O.index, columns=np.arange(M0, M_LAST))
    for m in range(M0, M_LAST):
        e, x = m + 1, min(m + hold, CUT)
        if x <= e:
            continue
        R[m] = (O[x] / O[e] - 1.0) * 1e4 - COST
    return R


def grid_counts(times, window=30):
    """B(t) for t in 570..959 = #{times in (t-window, t]} (times already integer minutes)."""
    grid = np.arange(M0, M_LAST + 1)
    arr = np.sort(np.asarray(times))
    return grid, np.searchsorted(arr, grid, side='right') - np.searchsorted(arr, grid - window, side='right')


def run_variant(name, fills, time_col, R60, day_hold, entry_shift=0):
    """Builder construction on a chosen event-time column: TRAIN-H2 pooled-grid p90 tau, first crossing m*."""
    by_day = {}
    for d, sub in fills.groupby('day'):
        grid, cnt = grid_counts(np.floor(sub[time_col].to_numpy()).astype(int))
        by_day[d] = cnt
    pooled = np.concatenate([v for d, v in by_day.items() if day_hold[d] == 'TRAIN-H2'])
    tau = float(np.percentile(pooled, 90))
    rows = []
    for d in sorted(by_day):
        hit = np.nonzero(by_day[d] >= tau)[0]
        if len(hit) == 0:
            continue
        ms = int(grid[hit[0]]) + entry_shift
        rows.append(dict(day=d, split=day_hold[d], m_star=ms, r=R60.at[d, ms] if ms in R60.columns else np.nan))
    df = pd.DataFrame(rows)
    out = dict(tau=tau, share_ge_tau=round(float((pooled >= tau).mean()), 4),
               share_gt_tau=round(float((pooled > tau).mean()), 4),
               share_zero=round(float((pooled == 0).mean()), 4))
    for h in ('TRAIN-H2', 'VAL'):
        s = df[df.split == h]
        out[h] = stats(s.r)
        out[h]['median_m'] = float(s.m_star.median()) if len(s) else None
        out[h]['ge10'] = stats(s[s.m_star >= 600].r)
    say(name, json.dumps(out))
    return out, df


def main():
    sig = pd.read_csv(os.path.join(HERE, 'cell_1625_signals.csv'))
    fills = load_fills()
    day_hold = fills.drop_duplicates('day').set_index('day').holdout.to_dict()
    days = sorted(day_hold)

    # ---------------- A. timestamps / source splice
    bars = pd.read_parquet(os.path.join(HERE, 'index_bars_1625.parquet'),
                           columns=['symbol', 'day', 't', 'o', 'c', 'v', 'source'])
    ts = pd.to_datetime(bars['t'], utc=True).dt.tz_convert(ET)
    bars['m'] = ts.dt.hour * 60 + ts.dt.minute
    bars['d_et'] = ts.dt.strftime('%Y-%m-%d')
    A = dict(day_label_mismatch=int((bars.d_et != bars.day).sum()))
    rth = bars[(bars.m >= 540) & (bars.m <= 1000)]
    for (sym, src), sub in rth.groupby(['symbol', 'source']):
        am = sub.loc[sub.groupby('d_et').v.idxmax(), 'm']
        A[f'{sym}_{src}'] = dict(days=int(sub.d_et.nunique()), rows=int(len(sub)),
                                 first_m_min=int(sub.groupby('d_et').m.min().min()),
                                 maxvol_bar_at_0930_or_close_share=round(float(am.isin([570, 959]).mean()), 3),
                                 maxvol_m_values=sorted(set(int(x) for x in am))[:8])
    con = sqlite3.connect(CACHE_URI, uri=True)
    A['cache_raw_ts_sample'] = [r[0] for r in con.execute(
        "select timestamp from intraday_bars_1min where symbol='SPY' order by bar_date limit 2")]
    dly = pd.read_sql_query("select symbol, bar_date, open, close from daily_bars where symbol in ('SPY','IWM')", con)
    con.close()
    dly['day'] = dly.bar_date.astype(str).str.slice(0, 10)
    for sym in ('SPY', 'IWM'):
        b = bars[bars.symbol == sym]
        o930 = b[b.m == 570].set_index('d_et').o
        c359 = b[b.m == 959].set_index('d_et').c
        src = b.drop_duplicates('d_et').set_index('d_et').source
        dd = dly[dly.symbol == sym].set_index('day')
        j = pd.DataFrame({'o930': o930, 'c359': c359, 'src': src}).join(dd[['open', 'close']], how='inner')
        j = j[j.index.isin(days)]
        j['do_bps'] = (j.o930 / j.open - 1) * 1e4
        j['dc_bps'] = (j.c359 / j.close - 1) * 1e4
        for s, g in j.groupby('src'):
            A[f'{sym}_{s}_vs_daily'] = dict(n=int(len(g)), open_med_abs_bps=round(float(g.do_bps.abs().median()), 2),
                                           open_max_abs_bps=round(float(g.do_bps.abs().max()), 2),
                                           close_med_abs_bps=round(float(g.dc_bps.abs().median()), 2),
                                           close_max_abs_bps=round(float(g.dc_bps.abs().max()), 2),
                                           n_open_gt_20bps=int((g.do_bps.abs() > 20).sum()))
    RES['A_timestamps'] = A
    say('A', json.dumps(A))

    # ---------------- B. independent recompute
    O = {s: open_matrix(bars, s, days) for s in ('SPY', 'IWM')}
    R60 = {s: ret_matrix(O[s], 61) for s in O}
    B = {}
    for s, col in (('SPY', 'spy_ret_bps_60'), ('IWM', 'iwm_ret_bps_60')):
        mine = np.array([R60[s].at[d, m] for d, m in zip(sig.day, sig.m_star)])
        B[s] = dict(max_abs_diff=round(float(np.nanmax(np.abs(mine - sig[col]))), 4),
                    n_nan_mine=int(np.isnan(mine).sum()))
    sig['spy_R'] = [R60['SPY'].at[d, m] for d, m in zip(sig.day, sig.m_star)]
    sig['iwm_R'] = [R60['IWM'].at[d, m] for d, m in zip(sig.day, sig.m_star)]
    RES['B_recompute'] = B
    say('B', B)

    # ---------------- C/D. signal-time distribution; excluding < 10:00
    bins = [569, 584, 599, 629, 659, 719, 960]
    labels = ['09:30-09:44', '09:45-09:59', '10:00-10:29', '10:30-10:59', '11:00-11:59', '12:00+']
    sig['bucket'] = pd.cut(sig.m_star, bins=bins, labels=labels)
    C = {h: sig[sig.split == h].bucket.value_counts().reindex(labels).fillna(0).astype(int).to_dict()
         for h in ('TRAIN-H2', 'VAL')}
    D = {}
    for h in ('TRAIN-H2', 'VAL'):
        s = sig[sig.split == h]
        D[h] = dict(all_spy=stats(s.spy_R), all_iwm=stats(s.iwm_R),
                    pre10_spy=stats(s[s.m_star < 600].spy_R), pre10_iwm=stats(s[s.m_star < 600].iwm_R),
                    ge10_spy=stats(s[s.m_star >= 600].spy_R), ge10_iwm=stats(s[s.m_star >= 600].iwm_R),
                    by_bucket_spy={b: stats(g.spy_R)['mean'] for b, g in s.groupby('bucket', observed=True)})
    RES['C_signal_time'] = C
    RES['D_excl_pre10'] = D
    say('C', C)
    say('D', json.dumps(D))

    # ---------------- E. time-of-day-matched placebo (same minute, every other day of the same holdout)
    E = {}
    rng = np.random.default_rng(1625)
    for s, col in (('SPY', 'spy_R'), ('IWM', 'iwm_R')):
        for h in ('TRAIN-H2', 'VAL'):
            hd = [d for d in days if day_hold[d] == h]
            sub = sig[sig.split == h]
            base, rnd = [], []
            for d, m in zip(sub.day, sub.m_star):
                others = [x for x in hd if x != d]
                base.append(np.nanmean(R60[s].loc[others, m].to_numpy()))
                rnd.append(R60[s].at[others[int(rng.integers(0, len(others)))], m])
            mg = sub[col].to_numpy() - np.array(base)
            mr = sub[col].to_numpy() - np.array(rnd)
            E[f'{s}_{h}'] = dict(real=stats(sub[col])['mean'], tod_baseline=round(float(np.nanmean(base)), 2),
                                 margin_vs_tod=stats(mg), margin_vs_random_other_day=stats(mr))
        # unconditional hold-60 mean by entry hour, all days of the holdout (what a uniform placebo averages)
        for h in ('TRAIN-H2', 'VAL'):
            hd = [d for d in days if day_hold[d] == h]
            Rh = R60[s].loc[hd]
            E[f'{s}_{h}_uncond_by_hour'] = {f'{hh:02d}': round(float(np.nanmean(
                Rh[[m for m in Rh.columns if hh * 60 <= m < hh * 60 + 60 and m <= 954]].to_numpy())), 2)
                for hh in range(9, 16)}
    RES['E_tod_placebo'] = E
    say('E', json.dumps(E))

    # ---------------- F. builder placebo: replicate draws; late-minute defect
    rng = np.random.default_rng(1625)
    sig_days = set(sig.day)
    draws = {d: int(rng.integers(M0, M_LAST)) for d in sorted(day_hold) if d in sig_days}
    rep_ok = bool(all(draws[d] == m for d, m in zip(sig.day, sig.placebo_m)))
    late = sig[sig.placebo_m >= 954]
    F = dict(draws_replicate=rep_ok, n_placebo_m_ge_954=int(len(late)),
             late_rows=late[['day', 'split', 'placebo_m', 'placebo_ret_bps']].to_dict('records'),
             placebo_m_median=float(sig.placebo_m.median()))
    for h in ('TRAIN-H2', 'VAL'):
        s = sig[(sig.split == h)]
        ok = s[s.placebo_m < 954]
        F[h] = dict(margin_builder=stats(s.spy_R - s.placebo_ret_bps),
                    margin_excl_late=stats(ok.spy_R - ok.placebo_ret_bps),
                    margin_iwm_builder=stats(s.iwm_R - s.placebo_ret_bps_iwm))
    RES['F_builder_placebo'] = F
    say('F', json.dumps(F, default=str))

    # ---------------- G. signal-minute causality
    G = {}
    unk = []
    for d, m in zip(sig.day, sig.m_star):
        f = fills[(fills.day == d) & (fills.arm_m > m - 30) & (fills.arm_m <= m)]
        n_unknown = int((f.fill_min >= m + 1).sum())
        unk.append(dict(n=len(f), n_unknown_at_entry=n_unknown, n_arm_at_mstar=int((f.arm_m == m).sum())))
    u = pd.DataFrame(unk)
    sig['n_unknown'] = u.n_unknown_at_entry.to_numpy()
    G['days_with_unknown_fill_in_count'] = int((u.n_unknown_at_entry > 0).sum())
    G['days_where_count_minus_unknown_below_tau'] = int(((u.n - u.n_unknown_at_entry) < 7).sum())
    G['lag_fill_minus_arm_ge1_share'] = round(float(((fills.fill_min - fills.arm_m) >= 1).mean()), 4)
    G['builder_reproduced'], _ = run_variant('builder(arm_m)', fills, 'arm_m', R60['SPY'], day_hold)
    G['causal_fill_floor'], dfc = run_variant('causal(floor fill_min)', fills, 'fill_min', R60['SPY'], day_hold)
    G['delay_plus1min'], _ = run_variant('arm_m, entry +1 min', fills, 'arm_m', R60['SPY'], day_hold, entry_shift=1)
    RES['G_causality'] = G

    # ---------------- H. tails / day concentration (VAL)
    H = {}
    for col in ('spy_R', 'iwm_R'):
        for h in ('TRAIN-H2', 'VAL'):
            s = sig[sig.split == h].copy()
            y = s[col].sort_values(ascending=False)
            s['month'] = s.day.str.slice(0, 7)
            H[f'{col}_{h}'] = dict(sum=round(float(y.sum()), 1), top5_days=[(s.at[i, 'day'], round(float(y[i]), 1))
                                                                             for i in y.index[:5]],
                                   bottom5_days=[(s.at[i, 'day'], round(float(y[i]), 1)) for i in y.index[-5:]],
                                   top5_sum=round(float(y.iloc[:5].sum()), 1),
                                   mean_ex_top1=round(float(y.iloc[1:].mean()), 2),
                                   mean_ex_top5=round(float(y.iloc[5:].mean()), 2),
                                   by_month={k: round(float(v), 2) for k, v in s.groupby('month')[col].mean().items()})
    s = sig[sig.split == 'VAL']
    mg = (s.spy_R - s.placebo_ret_bps).sort_values(ascending=False)
    H['VAL_builder_margin_top5_sum'] = round(float(mg.iloc[:5].sum()), 1)
    H['VAL_builder_margin_sum'] = round(float(mg.sum()), 1)
    H['VAL_builder_margin_ex_top5days'] = stats(mg.iloc[5:])
    RES['H_tails'] = H
    say('H', json.dumps(H, default=str))

    with open(OUT_JSON, 'w') as fh:
        json.dump(RES, fh, indent=1, default=str)
    say('wrote', OUT_JSON)


if __name__ == '__main__':
    main()
