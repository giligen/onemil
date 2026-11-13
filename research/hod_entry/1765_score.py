"""Cell 1,765 score: does what is known before 09:30 (pre-market tape F6, short interest E5, Reg SHO list E6) split the HOD book?

PREREG: research/hod_entry/PREREG_1765.md (frozen 2026-11-13 20:55 UTC).  Outcomes = the 1,763 HOD comparator trades
(1763_trades_{TRAIN,VAL}.parquet, read == 'HOD', r_pct >= 1.0 %), imported not re-simulated; fast loser = stop exit
<= 15 min after entry (1764_cache/fast_*.parquet).
Features (causal, all public before 09:30 of day t):
  pm_dv_ratio : sum(vw*v) over ET 04:00-09:29 / population adv20 (dollar).  Symbol absent from the day's pre-market
                file = no pre-market trade = 0 (day file exists for every population day).
  pm_bars     : # pre-market 1-min bars with trades.
  pm_gap      : pm high / previous-session raw close - 1 (NaN if no pre-market bar).
  dtc20       : short interest / ADV20 shares (mean daily volume of the 20 prior sessions), latest FINRA settlement whose
                usable_from (= settlement + 13 calendar days, conservative) is strictly before t.
  si_chg      : current / previous reported short position - 1 (same report).
  E6 Reg SHO threshold list: NOT ON DISK (1746_cache/finra = daily short VOLUME files, not threshold lists) -> VOID.
Quintile edges are fixed on TRAIN (ties collapse bins) and applied to H1/H2/TRAIN/VAL.  Filter read = keep the top two
bins; placebo = 200 random drops of the same count among the feature's valid trades.
TRAIN tables are written before VAL is read.
Usage: cd /home/ec2-user/onemil && ulimit -v 2500000 && nice -n 19 timeout 600 python3 research/hod_entry/1765_score.py
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'ARROW_IO_THREADS'):
    os.environ.setdefault(_v, '1')
import sys, time, resource, importlib.util
import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
HE = f'{ROOT}/research/hod_entry'
SI_CSV = f'{ROOT}/research/mature_method/hod_frames4/short_interest.csv'
sys.path.insert(0, ROOT)
sp = importlib.util.spec_from_file_location('s1763', f'{HE}/1763_score.py')
S3 = importlib.util.module_from_spec(sp); sp.loader.exec_module(S3)
S = S3.S
FLOOR = 1.0
FEATS = ['pm_dv_ratio', 'pm_bars', 'pm_gap', 'dtc20', 'si_chg']
N_PLACEBO = 200


def log(m):
    """Timestamped progress line with peak RSS."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f'[{time.strftime("%H:%M:%S")}] {m} (peak RSS {rss:.0f} MB)', flush=True)


def pm_features(pop):
    """F6 per symbol-day from the pre-market parquet files (ET 04:00-09:29)."""
    rows = []
    for day in sorted(pop.day.unique()):
        p = f'{HE}/1762_cache/premarket/{day}.parquet'
        if not os.path.exists(p):
            raise FileNotFoundError(f'pre-market file missing for {day}')
        b = pd.read_parquet(p)
        et = pd.to_datetime(b.t, utc=True).dt.tz_convert('America/New_York')
        mm = et.dt.hour * 60 + et.dt.minute
        b = b[(mm >= 240) & (mm <= 569) & (b.v > 0)].copy()
        b['dv'] = b.vw.fillna(b.c) * b.v
        g = b.groupby('symbol').agg(pm_dv=('dv', 'sum'), pm_bars=('v', 'size'), pm_high=('h', 'max'))
        g['day'] = day
        rows.append(g.reset_index())
    pm = pd.concat(rows, ignore_index=True)
    out = pop.merge(pm, on=['day', 'symbol'], how='left')
    out['pm_present'] = out.pm_bars.notna()
    out['pm_dv'] = out.pm_dv.fillna(0.0)
    out['pm_bars'] = out.pm_bars.fillna(0.0)
    out['pm_dv_ratio'] = out.pm_dv / out.adv20
    return out


def daily_features(f):
    """prev raw close (pm_gap) and ADV20 shares from daily.parquet; adds pm_gap, adv20_sh, prev_close, scale flag."""
    import sqlite3
    con = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=30)
    syms = sorted(f.symbol.unique())
    d = pd.concat([pd.read_sql_query(
        "select symbol, bar_date as date, close as c, volume as v from daily_bars where bar_date >= '2025-04-01' "
        f"and bar_date <= '2026-06-01' and symbol in ({','.join('?' * len(syms[i:i + 500]))})", con, params=syms[i:i + 500])
        for i in range(0, len(syms), 500)], ignore_index=True)
    con.close()
    d = d.sort_values(['symbol', 'date']).reset_index(drop=True)
    d['adv20_sh'] = d.groupby('symbol').v.transform(lambda s: s.rolling(20, min_periods=15).mean())
    chk = pd.read_parquet(f'{HE}/1762_cache/daily.parquet')[['symbol', 'date', 'c']].merge(d[['symbol', 'date', 'c']], on=['symbol', 'date'])
    log(f'daily_bars price-scale check vs 1762 daily: {len(chk)} overlap rows, |ratio-1| > 1 % in {(abs(chk.c_x / chk.c_y - 1) > 0.01).mean():.4f}')
    d['dts'] = pd.to_datetime(d.date)
    f = f.copy()
    f['dts'] = pd.to_datetime(f.day)
    f = f.sort_values('dts')
    # as-of join, exact match excluded: the latest session STRICTLY before t (<= 5 calendar days stale)
    m = pd.merge_asof(f, d.sort_values('dts')[['symbol', 'dts', 'c', 'adv20_sh']].rename(columns={'c': 'prev_close'}),
                      on='dts', by='symbol', allow_exact_matches=False, tolerance=pd.Timedelta(days=5)).sort_index()
    m['pm_gap'] = m.pm_high / m.prev_close - 1
    return m


def si_features(f):
    """E5: latest FINRA report public (usable_from < t) per symbol -> dtc20 and si_chg."""
    si = pd.read_csv(SI_CSV, usecols=['symbolCode', 'currentShortPositionQuantity', 'previousShortPositionQuantity',
                                      'settlementDate', 'usable_from'], dtype={'symbolCode': str}, keep_default_na=False)
    si['usable_from'] = pd.to_datetime(si.usable_from)
    si['settle'] = pd.to_datetime(si.settlementDate)
    si = si.sort_values('usable_from')
    f = f.copy()
    f['day_ts'] = pd.to_datetime(f.day)
    f = f.sort_values('day_ts')
    m = pd.merge_asof(f, si.rename(columns={'symbolCode': 'symbol'}), left_on='day_ts', right_on='usable_from',
                      by='symbol', allow_exact_matches=False)
    m['si_lag_days'] = (m.day_ts - m.settle).dt.days
    m['dtc20'] = m.currentShortPositionQuantity / m.adv20_sh
    prev = m.previousShortPositionQuantity.where(m.previousShortPositionQuantity > 0)
    m['si_chg'] = m.currentShortPositionQuantity / prev - 1
    m.loc[~np.isfinite(m.dtc20), 'dtc20'] = np.nan
    return m.sort_index()


def build_features():
    """All feature columns for the whole population -> 1765_features.parquet."""
    pop = pd.read_csv(f'{HE}/cell_1428_population.csv', dtype={'symbol': str}, keep_default_na=False, na_values=[''])
    f = si_features(daily_features(pm_features(pop)))
    keep = ['day', 'symbol', 'split', 'wk', 'adv20', 'pm_present', 'pm_dv', 'pm_dv_ratio', 'pm_bars', 'pm_high',
            'prev_close', 'pm_gap', 'adv20_sh', 'dtc20', 'si_chg', 'si_lag_days']
    f = f[keep].reset_index(drop=True)
    f.to_parquet(f'{HE}/1765_features.parquet', index=False)
    log(f'features {len(f)} rows; pm_present {f.pm_present.mean():.3f}; si lag median {f.si_lag_days.median()} d')
    return f


def load_trades(split, ft):
    """HOD comparator trades (R floor) joined to features and the fast-loser flag."""
    t = pd.read_parquet(f'{HE}/1763_trades_{split}.parquet')
    t = t[(t.read == 'HOD') & (t.r_pct >= FLOOR)].copy()
    fa = pd.read_parquet(f'{HE}/1764_cache/fast_{split}.parquet')
    t = t.merge(fa, on=['day', 'symbol'], how='left').merge(
        ft[['day', 'symbol', 'pm_present'] + FEATS], on=['day', 'symbol'], how='left')
    t['fastloser'] = ((t.why == 'stop') & (t.fast.fillna(0) == 1.0)).astype(float)
    return t


def bins_for(x, edges):
    """Bin index 0..len(edges) from TRAIN edges (ties collapsed; NaN stays NaN)."""
    x = np.asarray(x, float)
    b = np.searchsorted(edges, x, side='right').astype(float)
    b[np.isnan(x)] = np.nan
    return b


def edges_for(x):
    """Unique interior quintile edges from TRAIN values."""
    e = np.unique(np.nanquantile(np.asarray(x, float), [0.2, 0.4, 0.6, 0.8]))
    return e


def placebo_delta(t, nkeep, nweeks, rng):
    """Mean and p95 of dR/wk over N_PLACEBO random keeps of nkeep trades among t."""
    r = t.net_R.to_numpy()
    tot = r.sum()
    d = np.array([(r[rng.choice(len(r), nkeep, replace=False)].sum() - tot) / nweeks for _ in range(N_PLACEBO)])
    return float(d.mean()), float(np.quantile(d, 0.95))


def feature_rows(t, feat, edges, weeks, label):
    """Bucket rows + the top-minus-bottom summary and the keep-top-two filter read with placebo (one split label)."""
    t = t[t[feat].notna()].copy()
    t['bin'] = bins_for(t[feat], edges)
    top = int(len(edges))
    rows = []
    for b in range(top + 1):
        s = t[t.bin == b]
        mu, tt, _ = S.day_t(s.net_R, s.day) if len(s) > 2 else (np.nan, np.nan, np.nan)
        rows.append(dict(feature=feat, split=label, bucket=f'Q{b + 1}/{top + 1}', n=len(s), mean_net=mu,
                         extop5=S.ex_top5(s.net_R) if len(s) > 4 else np.nan,
                         fast_share=float(s.fastloser.mean()) if len(s) else np.nan))
    sel = t[t.bin.isin([0, top])]
    d = (sel.bin == top).astype(float).to_numpy()
    diff, dt = S3.diff_t(sel.net_R.to_numpy(), d, sel.day.to_numpy())
    fd, fdt = S3.diff_t(sel.fastloser.to_numpy(), d, sel.day.to_numpy())
    means = [r['mean_net'] for r in rows]
    steps = np.diff(means)
    mono = bool(np.all(steps * np.sign(diff) > 0)) if diff == diff and len(steps) else False
    keep = t[t.bin >= max(top - 1, 0)]
    nw = len(weeks)
    dR = (keep.net_R.sum() - t.net_R.sum()) / nw
    pm, p95 = placebo_delta(t, len(keep), nw, np.random.default_rng(1765))
    rows.append(dict(feature=feat, split=label, bucket='SUMMARY', n=len(t), top_minus_bottom=diff, day_t=dt, monotone=mono,
                     fast_diff=fd, fast_diff_t=fdt, filt_dR_wk=float(dR), placebo_mean_dR_wk=pm, placebo_p95_dR_wk=p95,
                     keep_fills_wk=len(keep) / nw, base_mean=float(t.net_R.mean()), nbins=top + 1))
    return rows


def coverage_rows(t, label):
    """Availability rail: share of trades with a value per feature and winner-minus-loser gap in pp."""
    rows = []
    for f in FEATS:
        w, l = t[t.net_R > 0][f].notna().mean(), t[t.net_R <= 0][f].notna().mean()
        rows.append(dict(feature=f, split=label, bucket='COVERAGE', n=len(t), coverage=float(t[f].notna().mean()),
                         win_loss_gap_pp=100 * (w - l)))
    return rows


def main():
    """Features -> TRAIN tables (TRAIN, H1, H2) written -> VAL read once -> structure verdicts."""
    ft = build_features()
    pop = pd.read_csv(f'{HE}/cell_1428_population.csv', dtype={'symbol': str}, keep_default_na=False, na_values=[''])
    tr_days = sorted(pop.day[pop.split == 'TRAIN'].unique())
    h1 = set(tr_days[:len(tr_days) // 2])
    wk_tr = sorted(pop.wk[pop.split == 'TRAIN'].unique())
    wk_h = {'H1': sorted(pop.wk[pop.day.isin(h1)].unique()), 'H2': sorted(pop.wk[(pop.split == 'TRAIN') & ~pop.day.isin(h1)].unique())}
    ttr = load_trades('TRAIN', ft)
    log(f'TRAIN trades {len(ttr)}')
    edges = {f: edges_for(ttr[f]) for f in FEATS}
    log('TRAIN edges: ' + '; '.join(f'{f}={np.round(e, 4).tolist()}' for f, e in edges.items()))
    rows = coverage_rows(ttr, 'TRAIN')
    for lab, sub, wk in (('TRAIN', ttr, wk_tr), ('H1', ttr[ttr.day.isin(h1)], wk_h['H1']), ('H2', ttr[~ttr.day.isin(h1)], wk_h['H2'])):
        for f in FEATS:
            rows += feature_rows(sub, f, edges[f], wk, lab)
    out = f'{HE}/1765_buckets.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    log('TRAIN tables written; reading VAL once')
    tv = load_trades('VAL', ft)
    wk_v = sorted(pop.wk[pop.split == 'VAL'].unique())
    rows += coverage_rows(tv, 'VAL')
    for f in FEATS:
        rows += feature_rows(tv, f, edges[f], wk_v, 'VAL')
    b = pd.DataFrame(rows)
    b.to_csv(out, index=False)
    sm = b[b.bucket == 'SUMMARY'].set_index(['feature', 'split'])
    cov = b[b.bucket == 'COVERAGE'].set_index(['feature', 'split'])
    for f in FEATS:
        v, a, c = sm.loc[(f, 'VAL')], sm.loc[(f, 'H1')], sm.loc[(f, 'H2')]
        tr = sm.loc[(f, 'TRAIN')]
        same = np.sign(a.top_minus_bottom) == np.sign(c.top_minus_bottom) == np.sign(v.top_minus_bottom) != 0
        ok = dict(same_sign=bool(same), val_t=abs(v.day_t) >= 2 and np.sign(v.top_minus_bottom) == np.sign(a.top_minus_bottom),
                  mono_train=bool(tr.monotone), filt_gt_p95=v.filt_dR_wk > v.placebo_p95_dR_wk, fills=v.keep_fills_wk >= 3,
                  cov_ok=min(cov.loc[(f, 'TRAIN')].coverage, cov.loc[(f, 'VAL')].coverage) >= 0.8)
        log(f'{f}: VAL top-bottom {v.top_minus_bottom:+.3f} t {v.day_t:+.2f} mono(TRAIN) {tr.monotone} H1 {a.top_minus_bottom:+.3f} '
            f'H2 {c.top_minus_bottom:+.3f} filt dR/wk {v.filt_dR_wk:+.2f} vs p95 {v.placebo_p95_dR_wk:+.2f} keep {v.keep_fills_wk:.1f}/wk '
            f'cov T {cov.loc[(f, "TRAIN")].coverage:.2f} V {cov.loc[(f, "VAL")].coverage:.2f} -> STRUCTURE {all(ok.values())} {ok}')


if __name__ == '__main__':
    main()
