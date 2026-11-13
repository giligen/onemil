"""Cell 1,764 score: do EDGAR dilution filings (E1) / earnings-day 8-Ks (E2) split the regular HOD-break book?

PREREG: research/hod_entry/PREREG_1764.md (frozen 2026-11-13 20:45 UTC).  Outcomes = the 1,763 HOD comparator trades
(1763_trades_{TRAIN,VAL}.parquet, written by 1763_score.day_rows -> first_break/trade; same costs, R floor 1.0 %), NOT
re-simulated.  Fast loser = stop exit within 15 min of entry (stop walk replicates 1752_build.simulate's stop test).
Features (1764_features.parquet, causal, filings public before the day):
  dil_90/dil_30 : any of S-1,S-3,F-1,F-3 (+/A, ASR, MEF variants), 424B2/3/4/5, or an 8-K with item 3.02 (unregistered
                  equity sale; PROXY for 'securities-purchase / ATM agreement' - no document text is fetched) with
                  filingDate in [t-90|30, t-1].   n_dil_365 : count in [t-365, t-1].
  earn_day      : 8-K with item 2.02 or 7.01 accepted in (prev business day 16:00 ET, t 09:30 ET].
  any_8k_2d     : any 8-K filed in [t-2, t].
Unmapped tickers (not in company_tickers.json) have NaN flags and are excluded from every table.
Stage A: features + TRAIN tables (TRAIN, H1, H2) -> 1764_buckets.csv.  Stage B (needs stage A): VAL once, appended.
Usage: cd /home/ec2-user/onemil && ulimit -v 2500000 && nice -n 19 timeout 600 python3 research/hod_entry/1764_score.py --stage A|B
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'ARROW_IO_THREADS'):
    os.environ.setdefault(_v, '1')
import sys, json, glob, time, sqlite3, resource, argparse, importlib.util
import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
HE = f'{ROOT}/research/hod_entry'
SUB = f'{HE}/1764_cache/submissions'
sys.path.insert(0, ROOT)
sp = importlib.util.spec_from_file_location('s1763', f'{HE}/1763_score.py')
S3 = importlib.util.module_from_spec(sp); sp.loader.exec_module(S3)
C, S = S3.C, S3.S
DIL_FORMS = ('S-1', 'S-3', 'F-1', 'F-3', '424B2', '424B3', '424B4', '424B5')
FLOOR = 1.0
FAST_MIN = 15


def log(m):
    """Timestamped progress line with peak RSS."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f'[{time.strftime("%H:%M:%S")}] {m} (peak RSS {rss:.0f} MB)', flush=True)


def is_dil(form, items):
    """E1 filing test: shelf / registration / prospectus supplement form, or an 8-K carrying item 3.02."""
    f = str(form).upper()
    if f.startswith('424B'):
        return f[:5] in ('424B2', '424B3', '424B4', '424B5')
    if f.startswith(('S-1', 'S-3', 'F-1', 'F-3')):
        return True
    return f.startswith('8-K') and '3.02' in str(items).split(',')


def load_filings():
    """All cached submissions pages -> long frame (cik, form, fdate, acc_utc, items); also #CIKs that needed older pages."""
    rows, old = [], set()
    for p in sorted(glob.glob(f'{SUB}/CIK*.json')):
        base = os.path.basename(p)
        cik = int(base[3:13])
        j = json.load(open(p))
        blk = j.get('filings', {}).get('recent', j) if 'filings' in j else j
        if '_CIK' in base or base[13:14] == '_':
            old.add(cik)
        if 'form' not in blk:
            continue
        rows.append(pd.DataFrame({'cik': cik, 'form': blk['form'], 'fdate': blk['filingDate'],
                                  'acc': blk.get('acceptanceDateTime', [None] * len(blk['form'])),
                                  'items': blk.get('items', [''] * len(blk['form']))}))
    f = pd.concat(rows, ignore_index=True)
    f['fdate'] = pd.to_datetime(f.fdate)
    f['acc_et'] = pd.to_datetime(f.acc, utc=True, errors='coerce').dt.tz_convert('America/New_York')
    return f.drop_duplicates(['cik', 'form', 'fdate', 'acc']), len(old)


def build_features():
    """Per symbol-day flags for the whole population -> 1764_features.parquet."""
    pop = pd.read_csv(f'{HE}/cell_1428_population.csv', dtype={'symbol': str}, keep_default_na=False)
    pop['day_ts'] = pd.to_datetime(pop.day)
    cmap = pd.read_csv(f'{HE}/1764_cache/symbol_cik.csv', dtype={'symbol': str}, keep_default_na=False)
    pop = pop.merge(cmap, on='symbol', how='left')
    pop['cik'] = pd.to_numeric(pop.cik, errors='coerce').astype('Int64')
    f, n_old = load_filings()
    log(f'{len(f)} filings, {f.cik.nunique()} CIKs; CIKs with older pages followed: {n_old}')
    f['dil'] = [is_dil(a, b) for a, b in zip(f.form, f['items'])]
    f['is8k'] = f.form.str.upper().str.startswith('8-K')
    f['earn'] = f.is8k & f['items'].astype(str).map(lambda s: bool({'2.02', '7.01'} & set(s.split(','))))
    by = {c: g for c, g in f.groupby('cik')}
    recs = []
    for r in pop.itertuples(index=False):
        rec = dict(day=r.day, symbol=r.symbol, split=r.split, wk=r.wk, cik=(None if pd.isna(r.cik) else int(r.cik)), mapped=not pd.isna(r.cik))
        if rec['mapped'] and int(r.cik) in by:
            g = by[int(r.cik)]
            t = r.day_ts
            d = g[g.dil]
            ago = (t - d.fdate).dt.days
            rec.update(dil_90=bool(((ago >= 1) & (ago <= 90)).any()), dil_30=bool(((ago >= 1) & (ago <= 30)).any()),
                       n_dil_365=int(((ago >= 1) & (ago <= 365)).sum()))
            pb = t - pd.offsets.BDay(1)
            lo = pd.Timestamp(pb.date(), tz='America/New_York') + pd.Timedelta(hours=16)
            hi = pd.Timestamp(t.date(), tz='America/New_York') + pd.Timedelta(hours=9, minutes=30)
            e = g[g.earn]
            rec['earn_day'] = bool(((e.acc_et > lo) & (e.acc_et <= hi)).any())
            a8 = g[g.is8k]
            rec['any_8k_2d'] = bool(((a8.fdate >= t - pd.Timedelta(days=2)) & (a8.fdate <= t)).any())
        elif rec['mapped']:
            rec.update(dil_90=False, dil_30=False, n_dil_365=0, earn_day=False, any_8k_2d=False)
        else:
            rec.update(dil_90=np.nan, dil_30=np.nan, n_dil_365=np.nan, earn_day=np.nan, any_8k_2d=np.nan)
        recs.append(rec)
    ft = pd.DataFrame(recs)
    ft.to_parquet(f'{HE}/1764_features.parquet', index=False)
    json.dump({'ciks_with_older_pages': n_old}, open(f'{HE}/1764_cache/meta.json', 'w'))
    return ft


def fast_flags(split, tr):
    """Mark stop exits within FAST_MIN minutes of entry by walking the minute bars (cached per split)."""
    p = f'{HE}/1764_cache/fast_{split}.parquet'
    if os.path.exists(p):
        return pd.read_parquet(p)
    pr = pd.read_parquet(f'{HE}/1762_cache/presence.parquet')
    S3.C.walker.wait_for_db_window()
    con = sqlite3.connect(S3.B.DB, uri=True)
    out = []
    for day, g in tr[tr.why == 'stop'].groupby('day'):
        bars = C.day_bars(con, day, pr[pr.day == day], list(g.symbol))
        by = {s: x.sort_values('m') for s, x in bars.groupby('symbol')}
        for r in g.itertuples(index=False):
            sb = by.get(r.symbol)
            fast = np.nan
            if sb is not None:
                m, o, l = sb.m.to_numpy(), sb.o.to_numpy(float), sb.l.to_numpy(float)
                j = np.flatnonzero((m > r.entry_m) & ((o <= r.stop) | (l <= r.stop)))
                fast = float(m[j[0]] - r.entry_m <= FAST_MIN) if len(j) else np.nan
            out.append(dict(day=day, symbol=r.symbol, fast=fast))
    con.close()
    fa = pd.DataFrame(out)
    log(f'{split}: fast-loser walk {len(fa)} stop trades, NaN {int(fa.fast.isna().sum())}')
    fa.to_parquet(p, index=False)
    return fa


def trades(split, ft):
    """HOD comparator trades (R floor) joined to features and the fast-loser flag."""
    t = pd.read_parquet(f'{HE}/1763_trades_{split}.parquet')
    t = t[(t.read == 'HOD') & (t.r_pct >= FLOOR)].copy()
    fa = fast_flags(split, t)
    t = t.merge(fa, on=['day', 'symbol'], how='left').merge(
        ft[['day', 'symbol', 'cik', 'mapped', 'dil_90', 'dil_30', 'n_dil_365', 'earn_day', 'any_8k_2d']],
        on=['day', 'symbol'], how='left')
    t['fast'] = t.fast.fillna(0.0)
    t['fastloser'] = ((t.why == 'stop') & (t.fast == 1.0)).astype(float)
    return t


def bucket_rows(t, flag, label, weeks):
    """Yes / no bucket rows for one boolean flag plus yes-minus-no diff, day-clustered t, filter delta."""
    t = t[t.mapped == True].copy()  # noqa: E712
    d = t[flag].astype(float).to_numpy()
    rows = []
    for name, sub in (('yes', t[d == 1]), ('no', t[d == 0])):
        mu, tt, se = S.day_t(sub.net_R, sub.day)
        rows.append(dict(label=label, flag=flag, bucket=name, n=len(sub), n_wk=len(sub) / len(weeks), mean_net=mu,
                         extop5=S.ex_top5(sub.net_R), fast_share=float(sub.fastloser.mean()) if len(sub) else np.nan))
    diff, dt = S3.diff_t(t.net_R.to_numpy(), d, t.day.to_numpy())
    fshare, ft_ = S3.diff_t(t.fastloser.to_numpy(), d, t.day.to_numpy())
    wk_all = t.groupby('wk').net_R.sum().reindex(weeks, fill_value=0.0)
    wk_keep = t[d == 0].groupby('wk').net_R.sum().reindex(weeks, fill_value=0.0)
    wk_only = t[d == 1].groupby('wk').net_R.sum().reindex(weeks, fill_value=0.0)
    dw = wk_keep - wk_all
    se = dw.std(ddof=1) / np.sqrt(len(dw)) if len(dw) > 2 else np.nan
    for r in rows:
        r.update(diff_yes_minus_no=diff, diff_day_t=dt, fast_diff=fshare, fast_diff_t=ft_,
                 filt_dR_wk=float(dw.mean()), filt_dR_wk_t=float(dw.mean() / se) if se and se > 0 else np.nan,
                 filt_keep_fills_wk=float((d == 0).sum() / len(weeks)),
                 green_all=float((wk_all > 0).mean()), green_filtered=float((wk_keep > 0).mean()),
                 green_flag_only=float((wk_only > 0).mean()))
    return rows


def placebo(t, weeks, seed=1764):
    """Same split by a random 50 % of CIKs (E1 shape): diff / t / filtered dR per week."""
    t = t[t.mapped == True].copy()  # noqa: E712
    rng = np.random.default_rng(seed)
    ciks = np.sort(t.cik.dropna().unique())
    pick = set(rng.choice(ciks, len(ciks) // 2, replace=False))
    t['placebo'] = t.cik.isin(pick)
    return bucket_rows(t, 'placebo', 'placebo', weeks)


def tables(split, ft, halves):
    """All bucket rows for one split label (halves = {label: (day set, weeks)})."""
    out = []
    t_all = trades(split if split in ('TRAIN', 'VAL') else 'TRAIN', ft)
    for label, (days, weeks) in halves.items():
        t = t_all[t_all.day.isin(days)] if days is not None else t_all
        for flag in ('dil_90', 'earn_day'):
            out += [{**r, 'split': label} for r in bucket_rows(t, flag, flag, weeks)]
        out += [{**r, 'split': label} for r in placebo(t, weeks)]
        # availability rail: mapped share among winners vs losers (all HOD trades incl. unmapped)
        w, l = t[t.net_R > 0].mapped.mean(), t[t.net_R <= 0].mapped.mean()
        out.append(dict(split=label, label='rail', flag='mapped_share_winner_vs_loser', bucket='win', n=len(t),
                        mean_net=w, extop5=l))
    return out


def main(stage):
    pop = pd.read_csv(f'{HE}/cell_1428_population.csv', dtype={'symbol': str}, keep_default_na=False)
    bp = f'{HE}/1764_buckets.csv'
    if stage == 'A':
        ft = build_features()
        m = ft.mapped
        log(f'mapped symbol-days {m.mean():.3f}; symbols {ft[["symbol","mapped"]].drop_duplicates().mapped.mean():.3f}; '
            f'dil_90 {ft.dil_90[m].mean():.3f} dil_30 {ft.dil_30[m].mean():.3f} earn_day {ft.earn_day[m].mean():.3f} '
            f'any_8k_2d {ft.any_8k_2d[m].mean():.3f}')
        tr_days = sorted(pop.day[pop.split == 'TRAIN'].unique())
        h1, h2 = set(tr_days[:len(tr_days) // 2]), set(tr_days[len(tr_days) // 2:])
        wk = lambda ds: sorted(pop.wk[pop.day.isin(ds)].unique())
        halves = {'TRAIN': (None, wk(tr_days)), 'H1': (h1, wk(h1)), 'H2': (h2, wk(h2))}
        pd.DataFrame(tables('TRAIN', ft, halves)).to_csv(bp, index=False)
        log(f'TRAIN tables written -> {bp}')
    else:
        if not os.path.exists(bp):
            sys.exit('ERROR: run stage A first (VAL is read once after TRAIN tables exist)')
        ft = pd.read_parquet(f'{HE}/1764_features.parquet')
        va = sorted(pop.day[pop.split == 'VAL'].unique())
        halves = {'VAL': (None, sorted(pop.wk[pop.split == 'VAL'].unique()))}
        old = pd.read_csv(bp)
        if (old.split == 'VAL').any():
            sys.exit('ERROR: VAL already read')
        pd.concat([old, pd.DataFrame(tables('VAL', ft, halves))], ignore_index=True).to_csv(bp, index=False)
        log('VAL tables appended')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', required=True, choices=['A', 'B'])
    main(ap.parse_args().stage)
