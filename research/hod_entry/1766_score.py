"""Cell 1,766 score: F5 sector sympathy -- is the HOD-break trade better when the same SIC sector already broke earlier that day?

PREREG: research/hod_entry/PREREG_1766.md (frozen 2026-11-13 21:00 UTC) + coordinator amendment: the asset-class map has NO sector
column (symbol, asset_class, name) so SECTOR = 2-digit SIC (secondary 3-digit) from 1764_cache/submissions/CIK*.json via the
1764_features.parquet cik map; wrappers (asset_class == 'wrapper') carry no sector.  If the mapped share is < 80 % the whole cell is
labelled EXPLORATORY (not VOID).
Break minute: the regular HOD rule's `first_break` (imported from 1763_score) applied to every covered population symbol-day (the
bars come from `1762_build.day_bars`, DB + 1762_cache/bars) -- cached in 1766_breaks_{split}.parquet.  Outcome = the 1,763 HOD
comparator trade (read == 'HOD', r_pct >= 1 %), net_R, imported not re-simulated.
Heat for name X = number of OTHER mapped names in X's sector with break minute < X's break minute (same day).
  SOLO = heat 0; LAGGARD = heat >= 2 and X's break >= 10 min after the 2nd earlier sector break; LEADER = heat >= 2, not LAGGARD;
  heat 1 reported as ONE.  Reads: class table per H1/H2/TRAIN/VAL, LAGGARD-SOLO day-clustered t, heat 0/1/2/>=3 monotone on
  TRAIN, 'trade only LAGGARD' filter dR/wk vs 200 count-matched random drops (p95), other-sector heat placebo (200 draws).
TRAIN tables are written before VAL is read.  TEST sealed (not in the population file).
Usage: cd /home/ec2-user/onemil && ulimit -v 2500000 && nice -n 19 timeout 600 python3 research/hod_entry/1766_score.py
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'ARROW_IO_THREADS'):
    os.environ.setdefault(_v, '1')
import sys, glob, json, time, sqlite3, resource, importlib.util
import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
HE = f'{ROOT}/research/hod_entry'
SUBS = f'{HE}/1764_cache/submissions'
ASSET_MAP = f'{ROOT}/data/research/orb_asset_class_map_20260711.csv'
sys.path.insert(0, ROOT)
sp = importlib.util.spec_from_file_location('s1763', f'{HE}/1763_score.py')
S3 = importlib.util.module_from_spec(sp); sp.loader.exec_module(S3)
S, C, B = S3.S, S3.C, S3.B
FLOOR = 1.0
LAG_GAP = 10
N_PLACEBO = 200


def log(m):
    """Timestamped progress line with peak RSS."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f'[{time.strftime("%H:%M:%S")}] {m} (peak RSS {rss:.0f} MB)', flush=True)


def load_pop():
    """Population symbol-days with split, week and TRAIN half."""
    pop = pd.read_csv(f'{HE}/cell_1428_population.csv', dtype={'symbol': str}, keep_default_na=False, na_values=[''])
    tr_days = sorted(pop.day[pop.split == 'TRAIN'].unique())
    halfmap = {d: ('H1' if k < len(tr_days) // 2 else 'H2') for k, d in enumerate(tr_days)}
    pop['half'] = pop.day.map(halfmap).fillna('V')
    return pop


def sector_map(pop):
    """symbol-day -> sic2 / sic3 strings (NaN if unmapped or wrapper) via the 1764 cik map and the submissions JSON `sic` field."""
    f = pd.read_parquet(f'{HE}/1764_cache/../1764_features.parquet', columns=['day', 'symbol', 'cik', 'mapped'])
    sic = {}
    for p in glob.glob(f'{SUBS}/CIK*.json'):
        nm = os.path.basename(p)
        if '_p' in nm:
            continue
        try:
            d = json.load(open(p))
        except (OSError, ValueError) as e:
            log(f'WARNING: unreadable submissions file {nm}: {e}')
            continue
        s = str(d.get('sic') or '').strip()
        if s.isdigit() and int(s) > 0:
            sic[int(nm[3:13])] = s.zfill(4)
    log(f'SIC read for {len(sic)} CIKs')
    f['sic4'] = f.cik.map(lambda c: sic.get(int(c)) if pd.notna(c) else None)
    am = pd.read_csv(ASSET_MAP, dtype=str, keep_default_na=False)
    wr = set(am.symbol[am.asset_class == 'wrapper'])
    f.loc[f.symbol.isin(wr), 'sic4'] = None
    f['sic2'] = f.sic4.str[:2]
    f['sic3'] = f.sic4.str[:3]
    out = pop[['day', 'symbol']].merge(f[['day', 'symbol', 'sic2', 'sic3']], on=['day', 'symbol'], how='left')
    log(f'population symbol-days {len(out)}; sic mapped {out.sic2.notna().mean():.3f}; wrappers in population '
        f'{pop.symbol.isin(wr).mean():.3f}')
    return out


def breaks_for(split, pop):
    """Break minute (regular HOD rule) per covered symbol-day of one split; cached atomically."""
    path = f'{HE}/1766_breaks_{split}.parquet'
    if os.path.exists(path):
        return pd.read_parquet(path)
    p = pop[pop.split == split]
    pr = pd.read_parquet(f'{HE}/1762_cache/presence.parquet')
    C.walker.wait_for_db_window()
    con = sqlite3.connect(B.DB, uri=True)
    rows, days = [], sorted(p.day.unique())
    for k, day in enumerate(days):
        bars = C.day_bars(con, day, pr[pr.day == day], list(p.symbol[p.day == day]))
        by = {s: g.sort_values('m') for s, g in bars[bars.symbol != 'SPY'].groupby('symbol')}
        for s in p.symbol[p.day == day]:
            sb = by.get(s)
            ok = sb is not None and len(sb) >= 30 and sb.m.iloc[0] <= S3.OPEN_M + 5
            bm = np.nan
            if ok:
                b = {c: sb[c].to_numpy(float) for c in ('o', 'h', 'l', 'c', 'v')}
                b['m'] = sb.m.to_numpy()
                ib = S3.first_break(b, S3.START_M)
                bm = float(b['m'][ib]) if ib is not None else np.nan
            rows.append(dict(day=day, symbol=s, covered=bool(ok), break_m=bm))
        if (k + 1) % 20 == 0:
            log(f'{split} breaks {k + 1}/{len(days)} days')
    con.close()
    df = pd.DataFrame(rows)
    df.to_parquet(path + '.tmp', index=False)
    os.replace(path + '.tmp', path)
    log(f'{split} breaks: covered {df.covered.mean():.3f}; broke {df.break_m.notna().mean():.3f}')
    return df


def heat_table(br, key):
    """Per symbol-day: heat (earlier other-name breaks in the same sector), second earlier break minute, class."""
    d = br[br.break_m.notna() & br[key].notna()].copy()
    d['heat'], d['second_m'] = 0, np.nan
    for (_, _), g in d.groupby(['day', key]):
        m = g.break_m.to_numpy()
        for idx, mx in zip(g.index, m):
            others = np.sort(m[m < mx])
            d.at[idx, 'heat'] = len(others)
            if len(others) >= 2:
                d.at[idx, 'second_m'] = others[1]
    d['cls'] = classify(d.heat.to_numpy(), d.break_m.to_numpy(), d.second_m.to_numpy())
    return d


def classify(heat, bm, second):
    """SOLO / ONE / LEADER / LAGGARD from heat, own break minute and the second earlier sector break."""
    lag = (heat >= 2) & (bm - second >= LAG_GAP)
    return np.where(heat == 0, 'SOLO', np.where(heat == 1, 'ONE', np.where(lag, 'LAGGARD', 'LEADER')))


def load_trades(split, pop, sec, br, key):
    """HOD comparator trades joined to sector, break minute, heat class and the fast-loser flag."""
    t = pd.read_parquet(f'{HE}/1763_trades_{split}.parquet')
    t = t[(t.read == 'HOD') & (t.r_pct >= FLOOR)].copy()
    fa = pd.read_parquet(f'{HE}/1764_cache/fast_{split}.parquet')
    t = t.merge(fa, on=['day', 'symbol'], how='left').merge(sec, on=['day', 'symbol'], how='left')
    t = t.merge(br[['day', 'symbol', 'break_m']], on=['day', 'symbol'], how='left')
    t['fastloser'] = ((t.why == 'stop') & (t.fast.fillna(0) == 1.0)).astype(float)
    ht = heat_table(br.merge(sec, on=['day', 'symbol'], how='left'), key)
    t = t.merge(ht[['day', 'symbol', 'heat', 'second_m', 'cls']], on=['day', 'symbol'], how='left')
    t['cls'] = t.cls.fillna('UNMAPPED')
    t['half'] = t.day.map(pop.drop_duplicates('day').set_index('day').half)
    return t


def placebo_delta(r, nkeep, nweeks, rng):
    """Mean and p95 of dR/wk over N_PLACEBO random keeps of nkeep trades among r."""
    tot = r.sum()
    d = np.array([(r[rng.choice(len(r), nkeep, replace=False)].sum() - tot) / nweeks for _ in range(N_PLACEBO)])
    return float(d.mean()), float(np.quantile(d, 0.95))


def class_rows(t, weeks, label):
    """Per-class table + LAGGARD-SOLO + heat buckets + filter read for one split label."""
    nw, rows = len(weeks), []
    for c in ('SOLO', 'ONE', 'LEADER', 'LAGGARD', 'UNMAPPED'):
        s = t[t.cls == c]
        mu, tt, se = S.day_t(s.net_R, s.day) if len(s) else (np.nan, np.nan, np.nan)
        rows.append(dict(split=label, table='CLASS', bucket=c, n=len(s), per_wk=len(s) / nw, mean_net=mu, day_t=tt,
                         extop5=S.ex_top5(s.net_R), fast_share=float(s.fastloser.mean()) if len(s) else np.nan))
    ls = t[t.cls.isin(['LAGGARD', 'SOLO'])]
    beta, tb = S3.diff_t(ls.net_R, (ls.cls == 'LAGGARD').astype(float), ls.day)
    rows.append(dict(split=label, table='DIFF', bucket='LAGGARD-SOLO', n=len(ls), mean_net=beta, day_t=tb))
    m = t[t.cls != 'UNMAPPED']
    hb = np.minimum(m.heat, 3)
    for h in range(4):
        s = m[hb == h]
        rows.append(dict(split=label, table='HEAT', bucket=f'{h}{"+" if h == 3 else ""}', n=len(s), per_wk=len(s) / nw,
                         mean_net=float(s.net_R.mean()) if len(s) else np.nan))
    r, lag = t.net_R.to_numpy(), t[t.cls == 'LAGGARD']
    if len(lag):
        pm, p95 = placebo_delta(r, len(lag), nw, np.random.default_rng(1766))
        rows.append(dict(split=label, table='FILTER', bucket='LAGGARD_only', n=len(lag), per_wk=len(lag) / nw,
                         mean_net=float(lag.net_R.mean()), dR_wk=(lag.net_R.sum() - r.sum()) / nw, placebo_mean=pm,
                         placebo_p95=p95))
    return rows


def other_sector_placebo(t, br, sec, key, obs_diff, label, rng):
    """Same class definition but heat read from a random OTHER sector of the same day; returns row with draws' mean/p95."""
    d = br.merge(sec, on=['day', 'symbol'], how='left')
    d = d[d.break_m.notna() & d[key].notna()]
    by = {day: {k: np.sort(g.break_m.to_numpy()) for k, g in gg.groupby(key)} for day, gg in d.groupby('day')}
    tm = t[t.cls != 'UNMAPPED']
    heats, lags, offs, rs, days = [], [], [], [], []
    for r in tm.itertuples(index=False):
        sectors = by.get(r.day, {})
        cand = [k for k in sectors if k != getattr(r, key)]
        if not cand:
            continue
        start = len(heats); days.append(r.day); rs.append(r.net_R)
        for k in cand:
            o = sectors[k][sectors[k] < r.break_m]
            heats.append(len(o))
            lags.append(len(o) >= 2 and r.break_m - o[1] >= LAG_GAP)
        offs.append((start, len(heats)))
    offs = np.array(offs)
    heats, lags, rs = np.array(heats), np.array(lags), np.array(rs)
    cnt = offs[:, 1] - offs[:, 0]
    diffs = []
    for _ in range(N_PLACEBO):
        pick = offs[:, 0] + np.floor(rng.random(len(cnt)) * cnt).astype(int)
        h, lg = heats[pick], lags[pick]
        a, b = rs[lg], rs[h == 0]
        diffs.append(a.mean() - b.mean() if len(a) >= 5 and len(b) >= 5 else np.nan)
    diffs = np.array(diffs)
    return dict(split=label, table='OTHER_SECTOR', bucket='LAGGARD-SOLO placebo', n=len(rs), mean_net=float(np.nanmean(diffs)),
                placebo_p95=float(np.nanquantile(diffs, 0.95)), obs=obs_diff, share_ge_obs=float(np.nanmean(diffs >= obs_diff)))


def run_split(split, pop, sec, key, weeks, label_halves):
    """Trades with classes for one split; returns (trades, tables rows)."""
    br = breaks_for(split, pop)
    t = load_trades(split, pop, sec, br, key)
    par = t[t.break_m.notna()]
    log(f'{split}: {len(t)} HOD trades; entry_m == regular break_m for {(par.entry_m == par.break_m).mean():.4f} of '
        f'{len(par)}; break coverage {t.break_m.notna().mean():.3f}; sector-mapped trade share {(t.cls != "UNMAPPED").mean():.3f}')
    rows = class_rows(t, weeks, split)
    for h, wk in label_halves.items():
        rows += class_rows(t[t.half == h], wk, h)
    return t, br, rows


def main():
    """Sector map -> TRAIN tables (TRAIN, H1, H2; SIC2 then SIC3) written -> VAL read once -> structure verdict."""
    pop = load_pop()
    sec = sector_map(pop)
    wk = {s: sorted(pop.wk[pop.split == s].unique()) for s in ('TRAIN', 'VAL')}
    wh = {h: sorted(pop.wk[pop.half == h].unique()) for h in ('H1', 'H2')}
    rng = np.random.default_rng(1766)
    rows, tabs = [], {}
    for key in ('sic2', 'sic3'):
        t, br, r = run_split('TRAIN', pop, sec, key, wk['TRAIN'], wh)
        for x in r:
            x['key'] = key
        rows += r
        obs = [x for x in r if x['table'] == 'DIFF' and x['split'] == 'TRAIN'][0]['mean_net']
        o = other_sector_placebo(t, br, sec, key, obs, 'TRAIN', rng); o['key'] = key; rows.append(o)
        tabs[key, 'TRAIN'] = t
        if key == 'sic2':
            ms = pop.merge(sec, on=['day', 'symbol'], how='left')
            ms = ms[ms.split == 'TRAIN']
            rows.append(dict(split='TRAIN', table='MAPPED', bucket='all symbol-days', n=len(ms), mean_net=ms.sic2.notna().mean(), key=key))
            rows.append(dict(split='TRAIN', table='MAPPED', bucket='HOD trades', n=len(t), mean_net=(t.cls != 'UNMAPPED').mean(), key=key))
    out = f'{HE}/1766_tables.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    cl_cols = ['day', 'symbol', 'split', 'half', 'wk', 'net_R', 'break_m', 'sic2', 'sic3', 'heat', 'second_m', 'cls', 'fastloser']
    tabs['sic2', 'TRAIN'][cl_cols].to_csv(f'{HE}/1766_classes_TRAIN.tmp', index=False)
    log('TRAIN tables written; reading VAL once')
    vals = {}
    for key in ('sic2', 'sic3'):
        t, br, r = run_split('VAL', pop, sec, key, wk['VAL'], {})
        for x in r:
            x['key'] = key
        rows += r
        obs = [x for x in r if x['table'] == 'DIFF' and x['split'] == 'VAL'][0]['mean_net']
        o = other_sector_placebo(t, br, sec, key, obs, 'VAL', rng); o['key'] = key; rows.append(o)
        vals[key] = t
        if key == 'sic2':
            ms = pop.merge(sec, on=['day', 'symbol'], how='left')
            ms = ms[ms.split == 'VAL']
            rows.append(dict(split='VAL', table='MAPPED', bucket='all symbol-days', n=len(ms), mean_net=ms.sic2.notna().mean(), key=key))
            rows.append(dict(split='VAL', table='MAPPED', bucket='HOD trades', n=len(t), mean_net=(t.cls != 'UNMAPPED').mean(), key=key))
    tb = pd.DataFrame(rows)
    tb.to_csv(out, index=False)
    pd.concat([tabs['sic2', 'TRAIN'][cl_cols], vals['sic2'].assign(split='VAL')[cl_cols]]).to_csv(f'{HE}/1766_classes.csv', index=False)
    os.remove(f'{HE}/1766_classes_TRAIN.tmp')
    verdict(tb)


def verdict(tb):
    """Log the PREREG kill list for SIC2 (primary) and SIC3 (secondary)."""
    for key in ('sic2', 'sic3'):
        x = tb[tb.key == key]
        g = lambda tab, sp, bk=None: x[(x.table == tab) & (x.split == sp) & ((x.bucket == bk) if bk else True)]
        d = {s: g('DIFF', s).iloc[0] for s in ('H1', 'H2', 'TRAIN', 'VAL')}
        sg = [np.sign(d[s].mean_net) for s in ('H1', 'H2', 'VAL')]
        mono = g('HEAT', 'TRAIN').mean_net.to_numpy()
        inc = bool(np.all(np.diff(mono) > 0))
        fv = g('FILTER', 'VAL').iloc[0]; ft = g('FILTER', 'TRAIN').iloc[0]
        osp = g('OTHER_SECTOR', 'VAL').iloc[0]
        ok = dict(same_sign=len(set(sg)) == 1 and sg[0] != 0, val_t=bool(abs(d['VAL'].day_t) >= 2), heat_mono_train=inc,
                  filt_gt_p95=bool(fv.dR_wk > fv.placebo_p95), fills3=bool(fv.per_wk >= 3), other_sector_null=bool(osp.share_ge_obs > 0.05))
        log(f'[{key}] LAGGARD-SOLO H1 {d["H1"].mean_net:+.3f} H2 {d["H2"].mean_net:+.3f} TRAIN {d["TRAIN"].mean_net:+.3f} '
            f't {d["TRAIN"].day_t:+.2f} VAL {d["VAL"].mean_net:+.3f} t {d["VAL"].day_t:+.2f}; heat TRAIN {np.round(mono, 3).tolist()} '
            f'inc={inc}; filter dR/wk TRAIN {ft.dR_wk:+.2f}/p95 {ft.placebo_p95:+.2f} VAL {fv.dR_wk:+.2f}/p95 {fv.placebo_p95:+.2f} '
            f'LAGGARD {fv.per_wk:.2f}/wk VAL; other-sector VAL mean {osp.mean_net:+.3f} p95 {osp.placebo_p95:+.3f} obs {osp.obs:+.3f} '
            f'-> STRUCTURE {all(ok.values())} {ok}')


if __name__ == '__main__':
    main()
