"""Cell 1,762 build - 1,752 event logic on the HONEST population (all PIT gapper symbol-days, TRAIN+VAL, no arm days).

PREREG: research/hod_entry/PREREG_1762.md.  The event/simulation logic is IMPORTED from 1752_build.py
(entries_for, stop_menu, simulate, features); this file only supplies the data plumbing:
  * minute bars: bars_sip.db (read-only) where 1762_fetch.py found them, else 1762_cache/bars/<day>.parquet;
  * daily features: 1746/1748 caches + 1762_cache/daily.parquet (fetched gaps);
  * comparators: BREAK / HOLD / RETEST from a CAUSAL break (first bar with high > prior HOD, same ok-mask as the
    approach), entered through the same entries_for code by synthesising the arm (entry = max(open, level)).
Every event row keeps r_pct, so the R >= 1 % floor is applied (and counted) in 1762_score.py.
Cost: symbol-median half-spread of the arms where the symbol is known, else the price-decile band (no NBBO exists for
these days - flagged in RESULT_1762.md).  TEST days are never in the population file's TRAIN/VAL filter.

Usage: ulimit -v 2500000 && nice -n 19 timeout 600 python3 research/hod_entry/1762_build.py
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'ARROW_IO_THREADS'):
    os.environ.setdefault(_v, '1')
import sys, glob, json, time, sqlite3, resource, zlib, importlib.util
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

ROOT = '/home/ec2-user/onemil'
sys.path.insert(0, ROOT)
HE = f'{ROOT}/research/hod_entry'
spec = importlib.util.spec_from_file_location('b1752', f'{HE}/1752_build.py')
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)
from research.hod_exit_lab import walker  # noqa: E402

CACHE = f'{HE}/1762_cache'
PARTS = f'{CACHE}/parts'
LOG = f'{HE}/1762_build.log'
CHUNK = 8
OPEN_M, START_M, LAST_M = B.OPEN_M, B.START_M, B.LAST_M


def log(msg):
    """Print and append a timestamped line with the current peak RSS (MB) to 1762_build.log."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    line = f'[{time.strftime("%H:%M:%S")}] rss={rss:.0f}MB {msg}'
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


def load_daily(symbols):
    """Daily causal features per (symbol, day): 1746/1748 caches + fetched gaps, same formulas as 1752_build."""
    flt = ds.field('symbol').isin(sorted(symbols)) & (ds.field('date') >= '2025-03-01')
    frames = []
    for pri, c in enumerate(('1748_cache', '1746_cache')):
        raw = ds.dataset(f'{B.SS}/{c}/bars_raw.parquet').to_table(
            columns=['symbol', 'date', 'h', 'l', 'c', 'v'], filter=flt, use_threads=False).to_pandas()
        adj = ds.dataset(f'{B.SS}/{c}/bars_all.parquet').to_table(
            columns=['symbol', 'date', 'c'], filter=flt, use_threads=False).to_pandas()
        raw['cadj'] = raw.set_index(['symbol', 'date']).index.map(adj.set_index(['symbol', 'date']).c).to_numpy()
        raw['pri'] = pri
        frames.append(raw)
        log(f'daily {c}: {len(raw):,} rows')
    fx = pd.read_parquet(f'{CACHE}/daily.parquet')
    fx = fx[fx.symbol.isin(symbols) & (fx.date >= '2025-03-01')].assign(pri=2)
    frames.append(fx)
    log(f'daily fetched: {len(fx):,} rows, {fx.symbol.nunique()} symbols')
    d = pd.concat(frames).sort_values(['symbol', 'date', 'pri']).drop_duplicates(['symbol', 'date'])
    d = d.sort_values(['symbol', 'date']).reset_index(drop=True)
    for k in ('h', 'l', 'c', 'v', 'cadj'):
        d[k] = d[k].astype('float32')
    g = d.groupby('symbol', sort=False)
    out = pd.DataFrame({'symbol': d.symbol, 'date': d.date})
    for n in (5, 10, 20, 50):
        ma = g.cadj.transform(lambda s, n=n: s.rolling(n, min_periods=n).mean())
        out[f'pc_ma{n}'] = (d.cadj / ma - 1).groupby(d.symbol).shift(1)
        if n == 20:
            ma20 = ma
    out['ma20_slope5'] = (ma20 / ma20.groupby(d.symbol).shift(5) - 1).groupby(d.symbol).shift(1)
    out['prev_close'] = g.c.shift(1)
    out['prev_range'] = ((d.h - d.l) / d.c).groupby(d.symbol).shift(1)
    out['high20'] = g.h.transform(lambda s: s.rolling(20, min_periods=10).max()).groupby(d.symbol).shift(1)
    out['adv20'] = g.v.transform(lambda s: s.rolling(20, min_periods=10).mean()).groupby(d.symbol).shift(1)
    out['day_up'] = (d.c > out.prev_close).astype(float).where(out.prev_close.notna())   # coverage audit only
    log(f'daily features: {len(out):,} rows, {out.symbol.nunique()} symbols')
    return out.rename(columns={'date': 'day'})


def causal_break_arm(b):
    """First bar with high > prior HOD under the approach ok-mask (set >= 10 min before, after 09:45, <= 15:30).
    Returns the synthetic arm {'entry_m', 'entry'} consumed by 1752's entries_for, or None."""
    m, h, o = b['m'], b['h'], b['o']
    cm = np.maximum.accumulate(h)
    idx = np.arange(len(m))
    set_idx = np.maximum.accumulate(np.where(h >= cm, idx, 0))
    Hp = np.r_[np.nan, cm[:-1]]
    setm = m[np.r_[0, set_idx[:-1]]]
    ok = (idx >= 1) & (m > START_M) & (m <= LAST_M) & (m - setm >= 10) & (h > Hp)
    bb = np.flatnonzero(ok)
    if not len(bb):
        return None
    return {'entry_m': m[bb[0]], 'entry': max(o[bb[0]], Hp[bb[0]])}


def day_bars(con, day, pr_day, syms):
    """Minute bars (RTH) of one day for the population symbols + SPY from the DB (present) and the parquet (fetched)."""
    in_db = list(pr_day.symbol[pr_day.in_db])
    parts = []
    if in_db:
        q = f"select symbol,t,o,h,l,c,v from bars where day=? and symbol in ({','.join('?' * len(in_db))})"
        parts.append(pd.read_sql_query(q, con, params=[day] + in_db))
    pq = f'{CACHE}/bars/{day}.parquet'
    if os.path.exists(pq):
        parts.append(pd.read_parquet(pq))
    bars = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=['symbol', 't', 'o', 'h', 'l', 'c', 'v'])
    bars['m'] = B.to_minutes(bars.t)
    return bars[(bars.m >= OPEN_M) & (bars.m < 960)].drop_duplicates(['symbol', 'm'])


def build_day(con, day, rows, pr_day, dly, band, half):
    """All events x S x K for one day; also the per-symbol-day coverage records."""
    bars = day_bars(con, day, pr_day, list(rows.symbol))
    spy = bars[bars.symbol == 'SPY'].sort_values('m')
    spyd = dict(zip(spy.m, spy.c / spy.o.iloc[0] - 1)) if len(spy) else {}
    by_sym = {s: g.sort_values('m') for s, g in bars[bars.symbol != 'SPY'].groupby('symbol')}
    out, cov = [], []
    for r in rows.itertuples(index=False):
        sb = by_sym.get(r.symbol)
        dfe = dly.get((r.symbol, day))
        ok = sb is not None and len(sb) >= 30 and sb.m.iloc[0] <= OPEN_M + 5
        cov.append(dict(day=day, symbol=r.symbol, split=r.split, covered=bool(ok),
                        in_db=bool(pr_day.in_db[pr_day.symbol == r.symbol].any()),
                        daily_ok=bool(dfe is not None and np.isfinite(dfe['pc_ma50'])),
                        day_up=dfe['day_up'] if dfe is not None else np.nan))
        if not ok:
            continue
        b = {k: sb[k].to_numpy(float) for k in ('o', 'h', 'l', 'c', 'v')}
        b['m'] = sb.m.to_numpy()
        price = b['o'][0]
        if np.isfinite(r.hs_sym_pct):
            hs_pct, csrc = r.hs_sym_pct, 'symbol_median'
        else:
            hs_pct, csrc = band(price), 'price_band'
        rng = np.random.default_rng(zlib.crc32(f'{day}{r.symbol}'.encode()))
        evs, att = B.entries_for(b, causal_break_arm(b), rng)
        for et, i, e_mid, kind, level, H, set_m, fillbar in evs:
            hs = hs_pct * e_mid
            e_cost = B.SLIP_BP * e_mid + (0 if kind == 'lmt' else hs)
            f = B.features(b, i, H, set_m, att(i), dfe, spyd, hs_pct)
            for sname, stop in B.stop_menu(b, i, e_mid, kind != 'mkt').items():
                for K in B.KS:
                    res = B.simulate(b, i, e_mid, e_cost, stop, level, K, hs, fillbar)
                    if res is None:
                        continue
                    net, gross, why, brk = res
                    out.append(dict(day=day, symbol=r.symbol, split=r.split, half=half[day], src='gapper',
                                    cost_src=csrc, etype=et, S=sname, K=K, entry_m=int(b['m'][i]), e_mid=e_mid,
                                    stop=stop, r_pct=(e_mid + e_cost - stop) / e_mid * 100, level=level, net_R=net,
                                    gross_R=gross, cost_R=gross - net, why=why, brk=brk, **f))
    return out, cov


def main():
    """Chunked build with resume; merges 1762_events.parquet and 1762_coverage.csv at the end."""
    os.makedirs(PARTS, exist_ok=True)
    pop = pd.read_csv(f'{HE}/cell_1428_population.csv', usecols=['day', 'symbol', 'split'],
                      dtype={'symbol': str}, keep_default_na=False, na_values=[''])
    pop = pop[pop.split.isin(['TRAIN', 'VAL'])].reset_index(drop=True)
    tr_days = sorted(pop.day[pop.split == 'TRAIN'].unique())
    half = {d: ('H1' if k < len(tr_days) // 2 else 'H2') for k, d in enumerate(tr_days)}
    half.update({d: 'V' for d in pop.day[pop.split == 'VAL'].unique()})
    sig = pd.read_parquet(f'{ROOT}/research/hod_exit_lab/signals.parquet', columns=['symbol', 'half_spread', 'price', 'split'])
    sig = sig[sig.split != 'TEST']
    sig['hs_pct'] = sig.half_spread / sig.price
    pop['hs_sym_pct'] = pop.symbol.map(sig.groupby('symbol').hs_pct.median())
    sig['dec'] = pd.qcut(np.log(sig.price), 10, labels=False)
    edges = np.quantile(np.log(sig.price), np.linspace(0, 1, 11))
    band_tab = sig.groupby('dec').hs_pct.median().to_numpy()
    band = lambda p: band_tab[int(np.clip(np.searchsorted(edges, np.log(p)) - 1, 0, 9))]
    log(f'population {len(pop):,}; symbol-median cost known for {pop.hs_sym_pct.notna().mean():.1%}')
    pr = pd.read_parquet(f'{CACHE}/presence.parquet')
    daily = load_daily(set(pop.symbol))
    keyset = set(zip(pop.symbol, pop.day))
    daily = daily[[k in keyset for k in zip(daily.symbol, daily.day)]]
    dly = {(s, d): rec for (s, d), rec in zip(zip(daily.symbol, daily.day),
                                               daily.drop(columns=['symbol', 'day']).to_dict('records'))}
    del daily
    log(f'daily rows matching the population: {len(dly):,}/{len(pop):,}')
    days = sorted(pop.day.unique())
    walker.wait_for_db_window()
    con = sqlite3.connect(B.DB, uri=True)
    t0 = time.time()
    for k in range(0, len(days), CHUNK):
        ch = days[k:k + CHUNK]
        name = f'{PARTS}/p_{ch[0]}_{ch[-1]}'
        if os.path.exists(name + '.parquet'):
            continue
        walker.wait_for_db_window()
        rows, covs = [], []
        for d in ch:
            o, c = build_day(con, d, pop[pop.day == d], pr[pr.day == d], dly, band, half)
            rows += o; covs += c
        pd.DataFrame(covs).to_csv(name + '.cov.csv', index=False)
        df = pd.DataFrame(rows)
        df.to_parquet(name + '.parquet.tmp', index=False)
        os.replace(name + '.parquet.tmp', name + '.parquet')
        log(f'chunk {ch[0]}..{ch[-1]}: {len(df):,} rows, {sum(x["covered"] for x in covs)}/{len(covs)} covered; '
            f'{k + len(ch)}/{len(days)} days, {time.time() - t0:.0f}s')
        if time.time() - t0 > 540:
            log('time budget - rerun to resume'); return
    con.close()
    ev = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f'{PARTS}/*.parquet'))], ignore_index=True)
    ev.to_parquet(f'{HE}/1762_events.parquet.tmp', index=False)
    os.replace(f'{HE}/1762_events.parquet.tmp', f'{HE}/1762_events.parquet')
    cov = pd.concat([pd.read_csv(p, dtype={'symbol': str}, keep_default_na=False, na_values=['']) for p in sorted(glob.glob(f'{PARTS}/*.cov.csv'))])
    cov.to_csv(f'{HE}/1762_coverage.csv', index=False)
    up, dn = cov[cov.day_up == 1], cov[cov.day_up == 0]
    summ = dict(n_symdays=len(cov), minute_cov=float(cov.covered.mean()), minute_cov_db=float(cov.in_db.mean()),
                daily_cov=float(cov.daily_ok.mean()), cov_up_days=float(up.covered.mean()),
                cov_down_days=float(dn.covered.mean()), gap_pp=float((up.covered.mean() - dn.covered.mean()) * 100),
                by_split=cov.groupby('split').covered.mean().to_dict(), event_rows=len(ev))
    json.dump(summ, open(f'{HE}/1762_cache/coverage.json', 'w'), indent=1)
    log(f'merged {len(ev):,} rows; {summ}')


if __name__ == '__main__':
    main()
