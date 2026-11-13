"""Cell 1,752 build — anticipation (APPROACH) entries + every other entry definition on one event table.

PREREG: research/hod_entry/PREREG_1752.md (+ Amendment 1). Read-only on bars_sip.db, opened only in the DB window.

Population (stated in RESULT_1752.md): symbol-days of 1735_per_signal.csv (HOD arms, break-conditioned) UNION
cell_1428_population.csv (causal PIT gapper universe: gap >= 5 %, $3-30, prior-day vol >= 500K; 2025-07..2026-05;
NOT break-conditioned), restricted to the arm days, TRAIN + VAL only (TEST days are never built).

Entries per symbol-day (all fills obtainable: market at a bar close, the arm's own fill, or a resting limit):
  APPR_d / APPR2_d  d in {0.5 %, 1.0 %}: first (second, same level, no break between) bar t > 09:45 with
                    HOD(t-1) set >= 10 min before, high_t >= HOD(1-d), high_t <= HOD (no intrabar break), close_t <
                    HOD, t <= 15:30.  Entry close_t + half_spread + 2 bp.
  BREAK             the arm's fill (signals.parquet entry at entry_m), + half_spread + 2 bp.
  HOLD{5,10,20}     close of bar b+N if every close b..b+N >= level, + half_spread + 2 bp.
  RETEST            limit at level, first bar within 30 min after b with low < level (strict), + 2 bp.
  PLACEBO           close of a seeded random bar 09:45-15:30 with close < HOD(1-2 %) (away from the HOD).
Stops S: bar = low of the entry bar (bar before for intrabar fills), min3 = min low of last 3 bars, pct = entry-0.5 %.
Time-out K in {10, 20, 30}: no high > level in K minutes -> exit close_{t+K}; else stop S / target E + 2R / 15:55.
Costs (PREREG_1740): stop fills at the stop (gap-through: the open) - 13 bp - half_spread - 2 bp; target - 2 bp;
time-out / 15:55 - half_spread - 2 bp. R = entry_fill - S.

Usage: ulimit -v 2500000 && nice -n 19 timeout 600 python3 research/hod_entry/1752_build.py [--merge]
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'ARROW_IO_THREADS'):
    os.environ.setdefault(_v, '1')
import sys, glob, json, time, sqlite3, resource, zlib, argparse
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
pa.set_cpu_count(1); pa.set_io_thread_count(1)

ROOT = '/home/ec2-user/onemil'
sys.path.insert(0, ROOT)
from research.hod_exit_lab import walker  # noqa: E402

HE = f'{ROOT}/research/hod_entry'
CACHE = f'{HE}/1752_cache'
PARTS = f'{CACHE}/parts'
LOG = f'{HE}/1752_build.log'
DB = 'file:/home/ec2-user/onemil/research/bf_zero/bars_sip.db?mode=ro'
SS = f'{ROOT}/research/signal_search'
CHUNK = 8
SLIP_BP, STOP_SLIP_BP = 2e-4, 13e-4
KS = (10, 20, 30)
STOPS = ('bar', 'min3', 'pct')
OPEN_M, START_M, LAST_M, EOD_M = 570, 585, 930, 955


def log(msg):
    """Print and append a timestamped line with the current peak RSS (MB) to 1752_build.log."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    line = f'[{time.strftime("%H:%M:%S")}] rss={rss:.0f}MB {msg}'
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


def load_population():
    """Arms (1735 x signals.parquet) UNION the 1428 gapper universe on the arm days; TRAIN+VAL only."""
    arms = pd.read_csv(f'{HE}/1735_per_signal.csv', usecols=['day', 'symbol', 'split', 'half', 'b0_net_R'],
                       dtype={'symbol': str}, keep_default_na=False, na_values=[''])
    sig = pd.read_parquet(f'{ROOT}/research/hod_exit_lab/signals.parquet')
    sig = sig[sig.split != 'TEST'].drop_duplicates(['day', 'symbol'])
    arms = arms[arms.split != 'TEST'].merge(sig[['day', 'symbol', 'entry_m', 'entry', 'half_spread', 'price']],
                                            on=['day', 'symbol'], how='left')
    arms['src'] = 'arm'
    p = pd.read_csv(f'{HE}/cell_1428_population.csv', usecols=['day', 'symbol', 'split'],
                    dtype={'symbol': str}, keep_default_na=False, na_values=[''])
    days = set(arms.day)
    p = p[p.day.isin(days) & (p.split != 'TEST')]
    half = arms.drop_duplicates('day').set_index('day').half
    p['half'] = p.day.map(half)
    both = p.merge(arms[['day', 'symbol']], on=['day', 'symbol'], how='inner')
    p = p.merge(arms[['day', 'symbol']], on=['day', 'symbol'], how='left', indicator=True)
    p = p[p._merge == 'left_only'].drop(columns='_merge')
    p['src'] = 'gapper'
    arms.loc[arms.set_index(['day', 'symbol']).index.isin(both.set_index(['day', 'symbol']).index), 'src'] = 'arm+gapper'
    pop = pd.concat([arms, p], ignore_index=True)
    # cost source: same-day measured (arms) > symbol median half-spread % (arms TRAIN+VAL) > price-decile band
    sig['hs_pct'] = sig.half_spread / sig.price
    pop['hs_sym_pct'] = pop.symbol.map(sig.groupby('symbol').hs_pct.median())
    log(f'population: {len(pop):,} symbol-days on {pop.day.nunique()} days; src {pop.src.value_counts().to_dict()}')
    return pop, sig


def load_daily(symbols):
    """Daily features per (symbol, date) from the 1748 (wins) / 1746 caches: MAs on adjusted closes, raw levels."""
    frames = []
    flt = ds.field('symbol').isin(sorted(symbols)) & (ds.field('date') >= '2024-09-01')
    for c in ('1748_cache', '1746_cache'):
        raw = ds.dataset(f'{SS}/{c}/bars_raw.parquet').to_table(
            columns=['symbol', 'date', 'h', 'l', 'c', 'v'], filter=flt, use_threads=False).to_pandas()
        adj = ds.dataset(f'{SS}/{c}/bars_all.parquet').to_table(
            columns=['symbol', 'date', 'c'], filter=flt, use_threads=False).to_pandas()
        for k in ('h', 'l', 'c', 'v'):
            raw[k] = raw[k].astype('float32')
        raw['cadj'] = raw.set_index(['symbol', 'date']).index.map(
            adj.set_index(['symbol', 'date']).c.astype('float32')).to_numpy()
        raw['pri'] = np.int8(0 if c.startswith('1748') else 1)
        frames.append(raw)
        log(f'daily {c}: {len(raw):,} rows')
        del adj
    d = pd.concat(frames).sort_values(['symbol', 'date', 'pri']).drop_duplicates(['symbol', 'date'])
    d = d.sort_values(['symbol', 'date']).reset_index(drop=True)
    g = d.groupby('symbol', sort=False)
    out = pd.DataFrame({'symbol': d.symbol, 'date': d.date})
    # everything below is known at the prior close (shift 1) -> causal for day t
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
    log(f'daily features: {len(out):,} rows, {out.symbol.nunique()} symbols')
    return out.rename(columns={'date': 'day'})


def to_minutes(t):
    """ISO-8601 UTC strings -> ET minute of day."""
    ts = pd.to_datetime(t, utc=True).dt.tz_convert('America/New_York')
    return (ts.dt.hour * 60 + ts.dt.minute).to_numpy()


def simulate(b, i, e_mid, e_cost, stop, level, K, hs, fill_bar_counts):
    """One trade from bar i (entry at e_mid + e_cost). Returns (net_R, gross_R, why, brk) or None if R <= 0."""
    e_fill = e_mid + e_cost
    R = e_fill - stop
    if not (R > 0) or not np.isfinite(R):
        return None
    m, o, h, l, c = b['m'], b['o'], b['h'], b['l'], b['c']
    tgt = e_fill + 2 * R
    t_end = m[i] + K
    brk = False
    raw = None
    if fill_bar_counts and l[i] <= stop:
        raw, why, cst = stop, 'stop', stop * STOP_SLIP_BP + hs + SLIP_BP * stop
    else:
        for j in range(i + 1, len(m)):
            if m[j] >= EOD_M:
                raw, why = o[j], 'eod'; cst = hs + SLIP_BP * o[j]; break
            if not brk and m[j] > t_end:
                k = j - 1
                raw, why = c[k], 'timeout'; cst = hs + SLIP_BP * c[k]; break
            if o[j] <= stop:
                raw, why = o[j], 'stop'; cst = o[j] * STOP_SLIP_BP + hs + SLIP_BP * o[j]; break
            if l[j] <= stop:
                raw, why = stop, 'stop'; cst = stop * STOP_SLIP_BP + hs + SLIP_BP * stop; break
            if h[j] > level:
                brk = True
            if h[j] >= tgt:
                raw, why = tgt, 'target'; cst = SLIP_BP * tgt; break
        if raw is None:
            raw, why = c[-1], 'eod'; cst = hs + SLIP_BP * c[-1]
    if not brk:
        brk = bool(np.any(h[i + 1:][(m[i + 1:] <= t_end)] > level))
    gross = (raw - e_mid) / R
    net = (raw - cst - e_fill) / R
    return net, gross, why, brk


def features(b, i, H, set_m, attempts, dfe, spy, hs_pct):
    """Causal feature block at entry bar i (everything through bar i's close)."""
    m, h, l, c, v = b['m'], b['h'], b['l'], b['c'], b['v']
    cumv = v[:i + 1].sum()
    v5 = v[max(0, i - 4):i + 1].mean()
    vprev5 = v[max(0, i - 5):i].mean() if i > 0 else np.nan
    f = dict(minute=m[i], min_since_hod=m[i] - set_m, attempts=attempts,
             min_since_low=m[i] - m[int(np.argmin(l[:i + 1]))],
             vol5_dayavg=v5 / (cumv / (i + 1)) if cumv > 0 else np.nan,
             volt_vol5=v[i] / vprev5 if vprev5 and vprev5 > 0 else np.nan,
             dist_open=c[i] / b['o'][0] - 1, dist_hod=c[i] / H - 1, hs_pct=hs_pct,
             spy_ret=spy.get(m[i], np.nan))
    if dfe is not None:
        adv = dfe['adv20']
        f.update(cumv_adv=cumv / adv if adv and adv > 0 else np.nan,
                 pc_ma5=dfe['pc_ma5'], pc_ma10=dfe['pc_ma10'], pc_ma20=dfe['pc_ma20'], pc_ma50=dfe['pc_ma50'],
                 ma20_slope5=dfe['ma20_slope5'], hod_vs_20dh=H / dfe['high20'] - 1 if dfe['high20'] else np.nan,
                 gap=b['o'][0] / dfe['prev_close'] - 1 if dfe['prev_close'] else np.nan, prev_range=dfe['prev_range'])
    return f


def entries_for(b, arm, rng):
    """All entry events for one symbol-day: list of (etype, i, e_mid, e_cost_kind, level, H, set_m, fill_bar, lows)."""
    m, h, l, c = b['m'], b['h'], b['l'], b['c']
    n = len(m)
    cm = np.maximum.accumulate(h)
    idx = np.arange(n)
    set_idx = np.maximum.accumulate(np.where(h >= cm, idx, 0))
    Hp = np.r_[np.nan, cm[:-1]]
    setp = np.r_[0, set_idx[:-1]]
    setm = m[setp]
    ok = (idx >= 1) & (m > START_M) & (m <= LAST_M) & (m - setm >= 10)
    ev = []
    zone1 = ok & (h >= Hp * 0.99) & (h <= Hp) & (c < Hp)
    ep1 = np.flatnonzero(zone1 & ~np.r_[False, zone1[:-1]])
    attempts_before = lambda i: int((ep1 < i).sum())
    for d, tag in ((0.005, '0.5'), (0.01, '1.0')):
        z = ok & (h >= Hp * (1 - d)) & (h <= Hp) & (c < Hp)
        ep = np.flatnonzero(z & ~np.r_[False, z[:-1]])
        if len(ep):
            i0 = ep[0]
            ev.append(('APPR_' + tag, i0, c[i0], 'mkt', Hp[i0], Hp[i0], setm[i0], False))
            later = [k for k in ep[1:] if Hp[k] == Hp[i0]]
            if later:
                ev.append(('APPR2_' + tag, later[0], c[later[0]], 'mkt', Hp[later[0]], Hp[later[0]], setm[later[0]], False))
    if arm is not None and np.isfinite(arm['entry_m']):
        bi = np.flatnonzero(m >= arm['entry_m'])
        if len(bi) and bi[0] >= 1:
            bb = bi[0]
            L = Hp[bb]
            ev.append(('BREAK', bb, arm['entry'], 'mkt_intrabar', max(h[bb], L), L, setm[bb], True))
            for N in (5, 10, 20):
                jj = np.flatnonzero(m >= m[bb] + N)
                if len(jj) and m[jj[0]] <= LAST_M and np.all(c[bb:jj[0] + 1] >= L):
                    j = jj[0]
                    ev.append((f'HOLD{N}', j, c[j], 'mkt', max(h[bb:j + 1].max(), L), L, setm[bb], False))
            for j in range(bb + 1, n):
                if m[j] > m[bb] + 30:
                    break
                if l[j] < L:
                    ev.append(('RETEST', j, L, 'lmt', max(h[bb:j].max(), L), L, setm[bb], True))
                    break
    away = np.flatnonzero(ok & (c < Hp * 0.98))
    if len(away):
        i = away[rng.integers(len(away))]
        ev.append(('PLACEBO', i, c[i], 'mkt', Hp[i], Hp[i], setm[i], False))
    return ev, attempts_before


def stop_menu(b, i, e_mid, intrabar):
    """Strict stops: bar low (bar before for intrabar fills), min low of the last 3 bars, entry - 0.5 %."""
    l = b['l']
    last = i - 1 if intrabar else i
    if last < 0:
        return {}
    return {'bar': l[last], 'min3': l[max(0, last - 2):last + 1].min(), 'pct': e_mid * 0.995}


def build_day(con, day, rows, dly, sig_hs, band):
    """All events x S x K for one day."""
    syms = list(rows.symbol)
    q = f"select symbol,t,o,h,l,c,v from bars where day=? and symbol in ({','.join('?' * len(syms))})"
    bars = pd.read_sql_query(q, con, params=[day] + syms)
    spy = pd.read_sql_query("select t,o,c from bars where day=? and symbol='SPY'", con, params=[day])
    spyd = {}
    if len(spy):
        spy['m'] = to_minutes(spy.t)
        spy = spy[(spy.m >= OPEN_M) & (spy.m < 960)]
        if len(spy):
            spyd = dict(zip(spy.m, spy.c / spy.o.iloc[0] - 1))
    if len(bars) == 0:
        return [], 0
    bars['m'] = to_minutes(bars.t)
    bars = bars[(bars.m >= OPEN_M) & (bars.m < 960)]
    out, covered = [], 0
    for r in rows.itertuples(index=False):
        sb = bars[bars.symbol == r.symbol].sort_values('m')
        if len(sb) < 30 or sb.m.iloc[0] > OPEN_M + 5:
            continue
        covered += 1
        b = {k: sb[k].to_numpy(float) for k in ('o', 'h', 'l', 'c', 'v')}
        b['m'] = sb.m.to_numpy()
        price = b['o'][0]
        if r.src != 'gapper' and np.isfinite(r.half_spread):
            hs_pct, csrc = r.half_spread / r.price, 'measured'
        elif np.isfinite(r.hs_sym_pct):
            hs_pct, csrc = r.hs_sym_pct, 'symbol_median'
        else:
            hs_pct, csrc = band(price), 'price_band'
        dfe = dly.get((r.symbol, day))
        arm = {'entry_m': r.entry_m, 'entry': r.entry} if r.src != 'gapper' else None
        rng = np.random.default_rng(zlib.crc32(f'{day}{r.symbol}'.encode()))
        evs, att = entries_for(b, arm, rng)
        for et, i, e_mid, kind, level, H, set_m, fillbar in evs:
            hs = hs_pct * e_mid
            e_cost = SLIP_BP * e_mid + (0 if kind == 'lmt' else hs)
            f = features(b, i, H, set_m, att(i), dfe, spyd, hs_pct)
            for sname, stop in stop_menu(b, i, e_mid, kind != 'mkt').items():
                for K in KS:
                    res = simulate(b, i, e_mid, e_cost, stop, level, K, hs, fillbar)
                    if res is None:
                        continue
                    net, gross, why, brk = res
                    out.append(dict(day=day, symbol=r.symbol, split=r.split, half=r.half, src=r.src, cost_src=csrc,
                                    etype=et, S=sname, K=K, entry_m=int(b['m'][i]), e_mid=e_mid, stop=stop,
                                    r_pct=(e_mid + e_cost - stop) / e_mid * 100, level=level, net_R=net,
                                    gross_R=gross, cost_R=gross - net, why=why, brk=brk,
                                    b0_net_R=r.b0_net_R if r.src != 'gapper' else np.nan, **f))
    return out, covered


def main():
    """Chunked build with resume; --merge writes 1752_events.parquet from the parts."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--merge', action='store_true')
    a = ap.parse_args()
    os.makedirs(PARTS, exist_ok=True)
    if a.merge:
        ev = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f'{PARTS}/*.parquet'))], ignore_index=True)
        ev.to_parquet(f'{HE}/1752_events.parquet.tmp', index=False)
        os.replace(f'{HE}/1752_events.parquet.tmp', f'{HE}/1752_events.parquet')
        cov = [json.load(open(p)) for p in sorted(glob.glob(f'{PARTS}/*.json'))]
        tot = {k: sum(c[k] for c in cov) for k in ('n_symdays', 'covered')}
        json.dump(tot, open(f'{CACHE}/coverage.json', 'w'))
        log(f'merged {len(ev):,} rows; coverage {tot}')
        return
    pop, sig = load_population()
    sig['dec'] = pd.qcut(np.log(sig.price), 10, labels=False)
    edges = np.quantile(np.log(sig.price), np.linspace(0, 1, 11))
    band_tab = sig.groupby('dec').hs_pct.median().to_numpy()
    band = lambda p: band_tab[int(np.clip(np.searchsorted(edges, np.log(p)) - 1, 0, 9))]
    daily = load_daily(set(pop.symbol))
    keyset = set(zip(pop.symbol, pop.day))
    daily = daily[[k in keyset for k in zip(daily.symbol, daily.day)]]
    dly = {(s, d): rec for (s, d), rec in zip(zip(daily.symbol, daily.day),
                                               daily.drop(columns=['symbol', 'day']).to_dict('records'))}
    log(f'daily coverage of population: {len(dly) / len(pop):.1%}')
    days = sorted(pop.day.unique())
    walker.wait_for_db_window()
    con = sqlite3.connect(DB, uri=True)
    t0 = time.time()
    for k in range(0, len(days), CHUNK):
        ch = days[k:k + CHUNK]
        name = f'{PARTS}/p_{ch[0]}_{ch[-1]}'
        if os.path.exists(name + '.parquet'):
            continue
        walker.wait_for_db_window()
        rows, n_sd, cov = [], 0, 0
        for d in ch:
            pr = pop[pop.day == d]
            out, c = build_day(con, d, pr, dly, None, band)
            rows += out; n_sd += len(pr); cov += c
        df = pd.DataFrame(rows)
        df.to_parquet(name + '.parquet.tmp', index=False)
        os.replace(name + '.parquet.tmp', name + '.parquet')
        json.dump({'n_symdays': n_sd, 'covered': cov}, open(name + '.json', 'w'))
        log(f'chunk {ch[0]}..{ch[-1]}: {len(df):,} rows, {cov}/{n_sd} symbol-days with bars; '
            f'{k + len(ch)}/{len(days)} days, {time.time() - t0:.0f}s')
    con.close()
    log('build done')


if __name__ == '__main__':
    main()
