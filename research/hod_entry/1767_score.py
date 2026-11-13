"""Cell 1,767 (J2): where in the day does the gapper fade live?  PREREG: research/hod_entry/PREREG_1767.md (frozen 2026-11-13 21:05 UTC).

Data: 1762_cache/bars/<day>.parquet (raw SIP minute bars, UTC stamps) restricted to cell_1428_population (day, symbol);
prev close = latest RAW daily close strictly before the day (1762_cache/daily.parquet column c, <= 5 days stale).
Timing convention (causal, 'last bar before the instant'): a price 'at HH:MM' = close of the last bar stamped <= HH:MM - 1 min
(i.e. the bar that ENDS at HH:MM); open = open of the 09:30 bar; cover 'at 15:55' = close of the last bar stamped <= 15:54.
Reads: 1 clock, 2 the 10:00 rule, 3 high clock, 4 short rule with hard stop.  Reuses 1763_score.cost_table (symbol-median
half-spread of the HOD arms, price-decile band fallback; NOT a per-trade NBBO -- no NBBO on these days) and 1752_score.day_t/ex_top5.
TRAIN tables are written before any VAL row is computed.  Per-day processing, peak RSS logged.
Usage: cd /home/ec2-user/onemil && ulimit -v 2500000 && nice -n 19 timeout 600 python3 research/hod_entry/1767_score.py
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'ARROW_IO_THREADS'):
    os.environ.setdefault(_v, '1')
import sys, time, resource, importlib.util
import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
HE = f'{ROOT}/research/hod_entry'
sys.path.insert(0, ROOT)
sp = importlib.util.spec_from_file_location('s1763', f'{HE}/1763_score.py')
S3 = importlib.util.module_from_spec(sp); sp.loader.exec_module(S3)
S = S3.S
CK = [575, 585, 600, 630, 660, 720, 840, 900, 955]       # clock instants (minutes of ET day)
CKN = ['0935', '0945', '1000', '1030', '1100', '1200', '1400', '1500', '1555']
OPEN_M, T10, LAST_M = 570, 600, 955                      # bar stamps used: < T10 morning, <= LAST_M-1 for the cover
SLIP_STOP_BP, FEE_BP, BORROW_BP, GAP_SCALE_MAX = 50.0, 2.0, 50.0, 2.0


def log(m):
    """Timestamped progress line with peak RSS."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f'[{time.strftime("%H:%M:%S")}] {m} (peak RSS {rss:.0f} MB)', flush=True)


def prev_closes(pop):
    """Attach prev_close = latest raw daily close strictly before the day (<= 5 calendar days stale)."""
    import pyarrow.dataset as ds
    syms = sorted(pop.symbol.unique()); frames = []
    flt = ds.field('symbol').isin(syms) & (ds.field('date') >= '2025-03-01')
    for pri, c in enumerate(('1748_cache', '1746_cache')):      # same RAW daily sources as 1762_build.load_daily
        frames.append(ds.dataset(f'{ROOT}/research/signal_search/{c}/bars_raw.parquet').to_table(
            columns=['symbol', 'date', 'c'], filter=flt, use_threads=False).to_pandas().assign(pri=pri))
    fx = pd.read_parquet(f'{HE}/1762_cache/daily.parquet', columns=['symbol', 'date', 'c'])
    frames.append(fx[fx.symbol.isin(syms)].assign(pri=2))
    d = pd.concat(frames).sort_values(['symbol', 'date', 'pri']).drop_duplicates(['symbol', 'date'])
    d['dts'] = pd.to_datetime(d.date)
    p = pop.copy(); p['dts'] = pd.to_datetime(p.day)
    m = pd.merge_asof(p.sort_values('dts'), d.sort_values('dts')[['symbol', 'dts', 'c']].rename(columns={'c': 'prev_close'}),
                      on='dts', by='symbol', allow_exact_matches=False, tolerance=pd.Timedelta(days=5))
    return m.drop(columns='dts').sort_values(['day', 'symbol']).reset_index(drop=True)


def price_at(m, c, T):
    """Close of the last bar stamped <= T-1 (the bar ending at T); NaN if none."""
    i = np.searchsorted(m, T - 1, side='right') - 1
    return float(c[i]) if i >= 0 else np.nan


def symbol_day(g, row, hs_sym, hs_band):
    """All reads' fields for one (day, symbol): clock, high clock, and the short trade of read 4 (or NaNs)."""
    m = g['m'].to_numpy(); o = g.o.to_numpy(); h = g.h.to_numpy(); c = g.c.to_numpy()
    if len(m) == 0 or m[0] != OPEN_M:
        return None
    op = float(o[0]); gap = op / row.prev_close - 1
    r = dict(day=row.day, symbol=row.symbol, split=row.split, wk=row.wk, open=op, gap=gap, nbars=len(m))
    px = {n: price_at(m, c, T) for n, T in zip(CKN, CK)}
    for n in CKN:
        r['bp' + n] = (px[n] / op - 1) * 1e4
    r['stale1555'] = LAST_M - 1 - m[m <= LAST_M - 1][-1] if (m <= LAST_M - 1).any() else np.nan
    # read 3: minute of the day's high (first occurrence), drawdown from the first-15-min high to the 15:55 price
    r['hod_m'] = int(m[int(np.argmax(h))])
    k15 = m < OPEN_M + 15
    r['dd15_bp'] = (px['1555'] / h[k15].max() - 1) * 1e4
    r['fade'] = px['1000'] < op
    # read 4: short at the 10:00 close if fade declared AND price >= $5, stop at the 09:30-10:00 high (+50 bp slip)
    kmo = m < T10
    r.update(sh_entry=np.nan, sh_net=np.nan, sh_stop=np.nan, sh_gross=np.nan, sh_hs_bp=np.nan)
    if r['fade'] and px['1000'] >= 5.0 and kmo.any():
        entry, H = px['1000'], float(h[kmo].max())
        late = (m >= T10) & (m <= LAST_M - 1)
        hit = np.flatnonzero(late & (h >= H))
        if len(hit):
            j = hit[0]; exit_p = max(H, float(o[j])) * (1 + SLIP_STOP_BP / 1e4); stop = 1
        else:
            exit_p = px['1555']; stop = 0
        hs = hs_sym.get(row.symbol)
        hs = hs_band(entry) if hs is None or not np.isfinite(hs) else hs
        gross = (entry - exit_p) / entry * 1e4
        net = gross - 2 * FEE_BP - 2 * hs * 1e4 - BORROW_BP
        r.update(sh_entry=entry, sh_net=net, sh_stop=stop, sh_gross=gross, sh_hs_bp=hs * 1e4)
    return r


def build(pop, days, hs_sym, hs_band, label):
    """Per-day pass over the bar files for the given days; returns the per-symbol-day record frame."""
    out, skipped = [], 0
    for k, day in enumerate(days):
        f = f'{HE}/1762_cache/bars/{day}.parquet'
        if not os.path.exists(f):
            continue
        b = pd.read_parquet(f)
        pr = pop[pop.day == day].set_index('symbol')
        b = b[b.symbol.isin(pr.index)]
        et = pd.to_datetime(b.t, utc=True).dt.tz_convert('America/New_York')
        b = b.assign(m=et.dt.hour * 60 + et.dt.minute)
        b = b[(b.m >= OPEN_M) & (b.m < 960)].sort_values(['symbol', 'm'])
        for sym, g in b.groupby('symbol', sort=False):
            row = _R(pr.loc[sym], sym, day)
            if not np.isfinite(row.prev_close):
                skipped += 1; continue
            rec = symbol_day(g, row, hs_sym, hs_band)
            if rec is None:
                skipped += 1
            else:
                out.append(rec)
        if k % 25 == 0:
            log(f'{label}: day {k + 1}/{len(days)} {day}, records {len(out)}, skipped {skipped}')
    log(f'{label}: done, {len(out)} records, {skipped} skipped (no 09:30 bar / no prev close)')
    return pd.DataFrame(out)


class _R:
    """Light row carrier (attribute access) for symbol_day."""
    def __init__(self, row, sym, day):
        self.symbol, self.day, self.split, self.wk, self.prev_close = sym, day, row['split'], row['wk'], row['prev_close']


def diff_t(y1, d1, y0, d0):
    """Difference of means (group1 - group0) and its day-clustered t (cluster sums of centred values per group)."""
    y1, y0 = np.asarray(y1, float), np.asarray(y0, float)
    if len(y1) < 3 or len(y0) < 3:
        return np.nan, np.nan
    s1 = pd.Series(y1 - y1.mean()).groupby(np.asarray(d1)).sum() / len(y1)
    s0 = pd.Series(y0 - y0.mean()).groupby(np.asarray(d0)).sum() / len(y0)
    a = s1.reindex(s1.index.union(s0.index), fill_value=0); b = s0.reindex(a.index, fill_value=0)
    se = np.sqrt(((a - b) ** 2).sum())
    d = y1.mean() - y0.mean()
    return d, (d / se if se > 0 else np.nan)


def subpops(t):
    """Named sub-populations for the clock (gap buckets and price tiers within gap >= 10 %)."""
    g, p = t.gap, t.open
    return {'gap5-10': t[(g >= .05) & (g < .10)], 'gap>=10': t[g >= .10], 'gap>=20': t[g >= .20],
            'gap>=10 $1-5': t[(g >= .10) & (p >= 1) & (p < 5)], 'gap>=10 $5-20': t[(g >= .10) & (p >= 5) & (p < 20)],
            'gap>=10 $>20': t[(g >= .10) & (p >= 20)]}


def clock_rows(t, sample):
    """Read 1 (clock), read 2 (10:00 rule difference) and read 3 (high clock) rows for one sample."""
    rows = []
    for name, s in subpops(t).items():
        if len(s) < 5:
            continue
        r = dict(sample=sample, subpop=name, n=len(s), days=s.day.nunique())
        for n in CKN:
            r['bp' + n] = s['bp' + n].mean()
        r['t1555'] = S.day_t(s.bp1555.to_numpy(), s.day.to_numpy())[1]
        r['share_by_1000'] = r['bp1000'] / r['bp1555'] if r['bp1555'] < 0 else np.nan
        r['share_by_1030'] = r['bp1030'] / r['bp1555'] if r['bp1555'] < 0 else np.nan
        # read 2: 10:00 -> 15:55 move (bp of the open) split by the 10:00 state
        s = s.assign(late=s.bp1555 - s.bp1000)
        f, hld = s[s.fade], s[~s.fade]
        r.update(n_fade=len(f), n_hold=len(hld), late_fade=f.late.mean(), late_hold=hld.late.mean())
        r['late_diff'], r['late_diff_t'] = diff_t(hld.late, hld.day, f.late, f.day)   # holding - fade
        # read 3: the minute of the day's high
        hm = s.hod_m
        r.update(hi_by0935=(hm < 575).mean(), hi_by0945=(hm < 585).mean(), hi_by1000=(hm < 600).mean(),
                 hi_later=(hm >= 600).mean(), hi_median_m=hm.median(), dd15_bp=s.dd15_bp.mean())
        rows.append(r)
    return rows


def short_rows(t, sample, weeks):
    """Read 4 rows (the owner's short rule) for the variants gap>=10 % (primary), >=20 %, and price tiers."""
    rows = []
    var = {'gap>=10 p>=5': t[t.gap >= .10], 'gap>=20 p>=5': t[t.gap >= .20],
           'gap>=10 $5-20': t[(t.gap >= .10) & (t.open < 20)], 'gap>=10 $>20': t[(t.gap >= .10) & (t.open >= 20)]}
    for name, s in var.items():
        s = s[np.isfinite(s.sh_net)]
        if len(s) < 5:
            continue
        x, d = s.sh_net.to_numpy(), s.day.to_numpy()
        mu, tt, se = S.day_t(x, d)
        k = max(1, int(np.floor(len(x) * 0.05)))
        rows.append(dict(sample=sample, variant=name, n=len(x), weeks=weeks, per_wk=len(x) / weeks, mean_net_bp=mu, day_t=tt,
                         se=se, MDE80_bp=2.8 * se, ex_top5_bp=S.ex_top5(x), tail5_bp=float(np.sort(x)[:k].mean()),
                         stop_hit=float(s.sh_stop.mean()), gross_bp=float(s.sh_gross.mean()), hs_bp=float(s.sh_hs_bp.mean()),
                         win_rate=float((x > 0).mean()), y2025=float(s[s.day < '2026'].sh_net.mean()) if (s.day < '2026').any() else np.nan,
                         y2026=float(s[s.day >= '2026'].sh_net.mean()) if (s.day >= '2026').any() else np.nan))
    return rows


def main():
    """TRAIN pass + tables, then VAL pass + tables, then the kill verdict."""
    pop = pd.read_csv(f'{HE}/cell_1428_population.csv', keep_default_na=False)
    pop = prev_closes(pop)
    log(f'population {len(pop)} rows, prev_close missing {pop.prev_close.isna().sum()}')
    hs_sym, hs_band = S3.cost_table()
    tr_days = sorted(pop[pop.split == 'TRAIN'].day.unique()); va_days = sorted(pop[pop.split == 'VAL'].day.unique())
    half = {d: ('H1' if k < len(tr_days) // 2 else 'H2') for k, d in enumerate(tr_days)}
    wk = {s: pop[pop.split == s].wk.nunique() for s in ('TRAIN', 'VAL')}
    wkh = {h: pop[pop.day.map(half) == h].wk.nunique() for h in ('H1', 'H2')}
    # ---- TRAIN
    t = build(pop, tr_days, hs_sym, hs_band, 'TRAIN')
    sus = t[t.gap.abs() > GAP_SCALE_MAX]
    log(f'TRAIN price-scale check: {len(sus)} symbol-days with |gap| > {GAP_SCALE_MAX:.0%} dropped; gap quantiles {t.gap.quantile([.01,.1,.5,.9,.99]).round(3).tolist()}')
    t = t[t.gap.abs() <= GAP_SCALE_MAX].copy(); t['half'] = t.day.map(half)
    c_rows = clock_rows(t[t.half == 'H1'], 'H1') + clock_rows(t[t.half == 'H2'], 'H2') + clock_rows(t, 'TRAIN')
    s_rows = (short_rows(t[t.half == 'H1'], 'H1', wkh['H1']) + short_rows(t[t.half == 'H2'], 'H2', wkh['H2'])
              + short_rows(t, 'TRAIN', wk['TRAIN']))
    pd.DataFrame(c_rows).to_csv(f'{HE}/1767_clock.csv', index=False); pd.DataFrame(s_rows).to_csv(f'{HE}/1767_short.csv', index=False)
    log('TRAIN tables written; reading VAL once')
    # ---- VAL (read once)
    v = build(pop, va_days, hs_sym, hs_band, 'VAL')
    log(f'VAL price-scale check: {(v.gap.abs() > GAP_SCALE_MAX).sum()} dropped')
    v = v[v.gap.abs() <= GAP_SCALE_MAX].copy()
    c_rows += clock_rows(v, 'VAL'); s_rows += short_rows(v, 'VAL', wk['VAL'])
    cl, sh = pd.DataFrame(c_rows), pd.DataFrame(s_rows)
    cl.to_csv(f'{HE}/1767_clock.csv', index=False); sh.to_csv(f'{HE}/1767_short.csv', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    print(cl[cl.subpop == 'gap>=10'].T.to_string()); print(cl[cl.subpop != 'gap>=10'][['sample', 'subpop', 'n', 'bp1000', 'bp1030', 'bp1555', 'share_by_1000']].to_string())
    print(sh.to_string())
    # stale-bar check on the 15:55 exit
    allr = pd.concat([t, v]); sub = allr[allr.gap >= .10]
    log(f'gap>=10: n {len(sub)}, share with last bar > 30 min before 15:55 = {(sub.stale1555 > 30).mean():.3f}')
    k = sh[(sh['sample'] == 'VAL') & (sh.variant == 'gap>=10 p>=5')].iloc[0]
    h1 = sh[(sh['sample'] == 'H1') & (sh.variant == 'gap>=10 p>=5')].iloc[0].mean_net_bp
    h2 = sh[(sh['sample'] == 'H2') & (sh.variant == 'gap>=10 p>=5')].iloc[0].mean_net_bp
    kills = {'VAL net<=50bp': k.mean_net_bp <= 50, 'day-t<2': k.day_t < 2, 'TRAIN halves opposite': np.sign(h1) != np.sign(h2),
             'ex-top5<=0': k.ex_top5_bp <= 0, 'tail5 < -800bp': k.tail5_bp < -800, '<3 names/wk': k.per_wk < 3}
    log(f'KILL CHECKS (VAL, gap>=10 p>=5): {kills} => {"KILL" if any(kills.values()) else "SURVIVE"}')


if __name__ == '__main__':
    main()
