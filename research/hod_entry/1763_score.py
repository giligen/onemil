"""Cell 1,763 - LEVELS & CLOCKS on the honest PIT gapper bars: six reads (B1, B2, B4, B6, B7, B8), one table.

PREREG: research/hod_entry/PREREG_1763.md (frozen 2026-11-13 20:35 UTC).  Population: cell_1428_population.csv,
TRAIN/VAL only (TEST sealed).  Minute bars: 1762_build.day_bars (bars_sip.db read-only + 1762_cache/bars parquet),
RTH 09:30-15:59.  Pre-market: 1762_cache/premarket/<day>.parquet (1763_fetch_premarket.py).
REUSED by import (not rewritten): 1752_build.simulate (stop / +2 R target / 15:55 exit, PREREG_1740 costs:
stop + 13 bp + half-spread + 2 bp, target 2 bp, EOD half-spread + 2 bp), SLIP_BP, the 1762 cost source (symbol-
median half-spread from the arms, else the price-decile band - no NBBO on these days), 1752_score.day_t / ex_top5.
Shared: entry = close of the bar that confirms the trigger + half_spread + 2 bp; R floor 1.0 % of price; one trade
per read per symbol-day; time-out = 15:55 only (simulate called with K = 10**6, level = +inf).
Reads (causal at the trigger bar):
  HOD   comparator: first bar 09:46-15:30 with high > prior HOD set >= 10 min before; stop min low of last 3 bars.
  B1    pre-market high (04:00-09:29): first 1-min close above it from 09:31; stop min3.  VOID if coverage < 90 %.
  B2    5-min (09:30-aligned) bar starting 09:45-15:25 closing above the prior HOD (set >= 10 min before the bucket
        start); stop = the 5-min bar low.  Paired 1-min rule: the HOD break bar's close, stop = that bar's low.
  B4    nearest $1 (<$10) / $5 ($10-50) / $10 (>$50) above the open that is >= 1 % away; first close above it
        09:46-15:30; stop min3.
  B6    HOD age at 10:30 (minutes since the 09:30-10:29 high was FIRST set): <=15 / 15-45 / >45; outcomes = the HOD
        rule restricted to breaks >= 10:30 (causal filter) and the 10:30 -> last-bar return (%).
  B7    new HOD (high > every earlier bar) on a bar 09:31-09:59; then the first bar <= 15:30 whose low <= VWAP and
        close > VWAP; stop = that bar's low x 0.998.  VWAP = cumulative typical price (h+l+c)/3 x volume from 09:30
        (no bar vwap on disk).
  B8    LULD proxy: gap of >= 5 missing minutes, last bar before >= 09:35, resumption bar <= 15:30, resumption open
        >= 5 % from the last close; entry = first close above the resumption bar's high (<= 15:30); stop = its low.
Stage A: TRAIN only -> 1763_trades_TRAIN.parquet, 1763_reads.csv.  Stage B (refuses unless 1763_reads.csv exists):
VAL once -> 1763_trades_VAL.parquet, 1763_val.csv.

Usage: cd /home/ec2-user/onemil && ulimit -v 2500000 && nice -n 19 timeout 600 python3 research/hod_entry/1763_score.py --stage A|B
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'ARROW_IO_THREADS'):
    os.environ.setdefault(_v, '1')
import sys, glob, time, sqlite3, resource, argparse, importlib.util
import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
HE = f'{ROOT}/research/hod_entry'
PM = f'{HE}/1762_cache/premarket'
PARTS = f'{HE}/1763_parts'


def _imp(name, path):
    """Import a research module whose file name starts with a digit."""
    sp = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


C = _imp('b1762', f'{HE}/1762_build.py')      # day_bars, causal_break_arm, walker, B (= 1752_build)
B = C.B                                         # simulate, SLIP_BP, constants
S = _imp('s1752', f'{HE}/1752_score.py')       # day_t, ex_top5
OPEN_M, START_M, LAST_M = B.OPEN_M, B.START_M, B.LAST_M
FLOOR = 1.0
BIGK = 10 ** 6
READS = ('HOD', 'B1', 'B1_HOD', 'B2_5m', 'B2_1m', 'B4', 'B7', 'B8', 'B6_HOD1030')


def log(m):
    """Timestamped progress line with peak RSS."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f'[{time.strftime("%H:%M:%S")}] {m} (peak RSS {rss:.0f} MB)', flush=True)


def cost_table():
    """1762 cost source: symbol-median half-spread % of the HOD arms (non-TEST), price-decile band fallback."""
    sig = pd.read_parquet(f'{ROOT}/research/hod_exit_lab/signals.parquet', columns=['symbol', 'half_spread', 'price', 'split'])
    sig = sig[sig.split != 'TEST']
    sig['hs_pct'] = sig.half_spread / sig.price
    sym = sig.groupby('symbol').hs_pct.median().to_dict()
    sig['dec'] = pd.qcut(np.log(sig.price), 10, labels=False)
    edges = np.quantile(np.log(sig.price), np.linspace(0, 1, 11))
    tab = sig.groupby('dec').hs_pct.median().to_numpy()
    return sym, (lambda p: tab[int(np.clip(np.searchsorted(edges, np.log(p)) - 1, 0, 9))])


def first_break(b, start_m):
    """Index of the first bar with m > start_m, m <= 15:30, high > prior HOD set >= 10 min before (causal), or None."""
    m, h = b['m'], b['h']
    cm = np.maximum.accumulate(h)
    idx = np.arange(len(m))
    set_idx = np.maximum.accumulate(np.where(h >= cm, idx, 0))
    Hp = np.r_[np.nan, cm[:-1]]
    setm = m[np.r_[0, set_idx[:-1]]]
    ok = (idx >= 1) & (m > start_m) & (m <= LAST_M) & (m - setm >= 10) & (h > Hp)
    bb = np.flatnonzero(ok)
    return int(bb[0]) if len(bb) else None


def trade(b, i, stop, hs_pct):
    """Market entry at close[i] + half-spread + 2 bp, imported 1752 simulate; None if R <= 0. Adds r_pct for the floor."""
    e_mid = b['c'][i]
    hs = hs_pct * e_mid
    e_cost = B.SLIP_BP * e_mid + hs
    res = B.simulate(b, i, e_mid, e_cost, stop, np.inf, BIGK, hs, False)
    if res is None:
        return None
    net, gross, why, _ = res
    return dict(entry_m=int(b['m'][i]), e_mid=e_mid, stop=stop, r_pct=(e_mid + e_cost - stop) / e_mid * 100,
                net_R=net, gross_R=gross, why=why)


def min3(b, i):
    """Min low of the last 3 bars ending at i."""
    return float(b['l'][max(0, i - 2):i + 1].min())


def first_close_above(b, level, lo_m, hi_m):
    """Index of the first bar with lo_m <= m <= hi_m and close > level, or None."""
    k = np.flatnonzero((b['m'] >= lo_m) & (b['m'] <= hi_m) & (b['c'] > level))
    return int(k[0]) if len(k) else None


def read_b2(b):
    """5-min (09:30-aligned) close above the prior HOD (set >= 10 min before the bucket start). Returns (i, stop)."""
    m, h, l, c = b['m'], b['h'], b['l'], b['c']
    for s in range(START_M, LAST_M - 4, 5):
        prior = m < s
        if prior.sum() < 1:
            continue
        hp = h[prior]
        H = hp.max()
        set_m = m[prior][int(np.argmax(hp))]
        if s - set_m < 10:
            continue
        inb = np.flatnonzero((m >= s) & (m < s + 5))
        if not len(inb):
            continue
        last = inb[-1]
        if c[last] > H:
            return int(last), float(l[inb].min())
    return None


def read_b4_level(o0):
    """Nearest $1/$5/$10 round number above the open that is >= 1 % away."""
    inc = 1.0 if o0 < 10 else (5.0 if o0 <= 50 else 10.0)
    lv = np.floor(o0 / inc) * inc + inc
    while lv / o0 - 1 < 0.01:
        lv += inc
    return lv


def read_b6(b):
    """HOD age at 10:30 (minutes since the 09:30-10:29 high was first set) and the 10:30 -> last-bar return (%)."""
    m, h, c = b['m'], b['h'], b['c']
    pre = np.flatnonzero(m < 630)
    post = np.flatnonzero(m >= 630)
    if not len(pre) or not len(post):
        return None
    hp = h[pre]
    age = 630 - m[pre][int(np.argmax(hp))]
    return age, (c[-1] / c[pre[-1]] - 1) * 100


def read_b7(b):
    """New HOD on a bar 09:31-09:59, then the first bar (<= 15:30) with low <= VWAP < close. Returns (i, stop)."""
    m, h, l, c, v = b['m'], b['h'], b['l'], b['c'], b['v']
    cm = np.r_[-np.inf, np.maximum.accumulate(h)[:-1]]
    nh = np.flatnonzero((h > cm) & (m > OPEN_M) & (m < 600) & (np.arange(len(m)) >= 1))
    if not len(nh):
        return None
    tp = (h + l + c) / 3
    cv = np.cumsum(v)
    vw = np.where(cv > 0, np.cumsum(tp * v) / np.maximum(cv, 1), np.nan)
    k = np.flatnonzero((np.arange(len(m)) > nh[0]) & (m <= LAST_M) & (l <= vw) & (c > vw))
    return (int(k[0]), float(l[k[0]] * 0.998)) if len(k) else None


def read_b8(b):
    """LULD-proxy halt resumption: first long trigger over the resumption bar high across qualifying gaps."""
    m, o, h, l, c = b['m'], b['o'], b['h'], b['l'], b['c']
    gaps = np.flatnonzero((np.diff(m) >= 6) & (m[:-1] >= 575) & (m[1:] <= LAST_M)
                          & (np.abs(o[1:] / c[:-1] - 1) >= 0.05)) + 1
    for j in gaps:
        k = np.flatnonzero((np.arange(len(m)) > j) & (m <= LAST_M) & (c > h[j]))
        if len(k):
            return int(k[0]), float(l[j]), int(len(gaps))
    return None


def premarket_high(day):
    """{symbol: pre-market high 04:00-09:29 ET} for one day (empty dict when the day file is missing)."""
    fn = f'{PM}/{day}.parquet'
    if not os.path.exists(fn):
        log(f'WARNING: no pre-market file for {day} - B1 treats every symbol as uncovered')
        return {}
    p = pd.read_parquet(fn, columns=['symbol', 't', 'h'])
    if not len(p):
        return {}
    p['m'] = B.to_minutes(p.t)
    p = p[(p.m >= 240) & (p.m <= 569)]
    return p.groupby('symbol').h.max().to_dict()


def day_rows(con, day, rows, pr_day, hs_sym, band, half):
    """All six reads for one day's population symbols -> (trade dicts, symbol-day records)."""
    bars = C.day_bars(con, day, pr_day, list(rows.symbol))
    by_sym = {s: g.sort_values('m') for s, g in bars[bars.symbol != 'SPY'].groupby('symbol')}
    pmh = premarket_high(day)
    out, recs = [], []
    for r in rows.itertuples(index=False):
        sb = by_sym.get(r.symbol)
        ok = sb is not None and len(sb) >= 30 and sb.m.iloc[0] <= OPEN_M + 5
        rec = dict(day=day, symbol=r.symbol, split=r.split, wk=r.wk, half=half, covered=bool(ok),
                   pm_cov=bool(r.symbol in pmh))
        recs.append(rec)
        if not ok:
            continue
        b = {k: sb[k].to_numpy(float) for k in ('o', 'h', 'l', 'c', 'v')}
        b['m'] = sb.m.to_numpy()
        hs_pct = hs_sym.get(r.symbol, np.nan)
        csrc = 'symbol_median'
        if not np.isfinite(hs_pct):
            hs_pct, csrc = band(b['o'][0]), 'price_band'
        base = dict(day=day, symbol=r.symbol, split=r.split, wk=r.wk, half=half, cost_src=csrc)

        def add(read, i, stop, **extra):
            """Simulate and append one trade (None-safe)."""
            if i is None:
                return
            t = trade(b, i, stop, hs_pct)
            if t is not None:
                out.append({**base, 'read': read, **t, **extra})
        ib = first_break(b, START_M)
        if ib is not None:
            add('HOD', ib, min3(b, ib))
            add('B2_1m', ib, float(b['l'][ib]))
            if rec['pm_cov']:
                add('B1_HOD', ib, min3(b, ib))
        if rec['pm_cov']:
            i1 = first_close_above(b, pmh[r.symbol], 571, LAST_M)
            if i1 is not None:
                add('B1', i1, min3(b, i1), level=pmh[r.symbol])
        r2 = read_b2(b)
        if r2:
            add('B2_5m', r2[0], r2[1])
        lv = read_b4_level(b['o'][0])
        i4 = first_close_above(b, lv, START_M + 1, LAST_M)
        if i4 is not None:
            add('B4', i4, min3(b, i4), level=lv)
        r6 = read_b6(b)
        if r6:
            rec['age'], rec['ret_1030'] = r6
            i6 = first_break(b, 629)
            if i6 is not None:
                add('B6_HOD1030', i6, min3(b, i6), age=r6[0])
        r7 = read_b7(b)
        if r7:
            add('B7', r7[0], r7[1])
        r8 = read_b8(b)
        if r8:
            add('B8', r8[0], r8[1], n_halts=r8[2])
    return out, recs


def build(split):
    """Per-day processing of one split with per-chunk checkpoints in 1763_parts/; returns (trades, symbol-days)."""
    os.makedirs(PARTS, exist_ok=True)
    pop = pd.read_csv(f'{HE}/cell_1428_population.csv', dtype={'symbol': str}, keep_default_na=False, na_values=[''])
    tr_days = sorted(pop.day[pop.split == 'TRAIN'].unique())
    halfmap = {d: ('H1' if k < len(tr_days) // 2 else 'H2') for k, d in enumerate(tr_days)}
    pop = pop[pop.split == split].reset_index(drop=True)
    hs_sym, band = cost_table()
    pr = pd.read_parquet(f'{HE}/1762_cache/presence.parquet')
    days = sorted(pop.day.unique())
    C.walker.wait_for_db_window()
    con = sqlite3.connect(B.DB, uri=True)
    t0 = time.time()
    for k in range(0, len(days), 10):
        ch = days[k:k + 10]
        name = f'{PARTS}/{split}_{ch[0]}'
        if os.path.exists(name + '.t.parquet'):
            continue
        C.walker.wait_for_db_window()
        tr, rc = [], []
        for d in ch:
            o, r = day_rows(con, d, pop[pop.day == d], pr[pr.day == d], hs_sym, band, halfmap.get(d, 'V'))
            tr += o; rc += r
        pd.DataFrame(rc).to_parquet(name + '.r.parquet', index=False)
        pd.DataFrame(tr).to_parquet(name + '.t.tmp', index=False)
        os.replace(name + '.t.tmp', name + '.t.parquet')
        log(f'{split} {ch[0]}..{ch[-1]}: {len(tr)} trades; {k + len(ch)}/{len(days)} days, {time.time() - t0:.0f}s')
    con.close()
    t = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f'{PARTS}/{split}_*.t.parquet'))], ignore_index=True)
    r = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f'{PARTS}/{split}_*.r.parquet'))], ignore_index=True)
    t.to_parquet(f'{HE}/1763_trades_{split}.parquet', index=False)
    return t, r, sorted(pop.wk.unique()), {h: sorted(pop.wk[pop.day.map(halfmap) == h].unique()) for h in ('H1', 'H2')}


def book(sub, weeks, pool=None, seed=1763):
    """Book stats: n, n/wk, net, day-t, se, ex-top-5 %, capped +3 R, weekly P10, green share vs count-matched null."""
    sub = sub[sub.r_pct >= FLOOR]
    mu, t, se = S.day_t(sub.net_R, sub.day)
    wk = sub.groupby('wk').net_R.sum().reindex(weeks, fill_value=0.0)
    out = dict(n=len(sub), per_wk=len(sub) / len(weeks), mean_net=mu, day_t=t, se=se, MDE80=2.8 * se,
               gross=float(sub.gross_R.mean()) if len(sub) else np.nan, extop5=S.ex_top5(sub.net_R),
               wincap3=float(np.minimum(sub.net_R, 3).mean()) if len(sub) else np.nan,
               wk_P10=float(np.quantile(wk, 0.10)), green=float((wk > 0).mean()))
    if pool is not None and len(sub):
        pool = pool[pool.r_pct >= FLOOR]
        rng = np.random.default_rng(seed)
        cnt = sub.groupby('wk').size()
        pools = {w: pool[pool.wk == w].net_R.to_numpy() for w in cnt.index}
        nul = []
        for _ in range(500):
            s = pd.Series({w: (rng.choice(pools[w], cnt[w], replace=True).sum() if len(pools[w]) else 0.0)
                           for w in cnt.index}).reindex(weeks, fill_value=0.0)
            nul.append(float((s > 0).mean()))
        out['null_green'] = float(np.mean(nul))
    return out


def diff_t(x, d, days):
    """Mean difference (in-group minus rest) with a day-clustered t (OLS on an indicator, clusters = days)."""
    x, d = np.asarray(x, float), np.asarray(d, float)
    if d.sum() < 5 or (1 - d).sum() < 5:
        return np.nan, np.nan
    dc = d - d.mean()
    beta = (dc * x).sum() / (dc ** 2).sum()
    e = x - x.mean() - beta * dc
    g = pd.Series(dc * e).groupby(np.asarray(days)).sum()
    se = np.sqrt((g ** 2).sum()) / (dc ** 2).sum()
    return beta, (beta / se if se > 0 else np.nan)


def b6_table(t, r, label):
    """B6 bucket rows: HOD-rule (>= 10:30) net R and 10:30 -> close return by HOD-age bucket, diff vs rest + t."""
    rows = []
    bk = lambda a: np.where(a <= 15, '<=15', np.where(a <= 45, '15-45', '>45'))
    h6 = t[(t.read == 'B6_HOD1030') & (t.r_pct >= FLOOR)].copy()
    h6['bucket'] = bk(h6.age.to_numpy())
    rr = r[r.covered & r.age.notna()].copy() if 'age' in r else r.iloc[:0]
    rr['bucket'] = bk(rr.age.to_numpy())
    for b_ in ('<=15', '15-45', '>45'):
        g = h6[h6.bucket == b_]
        d1, t1 = diff_t(h6.net_R, h6.bucket == b_, h6.day)
        d2, t2 = diff_t(rr.ret_1030, rr.bucket == b_, rr.day)
        rows.append(dict(sample=label, bucket=b_, n_break=len(g), break_net=float(g.net_R.mean()) if len(g) else np.nan,
                         break_diff=d1, break_diff_t=t1, n_days=int((rr.bucket == b_).sum()),
                         ret_1030_mean=float(rr.ret_1030[rr.bucket == b_].mean()), ret_diff=d2, ret_diff_t=t2))
    return rows


def table(t, r, weeks, label, b2pair=True):
    """One row per read for one sample (H1/H2/TRAIN/VAL)."""
    rows = []
    for rd in READS:
        s = t[t.read == rd]
        if not len(s):
            rows.append(dict(sample=label, read=rd, n=0)); continue
        bk = book(s, weeks, pool=t[t.read == 'HOD'])
        bk['floor_removed'] = int((s.r_pct < FLOOR).sum())
        bk['symbol_median_cost_share'] = float((s.cost_src == 'symbol_median').mean())
        rows.append(dict(sample=label, read=rd, **bk))
    if b2pair:
        a = t[(t.read == 'B2_5m') & (t.r_pct >= FLOOR)].set_index(['day', 'symbol']).net_R
        b_ = t[(t.read == 'B2_1m') & (t.r_pct >= FLOOR)].set_index(['day', 'symbol']).net_R
        j = pd.concat([a.rename('r5'), b_.rename('r1')], axis=1, join='inner').reset_index()
        mu, tt, se = S.day_t(j.r5 - j.r1, j.day)
        rows.append(dict(sample=label, read='B2_pairedDelta', n=len(j), mean_net=mu, day_t=tt, se=se))
    rc = r[r.covered]
    rows.append(dict(sample=label, read='PM_coverage', n=len(rc), mean_net=float(rc.pm_cov.mean()) if len(rc) else np.nan))
    return rows


def stage_a():
    """TRAIN halves + TRAIN tables -> 1763_reads.csv (+ B6 bucket rows)."""
    t, r, weeks, wh = build('TRAIN')
    log(f'TRAIN trades {len(t):,}; reads {t.read.value_counts().to_dict()}')
    rows, b6 = [], []
    for lab, tt, rr, ww in (('H1', t[t.half == 'H1'], r[r.half == 'H1'], wh['H1']),
                            ('H2', t[t.half == 'H2'], r[r.half == 'H2'], wh['H2']), ('TRAIN', t, r, weeks)):
        rows += table(tt, rr, ww, lab)
        b6 += b6_table(tt, rr, lab)
    pd.concat([pd.DataFrame(rows), pd.DataFrame(b6).assign(read='B6_bucket')], ignore_index=True).to_csv(
        f'{HE}/1763_reads.csv', index=False)
    log('1763_reads.csv written (TRAIN frozen before VAL)')


def stage_b():
    """One VAL read -> 1763_val.csv with the per-read kill verdicts."""
    assert os.path.exists(f'{HE}/1763_reads.csv'), 'TRAIN table not written - run stage A first'
    tr = pd.read_csv(f'{HE}/1763_reads.csv')
    t, r, weeks, _ = build('VAL')
    log(f'VAL trades {len(t):,}')
    v = pd.DataFrame(table(t, r, weeks, 'VAL'))
    b6v = pd.DataFrame(b6_table(t, r, 'VAL'))
    cov_tr = float(tr[(tr['sample'] == 'TRAIN') & (tr.read == 'PM_coverage')].mean_net.iloc[0])
    cov_v = float(v[v.read == 'PM_coverage'].mean_net.iloc[0])
    pdv = v[v.read == 'B2_pairedDelta'].iloc[0]
    out = []
    for rd, minwk in (('B1', 3), ('B2_5m', 3), ('B4', 3), ('B7', 3), ('B8', 1)):
        x = v[v.read == rd].iloc[0]
        h1 = tr[(tr['sample'] == 'H1') & (tr.read == rd)].mean_net.iloc[0]
        h2 = tr[(tr['sample'] == 'H2') & (tr.read == rd)].mean_net.iloc[0]
        k = {'val_net_le0': not (x.mean_net > 0), 'day_t_lt2': not (x.day_t >= 2),
             'train_halves_opposite': bool(np.sign(h1) != np.sign(h2)), 'val_extop5_le0': not (x.extop5 > 0),
             f'fills_lt{minwk}': not (x.per_wk >= minwk)}
        if rd == 'B1':
            k['coverage_lt90_VOID'] = min(cov_tr, cov_v) < 0.90
        if rd == 'B2_5m':
            k['paired_delta_le0'] = not (pdv.mean_net > 0)
        out.append(dict(read=rd, **x.drop(['sample', 'read']).to_dict(), H1=h1, H2=h2,
                        **{f'kill_{a}': bool(b) for a, b in k.items()}, verdict='KILL' if any(k.values()) else 'SURVIVE'))
    b6t = pd.concat([tr[tr.read == 'B6_bucket'], b6v.assign(read='B6_bucket')])
    surv = []
    for b_ in ('<=15', '15-45', '>45'):
        for col in ('break_diff_t', 'ret_diff_t'):
            ts = [b6t[(b6t['sample'] == s) & (b6t.bucket == b_)][col].iloc[0] for s in ('H1', 'H2', 'VAL')]
            if all(abs(z) >= 2 for z in ts) and len({np.sign(z) for z in ts}) == 1:
                surv.append(f'{b_}:{col}')
    out.append(dict(read='B6', verdict='SURVIVE' if surv else 'KILL', b6_passing=';'.join(surv)))
    for rd in ('HOD', 'B1_HOD', 'B2_1m', 'B2_pairedDelta', 'B6_HOD1030', 'PM_coverage'):
        out.append(dict(read=rd, **v[v.read == rd].iloc[0].drop(['sample', 'read']).dropna().to_dict()))
    o = pd.DataFrame(out)
    o = pd.concat([o, b6v.assign(read='B6_bucket')], ignore_index=True)
    o.to_csv(f'{HE}/1763_val.csv', index=False)
    for rec in out:
        log(str({k: (round(x, 3) if isinstance(x, float) else x) for k, x in rec.items() if not str(k).startswith('kill_') or x}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', required=True, choices=['A', 'B'])
    a = ap.parse_args()
    stage_a() if a.stage == 'A' else stage_b()
