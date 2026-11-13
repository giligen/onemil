#!/usr/bin/env python3
"""Cell 1,757 scorer (PREREG_1757.md, frozen): three flow reads on the HOD-break arm population.

F1 late-day momentum : arms within 1 % of the path high at 15:30 (volume proxy impossible: no volume column),
                       buy close(15:30 bar) + half_spread + 2 bp, sell close of the LAST bar (15:55; paths end at m=955,
                       there is no 15:59 bar) less 2 bp.  Comparator = arms NOT near the high.
F2 overnight drift   : arms whose RAW daily close sits in the top / bottom third of the RAW daily range; buy the daily
                       close, sell the next session's raw open, 2 bp per leg, no spread; excess vs the equal-weight
                       overnight mean of the liquid universe (close >= $1, day $vol >= $1M) on the same day.
                       Overnight corporate actions (raw return differs from adjusted by > 2 %) are dropped and counted.
F4 breadth -> index  : per day count of arms with entry_m < 630 (10:30 ET), divided by its trailing 60-day mean (>= 20 prior
                       days); quintile cuts frozen on TRAIN; outcome = SPY 10:30 -> last-bar return from paths.spy_close.

TEST rows are dropped at load and counted.  Each read writes its TRAIN rows to the CSV before VAL is computed.
Usage: cd /home/ec2-user/onemil && ulimit -v 2500000 && nice -n 19 timeout 600 python3 research/hod_entry/1757_score.py
"""
import json
import os
import time

os.environ.setdefault('ARROW_DEFAULT_MEMORY_POOL', 'system')   # jemalloc reserves too much virtual memory under ulimit -v

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

BASE = '/home/ec2-user/onemil/research'
EL = f'{BASE}/hod_exit_lab'
OUT = f'{BASE}/hod_entry'
C46, C48 = f'{BASE}/signal_search/1746_cache', f'{BASE}/signal_search/1748_cache'
pa.set_cpu_count(1)
pa.set_io_thread_count(1)
LEG = 2.0           # bp per leg
NEAR = 0.01         # within 1 % of the path high
M_1530, M_LAST, M_1030 = 930, 955, 630
CORP = 0.02         # raw-vs-adjusted overnight return gap that flags a corporate action


def log(*a):
    """Timestamped progress line, flushed."""
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def cl_t(x, d):
    """(mean, day-clustered t) of x grouped by day label array d."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 5:
        return np.nan, np.nan
    mu = x.mean()
    e = pd.Series(x - mu).groupby(np.asarray(d)).sum()
    se = np.sqrt((e ** 2).sum()) / n
    return mu, (mu / se if se > 0 else np.nan)


def ex_top5(x):
    """Mean after dropping the top 5 % of values."""
    x = np.sort(np.asarray(x, dtype=float))
    k = int(np.ceil(0.05 * len(x)))
    return x[:len(x) - k].mean() if len(x) > k else np.nan


def stat_row(label, part, x, days, ndays):
    """One table row: n, n/wk (5 trading days), mean bp, day-clustered t, ex-top-5 % mean, hit rate."""
    x = np.asarray(x, dtype=float)
    mu, t = cl_t(x, days) if len(x) else (np.nan, np.nan)
    return {'group': label, 'part': part, 'n': len(x), 'n_per_wk': len(x) / (ndays / 5.0) if ndays else np.nan,
            'mean_bp': mu, 'day_t': t, 'ex_top5_bp': ex_top5(x) if len(x) else np.nan,
            'hit': float((x > 0).mean()) if len(x) else np.nan}


def load_signals():
    """All non-TEST signals (TRAIN, VAL) with half_spread in bp; counts the TEST signals dropped."""
    sig = pd.read_parquet(f'{EL}/signals.parquet')
    assert not sig.duplicated(['day', 'symbol']).any()
    n_test = int((sig.split == 'TEST').sum())
    sig = sig[sig.split != 'TEST'].reset_index(drop=True)
    sig['hs_bp'] = sig.half_spread / sig.price * 1e4
    sig['arm'] = np.arange(len(sig), dtype=np.int32)
    log(f'signals: TRAIN {int((sig.split == "TRAIN").sum())}, VAL {int((sig.split == "VAL").sum())}; TEST dropped {n_test}')
    return sig, n_test


def read_paths(sig):
    """One pass over paths.parquet: per-arm path stats (F1) and per-day SPY anchors (F4). TEST rows dropped, counted."""
    keys = pd.Index(sig.day + '|' + sig.symbol)
    sp_days = set(sig.day)
    pf = pq.ParquetFile(f'{EL}/paths.parquet')
    parts, spy_parts, n_other = [], [], 0
    for i in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(i, columns=['day', 'symbol', 'm', 'h', 'c', 'spy_close'])
        k = pc.binary_join_element_wise(t['day'], t['symbol'], '|').to_pandas()
        idx = keys.get_indexer(k)
        sel = idx >= 0
        n_other += int((~sel).sum())
        m = t['m'].to_numpy().astype(np.int16)
        parts.append(pd.DataFrame({'arm': idx[sel].astype(np.int32), 'm': m[sel],
                                   'h': t['h'].to_numpy()[sel].astype(np.float32),
                                   'c': t['c'].to_numpy()[sel].astype(np.float32)}))
        d = t['day'].to_pandas()
        sp = t['spy_close'].to_numpy().astype(np.float64)
        s2 = d.isin(sp_days).to_numpy() & ~np.isnan(sp) & (((m >= 620) & (m <= 660)) | (m >= 945))
        spy_parts.append(pd.DataFrame({'day': d[s2].to_numpy(), 'm': m[s2], 'spy': sp[s2]}).drop_duplicates(['day', 'm']))
        log(f'row group {i}: kept {int(sel.sum()):,}; other (TEST/unkept) rows so far {n_other:,}')
        del t, k
    df = pd.concat(parts, ignore_index=True).sort_values(['arm', 'm'], kind='stable').reset_index(drop=True)
    spy = pd.concat(spy_parts, ignore_index=True).drop_duplicates(['day', 'm'])
    return df, spy, n_other


def f1_arm_table(df, sig):
    """Per-arm F1 return (bp, net) and near-high flag; arms whose path lacks the 15:30 or the last bar are counted out."""
    pre = df[df.m <= M_1530]
    hmax = pre.groupby('arm').h.max()
    c30 = df[df.m == M_1530].set_index('arm').c
    cend = df[df.m == M_LAST].set_index('arm').c
    t = sig.set_index('arm')[['day', 'symbol', 'split', 'half', 'hs_bp', 'price']].join(
        pd.DataFrame({'hmax': hmax, 'c30': c30, 'cend': cend}))
    t = t.dropna(subset=['hmax', 'c30', 'cend', 'hs_bp'])
    t['dist'] = 1.0 - t.c30 / t.hmax
    t['near'] = t.dist <= NEAR
    buy = t.c30 * (1 + t.hs_bp / 1e4 + LEG / 1e4)
    sell = t.cend * (1 - LEG / 1e4)
    t['net_bp'] = (sell / buy - 1.0) * 1e4
    t['gross_bp'] = (t.cend / t.c30 - 1.0) * 1e4
    return t.reset_index(), len(sig) - len(t)


def table_for(frame, value_col, groups, parts, ndays_by_part, label_fn):
    """Rows for each (group, part): parts = {name: boolean-mask-fn over frame}."""
    rows = []
    for gname, gmask in groups.items():
        for pname, pmask in parts.items():
            f = frame[gmask(frame) & pmask(frame)]
            rows.append(stat_row(gname, pname, f[value_col].to_numpy(), f.day.to_numpy(), ndays_by_part[pname]))
    return rows


def split_parts(sig):
    """Part definitions (TRAIN, TRAIN H1/H2, VAL) and their trading-day counts."""
    parts = {'TRAIN': lambda f: f.split == 'TRAIN', 'TRAIN_H1': lambda f: (f.split == 'TRAIN') & (f.half == 'H1'),
             'TRAIN_H2': lambda f: (f.split == 'TRAIN') & (f.half == 'H2'), 'VAL': lambda f: f.split == 'VAL'}
    nd = {'TRAIN': sig[sig.split == 'TRAIN'].day.nunique(),
          'TRAIN_H1': sig[(sig.split == 'TRAIN') & (sig.half == 'H1')].day.nunique(),
          'TRAIN_H2': sig[(sig.split == 'TRAIN') & (sig.half == 'H2')].day.nunique(),
          'VAL': sig[sig.split == 'VAL'].day.nunique()}
    return parts, nd


def run_read(name, frame, col, groups, sig, fname):
    """Write TRAIN rows, then (only after the file exists) VAL rows; return the full table."""
    parts, nd = split_parts(sig)
    tr_parts = {k: v for k, v in parts.items() if k != 'VAL'}
    rows = table_for(frame, col, groups, tr_parts, nd, None)
    tr = pd.DataFrame(rows)
    tr.to_csv(fname, index=False)
    log(f'{name} TRAIN table written:\n{tr.round(2).to_string()}')
    rows += table_for(frame, col, groups, {'VAL': parts['VAL']}, nd, None)
    full = pd.DataFrame(rows)
    full.to_csv(fname, index=False)
    log(f'{name} VAL rows appended:\n{full[full.part == "VAL"].round(2).to_string()}')
    return full


def verdict(full, g, tag, need_excess=False):
    """Apply the PREREG kill rules to group g: VAL mean <= 20 bp or t < 2; halves opposite; ex-top-5 % <= 0; < 3/wk."""
    r = lambda p: full[(full.group == g) & (full.part == p)].iloc[0]
    v, h1, h2 = r('VAL'), r('TRAIN_H1'), r('TRAIN_H2')
    why = []
    if not (v.mean_bp > 20): why.append(f'VAL mean {v.mean_bp:.1f} <= 20 bp')
    if not (v.day_t >= 2): why.append(f'VAL day-t {v.day_t:.2f} < 2')
    if np.sign(h1.mean_bp) != np.sign(h2.mean_bp): why.append(f'TRAIN halves opposite ({h1.mean_bp:.1f}/{h2.mean_bp:.1f})')
    if not (v.ex_top5_bp > 0): why.append(f'VAL ex-top-5 % {v.ex_top5_bp:.1f} <= 0')
    if not (v.n_per_wk >= 3): why.append(f'VAL {v.n_per_wk:.2f}/wk < 3')
    return {'read': tag, 'group': g, 'verdict': 'KILL' if why else 'SURVIVE', 'reasons': why}


def run_f1(sig, df):
    """F1 late-day momentum read."""
    t, n_out = f1_arm_table(df, sig)
    log(f'F1: arms scored {len(t)}, dropped (path lacks 15:30/last bar or no spread) {n_out}; near-high share {t.near.mean():.3f}')
    groups = {'near_high': lambda f: f.near, 'not_near': lambda f: ~f.near}
    full = run_read('F1', t, 'net_bp', groups, sig, f'{OUT}/1757_f1.csv')
    gross = {g: float(t[(t.near == (g == 'near_high')) & (t.split == 'VAL')].gross_bp.mean()) for g in groups}
    return full, verdict(full, 'near_high', 'F1'), {'n_dropped': n_out, 'gross_bp_val': gross}


SYM = {}   # symbol -> int code shared by the daily loader and the arm join


def load_daily():
    """RAW + adjusted daily bars 2025+ from both caches (1748 wins ties); returns frame sorted by symbol, date."""
    out = []
    for c in (C48, C46):
        pr, pa_ = (pq.ParquetFile(f'{c}/{f}') for f in ('bars_raw.parquet', 'bars_all.parquet'))
        assert pr.metadata.num_row_groups == pa_.metadata.num_row_groups
        n = 0
        for g in range(pr.metadata.num_row_groups):
            raw = pr.read_row_group(g, use_threads=False).to_pandas()
            adj = pa_.read_row_group(g, columns=['symbol', 'date', 'o', 'c'], use_threads=False).to_pandas()
            raw = raw[raw.date >= '2025-01-01']
            adj = adj[adj.date >= '2025-01-01']
            m = raw.merge(adj, on=['symbol', 'date'], suffixes=('', '_a'))
            for sy in m.symbol.unique():
                SYM.setdefault(sy, len(SYM))
            m['symbol'] = m.symbol.map(SYM).astype(np.int32)
            m['date'] = pd.to_datetime(m.date).to_numpy().astype('datetime64[D]').astype(np.int32)
            for col in ('o', 'h', 'l', 'c', 'v', 'o_a', 'c_a'):
                m[col] = m[col].astype(np.float32 if col != 'v' else np.float64)
            out.append(m)
            n += len(m)
            del raw, adj
        log(f'daily {c}: merged {n:,}')
    d = pd.concat(out, ignore_index=True)
    del out
    d = d.drop_duplicates(['symbol', 'date'], keep='first')
    return d.reset_index(drop=True)


def overnight_frame(d):
    """Per symbol-day overnight return (raw close -> next session raw open), flagging overnight corporate actions."""
    cnt = d.groupby('date').size()
    cal = list(cnt[cnt >= 1000].index)
    nxt = dict(zip(cal[:-1], cal[1:]))
    d = d[d.date.isin(nxt)].copy()
    d['ndate'] = d.date.map(nxt)
    nx = d[['symbol', 'date', 'o', 'o_a']].rename(columns={'date': 'ndate', 'o': 'o_n', 'o_a': 'oa_n'})
    d = d.merge(nx, on=['symbol', 'ndate'], how='left')
    d = d.dropna(subset=['o_n'])
    d['ret'] = d.o_n / d.c - 1.0
    d['ret_a'] = d.oa_n / d.c_a - 1.0
    d['corp'] = (d.ret - d.ret_a).abs() > CORP
    return d


def run_f2(sig):
    """F2 overnight drift read: top-third vs bottom-third closers, excess vs liquid-universe overnight mean."""
    d = overnight_frame(load_daily())
    n_corp = int(d.corp.sum())
    d = d[~d.corp]
    liq = d[(d.c >= 1.0) & (d.c * d.v >= 1e6)]
    umean = liq.groupby('date').ret.mean().rename('u_mean')
    log(f'F2: daily overnight rows {len(d):,}, corporate-action rows dropped {n_corp}, liquid-universe days {len(umean)}')
    s2 = sig.copy()
    s2['sym'] = s2.symbol.map(SYM).fillna(-1).astype(np.int32)
    s2['dord'] = pd.to_datetime(s2.day).to_numpy().astype('datetime64[D]').astype(np.int32)
    dd = d[['symbol', 'date', 'h', 'l', 'c', 'ret']].rename(columns={'symbol': 'sym', 'date': 'dord'})
    a = s2.merge(dd, on=['sym', 'dord'], how='left')
    a = a.merge(umean, left_on='dord', right_index=True, how='left')
    n_nodaily = int(a.ret.isna().sum())
    a = a.dropna(subset=['ret', 'u_mean']).copy()
    a = a[a.h > a.l]
    a['pos'] = (a.c - a.l) / (a.h - a.l)
    a['exc_bp'] = (a.ret - a.u_mean) * 1e4 - 2 * LEG
    a['raw_bp'] = a.ret * 1e4 - 2 * LEG
    log(f'F2: arms {len(a)}, without a daily/overnight row {n_nodaily}; max |ret| {a.ret.abs().max():.3f}, '
        f'|ret|>50 %: {int((a.ret.abs() > 0.5).sum())}')
    groups = {'top_third': lambda f: f.pos >= 2 / 3, 'bottom_third': lambda f: f.pos <= 1 / 3, 'all_arms': lambda f: f.pos >= 0}
    full = run_read('F2', a, 'exc_bp', groups, sig, f'{OUT}/1757_f2.csv')
    v = a[a.split == 'VAL']
    cov = len(a) / len(sig)
    log(f'F2 availability: {cov:.3f} of arms scored (rail >= 0.80); unscored are mostly 2x wrapper ETFs absent from the daily cache')
    extra = {'n_corp_dropped': n_corp, 'n_no_daily': n_nodaily, 'coverage': cov, 'coverage_rail_pass': bool(cov >= 0.80),
             'val_raw_net_bp': {g: float(v[f(v)].raw_bp.mean()) for g, f in groups.items()},
             'val_universe_mean_bp': float(v.u_mean.mean() * 1e4),
             'val_top_minus_bottom_bp': float(v[groups['top_third'](v)].exc_bp.mean() - v[groups['bottom_third'](v)].exc_bp.mean())}
    ver = verdict(full, 'top_third', 'F2')
    if cov < 0.80:
        ver['reasons'].append(f'availability rail FAILED: coverage {cov:.3f} < 0.80 (VOID)')
        ver['verdict'] = 'KILL'
    # year-sign: 2025 = TRAIN, 2026 = VAL; 2022 not in this population (data start 2025-01)
    return full, ver, extra


def spy_returns(spy):
    """Per day SPY 10:30 -> last-bar return (bp). Start = last bar m <= 629, else first bar in 630..660 (flagged late)."""
    rows = []
    for day, g in spy.groupby('day'):
        pre = g[g.m <= 629]
        if len(pre):
            b = pre.sort_values('m').iloc[-1]
            late = False
        else:
            lt = g[(g.m >= 630) & (g.m <= 660)].sort_values('m')
            if not len(lt):
                continue
            b, late = lt.iloc[0], True
        e = g[g.m >= 945].sort_values('m')
        if not len(e):
            continue
        rows.append({'day': day, 'spy_ret_bp': (e.iloc[-1].spy / b.spy - 1.0) * 1e4, 'late_start': late})
    return pd.DataFrame(rows)


def run_f4(sig_all, spy):
    """F4 breadth -> index read. Quintile cuts frozen on TRAIN; day-level."""
    days = sig_all.groupby('day').agg(split=('split', 'first'), half=('half', 'first')).sort_index()
    days['breadth'] = sig_all[sig_all.entry_m < M_1030].groupby('day').size().reindex(days.index).fillna(0.0)
    trail = days.breadth.shift(1).rolling(60, min_periods=20).mean()
    days['ratio'] = days.breadth / trail
    sp = spy_returns(spy).set_index('day')
    days = days.join(sp)
    log(f'F4: days {len(days)}, ratio defined {int(days.ratio.notna().sum())}, SPY return available {int(days.spy_ret_bp.notna().sum())}, '
        f'late-start {int(days.late_start.fillna(False).sum())}')
    ok = days.dropna(subset=['ratio', 'spy_ret_bp']).copy()
    tr = ok[ok.split == 'TRAIN']
    cuts = np.quantile(tr.ratio, [0.2, 0.4, 0.6, 0.8])
    ok['q'] = np.searchsorted(cuts, ok.ratio.to_numpy(), side='right') + 1
    ok['net_bp'] = ok.spy_ret_bp - 1.0

    def qtab(part, f):
        """Per-quintile mean/t/n for the part mask f."""
        rows = []
        s = ok[f(ok)]
        for q in range(1, 6):
            x = s[s.q == q].spy_ret_bp.to_numpy()
            mu, t = (x.mean(), x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 and x.std(ddof=1) > 0 else (np.nan, np.nan)
            rows.append({'part': part, 'row': f'Q{q}', 'n_days': len(x), 'mean_bp': mu, 't': t,
                         'pos_share': float((x > 0).mean()) if len(x) else np.nan})
        top, bot = s[s.q == 5].spy_ret_bp.to_numpy(), s[s.q == 1].spy_ret_bp.to_numpy()
        diff = top.mean() - bot.mean() if len(top) and len(bot) else np.nan
        se = np.sqrt(top.var(ddof=1) / len(top) + bot.var(ddof=1) / len(bot)) if len(top) > 2 and len(bot) > 2 else np.nan
        rows.append({'part': part, 'row': 'Q5-Q1', 'n_days': len(top) + len(bot), 'mean_bp': diff, 't': diff / se if se else np.nan,
                     'pos_share': np.nan})
        xl = top - 1.0
        rows.append({'part': part, 'row': 'long_SPY_Q5_net1bp', 'n_days': len(xl), 'mean_bp': xl.mean() if len(xl) else np.nan,
                     't': xl.mean() / (xl.std(ddof=1) / np.sqrt(len(xl))) if len(xl) > 2 and xl.std(ddof=1) > 0 else np.nan,
                     'pos_share': float((xl > 0).mean()) if len(xl) else np.nan})
        return rows

    fname = f'{OUT}/1757_f4.csv'
    rows = []
    for p, f in (('TRAIN', lambda s: s.split == 'TRAIN'), ('TRAIN_H1', lambda s: (s.split == 'TRAIN') & (s.half == 'H1')),
                 ('TRAIN_H2', lambda s: (s.split == 'TRAIN') & (s.half == 'H2'))):
        rows += qtab(p, f)
    pd.DataFrame(rows).to_csv(fname, index=False)
    log(f'F4 TRAIN table written (cuts {np.round(cuts, 3).tolist()}):\n{pd.DataFrame(rows).round(2).to_string()}')
    rows += qtab('VAL', lambda s: s.split == 'VAL')
    full = pd.DataFrame(rows)
    full.to_csv(fname, index=False)
    log(f'F4 VAL rows:\n{full[full.part == "VAL"].round(2).to_string()}')

    g = lambda p, r: full[(full.part == p) & (full.row == r)].iloc[0]
    tq = full[(full.part == 'TRAIN') & full.row.str.match(r'Q\d$')].mean_bp.to_numpy()
    spear = float(pd.Series(tq).rank().corr(pd.Series(np.arange(5.0)).rank())) if not np.isnan(tq).any() else np.nan
    v, h1, h2 = g('VAL', 'Q5-Q1'), g('TRAIN_H1', 'Q5-Q1'), g('TRAIN_H2', 'Q5-Q1')
    why = []
    if not (v.mean_bp > 10): why.append(f'VAL Q5-Q1 {v.mean_bp:.1f} <= 10 bp')
    if not (v.t >= 2): why.append(f'VAL t {v.t:.2f} < 2')
    if np.sign(h1.mean_bp) != np.sign(h2.mean_bp): why.append(f'TRAIN halves opposite ({h1.mean_bp:.1f}/{h2.mean_bp:.1f})')
    if not (spear >= 0.8): why.append(f'TRAIN not monotone (Spearman {spear:.2f})')
    late = ok.groupby('q').late_start.mean().round(2).to_dict()
    return full, {'read': 'F4', 'verdict': 'KILL' if why else 'SURVIVE', 'reasons': why}, \
        {'train_spearman': spear, 'cuts': cuts.tolist(), 'late_start_share_by_q': late,
         'days_total': len(days), 'days_scored': len(ok)}


def main():
    """Run F1, F2, F4; write the verdict json."""
    sig, n_test = load_signals()
    df, spy, n_other = read_paths(sig)
    log(f'path rows kept {len(df):,}; rows of other keys (TEST + unkept arms) {n_other:,}')
    res = {'test_signals_dropped': n_test, 'other_path_rows_dropped': n_other}
    f1, v1, x1 = run_f1(sig, df)
    del df
    f4, v4, x4 = run_f4(sig, spy)
    f2, v2, x2 = run_f2(sig)
    res.update({'F1': {'verdict': v1, 'extra': x1}, 'F2': {'verdict': v2, 'extra': x2}, 'F4': {'verdict': v4, 'extra': x4}})
    with open(f'{OUT}/1757_verdicts.json', 'w') as f:
        json.dump(res, f, indent=1, default=str)
    log('VERDICTS', json.dumps(res, indent=1, default=str))


if __name__ == '__main__':
    main()
