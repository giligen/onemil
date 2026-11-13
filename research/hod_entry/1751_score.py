#!/usr/bin/env python3
"""Cell 1,751 — confirmation entry on the HOD break, rule learned from winners vs losers.

Spec: research/hod_entry/PREREG_1751.md (frozen). Two stages, run separately:
  --stage A : TRAIN only (VAL and TEST dropped at load). Anatomy (winners vs losers of the
              forward return from minute N), the full one-rule grid, pick by TRAIN-H2 net among
              cells with >= 3 fills/wk; writes 1751_anatomy.csv, 1751_grid.csv, 1751_rule.json
              and the frozen-rule header of RESULT_1751.md.
  --stage B : VAL only, read once; scores the frozen rule from 1751_rule.json; writes 1751_val.csv.

Conventions (PREREG_1751 + PREREG_1740): level = B0 `entry` (the open of bar entry_m); R0 = `R`;
state at the close of bar N (minutes_since_entry == N, else the last bar <= N); held = min(l) over
bars 1..N > level. Forward exit restarted at N from bar N+1: stop at the original stop (gap-through
fills at the open), target at the original target (fills at target), a bar touching both = stop,
else the open of the 15:55 bar. Costs: entry c_N + half_spread + 2 bp; every exit leg half_spread
+ 2 bp; stop exits a further 13 bp slip. half_spread in signals.parquet is in dollars per share
(verified against b0_trades cost_R).
"""
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

BASE = '/home/ec2-user/onemil/research'
EL = f'{BASE}/hod_exit_lab'
OUT = f'{BASE}/hod_entry'
NS = [3, 5, 10, 15, 20, 30]
LEG = 0.0002      # 2 bp per leg
SLIP = 0.0013     # stop slip, PREREG_1740
EOD_M = 955       # 15:55 bar, minute of day
OPEN_M = 570      # 09:30


def log(*a):
    """Timestamped progress line, flushed."""
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def load(keep_split):
    """Load signals, B0 and paths for ONE split; TEST (and the other split) dropped at load, counted."""
    sig = pd.read_parquet(f'{EL}/signals.parquet')
    assert not sig.duplicated(['day', 'symbol']).any(), 'day,symbol not unique'
    test_keys = pd.Index((sig.day + '|' + sig.symbol)[sig.split == 'TEST'])
    n_test_sig = int((sig.split == 'TEST').sum())
    sig = sig[sig.split == keep_split]
    nohs = int(sig.half_spread.isna().sum())
    if nohs:
        log(f'WARNING: {keep_split}: {nohs} arms have no measured half_spread; dropped (cost unknowable)')
    sig = sig[sig.half_spread.notna()].reset_index(drop=True)
    sig['arm'] = np.arange(len(sig), dtype=np.int32)
    b0 = pd.read_csv(f'{EL}/b0_trades.csv')
    n_test_b0 = int((b0.split == 'TEST').sum())
    b0 = b0[b0.split == keep_split]
    sig = sig.merge(b0[['day', 'symbol', 'entry_m', 'net_R', 'exit_price', 'why']].rename(
        columns={'net_R': 'b0_net', 'exit_price': 'b0_exit', 'why': 'b0_why'}),
        on=['day', 'symbol', 'entry_m'], how='left', validate='1:1')
    if sig.b0_net.isna().any():
        log('WARNING: arms without a B0 row:', int(sig.b0_net.isna().sum()))
    keys = pd.Index(sig.day + '|' + sig.symbol)
    pf = pq.ParquetFile(f'{EL}/paths.parquet')
    parts, n_test_rows, n_other = [], 0, 0
    cols = ['day', 'symbol', 'm', 'o', 'h', 'l', 'c', 'vwap', 'spy_close', 'minutes_since_entry']
    for i in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(i, columns=cols)
        k = pc.binary_join_element_wise(t['day'], t['symbol'], '|').to_pandas()
        n_test_rows += int(test_keys.get_indexer(k).__ge__(0).sum())
        idx = keys.get_indexer(k)
        sel = idx >= 0
        n_other += int((~sel).sum())
        d = pd.DataFrame({
            'arm': idx[sel].astype(np.int32),
            'mse': t['minutes_since_entry'].to_numpy()[sel].astype(np.int16),
            'm': t['m'].to_numpy()[sel].astype(np.int16),
            'o': t['o'].to_numpy()[sel], 'h': t['h'].to_numpy()[sel],
            'l': t['l'].to_numpy()[sel], 'c': t['c'].to_numpy()[sel],
            'vwap': t['vwap'].to_numpy()[sel], 'spy': t['spy_close'].to_numpy()[sel]})
        parts.append(d)
        del t, k
        log(f'row group {i}: kept {int(sel.sum()):,}')
    df = pd.concat(parts, ignore_index=True).sort_values(['arm', 'mse'], kind='stable').reset_index(drop=True)
    log(f'{keep_split}: arms {len(sig):,}, path rows {len(df):,}; TEST signals dropped {n_test_sig:,}, '
        f'TEST B0 rows dropped {n_test_b0:,}, TEST path rows dropped {n_test_rows:,}, other-split rows {n_other - n_test_rows:,}')
    meta = {'test_signals_dropped': n_test_sig, 'test_path_rows_dropped': n_test_rows}
    return sig, df, meta


def walk(df, sig, after_mse, entry_px):
    """Exit under the live rule restarted after bar `after_mse` (per arm Series). Returns exit frame per arm."""
    st_all, tg_all = sig.stop.to_numpy(), sig.target.to_numpy()
    a = df.arm.to_numpy()
    after = after_mse.reindex(np.arange(len(sig))).to_numpy()
    sel = (df.mse.to_numpy() > after[a])
    sub = df.loc[sel, ['arm', 'm', 'o', 'h', 'l', 'c']]
    sa = sub.arm.to_numpy()
    st, tg = st_all[sa], tg_all[sa]
    pre = sub.m.to_numpy() < EOD_M
    hs = pre & (sub.l.to_numpy() <= st)
    ht = pre & (sub.h.to_numpy() >= tg)
    first = sub[hs | ht].groupby('arm', sort=False).head(1)
    fa = first.arm.to_numpy()
    is_stop = first.l.to_numpy() <= st_all[fa]
    px = np.where(is_stop, np.minimum(first.o.to_numpy(), st_all[fa]), tg_all[fa])
    out = pd.DataFrame({'exit': px, 'why': np.where(is_stop, 'stop', 'target')}, index=fa)
    eod = sub[sub.m.to_numpy() >= EOD_M].groupby('arm', sort=False).head(1).set_index('arm')
    last = sub.groupby('arm', sort=False).tail(1).set_index('arm')
    close1555 = eod.c.reindex(last.index).fillna(last.c)
    rest = last.index.difference(out.index)
    eod_px = eod.o.reindex(rest)
    n_missing = int(eod_px.isna().sum())
    if n_missing:
        log(f'WARNING: {n_missing} arms have no 15:55 bar after the restart; exit at the last bar close')
    eod_px = eod_px.fillna(last.c.reindex(rest))
    out = pd.concat([out, pd.DataFrame({'exit': eod_px.to_numpy(), 'why': 'eod'}, index=rest)])
    out['close1555'] = close1555.reindex(out.index)
    nxt_open = sub.groupby('arm', sort=False).head(1).set_index('arm').o
    out['next_open'] = nxt_open.reindex(out.index)
    return out.sort_index()


def net_r(entry, exit_, why, hs, r0, k=1.0):
    """Net R0 units: entry + half_spread + 2 bp; exit − half_spread − 2 bp; stop exits − 13 bp more."""
    ent = entry + k * (hs + LEG * entry)
    ex = exit_ - k * (hs + LEG * exit_) - k * SLIP * exit_ * (why == 'stop')
    return (ex - ent) / r0


def state_at(df, sig, n):
    """Causal state at the close of bar N plus the forward outcome restarted at N, one row per arm."""
    s0 = df[df.mse <= n]
    last = s0.groupby('arm', sort=False).tail(1).set_index('arm')
    spy0 = df[df.mse == 0].set_index('arm').spy
    win = df[(df.mse >= 1) & (df.mse <= n)]
    minl = win.groupby('arm').l.min()
    st = pd.DataFrame({'c': last.c, 'vwap': last.vwap, 'spy': last.spy, 'm_state': last.m, 'mse_state': last.mse})
    st['minl'] = minl.reindex(st.index)
    st['spy0'] = spy0.reindex(st.index)
    g = sig.loc[st.index]
    lvl, r0, stop, tgt = g.entry.to_numpy(), g.R.to_numpy(), g.stop.to_numpy(), g.target.to_numpy()
    st['N'] = n
    st['held'] = (st.minl.to_numpy() > lvl).astype(int)
    st['dist'] = (st.c.to_numpy() - lvl) / r0
    st['maxdd'] = (st.minl.to_numpy() - lvl) / r0
    st['above_vwap'] = (st.c > st.vwap).astype(int)
    st['ret_spy'] = st.spy / st.spy0 - 1
    st['bars_since_open'] = (g.entry_m.to_numpy() + n - OPEN_M)
    ok = (st.minl.notna() & (st.c.to_numpy() > stop) & (st.c.to_numpy() < tgt) & (g.entry_m.to_numpy() + n < EOD_M))
    excl = {'no_bars_1_to_N': int(st.minl.isna().sum()), 'c_le_stop': int((st.c.to_numpy() <= stop).sum()),
            'c_ge_target': int((st.c.to_numpy() >= tgt).sum()), 'too_late': int((g.entry_m.to_numpy() + n >= EOD_M).sum())}
    st = st[ok.to_numpy()]
    after = pd.Series(n, index=st.index)
    ex = walk(df, sig, after, None).reindex(st.index)
    g = sig.loc[st.index]
    r0, hs, c = g.R.to_numpy(), g.half_spread.to_numpy(), st.c.to_numpy()
    bad = ex.exit.isna().to_numpy()
    if bad.any():
        log(f'WARNING: N={n}: {int(bad.sum())} arms have no bar after N; dropped')
        st, ex = st[~bad], ex[~bad]
        g = sig.loc[st.index]
        r0, hs, c = g.R.to_numpy(), g.half_spread.to_numpy(), st.c.to_numpy()
    st['why'] = ex.why
    st['fwd_R'] = (ex.exit.to_numpy() - c) / r0
    st['fwd_close_R'] = (ex.close1555.to_numpy() - c) / r0
    st['net_R0'] = net_r(c, ex.exit.to_numpy(), ex.why.to_numpy(), hs, r0)
    st['net_R0_x15'] = net_r(c, ex.exit.to_numpy(), ex.why.to_numpy(), hs, r0, 1.5)
    st['net_new'] = st.net_R0 * r0 / (c - g.stop.to_numpy())
    no = ex.next_open.to_numpy()
    st['net_R0_nextopen'] = np.where(no > g.stop.to_numpy(), net_r(no, ex.exit.to_numpy(), ex.why.to_numpy(), hs, r0), np.nan)
    st['b0_net'] = g.b0_net.to_numpy()
    st['day'] = g.day.to_numpy()
    st['wk'] = g.wk.to_numpy()
    st['half'] = g.half.to_numpy()
    st['symbol'] = g.symbol.to_numpy()
    return st.reset_index().rename(columns={'index': 'arm'}), excl


def cluster_t_mean(x, day):
    """Mean and day-clustered SE / t of a mean."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 3:
        return np.nan, np.nan, np.nan
    mu = x.mean()
    s = pd.Series(x - mu).groupby(np.asarray(day)).sum().to_numpy()
    g = len(s)
    se = np.sqrt((s ** 2).sum() * g / max(g - 1, 1)) / n
    return mu, se, mu / se if se > 0 else np.nan


def cluster_t_diff(v, w, day):
    """Day-clustered t of mean(v | w=1) − mean(v | w=0) (OLS of v on [1,w], cluster sandwich)."""
    v, w = np.asarray(v, float), np.asarray(w, float)
    X = np.column_stack([np.ones_like(w), w])
    xtx_inv = np.linalg.inv(X.T @ X)
    b = xtx_inv @ X.T @ v
    u = v - X @ b
    sc = pd.DataFrame(X * u[:, None]).groupby(np.asarray(day)).sum().to_numpy()
    g = len(sc)
    V = xtx_inv @ (sc.T @ sc) @ xtx_inv * g / max(g - 1, 1)
    return b[1], b[1] / np.sqrt(V[1, 1])


def auc(v, w):
    """AUC of v for the binary label w (Mann-Whitney, average ranks)."""
    r = pd.Series(v).rank().to_numpy()
    w = np.asarray(w, bool)
    nw, nl = w.sum(), (~w).sum()
    return (r[w].sum() - nw * (nw + 1) / 2) / (nw * nl)


def ex_top(x, q=0.05):
    """Mean after dropping the top q share of trades."""
    x = np.sort(np.asarray(x, float))
    k = int(np.ceil(len(x) * q))
    return x[:len(x) - k].mean() if len(x) - k > 0 else np.nan


VARS = ['held', 'dist', 'maxdd', 'above_vwap', 'ret_spy', 'bars_since_open']


def anatomy(st):
    """Winners (fwd_R > 0) vs losers per state variable; AUC vs fwd and vs B0 sign; day-clustered t."""
    rows = []
    w = (st.fwd_R > 0).to_numpy()
    wb = (st.b0_net > 0).to_numpy()
    for v in VARS:
        x = st[v].to_numpy(float)
        ok = np.isfinite(x)
        d, t = cluster_t_diff(x[ok], w[ok], st.day.to_numpy()[ok])
        row = {'N': int(st.N.iloc[0]), 'var': v, 'n': int(ok.sum()), 'n_win': int(w[ok].sum()),
               'mean_win': x[ok & w].mean(), 'mean_los': x[ok & ~w].mean(), 'diff': d, 'day_t': t,
               'auc_fwd': auc(x[ok], w[ok]), 'auc_b0': auc(x[ok], wb[ok])}
        for q in range(1, 10):
            row[f'win_p{q}0'] = np.quantile(x[ok & w], q / 10)
            row[f'los_p{q}0'] = np.quantile(x[ok & ~w], q / 10)
        rows.append(row)
    for h in (1, 0):
        s = st[st.held == h]
        rows.append({'N': int(st.N.iloc[0]), 'var': f'SPLIT_held={h}', 'n': len(s),
                     'split_fwd_R': s.fwd_R.mean(), 'split_fwd_close_R': s.fwd_close_R.mean(),
                     'split_earned_R': s.dist.mean(), 'split_b0_net': s.b0_net.mean(),
                     'split_fwd_win_share': (s.fwd_R > 0).mean(), 'split_net_R0': s.net_R0.mean()})
    return rows


def cells(st, cuts):
    """All single-threshold rule cells at one N: (name, var, op, thr, mask)."""
    out = [('all', 'all', 'all', np.nan, np.ones(len(st), bool))]
    for v in ('held', 'above_vwap'):
        for val in (1, 0):
            out.append((f'{v}=={val}', v, '==', val, (st[v] == val).to_numpy()))
    for v in ('dist', 'maxdd'):
        for thr in cuts[v]:
            out.append((f'{v}>={thr:.4g}', v, '>=', thr, (st[v] >= thr).to_numpy()))
            out.append((f'{v}<{thr:.4g}', v, '<', thr, (st[v] < thr).to_numpy()))
    return out


def summarise(s, weeks):
    """Book statistics of one cell's trades."""
    mu, se, t = cluster_t_mean(s.net_R0, s.day)
    return {'n': len(s), 'fills_wk': len(s) / weeks, 'gross_R0': s.fwd_R.mean(), 'net_R0': mu, 'day_t': t, 'se': se,
            'net_new': s.net_new.mean(), 'ex_top5': ex_top(s.net_R0), 'capped3': np.minimum(s.net_R0, 3).mean(),
            'fwd_close_R': s.fwd_close_R.mean(), 'b0_same': s.b0_net.mean(), 'stop_share': (s.why == 'stop').mean()}


def stage_a():
    """TRAIN-only anatomy, grid, pick; freezes the rule before VAL is touched."""
    sig, df, meta = load('TRAIN')
    # B0 parity: replay from bar 0 at `entry`
    ex = walk(df, sig, pd.Series(-1, index=sig.index), None)
    ok = np.isclose(ex.exit.to_numpy(), sig.b0_exit.reindex(ex.index).to_numpy(), atol=1e-6)
    wok = ex.why.to_numpy() == sig.b0_why.reindex(ex.index).to_numpy()
    log(f'B0 parity on TRAIN: exit price match {ok.mean():.4f}, why match {wok.mean():.4f}, n {len(ok)}')
    meta['b0_parity_price'] = float(ok.mean())
    meta['b0_parity_why'] = float(wok.mean())
    weeks = sig.wk.nunique()
    wk_h = sig.groupby('half').wk.nunique().to_dict()
    an_rows, grid_rows, excl_all = [], [], {}
    for n in NS:
        st, excl = state_at(df, sig, n)
        excl_all[n] = excl
        log(f'N={n}: eligible {len(st):,} excluded {excl}')
        an_rows += anatomy(st)
        h1 = st[st.half == 'H1']
        cuts = {v: np.unique(np.round(np.quantile(h1[v], np.arange(1, 10) / 10), 4)) for v in ('dist', 'maxdd')}
        for name, var, op, thr, mask in cells(st, cuts):
            s = st[mask]
            if len(s) < 10:
                continue
            row = {'N': n, 'cell': name, 'var': var, 'op': op, 'thr': thr}
            row.update(summarise(s, weeks))
            for h in ('H1', 'H2'):
                sh = s[s.half == h]
                row[f'net_{h}'] = sh.net_R0.mean()
                row[f'fills_wk_{h}'] = len(sh) / wk_h[h]
            row['halves_same_sign'] = bool(np.sign(row['net_H1']) == np.sign(row['net_H2']))
            grid_rows.append(row)
    an = pd.DataFrame(an_rows)
    an.to_csv(f'{OUT}/1751_anatomy.csv', index=False)
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(f'{OUT}/1751_grid.csv', index=False)
    elig = grid[grid.fills_wk >= 3].sort_values('net_H2', ascending=False)
    pick = elig.iloc[0].to_dict()
    rule = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in pick.items()}
    rule.update({'frozen_utc': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()), 'n_cells': len(grid),
                 'meta': meta, 'excluded': excl_all, 'train_weeks': weeks})
    with open(f'{OUT}/1751_rule.json', 'w') as f:
        json.dump(rule, f, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    log('PICKED', json.dumps({k: rule[k] for k in ('N', 'cell', 'n', 'fills_wk', 'net_R0', 'net_H1', 'net_H2', 'day_t', 'ex_top5', 'b0_same')}, default=str))
    log(f'cells: {len(grid)}; top 10 by H2 (fills/wk >= 3):')
    print(elig.head(10)[['N', 'cell', 'n', 'fills_wk', 'gross_R0', 'net_R0', 'net_H1', 'net_H2', 'day_t', 'ex_top5', 'b0_same']].to_string(), flush=True)


def rule_mask(st, rule):
    """Apply a frozen rule (var, op, thr) and its placebo complement."""
    v, op, thr = rule['var'], rule['op'], rule['thr']
    if op == 'all':
        m = np.ones(len(st), bool)
    elif op == '==':
        m = (st[v] == thr).to_numpy()
    elif op == '>=':
        m = (st[v] >= thr).to_numpy()
    else:
        m = (st[v] < thr).to_numpy()
    return m


def week_stats(s, pop, weeks_all, rng, reps=2000):
    """Weekly P10 of summed net R0, green-week share, count-matched null from the same-N population."""
    wsum = s.groupby('wk').net_R0.sum().reindex(weeks_all).fillna(0.0)
    cnt = s.groupby('wk').size()
    traded = cnt.index
    green = (wsum.loc[traded] > 0).mean()
    null = np.zeros(reps)
    for wk, k in cnt.items():
        x = pop.loc[pop.wk == wk, 'net_R0'].to_numpy()
        draws = np.array([rng.choice(x, size=k, replace=False).sum() for _ in range(reps)])
        null += (draws > 0)
    null /= len(cnt)
    return float(np.quantile(wsum, 0.10)), float(green), float(null.mean()), float((null >= green).mean())


def stage_b():
    """VAL, read once: the frozen rule, placebo, B0 comparators, cost x1.5, next-open sensitivity."""
    with open(f'{OUT}/1751_rule.json') as f:
        rule = json.load(f)
    sig, df, meta = load('VAL')
    st, excl = state_at(df, sig, int(rule['N']))
    weeks_all = sorted(sig.wk.unique())
    weeks = len(weeks_all)
    m = rule_mask(st, rule)
    s, pl = st[m], st[~m]
    rng = np.random.default_rng(1751)
    rows = []
    for name, x in (('rule', s), ('placebo_complement', pl), ('all_N', st)):
        r = {'set': name}
        r.update(summarise(x, weeks))
        r['net_x15'] = x.net_R0_x15.mean()
        r['net_nextopen'] = np.nanmean(x.net_R0_nextopen)
        r['p10_wk'], r['green'], r['green_null'], r['green_null_p'] = week_stats(x, st, weeks_all, rng)
        d = x.net_R0.to_numpy() - x.b0_net.to_numpy()
        r['lift_b0_same'], r['lift_se'], r['lift_t'] = cluster_t_mean(d, x.day)
        r['mde80'] = 2.8 * r['se']
        for y, gy in x.groupby(x.day.str[:7]):
            r[f'm_{y}'] = gy.net_R0.mean()
        r['year_2026'] = x[x.day.str[:4] == '2026'].net_R0.mean()
        rows.append(r)
    out = pd.DataFrame(rows)
    out['b0_all_val'] = sig.b0_net.mean()
    out['test_dropped_signals'] = meta['test_signals_dropped']
    out['test_dropped_rows'] = meta['test_path_rows_dropped']
    out.to_csv(f'{OUT}/1751_val.csv', index=False)
    log(f'VAL excluded at N: {excl}; weeks {weeks}')
    print(out.T.to_string(), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--stage', choices=['A', 'B'], required=True)
    a = ap.parse_args()
    stage_a() if a.stage == 'A' else stage_b()
    sys.exit(0)
