#!/usr/bin/env python3
"""Cell 1,753 — harvest the post-break mean reversion, LONG only (reads D1 dip / D2 range-low retest / D4 geometry).

Spec: research/hod_entry/PREREG_1753.md (frozen 2026-11-13 19:25 UTC). Two stages, run separately:
  --stage A : TRAIN only (VAL and TEST dropped at load). Grid (D1 a x b, D2 one cell, D4 N), placebo, pick one
              cell per read by TRAIN-H2 net among cells with >= 3 fills/wk; winners-vs-losers anatomy of the picks;
              writes 1753_grid.csv, 1753_anatomy.csv, 1753_rule.json.
  --stage B : VAL only, read once; scores the three frozen picks + placebos; writes 1753_val.csv.

Conventions (data layout and cost of cell 1,751 / PREREG_1740):
  level = signals.entry (open of the B0 entry bar, minutes_since_entry 0); R0 = signals.R (= level - signals.stop).
  D1: first bar with mse in [0, 30] whose low <= level - a R0 -> buy at that bar's CLOSE + half_spread + 2 bp
      (valid only if stop < close < target); target level - b R0 (limit sell, fills on a strict through-trade
      h > target, 2 bp); stop level - (a + 0.5) R0 (fills at min(open, stop), half_spread + 2 bp + 13 bp slip;
      a bar touching both = stop); time-out = close of the 30th bar after entry (or the 15:55 open if earlier),
      half_spread + 2 bp. Primary unit R = entry close - stop.
  D2: the pre-break 5-bar low is NOT in paths.parquet (paths start at the entry bar); signals.stop is used as the
      range-low proxy. signals.stop = level - 1 R0, so D2 = the D1 machinery with a = 1, b = 0 and the trigger
      window extended to the whole session (any bar up to 15:25) — the distinct part of D2 vs D1(1, 0).
  D4: held = min low over bars 1..N > level; buy at the close of bar N; trailing stop max(level, high since bar 0
      through the previous bar - 1 R0), fill min(open, trail) + stop costs; else the 15:55 open. Unit R0
      (the PREREG gives no D4 unit; B0 net_R is in R0, so the comparison is like for like).
  Placebo (per arm, one random anchor minute t0 >= 31 after the break, m(t0) <= 14:54): D1/D2 level = local high
      max(h) over bars t0-10..t0, trigger scan from t0+1; D4 level = open of bar t0, bars counted from t0.
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
LEG = 0.0002      # 2 bp per leg
SLIP = 0.0013     # stop slip, PREREG_1740
EOD_M = 955       # 15:55 bar
OPEN_M = 570      # 09:30
D2_LAST_M = 925   # 15:25: last D2 trigger bar so the 30-min time-out fits
TIMEOUT = 30
D1_CELLS = [(0.5, 0.0), (0.5, 0.25), (1.0, 0.0), (1.0, 0.25), (1.0, 0.5)]
D4_NS = [10, 20]


def log(*a):
    """Timestamped progress line, flushed."""
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def load(keep_split):
    """Load signals, B0 and paths for ONE split (as 1751_score.load); TEST and the other split dropped, counted."""
    sig = pd.read_parquet(f'{EL}/signals.parquet')
    assert not sig.duplicated(['day', 'symbol']).any(), 'day,symbol not unique'
    test_keys = pd.Index((sig.day + '|' + sig.symbol)[sig.split == 'TEST'])
    n_test_sig = int((sig.split == 'TEST').sum())
    sig = sig[sig.split == keep_split]
    nohs = int(sig.half_spread.isna().sum())
    if nohs:
        log(f'WARNING: {keep_split}: {nohs} arms have no measured half_spread; dropped (cost unknowable)')
    sig = sig[sig.half_spread.notna()].reset_index(drop=True)
    b0 = pd.read_csv(f'{EL}/b0_trades.csv')
    n_test_b0 = int((b0.split == 'TEST').sum())
    b0 = b0[b0.split == keep_split]
    sig = sig.merge(b0[['day', 'symbol', 'entry_m', 'net_R']].rename(columns={'net_R': 'b0_net'}),
                    on=['day', 'symbol', 'entry_m'], how='left', validate='1:1')
    if sig.b0_net.isna().any():
        log('WARNING: arms without a B0 net_R (excluded from B0 means only):', int(sig.b0_net.isna().sum()))
    keys = pd.Index(sig.day + '|' + sig.symbol)
    pf = pq.ParquetFile(f'{EL}/paths.parquet')
    parts, n_test_rows, n_other = [], 0, 0
    cols = ['day', 'symbol', 'm', 'o', 'h', 'l', 'c', 'vwap', 'spy_close', 'minutes_since_entry']
    for i in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(i, columns=cols)
        k = pc.binary_join_element_wise(t['day'], t['symbol'], '|').to_pandas()
        n_test_rows += int((test_keys.get_indexer(k) >= 0).sum())
        idx = keys.get_indexer(k)
        sel = idx >= 0
        n_other += int((~sel).sum())
        parts.append(pd.DataFrame({
            'arm': idx[sel].astype(np.int32),
            'mse': t['minutes_since_entry'].to_numpy()[sel].astype(np.int16),
            'm': t['m'].to_numpy()[sel].astype(np.int16),
            'o': t['o'].to_numpy()[sel], 'h': t['h'].to_numpy()[sel],
            'l': t['l'].to_numpy()[sel], 'c': t['c'].to_numpy()[sel],
            'vwap': t['vwap'].to_numpy()[sel], 'spy': t['spy_close'].to_numpy()[sel]}))
        del t, k
    df = pd.concat(parts, ignore_index=True).sort_values(['arm', 'mse'], kind='stable').reset_index(drop=True)
    log(f'{keep_split}: arms {len(sig):,}, path rows {len(df):,}; TEST signals dropped {n_test_sig:,}, '
        f'TEST B0 rows dropped {n_test_b0:,}, TEST path rows dropped {n_test_rows:,}')
    meta = {'test_signals_dropped': n_test_sig, 'test_b0_dropped': n_test_b0, 'test_path_rows_dropped': n_test_rows}
    return sig, df, meta


def first_per_arm(mask, arm):
    """Row index of the first True row per arm (rows sorted by arm, mse). Returns (arms, row_idx)."""
    idx = np.flatnonzero(mask)
    a, pos = np.unique(arm[idx], return_index=True)
    return a, idx[pos]


def last_per_arm(mask, arm):
    """Row index of the last True row per arm."""
    idx = np.flatnonzero(mask)[::-1]
    a, pos = np.unique(arm[idx], return_index=True)
    return a, idx[pos]


def net_r(entry, exit_, why, hs, unit, k=1.0):
    """Net in `unit`: entry + hs + 2 bp; target exit 2 bp; stop/time-out exit hs + 2 bp; stop +13 bp slip."""
    ent = entry + k * (hs + LEG * entry)
    is_t = why == 'target'
    ex = exit_ - k * (np.where(is_t, 0.0, hs) + LEG * exit_) - k * SLIP * exit_ * (why == 'stop')
    return (ex - ent) / unit


def exit_walk(R, n_arms, e_mse, stop, tgt, horizon):
    """Exit after entry bar e_mse (per-arm arrays, NaN = no trade): fixed stop / target / horizon or 15:55 open."""
    arm, mse, m = R['arm'], R['mse'], R['m']
    em = e_mse[arm]
    inwin = (mse > em) & (mse <= em + horizon)
    pre = inwin & (m < EOD_M)
    hs_ = pre & (R['l'] <= stop[arm])
    ht_ = pre & (R['h'] > tgt[arm])
    ex_px, why = np.full(n_arms, np.nan), np.full(n_arms, '', dtype=object)
    a, i = first_per_arm(hs_ | ht_, arm)
    st_hit = R['l'][i] <= stop[a]
    ex_px[a] = np.where(st_hit, np.minimum(R['o'][i], stop[a]), tgt[a])
    why[a] = np.where(st_hit, 'stop', 'target')
    a2, i2 = first_per_arm(inwin & (m >= EOD_M), arm)
    k = np.isnan(ex_px[a2])
    ex_px[a2[k]] = R['o'][i2[k]]
    why[a2[k]] = 'eod'
    a3, i3 = last_per_arm(inwin, arm)
    k = np.isnan(ex_px[a3])
    ex_px[a3[k]] = R['c'][i3[k]]
    why[a3[k]] = 'timeout'
    return ex_px, why


def dip_trades(R, sig, a, b, lvl, start, last_trig_m, window):
    """D1/D2 long-the-dip trades for every arm with per-arm level and scan start (real or placebo)."""
    n = len(sig)
    r0, hs = sig.R.to_numpy(), sig.half_spread.to_numpy()
    arm, mse, m = R['arm'], R['mse'], R['m']
    trig = lvl - a * r0
    stop, tgt = lvl - (a + 0.5) * r0, lvl - b * r0
    rel = mse - start[arm]
    scan = (rel >= 0) & (m < EOD_M) & (m <= last_trig_m)
    if window is not None:
        scan &= rel <= window
    ta, ti = first_per_arm(scan & (R['l'] <= trig[arm]), arm)
    e_px, e_mse = np.full(n, np.nan), np.full(n, np.nan)
    e_px[ta], e_mse[ta] = R['c'][ti], mse[ti]
    excl = {'triggered': int(len(ta)), 'close_le_stop': int((e_px <= stop).sum()),
            'close_ge_target': int((e_px >= tgt).sum())}
    bad = ~((e_px > stop) & (e_px < tgt))
    e_px[bad], e_mse[bad] = np.nan, np.nan
    ex_px, why = exit_walk(R, n, e_mse, stop, tgt, TIMEOUT)
    ok = ~np.isnan(e_px) & ~np.isnan(ex_px)
    excl['no_bar_after_entry'] = int((~np.isnan(e_px) & np.isnan(ex_px)).sum())
    idx = np.flatnonzero(ok)
    unit = e_px[idx] - stop[idx]
    t = pd.DataFrame({'arm': idx, 'entry': e_px[idx], 'exit': ex_px[idx], 'why': why[idx].astype(str),
                      'e_mse': e_mse[idx], 'lvl': lvl[idx], 'unit': unit})
    _fill(t, sig, hs, r0)
    t['net'] = t.net_new
    t['gross'] = (t.exit - t.entry) / t.unit
    t['net_x15'] = net_r(t.entry.to_numpy(), t.exit.to_numpy(), t.why.to_numpy(), hs[idx], unit, 1.5)
    return t, excl


def trail_trades(R, sig, N, lvl, start):
    """D4 geometry trades: held through bar N, buy the close of N, trail max(level, prior high - 1 R0) to 15:55."""
    n = len(sig)
    r0, hs = sig.R.to_numpy(), sig.half_spread.to_numpy()
    arm, mse, m = R['arm'], R['mse'], R['m']
    rel = mse - start[arm]
    h_eff = np.where(rel >= 0, R['h'], -np.inf)
    hmax = pd.Series(h_eff).groupby(arm, sort=False).cummax().to_numpy()
    hprev = np.r_[-np.inf, hmax[:-1]]
    hprev[np.r_[True, arm[1:] != arm[:-1]]] = -np.inf
    trail = np.maximum(lvl[arm], hprev - r0[arm])
    w = (rel >= 1) & (rel <= N)
    minl = pd.Series(np.where(w, R['l'], np.inf)).groupby(arm, sort=False).min().reindex(np.arange(n)).to_numpy()
    sa, si = last_per_arm((rel >= 0) & (rel <= N), arm)
    c_n, mse_n, m_n, hmax_n = (np.full(n, np.nan) for _ in range(4))
    c_n[sa], mse_n[sa], m_n[sa], hmax_n[sa] = R['c'][si], mse[si], m[si], hmax[si]
    stop0 = np.maximum(lvl, hmax_n - r0)
    held = np.isfinite(minl) & (minl > lvl)   # an arm with no bar in 1..N is not 'held'
    ok = held & (c_n > stop0) & (m_n < EOD_M)
    excl = {'held': int(held.sum()), 'held_but_close_le_trail': int((held & ~(c_n > stop0)).sum()),
            'too_late': int((held & (m_n >= EOD_M)).sum())}
    e_mse = np.where(ok, mse_n, np.nan)
    em = e_mse[arm]
    after = mse > em
    pre = after & (m < EOD_M)
    ex_px, why = np.full(n, np.nan), np.full(n, '', dtype=object)
    a1, i1 = first_per_arm(pre & (R['l'] <= trail), arm)
    ex_px[a1], why[a1] = np.minimum(R['o'][i1], trail[i1]), 'stop'
    a2, i2 = first_per_arm(after & (m >= EOD_M), arm)
    k = np.isnan(ex_px[a2])
    ex_px[a2[k]], why[a2[k]] = R['o'][i2[k]], 'eod'
    a3, i3 = last_per_arm(after, arm)
    k = np.isnan(ex_px[a3])
    ex_px[a3[k]], why[a3[k]] = R['c'][i3[k]], 'last_close'
    if k.any():
        log(f'WARNING: D4 N={N}: {int(k.sum())} arms have no 15:55 bar; exit at the last bar close')
    idx = np.flatnonzero(ok & ~np.isnan(ex_px))
    excl['no_bar_after_entry'] = int((ok & np.isnan(ex_px)).sum())
    t = pd.DataFrame({'arm': idx, 'entry': c_n[idx], 'exit': ex_px[idx], 'why': why[idx].astype(str),
                      'e_mse': mse_n[idx], 'lvl': lvl[idx], 'unit': c_n[idx] - stop0[idx],
                      'runup': (hmax_n[idx] - lvl[idx]) / r0[idx]})
    _fill(t, sig, hs, r0)
    t['net'] = t.net_R0
    t['gross'] = (t.exit - t.entry) / r0[idx]
    t['net_x15'] = net_r(t.entry.to_numpy(), t.exit.to_numpy(), t.why.to_numpy(), hs[idx], r0[idx], 1.5)
    return t, excl


def _fill(t, sig, hs, r0):
    """Attach costs in R0 and in the trade's own unit, plus arm metadata."""
    idx = t.arm.to_numpy()
    t['net_R0'] = net_r(t.entry.to_numpy(), t.exit.to_numpy(), t.why.to_numpy(), hs[idx], r0[idx])
    t['net_new'] = t.net_R0 * r0[idx] / t.unit
    g = sig.iloc[idx]
    for col in ('day', 'symbol', 'wk', 'half', 'b0_net', 'entry_m'):
        t[col] = g[col].to_numpy()
    t['r0_bps'] = r0[idx] / g.entry.to_numpy() * 1e4
    t['hs_over_r0'] = hs[idx] / r0[idx]


def features(t, R, sig):
    """Causal entry-bar features for the winners-vs-losers anatomy."""
    arm, mse = R['arm'], R['mse']
    key = pd.MultiIndex.from_arrays([arm, mse])
    row = key.get_indexer(pd.MultiIndex.from_arrays([t.arm.to_numpy(), t.e_mse.to_numpy().astype(np.int16)]))
    spy0 = pd.Series(R['spy'][mse == 0], index=arm[mse == 0]).reindex(t.arm).to_numpy()
    t['bars_since_open'] = R['m'][row] - OPEN_M
    t['above_vwap'] = (R['c'][row] > R['vwap'][row]).astype(int)
    t['ret_spy'] = R['spy'][row] / spy0 - 1
    t['depth'] = (t.entry - t.lvl) / sig.R.to_numpy()[t.arm.to_numpy()]
    return t


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


def ex_top(x, q=0.05):
    """Mean after dropping the top q share of trades."""
    x = np.sort(np.asarray(x, float))
    k = int(np.ceil(len(x) * q))
    return x[:len(x) - k].mean() if len(x) - k > 0 else np.nan


def auc(v, w):
    """AUC of v for the binary label w (Mann-Whitney, average ranks)."""
    r = pd.Series(v).rank().to_numpy()
    w = np.asarray(w, bool)
    nw, nl = w.sum(), (~w).sum()
    return (r[w].sum() - nw * (nw + 1) / 2) / (nw * nl) if nw and nl else np.nan


def week_stats(s, pool, weeks_all, rng, reps=500):
    """Weekly P10 of summed net, green-week share, count-matched null drawn from the same week's placebo trades."""
    wsum = s.groupby('wk').net.sum().reindex(weeks_all).fillna(0.0)
    cnt = s.groupby('wk').size()
    green = float((wsum.loc[cnt.index] > 0).mean())
    allx = pool.net.to_numpy()
    null = np.zeros(reps)
    for wk, k in cnt.items():
        x = pool.loc[pool.wk == wk, 'net'].to_numpy()
        x = x if len(x) else allx
        if len(x) >= k:   # k draws without replacement per rep
            draws = x[np.argsort(rng.random((reps, len(x))), axis=1)[:, :k]]
        else:
            draws = rng.choice(x, size=(reps, k), replace=True)
        null += draws.sum(1) > 0
    null /= len(cnt)
    return float(np.quantile(wsum, 0.10)), green, float(null.mean()), float((null >= green).mean())


def summarise(s, weeks, wk_h=None):
    """Book statistics of one cell's trades (primary unit column `net`)."""
    mu, se, t = cluster_t_mean(s.net, s.day)
    r = {'n': len(s), 'fills_wk': len(s) / weeks, 'gross': s.gross.mean(), 'net': mu, 'day_t': t, 'se': se,
         'iid_t': mu / (s.net.std() / np.sqrt(len(s))) if len(s) > 2 else np.nan,
         'net_R0': s.net_R0.mean(), 'net_x15': s.net_x15.mean(), 'ex_top5': ex_top(s.net), 'ex_top1': ex_top(s.net, 0.01),
         'capped3': np.minimum(s.net, 3).mean(), 'mde80': 2.8 * se, 'b0_same_R0': s.b0_net.mean(),
         'stop_share': (s.why == 'stop').mean(), 'target_share': (s.why == 'target').mean()}
    if wk_h is not None:
        for h in ('H1', 'H2'):
            sh = s[s.half == h]
            r[f'net_{h}'] = sh.net.mean()
            r[f'fills_wk_{h}'] = len(sh) / wk_h[h]
        r['halves_same_sign'] = bool(np.sign(r['net_H1']) == np.sign(r['net_H2']))
    for y, gy in s.groupby(s.day.str[:4]):
        r[f'y{y}'] = gy.net.mean()
    return r


def placebo_anchor(R, sig, seed):
    """One random anchor row per arm: mse >= 31 and m <= 894 (14:54). Returns (t0 mse, local high, open at t0)."""
    n = len(sig)
    arm, mse, m = R['arm'], R['mse'], R['m']
    elig = np.flatnonzero((mse >= 31) & (m <= EOD_M - 61))
    key = np.random.default_rng(seed).random(len(elig))
    o = np.lexsort((key, arm[elig]))
    a, pos = np.unique(arm[elig][o], return_index=True)
    rows = elig[o][pos]
    t0 = np.full(n, np.nan)
    t0[a] = mse[rows]
    o_t0 = np.full(n, np.nan)
    o_t0[a] = R['o'][rows]
    rel = mse - t0[arm]
    lh = pd.Series(np.where((rel >= -10) & (rel <= 0), R['h'], -np.inf)).groupby(arm).max()
    loc_high = lh.reindex(np.arange(n)).to_numpy()
    loc_high[np.isnan(t0)] = np.nan
    log(f'placebo anchors: {len(a):,} of {n:,} arms')
    return t0, loc_high, o_t0


def run_read(R, sig, read, par, placebo):
    """One cell, real (start at the break, level = signals.entry) or placebo (random anchor)."""
    n = len(sig)
    if placebo is None:
        lvl, start = sig.entry.to_numpy(), np.zeros(n)
    else:
        t0, loc_high, o_t0 = placebo
    if read == 'D4':
        if placebo is not None:
            lvl, start = o_t0, t0
        return trail_trades(R, sig, par['N'], lvl, start)
    if placebo is not None:
        lvl, start = loc_high, t0 + 1
    if read == 'D1':
        return dip_trades(R, sig, par['a'], par['b'], lvl, start, EOD_M - 1, TIMEOUT)
    return dip_trades(R, sig, 1.0, 0.0, lvl, start, D2_LAST_M, None)


def cells():
    """The frozen grid: 5 D1 cells, 1 D2 cell, 2 D4 cells."""
    out = [('D1', {'a': a, 'b': b}) for a, b in D1_CELLS] + [('D2', {})]
    return out + [('D4', {'N': n}) for n in D4_NS]


def rows_of(df):
    """Numpy views of the path columns."""
    return {k: df[k].to_numpy() for k in ('arm', 'mse', 'm', 'o', 'h', 'l', 'c', 'vwap', 'spy')}


def stage_a():
    """TRAIN-only grid, placebo, pick per read, anatomy of the picks; freezes before VAL is touched."""
    sig, df, meta = load('TRAIN')
    R = rows_of(df)
    weeks = sig.wk.nunique()
    wk_h = sig.groupby('half').wk.nunique().to_dict()
    pl = placebo_anchor(R, sig, 1753)
    grid, trades = [], {}
    for read, par in cells():
        t, excl = run_read(R, sig, read, par, None)
        p, _ = run_read(R, sig, read, par, pl)
        row = {'read': read, 'par': json.dumps(par)}
        row.update(summarise(t, weeks, wk_h))
        row['placebo_net'], row['placebo_n'] = p.net.mean(), len(p)
        row['excl'] = json.dumps(excl)
        grid.append(row)
        trades[(read, json.dumps(par))] = t
        log(f"{read} {par}: n {len(t)} f/wk {row['fills_wk']:.2f} net {row['net']:+.3f} t {row['day_t']:.2f} "
            f"H1 {row['net_H1']:+.3f} H2 {row['net_H2']:+.3f} placebo {row['placebo_net']:+.3f} excl {excl}")
    g = pd.DataFrame(grid)
    g.to_csv(f'{OUT}/1753_grid.csv', index=False)
    picks, an = {}, []
    for read in ('D1', 'D2', 'D4'):
        e = g[(g.read == read) & (g.fills_wk >= 3)].sort_values('net_H2', ascending=False)
        if e.empty:
            log(f'WARNING: {read}: no cell with >= 3 fills/wk on TRAIN; taking the best H2 cell, flagged')
            e = g[g.read == read].sort_values('net_H2', ascending=False)
        pk = e.iloc[0]
        picks[read] = {'par': json.loads(pk.par), 'train': {k: (v.item() if hasattr(v, 'item') else v)
                                                            for k, v in pk.items() if k not in ('par', 'excl')}}
        t = features(trades[(read, pk.par)], R, sig)
        w = (t.net > 0).to_numpy()
        for v in ('e_mse', 'bars_since_open', 'depth', 'above_vwap', 'ret_spy', 'r0_bps', 'hs_over_r0') + (
                ('runup',) if read == 'D4' else ()):
            x = t[v].to_numpy(float)
            an.append({'read': read, 'var': v, 'win_mean': np.nanmean(x[w]), 'lose_mean': np.nanmean(x[~w]),
                       'auc': auc(x, w), 'n_win': int(w.sum()), 'n_lose': int((~w).sum())})
    pd.DataFrame(an).to_csv(f'{OUT}/1753_anatomy.csv', index=False)
    rule = {'frozen_utc': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()), 'picks': picks, 'meta': meta,
            'train_weeks': weeks, 'n_cells': len(g)}
    with open(f'{OUT}/1753_rule.json', 'w') as f:
        json.dump(rule, f, indent=1, default=str)
    print(g[['read', 'par', 'n', 'fills_wk', 'gross', 'net', 'day_t', 'net_H1', 'net_H2', 'ex_top5', 'b0_same_R0',
             'placebo_net']].to_string(), flush=True)
    print(pd.DataFrame(an).to_string(), flush=True)
    log('PICKS', json.dumps({k: v['par'] for k, v in picks.items()}))


def stage_b():
    """VAL, read once: the three frozen picks, placebo, B0 on the same arms, cost x1.5, weekly stats."""
    with open(f'{OUT}/1753_rule.json') as f:
        rule = json.load(f)
    sig, df, meta = load('VAL')
    R = rows_of(df)
    weeks_all = sorted(sig.wk.unique())
    weeks = len(weeks_all)
    pl = placebo_anchor(R, sig, 1754)
    rng = np.random.default_rng(1753)
    out = []
    for read, pk in rule['picks'].items():
        par = pk['par']
        t, excl = run_read(R, sig, read, par, None)
        p, _ = run_read(R, sig, read, par, pl)
        for name, s in (('rule', t), ('placebo', p)):
            r = {'read': read, 'par': json.dumps(par), 'set': name}
            r.update(summarise(s, weeks))
            r['p10_wk'], r['green'], r['green_null'], r['green_null_p'] = week_stats(s, p, weeks_all, rng)
            if name == 'rule':
                d = (s.net_R0 - s.b0_net).dropna()
                r['lift_b0_R0'], _, r['lift_t'] = cluster_t_mean(d, s.day[d.index])
                r['excl'] = json.dumps(excl)
            out.append(r)
        log(f"VAL {read} {par}: n {len(t)} net {out[-2]['net']:+.3f} t {out[-2]['day_t']:.2f} placebo {out[-1]['net']:+.3f}")
    o = pd.DataFrame(out)
    o['b0_all_val_R0'] = sig.b0_net.mean()
    o['val_weeks'] = weeks
    for k, v in meta.items():
        o[k] = v
    o.to_csv(f'{OUT}/1753_val.csv', index=False)
    print(o.T.to_string(), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--stage', choices=['A', 'B'], required=True)
    stage_a() if ap.parse_args().stage == 'A' else stage_b()
    sys.exit(0)
