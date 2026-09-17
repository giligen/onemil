#!/usr/bin/env python3
"""Stage G step 0 — the SHORT universe, the SSR flags and the M18 attention list.

Writes ONLY research/fuckup_audit/G/{members_g.csv, attention_list.csv, g0_universe.md}.
Every other path is read-only.

WHAT IT BUILDS (all of it declared in G/PREREG.md before this script ran):

  members_g.csv   one row per (bar_date, symbol) the Stage-G builder will walk:
                  (a) E/members.csv rows with u1|u2  AND  dvol20_med >= $10M  AND  open >= $10
                      AND asset_class == 'stock'                                      -> in_u12 = 1
                  (b) the M18 attention/control names on their trade day, same borrowable filter,
                      fields recomputed from research/lit_review_2026/daily_panel.parquet (the same
                      Databento panel E/members.csv came from, NULL tickers already dropped)
                  plus `ssr` (Reg SHO 201 fired on t-1) and `prev_ret_cc` for every row.

  attention_list.csv  the M18 selection reproduced EXACTLY from test_open_fade.py: on day t-1 keep
                  close >= 5, dvol20 >= $2M, |ret_cc| <= 0.5, ret_cc/vol_ratio present, symbol
                  ^[A-Z]{1,5}(\\.[A-Z]{1,2})?$, drop the Z-prefixed placeholder tickers; score
                  attn = |ret_cc| * vol_ratio; rank descending per day (method='first'); top 20 =
                  attention, ranks 101-120 = control; trade on the next day of the filtered frame.

  SSR: day t is an SSR day for s iff low_{t-1} <= 0.90 * close_{t-2} (the Rule 201 trigger fired on
  t-1, so the restriction covers all of t). prev_ret_cc = close_{t-1}/close_{t-2} - 1 is reported
  beside it. Both from the panel, causal, with the same per-symbol shift convention E/universes.py
  uses (block-aware, NaN at a symbol's first rows).

RUN: ulimit -v 2600000; nice -n 10 python3 research/fuckup_audit/G/g0_universe.py
"""
import os
import re
import sys

os.environ.setdefault('ARROW_DEFAULT_MEMORY_POOL', 'system')

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)

G = f'{ROOT}/research/fuckup_audit/G'
MEMBERS = f'{ROOT}/research/fuckup_audit/E/members.csv'
PANEL = f'{ROOT}/research/lit_review_2026/daily_panel.parquet'
EIDX = f'{ROOT}/research/fuckup_audit/E/bars_causal_index.csv'

DVOL_MIN = 10e6        # 20-day median dollar volume
PRICE_MIN = 10.0       # 09:30 open
SYM_RE = re.compile(r'^[A-Z]{1,5}(\.[A-Z]{1,2})?$')
L = []


def log(m):
    print(m, flush=True)
    L.append(m)


def split_of(day):
    return np.where(day < '2026-01-01', 'TRAIN', np.where(day < '2026-06-01', 'VAL', 'TEST'))


# ------------------------------------------------------------------ the panel
FCOLS = ('open', 'high', 'low', 'close', 'volume', 'adv20', 'dvol20', 'ret_cc', 'vol_ratio')


def load_panel():
    """Stream the panel by row group into int32 codes + float32 columns (this node cannot hold a
    5M-row pandas block); sort by (symbol, date), drop duplicate keys, return a frame whose `symbol`
    and `bar_date` are Categoricals."""
    import gc
    import pyarrow.parquet as pq
    parts = {k: [] for k in FCOLS}
    sym_parts, day_parts = [], []
    sym_map, day_map = {}, {}
    pf = pq.ParquetFile(PANEL)
    for b in pf.iter_batches(batch_size=200_000, columns=['symbol', 'bar_date'] + list(FCOLS)):
        sym_parts.append(np.fromiter(
            (sym_map.setdefault(str(s or ''), len(sym_map)) for s in b.column('symbol').to_pylist()),
            dtype='int32', count=b.num_rows))
        day_parts.append(np.fromiter(
            (day_map.setdefault(str(s or '')[:10], len(day_map)) for s in b.column('bar_date').to_pylist()),
            dtype='int32', count=b.num_rows))
        for c in FCOLS:
            parts[c].append(b.column(c).to_numpy(zero_copy_only=False).astype('float32'))
        del b
    del pf
    gc.collect()
    arr = {k: np.concatenate(v) for k, v in parts.items()}
    del parts
    sym_codes = np.concatenate(sym_parts)
    day_codes = np.concatenate(day_parts)
    del sym_parts, day_parts
    gc.collect()
    sym_uniq = pd.Index(list(sym_map))
    day_uniq_seen = pd.Index(list(day_map))
    order_d = np.argsort(np.argsort(day_uniq_seen.values))       # first-seen -> chronological code
    day_codes = order_d[day_codes].astype('int32')
    day_uniq = pd.Index(sorted(day_map))
    sym_order = np.argsort(np.argsort(sym_uniq.values))
    sym_codes = sym_order[sym_codes].astype('int32')
    sym_uniq = pd.Index(sorted(sym_map))
    n0 = len(sym_codes)

    o = np.lexsort((day_codes, sym_codes))
    sym_codes, day_codes = sym_codes[o], day_codes[o]
    for k in arr:
        arr[k] = arr[k][o]
    del o
    gc.collect()
    keep = np.empty(n0, dtype=bool)
    keep[0] = True
    keep[1:] = (sym_codes[1:] != sym_codes[:-1]) | (day_codes[1:] != day_codes[:-1])
    if not keep.all():
        sym_codes, day_codes = sym_codes[keep], day_codes[keep]
        for k in arr:
            arr[k] = arr[k][keep]
    n1 = len(sym_codes)
    del keep
    gc.collect()
    d = pd.DataFrame({'symbol': pd.Categorical.from_codes(sym_codes, sym_uniq)})
    d['bar_date'] = pd.Categorical.from_codes(day_codes, day_uniq)
    d['_sym'] = sym_codes
    for k in FCOLS:
        d[k] = arr[k]
        del arr[k]
    del arr
    gc.collect()
    log(f'- panel rows {n0:,} -> {n1:,} after (symbol, bar_date) dedup '
        f'({n0 - n1:,} duplicate keys dropped); symbols {len(sym_uniq):,}; '
        f'days {len(day_uniq):,} {day_uniq[0]}..{day_uniq[-1]}')
    return d


def add_causal_prev(d):
    """prev_high / prev_low / prev_close / prev2_close / ssr / prev_ret_cc, block-aware per symbol."""
    sym = d['_sym'].values
    first = np.empty(len(d), dtype=bool)
    first[0] = True
    first[1:] = sym[1:] != sym[:-1]

    def sh(a):
        out = np.empty(len(a), dtype='float64')
        out[1:] = a[:-1]
        out[0] = np.nan
        out[first] = np.nan
        return out

    d['prev_high'] = sh(d.high.values.astype('float64'))
    d['prev_low'] = sh(d.low.values.astype('float64'))
    pc = sh(d.close.values.astype('float64'))
    d['prev_close_p'] = pc
    p2c = sh(pc)
    d['prev2_close'] = p2c
    with np.errstate(invalid='ignore', divide='ignore'):
        d['ssr'] = (d.prev_low.values <= 0.90 * p2c)
        d['prev_ret_cc'] = pc / p2c - 1.0
    d.loc[~np.isfinite(p2c), 'ssr'] = False        # unknown t-2 -> not flagged, counted below
    d['ssr_unknown'] = ~np.isfinite(p2c)
    return d


# ------------------------------------------------------------------ M18 attention list
def attention_list(d):
    """test_open_fade.py's selection, reproduced field for field."""
    cats = d.symbol.cat.categories
    bad = {s for s in cats if re.match(r'^Z[VWX]ZZ|^ZZ', str(s))}
    okcat = np.array([(s not in bad) and bool(SYM_RE.match(str(s))) for s in cats])
    symok = okcat[d.symbol.cat.codes.values]
    m = (symok & (d.close.values >= 5) & (d.dvol20.values >= 2e6)
         & np.isfinite(d.ret_cc.values) & np.isfinite(d.vol_ratio.values)
         & (np.abs(d.ret_cc.values) <= 0.5))
    x = pd.DataFrame({'symbol': d.symbol.values[m].astype(str),
                      'bar_date': d.bar_date.values[m].astype(str),
                      'ret_cc': d.ret_cc.values[m],
                      'vol_ratio': d.vol_ratio.values[m]})
    x['attn'] = np.abs(x.ret_cc) * x.vol_ratio
    x = x.sort_values(['bar_date', 'attn'], ascending=[True, False])
    x['rank'] = x.groupby('bar_date').attn.rank(method='first', ascending=False)
    days = sorted(x.bar_date.unique())
    nxt = {a: b for a, b in zip(days[:-1], days[1:])}
    sel = x[(x['rank'] <= 20) | ((x['rank'] > 100) & (x['rank'] <= 120))].copy()
    sel['group'] = np.where(sel['rank'] <= 20, 'attention', 'control')
    sel['trade_day'] = sel.bar_date.map(nxt)
    sel = sel[sel.trade_day.notna()]
    log(f'- M18 selection: {len(bad)} placeholder tickers and '
        f'{int((~okcat).sum()) - len(bad)} non-matching tickers dropped; ranking pool '
        f'{len(x):,} symbol-days over {len(days)} days; selected {len(sel):,} '
        f'({int((sel.group == "attention").sum()):,} attention / '
        f'{int((sel.group == "control").sum()):,} control)')
    return sel[['symbol', 'bar_date', 'trade_day', 'group', 'rank', 'attn', 'ret_cc']]


# ------------------------------------------------------------------ main
def main():
    from trading.orb_asset_class import classify_asset, load_class_map
    cmap = load_class_map()

    def acl(s):
        return cmap.get(s) or classify_asset(s, None)

    log('# Stage G step 0 — the short universe, SSR and the attention list')
    log('')
    d = load_panel()
    d = add_causal_prev(d)

    # ---------------- (a) the U1uU2 borrowable set
    mem = pd.read_csv(MEMBERS, dtype={'symbol': str, 'bar_date': str, 'split': str},
                      keep_default_na=False, na_values=[''])
    for c in ('u1', 'u2', 'u3'):
        mem[c] = mem[c].astype(str).str.lower().isin(('true', '1'))
    for c in ('open', 'prev_close', 'prev_high', 'prev_low', 'gap_pct', 'prev_day_range_pct',
              'adv20', 'dvol20_med'):
        mem[c] = pd.to_numeric(mem[c], errors='coerce')
    cal = sorted(mem.bar_date.unique())
    calset = set(cal)
    log(f'- E/members.csv {len(mem):,} rows, calendar {len(cal)} days '
        f'{cal[0]}..{cal[-1]}')
    u = mem[mem.u1 | mem.u2].copy()
    n_u12 = len(u)
    u = u[u.dvol20_med >= DVOL_MIN]
    n_dv = len(u)
    u = u[u.open >= PRICE_MIN]
    n_px = len(u)
    u['asset_class'] = [acl(s) for s in u.symbol]
    u = u[u.asset_class == 'stock'].copy()
    log(f'- U1uU2 {n_u12:,} -> dvol20_med >= ${DVOL_MIN/1e6:.0f}M {n_dv:,} '
        f'-> open >= ${PRICE_MIN:.0f} {n_px:,} -> asset_class==stock **{len(u):,}** '
        f'({len(u)/n_u12*100:.1f}% of U1uU2, {u.symbol.nunique():,} symbols, '
        f'{u.bar_date.nunique()} days)')
    u['in_u12'] = 1
    u['attn_grp'] = ''
    u['attn_rank'] = np.nan
    keep = ['bar_date', 'symbol', 'open', 'prev_close', 'prev_high', 'prev_low', 'gap_pct',
            'prev_day_range_pct', 'adv20', 'dvol20_med', 'u1', 'u2', 'split', 'asset_class',
            'in_u12', 'attn_grp', 'attn_rank']
    ua = u[keep].copy()

    # ---------------- (b) the attention / control names on their trade day
    sel = attention_list(d)
    sel.to_csv(f'{G}/attention_list.csv', index=False)
    n_sel = len(sel)
    sel = sel[sel.trade_day.isin(calset)]
    log(f'- attention/control on the 410-day calendar: {len(sel):,} of {n_sel:,} '
        f'({len(sel)/n_sel*100:.1f}%) — the rest fall on early-close days the calendar excludes')

    asyms = set(sel.symbol.unique())
    msk = np.array([s in asyms for s in d.symbol.cat.categories])[d.symbol.cat.codes.values]
    sub = pd.DataFrame({'symbol': d.symbol.values[msk].astype(str),
                        'bar_date': d.bar_date.values[msk].astype(str)})
    for c in ('close', 'volume', 'open', 'prev_close_p', 'prev_high', 'prev_low', 'adv20'):
        sub[c] = d[c].values[msk].astype('float64')
    sub['dv'] = sub.close * sub.volume
    sub['dvol20_med'] = (sub.groupby('symbol')['dv']
                         .transform(lambda s: s.shift(1).rolling(20, min_periods=10).median()))
    dvm = dict(zip(zip(sub.symbol, sub.bar_date), sub.dvol20_med))

    pan = sub[sub.bar_date.isin(calset)][
        ['symbol', 'bar_date', 'open', 'prev_close_p', 'prev_high', 'prev_low', 'adv20']].copy()
    del sub
    b = sel.rename(columns={'bar_date': 'sel_day', 'trade_day': 'bar_date'})[
        ['symbol', 'bar_date', 'group', 'rank']]
    b = b.merge(pan, on=['symbol', 'bar_date'], how='left')
    n_b0 = len(b)
    b = b[b.open.notna()]
    b['dvol20_med'] = [dvm.get((s, dd), np.nan) for s, dd in zip(b.symbol, b.bar_date)]
    b['prev_close'] = b.prev_close_p
    b['gap_pct'] = (b.open / b.prev_close - 1.0) * 100.0
    b['prev_day_range_pct'] = (b.prev_high - b.prev_low) / b.prev_low * 100.0
    n_b1 = len(b)
    b = b[b.dvol20_med >= DVOL_MIN]
    n_b2 = len(b)
    b = b[b.open >= PRICE_MIN]
    n_b3 = len(b)
    b['asset_class'] = [acl(s) for s in b.symbol]
    b = b[b.asset_class == 'stock'].copy()
    log(f'- attention/control rows: {n_b0:,} -> panel row on the trade day {n_b1:,} '
        f'-> dvol20_med >= ${DVOL_MIN/1e6:.0f}M {n_b2:,} -> open >= ${PRICE_MIN:.0f} {n_b3:,} '
        f'-> stock **{len(b):,}** ({int((b.group=="attention").sum()):,} attention / '
        f'{int((b.group=="control").sum()):,} control)')
    b['u1'] = False
    b['u2'] = False
    b['split'] = split_of(b.bar_date.values)
    b['in_u12'] = 0
    b = b.rename(columns={'group': 'attn_grp', 'rank': 'attn_rank'})
    ub = b[keep].copy()

    # ---------------- union
    m = pd.concat([ua, ub], ignore_index=True)
    m = m.sort_values(['bar_date', 'symbol', 'in_u12'], ascending=[True, True, False])
    dup = m.duplicated(['bar_date', 'symbol'], keep=False)
    # a key can arrive from both sides: keep the U1uU2 row but carry the attention flag onto it
    agrp = {(r.bar_date, r.symbol): (r.attn_grp, r.attn_rank)
            for r in ub.itertuples() if r.attn_grp}
    m = m.drop_duplicates(['bar_date', 'symbol'], keep='first').reset_index(drop=True)
    hit = 0
    grp, rnk = [], []
    for bd, sy, g0, r0 in zip(m.bar_date, m.symbol, m.attn_grp, m.attn_rank):
        v = agrp.get((bd, sy))
        if v and not g0:
            hit += 1
            grp.append(v[0])
            rnk.append(v[1])
        else:
            grp.append(g0)
            rnk.append(r0)
    m['attn_grp'] = grp
    m['attn_rank'] = rnk
    log(f'- union: {len(ua):,} U1uU2 + {len(ub):,} attention/control -> {len(m):,} distinct keys '
        f'({int(dup.sum()/2):,} keys in both, attention flag carried onto {hit:,} U1uU2 rows)')
    a_in = int(((m.attn_grp == 'attention') & (m.in_u12 == 1)).sum())
    a_all = int((m.attn_grp == 'attention').sum())
    log(f'- **attention names that are ALSO in U1uU2: {a_in:,} of {a_all:,} = '
        f'{a_in/max(a_all,1)*100:.1f}%** (the S5-strict variant of PREREG §1)')

    # ---------------- SSR (reduced to the G key set before any string materialisation)
    gsyms = set(m.symbol.unique())
    gdays = set(m.bar_date.unique())
    k1 = np.array([s in gsyms for s in d.symbol.cat.categories])[d.symbol.cat.codes.values]
    k2 = np.array([s in gdays for s in d.bar_date.cat.categories])[d.bar_date.cat.codes.values]
    kk = k1 & k2
    ss = pd.DataFrame({'symbol': d.symbol.values[kk].astype(str),
                       'bar_date': d.bar_date.values[kk].astype(str),
                       'ssr': d.ssr.values[kk], 'prev_ret_cc': d.prev_ret_cc.values[kk],
                       'ssr_unknown': d.ssr_unknown.values[kk],
                       'open_p': d.open.values[kk].astype('float64'),
                       'prev_close_p': d.prev_close_p.values[kk]})
    m = m.merge(ss[['symbol', 'bar_date', 'ssr', 'prev_ret_cc', 'ssr_unknown']],
                on=['symbol', 'bar_date'], how='left')
    m['ssr'] = m.ssr.fillna(False).astype(bool)
    m['ssr_unknown'] = m.ssr_unknown.fillna(True).astype(bool)
    tab = m.groupby('split').agg(n=('ssr', 'size'), ssr=('ssr', 'sum'),
                                 unknown=('ssr_unknown', 'sum'),
                                 ccdown=('prev_ret_cc', lambda s: int((s <= -0.10).sum())))
    tab['ssr_pct'] = (tab.ssr / tab.n * 100).round(2)
    tab['ccdown_pct'] = (tab.ccdown / tab.n * 100).round(2)
    log('')
    log('## SSR (Rule 201 fired on t-1: low_{t-1} <= 0.90 x close_{t-2})')
    log('')
    log(tab.to_string())
    log('')
    log(f'- close-to-close marker (prev_ret_cc <= -10%) agrees with the low-based trigger on '
        f'{int((m.ssr == (m.prev_ret_cc <= -0.10)).sum()):,} of {len(m):,} rows '
        f'({(m.ssr == (m.prev_ret_cc <= -0.10)).mean()*100:.1f}%); the low-based trigger is the rule.')

    # ---------------- provenance cross-check: members.csv vs the lit panel on shared keys
    chk = ua.merge(ss[['symbol', 'bar_date', 'open_p', 'prev_close_p']],
                   on=['symbol', 'bar_date'], how='inner')
    if len(chk):
        do = (chk.open - chk.open_p).abs() / chk.open * 100
        dp = (chk.prev_close - chk.prev_close_p).abs() / chk.prev_close * 100
        log('')
        log(f'- provenance check, E/members.csv vs research/lit_review_2026/daily_panel.parquet on '
            f'{len(chk):,} shared keys (both built from the SAME Databento parquet): open median '
            f'|diff| {do.median():.6f}%, p99 {do.quantile(0.99):.6f}%, max {do.max():.4f}%; '
            f'prev_close median {dp.median():.6f}%, p99 {dp.quantile(0.99):.6f}%')

    # ---------------- bar-store coverage probe (E fetch index only; the builder records the truth)
    if os.path.exists(EIDX):
        idx = pd.read_csv(EIDX, dtype={'symbol': str, 'day': str}, keep_default_na=False,
                          na_values=[''])
        have = set(zip(idx[idx.n_bars > 0].symbol, idx[idx.n_bars > 0].day))
        inidx = np.array([(s, dd) in have for s, dd in zip(m.symbol, m.bar_date)])
        log(f'- E/bars_causal index covers {int(inidx.sum()):,} of {len(m):,} G keys '
            f'({inidx.mean()*100:.1f}%); the rest must come from bars_sip.db or attention.db and '
            f'the builder records every miss in G/coverage_short_missing.csv')

    m['ssr'] = m.ssr.astype(int)
    m['ssr_unknown'] = m.ssr_unknown.astype(int)
    m = m.sort_values(['bar_date', 'symbol']).reset_index(drop=True)
    m.to_csv(f'{G}/members_g.csv', index=False)
    log('')
    log(f'- **members_g.csv: {len(m):,} symbol-days over {m.bar_date.nunique()} days, '
        f'{m.symbol.nunique():,} symbols** '
        f'(TRAIN {int((m.split=="TRAIN").sum()):,} / VAL {int((m.split=="VAL").sum()):,} / '
        f'TEST {int((m.split=="TEST").sum()):,})')
    open(f'{G}/g0_universe.md', 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
