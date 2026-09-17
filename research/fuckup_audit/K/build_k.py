#!/usr/bin/env python3
"""Stage K — multi-day holds on the liquid daily panel.  Pre-registered in K/PREREG.md.

Phase A (`python3 build_k.py A`)  : builds the daily candidate table for the five declared
                                    families, scores the 20 cells on TRAIN and VAL ONLY, runs G1/G2,
                                    tails, permutation p, availability audit and capacity.
                                    Writes K/cells_trainval.csv, K/phaseA.md, K/signals_*.csv,
                                    K/trades/*.csv.
Phase B (`python3 build_k.py B`)  : reads TEST once, for the frozen G2 survivors named in K/FREEZE.md.

NOTHING here is tuned: every cut, hold, book size, stop and ranking key is the text of PREREG.md.
Where the declaration cannot be represented on daily data (K4's "half hold" of a one-day hold) the
deviation is named in the output rather than silently resolved.

Data: research/lit_review_2026/daily_panel.parquet (the panel the earlier daily tests used -- the
same loader, same causal conventions: adv20 = volume.shift(1).rolling(20, min_periods=10).mean(),
high52 = high.shift(1).rolling(250, min_periods=60).max(), ret5 = close/close.shift(5)-1).
Everything this stage adds (dvol20_med, ret3, sma20, rolling 50-day high, mean overnight) is built
with the same shift-by-one-or-inclusive-of-today rule stated per field below.

READ-ONLY outside research/fuckup_audit/K/.
"""
import gc
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

os.environ.setdefault('ARROW_DEFAULT_MEMORY_POOL', 'system')
ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
K = f'{ROOT}/research/fuckup_audit/K'
PANEL = f'{ROOT}/research/lit_review_2026/daily_panel.parquet'
CLASSMAP = f'{ROOT}/data/research/orb_asset_class_map_20260711.csv'
TEST_TICKER = re.compile(r'^Z[A-Z]ZZT$')

SPLITS = [('TRAIN', '2025-01-02', '2025-12-31'),
          ('VAL', '2026-01-01', '2026-05-31'),
          ('TEST', '2026-06-01', '2026-09-11')]
EARLY_CLOSES = {'2025-07-03', '2025-11-28', '2025-12-24', '2026-11-27', '2026-12-24'}

MIN_DVOL20_MED = 1e7          # PREREG: 20-day median dollar volume >= $10M
MIN_PRICE = 5.0               # PREREG: price >= $5 (the signal day's close)

# PREREG cost model: half the liquidity-band spread + 5 bps, per side.  Band spreads are the
# `## by liquidity` table of research/lit_review_2026/cost_curve.md, averaged over its five
# time-of-day bands (that table is measured in the SIGNAL MINUTE of a >=5%-range intraday mover, so
# for an opening/closing auction on a $10M+/day name it is a conservative UPPER bound).
BAND_SPREAD_BPS = {'>$50M': 19.334, '$5-50M': 53.876, '<$5M': 79.343}
SLIP_BPS_PER_SIDE = 5.0
# Secondary cost models, reported beside the primary, never substituted for it:
#   daily  = the band table the earlier daily tests used (daily_addons.py:25), round trip
#   auction = 5 bps per side and NO quoted spread (CLAUDE.md fill-realism rule 4: an auction
#             execution does not cross a quoted spread)
AUCTION_RT_BPS = 10.0


def log(m):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {m}', flush=True)


def split_of(d):
    for n, a, b in SPLITS:
        if a <= d <= b:
            return n
    return ''


# ---------------------------------------------------------------- causal block helpers

def block_edges(codes):
    e = np.flatnonzero(np.diff(codes)) + 1
    return np.concatenate(([0], e)), np.concatenate((e, [len(codes)]))


def trailing_median(vals, starts, ends, win, minp):
    """out[i] = median(vals[i-win .. i-1]) inside the symbol block (E/universes.py convention)."""
    out = np.full(len(vals), np.nan, dtype='float64')
    sw = np.lib.stride_tricks.sliding_window_view
    for a, b in zip(starts, ends):
        v = vals[a:b].astype('float64')
        n = len(v)
        if n < minp + 1:
            continue
        for i in range(minp, min(win, n)):
            out[a + i] = np.median(v[max(0, i - win):i])
        if n > win:
            out[a + win:a + n] = np.median(sw(v, win)[:n - win], axis=1)
    return out


def rolling_mean_incl(vals, starts, ends, win, minp):
    """out[i] = mean(vals[i-win+1 .. i]) -- INCLUDES day i (known at day i's close).

    Non-finite inputs (the panel divides by a zero prev_close on junk rows, producing inf) are
    treated as MISSING, never as a number: an inf inside a cumsum silently poisons every later
    window in that symbol.
    """
    out = np.full(len(vals), np.nan, dtype='float64')
    for a, b in zip(starts, ends):
        v = vals[a:b].astype('float64')
        ok = np.isfinite(v)
        v = np.where(ok, v, 0.0)
        n = len(v)
        if n < minp:
            continue
        cs = np.concatenate(([0.0], np.cumsum(v)))
        cnt = np.concatenate(([0], np.cumsum(ok)))
        idx = np.arange(n)
        lo = np.maximum(0, idx - win + 1)
        s = cs[idx + 1] - cs[lo]
        c = cnt[idx + 1] - cnt[lo]
        ok = c >= minp
        o = np.full(n, np.nan)
        o[ok] = s[ok] / c[ok]
        out[a:b] = o
    return out


def rolling_max_incl(vals, starts, ends, win):
    """out[i] = max(vals[i-win+1 .. i]) -- INCLUDES day i."""
    out = np.full(len(vals), np.nan, dtype='float64')
    sw = np.lib.stride_tricks.sliding_window_view
    for a, b in zip(starts, ends):
        v = vals[a:b].astype('float64')
        n = len(v)
        o = np.full(n, np.nan)
        for i in range(min(win, n)):
            o[i] = np.nanmax(v[:i + 1])
        if n > win:
            o[win:] = np.nanmax(sw(v, win)[1:], axis=1)
        out[a:b] = o
    return out


def lag_ratio(vals, starts, ends, k):
    """out[i] = vals[i] / vals[i-k] - 1 inside the block."""
    out = np.full(len(vals), np.nan, dtype='float64')
    with np.errstate(divide='ignore', invalid='ignore'):
        for a, b in zip(starts, ends):
            n = b - a
            if n > k:
                out[a + k:b] = vals[a + k:b] / vals[a:b - k] - 1.0
    return out


def any_within(flag, starts, ends, win):
    """out[i] = flag[i-win+1 .. i].any() inside the block."""
    out = np.zeros(len(flag), dtype=bool)
    for a, b in zip(starts, ends):
        v = flag[a:b].astype('int32')
        n = len(v)
        cs = np.concatenate(([0], np.cumsum(v)))
        idx = np.arange(n)
        lo = np.maximum(0, idx - win + 1)
        out[a:b] = (cs[idx + 1] - cs[lo]) > 0
    return out


def day_decile(daycode, value, mask):
    """Cross-sectional percentile rank of `value` among `mask` rows sharing a day.

    Returns an array of NaN except on masked rows, where it is the rank in [0, 1): 0 = smallest.
    Causal: every input is a day-t value, the decision is taken at day t's close.
    """
    out = np.full(len(value), np.nan)
    idx = np.flatnonzero(mask & np.isfinite(value))
    if not len(idx):
        return out
    order = idx[np.lexsort((value[idx], daycode[idx]))]
    d = daycode[order]
    newday = np.flatnonzero(np.diff(d)) + 1
    starts = np.concatenate(([0], newday))
    ends = np.concatenate((newday, [len(d)]))
    rank = np.empty(len(d))
    for a, b in zip(starts, ends):
        rank[a:b] = np.arange(b - a) / max(b - a, 1)
    out[order] = rank
    return out


# ---------------------------------------------------------------- panel

def load_panel():
    import pyarrow.parquet as pq
    log('reading daily panel by row group')
    want = ['symbol', 'bar_date', 'open', 'high', 'low', 'close', 'volume',
            'prev_close', 'adv20', 'ret5', 'high52', 'ret_on', 'vol_ratio']
    parts = {c: [] for c in want if c not in ('symbol', 'bar_date')}
    sym_parts, day_parts = [], []
    sym_map, day_map = {}, {}
    pf = pq.ParquetFile(PANEL)
    for b in pf.iter_batches(batch_size=100_000, columns=want):
        sym_parts.append(np.fromiter(
            (sym_map.setdefault(s or '', len(sym_map)) for s in b.column('symbol').to_pylist()),
            dtype='int32', count=b.num_rows))
        day_parts.append(np.fromiter(
            (day_map.setdefault((s or '')[:10], len(day_map)) for s in b.column('bar_date').to_pylist()),
            dtype='int32', count=b.num_rows))
        for c in parts:
            parts[c].append(b.column(c).to_numpy(zero_copy_only=False).astype('float32'))
        del b
    del pf
    gc.collect()
    # one column at a time: this node has 1.5 GB of address space and pyarrow's arenas are in it
    arr = {}
    for c in list(parts):
        arr[c] = np.concatenate(parts[c])
        del parts[c]
        gc.collect()
    del parts
    sym_codes = np.concatenate(sym_parts)
    day_codes = np.concatenate(day_parts)
    del sym_parts, day_parts
    gc.collect()
    syms = np.array(list(sym_map), dtype=object)
    days = np.array(list(day_map), dtype=object)
    # chronological day codes
    order_d = np.argsort(np.argsort(days))
    day_codes = order_d[day_codes].astype('int32')
    days = np.array(sorted(day_map), dtype=object)
    # the panel is written sorted by (symbol, bar_date); re-assert it
    order = np.lexsort((day_codes, sym_codes))
    if not np.array_equal(order, np.arange(len(order))):
        sym_codes = sym_codes[order]
        day_codes = day_codes[order]
        for c in arr:
            arr[c] = arr[c][order]
    del order
    gc.collect()
    log(f'panel {len(sym_codes):,} rows  {len(syms):,} symbols  {len(days)} days '
        f'{days[0]}..{days[-1]}')
    return sym_codes, day_codes, syms, days, arr


def build_features():
    sc, dc, syms, days, a = load_panel()
    starts, ends = block_edges(sc)
    log(f'{len(starts):,} symbol blocks')

    # --- price-scale sanity: the panel carries zero-price rows (K/step0.md) -------------
    bad = ~(np.isfinite(a['open']) & np.isfinite(a['close']) & (a['open'] > 0) & (a['close'] > 0)
            & (a['high'] > 0) & (a['low'] > 0))
    log(f'zero/NaN price rows in the panel: {int(bad.sum()):,} (excluded from every rule)')
    # Those rows must not enter a rolling statistic either: a zero price divides into inf and an
    # inf inside a cumsum poisons every later window of that symbol.  NaN = "no bar", everywhere.
    for k in ('open', 'high', 'low', 'close', 'volume'):
        a[k] = np.where(bad, np.nan, a[k]).astype('float32')
    a['prev_close'] = np.where(np.isfinite(a['prev_close']) & (a['prev_close'] > 0),
                               a['prev_close'], np.nan).astype('float32')
    for k in ('adv20', 'ret5', 'high52', 'ret_on', 'vol_ratio'):
        a[k] = np.where(np.isfinite(a[k]), a[k], np.nan).astype('float32')
    n_inf = int((~np.isfinite(a['ret_on']) & ~bad).sum())
    log(f'panel ret_on non-finite on otherwise-valid rows: {n_inf:,} (now NaN, not inf)')

    log('dvol20_med (trailing 20-day median of close*volume, shifted 1)')
    dvol = a['close'].astype('float64') * a['volume'].astype('float64')
    dvol20_med = trailing_median(dvol, starts, ends, 20, 10).astype('float32')
    del dvol
    gc.collect()
    log('ret3, sma20, rolling 50-day high, mean overnight 20d')
    ret3 = lag_ratio(a['close'], starts, ends, 3).astype('float32')
    sma20 = rolling_mean_incl(a['close'], starts, ends, 20, 20).astype('float32')
    gc.collect()
    hi50 = rolling_max_incl(a['high'], starts, ends, 50).astype('float32')
    is50 = np.isfinite(hi50) & (a['high'] >= hi50 * (1 - 1e-7))
    had50_10 = any_within(is50, starts, ends, 10)
    del is50
    gc.collect()
    on20 = rolling_mean_incl(a['ret_on'], starts, ends, 20, 15).astype('float32')
    rng = a['high'] - a['low']
    close_pos = np.where(rng > 0, (a['close'] - a['low']) / np.where(rng > 0, rng, 1.0),
                         0.5).astype('float32')
    close_pos = np.where(bad, np.nan, close_pos).astype('float32')
    del rng
    with np.errstate(divide='ignore', invalid='ignore'):
        gap = (a['open'] / a['prev_close'] - 1.0).astype('float32')
    gap = np.where(np.isfinite(gap), gap, np.nan).astype('float32')
    ret3 = np.where(np.isfinite(ret3), ret3, np.nan).astype('float32')
    gc.collect()

    f = dict(sym=sc, day=dc, syms=syms, days=days, starts=starts, ends=ends, bad=bad,
             open=a['open'], high=a['high'], low=a['low'], close=a['close'],
             volume=a['volume'], prev_close=a['prev_close'], adv20=a['adv20'],
             vol_ratio=a['vol_ratio'], ret5=a['ret5'], high52=a['high52'], ret_on=a['ret_on'],
             dvol20_med=dvol20_med, ret3=ret3, sma20=sma20, hi50=hi50, had50_10=had50_10,
             on20=on20, close_pos=close_pos, gap=gap)
    del a
    gc.collect()
    return f


def build_universe(f):
    """PREREG universe: dvol20_med >= $10M, close >= $5, common stock only, test tickers out."""
    cm = pd.read_csv(CLASSMAP, dtype=str, keep_default_na=False, na_values=[''])
    cls = dict(zip(cm.symbol, cm.asset_class))
    cls_of = np.array([cls.get(str(s), 'not_in_map') for s in f['syms']], dtype=object)
    is_test = np.array([bool(TEST_TICKER.match(str(s))) for s in f['syms']])
    stock = (cls_of == 'stock')
    wrapper = (cls_of == 'wrapper')
    row_stock = stock[f['sym']]
    row_wrapper = wrapper[f['sym']]
    row_test = is_test[f['sym']]
    base = (~f['bad']) & (~row_test) & np.isfinite(f['dvol20_med']) \
        & (f['dvol20_med'] >= MIN_DVOL20_MED) & (f['close'] >= MIN_PRICE)
    u_primary = base & row_stock                      # PREREG "common stock only"
    u_secondary = base & (~row_wrapper)               # stock or not-in-map (survivorship control)
    log(f'universe rows: base {int(base.sum()):,}  primary(stock) {int(u_primary.sum()):,}  '
        f'secondary(non-wrapper) {int(u_secondary.sum()):,}')
    return u_primary, u_secondary, cls_of


# ---------------------------------------------------------------- families

FAMILIES = {
    'K1': dict(desc='gap >= +8% on >= 3x ADV, close in the top third of the range',
               hold=5, stop='gap_low', strength='gap', asc=False),
    'K2': dict(desc='close at a new 250-day high (panel high52) with volume >= 2x ADV',
               hold=10, stop='pct7', strength='vol_ratio', asc=False),
    'K3': dict(desc='5-day return in the bottom decile of the universe, close > open',
               hold=3, stop=None, strength='ret5', asc=True),
    'K4': dict(desc='top decile of the trailing 20-day mean overnight return',
               hold=1, stop=None, strength='on20', asc=False),
    'K5': dict(desc='50-day high within 10 days, 3-day decline >= 5%, close > 20-day SMA',
               hold=5, stop='pct5', strength='ret3', asc=True),
}


def family_mask(fam, f, uni):
    if fam == 'K1':
        return uni & np.isfinite(f['gap']) & (f['gap'] >= 0.08) \
            & np.isfinite(f['vol_ratio']) & (f['vol_ratio'] >= 3.0) \
            & np.isfinite(f['close_pos']) & (f['close_pos'] >= 2.0 / 3.0)
    if fam == 'K2':
        return uni & np.isfinite(f['high52']) & (f['close'] >= f['high52']) \
            & np.isfinite(f['vol_ratio']) & (f['vol_ratio'] >= 2.0)
    if fam == 'K3':
        dec = day_decile(f['day'], f['ret5'], uni)
        return uni & np.isfinite(dec) & (dec < 0.10) & (f['close'] > f['open'])
    if fam == 'K4':
        dec = day_decile(f['day'], f['on20'], uni)
        return uni & np.isfinite(dec) & (dec >= 0.90)
    if fam == 'K5':
        return uni & f['had50_10'] & np.isfinite(f['ret3']) & (f['ret3'] <= -0.05) \
            & np.isfinite(f['sma20']) & (f['close'] > f['sma20'])
    raise KeyError(fam)


def band_of(dv):
    return '>$50M' if dv >= 5e7 else ('$5-50M' if dv >= 5e6 else '<$5M')


def cost_rt(dv):
    """(primary, daily-convention, auction) round-trip costs in decimal return."""
    prim = (BAND_SPREAD_BPS[band_of(dv)] + 2 * SLIP_BPS_PER_SIDE) / 1e4
    dly = (6 if dv >= 5e7 else 12 if dv >= 1e7 else 25 if dv >= 5e6 else 40) / 1e4
    return prim, dly, AUCTION_RT_BPS / 1e4


def simulate(fam, f, mask, hold, n_days):
    """Every signal of `fam` walked to its exit under the declared fill and exit rules.

    Entry  : the next trading bar's OPEN (market-on-open the morning after the signal close).
    Exit   : the close of the hold's last bar, or the close of the first bar whose CLOSE breaches
             the family's stop (PREREG: "a stop on a daily close").
    Dropped: signals whose full hold would run past the panel's last day (counted, never scored).
    """
    cfg = FAMILIES[fam]
    idx = np.flatnonzero(mask)
    sym, day = f['sym'], f['day']
    op, hi, lo, cl = f['open'], f['high'], f['low'], f['close']
    ends_of = {}
    for a, b in zip(f['starts'], f['ends']):
        ends_of[sym[a]] = b
    rows = []
    n_past_end = n_no_next = 0
    for i in idx:
        if day[i] + hold > n_days - 1:
            n_past_end += 1
            continue
        b = ends_of[sym[i]]
        if i + 1 >= b:
            n_no_next += 1
            continue
        entry = op[i + 1]
        if not np.isfinite(entry) or entry <= 0:
            n_no_next += 1
            continue
        if cfg['stop'] == 'gap_low':
            stop = lo[i]
        elif cfg['stop'] == 'pct7':
            stop = entry * 0.93
        elif cfg['stop'] == 'pct5':
            stop = entry * 0.95
        else:
            stop = np.nan
        last = min(i + hold, b - 1)
        exit_i, why = last, ('hold' if last == i + hold else 'truncated')
        if np.isfinite(stop):
            for j in range(i + 1, last + 1):
                if cl[j] <= stop:
                    exit_i, why = j, 'stop'
                    break
        exit_px = cl[exit_i]
        if not np.isfinite(exit_px) or exit_px <= 0:
            n_no_next += 1
            continue
        dv = f['dvol20_med'][i]
        cp, cd, ca = cost_rt(dv)
        gross = exit_px / entry - 1.0
        r_pct = (entry - stop) / entry if np.isfinite(stop) else np.nan
        rows.append((int(sym[i]), int(day[i]), int(day[i + 1]), int(day[exit_i]),
                     float(entry), float(exit_px), float(stop) if np.isfinite(stop) else np.nan,
                     float(r_pct) if np.isfinite(r_pct) else np.nan, why, float(gross),
                     float(gross - cp), float(gross - cd), float(gross - ca), float(dv),
                     float(f['gap'][i]), float(f['vol_ratio'][i]), float(f['ret5'][i]),
                     float(f['ret3'][i]), float(f['on20'][i]), float(f['close'][i])))
    cols = ['sym', 'sig_day', 'entry_day', 'exit_day', 'entry', 'exit', 'stop', 'r_pct', 'why',
            'gross', 'net', 'net_daily', 'net_auction', 'dvol20_med', 'gap', 'vol_ratio',
            'ret5', 'ret3', 'on20', 'sig_close']
    t = pd.DataFrame(rows, columns=cols)
    t['fam'] = fam
    t['hold'] = hold
    log(f'  {fam} hold {hold}: {len(t):,} simulated  (dropped: {n_past_end:,} past panel end, '
        f'{n_no_next:,} no next/exit bar)')
    return t, n_past_end, n_no_next


def run_book(t, f, slots):
    """First-come book: on each entry day fill free slots with the day's strongest signals.

    A position entered at the OPEN of day E and exited at the CLOSE of day X occupies a slot on
    every day E..X, so the slot is free for an entry on day X+1.  One open position per name.
    """
    if not len(t):
        return t.assign(booked=False)
    cfg = FAMILIES[t.fam.iat[0]]
    key = cfg['strength']
    t = t.sort_values(['entry_day', key], ascending=[True, cfg['asc']], kind='mergesort')
    open_pos = []                       # (exit_day, sym)
    booked = np.zeros(len(t), dtype=bool)
    sy = t.sym.to_numpy()
    ed = t.entry_day.to_numpy()
    xd = t.exit_day.to_numpy()
    for p, day in enumerate(ed):
        open_pos = [q for q in open_pos if q[0] >= day]
        if len(open_pos) >= slots:
            continue
        if any(q[1] == sy[p] for q in open_pos):
            continue
        booked[p] = True
        open_pos.append((xd[p], sy[p]))
    t = t.assign(booked=booked)
    return t


# ---------------------------------------------------------------- stats

def tstat(x):
    x = np.asarray(x, dtype='float64')
    x = x[np.isfinite(x)]
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float('nan')
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def weekly(t, days, col='net'):
    """Book return per ISO week, attributed to the EXIT week, in units of one slot."""
    if not len(t):
        return pd.Series(dtype='float64')
    d = pd.to_datetime([days[i] for i in t.exit_day])
    wk = pd.Series(d).dt.isocalendar()
    key = wk.year.astype(str) + '-W' + wk.week.astype(str).str.zfill(2)
    return pd.Series(t[col].to_numpy()).groupby(key.to_numpy()).sum()


HOLDS = {'K1': (5, 2), 'K2': (10, 5), 'K3': (3, 1), 'K4': (1, 2), 'K5': (5, 2)}
HOLD_NOTE = {'K4': 'declared hold is 1 day; a HALF of a one-day close-to-close hold is not '
                   'representable on daily bars, so the second cell EXTENDS to 2 days. This is '
                   'the one deviation from PREREG.md and it is reported, not hidden.'}
SLOTS = (10, 20)
N_PERM = 5000
PERM_SEED = 23


def per_trade_stats(t, days, split, slots):
    """Every number the gate, the tails and the capacity line need, for one cell on one split."""
    x = t[t.booked & (t.split == split)]
    out = dict(split=split, n=len(x))
    if not len(x):
        return out
    # weekly BOOK return: each position is 1/slots of the book, so the week's trade returns are
    # summed and divided by the slot count.  (Sign, t and green-week share are unaffected; the
    # magnitude becomes readable as "what the book made that week".)
    wk = weekly(x, days) / slots
    ndays = len(set(x.entry_day))
    span_days = (x.entry_day.max() - x.entry_day.min() + 1) if len(x) else 0
    out.update(
        n_weeks=len(wk),
        trades_per_week=len(x) / max(len(wk), 1),
        gross_bps=x.gross.mean() * 1e4,
        net_bps=x.net.mean() * 1e4,
        net_daily_bps=x.net_daily.mean() * 1e4,
        net_auction_bps=x.net_auction.mean() * 1e4,
        med_net_bps=x.net.median() * 1e4,
        t_trade=tstat(x.net),
        wr=(x.net > 0).mean() * 100,
        sd_trade_bps=x.net.std(ddof=1) * 1e4,
        mde_trade_bps=2.0 * x.net.std(ddof=1) / np.sqrt(len(x)) * 1e4,
        wk_mean_bps=wk.mean() * 1e4 if len(wk) else np.nan,
        wk_green=(wk > 0).mean() * 100 if len(wk) else np.nan,
        t_week=tstat(wk.to_numpy()),
        mde_week_bps=(2.0 * wk.std(ddof=1) / np.sqrt(len(wk)) * 1e4) if len(wk) > 1 else np.nan,
        stop_rate=(x.why == 'stop').mean() * 100,
        trunc_rate=(x.why == 'truncated').mean() * 100,
        span_days=int(span_days), entry_days=ndays,
    )
    r = x.net / x.r_pct
    out['net_R'] = float(r.mean()) if np.isfinite(r).any() else np.nan
    out['t_R'] = tstat(r) if np.isfinite(r).any() else np.nan
    # tails
    q99, q95 = x.net.quantile(0.99), x.net.quantile(0.95)
    out['net_bps_no_top1'] = x.net[x.net <= q99].mean() * 1e4
    out['net_bps_no_top5'] = x.net[x.net <= q95].mean() * 1e4
    win = x.net[x.net > 0]
    cap = win.quantile(0.95) if len(win) else np.nan
    out['net_bps_wincap'] = np.minimum(x.net, cap).mean() * 1e4 if np.isfinite(cap) else np.nan
    # Capacity: PREREG sizes a position at 1% of the name's 20-day median dollar volume, and PREREG
    # also sizes the book EQUAL-$ per position.  Both at once means ONE position size for the whole
    # book, set by what the names can absorb -- not a different size per name.  Weighting each trade
    # by its own dvol turns the book into a mega-cap index (1% of NVDA's $20B/day is a $200M
    # position) and the monthly P&L into that one name's return; the median is the honest size and
    # p25 the conservative one.
    pos = x.dvol20_med * 0.01
    pos_med = float(pos.median())
    out['pos_usd_med'] = pos_med
    out['pos_usd_p25'] = float(pos.quantile(0.25))
    out['book_usd'] = pos_med * slots
    mon = pd.Series(x.net.to_numpy()).groupby(
        pd.Series([days[i][:7] for i in x.exit_day]).to_numpy()).sum() * pos_med
    out['usd_per_month'] = float(mon.mean()) if len(mon) else np.nan
    out['usd_worst_month'] = float(mon.min()) if len(mon) else np.nan
    out['pct_book_per_month'] = (float(mon.mean()) / (pos_med * slots) * 100
                                 if len(mon) and pos_med > 0 else np.nan)
    out['n_months'] = len(mon)
    return out


def main(phase):
    f = build_features()
    days = f['days']
    n_days = len(days)
    u_prim, u_sec, cls_of = build_universe(f)
    early = np.array([d in EARLY_CLOSES for d in days])
    not_early = ~early[f['day']]
    u_prim = u_prim & not_early           # PLAN §1: early-close days are not SIGNAL days
    u_sec = u_sec & not_early
    split_of_day = np.array([split_of(d) for d in days], dtype=object)

    # ------------------------------------------------------------ availability audit
    log('availability audit')
    AL = ['## Availability audit (PLAN §1 standing rule)', '',
          'Coverage = share of BASE-universe symbol-days (dvol20_med >= $10M, close >= $5, common '
          'stock, non-test, valid prices, non-early-close) on which the field is finite. A field '
          'below 100% cannot be used without knowing WHO is missing.', '',
          '| field | TRAIN | VAL | TEST | what missing means |']
    AL.append('|---|---:|---:|---:|---|')
    why_missing = {
        'gap': 'no prior close (first bar of a symbol)',
        'vol_ratio': 'adv20 undefined (<10 prior bars)',
        'close_pos': 'never missing (high==low falls back to 0.5)',
        'high52': '<60 prior bars (young listing) — K2 cannot fire',
        'ret5': '<5 prior bars',
        'ret3': '<3 prior bars',
        'sma20': '<20 prior bars',
        'hi50': 'never missing (expanding window until 50 bars exist)',
        'on20': '<15 of the last 20 overnight returns',
        'dvol20_med': 'gate itself — 100% by construction',
    }
    for col in ('gap', 'vol_ratio', 'close_pos', 'high52', 'ret5', 'ret3', 'sma20', 'hi50',
                'on20', 'dvol20_med'):
        cells = []
        for sp in ('TRAIN', 'VAL', 'TEST'):
            m = u_prim & (split_of_day[f['day']] == sp)
            v = f[col][m]
            cells.append(f'{np.isfinite(v).mean() * 100:.2f}%' if len(v) else 'n/a')
        AL.append(f'| `{col}` | {cells[0]} | {cells[1]} | {cells[2]} | {why_missing[col]} |')
    AL.append('')
    AL.append('The class map is audited in `K/step0.md`: 73% of the liquid slice is positively '
              'identified as common stock, 24.5% as a leveraged/inverse wrapper, 2.1% is not in '
              'the 2026-07-11 dump at all (2025-weighted — the survivorship channel). Every cell '
              'is therefore scored on BOTH the primary (stock-only) and the secondary '
              '(non-wrapper) universe.')
    AL.append('')

    # ------------------------------------------------------------ signals + simulation
    sig_counts = []
    trades = {}
    for uni_name, uni in (('primary', u_prim), ('secondary', u_sec)):
        for fam in FAMILIES:
            m = family_mask(fam, f, uni)
            n_tr = int((m & (split_of_day[f['day']] == 'TRAIN')).sum())
            n_va = int((m & (split_of_day[f['day']] == 'VAL')).sum())
            n_te = int((m & (split_of_day[f['day']] == 'TEST')).sum())
            wk_tr = n_tr / 52.0
            sig_counts.append(dict(universe=uni_name, fam=fam, train=n_tr, val=n_va, test=n_te,
                                   train_per_week=wk_tr))
            log(f'{uni_name} {fam}: signals TRAIN {n_tr:,} ({wk_tr:.1f}/wk) VAL {n_va:,} TEST {n_te:,}')
            if uni_name == 'primary':
                for hold in HOLDS[fam]:
                    t, _pe, _nn = simulate(fam, f, m, hold, n_days)
                    trades[(fam, hold)] = t
            else:
                t, _pe, _nn = simulate(fam, f, m, HOLDS[fam][0], n_days)
                trades[(fam, HOLDS[fam][0], 'sec')] = t
    pd.DataFrame(sig_counts).to_csv(f'{K}/signal_counts.csv', index=False)

    # ------------------------------------------------------------ book + score
    rows = []
    wkser = {}
    os.makedirs(f'{K}/trades', exist_ok=True)
    for key, t in trades.items():
        fam, hold = key[0], key[1]
        sec = len(key) > 2
        if not len(t):
            continue
        t = t.copy()
        t['split'] = [split_of(days[i]) for i in t.sig_day]
        for slots in SLOTS:
            if sec and slots != 10:
                continue
            tb = run_book(t, f, slots)
            cell = f'{fam}_h{hold}_n{slots}' + ('_sec' if sec else '')
            x = tb[tb.booked]
            outp = x.assign(symbol=[str(f['syms'][s]) for s in x.sym],
                            sig_date=[days[i] for i in x.sig_day],
                            entry_date=[days[i] for i in x.entry_day],
                            exit_date=[days[i] for i in x.exit_day])
            keep = ['symbol', 'sig_date', 'entry_date', 'exit_date', 'split', 'entry', 'exit',
                    'stop', 'r_pct', 'why', 'gross', 'net', 'net_daily', 'net_auction',
                    'dvol20_med', 'gap', 'vol_ratio', 'ret5', 'ret3', 'on20', 'sig_close']
            outp[keep].to_csv(f'{K}/trades/{cell}.csv', index=False, float_format='%.6g')
            for sp in (('TRAIN', 'VAL') if phase == 'A' else ('TRAIN', 'VAL', 'TEST')):
                s = per_trade_stats(tb, days, sp, slots)
                s.update(cell=cell, fam=fam, hold=hold, slots=slots,
                         universe='secondary' if sec else 'primary')
                rows.append(s)
                xx = tb[tb.booked & (tb.split == sp)]
                wkser[(cell, sp)] = weekly(xx, days) / slots
    cells = pd.DataFrame(rows)
    cells.to_csv(f'{K}/cells_{"trainval" if phase == "A" else "all"}.csv', index=False,
                 float_format='%.6g')
    log(f'wrote {len(cells)} cell-split rows')

    # ------------------------------------------------------------ permutation p (search-adjusted)
    perm = {}
    for sp in (('TRAIN', 'VAL') if phase == 'A' else ('TRAIN', 'VAL', 'TEST')):
        prim = [c for c in cells[cells.universe == 'primary'].cell.unique()]
        series = {c: wkser.get((c, sp), pd.Series(dtype='float64')) for c in prim}
        allwk = sorted(set().union(*[set(s.index) for s in series.values() if len(s)]))
        if not allwk:
            continue
        M = np.zeros((len(prim), len(allwk)))
        present = np.zeros((len(prim), len(allwk)), dtype=bool)
        for i, c in enumerate(prim):
            s = series[c].reindex(allwk)
            present[i] = s.notna().to_numpy()
            M[i] = s.fillna(0.0).to_numpy()
        obs_t = np.array([tstat(M[i][present[i]]) if present[i].sum() > 1 else np.nan
                          for i in range(len(prim))])
        rng = np.random.default_rng(PERM_SEED)
        maxt = np.empty(N_PERM)
        for b in range(N_PERM):
            fl = rng.choice([-1.0, 1.0], size=len(allwk))
            ts = []
            for i in range(len(prim)):
                v = (M[i] * fl)[present[i]]
                ts.append(tstat(v) if len(v) > 1 else np.nan)
            maxt[b] = np.nanmax(ts) if np.isfinite(ts).any() else -np.inf
        for i, c in enumerate(prim):
            perm[(c, sp)] = float(np.mean(maxt >= obs_t[i])) if np.isfinite(obs_t[i]) else np.nan
        log(f'permutation {sp}: {len(prim)} cells x {N_PERM} sign-flip draws done')
    pd.DataFrame([dict(cell=c, split=s, p_adj=v) for (c, s), v in perm.items()]).to_csv(
        f'{K}/perm_p.csv', index=False)

    pd.DataFrame(sig_counts).to_csv(f'{K}/signal_counts.csv', index=False)
    with open(f'{K}/availability.md', 'w') as fh:
        fh.write('\n'.join(AL) + '\n')
    log('phase %s done' % phase)
    return cells, perm, sig_counts, wkser, days


if __name__ == '__main__':
    ph = sys.argv[1] if len(sys.argv) > 1 else 'A'
    main(ph)
