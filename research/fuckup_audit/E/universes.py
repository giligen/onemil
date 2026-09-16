#!/usr/bin/env python3
"""Stage E Part 1 — the CAUSAL universes, defined only from data known at 09:30 ET on day t.

WHY (PLAN.md H6): every family scan so far ran on `research/bf_zero/universe.csv` — days that
ENDED with a >= 5% range — plus a causal floor (`range_so_far_pct >= 5` before the signal), which
forces every F5-F10 entry to be a LATE entry. Stage D0b then found that the ORB two-leg rule
(premarket news AND premarket $ volume) is NEGATIVE on entries after 10:00 in that universe; the
rule's home is the 09:35 window on a gap-up universe, which this universe cannot express.

THE RULE (causal by construction, no range gate at all):
  U1 gap        open_t / close_{t-1} - 1 >= +3%,  open >= $5,  adv20 >= 100K shares
  U2 prior-day  (high_{t-1} - low_{t-1}) / low_{t-1} >= 8%,    open >= $5,  adv20 >= 100K
  U3 liquid     median_20d(close * volume) >= $5M,             open >= $5   (no other gate)
  U4 premarket  NOT computable from the daily panel — defined in Part 3 from the fetched
                04:00-09:29 bars (pm_dollar_vol >= $500K). Reported, not built here.

`open_t` is the day's own open, which IS known at 09:30; every other input is from days < t.
adv20 = volume.shift(1).rolling(20, min_periods=10).mean()  -- the convention of
research/lit_review_2026/build_daily_panel.py:17 and therefore of bf_zero/universe.csv.
dvol20_med = (close*volume).shift(1).rolling(20, min_periods=10).median().

TRADING DAYS: the 420-day set of research/bf_zero/universe.csv (early closes already excluded
there), so the splits line up with every earlier stage of this program.

PRICE SCALE (PLAN.md §1): the Databento daily panel may be ADJUSTED while the Alpaca 1-min tape
is raw. 200 random (symbol, day) keys present in both are compared: panel `open` vs the 09:30 ET
minute bar's open from research/bf_zero/bars_sip.db. Agreement rate reported in E/universes.md.

WRITES ONLY research/fuckup_audit/E/{u1_keys.csv,u2_keys.csv,u3_keys.csv,universes.md,
fetch_keys.csv,pricescale.csv}. Every other path is opened read-only.
"""
import gc
import os
import random
import sqlite3
import sys
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

os.environ.setdefault('ARROW_DEFAULT_MEMORY_POOL', 'system')   # jemalloc arenas blow ulimit -v

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
E = f'{ROOT}/research/fuckup_audit/E'
PANEL = f'{ROOT}/data/research/databento/equs_daily_2025_2026.parquet'
UNI5 = f'{ROOT}/research/bf_zero/universe.csv'
SIP = f'{ROOT}/research/bf_zero/bars_sip.db'
CACHE = f'{ROOT}/data/cache.db'
ET = ZoneInfo('America/New_York')

SPLITS = [('TRAIN', '2025-01-02', '2025-12-31'),
          ('VAL', '2026-01-01', '2026-05-31'),
          ('TEST', '2026-06-01', '2026-09-11')]

MIN_OPEN = 5.0
MIN_ADV20 = 100_000.0
GAP_MIN = 0.03
PDR_MIN = 0.08
DVOL20_MIN = 5_000_000.0
FETCH_CAP = 120_000
MEMBERS = f'{E}/members.csv'   # phase `panel` -> phase `report` hand-off (U1|U2|U3 rows only)


def log(m):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {m}', flush=True)


def split_of(d):
    for name, a, b in SPLITS:
        if a <= d <= b:
            return name
    return ''


# ------------------------------------------------------------------ panel

def _trailing(vals, starts, ends, win, minp, how):
    """Causal trailing statistic: out[i] uses vals[i-win .. i-1] within its own symbol block.

    `how` is 'mean' or 'median'. NaN until `minp` prior observations exist.
    """
    out = np.full(len(vals), np.nan, dtype='float64')
    sw = np.lib.stride_tricks.sliding_window_view
    for a, b in zip(starts, ends):
        v = vals[a:b].astype('float64')
        n = len(v)
        if n < minp + 1:
            continue
        idx = np.arange(minp, n)
        lo = np.maximum(0, idx - win)
        if how == 'mean':
            cs = np.concatenate(([0.0], np.cumsum(v)))
            out[a + idx] = (cs[idx] - cs[lo]) / (idx - lo)
        else:
            # ragged head (at most `win - minp` rows), then the vectorised full windows
            for i in range(minp, min(win, n)):
                out[a + i] = np.median(v[max(0, i - win):i])
            if n > win:
                w = sw(v, win)[:n - win]           # window starting at i-win, for i in [win, n)
                out[a + win:a + n] = np.median(w, axis=1)
    return out


def build_panel():
    import pyarrow.parquet as pq
    log('reading daily panel (streamed by row group, downcast on arrival)')
    cols = {k: [] for k in ('open', 'high', 'low', 'close', 'volume')}
    sym_parts, day_parts = [], []
    sym_map, day_map = {}, {}                      # string -> int code, never retained as arrays
    pf = pq.ParquetFile(PANEL)
    for b in pf.iter_batches(batch_size=100_000,
                             columns=['bar_date', 'symbol', 'open', 'high', 'low',
                                      'close', 'volume']):
        sym_parts.append(np.fromiter(
            (sym_map.setdefault(s or '', len(sym_map)) for s in b.column('symbol').to_pylist()),
            dtype='int32', count=b.num_rows))
        day_parts.append(np.fromiter(
            (day_map.setdefault((s or '')[:10], len(day_map)) for s in b.column('bar_date').to_pylist()),
            dtype='int32', count=b.num_rows))
        for c in ('open', 'high', 'low', 'close'):
            cols[c].append(b.column(c).to_numpy(zero_copy_only=False).astype('float32'))
        cols['volume'].append(b.column('volume').to_numpy(zero_copy_only=False).astype('float64'))
        del b
    del pf
    gc.collect()
    arr = {k: np.concatenate(v) for k, v in cols.items()}
    del cols
    sym_codes = np.concatenate(sym_parts)
    day_codes = np.concatenate(day_parts)
    del sym_parts, day_parts
    sym_uniq = pd.Index(list(sym_map))
    day_uniq = pd.Index(list(day_map))
    # codes are first-seen order; re-code days to chronological so lexsort orders by date
    day_order = np.argsort(np.argsort(day_uniq.values))
    day_codes = day_order[day_codes].astype('int32')
    day_uniq = pd.Index(sorted(day_map))
    log(f'panel rows {len(sym_codes):,}  symbols {len(sym_uniq):,}  days {len(day_uniq):,}')

    order = np.lexsort((day_codes, sym_codes))
    sym_codes = sym_codes[order]
    day_codes = day_codes[order]
    for k in arr:
        arr[k] = arr[k][order]
    del order

    # contiguous symbol blocks
    edges = np.flatnonzero(np.diff(sym_codes)) + 1
    starts = np.concatenate(([0], edges))
    ends = np.concatenate((edges, [len(sym_codes)]))
    del edges

    first = np.zeros(len(sym_codes), dtype=bool)
    first[starts] = True

    def prev(a):
        p = np.empty(len(a), dtype='float32')
        p[1:] = a[:-1]
        p[0] = np.nan
        p[first] = np.nan
        return p

    log('computing causal prev-day and trailing-20 fields')
    gc.collect()
    prev_close = prev(arr['close'])
    prev_high = prev(arr['high'])
    prev_low = prev(arr['low'])
    del arr['high'], arr['low']
    pdr = (prev_high - prev_low) / prev_low * 100.0     # float32 throughout
    adv20 = _trailing(arr['volume'], starts, ends, 20, 10, 'mean').astype('float32')
    dvol = arr['close'].astype('float64') * arr['volume']
    del arr['close'], arr['volume']
    gc.collect()
    dvol20_med = _trailing(dvol, starts, ends, 20, 10, 'median').astype('float32')
    del dvol
    gc.collect()

    # only what the membership rules need survives — the frame is built column by column so
    # pandas never consolidates a wide block (this node has 1.3 GB of address space)
    op = arr['open']
    del arr
    gap = op / prev_close
    gap -= 1.0
    gap *= 100.0
    df = pd.DataFrame({'bar_date': pd.Categorical.from_codes(day_codes, day_uniq)})
    df['symbol'] = pd.Categorical.from_codes(sym_codes, sym_uniq)
    df['open'] = op
    del op
    df['prev_close'] = prev_close
    del prev_close
    df['prev_high'] = prev_high
    del prev_high
    df['prev_low'] = prev_low
    del prev_low
    df['gap_pct'] = gap
    del gap
    df['prev_day_range_pct'] = pdr
    del pdr
    df['adv20'] = adv20
    del adv20
    df['dvol20_med'] = dvol20_med
    del dvol20_med
    return df


def phase_panel():
    """Read the daily panel, apply the three causal rules, write E/members.csv, EXIT.

    Kept as its own process: the 5M-row panel plus pyarrow's buffers do not coexist with the
    coverage/report work inside `ulimit -v 1300000` (PLAN.md §1, one memory-capped process).
    """
    t0 = time.time()
    day_set = set(pd.read_csv(UNI5, usecols=['bar_date'], dtype=str,
                              keep_default_na=False, na_values=['']).bar_date)
    big = build_panel()
    panel_days = set(big.bar_date.cat.categories)
    extra = sorted(panel_days - day_set)
    log(f'panel rows {len(big):,}; panel days not on the bf_zero calendar: {len(extra)} {extra[:5]}')

    # masks as plain numpy — the 5M-row frame is never copied or widened on this node
    # the panel carries rows with a NULL ticker (and a symbol can appear twice on a day under
    # two instrument_ids); neither is a tradable key
    named = (big.symbol.values.astype(str) != '') & (big.symbol.values.astype(str) != 'nan')
    base = named & (big.open.values >= MIN_OPEN)
    liq = base & (big.adv20.values >= MIN_ADV20)
    on_cal = big.bar_date.isin(day_set).values
    m1 = on_cal & liq & (big.gap_pct.values >= GAP_MIN * 100)
    m2 = on_cal & liq & (big.prev_day_range_pct.values >= PDR_MIN * 100)
    m3 = on_cal & base & (big.dvol20_med.values >= DVOL20_MIN)
    keep = m1 | m2 | m3
    df = big[keep].reset_index(drop=True)
    del big
    gc.collect()
    df['u1'] = m1[keep]
    df['u2'] = m2[keep]
    df['u3'] = m3[keep]
    del m1, m2, m3, keep, base, liq, on_cal
    df['bar_date'] = df.bar_date.astype(str)
    df['symbol'] = df.symbol.astype(str)
    n0 = len(df)
    df = df.drop_duplicates(['bar_date', 'symbol'], keep='first').reset_index(drop=True)
    log(f'dropped {n0 - len(df):,} duplicate (day, symbol) rows (repeated instrument_ids)')
    df['split'] = df.bar_date.map(split_of)
    log(f'U1|U2|U3 = {len(df):,} symbol-days')
    df.to_csv(MEMBERS, index=False, float_format='%.6g')
    log(f'wrote {MEMBERS} ({len(df):,} rows) in {(time.time() - t0) / 60:.1f} min '
        '— phase `panel` done, re-run with `report`')


def main():
    t0 = time.time()
    uni5 = pd.read_csv(UNI5, usecols=['symbol', 'bar_date'], dtype=str,
                       keep_default_na=False, na_values=[''])
    days = sorted(set(uni5.bar_date))
    day_set = set(days)
    uni5_keys = set(zip(uni5.bar_date, uni5.symbol))
    log(f'bf_zero universe: {len(uni5):,} symbol-days, {len(days)} trading days '
        f'{days[0]}..{days[-1]}')
    df = pd.read_csv(MEMBERS, dtype={'symbol': str, 'bar_date': str, 'split': str},
                     keep_default_na=False, na_values=[''])
    for c in ('u1', 'u2', 'u3'):
        df[c] = df[c].astype(str).str.lower().isin(('true', '1'))
    for c in ('open', 'prev_close', 'gap_pct', 'prev_day_range_pct', 'adv20', 'dvol20_med'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    log(f'members: {len(df):,} rows  U1 {int(df.u1.sum()):,}  U2 {int(df.u2.sum()):,}  '
        f'U3 {int(df.u3.sum()):,}')

    cols = ['bar_date', 'symbol', 'split', 'open', 'prev_close', 'gap_pct',
            'prev_day_range_pct', 'adv20', 'dvol20_med']
    for u in ('u1', 'u2', 'u3'):
        sub = df.loc[df[u].values, cols]
        sub.to_csv(f'{E}/{u}_keys.csv', index=False, float_format='%.6g')
        log(f'{u}: {len(sub):,} symbol-days -> {u}_keys.csv')
        del sub
    gc.collect()

    # ---------------------------------------------------------------- coverage
    log('coverage: bars_sip.db (one distinct-symbol query per day)')
    sip = sqlite3.connect(f'file:{SIP}?mode=ro', uri=True, timeout=180)
    sip_by_day = {}
    for i, d in enumerate(days):
        sip_by_day[d] = set(r[0] for r in sip.execute(
            'select distinct symbol from bars where day=?', (d,)))
        if i % 100 == 0:
            log(f'  sip day {i}/{len(days)}')
    sip.close()
    n_sip_total = sum(len(v) for v in sip_by_day.values())
    log(f'bars_sip.db holds {n_sip_total:,} symbol-days on this calendar')

    def in_sip(bd, sym):
        return sym in sip_by_day.get(bd, ())

    for u in ('u1', 'u2', 'u3'):
        df[f'{u}_sip'] = False
    m = df.u1 | df.u2 | df.u3
    have = np.array([in_sip(b, s) for b, s in zip(df.bar_date[m], df.symbol[m])])
    df.loc[m, 'in_sip'] = have
    df['in_sip'] = df['in_sip'].fillna(False)
    df.loc[m, 'in_uni5'] = [(b, s) in uni5_keys for b, s in zip(df.bar_date[m], df.symbol[m])]
    df['in_uni5'] = df['in_uni5'].fillna(False)

    # cache.db intraday_bars_1min — INFORMATION ONLY (different feed provenance; never a source
    # for this store). Point queries on (symbol, bar_date), which is the covering index.
    log('coverage: cache.db intraday_bars_1min (point queries, U1|U2 only, sampled)')
    cache_hit = cache_n = 0
    try:
        cc = sqlite3.connect(f'file:{CACHE}?mode=ro', uri=True, timeout=180)
        keys = list(zip(df.bar_date[m], df.symbol[m]))
        random.seed(11)
        samp = random.sample(keys, min(4000, len(keys)))
        for b, s in samp:
            cache_n += 1
            r = cc.execute('select 1 from intraday_bars_1min where symbol=? and bar_date=? '
                           'limit 1', (s, b)).fetchone()
            if r:
                cache_hit += 1
        cc.close()
    except Exception as e:                                     # pragma: no cover
        log(f'cache.db probe failed (reported, not fatal): {e}')

    # ---------------------------------------------------------------- price scale
    log('price-scale check: 200 random keys, panel open vs 09:30 ET minute open')
    sip = sqlite3.connect(f'file:{SIP}?mode=ro', uri=True, timeout=180)
    cand = df.loc[m & df.in_sip, ['bar_date', 'symbol', 'open']].reset_index(drop=True)
    random.seed(3)
    idx = random.sample(range(len(cand)), min(200, len(cand)))
    ps = []
    for i in idx:
        b, s, op = cand.bar_date[i], cand.symbol[i], float(cand.open[i])
        d = datetime.strptime(b, '%Y-%m-%d')
        t930 = datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET).astimezone(
            timezone.utc).isoformat()
        r = sip.execute('select o from bars where symbol=? and day=? and t=?',
                        (s, b, t930)).fetchone()
        if not r:
            ps.append((s, b, op, None, None))
            continue
        mo = float(r[0])
        ps.append((s, b, op, mo, (op / mo - 1.0) * 100.0 if mo else None))
    sip.close()
    psdf = pd.DataFrame(ps, columns=['symbol', 'day', 'panel_open', 'min930_open', 'diff_pct'])
    psdf.to_csv(f'{E}/pricescale.csv', index=False)
    got = psdf.dropna(subset=['diff_pct'])
    within = {k: float((got.diff_pct.abs() <= k).mean()) if len(got) else float('nan')
              for k in (0.01, 0.1, 0.5, 2.0)}

    # ---------------------------------------------------------------- fetch keys
    fk = df.loc[(df.u1 | df.u2) & (~df.in_sip), ['bar_date', 'symbol', 'split', 'u1', 'u2']]
    fk = fk.sort_values(['bar_date', 'symbol'])
    truncated = ''
    if len(fk) > FETCH_CAP:
        keep = fk[fk.bar_date >= '2025-07-01']
        over = fk[fk.bar_date < '2025-07-01']
        over.to_csv(f'{E}/fetch_keys_overflow.csv', index=False)
        truncated = (f'U1 u U2 minus bars_sip.db = {len(fk):,} keys > the {FETCH_CAP:,} cap; '
                     f'the fetched set is bar_date >= 2025-07-01 -> **{len(keep):,} keys**. '
                     f'The other {len(over):,} keys (2025-01-02..2025-06-30, all TRAIN) are '
                     'written to `E/fetch_keys_overflow.csv` and are NOT fetched by default; '
                     '`fetch_causal.py --overflow` fetches them into the same store. Nothing is '
                     'dropped silently, and TRAIN coverage of the causal universe is therefore '
                     'HALF a year unless the overflow pass is run.')
        fk = keep
    fk.to_csv(f'{E}/fetch_keys.csv', index=False)
    log(f'fetch keys: {len(fk):,} -> fetch_keys.csv  {truncated}')

    # ---------------------------------------------------------------- report
    L = []
    L.append('# Stage E Part 1 — causal universes (counts only)\n')
    L.append(f'Generated {datetime.now(timezone.utc).isoformat(timespec="seconds")} by '
             '`research/fuckup_audit/E/universes.py`.\n')
    L.append('Membership uses ONLY data known at 09:30 ET on day t: the day\'s own open, and '
             'prior-day close / high / low / volume. No range gate, no hindsight.\n')
    L.append('| universe | rule |')
    L.append('|---|---|')
    L.append(f'| U1 gap | open/prev_close - 1 >= {GAP_MIN * 100:.0f}%, open >= ${MIN_OPEN:.0f}, '
             f'adv20 >= {MIN_ADV20:,.0f} sh |')
    L.append(f'| U2 prior-day range | (prev_high-prev_low)/prev_low >= {PDR_MIN * 100:.0f}%, '
             f'open >= ${MIN_OPEN:.0f}, adv20 >= {MIN_ADV20:,.0f} sh |')
    L.append(f'| U3 liquid slice | median_20d(close*volume) >= ${DVOL20_MIN:,.0f}, '
             f'open >= ${MIN_OPEN:.0f} (no other gate) |')
    L.append('| U4 premarket | pm_dollar_vol >= $500K — NOT computable from a daily panel; '
             'built in Part 3 from the fetched 04:00-09:29 bars |')
    L.append('')
    L.append('adv20 = `volume.shift(1).rolling(20, min_periods=10).mean()`, '
             'dvol20_med = `(close*volume).shift(1).rolling(20, min_periods=10).median()` — the '
             'convention of `research/lit_review_2026/build_daily_panel.py:17`, i.e. the one '
             'behind `research/bf_zero/universe.csv`.\n')
    L.append('## Size, per universe per split\n')
    L.append('| universe | split | symbol-days | distinct symbols | in bf_zero universe (>=5% '
             'range) | already in bars_sip.db |')
    L.append('|---|---|---:|---:|---:|---:|')
    sym = df.symbol.values
    spl = df.split.values
    uni5v = df.in_uni5.values.astype(bool)
    sipv = df.in_sip.values.astype(bool)
    for u in ('u1', 'u2', 'u3'):
        uv = df[u].values.astype(bool)
        for name, _a, _b in SPLITS + [('ALL', '', '')]:
            k = uv if name == 'ALL' else (uv & (spl == name))
            n = int(k.sum())
            nov = int((uni5v & k).sum())
            nsip = int((sipv & k).sum())
            L.append(f'| {u.upper()} | {name} | {n:,} | {len(np.unique(sym[k])):,} | '
                     f'{nov:,} ({nov / max(n, 1) * 100:.1f}%) | '
                     f'{nsip:,} ({nsip / max(n, 1) * 100:.1f}%) |')
    L.append('')
    n_u12 = int((df.u1 | df.u2).sum())
    n_u12_sip = int(((df.u1 | df.u2) & df.in_sip).sum())
    L.append(f'**U1 u U2 = {n_u12:,} symbol-days**, of which {n_u12_sip:,} '
             f'({n_u12_sip / n_u12 * 100:.1f}%) are already in `bars_sip.db`; '
             f'**{n_u12 - n_u12_sip:,} to fetch**.\n')
    L.append(f'`bars_sip.db` holds {n_sip_total:,} symbol-days on this 420-day calendar.\n')
    if cache_n:
        L.append(f'`data/cache.db::intraday_bars_1min` probe on {cache_n:,} random U1uU2 keys: '
                 f'{cache_hit:,} hits ({cache_hit / cache_n * 100:.1f}%). **Information only** — '
                 'cache.db bars are a different fetch provenance (and RTH-only); this stage '
                 'fetches SIP so the new store is one tape with `bars_sip.db`.\n')
    L.append('## U3 — not fetched in this stage\n')
    n_u3 = int(df.u3.sum())
    n_u3_missing = int((df.u3 & ~df.in_sip).sum())
    L.append(f'U3 is {n_u3:,} symbol-days, {n_u3_missing:,} of them not in `bars_sip.db`. '
             'At the KB/symbol-day this stage measures (see E/REPORT.md) the disk it would need '
             'is reported there; it is a later decision, not this stage\'s.\n')
    L.append('## Price-scale check (PLAN.md §1)\n')
    L.append(f'{len(got)} of {len(psdf)} random keys had a 09:30 ET minute bar in '
             '`bars_sip.db`. `diff = panel_open / minute_open - 1`:\n')
    L.append('| |within 0.01%|within 0.1%|within 0.5%|within 2%|median |diff||p95 |diff||')
    L.append('|---|---:|---:|---:|---:|---:|---:|')
    if len(got):
        L.append(f'| panel open vs 09:30 minute open | {within[0.01] * 100:.1f}% | '
                 f'{within[0.1] * 100:.1f}% | {within[0.5] * 100:.1f}% | '
                 f'{within[2.0] * 100:.1f}% | {got.diff_pct.abs().median():.4f}% | '
                 f'{got.diff_pct.abs().quantile(0.95):.4f}% |')
    L.append('')
    L.append('Rows: `E/pricescale.csv`.\n')
    if truncated:
        L.append(f'## Fetch truncation\n\n{truncated}\n')
    L.append(f'Built in {(time.time() - t0) / 60:.1f} min.\n')
    with open(f'{E}/universes.md', 'w') as f:
        f.write('\n'.join(L) + '\n')
    log(f'wrote {E}/universes.md in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    phase = sys.argv[1] if len(sys.argv) > 1 else 'panel'
    if phase == 'panel':
        phase_panel()
    elif phase == 'report':
        main()
    else:
        raise SystemExit('usage: universes.py [panel|report]')
