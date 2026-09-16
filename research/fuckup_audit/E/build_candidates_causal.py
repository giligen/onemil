#!/usr/bin/env python3
"""Stage E Part 3 — pass 1 on the CAUSAL universes (U1 gap / U2 prior-day range).

WHAT IS DIFFERENT FROM STAGE B, AND ONLY THIS:
  1. the UNIVERSE is causal at 09:30 (E/members.csv, built by E/universes.py) instead of
     research/bf_zero/universe.csv (a day list defined by the day's FINAL range);
  2. the BARS come from E/bars_causal/ (the Alpaca-SIP fetch of Part 2) and, for keys it already
     held, research/bf_zero/bars_sip.db — ONE tape, SIP, adjustment raw. `data/cache.db` is NOT a
     source here (different fetch provenance, and the parity review of 2026-09-15 is the reason
     this program keeps one tape);
  3. NO range-so-far floor is needed or applied — membership is causal by construction, so a
     09:35 entry is legitimate. `range_so_far_pct` is still emitted so the two universes can be
     compared under the same scorer;
  4. five family-configs only: F8 N=5 (the ORB entry, 09:35), F8 N=15, F8 N=30, F6, F13.

EVERYTHING ELSE IS STAGE B's CODE, IMPORTED, NOT COPIED: the family detectors, both fill models,
the exit walks and every feature come from research/fuckup_audit/B/build_candidates4.py via
`B4.build_day`. Parity with `B/candidates4.csv` is therefore by construction and is MEASURED by
E/parity_causal.py on the keys both files hold with the same tape.

THE THIN IMPORT.  `build_candidates4` imports `build_candidates`, which at module scope loads the
whole 5M-row Databento daily panel and the >=5%-range universe (>3 GB; Stage B ran at
`ulimit -v 3500000`, which this node cannot spare while the Stage-B build is still running). This
script therefore imports `build_candidates` FIRST with `pd.read_parquet`/`pd.read_csv` stubbed to
return one-row frames, registers it in `sys.modules`, and then imports `build_candidates4`, whose
own `import build_candidates` is a cache hit. Nothing that this script uses is affected: the
panel-derived fields (prev_close / prev_high / prev_low / adv20) are supplied per row from
E/members.csv, which computes them from the SAME parquet panel with the same causal convention,
and `B.uni` / `B.daily` / `B.spy` are never read.

EXTRA COLUMNS (on top of B4.COLS, same order, then these):
  u1, u2, u3, split               universe membership and the PLAN §1 split
  open_daily, gap_pct_daily       the daily panel's 09:30 open and gap (`gap_pct` in B4.COLS is
                                  the same quantity computed from the MINUTE tape's 09:30 open —
                                  the two are a free price-scale cross-check per row)
  prev_day_range_pct_daily        (prev_high-prev_low)/prev_low from the panel; B4's
                                  `prev_day_range_pct` is the same number by the same formula
  dvol20_med                      20-day median dollar volume (the U3 rule's input)
  pm_dollar_vol                   ALREADY in B4.COLS — computed by B4.build_day from the 04:00-09:29
                                  bars of this store, which is why Part 2 fetched the full session
  news_key, has_news, n_news_prev15
                                  joined from research/fuckup_audit/D/news_presence.csv on
                                  (day, symbol). That file's key set is Stage D's population
                                  (F6/F8 entries >= 10:00 on the >=5%-range universe), so most
                                  causal-universe keys have NO row: news_key=0 and has_news EMPTY.
                                  The null share is printed at the end and belongs in E/REPORT.md.
                                  D/d0_news.py can be pointed at this file's key set later.

WRITES ONLY research/fuckup_audit/E/{candidates_causal*.csv, build_causal_state*.json,
coverage_causal_missing*.csv}. Every other path is read-only.

RUN:
  setsid nohup bash -c "cd /home/ec2-user/onemil; ulimit -v 1300000; nice -n 10 \
    python3 research/fuckup_audit/E/build_candidates_causal.py \
    > research/fuckup_audit/E/build_causal.log 2>&1; \
    echo EXIT=\\$? >> research/fuckup_audit/E/build_causal.log" >/dev/null 2>&1 </dev/null &
  E_DAYS=3 E_TAG=_smoke python3 research/fuckup_audit/E/build_candidates_causal.py   # smoke test
"""
import gc
import json
import os
import sqlite3
import sys
import time

os.environ.setdefault('ARROW_DEFAULT_MEMORY_POOL', 'system')

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
os.environ.setdefault('BFZ_SIP_STORE', f'{ROOT}/research/bf_zero/bars_sip.db')
os.environ['BFZ_SLIP'] = '0.0'                     # exactly what build_candidates4 sets
sys.path.insert(0, f'{ROOT}/research/bf_zero')
sys.path.insert(0, f'{ROOT}/research/fuckup_audit/B')
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

E = f'{ROOT}/research/fuckup_audit/E'
STORE = f'{E}/bars_causal'
MEMBERS = f'{E}/members.csv'
SIP = f'{ROOT}/research/bf_zero/bars_sip.db'
NEWS = f'{ROOT}/research/fuckup_audit/D/news_presence.csv'
TAG = os.environ.get('E_TAG', '')
OUT = f'{E}/candidates_causal{TAG}.csv'
STATE = f'{E}/build_causal_state{TAG}.json'
MISS = f'{E}/coverage_causal_missing{TAG}.csv'


def log(m):
    print(f'{time.strftime("%H:%M:%S")} {m}', flush=True)


# ---------------------------------------------------------------- the thin import
def _stub_import():
    """Import build_candidates with its two heavy module-scope loads stubbed out, then
    build_candidates4 on top of it. Returns the build_candidates4 module."""
    _rp, _rc = pd.read_parquet, pd.read_csv

    def stub_parquet(path, *a, **k):
        if 'equs_daily' in str(path):
            return pd.DataFrame({'symbol': ['SPY'], 'bar_date': ['2025-01-02'],
                                 'open': [1.0], 'high': [1.0], 'low': [1.0],
                                 'close': [1.0], 'volume': [1.0]})
        return _rp(path, *a, **k)

    def stub_csv(path, *a, **k):
        if str(path).endswith('bf_zero/universe.csv'):
            d = pd.DataFrame({'symbol': ['SPY'], 'bar_date': ['2025-01-02'], 'open': [1.0],
                              'high': [1.0], 'low': [1.0], 'close': [1.0], 'volume': [1.0],
                              'adv20': [1.0], 'prev_vol': [1.0]})
            uc = k.get('usecols')
            return d[list(uc)] if uc else d
        return _rc(path, *a, **k)

    pd.read_parquet, pd.read_csv = stub_parquet, stub_csv
    try:
        import build_candidates                      # noqa: F401  (registers in sys.modules)
    finally:
        pd.read_parquet, pd.read_csv = _rp, _rc
    import build_candidates4 as B4                   # its `import build_candidates` is a cache hit
    gc.collect()
    return B4


B4 = _stub_import()
B = B4.B
OPEN_M = B.OPEN_M

FAMS = [('F8', dict(N=5)), ('F8', dict(N=15)), ('F8', dict(N=30)),
        ('F6', {}), ('F13', dict(K=5, X=0.04))]
B4.FAMS = FAMS

EXTRA = ['u1', 'u2', 'u3', 'split', 'open_daily', 'gap_pct_daily',
         'prev_day_range_pct_daily', 'dvol20_med', 'news_key', 'has_news', 'n_news_prev15']
COLS = B4.COLS + EXTRA


# ---------------------------------------------------------------- bars
_sip = sqlite3.connect(f'file:{SIP}?mode=ro', uri=True, timeout=180)


def _minute_from_iso(s):
    """ISO-UTC string -> ET minute of day. bars_sip.db's `t` is a fixed 25-char ISO string."""
    ts = pd.to_datetime(s, utc=True, format='ISO8601').dt.tz_convert('America/New_York')
    return (ts.dt.hour * 60 + ts.dt.minute).values


def load_bars_causal(day, syms):
    """{symbol: DataFrame(m,o,h,l,c,v)} — E/bars_causal parquet first, then bars_sip.db.

    Same contract as build_candidates.load_bars (minute-of-day ET, sorted, one row per minute,
    premarket INCLUDED so pm_dollar_vol is computable), with cache.db deliberately excluded.
    """
    want = set(syms)
    out = {}
    p = f'{STORE}/day={day}/bars.parquet'
    if os.path.exists(p):
        import pyarrow.parquet as pq
        t = pq.read_table(p, columns=['symbol', 't', 'o', 'h', 'l', 'c', 'v'])
        d = t.to_pandas()
        del t
        d = d[d.symbol.isin(want)]
        for s, gg in d.groupby('symbol', sort=False):
            out[s] = gg[['t', 'o', 'h', 'l', 'c', 'v']].rename(columns={'t': 'm'})
        del d
    left = [s for s in syms if s not in out]
    if left:
        q = ('select symbol, t, o, h, l, c, v from bars where day=? and symbol in '
             f'({",".join("?" * len(left))})')
        d = pd.read_sql(q, _sip, params=[day] + left)
        if len(d):
            d['m'] = _minute_from_iso(d.t)
            for s, gg in d.groupby('symbol', sort=False):
                out[s] = gg[['m', 'o', 'h', 'l', 'c', 'v']]
        del d
    res = {}
    for s, gg in out.items():
        gg = gg.astype({'m': int}).sort_values('m').drop_duplicates('m')
        res[s] = gg[['m', 'o', 'h', 'l', 'c', 'v']].reset_index(drop=True)
    return res


B.load_bars = load_bars_causal                        # B4.build_day calls B.load_bars


# ---------------------------------------------------------------- inputs
def load_members():
    d = pd.read_csv(MEMBERS, dtype={'symbol': str, 'bar_date': str, 'split': str},
                    keep_default_na=False, na_values=[''])
    for c in ('u1', 'u2', 'u3'):
        d[c] = d[c].astype(str).str.lower().isin(('true', '1'))
    for c in ('open', 'prev_close', 'prev_high', 'prev_low', 'gap_pct',
              'prev_day_range_pct', 'adv20', 'dvol20_med'):
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d[d.u1 | d.u2].reset_index(drop=True)
    log(f'members U1|U2: {len(d):,} symbol-days over {d.bar_date.nunique()} days')
    return d


def load_news():
    if not os.path.exists(NEWS):
        log('news_presence.csv absent — news columns will be empty')
        return {}
    d = pd.read_csv(NEWS, dtype={'symbol': str, 'day': str},
                    keep_default_na=False, na_values=[''])
    d['n'] = pd.to_numeric(d.n_prev15_to_0930, errors='coerce').fillna(0).astype(int)
    d = d[pd.to_numeric(d.fetch_ok, errors='coerce').fillna(0) > 0]
    return {(r.day, r.symbol): int(r.n) for r in d.itertuples()}


# ---------------------------------------------------------------- main
def main():
    mem = load_members()
    news = load_news()
    state = json.load(open(STATE)) if os.path.exists(STATE) else {'done': []}
    done = set(state['done'])
    days = sorted(mem.bar_date.unique())
    if os.environ.get('E_DAYS'):
        want = os.environ.get('E_DAY_LIST')
        days = want.split(',') if want else days[:int(os.environ['E_DAYS'])]
    todo = [d for d in days if d not in done]
    log(f'candidates_causal | days {len(days)} | done {len(done)} | todo {len(todo)} | '
        f'fam-configs {len(FAMS)} | out {OUT}')
    t0 = time.time()
    n = n_news = n_rows_news = 0
    for k, day in enumerate(todo):
        sub = mem[mem.bar_date == day]
        rows, miss = B4.build_day(day, sub)
        if rows:
            flags = {(r.bar_date, r.symbol): r for r in sub.itertuples()}
            for d in rows:
                f = flags[(d['day'], d['symbol'])]
                d['u1'], d['u2'], d['u3'] = int(f.u1), int(f.u2), int(f.u3)
                d['split'] = f.split
                d['open_daily'] = f.open
                d['gap_pct_daily'] = f.gap_pct
                d['prev_day_range_pct_daily'] = f.prev_day_range_pct
                d['dvol20_med'] = f.dvol20_med
                key = (d['day'], d['symbol'])
                if key in news:
                    d['news_key'] = 1
                    d['n_news_prev15'] = news[key]
                    d['has_news'] = int(news[key] > 0)
                    n_rows_news += 1
                else:
                    d['news_key'] = 0
                    d['n_news_prev15'] = ''
                    d['has_news'] = ''
            pd.DataFrame(rows).reindex(columns=COLS).to_csv(
                OUT, mode='a', header=not os.path.exists(OUT), index=False)
        if miss:
            pd.DataFrame({'symbol': miss, 'bar_date': day}).to_csv(
                MISS, mode='a', header=not os.path.exists(MISS), index=False)
        n += len(rows)
        n_news += len(miss) * 0
        state['done'].append(day)
        json.dump(state, open(STATE, 'w'))
        el = time.time() - t0
        log(f'{k + 1}/{len(todo)} {day} keys {len(sub)} rows+={len(rows)} miss={len(miss)} '
            f'total {n:,} | {el / 60:.1f} min, {el / (k + 1):.1f} s/day, '
            f'ETA {(len(todo) - k - 1) * el / (k + 1) / 60:.0f} min')
    log(f'DONE rows {n:,} | news key present on {n_rows_news:,} rows '
        f'({n_rows_news / max(n, 1) * 100:.1f}%)')


if __name__ == '__main__':
    main()
