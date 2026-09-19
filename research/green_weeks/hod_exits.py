#!/usr/bin/env python3
"""green_weeks — HOD-break exit cells E0..E5 on the HONEST SIP-tape population.

Population: `research/bf_zero/spec_trades.csv` — the 60,461 spec signals of
REPORT §6a, the population on which the HOD edge was REFUTED.  Entries are NOT
touched: every cell takes exactly the same signals at exactly the same fills;
only `walk_exit` moves.  Because a variant changes `exit_m`, the executable book
(`trading.hod_break.run_book`, causal slot freeing) is recomputed per cell.

Scope, to keep this to one modest process: only the LIVE-CONFIG pool is walked
(price >= `min_price`, entry minute <= `last_entry_minute` + 1) — that is the
book the owner named and the one §6a's 24/53 · 9/23 · 9/15 green weeks describe.

Parity gate (PREREG §6.1): the E0 cell must reproduce `spec_trades.csv`'s own
`rr` on every walked signal to < 1e-9, and the E0 book must reproduce §6a's
green-week counts.

Tape: cache.db (SIP-verified 99.7%) then `research/bf_zero/bars_sip.db` — the
same loader order the §6a re-simulation used, both opened READ-ONLY.  The loader
is copied verbatim from `research/bf_zero/build_candidates.load_bars` rather than
imported, because importing that module also loads a 41 MB universe CSV and a
114 MB daily parquet this walk has no use for.  The copy is not trusted on faith:
the E0 parity gate below reproduces `spec_trades.csv`'s `rr` to < 1e-9 on every
walked signal, which cannot happen unless the bars are identical.
"""
from __future__ import annotations

import os
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from trading.hod_break import (                      # noqa: E402
    HodBreakParams, STOP_FILL_SLIP, walk_exit, run_book)

OPEN_M = 570
_cache = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True,
                         timeout=120)
_side = [sqlite3.connect(f'file:{ROOT}/research/bf_zero/bars_sip.db?mode=ro',
                         uri=True, timeout=120)]


def load_bars(day, syms):
    """{symbol: DataFrame(m,o,h,l,c,v)} — verbatim `build_candidates.load_bars`
    with SIDE = the Alpaca-SIP re-fetch only (BFZ_SIP_STORE semantics)."""
    out = {}
    q = ("select symbol, timestamp as t, open as o, high as h, low as l, "
         "close as c, volume as v from intraday_bars_1min where bar_date=? "
         f"and symbol in ({','.join('?' * len(syms))})")
    for s, gg in pd.read_sql(q, _cache, params=[day] + list(syms)).groupby('symbol'):
        out[s] = gg
    for con in _side:
        left = [s for s in syms if s not in out]
        if not left:
            break
        t = pd.read_sql("select symbol, t, o, h, l, c, v from bars where day=?",
                        con, params=[day])
        for s, gg in t[t.symbol.isin(left)].groupby('symbol'):
            out[s] = gg
    res = {}
    for s, gg in out.items():
        ts = pd.to_datetime(gg.t, utc=True).dt.tz_convert('America/New_York')
        gg = (gg.assign(m=(ts.dt.hour * 60 + ts.dt.minute).values)
              .sort_values('m').drop_duplicates('m'))
        res[s] = gg[['m', 'o', 'h', 'l', 'c', 'v']].reset_index(drop=True)
    return res

D = 'research/green_weeks'
SPEC = 'research/bf_zero/spec_trades.csv'
OUT = f'{D}/hod_cells.csv'

# ---- cells: (target_r, partial_r, partial_frac, trail_r, be_at_r, ts_min, ts_r)
CELLS = {
    'E0':  (2.00, None, None, None, None, None, None),   # shipped
    'E1a': (0.50, None, None, None, None, None, None),
    'E1b': (0.75, None, None, None, None, None, None),
    'E1c': (1.00, None, None, None, None, None, None),
    'E1d': (1.50, None, None, None, None, None, None),
    'E2a': (2.00, 0.50, 0.50, None, None, None, None),
    'E2b': (2.00, 1.00, 0.50, None, None, None, None),
    'E2c': (2.00, 1.50, 0.50, None, None, None, None),
    'E3':  (None, 1.00, 0.50, 0.50, None, None, None),
    'E4a': (2.00, None, None, None, None, 15, 0.50),
    'E4b': (2.00, None, None, None, None, 30, 0.50),
    'E5':  (None, None, None, None, 0.75, None, None),
}
LABEL = {
    'E0': 'shipped: +2R resting target, stop = consol low, flat 15:55',
    'E1a': 'target +0.5R (whole)', 'E1b': 'target +0.75R (whole)',
    'E1c': 'target +1.0R (whole)', 'E1d': 'target +1.5R (whole)',
    'E2a': '50% @ +0.5R, BE, rest to +2R', 'E2b': '50% @ +1.0R, BE, rest to +2R',
    'E2c': '50% @ +1.5R, BE, rest to +2R',
    'E3': '50% @ +1.0R, rest trailed 0.5R (no target)',
    'E4a': 'time box 15 min at +0.5R', 'E4b': 'time box 30 min at +0.5R',
    'E5': 'breakeven stop at +0.75R, NO target',
}


def walk_variant(o, h, l, c, m, entry_idx, entry, stop, r_unit, p,
                 target_r=None, partial_r=None, partial_frac=None,
                 trail_r=None, be_at_r=None, ts_min=None, ts_r=None):
    """One trade's exit under a cell.  Returns (exit_idx, rr).

    Ordering inside a bar (PREREG §5), from the bar AFTER entry:
      1. flat minute -> that bar's OPEN;
      2. time box -> that bar's OPEN (first bar at/after entry + N minutes);
      3. breakeven arm off the bar's HIGH;
      4. trail ratchet from CLOSED bars at or before k-1, only after the partial;
      5. protective stop off the bar's LOW, filled min(stop, open) x (1 - 10bps)
         — a stop WINS a same-bar tie against any profit leg;
      6. target: the bar must CLOSE at or above the level; fills AT the level;
      7. partial: same trigger and fill; stop -> entry.
    With (target_r=2, everything else None) this is `hod_break.walk_exit`.
    """
    n = len(o)
    tgt = entry + target_r * r_unit if target_r is not None else None
    ppl = entry + partial_r * r_unit if partial_r is not None else None
    bel = entry + be_at_r * r_unit if be_at_r is not None else None
    ts_m = (int(m[entry_idx]) + ts_min) if ts_min is not None else None
    f_pp, px_pp = 0.0, None
    run_hi = float(h[entry_idx])
    ts_done = False

    def _rr(k, px, rsn):
        rem = 1.0 - f_pp
        r = rem * (px - entry) / r_unit
        if px_pp is not None:
            r += f_pp * (px_pp - entry) / r_unit
        return k, r, ('pp+' + rsn if px_pp is not None else rsn)

    for k in range(entry_idx + 1, n):
        if int(m[k]) >= p.flat_minute:
            return _rr(k, float(o[k]), 'eod')
        if ts_m is not None and not ts_done and int(m[k]) >= ts_m:
            ts_done = True
            if (float(o[k]) - entry) / r_unit < ts_r:
                return _rr(k, float(o[k]), 'time_stop')
        if bel is not None and h[k] >= bel:
            stop = max(stop, entry)
        if trail_r is not None and px_pp is not None:
            stop = max(stop, run_hi - trail_r * r_unit)
        if l[k] <= stop:
            return _rr(k, float(min(stop, o[k]) * (1.0 - STOP_FILL_SLIP)), 'stop')
        if tgt is not None and c[k] >= tgt:
            return _rr(k, float(tgt), 'target')
        if ppl is not None and px_pp is None and c[k] >= ppl:
            f_pp = float(partial_frac)
            px_pp = float(ppl)
            stop = max(stop, entry)
        run_hi = max(run_hi, float(h[k]))
    return _rr(n - 1, float(c[-1]), 'eod')


def main():
    p_det = HodBreakParams()
    from config import Config
    cfg = Config().hod_break_cfg
    LIVE = HodBreakParams(**(cfg.get('params') or {}))
    floor = float(cfg.get('min_price') or 0)
    print(f"live config: price >= {floor}, last entry {LIVE.last_entry_minute}, "
          f"{LIVE.max_per_day}/day, {LIVE.max_concurrent} concurrent", flush=True)

    T = pd.read_csv(SPEC, dtype={'symbol': str}, keep_default_na=False)
    for col in ('entry', 'stop', 'target', 'rr', 'price', 'adv20'):
        T[col] = pd.to_numeric(T[col], errors='coerce')
    print(f"spec signals: {len(T)}", flush=True)
    T = T[(T.entry_m <= LIVE.last_entry_minute + 1) & (T.price >= floor)]
    T = T.reset_index(drop=True)
    print(f"live-config pool: {len(T)} signals over {T.day.nunique()} days",
          flush=True)

    cells = list(CELLS)
    rr = {ce: np.full(len(T), np.nan) for ce in cells}
    xm = {ce: np.zeros(len(T), dtype=int) for ce in cells}
    why = {ce: [''] * len(T) for ce in cells}
    n_par, n_miss, max_dev = 0, 0, 0.0

    for di, (day, sub) in enumerate(T.groupby('day'), 1):
        bars = load_bars(day, sorted(set(sub.symbol)))
        for i, r in zip(sub.index, sub.itertuples()):
            gg = bars.get(r.symbol)
            if gg is None:
                n_miss += 1
                continue
            rth = gg[(gg.m >= OPEN_M) & (gg.m < 960)].reset_index(drop=True)
            if len(rth) < 10:
                n_miss += 1
                continue
            o, h, l, c = (rth[k].values.astype(float) for k in ('o', 'h', 'l', 'c'))
            m = rth.m.values.astype(int)
            hit = np.flatnonzero(m == int(r.entry_m))
            if not len(hit):
                n_miss += 1
                continue
            ei = int(hit[0])
            entry, stop = float(r.entry), float(r.stop)
            r_unit = entry - stop
            if r_unit <= 0:
                n_miss += 1
                continue
            for ce in cells:
                tr, pr, pf, trl, be, tsm, tsr = CELLS[ce]
                k, v, w = walk_variant(o, h, l, c, m, ei, entry, stop, r_unit,
                                       LIVE, target_r=tr, partial_r=pr,
                                       partial_frac=pf, trail_r=trl, be_at_r=be,
                                       ts_min=tsm, ts_r=tsr)
                rr[ce][i] = v
                xm[ce][i] = int(m[k])
                why[ce][i] = w
            # PREREG §6.1 — E0 must reproduce the shipped walker on this signal
            k0, px0, w0 = walk_exit(o, h, l, c, m, ei, entry, stop,
                                    entry + p_det.target_r * r_unit, LIVE)
            ref = (px0 - entry) / r_unit
            dev = abs(ref - rr['E0'][i])
            max_dev = max(max_dev, dev)
            assert dev < 1e-9 and w0 == why['E0'][i], (
                f"PARITY BREAK {r.symbol} {day}: shipped {ref}/{w0} vs "
                f"E0 {rr['E0'][i]}/{why['E0'][i]}")
            n_par += 1
        if di % 40 == 0:
            print(f"  {di} days · parity-checked {n_par} · missing {n_miss}",
                  flush=True)

    print(f"walked {n_par} signals · missing tape {n_miss} · "
          f"E0 max |dev| vs shipped walk_exit = {max_dev:.3e}", flush=True)

    # PREREG §6.1 second leg — E0's rr must equal the committed spec_trades.csv
    ok = np.isfinite(rr['E0'])
    d_spec = float(np.abs(rr['E0'][ok] - T.rr.to_numpy()[ok]).max())
    print(f"E0 vs spec_trades.csv rr: max |dev| = {d_spec:.3e} over {ok.sum()} rows",
          flush=True)
    assert d_spec < 1e-9, "E0 does not reproduce spec_trades.csv — STOP"

    out = T[['day', 'symbol', 'entry_m', 'entry', 'stop', 'price']].copy()
    for ce in cells:
        out[f'rr_{ce}'] = rr[ce]
        out[f'xm_{ce}'] = xm[ce]
        out[f'why_{ce}'] = why[ce]
    out = out[ok].reset_index(drop=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(out)} rows)", flush=True)

    # per-cell executable book (exits move -> slot freeing moves -> book moves)
    for ce in cells:
        rows = [(r.day, int(r.entry_m), int(getattr(r, f'xm_{ce}')), r.symbol,
                 r.Index) for r in out.itertuples()]
        idx = [t[4] for t in run_book(rows, LIVE.max_per_day, LIVE.max_concurrent)]
        bk = out.loc[idx, ['day', 'symbol', 'entry_m', f'rr_{ce}', f'why_{ce}']]
        bk = bk.rename(columns={f'rr_{ce}': 'rr', f'why_{ce}': 'why'})
        bk.to_csv(f'{D}/hod_book_{ce}.csv', index=False)
        print(f"  book {ce}: {len(bk)} trades  totR {bk.rr.sum():+.1f}", flush=True)


if __name__ == '__main__':
    main()
