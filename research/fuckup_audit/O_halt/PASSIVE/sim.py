#!/usr/bin/env python3
"""S1-PASSIVE step 2 — simulate the passive short LIMIT fill and the cover, per PREREG.md §2/§3.

Reuses `O_halt.score_cells.Bars` for the bar tape (SIP first, Alpaca cache second) — not rewritten.

For each candidate event (reopen >= ref * 0.994) and each of the 3 x 2 cells:
    limit = max(NBB_at_entry_t + 0.01, reopen * (1 + b))       b in {0, 0.005, 0.010}
    touch  arm: first bar in [entry_t, resume_ts + 5 min] with HIGH >= limit
    strict arm: first bar in the same window with OPEN >= limit
    fill price = limit in both arms (never the better print — conservative for a short)
Exit horizons +5m / +30m / close measured from the FILL bar; the primary cover is the OPEN of the
bar AFTER the horizon bar (marketable buy-to-cover); the O_halt convention (horizon bar's CLOSE) is
carried alongside.

Emits sim_rows.csv (one row per event x cell) and cover_keys.csv (distinct quote instants to fetch).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, f'{ROOT}/research/fuckup_audit/O_halt')

from score_cells import Bars                                   # noqa: E402

P = f'{ROOT}/research/fuckup_audit/O_halt/PASSIVE'
SRC = f'{ROOT}/research/fuckup_audit/O_halt/trades.parquet'
FLOOR = 0.994
TICK = 0.01
BS = [0.0, 0.005, 0.010]
FILL_WINDOW_MIN = 5          # "within 5 minutes of the resume"
HORIZONS = {'h5': 5, 'h30': 30}
EOD = pd.Timedelta(hours=15, minutes=55)


def main() -> int:
    t = pd.read_parquet(SRC)
    t = t[t['fill'] >= t['ref'] * FLOOR].copy()
    for c in ('halt_ts', 'resume_ts', 'entry_t'):
        t[c] = pd.to_datetime(t[c], utc=True).dt.tz_convert('America/New_York')

    q = pd.read_csv(f'{P}/entry_nbbo.csv', keep_default_na=False, na_values=[''],
                    dtype={'symbol': str, 'day': str})
    q['nbb'] = pd.to_numeric(q['nbb'], errors='coerce')
    q['mean_spread'] = pd.to_numeric(q['mean_spread'], errors='coerce')
    q = q.drop_duplicates(subset=['symbol', 'day', 'halt_seq'], keep='last')
    t = t.merge(q[['symbol', 'day', 'halt_seq', 'nbb', 'nbo', 'spread', 'mean_spread', 'err']],
                on=['symbol', 'day', 'halt_seq'], how='left')

    excl = t['nbb'].isna() | t['mean_spread'].isna()
    print(f'candidates {len(t):,}   excluded for missing entry NBBO/spread {int(excl.sum()):,}')
    exc = t[excl].copy()
    exc['split'] = np.where(exc.day <= '2025-12-31', 'TRAIN',
                            np.where(exc.day <= '2026-05-31', 'VAL', 'TEST'))
    exc[['day', 'symbol', 'halt_seq', 'split', 'err']].to_csv(f'{P}/excluded_no_quote.csv', index=False)
    t = t[~excl].copy()

    bars, cache = Bars(), {}
    rows, cover_keys = [], set()
    for e in t.itertuples():
        key = (e.symbol, e.day)
        if key not in cache:
            cache[key] = bars.get(*key)
        bd, _src = cache[key]
        if bd is None or bd.empty:
            continue
        post = bd[bd.index >= e.entry_t]
        if post.empty:
            continue
        assert abs(float(post['o'].iloc[0]) - float(e.fill)) < 1e-6, f'reopen mismatch {e.symbol} {e.day}'
        win_end = e.resume_ts + pd.Timedelta(minutes=FILL_WINDOW_MIN)
        win = post[post.index <= win_end]
        eod_t = e.entry_t.normalize() + EOD
        base = dict(day=e.day, symbol=e.symbol, halt_seq=e.halt_seq, side=e.side,
                    ref=e.ref, reopen=e.fill, nbb=e.nbb, prev_close=e.prev_close,
                    adv20=e.adv20, entry_t=e.entry_t, resume_ts=e.resume_ts,
                    entry_vol=float(post['v'].iloc[0]),
                    mean_spread=e.mean_spread, n_win_bars=len(win))
        for b in BS:
            limit = max(e.nbb + TICK, e.fill * (1.0 + b))
            for arm in ('touch', 'strict'):
                hit = win[win['h'] >= limit] if arm == 'touch' else win[win['o'] >= limit]
                r = dict(base, b=b, arm=arm, limit=limit, filled=0)
                if len(hit):
                    ft = hit.index[0]
                    r.update(filled=1, fill_t=ft, fill_px=limit,
                             fill_vol=float(hit['v'].iloc[0]),
                             fill_delay_min=int((ft - e.entry_t).total_seconds() // 60))
                    after = bd[bd.index >= ft]
                    for name, mins in HORIZONS.items():
                        tgt = min(ft + pd.Timedelta(minutes=mins), eod_t)
                        w = after[after.index <= tgt]
                        r[f'close_{name}'] = float(w['c'].iloc[-1]) if len(w) else limit
                        nxt = after[after.index > tgt]
                        if len(nxt):
                            r[f'cov_{name}'] = float(nxt['o'].iloc[0])
                            r[f'cov_{name}_t'] = nxt.index[0]
                            cover_keys.add((e.symbol, e.day, nxt.index[0].isoformat()))
                        else:
                            r[f'cov_{name}'] = r[f'close_{name}']
                            r[f'cov_{name}_t'] = w.index[-1] if len(w) else ft
                            r[f'cov_{name}_flag'] = 'no_next_bar'
                            cover_keys.add((e.symbol, e.day, str(r[f'cov_{name}_t'])))
                    w = after[after.index <= eod_t]
                    r['close_eod'] = float(w['c'].iloc[-1]) if len(w) else limit
                    r['cov_eod'] = r['close_eod']
                    r['cov_eod_t'] = w.index[-1] if len(w) else ft
                    cover_keys.add((e.symbol, e.day, str(r['cov_eod_t'])))
                rows.append(r)
    out = pd.DataFrame(rows)
    out.to_parquet(f'{P}/sim_rows.parquet', index=False)
    ck = pd.DataFrame(sorted(cover_keys), columns=['symbol', 'day', 'ts'])
    ck.to_csv(f'{P}/cover_keys.csv', index=False)
    print(f'sim rows {len(out):,}   events {out[["day","symbol","halt_seq"]].drop_duplicates().shape[0]:,}'
          f'   distinct cover instants {len(ck):,}')
    print(out.groupby(['b', 'arm'])['filled'].agg(['size', 'mean']).round(4).to_string())
    return 0


if __name__ == '__main__':
    sys.exit(main())
