#!/usr/bin/env python3
"""S1-REVIVE step 2 — simulate the limit entry (both sides, all rungs) and the bracketed exit.

Contract frozen in REVIVE/PREREG.md.  Reuses `O_halt.score_cells.Bars` (SIP tape first, Alpaca cache
second) — not rewritten.

Entry (PREREG §1/§3): a resting LIMIT at a price we declare. Fill iff the market trades at or through
it inside [entry_t, resume_ts + 5 min]. Fill price = the limit, never the better print. Unfilled =
no trade, 0 R, counted. A pseudo-rung `print` (entry at the reopen print itself, always filled) is
emitted for EVERY event so the non-fill counterfactual is directly measurable.

Exit (PREREG §1): stop = 1R adverse, target = 2R favourable, else horizon.
  * the FILL BAR IS EXCLUDED from both legs — the repo's standing convention
    (`trading/bf_trail.entry_bar_excluded`, `skip_exits_until_ts`): the bracket legs are not resting
    until the fill is confirmed, i.e. from the next closed bar. Disclosed, not silently chosen.
  * later bars: STOP first, then target (conservative tie-break)
  * a bar that OPENS through the stop exits at that open (gap), not at the stop level
  * horizon exit = the OPEN of the bar AFTER the horizon bar (marketable); if none, the last close
  * r_pct = 0 is the NO-STOP arm: no stop, no target, horizon exit only — the O_halt/PASSIVE
    construction, carried so the revival is comparable to what was killed. Scored in R units at
    R = 2% of the fill price (the declared O_halt scale).
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

P = f'{ROOT}/research/fuckup_audit/O_halt/REVIVE'
PAS = f'{ROOT}/research/fuckup_audit/O_halt/PASSIVE'
SRC = f'{ROOT}/research/fuckup_audit/O_halt/trades.parquet'

FLOOR, CEIL = 0.994, 1.006
TICK = 0.01
WINDOW_MIN = 5
R_PCTS = [0.0, 2.0, 4.0, 6.0, 8.0]      # 0.0 = the NO-STOP arm (O_halt/PASSIVE construction)
HORIZONS = {'h5': 5, 'h30': 30, 'eod': None}
EOD = pd.Timedelta(hours=15, minutes=55)
SHORT_RUNGS = [('b0', 0.0)]
LONG_RUNGS = [('d0', 0.0), ('d05', -0.005), ('d10', -0.010), ('d20', -0.020)]


def load_quotes() -> pd.DataFrame:
    frames = []
    for path in (f'{PAS}/entry_nbbo.csv', f'{P}/entry_nbbo_long.csv'):
        q = pd.read_csv(path, keep_default_na=False, na_values=[''],
                        dtype={'symbol': str, 'day': str})
        frames.append(q)
    q = pd.concat(frames, ignore_index=True)
    for c in ('nbb', 'nbo', 'spread', 'mean_spread'):
        q[c] = pd.to_numeric(q[c], errors='coerce')
    q = q.drop_duplicates(subset=['symbol', 'day', 'halt_seq'], keep='last')
    return q[['symbol', 'day', 'halt_seq', 'entry_t', 'q_ts', 'nbb', 'nbo',
              'spread', 'mean_spread', 'err']]


def resolve_exit(bars, fill_t, fill_px, is_short, r_pct, hz_min, eod_t):
    """Walk the bars and return (exit_px, why, exit_t). PREREG §1 exit contract."""
    r_abs = fill_px * r_pct / 100.0
    stop_px = fill_px + r_abs if is_short else fill_px - r_abs
    tgt_px = fill_px - 2 * r_abs if is_short else fill_px + 2 * r_abs
    end_t = eod_t if hz_min is None else min(fill_t + pd.Timedelta(minutes=hz_min), eod_t)
    seg = bars[(bars.index >= fill_t) & (bars.index <= end_t)]
    if seg.empty:
        return fill_px, 'nobar', fill_t
    o = seg['o'].values
    h = seg['h'].values
    lo = seg['l'].values
    idx = seg.index
    for i in range(1, len(seg)):                      # fill bar EXCLUDED on both legs
        if r_pct <= 0:
            break                                     # no-stop arm: horizon only
        stop_hit = (h[i] >= stop_px) if is_short else (lo[i] <= stop_px)
        if stop_hit:
            gapped = (o[i] > stop_px) if is_short else (o[i] < stop_px)
            return (float(o[i]) if gapped else stop_px), 'stop', idx[i]
        tgt_hit = (lo[i] <= tgt_px) if is_short else (h[i] >= tgt_px)
        if tgt_hit:
            return tgt_px, 'target', idx[i]
    nxt = bars[bars.index > end_t]
    if len(nxt):
        return float(nxt['o'].iloc[0]), 'horizon', nxt.index[0]
    return float(seg['c'].iloc[-1]), 'horizon', idx[-1]


def main() -> int:
    t = pd.read_parquet(SRC)
    for c in ('halt_ts', 'resume_ts', 'entry_t'):
        t[c] = pd.to_datetime(t[c], utc=True).dt.tz_convert('America/New_York')
    t['in_short'] = t['fill'] >= t['ref'] * FLOOR
    t['in_long'] = t['fill'] <= t['ref'] * CEIL
    t = t[t['in_short'] | t['in_long']].copy()

    q = load_quotes()
    t = t.merge(q.drop(columns=['entry_t']), on=['symbol', 'day', 'halt_seq'], how='left')
    miss = t['nbb'].isna() | t['nbo'].isna() | t['mean_spread'].isna()
    t['split'] = np.where(t.day <= '2025-12-31', 'TRAIN',
                          np.where(t.day <= '2026-05-31', 'VAL', 'TEST'))
    print(f'events {len(t):,}  short-book {int(t.in_short.sum()):,}  long-book {int(t.in_long.sum()):,}')
    print(f'excluded for missing entry NBBO/spread {int(miss.sum()):,}')
    t[miss][['day', 'symbol', 'halt_seq', 'split', 'err']].to_csv(
        f'{P}/excluded_no_quote.csv', index=False)
    print(t[miss].groupby('split').size().to_string())
    t = t[~miss].copy()

    # ---- causality assertions (PREREG §1) ----
    qts = pd.to_datetime(t['q_ts'], utc=True, errors='coerce').dt.tz_convert('America/New_York')
    assert (qts <= t['entry_t']).all(), 'gating/limit quote is not at or before the decision instant'
    assert (t['entry_t'] > t['resume_ts']).all(), 'entry bar does not start after the resume'

    bars, cache = Bars(), {}
    rows = []
    for e in t.itertuples():
        key = (e.symbol, e.day)
        if key not in cache:
            cache[key] = bars.get(*key)[0]
        bd = cache[key]
        if bd is None or bd.empty:
            continue
        post = bd[bd.index >= e.entry_t]
        if post.empty:
            continue
        assert abs(float(post['o'].iloc[0]) - float(e.fill)) < 1e-6
        win = post[post.index <= e.resume_ts + pd.Timedelta(minutes=WINDOW_MIN)]
        eod_t = e.entry_t.normalize() + EOD
        base = dict(day=e.day, symbol=e.symbol, halt_seq=e.halt_seq, halt_side=e.side,
                    split=e.split, ref=e.ref, reopen=e.fill, prev_close=e.prev_close,
                    adv20=e.adv20, entry_t=e.entry_t,
                    entry_vol=float(post['v'].iloc[0]),
                    nbb=e.nbb, nbo=e.nbo, spread=e.spread, mean_spread=e.mean_spread)

        plans = []
        if e.in_short:
            for name, b in SHORT_RUNGS:
                plans.append(('short', name, max(e.nbb + TICK, e.fill * (1.0 + b)), True))
            plans.append(('short', 'print', float(e.fill), True))
        if e.in_long:
            for name, d in LONG_RUNGS:
                plans.append(('long', name, min(e.nbo - TICK, e.fill * (1.0 + d)), False))
            plans.append(('long', 'print', float(e.fill), False))

        for side, rung, limit, is_short in plans:
            if rung == 'print':
                filled, fill_t, fill_px, fill_vol = 1, post.index[0], float(e.fill), float(post['v'].iloc[0])
            else:
                hit = win[win['h'] >= limit] if is_short else win[win['l'] <= limit]
                if len(hit) == 0:
                    rows.append(dict(base, side=side, rung=rung, limit=limit, filled=0))
                    continue
                filled, fill_t, fill_px = 1, hit.index[0], float(limit)
                fill_vol = float(hit['v'].iloc[0])
            after = bd[bd.index >= fill_t]
            for r_pct in R_PCTS:
                for hz, mins in HORIZONS.items():
                    ex_px, why, ex_t = resolve_exit(after, fill_t, fill_px, is_short, r_pct,
                                                    mins, eod_t)
                    raw = ((fill_px - ex_px) if is_short else (ex_px - fill_px)) / fill_px * 100.0
                    rows.append(dict(base, side=side, rung=rung, limit=limit, filled=filled,
                                     fill_t=fill_t, fill_px=fill_px, fill_vol=fill_vol,
                                     r_pct=r_pct, hz=hz, exit_px=ex_px, why=why, exit_t=ex_t,
                                     raw_pct=raw))
    out = pd.DataFrame(rows)
    out.to_parquet(f'{P}/sim_rows.parquet', index=False)
    print(f'\nsim rows {len(out):,}')
    fr = out[out['r_pct'].isna() | (out['r_pct'] == 2.0)]
    fr = fr[fr['hz'].isna() | (fr['hz'] == 'h5')]
    print('\nfill rate by side x rung x split:')
    print(fr.groupby(['side', 'rung', 'split'])['filled'].agg(['size', 'mean']).round(4).to_string())
    return 0


if __name__ == '__main__':
    sys.exit(main())
