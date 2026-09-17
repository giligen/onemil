#!/usr/bin/env python3
"""Stage L step 1 — build the five run_book populations (B1, B2, B3, B4, B6) in ONE schema,
book each with its own slot rule, and check the base statistics against the published anchors.

Writes  L/pop_<B>.csv  (the scoreable population, one row per candidate trade)
        L/base_stats.csv (the base book per split)
        L/l1_parity.md  (the anchor comparison + availability audit of the columns that exist here)

No cell of the Stage-L grid is scored in this file.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lcore as C                                                             # noqa: E402

FA, L = C.FA, C.L
KEEP = ['book', 'day', 'symbol', 'side', 'sig_m', 'entry_m', 'exit_m', 'entry', 'stop',
        'r_pct', 'R_ps', 'gross', 'net', 'half', 'why', 'pdr', 'rsf']


def fin(d):
    """The common schema, in a fixed column order."""
    for c in KEEP:
        if c not in d.columns:
            d[c] = np.nan
    return d[KEEP].reset_index(drop=True)


def drop_test_tickers(d):
    m = d.symbol.astype(str).str.match(C.TEST_TICKER)
    if m.any():
        C.log(f'  dropped {int(m.sum())} test-ticker rows')
    return d[~m]


def hhmm(s):
    """'HH:MM[:SS]' -> minute of day."""
    p = str(s).split(':')
    return int(p[0]) * 60 + int(p[1])


# ------------------------------------------------------------------ B1 bull flag raw cache
def b1():
    src = 'data/bull_flag_cache_causal_full_20260905.csv'
    c = pd.read_csv(src, keep_default_na=False, na_values=[''], dtype={'symbol': str, 'date': str})
    C.log(f'B1 raw rows {len(c)}')
    c = drop_test_tickers(c.rename(columns={'date': 'day'}))
    c['R_ps'] = c.entry_price - c.stop_loss
    c = c[(c.R_ps > 0) & (c.shares > 0) & c.entry_time_et.astype(str).str.contains(':')
          & c.exit_time_et.astype(str).str.contains(':')].copy()
    c['net'] = c.pnl / (c.R_ps * c.shares)
    c['gross'] = c.net
    c['entry_m'] = [hhmm(x) for x in c.entry_time_et]
    c['exit_m'] = [hhmm(x) for x in c.exit_time_et]
    c['sig_m'] = c.entry_m - 1                       # the breakout bar closes the minute before the fill
    c['entry'] = c.entry_price
    c['stop'] = c.stop_loss
    c['r_pct'] = c.R_ps / c.entry * 100.0
    c['why'] = c.exit_reason.astype(str)
    c['side'] = 1
    c['book'] = 'B1'
    c['half'] = np.nan
    pdr = C.prev_day_range_map()
    c['pdr'] = [pdr.get((s, d), np.nan) for s, d in zip(c.symbol, c.day)]
    c['rsf'] = np.nan                                # filled by l2 from the tape
    C.log(f'B1 scoreable {len(c)}  meanR {c.net.mean():+.4f}')
    return fin(c)


# ------------------------------------------------------------------ B2 F6 first-break hold
def b2():
    sys.path.insert(0, f'{FA}/H/F6_rebuild')
    import book as F6BOOK                                                     # noqa: E402
    tbl = F6BOOK.cost_table()
    d = pd.read_csv(f'{FA}/H/F6_rebuild/scan_ai.csv', keep_default_na=False, na_values=[''],
                    dtype={'symbol': str, 'day': str})
    C.log(f'B2 raw rows {len(d)}')
    d = drop_test_tickers(d).copy()
    S = np.array([tbl[(F6BOOK.price_band(e), F6BOOK.hour_band(m))] for e, m in zip(d.entry, d.entry_min)])
    r_pct = d.R / d.entry * 100.0
    half = 0.5 * S / np.maximum(r_pct, 0.05)
    cost = 0.25 * half + half * d.hold_exit_type.map(F6BOOK.EXIT_K).fillna(0.875).values
    d['half'] = half
    d['net'] = d.hold_grossR - cost
    d['gross'] = d.hold_grossR
    d['entry_m'] = d.entry_min
    d['exit_m'] = d.hold_exit_min
    d['sig_m'] = d.sig_min
    d['r_pct'] = r_pct
    d['R_ps'] = d.R
    d['why'] = d.hold_exit_type
    d['side'] = 1
    d['book'] = 'B2'
    d['pdr'] = np.nan                    # degenerate: the universe is prev_day_range_pct >= 8
    d['rsf'] = np.nan                    # degenerate: the scan applies the >= 5 floor
    C.log(f'B2 scoreable {len(d)}')
    return fin(d)


# ------------------------------------------------------------------ B3 / B4 Stage-C contract
def b34(tag, bookid):
    sys.path.insert(0, f'{FA}/H/F14_F8_F11')
    import hcore as H                                                         # noqa: E402
    d = H.load(tag)
    C.log(f'{bookid} pop rows {len(d)}')
    d = drop_test_tickers(d)
    x = H.scoreable(d, 'hold', floor=True)
    x = x.rename(columns={'next_entry_m': 'entry_m', 'next_entry': 'entry', 'rp': 'r_pct',
                          'prev_day_range_pct': 'pdr', 'range_so_far_pct': 'rsf'})
    x['exit_m'] = x.xm
    x['half'] = 0.5 * (x.spread_cc_bps / 100.0) / x.r_pct.clip(lower=0.05)
    x['R_ps'] = x.entry - x.stop
    x['side'] = 1
    x['book'] = bookid
    C.log(f'{bookid} scoreable {len(x)}')
    return fin(x)


# ------------------------------------------------------------------ B6 S1 gap-fade short
def b6():
    USE = ['day', 'symbol', 'fam', 'cfg', 'sig_m', 'range_so_far_pct', 'spread_cc_bps', 'in_u12',
           'ssr', 'entry', 'entry_m', 'r_pct', 'prev_day_range_pct', 'rr_hold', 'why_hold',
           'exit_m_hold', 'level', 'stop']
    EXIT_RATIO = {'stop': 0.875, 'target': 0.875, 'eod': 0.412, 'cover': 0.412, 'none': 0.875}
    keep, n_in = [], 0
    for ch in pd.read_csv(f'{FA}/G/candidates_short.csv', usecols=USE,
                          dtype={'symbol': str, 'day': str, 'fam': str, 'cfg': str, 'why_hold': str},
                          keep_default_na=False, na_values=[''], chunksize=250_000, low_memory=True):
        n_in += len(ch)
        ok = ((ch.fam == 'S1') & (ch.in_u12 == 1) & (ch.range_so_far_pct >= 5) & (ch.ssr != 1)
              & ch.entry.notna() & (ch.entry >= 10.0) & (ch.entry_m <= 841) & (ch.r_pct >= 1.0)
              & ch.rr_hold.notna() & ch.exit_m_hold.notna())
        if ok.any():
            keep.append(ch.loc[ok].copy())
    d = pd.concat(keep, ignore_index=True)
    C.log(f'B6 read {n_in:,} rows -> S1/UB/hold scoreable {len(d)}')
    d = drop_test_tickers(d).copy()
    half = 0.5 * (d.spread_cc_bps / 100.0) / d.r_pct.clip(lower=0.05)
    d['half'] = half
    d['net'] = d.rr_hold - 0.25 * half - half * d.why_hold.map(EXIT_RATIO).fillna(0.875)
    d['gross'] = d.rr_hold
    d['exit_m'] = d.exit_m_hold
    d['why'] = d.why_hold
    d['R_ps'] = d.entry * d.r_pct / 100.0
    d['side'] = -1
    d['book'] = 'B6'
    d['pdr'] = d.prev_day_range_pct
    d['rsf'] = d.range_so_far_pct
    return fin(d)


ANCHORS = {   # book -> (split -> published base mean net R), and the source of the anchor
    'B1': ({'TRAIN': -0.014, 'VAL': -0.058, 'TEST': -0.088}, 'live_followthrough.md (UN-BOOKED population)'),
    'B2': ({'TRAIN': +0.0626, 'VAL': +0.2066, 'TEST': -0.0420}, 'H/F6_rebuild/REPORT.md §2 hold'),
    'B3': ({'TRAIN': +0.0564, 'VAL': +0.0503}, 'H/F14_F8_F11/REPORT.md F14 hold base'),
    'B4': ({'TRAIN': +0.0079, 'VAL': -0.0107}, 'H/F14_F8_F11/REPORT.md F8 N=30 hold base'),
    'B6': ({'TRAIN': +0.0261, 'VAL': +0.0235}, 'G/REPORT.md S1 UB hold'),
}


def main():
    pops = {'B1': b1, 'B2': b2, 'B3': lambda: b34('F14', 'B3'),
            'B4': lambda: b34('F8N30', 'B4'), 'B6': b6}
    rows, lines = [], ['# Stage L step 1 — populations, base books, anchor parity', '']
    for bid, fn in pops.items():
        p = C.add_split_cols(fn())
        p.to_csv(f'{L}/pop_{bid}.csv', index=False)
        bk = C.book(p)
        bk.to_csv(f'{L}/base_{bid}.csv', index=False)
        anc, src = ANCHORS[bid]
        lines += [f'## {bid} — population {len(p):,} rows, booked {len(bk):,}  (anchor: {src})', '',
                  '| split | pop n | booked n | tr/wk | base mean net R | anchor | delta | t | MDE | week green |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for sp, _lo, _hi in C.SPLITS:
            wk = C.weeks_of(p, sp)
            t = bk[bk.split == sp]
            st = C.stats(t, wk, C.months_in(sp))
            a = anc.get(sp, np.nan)
            # B1's published anchor is the UN-BOOKED population mean
            ref = float(p.loc[p.split == sp, 'net'].mean()) if bid == 'B1' else st['meanR']
            rows.append(dict(book=bid, split=sp, pop_n=int((p.split == sp).sum()), **st))
            lines.append(f"| {sp} | {int((p.split == sp).sum()):,} | {st['n']:,} | {st['tpw']:.1f} | "
                         f"{st['meanR']:+.4f} | {('%+.4f' % a) if a == a else '—'} | "
                         f"{('%+.4f' % (ref - a)) if a == a else '—'} | {st['t']:.2f} | "
                         f"{st['mde']:.3f} | {st['green']:.2f} |")
        lines.append('')
    pd.DataFrame(rows).to_csv(f'{L}/base_stats.csv', index=False)
    open(f'{L}/l1_parity.md', 'w').write('\n'.join(lines))
    C.log('\n'.join(lines))


if __name__ == '__main__':
    main()
