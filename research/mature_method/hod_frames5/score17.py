#!/usr/bin/env python3
"""hod_frames5 / F17 — THE HORIZON: what the signal is worth after the bell.

Cells exactly as declared in PREREG.md §5 (committed 7144d3c, before any cell was scored).
Entry, stop, target and every intraday rule are UNCHANGED; the only change is what happens to a
position still open at 15:55.  Multi-day P&L comes from `cache.db daily_bars` for the trade's OWN
symbol (so a wrapper's daily-rebalancing decay is in the price by construction — verified, not
assumed, by F17-D2).  TEST sealed.  Read-only.  One process.
"""
import sqlite3, sys
import numpy as np, pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/mature_method/hod_frames5')
from common5 import (ROOT, D5, S, S2, SPLITS, RISK, load_breaks4, sigset5, admit,   # noqa: E402
                     book_ranked, attach_instrument, clustered_t, halves, Sheet)

TOL = 0.005          # price-scale tolerance: the entry-day daily bar must bracket the entry


def daily_panel(symbols):
    con = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=120)
    out = []
    syms = sorted(set(symbols))
    for i in range(0, len(syms), 400):
        ch = syms[i:i + 400]
        q = ('select symbol, bar_date, open, high, low, close from daily_bars where symbol in (%s)'
             % ','.join('?' * len(ch)))
        out.append(pd.read_sql(q, con, params=ch))
    con.close()
    d = pd.concat(out, ignore_index=True)
    d['symbol'] = d.symbol.astype(str); d['bar_date'] = d.bar_date.astype(str)
    d = d.sort_values(['symbol', 'bar_date'], kind='mergesort').reset_index(drop=True)
    d['k'] = d.groupby('symbol').cumcount()
    return d


def horizon(b, dp, h, stop_arm):
    """Re-write the `eod` exits of `b` at horizon `h`.

    h = 0  -> exit at the NEXT session's OPEN
    h >= 1 -> exit at the close of the h-th session after the entry day
    `stop_arm`: False -> the position rides with no stop overnight.
                True  -> the PRIOR session's CLOSE is the stop; if a session's low <= it the exit
                         is there, and if the session's OPEN gaps below it the exit is the OPEN
                         (the gap is charged, never netted).
    Returns a copy of `b` with `rr`, `net`, `pnl`, `days_held` re-written for carried rows.
    """
    idx = {(r.symbol, r.bar_date): (r.k, r.open, r.high, r.low, r.close)
           for r in dp.itertuples()}
    bysym = {}
    for r in dp.itertuples():
        bysym.setdefault(r.symbol, []).append((r.bar_date, r.open, r.high, r.low, r.close))
    o = b.copy()
    rr = o.rr.values.astype(float).copy()
    days = np.zeros(len(o))
    carried = (o.why == 'eod').values
    dropped = 0
    for i, r in enumerate(o.itertuples()):
        if not carried[i]:
            continue
        key = (r.symbol, r.day)
        if key not in idx:
            rr[i] = np.nan; dropped += 1; continue
        k0, o0, h0, l0, c0 = idx[key]
        # ---- price-scale check (CLAUDE.md rule 3): the daily bar must bracket the intraday entry
        if not (l0 * (1 - TOL) <= r.price <= h0 * (1 + TOL)):
            rr[i] = np.nan; dropped += 1; continue
        seq = bysym[r.symbol]
        rps = r.price - r.stop                       # R per share, the trade's own risk
        if rps <= 0:
            rr[i] = np.nan; dropped += 1; continue
        exit_px = None; nd = 0; prior_close = c0
        if h == 0:                                    # the next session's OPEN
            if k0 + 1 >= len(seq):
                rr[i] = np.nan; dropped += 1; continue
            exit_px = seq[k0 + 1][1]; nd = 1
        else:
            for j in range(1, h + 1):
                if k0 + j >= len(seq):
                    break
                _, oj, hj, lj, cj = seq[k0 + j]
                nd = j
                if stop_arm:
                    if oj <= prior_close:             # the gap is the fill, never the level
                        exit_px = oj; break
                    if lj <= prior_close:
                        exit_px = prior_close; break
                prior_close = cj
                if j == h:
                    exit_px = cj
            if exit_px is None:
                if k0 + 1 >= len(seq):
                    rr[i] = np.nan; dropped += 1; continue
                exit_px = seq[min(k0 + h, len(seq) - 1)][4]
        rr[i] = (exit_px - r.price) / rps
        days[i] = nd
    o['rr'] = rr
    o['days_held'] = days
    # the carried exit pays a FULL half-spread (ratio 1.0 — a marketable exit at an open/close);
    # the shipped eod exit pays 0.412.  Uncarried rows keep the shipped charge.
    half = 0.5 * o.sp_pct / o.r_pct.clip(lower=0.05)
    ratio = o.why.map(S.RATIO).fillna(0.875)
    ratio = np.where(carried, 1.0, ratio)
    o['net'] = o.rr - half - half * ratio
    o['pnl'] = o.net * RISK
    n_drop = int(np.isnan(rr).sum())
    o = o[o.rr.notna()].copy()
    return o, n_drop, int(carried.sum())


def leverage_of(name):
    """Stated leverage and direction parsed from the Alpaca asset name ('2X Long TSLA')."""
    if not isinstance(name, str):
        return np.nan
    n = name.upper()
    lev = np.nan
    for k, v in (('3X', 3.0), ('3.0X', 3.0), ('2X', 2.0), ('2.0X', 2.0), ('1.5X', 1.5),
                 ('1.25X', 1.25), ('1X', 1.0)):
        if k in n:
            lev = v; break
    if lev != lev:
        lev = 1.0 if ('LONG' in n or 'SHORT' in n or 'BULL' in n or 'BEAR' in n) else np.nan
    if any(w in n for w in ('SHORT', 'INVERSE', 'BEAR', '-1X')):
        lev = -abs(lev)
    return lev


def main():
    br = load_breaks4()
    S.build_impute(S2.load_pop())
    s = sigset5(admit(br, pd.Series(True, index=br.index)))
    s = attach_instrument(s)
    b0 = book_ranked(s, 12, 4)
    for sp in SPLITS:
        w = S.week_stats(b0, sp)
        print(f'   B2 {sp:5s} n {w["n"]:5d} gross {w["gross"]:+.3f} net {w["net"]:+.3f} '
              f'total ${w["total"]:+,.0f}', flush=True)
    print(f'\n   exit-reason mix of the booked set: {dict(b0.why.value_counts())}', flush=True)
    print(f'   CARRIED (why == eod) share: TRAIN '
          f'{(b0[b0.split=="TRAIN"].why=="eod").mean():.1%} | VAL '
          f'{(b0[b0.split=="VAL"].why=="eod").mean():.1%}', flush=True)

    dp = daily_panel(b0.symbol.unique())
    print(f'   daily panel: {len(dp)} rows, {dp.symbol.nunique()} symbols, '
          f'{dp.bar_date.min()} -> {dp.bar_date.max()}', flush=True)

    sh = Sheet()
    print('\n== F17 CELLS ==', flush=True)
    sh.show('F17-h0  15:55 flat [shipped, =B2]', b0)
    drops = {}
    for h, lab in ((0, 'h1'), (1, 'h2'), (2, 'h3'), (5, 'h5')):
        for arm, sa in (('a', False), ('b', True)):
            o, nd, nc = horizon(b0, dp, h, sa)
            nm = (f'F17-{lab}{arm}  '
                  + ('next OPEN' if h == 0 else f'+{h} session close')
                  + ('' if arm == 'a' else ' , prior-close stop'))
            drops[nm] = (nd, nc)
            sh.show(nm, o)
    print('\n   price-scale / coverage drops (carried rows removed, PREREG rail 10):', flush=True)
    for k, (nd, nc) in drops.items():
        print(f'     {k}: dropped {nd} of {nc} carried rows ({nd/max(nc,1):.1%})', flush=True)

    # ------------------------------------------------------------------ per unit of TIME
    print('\n== net R per trade AND per calendar day held ==', flush=True)
    print('| cell | split | n | net R | mean days held | net R per day | gross R |')
    print('|' + '|'.join(['-' * 6] * 7) + '|')
    for name, b in sh.books.items():
        for sp in SPLITS:
            d = b[b.split == sp]
            if not len(d):
                continue
            dh = float(d.days_held.mean()) if 'days_held' in d else 0.0
            per = d.net.mean() / max(dh, 1.0)
            print(f'| {name} | {sp} | {len(d)} | {d.net.mean():+.3f} | {dh:.2f} | {per:+.4f} | '
                  f'{d.rr.mean():+.3f} |', flush=True)

    # ------------------------------------------------------------------ F17-D1 two cohorts
    print('\n== F17-D1  the two-cohort diagnostic (day_range_pct >= 10 % vs the rest) ==', flush=True)
    print('| cell | split | mover n | mover gross | mover net | quiet n | quiet gross | quiet net |')
    print('|' + '|'.join(['-' * 6] * 8) + '|')
    for name, b in sh.books.items():
        for sp in SPLITS:
            d = b[b.split == sp]
            if not len(d):
                continue
            mv = d[d.day_range_pct >= 10]; qt = d[d.day_range_pct < 10]
            print(f'| {name} | {sp} | {len(mv)} | {mv.rr.mean():+.3f} | {mv.net.mean():+.3f} | '
                  f'{len(qt)} | {qt.rr.mean():+.3f} | {qt.net.mean():+.3f} |', flush=True)

    # ------------------------------------------------------------------ F17-D2 wrapper decay
    print('\n== F17-D2  the WRAPPER DECAY, measured (not assumed) ==', flush=True)
    ass = pd.read_csv(f'{ROOT}/data/research/alpaca_assets_all_20260905.csv', dtype=str,
                      keep_default_na=False, na_values=[''])
    nm = dict(zip(ass.symbol, ass.name))
    wr = b0[b0.asset_class == 'wrapper'][['symbol', 'anchor']].drop_duplicates()
    wr['lev'] = wr.symbol.map(lambda x: leverage_of(nm.get(x)))
    wr = wr[wr.lev.notna() & wr.anchor.notna()]
    need = sorted(set(wr.symbol) | set(wr.anchor))
    dpx = daily_panel(need)
    px = {(r.symbol, r.bar_date): r.close for r in dpx.itertuples()}
    ser = {}
    for sym, g in dpx.groupby('symbol'):
        ser[sym] = g[['bar_date', 'close']].values.tolist()
    rows = []
    for r in wr.itertuples():
        a = ser.get(r.symbol); u = ser.get(r.anchor)
        if not a or not u:
            continue
        ud = dict(u)
        prev_a = prev_u = None
        for dt, c in a:
            cu = ud.get(dt)
            if cu is None:
                prev_a = prev_u = None; continue
            if prev_a and prev_u and prev_a > 0 and prev_u > 0 and c > 0 and cu > 0:
                ra = np.log(c / prev_a); ru = np.log(cu / prev_u)
                rows.append(dict(symbol=r.symbol, anchor=r.anchor, lev=r.lev,
                                 drag=ra - r.lev * ru))
            prev_a, prev_u = c, cu
    dd = pd.DataFrame(rows)
    if len(dd):
        print(f'   wrapper-days measured: {len(dd)} over {dd.symbol.nunique()} wrappers')
        print(f'   mean daily drag (wrapper log-ret MINUS lev x underlying log-ret): '
              f'{dd.drag.mean()*100:+.4f} %/day  (median {dd.drag.median()*100:+.4f}, '
              f'sd {dd.drag.std()*100:.3f})')
        for L, g in dd.groupby('lev'):
            print(f'     lev {L:+.1f}: n {len(g)}  mean drag {g.drag.mean()*100:+.4f} %/day')
        med_r = float(b0[b0.asset_class == "wrapper"].r_pct.median())
        print(f'   median R of a booked wrapper: {med_r:.2f} % of price -> the measured drag is '
              f'{abs(dd.drag.mean())*100/med_r:.3f} R per calendar day held.')
        print('   NOTE: every multi-day cell above is priced from the WRAPPER\'S OWN daily bars, '
              'so this drag is ALREADY inside those numbers; it is measured here to prove it is '
              'not being netted out.')
    else:
        print('   no wrapper/underlying pair had a usable daily panel — reported, not assumed.')

    sh.nulls(f'{D5}/nulls17.csv')
    sh.dump(f'{D5}/cells17.csv')

    print('\n== MDE and the tail ==', flush=True)
    print('| cell | split | n | net | MDE | ex-top5% | ex-top1% |')
    print('|' + '|'.join(['-' * 6] * 7) + '|')
    for name, b in sh.books.items():
        for sp in SPLITS:
            d = b[b.split == sp]
            if len(d) < 5:
                continue
            print(f'| {name} | {sp} | {len(d)} | {d.net.mean():+.3f} | '
                  f'{2.80*float(d.net.std(ddof=1)/np.sqrt(len(d))):.3f} | '
                  f'{d.net[d.net <= d.net.quantile(0.95)].mean():+.3f} | '
                  f'{d.net[d.net <= d.net.quantile(0.99)].mean():+.3f} |', flush=True)

    # overnight risk, stated
    print('\n== overnight risk actually taken (carried rows only) ==', flush=True)
    for name in [k for k in sh.books if k.startswith('F17-h') and not k.startswith('F17-h0')]:
        b = sh.books[name]
        c = b[b.why == 'eod']
        if not len(c):
            continue
        print(f'   {name}: carried {len(c)} | worst single trade {c.rr.min():+.2f} R | '
              f'best {c.rr.max():+.2f} R | share worse than -1R (the intraday stop) '
              f'{(c.rr < -1).mean():.1%}', flush=True)
    print('\nDONE F17', flush=True)


if __name__ == '__main__':
    main()
