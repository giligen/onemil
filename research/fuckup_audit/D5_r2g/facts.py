"""D5 red-to-green by-hand loser dive — stage 1: per-trade tape facts.

Input : research/fuckup_audit/H/F6_rebuild/trades_hold_ai.csv  (implementation B,
        first-break rule, HOLD exit — the booked 12/day, 4-concurrent book)
        + scan_ai.csv (pre-book candidates: level, sig_min)
        + prefilter_all.csv (daily context: day OHLC, prior day OHLC)
Bars  : research/bf_zero/bars_sip.db  (table bars) with data/cache.db
        intraday_bars_1min as the fallback, exactly as the book's own scan did
        (the `src` column on each trade says which store served it).
ETFs  : research/lit_review_2026/etf_1min.db (SPY, IWM)

Output: trades_facts.csv  (one row per booked trade, every entry-time fact and
        every tape fact needed for the qualitative classes)
        days_facts.csv    (one row per trading day in the book)

Read-only on every store.  ZVZZT / Z?ZZT test tickers dropped.
"""
import os
import re
import sys
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REB = os.path.join(HERE, '..', 'H', 'F6_rebuild')
SIP = 'file:research/bf_zero/bars_sip.db?mode=ro'
CACHE = 'file:data/cache.db?mode=ro'
ETF = 'file:research/lit_review_2026/etf_1min.db?mode=ro'

ET = ZoneInfo('America/New_York')
UTC = ZoneInfo('UTC')
RTH_LO, RTH_HI = 570, 959
TEST_TICKER = re.compile(r'^Z[A-Z]ZZT$')

VAL_LO, VAL_HI = '2026-01-01', '2026-05-31'
TEST_LO = '2026-06-01'


def log(*a):
    print(*a)
    sys.stdout.flush()


_off = {}


def utc_offset_min(day):
    if day not in _off:
        d = datetime.fromisoformat(day + 'T12:00:00').replace(tzinfo=UTC)
        _off[day] = -int(d.astimezone(ET).utcoffset().total_seconds() // 60)
    return _off[day]


def et_minute(ts, off):
    return int(ts[11:13]) * 60 + int(ts[14:16]) - off


def split_of(day):
    if day < VAL_LO:
        return 'TRAIN'
    if day <= VAL_HI:
        return 'VAL'
    return 'TEST'


# ---------------------------------------------------------------- ETF tape
def load_etf(sym):
    """day -> (open570, close_by_minute array indexed minute-570, last_min)."""
    con = sqlite3.connect(ETF, uri=True)
    rows = con.execute(
        "select t,o,h,l,c from bars where symbol=? and t>='2025-01-01' "
        "and t<'2026-09-15' and substr(t,12,2) between '12' and '21'",
        (sym,)).fetchall()
    con.close()
    out = {}
    for t, o, h, l, c in rows:
        day = t[:10]
        off = utc_offset_min(day)
        m = et_minute(t, off)
        if not (RTH_LO <= m <= RTH_HI):
            continue
        d = out.get(day)
        if d is None:
            d = out[day] = {'o': np.full(390, np.nan), 'c': np.full(390, np.nan)}
        d['o'][m - RTH_LO] = o
        d['c'][m - RTH_LO] = c
    log(f'  ETF {sym}: {len(rows)} raw rows -> {len(out)} RTH days')
    return out


def px_at(arr, m):
    """Last non-nan value at or before ET minute m."""
    i = m - RTH_LO
    if i < 0:
        return np.nan
    i = min(i, 389)
    seg = arr[:i + 1]
    ok = ~np.isnan(seg)
    if not ok.any():
        return np.nan
    return float(seg[ok][-1])


def first_px(arr):
    ok = ~np.isnan(arr)
    return float(arr[ok][0]) if ok.any() else np.nan


def last_px(arr):
    ok = ~np.isnan(arr)
    return float(arr[ok][-1]) if ok.any() else np.nan


# ---------------------------------------------------------------- stock tape
def load_bars(sip, cache, sym, day, src):
    off = utc_offset_min(day)
    if src == 'sip':
        rows = sip.execute('select t,o,h,l,c,v from bars where symbol=? and day=?',
                           (sym, day)).fetchall()
    else:
        rows = cache.execute(
            'select timestamp,open,high,low,close,volume from intraday_bars_1min '
            'where symbol=? and bar_date=?', (sym, day)).fetchall()
    out = []
    for t, o, h, l, c, v in rows:
        m = et_minute(t, off)
        if RTH_LO <= m <= RTH_HI:
            out.append((m, o, h, l, c, v))
    out.sort()
    return out


def main():
    log('reading book...')
    tr = pd.read_csv(os.path.join(REB, 'trades_hold_ai.csv'), keep_default_na=False)
    n0 = len(tr)
    tr = tr[~tr['symbol'].str.match(TEST_TICKER)].copy()
    log(f'  trades {n0} -> {len(tr)} after dropping Z?ZZT '
        f'({n0 - len(tr)} test-ticker rows)')

    for c in ('entry', 'stop', 'R', 'gross_R', 'net_R', 'half'):
        tr[c] = tr[c].astype(float)
    for c in ('entry_min', 'exit_min', 'sig_min'):
        tr[c] = tr[c].astype(int)

    sc = pd.read_csv(os.path.join(REB, 'scan_ai.csv'), keep_default_na=False)
    sc = sc[~sc['symbol'].str.match(TEST_TICKER)].copy()
    lvl = {(r.day, r.symbol): float(r.level) for r in sc.itertuples()}
    cand_per_day = sc.groupby('day').size().to_dict()

    pf = pd.read_csv(os.path.join(REB, 'prefilter_all.csv'), keep_default_na=False)
    pfm = {}
    for r in pf.itertuples():
        pfm[(r.bar_date, r.symbol)] = (
            float(r.open), float(r.high), float(r.low), float(r.close),
            float(r.volume) if r.volume != '' else np.nan,
            float(r.adv20) if r.adv20 != '' else np.nan,
            float(r.prev_close), float(r.prev_range_pct))
    del pf
    log(f'  scan candidates {len(sc)}, prefilter rows {len(pfm)}')

    log('loading SPY / IWM...')
    spy = load_etf('SPY')
    iwm = load_etf('IWM')

    sip = sqlite3.connect(SIP, uri=True)
    cache = sqlite3.connect(CACHE, uri=True)

    tr = tr.sort_values(['day', 'entry_min', 'symbol']).reset_index(drop=True)
    tr['seq'] = tr.groupby('day').cumcount() + 1
    ntd = tr.groupby('day').size().to_dict()

    rows = []
    for i, r in enumerate(tr.itertuples()):
        if i % 250 == 0:
            log(f'  trade {i}/{len(tr)}')
        day, sym = r.day, r.symbol
        bars = load_bars(sip, cache, sym, day, r.src)
        mins = [b[0] for b in bars]
        if not bars:
            continue
        try:
            fi = mins.index(r.entry_min)
        except ValueError:
            fi = min(range(len(mins)), key=lambda k: abs(mins[k] - r.entry_min))
        try:
            xi = mins.index(r.exit_min)
        except ValueError:
            xi = min(range(len(mins)), key=lambda k: abs(mins[k] - r.exit_min))

        day_open = bars[0][1]
        pfk = pfm.get((day, sym))
        prev_close = pfk[6] if pfk else np.nan
        prev_range = pfk[7] if pfk else np.nan
        adv20 = pfk[5] if pfk else np.nan
        day_close = pfk[3] if pfk else bars[-1][4]
        level = lvl.get((day, sym), np.nan)

        R = r.R
        entry = r.entry

        # ---- entry-time facts (all computable at the fill)
        gap_pct = (day_open - prev_close) / prev_close * 100 if prev_close else np.nan
        e_vs_pc = (entry - prev_close) / prev_close * 100 if prev_close else np.nan
        o2e = (entry - day_open) / day_open * 100 if day_open else np.nan
        stop_dist_pct = R / entry * 100

        pre = bars[max(0, fi - 5):fi]
        dv5 = sum(b[4] * b[5] for b in pre)
        vol5 = sum(b[5] for b in pre)
        si = mins.index(r.sig_min) if r.sig_min in mins else max(0, fi - 1)
        run_hi = max(b[2] for b in bars[:si + 1])
        run_lo = min(b[3] for b in bars[:si + 1])
        rsf = (run_hi - run_lo) / run_lo * 100 if run_lo else np.nan

        # ---- tape facts after the fill
        post = bars[fi + 1:xi + 1]
        if post:
            mfe_px = max(b[2] for b in post)
            mae_px = min(b[3] for b in post)
            mfe_min = [b[0] for b in post if b[2] == mfe_px][0]
        else:
            mfe_px, mae_px, mfe_min = entry, entry, r.entry_min
        mfe_R = (mfe_px - entry) / R
        mae_R = (mae_px - entry) / R
        hold_min = r.exit_min - r.entry_min

        post_hi = max((b[2] for b in bars[fi + 1:]), default=entry)
        level_fail = bool(post_hi <= level) if level == level else False
        never_green = bool(post_hi <= entry)

        # halt / news markers between the fill and the exit
        win = bars[fi:xi + 1]
        max_bar_range = max(((b[2] - b[3]) / b[3] * 100) for b in win if b[3] > 0) \
            if win else 0.0
        wm = [b[0] for b in win]
        biggest_gap = 0
        for a, b in zip(wm, wm[1:]):
            biggest_gap = max(biggest_gap, b - a - 1)

        # wick stop
        wick = False
        recovered = False
        if r.exit_type == 'stop':
            xb = bars[xi]
            wick = xb[4] > r.stop
            for b in bars[xi + 1:]:
                if b[0] > r.exit_min + 30:
                    break
                if b[2] > entry:
                    recovered = True
                    break

        # market context
        sd, idd = spy.get(day), iwm.get(day)
        def etf_ret(d, m0, m1):
            if d is None:
                return np.nan
            a, b = px_at(d['c'], m0), px_at(d['c'], m1)
            if a != a or b != b or a == 0:
                return np.nan
            return (b - a) / a * 100
        spy_h1 = etf_ret(sd, r.entry_min, min(r.entry_min + 60, RTH_HI))
        iwm_h1 = etf_ret(idd, r.entry_min, min(r.entry_min + 60, RTH_HI))
        spy_o2e = ((px_at(sd['c'], r.entry_min) - first_px(sd['o'])) /
                   first_px(sd['o']) * 100) if sd else np.nan
        iwm_o2e = ((px_at(idd['c'], r.entry_min) - first_px(idd['o'])) /
                   first_px(idd['o']) * 100) if idd else np.nan

        rows.append(dict(
            split=split_of(day), day=day, symbol=sym, seq=r.seq,
            n_trades_day=ntd[day], dow=datetime.fromisoformat(day).strftime('%a'),
            entry_min=r.entry_min, sig_min=r.sig_min, exit_min=r.exit_min,
            hold_min=hold_min, exit_type=r.exit_type,
            entry=entry, stop=r.stop, R=R, level=level,
            prev_close=prev_close, day_open=day_open, day_close=day_close,
            gross_R=r.gross_R, net_R=r.net_R,
            gap_pct=gap_pct, prev_range_pct=prev_range,
            entry_vs_prev_close_pct=e_vs_pc, open_to_entry_pct=o2e,
            stop_dist_pct=stop_dist_pct, range_so_far_pct=rsf,
            dollar_vol_5m=dv5, vol_5m=vol5, adv20=adv20,
            mfe_R=mfe_R, mae_R=mae_R, mfe_min=mfe_min,
            post_high=post_hi, level_fail=int(level_fail),
            never_green=int(never_green),
            max_bar_range_pct=max_bar_range, biggest_bar_gap=biggest_gap,
            wick=int(wick), recovered_30m=int(recovered),
            spy_ret_1h=spy_h1, iwm_ret_1h=iwm_h1,
            spy_open_to_entry=spy_o2e, iwm_open_to_entry=iwm_o2e,
            src=r.src, n_bars=len(bars),
        ))

    sip.close()
    cache.close()
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(HERE, 'trades_facts.csv'), index=False)
    log(f'wrote trades_facts.csv  n={len(out)}')

    # ------------------------------------------------------------ day table
    drows = []
    for day, g in out.groupby('day'):
        sd, idd = spy.get(day), iwm.get(day)
        def oc(d):
            if d is None:
                return (np.nan, np.nan)
            o = first_px(d['o'])
            return ((last_px(d['c']) - o) / o * 100,
                    (px_at(d['c'], 600) - o) / o * 100)
        s_oc, s_o10 = oc(sd)
        i_oc, i_o10 = oc(idd)
        drows.append(dict(
            day=day, split=split_of(day),
            book_R=g['net_R'].sum(), n=len(g),
            wins=int((g['net_R'] > 0).sum()),
            spy_oc=s_oc, spy_o10=s_o10, iwm_oc=i_oc, iwm_o10=i_o10,
            candidates=cand_per_day.get(day, 0),
        ))
    dd = pd.DataFrame(drows).sort_values('day')
    dd.to_csv(os.path.join(HERE, 'days_facts.csv'), index=False)
    log(f'wrote days_facts.csv  n={len(dd)}')


if __name__ == '__main__':
    main()
