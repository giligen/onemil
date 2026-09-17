"""Stage 2 of the F6 independent rebuild: 1-minute signal + exit walk.

For every daily-prefiltered (symbol, day) this loads the regular-session
1-minute tape and produces, for each of four spec variants
(signal scan start x stop window), the entry, stop, R and the three exit
walks (E-hold, E-2R, E-partial) as gross R with exit minute and exit type.

Variants
  sig 'a' : signal bar searched from 09:31 ET onward   (PRIMARY)
  sig 'b' : the 09:30 bar itself may be the signal
  stp 'i' : stop = lowest low 09:30 .. signal bar INCLUSIVE (PRIMARY)
  stp 'x' : stop = lowest low 09:30 .. bar before the signal bar

Output: scan_<sig><stp>.csv, one row per taken candidate (pre-book).
"""
import os
import sys
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PRE = os.path.join(HERE, 'prefilter_all.csv')
SIP = 'file:research/bf_zero/bars_sip.db?mode=ro'
CACHE = 'file:data/cache.db?mode=ro'

ET = ZoneInfo('America/New_York')
UTC = ZoneInfo('UTC')

RTH_LO, RTH_HI = 570, 959          # 09:30 .. 15:59 ET inclusive (minutes of day)
FLAT_MIN = 955                     # 15:55 ET
LAST_ENTRY_MIN = 841               # 14:01 ET
CAP = 1.006                        # fill must be <= level * CAP
LEVEL_MULT = 1.003
MIN_PRE_RANGE_PCT = 5.0
MIN_R_PCT = 1.0
MIN_PRICE = 5.0
STOP_SLIP = 0.999


def log(*a):
    print(*a)
    sys.stdout.flush()


_off_cache = {}


def utc_offset_min(day):
    """Minutes to SUBTRACT from a UTC minute-of-day to get ET minute-of-day."""
    if day not in _off_cache:
        d = datetime.fromisoformat(day + 'T12:00:00').replace(tzinfo=UTC)
        _off_cache[day] = -int(d.astimezone(ET).utcoffset().total_seconds() // 60)
    return _off_cache[day]


def et_minute(ts, off):
    """ISO UTC timestamp -> ET minute of day."""
    hh = int(ts[11:13]); mm = int(ts[14:16])
    return hh * 60 + mm - off


def load_bars(sip, cache, sym, day):
    """Return RTH 1-min bars for (sym, day) as list of (et_min, o, h, l, c, v)."""
    off = utc_offset_min(day)
    rows = sip.execute('select t,o,h,l,c,v from bars where symbol=? and day=?',
                       (sym, day)).fetchall()
    src = 'sip'
    if not rows:
        rows = cache.execute(
            'select timestamp,open,high,low,close,volume from intraday_bars_1min '
            'where symbol=? and bar_date=?', (sym, day)).fetchall()
        src = 'cache'
    out = []
    for t, o, h, l, c, v in rows:
        m = et_minute(t, off)
        if RTH_LO <= m <= RTH_HI:
            out.append((m, o, h, l, c, v))
    out.sort()
    return out, src


def walk_exits(bars, fi, entry, stop):
    """Walk exits from the bar AFTER the fill bar.  Returns dict of exit rule ->
    (exit_min, exit_type, gross_R, legs) where legs is a list of
    (weight, exit_type) used for the cost model."""
    R = entry - stop
    target = entry + 2.0 * R
    res = {}

    # ---- E-hold ----------------------------------------------------------
    ex_min = ex_type = None
    ex_px = None
    for m, o, h, l, c, v in bars[fi + 1:]:
        if m >= FLAT_MIN:
            ex_min, ex_type, ex_px = m, 'eod', o
            break
        if l <= stop:
            ex_min, ex_type, ex_px = m, 'stop', min(stop, o) * STOP_SLIP
            break
    if ex_min is None:
        if len(bars) > fi + 1:
            m, o, h, l, c, v = bars[-1]
            ex_min, ex_type, ex_px = m, 'eod', c
        else:
            ex_min, ex_type, ex_px = bars[fi][0], 'eod', bars[fi][4]
    res['hold'] = (ex_min, ex_type, (ex_px - entry) / R, [(1.0, ex_type)])

    # ---- E-2R ------------------------------------------------------------
    ex_min = ex_type = None
    ex_px = None
    for m, o, h, l, c, v in bars[fi + 1:]:
        if m >= FLAT_MIN:
            ex_min, ex_type, ex_px = m, 'eod', o
            break
        if l <= stop:
            ex_min, ex_type, ex_px = m, 'stop', min(stop, o) * STOP_SLIP
            break
        if c >= target:
            ex_min, ex_type, ex_px = m, 'target', target
            break
    if ex_min is None:
        if len(bars) > fi + 1:
            m, o, h, l, c, v = bars[-1]
            ex_min, ex_type, ex_px = m, 'eod', c
        else:
            ex_min, ex_type, ex_px = bars[fi][0], 'eod', bars[fi][4]
    res['r2'] = (ex_min, ex_type, (ex_px - entry) / R, [(1.0, ex_type)])

    # ---- E-partial -------------------------------------------------------
    legs = []
    grossR = 0.0
    ex_min = None
    ex_type = None
    part_done = False
    cur_stop = stop
    for m, o, h, l, c, v in bars[fi + 1:]:
        if m >= FLAT_MIN:
            w = 0.5 if part_done else 1.0
            grossR += w * (o - entry) / R
            legs.append((w, 'eod'))
            ex_min, ex_type = m, ('pp+eod' if part_done else 'eod')
            break
        if l <= cur_stop:
            px = min(cur_stop, o) * STOP_SLIP
            w = 0.5 if part_done else 1.0
            grossR += w * (px - entry) / R
            legs.append((w, 'stop'))
            ex_min, ex_type = m, ('pp+stop' if part_done else 'stop')
            break
        if (not part_done) and c >= target:
            grossR += 0.5 * (target - entry) / R
            legs.append((0.5, 'target'))
            part_done = True
            cur_stop = entry
    if ex_min is None:
        if len(bars) > fi + 1:
            m, o, h, l, c, v = bars[-1]
            px = c
        else:
            m, px = bars[fi][0], bars[fi][4]
        w = 0.5 if part_done else 1.0
        grossR += w * (px - entry) / R
        legs.append((w, 'eod'))
        ex_min, ex_type = m, ('pp+eod' if part_done else 'eod')
    res['partial'] = (ex_min, ex_type, grossR, legs)
    return res


def main():
    pre = pd.read_csv(PRE, keep_default_na=False, na_values=[''])
    log('prefiltered symbol-days:', len(pre))

    sip = sqlite3.connect(SIP, uri=True)
    cache = sqlite3.connect(CACHE, uri=True)

    variants = [('a', 'i'), ('a', 'x'), ('b', 'i'), ('b', 'x')]
    rows = {v: [] for v in variants}

    stats = dict(nokey=0, nobars=0, no0930=0, red_bar=0, red_daily=0, red_disagree=0,
                 src_sip=0, src_cache=0, done=0)
    # reason counters for the primary variant
    prim = dict(no_signal=0, range_floor=0, late=0, no_next=0, cap=0, price=0, r_small=0,
                taken=0, sig_is_0930=0, gap_next=0)

    for i, row in enumerate(pre.itertuples(index=False)):
        if i % 5000 == 0:
            log('  %d/%d  taken(primary)=%d' % (i, len(pre), prim['taken']))
        bars, src = load_bars(sip, cache, row.symbol, row.bar_date)
        if not bars:
            stats['nobars'] += 1
            continue
        stats['src_' + src] += 1
        if bars[0][0] != RTH_LO:
            stats['no0930'] += 1
        day_open = bars[0][1]
        red_bar = day_open < row.prev_close
        red_daily = row.open < row.prev_close
        if red_bar != red_daily:
            stats['red_disagree'] += 1
        if red_bar:
            stats['red_bar'] += 1
        if red_daily:
            stats['red_daily'] += 1
        if not red_bar:
            continue

        level = row.prev_close * LEVEL_MULT

        for sig, stp in variants:
            start = 1 if sig == 'a' else 0
            if len(bars) <= start:
                continue
            si = None
            for j in range(start, len(bars)):
                if bars[j][2] >= level:
                    si = j
                    break
            if si is None:
                if (sig, stp) == ('a', 'i'):
                    prim['no_signal'] += 1
                continue
            # range-so-far floor on bars STRICTLY BEFORE the signal bar
            if si == 0:
                ok_range = False
            else:
                hi = max(b[2] for b in bars[:si])
                lo = min(b[3] for b in bars[:si])
                ok_range = lo > 0 and (hi - lo) / lo * 100.0 >= MIN_PRE_RANGE_PCT
            if not ok_range:
                if (sig, stp) == ('a', 'i'):
                    prim['range_floor'] += 1
                continue
            if si + 1 >= len(bars):
                if (sig, stp) == ('a', 'i'):
                    prim['no_next'] += 1
                continue
            fi = si + 1
            fill_min = bars[fi][0]
            if fill_min > LAST_ENTRY_MIN:
                if (sig, stp) == ('a', 'i'):
                    prim['late'] += 1
                continue
            entry = bars[fi][1]
            if entry > level * CAP:
                if (sig, stp) == ('a', 'i'):
                    prim['cap'] += 1
                continue
            if entry < MIN_PRICE:
                if (sig, stp) == ('a', 'i'):
                    prim['price'] += 1
                continue
            upto = si if stp == 'i' else si - 1
            if upto < 0:
                continue
            stop = min(b[3] for b in bars[:upto + 1])
            R = entry - stop
            if R <= 0 or R < MIN_R_PCT / 100.0 * entry:
                if (sig, stp) == ('a', 'i'):
                    prim['r_small'] += 1
                continue
            ex = walk_exits(bars, fi, entry, stop)
            if (sig, stp) == ('a', 'i'):
                prim['taken'] += 1
                if bars[si][0] == RTH_LO:
                    prim['sig_is_0930'] += 1
                if bars[fi][0] != bars[si][0] + 1:
                    prim['gap_next'] += 1
            rec = dict(day=row.bar_date, symbol=row.symbol, src=src,
                       sig_min=bars[si][0], entry_min=fill_min, entry=entry,
                       stop=stop, R=R, level=level,
                       fill_bar_l=bars[fi][3], fill_bar_h=bars[fi][2])
            for k, (em, et, gr, legs) in ex.items():
                rec['%s_exit_min' % k] = em
                rec['%s_exit_type' % k] = et
                rec['%s_grossR' % k] = gr
                rec['%s_legs' % k] = ';'.join('%g:%s' % (w, t) for w, t in legs)
            rows[(sig, stp)].append(rec)
        stats['done'] += 1

    log('stats:', stats)
    log('primary reasons:', prim)
    for v, rs in rows.items():
        df = pd.DataFrame(rs)
        out = os.path.join(HERE, 'scan_%s%s.csv' % v)
        df.to_csv(out, index=False)
        log('wrote', out, len(df))


if __name__ == '__main__':
    main()
