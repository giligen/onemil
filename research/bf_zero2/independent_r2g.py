#!/usr/bin/env python3
"""Independent re-implementation of the "red to green" simulation, written from the
written specification only (no reference to build_candidates2/score2/score3/verify_f6).

Spec implemented here, verbatim:

  Universe   : every row of research/bf_zero/universe.csv (point-in-time, delisted
               included).  prev_close = the previous trading day's close for that
               symbol from data/research/databento/equs_daily_2025_2026.parquet.
  Bars       : 1-min SIP, RTH only (ET minute 570..959 = 09:30..15:59).
  Setup      : bar-0 open (o0) strictly below prev_close.
  Entry      : first bar i>=1 with high >= prev_close*1.003; entry = prev_close*1.003.
  Stop       : min low of bars 0..i-1; require stop < entry; R = entry-stop;
               require R/entry >= 1%.
  Causal     : (max high 0..i - min low 0..i)/o0 >= 5%.
  Filters    : entry >= $5 ; entry bar minute <= 841.
  Exit A     : bars i+1.. in order:  m>=955 -> open ('eod')
                                     low<=stop -> min(stop, open)*0.999 ('stop')
                                     close>=entry+2R -> entry+2R ('target')
               fallback: last bar close ('end').
  Exit B     : same, no target leg.
  rr         : (exit-entry)/R, minus cost 0.20/r_pct (r_pct = R/entry*100) unless
               the exit reason is 'target' (resting limit -> zero cost).
  Book       : per day sort by (entry minute, symbol); first-come, <=4 concurrent
               (a slot frees only when a prior trade's exit minute is STRICTLY
               before the new entry minute), <=4 fills/day.
  Splits     : TRAIN < 2026-01-01, VAL 2026-01-01..2026-05-31, TEST >= 2026-06-01.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.environ.setdefault('BFZ_SIP_STORE', f'{ROOT}/research/bf_zero/bars_sip.db')
sys.path.insert(0, f'{ROOT}/research/bf_zero')

UNIVERSE = f'{ROOT}/research/bf_zero/universe.csv'
DAILY_PARQUET = f'{ROOT}/data/research/databento/equs_daily_2025_2026.parquet'
OUT_DIR = f'{ROOT}/research/bf_zero2'
TAG = os.environ.get('IR2G_TAG', '')
OUT_TRADES = {'A': f'{OUT_DIR}/independent_trades_A{TAG}.csv',
              'B': f'{OUT_DIR}/independent_trades_B{TAG}.csv'}

# --- reconciliation knobs (default = the spec exactly as handed to me) ---
# IR2G_PRICE_ON=o0     : apply the $5 floor to the day's first RTH open instead of
#                        the entry price (this is what the reference implementation
#                        demonstrably does -- 0/1676 of its trades have o0 < $5).
# IR2G_RANGE_EPS=1e-9  : relative tolerance on the >=5% causal-range test, so that
#                        symbol-days sitting EXACTLY on 5.000% are not lost to binary
#                        floating point (e.g. (8.35-7.95)/8.00 == 0.04999999999999993).
PRICE_ON = os.environ.get('IR2G_PRICE_ON', 'entry')
RANGE_EPS = float(os.environ.get('IR2G_RANGE_EPS', '0'))

TRIGGER_MULT = 1.003     # 0.3% slippage through the level
MIN_R_PCT = 1.0          # R/entry >= 1%
MIN_RANGE_PCT = 5.0      # causal universe guarantee
MIN_PRICE = 5.0
MAX_ENTRY_MIN = 841      # 14:01
EOD_MIN = 955            # 15:55
RTH_LO, RTH_HI = 570, 959
STOP_SLIP = 0.999        # 10 bps through the stop
COST_PCT_HALF = 0.20     # 0.5 * 0.40% expressed in "percent of price"
TARGET_R = 2.0
MAX_CONCURRENT = 4
MAX_FILLS_DAY = 4

COLS = ['bar_date', 'symbol', 'prev_close', 'o0', 'entry_m', 'entry', 'stop',
        'r_pct', 'exit_m', 'exit', 'reason', 'rr_gross', 'cost_r', 'rr', 'in_book']


def log(*a):
    print(*a, flush=True)


def build_prev_close(u):
    """prev_close per (symbol, bar_date) from the databento daily parquet."""
    log('reading daily parquet ...')
    d = pd.read_parquet(DAILY_PARQUET, columns=['symbol', 'bar_date', 'close'])
    log(f'  daily rows={len(d):,}')
    d = d.sort_values(['symbol', 'bar_date'], kind='mergesort')
    d['prev_close'] = d.groupby('symbol', sort=False)['close'].shift(1)
    d = d.dropna(subset=['prev_close'])[['symbol', 'bar_date', 'prev_close']]
    out = u.merge(d, on=['symbol', 'bar_date'], how='left')
    log(f'  universe rows with prev_close: {out.prev_close.notna().sum():,} / {len(out):,}')
    return out


def simulate(m, o, h, l, c, prev_close, want_target):
    """Return a trade dict or None for one symbol-day."""
    n = len(m)
    if n < 2:
        return None
    o0 = o[0]
    if not (o0 < prev_close):
        return None
    entry = prev_close * TRIGGER_MULT
    # first bar i>=1 whose high reaches the level
    hit = np.flatnonzero(h[1:] >= entry)
    if not len(hit):
        return None
    i = int(hit[0]) + 1
    if m[i] > MAX_ENTRY_MIN:
        return None
    if (o0 if PRICE_ON == 'o0' else entry) < MIN_PRICE:
        return None
    stop = float(l[:i].min())
    if not (stop < entry):
        return None
    R = entry - stop
    r_pct = R / entry * 100.0
    if r_pct < MIN_R_PCT:
        return None
    rng = (float(h[:i + 1].max()) - float(l[:i + 1].min())) / o0 * 100.0
    if rng < MIN_RANGE_PCT * (1 - RANGE_EPS):
        return None

    target = entry + TARGET_R * R
    exit_px = exit_m = None
    reason = None
    for j in range(i + 1, n):
        if m[j] >= EOD_MIN:
            exit_px, exit_m, reason = float(o[j]), int(m[j]), 'eod'
            break
        if l[j] <= stop:
            exit_px, exit_m, reason = min(stop, float(o[j])) * STOP_SLIP, int(m[j]), 'stop'
            break
        if want_target and c[j] >= target:
            exit_px, exit_m, reason = float(target), int(m[j]), 'target'
            break
    if exit_px is None:
        exit_px, exit_m, reason = float(c[n - 1]), int(m[n - 1]), 'end'

    rr_gross = (exit_px - entry) / R
    cost_r = 0.0 if reason == 'target' else COST_PCT_HALF / r_pct
    return dict(prev_close=float(prev_close), o0=float(o0), entry_m=int(m[i]),
                entry=float(entry), stop=float(stop), r_pct=float(r_pct),
                exit_m=exit_m, exit=float(exit_px), reason=reason,
                rr_gross=float(rr_gross), cost_r=float(cost_r),
                rr=float(rr_gross - cost_r))


def pick_book(trades):
    """first-come, <=4 concurrent (slot frees when exit_m < new entry_m), <=4 fills."""
    trades = sorted(trades, key=lambda t: (t['entry_m'], t['symbol']))
    open_exits = []
    taken = 0
    for t in trades:
        open_exits = [x for x in open_exits if not (x < t['entry_m'])]
        if len(open_exits) >= MAX_CONCURRENT:
            continue
        t['in_book'] = 1
        open_exits.append(t['exit_m'])
        taken += 1
        if taken >= MAX_FILLS_DAY:
            break
    return trades


def main():
    import build_candidates as bc  # only load_bars is used

    # keep_default_na=False: the ticker "NA" must not be parsed as a missing value
    # (165 universe rows; the reference book contains 3 such trades).
    u = pd.read_csv(UNIVERSE, usecols=['symbol', 'bar_date'],
                    keep_default_na=False, na_values=[''])
    u = build_prev_close(u).dropna(subset=['prev_close'])
    days = sorted(u.bar_date.unique())
    if os.environ.get('IR2G_MAX_DAYS'):
        days = days[:int(os.environ['IR2G_MAX_DAYS'])]
    log(f'days={len(days)}  rows={len(u):,}')

    handles = {}
    for v, p in OUT_TRADES.items():
        f = open(p, 'w')
        f.write(','.join(COLS) + '\n')
        handles[v] = f

    t_start = time.time()
    counts = {'A': 0, 'B': 0}
    for di, day in enumerate(days):
        sub = u[u.bar_date == day]
        syms = sorted(sub.symbol.unique())
        pc = dict(zip(sub.symbol, sub.prev_close))
        try:
            bars = bc.load_bars(day, syms)
        except Exception as e:                       # noqa: BLE001
            log(f'{day} LOAD FAIL {e!r}')
            continue
        per_variant = {'A': [], 'B': []}
        for s, gg in bars.items():
            if s not in pc:
                continue
            gg = gg[(gg.m >= RTH_LO) & (gg.m <= RTH_HI)]
            if len(gg) < 2:
                continue
            m = gg.m.values.astype(np.int64)
            o = gg.o.values.astype(np.float64)
            h = gg.h.values.astype(np.float64)
            l = gg.l.values.astype(np.float64)
            c = gg.c.values.astype(np.float64)
            if not np.isfinite(o[0]):
                continue
            for v, want_t in (('A', True), ('B', False)):
                r = simulate(m, o, h, l, c, pc[s], want_t)
                if r is not None:
                    r['bar_date'] = day
                    r['symbol'] = s
                    r['in_book'] = 0
                    per_variant[v].append(r)
        for v, rows in per_variant.items():
            rows = pick_book(rows)
            counts[v] += len(rows)
            f = handles[v]
            for r in rows:
                f.write(','.join(str(r[k]) for k in COLS) + '\n')
            f.flush()
        if di % 10 == 0 or di == len(days) - 1:
            el = time.time() - t_start
            log(f'[{di + 1}/{len(days)}] {day} syms={len(syms)} bars={len(bars)} '
                f'candA={counts["A"]} candB={counts["B"]} elapsed={el / 60:.1f}m')
    for f in handles.values():
        f.close()
    log('candidate generation done')
    report()


# ------------------------------- reporting -------------------------------
def split_of(d):
    if d < '2026-01-01':
        return 'TRAIN'
    if d <= '2026-05-31':
        return 'VAL'
    return 'TEST'


def stats(df):
    n = len(df)
    if n == 0:
        return dict(n=0)
    r = df.rr.values
    mean = r.mean()
    sd = r.std(ddof=1) if n > 1 else float('nan')
    t = mean / (sd / np.sqrt(n)) if n > 1 and sd > 0 else float('nan')
    wk = pd.to_datetime(df.bar_date).dt.isocalendar()
    key = wk.year.astype(str) + '-' + wk.week.astype(str).str.zfill(2)
    g = df.groupby(key.values).rr.sum()
    return dict(n=n, mean=mean, t=t, wr=(r > 0).mean(),
                total=r.sum(), weeks=len(g), rpw=r.sum() / len(g),
                green=int((g > 0).sum()))


def report():
    for v in ('A', 'B'):
        df = pd.read_csv(OUT_TRADES[v], keep_default_na=False, na_values=[''])
        df = df[df.in_book == 1]
        df['split'] = df.bar_date.map(split_of)
        log(f'\n===== VARIANT {v} (book, <=4/day) =====')
        log(f'{"split":6s} {"n":>6s} {"meanR":>8s} {"t":>6s} {"win":>6s} '
            f'{"totR":>9s} {"R/wk":>7s} {"weeks green":>12s}')
        for sp in ('TRAIN', 'VAL', 'TEST'):
            s = stats(df[df.split == sp])
            if not s['n']:
                log(f'{sp:6s}      0')
                continue
            log(f'{sp:6s} {s["n"]:6d} {s["mean"]:+8.3f} {s["t"]:6.2f} {s["wr"]:6.1%} '
                f'{s["total"]:+9.1f} {s["rpw"]:+7.2f} {s["green"]:6d}/{s["weeks"]:<5d}')
        s = stats(df)
        log(f'{"ALL":6s} {s["n"]:6d} {s["mean"]:+8.3f} {s["t"]:6.2f} {s["wr"]:6.1%} '
            f'{s["total"]:+9.1f} {s["rpw"]:+7.2f} {s["green"]:6d}/{s["weeks"]:<5d}')
        log('  exit-reason mix: ' + df.reason.value_counts().to_dict().__repr__())


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--report':
        report()
    else:
        main()
