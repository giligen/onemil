"""Stage 3 of the F6 independent rebuild: costs, the daily book, and stats.

Reads scan_<sig><stp>.csv, applies the spread cost contract, runs the 12/day +
4-concurrent book per exit rule, and writes per-trade CSVs plus a stats table.
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
COST = 'research/lit_review_2026/cost_curve.csv'

MAX_PER_DAY = 12
MAX_CONCURRENT = 4
EXIT_K = {'stop': 0.875, 'eod': 0.412, 'target': 0.875}

SPLITS = [('TRAIN', '2025-01-01', '2025-12-31'),
          ('VAL', '2026-01-01', '2026-05-31'),
          ('TEST', '2026-06-01', '2026-09-11')]


def log(*a):
    print(*a)
    sys.stdout.flush()


def price_band(p):
    if p < 10:
        return '$5-10'
    if p < 20:
        return '$10-20'
    if p < 50:
        return '$20-50'
    if p < 200:
        return '$50-200'
    return '$200+'


def hour_band(m):
    if m < 575:
        return '09:30-09:35'
    if m < 600:
        return '09:35-10:00'
    if m < 660:
        return '10:00-11:00'
    if m < 780:
        return '11:00-13:00'
    return '13:00+'


def cost_table():
    d = pd.read_csv(COST, keep_default_na=False, na_values=[''])
    d = d[d.n_q > 0].copy()
    d['sp_bps'] = d.spread / d.price * 1e4
    t = d.groupby(['pb', 'hb']).sp_bps.median()
    return {k: v / 100.0 for k, v in t.items()}      # spread as % of price


def net_r(row, exit_key, tbl):
    S = tbl[(price_band(row.entry), hour_band(row.entry_min))]
    r_pct = row.R / row.entry * 100.0
    half = 0.5 * S / max(r_pct, 0.05)
    c = 0.25 * half
    for leg in str(row['%s_legs' % exit_key]).split(';'):
        w, t = leg.split(':')
        c += float(w) * half * EXIT_K[t]
    return row['%s_grossR' % exit_key] - c, half


def run_book_idx(df, exit_key):
    """Apply the 12/day, 4-concurrent, one-trade-per-symbol book."""
    emin = df['%s_exit_min' % exit_key]
    d = df.assign(_ex=emin).sort_values(['day', 'entry_min', 'symbol'])
    keep = []
    for day, g in d.groupby('day', sort=True):
        n = 0
        open_ex = []
        seen = set()
        for ix, row in g.iterrows():
            if n >= MAX_PER_DAY:
                break
            if row['symbol'] in seen:
                continue
            open_ex = [x for x in open_ex if x >= row['entry_min']]
            if len(open_ex) >= MAX_CONCURRENT:
                continue
            keep.append(ix)
            seen.add(row['symbol'])
            open_ex.append(row['_ex'])
            n += 1
    return d.loc[keep]


def weekly(df):
    wk = pd.to_datetime(df.day).dt.to_period('W')
    return df.groupby(wk).net_R.sum()


def stats_row(df, label):
    if len(df) == 0:
        return dict(split=label, n=0)
    w = weekly(df)
    nweeks = len(w)
    sd = df.net_R.std(ddof=1)
    t = df.net_R.mean() / (sd / np.sqrt(len(df))) if sd and len(df) > 1 else float('nan')
    stop_rate = df.exit_type.str.contains('stop').mean()
    return dict(split=label, n=len(df),
                trades_per_week=len(df) / nweeks if nweeks else float('nan'),
                mean_net_R=df.net_R.mean(), mean_gross_R=df.gross_R.mean(),
                t=t, WR=(df.net_R > 0).mean(), stop_rate=stop_rate,
                weekly_R=df.net_R.sum() / nweeks if nweeks else float('nan'),
                weeks=nweeks, weeks_green=(w > 0).mean(),
                worst_week=w.min(), total_net_R=df.net_R.sum())


def main(sig='a', stp='i'):
    tbl = cost_table()
    src = os.path.join(HERE, 'scan_%s%s.csv' % (sig, stp))
    df = pd.read_csv(src, keep_default_na=False, na_values=[''])
    log('scan rows:', len(df), src)

    all_stats = []
    for key in ['hold', 'r2', 'partial']:
        b = run_book_idx(df, key).copy()
        nr, half = zip(*[net_r(r, key, tbl) for _, r in b.iterrows()])
        b['net_R'] = nr
        b['half'] = half
        b['gross_R'] = b['%s_grossR' % key]
        b['exit_min'] = b['%s_exit_min' % key]
        b['exit_type'] = b['%s_exit_type' % key]
        out = b[['day', 'symbol', 'entry_min', 'entry', 'stop', 'R', 'exit_min',
                 'exit_type', 'gross_R', 'net_R', 'half', 'sig_min', 'src',
                 '%s_legs' % key]].sort_values(['day', 'entry_min', 'symbol'])
        p = os.path.join(HERE, 'trades_%s_%s%s.csv' % (key, sig, stp))
        out.to_csv(p, index=False)
        log('wrote', p, len(out))
        for name, lo, hi in SPLITS:
            s = out[(out.day >= lo) & (out.day <= hi)]
            r = stats_row(s, name)
            r['exit'] = key
            all_stats.append(r)
        r = stats_row(out, 'ALL')
        r['exit'] = key
        all_stats.append(r)

    st = pd.DataFrame(all_stats)
    cols = ['exit', 'split', 'n', 'trades_per_week', 'mean_net_R', 'mean_gross_R', 't',
            'WR', 'stop_rate', 'weekly_R', 'weeks', 'weeks_green', 'worst_week',
            'total_net_R']
    st = st[cols]
    p = os.path.join(HERE, 'stats_%s%s.csv' % (sig, stp))
    st.to_csv(p, index=False)
    log(st.to_string(index=False))
    log('wrote', p)


if __name__ == '__main__':
    main(*(sys.argv[1:3] or ['a', 'i']))
