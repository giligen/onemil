#!/usr/bin/env python3
"""Independent rebuild of cell 1,708: ORB latency replay (cell 1,426) with true share counts.

Rules (from replay.py docstring / PREREG): t* = first XNAS trade print >= trigger in [0,300) s after 09:35 ET;
d <= t*: fill at last ask strictly before t* if ask <= limit; d > t*: fill at ask before d if ask+0.02 <= limit.
P&L = _sized_pnl + shares*(entry_price - fill), shares = _rp_position/entry_price. R = P&L/375.
"""
import json, numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil/research/orb_latency_bt'
OUT = '/home/ec2-user/onemil/research/exec_learnings_oct2026/rebuild_1708.json'
DELAYS = [0, 30]
OOS = ['2023-24', '2025H2-2026-09']


def sec_since_0935(ts_ns, date):
    """Seconds since 09:35:00 America/New_York on `date` for UTC-ns timestamps."""
    anchor = pd.Timestamp(f'{date} 09:35:00', tz='America/New_York').tz_convert('UTC').value
    return (np.asarray(ts_ns).astype('int64') - anchor) / 1e9


def replay(tr_t, tr_p, q_t, q_ask, trigger, limit, d):
    """Return (status, fill) for one fill at delay d."""
    m = (tr_t >= 0) & (tr_t < 300) & (tr_p >= trigger)
    if not m.any():
        return 'nots', np.nan
    ts = tr_t[m][0]
    ref = ts if ts >= d else float(d)
    i = np.searchsorted(q_t, ref, side='left') - 1  # last record strictly before ref
    if i < 0 or not np.isfinite(q_ask[i]) or q_ask[i] <= 0:
        return 'noquote', np.nan
    ask = q_ask[i]
    buf = 0.0 if ts >= d else 0.02
    return ('filled', ask) if ask + buf <= limit else ('skip', np.nan)


def stats(df):
    """df: one row per population fill in the set at one delay; returns metric dict."""
    n = len(df); R = df.pnl / 375.0
    mean = R.mean()
    dev = df.pnl - df.pnl.mean()
    cs = dev.groupby(df.date).sum()
    se = np.sqrt((cs ** 2).sum()) / n / 375.0
    k = int(np.ceil(0.05 * n))
    ex = (df.pnl.sort_values(ascending=False).iloc[k:] / 375.0).mean()
    f = df[df.status == 'filled']
    bps = ((f.fill - f.entry_price) / f.entry_price * 1e4).mean()
    return dict(n=n, n_filled=int(len(f)), mean_R=mean, ci95=[mean - 1.96 * se, mean + 1.96 * se],
                se_R=se, sum_usd=float(df.pnl.sum()), ex_top5pct_R=float(ex), mean_fill_vs_bt_bps=float(bps),
                n_days=int(df.date.nunique()))


def main():
    pop = pd.read_csv(f'{ROOT}/population.csv')
    pop = pop[pop.entered == 1]
    pop = pop[pop.period.isin(OOS)].copy()
    pop['shares'] = pop['_rp_position'] / pop['entry_price']
    print('n pop OOS', len(pop), 'mean shares', pop.shares.mean(), 'median', pop.shares.median(), flush=True)
    rows = []; n_missing = 0
    pop = pop.rename(columns={'_sized_pnl': 'sized_pnl'})
    for r in pop.itertuples():
        path = f'{ROOT}/raw/{r.date}__{r.symbol}.parquet'
        try:
            raw = pd.read_parquet(path)
        except FileNotFoundError:
            n_missing += 1  # missing tick data: excluded from the denominator (as REPORT.md does)
            continue
        t = raw[raw.schema == 'trades'].sort_values('ts_event')
        q = raw[raw.schema == 'mbp-1'].sort_values('ts_event')
        tr_t = sec_since_0935(t.ts_event.values, r.date); tr_p = t.price.values
        q_t = sec_since_0935(q.ts_event.values, r.date); q_ask = q.ask_px_00.values.astype(float)
        for d in DELAYS:
            st, fill = replay(tr_t, tr_p, q_t, q_ask, r.trigger, r.limit, d)
            pnl = r.sized_pnl + r.shares * (r.entry_price - fill) if st == 'filled' else 0.0
            rows.append(dict(date=r.date, symbol=r.symbol, delay=d, status=st, fill=fill,
                             entry_price=r.entry_price, pnl=pnl))
    res = pd.DataFrame(rows)
    out = dict(n_missing_excluded=n_missing, mean_shares=float(pop.shares.mean()), n_pop_oos=len(pop),
               status_counts={int(d): res[res.delay == d].status.value_counts().to_dict() for d in DELAYS})
    for d in DELAYS:
        out[f'delay_{d}s'] = stats(res[res.delay == d])
    json.dump(out, open(OUT, 'w'), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float), flush=True)


main()
