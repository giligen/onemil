#!/usr/bin/env python3
"""Step 3 - MEASURED cost: Alpaca SIP NBBO at the red-to-green decision instant.

The decision is the CLOSE of the signal bar; the modelled fill is the OPEN of the next printed bar. We fetch the
first SIP quote at or after that fill instant and record:

  full_bps   (ask - bid) / mid * 1e4         the quoted spread, the thing cost_curve.csv estimates with a band
  direct_bps (ask - fill) / fill * 1e4       what a marketable BUY actually pays on top of the modelled bar-open fill

TEST (2026-06-01 onward) is SEALED: no signal from TEST is sampled.

Output `nbbo.csv` (per sampled signal) and `cost_curve_measured.csv` (median full_bps by price band x hour band,
the drop-in replacement for research/lit_review_2026/cost_curve.csv used by score.py).
"""
import os, sys, time
import numpy as np, pandas as pd
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockQuotesRequest

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
OUT = 'research/mature_method/red_to_green'
PER_STRATUM = int(sys.argv[1]) if len(sys.argv) > 1 else 60
log = lambda *a: (print(*a), sys.stdout.flush())

PB = [(0, 5, '$1-5'), (5, 10, '$5-10'), (10, 20, '$10-20'), (20, 50, '$20-50'), (50, 200, '$50-200'), (200, 1e9, '$200+')]
HB = [(0, 575, '09:30-09:35'), (576, 600, '09:35-10:00'), (601, 660, '10:00-11:00'),
      (661, 780, '11:00-13:00'), (781, 9999, '13:00+')]
pband = lambda p: next(n for lo, hi, n in PB if lo < p <= hi)
hband = lambda m: next(n for lo, hi, n in HB if lo <= m <= hi)

d = pd.read_csv(f'{OUT}/cands.csv', dtype={'symbol': str, 'day': str}, keep_default_na=False, na_values=[''])
d = d[d.variants.str.contains('lvl1003_f5', regex=False)]
d = d[(d.day < '2026-06-01') & (d.entry > 1.0) & (d.r_pct >= 0.5)]
d = d.drop_duplicates(['day', 'symbol', 'entry_m'])
d['pb'] = [pband(p) for p in d.entry]; d['hb'] = [hband(m) for m in d.entry_m]
d['split'] = np.where(d.day < '2026-01-01', 'TRAIN', 'VAL')
log('sampling frame:', len(d), 'signals (TRAIN+VAL, lvl1003_f5, TEST sealed)')

rng = np.random.default_rng(20260919)
take = []
for (pb, hb, sp), g in d.groupby(['pb', 'hb', 'split']):
    k = min(PER_STRATUM, len(g))
    take.append(g.iloc[rng.choice(len(g), size=k, replace=False)])
S = pd.concat(take, ignore_index=True)
log('sampled', len(S), 'signals across', S.groupby(['pb', 'hb']).ngroups, 'price x hour strata')

load_dotenv()
cli = StockHistoricalDataClient(os.getenv('ALPACA_API_KEY'),
                               os.getenv('ALPACA_SECRET_KEY') or os.getenv('ALPACA_API_SECRET'))
rows = []
t0 = time.time()
for i, r in enumerate(S.itertuples(index=False)):
    if i % 200 == 0:
        log('  %d/%d  %.1f min' % (i, len(S), (time.time() - t0) / 60))
    ts = (pd.Timestamp(r.day) + pd.Timedelta(minutes=int(r.entry_m))).tz_localize(
        'America/New_York', nonexistent='shift_forward', ambiguous=True).tz_convert('UTC')
    bid = ask = np.nan
    try:
        q = cli.get_stock_quotes(StockQuotesRequest(symbol_or_symbols=r.symbol, start=ts,
                                                   end=ts + pd.Timedelta(seconds=20), limit=1, feed='sip'))
        dd = q.data.get(r.symbol, [])
        if dd:
            bid, ask = float(dd[0].bid_price), float(dd[0].ask_price)
    except Exception as e:                                          # noqa: BLE001
        log('  WARNING quote fetch failed %s %s: %s' % (r.symbol, r.day, e))
    mid = 0.5 * (bid + ask) if bid == bid and ask == ask and bid > 0 and ask > 0 else np.nan
    rows.append(dict(day=r.day, symbol=r.symbol, split=r.split, pb=r.pb, hb=r.hb, entry_m=int(r.entry_m),
                     entry=r.entry, r_pct=r.r_pct, bid=bid, ask=ask,
                     full_bps=(ask - bid) / mid * 1e4 if mid == mid else np.nan,
                     direct_bps=(ask - r.entry) / r.entry * 1e4 if ask == ask and ask > 0 else np.nan))
q = pd.DataFrame(rows)
q.to_csv(f'{OUT}/nbbo.csv', index=False)
ok = q[q.full_bps.notna() & (q.full_bps > 0) & (q.full_bps < 5000)]
log('coverage: %d of %d (%.1f%%)' % (len(ok), len(q), 100.0 * len(ok) / max(len(q), 1)))
log('measured full spread bps  median %.1f  mean %.1f  p90 %.1f' %
    (ok.full_bps.median(), ok.full_bps.mean(), ok.full_bps.quantile(0.9)))
log('direct (ask - modelled fill) bps  median %.1f  mean %.1f' % (ok.direct_bps.median(), ok.direct_bps.mean()))
log('above the live max_spread_bps 300 gate: %.1f%%' % (100.0 * (ok.full_bps > 300).mean()))

cur = ok.groupby(['pb', 'hb']).full_bps.agg(['median', 'mean', 'count']).reset_index()
cur.columns = ['pb', 'hb', 'bps_median', 'bps_mean', 'n']
cur.to_csv(f'{OUT}/cost_curve_measured.csv', index=False)
log(cur.to_string(index=False))

# the band table's own charge on the SAME signals, for the direction-of-error statement
b = pd.read_csv('research/lit_review_2026/cost_curve.csv', dtype={'symbol': str, 'day': str},
                keep_default_na=False, na_values=[''])
b = b[(b.n_q > 0) & b.spread.notna() & (b.price > 0)].copy(); b['bps'] = b.spread / b.price * 1e4
BAND = {k: float(v) for k, v in b.groupby(['pb', 'hb']).bps.median().items()}
ok = ok.assign(band_bps=[BAND.get((p, h), np.nan) for p, h in zip(ok.pb, ok.hb)])
log('BAND on the same signals   median %.1f  mean %.1f' % (ok.band_bps.median(), ok.band_bps.mean()))
log('band / measured            median %.2fx  mean %.2fx' %
    (ok.band_bps.median() / ok.full_bps.median(), ok.band_bps.mean() / ok.full_bps.mean()))
ok.to_csv(f'{OUT}/nbbo_ok.csv', index=False)
