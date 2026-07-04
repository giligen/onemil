"""Premarket-volume monster-detection study (ledger debt #3, 2026-07-04).

The only unrefuted monster-detection channels are float and premarket
activity. This tests premarket: for each post-veto defended trade, fetch
4:00-9:30 ET 1-min bars and compute pm_volume, pm_dollar_vol,
pm_vol_vs_prev (pm_volume / prev-day volume), pm_range_pct. Then:
  1. Monster separation: do top-decile trades sit in high-PM terciles?
  2. Loser separation: does low PM volume mark the stop-out cohort?
  3. Era consistency: does any gradient hold in 2025H1 / H2 / 2026?
No cache writes (API -> /tmp CSV only; runs concurrently with other jobs).

Honest caveats: IEX feed premarket coverage is partial for thin names —
coverage is REPORTED; a no-data symbol-day is 'unknown', never zero.

Artifacts: /tmp/orb_premarket_features.csv, /tmp/orb_premarket_report.txt
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = '/home/ec2-user/onemil'
sys.path.insert(0, ROOT)
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, '.env'))

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

ET = ZoneInfo('America/New_York')
REPORT = []


def log(m=''):
    print(m, flush=True)
    REPORT.append(str(m))


def pm_features(client, sym, day):
    d = datetime(int(day[:4]), int(day[5:7]), int(day[8:10]), tzinfo=ET)
    try:
        req = StockBarsRequest(
            symbol_or_symbols=sym, timeframe=TimeFrame.Minute,
            start=d.replace(hour=4, minute=0),
            end=d.replace(hour=9, minute=29))
        bars = client.get_stock_bars(req).data.get(sym, [])
    except Exception:
        return None
    if not bars:
        return None
    vol = sum(b.volume for b in bars)
    dol = sum(b.volume * b.vwap for b in bars if b.vwap)
    hi = max(b.high for b in bars)
    lo = min(b.low for b in bars)
    last = bars[-1].close
    return dict(pm_volume=vol, pm_dollar_vol=dol,
                pm_range_pct=(hi - lo) / last * 100 if last else None,
                pm_n_bars=len(bars))


def main():
    sel = pd.read_csv('/tmp/orb_base_sel.csv')
    sel = sel[sel['prev_day_range_pct'] > 8.0].copy()
    sel['day'] = sel['date'].astype(str).str[:10]
    client = StockHistoricalDataClient(os.getenv('ALPACA_API_KEY'),
                                       os.getenv('ALPACA_API_SECRET'))
    rows = []
    for i, (_, r) in enumerate(sel.iterrows()):
        f = pm_features(client, r['symbol'], r['day'])
        rec = dict(symbol=r['symbol'], day=r['day'],
                   sized_pnl=r['_sized_pnl'], exit_reason=r['exit_reason'],
                   prev_vol=r.get('avg_daily_volume_20d'))
        if f:
            rec.update(f)
        rows.append(rec)
        if (i + 1) % 100 == 0:
            log(f"  ... {i + 1}/{len(sel)}")
    df = pd.DataFrame(rows)
    df.to_csv('/tmp/orb_premarket_features.csv', index=False)

    got = df.dropna(subset=['pm_volume'])
    log(f"\npremarket coverage: {len(got)}/{len(df)} "
        f"({len(got) / len(df) * 100:.0f}%)")
    if len(got) < 100:
        log("insufficient coverage — feed lacks premarket for these names")
        return
    got = got.copy()
    got['pm_vol_vs_prev'] = got['pm_volume'] / got['prev_vol'].clip(lower=1)
    for feat in ('pm_volume', 'pm_dollar_vol', 'pm_vol_vs_prev', 'pm_range_pct'):
        g = got.dropna(subset=[feat])
        if len(g) < 100:
            continue
        g = g.copy()
        g['tercile'] = pd.qcut(g[feat], 3, labels=['low', 'mid', 'high'],
                               duplicates='drop')
        log(f"\n=== {feat} terciles ===")
        agg = g.groupby('tercile', observed=True)['sized_pnl'] \
            .agg(['count', 'sum', 'mean']).round(0)
        # monster rate: share of book top-decile trades per tercile
        top = set(g.nlargest(max(len(g) // 10, 5), 'sized_pnl').index)
        agg['monster_n'] = g.groupby('tercile', observed=True) \
            .apply(lambda x: len(set(x.index) & top), include_groups=False)
        log(agg.to_string())
    # era consistency for the most promising feature is read manually from
    # the tercile tables; dump era split for pm_vol_vs_prev
    g = got.dropna(subset=['pm_vol_vs_prev']).copy()
    if len(g) >= 150:
        g['tercile'] = pd.qcut(g['pm_vol_vs_prev'], 3,
                               labels=['low', 'mid', 'high'], duplicates='drop')
        for era, lo, hi in (('2025H1', '2025-01', '2025-07'),
                            ('2025H2', '2025-07', '2026-01'),
                            ('2026', '2026-01', '2027-01')):
            e = g[(g['day'] >= lo) & (g['day'] < hi)]
            if len(e) < 30:
                continue
            m = e.groupby('tercile', observed=True)['sized_pnl'].mean().round(0)
            log(f"\npm_vol_vs_prev mean $/trade {era}: {m.to_dict()}")
    with open('/tmp/orb_premarket_report.txt', 'w') as fh:
        fh.write('\n'.join(REPORT))
    log('\ndone')


if __name__ == '__main__':
    main()
