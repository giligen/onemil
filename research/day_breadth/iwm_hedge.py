"""EXPLORATION, no claim: can an IWM hedge strip the tape component out of the risk-on base-entry book?

Every split of this population has been seen, so nothing here is a result; it shapes the forward ledger.
For each kept trade (cell 1,412 rule, all signals): IWM return from the entry bar's open to the exit bar's open,
in the trade's R units. Beta fitted on TRAIN (OLS of trade R on IWM R); hedged R = R − beta × IWM R − hedge cost
(1 bp/side on the IWM leg). Per split: beta, R², mean, day-level std, worst day, worst week, green weeks —
unhedged vs hedged.

Usage: python3 research/day_breadth/iwm_hedge.py
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/ec2-user/onemil')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'research/hod_consol'))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import run_consol as rc  # noqa: E402
import test_1412 as T  # noqa: E402

IWM_CACHE = HERE / 'IWM_1min.parquet'
BR_TEST = HERE / 'breadth_test.parquet'


def fetch_iwm():
    """IWM 1-min SIP bars 2025-01-01 .. 2026-09-19, RTH only, cached."""
    if IWM_CACHE.exists():
        return pd.read_parquet(IWM_CACHE)
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    from config import Config
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    cfg = Config()
    cl = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_api_secret)
    frames = []
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 9, 19, tzinfo=timezone.utc)
    while start < end:
        stop = min(start + timedelta(days=31), end)
        df = cl.get_stock_bars(StockBarsRequest(symbol_or_symbols='IWM', timeframe=TimeFrame.Minute,
                                                start=start, end=stop, feed='sip')).df
        if df is not None and len(df):
            frames.append(df.reset_index())
        print(f'IWM {start.date()} +{0 if df is None else len(df)}', flush=True)
        start = stop
    d = pd.concat(frames, ignore_index=True)
    ts = pd.to_datetime(d.timestamp, utc=True).dt.tz_convert('America/New_York')
    d = pd.DataFrame(dict(day=ts.dt.strftime('%Y-%m-%d'), m=ts.dt.hour * 60 + ts.dt.minute, o=d.open, c=d.close))
    d = d[(d.m >= 570) & (d.m <= 959)].drop_duplicates(['day', 'm'])
    d.to_parquet(IWM_CACHE, index=False)
    return d


def kept_book():
    """All-split kept trades (BR at signal_m >= EDGE) with entry/exit minutes, R and r_pct."""
    br = pd.read_parquet(HERE / 'breadth.parquet')[['day', 'm', 'BR']]
    if BR_TEST.exists():
        brt = pd.read_parquet(BR_TEST)
    else:
        brt = T.test_breadth()
        brt.to_parquet(BR_TEST, index=False)
    brmap = dict(zip(zip(pd.concat([br.day, brt.day]), pd.concat([br.m, brt.m])), pd.concat([br.BR, brt.BR])))
    sig = pd.read_parquet(ROOT / 'research/hod_consol/signals.parquet')
    paths = pd.read_parquet(ROOT / 'research/hod_consol/paths.parquet')
    idx = paths.sort_values(['day', 'symbol', 'm']).set_index(['day', 'symbol']).sort_index()
    w = rc.walk(sig, idx, rc.fill_c1)
    w['BR'] = [brmap.get((d, int(m)), np.nan) for d, m in zip(w.day, w.signal_m)]
    w = w[w.BR >= T.EDGE].copy()
    w['r_frac'] = w.R / w.entry
    return w


def main():
    iwm = fetch_iwm()
    op = dict(zip(zip(iwm.day, iwm.m), iwm.o))
    k = kept_book()

    def iwm_ret(r):
        a = op.get((r.day, int(r.entry_m)))
        b = op.get((r.day, int(r.exit_m))) or op.get((r.day, int(r.exit_m) - 1))
        return (b / a - 1) if a and b else np.nan

    k['iwm_ret'] = [iwm_ret(r) for r in k.itertuples()]
    k = k.dropna(subset=['iwm_ret'])
    k['iwm_R'] = k.iwm_ret / k.r_frac
    tr = k[k.split == 'TRAIN']
    beta = float(np.polyfit(tr.iwm_R, tr.net_R, 1)[0])
    cost = 0.0002 * abs(beta) / k.r_frac
    k['hedged_R'] = k.net_R - beta * k.iwm_R - cost
    print(f'beta (TRAIN OLS, trade R on IWM R) = {beta:+.2f}; trades with IWM data {len(k)}')
    print('| split | n | corr | mean R → hedged | day std → hedged | worst day → hedged | worst week → hedged '
          '| green weeks → hedged |')
    for split in ('TRAIN', 'VAL', 'TEST'):
        s = k[k.split == split]
        dr, dh = s.groupby('day').net_R.sum(), s.groupby('day').hedged_R.sum()
        wr, wh = s.groupby('wk').net_R.sum(), s.groupby('wk').hedged_R.sum()
        corr = float(np.corrcoef(s.iwm_R, s.net_R)[0, 1])
        print(f'| {split} | {len(s)} | {corr:+.2f} | {s.net_R.mean():+.3f} → {s.hedged_R.mean():+.3f} '
              f'| {dr.std():.1f} → {dh.std():.1f} | {dr.min():+.1f} → {dh.min():+.1f} | {wr.min():+.1f} → {wh.min():+.1f} '
              f'| {(wr > 0).mean():.0%} → {(wh > 0).mean():.0%} |', flush=True)


if __name__ == '__main__':
    main()
