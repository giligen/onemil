"""Cell 1,732 — re-walk every ORB paper fill (9/25-11/10) on SIP and IEX 1-min bars under rule L8.

Read-only: reads data/trades.db (ro URI), analysis_results/orb_bplus_book.csv, the 11/10 features CSV and
Alpaca bars; writes research/book_review_20261111/stops_1732.csv only.
"""
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, '/home/ec2-user/onemil')
from trading.orb_touchgo_filter import TouchgoConfig, evaluate_rule_m, evaluate_rule_d  # noqa: E402
from alpaca.data.historical import StockHistoricalDataClient  # noqa: E402
from alpaca.data.requests import StockBarsRequest  # noqa: E402
from alpaca.data.timeframe import TimeFrame  # noqa: E402
from alpaca.data.enums import DataFeed  # noqa: E402

ROOT = '/home/ec2-user/onemil'
OUT = ROOT + '/research/book_review_20261111/stops_1732.csv'
load_dotenv(ROOT + '/.env')
CLIENT = StockHistoricalDataClient(os.environ['ALPACA_API_KEY'], os.environ['ALPACA_API_SECRET'])
TG = TouchgoConfig()
LOCK_ARM, LOCK_R, SCALE_R, SCALE_FRAC = 1.75, 0.5, 3.0, 0.4


def ledger():
    """Return the closed ORB paper fills since 9/25 with range levels parsed from pattern_data."""
    c = sqlite3.connect(f'file:{ROOT}/data/trades.db?mode=ro', uri=True)
    df = pd.read_sql("select trade_date,symbol,fill_price,stop_loss_price,shares,exit_reason,exit_price,filled_at,"
                     "exited_at,exit_trigger_price,pnl,pattern_data from trades where strategy='orb' and "
                     "trade_date>='2026-09-25' and exit_price is not null order by filled_at", c)
    pdx = df.pattern_data.map(json.loads)
    df['range_high'] = pdx.map(lambda p: p['range_high'])
    df['range_low'] = pdx.map(lambda p: p['range_low'])
    df['pool'] = pdx.map(lambda p: p.get('pool'))
    return df.drop(columns='pattern_data')


def bars(sym, day, feed):
    """Fetch regular-session 1-min bars for one symbol-day on the given Alpaca feed (UTC-indexed)."""
    d0 = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
    req = StockBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame.Minute, start=d0 + timedelta(hours=13),
                           end=d0 + timedelta(hours=21), feed=feed)
    try:
        b = CLIENT.get_stock_bars(req).df
    except Exception as e:  # noqa: BLE001 — research script, loud
        print(f'WARNING bars {sym} {day} {feed}: {e}', flush=True)
        return pd.DataFrame()
    if b is None or len(b) == 0:
        print(f'WARNING no bars {sym} {day} {feed}', flush=True)
        return pd.DataFrame()
    b = b.reset_index().set_index('timestamp').sort_index()
    return b


def walk(b, r, fill_ts):
    """Rule L8 from the live fill minute: touchgo M/D, floored stop, lock 1.75R->0.5R, 40%@+3R, 15:45 ET close.

    Same-bar ambiguity is resolved stop-first (conservative). Returns (reason, exit_ts, blended_exit_px).
    """
    if b.empty:
        return 'no_bars', None, None
    e, stop, rs = r.fill_price, r.stop_loss_price, r.range_high - r.range_low
    rrisk = e - r.range_low
    fc = pd.Timestamp(r.trade_date + ' 15:45', tz='America/New_York').tz_convert('UTC')
    post = b[(b.index >= fill_ts.floor('min')) & (b.index <= fc)]
    if post.empty:
        return 'no_bars', None, None
    bb = post.iloc[0]
    fm, xm = evaluate_rule_m(bb.open, bb.high, bb.low, bb.close, TG)
    if fm and xm is not None:
        return 'tag_bb', post.index[0], xm
    if len(post) >= 2:
        fd, xd = evaluate_rule_d(e, post.iloc[1].low, rs, TG)
        if fd and xd is not None:
            return 'tag_b1', post.index[1], xd
    scaled, locked, scale_px = False, False, e + SCALE_R * rs
    for ts, x in post.iloc[1:].iterrows():
        if x.low <= stop:
            px = stop
            reason = 'lock_stop' if locked else 'stop'
            return reason, ts, (SCALE_FRAC * scale_px + (1 - SCALE_FRAC) * px) if scaled else px
        if not scaled and x.high >= scale_px:
            scaled = True
        if not locked and x.high >= e + LOCK_ARM * rrisk:
            locked, stop = True, max(stop, e + LOCK_R * rrisk)
    last = post.iloc[-1].close
    return 'eod', post.index[-1], (SCALE_FRAC * scale_px + (1 - SCALE_FRAC) * last) if scaled else last


def main():
    """Build stops_1732.csv: one row per paper fill with SIP/IEX re-walks and the BT join."""
    print('TouchgoConfig:', TG, flush=True)
    led = ledger()
    book = pd.read_csv(ROOT + '/analysis_results/orb_bplus_book.csv')
    book = book[book.entered == 1].set_index(['date', 'symbol'])
    feat = pd.read_csv(ROOT + '/analysis_results/orb_features_20261110_2203.csv', low_memory=False)
    feat = feat.set_index(['date', 'symbol'])
    rows = []
    for r in led.itertuples():
        fill_ts = pd.Timestamp(r.filled_at).tz_convert('UTC')
        ex_ts = pd.Timestamp(r.exited_at).tz_convert('UTC')
        out = dict(date=r.trade_date, symbol=r.symbol, pool=r.pool, fill=r.fill_price, stop=r.stop_loss_price,
                   range_low=r.range_low, range_high=r.range_high, shares=r.shares,
                   live_exit_reason=r.exit_reason, live_exit_ts=ex_ts.isoformat(), live_exit_px=r.exit_price,
                   live_pnl=r.pnl, trigger_px=r.exit_trigger_price,
                   fill_min_et=fill_ts.tz_convert('America/New_York').strftime('%H:%M:%S'),
                   mins_to_exit=round((ex_ts - fill_ts).total_seconds() / 60, 1))
        for feed, tag in ((DataFeed.SIP, 'sip'), (DataFeed.IEX, 'iex')):
            b = bars(r.symbol, r.trade_date, feed)
            win = b[(b.index >= ex_ts.floor('min') - pd.Timedelta(minutes=1)) & (b.index <= ex_ts.floor('min'))] \
                if not b.empty else b
            out[f'{tag}_minute_low_at_trigger'] = round(float(win.low.min()), 4) if len(win) else None
            pre = b[(b.index > fill_ts.floor('min')) & (b.index <= ex_ts)] if not b.empty else b
            out[f'{tag}_min_low_fill_to_exit'] = round(float(pre.low.min()), 4) if len(pre) else None
            reason, ts, px = walk(b, r, fill_ts)
            out[f'sim_exit_{tag}'] = reason
            out[f'sim_exit_{tag}_ts'] = ts.isoformat() if ts is not None else None
            out[f'sim_pnl_{tag}'] = round((px - r.fill_price) * r.shares, 1) if px is not None else None
        key = (r.trade_date, r.symbol)
        out['bt_taken'] = key in book.index
        out['bt_exit'] = book.loc[key, 'exit_reason'] if key in book.index else None
        out['bt_entry'] = round(float(book.loc[key, 'entry_price']), 4) if key in book.index else None
        out['feat_present'] = key in feat.index
        out['feat_entered'] = int(feat.loc[key, 'entered']) if key in feat.index else None
        out['feat_exit'] = feat.loc[key, 'exit_reason'] if key in feat.index else None
        print(r.trade_date, r.symbol, r.exit_reason, '| sip', out['sim_exit_sip'], '| iex', out['sim_exit_iex'],
              '| bt', out['bt_exit'], flush=True)
        rows.append(out)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print('wrote', OUT, flush=True)


if __name__ == '__main__':
    main()
