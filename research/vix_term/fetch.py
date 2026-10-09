"""Fetch stage for PREREG_1: CBOE VIX/VIX3M closes, Alpaca daily bars (SPY calendar, SVXY, VXX), 15:58 ET quotes.

Everything is cached under research/vix_term/data/ (atomic writes). Verbose progress; any fallback logs WARNING.
"""
import io, os, sys, logging, urllib.request
from datetime import datetime, timedelta
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data')
logger = logging.getLogger('vix_term.fetch')
CBOE_URL = 'https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv'  # same as scripts/momentum_sleeve.py
START, END = '2011-01-03', '2026-10-08'


def _atomic_csv(df, name):
    """Write ``df`` to data/name via tmp + rename (never a half-written cache)."""
    p = os.path.join(DATA, name)
    df.to_csv(p + '.tmp')
    os.replace(p + '.tmp', p)
    logger.info("wrote %s (%d rows)", p, len(df))


def fetch_cboe_close(name):
    """CBOE daily history of index ``name`` as a close series (copy of scripts/momentum_sleeve.fetch_cboe_close)."""
    req = urllib.request.Request(CBOE_URL.format(name=name), headers={'User-Agent': 'Mozilla/5.0 research'})
    d = pd.read_csv(io.StringIO(urllib.request.urlopen(req, timeout=60).read().decode()))
    d['DATE'] = pd.to_datetime(d['DATE'], format='%m/%d/%Y')
    return d.set_index('DATE')['CLOSE'].astype(float).rename(name)


def _client():
    """Alpaca historical data client from the env keys (read-only; keys loaded from .env via dotenv if present)."""
    from dotenv import load_dotenv
    load_dotenv('/home/ec2-user/onemil/.env')
    from alpaca.data.historical import StockHistoricalDataClient
    k, s = os.environ.get('ALPACA_API_KEY'), os.environ.get('ALPACA_API_SECRET')
    if not k or not s:
        raise SystemExit("ERROR: ALPACA_API_KEY / ALPACA_API_SECRET missing")
    return StockHistoricalDataClient(k, s)


def fetch_daily(client, sym, feed):
    """Daily bars (adjustment=all) for ``sym`` -> DataFrame[open, close] indexed by date."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment
    req = StockBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame.Day, start=datetime(2010, 12, 20),
                           end=datetime(2026, 10, 9), adjustment=Adjustment.ALL, feed=feed)
    b = client.get_stock_bars(req).df
    b = b.reset_index()
    b['date'] = pd.to_datetime(b['timestamp']).dt.tz_convert('America/New_York').dt.tz_localize(None).dt.normalize()
    return b.set_index('date')[['open', 'high', 'low', 'close', 'volume']]


def fetch_quotes(client, syms, sessions, feed):
    """First quote at/after 15:58:00 ET for each (sym, session): rows (sym, date, bid, ask, half-spread bps)."""
    from alpaca.data.requests import StockQuotesRequest
    rows = []
    for d in sessions:
        t0 = pd.Timestamp(f"{d.date()} 15:58:00", tz='America/New_York').tz_convert('UTC')
        for s in syms:
            try:
                q = client.get_stock_quotes(StockQuotesRequest(symbol_or_symbols=s, start=t0.to_pydatetime(),
                        end=(t0 + pd.Timedelta(seconds=45)).to_pydatetime(), limit=5, feed=feed)).df
            except Exception as e:
                logger.warning("quote fetch %s %s failed: %s", s, d.date(), e); continue
            q = q.reset_index()
            q = q[(q.bid_price > 0) & (q.ask_price > q.bid_price)]
            if q.empty:
                logger.warning("no valid 15:58 quote for %s %s", s, d.date()); continue
            r = q.iloc[0]
            mid = (r.bid_price + r.ask_price) / 2
            rows.append((s, d.date(), r.bid_price, r.ask_price, (r.ask_price - r.bid_price) / 2 / mid * 1e4))
        logger.info("quotes %s done (%d rows so far)", d.date(), len(rows))
    return pd.DataFrame(rows, columns=['sym', 'date', 'bid', 'ask', 'half_spread_bps'])


def main():
    """Fetch and cache everything; prints coverage summaries."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    os.makedirs(DATA, exist_ok=True)
    vix, v3m = fetch_cboe_close('VIX'), fetch_cboe_close('VIX3M')
    _atomic_csv(pd.concat([vix, v3m], axis=1), 'cboe.csv')
    c = _client()
    bars = {}
    for sym in ['SPY', 'SVXY', 'VXX', 'UVXY']:
        for feed in ('sip', 'iex'):
            try:
                bars[sym] = fetch_daily(c, sym, feed); logger.info("%s daily via %s: %d bars %s..%s", sym, feed, len(bars[sym]),
                    bars[sym].index.min().date(), bars[sym].index.max().date())
                break
            except Exception as e:
                logger.warning("%s daily feed=%s failed (%s)", sym, feed, e)
        else:
            raise SystemExit(f"ERROR: no daily bars for {sym}")
        _atomic_csv(bars[sym], f'{sym}_daily.csv')
    sess = bars['SPY'].index[-62:-2]
    for feed in ('sip', 'iex'):
        try:
            qd = fetch_quotes(c, ['SVXY', 'VXX'], sess, feed)
            if len(qd) > 60:
                logger.info("quotes via feed=%s", feed); break
            logger.warning("feed=%s returned only %d quotes, trying next", feed, len(qd))
        except Exception as e:
            logger.warning("quotes feed=%s failed: %s", feed, e)
    qd['feed'] = feed
    _atomic_csv(qd.set_index('sym'), 'quotes_1558.csv')
    print(qd.groupby('sym').half_spread_bps.describe())


if __name__ == '__main__':
    main()
