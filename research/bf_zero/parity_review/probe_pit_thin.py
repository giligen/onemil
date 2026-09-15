"""Is the PIT/topup tape thin across the board? 20 random superset keys per side DB vs Alpaca SIP REST (cum vol @10:30)."""
import os, sys, sqlite3, random
os.chdir('/home/ec2-user/onemil'); sys.path.insert(0, '/home/ec2-user/onemil')
import logging; logging.basicConfig(level=logging.ERROR)
import pandas as pd, numpy as np
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from config import Config
from data_sources.alpaca_client import AlpacaClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed
ET = ZoneInfo('America/New_York'); ROOT = '/home/ec2-user/onemil'
cfg = Config(); client = AlpacaClient(cfg.alpaca_api_key, cfg.alpaca_api_secret, paper=cfg.alpaca_paper)
K = pd.read_csv('research/bf_zero/parity_review/thin_tape_superset_keys.csv', dtype=str)
random.seed(7)
for src, path in (('pit_bars_1min.db', 'data/research/databento/pit_bars_1min.db'), ('topup.db', 'research/ignition_capcheck/topup.db')):
    con = sqlite3.connect(f'file:{ROOT}/{path}?mode=ro', uri=True)
    keys = K[K.src == src][['symbol', 'day']].values.tolist(); random.shuffle(keys); ratios = []
    for sym, day in keys[:20]:
        g = pd.read_sql("select t, v from bars where day=? and symbol=?", con, params=[day, sym])
        ts = pd.to_datetime(g.t, utc=True).dt.tz_convert('America/New_York'); m = ts.dt.hour * 60 + ts.dt.minute
        sv = float(g.v[(m >= 570) & (m <= 630)].sum())
        d = datetime.strptime(day, '%Y-%m-%d')
        req = StockBarsRequest(symbol_or_symbols=[sym], timeframe=TimeFrame(1, TimeFrameUnit.Minute), start=datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET).astimezone(timezone.utc), end=datetime(d.year, d.month, d.day, 10, 31, tzinfo=ET).astimezone(timezone.utc), feed=DataFeed.SIP)
        raw = client._to_dict(client.data_client.get_stock_bars(req)).get(sym) or []
        av = float(sum(b.volume for b in raw))
        r = sv / av if av else float('nan'); ratios.append(r)
        print(f'{src:16s} {sym:6s} {day} study {sv:10.0f} alpaca {av:10.0f} ratio {r:.3f}')
    a = np.array(ratios); print(f'== {src}: n={len(a)} median ratio {np.nanmedian(a):.3f} max {np.nanmax(a):.3f} | ratio>0.5: {int((a>0.5).sum())}')
