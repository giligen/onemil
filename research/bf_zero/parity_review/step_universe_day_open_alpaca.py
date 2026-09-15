"""Read-only Alpaca REST: for today's scan-admitted symbols compare (a) the open the scanner logged, (b) get_current_bars
first-1-min open, (c) the snapshot DAILY bar open, (d) get_1min_bars_multi first bar open + its minute."""
import os, sys
os.chdir('/home/ec2-user/onemil'); sys.path.insert(0, '/home/ec2-user/onemil')
import logging; logging.basicConfig(level=logging.ERROR)
from datetime import timezone
from zoneinfo import ZoneInfo
from config import Config
from data_sources.alpaca_client import AlpacaClient
ET = ZoneInfo('America/New_York')
ADMIT = {'IONX': 19.26, 'BEX': 33.48, 'KMTS': 21.82, 'TEMT': 23.29, 'AVXX': 24.88, 'BAND': 49.51, 'CVI': 48.87, 'MVLL': 25.63, 'FBL': 26.85,
         'METU': 25.93, 'QMCO': 22.90, 'ASTN': 19.25, 'CRWL': 82.18, 'AZTA': 31.93, 'CSTL': 33.64, 'TANH': 19.60, 'FIGG': 19.63, 'DUOG': 58.96,
         'WTIU': 25.96, 'RFAI': 30.64, 'ALMR': 28.01, 'GLOB': 39.60, 'IBEX': 39.78, 'FLOC': 19.86, 'KIDS': 22.40, 'SCTX': 30.19}
cfg = Config()
client = AlpacaClient(cfg.alpaca_api_key, cfg.alpaca_api_secret, paper=cfg.alpaca_paper)
syms = sorted(ADMIT)
cur = client.get_current_bars(syms)
snap = client.get_snapshots(syms)
m1 = client.get_1min_bars_multi(syms, lookback_minutes=420)
print(f"{'sym':6s} {'logged':>8s} {'curbars':>8s} {'snapDaily':>9s} {'1min[0]':>8s} {'1min[0]_ET':>10s} {'n1min':>5s} flags")
for s in syms:
    lo = ADMIT[s]; cb = (cur.get(s) or {}).get('open'); sn = snap.get(s) or {}
    dbar = sn.get('daily_bar') or sn.get('dailyBar') or {}
    so = sn.get('open'); dd = sn.get('daily_bar_date')
    df = m1.get(s)
    if df is not None and len(df):
        df = df.sort_values('timestamp'); f = df.iloc[0]; ts = f['timestamp']
        ts = ts.to_pydatetime() if hasattr(ts, 'to_pydatetime') else ts
        et = ts.astimezone(ET); mo = float(f['open']); mstr = et.strftime('%H:%M'); n = len(df)
    else:
        mo = None; mstr = '-'; n = 0
    flags = []
    if cb is not None and abs(cb - lo) > 0.011: flags.append('LOGGED!=CUR')
    if mo is not None and abs(mo - lo) > 0.011: flags.append('LOGGED!=1MIN')
    if so is not None and mo is not None and abs(float(so) - mo) > 0.011: flags.append('DAILY!=1MIN')
    if mstr != '09:30' and mstr != '-': flags.append('FIRST_BAR_NOT_0930')
    print(f"{s:6s} {str(dd)[:10]:>10s} {lo:8.2f} {cb if cb is None else round(cb,2)!s:>8s} {so if so is None else round(float(so),2)!s:>9s} {mo if mo is None else round(mo,2)!s:>8s} {mstr:>10s} {n:5d} {' '.join(flags)}")
print('snapshot keys sample:', list((snap.get(syms[0]) or {}).keys()))
