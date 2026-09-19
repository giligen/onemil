#!/usr/bin/env python3
"""hod_preopen_regime — build the day-regime field table + the availability / causality trace.

Every field here must be computable from data whose RAW timestamp is strictly before 09:35:00 ET.
The trace (field -> source -> raw timestamp -> decision time) is printed and written to
availability.csv; nothing is imputed.

Output: day_fields.csv (one row per session), availability.csv, and the assertion log on stdout.
"""
import os, sqlite3, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT); sys.path.insert(0, ROOT)
D = f'{ROOT}/research/mature_method/hod_preopen_regime'
EARLY_CLOSE = {'2025-07-03', '2025-11-28', '2025-12-24'}

# ---------------------------------------------------------------- SPY / QQQ 1-min (Alpaca SIP)
ix = pd.read_csv(f'{D}/idx_1min.csv', dtype={'symbol': str, 'day': str, 'ts_utc': str})
print(f'idx_1min: {len(ix)} bars | SPY sessions {ix[ix.symbol=="SPY"].day.nunique()} | '
      f'QQQ sessions {ix[ix.symbol=="QQQ"].day.nunique()}')

rows = {}
for sym in ('SPY', 'QQQ'):
    s = ix[ix.symbol == sym]
    # prior RTH close: the 15:59 ET bar's close of the previous session (same source, no adjustment
    # mismatch with the intraday tape).
    cl_1559 = s[s.m_et == 959].set_index('day').c
    # 09:30 RTH opening bar's OPEN -- the open print, timestamped 09:30:00.
    op_0930 = s[s.m_et == 570].set_index('day').o
    # last premarket print strictly before 09:30 (m_et <= 569)
    pm = s[s.m_et <= 569].sort_values(['day', 'm_et']).groupby('day').tail(1).set_index('day')
    pm_c, pm_m = pm.c, pm.m_et
    # the 09:34 bar's CLOSE -- the last fact of the 09:30->09:35 window, timestamped 09:35:00.
    cl_0934 = s[s.m_et == 574].set_index('day').c
    days = sorted(op_0930.index)
    prev = {d: p for d, p in zip(days[1:], days[:-1])}
    f = pd.DataFrame(index=days)
    f['prev_close'] = [cl_1559.get(prev.get(d), np.nan) for d in days]
    f['prev_close2'] = [cl_1559.get(prev.get(prev.get(d, ''), ''), np.nan) for d in days]
    f['open0930'] = op_0930.reindex(days).values
    f['pm_last'] = pm_c.reindex(days).values
    f['pm_last_m'] = pm_m.reindex(days).values
    f['close0934'] = cl_0934.reindex(days).values
    p = sym.lower()
    rows[f'{p}_gap_pct'] = (f.open0930 / f.prev_close - 1) * 100
    rows[f'{p}_pm_ret_pct'] = (f.pm_last / f.prev_close - 1) * 100
    rows[f'{p}_r5_pct'] = (f.close0934 / f.open0930 - 1) * 100
    rows[f'{p}_prev_c2c_pct'] = (f.prev_close / f.prev_close2 - 1) * 100
    rows[f'{p}_pm_last_m'] = f.pm_last_m

df = pd.DataFrame(rows)
df.index.name = 'day'

# ---------------------------------------------------------------- 20d realised vol at T-1
con = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=180)
db = pd.read_sql("select symbol,bar_date,close from daily_bars where symbol in ('SPY','QQQ')", con)
con.close()
for sym in ('SPY', 'QQQ'):
    s = db[db.symbol == sym].sort_values('bar_date').set_index('bar_date').close
    v = (np.log(s / s.shift(1)).rolling(20).std() * np.sqrt(252) * 100).shift(1)   # T-1, no lookahead
    df[f'{sym.lower()}_vol20'] = df.index.map(v)

df['dow'] = pd.to_datetime(df.index).dayofweek

# ---------------------------------------------------------------- QQQ opening-auction imbalance
imb = pd.read_csv(f'{D}/qqq_imbalance.csv', dtype={'day': str}, low_memory=False)
pre = imb[(imb.auction_type == 'O') & (imb.m_et < 570)]
assert pre.m_et.max() <= 569, 'imbalance message at or after 09:30 -- causality violation'
last = pre.sort_values(['day', 'm_et', 'sec_et']).groupby('day').tail(1).set_index('day')
sgn = np.where(last.side == 'B', 1.0, np.where(last.side == 'A', -1.0, 0.0))
df['qqq_imb_signed'] = df.index.map(pd.Series(sgn * last.total_imbalance_qty.values, index=last.index))
df['qqq_imb_ratio'] = df.index.map(pd.Series(sgn * last.total_imbalance_qty.values
                                             / last.paired_qty.clip(lower=1).values, index=last.index))
df['qqq_imb_side'] = df.index.map(last.side)
df['qqq_imb_last_m'] = df.index.map(last.m_et)
df['qqq_imb_last_s'] = df.index.map(last.sec_et)

# ---------------------------------------------------------------- the 10:00 field that FAILED
dc = pd.read_csv(f'{ROOT}/research/mature_method/hod_filter_stack/day_ctx.csv',
                 dtype={'day': str}).set_index('day')
df['spy_ret_0930_1000'] = df.index.map(dc.spy_ret_0930_1000)          # the look-ahead reference only
df['breadth_by_1000'] = df.index.map(dc.breadth_by_1000)

df = df[~df.index.isin(EARLY_CLOSE)]
df = df[(df.index >= '2025-01-02') & (df.index <= '2026-09-12')]
df.to_csv(f'{D}/day_fields.csv')

# ---------------------------------------------------------------- availability / causality trace
def split_of(d):
    return 'TRAIN' if d < '2026-01-01' else ('VAL' if d < '2026-06-01' else 'TEST')


df['split'] = [split_of(d) for d in df.index]
TRACE = {
    'spy_gap_pct':      ('Alpaca SIP 1-min SPY', 'prev session 15:59 bar close; 09:30 bar OPEN', '09:30:00'),
    'spy_pm_ret_pct':   ('Alpaca SIP 1-min SPY', 'last premarket bar close, m_et <= 569', '<= 09:29:59'),
    'spy_r5_pct':       ('Alpaca SIP 1-min SPY', '09:30 bar open -> 09:34 bar close', '09:35:00'),
    'spy_prev_c2c_pct': ('Alpaca SIP 1-min SPY', 'T-1 and T-2 15:59 closes', 'T-1 16:00'),
    'spy_vol20':        ('cache.db daily_bars SPY', '20 log returns ending T-1', 'T-1 16:00'),
    'qqq_gap_pct':      ('Alpaca SIP 1-min QQQ', 'prev session 15:59 close; 09:30 bar OPEN', '09:30:00'),
    'qqq_pm_ret_pct':   ('Alpaca SIP 1-min QQQ', 'last premarket bar close, m_et <= 569', '<= 09:29:59'),
    'qqq_r5_pct':       ('Alpaca SIP 1-min QQQ', '09:30 bar open -> 09:34 bar close', '09:35:00'),
    'qqq_prev_c2c_pct': ('Alpaca SIP 1-min QQQ', 'T-1 and T-2 15:59 closes', 'T-1 16:00'),
    'qqq_vol20':        ('cache.db daily_bars QQQ', '20 log returns ending T-1', 'T-1 16:00'),
    'qqq_imb_ratio':    ('Databento XNAS.ITCH imbalance', 'last type-O msg before 09:30', '<= 09:29:59'),
    'qqq_imb_signed':   ('Databento XNAS.ITCH imbalance', 'last type-O msg before 09:30', '<= 09:29:59'),
    'dow':              ('calendar', 'the date itself', 'T-1 or earlier'),
    'spy_ret_0930_1000': ('causal_filter spy_1min', '09:30 -> 10:00 (THE FAILED FIELD)', '10:00:00'),
}
av = []
print('\n== AVAILABILITY / CAUSALITY TRACE (decision instant: earliest signal bar close 09:36) ==')
print('| field | source | construction | KNOWN AT | cov TRAIN | cov VAL | causal <09:35 |')
for f, (src, con_, known) in TRACE.items():
    v = pd.to_numeric(df[f], errors='coerce')
    ct = float(v[df.split == 'TRAIN'].notna().mean()); cv = float(v[df.split == 'VAL'].notna().mean())
    ok = known <= '09:35:00' or known.startswith('T-1') or known.startswith('<=')
    av.append(dict(field=f, source=src, construction=con_, known_at=known,
                   cov_train=ct, cov_val=cv, causal=ok))
    print(f'| {f:18s} | {src:31s} | {con_:44s} | {known:8s} | {ct:6.1%} | {cv:6.1%} | {str(ok)} |')
pd.DataFrame(av).to_csv(f'{D}/availability.csv', index=False)

# raw-timestamp assertions
print('\n== RAW-TIMESTAMP ASSERTIONS ==')
s = ix[(ix.symbol == 'SPY') & (ix.m_et == 574)]
print(f'  SPY 09:34 bar: {len(s)} sessions, ts_utc examples {list(s.ts_utc.head(2))} '
      f'(bar stamped at its OPEN; it closes 09:35:00 ET)')
print(f'  last premarket bar minute, SPY: median m_et {df.spy_pm_last_m.median():.0f} '
      f'(569 = 09:29), max {df.spy_pm_last_m.max():.0f}')
print(f'  QQQ imbalance last pre-cross message: median m_et {df.qqq_imb_last_m.median():.0f} '
      f'sec {df.qqq_imb_last_s.median():.0f}; MAX m_et {df.qqq_imb_last_m.max():.0f} -- '
      f'{"OK (<570)" if df.qqq_imb_last_m.max() < 570 else "VIOLATION"}')
print(f'  sessions in day_fields.csv: {len(df)}  '
      f'(TRAIN {(df.split=="TRAIN").sum()} VAL {(df.split=="VAL").sum()} TEST {(df.split=="TEST").sum()})')
print('\nwrote day_fields.csv / availability.csv')
