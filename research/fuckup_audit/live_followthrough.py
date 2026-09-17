#!/usr/bin/env python3
"""(1) Bull flag: the follow-through-volume rule found on the 53 REAL trades, validated on the honest raw BF cache
    (data/bull_flag_cache_causal_full_20260905.csv, 896 Stage-1 detections 2025-01..2026-08, regen-7 exits) —
    volume of the FILL minute vs the prior 5 bars (known one minute after entry) and the gap (known at 09:30).
    Counterfactual exit for low-follow-through trades: sell at the open of the bar AFTER the fill minute closes.
(2) ORB: a 10-minute time stop on the 116 real trades (exit at the 09:45 bar open if the trade is below +0.25R then).
Both are POST-HOC rules on small samples; cells are counted in the output."""
import sqlite3, numpy as np, pandas as pd
from zoneinfo import ZoneInfo
ET = ZoneInfo('America/New_York'); pd.set_option('display.width', 250)
cache = sqlite3.connect('file:data/cache.db?mode=ro', uri=True)


def bars(sym, day):
    b = pd.read_sql("select timestamp, open, high, low, close, volume from intraday_bars_1min where symbol=? and bar_date=? order by timestamp", cache, params=(sym, day))
    if not len(b): return b
    ts = pd.to_datetime(b.timestamp, utc=True).dt.tz_convert(ET); b['m'] = ts.dt.hour * 60 + ts.dt.minute
    return b[(b.m >= 570) & (b.m < 960)].reset_index(drop=True)


def gap(sym, day):
    d = pd.read_sql("select open, close from daily_bars where symbol=? and bar_date<=? order by bar_date desc limit 2", cache, params=(sym, day))
    return (d.iloc[0].open / d.iloc[1].close - 1) * 100 if len(d) == 2 else np.nan


# ---------- (1) BF cache validation
c = pd.read_csv('data/bull_flag_cache_causal_full_20260905.csv', keep_default_na=False, na_values=[''], dtype={'symbol': str})
c['R_ps'] = c.entry_price - c.stop_loss
c = c[(c.R_ps > 0) & (c.shares > 0)].copy(); c['R'] = c.pnl / (c.R_ps * c.shares)
rows = []
for r in c.itertuples():
    b = bars(r.symbol, r.date)
    if not len(b): continue
    hh, mm = [int(x) for x in str(r.entry_time_et)[:5].split(':')]; fm = hh * 60 + mm
    fb = b[b.m == fm]; pre = b[b.m < fm].tail(5); nxt = b[b.m > fm]
    if not len(fb) or len(pre) < 5 or not len(nxt): continue
    vr = float(fb.iloc[0].volume) / max(pre.volume.mean(), 1)
    early_exit = float(nxt.iloc[0].open) * 0.999          # sell at the open of the bar after the fill minute
    rows.append(dict(symbol=r.symbol, date=r.date, split='TRAIN' if r.date < '2026-01-01' else 'VAL' if r.date < '2026-06-01' else 'TEST',
                     R=r.R, vol_ft=vr, gap=gap(r.symbol, r.date), R_early=(early_exit - r.entry_price) / r.R_ps))
V = pd.DataFrame(rows)
L = ['# Follow-through volume on the honest BF cache (raw detector, 896 detections)', '', f'rows with bars {len(V)} of {len(c)}', '']
for sp in ('TRAIN', 'VAL', 'TEST', 'ALL'):
    x = V if sp == 'ALL' else V[V.split == sp]
    hi = x[x.vol_ft >= 1.5]; lo = x[x.vol_ft < 1.5]; g = x[x.gap >= 2]; ng = x[x.gap < 2]; both = x[(x.vol_ft >= 1.5) & (x.gap >= 2)]
    rule = x.copy(); rule['R_rule'] = np.where(rule.vol_ft >= 1.5, rule.R, rule.R_early)   # exit low-follow-through at the next open
    L.append(f'{sp}: n {len(x)} meanR {x.R.mean():+.3f} WR {(x.R > 0).mean():.0%} | vol>=1.5x n {len(hi)} {hi.R.mean():+.3f} WR {(hi.R > 0).mean():.0%} | vol<1.5x n {len(lo)} {lo.R.mean():+.3f} WR {(lo.R > 0).mean():.0%} '
             f'| gap>=2 n {len(g)} {g.R.mean():+.3f} | gap<2 n {len(ng)} {ng.R.mean():+.3f} | both n {len(both)} {both.R.mean():+.3f} '
             f'| EXIT-RULE book (all trades, low-vol sold at next open): meanR {rule.R_rule.mean():+.3f} vs {x.R.mean():+.3f}, sum {rule.R_rule.sum():+.1f} vs {x.R.sum():+.1f}')
V.to_csv('research/fuckup_audit/bf_cache_followthrough.csv', index=False)
# ---------- (2) ORB time stop on the real trades
P = pd.read_csv('research/fuckup_audit/live_loser_paths.csv', keep_default_na=False, na_values=['']); T = pd.read_csv('research/fuckup_audit/live_trades_dump.csv', keep_default_na=False, na_values=[''])
o = T[T.strategy == 'orb'].merge(P[P.strategy == 'orb'][['day', 'sym', 'fill_m']], left_on=['trade_date', 'symbol'], right_on=['day', 'sym'])
res = []
for r in o.itertuples():
    b = bars(r.symbol, r.trade_date)
    stop = r.real_stop_loss_price if r.real_stop_loss_price == r.real_stop_loss_price and r.real_stop_loss_price else r.stop_loss_price
    R = r.fill_price - stop
    if not len(b) or not (R and R > 0): continue
    tm = int(r.fill_m) + 10; at = b[b.m == tm]
    if not len(at): continue
    ex = pd.to_datetime(r.exited_at, utc=True).tz_convert(ET); xm = ex.hour * 60 + ex.minute
    if xm <= tm: res.append(dict(R_real=r.R, R_ts=r.R)); continue      # already exited before the check
    prog = (float(at.iloc[0].open) - r.fill_price) / R
    res.append(dict(R_real=r.R, R_ts=(prog - 0.003) if prog < 0.25 else r.R))
Q = pd.DataFrame(res)
L += ['', '# ORB 10-minute time stop on the real trades (exit at fill+10 min open if below +0.25R)', f'n {len(Q)} real {Q.R_real.sum():+.1f}R ({Q.R_real.mean():+.3f}) -> time-stop {Q.R_ts.sum():+.1f}R ({Q.R_ts.mean():+.3f}); trades cut early {(Q.R_ts != Q.R_real).sum()}',
      '', 'cells looked at in this file: BF 6 buckets x 4 splits + 1 exit rule; ORB 1 rule (0.25R at 10 min).']
open('research/fuckup_audit/live_followthrough.md', 'w').write('\n'.join(L)); print('\n'.join(L))
