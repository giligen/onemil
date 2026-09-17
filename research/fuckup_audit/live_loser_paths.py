#!/usr/bin/env python3
"""Bottom-up on the REAL losers: every closed live ORB / bull-flag trade, its bars, its path, its day.
Reads data/trades.db (ro), data/cache.db intraday_bars_1min + daily_bars (ro), research/lit_review_2026/etf_1min.db (SPY).
Writes research/fuckup_audit/live_loser_paths.csv and prints the loser-vs-winner picture."""
import sqlite3, numpy as np, pandas as pd
from zoneinfo import ZoneInfo
ET = ZoneInfo('America/New_York')
pd.set_option('display.width', 260); pd.set_option('display.max_rows', 300); pd.set_option('display.max_columns', 60)
t = pd.read_csv('research/fuckup_audit/live_trades_dump.csv', keep_default_na=False, na_values=[''])
cache = sqlite3.connect('file:data/cache.db?mode=ro', uri=True); etf = sqlite3.connect('file:research/lit_review_2026/etf_1min.db?mode=ro', uri=True)


def bars(sym, day):
    b = pd.read_sql("select timestamp, open, high, low, close, volume from intraday_bars_1min where symbol=? and bar_date=? order by timestamp", cache, params=(sym, day))
    if not len(b): return b
    ts = pd.to_datetime(b.timestamp, utc=True).dt.tz_convert(ET); b['m'] = ts.dt.hour * 60 + ts.dt.minute
    return b[(b.m >= 570) & (b.m < 960)].reset_index(drop=True)


def spy(day):
    b = pd.read_sql("select t, o, h, l, c from bars where symbol='SPY' and t >= ? and t < ?", etf, params=(f'{day}T00:00:00', f'{day}T23:59:59'))
    if not len(b): return b
    ts = pd.to_datetime(b.t, utc=True).dt.tz_convert(ET); b['m'] = ts.dt.hour * 60 + ts.dt.minute
    return b[(b.m >= 570) & (b.m < 960)].reset_index(drop=True)


def daily(sym, day):
    d = pd.read_sql("select bar_date, open, high, low, close from daily_bars where symbol=? and bar_date<=? order by bar_date desc limit 2", cache, params=(sym, day))
    return d


rows = []
for r in t.itertuples():
    day = r.trade_date; b = bars(r.symbol, day)
    fa = pd.to_datetime(r.filled_at, utc=True).tz_convert(ET); fm = fa.hour * 60 + fa.minute
    ex = pd.to_datetime(r.exited_at, utc=True).tz_convert(ET) if isinstance(r.exited_at, str) else None; xm = ex.hour * 60 + ex.minute if ex is not None else None
    stop = r.real_stop_loss_price if r.real_stop_loss_price == r.real_stop_loss_price and r.real_stop_loss_price else r.stop_loss_price
    fill = r.fill_price; R = fill - stop if stop and fill else np.nan
    d = daily(r.symbol, day); gap = pdr = None
    if len(d) == 2:
        gap = (d.iloc[0].open / d.iloc[1].close - 1) * 100; pdr = (d.iloc[1].high - d.iloc[1].low) / d.iloc[1].low * 100
    o = dict(strategy=r.strategy, day=day, sym=r.symbol, fill_m=fm, fill_et=r.fill_et, fill=fill, stop_pct=round(r.stop_pct, 2), slip_bps=round(r.fill_slip_bps, 0),
             exit=r.exit_reason, hold_min=round(r.hold_min, 0), pnl=round(r.pnl, 0), R=round(r.R, 2), gap_pct=None if gap is None else round(gap, 1), pdr_pct=None if pdr is None else round(pdr, 1))
    if len(b) and R == R and R > 0:
        after = b[b.m > fm]; day_open = float(b.iloc[0].open)
        o['open_to_fill_pct'] = round((fill / day_open - 1) * 100, 1)
        if len(after):
            upto = after if xm is None else after[after.m <= xm]
            if len(upto):
                o['mfe_R'] = round((upto.high.max() - fill) / R, 2); o['mfe_min'] = int(upto.m[upto.high.idxmax()] - fm)
                o['mae_R'] = round((fill - upto.low.min()) / R, 2)
            o['close_R'] = round((float(b.iloc[-1].close) - fill) / R, 2)            # hold-to-close counterfactual
            o['hi_after_exit_R'] = round((after[after.m > (xm or fm)].high.max() - fill) / R, 2) if xm and len(after[after.m > xm]) else None
            if xm and 'stop' in str(r.exit_reason):
                xb = b[b.m == xm]
                if len(xb): o['stopbar_close_above_stop'] = bool(float(xb.iloc[0].close) > stop)
                post = after[(after.m > xm) & (after.m <= xm + 60)]
                if len(post): o['recover_above_fill_60m'] = bool(post.high.max() > fill)
        pre = b[b.m < fm]
        if len(pre) >= 5:
            o['vol_break_vs_prior5'] = round(float(b[b.m == fm].volume.sum() or 0) / max(pre.tail(5).volume.mean(), 1), 1)
        rng = b[(b.m >= 570) & (b.m < 575)]
        if len(rng): o['or5_range_pct'] = round((rng.high.max() - rng.low.min()) / rng.low.min() * 100, 1)
    s = spy(day)
    if len(s):
        so = float(s.iloc[0].o); sf = s[s.m <= fm]; s1030 = s[s.m <= 630]
        o['spy_open_to_fill'] = round((float(sf.iloc[-1].c) / so - 1) * 100, 2) if len(sf) else None
        o['spy_open_to_1030'] = round((float(s1030.iloc[-1].c) / so - 1) * 100, 2) if len(s1030) else None
        o['spy_open_to_close'] = round((float(s.iloc[-1].c) / so - 1) * 100, 2)
    rows.append(o)
P = pd.DataFrame(rows); P.to_csv('research/fuckup_audit/live_loser_paths.csv', index=False)
P['win'] = P.R > 0
num = ['stop_pct', 'slip_bps', 'gap_pct', 'pdr_pct', 'open_to_fill_pct', 'mfe_R', 'mfe_min', 'mae_R', 'close_R', 'vol_break_vs_prior5', 'or5_range_pct', 'spy_open_to_fill', 'spy_open_to_1030', 'spy_open_to_close', 'hold_min']
for strat in ('orb', 'bull_flag'):
    x = P[P.strategy == strat]
    print(f'\n===== {strat}: losers vs winners (median | mean)')
    g = x.groupby('win')[num].agg(['median', 'mean']).round(2).T
    print(g.to_string())
    st = x[x.exit.astype(str).str.contains('stop')]
    print(f'\n-- {strat} stopped trades: stop bar closed back above stop {st.stopbar_close_above_stop.mean():.0%} (n={st.stopbar_close_above_stop.notna().sum()}), '
          f'recovered above fill within 60m {st.recover_above_fill_60m.mean():.0%}, day close above fill {(st.close_R > 0).mean():.0%}, '
          f'had >= +0.5R before the stop {(st.mfe_R >= 0.5).mean():.0%}, >= +1R {(st.mfe_R >= 1).mean():.0%}')
    print(f'-- hold-to-close counterfactual: mean close_R losers {x[~x.win].close_R.mean():.2f}, winners {x[x.win].close_R.mean():.2f}, all {x.close_R.mean():.2f}')
    print('\n-- every loser, worst first:')
    print(x[~x.win].sort_values('pnl')[['day', 'sym', 'fill_et', 'stop_pct', 'slip_bps', 'gap_pct', 'pdr_pct', 'open_to_fill_pct', 'or5_range_pct', 'vol_break_vs_prior5', 'mfe_R', 'mfe_min', 'exit', 'hold_min', 'stopbar_close_above_stop', 'recover_above_fill_60m', 'close_R', 'spy_open_to_fill', 'spy_open_to_close', 'R']].to_string(index=False))
