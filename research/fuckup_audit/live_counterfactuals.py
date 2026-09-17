#!/usr/bin/env python3
"""Counterfactuals on the REAL live trades (ORB + bull flag): what would the P&L have been with
 (a) a CLOSE-based stop (exit at the next bar's open after a 1-min bar CLOSES below the stop),
 (b) a buffered touch stop (stop x 0.995),
 (c) a breakout-volume gate (skip trades whose breakout bar volume < 1.5x the prior 5-bar mean),
 (d) a gap gate (skip non-gappers, gap < +2%),
 each with the original exits otherwise (target/lock/force-close kept where they happened first). Real fills, real
 stops from trades.db; bars from cache.db. Costs: the real trades already carry their costs; counterfactual exits are
 filled at the bar open x 0.999 (stop) — conservative. Output: research/fuckup_audit/live_counterfactuals.md"""
import sqlite3, numpy as np, pandas as pd
from zoneinfo import ZoneInfo
ET = ZoneInfo('America/New_York')
pd.set_option('display.width', 250)
P = pd.read_csv('research/fuckup_audit/live_loser_paths.csv', keep_default_na=False, na_values=[''])
T = pd.read_csv('research/fuckup_audit/live_trades_dump.csv', keep_default_na=False, na_values=[''])
cache = sqlite3.connect('file:data/cache.db?mode=ro', uri=True)


def bars(sym, day):
    b = pd.read_sql("select timestamp, open, high, low, close, volume from intraday_bars_1min where symbol=? and bar_date=? order by timestamp", cache, params=(sym, day))
    if not len(b): return b
    ts = pd.to_datetime(b.timestamp, utc=True).dt.tz_convert(ET); b['m'] = ts.dt.hour * 60 + ts.dt.minute
    return b[(b.m >= 570) & (b.m < 960)].reset_index(drop=True)


def rewalk(b, fm, xm, fill, stop, R, orig_exit_reason, orig_exit_price, mode):
    """Walk from the bar after the fill. Returns the counterfactual exit price. Non-stop original exits (target, lock,
    tag_bb, force_close) are honoured at their original minute if they come first."""
    after = b[b.m > fm]
    stop_cf = stop * 0.995 if mode == 'buffer' else stop
    below_close = False
    for r in after.itertuples():
        if xm is not None and r.m >= xm and 'stop' not in str(orig_exit_reason):
            return orig_exit_price                                   # the original non-stop exit happened here
        if mode == 'close':
            if below_close:                                          # exit at this bar's open after a close below the stop
                return min(float(r.open), stop) * 0.999 if float(r.open) > stop * 0.98 else float(r.open) * 0.999
            if float(r.close) < stop: below_close = True
        else:
            if float(r.low) <= stop_cf:
                return min(stop_cf, float(r.open)) * 0.999
    last = after[after.m >= 955]
    return float(last.iloc[0].open) if len(last) else float(after.iloc[-1].close) if len(after) else fill


rows = []
for r in T.itertuples():
    p = P[(P.day == r.trade_date) & (P.sym == r.symbol) & (P.strategy == r.strategy)]
    if not len(p) or r.R != r.R: continue
    p = p.iloc[0]
    stop = r.real_stop_loss_price if r.real_stop_loss_price == r.real_stop_loss_price and r.real_stop_loss_price else r.stop_loss_price
    fill = r.fill_price; R = fill - stop
    if not (R and R > 0): continue
    b = bars(r.symbol, r.trade_date)
    if not len(b): continue
    fm = int(p.fill_m); xm = None
    if isinstance(r.exited_at, str):
        ex = pd.to_datetime(r.exited_at, utc=True).tz_convert(ET); xm = ex.hour * 60 + ex.minute
    o = dict(strategy=r.strategy, day=r.trade_date, sym=r.symbol, R_real=r.R, pnl_real=r.pnl, exit=r.exit_reason, vol_break=p.vol_break_vs_prior5, gap=p.gap_pct, mfe=p.mfe_R)
    for mode in ('close', 'buffer'):
        xp = rewalk(b, fm, xm, fill, stop, R, r.exit_reason, r.exit_price, mode)
        o[f'R_{mode}'] = (xp - fill) / R
    rows.append(o)
C = pd.DataFrame(rows)
L = ['# Counterfactuals on the real live trades', '']
for s in ('orb', 'bull_flag'):
    x = C[C.strategy == s]
    def book(y, lab):
        return dict(rule=lab, n=len(y), R_real=round(y.R_real.sum(), 1), meanR_real=round(y.R_real.mean(), 3), WR_real=round((y.R_real > 0).mean(), 2),
                    R_closestop=round(y.R_close.sum(), 1), meanR_closestop=round(y.R_close.mean(), 3), R_buffer=round(y.R_buffer.sum(), 1), meanR_buffer=round(y.R_buffer.mean(), 3))
    rows2 = [book(x, 'all trades'),
             book(x[x.vol_break >= 1.5], 'vol_break >= 1.5x'), book(x[x.vol_break < 1.5], 'vol_break < 1.5x (would be skipped)'),
             book(x[x.gap >= 2], 'gap >= +2%'), book(x[x.gap < 2], 'gap < +2% (would be skipped)'),
             book(x[(x.vol_break >= 1.5) & (x.gap >= 2)], 'vol >= 1.5x AND gap >= 2%')]
    L += [f'## {s} ({len(x)} real trades)', pd.DataFrame(rows2).to_string(index=False), '']
    st = x[x.exit.astype(str).str.contains('stop')]
    L += [f'stopped trades: {len(st)} — real {st.R_real.sum():.1f}R, close-stop {st.R_close.sum():.1f}R, buffer {st.R_buffer.sum():.1f}R; '
          f'close-stop turns {(st.R_close > 0).sum()} of them positive, buffer {(st.R_buffer > 0).sum()}', '']
open('research/fuckup_audit/live_counterfactuals.md', 'w').write('\n'.join(L)); C.to_csv('research/fuckup_audit/live_counterfactuals.csv', index=False)
print('\n'.join(L))
# ORB summary that scrolled off
x = P[P.strategy == 'orb']; x['win'] = x.R > 0
num = ['stop_pct', 'slip_bps', 'gap_pct', 'pdr_pct', 'open_to_fill_pct', 'mfe_R', 'mfe_min', 'mae_R', 'close_R', 'vol_break_vs_prior5', 'or5_range_pct', 'spy_open_to_fill', 'spy_open_to_1030', 'spy_open_to_close']
print('\n===== orb: losers vs winners (median)'); print(x.groupby('win')[num].median().round(2).T.to_string())
st = x[x.exit.astype(str).str.contains('stop')]
print(f"orb stopped: n={len(st)} wick {st.stopbar_close_above_stop.mean():.0%} recover60 {st.recover_above_fill_60m.mean():.0%} close>fill {(st.close_R > 0).mean():.0%} mfe>=0.5R {(st.mfe_R >= 0.5).mean():.0%} mfe>=1R {(st.mfe_R >= 1).mean():.0%}")
print(f"orb hold-to-close counterfactual: losers {x[~x.win].close_R.mean():.2f} winners {x[x.win].close_R.mean():.2f} all {x.close_R.mean():.2f}")
