"""Fresh-eyes review helper: the cuts of the 169 real live trades the audit program never printed.

Read-only. Inputs: research/fuckup_audit/live_trades_dump.csv + live_loser_paths.csv (already dumped from
data/trades.db). Every CSV read with keep_default_na=False. Output: stdout only.
"""
import json
import sys
import pandas as pd
import numpy as np

D = '/home/ec2-user/onemil/research/fuckup_audit/'
t = pd.read_csv(D + 'live_trades_dump.csv', keep_default_na=False, na_values=[''])
p = pd.read_csv(D + 'live_loser_paths.csv', keep_default_na=False, na_values=[''])
t = t[t.strategy.isin(['bull_flag', 'orb'])].copy()
t['pnl'] = pd.to_numeric(t.pnl, errors='coerce')
t['R'] = pd.to_numeric(t.R, errors='coerce')
t['total_risk'] = pd.to_numeric(t.total_risk, errors='coerce')
t['fill_slip_bps'] = pd.to_numeric(t.fill_slip_bps, errors='coerce')
t['stop_pct'] = pd.to_numeric(t.stop_pct, errors='coerce')
t['hold_min'] = pd.to_numeric(t.hold_min, errors='coerce')
t['shares'] = pd.to_numeric(t.shares, errors='coerce')
t['fill_price'] = pd.to_numeric(t.fill_price, errors='coerce')
t['d'] = pd.to_datetime(t.trade_date)
t['month'] = t.d.dt.strftime('%Y-%m')
t['dow'] = t.d.dt.day_name().str[:3]
t['hour'] = t.fill_et.str.slice(0, 2)
t['notional'] = t.shares * t.fill_price

def blk(title):
    print('\n## ' + title)

for s, g in t.groupby('strategy'):
    blk(f'{s}: n={len(g)} pnl={g.pnl.sum():.0f} sumR={g.R.sum():.1f} meanR={g.R.mean():.3f} WR={(g.pnl>0).mean():.2f} '
        f'median risk={g.total_risk.median():.0f} median notional={g.notional.median():.0f}')
    print('by month:')
    print(g.groupby('month').agg(n=('pnl', 'size'), pnl=('pnl', 'sum'), R=('R', 'sum'), risk=('total_risk', 'median')).round(1).to_string())
    print('by day of week:')
    print(g.groupby('dow').agg(n=('pnl', 'size'), pnl=('pnl', 'sum'), meanR=('R', 'mean')).round(2).to_string())
    print('by fill hour (ET):')
    print(g.groupby('hour').agg(n=('pnl', 'size'), pnl=('pnl', 'sum'), meanR=('R', 'mean'), WR=('pnl', lambda x: (x > 0).mean())).round(2).to_string())
    print('by exit reason:')
    print(g.groupby('exit_reason').agg(n=('pnl', 'size'), pnl=('pnl', 'sum'), meanR=('R', 'mean'), hold=('hold_min', 'median')).round(2).to_string())
    # sequence within day
    g = g.sort_values(['trade_date', 'filled_at'])
    g['seq'] = g.groupby('trade_date').cumcount() + 1
    print('by sequence within the day (1 = first fill of the day):')
    print(g.groupby(g.seq.clip(upper=4)).agg(n=('pnl', 'size'), pnl=('pnl', 'sum'), meanR=('R', 'mean'), WR=('pnl', lambda x: (x > 0).mean())).round(2).to_string())
    # after a loser earlier the same day
    g['prev_day_pnl'] = g.groupby('trade_date').pnl.transform(lambda x: x.shift(1).fillna(0).cumsum())
    print('trades taken when the day was already red / not red:')
    print(g.groupby(g.prev_day_pnl < 0).agg(n=('pnl', 'size'), pnl=('pnl', 'sum'), meanR=('R', 'mean')).round(2).to_string())
    # slippage as P&L component
    slip_usd = (g.fill_slip_bps / 1e4 * g.notional)
    print(f'entry slip vs planned level: median {g.fill_slip_bps.median():.0f} bps, total ${slip_usd.sum():.0f} on pnl ${g.pnl.sum():.0f}; '
          f'slip/risk median {(slip_usd / g.total_risk).median():.2f}R')
    # stop distance vs realized
    print(f'stop_pct: median {g.stop_pct.median():.2f}% p10 {g.stop_pct.quantile(.1):.2f} p90 {g.stop_pct.quantile(.9):.2f}; '
          f'median hold {g.hold_min.median():.0f} min')
    # weekly
    g['wk'] = g.d.dt.to_period('W').astype(str)
    w = g.groupby('wk').pnl.sum()
    print(f'weeks: {len(w)} green {(w>0).mean():.2f} worst {w.min():.0f} best {w.max():.0f}; day-level: {g.groupby("trade_date").pnl.sum().lt(0).mean():.2f} red days')
    # concentration
    r = g.R.sort_values(ascending=False)
    print(f'top-3 winners R {r.head(3).round(2).tolist()} = {r.head(3).sum():.1f} of total {r.sum():.1f}; bottom-3 {r.tail(3).round(2).tolist()}')

blk('ORB by config era (B+ live from 2026-08-17; pre-0/stage before)')
o = t[t.strategy == 'orb'].copy()
o['era'] = np.where(o.d >= '2026-08-17', 'B+ (8/17+)', 'pre-B+')
print(o.groupby('era').agg(n=('pnl', 'size'), pnl=('pnl', 'sum'), R=('R', 'sum'), meanR=('R', 'mean'), WR=('pnl', lambda x: (x > 0).mean()), risk=('total_risk', 'median')).round(2).to_string())

blk('loser paths: SPY context of losing vs winning days (from live_loser_paths.csv)')
p['pnl'] = pd.to_numeric(p.pnl, errors='coerce')
for s, g in p.groupby('strategy'):
    dd = g.groupby('day').agg(pnl=('pnl', 'sum'), spy_oc=('spy_open_to_close', 'first'), spy_1030=('spy_open_to_1030', 'first'))
    print(f'{s}: days {len(dd)}; corr(day pnl, SPY open->close) = {dd.pnl.corr(dd.spy_oc):.2f}; '
          f'red days SPY o->c mean {dd[dd.pnl<0].spy_oc.mean():.2f}% vs green days {dd[dd.pnl>0].spy_oc.mean():.2f}%')
    lo = g[g.pnl < 0]; wi = g[g.pnl > 0]
    print(f'  losers n={len(lo)} median mfe_R {lo.mfe_R.median():.2f} at {lo.mfe_min.median():.0f} min; winners mfe {wi.mfe_R.median():.2f}; '
          f'losers vol_break {lo.vol_break_vs_prior5.median():.2f} winners {wi.vol_break_vs_prior5.median():.2f}; '
          f'losers gap {lo.gap_pct.median():.1f} winners {wi.gap_pct.median():.1f}; losers or5 {lo.or5_range_pct.median():.1f} winners {wi.or5_range_pct.median():.1f}')

blk('BF pattern_data: pole gain / retracement / price band of winners vs losers')
b = t[t.strategy == 'bull_flag'].copy()
pd_ = b.pattern_data.apply(lambda s: json.loads(s) if isinstance(s, str) and s.startswith('{') else {})
b['pole'] = pd_.apply(lambda d: d.get('pole_gain_pct'))
b['retr'] = pd_.apply(lambda d: d.get('retracement_pct'))
b['won'] = b.pnl > 0
print(b.groupby('won').agg(n=('pnl', 'size'), pole=('pole', 'median'), retr=('retr', 'median'), px=('fill_price', 'median'), stop=('stop_pct', 'median'), slip=('fill_slip_bps', 'median')).round(2).to_string())
print('BF by price band:')
print(b.groupby(pd.cut(b.fill_price, [0, 5, 10, 20, 100])).agg(n=('pnl', 'size'), pnl=('pnl', 'sum'), meanR=('R', 'mean')).round(2).to_string())
