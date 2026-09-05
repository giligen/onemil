"""Ignition — BT vs SHADOW vs LIVE on the live window (2026-09-05).

Three layers for every (day, symbol) from 2026-08-21 (first live trade):
  LIVE    data/trades.db strategy='ignition' (fill, stop, exit, pnl, R realized)
  SHADOW  logs/ignition_shadow_<day>.jsonl SHADOW_TRIGGER records (hypo entry/
          stop, catalyst) — selection parity by construction with live
  BT      capsim trades_LIVEWIN_annotated.csv (chase-entry baseline, $3K-risk
          model) + resting_LIVEWIN.csv (resting cap300 fill model)
Sizing-free comparison = R multiples (pnl / (shares * (fill - stop))); dollar
comparison rescales the BT model to the live $50 risk per trade.

Usage: python3 live_window_compare.py [START] [END]
"""
import glob
import json
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
sys.path.insert(0, ROOT)
from trading.orb_csv import read_orb_csv  # noqa: E402

D = f'{ROOT}/research/ignition_capcheck'
START = sys.argv[1] if len(sys.argv) > 1 else '2026-08-21'
END = sys.argv[2] if len(sys.argv) > 2 else '2026-09-04'
LIVE_RISK = 50.0


def live_trades() -> pd.DataFrame:
    con = sqlite3.connect(f'file:{ROOT}/data/trades.db?mode=ro', uri=True)
    t = pd.read_sql("SELECT trade_date AS day, symbol, shares, fill_price, entry_price, "
                    "stop_loss_price, real_stop_loss_price, exit_price, exit_reason, pnl "
                    "FROM trades WHERE strategy='ignition' AND trade_date BETWEEN ? AND ? "
                    "ORDER BY trade_date, symbol", con, params=[START, END])
    stop = t['real_stop_loss_price'].where(t['real_stop_loss_price'].notna(), t['stop_loss_price'])
    fill = t['fill_price'].where(t['fill_price'].notna(), t['entry_price'])
    t['live_fill'] = fill
    t['live_stop'] = stop
    t['live_R'] = t['pnl'] / (t['shares'] * (fill - stop)).replace(0, np.nan)
    return t


def shadow_triggers() -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(f'{ROOT}/logs/ignition_shadow_*.jsonl')):
        day = p.split('_')[-1][:10]
        if not (START <= day <= END):
            continue
        for line in open(p):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get('verdict') == 'SHADOW_TRIGGER':
                rows.append({'day': r['day'], 'symbol': r['symbol'], 'sh_entry': r.get('hypo_entry') or r.get('_entry'),
                             'sh_stop': r.get('hypo_stop') or r.get('_stop'), 'sh_trig_m': r.get('trigger_m'),
                             'sh_r_pct': r.get('r_pct'), 'catalyst': r.get('catalyst'), 'cohort': r.get('anchor_cohort'),
                             'spread_bps': r.get('spread_bps'), 'ask': r.get('ask')})
    return pd.DataFrame(rows).drop_duplicates(['day', 'symbol'], keep='first')


def main() -> None:
    live, sh = live_trades(), shadow_triggers()
    bt = read_orb_csv(f'{D}/trades_LIVEWIN_annotated.csv')
    bt = bt[(bt['day'] >= START) & (bt['day'] <= END)].copy()
    rs = read_orb_csv(f'{D}/resting_LIVEWIN.csv')
    rs = rs[(rs['day'] >= START) & (rs['day'] <= END)]
    bt = bt.merge(rs[['day', 'symbol', 'k300_cls', 'k300_fill', 'k300_rr0', 'k300_pnl10', 'k300_pos']],
                  on=['day', 'symbol'], how='left')
    # rescale the $3K-risk model to the live $50 risk (caps re-applied)
    pos50 = np.minimum.reduce([LIVE_RISK / (bt['r_pct'] / 100.0), np.full(len(bt), 25000.0), 0.15 * bt['bar_dollar']])
    bt['bt_pnl50'] = bt['pnl'] * pos50 / bt['pos'].replace(0, np.nan)
    bt['rest_pnl50'] = bt['k300_pnl10'] * pos50 / bt['k300_pos'].replace(0, np.nan)

    print(f"window {START}..{END}: LIVE {len(live)} trades ${live['pnl'].sum():+,.0f} | "
          f"SHADOW triggers {len(sh)} | BT triggers {len(bt)} (complex-confirmed {int(bt['complex_conf'].astype(str).eq('True').sum())})")
    m = live.merge(sh, on=['day', 'symbol'], how='left').merge(
        bt[['day', 'symbol', 'trig_m', 'entry', 'stop', 'r_pct', 'rr', 'reason', 'complex_conf', 'k300_cls', 'k300_fill', 'k300_rr0', 'bt_pnl50', 'rest_pnl50']],
        on=['day', 'symbol'], how='left')
    m['in_shadow'] = m['sh_entry'].notna()
    m['in_bt'] = m['entry'].notna()
    m['fill_vs_bt_bps'] = (m['live_fill'] / m['entry'] - 1) * 1e4
    m['fill_vs_shadow_bps'] = (m['live_fill'] / m['sh_entry'] - 1) * 1e4
    cols = ['day', 'symbol', 'in_shadow', 'in_bt', 'catalyst', 'live_fill', 'entry', 'k300_fill', 'fill_vs_bt_bps',
            'live_R', 'rr', 'k300_rr0', 'pnl', 'bt_pnl50', 'rest_pnl50', 'exit_reason', 'reason', 'k300_cls']
    pd.set_option('display.width', 250)
    print("\nPER LIVE TRADE (entry = BT chase model, k300_fill = resting model; R = per-R multiples):")
    print(m[cols].round(3).to_string(index=False))
    mb = m[m['in_bt']]
    print(f"\nmatched in BT: {len(mb)}/{len(live)} | live ${mb['pnl'].sum():+,.0f} vs BT-chase(@$50) ${mb['bt_pnl50'].sum():+,.0f} "
          f"vs resting(@$50, s10) ${mb['rest_pnl50'].sum():+,.0f} | mean R live {mb['live_R'].mean():+.2f} vs BT {mb['rr'].mean():+.2f} "
          f"vs resting {mb['k300_rr0'].mean():+.2f} | live fill vs BT entry median {mb['fill_vs_bt_bps'].median():+.0f} bps")
    print(f"live fill vs shadow hypo entry: median {m['fill_vs_shadow_bps'].median():+.0f} bps, p90 {m['fill_vs_shadow_bps'].quantile(.9):+.0f} bps")
    not_live = bt[[k not in set(zip(live['day'], live['symbol'])) for k in zip(bt['day'], bt['symbol'])]]
    shk = set(zip(sh['day'], sh['symbol']))
    print(f"\nBT triggers NOT traded live: {len(not_live)} (in shadow journal as trigger: {sum(k in shk for k in zip(not_live['day'], not_live['symbol']))}) "
          f"| their BT pnl@$50 ${not_live['bt_pnl50'].sum():+,.0f} | complex-confirmed among them: "
          f"{int(not_live['complex_conf'].astype(str).eq('True').sum())} (${not_live.loc[not_live['complex_conf'].astype(str).eq('True'), 'bt_pnl50'].sum():+,.0f})")
    print(not_live.sort_values('bt_pnl50')[['day', 'symbol', 'trig_m', 'rr', 'bt_pnl50', 'complex_conf', 'k300_cls']].to_string(index=False))
    m.to_csv(f'{D}/live_window_compare.csv', index=False)


if __name__ == '__main__':
    main()
