"""Root-cause the BT-LIVE divergence: do they detect the same MACD cross?

For each LIVE trade, look up the corresponding BT cache entry (if any) and
compare the cross_time_min and macd_hist_pct. If they differ, the two
systems are detecting DIFFERENT signals on the same day.
"""
import json
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
LIVE_DB = ROOT / 'data' / 'trades.db'
BT_CACHE = ROOT / 'data' / 'macd_signal_cache_t30_s40.csv'


def load_live() -> pd.DataFrame:
    conn = sqlite3.connect(str(LIVE_DB))
    df = pd.read_sql("""
        SELECT trade_date as date, symbol, fill_price, exit_price, exit_reason,
               COALESCE(pnl, 0) as pnl, pattern_data, filled_at
          FROM trades
         WHERE strategy='macd_wave' AND exit_price IS NOT NULL
           AND fill_price IS NOT NULL
           AND trade_date BETWEEN '2026-03-30' AND '2026-05-06'
    """, conn)
    conn.close()
    for c in ['live_cross_min', 'live_macd_pct', 'live_vol_at_cross']:
        df[c] = pd.NA
    for i, r in df.iterrows():
        if not r['pattern_data']: continue
        try:
            d = json.loads(r['pattern_data'])
            df.at[i, 'live_cross_min'] = d.get('cross_time_min')
            df.at[i, 'live_macd_pct'] = d.get('macd_hist_pct')
            df.at[i, 'live_vol_at_cross'] = d.get('vol_at_cross')
        except Exception:
            pass
    return df


def main():
    live = load_live()
    bt = pd.read_csv(BT_CACHE)
    bt['date'] = bt['date'].astype(str)

    # Aggregate BT cache: per (symbol,date) keep the EARLIEST cross signal
    bt_agg = bt.sort_values(['symbol','date','cross_time_min']).groupby(['symbol','date']).first().reset_index()
    bt_agg = bt_agg.rename(columns={'cross_time_min':'bt_cross_min',
                                     'macd_hist_pct':'bt_macd_pct',
                                     'vol_at_cross':'bt_vol_at_cross',
                                     'pnl_dollar':'bt_pnl', 'exit_reason':'bt_exit',
                                     'entry_price':'bt_entry'})

    merged = live.merge(bt_agg[['symbol','date','bt_cross_min','bt_macd_pct',
                                 'bt_vol_at_cross','bt_pnl','bt_exit','bt_entry']],
                         on=['symbol','date'], how='left')

    # Compute mismatch
    merged['cross_diff'] = merged['bt_cross_min'].astype(float) - merged['live_cross_min'].astype(float)
    merged['macd_diff'] = merged['bt_macd_pct'].astype(float) - merged['live_macd_pct'].astype(float)
    merged['in_bt_cache'] = merged['bt_cross_min'].notna()

    n = len(merged)
    in_cache = merged['in_bt_cache'].sum()
    not_in_cache = n - in_cache

    print(f"Total LIVE closed trades 3/30-5/6: {n}")
    print(f"  In BT cache (universe matched):     {in_cache}")
    print(f"  NOT in BT cache (universe missed):  {not_in_cache}")
    print()

    # Of those in cache, how many have matching cross detection?
    in_both = merged[merged['in_bt_cache']].copy()
    if len(in_both):
        # Mismatch defined as cross_diff > 5 min OR macd_diff > 0.2%
        in_both['cross_match'] = (in_both['cross_diff'].abs() <= 5) | in_both['cross_diff'].isna()
        in_both['macd_match'] = (in_both['macd_diff'].abs() <= 0.2) | in_both['macd_diff'].isna()
        in_both['signals_match'] = in_both['cross_match'] & in_both['macd_match']

        match = in_both['signals_match'].sum()
        mismatch = len(in_both) - match
        print(f"Of the {len(in_both)} LIVE trades in BT cache:")
        print(f"  Same-signal detection (cross±5min, macd±0.2%):  {match}")
        print(f"  DIFFERENT signal detection:                      {mismatch}")
        print()

        # What was BT's projected outcome on those mismatch trades?
        mm = in_both[~in_both['signals_match']].copy()
        if len(mm):
            print(f"Top 15 worst LIVE outcomes where signals MISMATCH:")
            print(f"{'date':>11} {'sym':>6} {'L_cross':>8} {'B_cross':>8} {'L_macd%':>8} {'B_macd%':>8} {'L_pnl':>9} {'B_pnl':>9} {'B_filt?':>8}")
            for _, r in mm.sort_values('pnl').head(15).iterrows():
                bt_passed = (float(r['bt_cross_min']) <= 10 and 0 < float(r['bt_vol_at_cross']) < 300000
                              and float(r['bt_macd_pct']) >= 0.5)
                # but BT requires above filters; for the row to be in BT cache, only that BT generated a signal
                # for it. The actual filter pass is at backtest time. Here we evaluate would it pass.
                pass_str = 'PASS' if bt_passed else 'SKIP'
                print(f"{r['date']:>11} {r['symbol']:>6} "
                      f"{float(r['live_cross_min']):>8.1f} {float(r['bt_cross_min']):>8.1f} "
                      f"{float(r['live_macd_pct']):>8.3f} {float(r['bt_macd_pct']):>8.3f} "
                      f"${float(r['pnl']):>+8,.0f} ${float(r['bt_pnl']):>+8,.0f} {pass_str:>8}")

        # Why was each mismatch trade NOT picked by BT proper?
        print()
        print("BT-skip reasons for mismatch trades (would BT take this signal?)")
        mm['bt_pass_filters'] = (
            (mm['bt_cross_min'].astype(float) <= 10) &
            (mm['bt_vol_at_cross'].astype(float) < 300000) &
            (mm['bt_macd_pct'].astype(float) >= 0.5)
        )
        n_bt_skip = (~mm['bt_pass_filters']).sum()
        n_bt_take = mm['bt_pass_filters'].sum()
        print(f"  Mismatch trades where BT would have ALSO entered: {n_bt_take}")
        print(f"  Mismatch trades where BT would have REJECTED:     {n_bt_skip}")
        if n_bt_skip:
            mm_skipped = mm[~mm['bt_pass_filters']]
            # Why skipped
            for col, name, thresh, op in [('bt_cross_min','cross>10', 10, '>'),
                                            ('bt_vol_at_cross','vol>=300K', 300000, '>='),
                                            ('bt_macd_pct','macd<0.5', 0.5, '<')]:
                if op == '>':
                    cnt = (mm_skipped[col].astype(float) > thresh).sum()
                elif op == '>=':
                    cnt = (mm_skipped[col].astype(float) >= thresh).sum()
                elif op == '<':
                    cnt = (mm_skipped[col].astype(float) < thresh).sum()
                print(f"    {name}: {cnt} trades")

    # NOT IN CACHE — universe-level mismatch
    notc = merged[~merged['in_bt_cache']]
    if len(notc):
        print()
        print(f"LIVE trades NOT IN BT cache ({len(notc)} trades) — universe mismatch:")
        print(f"  Sum P&L of these: ${notc['pnl'].sum():+,.0f}")
        print(f"  Worst 10:")
        for _, r in notc.sort_values('pnl').head(10).iterrows():
            entry = r['fill_price']
            print(f"    {r['date']} {r['symbol']:>6}  entry=${entry:>6.2f}  "
                  f"L_cross={r['live_cross_min']}  L_macd={r['live_macd_pct']:.3f}  "
                  f"L_pnl=${r['pnl']:>+8,.0f}  exit={r['exit_reason']}")

    merged.to_csv(ROOT/'analysis_results'/'macd_drift_root_cause.csv', index=False)
    print(f"\nSaved: analysis_results/macd_drift_root_cause.csv ({len(merged)} rows)")


if __name__ == '__main__':
    main()
