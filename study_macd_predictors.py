"""MACD wave win/loss predictor analysis — exploratory + confirmatory.

GOAL
----
For each candidate predictor feature, ask: does it discriminate winners
from losers? Use both the LIVE trades table (real fills, n~65) and the
BT trail03 CSV (Jan 2025 - Mar 2026, n~1170) for triangulation.

A feature is a real predictor only if:
  1. Effect direction is consistent across LIVE and BT
  2. Bootstrap CI on Δmean_pnl(top_tertile − bottom_tertile) excludes zero
  3. Effect is monotonic across tertiles (top > mid > bottom or reverse)

Multi-hypothesis: testing N features, Bonferroni-correct (require p ~ 0.05/N).
With small samples, exploratory analysis only — never ship a filter purely
on this; require BT validation with walk-forward + production-faithful
implementation (per the H3-A lesson).

USAGE
-----
    python3 study_macd_predictors.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).parent
LIVE_DB = ROOT / 'data' / 'trades.db'
BT_CSV = ROOT / 'macd_wave_trail03.csv'


def load_live() -> pd.DataFrame:
    conn = sqlite3.connect(str(LIVE_DB))
    df = pd.read_sql("""
        SELECT id, trade_date, symbol, fill_price, exit_price, exit_reason,
               COALESCE(pnl, 0) as pnl, shares, news_catalyst,
               entry_quote_spread, exit_quote_spread,
               filled_at, exited_at, pattern_data
          FROM trades
         WHERE strategy='macd_wave' AND exit_price IS NOT NULL
           AND fill_price IS NOT NULL
         ORDER BY id
    """, conn)
    conn.close()
    # Parse pattern_data JSON
    for col in ['cross_time_min', 'vol_at_cross', 'macd_hist_pct',
                'conviction_mult', 'conv_cross_speed', 'conv_vol_at_cross']:
        df[col] = np.nan
    for i, row in df.iterrows():
        if not row['pattern_data']: continue
        try:
            d = json.loads(row['pattern_data'])
            for k in ['cross_time_min','vol_at_cross','macd_hist_pct',
                      'conviction_mult','conv_cross_speed','conv_vol_at_cross']:
                if k in d: df.at[i, k] = float(d[k])
        except Exception:
            continue
    df['win'] = (df['pnl'] > 0).astype(int)
    df['filled_at'] = pd.to_datetime(df['filled_at'], errors='coerce')
    df['minute_of_day'] = df['filled_at'].dt.hour * 60 + df['filled_at'].dt.minute
    df['weekday'] = df['filled_at'].dt.weekday
    df['entry_spread_bps'] = df['entry_quote_spread'] * 10000.0 / df['fill_price']
    df['notional'] = df['fill_price'] * df['shares']
    return df


def load_bt() -> pd.DataFrame:
    df = pd.read_csv(BT_CSV)
    df = df[df['exit_reason'].notna()].copy()
    df['win'] = (df['pnl_dollar'] > 0).astype(int)
    df['date'] = pd.to_datetime(df['date'])
    df['filled_at'] = pd.to_datetime(df['entry_time'], errors='coerce', utc=True)
    df['exited_at'] = pd.to_datetime(df['exit_time'], errors='coerce', utc=True)
    df['minute_of_day'] = df['filled_at'].dt.hour * 60 + df['filled_at'].dt.minute
    df['weekday'] = df['filled_at'].dt.weekday
    df['hold_min'] = (df['exited_at'] - df['filled_at']).dt.total_seconds() / 60.0
    df['fill_price'] = df['entry_price']
    df['pnl'] = df['pnl_dollar']
    return df


def tertile_summary(df: pd.DataFrame, feat: str, label: str = None) -> pd.DataFrame:
    """Split df into tertiles by feat, return per-tertile mean_pnl + WR + n."""
    label = label or feat
    s = df[feat].dropna()
    if len(s) < 12:
        return pd.DataFrame()
    cuts = [s.quantile(0.0), s.quantile(0.333), s.quantile(0.667), s.quantile(1.0)]
    cuts[-1] = cuts[-1] + 1e-9
    df = df[df[feat].notna()].copy()
    try:
        df['_tertile'] = pd.cut(df[feat], bins=cuts, labels=['T1_low','T2_mid','T3_high'],
                                 include_lowest=True, duplicates='drop')
    except ValueError:
        return pd.DataFrame()
    rows = []
    for t in ['T1_low','T2_mid','T3_high']:
        sub = df[df['_tertile'] == t]
        if len(sub) == 0: continue
        rows.append({
            'feature': label,
            'tertile': t,
            'n': len(sub),
            'mean_pnl': float(sub['pnl'].mean()),
            'sum_pnl': float(sub['pnl'].sum()),
            'wr_pct': float((sub['pnl'] > 0).mean() * 100),
            'range': f"[{sub[feat].min():.3f} – {sub[feat].max():.3f}]",
        })
    return pd.DataFrame(rows)


def bootstrap_ci(arr_a, arr_b, n_iter=2000, seed=42):
    if len(arr_a) == 0 or len(arr_b) == 0:
        return float('nan'), float('nan'), float('nan')
    rng = np.random.RandomState(seed)
    a = np.array(arr_a); b = np.array(arr_b); diffs = []
    for _ in range(n_iter):
        ar = rng.choice(a, size=len(a), replace=True)
        br = rng.choice(b, size=len(b), replace=True)
        diffs.append(ar.mean() - br.mean())
    diffs = np.sort(diffs)
    return float(np.mean(diffs)), float(diffs[int(0.025*n_iter)]), float(diffs[int(0.975*n_iter)])


def analyse_feature(df: pd.DataFrame, feat: str, df_label: str, label=None):
    label = label or feat
    s = df[feat].dropna()
    if len(s) < 12:
        return None
    df = df[df[feat].notna()].copy()
    cuts = [s.quantile(0.0), s.quantile(0.333), s.quantile(0.667), s.quantile(1.0)]
    cuts[-1] = cuts[-1] + 1e-9
    try:
        df['_t'] = pd.cut(df[feat], bins=cuts, labels=['T1','T2','T3'],
                           include_lowest=True, duplicates='drop')
    except ValueError:
        return None  # too few unique values for tertiles
    t1 = df[df['_t']=='T1']['pnl'].values
    t3 = df[df['_t']=='T3']['pnl'].values
    md, lo, hi = bootstrap_ci(t3, t1)
    sig = '*' if (lo < 0 and hi < 0) or (lo > 0 and hi > 0) else ' '
    return {
        'dataset': df_label,
        'feature': label,
        'n_total': len(df),
        'mean_T1': float(t1.mean()) if len(t1) else float('nan'),
        'mean_T3': float(t3.mean()) if len(t3) else float('nan'),
        'wr_T1': float((t1>0).mean()*100) if len(t1) else float('nan'),
        'wr_T3': float((t3>0).mean()*100) if len(t3) else float('nan'),
        'delta_T3_minus_T1': float(t3.mean() - t1.mean()) if len(t1) and len(t3) else float('nan'),
        'ci_lo': lo, 'ci_hi': hi, 'sig': sig,
    }


def main():
    print("Loading live trades.db...")
    live = load_live()
    print(f"  Live MACD trades closed: {len(live)}")
    print(f"  Live: {(live['pnl']>0).sum()} wins / {(live['pnl']<0).sum()} losses / sum ${live['pnl'].sum():+,.0f}")

    print("\nLoading BT trail03 CSV...")
    bt = load_bt()
    print(f"  BT MACD trades: {len(bt)}")
    print(f"  BT: {(bt['pnl']>0).sum()} wins / {(bt['pnl']<0).sum()} losses / sum ${bt['pnl'].sum():+,.0f}")

    # --------------- Per-exit-reason — diagnostic only ---------------
    print()
    print(f"{'='*100}")
    print("PER-EXIT-REASON BREAKDOWN (post-hoc; not predictive but informative)")
    print(f"{'='*100}")
    for label, df in [('LIVE', live), ('BT', bt)]:
        print(f"\n{label}:")
        agg = df.groupby('exit_reason').agg(n=('pnl','count'), mean_pnl=('pnl','mean'),
                                             sum_pnl=('pnl','sum'),
                                             wr=('win', lambda x: x.mean()*100)).sort_values('n', ascending=False)
        print(agg.to_string())

    # --------------- Univariate predictors ---------------
    features = [
        ('fill_price', 'price_at_entry'),
        ('cross_time_min', 'cross_time_min'),
        ('vol_at_cross', 'vol_at_cross'),
        ('macd_hist_pct', 'macd_hist_pct'),
        ('conviction_mult', 'conviction_mult'),
        ('minute_of_day', 'minute_of_day_ET'),
        ('weekday', 'weekday'),
    ]
    if 'entry_spread_bps' in live.columns:
        features.append(('entry_spread_bps', 'entry_spread_bps'))
    if 'hold_min' in bt.columns:
        # Skip — hold_min is post-hoc, already conditional on outcome
        pass

    rows = []
    print()
    print(f"{'='*100}")
    print("UNIVARIATE PREDICTOR ANALYSIS — tertile T1(low) vs T3(high)")
    print(f"{'='*100}")
    for col, label in features:
        for ds_name, ds in [('LIVE', live), ('BT', bt)]:
            if col not in ds.columns:
                continue
            r = analyse_feature(ds, col, ds_name, label=label)
            if r is None: continue
            rows.append(r)

    res = pd.DataFrame(rows)
    if len(res):
        # Pretty print
        print()
        print(f"{'feature':>22} | {'ds':>4} | {'n':>4} | {'mean T1':>10} | {'mean T3':>10} | {'WR T1':>5} | {'WR T3':>5} | {'Δ(T3-T1)':>10} | {'CI':>22} | sig")
        print('-' * 130)
        for _, r in res.iterrows():
            ci_str = f"[${r['ci_lo']:>+5,.0f}, ${r['ci_hi']:>+5,.0f}]"
            print(f"{r['feature']:>22} | {r['dataset']:>4} | {r['n_total']:>4} | "
                  f"${r['mean_T1']:>+8,.0f} | ${r['mean_T3']:>+8,.0f} | "
                  f"{r['wr_T1']:>4.1f}% | {r['wr_T3']:>4.1f}% | "
                  f"${r['delta_T3_minus_T1']:>+8,.0f} | {ci_str:>22} | {r['sig']}")
        res.to_csv(ROOT/'analysis_results'/'macd_predictors.csv', index=False)

    # --------------- Tertile detail for top candidates ---------------
    print()
    print(f"{'='*100}")
    print("TERTILE DETAIL — features showing direction-aligned signal in BOTH live AND BT")
    print(f"{'='*100}")
    # Find features where direction is consistent across live and BT
    if len(res):
        consistent = []
        for f in res['feature'].unique():
            sub = res[res['feature']==f]
            if len(sub) < 2: continue
            l = sub[sub['dataset']=='LIVE']['delta_T3_minus_T1'].iloc[0] if (sub['dataset']=='LIVE').any() else None
            b = sub[sub['dataset']=='BT']['delta_T3_minus_T1'].iloc[0] if (sub['dataset']=='BT').any() else None
            if l is None or b is None: continue
            if (l > 0 and b > 0) or (l < 0 and b < 0):
                consistent.append((f, l, b, np.sign(l)))
        if consistent:
            print(f"\nConsistent-direction features:")
            for f, l, b, s in sorted(consistent, key=lambda x: abs(x[1])+abs(x[2]), reverse=True):
                arrow = '↑ T3 better' if s > 0 else '↓ T1 better'
                print(f"  {f:>22}: live Δ=${l:+,.0f}, BT Δ=${b:+,.0f}  [{arrow}]")
                # Tertile detail
                for ds_name, ds in [('LIVE', live), ('BT', bt)]:
                    col = next((c for c, lab in features if lab == f), None)
                    if col is None: continue
                    t = tertile_summary(ds, col, label=f)
                    if len(t):
                        for _, r in t.iterrows():
                            print(f"    {ds_name:>4} {r['tertile']:>8} n={r['n']:>4}  mean=${r['mean_pnl']:>+7,.0f}  "
                                  f"sum=${r['sum_pnl']:>+9,.0f}  WR={r['wr_pct']:>4.1f}%  range={r['range']}")
        else:
            print("\nNo features show consistent direction across live and BT.")

    # --------------- Multi-hypothesis caveat ---------------
    print()
    print(f"{'='*100}")
    print("MULTI-HYPOTHESIS / SAMPLE-SIZE WARNING")
    print(f"{'='*100}")
    print(f"Tested {len(features)} features × 2 datasets = {len(features)*2} comparisons.")
    print(f"Bonferroni-corrected alpha for joint 5%: ~{0.05/(len(features)*2):.4f} per test.")
    print(f"Live n={len(live)} closed trades is small; CIs are wide. Treat any 'sig' below as exploratory.")
    print(f"Production decision requires BT validation with full pipeline + production-faithful")
    print(f"implementation, like H3-A lesson learned.")


if __name__ == '__main__':
    main()
