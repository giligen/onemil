"""
Regime V5 — Push DD reduction further by stacking regime + daily rules.

Best from V4 (daily rules): max 5 trades (-1.5% PnL, -21% DD)
Best from deep (regime): dist<-3% + vol>1.2%, 25% size (-1.4% PnL, -39% DD)

Now: Stack them. Also explore:
1. Tighter regime signals (more aggressive on HR days)
2. Multiple regime tiers (moderate risk vs extreme risk)
3. Combine regime + cushion (start small on normal days too, but scale back to base on HR)
4. Rolling strategy WR as an adaptive signal
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import date, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def load_spy():
    conn = sqlite3.connect('data/onemil.db')
    df = pd.read_sql_query("""
        SELECT bar_date as date, open, high, low, close, volume
        FROM daily_bars WHERE symbol = 'SPY' ORDER BY bar_date
    """, conn)
    conn.close()
    df['date'] = pd.to_datetime(df['date']).dt.date
    for c in ['open','high','low','close','volume']: df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def load_trades():
    df = pd.read_csv('C:/Work/onemil/full_15mo.csv')
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['entry_time'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['entry_time_et'])
    df['pnl'] = df['pnl'].astype(float)
    df = df.sort_values(['date', 'entry_time']).reset_index(drop=True)
    return df


def compute_spy_indicators(spy_df):
    s = spy_df.copy().sort_values('date').reset_index(drop=True)
    s['prev_close'] = s['close'].shift(1)
    s['daily_range_pct'] = (s['high'] - s['low']) / s['close'] * 100
    s['true_range'] = np.maximum(s['high'] - s['low'],
        np.maximum(abs(s['high'] - s['prev_close']), abs(s['low'] - s['prev_close'])))
    s['tr_pct'] = s['true_range'] / s['close'] * 100
    s['daily_return'] = (s['close'] / s['prev_close'] - 1) * 100

    for w in [3, 5, 10, 20]:
        s[f'vol_{w}d'] = s['daily_range_pct'].rolling(w).mean().shift(1)
        s[f'atr_{w}d'] = s['tr_pct'].rolling(w).mean().shift(1)
        s[f'ret_{w}d'] = ((s['close'] / s['close'].shift(w)) - 1).shift(1) * 100
        s[f'ret_std_{w}d'] = s['daily_return'].rolling(w).std().shift(1)

    s['high_20d'] = s['high'].rolling(20).max().shift(1)
    s['dist_from_20d_high'] = ((s['close'].shift(1) / s['high_20d']) - 1) * 100

    for sma in [10, 20, 50]:
        s[f'sma_{sma}'] = s['close'].rolling(sma).mean().shift(1)
        s[f'above_sma_{sma}'] = (s['close'].shift(1) > s[f'sma_{sma}']).astype(int)

    return s


def run_regime_strategy(trades_df, spy, regime_func, hr_scale, normal_scale=1.0,
                         max_trades_hr=0, max_trades_normal=0,
                         consec_stop_hr=0, consec_stop_normal=0,
                         cushion_scale_normal=False, cushion_start=1.0):
    """
    Run strategy with regime-conditional rules.

    On high-risk days: trade at hr_scale, apply hr stop rules
    On normal days: trade at normal_scale, apply normal stop rules
    cushion_scale_normal: if True, on normal days start at cushion_start, scale to 1.0 on win, back on loss
    """
    # Pre-compute HR dates
    spy_dict = {}
    for _, r in spy.iterrows():
        spy_dict[r['date']] = r.to_dict()

    hr_dates = set()
    for d in trades_df['date'].unique():
        if d in spy_dict:
            if regime_func(spy_dict[d]):
                hr_dates.add(d)

    tp, dpd = [], defaultdict(float)
    for td, dt in trades_df.groupby('date'):
        dt = dt.sort_values('entry_time')
        is_hr = td in hr_dates

        if is_hr:
            scale = hr_scale
            mt = max_trades_hr
            cs = consec_stop_hr
        else:
            if cushion_scale_normal:
                scale = cushion_start
            else:
                scale = normal_scale
            mt = max_trades_normal
            cs = consec_stop_normal

        cl, count, stopped = 0, 0, False
        for _, r in dt.iterrows():
            if stopped: continue
            if mt > 0 and count >= mt: continue

            pnl = r['pnl'] * scale
            tp.append(pnl); dpd[td] += pnl; count += 1

            if r['pnl'] > 0:
                cl = 0
                if not is_hr and cushion_scale_normal:
                    scale = 1.0  # scale up on win (normal days)
            else:
                cl += 1
                if not is_hr and cushion_scale_normal:
                    scale = cushion_start  # back to base on loss
                if cs > 0 and cl >= cs:
                    stopped = True

    return tp, dpd, hr_dates


def metrics(dpd, tp):
    daily = [dpd[d] for d in sorted(dpd)]
    cum = np.cumsum(daily)
    total = sum(daily)
    pk = np.maximum.accumulate(cum)
    dd = abs(min(cum - pk)) if len(cum) > 0 else 0

    wins = sum(1 for p in tp if p > 0)
    wr = wins / len(tp) * 100 if tp else 0

    mm_pnls = [v for d, v in sorted(dpd.items()) if date(2025, 3, 1) <= d <= date(2025, 5, 31)]
    mm_pnl = sum(mm_pnls)
    if mm_pnls:
        mc = np.cumsum(mm_pnls)
        mp = np.maximum.accumulate(mc)
        mm_dd = abs(min(mc - mp))
    else:
        mm_dd = 0

    calmar = total / dd if dd > 0 else 0
    sharpe = (np.mean(daily) / np.std(daily)) * np.sqrt(252) if len(daily) > 1 and np.std(daily) > 0 else 0

    return total, dd, mm_pnl, mm_dd, wr, len(tp), calmar, sharpe


def pr(label, total, dd, mm_pnl, mm_dd, wr, trades, calmar, sharpe, bp, bd, hr_count=0):
    pd_ = (total / bp - 1) * 100
    dd_ = (dd / bd - 1) * 100
    print(f"  {label:<62} ${total:>9,.0f} ({pd_:>+5.1f}%) ${dd:>7,.0f} ({dd_:>+5.1f}%) ${mm_pnl:>8,.0f} ${mm_dd:>7,.0f} {calmar:>6.1f} {sharpe:>6.2f} {hr_count:>4}")


def main():
    trades_df = load_trades()
    spy = compute_spy_indicators(load_spy())
    print(f"Loaded {len(trades_df)} trades, {trades_df['date'].nunique()} days\n")

    BP, BD = 247088, 28781

    hdr = f"  {'Strategy':<62} {'PnL':>10} {'Δ%':>8} {'MaxDD':>8} {'Δ%':>8} {'MM PnL':>9} {'MM DD':>8} {'Calmr':>6} {'Shrpe':>6} {'HR#':>4}"
    sep = "  " + "-" * 145

    # =========================================================================
    # REGIME DEFINITIONS
    # =========================================================================
    R1 = ("dist<-3%+vol>1.2", lambda r: r.get('dist_from_20d_high', 0) < -3 and r.get('vol_5d', 0) > 1.2)
    R2 = ("dist<-5%+vol>1.0", lambda r: r.get('dist_from_20d_high', 0) < -5 and r.get('vol_5d', 0) > 1.0)
    R3 = ("vol>1.5+<SMA50", lambda r: r.get('vol_5d', 0) > 1.5 and r.get('above_sma_50', 1) == 0)
    R4 = ("vol>1.5+<SMA20", lambda r: r.get('vol_5d', 0) > 1.5 and r.get('above_sma_20', 1) == 0)
    R5 = ("dist<-3%+vol>1.0", lambda r: r.get('dist_from_20d_high', 0) < -3 and r.get('vol_5d', 0) > 1.0)
    R6 = ("atr>1.5+<SMA20", lambda r: r.get('atr_5d', 0) > 1.5 and r.get('above_sma_20', 1) == 0)
    R7 = ("vol>1.2+<SMA50", lambda r: r.get('vol_5d', 0) > 1.2 and r.get('above_sma_50', 1) == 0)

    # =========================================================================
    print("=" * 160)
    print("SECTION 1: Best regime signals + stacking with max-trades rule")
    print("=" * 160)
    print(hdr); print(sep)

    # Baseline
    tp, dpd, hr = run_regime_strategy(trades_df, spy, lambda r: False, 1.0)
    pr("Baseline", *metrics(dpd, tp), BP, BD)

    configs = [
        # (label, regime, hr_scale, normal_scale, mt_hr, mt_normal, cs_hr, cs_normal)
        # Pure regime (best signals from deep analysis)
        ("R1: dist<-3%+vol>1.2, SKIP", R1, 0.0, 1.0, 0, 0, 0, 0),
        ("R1: dist<-3%+vol>1.2, 25%", R1, 0.25, 1.0, 0, 0, 0, 0),
        ("R2: dist<-5%+vol>1.0, SKIP", R2, 0.0, 1.0, 0, 0, 0, 0),
        ("R3: vol>1.5+<SMA50, SKIP", R3, 0.0, 1.0, 0, 0, 0, 0),
        ("R7: vol>1.2+<SMA50, SKIP", R7, 0.0, 1.0, 0, 0, 0, 0),
        ("R7: vol>1.2+<SMA50, 25%", R7, 0.25, 1.0, 0, 0, 0, 0),

        # Regime + max 5 trades (stacking)
        ("R1 SKIP + max5 normal", R1, 0.0, 1.0, 0, 5, 0, 0),
        ("R1 25% + max5 normal", R1, 0.25, 1.0, 5, 5, 0, 0),
        ("R2 SKIP + max5 normal", R2, 0.0, 1.0, 0, 5, 0, 0),
        ("R3 SKIP + max5 normal", R3, 0.0, 1.0, 0, 5, 0, 0),
        ("R7 SKIP + max5 normal", R7, 0.0, 1.0, 0, 5, 0, 0),
        ("R7 25% + max5 normal", R7, 0.25, 1.0, 5, 5, 0, 0),

        # Regime + max 5 + 3 consec
        ("R1 SKIP + max5 + 3consec", R1, 0.0, 1.0, 0, 5, 0, 3),
        ("R1 25% max3 + max5 normal + 3consec", R1, 0.25, 1.0, 3, 5, 2, 3),
        ("R3 SKIP + max5 + 3consec", R3, 0.0, 1.0, 0, 5, 0, 3),
        ("R7 SKIP + max5 + 3consec", R7, 0.0, 1.0, 0, 5, 0, 3),
        ("R7 25% max3 + max5 normal + 3consec", R7, 0.25, 1.0, 3, 5, 2, 3),

        # Max 4 trades on HR, max 5 normal
        ("R1 SKIP + max4 all", R1, 0.0, 1.0, 0, 4, 0, 0),
        ("R1 50% max3 HR + max5 normal", R1, 0.50, 1.0, 3, 5, 0, 0),
        ("R3 50% max3 HR + max5 normal", R3, 0.50, 1.0, 3, 5, 0, 0),
        ("R7 50% max3 HR + max5 normal", R7, 0.50, 1.0, 3, 5, 0, 0),
    ]

    for label, (rname, rfunc), hrs, ns, mt_hr, mt_n, cs_hr, cs_n in configs:
        tp, dpd, hr = run_regime_strategy(trades_df, spy, rfunc, hrs, ns, mt_hr, mt_n, cs_hr, cs_n)
        pr(label, *metrics(dpd, tp), BP, BD, len(hr))

    # =========================================================================
    print()
    print("=" * 160)
    print("SECTION 2: Two-tier regime (moderate risk + extreme risk)")
    print("=" * 160)
    print(hdr); print(sep)

    # Two-tier: extreme risk (SKIP), moderate risk (50%), normal (100%)
    spy_dict = {}
    for _, r in spy.iterrows():
        spy_dict[r['date']] = r.to_dict()

    def run_two_tier(trades_df, spy_dict, extreme_func, moderate_func,
                      extreme_scale, moderate_scale, normal_scale=1.0,
                      max_trades_extreme=0, max_trades_moderate=0, max_trades_normal=0,
                      consec_stop=0):
        tp, dpd = [], defaultdict(float)
        extreme_dates, moderate_dates = set(), set()

        for d in trades_df['date'].unique():
            if d in spy_dict:
                if extreme_func(spy_dict[d]):
                    extreme_dates.add(d)
                elif moderate_func(spy_dict[d]):
                    moderate_dates.add(d)

        for td, dt in trades_df.groupby('date'):
            dt = dt.sort_values('entry_time')

            if td in extreme_dates:
                scale, mt = extreme_scale, max_trades_extreme
            elif td in moderate_dates:
                scale, mt = moderate_scale, max_trades_moderate
            else:
                scale, mt = normal_scale, max_trades_normal

            cl, count, stopped = 0, 0, False
            for _, r in dt.iterrows():
                if stopped: continue
                if mt > 0 and count >= mt: continue

                pnl = r['pnl'] * scale
                tp.append(pnl); dpd[td] += pnl; count += 1

                if r['pnl'] > 0: cl = 0
                else:
                    cl += 1
                    if consec_stop > 0 and cl >= consec_stop: stopped = True

        return tp, dpd, len(extreme_dates), len(moderate_dates)

    # Define tier functions
    extreme_defs = [
        ("dist<-5%+vol>1.0", lambda r: r.get('dist_from_20d_high', 0) < -5 and r.get('vol_5d', 0) > 1.0),
        ("vol>1.5+<SMA50", lambda r: r.get('vol_5d', 0) > 1.5 and r.get('above_sma_50', 1) == 0),
    ]
    moderate_defs = [
        ("dist<-3%+vol>1.0", lambda r: r.get('dist_from_20d_high', 0) < -3 and r.get('vol_5d', 0) > 1.0),
        ("vol>1.2+<SMA50", lambda r: r.get('vol_5d', 0) > 1.2 and r.get('above_sma_50', 1) == 0),
        ("<SMA20+<SMA50", lambda r: r.get('above_sma_20', 1) == 0 and r.get('above_sma_50', 1) == 0),
        ("dist<-3%+vol>1.2", lambda r: r.get('dist_from_20d_high', 0) < -3 and r.get('vol_5d', 0) > 1.2),
    ]

    two_tier_configs = [
        # (label, extreme_def, moderate_def, e_scale, m_scale, n_scale, mt_e, mt_m, mt_n, cs)
        # Extreme=SKIP, Moderate=50%, Normal=100%
        ("SKIP extreme(d<-5%+v>1) / 50% moderate(d<-3%+v>1) / 100%",
         extreme_defs[0], moderate_defs[0], 0.0, 0.5, 1.0, 0, 5, 0, 0),
        ("SKIP extreme(d<-5%+v>1) / 50% moderate(d<-3%+v>1) / max5",
         extreme_defs[0], moderate_defs[0], 0.0, 0.5, 1.0, 0, 3, 5, 0),
        ("SKIP extreme(v>1.5+<50) / 50% moderate(v>1.2+<50) / 100%",
         extreme_defs[1], moderate_defs[1], 0.0, 0.5, 1.0, 0, 5, 0, 0),
        ("SKIP extreme(v>1.5+<50) / 50% moderate(v>1.2+<50) / max5",
         extreme_defs[1], moderate_defs[1], 0.0, 0.5, 1.0, 0, 3, 5, 0),
        ("SKIP extreme(v>1.5+<50) / 25% moderate(v>1.2+<50) / max5",
         extreme_defs[1], moderate_defs[1], 0.0, 0.25, 1.0, 0, 3, 5, 0),
        # With 3 consec stop on all tiers
        ("SKIP ext(d<-5%+v>1) / 50% mod(d<-3%+v>1) / max5 + 3c",
         extreme_defs[0], moderate_defs[0], 0.0, 0.5, 1.0, 0, 3, 5, 3),
        ("SKIP ext(v>1.5+<50) / 50% mod(v>1.2+<50) / max5 + 3c",
         extreme_defs[1], moderate_defs[1], 0.0, 0.5, 1.0, 0, 3, 5, 3),
        ("SKIP ext(v>1.5+<50) / 25% mod(v>1.2+<50) / max5 + 3c",
         extreme_defs[1], moderate_defs[1], 0.0, 0.25, 1.0, 0, 3, 5, 3),
        # Extreme=SKIP, Moderate=25%, Normal=100% + max5 + 3consec
        ("SKIP ext(d<-5%+v>1) / 25% mod(d<-3%+v>1.2) / max5 + 3c",
         extreme_defs[0], moderate_defs[3], 0.0, 0.25, 1.0, 0, 3, 5, 3),
        # Extreme=SKIP, Moderate=50% max2, Normal=max5
        ("SKIP ext(d<-5%+v>1) / 50% max2 mod(d<-3%+v>1) / max5",
         extreme_defs[0], moderate_defs[0], 0.0, 0.5, 1.0, 0, 2, 5, 0),
        ("SKIP ext(v>1.5+<50) / 50% max2 mod(v>1.2+<50) / max5",
         extreme_defs[1], moderate_defs[1], 0.0, 0.5, 1.0, 0, 2, 5, 0),
        # SMA-based moderate
        ("SKIP ext(d<-5%+v>1) / 50% mod(<SMA20+<SMA50) / max5",
         extreme_defs[0], moderate_defs[2], 0.0, 0.5, 1.0, 0, 3, 5, 0),
        ("SKIP ext(v>1.5+<50) / 50% mod(<SMA20+<SMA50) / max5",
         extreme_defs[1], moderate_defs[2], 0.0, 0.5, 1.0, 0, 3, 5, 0),
    ]

    for label, (ename, efunc), (mname, mfunc), es, ms, ns, mte, mtm, mtn, cs in two_tier_configs:
        tp, dpd, ne, nm = run_two_tier(trades_df, spy_dict, efunc, mfunc,
                                         es, ms, ns, mte, mtm, mtn, cs)
        t, d, mp, md, wr, n, cal, sh = metrics(dpd, tp)
        pd_ = (t / BP - 1) * 100
        dd_ = (d / BD - 1) * 100
        print(f"  {label:<62} ${t:>9,.0f} ({pd_:>+5.1f}%) ${d:>7,.0f} ({dd_:>+5.1f}%) ${mp:>8,.0f} ${md:>7,.0f} {cal:>6.1f} {sh:>6.2f} {ne:>2}E/{nm:>2}M")

    # =========================================================================
    print()
    print("=" * 160)
    print("SECTION 3: Regime + normal-day cushion (mild scaling on normal days too)")
    print("=" * 160)
    print(hdr); print(sep)

    # Baseline
    tp, dpd, hr = run_regime_strategy(trades_df, spy, lambda r: False, 1.0)
    pr("Baseline", *metrics(dpd, tp), BP, BD)

    cushion_configs = [
        # (label, regime, hr_scale, cushion_start, mt_hr, mt_normal, cs_hr, cs_normal)
        ("R1 SKIP + 90%↔100% normal + max5", R1, 0.0, 0.90, 0, 5, 0, 0),
        ("R1 25% + 90%↔100% normal + max5", R1, 0.25, 0.90, 3, 5, 0, 0),
        ("R3 SKIP + 90%↔100% normal + max5", R3, 0.0, 0.90, 0, 5, 0, 0),
        ("R7 SKIP + 90%↔100% normal + max5", R7, 0.0, 0.90, 0, 5, 0, 0),
        ("R1 SKIP + 85%↔100% normal + max5", R1, 0.0, 0.85, 0, 5, 0, 0),
        ("R3 SKIP + 85%↔100% normal + max5", R3, 0.0, 0.85, 0, 5, 0, 0),
        ("R7 SKIP + 85%↔100% normal + max5", R7, 0.0, 0.85, 0, 5, 0, 0),
        # With 3consec
        ("R1 SKIP + 90%↔100% + max5 + 3consec", R1, 0.0, 0.90, 0, 5, 0, 3),
        ("R3 SKIP + 90%↔100% + max5 + 3consec", R3, 0.0, 0.90, 0, 5, 0, 3),
        ("R7 SKIP + 90%↔100% + max5 + 3consec", R7, 0.0, 0.90, 0, 5, 0, 3),
        ("R1 SKIP + 85%↔100% + max5 + 3consec", R1, 0.0, 0.85, 0, 5, 0, 3),
        ("R7 SKIP + 85%↔100% + max5 + 3consec", R7, 0.0, 0.85, 0, 5, 0, 3),
    ]

    for label, (rname, rfunc), hrs, cush_start, mt_hr, mt_n, cs_hr, cs_n in cushion_configs:
        tp, dpd, hr = run_regime_strategy(trades_df, spy, rfunc, hrs, 1.0,
                                            mt_hr, mt_n, cs_hr, cs_n,
                                            cushion_scale_normal=True,
                                            cushion_start=cush_start)
        pr(label, *metrics(dpd, tp), BP, BD, len(hr))

    # =========================================================================
    print()
    print("=" * 160)
    print("SECTION 4: Rolling strategy WR as adaptive regime signal")
    print("=" * 160)
    print(hdr); print(sep)

    # Compute rolling WR over last N trading days (from our OWN trades)
    dates_sorted = sorted(trades_df['date'].unique())
    daily_wr = {}
    for td, dt in trades_df.groupby('date'):
        wins = (dt['pnl'] > 0).sum()
        daily_wr[td] = wins / len(dt) * 100

    for lookback in [5, 10, 15]:
        for threshold in [25, 30, 35]:
            tp, dpd = [], defaultdict(float)
            for i, td in enumerate(dates_sorted):
                dt = trades_df[trades_df['date'] == td].sort_values('entry_time')

                # Rolling WR from previous N days
                past = [daily_wr[dates_sorted[j]] for j in range(max(0, i - lookback), i) if dates_sorted[j] in daily_wr]
                rolling_wr = np.mean(past) if past else 50

                # If rolling WR is low, scale down
                if rolling_wr < threshold:
                    scale = 0.5
                    mt = 3
                else:
                    scale = 1.0
                    mt = 5

                count, cl, stopped = 0, 0, False
                for _, r in dt.iterrows():
                    if stopped: continue
                    if mt > 0 and count >= mt: continue
                    pnl = r['pnl'] * scale
                    tp.append(pnl); dpd[td] += pnl; count += 1
                    if r['pnl'] <= 0:
                        cl += 1
                        if cl >= 3: stopped = True
                    else: cl = 0

                # Count HR days
            hr_count = sum(1 for i, td in enumerate(dates_sorted)
                          if np.mean([daily_wr[dates_sorted[j]]
                                      for j in range(max(0, i - lookback), i)
                                      if dates_sorted[j] in daily_wr] or [50]) < threshold)

            label = f"Rolling {lookback}d WR < {threshold}% → 50% max3 + 3consec"
            pr(label, *metrics(dpd, tp), BP, BD, hr_count)

    # =========================================================================
    # Combine rolling WR with SPY regime
    print()
    print("  Combining SPY regime + rolling WR:")
    print(sep)

    for lookback in [5, 10]:
        for wr_thresh in [30, 35]:
            tp, dpd = [], defaultdict(float)
            hr_count = 0
            for i, td in enumerate(dates_sorted):
                dt = trades_df[trades_df['date'] == td].sort_values('entry_time')

                # SPY regime
                spy_data = spy_dict.get(td, {})
                spy_hr = (spy_data.get('dist_from_20d_high', 0) < -3 and
                          spy_data.get('vol_5d', 0) > 1.2)

                # Rolling WR
                past = [daily_wr[dates_sorted[j]] for j in range(max(0, i - lookback), i) if dates_sorted[j] in daily_wr]
                rolling_wr = np.mean(past) if past else 50
                wr_hr = rolling_wr < wr_thresh

                # Combined: either signal triggers HR
                is_hr = spy_hr or wr_hr
                if is_hr: hr_count += 1

                if is_hr:
                    scale, mt = 0.25, 3
                else:
                    scale, mt = 1.0, 5

                count, cl, stopped = 0, 0, False
                for _, r in dt.iterrows():
                    if stopped: continue
                    if mt > 0 and count >= mt: continue
                    pnl = r['pnl'] * scale
                    tp.append(pnl); dpd[td] += pnl; count += 1
                    if r['pnl'] <= 0:
                        cl += 1
                        if cl >= 3: stopped = True
                    else: cl = 0

            label = f"SPY(dist<-3%+vol>1.2) OR Roll{lookback}d WR<{wr_thresh}% → 25% max3 3c"
            pr(label, *metrics(dpd, tp), BP, BD, hr_count)

    # =========================================================================
    print()
    print("=" * 160)
    print("SECTION 5: TOP 10 FINALISTS — Monthly breakdown")
    print("=" * 160)

    finalists = {
        'Baseline': (lambda r: False, 1.0, 1.0, 0, 0, 0, 0, False, 1.0),
        'R1 SKIP+m5': (R1[1], 0.0, 1.0, 0, 5, 0, 0, False, 1.0),
        'R3 SKIP+m5': (R3[1], 0.0, 1.0, 0, 5, 0, 0, False, 1.0),
        'R7 SKIP+m5': (R7[1], 0.0, 1.0, 0, 5, 0, 0, False, 1.0),
        'R1 SKIP+m5+3c': (R1[1], 0.0, 1.0, 0, 5, 0, 3, False, 1.0),
        'R3 SKIP+m5+3c': (R3[1], 0.0, 1.0, 0, 5, 0, 3, False, 1.0),
        'R7 SKIP+m5+3c': (R7[1], 0.0, 1.0, 0, 5, 0, 3, False, 1.0),
        'R7 SKIP+90%+m5': (R7[1], 0.0, 1.0, 0, 5, 0, 0, True, 0.90),
    }

    strat_dpd = {}
    for name, (rfunc, hrs, ns, mt_hr, mt_n, cs_hr, cs_n, cush, cush_s) in finalists.items():
        tp, dpd, hr = run_regime_strategy(trades_df, spy, rfunc, hrs, ns, mt_hr, mt_n, cs_hr, cs_n, cush, cush_s)
        strat_dpd[name] = dpd

    months = sorted(set((d.year, d.month) for d in strat_dpd['Baseline']))

    header = f"{'Month':>10}"
    for n in finalists: header += f" {n:>14}"
    print(header)
    print("-" * (10 + 15 * len(finalists)))

    for yr, mo in months:
        line = f"  {yr}-{mo:02d}"
        for n in finalists:
            pnl = sum(v for d, v in strat_dpd[n].items() if d.year == yr and d.month == mo)
            line += f" ${pnl:>12,.0f}"
        print(line)

    print()
    def calc_total(dpd):
        return sum(dpd.values())

    def calc_dd(dpd):
        daily = [dpd[d] for d in sorted(dpd)]
        cum = np.cumsum(daily)
        pk = np.maximum.accumulate(cum)
        return abs(min(cum - pk))

    def calc_mm_dd(dpd):
        mm = [v for d, v in sorted(dpd.items()) if date(2025,3,1) <= d <= date(2025,5,31)]
        if not mm: return 0
        mc = np.cumsum(mm)
        mp = np.maximum.accumulate(mc)
        return abs(min(mc - mp))

    for metric_name, metric_func in [("TOTAL", calc_total), ("MaxDD", calc_dd), ("MM DD", calc_mm_dd)]:
        line = f"  {metric_name:<8}"
        for n in finalists:
            val = metric_func(strat_dpd[n])
            line += f" ${val:>12,.0f}"
        print(line)

    # Calmar
    line = f"  {'Calmar':<8}"
    for n in finalists:
        dpd = strat_dpd[n]
        daily = [dpd[d] for d in sorted(dpd)]
        total = sum(daily)
        cum = np.cumsum(daily)
        pk = np.maximum.accumulate(cum)
        dd = abs(min(cum - pk))
        cal = total / dd if dd > 0 else 0
        line += f" {cal:>14.1f}"
    print(line)


if __name__ == '__main__':
    main()
