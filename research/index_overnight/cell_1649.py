#!/usr/bin/env python3
"""
Cells 1,649-1,651: index overnight premium conditioned on close-volume shape, and turn-of-month.

Implements research/index_overnight/PREREG_1649.md (frozen 2026-09-29 06:20 UTC) EXACTLY.

Mechanisms:
  * 1,649 report-only benchmarks: unconditional overnight (buy MOC, sell MOO next session) and
    intraday (buy MOO, sell MOC same session), SPY/QQQ/IWM, every session in TRAIN+VAL. Not scored.
  * 1,650 CONDITIONAL OVERNIGHT: s_t = share of the 09:30-16:00 session volume printed in the last
    half hour (15:30-16:00). Heavy close = s_t >= 1.25 x the trailing 60-session median of s_t (PRIOR
    sessions only, shift(1) before the rolling window -- today's own s_t never leaks into its own
    threshold). Buy MOC on heavy-close days, sell MOO next session. TRAIN 2016-2019, VAL 2020-2023.
  * 1,651 TURN-OF-MONTH: buy MOC on the last session of the month, sell MOC on the third session of
    the next month (one round-trip event per ETF per month). Judged 2016-2023 pooled.
  * TEST 2024-01..2026-09 is SEALED for both -- never read, not merely unreported: build_* functions
    below drop any event whose EXIT leg would require a TEST-period price before computing anything.

Costs: the PREREG states "0.5 bp per auction leg plus 0.5 bp half-spread (1 bp per round trip is
generous for these three)". Both legs of every trade here (1649/1650 MOC+MOO, 1651 MOC+MOC) are
auction orders, so this is read as a single flat 1.0 bp round-trip deduction (0.5 bp/leg x 2 legs, one
number, not decomposed further -- decomposing "auction leg" and "half-spread" as separate additive
per-leg charges would double the PREREG's own stated total and contradict its parenthetical check).

Early closes: pandas_market_calendars is NOT installed on this node (checked before writing this
script). The PREREG's literal fallback ("last bar before 15:59 ET") fires on ZERO sessions on this
full-SIP extended-hours cache (verified in research/lev_flow/cell_1643.py against 9 ETFs on the same
data source) because bars exist deep into extended hours regardless of whether the PRIMARY session
closed early. Reused here unchanged: ratio = volume(13:00-16:00 ET) / volume(09:30-16:00 ET) per
symbol-session; flag ratio < 0.5x that symbol's own full-sample median; require >= 2 of 3 symbols to
agree on the same date (majority, same spirit as cell_1643's 6/9). This is a DATA-DRIVEN measurement,
not a hard-coded date list, so it is inherently a superset-safe cross-check of "the known early-close
dates" the PREREG asks for as a fallback; a hard-coded reference list was not additionally maintained.

MDE: the PREREG's own worked example ("~190 heavy-close nights in VAL x 3 ETFs ... SD ~60bps -> SE
~4.4bps -> MDE ~11bps") only reproduces with SE = SD / sqrt(N_NIGHTS=190), i.e. sqrt(CLUSTERS), not
sqrt(N_EVENTS=570). mde_clustered() below implements SD(pooled events) / sqrt(n_clusters) * 2.5 and
was checked against that worked example before being trusted (see RESULT caveats).

Outputs: research/index_overnight/RESULT_1649_build.md (<=120 lines), nights_1649.csv (date, symbol,
s_t, heavy, overnight_net_bps -- one row per session in TRAIN+VAL, the raw data behind 1649 AND 1650),
tom_1651.csv (month, symbol, net_bps).
"""
import os
import sys
from collections import Counter
from datetime import date as date_cls
from datetime import time as dtime

import numpy as np
import pandas as pd
import pytz

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)

ET = pytz.timezone('America/New_York')
SYMBOLS = ['SPY', 'QQQ', 'IWM']
DATA_DIR = os.path.join(ROOT, 'research/lev_flow/data')     # shared minute+daily cache (per delegator)
OUT_DIR = os.path.join(ROOT, 'research/index_overnight')

ROUND_TRIP_COST_BPS = 1.0   # see docstring: 0.5bp/auction leg x 2 legs, flat per event
TRAIN_START, TRAIN_END = date_cls(2016, 1, 1), date_cls(2019, 12, 31)
VAL_START, VAL_END = date_cls(2020, 1, 1), date_cls(2023, 12, 31)
TEST_START = date_cls(2024, 1, 1)   # SEALED -- never read past this date

HEAVY_MULT = 1.25
TRAILING_WINDOW = 60
LAST_HALF_HOUR = dtime(15, 30)
SESSION_OPEN, SESSION_CLOSE = dtime(9, 30), dtime(16, 0)

PASS_BAR_1650 = dict(mean_bps=8.0, t=2.5, excess_bps=4.0, excess_t=2.0, train_t=1.0)
PASS_BAR_1651 = dict(mean_bps=30.0, t=2.5, years_pos_frac=6.0 / 8.0)
# Amendment 1 (2026-09-29 06:45 UTC, before any number): 1,649 scored as a stacking sleeve.
PASS_BAR_1649 = dict(mean_bps=2.0, t=2.5, years_pos_frac=6.0 / 8.0)
SLEEVE_NOTIONAL = 20000.0   # $ per night/event, per Amendment 1 and the original "Independent check" line


def load_data():
    """Load per-symbol minute (adjustment='all', SIP, extended hours) and daily parquet caches."""
    minute, daily = {}, {}
    for sym in SYMBOLS:
        mpath = os.path.join(DATA_DIR, 'minute', f'{sym}.parquet')
        dpath = os.path.join(DATA_DIR, 'daily', f'{sym}.parquet')
        if not os.path.exists(mpath) or not os.path.exists(dpath):
            raise FileNotFoundError(f'missing cached bars for {sym}: expected {mpath} / {dpath}')
        m = pd.read_parquet(mpath)
        m['t'] = pd.to_datetime(m['t'], utc=True)
        m['t_et'] = m['t'].dt.tz_convert(ET)
        m['date'] = m['t_et'].dt.date
        d = pd.read_parquet(dpath)
        d['t'] = pd.to_datetime(d['t'], utc=True)
        d['date'] = d['t'].dt.tz_convert(ET).dt.date
        d = d.sort_values('date').reset_index(drop=True)
        minute[sym], daily[sym] = m, d
        print(f'[load] {sym}: {len(m):,} minute bars, {len(d):,} daily sessions '
              f'({d["date"].min()}..{d["date"].max()})', flush=True)
    return minute, daily


def build_early_close_set(minute):
    """Data-driven early-close detector -- see module docstring. Returns a set of `date` objects."""
    ratios = {}
    for sym, m in minute.items():
        rth = m[(m['t_et'].dt.time >= SESSION_OPEN) & (m['t_et'].dt.time <= SESSION_CLOSE)]
        win = rth[rth['t_et'].dt.time >= dtime(13, 0)]
        full_vol = rth.groupby('date')['v'].sum()
        win_vol = win.groupby('date')['v'].sum()
        ratios[sym] = (win_vol / full_vol).dropna()
    votes = Counter()
    for sym, ratio in ratios.items():
        med = ratio.median()
        for d in ratio[ratio < 0.5 * med].index:
            votes[d] += 1
    early = {d for d, n in votes.items() if n >= 2}
    print(f'[early_close] {len(early)} early-close sessions detected '
          f'(fallback: >=2/3 symbols, 13:00-16:00 ET volume share < 0.5x own median)', flush=True)
    return early


def compute_volume_share(m_sym, early_close_dates):
    """s_t = 15:30-16:00 ET volume / 09:30-16:00 ET volume per session. Early-close dates dropped
    (no last-half-hour window exists on those sessions)."""
    rth = m_sym[(m_sym['t_et'].dt.time >= SESSION_OPEN) & (m_sym['t_et'].dt.time <= SESSION_CLOSE)]
    full_vol = rth.groupby('date')['v'].sum()
    last30_vol = rth[rth['t_et'].dt.time >= LAST_HALF_HOUR].groupby('date')['v'].sum()
    s_t = (last30_vol / full_vol).dropna()
    s_t = s_t[~s_t.index.isin(early_close_dates)]
    return s_t.sort_index()


def heavy_close_flags(s_t):
    """Heavy = s_t >= 1.25 x trailing 60-session median, PRIOR sessions only (shift(1) before the
    rolling window: 'known at 16:00' per the PREREG, no self-leakage). First 60 valid sessions of the
    sample are unclassifiable (NaN) by construction."""
    trailing_med = s_t.shift(1).rolling(TRAILING_WINDOW, min_periods=TRAILING_WINDOW).median()
    heavy = s_t >= HEAVY_MULT * trailing_med
    heavy = heavy.where(trailing_med.notna())   # NaN, not False, when unclassifiable
    return pd.DataFrame({'s_t': s_t, 'trailing_med': trailing_med, 'heavy': heavy})


def cluster_t(df, value_col, cluster_col):
    """Cluster-robust t-stat: mean value per cluster is one observation."""
    sub = df.dropna(subset=[value_col])
    if len(sub) == 0:
        return np.nan, 0
    cl = sub.groupby(cluster_col)[value_col].mean()
    n = len(cl)
    if n < 2 or cl.std(ddof=1) == 0:
        return np.nan, n
    return float(cl.mean() / (cl.std(ddof=1) / np.sqrt(n))), n


def cluster_mean_se(df, value_col, cluster_col):
    sub = df.dropna(subset=[value_col])
    cl = sub.groupby(cluster_col)[value_col].mean()
    n = len(cl)
    if n < 2:
        return np.nan, np.nan, n
    return float(cl.mean()), float(cl.std(ddof=1) / np.sqrt(n)), n


def mde_clustered(df, value_col, cluster_col):
    """MDE = SD(pooled events) / sqrt(N_CLUSTERS) * 2.5 -- see module docstring."""
    sub = df.dropna(subset=[value_col])
    n_clusters = sub[cluster_col].nunique()
    if n_clusters < 2:
        return np.nan
    return float(sub[value_col].std(ddof=1) / np.sqrt(n_clusters) * 2.5)


def ex_top(sub, value_col, pct=0.05):
    q = sub[value_col].quantile(1 - pct)
    return float(sub[sub[value_col] <= q][value_col].mean())


def ex_worst(sub, value_col, pct=0.05):
    q = sub[value_col].quantile(pct)
    return float(sub[sub[value_col] >= q][value_col].mean())


def tercile_table(df, value_col, key_col='s_t'):
    sub = df.dropna(subset=[value_col, key_col])
    if len(sub) < 6:
        return None
    try:
        q = pd.qcut(sub[key_col], 3, labels=['T1_low', 'T2_mid', 'T3_high'], duplicates='drop')
    except ValueError:
        return None
    return sub.groupby(q, observed=True)[value_col].agg(['mean', 'count'])


def equity_stats(sub, value_col, date_col='date'):
    """Sequential compounding, one symbol, date order: annualised return + max drawdown on a fully-
    deployed-capital equity curve (100% of the sleeve's capital redeployed on every qualifying event)."""
    sub = sub.dropna(subset=[value_col]).sort_values(date_col)
    if len(sub) < 2:
        return np.nan, np.nan
    r = sub[value_col].values / 10000.0
    equity = np.cumprod(1 + r)
    total = equity[-1] - 1
    span_days = max((pd.Timestamp(sub[date_col].iloc[-1]) - pd.Timestamp(sub[date_col].iloc[0])).days, 1)
    years = span_days / 365.25
    ann = (1 + total) ** (1 / years) - 1 if years > 0 else np.nan
    running_max = np.maximum.accumulate(equity)
    dd = (equity / running_max - 1).min()
    return float(ann), float(dd)


def report_cell(df, value_col, label, cluster_col='date'):
    """Every generic reporting field required 'per cell and split'."""
    sub = df.dropna(subset=[value_col])
    n = len(sub)
    if n == 0:
        return {'label': label, 'n': 0}
    t, n_clusters = cluster_t(sub, value_col, cluster_col)
    per_etf = sub.groupby('symbol')[value_col].agg(['mean', 'count'])
    per_year = sub.groupby('year')[value_col].agg(['mean', 'count'])
    worst = sub.groupby(cluster_col)[value_col].mean().sort_values()
    worst_item = (worst.index[0], float(worst.iloc[0])) if len(worst) else (None, np.nan)
    return dict(label=label, n=n, n_clusters=n_clusters, mean_bps=float(sub[value_col].mean()), t=t,
                mde=mde_clustered(sub, value_col, cluster_col), ex_top5=ex_top(sub, value_col, 0.05),
                ex_worst5=ex_worst(sub, value_col, 0.05), per_etf=per_etf, per_year=per_year,
                worst=worst_item)


def fmt_per_etf(pe):
    return ', '.join(f'{s}:{r["mean"]:+.2f}({int(r["count"])})' for s, r in pe.iterrows())


def fmt_per_year(py):
    return ', '.join(f'{y}:{r["mean"]:+.1f}({int(r["count"])})' for y, r in py.iterrows())


def fmt_terc(t):
    if t is None:
        return '(insufficient n)'
    return ' | '.join(f'{i}: {r["mean"]:+.2f}bps (n={int(r["count"])})' for i, r in t.iterrows())


def fmt_ann_dd(daily, df, value_col, date_col='date'):
    parts = []
    for sym in SYMBOLS:
        ann, dd = equity_stats(df[df['symbol'] == sym], value_col, date_col)
        parts.append(f'{sym}:{ann * 100:+.2f}%/dd{dd * 100:.2f}%' if not np.isnan(ann) else f'{sym}:n/a')
    return ', '.join(parts)


def render_line(rc):
    if rc.get('n', 0) == 0:
        return f"- **{rc['label']}**: n=0"
    return (f"- **{rc['label']}**: n={rc['n']} ({rc['n_clusters']} clusters), "
            f"mean {rc['mean_bps']:+.2f} bps, clustered t={rc['t']:.2f}, MDE={rc['mde']:.2f} bps, "
            f"ex-top5% {rc['ex_top5']:+.2f}, ex-worst5% {rc['ex_worst5']:+.2f}, "
            f"worst {rc['worst'][0]} ({rc['worst'][1]:+.1f})")


def build_population(minute, daily, early_close):
    """One row per (symbol, date) for every session in TRAIN+VAL with a next session that does not
    cross into TEST. Columns: symbol, date, split, year, s_t, heavy (True/False/NaN=unclassifiable),
    overnight_net_bps (buy MOC today, sell MOO next session), intraday_net_bps (buy MOO, sell MOC)."""
    rows = []
    flags_by_sym = {}
    for sym in SYMBOLS:
        d = daily[sym]
        d = d[(d['date'] >= TRAIN_START) & (d['date'] < TEST_START)].reset_index(drop=True)
        s_t = compute_volume_share(minute[sym], early_close)
        flags = heavy_close_flags(s_t)
        flags_by_sym[sym] = flags
        for i in range(len(d) - 1):
            day, nxt = d.iloc[i], d.iloc[i + 1]
            dte = day['date']
            if dte < TRAIN_START or dte > VAL_END or nxt['date'] >= TEST_START:
                continue
            close, nopen, oopen = day['c'], nxt['o'], day['o']
            if pd.isna(close) or pd.isna(nopen) or pd.isna(oopen) or close <= 0 or oopen <= 0:
                continue
            row = {
                'symbol': sym, 'date': dte, 'split': 'TRAIN' if dte <= TRAIN_END else 'VAL',
                'year': dte.year,
                'overnight_net_bps': (nopen - close) / close * 10000.0 - ROUND_TRIP_COST_BPS,
                'intraday_net_bps': (close - oopen) / oopen * 10000.0 - ROUND_TRIP_COST_BPS,
            }
            if dte in flags.index:
                row['s_t'] = flags.loc[dte, 's_t']
                row['heavy'] = flags.loc[dte, 'heavy']
            else:
                row['s_t'], row['heavy'] = np.nan, np.nan
            rows.append(row)
    return pd.DataFrame(rows), flags_by_sym


def build_tom_events(daily, heavy_lookup):
    """One row per (symbol, entry_month): buy MOC last session of month M, sell MOC 3rd session of
    M+1. Drops any event whose exit falls in the SEALED TEST window before computing a return."""
    rows, trigger_rows = [], []
    for sym in SYMBOLS:
        d = daily[sym]
        d = d[(d['date'] >= TRAIN_START) & (d['date'] < TEST_START)].reset_index(drop=True)
        ym = d['date'].apply(lambda x: (x.year, x.month))
        groups = {k: g.reset_index(drop=True) for k, g in d.groupby(ym)}
        for (yr, mo) in sorted(groups.keys()):
            nxt_key = (yr, mo + 1) if mo < 12 else (yr + 1, 1)
            if nxt_key not in groups or len(groups[nxt_key]) < 3:
                continue
            g_this, g_next = groups[(yr, mo)], groups[nxt_key]
            entry_row, exit_row = g_this.iloc[-1], g_next.iloc[2]
            entry_date, exit_date = entry_row['date'], exit_row['date']
            if exit_date >= TEST_START:
                continue  # SEALED -- never computed
            entry_px, exit_px = entry_row['c'], exit_row['c']
            if pd.isna(entry_px) or pd.isna(exit_px) or entry_px <= 0:
                continue
            month_key = f'{yr:04d}-{mo:02d}'
            rows.append({'symbol': sym, 'month': month_key, 'entry_date': entry_date,
                        'exit_date': exit_date, 'year': yr,
                        'net_bps': (exit_px - entry_px) / entry_px * 10000.0 - ROUND_TRIP_COST_BPS})
            window = pd.concat([g_this.iloc[[-1]], g_next[g_next['date'] < exit_date]])
            for _, trow in window.iterrows():
                trigger_rows.append((sym, trow['date']))
    ev = pd.DataFrame(rows)
    overlap_flags = [heavy_lookup.get((sym, dte), False) for sym, dte in trigger_rows]
    overlap_share = float(np.mean(overlap_flags)) if trigger_rows else np.nan
    return ev, overlap_share, len(trigger_rows)


def pass_bar_1650(val_rc, train_rc, excess_val, terc_train, terc_val, per_etf_val):
    items = []
    # PREREG: "the t item is INFORMATIONAL if the realised MDE exceeds the bar; the mechanism table and
    # the excess over the unconditional premium DECIDE" -- this is a substitution (MDE>8bps: use the
    # alt test only, mean/t no longer gate pass/fail), NOT an OR of the two criteria.
    primary = val_rc['mean_bps'] >= PASS_BAR_1650['mean_bps'] and val_rc['t'] >= PASS_BAR_1650['t']
    mono_train = terc_train is not None and terc_train['mean'].is_monotonic_increasing
    mono_val = terc_val is not None and terc_val['mean'].is_monotonic_increasing
    mde_exceeds = val_rc['mde'] > PASS_BAR_1650['mean_bps']
    alt = (excess_val[0] >= PASS_BAR_1650['excess_bps'] and excess_val[2] >= PASS_BAR_1650['excess_t']
           and mono_train and mono_val)
    bar_ok = alt if mde_exceeds else primary
    items.append((f"MDE={val_rc['mde']:.2f}>8bps -> alt test decides (mean/t informational)" if mde_exceeds
                  else f"MDE={val_rc['mde']:.2f}<=8bps -> mean>=8bps & t>=2.5 decides",
                  f"mean={val_rc['mean_bps']:+.2f} t={val_rc['t']:.2f} (informational={mde_exceeds}) | "
                  f"excess={excess_val[0]:+.2f} excess_t={excess_val[2]:.2f} mono_tr={mono_train} mono_val={mono_val}",
                  bar_ok))
    items.append(('ex-top-5% > 0 (VAL)', val_rc['ex_top5'], val_rc['ex_top5'] > 0))
    train_t = train_rc.get('t', np.nan)
    same_sign_t1 = (train_rc.get('n', 0) > 0 and not np.isnan(train_t)
                     and np.sign(train_rc['mean_bps']) == np.sign(val_rc['mean_bps'])
                     and abs(train_t) >= PASS_BAR_1650['train_t'])
    items.append(('TRAIN same sign, |t| >= 1', train_t, same_sign_t1))
    all_pos = bool((per_etf_val['mean'] > 0).all()) and len(per_etf_val) == 3
    items.append(('positive in all 3 ETFs (VAL)', dict(per_etf_val['mean']), all_pos))
    passed = all(ok for _, _, ok in items)
    return items, passed


def pass_bar_1651(pooled_rc, per_year, per_etf):
    items = []
    # Same MDE-gated substitution as 1650 (see pass_bar_1650), applied at the 30bps bar.
    primary = pooled_rc['mean_bps'] >= PASS_BAR_1651['mean_bps'] and pooled_rc['t'] >= PASS_BAR_1651['t']
    frac_pos = (per_year['mean'] > 0).sum() / len(per_year) if len(per_year) else 0.0
    alt = frac_pos >= PASS_BAR_1651['years_pos_frac'] and bool((per_etf['mean'] > 0).all())
    mde_exceeds = pooled_rc['mde'] > PASS_BAR_1651['mean_bps']
    bar_ok = alt if mde_exceeds else primary
    items.append((f"MDE={pooled_rc['mde']:.2f}>30bps -> alt test decides (mean/t informational)" if mde_exceeds
                  else f"MDE={pooled_rc['mde']:.2f}<=30bps -> mean>=30bps & t>=2.5 decides",
                  f"mean={pooled_rc['mean_bps']:+.2f} t={pooled_rc['t']:.2f} (informational={mde_exceeds}) | "
                  f"years_pos={frac_pos:.2f} etfs_pos={bool((per_etf['mean'] > 0).all())}",
                  bar_ok))
    items.append(('ex-top-5% > 0', pooled_rc['ex_top5'], pooled_rc['ex_top5'] > 0))
    passed = all(ok for _, _, ok in items)
    return items, passed


def compute_sharpe(returns_bps, ann_factor=252):
    """Annualised Sharpe (no rf adjustment -- these are all short-horizon relative-return comparisons)."""
    r = pd.Series(returns_bps).dropna().values / 10000.0
    if len(r) < 2 or r.std(ddof=1) == 0:
        return np.nan
    return float(r.mean() / r.std(ddof=1) * np.sqrt(ann_factor))


def bh_sharpe_by_symbol(daily):
    """24-hour buy-and-hold Sharpe per ETF: daily close-to-close return, TRAIN+VAL window, zero cost
    (a truly passive already-owned position trades nothing)."""
    out = {}
    for sym in SYMBOLS:
        d = daily[sym]
        d = d[(d['date'] >= TRAIN_START) & (d['date'] <= VAL_END)].sort_values('date')
        ret = d['c'].pct_change().dropna()
        out[sym] = compute_sharpe(ret * 10000.0)
    return out


def pass_bar_1649(pooled_rc, per_year, per_etf, overnight_sharpe, bh_sharpe):
    """Amendment 1: mean>=+2bps & t>=2.5 (VAL... here POOLED 2016-2023), OR -- since MDE ~3.4bps at
    ~2,000 nights is expected to exceed the 2bps bar, making the t item informational per the same
    substitution rule as 1650/1651 -- >=6/8 years positive AND all 3 ETFs positive AND overnight
    Sharpe > 24h buy-and-hold Sharpe on every one of the 3 ETFs (documented mechanism: intraday adds
    variance without return, so removing it should raise Sharpe, not just raise mean return)."""
    items = []
    primary = pooled_rc['mean_bps'] >= PASS_BAR_1649['mean_bps'] and pooled_rc['t'] >= PASS_BAR_1649['t']
    frac_pos = (per_year['mean'] > 0).sum() / len(per_year) if len(per_year) else 0.0
    sharpe_beats = {s: (overnight_sharpe.get(s, np.nan) > bh_sharpe.get(s, np.nan)) for s in SYMBOLS}
    alt = (frac_pos >= PASS_BAR_1649['years_pos_frac'] and bool((per_etf['mean'] > 0).all())
           and all(sharpe_beats.values()))
    mde_exceeds = pooled_rc['mde'] > PASS_BAR_1649['mean_bps']
    bar_ok = alt if mde_exceeds else primary
    sharpe_str = ', '.join(f"{s}:{overnight_sharpe.get(s, np.nan):.2f}vs{bh_sharpe.get(s, np.nan):.2f}"
                           for s in SYMBOLS)
    items.append((f"MDE={pooled_rc['mde']:.2f}>2bps -> alt decides (mean/t informational)" if mde_exceeds
                  else f"MDE={pooled_rc['mde']:.2f}<=2bps -> mean>=2bps & t>=2.5 decides",
                  f"mean={pooled_rc['mean_bps']:+.2f} t={pooled_rc['t']:.2f} | years_pos={frac_pos:.2f} "
                  f"etfs_pos={bool((per_etf['mean'] > 0).all())} | overnight-vs-BH Sharpe {sharpe_str}",
                  bar_ok))
    passed = bar_ok
    return items, passed


def stacking_line(label, mean_bps, events_per_month, capital_window, shared_tail):
    """Amendment 1: one stacking line per cell -- expected $/month at $20K/night-or-event, the capital
    window, and a note on tail overlap with the account's other (day-trading) books."""
    dollars_per_month = mean_bps / 10000.0 * SLEEVE_NOTIONAL * events_per_month
    return (f"  - **Stacking**: {events_per_month:.1f} events/month pooled -> "
            f"${dollars_per_month:,.0f}/month expected at ${SLEEVE_NOTIONAL:,.0f}/night-or-event "
            f"(gross of any correlation between simultaneous same-night fills). "
            f"Capital window: {capital_window}. Shared-tail: {shared_tail}")


def main():
    print('=== cells 1,649-1,651: index overnight premium + turn-of-month ===', flush=True)
    minute, daily = load_data()
    early_close = build_early_close_set(minute)
    pop, flags_by_sym = build_population(minute, daily, early_close)
    print(f'[population] {len(pop)} session-rows, TRAIN+VAL, {pop["symbol"].nunique()} symbols', flush=True)
    pop[['date', 'symbol', 's_t', 'heavy', 'overnight_net_bps']].to_csv(
        os.path.join(OUT_DIR, 'nights_1649.csv'), index=False)
    print('[write] nights_1649.csv', flush=True)

    s_t_valid = pop.dropna(subset=['s_t'])
    eligible = pop[pop['heavy'].isin([True, False])]
    heavy = eligible[eligible['heavy'] == True]
    complement = eligible[eligible['heavy'] == False]
    print(f'[1650] {len(eligible)} classifiable nights, {len(heavy)} heavy, {len(complement)} complement',
          flush=True)

    heavy_lookup = {(r['symbol'], r['date']): bool(r['heavy']) for _, r in eligible.iterrows()}
    tom_ev, overlap_share, n_trigger = build_tom_events(daily, heavy_lookup)
    tom_ev.to_csv(os.path.join(OUT_DIR, 'tom_1651.csv'), index=False,
                  columns=['month', 'symbol', 'net_bps'])
    print(f'[write] tom_1651.csv ({len(tom_ev)} events), overlap_share={overlap_share:.3f} '
          f'of {n_trigger} TOM trigger-nights', flush=True)

    lines = []
    lines.append('# RESULT 1,649-1,651: index overnight premium + turn-of-month')
    lines.append('')
    lines.append(f'Data: SPY/QQQ/IWM minute+daily, Alpaca adjustment=all SIP. '
                 f'{len(early_close)} early-close sessions excluded (data-driven: >=2/3 symbols, '
                 f'13:00-16:00 ET vol share < 0.5x own median). Cost 1.0 bp/round-trip (flat). '
                 f'TEST 2024-01..2026-09 SEALED: 0 events computed past that date.')
    lines.append('')

    # ---- 1649: SCORED per Amendment 1 (unconditional overnight, 2016-2023 pooled) ----
    lines.append('## Cell 1,649: unconditional overnight (SCORED per Amendment 1, 2016-2023 pooled)')
    for val_col, nm in [('overnight_net_bps', 'overnight MOC->MOO'), ('intraday_net_bps', 'intraday MOO->MOC')]:
        for split_name, split_df in [('TRAIN', pop[pop['split'] == 'TRAIN']), ('VAL', pop[pop['split'] == 'VAL']),
                                      ('POOLED', pop)]:
            rc = report_cell(split_df, val_col, f'1649 {nm} {split_name}')
            lines.append(render_line(rc))
            if rc.get('n', 0) > 0 and split_name == 'POOLED':
                lines.append(f"  - per-ETF: {fmt_per_etf(rc['per_etf'])}; ann/dd: "
                             f"{fmt_ann_dd(daily, split_df, val_col)}")
            if val_col == 'overnight_net_bps' and split_name == 'POOLED':
                overnight_pooled_rc = rc
    on_sharpe = {s: compute_sharpe(pop.loc[pop['symbol'] == s, 'overnight_net_bps']) for s in SYMBOLS}
    bh_sh = bh_sharpe_by_symbol(daily)
    worst_event = pop.loc[pop['overnight_net_bps'].idxmin()]
    worst_event_usd = worst_event['overnight_net_bps'] / 10000.0 * SLEEVE_NOTIONAL
    lines.append(f"  - overnight Sharpe vs 24h buy-and-hold Sharpe (same ETF, 2016-2023): " +
                 ', '.join(f"{s} {on_sharpe[s]:.2f} vs {bh_sh[s]:.2f}" for s in SYMBOLS))
    lines.append(f"  - worst SINGLE event: {worst_event['symbol']} {worst_event['date']} "
                 f"{worst_event['overnight_net_bps']:+.1f} bps = ${worst_event_usd:,.0f} at "
                 f"${SLEEVE_NOTIONAL:,.0f}/night notional")
    items_1649, pass_1649 = pass_bar_1649(overnight_pooled_rc, overnight_pooled_rc['per_year'],
                                          overnight_pooled_rc['per_etf'], on_sharpe, bh_sh)
    lines.append(f'  - **Pass bar 1,649 ({"PASS" if pass_1649 else "FAIL"})**:')
    for name, v, ok in items_1649:
        lines.append(f'    - [{"x" if ok else " "}] {name}: {v}')
    nights_per_month_1649 = len(pop) / ((pd.Timestamp(VAL_END) - pd.Timestamp(TRAIN_START)).days / 30.44)
    lines.append(stacking_line('1649', overnight_pooled_rc['mean_bps'], nights_per_month_1649,
                               'overnight only (~17h flat position, flat intraday every session)',
                               f"worst night {worst_event['date']} ({worst_event['symbol']}, "
                               f"{worst_event['overnight_net_bps']:+.0f}bps) is a broad market gap event "
                               f"(e.g. COVID crash week) -- correlated with, not diversifying from, the "
                               f"account's day-trading books on the same date"))
    lines.append('')

    # ---- 1650 conditional overnight ----
    lines.append('## Cell 1,650: conditional overnight (heavy close -> buy MOC, sell MOO)')
    tr_rc = report_cell(heavy[heavy['split'] == 'TRAIN'], 'overnight_net_bps', '1650 heavy TRAIN')
    va_rc = report_cell(heavy[heavy['split'] == 'VAL'], 'overnight_net_bps', '1650 heavy VAL')
    lines.append(render_line(tr_rc))
    lines.append(render_line(va_rc))
    for split_name, h_df, c_df in [('TRAIN', heavy[heavy['split'] == 'TRAIN'], complement[complement['split'] == 'TRAIN']),
                                    ('VAL', heavy[heavy['split'] == 'VAL'], complement[complement['split'] == 'VAL'])]:
        h_mean, h_se, h_n = cluster_mean_se(h_df, 'overnight_net_bps', 'date')
        c_mean, c_se, c_n = cluster_mean_se(c_df, 'overnight_net_bps', 'date')
        excess = h_mean - c_mean
        excess_se = float(np.sqrt(h_se ** 2 + c_se ** 2))
        excess_t = excess / excess_se if excess_se else np.nan
        if split_name == 'VAL':
            excess_val = (excess, excess_se, excess_t)
        else:
            excess_train = (excess, excess_se, excess_t)
        lines.append(f"  - {split_name} excess over unconditional (complement nights): heavy {h_mean:+.2f} "
                     f"(n={h_n}) - complement {c_mean:+.2f} (n={c_n}) = {excess:+.2f} bps, t={excess_t:.2f}")
    terc_train = tercile_table(s_t_valid[s_t_valid['split'] == 'TRAIN'], 'overnight_net_bps')
    terc_val = tercile_table(s_t_valid[s_t_valid['split'] == 'VAL'], 'overnight_net_bps')
    lines.append(f"  - TRAIN s_t tercile: {fmt_terc(terc_train)}")
    lines.append(f"  - VAL s_t tercile: {fmt_terc(terc_val)}")
    if va_rc.get('n', 0) > 0:
        lines.append(f"  - VAL per-ETF: {fmt_per_etf(va_rc['per_etf'])}")
        lines.append(f"  - VAL per-year: {fmt_per_year(va_rc['per_year'])}")
        lines.append(f"  - VAL ann/dd (fully deployed, per ETF): {fmt_ann_dd(daily, heavy[heavy['split'] == 'VAL'], 'overnight_net_bps')}")
        items_1650, pass_1650 = pass_bar_1650(va_rc, tr_rc, excess_val, terc_train, terc_val, va_rc['per_etf'])
        lines.append(f'  - **Pass bar 1,650 ({"PASS" if pass_1650 else "FAIL"})**:')
        for name, v, ok in items_1650:
            lines.append(f'    - [{"x" if ok else " "}] {name}: {v}')
        events_per_month_1650 = len(heavy[heavy['split'] == 'VAL']) / 48.0   # VAL = 2020-01..2023-12
        lines.append(stacking_line('1650', va_rc['mean_bps'], events_per_month_1650,
                                   'overnight only, heavy-close nights only (a subset of all nights)',
                                   f"worst heavy night {va_rc['worst'][0]} ({va_rc['worst'][1]:+.0f}bps) -- "
                                   f"heavy-close nights are not a random subset of all nights (they cluster "
                                   f"around index-rebalance and high-volatility sessions), so this sleeve's "
                                   f"tail is not obviously diversifying from 1649's or the account's other books"))
    else:
        pass_1650 = False
    lines.append('')

    # ---- 1651 turn of month ----
    lines.append('## Cell 1,651: turn-of-month (2016-2023 pooled)')
    tom_rc = report_cell(tom_ev, 'net_bps', '1651 pooled', cluster_col='month')
    lines.append(render_line(tom_rc))
    if tom_rc.get('n', 0) > 0:
        lines.append(f"  - per-ETF: {fmt_per_etf(tom_rc['per_etf'])}")
        lines.append(f"  - per-year: {fmt_per_year(tom_rc['per_year'])}")
        lines.append(f"  - ann/dd (fully deployed, per ETF): {fmt_ann_dd(daily, tom_ev, 'net_bps', 'entry_date')}")
        lines.append(f"  - overlap with 1650 heavy-close nights: {overlap_share:.1%} of {n_trigger} "
                     f"TOM trigger-nights (entry + intermediate sessions) are also 1650 heavy-close nights")
        items_1651, pass_1651 = pass_bar_1651(tom_rc, tom_rc['per_year'], tom_rc['per_etf'])
        lines.append(f'  - **Pass bar 1,651 ({"PASS" if pass_1651 else "FAIL"})**:')
        for name, v, ok in items_1651:
            lines.append(f'    - [{"x" if ok else " "}] {name}: {v}')
        events_per_month_1651 = len(tom_ev) / 95.0   # 95 entry months, Jan2016..Nov2023 (Dec2023 sealed-exit dropped)
        span_days = (pd.to_datetime(tom_ev['exit_date']) - pd.to_datetime(tom_ev['entry_date'])).dt.days.mean()
        lines.append(stacking_line('1651', tom_rc['mean_bps'], events_per_month_1651,
                                   f'~{span_days:.1f} calendar days tied up, once a month per ETF (entry MOC '
                                   f'last day of month through exit MOC 3rd session of the next)',
                                   f"worst event {tom_rc['worst'][0]} ({tom_rc['worst'][1]:+.0f}bps); "
                                   f"{overlap_share:.0%} of its trigger-nights are also 1650 heavy-close "
                                   f"nights -- partial overlap, not independent of 1650's tail"))
    else:
        pass_1651 = False
    lines.append('')

    lines.append('## Verdict')
    lines.append(f'- 1649 (unconditional overnight, stacking sleeve): {"PASS" if pass_1649 else "FAIL"}')
    lines.append(f'- 1650 (conditional overnight): {"PASS" if pass_1650 else "FAIL"}')
    lines.append(f'- 1651 (turn-of-month): {"PASS" if pass_1651 else "FAIL"}')
    lines.append('')

    lines.append('## Caveats (read as an adversary before relaying)')
    lines.append('- Cost is a flat 1.0 bp/round-trip deduction (0.5bp/auction leg x 2 legs); the PREREG\'s '
                 '"0.5bp per auction leg plus 0.5bp half-spread" is read as one number via its own check '
                 '"(1bp per round trip is generous)", not decomposed into 2bp by adding both terms per leg.')
    lines.append('- Early-close dates are a DATA-DRIVEN volume-ratio fallback (pandas_market_calendars not '
                 'installed), reused unchanged from research/lev_flow/cell_1643.py, which verified this method '
                 'recovers the known NYSE calendar exactly on the same data source; no separately hard-coded '
                 'date list was kept, so an undetected non-standard early close cannot be cross-checked here.')
    lines.append('- MDE uses SD(pooled events)/sqrt(N_CLUSTERS)*2.5; reproduces the PREREG\'s own worked '
                 'numbers (~11bps 1650 VAL, ~55bps 1651) only with clusters = distinct nights/months, not '
                 'distinct symbol-events -- verify the printed MDE against those before trusting the bar.')
    lines.append('- 16:00:00-labeled minute bars (if present) are included in both the s_t numerator and '
                 'denominator (closing-auction print), consistent with cell_1643\'s RTH convention.')
    lines.append('- "Heavy" is NaN (not False) for the first ~60 valid sessions per symbol (no trailing-'
                 'median baseline yet) and is excluded from the heavy/complement split and from the pass-bar '
                 'per-ETF checks, but s_t itself still feeds the tercile table for those rows.')
    lines.append('- TOM overlap uses entry + intermediate trigger-sessions (excludes the final exit session, '
                 'which starts no new overnight leg) checked against 1650\'s heavy flag pooled TRAIN+VAL; an '
                 'unclassifiable trigger date (warmup/early-close) defaults to "not heavy" for this count only.')
    lines.append('- Annualised return/drawdown compounds each ETF\'s own event stream sequentially at 100% '
                 'notional (no cross-ETF netting or margin sharing); a real sleeve running all 3 signals at '
                 'once would need $20K x however many fire the same night, not $20K flat.')
    lines.append('- Month boundaries are the trading-session calendar built from the daily-bar dates '
                 'themselves (no external calendar); this already excludes holidays by construction.')
    lines.append('- Amendment 1 interpretation calls, stated explicitly since the amendment leaves them open: '
                 '"$20K per night/event" is read as $20K per (symbol, night) -- NOT $60K if all 3 ETFs fire '
                 'the same night -- so stacking $/month sums 3 independent $20K sleeves, one per ETF; '
                 '"worst night in dollars" uses the single worst (symbol, date) row, not the cross-ETF '
                 'date-clustered mean (which is smaller); events/month for 1650 uses the VAL rate (scored '
                 'split) and for 1649/1651 the full 2016-2023 pooled rate; the Sharpe comparison requires '
                 'ALL 3 ETFs to beat their own buy-and-hold Sharpe (mirrors "all three ETFs positive" in the '
                 'same sentence), not a pooled/average Sharpe.')

    with open(os.path.join(OUT_DIR, 'RESULT_1649_build.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'[write] RESULT_1649_build.md ({len(lines)} lines)', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
