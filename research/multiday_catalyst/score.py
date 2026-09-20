#!/usr/bin/env python3
"""multiday_catalyst: cells 1,285 (long continuation) / 1,286 (matched control).

Scores the PREREG spec (research/multiday_catalyst/PREREG.md) exactly. Read-only against
data/cache.db, data/research/orb_news_catalyst_nightly.csv, data/research/orb_asset_class_map_20260711.csv,
data/research/databento/pit_definition/*.parquet. Writes trades_signal.csv, trades_control.csv,
diagnostics.json under research/multiday_catalyst/.
"""
import sys, os, json, sqlite3, re
from datetime import date
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil')
from research.scripts.pit_listings import PitListings, is_test_ticker

ROOT = '/home/ec2-user/onemil'
OUT = os.path.join(ROOT, 'research/multiday_catalyst')
CACHE = os.path.join(ROOT, 'data/cache.db')
NEWS_CSV = os.path.join(ROOT, 'data/research/orb_news_catalyst_nightly.csv')
CLASS_MAP_CSV = os.path.join(ROOT, 'data/research/orb_asset_class_map_20260711.csv')

HALF_SPREAD_1555 = 0.001725  # frames14 f45_minute_table.csv, clock_m=955, median, % of price
R_MIN_PCT = 0.01
ADV20_MIN = 500_000
PRICE_MIN = 5.0
DVOL_PCTL = 0.90
BOOK_SLOTS = 10
BOOK_EQUITY = 66_000.0
BOOK_RISK_PCT = 0.01

def log(*a):
    print(*a, flush=True)

def load_class_map():
    m = {}
    with open(CLASS_MAP_CSV, newline='') as fh:
        import csv
        r = csv.DictReader(fh)
        for row in r:
            m[row['symbol']] = row['asset_class']
    return m

def load_daily_bars():
    conn = sqlite3.connect(f'file:{CACHE}?mode=ro', uri=True, timeout=60)
    q = "SELECT symbol, bar_date, open, high, low, close, volume FROM daily_bars WHERE bar_date >= '2024-06-01' AND bar_date <= '2026-09-18'"
    df = pd.read_sql_query(q, conn, parse_dates=['bar_date'])
    conn.close()
    df['symbol'] = df['symbol'].astype('category')
    for c in ('open', 'high', 'low', 'close'):
        df[c] = df[c].astype('float32')
    df['volume'] = df['volume'].astype('int64')
    df = df.sort_values(['symbol', 'bar_date']).reset_index(drop=True)
    return df

def load_news():
    df = pd.read_csv(NEWS_CSV, usecols=['symbol', 'day', 'n_articles'])
    df['day'] = pd.to_datetime(df['day'])
    df['has_news'] = df['n_articles'].astype(int) > 0
    return df[['symbol', 'day', 'has_news']]

def build_pit_month_universe(pit, class_map, months):
    """{month_key: frozenset(symbols)} for symbols pit-listed that month AND classified stock/wrapper."""
    out = {}
    good_classes = {'stock', 'wrapper'}
    for mk in months:
        d = pd.Timestamp(f'{mk[:4]}-{mk[4:]}-15')
        listed = pit.listed_symbols(d)
        out[mk] = frozenset(s for s in listed if class_map.get(s) in good_classes)
    return out

def build_corp_action_months(pit, months):
    """{month_key: frozenset(symbols with a security_update_action == 'M' record that month)}"""
    out = {}
    for mk in months:
        df = pit._month(mk)
        m = df.loc[df['security_update_action'] == 'M', 'raw_symbol']
        out[mk] = frozenset(m.tolist())
    return out

def main():
    log('=== load daily_bars ===')
    db = load_daily_bars()
    log(f'daily_bars rows={len(db):,} symbols={db.symbol.nunique():,} '
        f'range={db.bar_date.min().date()}..{db.bar_date.max().date()}')

    log('=== per-symbol causal features ===')
    db['month'] = db['bar_date'].dt.strftime('%Y%m')
    g = db.groupby('symbol', observed=True)
    db['prior_close'] = g['close'].shift(1)
    db['overnight_ret'] = db['open'] / db['prior_close'] - 1.0
    db['adv20'] = g['volume'].apply(lambda s: s.shift(1).rolling(20, min_periods=10).mean()).reset_index(level=0, drop=True)
    db['dollar_vol'] = db['close'].astype('float64') * db['volume']
    db['is_green'] = db['close'] > db['open']

    log('=== pit universe by month ===')
    pit = PitListings()
    class_map = load_class_map()
    months = sorted(db['month'].unique())
    lo_cov, hi_cov = pit.coverage
    log(f'pit coverage {lo_cov}..{hi_cov}')
    months_in_cov = [m for m in months if lo_cov <= m <= hi_cov]
    dropped_months = [m for m in months if m not in months_in_cov]
    if dropped_months:
        log(f'WARNING: {len(dropped_months)} months outside PIT coverage, excluded entirely: {dropped_months}')
    pit_uni = build_pit_month_universe(pit, class_map, months_in_cov)
    corp_action = build_corp_action_months(pit, months_in_cov)

    db = db[db['month'].isin(months_in_cov)].copy()
    valid_keys = set()
    for mk, syms in pit_uni.items():
        for s in syms:
            valid_keys.add(mk + '|' + s)
    db['mkey'] = db['month'] + '|' + db['symbol'].astype(str)
    db['pit_ok'] = db['mkey'].isin(valid_keys)
    db = db.drop(columns=['mkey'])
    n_all = len(db)
    log(f'rows after pit-coverage-month restriction: {n_all:,}')

    log('=== universe day gate ===')
    db['uni_ok'] = db['pit_ok'] & (db['close'] >= PRICE_MIN) & (db['adv20'] >= ADV20_MIN) & (~db['symbol'].astype(str).str.match(r'^Z[A-Z]ZZT'))
    log(f'universe-eligible rows: {int(db.uni_ok.sum()):,} of {n_all:,}')

    log('=== day-D 90th pctile dollar volume among universe ===')
    uni = db[db.uni_ok]
    pctl = uni.groupby('bar_date')['dollar_vol'].quantile(DVOL_PCTL)
    db = db.merge(pctl.rename('dvol_p90'), left_on='bar_date', right_index=True, how='left')
    db['dvol_ok'] = db['dollar_vol'] >= db['dvol_p90']

    log('=== news merge ===')
    news = load_news()
    news_cov_lo, news_cov_hi = news.day.min(), news.day.max()
    log(f'news coverage {news_cov_lo.date()}..{news_cov_hi.date()}')
    db = db.merge(news, left_on=['symbol', 'bar_date'], right_on=['symbol', 'day'], how='left')
    db['has_news'] = db['has_news'].fillna(False)
    db['news_covered_day'] = db['bar_date'].between(news_cov_lo, news_cov_hi)

    candidate = db[db.uni_ok & db.dvol_ok & db.is_green].copy()
    log(f'candidates (universe & dvol90 & green), pre news split: {len(candidate):,}')

    sig_days = candidate[candidate.has_news & candidate.news_covered_day].copy()
    ctl_days = candidate[(~candidate.has_news) ].copy()
    log(f'signal-day rows (has_news, news-covered): {len(sig_days):,}')
    log(f'control-day rows (no news, incl. pre-news-coverage era): {len(ctl_days):,}')

    log('=== index daily_bars by (symbol, bar_date) for D+1..D+3 lookups ===')
    db_idx = db.set_index(['symbol', 'bar_date']).sort_index()
    by_symbol_dates = {sym: sub['bar_date'].values for sym, sub in db.groupby('symbol', observed=True)}

    def next_n_trading_days(sym, d, n):
        arr = by_symbol_dates.get(sym)
        if arr is None:
            return []
        pos = np.searchsorted(arr, np.datetime64(d))
        if pos >= len(arr) or arr[pos] != np.datetime64(d):
            return []
        out = []
        for k in range(1, n + 1):
            if pos + k < len(arr):
                out.append(arr[pos + k])
        return out

    def score_cell(rows, label):
        log(f'--- scoring {label}: {len(rows)} candidate signal-days ---')
        trades = []
        n_split_excl = 0
        n_corp_excl = 0
        n_r_skip = 0
        n_no_hold = 0
        held = {}  # symbol -> exit_date, for the one-position-per-symbol / no-resignal-while-held rule
        rows = rows.sort_values(['bar_date', 'symbol'])
        for row in rows.itertuples():
            sym = row.symbol
            d = row.bar_date
            if sym in held and held[sym] >= d:
                continue  # re-signal while held: skip
            fut = next_n_trading_days(sym, d, 3)
            if len(fut) < 1:
                n_no_hold += 1
                continue
            close_d = float(row.close)
            low_d = float(row.low)
            R_raw = close_d - low_d
            if R_raw < R_MIN_PCT * close_d or R_raw <= 0:
                n_r_skip += 1
                continue
            window_dates = [d] + list(fut)
            month_keys = sorted(set(pd.Timestamp(x).strftime('%Y%m') for x in window_dates))
            corp_hit = any(sym in corp_action.get(mk, frozenset()) for mk in month_keys)
            split_hit = False
            for wd in fut:
                r = db_idx.loc[(sym, wd)]
                ov = r['overnight_ret'] if not isinstance(r, pd.DataFrame) else r['overnight_ret'].iloc[0]
                if pd.notna(ov) and abs(ov) > 0.40:
                    split_hit = True
                    break
            if split_hit:
                n_split_excl += 1
                continue
            if corp_hit:
                n_corp_excl += 1
                continue
            entry_paid = close_d * (1 + HALF_SPREAD_1555)
            entry_gross = close_d
            closes = []  # D+1, D+2, D+3 closes (gross, whatever exist)
            for wd in fut:
                r = db_idx.loc[(sym, wd)]
                if isinstance(r, pd.DataFrame):
                    r = r.iloc[0]
                closes.append(float(r['close']))
            exit_price = None
            exit_price_gross = None
            exit_day = None
            exit_kind = None
            stop_price = low_d
            for i, wd in enumerate(fut):
                r = db_idx.loc[(sym, wd)]
                if isinstance(r, pd.DataFrame):
                    r = r.iloc[0]
                if float(r['low']) <= stop_price:
                    fill = min(float(r['open']), stop_price) if float(r['open']) < stop_price else stop_price
                    exit_price = fill * (1 - HALF_SPREAD_1555)
                    exit_price_gross = fill
                    exit_day = wd
                    exit_kind = 'stop'
                    break
            if exit_price is None:
                last_d = fut[-1]
                r = db_idx.loc[(sym, last_d)]
                if isinstance(r, pd.DataFrame):
                    r = r.iloc[0]
                exit_price = float(r['close'])
                exit_price_gross = exit_price
                exit_day = last_d
                exit_kind = 'time' if len(fut) == 3 else 'time_short'
            pnl_R = (exit_price - entry_paid) / R_raw
            pnl_R_gross = (exit_price_gross - entry_gross) / R_raw
            held[sym] = exit_day
            rec = dict(
                date=pd.Timestamp(d).strftime('%Y-%m-%d'), symbol=str(sym), pnl_R=pnl_R,
                pnl_R_gross=pnl_R_gross,
                close_d=close_d, low_d=low_d, R_raw=R_raw, entry_paid=entry_paid,
                exit_price=exit_price, exit_day=pd.Timestamp(exit_day).strftime('%Y-%m-%d'),
                exit_kind=exit_kind, hold_days=len(fut),
                asset_class=class_map.get(str(sym), 'unknown'),
            )
            for i in range(3):
                rec[f'close_d{i+1}'] = closes[i] if i < len(closes) else None
                rec[f'R_d{i+1}'] = ((closes[i] - entry_gross) / R_raw) if i < len(closes) else None
            trades.append(rec)
        log(f'{label}: trades={len(trades)} split_excl={n_split_excl} corp_excl={n_corp_excl} '
            f'r_skip={n_r_skip} no_holding_days={n_no_hold}')
        return pd.DataFrame(trades), dict(split_excl=n_split_excl, corp_excl=n_corp_excl,
                                           r_skip=n_r_skip, no_hold=n_no_hold, n_candidates=len(rows))

    sig_trades, sig_meta = score_cell(sig_days, 'SIGNAL (1,285)')
    ctl_trades, ctl_meta = score_cell(ctl_days, 'CONTROL (1,286)')

    sig_trades.to_csv(os.path.join(OUT, 'trades_signal_raw.csv'), index=False)
    ctl_trades.to_csv(os.path.join(OUT, 'trades_control_raw.csv'), index=False)

    log('=== price-scale check (200-sample) ===')
    conn = sqlite3.connect(f'file:{CACHE}?mode=ro', uri=True, timeout=60)
    sample = sig_trades.sample(n=min(200, len(sig_trades)), random_state=0) if len(sig_trades) else sig_trades
    mism = 0
    checked = 0
    for row in sample.itertuples():
        cur = conn.execute(
            "SELECT close FROM intraday_bars_1min WHERE symbol=? AND bar_date=? AND "
            "(strftime('%H:%M', timestamp) = '15:59' OR strftime('%H:%M', timestamp) = '16:00') "
            "ORDER BY timestamp DESC LIMIT 1", (row.symbol, row.date))
        r = cur.fetchone()
        if r is None:
            continue
        checked += 1
        if abs(r[0] - row.close_d) / row.close_d > 0.005:
            mism += 1
    conn.close()
    log(f'price-scale check: checked={checked} mismatched>0.5%={mism}')

    diag = dict(sig_meta=sig_meta, ctl_meta=ctl_meta, price_scale_checked=checked, price_scale_mismatch=mism,
                news_coverage=[str(news_cov_lo.date()), str(news_cov_hi.date())],
                pit_coverage=[lo_cov, hi_cov], months_dropped_pit_coverage=dropped_months,
                n_daily_bars_rows=n_all, n_universe_rows=int(db.uni_ok.sum()))
    with open(os.path.join(OUT, 'diagnostics_raw.json'), 'w') as f:
        json.dump(diag, f, indent=2, default=str)
    log('=== DONE ===')

if __name__ == '__main__':
    main()
