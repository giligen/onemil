"""
Independent rebuild of cells 1,646-1,648 (pre-announcement run-up around the
earnings announcement, expected date from the firm's own 8-K 2.02 history).

Built ONLY from the prose spec in research/edgar_desk/PREREG_1646.md -- the
first build's code/results were not opened before this script produced its
own numbers.

Usage:
    python3 rebuild_1646.py [--debug N]   # --debug limits to first N CIKs
"""
import re
import sys
import time

import numpy as np
import pandas as pd

RAW = "research/edgar_desk/events_raw.csv"
BARS1 = "research/overnight_high/alpaca_daily_2019_2024H1.parquet"
BARS2 = "research/overnight_high/panel_2024_2026.parquet"
OUT_CSV = "research/edgar_desk/events_1646_rebuild.csv"

TEST_TICKER_RE = re.compile(r'^Z[A-Z]ZZT$')
COST_BPS_PER_LEG = 5.0
SPLIT_GUARD_RET = 0.60


def log(msg):
    print(f"[rebuild_1646] {time.strftime('%H:%M:%S')} {msg}", flush=True)


# ---------------------------------------------------------------- bars ----
def load_bars():
    log("loading daily bars (both parquet files)...")
    cols = ['symbol', 'bar_date', 'open', 'high', 'low', 'close', 'volume']
    b1 = pd.read_parquet(BARS1, columns=cols)
    b2 = pd.read_parquet(BARS2, columns=cols)
    bars = pd.concat([b1, b2], ignore_index=True)
    before = len(bars)
    bars = bars.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
    bars = bars[(bars.open > 0) & (bars.high > 0) & (bars.low > 0) &
                (bars.close > 0) & (bars.volume > 0)]
    log(f"dropped {before - len(bars)} zero/NaN-OHLCV rows of {before}")
    bars['bar_date'] = pd.to_datetime(bars['bar_date']).dt.normalize()
    bars = bars.sort_values(['symbol', 'bar_date'])
    dup = bars.duplicated(['symbol', 'bar_date']).sum()
    bars = bars.drop_duplicates(['symbol', 'bar_date'], keep='first').reset_index(drop=True)
    log(f"dropped {dup} duplicate (symbol,bar_date) rows; {len(bars)} bars, "
        f"{bars.symbol.nunique()} symbols remain")
    return bars


def add_rolling(bars):
    log("computing dollar volume, adv20/dvol20, single-day-return split flag...")
    bars['dollar_vol'] = bars['close'] * bars['volume']
    g = bars.groupby('symbol', sort=False)
    bars['adv20'] = g['volume'].transform(lambda s: s.rolling(20, min_periods=20).mean())
    bars['dvol20'] = g['dollar_vol'].transform(lambda s: s.rolling(20, min_periods=20).mean())
    bars['ret1'] = g['close'].transform(lambda s: s.pct_change())
    bars['bad_day'] = bars['ret1'].abs() > SPLIT_GUARD_RET
    log(f"flagged {int(bars['bad_day'].sum())} single-day |return|>{SPLIT_GUARD_RET:.0%} bars "
        f"(unadjusted-split guard)")
    return bars


# -------------------------------------------------------------- events ----
def load_events():
    log("scanning events_raw.csv for form=='8-K' with item 2.02 ...")
    usecols = ['cik', 'symbol', 'form', 'acceptance_datetime', 'items']
    chunks = []
    nrows = 0
    for chunk in pd.read_csv(RAW, usecols=usecols, chunksize=500_000, dtype={'cik': str}):
        nrows += len(chunk)
        sub = chunk[chunk['form'] == '8-K'].copy()
        lst = sub['items'].astype(str).str.split(';')
        mask = lst.apply(lambda x: '2.02' in x)
        chunks.append(sub[mask])
    ev = pd.concat(chunks, ignore_index=True)
    log(f"scanned {nrows} rows -> {len(ev)} form=8-K item=2.02 rows")

    ts_utc = pd.to_datetime(ev['acceptance_datetime'], utc=True)
    ts_et = ts_utc.dt.tz_convert('America/New_York')
    ev['acc_et'] = ts_et
    ev['et_date'] = ts_et.dt.normalize().dt.tz_localize(None)
    ev['after_close'] = ts_et.dt.hour * 60 + ts_et.dt.minute >= 16 * 60

    ev = ev.sort_values(['cik', 'et_date', 'acc_et'])
    dup = ev.duplicated(['cik', 'et_date']).sum()
    ev = ev.drop_duplicates(['cik', 'et_date'], keep='first')
    log(f"dropped {dup} same-(cik,et_date) duplicate/amended filings")
    ev = ev.sort_values(['cik', 'et_date']).reset_index(drop=True)
    return ev


# --------------------------------------------------- firm-quarter match ----
def build_firm_quarters(ev, max_ciks=None):
    log("matching each filing to its same-fiscal-quarter filing one year earlier...")
    records = []
    n_ciks = 0
    n_skip_no_l1 = 0
    n_skip_drift = 0
    for cik, grp in ev.groupby('cik', sort=False):
        n_ciks += 1
        if max_ciks and n_ciks > max_ciks:
            break
        dates = grp['et_date'].values.astype('datetime64[D]')
        # strip tz while STILL tz-aware so this keeps the ET wall-clock instant
        # (Series.values on a tz-aware column silently converts to UTC instead).
        accs = grp['acc_et'].dt.tz_localize(None).values
        syms = grp['symbol'].values
        after = grp['after_close'].values
        n = len(dates)
        if n < 2:
            continue
        for i in range(1, n):
            D = dates[i]
            target1 = D - np.timedelta64(365, 'D')
            prior = dates[:i]
            diffs1 = np.abs((prior - target1).astype('timedelta64[D]').astype(int))
            j1 = int(np.argmin(diffs1))
            if diffs1[j1] > 45:
                n_skip_no_l1 += 1
                continue
            L1 = prior[j1]
            L1_acc = accs[j1]
            drift_days = 0
            L2 = None
            if j1 >= 1:
                target2 = L1 - np.timedelta64(365, 'D')
                prior2 = dates[:j1]
                diffs2 = np.abs((prior2 - target2).astype('timedelta64[D]').astype(int))
                j2 = int(np.argmin(diffs2))
                if diffs2[j2] <= 45:
                    L2 = prior2[j2]
                    drift_days = int((L1 - L2).astype('timedelta64[D]').astype(int)) - 365
                    if abs(drift_days) > 7:
                        n_skip_drift += 1
                        continue
            # E is ~one year AFTER L1 (last year's same-quarter release), projected
            # forward by the observed year-over-year drift.
            E_date_raw = L1 + np.timedelta64(365 + int(drift_days), 'D')
            records.append((
                cik, syms[i], pd.Timestamp(D), pd.Timestamp(L1), L1_acc,
                (pd.Timestamp(L2) if L2 is not None else pd.NaT),
                drift_days, pd.Timestamp(E_date_raw), accs[i], bool(after[i]),
            ))
    log(f"scanned {n_ciks} CIKs; {len(records)} candidate firm-quarters; "
        f"{n_skip_no_l1} skipped (no L1 within +-45d), {n_skip_drift} skipped (drift>7d)")
    cols = ['cik', 'symbol', 'D_et_date', 'L1', 'L1_acc', 'L2', 'drift_days',
            'E_date_raw', 'D_acc', 'D_after_close']
    fq = pd.DataFrame.from_records(records, columns=cols)
    return fq


# ------------------------------------------------------------ scoring ----
def score(fq, bars, calendar):
    log("mapping expected/actual dates to trading sessions and scoring...")
    cal_vals = calendar.values  # datetime64[ns], sorted

    def idx_of(dates):
        return np.searchsorted(cal_vals, dates, side='left')

    E_idx = idx_of(fq['E_date_raw'].values.astype('datetime64[ns]'))
    d_target = fq['D_et_date'].values.astype('datetime64[ns]') + \
        np.where(fq['D_after_close'].values, np.timedelta64(1, 'D'), np.timedelta64(0, 'D'))
    D_idx = idx_of(d_target)

    fq = fq.copy()
    fq['E_idx'] = E_idx
    fq['D_idx'] = D_idx
    fq['E6_idx'] = E_idx - 6
    fq['E5_idx'] = E_idx - 5
    fq['E3_idx'] = E_idx - 3
    fq['E1_idx'] = E_idx - 1
    fq['Ep1_idx'] = E_idx + 1
    fq['Ep3_idx'] = E_idx + 3

    n_cal = len(cal_vals)
    in_range = (fq['E6_idx'] >= 0) & (fq['Ep3_idx'] < n_cal)
    log(f"dropping {int((~in_range).sum())} events with a session window outside the calendar")
    fq = fq[in_range].reset_index(drop=True)

    # causality guard: L1 (and L2) must be accepted strictly before session E-6
    e6_date = calendar[fq['E6_idx'].values]
    causal_ok = fq['L1_acc'].dt.tz_localize(None).values.astype('datetime64[ns]') < e6_date.values
    l2_ok = fq['L2'].isna().values | (fq['L2'].values.astype('datetime64[ns]') < e6_date.values)
    guard = causal_ok & l2_ok
    log(f"causality guard (L1/L2 accepted before session E-6): {int((~guard).sum())} violations dropped")
    fq = fq[guard].reset_index(drop=True)

    # test ticker exclude
    is_test_ticker = fq['symbol'].str.match(TEST_TICKER_RE)
    log(f"excluding {int(is_test_ticker.sum())} test-ticker symbols")
    fq = fq[~is_test_ticker].reset_index(drop=True)

    # entry must be knowable before the actual event: D must not precede E-5
    early = fq['D_idx'] < fq['E5_idx']
    log(f"dropping {int(early.sum())} firm-quarters where the actual release preceded "
        f"session E-5 (estimate too late, not tradeable as pre-announcement)")
    fq = fq[~early].reset_index(drop=True)

    symbols_needed = set(fq['symbol'].unique()) | {'SPY'}
    bars_small = bars[bars['symbol'].isin(symbols_needed)]
    log(f"indexing bars for {len(symbols_needed)} symbols ({len(bars_small)} rows)...")
    frames = {sym: g.set_index('bar_date') for sym, g in bars_small.groupby('symbol', sort=False)}
    date_sets = {sym: set(f.index) for sym, f in frames.items()}
    bad_sets = {sym: set(f.index[f['bad_day']]) for sym, f in frames.items()}

    out_rows = []
    n_gap = n_univ = n_split = n_nodata = 0
    for row in fq.itertuples(index=False):
        sym = row.symbol
        if sym not in frames:
            n_nodata += 1
            continue
        fr = frames[sym]
        dset = date_sets[sym]
        bset = bad_sets[sym]
        req_idx = list(range(row.E6_idx, row.Ep1_idx + 1))
        req_dates = calendar[req_idx]
        if not all(d in dset for d in req_dates):
            n_gap += 1
            continue
        if any(d in bset for d in req_dates):
            n_split += 1
            continue
        e6_row = fr.loc[calendar[row.E6_idx]]
        if not (e6_row['close'] >= 3.0 and e6_row['dvol20'] >= 5_000_000):
            n_univ += 1
            continue

        e5_date = calendar[row.E5_idx]
        e1_date = calendar[row.E1_idx]
        e_date = calendar[row.E_idx]
        ep1_date = calendar[row.Ep1_idx]

        entry_open = fr.loc[e5_date, 'open']
        exit1646_idx = min(row.D_idx, row.E1_idx) if row.D_idx < row.E1_idx else row.E1_idx
        exit1646_date = calendar[exit1646_idx]
        exit1646_close = fr.loc[exit1646_date, 'close']
        exit1647_close = fr.loc[ep1_date, 'close']

        raw1646 = exit1646_close / entry_open - 1.0
        raw1647 = exit1647_close / entry_open - 1.0
        net1646 = raw1646 * 1e4 - 2 * COST_BPS_PER_LEG
        net1647 = raw1647 * 1e4 - 2 * COST_BPS_PER_LEG

        spy_fr = frames['SPY']
        spy_dset = date_sets['SPY']
        if e5_date in spy_dset and exit1646_date in spy_dset and ep1_date in spy_dset:
            spy_entry = spy_fr.loc[e5_date, 'open']
            spy_ret1646 = spy_fr.loc[exit1646_date, 'close'] / spy_entry - 1.0
            spy_ret1647 = spy_fr.loc[ep1_date, 'close'] / spy_entry - 1.0
            adj1646 = net1646 - spy_ret1646 * 1e4
            adj1647 = net1647 - spy_ret1647 * 1e4
        else:
            adj1646 = adj1647 = np.nan

        # 1648 mechanism: prior-year (L1) announcement-window volume ratio
        hist_ratio = np.nan
        l1_idx = int(np.searchsorted(cal_vals, np.datetime64(row.L1), side='left'))
        l1_e6 = l1_idx - 6
        l1_e5 = l1_idx - 5
        l1_ep1 = l1_idx + 1
        if l1_e6 >= 0 and l1_ep1 < n_cal:
            l1_req = calendar[l1_e6:l1_ep1 + 1]
            if all(d in dset for d in l1_req):
                base = fr.loc[calendar[l1_e6], 'adv20']
                # mean volume over L1's own [-5,+1] window
                win_dates = calendar[l1_e5:l1_ep1 + 1]
                vols = [fr.loc[d, 'volume'] for d in win_dates]
                if base and base > 0:
                    hist_ratio = float(np.mean(vols)) / float(base)

        early_arrival = bool(row.D_idx < row.E1_idx)
        entry_year = e5_date.year
        entry_month = e5_date.month
        if entry_year <= 2022:
            split = 'TRAIN'
        elif entry_year == 2023 or (entry_year == 2024 and entry_month <= 6):
            split = 'VAL'
        else:
            split = 'TEST'

        out_rows.append(dict(
            cik=row.cik, symbol=sym, split=split,
            D_et_date=row.D_et_date.date(), L1=row.L1.date(),
            L2=(row.L2.date() if pd.notna(row.L2) else None),
            drift_days=row.drift_days, E_date=e_date.date(),
            entry_session=e5_date.date(), exit1646_session=exit1646_date.date(),
            exit1647_session=ep1_date.date(), early_arrival=early_arrival,
            hit_e1_e1=bool(abs(row.D_idx - row.E_idx) <= 1),
            hit_e3_e3=bool(abs(row.D_idx - row.E_idx) <= 3),
            gap_sessions=int(row.D_idx - row.E_idx),
            net_bps_1646=(net1646 if split != 'TEST' else np.nan),
            net_bps_1647=(net1647 if split != 'TEST' else np.nan),
            spy_adj_1646=(adj1646 if split != 'TEST' else np.nan),
            spy_adj_1647=(adj1647 if split != 'TEST' else np.nan),
            hist_vol_ratio=(hist_ratio if split != 'TEST' else np.nan),
        ))
    log(f"guard drops -> gap:{n_gap} split:{n_split} universe:{n_univ} nodata:{n_nodata}; "
        f"{len(out_rows)} events survive")
    return pd.DataFrame(out_rows)


def main():
    debug = None
    if '--debug' in sys.argv:
        debug = int(sys.argv[sys.argv.index('--debug') + 1])

    bars = load_bars()
    bars = add_rolling(bars)
    calendar = pd.DatetimeIndex(sorted(bars.loc[bars.symbol == 'SPY', 'bar_date'].unique()))
    log(f"master calendar (SPY sessions): {len(calendar)} sessions "
        f"{calendar.min().date()} .. {calendar.max().date()}")

    ev = load_events()
    fq = build_firm_quarters(ev, max_ciks=debug)
    result = score(fq, bars, calendar)
    result.to_csv(OUT_CSV, index=False)
    log(f"wrote {len(result)} rows to {OUT_CSV}")

    for split in ('TRAIN', 'VAL'):
        sub = result[result.split == split]
        if len(sub) == 0:
            continue
        hit1 = sub['hit_e1_e1'].mean()
        hit3 = sub['hit_e3_e3'].mean()
        medgap = sub['gap_sessions'].abs().median()
        log(f"{split}: n={len(sub)} hit[E-1,E+1]={hit1:.1%} hit[E-3,E+3]={hit3:.1%} "
            f"median|gap|={medgap:.1f} sessions "
            f"1646 mean={sub['net_bps_1646'].mean():.1f}bps "
            f"1647 mean={sub['net_bps_1647'].mean():.1f}bps")


if __name__ == '__main__':
    main()
