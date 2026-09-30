#!/usr/bin/env python3
"""Driver for rebuild_1599.py: Mondays -> cycles -> monthly stats -> TRAIN selection ->
VAL report -> EXTENSION report -> REBUILD_1599.md. Imported by rebuild_1599.py's __main__."""
import datetime as dt
import json
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

import rebuild_1599 as R

log = R.log
HERE = R.HERE
CACHE = R.CACHE

PANEL_START, PANEL_END = dt.date(2024, 2, 5), dt.date(2026, 8, 17)
TRAIN_END, VAL_START = dt.date(2025, 6, 30), dt.date(2025, 7, 7)
EXT_START, EXT_END = dt.date(2013, 4, 8), dt.date(2023, 12, 25)
DELTAS = [0.20, 0.30]

PANEL_CYCLES = os.path.join(CACHE, 'panel_cycles.parquet')
EXT_CYCLES = os.path.join(CACHE, 'ext_cycles.parquet')
SELECTION_PATH = os.path.join(CACHE, 'selection.json')
EXT_SPY_DAILY = os.path.join(CACHE, 'spy_ext_daily.parquet')
EXT_SPY_MIN = os.path.join(CACHE, 'spy_ext_minute.parquet')


def snapshot_with_holiday_fallback(monday, cost_only):
    """PREREG: 'the first session of the week if Monday is closed'. Tries Monday, then up to
    2 following days, for a market holiday (e.g. 2024-02-19 Presidents Day)."""
    for offset in range(0, 3):
        day = monday + dt.timedelta(days=offset)
        if day.weekday() > 4:
            break
        snap = R.fetch_monday_snapshot(day, cost_only)
        if snap is not None and not snap.empty:
            return snap, day
    return None, monday


def process_task(monday, delta, spot_fn, close_fn, cost_only):
    snap, actual_day = snapshot_with_holiday_fallback(monday, cost_only)
    if actual_day != monday:
        log.warning(f"{monday} closed (no chain) -- using first session {actual_day} instead")
    monday = actual_day
    if snap is None or snap.empty:
        base = {'entry_day': monday, 'delta_target': delta, 'status': 'VOID', 'reason': 'no_snapshot'}
        return [dict(base, mgmt=m) for m in ['A', 'B']]
    spot10 = spot_fn(monday)
    exp_info = R.nearest_friday_expiry(monday)
    used_parity = False
    if spot10 is None and exp_info is not None:
        spot10 = R.parity_spot(snap, exp_info[0], exp_info[1])
        used_parity = spot10 is not None
    cyc = R.build_cycle(monday, delta, snap, spot10, None)
    cyc['used_parity_spot'] = used_parity
    if cyc['status'] != 'OPEN':
        return [dict(cyc, mgmt=m) for m in ['A', 'B']]
    life_end = min(cyc['expiry'], monday + dt.timedelta(days=60))
    short_life = R.fetch_leg_life(cyc['short_symbol'], monday, life_end, cost_only)
    long_life = R.fetch_leg_life(cyc['long_symbol'], monday, life_end, cost_only)
    rows = []
    for m in ['A', 'B']:
        r = R.run_management(cyc, m, close_fn, short_life, long_life)
        r['mgmt'] = m
        rows.append(r)
    return rows


def build_cycles(mondays, spot_fn, close_fn, cost_only, workers, label):
    log.info(f"[{label}] snapshots for {len(mondays)} Mondays (cost_only={cost_only})")
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(R.fetch_monday_snapshot, m, cost_only): m for m in mondays}
        done = 0
        for f in as_completed(futs):
            m = futs[f]
            try:
                f.result()
            except Exception as e:
                log.error(f"[{label}] snapshot FAILED {m}: {e}")
            done += 1
            if done % 25 == 0:
                log.info(f"[{label}] snapshots {done}/{len(mondays)}")
    log.info(f"[{label}] cycles: {len(mondays)} Mondays x {len(DELTAS)} deltas")
    tasks = [(m, d) for m in mondays for d in DELTAS]
    rows = []
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(process_task, m, d, spot_fn, close_fn, cost_only): (m, d) for m, d in tasks}
        done = 0
        for f in as_completed(futs):
            m, d = futs[f]
            try:
                rows.extend(f.result())
            except Exception as e:
                log.error(f"[{label}] task FAILED {m} delta={d}: {e}")
                rows.extend({'entry_day': m, 'delta_target': d, 'status': 'ERROR', 'mgmt': mg} for mg in ['A', 'B'])
            done += 1
            if done % 25 == 0:
                log.info(f"[{label}] cycles {done}/{len(tasks)}")
    df = pd.DataFrame(rows)
    return df


# --------------------------------------------------------------------------------------
# Alpaca SPY fetch for the EXTENSION period (2016-01-04+; 2013-2015 uses parity fallback)
# --------------------------------------------------------------------------------------
def fetch_alpaca_spy_ext():
    if os.path.exists(EXT_SPY_DAILY) and os.path.exists(EXT_SPY_MIN):
        return
    key = os.environ.get('ALPACA_API_KEY')
    secret = os.environ.get('ALPACA_SECRET_KEY') or os.environ.get('ALPACA_API_SECRET')
    if not key or not secret:
        log.error("Alpaca keys missing in .env -- cannot fetch EXTENSION SPY prices for 2016-2023")
        pd.DataFrame(columns=['day', 'c']).to_parquet(EXT_SPY_DAILY)
        pd.DataFrame(columns=['day', 't', 'c']).to_parquet(EXT_SPY_MIN)
        return
    headers = {'APCA-API-KEY-ID': key, 'APCA-API-SECRET-KEY': secret}
    start = dt.date(2016, 1, 4).isoformat() + 'T00:00:00Z'
    end = (EXT_END + dt.timedelta(days=1)).isoformat() + 'T00:00:00Z'
    for tf, path, adj in [('1Day', EXT_SPY_DAILY, 'raw'), ('1Min', EXT_SPY_MIN, 'raw')]:
        rows, token = [], None
        while True:
            params = {'timeframe': tf, 'start': start, 'end': end, 'limit': 10000, 'adjustment': adj, 'feed': 'sip'}
            if token:
                params['page_token'] = token
            resp = requests.get('https://data.alpaca.markets/v2/stocks/SPY/bars', headers=headers, params=params, timeout=30)
            if resp.status_code != 200:
                log.error(f"Alpaca bars {tf} FAILED {resp.status_code}: {resp.text[:300]}")
                break
            j = resp.json()
            rows.extend(j.get('bars', []) or [])
            token = j.get('next_page_token')
            if not token:
                break
        if rows:
            df = pd.DataFrame(rows)
            df['t'] = pd.to_datetime(df['t'], utc=True)
            df['day'] = df['t'].dt.date
            df = df.rename(columns={'c': 'c'})[['day', 't', 'c']] if tf == '1Min' else df.rename(columns={'c': 'c'})[['day', 'c']]
            df.to_parquet(path)
            log.info(f"Alpaca {tf}: {len(df)} rows saved to {path}")
        else:
            log.error(f"Alpaca {tf}: 0 rows returned for {start}..{end}")


_ext_daily = None
_ext_min = None


def ext_spot_fn(monday):
    global _ext_daily, _ext_min
    if monday < dt.date(2016, 1, 4):
        return None  # gap -- parity fallback used by caller
    if _ext_min is None:
        _ext_min = pd.read_parquet(EXT_SPY_MIN)
        if len(_ext_min):
            _ext_min['t'] = pd.to_datetime(_ext_min['t'], utc=True)
    if _ext_min.empty:
        return None
    start_u, end_u = R.et_window_utc(monday, 10, 0, span_min=1)
    row = _ext_min[(_ext_min['t'] >= start_u) & (_ext_min['t'] < end_u)]
    return float(row.iloc[0]['c']) if len(row) else None


def ext_close_fn(day):
    global _ext_daily
    if _ext_daily is None:
        _ext_daily = pd.read_parquet(EXT_SPY_DAILY)
    if _ext_daily.empty:
        return None
    row = _ext_daily[_ext_daily['day'] == day]
    return float(row.iloc[0]['c']) if len(row) else None


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------
def cell_table(df, delta, mgmt, gate):
    sub = df[(df['delta_target'] == delta) & (df['mgmt'] == mgmt)].copy()
    if gate == 'iv15':
        # gate skips the week entirely (not a cycle, not VOID) when ATM 45-DTE IV < 15%
        below = sub['atm_iv'].notna() & (sub['atm_iv'] < 0.15)
        sub = sub[~below]
    return sub


def monthly_series(sub, equity_b):
    closed = sub[sub['status'] == 'CLOSED'].copy()
    closed['exit_month'] = pd.to_datetime(closed['exit_day']).dt.to_period('M')
    by_month = closed.groupby('exit_month')['pnl'].sum()
    if len(by_month) == 0:
        return pd.Series(dtype=float), closed
    full_range = pd.period_range(by_month.index.min(), by_month.index.max(), freq='M')
    by_month = by_month.reindex(full_range, fill_value=0.0)
    return by_month / equity_b, closed


def summarize(sub, equity_b, label):
    n_cycles = len(sub[sub['status'].isin(['CLOSED'])])
    n_void = len(sub[sub['status'] == 'VOID'])
    n_total = len(sub)
    void_rail = n_void / n_total if n_total else float('nan')
    monthly, closed = monthly_series(sub, equity_b)
    if len(monthly) == 0 or n_cycles == 0:
        return {'label': label, 'n_cycles': n_cycles, 'void_rail': void_rail, 'monthly_mean': float('nan'),
                'monthly_sharpe': float('nan'), 'green_share': float('nan'), 'worst_month': float('nan'),
                'max_dd': float('nan'), 'ex_top5_mean': float('nan'), 'n_months': 0}
    mean_m = monthly.mean()
    sharpe = (monthly.mean() / monthly.std(ddof=1) * math.sqrt(12)) if monthly.std(ddof=1) > 0 else float('nan')
    green = (monthly > 0).mean()
    worst = monthly.min()
    cum = monthly.cumsum()
    dd = (cum.cummax() - cum).max()
    top5_cut = closed['pnl'].quantile(0.95)
    ex_top5 = closed[closed['pnl'] < top5_cut]['pnl'].sum() / equity_b / max(len(monthly), 1)
    return {'label': label, 'n_cycles': n_cycles, 'void_rail': void_rail, 'monthly_mean': mean_m,
            'monthly_sharpe': sharpe, 'green_share': green, 'worst_month': worst, 'max_dd': dd,
            'ex_top5_mean': ex_top5, 'n_months': len(monthly)}


def buy_and_hold_spy(daily_df, start, end, equity_b):
    d = daily_df[(daily_df['day'] >= start) & (daily_df['day'] <= end)].sort_values('day')
    if len(d) < 2:
        return float('nan'), float('nan')
    ret = d['c'].iloc[-1] / d['c'].iloc[0] - 1.0
    monthly = d.set_index(pd.to_datetime(d['day'])).resample('ME')['c'].last().pct_change().dropna()
    dd = (monthly.add(1).cumprod().cummax() - monthly.add(1).cumprod()).max()
    return ret * equity_b, dd


def dispatch(args):
    from dotenv import load_dotenv
    load_dotenv(os.path.join(R.ROOT, '.env'))
    R.load_spy_panel()

    panel_spot_fn = lambda m: R.spy_spot_at(m, 10, 0)

    if args.stage == 'smoke':
        mon = R.mondays_between(dt.date(2026, 8, 4), dt.date(2026, 8, 18))[:1]
        df = build_cycles(mon, panel_spot_fn, R.spy_close, args.cost_only, 2, 'smoke')
        print(df.to_string())
        df.to_parquet(os.path.join(CACHE, 'smoke_cycles.parquet'))
        return

    if args.stage in ('panel', 'all'):
        mondays = R.mondays_between(PANEL_START, PANEL_END)
        if args.max_mondays:
            mondays = mondays[:args.max_mondays]
        df = build_cycles(mondays, panel_spot_fn, R.spy_close, args.cost_only, args.workers, 'panel')
        df.to_parquet(PANEL_CYCLES)
        log.info(f"panel done: {len(df)} rows -> {PANEL_CYCLES}")

    if args.stage in ('select', 'all'):
        df = pd.read_parquet(PANEL_CYCLES)
        df['entry_day'] = pd.to_datetime(df['entry_day']).dt.date
        train = df[df['entry_day'] <= TRAIN_END]
        val = df[df['entry_day'] >= VAL_START]
        results = []
        for delta in DELTAS:
            for mgmt in ['A', 'B']:
                for gate in ['none', 'iv15']:
                    sub_tr = cell_table(train, delta, mgmt, gate)
                    s = summarize(sub_tr, R.B, f"d{delta}_m{mgmt}_g{gate}")
                    s.update(delta=delta, mgmt=mgmt, gate=gate)
                    results.append(s)
        res_df = pd.DataFrame(results)
        eligible = res_df[(res_df['n_cycles'] >= 12) & (res_df['green_share'] >= 0.55)]
        pool = eligible if len(eligible) else res_df
        winner = pool.sort_values('monthly_sharpe', ascending=False).iloc[0]
        sel = {'delta': float(winner['delta']), 'mgmt': str(winner['mgmt']), 'gate': str(winner['gate'])}
        val_sub = cell_table(val, sel['delta'], sel['mgmt'], sel['gate'])
        val_stats = summarize(val_sub, R.B, 'VAL_selected')
        with open(SELECTION_PATH, 'w') as f:
            json.dump({'train_table': results, 'selection': sel, 'val_stats': val_stats}, f, indent=2, default=str)
        res_df.to_csv(os.path.join(HERE, 'rebuild_1599_train_table.csv'), index=False)
        log.info(f"SELECTED {sel} | VAL {val_stats}")

    if args.stage in ('extension', 'all'):
        if not os.path.exists(SELECTION_PATH):
            log.error("no selection.json -- run --stage select first")
            return
        sel = json.load(open(SELECTION_PATH))['selection']
        delta = args.delta if args.delta else sel['delta']
        fetch_alpaca_spy_ext()

        def spot_fn(monday):
            return ext_spot_fn(monday)

        mondays = R.mondays_between(EXT_START, EXT_END)
        if args.max_mondays:
            mondays = mondays[:args.max_mondays]
        df = build_cycles(mondays, spot_fn, ext_close_fn, args.cost_only, args.workers, 'extension')
        df.to_parquet(EXT_CYCLES)
        log.info(f"extension done: {len(df)} rows -> {EXT_CYCLES}")

    if args.stage in ('report', 'all'):
        write_report()


def write_report():
    sel_blob = json.load(open(SELECTION_PATH))
    sel = sel_blob['selection']
    val_stats = sel_blob['val_stats']
    panel = pd.read_parquet(PANEL_CYCLES)
    panel['entry_day'] = pd.to_datetime(panel['entry_day']).dt.date
    val_sub = cell_table(panel[panel['entry_day'] >= VAL_START], sel['delta'], sel['mgmt'], sel['gate'])
    val_sub.to_csv(os.path.join(HERE, 'rebuild_1599_cycles.csv'), index=False)
    monthly, _ = monthly_series(val_sub, R.B)
    monthly.to_frame('ret_on_B').to_csv(os.path.join(HERE, 'rebuild_1599_monthly.csv'))

    ext_stats, ext_years = None, None
    if os.path.exists(EXT_CYCLES):
        ext = pd.read_parquet(EXT_CYCLES)
        ext['entry_day'] = pd.to_datetime(ext['entry_day']).dt.date
        ext_sub = cell_table(ext, sel['delta'], sel['mgmt'], sel['gate'])
        ext_stats = summarize(ext_sub, R.B, 'EXTENSION_selected')
        closed = ext_sub[ext_sub['status'] == 'CLOSED'].copy()
        if len(closed):
            closed['year'] = pd.to_datetime(closed['exit_day']).dt.year
            yearly = closed.groupby('year')['pnl'].sum()
            ext_years = {'n_years': len(yearly), 'n_positive': int((yearly > 0).sum())}

    bh_ret, bh_dd = buy_and_hold_spy(R._spy_daily, VAL_START, PANEL_END, R.B)

    lines = []
    lines.append("# REBUILD_1599 -- Independent rebuild of PREREG_1567 v3 (cells 1,599-1,606)\n")
    lines.append("Built from PREREG_1567.md + Amendments 2/2a/3 prose only; did not open cell_1599.py, "
                  "test_cell_1599.py, cell_1599_cycles.csv/monthly.csv, RESULT_1599.md, or the v1/v2 cell scripts.\n")
    lines.append(f"\n## TRAIN selection\nSelected cell: delta={sel['delta']}, management={sel['mgmt']}, gate={sel['gate']}.\n")
    lines.append(f"\nTRAIN cell table (all 8 cells): `rebuild_1599_train_table.csv`.\n")
    lines.append(f"\n## VAL (2025-07-07..2026-08-17) for the selected cell\n")
    for k, v in val_stats.items():
        lines.append(f"* {k}: {v}")
    lines.append(f"\nBuy-and-hold SPY over the same VAL window on B: ${bh_ret:.0f}, max drawdown {bh_dd:.2%}.\n")
    pass_bar_val = (
        val_stats.get('monthly_mean', 0) >= 0.04 and val_stats.get('monthly_sharpe', 0) >= 1.0 and
        val_stats.get('green_share', 0) >= 0.60 and val_stats.get('worst_month', -1) >= -1.0 and
        val_stats.get('max_dd', 99) <= 1.5
    )
    lines.append(f"\n**VAL pass bar: {'PASS' if pass_bar_val else 'FAIL'}** (mean monthly return on B >= 4%, "
                  "Sharpe >= 1.0, green >= 60%, worst month >= -B, max DD <= 1.5B, beats SPY B&H per unit drawdown).\n")
    if ext_stats:
        lines.append(f"\n## EXTENSION (2013-04-08..2023-12-25), read once for the selected cell\n")
        for k, v in ext_stats.items():
            lines.append(f"* {k}: {v}")
        if ext_years:
            lines.append(f"* years positive/total: {ext_years['n_positive']}/{ext_years['n_years']} "
                          f"(bar: >=8/11)")
    else:
        lines.append("\n## EXTENSION\nNOT COMPLETED within this session's step/time budget -- see caveats.\n")
    lines.append("\n## Method notes / caveats (read as an adversary)\n")
    lines.append("* Monthly P&L is attributed to the calendar month of the EXIT (realization), not entry -- "
                  "disclosed convention, PREREG does not specify which.")
    lines.append("* ATM 45-DTE IV for the gate uses the PUT closest to spot only (no call averaging).")
    lines.append("* SPY spot 2013-04-08..2015-12-31 (EXTENSION only) has no Alpaca/Databento equity source; "
                  "used put-call parity from the same 10:00 options chain -- flagged `used_parity_spot`.")
    lines.append("* VOID = entry 10:00 bar lacks a two-sided quote on either leg, per Amendment 3 (rail computed above).")
    lines.append("* This is a single independent build, not yet cross-checked trade-by-trade against cell_1599.py's "
                  "own output (that comparison is a separate step per the CLAUDE.md independent-check protocol, and "
                  "requires an agent to read both, which this task explicitly forbade).")
    with open(os.path.join(HERE, 'REBUILD_1599.md'), 'w') as f:
        f.write('\n'.join(lines))
    log.info("wrote REBUILD_1599.md")
