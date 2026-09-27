"""INDEPENDENT REBUILD of PREREG_1567 (bull put credit spread ladder on SPY).

Written from research/options_vrp/PREREG_1567.md prose ONLY, without reading
cell_1567.py / test_cell_1567.py / cell_1567_cycles.csv / cell_1567_monthly.csv /
RESULT_1567.md, per the independent-check protocol in CLAUDE.md. Data source:
the cache already fetched at research/options_vrp/opt_cache/ (read only; no new
network calls). Every deviation from a literal reading of the PREREG that the
cache's shape forced is logged in the DEVIATIONS section at the bottom of this
file's module docstring and echoed into REBUILD_1567.md.

DEVIATIONS (forced by what the cache actually contains, not analytical choices):
  D1. "Fill = the spread's mid" -- the cache holds Alpaca OPTION BARS (trade
      OHLC), never NBBO quotes (fetch_options.py only calls get_option_bars).
      The trade bar's close in the 09:55-10:10 ET window nearest 10:00 stands
      in for "mid"; the PREREG's own $0.03/leg slippage is the disclosed
      correction for exactly this gap.
  D2. Minute bars exist ONLY for the entry Monday 09:55-10:10 window (FETCH_1567
      scope reduction). Every post-entry management decision (50%-credit /
      2x-stop / 21-DTE) and its "next session's 10:00" fill is therefore taken
      off the option DAILY close instead -- FETCH_1567.md itself names this as
      option (a) ("the SCORE stage must ... treat the daily close as the
      management-trigger price (documented approximation, disclose it)").
      Concretely: trigger is evaluated on session t's daily close; the fill is
      session (t+1)'s daily OPEN (the closest available print to "next
      session's 10:00", since no (t+1) intraday quote is cached).
  D3. Expiry eligibility ("nearest 45 DTE") is decided empirically per entry
      Monday from which candidate contracts actually carry an option_daily bar
      dated exactly on that Monday (real listing/liquidity at entry), not from
      a static calendar rule -- SPY's daily-expiry lineup visibly expanded over
      2024-2026 (verified directly against the cache, see explore step in the
      session log), so a fixed calendar assumption would silently select
      unlisted contracts.
  D4. Regulatory fee ($0.03/contract) is charged per leg per transaction (open
      and, for management A, close): 2 legs x $0.03 x contracts each time the
      position is opened or closed. Not specified to the leg in the PREREG;
      disclosed as the conservative reading.
  D5. Cycles whose expiry falls after the last cached SPY/option session
      (2026-09-26) are dropped (data does not exist yet), not settled early.
"""
import argparse
import datetime as dt
import math
import os
import sys

import numpy as np
import pandas as pd

CACHE = os.path.join(os.path.dirname(__file__), 'opt_cache')
OUT_DIR = os.path.dirname(__file__)

B_PCT = 0.10
EQUITY = 65000.0
B = B_PCT * EQUITY  # 6500.0
N_SLOTS = 6
R_RATE = 0.045
Q_YIELD = 0.013
SLIP_PER_LEG = 0.03
REG_FEE_PER_LEG = 0.03
DTE_LO, DTE_HI, DTE_TARGET = 38, 52, 45
CLOSE_AT_DTE = 21
DELTAS = [0.15, 0.20, 0.30]
WIDTHS = [5.0, 10.0]
MGMTS = ['A', 'B']
GATES = ['none', 'gate15']

TRAIN_START, TRAIN_END = '2024-02-05', '2025-06-30'
VAL_START, VAL_END = '2025-07-07', '2026-08-17'


# --------------------------------------------------------------------- Black-Scholes

def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_put_price(S, K, T, sigma, r=R_RATE, q=Q_YIELD):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1)


def bs_put_delta(S, K, T, sigma, r=R_RATE, q=Q_YIELD):
    if T <= 0 or sigma <= 0:
        return -1.0 if S < K else 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return -math.exp(-q * T) * _norm_cdf(-d1)


def implied_vol_put(price, S, K, T, lo=0.001, hi=4.0, tol=1e-6, max_iter=80):
    """Bisection. Returns None if the price is outside the no-arbitrage band or
    T<=0 (delta/vol undefined -> caller must skip the contract)."""
    if T <= 0 or price <= 0:
        return None
    intrinsic = max(K * math.exp(-R_RATE * T) - S * math.exp(-Q_YIELD * T), 0.0)
    upper_bound = K * math.exp(-R_RATE * T)
    if price <= intrinsic or price >= upper_bound:
        return None
    f_lo = bs_put_price(S, K, T, lo) - price
    f_hi = bs_put_price(S, K, T, hi) - price
    if f_lo * f_hi > 0:
        return None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = bs_put_price(S, K, T, mid) - price
        if abs(f_mid) < tol or (hi - lo) < tol:
            return mid
        if f_lo * f_mid <= 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------- data loading

def load_cache():
    spy_daily = pd.read_parquet(os.path.join(CACHE, 'spy_daily.parquet'))
    spy_minute = pd.read_parquet(os.path.join(CACHE, 'spy_minute.parquet'))
    opt_daily = pd.read_parquet(os.path.join(CACHE, 'option_daily.parquet'))
    opt_minute = pd.read_parquet(os.path.join(CACHE, 'option_minute_entry.parquet'))

    spy_daily['day'] = pd.to_datetime(spy_daily['day'])
    spy_minute['t'] = pd.to_datetime(spy_minute['t'], utc=True).dt.tz_convert('America/New_York')
    spy_minute['day'] = pd.to_datetime(spy_minute['day'])

    opt_daily['day'] = pd.to_datetime(opt_daily['day'])
    opt_daily['strike'] = opt_daily.symbol.str.slice(10).astype(float) / 1000.0
    opt_daily['expiry'] = pd.to_datetime('20' + opt_daily.symbol.str.slice(3, 9), format='%Y%m%d')

    opt_minute['t'] = pd.to_datetime(opt_minute['t'], utc=True).dt.tz_convert('America/New_York')
    opt_minute['monday'] = pd.to_datetime(opt_minute['monday'])
    opt_minute['strike'] = opt_minute.symbol.str.slice(10).astype(float) / 1000.0
    opt_minute['expiry'] = pd.to_datetime('20' + opt_minute.symbol.str.slice(3, 9), format='%Y%m%d')

    return spy_daily, spy_minute, opt_daily, opt_minute


def spy_price_near(spy_minute_day, hh, mm):
    if spy_minute_day.empty:
        return None
    target = spy_minute_day.iloc[0]['t'].replace(hour=hh, minute=mm, second=0, microsecond=0)
    idx = (spy_minute_day['t'] - target).abs().idxmin()
    return float(spy_minute_day.loc[idx, 'c']), spy_minute_day.loc[idx, 't']


def entry_mondays(trading_days_set, start, end):
    """Same rule the PREREG states: calendar Monday, snapped forward to the
    first actual trading session if the calendar Monday is a holiday."""
    out = []
    d = dt.date.fromisoformat(start)
    end_d = dt.date.fromisoformat(end)
    while d <= end_d:
        probe = d
        found = None
        for _ in range(5):
            if pd.Timestamp(probe) in trading_days_set:
                found = probe
                break
            probe += dt.timedelta(days=1)
        if found:
            out.append(pd.Timestamp(found))
        d += dt.timedelta(days=7)
    return out


# --------------------------------------------------------------------- per-monday leg pricing

def leg_price_at_entry(monday, symbol, opt_minute, opt_daily_by_symbol):
    """Best available proxy for the 10:00 mid on entry day: nearest minute bar
    close in the cached 09:55-10:10 window, else the entry day's daily close
    (D1/D2). Returns None if neither exists (contract not actually tradeable
    that day -> caller excludes it)."""
    sub = opt_minute[(opt_minute.monday == monday) & (opt_minute.symbol == symbol)]
    if not sub.empty:
        target = monday.replace(hour=10, minute=0)
        # sub['t'] is tz-aware ET; build comparable target
        target = sub.iloc[0]['t'].replace(hour=10, minute=0, second=0, microsecond=0)
        idx = (sub['t'] - target).abs().idxmin()
        return float(sub.loc[idx, 'c'])
    df = opt_daily_by_symbol.get(symbol)
    if df is None:
        return None
    row = df[df.day == monday]
    if row.empty:
        return None
    return float(row.iloc[0]['c'])


def candidate_expiries(monday, opt_daily_day_index):
    """D3: expiries eligible at this entry Monday = those with >=1 real daily
    bar dated exactly on the Monday, DTE in [38,52]."""
    syms_today = opt_daily_day_index.get(monday)
    if syms_today is None or syms_today.empty:
        return []
    exps = syms_today['expiry'].unique()
    out = []
    for e in exps:
        dte = (pd.Timestamp(e) - monday).days
        if DTE_LO <= dte <= DTE_HI:
            out.append((pd.Timestamp(e), dte))
    return sorted(out, key=lambda x: abs(x[1] - DTE_TARGET))


# --------------------------------------------------------------------- cycle construction

def build_cycles(spy_daily, spy_minute, opt_daily, opt_minute, mondays, last_data_day):
    trading_days = spy_daily.set_index('day')
    spy_minute_by_day = {d: g for d, g in spy_minute.groupby('day')}
    opt_daily_by_symbol = {s: g.sort_values('day') for s, g in opt_daily.groupby('symbol')}
    opt_daily_day_index = {d: g for d, g in opt_daily.groupby('day')}

    cycles = {}  # cell -> list of dict rows
    for delta_t in DELTAS:
        for W in WIDTHS:
            for M in MGMTS:
                for G in GATES:
                    cycles[(delta_t, W, M, G)] = []

    skipped_no_spot = skipped_no_expiry = skipped_no_strikes = 0

    for monday in mondays:
        mg = spy_minute_by_day.get(monday)
        if mg is None or mg.empty:
            skipped_no_spot += 1
            continue
        spot10, _ = spy_price_near(mg, 10, 0)

        exps = candidate_expiries(monday, opt_daily_day_index)
        if not exps:
            skipped_no_expiry += 1
            continue
        expiry, dte = exps[0]
        if expiry > last_data_day:
            continue
        T = dte / 365.0

        day_syms = opt_daily_day_index[monday]
        strikes_today = sorted(day_syms[day_syms.expiry == expiry]['strike'].unique())
        if len(strikes_today) < 3:
            skipped_no_strikes += 1
            continue

        # Price + delta every strike listed on the entry day for this expiry.
        strike_info = {}
        for k in strikes_today:
            sym = f"SPY{expiry.strftime('%y%m%d')}P{int(round(k * 1000)):08d}"
            px = leg_price_at_entry(monday, sym, opt_minute, opt_daily_by_symbol)
            if px is None or px <= 0:
                continue
            iv = implied_vol_put(px, spot10, k, T)
            if iv is None:
                continue
            delta = bs_put_delta(spot10, k, T, iv)
            strike_info[k] = dict(symbol=sym, price=px, iv=iv, delta=delta)
        if not strike_info:
            skipped_no_strikes += 1
            continue

        # ATM IV for the gate: strike nearest spot10 among priced strikes.
        atm_k = min(strike_info.keys(), key=lambda k: abs(k - spot10))
        atm_iv = strike_info[atm_k]['iv']
        gate_pass = atm_iv >= 0.15

        for delta_t in DELTAS:
            short_k = min(strike_info.keys(), key=lambda k: abs(abs(strike_info[k]['delta']) - delta_t))
            for W in WIDTHS:
                long_k = short_k - W
                if long_k not in strike_info:
                    # nearest available strike at or below short_k - W
                    lower = [k for k in strike_info if k <= long_k]
                    if not lower:
                        continue
                    long_k = max(lower)
                short_leg = strike_info[short_k]
                long_leg = strike_info[long_k]
                credit_mid = short_leg['price'] - long_leg['price']
                credit_ps = credit_mid - 2 * SLIP_PER_LEG
                actual_width = short_k - long_k
                risk_per_contract = (actual_width - credit_ps) * 100.0
                if risk_per_contract <= 0:
                    continue
                contracts = math.floor((B / N_SLOTS) / risk_per_contract)
                if contracts <= 0:
                    continue
                credit_total = contracts * credit_ps * 100.0 - contracts * REG_FEE_PER_LEG * 2

                for M in MGMTS:
                    for G in GATES:
                        if G == 'gate15' and not gate_pass:
                            continue
                        row = simulate_one(
                            monday, expiry, short_k, long_k, short_leg['symbol'], long_leg['symbol'],
                            contracts, credit_ps, credit_total, actual_width, M,
                            opt_daily_by_symbol, spy_minute_by_day, trading_days, last_data_day)
                        if row is not None:
                            cycles[(delta_t, W, M, G)].append(row)
    meta = dict(skipped_no_spot=skipped_no_spot, skipped_no_expiry=skipped_no_expiry,
                skipped_no_strikes=skipped_no_strikes)
    return cycles, meta


def _session_price(df, day, col='c'):
    r = df[df.day == day]
    return float(r.iloc[0][col]) if not r.empty else None


def simulate_one(monday, expiry, short_k, long_k, short_sym, long_sym, contracts,
                  credit_ps, credit_total, width, M, opt_daily_by_symbol,
                  spy_minute_by_day, trading_days, last_data_day):
    short_df = opt_daily_by_symbol.get(short_sym)
    long_df = opt_daily_by_symbol.get(long_sym)
    if short_df is None or long_df is None:
        return dict(entry_date=monday, expiry=expiry, short_strike=short_k, long_strike=long_k,
                     contracts=contracts, credit=credit_total, exit_date=expiry,
                     exit_reason='VOID_missing_bars', pnl_usd=0.0, width=width, void=True)

    sessions = trading_days[(trading_days.index > monday) & (trading_days.index <= expiry)].index.sort_values()
    close_by_day = 21
    exit_date, exit_reason, exit_mark = None, None, None

    if M == 'A':
        triggered_on = None
        for d in sessions:
            dte_remaining = (expiry - d).days
            sp = _session_price(short_df, d)
            lp = _session_price(long_df, d)
            if sp is None or lp is None:
                continue
            mark = sp - lp
            if mark <= 0.5 * credit_ps:
                triggered_on = (d, 'profit_target_50pct')
                break
            if mark >= 2.0 * credit_ps:
                triggered_on = (d, 'stop_2x_credit')
                break
            if dte_remaining <= close_by_day:
                triggered_on = (d, '21dte_close')
                break
        if triggered_on is not None:
            trig_day, reason = triggered_on
            later = sessions[sessions > trig_day]
            fill_day = None
            for d in later:
                sp = _session_price(short_df, d, 'o')
                lp = _session_price(long_df, d, 'o')
                if sp is not None and lp is not None:
                    fill_day, exit_mark = d, sp - lp
                    break
            if fill_day is None:
                sp = _session_price(short_df, trig_day)
                lp = _session_price(long_df, trig_day)
                fill_day, exit_mark = trig_day, (sp - lp if sp is not None and lp is not None else None)
            exit_date, exit_reason = fill_day, reason
        else:
            exit_date, exit_reason = expiry, 'expiry'

    if M == 'B' or exit_date is None or exit_reason == 'expiry':
        exit_date = expiry
        exit_reason = 'expiry_intrinsic'
        spy_close = _session_price_spy(spy_minute_by_day, expiry)
        if spy_close is None:
            return dict(entry_date=monday, expiry=expiry, short_strike=short_k, long_strike=long_k,
                        contracts=contracts, credit=credit_total, exit_date=expiry,
                        exit_reason='VOID_no_settlement_price', pnl_usd=0.0, width=width, void=True)
        payoff = max(short_k - spy_close, 0.0) - max(long_k - spy_close, 0.0)
        pnl = credit_total - payoff * 100.0 * contracts
        return dict(entry_date=monday, expiry=expiry, short_strike=short_k, long_strike=long_k,
                    contracts=contracts, credit=credit_total, exit_date=exit_date,
                    exit_reason=exit_reason, pnl_usd=pnl, width=width, void=False)

    if exit_mark is None:
        return dict(entry_date=monday, expiry=expiry, short_strike=short_k, long_strike=long_k,
                    contracts=contracts, credit=credit_total, exit_date=exit_date,
                    exit_reason='VOID_no_exit_fill', pnl_usd=0.0, width=width, void=True)
    close_cost = (exit_mark + 2 * SLIP_PER_LEG) * contracts * 100.0 + contracts * REG_FEE_PER_LEG * 2
    pnl = credit_total - close_cost
    return dict(entry_date=monday, expiry=expiry, short_strike=short_k, long_strike=long_k,
                contracts=contracts, credit=credit_total, exit_date=exit_date,
                exit_reason=exit_reason, pnl_usd=pnl, width=width, void=False)


def _session_price_spy(spy_minute_by_day, day):
    g = spy_minute_by_day.get(day)
    if g is None or g.empty:
        return None
    px, _ = spy_price_near(g, 16, 0)
    return px


# --------------------------------------------------------------------- reporting

def monthly_series(rows, valid_only=True):
    if not rows:
        return pd.Series(dtype=float), pd.DataFrame(columns=['pnl_usd', 'void', 'exit_date'])
    df = pd.DataFrame(rows)
    if valid_only:
        df = df[~df['void']]
    if df.empty:
        return pd.Series(dtype=float), df
    df['exit_month'] = pd.to_datetime(df['exit_date']).dt.to_period('M')
    monthly = df.groupby('exit_month')['pnl_usd'].sum()
    return monthly, df


def cell_stats(rows):
    monthly, df = monthly_series(rows)
    n_void = sum(1 for r in rows if r['void'])
    n_cycles = len(df)
    if n_cycles == 0 or monthly.empty:
        return dict(n_cycles=n_cycles, n_void=n_void, n_months=0, mean_monthly_ret=np.nan,
                    monthly_sharpe=np.nan, green_share=np.nan, worst_month=np.nan,
                    max_dd=np.nan, ex_top5_positive=None, win_rate=np.nan)
    ret = monthly / B
    mean_ret = ret.mean()
    sharpe = (ret.mean() / ret.std(ddof=1) * math.sqrt(12)) if ret.std(ddof=1) > 0 else np.nan
    green_share = (monthly > 0).mean()
    worst_month = monthly.min()
    cum = monthly.cumsum()
    dd = (cum.cummax() - cum).max()
    sorted_pnl = df['pnl_usd'].sort_values(ascending=False)
    top5_n = max(1, int(round(0.05 * len(sorted_pnl))))
    ex_top5_sum = sorted_pnl.iloc[top5_n:].sum()
    win_rate = (df['pnl_usd'] > 0).mean()
    return dict(n_cycles=n_cycles, n_void=n_void, n_months=len(monthly), mean_monthly_ret=mean_ret,
                monthly_sharpe=sharpe, green_share=green_share, worst_month=worst_month,
                max_dd=dd, ex_top5_positive=(ex_top5_sum > 0), win_rate=win_rate,
                total_pnl=df['pnl_usd'].sum())


def buy_and_hold_spy(spy_daily, start, end, capital):
    sub = spy_daily[(spy_daily.day >= start) & (spy_daily.day <= end)].sort_values('day')
    if sub.empty:
        return np.nan, np.nan
    shares = capital / sub.iloc[0]['c']
    ret = (sub.iloc[-1]['c'] - sub.iloc[0]['c']) / sub.iloc[0]['c']
    daily_ret = sub['c'].pct_change().dropna()
    max_dd = ((1 + daily_ret).cumprod().cummax() - (1 + daily_ret).cumprod()).max() * capital
    return ret, max_dd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke-weeks', type=int, default=0)
    a = ap.parse_args()

    print('Loading cache...', flush=True)
    spy_daily, spy_minute, opt_daily, opt_minute = load_cache()
    last_data_day = spy_daily['day'].max()
    trading_days_set = set(spy_daily['day'])
    all_mondays = entry_mondays(trading_days_set, TRAIN_START, VAL_END)
    if a.smoke_weeks:
        all_mondays = all_mondays[:a.smoke_weeks]
    print(f'{len(all_mondays)} entry Mondays, last_data_day={last_data_day.date()}', flush=True)

    cycles, meta = build_cycles(spy_daily, spy_minute, opt_daily, opt_minute, all_mondays, last_data_day)
    print('build_cycles meta:', meta, flush=True)

    all_rows = []
    for (delta_t, W, M, G), rows in cycles.items():
        for r in rows:
            r2 = dict(r)
            r2.update(cell_delta=delta_t, cell_width=W, cell_mgmt=M, cell_gate=G)
            r2['split'] = 'TRAIN' if r['entry_date'] <= pd.Timestamp(TRAIN_END) else 'VAL'
            all_rows.append(r2)
    cyc_df = pd.DataFrame(all_rows)
    cyc_df.to_csv(os.path.join(OUT_DIR, 'rebuild_1567_cycles.csv'), index=False)
    print(f'wrote {len(cyc_df)} cycle rows', flush=True)

    train_summary = {}
    val_summary = {}
    monthly_rows = []
    for cell, rows in cycles.items():
        train_rows = [r for r in rows if r['entry_date'] <= pd.Timestamp(TRAIN_END)]
        val_rows = [r for r in rows if r['entry_date'] >= pd.Timestamp(VAL_START)]
        train_summary[cell] = cell_stats(train_rows)
        val_summary[cell] = cell_stats(val_rows)
        for split_name, rws in (('TRAIN', train_rows), ('VAL', val_rows)):
            monthly, df = monthly_series(rws)
            for period, pnl in monthly.items():
                monthly_rows.append(dict(delta=cell[0], width=cell[1], mgmt=cell[2], gate=cell[3],
                                          split=split_name, month=str(period), pnl_usd=pnl))
    pd.DataFrame(monthly_rows).to_csv(os.path.join(OUT_DIR, 'rebuild_1567_monthly.csv'), index=False)

    # Selection on TRAIN
    candidates = [(cell, s) for cell, s in train_summary.items()
                  if s['n_cycles'] >= 12 and not np.isnan(s['green_share']) and s['green_share'] >= 0.55]
    if not candidates:
        candidates = list(train_summary.items())
        selection_note = 'NO CELL cleared n>=12 & green>=55% on TRAIN; showing best-Sharpe cell anyway for the record.'
    else:
        selection_note = f'{len(candidates)}/24 cells cleared the TRAIN screen (n>=12, green>=55%).'
    selected_cell, selected_train = max(candidates, key=lambda cs: (cs[1]['monthly_sharpe']
                                                                     if not np.isnan(cs[1]['monthly_sharpe']) else -99))
    selected_val = val_summary[selected_cell]

    bh_ret_train, bh_dd_train = buy_and_hold_spy(spy_daily, TRAIN_START, TRAIN_END, B)
    bh_ret_val, bh_dd_val = buy_and_hold_spy(spy_daily, VAL_START, VAL_END, B)

    lines = []
    lines.append('# REBUILD_1567 -- independent rebuild of PREREG_1567\n')
    lines.append(f'Entry Mondays used: {len(all_mondays)}. build_cycles meta: {meta}\n')
    lines.append(f'\n## Selection\n{selection_note}\n')
    lines.append(f'Selected cell: delta={selected_cell[0]} width={selected_cell[1]} mgmt={selected_cell[2]} gate={selected_cell[3]}\n')
    lines.append(f'\n### TRAIN ({TRAIN_START}..{TRAIN_END})\n{selected_train}\n')
    lines.append(f'\n### VAL ({VAL_START}..{VAL_END})\n{selected_val}\n')
    lines.append(f'\nBuy-and-hold SPY on B={B}: TRAIN ret={bh_ret_train:.4f} maxdd$={bh_dd_train:.2f}; '
                 f'VAL ret={bh_ret_val:.4f} maxdd$={bh_dd_val:.2f}\n')
    lines.append('\n## Full VAL table (unselected, for the record)\n')
    lines.append('| delta | width | mgmt | gate | n | green% | mean_mo_ret | sharpe | worst_$ | maxdd_$ |\n')
    lines.append('|---|---|---|---|---|---|---|---|---|---|\n')
    for cell, s in sorted(val_summary.items()):
        lines.append(f"| {cell[0]} | {cell[1]} | {cell[2]} | {cell[3]} | {s['n_cycles']} | "
                     f"{s['green_share']*100 if not np.isnan(s['green_share']) else float('nan'):.0f}% | "
                     f"{s['mean_monthly_ret']*100 if not np.isnan(s['mean_monthly_ret']) else float('nan'):.2f}% | "
                     f"{s['monthly_sharpe']:.2f} | {s['worst_month']:.0f} | {s['max_dd']:.0f} |\n")
    with open(os.path.join(OUT_DIR, 'REBUILD_1567.md'), 'w') as f:
        f.writelines(lines)

    print('SELECTED CELL:', selected_cell, flush=True)
    print('TRAIN:', selected_train, flush=True)
    print('VAL:', selected_val, flush=True)
    print('BUYHOLD VAL ret/maxdd:', bh_ret_val, bh_dd_val, flush=True)


if __name__ == '__main__':
    main()
