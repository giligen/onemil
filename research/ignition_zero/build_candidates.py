#!/usr/bin/env python3
"""Ignition-from-zero — the ONE candidate table (DESIGN.md step 1).

Every symbol-day 2025-01-02..2026-09-11 with intraday high/open >= 1.05 and
open >= $1 (Databento EQUS daily, delisted included). For each cross level
(5/7/10/15% above the 9:30 RTH open) the FIRST 1-min bar in 9:35..11:30 ET
whose high crosses the level is the trigger; entry = next bar open x 1.003,
stop = min(prior-30-min low, entry*0.99) — byte-identical to
trading.ignition_rules.trigger_entry_stop on the +10% level inside 9:35..10:30
(the parity subset), and every BT gate (chase guard, pre-bars, R_MIN) is
recorded as a FLAG, never applied, so nothing is excluded before the study.
Features use bars <= trigger bar and daily bars < the day only. Exits are
re-simulated on the same bars: V0 lock (BT), hold-to-close, partial@1R/BE,
structure trail, time-60.

Resumable per day (state file); symbol-days without bars are written to
coverage_missing.csv (second pass after the Databento fetch: --retry-missing).
Second pass (pass2.py) adds cohort / theme / sympathy / news / short-interest
features that need every symbol's triggers of the day.
"""
import json, os, sqlite3, sys, time
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; sys.path.insert(0, ROOT); os.chdir(ROOT)
import trading.ignition_rules as R
D = 'research/ignition_zero'
LEVELS = (5, 7, 10, 15)
WIN_START, WIN_END = 575, 690          # 9:35 .. 11:30 ET (window itself is H18)
RETRY = '--retry-missing' in sys.argv
STATE = f'{D}/build_state.json'
OUT = f'{D}/candidates.csv'
MISS = f'{D}/coverage_missing.csv'

cache = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=120)
topup = sqlite3.connect(f'file:{ROOT}/research/ignition_capcheck/topup.db?mode=ro', uri=True, timeout=120)
pit = sqlite3.connect(f'file:{ROOT}/data/research/databento/pit_bars_1min.db?mode=ro', uri=True, timeout=120)
pit_tab = [t[0] for t in pit.execute("select name from sqlite_master where type='table'")][0]
pit_cols = [x[1] for x in pit.execute(f'PRAGMA table_info({pit_tab})')]
pit_day = [c for c in pit_cols if c in ('day', 'bar_date', 'date')][0]

daily = pd.read_parquet(f'{ROOT}/data/research/databento/equs_daily_2025_2026.parquet')
daily['bar_date'] = daily['bar_date'].astype(str).str[:10]
daily = daily[daily.symbol.notna() & (daily.symbol.astype(str).str.strip() != '')]
daily = daily.sort_values(['symbol', 'bar_date']).reset_index(drop=True)
# prior-day features (shifted -> strictly before the day)
g = daily.groupby('symbol')
daily['prev_close'] = g['close'].shift(1); daily['prev_high'] = g['high'].shift(1); daily['prev_low'] = g['low'].shift(1)
daily['prev_volume'] = g['volume'].shift(1)
daily['high20'] = g['high'].transform(lambda s: s.shift(1).rolling(20, min_periods=5).max())
daily['adv20'] = g['volume'].transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
daily['hist_n'] = g.cumcount()
daily['ratio'] = daily['high'] / daily['open']
cand = daily[(daily.ratio >= 1.05) & (daily.open >= 1.0) & (daily.bar_date >= '2025-01-02') & (daily.bar_date <= '2026-09-11')]
spy = pd.read_sql("select bar_date, timestamp, close from intraday_bars_1min where symbol='SPY' and bar_date>='2025-01-02'", cache)
spy['m'] = pd.to_datetime(spy.timestamp, utc=True).dt.tz_convert('America/New_York').pipe(lambda t: t.dt.hour * 60 + t.dt.minute)
spy_by_day = {d: dict(zip(gg.m, gg.close)) for d, gg in spy.groupby('bar_date')}
spyd = daily[daily.symbol == 'SPY'].set_index('bar_date')
spyd['range3'] = ((spyd.high - spyd.low) / spyd.close * 100).rolling(3).mean().shift(1)


def load_bars(day, syms):
    out = {}
    q = ("select symbol, timestamp as t, open, high, low, close, volume from intraday_bars_1min "
         f"where bar_date=? and symbol in ({','.join('?' * len(syms))})")
    for s, gg in pd.read_sql(q, cache, params=[day] + syms).groupby('symbol'): out[s] = gg
    left = [s for s in syms if s not in out]
    if left:
        t = pd.read_sql("select symbol, t, o as open, h as high, l as low, c as close, v as volume from bars where day=?", topup, params=[day])
        for s, gg in t[t.symbol.isin(left)].groupby('symbol'): out[s] = gg
        left = [s for s in syms if s not in out]
    if left:
        q2 = (f"select symbol, {'t' if 't' in pit_cols else 'timestamp'} as t, "
              f"{'o' if 'o' in pit_cols else 'open'} as open, {'h' if 'h' in pit_cols else 'high'} as high, "
              f"{'l' if 'l' in pit_cols else 'low'} as low, {'c' if 'c' in pit_cols else 'close'} as close, "
              f"{'v' if 'v' in pit_cols else 'volume'} as volume from {pit_tab} where {pit_day}=?")
        try:
            t = pd.read_sql(q2, pit, params=[day])
            for s, gg in t[t.symbol.isin(left)].groupby('symbol'): out[s] = gg
        except Exception as e:
            print('pit read failed', e, flush=True)
    res = {}
    for s, gg in out.items():
        ts = pd.to_datetime(gg['t'], utc=True).dt.tz_convert('America/New_York')
        gg = gg.assign(m=ts.dt.hour * 60 + ts.dt.minute).sort_values('m').reset_index(drop=True)
        res[s] = gg
    return res


def walk(post, entry, stop, entry_min, *, mode):
    """rr on the full position. mode: v0 (BT lock), hold, p1be, strail, t60."""
    Rd = entry - stop; cur = stop; armed = False; taken = False; rr_part = 0.0; frac = 1.0
    last_low = None; prev_low = None; hl_stop = None
    for r in post.itertuples():
        if r.m >= R.EOD_FLAT_MIN: return rr_part + frac * (r.open - entry) / Rd, 'eod'
        if r.low <= cur:
            fill = min(cur, r.open); return rr_part + frac * (fill * 0.999 - entry) / Rd, ('lock' if armed else ('be' if taken else 'stop'))
        if mode == 'p1be' and not taken and r.high >= entry + Rd:
            taken = True; rr_part = 0.5; frac = 0.5; cur = max(cur, entry)
        if mode == 'strail':
            if not taken and r.high >= entry + Rd: taken = True; cur = max(cur, entry)
            if taken:
                if prev_low is not None and last_low is not None and r.low > last_low and last_low > prev_low:
                    hl_stop = last_low
                if hl_stop is not None: cur = max(cur, hl_stop)
            prev_low, last_low = last_low, r.low
        if mode == 'v0' and not armed and r.high >= entry + R.ARM_R * Rd:
            armed = True; cur = entry + R.LOCK_R * Rd
        if mode == 't60' and r.m >= entry_min + 60:
            return rr_part + frac * (r.close - entry) / Rd, 'hold'
    if len(post): return rr_part + frac * (post.iloc[-1].close - entry) / Rd, 'eod'
    return 0.0, 'none'


def build_day(day, sub):
    syms = sub.symbol.tolist(); B = load_bars(day, syms); rows = []; missing = []
    spym = spy_by_day.get(day, {})
    for r in sub.itertuples():
        gg = B.get(r.symbol)
        if gg is None or len(gg) < 20: missing.append((r.symbol, day)); continue
        rth = gg[(gg.m >= 570) & (gg.m < 960)].reset_index(drop=True)
        if len(rth) < 20: missing.append((r.symbol, day)); continue
        o = float(rth.iloc[0].open)
        pm = gg[gg.m < 570]; pm_dollar = float((pm.volume * pm.close).sum()) if len(pm) else np.nan
        first5 = rth[(rth.high >= o * 1.05) & (rth.m >= WIN_START)]
        m5 = int(first5.iloc[0].m) if len(first5) else None
        for L in LEVELS:
            lvl = o * (1 + L / 100.0)
            trig = rth[(rth.high >= lvl) & (rth.m >= WIN_START) & (rth.m <= WIN_END)]
            if trig.empty: continue
            ti = trig.index[0]; tb = rth.loc[ti]; nxt = rth[rth.index > ti]
            if nxt.empty: continue
            nb = nxt.iloc[0]; entry = float(nb.open) * R.ENTRY_SLIP
            pre = rth[(rth.m >= tb.m - 30) & (rth.m < tb.m)]
            stop = R.stop_from_pre_lows(float(pre.low.min()) if len(pre) else entry * 0.99, entry)
            rp = R.r_pct_from_stop(entry, stop)
            upto = rth[rth.index <= ti]
            row = dict(day=day, symbol=r.symbol, level=L, trig_m=int(tb.m), min_from_open=int(tb.m) - 570, min_to_5pct=(m5 - 570) if m5 is not None else np.nan,
                       day_open=o, level_px=lvl, entry=entry, stop=stop, r_pct=rp,
                       flag_chase=int(entry > lvl * R.CHASE_MAX_RATIO), flag_prebars=int(len(pre) < R.PRE_BARS_MIN), flag_rmin=int(rp < R.R_MIN_PCT), in_bt_window=int(tb.m <= R.TRIGGER_MIN_END),
                       tb_close_pos=float((tb.close - tb.low) / (tb.high - tb.low)) if tb.high > tb.low else np.nan,
                       tb_vol=float(tb.volume), tb_vol_x_prior=float(tb.volume / upto.iloc[:-1].volume.mean()) if len(upto) > 1 and upto.iloc[:-1].volume.mean() > 0 else np.nan,
                       bar_dollar=float(tb.volume) * entry, dollar_to_trig=float((upto.volume * upto.close).sum()), pm_dollar=pm_dollar,
                       prev_close=r.prev_close, open_gap_pct=(o - r.prev_close) / r.prev_close * 100 if r.prev_close and r.prev_close > 0 else np.nan,
                       prev_range_pct=(r.prev_high - r.prev_low) / r.prev_close * 100 if r.prev_close and r.prev_close > 0 else np.nan,
                       prev_volume=r.prev_volume, adv20=r.adv20, dist_20d_high_pct=(lvl - r.high20) / r.high20 * 100 if r.high20 and r.high20 > 0 else np.nan, hist_n=int(r.hist_n),
                       spy_5m_ret=((spym.get(int(tb.m), np.nan) / spym.get(int(tb.m) - 5, np.nan)) - 1) * 100 if spym.get(int(tb.m)) and spym.get(int(tb.m) - 5) else np.nan,
                       spy_range3=float(spyd.range3.get(day, np.nan)))
            post = rth[rth.index > ti]
            for mode in ('v0', 'hold', 'p1be', 'strail', 't60'):
                rr, why = walk(post, entry, stop, int(tb.m), mode=mode); row[f'rr_{mode}'] = rr; row[f'why_{mode}'] = why
            rows.append(row)
    return rows, missing


def main():
    state = json.load(open(STATE)) if os.path.exists(STATE) else {'done': []}
    done = set(state['done'])
    if RETRY:
        miss = pd.read_csv(MISS); target = cand.merge(miss.rename(columns={'bar_date': 'bar_date'}), on=['symbol', 'bar_date'])
        days = sorted(target.bar_date.unique()); print(f'RETRY pass over {len(target)} missing symbol-days / {len(days)} days', flush=True)
    else:
        target = cand; days = [d for d in sorted(cand.bar_date.unique()) if d not in done]
        print(f'{len(cand):,} candidate symbol-days; {len(days)} days to build', flush=True)
    t0 = time.time(); header = not os.path.exists(OUT)
    for i, day in enumerate(days):
        sub = target[target.bar_date == day]
        rows, missing = build_day(day, sub)
        if rows: pd.DataFrame(rows).to_csv(OUT, mode='a', header=header, index=False); header = False
        if missing and not RETRY: pd.DataFrame(missing, columns=['symbol', 'bar_date']).to_csv(MISS, mode='a', header=not os.path.exists(MISS), index=False)
        if not RETRY:
            state['done'].append(day); json.dump(state, open(STATE, 'w'))
        if i % 5 == 0: print(f'{i + 1}/{len(days)} {day}: {len(sub)} cands, {len(rows)} rows, {len(missing)} no-bars, {time.time() - t0:.0f}s', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
