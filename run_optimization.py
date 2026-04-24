"""
Optimization matrix: find the best filter combination starting from the $274K baseline.
Tests risk scaling (not binary skip), wider entry windows, and volume-based sizing.
"""
import pandas as pd, numpy as np, os, logging, time
from datetime import date, timedelta
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv()
for n in ['trading.pattern_detector','backtest','trading.trade_planner',
          'trading.indicators','persistence.database']:
    logging.getLogger(n).setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('opt')

from backtest import BacktestRunner
from trading.market_regime import MarketRegimeFilter
from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from batch_backtest import fetch_daily_bars_cached, find_big_movers, _get_previous_trading_date

db = Database('data/onemil.db')
client = AlpacaClient(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_API_SECRET'))
start, end = date(2025, 1, 1), date(2026, 3, 21)

# Load data once
logger.info("Loading data...")
universe = db.get_active_universe()
symbols = [s['symbol'] for s in universe]
daily_bars = fetch_daily_bars_cached(symbols, start - timedelta(days=7), end, client, db)
movers = find_big_movers(daily_bars, universe_dict={s['symbol']: s for s in universe},
    price_min=2.0, price_max=30.0, float_max=10_000_000, start_date=start, end_date=end)

mbd = defaultdict(list)
for m in movers:
    mbd[m[1]].append((m[0], m[1], m[2] if len(m) > 2 else 0.0))
fd = sorted(mbd.keys())

sd = [(s, d.isoformat()) for td in fd for s, d, _ in mbd[td]]
ab = {k: pd.DataFrame(v) for k, v in db.get_intraday_bars_bulk(sd).items()}
psd = set()
for td in fd:
    p = _get_previous_trading_date(td)
    if p:
        for s, d, _ in mbd[td]:
            psd.add((s, p.isoformat()))
apb = {k: pd.DataFrame(v) for k, v in db.get_intraday_bars_bulk(list(psd)).items()}

# Base regime (vol+trend only)
regime = MarketRegimeFilter(enabled=True, vol_threshold=1.5, sma_period=50,
    min_spy_volume_ratio=0.70, thin_liquidity_breakout_vol_ratio=2.0)
spy_raw = fetch_daily_bars_cached(['SPY'], start - timedelta(days=90), end, client, db)
regime.load_spy_bars(spy_raw.get('SPY', []))

# SPY indicators for risk scaling
cursor = db.conn.execute(
    'SELECT bar_date, open, high, low, close, volume FROM daily_bars WHERE symbol="SPY" ORDER BY bar_date')
spy = pd.DataFrame([dict(r) for r in cursor.fetchall()])
spy['bar_date'] = pd.to_datetime(spy['bar_date'])
spy = spy.set_index('bar_date').sort_index()
spy['sma20'] = spy['close'].rolling(20).mean()
spy['prev_close'] = spy['close'].shift(1)
spy['uv'] = spy['volume'].where(spy['close'] > spy['prev_close'], 0)
spy['dv'] = spy['volume'].where(spy['close'] <= spy['prev_close'], 0)
spy['ud'] = spy['uv'].rolling(10).sum() / (spy['dv'].rolling(10).sum() + 1)
spy['p_above_sma20'] = (spy['close'] > spy['sma20']).shift(1)
spy['p_ud'] = spy['ud'].shift(1)

spy_info = {}
for d_raw, row in spy.iterrows():
    spy_info[d_raw.date()] = {
        'above_sma20': bool(row.get('p_above_sma20', False)),
        'ud': float(row.get('p_ud', 1.0)),
    }

logger.info(f"Loaded {len(ab)} bar sets. Running optimizations...")


def run_config(label, min_dist=0.0, last_entry=(15, 0),
               risk_scale_fn=None):
    """Run a backtest with optional risk scaling function.

    risk_scale_fn(trade_date, entry_hour, stop_dist, spy_data) -> float multiplier
    Returns 0 to skip, 0.5 for half size, 1.0 normal, 1.5 boost.
    Applied as post-hoc P&L scaling on the result.
    """
    runner = BacktestRunner(min_stop_distance=min_dist, last_entry_time_et=last_entry)
    results = []
    for td in fd:
        if not regime.is_regime_ok(td):
            continue
        runner._min_breakout_vol_override = (
            regime.get_min_breakout_volume_ratio(td, default=0)
            if regime.is_thin_liquidity(td) else 0)
        for sym, d, pc in mbd[td]:
            ds = d.isoformat()
            bars = ab.get((sym, ds))
            if bars is None or bars.empty:
                continue
            pdb = None
            p = _get_previous_trading_date(td)
            if p:
                pdb = apb.get((sym, p.isoformat()))
            try:
                r = runner.run(sym, bars, ds,
                               prev_close=pc if pc > 0 else None,
                               prev_day_bars=pdb)
                results.append(r)
            except Exception:
                pass

    rows = []
    for r in results:
        for t in r.trades_simulated:
            # Get entry hour (rough UTC-5 for EST, UTC-4 for EDT)
            eh = t.entry_time.hour - 5 if t.entry_time else 9  # rough
            if t.entry_time and t.entry_time.month >= 3 and t.entry_time.month <= 11:
                eh = t.entry_time.hour - 4  # EDT
            sd_val = t.entry_price - t.stop_loss
            td_date = date.fromisoformat(r.trade_date)
            spy_d = spy_info.get(td_date, {'above_sma20': True, 'ud': 1.0})

            if risk_scale_fn:
                scale = risk_scale_fn(td_date, eh, sd_val, spy_d)
                if scale == 0:
                    continue
                pnl = t.pnl * scale
            else:
                pnl = t.pnl

            rows.append({'date': r.trade_date, 'pnl': pnl})

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return None
    w = df[df['pnl'] > 0]
    l = df[df['pnl'] <= 0]
    mx = df.groupby(pd.to_datetime(df['date']).dt.to_period('M'))['pnl'].sum()
    sh = mx.mean() / mx.std() * np.sqrt(12) if mx.std() > 0 else 0
    cx = df.sort_values('date')['pnl'].cumsum()
    dd = (cx - cx.cummax()).min()
    pf = w['pnl'].sum() / abs(l['pnl'].sum()) if l['pnl'].sum() != 0 else 0
    return {
        'label': label, 'n': len(df),
        'wr': len(w) / len(df) * 100,
        'pnl': df['pnl'].sum(), 'pf': pf, 'sh': sh,
        'dd': dd, 'lm': (mx < 0).sum(),
    }


# ============================================================
# CONFIGS TO TEST
# ============================================================

configs = []

# 1. Raw baseline (vol+trend, no extra filters)
configs.append(('1. BASELINE (vol+trend only)', {}))

# 2. Current prod
configs.append(('2. CURRENT PROD (slope+euph+dist0.12+11AM)',
                {'min_dist': 0.12, 'last_entry': (11, 0)}))

# 3. Just min_dist 0.09 (no time restriction beyond default 15:00)
configs.append(('3. dist 0.09 only', {'min_dist': 0.09}))

# 4. dist 0.09 + last_entry 13:00 (wider window)
configs.append(('4. dist 0.09 + 13:00 cutoff',
                {'min_dist': 0.09, 'last_entry': (13, 0)}))

# 5. dist 0.09 + last_entry 12:00
configs.append(('5. dist 0.09 + 12:00 cutoff',
                {'min_dist': 0.09, 'last_entry': (12, 0)}))

# 6. dist 0.09 + RISK SCALE: half size when SPY below SMA20
def scale_sma20_half(td, eh, sd, spy_d):
    if not spy_d['above_sma20']:
        return 0.5
    return 1.0
configs.append(('6. dist 0.09 + half_size below SMA20',
                {'min_dist': 0.09, 'risk_scale_fn': scale_sma20_half}))

# 7. dist 0.09 + RISK SCALE: half size when UD > 1.2 (euphoria)
def scale_ud_half(td, eh, sd, spy_d):
    if spy_d['ud'] > 1.2:
        return 0.5
    return 1.0
configs.append(('7. dist 0.09 + half_size UD>1.2',
                {'min_dist': 0.09, 'risk_scale_fn': scale_ud_half}))

# 8. dist 0.09 + COMBINED: half below SMA20, half UD>1.2
def scale_combined(td, eh, sd, spy_d):
    mult = 1.0
    if not spy_d['above_sma20']:
        mult *= 0.5
    if spy_d['ud'] > 1.2:
        mult *= 0.5
    return mult
configs.append(('8. dist 0.09 + half SMA20 + half UD>1.2',
                {'min_dist': 0.09, 'risk_scale_fn': scale_combined}))

# 9. dist 0.09 + 12:00 + half below SMA20
def scale_sma20_12(td, eh, sd, spy_d):
    if not spy_d['above_sma20']:
        return 0.5
    return 1.0
configs.append(('9. dist 0.09 + 12:00 + half below SMA20',
                {'min_dist': 0.09, 'last_entry': (12, 0),
                 'risk_scale_fn': scale_sma20_12}))

# 10. dist 0.09 + skip tiny stops + half below SMA20
def scale_sma20_skip_tiny(td, eh, sd, spy_d):
    if sd < 0.09:
        return 0  # already handled by min_dist, but belt+suspenders
    mult = 1.0
    if not spy_d['above_sma20']:
        mult *= 0.5
    return mult
configs.append(('10. dist 0.09 + half below SMA20 (no time cut)',
                {'min_dist': 0.09, 'risk_scale_fn': scale_sma20_skip_tiny}))

# 11. dist 0.09 + 12:00 + half SMA20 + half UD>1.2
def scale_all(td, eh, sd, spy_d):
    mult = 1.0
    if not spy_d['above_sma20']:
        mult *= 0.5
    if spy_d['ud'] > 1.2:
        mult *= 0.5
    return mult
configs.append(('11. dist 0.09 + 12:00 + half SMA20 + half UD',
                {'min_dist': 0.09, 'last_entry': (12, 0),
                 'risk_scale_fn': scale_all}))

# 12. JUST half size below SMA20 (no dist, no time cut)
def scale_sma20_only(td, eh, sd, spy_d):
    return 0.5 if not spy_d['above_sma20'] else 1.0
configs.append(('12. half_size below SMA20 (no other filters)',
                {'risk_scale_fn': scale_sma20_only}))

# 13. dist 0.09 + 11:00 + half SMA20 (current-ish but with scaling not skip)
configs.append(('13. dist 0.09 + 11:00 + half SMA20',
                {'min_dist': 0.09, 'last_entry': (11, 0),
                 'risk_scale_fn': scale_sma20_half}))

# Run all
results = []
for label, kw in configs:
    logger.info(f"  {label}")
    r = run_config(label, **kw)
    if r:
        results.append(r)

# Sort by P&L * Sharpe (balance both)
results.sort(key=lambda x: x['pnl'] * x['sh'], reverse=True)

print()
print(f"{'#':<3} {'Filter':<52} {'N':>4} {'WR':>6} {'P&L':>11} {'PF':>5} {'Sh':>5} {'DD':>10} {'LM':>3}")
print('=' * 100)
for r in results:
    print(f"    {r['label']:<52} {r['n']:>4} {r['wr']:>5.1f}% ${r['pnl']:>+9,.0f} {r['pf']:>4.2f} {r['sh']:>4.2f} ${r['dd']:>+8,.0f} {r['lm']:>3}")
