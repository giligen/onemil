"""Quick comparison script: breakeven profit levels."""
import logging, sys, time, yaml, os
import numpy as np
from collections import defaultdict
from datetime import date, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# Suppress noisy loggers
for name in ['trading.pattern_detector', 'backtest', 'trading.trade_planner',
             'trading.indicators', 'persistence.database', 'batch_backtest']:
    logging.getLogger(name).setLevel(logging.WARNING)

from dotenv import load_dotenv
load_dotenv()

from config import Config
from persistence.database import get_database
from data_sources.alpaca_client import AlpacaClient
from trading.market_regime import MarketRegimeFilter
from batch_backtest import find_big_movers, fetch_daily_bars_cached, run_batch_backtest_fast

# Setup
db = get_database()
api_key = os.getenv("ALPACA_API_KEY")
api_secret = os.getenv("ALPACA_API_SECRET")
client = AlpacaClient(api_key, api_secret)

start_date = date(2025, 1, 1)
end_date = date(2026, 3, 25)

cfg = Config._load_yaml_only()
scanner_cfg = cfg.get("scanner", {})
trading_cfg = cfg.get("trading", {})
regime_cfg = trading_cfg.get("market_regime", {})

# Fetch daily bars + movers
logger.info("Loading daily bars and finding movers...")
sma_period = int(regime_cfg.get("sma_period", 50))
spy_lookback = int(sma_period * 1.5) + 14
daily_bars = fetch_daily_bars_cached(
    list({s['symbol'] for s in db.get_active_universe()}),
    start_date - timedelta(days=7), end_date, client, db
)
universe = {s['symbol']: s for s in db.get_active_universe()}
movers = find_big_movers(
    daily_bars, universe_dict=universe,
    price_min=float(scanner_cfg.get("price_min", 2.0)),
    price_max=float(scanner_cfg.get("price_max", 20.0)),
    float_max=int(scanner_cfg.get("float_max", 10_000_000)),
    start_date=start_date, end_date=end_date,
)
logger.info(f"Found {len(movers)} movers")

# Build regime
spy_bars_raw = fetch_daily_bars_cached(['SPY'], start_date - timedelta(days=spy_lookback), end_date, client, db)
regime = MarketRegimeFilter(
    enabled=bool(regime_cfg.get("enabled", False)),
    vol_threshold=float(regime_cfg.get("vol_threshold", 1.5)),
    sma_period=sma_period,
    max_trades_per_day=int(trading_cfg.get("max_trades_per_day", 0)),
    min_spy_volume_ratio=float(regime_cfg.get("min_spy_volume_ratio", 0.7)),
    thin_liquidity_breakout_vol_ratio=float(regime_cfg.get("thin_liquidity_breakout_vol_ratio", 2.0)),
)
regime.load_spy_bars(spy_bars_raw.get('SPY', []))

# Patch loader
original_load = Config._load_yaml_only

def make_patched_loader(be_r, profit_r):
    def patched_load():
        d = original_load()
        trail = d.setdefault('trading', {}).setdefault('trailing_stop', {})
        trail['breakeven_at_r'] = be_r
        trail['breakeven_profit_r'] = profit_r
        return d
    return patched_load

tests = [
    (0.0, 0.0, 'No BE (baseline)'),
    (1.5, 0.0, 'BE@1.5R +0.0R'),
    (1.5, 0.2, 'BE@1.5R +0.2R'),
    (1.5, 0.4, 'BE@1.5R +0.4R'),
    (1.5, 0.5, 'BE@1.5R +0.5R'),
    (1.5, 0.7, 'BE@1.5R +0.7R'),
    (1.5, 1.0, 'BE@1.5R +1.0R'),
]

print()
print(f"{'Config':25s} | {'P&L':>10s} | {'WR':>6s} | {'PF':>5s} | {'Sharpe':>6s} | {'MaxDD':>10s} | {'Trades':>6s} | {'AvgW':>7s} {'AvgL':>7s}")
print('-' * 110)

for be_r, profit_r, label in tests:
    Config._load_yaml_only = staticmethod(make_patched_loader(be_r, profit_r))
    
    t0 = time.time()
    results = run_batch_backtest_fast(movers, db, market_regime=regime)
    elapsed = time.time() - t0
    
    all_trades = []
    for r in results:
        all_trades.extend(r.trades_simulated)
    
    if not all_trades:
        print(f'{label:25s} | NO TRADES')
        continue
    
    total_pnl = sum(t.pnl for t in all_trades)
    wins = [t for t in all_trades if t.pnl > 0]
    losses = [t for t in all_trades if t.pnl <= 0]
    wr = len(wins) / len(all_trades) * 100
    
    equity = 0
    peak = 0
    max_dd = 0
    for t in sorted(all_trades, key=lambda x: x.entry_time):
        equity += t.pnl
        peak = max(peak, equity)
        dd = equity - peak
        max_dd = min(max_dd, dd)
    
    avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0
    pf = abs(sum(t.pnl for t in wins)) / abs(sum(t.pnl for t in losses)) if losses else float('inf')
    
    daily_pnl = defaultdict(float)
    for t in all_trades:
        d_str = str(t.entry_time)[:10]
        daily_pnl[d_str] += t.pnl
    daily_vals = list(daily_pnl.values())
    sharpe = (np.mean(daily_vals) / np.std(daily_vals)) * (252**0.5) if np.std(daily_vals) > 0 else 0
    
    print(f'{label:25s} | ${total_pnl:>9,.0f} | {wr:5.1f}% | {pf:5.2f} | {sharpe:6.2f} | ${max_dd:>9,.0f} | {len(all_trades):>6} | ${avg_win:>6,.0f} ${avg_loss:>6,.0f}')

Config._load_yaml_only = staticmethod(original_load)
print("\nDone.")
