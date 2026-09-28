#!/usr/bin/env python3
"""Cells 1,626-1,629 -- leveraged single-stock ETF decay pair.

PREREG: research/lev_decay/PREREG_1626.md (FROZEN 2026-09-28 18:30 UTC).
Idea 1 of research/IDEAS_20260928.md.

Mechanism: a daily-rebalanced L-times ETF loses ~= 1/2*(L^2-L)*sigma^2 per day
to the rebalancing/volatility drag. Short BOTH the 2x-long and 2x-short daily
ETF on the same single-stock underlying, dollar-balanced, to collect the drag
on both legs while staying approximately delta-neutral. Nothing here is a
directional bet on the underlying.

Cells:
  1,626  PAIR-SHORT static: $1 short long-leg + $1 short short-leg, rebalance
         to dollar-neutral every 5 sessions, hold indefinitely.
  1,627  PAIR-SHORT vol-gated: same engine, position ON only while the
         underlying's 20d realised vol (annualised) >= 60%, OFF below 40%.
  1,628  SINGLE-LEG hedged (report-only): short $1 of the 2x-long ETF, long
         $2 of the underlying, same 5-session rebalance.
  1,629  the 3x / 1.5x / 1.75x pairs (report-only), same engine as 1,626.

Usage:
    python3 research/lev_decay/cell_1626.py --match         # pair matching -> pairs.csv
    python3 research/lev_decay/cell_1626.py --fetch-flags   # shortable/ETB snapshot
    python3 research/lev_decay/cell_1626.py --fetch-bars    # resumable daily bars -> bars.parquet
    python3 research/lev_decay/cell_1626.py --score         # cells 1,626-1,629 -> RESULT_1626.md
    python3 research/lev_decay/cell_1626.py --all           # all four steps in order
"""
from __future__ import annotations

import os
import re
import sys
import time
import argparse
from datetime import datetime, timezone

REPO = '/home/ec2-user/onemil'
sys.path.insert(0, REPO)
sys.path.insert(0, f'{REPO}/research/hod_entry')
os.chdir(REPO)

import numpy as np
import pandas as pd

from cell_1445 import day_clustered_t  # noqa: E402  (shared BT/live helper, per CLAUDE.md)

OUT = f'{REPO}/research/lev_decay'
ASSET_LIST = f'{REPO}/data/research/databento/alpaca_assets_all_20260905.csv'
PARSED_CSV = f'{OUT}/leveraged_names_parsed.csv'
PAIRS_CSV = f'{OUT}/pairs.csv'
BARS_PARQUET = f'{OUT}/bars.parquet'
FLAGS_CSV = f'{OUT}/asset_flags.csv'
RESULT_MD = f'{OUT}/RESULT_1626.md'

START = '2022-01-03'
END = '2026-09-26'
TRAIN_END = '2025-03-31'    # TRAIN 2022-01..2025-03
VAL_START = '2025-04-01'    # VAL   2025-04..2026-09
BORROW_RAILS = {'5pct': 0.05, '15pct': 0.15, '30pct': 0.30}
COST_BPS_PER_LEG = 5.0
REBALANCE_EVERY = 5         # sessions
VOL_ON = 0.60                # 1,627 gate: enter >= 60% annualised
VOL_OFF = 0.40                # 1,627 gate: exit below 40% annualised
VOL_WINDOW = 20              # sessions, realised vol of the underlying

# ---------------------------------------------------------------------------
# 1. Pair matching
# ---------------------------------------------------------------------------

# Issuer / boilerplate tokens that are never the underlying ticker. Extended
# whenever the hand-check below finds a false positive (e.g. 'CORGI').
STOPWORDS = {
    'DAILY', 'LONG', 'BULL', 'SHORT', 'BEAR', 'INVERSE', 'TARGET', 'TRUST',
    'SHARES', 'FUND', 'ETF', 'II', 'III', 'MANAGERS', 'SERIES',
    'GRANITESHARES', 'DIREXION', 'THEMES', 'LEVERAGE', 'LEVERAGES', 'TIDAL',
    'DEFIANCE', 'TREX', 'TRADR', 'INVESTMENT', 'ISHARES', 'YIELDMAX', 'KURV',
    'AXS', 'REX', 'PROSHARES', 'STRATEGY', 'ULTRA', 'ULTRAPRO', 'PRO', 'THE',
    'AND', 'OF', 'A', 'DOWN', 'UP', 'DAY', 'NEW', 'GROUP', 'HOLDINGS',
    'CORP', 'CORPORATION', 'INC', 'CO', 'PLC', 'ADR', 'CLASS', 'COMMON',
    'STOCK', 'ORD', 'NOTES', 'LINKED', 'OPTION', 'OPTIONS', 'INCOME',
    'PREMIUM', 'WEEKLY', 'MONTHLY', 'ACCELERATED', 'AMPLIFY', 'SINGLE',
    'CORGI', 'REVERSE', 'HALF', 'DOUBLE', 'TRIPLE', 'CURRENCY', 'PLUS',
}

# Company-name -> ticker aliases for issuers (observed: T-Rex) that spell out
# the underlying's NAME rather than its ticker in the Alpaca `name` field.
NAME_ALIASES = {
    'NVIDIA': 'NVDA', 'TESLA': 'TSLA', 'AMAZON': 'AMZN', 'ALPHABET': 'GOOGL',
    'GOOGLE': 'GOOGL', 'FACEBOOK': 'META', 'APPLE': 'AAPL',
    'MICROSOFT': 'MSFT', 'BROADCOM': 'AVGO', 'ALIBABA': 'BABA',
    'COINBASE': 'COIN', 'MICROSTRATEGY': 'MSTR', 'PALANTIR': 'PLTR',
    'NETFLIX': 'NFLX', 'ROBINHOOD': 'HOOD', 'SUPERMICRO': 'SMCI',
    'MARATHON': 'MARA', 'ORACLE': 'ORCL', 'CARVANA': 'CVNA',
    'RIVIAN': 'RIVN', 'LUCID': 'LCID', 'SNOWFLAKE': 'SNOW',
    'SHOPIFY': 'SHOP', 'PAYPAL': 'PYPL', 'INTEL': 'INTC',
    'QUALCOMM': 'QCOM', 'DISNEY': 'DIS', 'BOEING': 'BA', 'GAMESTOP': 'GME',
    'ROBLOX': 'RBLX', 'DRAFTKINGS': 'DKNG', 'CROWDSTRIKE': 'CRWD',
    'MICRON': 'MU', 'APPLOVIN': 'APP', 'ROCKETLAB': 'RKLB',
    'ADVANCED': 'AMD', 'JPMORGAN': 'JPM', 'BERKSHIRE': 'BRK',
    'PINTEREST': 'PINS', 'TWILIO': 'TWLO', 'DOORDASH': 'DASH',
    'STARBUCKS': 'SBUX', 'NIKE': 'NKE', 'BOFA': 'BAC',
    'BANKOFAMERICA': 'BAC', 'WELLSFARGO': 'WFC', 'GOLDMANSACHS': 'GS',
    'MORGANSTANLEY': 'MS', 'VISA': 'V', 'MASTERCARD': 'MA',
    'EXXON': 'XOM', 'CHEVRON': 'CVX', 'ELILILLY': 'LLY', 'LILLY': 'LLY',
    'PFIZER': 'PFE', 'MODERNA': 'MRNA', 'NOVONORDISK': 'NVO',
    'COCACOLA': 'KO', 'PEPSICO': 'PEP', 'WALMART': 'WMT',
    'HOMEDEPOT': 'HD', 'MCDONALDS': 'MCD', 'CATERPILLAR': 'CAT',
    'DEERE': 'DE', 'HONEYWELL': 'HON', 'LOCKHEEDMARTIN': 'LMT',
    'AIRBNB': 'ABNB', 'BOOKING': 'BKNG', 'CHEWY': 'CHWY',
    'PELOTON': 'PTON', 'ZOOM': 'ZM', 'DOCUSIGN': 'DOCU',
    'CLOUDFLARE': 'NET', 'DATADOG': 'DDOG', 'MONGODB': 'MDB',
    'ZSCALER': 'ZS', 'AFFIRM': 'AFRM', 'UPSTART': 'UPST',
    'SEALIMITED': 'SE', 'MERCADOLIBRE': 'MELI', 'BAIDU': 'BIDU',
    'XPENG': 'XPEV', 'TAIWANSEMICONDUCTOR': 'TSM',
    'CLEANSPARK': 'CLSK', 'HUT8': 'HUT', 'CORESCIENTIFIC': 'CORZ',
    'APPLIEDMATERIALS': 'AMAT', 'LAMRESEARCH': 'LRCX', 'KLA': 'KLAC',
    'TEXASINSTRUMENTS': 'TXN', 'ONSEMICONDUCTOR': 'ON', 'WOLFSPEED': 'WOLF',
    'FIRSTSOLAR': 'FSLR', 'ENPHASE': 'ENPH', 'SOLAREDGE': 'SEDG',
    'VERIZON': 'VZ', 'TMOBILE': 'TMUS', 'COMCAST': 'CMCSA',
}

LEV_FACTOR_RE = re.compile(r'(\d+(?:\.\d+)?)\s*X\b')
LONG_WORDS = {'LONG', 'BULL', 'UP'}
SHORT_WORDS = {'SHORT', 'BEAR', 'INVERSE', 'DOWN', 'REVERSE'}
VALID_FACTORS = (1.5, 1.75, 2.0, 3.0)

# Extracted tokens that are NOT a single-stock underlying, found by the
# hand-check (PREREG "Not allowed: adding pairs on non-single-stock
# underlyings without a separate cell"): sector baskets (ENERGY, BANKS, GAS,
# HIGH beta, REAL estate, RETAIL, CLOUD), country/region indices (RUSSIA,
# JAPAN, LATIN America, FTSE, MSCI EM, DOW Jones Internet, CSI China), broad
# indices (MID cap, SMALL cap, INDEX, YEAR=20yr Treasury), commodities (GOLD,
# OIL, SILVER, COPPER, GAS), crypto (ETHER, BNB), a private company with no
# tradeable NBBO underlying (SPACEX), a curated-basket theme (DRAM, WORLD,
# MEMORY, SEVEN=Mag-7), and one CONFIRMED false pair where two DIFFERENT
# 'Top 5' sector baskets (Biotech vs Semiconductors) both extracted the
# shared word 'TOP'. Each token below is coincidentally either a real ticker
# for an unrelated company (GOLD=Barrick, MID=?, DOW=Dow Inc, HIGH=?,
# REAL=RealReal, YEAR=?, MSCI=MSCI Inc) or an unresolved theme word --
# confirmed by reading the raw asset name in leveraged_names_parsed.csv.
NON_SINGLE_STOCK_BLOCKLIST = {
    'ENERGY', 'GOLD', 'OIL', 'BANKS', 'DOW', 'FTSE', 'GAS', 'HIGH', 'INDEX',
    'MID', 'MSCI', 'REAL', 'SMALL', 'YEAR', 'CSI', 'RUSSIA', 'JAPAN',
    'LATIN', 'CLOUD', 'WORLD', 'MEMORY', 'RETAIL', 'SEVEN', 'SILVER',
    'ETHER', 'COPPER', 'XETFS', 'TUTTLE', 'ETFMG', 'SPACEX', 'BNB', 'TOP',
    'DRAM',
}


def parse_name(symbol: str, name: str, known_symbols: set[str]) -> dict | None:
    """Parse an Alpaca asset `name` into (underlying, factor, direction).

    Returns None if `name` does not look like a leveraged single-stock daily
    ETF. `underlying` resolution order: (1) an alias for a spelled-out company
    name [strong], (2) a token that is itself a symbol in the full Alpaca
    universe [strong], (3) the first remaining candidate token [weak -- flag
    for hand-check].
    """
    if not isinstance(name, str) or not name:
        return None
    up = name.upper()
    m = LEV_FACTOR_RE.search(up)
    if not m:
        return None
    factor = float(m.group(1))
    if factor not in VALID_FACTORS:
        return None
    tokens = re.findall(r'[A-Z]+', up)
    tokset = set(tokens)
    if 'DAILY' not in tokset:
        return None
    if tokset & LONG_WORDS:
        direction = 'long'
    elif tokset & SHORT_WORDS:
        direction = 'short'
    else:
        return None

    cands = [t for t in tokens if t not in STOPWORDS and t not in LONG_WORDS
              and t not in SHORT_WORDS and 1 < len(t) <= 6]
    alias_hits = [NAME_ALIASES[t] for t in cands if t in NAME_ALIASES]
    known_hits = [t for t in cands if t in known_symbols and t != symbol.upper()]
    if alias_hits:
        underlying, confidence = alias_hits[0], 'alias'
    elif known_hits:
        underlying, confidence = known_hits[0], 'known_symbol'
    elif cands:
        underlying, confidence = cands[0], 'weak_unconfirmed'
    else:
        return None
    return dict(symbol=symbol, name=name, factor=factor, direction=direction,
                underlying=underlying, match_confidence=confidence)


def build_pairs() -> pd.DataFrame:
    """Parse the asset list, match long/short legs per (underlying, factor)."""
    assets = pd.read_csv(ASSET_LIST, dtype=str)
    assets['name'] = assets['name'].fillna('')
    assets['tradable'] = assets['tradable'].astype(str).str.upper() == 'TRUE'
    known_symbols = set(assets['symbol'].str.upper())

    # one row per symbol: prefer the ACTIVE listing if one exists, else the
    # most recent inactive row (closed/delisted -- counted for survivorship).
    assets['is_active'] = assets['status'].astype(str).str.lower() == 'active'
    dedup = (assets.sort_values('is_active', ascending=False)
                    .drop_duplicates('symbol', keep='first'))

    parsed = []
    for r in dedup.itertuples(index=False):
        p = parse_name(r.symbol, r.name, known_symbols)
        if p:
            p['status'] = r.status
            p['tradable'] = r.tradable
            p['exchange'] = r.exchange
            parsed.append(p)
    lev = pd.DataFrame(parsed)
    lev.to_csv(PARSED_CSV, index=False)
    n_closed = int((lev['status'].astype(str).str.lower() != 'active').sum())
    print(f'[match] {len(lev)} rows parsed as leveraged single-stock daily ETFs '
          f'out of {len(dedup)} unique symbols ({len(assets)} raw rows); '
          f'{n_closed} are NOT status=active (closed/delisted -- survivorship note)',
          flush=True)
    conf_counts = lev['match_confidence'].value_counts().to_dict()
    print(f'[match] underlying-match confidence: {conf_counts}', flush=True)

    pairs = []
    skipped_alt = []
    for (underlying, factor), grp in lev.groupby(['underlying', 'factor']):
        longs = grp[grp.direction == 'long']
        shorts = grp[grp.direction == 'short']
        if longs.empty or shorts.empty:
            continue

        def pick(d):
            d = d.sort_values(['tradable', 'status', 'symbol'],
                               ascending=[False, True, True])
            return d.iloc[0], d.iloc[1:]

        L, Lrest = pick(longs)
        S, Srest = pick(shorts)
        for _, alt in pd.concat([Lrest, Srest]).iterrows():
            skipped_alt.append((underlying, factor, alt['symbol'], alt['direction']))
        pairs.append(dict(
            underlying=underlying, factor=factor,
            long_symbol=L.symbol, long_name=L.name, long_status=L.status,
            long_tradable=bool(L.tradable), long_confidence=L.match_confidence,
            short_symbol=S.symbol, short_name=S.name, short_status=S.status,
            short_tradable=bool(S.tradable), short_confidence=S.match_confidence,
        ))
    pairs_df = pd.DataFrame(pairs).sort_values(['factor', 'underlying']).reset_index(drop=True)
    pairs_df['pair_id'] = pairs_df['underlying'] + '_' + pairs_df['factor'].astype(str) + 'x'

    is_blocked = pairs_df['underlying'].isin(NON_SINGLE_STOCK_BLOCKLIST)
    blocked_df = pairs_df[is_blocked].copy()
    pairs_df = pairs_df[~is_blocked].reset_index(drop=True)
    if len(blocked_df):
        blocked_df.to_csv(f'{OUT}/pairs_excluded_non_single_stock.csv', index=False)
        print(f'[match] EXCLUDED {len(blocked_df)} non-single-stock pairs (sector/index/'
              f'country/commodity/crypto/private-co, or a confirmed false pair from a shared '
              f'brand word): {sorted(blocked_df.underlying.unique().tolist())} -> '
              f'pairs_excluded_non_single_stock.csv', flush=True)

    pairs_df.to_csv(PAIRS_CSV, index=False)
    with open(f'{OUT}/pairs_alt_legs_skipped.txt', 'w') as f:
        f.write('underlying,factor,symbol,direction  (alternate issuer legs NOT used in the pair)\n')
        for row in skipped_alt:
            f.write(','.join(str(x) for x in row) + '\n')
    n2 = int((pairs_df.factor == 2.0).sum())
    n3 = int((pairs_df.factor == 3.0).sum())
    n15 = int(pairs_df.factor.isin([1.5, 1.75]).sum())
    print(f'[match] {len(pairs_df)} pairs matched: {n2} at 2x, {n3} at 3x, '
          f'{n15} at 1.5x/1.75x; {len(skipped_alt)} alternate legs skipped '
          f'(multi-issuer underlyings) -> pairs_alt_legs_skipped.txt', flush=True)
    return pairs_df


# ---------------------------------------------------------------------------
# 2. Alpaca fetch: shortable/ETB flags + daily bars
# ---------------------------------------------------------------------------

def _alpaca_keys():
    from dotenv import load_dotenv
    load_dotenv(f'{REPO}/.env')
    key = os.getenv('ALPACA_API_KEY')
    secret = os.getenv('ALPACA_API_SECRET')
    if not key or not secret:
        print('[alpaca] ERROR: ALPACA_API_KEY/ALPACA_API_SECRET missing from .env -- abort',
              flush=True)
        sys.exit(1)
    return key, secret


def fetch_flags(symbols: list[str]) -> pd.DataFrame:
    """Today's shortable / easy_to_borrow snapshot from TradingClient.get_all_assets.
    Disclosed hindsight per PREREG: this is NOT the historical borrow state."""
    key, secret = _alpaca_keys()
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetAssetsRequest
    from alpaca.trading.enums import AssetClass

    cl = TradingClient(key, secret, paper=False)
    assets = cl.get_all_assets(GetAssetsRequest(asset_class=AssetClass.US_EQUITY))
    rows = [dict(symbol=a.symbol, status=str(a.status), tradable=bool(a.tradable),
                 shortable=bool(getattr(a, 'shortable', False)),
                 easy_to_borrow=bool(getattr(a, 'easy_to_borrow', False)),
                 exchange=str(a.exchange)) for a in assets]
    d = pd.DataFrame(rows)
    d.to_csv(FLAGS_CSV, index=False)
    want = set(symbols)
    sub = d[d.symbol.isin(want)]
    print(f'[fetch-flags] {len(d)} total active assets; {len(sub)}/{len(want)} '
          f'pair-member symbols found; shortable {sub.shortable.mean():.2f} '
          f'easy_to_borrow {sub.easy_to_borrow.mean():.2f}', flush=True)
    missing = want - set(d.symbol)
    if missing:
        print(f'[fetch-flags] WARNING: {len(missing)} symbols absent from the '
              f'ACTIVE asset endpoint (delisted) -> flags treated as NOT '
              f'shortable/ETB: {sorted(missing)[:15]}{"..." if len(missing) > 15 else ""}',
              flush=True)
    return d


def fetch_bars(symbols: list[str], force: bool = False) -> pd.DataFrame:
    """Resumable daily-bar pull, split-adjusted, batches of <=10 symbols.

    split-adjusted (not raw): these products carry frequent reverse splits as
    NAV decays toward zero; raw closes would show fabricated +/-90% jumps at
    each split that are not real trading P&L (CLAUDE.md price-scale check).
    """
    key, secret = _alpaca_keys()
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(key, secret)
    symbols = sorted(set(symbols))
    existing = pd.DataFrame()
    have = set()
    if os.path.exists(BARS_PARQUET) and not force:
        existing = pd.read_parquet(BARS_PARQUET)
        have = set(existing['symbol'].unique())
    todo = [s for s in symbols if s not in have]
    print(f'[fetch-bars] {len(symbols)} symbols total, {len(have)} already cached, '
          f'{len(todo)} to fetch', flush=True)

    start_dt = datetime.strptime(START, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(END, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    frames = [existing] if len(existing) else []
    lost = []
    BATCH = 10
    n_batches = (len(todo) + BATCH - 1) // BATCH
    t0 = time.time()
    for bi in range(n_batches):
        batch = todo[bi * BATCH:(bi + 1) * BATCH]
        try:
            req = StockBarsRequest(symbol_or_symbols=batch, timeframe=TimeFrame.Day,
                                    start=start_dt, end=end_dt, adjustment='split')
            bars = client.get_stock_bars(req)
            df = bars.df.reset_index() if bars.df.index.size else pd.DataFrame()
        except Exception as e:
            print(f'[fetch-bars] BATCH FAIL {batch}: {e}', flush=True)
            df = pd.DataFrame()
        got = set(df['symbol'].unique()) if len(df) else set()
        for s in batch:
            if s not in got:
                lost.append(s)
        if len(df):
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df['date'] = df['timestamp'].dt.strftime('%Y-%m-%d')
            frames.append(df[['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']])
        if (bi + 1) % 5 == 0 or bi == n_batches - 1:
            print(f'[fetch-bars] batch {bi + 1}/{n_batches}  lost so far={len(lost)}  '
                  f'elapsed={time.time() - t0:.0f}s', flush=True)
            if frames:
                pd.concat(frames, ignore_index=True).drop_duplicates(
                    ['symbol', 'date']).to_parquet(BARS_PARQUET, index=False)
        time.sleep(0.1)

    allbars = (pd.concat(frames, ignore_index=True).drop_duplicates(['symbol', 'date'])
               if frames else pd.DataFrame())
    allbars.to_parquet(BARS_PARQUET, index=False)
    with open(f'{OUT}/fetch_lost.txt', 'w') as f:
        f.write(f'LOST {len(lost)}/{len(symbols)} symbols (Alpaca returned zero daily bars '
                f'{START}..{END}):\n')
        f.write('\n'.join(lost) + '\n')
    got_syms = allbars.symbol.nunique() if len(allbars) else 0
    print(f'[fetch-bars] DONE rows={len(allbars):,} symbols={got_syms} '
          f'LOST={len(lost)}/{len(symbols)} ({len(lost)/max(len(symbols),1):.1%})', flush=True)
    return allbars


# ---------------------------------------------------------------------------
# 3. Simulation engine (shared by cells 1,626 / 1,627 / 1,629)
# ---------------------------------------------------------------------------

def _leg_returns(bars: pd.DataFrame, symbol: str) -> pd.Series:
    d = bars[bars.symbol == symbol].sort_values('date')
    if d.empty:
        return pd.Series(dtype=float)
    s = d.set_index('date')['close'].astype(float)
    return s


def simulate_pair(long_px: pd.Series, short_px: pd.Series, rebalance_every: int,
                   gate: pd.Series | None = None) -> pd.DataFrame:
    """Core pair-short engine: $1 short long-leg + $1 short short-leg, dollar-
    neutral rebalance every `rebalance_every` sessions. `gate` (optional,
    boolean, indexed like the merged dates) turns the position OFF/ON for
    cell 1,627; a gate transition forces close+reopen (cost both ways).

    Returns one row per available trading day with: r_long, r_short (leg
    returns), pnl_bps_gross (pre-cost/borrow, bps of the $2 gross basis),
    cost_bps, is_rebalance, notional_gap (delta drift, D_long - D_short),
    gross_notional_frac (D_long_prev + D_short_prev, for borrow costing).
    """
    idx = long_px.index.intersection(short_px.index)
    idx = idx.sort_values()
    if len(idx) < 2:
        return pd.DataFrame()
    lp = long_px.loc[idx].astype(float)
    sp = short_px.loc[idx].astype(float)
    r_long = lp.pct_change()
    r_short = sp.pct_change()
    dates = idx[1:]
    r_long = r_long.iloc[1:]
    r_short = r_short.iloc[1:]
    if gate is not None:
        gate = gate.reindex(dates).fillna(False).astype(bool)
    else:
        gate = pd.Series(True, index=dates)

    rows = []
    D_long, D_short = 0.0, 0.0   # 0 == flat/closed
    open_pos = False
    since_open = 0
    for dt, rl, rs, g in zip(dates, r_long.to_numpy(), r_short.to_numpy(), gate.to_numpy()):
        cost = 0.0
        rebal = False
        if not open_pos:
            if g and np.isfinite(rl) and np.isfinite(rs):
                # open: buy from flat to $1/$1 short each leg
                D_long, D_short = 1.0, 1.0
                cost = (1.0 + 1.0) * COST_BPS_PER_LEG / 1e4
                open_pos = True
                since_open = 0
                pnl_gross = 0.0
                gross_frac = 0.0    # flat before today -- no borrow base yet
                notional_gap = 0.0  # just opened at $1/$1, no drift yet
                rebal = True  # counts as the opening trade
            else:
                rows.append(dict(date=dt, r_long=rl, r_short=rs, pnl_bps_gross=0.0,
                                  cost_bps=0.0, is_rebalance=False,
                                  notional_gap=np.nan, gross_notional_frac=0.0,
                                  position_open=False))
                continue
        else:
            if not g:
                # gate turned off: close out at today's drifted notional
                D_long_prev, D_short_prev = D_long, D_short
                D_long_t = D_long_prev * (1 + rl) if np.isfinite(rl) else D_long_prev
                D_short_t = D_short_prev * (1 + rs) if np.isfinite(rs) else D_short_prev
                pnl_gross = (D_long_prev - D_long_t) + (D_short_prev - D_short_t)
                cost = (abs(D_long_t) + abs(D_short_t)) * COST_BPS_PER_LEG / 1e4  # closing trade
                gross_frac = D_long_prev + D_short_prev
                notional_gap = D_long_t - D_short_t
                D_long, D_short = 0.0, 0.0
                open_pos = False
                pnl_bps = pnl_gross / 2.0 * 1e4
                cost_bps = cost / 2.0 * 1e4
                rows.append(dict(date=dt, r_long=rl, r_short=rs, pnl_bps_gross=pnl_bps,
                                  cost_bps=cost_bps, is_rebalance=True,
                                  notional_gap=notional_gap,
                                  gross_notional_frac=gross_frac, position_open=True))
                continue
            since_open += 1
            D_long_prev, D_short_prev = D_long, D_short
            D_long = D_long_prev * (1 + rl) if np.isfinite(rl) else D_long_prev
            D_short = D_short_prev * (1 + rs) if np.isfinite(rs) else D_short_prev
            pnl_gross = (D_long_prev - D_long) + (D_short_prev - D_short)
            gross_frac = D_long_prev + D_short_prev
            notional_gap = D_long - D_short
            if since_open % rebalance_every == 0:
                cost = (abs(D_long - 1.0) + abs(D_short - 1.0)) * COST_BPS_PER_LEG / 1e4
                rebal = True
                D_long, D_short = 1.0, 1.0
        pnl_bps = pnl_gross / 2.0 * 1e4
        cost_bps = cost / 2.0 * 1e4
        rows.append(dict(date=dt, r_long=rl, r_short=rs, pnl_bps_gross=pnl_bps,
                          cost_bps=cost_bps, is_rebalance=rebal,
                          notional_gap=notional_gap, gross_notional_frac=gross_frac,
                          position_open=True))
    out = pd.DataFrame(rows)
    return out


def simulate_single_leg_hedge(long_px: pd.Series, und_px: pd.Series,
                               rebalance_every: int) -> pd.DataFrame:
    """Cell 1,628 (report-only): short $1 of the 2x-long ETF, long $2 of the
    underlying, rebalance to (1, 2) every `rebalance_every` sessions. Gross
    notional = $3; bps reported on that $3 basis."""
    idx = long_px.index.intersection(und_px.index).sort_values()
    if len(idx) < 2:
        return pd.DataFrame()
    lp = long_px.loc[idx].astype(float)
    up = und_px.loc[idx].astype(float)
    r_etf = lp.pct_change().iloc[1:]
    r_und = up.pct_change().iloc[1:]
    dates = r_etf.index
    rows = []
    D_etf, D_und = 1.0, 2.0
    since = 0
    for dt, re_, ru in zip(dates, r_etf.to_numpy(), r_und.to_numpy()):
        if not (np.isfinite(re_) and np.isfinite(ru)):
            rows.append(dict(date=dt, r_etf=re_, r_und=ru, pnl_bps_gross=0.0,
                              cost_bps=0.0, is_rebalance=False))
            continue
        since += 1
        D_etf_prev, D_und_prev = D_etf, D_und
        D_etf = D_etf_prev * (1 + re_)
        D_und = D_und_prev * (1 + ru)
        pnl_dollars = (D_etf_prev - D_etf) + (D_und - D_und_prev)  # short etf, long und
        cost = 0.0
        rebal = False
        if since % rebalance_every == 0:
            cost = (abs(D_etf - 1.0) + abs(D_und - 2.0)) * COST_BPS_PER_LEG / 1e4
            rebal = True
            D_etf, D_und = 1.0, 2.0
        rows.append(dict(date=dt, r_etf=re_, r_und=ru,
                          pnl_bps_gross=pnl_dollars / 3.0 * 1e4,
                          cost_bps=cost / 3.0 * 1e4, is_rebalance=rebal))
    return pd.DataFrame(rows)


def realised_vol(px: pd.Series, window: int = VOL_WINDOW) -> pd.Series:
    r = np.log(px.astype(float)).diff()
    return r.rolling(window).std() * np.sqrt(252)


# ---------------------------------------------------------------------------
# 4. Scoring / reporting
# ---------------------------------------------------------------------------

def split_of(date_str: str) -> str:
    if date_str <= TRAIN_END:
        return 'TRAIN'
    if date_str >= VAL_START:
        return 'VAL'
    return 'GAP'  # should not occur (TRAIN/VAL are contiguous by construction)


def net_bps(row_pnl: pd.Series, row_cost: pd.Series, row_grossfrac: pd.Series,
            rail: float) -> pd.Series:
    borrow_bps = row_grossfrac.fillna(0.0) / 2.0 * (rail / 252.0) * 1e4
    return row_pnl - row_cost - borrow_bps


def score() -> None:
    pairs = pd.read_csv(PAIRS_CSV)
    bars = pd.read_parquet(BARS_PARQUET)
    flags = pd.read_csv(FLAGS_CSV) if os.path.exists(FLAGS_CSV) else pd.DataFrame(
        columns=['symbol', 'shortable', 'easy_to_borrow'])
    # get_all_assets can list a symbol more than once (multiple exchange/class
    # records) -- OR the flags together so "known shortable via any listing".
    flag_map = flags.groupby('symbol')[['shortable', 'easy_to_borrow']].max().to_dict('index')

    def etb(sym):
        f = flag_map.get(sym, {})
        return bool(f.get('shortable', False)) and bool(f.get('easy_to_borrow', False))

    all_days_1626, all_days_1627, all_days_1628, all_days_1629 = [], [], [], []
    pair_first_date = {}
    for p in pairs.itertuples(index=False):
        lp = _leg_returns(bars, p.long_symbol)
        sp = _leg_returns(bars, p.short_symbol)
        up = _leg_returns(bars, p.underlying)
        if lp.empty or sp.empty:
            print(f'[score] SKIP {p.pair_id}: missing bars (long empty={lp.empty} '
                  f'short empty={sp.empty})', flush=True)
            continue
        common = lp.index.intersection(sp.index)
        if len(common) < 2:
            print(f'[score] SKIP {p.pair_id}: <2 common trading days', flush=True)
            continue
        pair_first_date[p.pair_id] = min(common)

        vol = realised_vol(up) if not up.empty else pd.Series(dtype=float)

        engine = simulate_pair(lp, sp, REBALANCE_EVERY)
        if not engine.empty:
            engine['pair_id'] = p.pair_id
            engine['underlying'] = p.underlying
            engine['factor'] = p.factor
            engine['long_symbol'] = p.long_symbol
            engine['short_symbol'] = p.short_symbol
            engine['split'] = engine['date'].apply(split_of)
            engine['vol_ann'] = engine['date'].map(vol.to_dict()) if len(vol) else np.nan
            engine['theory_drag_bps'] = (p.factor ** 2 - p.factor) / 2.0 * \
                (engine['vol_ann'] / np.sqrt(252)) ** 2 * 1e4
            engine['pair_etb_both'] = etb(p.long_symbol) and etb(p.short_symbol)
            for rname, rail in BORROW_RAILS.items():
                engine[f'net_bps_{rname}'] = net_bps(engine['pnl_bps_gross'],
                                                       engine['cost_bps'],
                                                       engine['gross_notional_frac'], rail)
            if p.factor == 2.0:
                all_days_1626.append(engine.copy())
            else:
                all_days_1629.append(engine.copy())

        # 1,627 vol-gated (2x pairs only, per PREREG's population)
        if p.factor == 2.0 and len(vol):
            state, gate_vals = False, []
            for v in vol.reindex(common).to_numpy():
                if not np.isfinite(v):
                    gate_vals.append(state)
                    continue
                if not state and v >= VOL_ON:
                    state = True
                elif state and v < VOL_OFF:
                    state = False
                gate_vals.append(state)
            gate = pd.Series(gate_vals, index=common)
            eng27 = simulate_pair(lp, sp, REBALANCE_EVERY, gate=gate)
            if not eng27.empty:
                eng27['pair_id'] = p.pair_id
                eng27['underlying'] = p.underlying
                eng27['factor'] = p.factor
                eng27['split'] = eng27['date'].apply(split_of)
                eng27['vol_ann'] = eng27['date'].map(vol.to_dict())
                eng27['theory_drag_bps'] = (p.factor ** 2 - p.factor) / 2.0 * \
                    (eng27['vol_ann'] / np.sqrt(252)) ** 2 * 1e4
                eng27['pair_etb_both'] = etb(p.long_symbol) and etb(p.short_symbol)
                for rname, rail in BORROW_RAILS.items():
                    eng27[f'net_bps_{rname}'] = net_bps(eng27['pnl_bps_gross'],
                                                          eng27['cost_bps'],
                                                          eng27['gross_notional_frac'], rail)
                all_days_1627.append(eng27.copy())

        # 1,628 single-leg hedge, report-only (2x pairs only)
        if p.factor == 2.0 and not up.empty:
            eng28 = simulate_single_leg_hedge(lp, up, REBALANCE_EVERY)
            if not eng28.empty:
                eng28['pair_id'] = p.pair_id
                eng28['underlying'] = p.underlying
                eng28['split'] = eng28['date'].apply(split_of)
                for rname, rail in BORROW_RAILS.items():
                    borrow_bps = 1.0 / 3.0 * (rail / 252.0) * 1e4  # only the ETF leg is short
                    eng28[f'net_bps_{rname}'] = eng28['pnl_bps_gross'] - eng28['cost_bps'] - borrow_bps
                all_days_1628.append(eng28.copy())

    d1626 = pd.concat(all_days_1626, ignore_index=True) if all_days_1626 else pd.DataFrame()
    d1627 = pd.concat(all_days_1627, ignore_index=True) if all_days_1627 else pd.DataFrame()
    d1628 = pd.concat(all_days_1628, ignore_index=True) if all_days_1628 else pd.DataFrame()
    d1629 = pd.concat(all_days_1629, ignore_index=True) if all_days_1629 else pd.DataFrame()

    keep_cols = ['pair_id', 'underlying', 'factor', 'date', 'split', 'long_symbol',
                 'short_symbol', 'r_long', 'r_short', 'is_rebalance', 'pnl_bps_gross',
                 'cost_bps', 'notional_gap', 'vol_ann', 'theory_drag_bps',
                 'pair_etb_both', 'net_bps_5pct', 'net_bps_15pct', 'net_bps_30pct']
    if not d1626.empty:
        d1626[keep_cols].to_csv(f'{OUT}/cell_1626_days.csv', index=False)
    for name, d in [('1627', d1627), ('1628', d1628), ('1629', d1629)]:
        if not d.empty:
            cols = [c for c in keep_cols if c in d.columns]
            d[cols].to_csv(f'{OUT}/cell_{name}_days.csv', index=False)

    rows_out = []
    for cell_id, d, rebalance_note in [
        ('1626', d1626, 'static, 5-session rebalance'),
        ('1627', d1627, 'vol-gated 60%/40%'),
        ('1629', d1629, '3x/1.5x/1.75x pairs, report-only'),
    ]:
        if d.empty:
            continue
        d = d[d.position_open] if 'position_open' in d.columns else d
        for split in ['TRAIN', 'VAL']:
            ds = d[d.split == split]
            if ds.empty:
                continue
            for rname in BORROW_RAILS:
                col = f'net_bps_{rname}'
                y = ds[col]
                t = day_clustered_t(y, ds['date'])
                per_pair = ds.groupby('pair_id')[col].mean()
                share_pos = float((per_pair > 0).mean()) if len(per_pair) else np.nan
                pm = ds.copy()
                pm['ym'] = pm['date'].str.slice(0, 7)
                pair_month = pm.groupby(['pair_id', 'ym'])[col].sum()
                worst_month = float(pair_month.min()) if len(pair_month) else np.nan
                port_day = ds.groupby('date')[col].mean().sort_index()
                worst_day = float(port_day.min()) if len(port_day) else np.nan
                cum = port_day.cumsum()
                dd = (cum.cummax() - cum)
                max_dd_bps = float(dd.max()) if len(dd) else np.nan
                n_etb_both = ds.loc[ds.pair_etb_both, 'pair_id'].nunique() \
                    if 'pair_etb_both' in ds.columns else 0
                # vol-tercile calibration (top-vol tercile only, rail-independent: gross)
                dv = ds.dropna(subset=['vol_ann'])
                drag_ratio = np.nan
                if len(dv) >= 30:
                    terc = pd.qcut(dv['vol_ann'], 3, labels=['low', 'mid', 'high'], duplicates='drop')
                    top = dv.loc[terc == 'high'] if 'high' in getattr(terc, 'categories', []) else dv
                    if len(top) and top['theory_drag_bps'].mean() not in (0, np.nan):
                        drag_ratio = float(top['pnl_bps_gross'].mean() / top['theory_drag_bps'].mean()) \
                            if top['theory_drag_bps'].mean() else np.nan
                rows_out.append(dict(
                    cell=cell_id, split=split, borrow_rail=rname,
                    n_pair_days=int(len(ds)), n_pairs=int(ds.pair_id.nunique()),
                    mean_bps_day=float(y.mean()), t=float(t) if pd.notna(t) else np.nan,
                    share_pairs_positive=share_pos, worst_month_pct=worst_month / 100.0,
                    worst_day_pct=worst_day / 100.0, max_dd_pct=max_dd_bps / 100.0,
                    n_pairs_etb_both=int(n_etb_both),
                    drag_realised_over_theory_topvol=drag_ratio,
                ))
    summary = pd.DataFrame(rows_out)
    summary.to_csv(f'{OUT}/cell_1626_summary.csv', index=False)
    print(f'[score] wrote {len(summary)} (cell,split,rail) rows -> cell_1626_summary.csv', flush=True)

    write_result_md(pairs, d1626, d1627, d1628, d1629, summary, pair_first_date)


def write_result_md(pairs, d1626, d1627, d1628, d1629, summary, pair_first_date) -> None:
    lines = []
    lines.append('# RESULT — cells 1,626-1,629: leveraged single-stock ETF decay pair')
    lines.append('')
    lines.append(f'Generated {datetime.now(timezone.utc).isoformat(timespec="seconds")} by '
                 f'research/lev_decay/cell_1626.py. PREREG: research/lev_decay/PREREG_1626.md '
                 f'(FROZEN 2026-09-28 18:30 UTC). This is the BUILDER run: numbers below are '
                 f'**NOT yet independently reimplemented** (PREREG "Independent check" step 1 '
                 f'is a separate, not-yet-run task) -- do not relay the headline to the owner '
                 f'until that rebuild, the obtainability check and the tail check (steps 1-5 of '
                 f'the CLAUDE.md research-claim gate) have run.')
    lines.append('')
    lines.append('## Pair matching')
    n2 = int((pairs.factor == 2.0).sum())
    n3 = int((pairs.factor == 3.0).sum())
    n15 = int(pairs.factor.isin([1.5, 1.75]).sum())
    lines.append(f'{len(pairs)} pairs matched from `{ASSET_LIST}`: {n2} at 2x, {n3} at 3x, '
                 f'{n15} at 1.5x/1.75x. Parser: `parse_name()` in this file -- leverage factor '
                 f'from a `\\d+(\\.\\d+)?X` regex on the asset name; direction from '
                 f'LONG/BULL/UP vs SHORT/BEAR/INVERSE/DOWN/REVERSE tokens; underlying resolved '
                 f'(1) via a company-name alias table [T-Rex spells out names like "NVIDIA"], '
                 f'(2) via direct match against the full Alpaca symbol universe, (3) weak '
                 f'fallback = first remaining candidate token (flagged `weak_unconfirmed` in '
                 f'`leveraged_names_parsed.csv` for manual review). Full parsed set: '
                 f'`leveraged_names_parsed.csv`; matched pairs: `pairs.csv`; alternate-issuer '
                 f'legs not used (e.g. TSLA has both TSLL/TSLR as 2x-long and TSDD/TSLQ as '
                 f'2x-short -- one of each picked deterministically, tradable+active first, '
                 f'then alphabetical): `pairs_alt_legs_skipped.txt`.')
    lines.append('')
    lines.append(f'**Cell 1,629 is EMPTY by finding, not by omission.** Every 3x/1.5x/1.75x '
                 f'name that matched the leverage-factor regex tracks a sector, country or '
                 f'broad-index basket (e.g. TNA/TZA=Russell small-cap, EDC/EDZ=MSCI EM, '
                 f'DRN/DRV=real estate, TMF/TMV=20yr Treasury, HIBL/HIBS=S&P500 high-beta, '
                 f'YINN/YANG=FTSE China, DPST/WDRW=regional banks, GASL/GASX=natural gas) -- '
                 f'NONE are single-stock. In this Alpaca asset snapshot, single-stock leveraged '
                 f'products are essentially all 2x (matches the PREREG examples: TSLR/TSDD, '
                 f'AMZU/AMZD, AVGG, BABX). The 17 excluded underlyings, and the one confirmed '
                 f'FALSE PAIR the hand-check caught (TBXU "Biotech Top 5 Bull 2X" wrongly paired '
                 f'with TSXD "Semiconductors Top 5 Bear 2X" on the shared word "TOP" -- two '
                 f'different baskets, not a real pair), are in '
                 f'`pairs_excluded_non_single_stock.csv` with the blocklist and rationale in '
                 f'`NON_SINGLE_STOCK_BLOCKLIST` at the top of this file.')
    lines.append('')
    n_inactive_pairs = int(((pairs.long_status.astype(str).str.lower() != 'active') |
                             (pairs.short_status.astype(str).str.lower() != 'active')).sum())
    lines.append(f'Survivorship: {n_inactive_pairs}/{len(pairs)} pairs have at least one leg '
                 f'NOT status=active in the (current, 2026-09-05) asset-list snapshot -- these '
                 f'are closed/delisted legs, kept in the population and dated by their available '
                 f'bars, per PREREG ("count the ones that closed").')
    lines.append('')
    lines.append('### Hand-check of 20 pairs')
    sample = pairs.sample(n=min(20, len(pairs)), random_state=1626).sort_values('pair_id')
    lines.append('| pair | underlying | factor | long | short | long_conf | short_conf |')
    lines.append('|---|---|---|---|---|---|---|')
    for r in sample.itertuples(index=False):
        lines.append(f'| {r.pair_id} | {r.underlying} | {r.factor} | {r.long_symbol} | '
                     f'{r.short_symbol} | {r.long_confidence} | {r.short_confidence} |')
    lines.append('')
    lines.append('Hand-checked against the source `name` strings in '
                 '`leveraged_names_parsed.csv`; any row not visibly a correct '
                 '(underlying, factor, direction) triple is a parser bug, not a data error.')
    lines.append('')

    lines.append('## First bar date per pair (listed-date coverage)')
    if pair_first_date:
        fd = pd.Series(pair_first_date).sort_values()
        fd_dt = pd.to_datetime(fd)
        median_date = fd_dt.iloc[len(fd_dt) // 2].date()
        lines.append(f'Earliest: {fd.min()} ({fd.index[0]}). Latest: {fd.max()} '
                     f'({fd.index[-1]}). Median: {median_date}. '
                     f'{int((fd > TRAIN_END).sum())}/{len(fd)} pairs first-traded after '
                     f'{TRAIN_END} (TRAIN-absent or TRAIN-partial, mostly VAL).')
    lines.append('')

    lines.append('## Cell 1,626 / 1,627 / 1,629 — per (cell, split, borrow rail)')
    lines.append('')
    if not summary.empty:
        lines.append('| cell | split | rail | n pair-days | n pairs | mean bps/day | '
                     't (day-clustered) | share pairs+ | worst month % | worst day % | '
                     'max DD % | pairs ETB-both | drag realised/theory (top-vol tercile) |')
        lines.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
        for r in summary.itertuples(index=False):
            lines.append(f'| {r.cell} | {r.split} | {r.borrow_rail} | {r.n_pair_days} | '
                         f'{r.n_pairs} | {r.mean_bps_day:.2f} | {r.t:.2f} | '
                         f'{r.share_pairs_positive:.2f} | {r.worst_month_pct:.2f} | '
                         f'{r.worst_day_pct:.2f} | {r.max_dd_pct:.2f} | '
                         f'{r.n_pairs_etb_both} | {r.drag_realised_over_theory_topvol:.2f} |')
    else:
        lines.append('EMPTY -- bars not fetched / scored yet.')
    lines.append('')

    lines.append('## Pass-bar checklist (frozen, VAL, per cell)')
    lines.append('Mean >= +4 bps/day net @ 15%/yr rail; day-clustered t >= 2.5; '
                 '>= 60% of pairs positive; TRAIN same sign t >= 1; realised drag within '
                 '30% of theory in the top-vol tercile; worst month >= -3%; '
                 '>= 10 pairs ETB-both at the snapshot.')
    lines.append('')
    if not summary.empty:
        for cell_id in ['1626', '1627', '1629']:
            val15 = summary[(summary.cell == cell_id) & (summary.split == 'VAL') &
                             (summary.borrow_rail == '15pct')]
            train15 = summary[(summary.cell == cell_id) & (summary.split == 'TRAIN') &
                               (summary.borrow_rail == '15pct')]
            if val15.empty:
                continue
            v = val15.iloc[0]
            checks = {
                'mean >= +4 bps/day (VAL, 15%)': v.mean_bps_day >= 4.0,
                't >= 2.5 (VAL, 15%)': v.t >= 2.5,
                '>=60% pairs positive (VAL)': v.share_pairs_positive >= 0.60,
                'TRAIN same sign, t>=1': (not train15.empty and
                    np.sign(train15.iloc[0].mean_bps_day) == np.sign(v.mean_bps_day) and
                    train15.iloc[0].t >= 1.0),
                'drag within 30% of theory (top-vol)': (pd.notna(v.drag_realised_over_theory_topvol)
                    and 0.7 <= v.drag_realised_over_theory_topvol <= 1.3),
                'worst month >= -3%': v.worst_month_pct >= -3.0,
                '>=10 pairs ETB-both': v.n_pairs_etb_both >= 10,
            }
            passed = sum(checks.values())
            lines.append(f'**Cell {cell_id}** ({passed}/{len(checks)} pass):')
            for k, ok in checks.items():
                lines.append(f'- [{"x" if ok else " "}] {k}')
            lines.append('')

    lines.append('## Cell 1,628 — single-leg hedge (report-only, no pass bar)')
    if not d1628.empty:
        for split in ['TRAIN', 'VAL']:
            ds = d1628[d1628.split == split]
            if ds.empty:
                continue
            for rname in BORROW_RAILS:
                col = f'net_bps_{rname}'
                t = day_clustered_t(ds[col], ds['date'])
                lines.append(f'- {split} {rname}: n={len(ds)} mean={ds[col].mean():.2f} '
                             f'bps/day (of $3 gross) t={t:.2f}')
    else:
        lines.append('EMPTY.')
    lines.append('')

    lines.append('## Costs, borrow, conventions (exact, for the independent rebuild)')
    lines.append(f'- Bars: Alpaca daily, `adjustment=split` (NOT raw -- these products carry '
                 f'frequent reverse splits; raw closes would fabricate +/-90% jump days).')
    lines.append(f'- Gross notional basis: $2 fixed (1,626/1,627/1,629) or $3 fixed (1,628), '
                 f'i.e. bps are relative to the STATIC target notional at the last rebalance, '
                 f'not a daily mark-to-market renormalisation -- this is exactly how the '
                 f'position drifts between rebalances (reported as `notional_gap`).')
    lines.append(f'- Rebalance: every {REBALANCE_EVERY} AVAILABLE trading sessions (not '
                 f'calendar days) since the last rebalance/open; cost = '
                 f'{COST_BPS_PER_LEG} bps * dollar size of the rebalancing trade, per leg '
                 f'(the opening trade is a full $1/leg; later rebalances trade only the drift '
                 f'back to $1).')
    lines.append(f'- Borrow: notional-based, `(D_long_prev + D_short_prev)/2 * rail/252`, '
                 f'charged daily on both short legs at the 5%/15%/30% annual rails; NOT a '
                 f'real historical rate (Alpaca has none) -- a sensitivity rail, per PREREG.')
    lines.append(f'- ETB flags are TODAY\'s snapshot ({datetime.now(timezone.utc).date()}) '
                 f'applied to the whole history -- disclosed hindsight, not a historical '
                 f'borrowability series.')
    lines.append(f'- 1,627 gate: uses the UNDERLYING\'S 20-session realised vol (log-return '
                 f'std, annualised by sqrt(252)); ON at >=60%, OFF below 40%; a gate '
                 f'transition forces close (cost) then reopen (cost) rather than netting.')
    lines.append(f'- Theory drag per pair-day: (L^2-L)/2 * sigma_daily^2 * 10000 bps, using '
                 f'that day\'s rolling 20-session realised vol as the sigma estimate (a proxy '
                 f'for the true path variance, not a perfect match).')
    lines.append('')

    lines.append('## Caveats (read as an adversary, per CLAUDE.md)')
    lines.append('- **No independent reimplementation yet.** PREREG requires an agent that has '
                 'not read this code to rebuild from prose and match trade-by-trade before this '
                 'goes in front of the owner. Not done in this BUILDER task.')
    lines.append('- **Obtainability not separately verified.** P&L uses CLOSE-to-close leg '
                 'returns with no explicit fill-price/spread model beyond the flat 5 bps/leg '
                 'cost; a close print is not always a restable order for an illiquid single-'
                 'stock leveraged ETF -- the 5 bps cost is a placeholder, not a measured NBBO '
                 'cost (CLAUDE.md #4 wants measured per-trade NBBO, not a band; not done here).')
    lines.append('- **"Weak_unconfirmed" underlying matches** should be treated as unverified '
                 'until hand-checked (see `leveraged_names_parsed.csv` `match_confidence` '
                 'column) -- the 20-pair sample above covers only part of the matched set.')
    lines.append('- **Multi-issuer underlyings**: only ONE long/short pair kept per '
                 '(underlying, factor); skipped alternates in `pairs_alt_legs_skipped.txt` are '
                 'a second, correlated pair on the same name -- not counted toward `n_pairs`, '
                 'by design (avoids pseudo-replication), but means true product coverage is '
                 'wider than `n_pairs` suggests.')
    lines.append('- **Theory-vs-realised drag ratio** uses the rolling 20-session vol as the '
                 'sigma proxy for that single day\'s theoretical drag, not the realised path '
                 'variance since the last rebalance -- a coarse calibration check, not exact.')
    lines.append('- **Trend-path tail** (the mechanism\'s stated failure mode) is only visible '
                 'via worst-month/worst-day/max-DD above; no separate decomposition of trend '
                 'vs chop regimes was run.')
    lines.append('- **Borrow cost is a rail, not a rate**: real borrow on illiquid single-stock '
                 '2x products can spike far above 30%/yr or be recalled outright; that scenario '
                 'is not modelled beyond the static-rail sensitivity.')
    lines.append('')
    with open(RESULT_MD, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'[score] wrote {RESULT_MD}', flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--match', action='store_true')
    ap.add_argument('--fetch-flags', action='store_true')
    ap.add_argument('--fetch-bars', action='store_true')
    ap.add_argument('--score', action='store_true')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--force-refetch', action='store_true')
    args = ap.parse_args()

    if args.all or args.match:
        pairs = build_pairs()
    if args.all or args.fetch_flags:
        pairs = pd.read_csv(PAIRS_CSV)
        symbols = sorted(set(pairs.long_symbol) | set(pairs.short_symbol))
        fetch_flags(symbols)
    if args.all or args.fetch_bars:
        pairs = pd.read_csv(PAIRS_CSV)
        symbols = sorted(set(pairs.long_symbol) | set(pairs.short_symbol) | set(pairs.underlying))
        fetch_bars(symbols, force=args.force_refetch)
    if args.all or args.score:
        score()


if __name__ == '__main__':
    main()
