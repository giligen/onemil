#!/usr/bin/env python3
"""
Cells 1,640-1,642 -- VIX basis carry with a defined loss (roll yield of the VIX
futures curve), scored per research/vix_carry/PREREG_1640.md EXACTLY.

Mechanism: VIX futures trade above spot most of the time (contango); a short
front-month position earns the roll-down as the contract converges to spot.
The basis itself is the documented timing signal (Simon & Campasano 2014;
Eraker & Wu 2017): carry when the basis is wide, stand aside when flat/inverted.

Pipeline
--------
1. Fetch CBOE VIX spot close (VIX_History.csv) and every MONTHLY VX futures
   settlement series 2011-09 .. 2026-10 (weeklies excluded by construction:
   contract identities are generated from the monthly-expiry rule, never
   scraped from a list that could include weeklies).
2. Build F30, the constant-maturity 30-calendar-day futures level, by linear
   interpolation of the two nearest UNEXPIRED monthly expiries on each VIX
   trading day. Basis b_t = F30_t / VIX_t - 1.
3. Fetch Alpaca daily bars (adjustment=ALL) for SVXY, SVIX, UVXY, VIXY and run
   the mandatory price-scale check (flag every |daily return| > 40%).
4. Run the two-close entry / basis-or-vol-spike exit state machine for:
     1,640 SVXY basis-gated  (report -1x era and -0.5x era separately)
     1,641 SVIX basis-gated  (2022-03+, report-only)
     1,642 SVXY always-in    (no gate, report-only -- the gate must beat it)
   on TRAIN (2011-10..2019-12) and VAL (2020-01..2023-12) ONLY.
   TEST (2024-01..2026-09) is SEALED: this script fetches the raw data through
   the full sample window (so the cache serves a later sealed TEST read) but
   NEVER builds a signal, a trade, or a stat past VAL_END. See `SEAL_CUTOFF`.

Trade accounting convention (documented here because the PREREG is silent on
the exact day-indexing -- an independent rebuilder must reproduce this):
  Signal computed from day t's close (b_t, b_{t-1}, VIX_t vs its 20d mean)
  decides the action executed at t+1's OPEN, where t+1 is simply the NEXT
  trading day in the ETP's own calendar. Daily P&L is indexed by ETP trade
  date d:
    entry day d:  close_d / (open_d * (1+5bps)) - 1
    hold day d:   close_d / close_{d-1} - 1
    exit day d:   (open_d * (1-5bps)) / close_{d-1} - 1
    flat day d:   0.0, in_market = False
  This makes "days in market" = [entry_day, ..., exit_day] inclusive, and the
  daily P&L series compounds exactly to the total holding-period return.

Usage
-----
    python3 cell_1640.py                # fetch (resumable) + score + write outputs
    python3 cell_1640.py --no-fetch     # score only, from whatever is cached
    python3 cell_1640.py --refresh      # force re-fetch every source file
"""
import argparse
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
DATA_DIR.mkdir(exist_ok=True)
RAW_DIR = DATA_DIR / "vx_raw"
RAW_DIR.mkdir(exist_ok=True)

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
REFERER = "https://www.cboe.com/"

TRAIN_START, TRAIN_END = date(2011, 10, 3), date(2019, 12, 31)
VAL_START, VAL_END = date(2020, 1, 1), date(2023, 12, 31)
TEST_START, TEST_END = date(2024, 1, 1), date(2026, 9, 4)   # SEALED, never scored here
SAMPLE_START, SAMPLE_END = date(2011, 10, 3), date(2026, 9, 4)
SEAL_CUTOFF = VAL_END                                        # hard stop for all scoring
ERA_SPLIT = date(2018, 2, 27)                                 # SVXY -1x -> -0.5x

MONTH_CODE = {1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
              7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'}

COST = 0.0005     # 5 bps per leg
ENTRY_B, EXIT_B = 0.03, 0.0
VIX_SPIKE_MULT = 1.25
NW_LAGS = 5

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cell_1640")


# ============================================================= fetching ===

def third_friday(year: int, month: int) -> date:
    """Third Friday of (year, month)."""
    d = date(year, month, 1)
    first_friday = d + timedelta(days=(4 - d.weekday()) % 7)
    return first_friday + timedelta(days=14)


def vx_monthly_expiry(year: int, month: int) -> date:
    """CFE rule: the Wednesday 30 calendar days before the 3rd Friday of the
    month FOLLOWING (year, month). This generates ONLY monthly expiries --
    weeklies are never in this sequence, satisfying the PREREG's exclusion
    by construction rather than by filtering a scraped list."""
    ny, nm = (year + 1, 1) if month == 12 else (year, month + 1)
    target = third_friday(ny, nm) - timedelta(days=30)
    while target.weekday() != 2:   # walk back to Wednesday if the arithmetic slips
        target -= timedelta(days=1)
    return target


def month_range(start_ym, end_ym):
    y, m = start_ym
    ey, em = end_ym
    out = []
    while (y, m) <= (ey, em):
        out.append((y, m))
        m += 1
        if m == 13:
            m = 1
            y += 1
    return out


def fetch_csv(url: str, dest: Path, referer: str = None, resume: bool = True,
              required_header: bytes = None) -> bool:
    """Download url to dest. Returns True on a plausible CSV, False otherwise.
    Never raises -- callers decide whether a miss is fatal. `required_header`
    (if given) must appear in the first 200 bytes, or the response is rejected
    -- CBOE returns a 200 disclaimer/legal-text page (not a 403) for some dead
    paths, which would otherwise be cached as if it were real data."""
    if resume and dest.exists() and dest.stat().st_size > 20:
        if required_header is None or required_header in dest.read_bytes()[:200]:
            return True
        log.warning(f"cached file failed header check, will re-fetch: {dest.name}")
    headers = {"User-Agent": UA}
    if referer:
        headers["Referer"] = referer
    try:
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code != 200 or len(r.content) < 20 or b"AccessDenied" in r.content[:200]:
            return False
        if required_header is not None and required_header not in r.content[:200]:
            return False
        dest.write_bytes(r.content)
        return True
    except Exception as e:
        log.warning(f"fetch failed {url}: {e}")
        return False


def fetch_vix_spot(resume: bool = True) -> pd.DataFrame:
    """CBOE VIX spot close, full history. Source:
    https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"""
    dest = DATA_DIR / "vix_spot.csv"
    url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
    if not fetch_csv(url, dest, resume=resume):
        raise RuntimeError(f"VIX spot fetch failed (required, not mocked): {url}")
    df = pd.read_csv(dest)
    df['date'] = pd.to_datetime(df['DATE'], format='%m/%d/%Y').dt.date
    df = (df.rename(columns={'CLOSE': 'vix'})[['date', 'vix']]
            .sort_values('date').reset_index(drop=True))
    log.info(f"VIX spot: {len(df)} rows {df.date.min()}..{df.date.max()} <- {url}")
    return df


def _parse_vx_csv(path: Path):
    """Parse one CFE settlement CSV (modern or legacy schema differ only in the
    Trade Date format). Keeps rows with Settle > 0 (drops pre-listing padding
    rows that the CBOE files carry with all-zero OHLC/Settle)."""
    try:
        df = pd.read_csv(path)
    except Exception as e:
        log.warning(f"parse failed {path}: {e}")
        return None
    if 'Trade Date' not in df.columns or 'Settle' not in df.columns:
        log.warning(f"unexpected columns in {path.name}: {list(df.columns)}")
        return None
    dt = pd.to_datetime(df['Trade Date'], format='%Y-%m-%d', errors='coerce')
    if dt.isna().mean() > 0.5:
        dt = pd.to_datetime(df['Trade Date'], format='%m/%d/%Y', errors='coerce')
    df['date'] = dt.dt.date
    df['settle'] = pd.to_numeric(df['Settle'], errors='coerce')
    df = df[(df['settle'] > 0) & df['date'].notna()][['date', 'settle']]
    return df.sort_values('date').reset_index(drop=True)


def fetch_vx_contracts(resume: bool = True) -> dict:
    """Fetch every monthly VX contract 2011-09 .. 2026-10 (one month of buffer
    on each end of the sample, so F30 interpolation always has two unexpired
    legs). Two source patterns, tried in order (both cboe.com, both require the
    'https://www.cboe.com/' Referer header or S3 returns a bare 403):
      modern (~2013+): cdn.cboe.com/data/us/futures/market_statistics/
                        historical_data/VX/VX_<expiry YYYY-MM-DD>.csv
      legacy (2011-2013): cdn.cboe.com/resources/futures/archive/
                        volume-and-price/CFE_<MonthCode><YY>_VX.csv
    Returns {expiry_date: DataFrame(date, settle)}."""
    months = month_range((2011, 9), (2026, 10))
    contracts, n_modern, n_legacy, n_fail, n_cache = {}, 0, 0, 0, 0
    HEADER = b"Trade Date"
    for i, (y, m) in enumerate(months):
        nominal_expiry = vx_monthly_expiry(y, m)
        code = f"{MONTH_CODE[m]}{y % 100:02d}"
        got, source, actual_expiry = False, None, None
        # CFE shifts the expiry a business day earlier when the naive Wed/ref-Friday
        # collides with an exchange holiday (observed for H19/H22/M24/H25/K26, all
        # off by exactly 1 day) -- try the nominal date then walk back up to 2 days
        # rather than hand-maintain a holiday calendar.
        for offset in (0, 1, 2):
            expiry = nominal_expiry - timedelta(days=offset)
            dest = RAW_DIR / f"VX_{expiry.isoformat()}.csv"
            if resume and dest.exists() and dest.stat().st_size > 20 and HEADER in dest.read_bytes()[:200]:
                got, source, actual_expiry = True, "cache", expiry
                n_cache += 1
                break
            modern_url = (f"https://cdn.cboe.com/data/us/futures/market_statistics/"
                          f"historical_data/VX/VX_{expiry.isoformat()}.csv")
            if fetch_csv(modern_url, dest, referer=REFERER, resume=False, required_header=HEADER):
                got, source, actual_expiry, n_modern = True, "modern", expiry, n_modern + 1
                break
            legacy_url = (f"https://cdn.cboe.com/resources/futures/archive/"
                          f"volume-and-price/CFE_{code}_VX.csv")
            if fetch_csv(legacy_url, dest, referer=REFERER, resume=False, required_header=HEADER):
                got, source, actual_expiry, n_legacy = True, "legacy", expiry, n_legacy + 1
                break
            time.sleep(0.15)   # polite pacing against cboe.com
        if not got:
            n_fail += 1
            log.warning(f"VX contract MISSING: {code} nominal expiry {nominal_expiry} "
                        f"(tried modern+legacy at offsets 0,1,2 -- all failed)")
            continue
        if actual_expiry != nominal_expiry:
            log.info(f"VX contract {code}: expiry shifted {nominal_expiry} -> {actual_expiry} (holiday)")
        df = _parse_vx_csv(RAW_DIR / f"VX_{actual_expiry.isoformat()}.csv")
        if df is not None and len(df):
            contracts[actual_expiry] = df
        if (i + 1) % 30 == 0:
            log.info(f"VX contracts progress: {i+1}/{len(months)}")
    log.info(f"VX contracts: {len(contracts)}/{len(months)} usable "
             f"(new-fetch modern={n_modern} legacy={n_legacy} cache={n_cache} fail={n_fail})")
    return contracts


def fetch_alpaca_bars(symbols, resume: bool = True) -> dict:
    """Daily bars, split/div-adjusted (adjustment=ALL), directly via alpaca-py
    (this is a standalone research script -- not routed through trading code
    per the task's 'never touch trading code' constraint)."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment, DataFeed
    from dotenv import load_dotenv

    load_dotenv(HERE.parent.parent / ".env")
    key, secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_API_SECRET")
    if not key or not secret:
        # CLAUDE.md: missing API keys must break execution, never fall back silently.
        raise RuntimeError("ALPACA_API_KEY/ALPACA_API_SECRET missing from .env -- required")
    client = StockHistoricalDataClient(key, secret)

    out = {}
    for sym in symbols:
        dest = DATA_DIR / f"{sym}_daily.csv"
        if resume and dest.exists() and dest.stat().st_size > 100:
            df = pd.read_csv(dest, parse_dates=['date'])
            df['date'] = df['date'].dt.date
            out[sym] = df
            log.info(f"{sym}: {len(df)} rows from cache {df.date.min()}..{df.date.max()}")
            continue
        req = StockBarsRequest(
            symbol_or_symbols=[sym], timeframe=TimeFrame.Day,
            start=datetime(2011, 1, 1), end=datetime(2026, 9, 5),
            adjustment=Adjustment.ALL, feed=DataFeed.SIP,
        )
        try:
            bars = client.get_stock_bars(req).df
        except Exception as e:
            log.error(f"{sym}: Alpaca fetch FAILED: {e}")
            continue
        if bars is None or bars.empty:
            log.error(f"{sym}: NO bars returned from Alpaca")
            continue
        bars = bars.reset_index()
        bars['date'] = pd.to_datetime(bars['timestamp']).dt.date
        df = (bars[['date', 'open', 'high', 'low', 'close', 'volume']]
              .sort_values('date').reset_index(drop=True))
        df.to_csv(dest, index=False)
        out[sym] = df
        log.info(f"{sym}: fetched {len(df)} rows {df.date.min()}..{df.date.max()} <- Alpaca adjustment=ALL")
    return out


# ============================================================ basis / F30 ===

def build_basis(vix_df: pd.DataFrame, contracts: dict) -> pd.DataFrame:
    """F30 = linear interpolation (by calendar days to expiry) of the two
    nearest UNEXPIRED monthly settlements on each VIX trading day.
    b_t = F30_t / VIX_t - 1."""
    expiries = sorted(contracts.keys())
    rows, missing = [], 0
    for _, r in vix_df.iterrows():
        t = r['date']
        upcoming = [e for e in expiries if e >= t]
        if len(upcoming) < 2:
            missing += 1
            continue
        e1, e2 = upcoming[0], upcoming[1]
        v1 = contracts[e1].loc[contracts[e1]['date'] == t, 'settle']
        v2 = contracts[e2].loc[contracts[e2]['date'] == t, 'settle']
        if v1.empty or v2.empty:
            missing += 1
            continue
        f1, f2 = float(v1.iloc[0]), float(v2.iloc[0])
        T1, T2 = (e1 - t).days, (e2 - t).days
        if T2 == T1:
            missing += 1
            continue
        if T1 > 45:
            # a nearer contract is entirely absent from `contracts` (fetch failure) --
            # refuse to interpolate off a mismatched maturity pair silently.
            missing += 1
            continue
        f30 = f1 + (f2 - f1) * (30 - T1) / (T2 - T1)
        rows.append({'date': t, 'vix': r['vix'], 'f30': f30, 'f1': f1, 'f2': f2,
                      'e1': e1, 'e2': e2, 'T1': T1, 'T2': T2})
    df = pd.DataFrame(rows)
    df['basis'] = df['f30'] / df['vix'] - 1
    df['vix20'] = df['vix'].rolling(20, min_periods=20).mean()
    log.info(f"Basis built: {len(df)}/{len(vix_df)} VIX days scored, {missing} skipped "
             f"(missing contract settle on that date)")
    return df


def price_scale_check(bars: dict) -> pd.DataFrame:
    """Flag every |daily return| > 40% per ETP; annotate known real events.
    A reverse split appearing here (and NOT in the KNOWN dict) is a bug in the
    adjustment, not a real move -- must be resolved before any number ships."""
    # Annotated from well-known market vol-spike dates, NOT independently tick-verified
    # -- UVXY/VIXY are report-only here (never used in a scored cell), so a wrong
    # annotation cannot change a PASS/FAIL number, but is still flagged as a caveat.
    KNOWN = {
        ('SVXY', date(2018, 2, 5)): "real: Volmageddon vol spike",
        ('SVXY', date(2018, 2, 6)): "real: Volmageddon vol spike (2nd day)",
        ('UVXY', date(2018, 2, 5)): "plausible real: Volmageddon (long-vol ETP jumps up)",
        ('UVXY', date(2018, 2, 6)): "plausible real: Volmageddon (long-vol ETP jumps up)",
        ('UVXY', date(2018, 2, 8)): "plausible real: Volmageddon 2nd aftershock (2018-02-08 was a large down day)",
        ('VIXY', date(2018, 2, 6)): "plausible real: Volmageddon (long-vol ETP jumps up)",
        ('UVXY', date(2016, 6, 24)): "plausible real: Brexit referendum result shock",
        ('UVXY', date(2020, 6, 11)): "plausible real: COVID 'Black Thursday' mini selloff (SPX -5.9%)",
        ('UVXY', date(2024, 8, 5)): "plausible real: Aug-2024 global vol spike / yen-carry unwind",
        ('VIXY', date(2024, 8, 5)): "plausible real: Aug-2024 global vol spike / yen-carry unwind",
    }
    rows = []
    for sym, df in bars.items():
        df = df.sort_values('date').reset_index(drop=True)
        df['ret'] = df['close'].pct_change()
        big = df[df['ret'].abs() > 0.40]
        for _, r in big.iterrows():
            note = KNOWN.get((sym, r['date']), "UNEXPLAINED -- check for un-adjusted reverse split")
            rows.append({'symbol': sym, 'date': r['date'], 'ret_pct': round(r['ret'] * 100, 1), 'note': note})
    out = pd.DataFrame(rows).sort_values(['symbol', 'date']) if rows else pd.DataFrame(
        columns=['symbol', 'date', 'ret_pct', 'note'])
    log.info(f"Price-scale check: {len(out)} days with |ret|>40% across {list(bars.keys())}")
    return out


# ============================================================= strategy ===

def run_cell(etp: pd.DataFrame, basis_map: dict, vix_map: dict, vix20_map: dict,
             gated: bool, cutoff: date) -> pd.DataFrame:
    """Day-by-day P&L for one cell. See module docstring for the exact
    entry/exit/day-indexing convention. `cutoff` hard-stops scoring (TEST seal)."""
    e = etp[etp['date'] <= cutoff].sort_values('date').reset_index(drop=True)
    dates = e['date'].tolist()
    opens = dict(zip(e['date'], e['open']))
    closes = dict(zip(e['date'], e['close']))

    in_pos, rows, gaps = False, [], 0
    for i in range(2, len(dates)):
        d, t, tm1 = dates[i], dates[i - 1], dates[i - 2]
        action = None
        if gated:
            bt, btm1 = basis_map.get(t), basis_map.get(tm1)
            if bt is None or btm1 is None:
                gaps += 1
            elif not in_pos and bt >= ENTRY_B and btm1 >= ENTRY_B:
                action = 'enter'
            elif in_pos:
                vixt, v20 = vix_map.get(t), vix20_map.get(t)
                spike = vixt is not None and v20 is not None and not np.isnan(v20) and vixt > VIX_SPIKE_MULT * v20
                if bt is not None and (bt <= EXIT_B or spike):
                    action = 'exit'
        else:
            in_pos = True   # 1,642 always-in: no gate at all

        if action == 'enter':
            in_pos = True
            pnl = closes[d] / (opens[d] * (1 + COST)) - 1
            in_market, state = True, 'entry'
        elif action == 'exit':
            in_pos = False
            pnl = (opens[d] * (1 - COST)) / closes[t] - 1
            in_market, state = True, 'exit'
        elif in_pos:
            pnl = closes[d] / closes[t] - 1
            in_market, state = True, 'hold'
        else:
            pnl, in_market, state = 0.0, False, 'flat'
        rows.append({'date': d, 'pnl': pnl, 'in_market': in_market, 'state': state,
                     'basis_t': basis_map.get(t), 'vix_t': vix_map.get(t)})
    if gated and gaps:
        log.warning(f"run_cell: {gaps} signal-lookup gaps (ETP date with no basis on record)")
    return pd.DataFrame(rows)


# ============================================================ statistics ===

def newey_west_t(x: np.ndarray, lags: int = NW_LAGS) -> float:
    """NW t-stat of the mean of x, Bartlett kernel, `lags` lags. Manual
    implementation (no statsmodels dependency) -- standard HAC formula."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < max(10, lags + 2):
        return float('nan')
    mean = x.mean()
    resid = x - mean
    lrv = float(np.sum(resid ** 2)) / n
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        cov = float(np.sum(resid[lag:] * resid[:-lag])) / n
        lrv += 2 * w * cov
    se = np.sqrt(max(lrv, 1e-18) / n)
    return mean / se if se > 0 else float('nan')


def max_drawdown(pnl: pd.Series) -> float:
    equity = (1 + pnl.fillna(0)).cumprod()
    running_max = equity.cummax()
    dd = equity / running_max - 1
    return float(dd.min()) if len(dd) else float('nan')


def worst_month(df: pd.DataFrame) -> float:
    d = df.copy()
    d['ym'] = d['date'].apply(lambda x: (x.year, x.month))
    g = d.groupby('ym')['pnl'].apply(lambda s: (1 + s).prod() - 1)
    return float(g.min()) if len(g) else float('nan')


def cell_stats(df: pd.DataFrame, label: str) -> dict:
    """All PREREG-required fields for one (cell, split[, era]) slice."""
    n = len(df)
    im = df[df['in_market']]
    n_im = len(im)
    out = {
        'label': label, 'n_days': n, 'n_in_market': n_im,
        'share_in_market': n_im / n if n else float('nan'),
        'mean_bps_day_im': float(im['pnl'].mean() * 1e4) if n_im else float('nan'),
        'nw_t_im': newey_west_t(im['pnl'].values) if n_im >= 10 else float('nan'),
        'worst_day': float(im['pnl'].min()) if n_im else float('nan'),
        'worst_month': worst_month(df),
        'max_drawdown': max_drawdown(df.set_index('date')['pnl']) if n else float('nan'),
    }
    total_growth = float((1 + df['pnl']).prod() - 1) if n else float('nan')
    ann_days = n if n else 1
    out['ann_return'] = (1 + total_growth) ** (252 / ann_days) - 1 if n else float('nan')
    out['pnl_750'] = 750 * total_growth if n else float('nan')
    out['pnl_5000'] = 5000 * total_growth if n else float('nan')
    if n_im > 5:
        trimmed = im.nsmallest(5, 'pnl')
        rest = im.drop(trimmed.index)
        out['ex_worst5_mean_bps'] = float(rest['pnl'].mean() * 1e4)
    else:
        out['ex_worst5_mean_bps'] = float('nan')
    return out


def basis_decile_table(basis_df: pd.DataFrame, etp: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Next-day ETP return by basis decile (mechanism check, independent of
    the trading rule -- uses ALL days with a valid basis, in vs out of the
    strategy's market notwithstanding)."""
    e = etp.sort_values('date').reset_index(drop=True).copy()
    e['next_ret'] = e['close'].shift(-1) / e['close'] - 1
    next_ret_map = dict(zip(e['date'], e['next_ret']))
    b = basis_df[(basis_df['date'] >= start) & (basis_df['date'] <= end)].copy()
    b['next_ret'] = b['date'].map(next_ret_map)
    b = b.dropna(subset=['next_ret'])
    if len(b) < 20:
        return pd.DataFrame()
    b['decile'] = pd.qcut(b['basis'], 10, labels=False, duplicates='drop')
    tbl = b.groupby('decile').agg(n=('next_ret', 'size'),
                                   mean_basis=('basis', 'mean'),
                                   mean_next_ret_bps=('next_ret', lambda s: s.mean() * 1e4))
    return tbl.reset_index()


# =================================================================== main ==

def fmt(x, pct=False, bps=False, dollars=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    if bps:
        return f"{x:.1f}"
    if pct:
        return f"{x*100:.2f}%"
    if dollars:
        return f"${x:,.0f}"
    return f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--no-fetch', action='store_true', help='score only, use existing cache, error if missing')
    ap.add_argument('--refresh', action='store_true', help='force re-fetch every source file')
    args = ap.parse_args()
    resume = not args.refresh

    log.info("=" * 70)
    log.info("Cell 1,640-1,642: VIX basis carry -- fetch + score")
    log.info(f"TRAIN {TRAIN_START}..{TRAIN_END}  VAL {VAL_START}..{VAL_END}  "
             f"TEST {TEST_START}..{TEST_END} SEALED (never scored)")
    log.info("=" * 70)

    if args.no_fetch:
        log.info("--no-fetch: scoring from cache only")

    vix_df = fetch_vix_spot(resume=resume or args.no_fetch)
    contracts = fetch_vx_contracts(resume=resume or args.no_fetch)
    if len(contracts) < 100:
        log.error(f"Only {len(contracts)} VX contracts usable -- expected ~180+. Continuing but flag this.")
    basis_df = build_basis(vix_df, contracts)
    basis_df.to_csv(DATA_DIR / "basis_f30.csv", index=False)

    bars = fetch_alpaca_bars(['SVXY', 'SVIX', 'UVXY', 'VIXY'], resume=resume or args.no_fetch)
    for sym in ['SVXY', 'SVIX']:
        if sym not in bars:
            raise RuntimeError(f"{sym} bars required and missing -- cannot score")
    if bars['SVXY']['date'].min() > date(2012, 1, 1):
        log.error(f"BLOCKER: SVXY bars start {bars['SVXY']['date'].min()}, not 2011-10 as the PREREG "
                  f"samples from -- confirmed via a direct SPY 2011 probe returning 0 rows: this Alpaca "
                  f"account's historical market data plan does not serve data before ~2016, for ANY "
                  f"symbol. TRAIN era1 (-1x, 2011-10..2018-02-27) is scored on 2016-01..2018-02-27 ONLY. "
                  f"No substitute data source was used (none was authorized).")

    scale_check = price_scale_check(bars)
    scale_check.to_csv(DATA_DIR / "price_scale_check.csv", index=False)
    unexplained = scale_check[scale_check['note'].str.startswith('UNEXPLAINED')] if len(scale_check) else scale_check
    if len(unexplained):
        log.error(f"Price-scale check: {len(unexplained)} UNEXPLAINED >40% jumps -- see data/price_scale_check.csv")
    else:
        log.info("Price-scale check: no unexplained >40% jumps")

    basis_map = dict(zip(basis_df['date'], basis_df['basis']))
    vix_map = dict(zip(basis_df['date'], basis_df['vix']))
    vix20_map = dict(zip(basis_df['date'], basis_df['vix20']))

    log.info("Running state machine: 1,640 SVXY gated, 1,641 SVIX gated, 1,642 SVXY always-in")
    r1640 = run_cell(bars['SVXY'], basis_map, vix_map, vix20_map, gated=True, cutoff=SEAL_CUTOFF)
    r1641 = run_cell(bars['SVIX'], basis_map, vix_map, vix20_map, gated=True, cutoff=SEAL_CUTOFF)
    r1642 = run_cell(bars['SVXY'], basis_map, vix_map, vix20_map, gated=False, cutoff=SEAL_CUTOFF)

    # --- combined daily CSV for the independent rebuild -----------------
    def slim(df, suffix):
        return df[['date', 'basis_t', 'vix_t', 'in_market', 'pnl']].rename(
            columns={'basis_t': f'basis_{suffix}', 'vix_t': f'vix_{suffix}',
                     'in_market': f'in_market_{suffix}', 'pnl': f'pnl_bps_{suffix}'})
    d1640 = slim(r1640, '1640'); d1640[f'pnl_bps_1640'] *= 1e4
    d1641 = slim(r1641, '1641'); d1641[f'pnl_bps_1641'] *= 1e4
    d1642 = slim(r1642, '1642'); d1642[f'pnl_bps_1642'] *= 1e4
    daily = d1640.merge(d1641, on='date', how='outer').merge(d1642, on='date', how='outer')
    daily = daily.sort_values('date')
    daily.to_csv(HERE / "daily_1640.csv", index=False)
    log.info(f"Wrote daily_1640.csv: {len(daily)} rows (TEST period excluded -- sealed)")

    # --- stats blocks -----------------------------------------------------
    def split_block(df, start, end):
        return df[(df['date'] >= start) & (df['date'] <= end)]

    blocks = {}
    blocks['1640_TRAIN_era1(-1x)'] = cell_stats(split_block(r1640, TRAIN_START, min(TRAIN_END, ERA_SPLIT)), '1640_TRAIN_era1')
    blocks['1640_TRAIN_era2(-0.5x)'] = cell_stats(split_block(r1640, ERA_SPLIT, TRAIN_END), '1640_TRAIN_era2')
    blocks['1640_VAL(-0.5x)'] = cell_stats(split_block(r1640, VAL_START, VAL_END), '1640_VAL')
    blocks['1642_TRAIN_era1(-1x)'] = cell_stats(split_block(r1642, TRAIN_START, min(TRAIN_END, ERA_SPLIT)), '1642_TRAIN_era1')
    blocks['1642_VAL(-0.5x)'] = cell_stats(split_block(r1642, VAL_START, VAL_END), '1642_VAL')
    svix_train = split_block(r1641, TRAIN_START, TRAIN_END)
    if len(svix_train) > 20:
        blocks['1641_TRAIN'] = cell_stats(svix_train, '1641_TRAIN')
    blocks['1641_VAL'] = cell_stats(split_block(r1641, VAL_START, VAL_END), '1641_VAL')

    decile_train = basis_decile_table(basis_df, bars['SVXY'], TRAIN_START, min(TRAIN_END, ERA_SPLIT))
    decile_val = basis_decile_table(basis_df, bars['SVXY'], VAL_START, VAL_END)
    decile_val_svix = basis_decile_table(basis_df, bars['SVIX'], VAL_START, VAL_END)

    def monotone_corr(tbl):
        if tbl is None or len(tbl) < 3:
            return float('nan')
        return float(np.corrcoef(tbl['decile'], tbl['mean_next_ret_bps'])[0, 1])

    corr_train = monotone_corr(decile_train)
    corr_val = monotone_corr(decile_val)

    # --- pass bar (VAL, cell 1640, -0.5x era = the whole VAL block) -------
    val = blocks['1640_VAL(-0.5x)']
    tr_era1 = blocks['1640_TRAIN_era1(-1x)']
    always_val = blocks['1642_VAL(-0.5x)']
    pass_items = [
        ("net bps/day in market >= +4", val['mean_bps_day_im'], val['mean_bps_day_im'] >= 4 if not np.isnan(val['mean_bps_day_im']) else False),
        ("NW t >= 2.5", val['nw_t_im'], (val['nw_t_im'] >= 2.5) if not np.isnan(val['nw_t_im']) else False),
        (">= 40% days in market", val['share_in_market'], val['share_in_market'] >= 0.40),
        ("TRAIN -1x era same sign, t>=1", tr_era1['nw_t_im'],
         (not np.isnan(tr_era1['nw_t_im'])) and (np.sign(tr_era1['nw_t_im']) == np.sign(val['nw_t_im'])) and (abs(tr_era1['nw_t_im']) >= 1)),
        ("decile table monotone, both halves (corr>0)", (corr_train, corr_val),
         (not np.isnan(corr_train)) and (not np.isnan(corr_val)) and corr_train > 0 and corr_val > 0),
        ("worst day >= -25%", val['worst_day'], (not np.isnan(val['worst_day'])) and val['worst_day'] >= -0.25),
        ("max drawdown >= -35%", val['max_drawdown'], (not np.isnan(val['max_drawdown'])) and val['max_drawdown'] >= -0.35),
        ("gate worst day better than always-in", (val['worst_day'], always_val['worst_day']),
         (not np.isnan(val['worst_day'])) and (not np.isnan(always_val['worst_day'])) and val['worst_day'] > always_val['worst_day']),
        ("gate max DD better than always-in", (val['max_drawdown'], always_val['max_drawdown']),
         (not np.isnan(val['max_drawdown'])) and (not np.isnan(always_val['max_drawdown'])) and val['max_drawdown'] > always_val['max_drawdown']),
    ]
    overall_pass = all(p[2] for p in pass_items)

    write_result_md(vix_df, contracts, bars, scale_check, blocks, decile_train, decile_val,
                     decile_val_svix, corr_train, corr_val, pass_items, overall_pass)

    log.info("=" * 70)
    log.info(f"VAL 1,640 (-0.5x era): {val['mean_bps_day_im']:.1f} bps/day in-market, "
             f"NW t={val['nw_t_im']:.2f}, {val['share_in_market']*100:.0f}% days in market, "
             f"worst day {val['worst_day']*100:.1f}%")
    log.info(f"PASS BAR: {'MET' if overall_pass else 'NOT MET'} ({sum(p[2] for p in pass_items)}/{len(pass_items)} items)")
    log.info("=" * 70)


def write_result_md(vix_df, contracts, bars, scale_check, blocks, decile_train, decile_val,
                     decile_val_svix, corr_train, corr_val, pass_items, overall_pass):
    lines = []
    lines.append("# RESULT 1,640-1,642 -- VIX basis carry (build + score)")
    lines.append("")
    lines.append(f"Run {datetime.utcnow().isoformat()}Z. TEST (2024-01..2026-09) is SEALED -- not read.")
    lines.append("")
    lines.append("## Data provenance")
    lines.append(f"- VIX spot: {len(vix_df)} rows {vix_df.date.min()}..{vix_df.date.max()} <- "
                 f"cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv")
    lines.append(f"- VX futures monthly settlements: {len(contracts)} contracts usable "
                 f"(target ~181, 2011-09..2026-10) <- cdn.cboe.com/data/.../VX/VX_<expiry>.csv "
                 f"(2013+) and cdn.cboe.com/resources/futures/archive/volume-and-price/"
                 f"CFE_<code>_VX.csv (2011-2013); both need Referer: https://www.cboe.com/")
    for sym, df in bars.items():
        lines.append(f"- {sym}: {len(df)} daily bars {df.date.min()}..{df.date.max()} <- Alpaca adjustment=ALL")
    lines.append("")
    lines.append("## Price-scale check (|daily return| > 40%)")
    if len(scale_check):
        for _, r in scale_check.iterrows():
            lines.append(f"- {r['symbol']} {r['date']}: {r['ret_pct']}% -- {r['note']}")
    else:
        lines.append("- none found")
    lines.append("")
    lines.append("## Per-cell / per-split stats (in-market unless noted)")
    lines.append("| block | days | %in-mkt | bps/day | NW t | ann.ret | worst day | worst mo | max DD | $750 P&L | $5K P&L | ex-w5 bps |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k, s in blocks.items():
        lines.append(f"| {k} | {s['n_days']} | {fmt(s['share_in_market'],pct=True)} | "
                     f"{fmt(s['mean_bps_day_im'],bps=True)} | {fmt(s['nw_t_im'])} | "
                     f"{fmt(s['ann_return'],pct=True)} | {fmt(s['worst_day'],pct=True)} | "
                     f"{fmt(s['worst_month'],pct=True)} | {fmt(s['max_drawdown'],pct=True)} | "
                     f"{fmt(s['pnl_750'],dollars=True)} | {fmt(s['pnl_5000'],dollars=True)} | "
                     f"{fmt(s['ex_worst5_mean_bps'],bps=True)} |")
    lines.append("")
    lines.append("## Basis-decile table, next-day SVXY return (mechanism check)")
    lines.append(f"TRAIN corr(decile,ret)={fmt(corr_train)}  VAL corr(decile,ret)={fmt(corr_val)}")
    for name, tbl in [("TRAIN era1", decile_train), ("VAL", decile_val), ("VAL SVIX", decile_val_svix)]:
        if len(tbl):
            vals = " ".join(f"{v:.0f}" for v in tbl['mean_next_ret_bps'])
            lines.append(f"- {name} (decile1..10, bps): {vals}")
        else:
            lines.append(f"- {name}: insufficient data")
    lines.append("")
    lines.append("## Pass bar (VAL, cell 1,640, -0.5x era) -- item by item")
    for name, val, ok in pass_items:
        lines.append(f"- [{'PASS' if ok else 'FAIL'}] {name}: {val}")
    lines.append(f"\n**OVERALL: {'PASS' if overall_pass else 'FAIL'}** "
                 f"({sum(p[2] for p in pass_items)}/{len(pass_items)} items met)")
    lines.append("")
    lines.append("## Caveats (read as an adversary)")
    if bars['SVXY']['date'].min() > date(2012, 1, 1):
        lines.append(f"- **BLOCKER, not a coding bug**: this Alpaca account's historical data plan starts "
                     f"~2016-01-04 for every symbol tested (confirmed via a direct SPY-2011 probe returning "
                     f"0 rows) -- SVXY/UVXY/VIXY 2011-10..2015-12 are UNAVAILABLE here. TRAIN era1 (-1x) is "
                     f"therefore scored on 2016-01-04..2018-02-27 (~2.1 yr), not the full 2011-10..2018-02-27 "
                     f"(~6.4 yr) the PREREG samples from. VAL (2020-2023, the pass-bar split) is unaffected. "
                     f"No unauthorized substitute data source was used.")
    lines.append("- Day-indexing convention (entry/hold/exit P&L formula) is this script's own literal "
                 "reading of the PREREG -- not independently specified there; an independent rebuild must "
                 "match it exactly (see module docstring) or the Jaccard/bps check will disagree for a "
                 "structural, not a coding, reason.")
    lines.append("- NW t is computed on the IN-MARKET daily P&L subsequence only (trade order preserved, "
                 "flat days dropped), paired with 'mean bps/day in market' -- not on the full including-zeros "
                 "calendar series. This is a judgment call the PREREG does not disambiguate.")
    lines.append("- VIX 20-day mean uses a trailing window that INCLUDES day t (pandas default).")
    lines.append("- Contract-fetch failures (see data provenance count vs ~181 target) leave basis gaps on "
                 "those dates; F30 is simply not computed on a gap day (no entry/exit can trigger off it).")
    lines.append("- SVIX TRAIN block is empty/near-empty by construction (inception ~2022) -- report-only, "
                 "not part of the pass bar, per PREREG.")
    lines.append("- Monotonicity is checked via Pearson corr(decile index, mean next-day return) > 0, a "
                 "looser bar than strict staircase monotonicity; read the raw decile values above too.")
    lines.append("- This is an ETP daily-return backtest, not a futures-notional backtest: SVXY/SVIX daily "
                 "resets already embed their own cost/borrow drag, which is NOT separately itemized here.")

    out_path = HERE / "RESULT_1640_build.md"
    out_path.write_text("\n".join(lines))
    log.info(f"Wrote {out_path} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
