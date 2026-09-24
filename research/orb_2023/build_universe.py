"""2023-01 .. 2024-06 survivorship-free gapper universe for the out-of-regime ORB test.

Point-in-time symbol list: every ticker that traded on XNAS.ITCH in the period (Databento, delisted included —
data/research/databento/xnas_daily_2023_2024H1.parquet). Nasdaq-venue volume is NOT consolidated, so the universe
thresholds use CONSOLIDATED Alpaca SIP daily bars fetched here for every one of those tickers (2022-11-15 ..
2024-06-30; the lead gives 20-day lookbacks). Writes daily_alpaca.parquet (bar_date, symbol, open, high, low,
close, volume) and candidates.csv (symbol, bar_date): open >= 1.03 x prior close, open in [$3, $50], prior
consolidated volume >= 500,000; test tickers and '-' preferreds excluded. The 09:30-09:35 volume floor is applied
from the minute bars later. Market-hours rule: refuses to fetch between 13:25 and 20:05 UTC.

Usage: python3 research/orb_2023/build_universe.py
"""
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path('/home/ec2-user/onemil')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
XNAS = ROOT / 'data/research/databento/xnas_daily_2023_2024H1.parquet'
DAILY = HERE / 'daily_alpaca.parquet'
TEST_TICKER = re.compile(r'^Z[A-Z]ZZT$')
BATCH = 200


def fetch_daily(symbols):
    """Consolidated SIP daily bars for `symbols`, batched; invalid symbols are dropped from a batch and retried."""
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    from config import Config
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    cfg = Config()
    client = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_api_secret)
    start, end = datetime(2022, 11, 15, tzinfo=timezone.utc), datetime(2024, 7, 1, tzinfo=timezone.utc)
    frames, dropped, lost, t0 = [], [], [], time.time()
    for i in range(0, len(symbols), BATCH):
        chunk = list(symbols[i:i + BATCH])
        errors = 0
        # 2026-09-24 fix: invalid-symbol removals no longer consume the retry budget. The first version gave each
        # batch 8 attempts in total, so a batch holding > 8 unfetchable tickers exhausted them removing symbols and
        # was DROPPED SILENTLY (NVDA, AAPL, LCID and ~5,000 others lost). Real errors get 5 tries, then a loud loss.
        while chunk:
            try:
                df = client.get_stock_bars(StockBarsRequest(symbol_or_symbols=chunk, timeframe=TimeFrame.Day,
                                                            start=start, end=end, feed='sip')).df
                if df is not None and len(df):
                    frames.append(df.reset_index())
                break
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                bad = msg.split('invalid symbol: ')[-1].split('"')[0] if 'invalid symbol' in msg else None
                if bad and bad in chunk:
                    chunk.remove(bad)
                    dropped.append(bad)
                    continue
                errors += 1
                print(f'[WARNING] batch {i // BATCH} error {errors}: {msg[:120]}', flush=True)
                if errors >= 5:
                    print(f'[ERROR] batch {i // BATCH} LOST after 5 errors: {len(chunk)} symbols', flush=True)
                    lost.extend(chunk)
                    break
                time.sleep(3 * errors)
        if (i // BATCH) % 10 == 0:
            print(f'[daily] {i + len(chunk)}/{len(symbols)} symbols, {time.time() - t0:.0f}s', flush=True)
    d = pd.concat(frames, ignore_index=True)
    ts = pd.to_datetime(d.timestamp, utc=True).dt.tz_convert('America/New_York')
    out = pd.DataFrame(dict(bar_date=ts.dt.strftime('%Y-%m-%d'), symbol=d.symbol, open=d.open, high=d.high,
                            low=d.low, close=d.close, volume=d.volume))
    print(f'[daily] {len(out):,} bars for {out.symbol.nunique():,} symbols; {len(dropped)} invalid symbols dropped '
          f'(e.g. {dropped[:8]}); {len(lost)} symbols LOST to errors', flush=True)
    for must in ('AAPL', 'NVDA', 'LCID', 'PLUG', 'SPY'):
        if must not in set(out.symbol):
            raise SystemExit(f'[ERROR] completeness check failed: {must} has no daily bars')
    return out


def main():
    now = int(datetime.now(timezone.utc).strftime('%H%M'))
    if 1325 <= now < 2005:
        sys.exit(f'refusing to fetch inside the market-hours blackout (UTC {now:04d})')
    syms = pd.read_parquet(XNAS, columns=['symbol']).symbol.dropna().unique()
    # Parity with 2025-26 / 2024H2: '+' warrants -> Alpaca '.WS'; units ('='), rights ('^') and '-' preferreds never
    # appear in those universes -> excluded before fetching (they are also what Alpaca rejects).
    syms = sorted({(s[:-1] + '.WS' if s.endswith('+') else s) for s in syms
                   if '-' not in s and '=' not in s and '^' not in s and not TEST_TICKER.match(s)})
    print(f'XNAS point-in-time tickers (preferreds / test tickers removed): {len(syms):,}', flush=True)
    daily = pd.read_parquet(DAILY) if DAILY.exists() else fetch_daily(syms)
    daily.to_parquet(DAILY, index=False)
    d = daily.sort_values(['symbol', 'bar_date']).copy()
    d['prev_close'] = d.groupby('symbol').close.shift(1)
    d['prev_vol'] = d.groupby('symbol').volume.shift(1)
    m = ((d.bar_date >= '2023-01-03') & (d.bar_date <= '2024-06-28') & (d.prev_close > 0)
         & (d.open >= 1.03 * d.prev_close) & (d.open >= 3.0) & (d.open <= 50.0) & (d.prev_vol >= 500_000))
    cand = d.loc[m, ['symbol', 'bar_date']]
    cand.to_csv(HERE / 'candidates.csv', index=False)
    print(f'candidates {len(cand):,} symbol-days over {cand.bar_date.nunique()} days '
          f'(~{len(cand) / max(1, cand.bar_date.nunique()):.0f}/day)', flush=True)
    print(cand.groupby(cand.bar_date.str[:7]).size().to_string(), flush=True)


if __name__ == '__main__':
    main()
