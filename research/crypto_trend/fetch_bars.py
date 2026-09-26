"""Fetch daily BTC/ETH/SOL bars for the crypto-trend PREREG (cells C1-C4).

Sources:
  * Binance public klines (no key) from each coin's inception through END_DATE.
  * Alpaca CryptoHistoricalDataClient (the executable venue) from 2021-01-01 through END_DATE.

Writes research/crypto_trend/bars_daily.csv with columns:
  date, coin, source, open, high, low, close, volume

Also prints the median |Alpaca/Binance - 1| relative close difference on the
overlap window per coin (PREREG requires < 0.3%). TEST-window rows (>= 2025-07-01)
ARE fetched and written here (data collection is not scoring) but the scorer
(cells.py) must never score them.
"""
import logging
import statistics
import time
from datetime import datetime, timezone

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fetch_bars")

BINANCE_HOSTS = ["https://api.binance.com", "https://data-api.binance.vision"]
END_DATE = datetime(2026, 9, 26, tzinfo=timezone.utc)
ALPACA_START = datetime(2021, 1, 1, tzinfo=timezone.utc)

COINS = {
    "BTC": {"binance_symbol": "BTCUSDT", "inception": datetime(2017, 8, 17, tzinfo=timezone.utc), "alpaca_symbol": "BTC/USD"},
    "ETH": {"binance_symbol": "ETHUSDT", "inception": datetime(2017, 8, 17, tzinfo=timezone.utc), "alpaca_symbol": "ETH/USD"},
    "SOL": {"binance_symbol": "SOLUSDT", "inception": datetime(2020, 8, 11, tzinfo=timezone.utc), "alpaca_symbol": "SOL/USD"},
}

PAUSE_S = 0.3


def _working_binance_host() -> str:
    """Return the first Binance host that answers a trivial klines request."""
    for host in BINANCE_HOSTS:
        try:
            r = requests.get(f"{host}/api/v3/klines", params={"symbol": "BTCUSDT", "interval": "1d", "limit": 1}, timeout=10)
            if r.status_code == 200:
                log.info(f"Binance host reachable: {host}")
                return host
        except requests.RequestException as e:
            log.warning(f"Binance host {host} unreachable: {e}")
    raise RuntimeError("No Binance host reachable (tried api.binance.com and data-api.binance.vision)")


def fetch_binance_klines(host: str, symbol: str, start: datetime, end: datetime) -> list[dict]:
    """Paginate Binance /klines from start to end (UTC, daily). Returns list of bar dicts."""
    rows = []
    cursor_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    url = f"{host}/api/v3/klines"
    while cursor_ms < end_ms:
        params = {"symbol": symbol, "interval": "1d", "startTime": cursor_ms, "limit": 1000}
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200:
            log.error(f"Binance klines fetch failed for {symbol} at {cursor_ms}: HTTP {r.status_code} {r.text[:200]}")
            r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        for k in batch:
            open_ms = k[0]
            d = datetime.fromtimestamp(open_ms / 1000, tz=timezone.utc).date().isoformat()
            rows.append({
                "date": d, "coin": None, "source": "binance",
                "open": float(k[1]), "high": float(k[2]), "low": float(k[3]),
                "close": float(k[4]), "volume": float(k[5]),
            })
        last_open_ms = batch[-1][0]
        next_cursor = last_open_ms + 24 * 3600 * 1000
        if next_cursor <= cursor_ms:
            break
        cursor_ms = next_cursor
        time.sleep(PAUSE_S)
    return [row for row in rows if row["date"] <= end.date().isoformat()]


def fetch_alpaca_daily(client, alpaca_symbol: str, start: datetime, end: datetime) -> list[dict]:
    """Fetch Alpaca daily crypto bars via CryptoHistoricalDataClient (executable venue, read-only)."""
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame

    req = CryptoBarsRequest(symbol_or_symbols=alpaca_symbol, timeframe=TimeFrame.Day, start=start, end=end)
    bars = client.get_crypto_bars(req)
    df = bars.df
    rows = []
    if df is None or len(df) == 0:
        log.warning(f"Alpaca returned zero daily bars for {alpaca_symbol} {start.date()}..{end.date()}")
        return rows
    df = df.reset_index()
    for _, r in df.iterrows():
        d = r["timestamp"].date().isoformat() if hasattr(r["timestamp"], "date") else str(r["timestamp"])[:10]
        rows.append({
            "date": d, "coin": None, "source": "alpaca",
            "open": float(r["open"]), "high": float(r["high"]), "low": float(r["low"]),
            "close": float(r["close"]), "volume": float(r["volume"]),
        })
    return rows


def main():
    from dotenv import load_dotenv
    load_dotenv(".env")
    from config import Config
    c = Config()
    from alpaca.data.historical import CryptoHistoricalDataClient

    alpaca_client = CryptoHistoricalDataClient(c.alpaca_api_key, c.alpaca_api_secret)

    binance_host = _working_binance_host()
    all_rows = []
    overlap_diffs = {}

    for coin, meta in COINS.items():
        log.info(f"Fetching Binance {coin} ({meta['binance_symbol']}) from {meta['inception'].date()}..{END_DATE.date()}")
        b_rows = fetch_binance_klines(binance_host, meta["binance_symbol"], meta["inception"], END_DATE)
        for row in b_rows:
            row["coin"] = coin
        log.info(f"  Binance {coin}: {len(b_rows)} daily bars")

        log.info(f"Fetching Alpaca {coin} ({meta['alpaca_symbol']}) from {ALPACA_START.date()}..{END_DATE.date()}")
        try:
            a_rows = fetch_alpaca_daily(alpaca_client, meta["alpaca_symbol"], ALPACA_START, END_DATE)
        except Exception as e:
            log.error(f"Alpaca fetch failed for {coin}: {e}")
            a_rows = []
        for row in a_rows:
            row["coin"] = coin
        log.info(f"  Alpaca {coin}: {len(a_rows)} daily bars")

        all_rows.extend(b_rows)
        all_rows.extend(a_rows)

        b_close = {r["date"]: r["close"] for r in b_rows}
        a_close = {r["date"]: r["close"] for r in a_rows}
        overlap_dates = sorted(set(b_close) & set(a_close))
        if overlap_dates:
            diffs = [abs(a_close[d] / b_close[d] - 1) for d in overlap_dates if b_close[d] != 0]
            med = statistics.median(diffs) if diffs else float("nan")
            overlap_diffs[coin] = (med, len(overlap_dates))
            status = "OK" if med < 0.003 else "FAIL (>= 0.3%)"
            log.info(f"  {coin} overlap n={len(overlap_dates)} median |Alpaca/Binance-1| = {med*100:.4f}% [{status}]")
        else:
            overlap_diffs[coin] = (float("nan"), 0)
            log.warning(f"  {coin} has zero overlapping dates between Binance and Alpaca")

    all_rows.sort(key=lambda r: (r["coin"], r["date"], r["source"]))

    out_path = "research/crypto_trend/bars_daily.csv"
    import csv
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "coin", "source", "open", "high", "low", "close", "volume"])
        w.writeheader()
        w.writerows(all_rows)
    log.info(f"Wrote {len(all_rows)} rows to {out_path}")

    print("\n=== Alpaca vs Binance overlap (median |Alpaca/Binance - 1|) ===")
    for coin, (med, n) in overlap_diffs.items():
        flag = "PASS" if med < 0.003 else "FAIL"
        print(f"{coin}: median={med*100:.4f}%  n_overlap_days={n}  [{flag}, bar < 0.3%]")


if __name__ == "__main__":
    main()
