"""Cell 1,760 fetch: Alpaca crypto hourly bars BTC/ETH/SOL 2021-01-01 -> 2023-12-31 (TEST never fetched).

Writes research/signal_search/1760_cache/bars.parquet atomically. Keys come from .env, never printed.
"""
import os, sys, time
import requests
import pandas as pd
from dotenv import load_dotenv

URL = "https://data.alpaca.markets/v1beta3/crypto/us/bars"
OUT = "/home/ec2-user/onemil/research/signal_search/1760_cache/bars.parquet"
SYMS = ["BTC/USD", "ETH/USD", "SOL/USD"]
START, END = "2021-01-01T00:00:00Z", "2023-12-31T23:59:59Z"


def get_pairs():
    """Return candidate (key_id, secret) pairs from env, in preference order."""
    load_dotenv('/home/ec2-user/onemil/.env')
    pairs = []
    for k, s in (("ALPACA_HOD_API_KEY", "ALPACA_HOD_API_SECRET"), ("ALPACA_API_KEY", "ALPACA_API_SECRET")):
        if os.getenv(k) and os.getenv(s):
            pairs.append((k, os.getenv(k), os.getenv(s)))
    if not pairs:
        raise SystemExit("ERROR: no Alpaca keys in .env")
    return pairs


def fetch_symbol(sym, hdr):
    """Fetch all hourly bars for one symbol following next_page_token; <= 2 req/s."""
    rows, token, pages = [], None, 0
    while True:
        p = dict(symbols=sym, timeframe="1Hour", start=START, end=END, limit=10000, sort="asc")
        if token:
            p["page_token"] = token
        r = requests.get(URL, params=p, headers=hdr, timeout=60)
        time.sleep(0.15)  # request latency ~0.5 s keeps the rate < 2 req/s
        if r.status_code != 200:
            return None, r.status_code
        j = r.json()
        rows += j.get("bars", {}).get(sym, [])
        pages += 1
        token = j.get("next_page_token")
        print(f"{sym} page {pages} rows so far {len(rows)}", flush=True)
        if not token:
            return rows, 200


def ckpt_path(sym):
    return OUT.replace("bars.parquet", f"raw_{sym.split('/')[0]}.json")


def load_ckpt(sym):
    """Resume: return (rows, 200) if this symbol was already fetched, else None."""
    import json
    p = ckpt_path(sym)
    if os.path.exists(p):
        print(f"resume {sym} from checkpoint", flush=True)
        return json.load(open(p)), 200
    return None


def save_ckpt(sym, rows):
    """Atomically store one symbol's raw rows."""
    import json
    tmp = ckpt_path(sym) + ".tmp"
    json.dump(rows, open(tmp, "w"))
    os.replace(tmp, ckpt_path(sym))


def main():
    pairs = get_pairs()
    frames = []
    hdr = None
    for name, k, s in pairs:
        h = {"APCA-API-KEY-ID": k, "APCA-API-SECRET-KEY": s}
        rows, code = load_ckpt(SYMS[0]) or fetch_symbol(SYMS[0], h)
        if rows is not None:
            save_ckpt(SYMS[0], rows)
            hdr = h
            print(f"keys ok: {name}", flush=True)
            frames.append((SYMS[0], rows))
            break
        print(f"WARNING: {name} rejected, HTTP {code}", flush=True)
    if hdr is None:
        raise SystemExit("ERROR: all key pairs rejected by crypto endpoint")
    for sym in SYMS[1:]:
        rows, code = load_ckpt(sym) or fetch_symbol(sym, hdr)
        if rows is None:
            raise SystemExit(f"ERROR: {sym} HTTP {code}")
        save_ckpt(sym, rows)
        frames.append((sym, rows))
    dfs = []
    for sym, rows in frames:
        d = pd.DataFrame(rows).rename(columns={"t": "ts", "o": "open", "h": "high", "l": "low", "c": "close",
                                               "v": "volume", "n": "trade_count", "vw": "vwap"})
        d["ts"] = pd.to_datetime(d["ts"], utc=True)
        d["symbol"] = sym.split("/")[0]
        dfs.append(d)
    df = pd.concat(dfs).sort_values(["symbol", "ts"]).reset_index(drop=True)
    assert df.ts.max() < pd.Timestamp("2024-01-01", tz="UTC"), "TEST data leaked"
    tmp = OUT + ".tmp"
    df.to_parquet(tmp)
    os.replace(tmp, OUT)
    print(df.groupby("symbol").agg(n=("ts", "size"), first=("ts", "min"), last=("ts", "max")), flush=True)


if __name__ == "__main__":
    main()
