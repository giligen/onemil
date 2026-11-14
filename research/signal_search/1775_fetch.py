"""PREREG 1,775 news fetch: Alpaca news history per symbol-day, window [prev business day 16:00 ET, day 09:30 ET].

Resumable: per-day parquet in 1775_cache/news/<day>.parquet (atomic). One row per headline, plus one marker row
(id empty) per symbol-day queried successfully. Keys ALPACA_MOM_* from .env, never printed. <=190 req/min.
"""
import os, sys, time
import pandas as pd, requests
from dotenv import load_dotenv

ROOT = "/home/ec2-user/onemil"; HERE = f"{ROOT}/research/signal_search"
OUT = f"{HERE}/1775_cache/news"; os.makedirs(OUT, exist_ok=True)
GAP = 60.0 / 190
URL = "https://data.alpaca.markets/v1beta1/news"


def log(*a):
    """Verbose progress line, flushed."""
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def window(day):
    """UTC ISO start/end of the pre-market window for `day`."""
    d = pd.Timestamp(day)
    prev = (d - pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    s = pd.Timestamp(f"{prev} 16:00:00", tz="America/New_York").tz_convert("UTC")
    e = pd.Timestamp(f"{day} 09:30:00", tz="America/New_York").tz_convert("UTC")
    return s.strftime("%Y-%m-%dT%H:%M:%SZ"), e.strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_one(sess, sym, day):
    """All headlines for sym in the window; None on persistent failure."""
    s, e = window(day); rows = []; tok = None
    for _ in range(10):
        p = dict(symbols=sym, start=s, end=e, limit=50, include_content="false", sort="asc")
        if tok:
            p["page_token"] = tok
        for att in range(6):
            time.sleep(GAP)
            try:
                r = sess.get(URL, params=p, timeout=60)
            except requests.RequestException as ex:
                log("WARN net", type(ex).__name__, sym, day); time.sleep(5 * (att + 1)); continue
            if r.status_code == 200:
                break
            log("WARN http", r.status_code, sym, day, r.text[:80].replace("\n", " "))
            time.sleep(5 * (att + 1))
        else:
            return None
        j = r.json()
        for a in j.get("news") or []:
            rows.append(dict(day=day, symbol=sym, id=str(a.get("id")), created_at=a.get("created_at"),
                             headline=a.get("headline"), source=a.get("source"), author=a.get("author")))
        tok = j.get("next_page_token")
        if not tok:
            break
    return rows


def main():
    """Fetch every symbol-day with a known r_gross."""
    load_dotenv(f"{ROOT}/.env", override=True)
    k, sec = os.environ["ALPACA_MOM_API_KEY"], os.environ["ALPACA_MOM_API_SECRET"]
    sess = requests.Session(); sess.headers.update({"APCA-API-KEY-ID": k, "APCA-API-SECRET-KEY": sec})
    e = pd.read_parquet(f"{HERE}/1773_events.parquet")
    e = e[e.gap_ok & e.r_gross.notna()][["day", "symbol"]]
    days = sorted(e.day.unique()); log("symbol-days", len(e), "days", len(days))
    done = 0
    for day in days:
        f = f"{OUT}/{day}.parquet"
        if os.path.exists(f):
            continue
        out = []; fail = 0
        for sym in e.loc[e.day == day, "symbol"]:
            r = fetch_one(sess, sym, day)
            if r is None:
                fail += 1; out.append(dict(day=day, symbol=sym, id="FAIL")); continue
            out.append(dict(day=day, symbol=sym, id=""))
            out += r
        pd.DataFrame(out).to_parquet(f + ".tmp"); os.replace(f + ".tmp", f)
        done += 1; log("day", day, "rows", len(out), "fail", fail, "days_done", done, "/", len(days))
    log("DONE")


if __name__ == "__main__":
    main()
