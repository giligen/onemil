"""PREREG 1,774 quote fetch: last NBBO at/before 10:00/10:30/11:00 ET per symbol-day (free Alpaca sip quotes).

Resumable: per-day parquet in 1774_cache/quotes/<day>.parquet (atomic). Keys from .env, never printed.
Window [t-30s, t] ET, paginated until the instant is passed; only the last quote is kept (plus count).
"""
import os, sys, time
import pandas as pd, requests
from dotenv import load_dotenv

ROOT = "/home/ec2-user/onemil"; HERE = f"{ROOT}/research/signal_search"
OUT = f"{HERE}/1774_cache/quotes"; os.makedirs(OUT, exist_ok=True)
WINS = {"1000": (10, 0), "1030": (10, 30), "1100": (11, 0)}
GAPMAX, CAPSZ, MIN_GAP_S = 200.0, 10, 60.0 / 190


def log(*a):
    """Verbose progress line, flushed."""
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def target_rows():
    """Executable rows: gap_ok, r_gross known, top-10/day by gap rank (declared)."""
    e = pd.read_parquet(f"{HERE}/1773_events.parquet")
    x = e[e.gap_ok & e.r_gross.notna()].sort_values(["day", "gap_pct"], ascending=[True, False])
    return x.groupby("day").head(CAPSZ)[["day", "symbol"]]


def get_last(sess, sym, day, h, m):
    """Last quote at or before day h:m:00 ET, window 30 s. Returns dict."""
    ny = pd.Timestamp(f"{day} {h:02d}:{m:02d}:00", tz="America/New_York")
    start = (ny - pd.Timedelta(seconds=30)).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    end = ny.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    tok, last, n = None, None, 0
    for _ in range(20):
        p = dict(start=start, end=end, limit=1000, feed="sip")
        if tok:
            p["page_token"] = tok
        for att in range(6):
            time.sleep(MIN_GAP_S)
            r = sess.get(f"https://data.alpaca.markets/v2/stocks/{sym}/quotes", params=p, timeout=60)
            if r.status_code == 200:
                break
            log("WARN http", r.status_code, sym, day, r.text[:80].replace("\n", " "))
            time.sleep(5 * (att + 1))
        else:
            return dict(err=1)
        j = r.json(); qs = j.get("quotes") or []
        n += len(qs)
        if qs:
            last = qs[-1]
        tok = j.get("next_page_token")
        if not tok:
            break
    if last is None:
        return dict(n=0)
    return dict(n=n, ts=last["t"], bid=last["bp"], ask=last["ap"], bsz=last["bs"], asz=last["as"])


def main():
    """Fetch all days, skipping cached ones."""
    load_dotenv(f"{ROOT}/.env")
    s = requests.Session()
    s.headers.update({"APCA-API-KEY-ID": os.environ["ALPACA_HOD_API_KEY"], "APCA-API-SECRET-KEY": os.environ["ALPACA_HOD_API_SECRET"]})
    t = target_rows(); log("symbol-days", len(t), "days", t.day.nunique(), "requests>=", 3 * len(t))
    done = 0
    for day, g in t.groupby("day"):
        f = f"{OUT}/{day}.parquet"
        if os.path.exists(f):
            done += len(g); continue
        rows = []
        for sym in g.symbol:
            for w, (h, m) in WINS.items():
                d = get_last(s, sym, day, h, m); d.update(day=day, symbol=sym, win=w); rows.append(d)
        pd.DataFrame(rows).to_parquet(f + ".tmp"); os.replace(f + ".tmp", f)
        prev = done; done += len(g)
        if done // 200 != prev // 200:
            log("progress symbol-days", done, "/", len(t), "day", day)
    log("DONE", done)


if __name__ == "__main__":
    main()
