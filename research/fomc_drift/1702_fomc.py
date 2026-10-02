"""Cell 1,702 -- pre-FOMC drift as a stacking sleeve. Executes PREREG_1702.md exactly (10 cells).

Stages (all in one process, chunk caches make it resumable):
  1. FOMC dates parsed from the Federal Reserve pages saved in raw/ (calendar page 2021+, per-year historical pages <=2020).
  2. SPY/QQQ/IWM daily + 1-minute bars from Alpaca SIP, adjustment=ALL, chunked by quarter -> minute_bars.parquet.
  3. Windows W1/W2/W3 (intraday, 2016-2026) and W4 (SPY daily via yfinance, 1994-2026), stats, placebo, pass rule.
Run: bash scripts/research_run.sh -m 2000M python3 research/fomc_drift/1702_fomc.py
"""
import os, re, sys, math, time, json
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
CH = os.path.join(RAW, "chunks")
os.makedirs(CH, exist_ok=True)
SYMS = ["SPY", "QQQ", "IWM"]
COST_BPS = 2.0  # 1 bp per side
TODAY = date(2026, 10, 2)
CAL_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
HIST_URL = "https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm"
MONTHS = {m: i + 1 for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                           "september", "october", "november", "december"])}
MONTHS.update({m[:3]: v for m, v in list(MONTHS.items())})


def log(*a):
    """Progress line, flushed."""
    print(datetime.utcnow().strftime("%H:%M:%S"), *a, flush=True)


def strip(s):
    """Remove tags and collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s)).strip()


def mnum(tok):
    """Month token -> 1..12 (first 3 letters)."""
    return MONTHS[tok.lower()[:3]] if tok.lower()[:3] in MONTHS else None


def last_day(year, month_txt, days_txt):
    """Announcement (last) day of a meeting from 'April/May' + '30-1' style text, or 'June 30-July 1' style."""
    m = re.match(r"^(\d+)(?:\s*-\s*(?:([A-Za-z]+)\s+)?(\d+))?", days_txt)
    d1, mon2, d2 = m.group(1), m.group(2), m.group(3)
    mt = month_txt.split("/")
    if d2 is None:
        return date(year, mnum(mt[-1]), int(d1)) if len(mt) == 1 else date(year, mnum(mt[0]), int(d1))
    if mon2:
        return date(year, mnum(mon2), int(d2))
    return date(year, mnum(mt[-1]), int(d2))


def parse_dates():
    """Build the full meeting list (scheduled and excluded) from the saved Fed pages."""
    rows = []
    for y in range(1994, 2021):
        t = open(f"{RAW}/h{y}.html").read()
        panels = t.split("<h5")[1:]
        for p in panels:
            head = strip(p.split("</h5>")[0].split(">", 1)[-1])
            body = p.split("</h5>", 1)[1]
            body = re.split(r'<div class="panel panel-default"', body)[0]
            h = head.replace(f" - {y}", "")
            m = re.match(r"^([A-Za-z/]+)\s+(\d+(?:-(?:[A-Za-z]+ )?\d+)?)\s*(.*)$", h)
            if not m:
                log("WARNING unparsed heading", y, head)
                continue
            mon, days, rest = m.groups()
            d = last_day(y, mon, days)
            low = rest.lower()
            sched, why = 1, ""
            if "conference call" in low:
                sched, why = 0, "conference call"
            elif "unscheduled" in low:
                sched, why = 0, "unscheduled"
            elif "cancelled" in low:
                sched, why = 0, "cancelled meeting"
            elif "notation vote" in low:
                sched, why = 0, "notation vote"
            elif "meeting" not in low:
                sched, why = 0, "not a meeting: " + rest
            pc = int(bool(re.search(r"press conference", body, re.I)))
            rows.append(dict(date=d, scheduled=sched, press_conf=pc, excluded_reason=why, source_url=HIST_URL.format(y=y), heading=head))
    t = open(f"{RAW}/cal.html").read()
    for y in range(2021, 2028):
        i = t.find(f"{y} FOMC Meetings")
        if i < 0:
            continue
        j = t.find(" FOMC Meetings", i + 20)
        blk = t[i:j if j > 0 else len(t)]
        for mt in re.finditer(r'fomc-meeting__month[^>]*><strong>([^<]+)</strong></div>\s*<div[^>]*fomc-meeting__date[^>]*>([^<]+)</div>(.*?)(?=fomc-meeting__month|$)', blk, re.S):
            mon, days, rest = mt.group(1), mt.group(2).replace("*", "").strip(), mt.group(3)
            d = last_day(y, mon, days)
            low = (days + " " + mon).lower()
            sched, why = 1, ""
            for kw in ("notation vote", "unscheduled", "cancelled", "conference call"):
                if kw in low:
                    sched, why = 0, kw
            link = re.search(r"monetary(\d{8})a", rest)
            if link and link.group(1) != d.strftime("%Y%m%d"):
                log("WARNING statement-link date mismatch", d, link.group(1))
            rows.append(dict(date=d, scheduled=sched, press_conf=int(bool(re.search("Press Conference", rest))), excluded_reason=why,
                             source_url=CAL_URL, heading=f"{mon} {days} {y}"))
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    # a panel the day before another scheduled panel is the pre-meeting (agenda-only) panel of the same meeting
    for i in range(len(df) - 1):
        if df.loc[i, "scheduled"] == 1 and df.loc[i + 1, "scheduled"] == 1 and (df.loc[i + 1, "date"] - df.loc[i, "date"]).days == 1:
            df.loc[i, "scheduled"], df.loc[i, "excluded_reason"] = 0, "pre-meeting panel of the next-day meeting (no announcement)"
            log("excluded pre-meeting panel", df.loc[i, "date"])
    df["time_et"] = np.where(df["date"] < date(2013, 1, 1), "14:15", "14:00")
    df["time_note"] = "era assumption (Fed pages carry no release time); irrelevant to 2016+ windows (all 14:00)"
    df["future"] = df["date"] > TODAY
    df.loc[df["future"], "scheduled"] = df.loc[df["future"], "scheduled"]
    df.to_csv(f"{HERE}/fomc_dates.csv", index=False)
    return df


def client():
    """Alpaca data client from .env (keys never printed)."""
    from dotenv import load_dotenv
    load_dotenv("/home/ec2-user/onemil/.env")
    from alpaca.data.historical import StockHistoricalDataClient
    k, s = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET")
    if not k or not s:
        raise SystemExit("ERROR missing ALPACA keys")
    return StockHistoricalDataClient(k, s)


def fetch_daily(cl):
    """Daily bars (adjustment=ALL) 2016-2026 for the three ETFs."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed, Adjustment
    f = f"{CH}/daily.parquet"
    if os.path.exists(f):
        return pd.read_parquet(f)
    out = []
    for s in SYMS:
        r = StockBarsRequest(symbol_or_symbols=s, timeframe=TimeFrame.Day, start=datetime(2015, 12, 20), end=datetime(2026, 10, 2),
                             feed=DataFeed.SIP, adjustment=Adjustment.ALL)
        d = cl.get_stock_bars(r).df.reset_index()
        d["date"] = pd.to_datetime(d["timestamp"]).dt.tz_convert("America/New_York").dt.date
        d["symbol"] = s
        out.append(d[["symbol", "date", "open", "close"]])
        log("daily", s, len(d))
    df = pd.concat(out)
    df.to_parquet(f)
    return df


def fetch_minutes(cl, need_full):
    """Quarterly minute chunks; keep 09:30-14:05 + 15:55-16:00 on need_full days, else only the 09:30/13:55/14:00 bars."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import DataFeed, Adjustment
    parts, lost = [], []
    for s in SYMS:
        for y in range(2016, 2027):
            for q in range(4):
                a = datetime(y, 3 * q + 1, 1)
                b = datetime(y + (q == 3), (3 * q + 4 - 1) % 12 + 1, 1)
                if a > datetime(2026, 10, 2):
                    continue
                f = f"{CH}/{s}_{y}Q{q+1}.parquet"
                if os.path.exists(f):
                    parts.append(pd.read_parquet(f))
                    continue
                for attempt in range(4):
                    try:
                        r = StockBarsRequest(symbol_or_symbols=s, timeframe=TimeFrame(1, TimeFrameUnit.Minute), start=a, end=min(b, datetime(2026, 10, 2)),
                                             feed=DataFeed.SIP, adjustment=Adjustment.ALL)
                        d = cl.get_stock_bars(r).df
                        break
                    except Exception as e:
                        log("WARNING fetch retry", s, y, q, attempt, repr(e)[:120])
                        time.sleep(5 * (attempt + 1))
                else:
                    log("ERROR LOST chunk", s, y, q)
                    lost.append((s, y, q))
                    continue
                d = d.reset_index()
                ts = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert("America/New_York")
                d["date"] = ts.dt.date
                mins = ts.dt.hour * 60 + ts.dt.minute
                keep3 = mins.isin([570, 835, 840])
                keepfull = d["date"].isin(need_full) & (((mins >= 570) & (mins <= 845)) | (mins >= 955))
                d = d[keep3 | keepfull].copy()
                d["mn"] = mins[d.index]
                d["symbol"] = s
                d = d[["symbol", "date", "mn", "open", "high", "low", "close", "volume"]]
                d.to_parquet(f)
                parts.append(d)
                log("chunk", s, y, q, len(d))
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(f"{HERE}/minute_bars.parquet")
    return df, lost


def tstat(x):
    """t of the mean."""
    x = np.asarray(x, float)
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))) if len(x) > 2 and x.std(ddof=1) > 0 else float("nan")


def welch(a, b):
    """Welch t of mean(a)-mean(b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se)


def cell_stats(ev, pl_all, pl_wd, half_cut):
    """ev: DataFrame(date, wd, net); pl_*: placebo net arrays. Returns dict of reads incl. pass flags."""
    x = ev["net"].values
    n = len(x)
    srt = np.sort(x)
    k = max(1, math.ceil(0.05 * n))
    sd = x.std(ddof=1)
    h1 = ev.loc[ev["date"].map(lambda d: d.year) < half_cut, "net"]
    h2 = ev.loc[ev["date"].map(lambda d: d.year) >= half_cut, "net"]
    wdm = ev["wd"].map(pl_wd).values
    d_wd = x - wdm
    r = dict(n=n, mean=x.mean(), t=tstat(x), pos=(x > 0).mean(), h1=h1.mean(), h1n=len(h1), h1t=tstat(h1), h2=h2.mean(), h2n=len(h2),
             h2t=tstat(h2), ex_top5=srt[:-k].mean(), ex_worst5=srt[k:].mean(), worst=x.min(), best=x.max(),
             plc=float(np.mean(pl_all)), diff=x.mean() - float(np.mean(pl_all)), diff_t=welch(x, pl_all),
             plc_wd=float(np.mean(wdm)), diff_wd=d_wd.mean(), diff_wd_t=tstat(d_wd), sd=sd, mde=2.8 * sd / math.sqrt(n))
    r["pass"] = bool(r["mean"] >= 15 and r["t"] >= 2 and r["h1"] > 0 and r["h2"] > 0 and r["ex_top5"] > 0 and min(r["diff"], r["diff_wd"]) >= 10 and r["pos"] >= 0.55)
    return r


def main():
    dates = parse_dates()
    sch = dates[dates["scheduled"] == 1]
    log("dates parsed", len(dates), "scheduled", len(sch))
    cov = sch.groupby(sch["date"].map(lambda d: d.year)).size()
    log("coverage per year", cov.to_dict())
    ev_dates = sorted(d for d in sch["date"] if d <= TODAY)
    cl = client()
    dly = fetch_daily(cl)
    spy_d = dly[dly.symbol == "SPY"].sort_values("date")
    sess = list(spy_d["date"])
    prev = {sess[i]: sess[i - 1] for i in range(1, len(sess))}
    ev_intraday = [d for d in ev_dates if d >= date(2016, 1, 4) and d in prev]
    need_full = set(ev_intraday) | {prev[d] for d in ev_intraday}
    mb, lost = fetch_minutes(cl, need_full)
    log("minute rows", len(mb), "lost chunks", lost)
    # tables
    piv = {}
    for s in SYMS:
        m = mb[mb.symbol == s]
        piv[s] = {k: m[m.mn == mn].drop_duplicates("date").set_index("date")["open"] for k, mn in (("o930", 570), ("o1355", 835), ("o1400", 840))}
        piv[s]["dc"] = dly[dly.symbol == s].drop_duplicates("date").set_index("date")["close"]
    # completeness gate
    gate = {}
    for s in SYMS:
        for k in ("o930", "o1355", "o1400"):
            sub = [d for d in sess if d >= date(2016, 1, 4)]
            miss = [d for d in sub if d not in piv[s][k].index]
            gate[f"{s}.{k}"] = (len(miss), len(sub))
    log("completeness (missing, sessions)", gate)
    # prior close: daily vs last minute bar (15:59 bar close) on prior days of events
    diffs = []
    spy_m = mb[(mb.symbol == "SPY") & (mb.mn >= 955)]
    last = spy_m.sort_values("mn").groupby("date").tail(1).set_index("date")["close"]
    for d in ev_intraday:
        p = prev[d]
        if p in last.index and p in piv["SPY"]["dc"].index:
            diffs.append((last[p] / piv["SPY"]["dc"][p] - 1) * 1e4)
    log("prior close daily vs last-minute-bar (bps): n", len(diffs), "mean", np.mean(diffs), "mean abs", np.mean(np.abs(diffs)), "max abs", np.max(np.abs(diffs)))

    def wret(s, w, d):
        """Gross return of window w on session d (needs prior session)."""
        p = prev.get(d)
        try:
            if w == "W1":
                return piv[s]["o1355"][d] / piv[s]["dc"][p] - 1
            if w == "W2":
                return piv[s]["o1355"][d] / piv[s]["o1400"][p] - 1
            return piv[s]["o1355"][d] / piv[s]["o930"][d] - 1
        except KeyError:
            return np.nan

    evset = set(ev_dates)
    all_ev = set(dates["date"])  # excludes every listed meeting day (incl. unscheduled) from placebo
    plac_days = [d for d in sess if d >= date(2016, 1, 4) and d not in all_ev]
    rows, res = [], {}
    for s in SYMS:
        for w in ("W1", "W2", "W3"):
            e = [(d, wret(s, w, d)) for d in ev_intraday]
            lostn = sum(1 for _, v in e if np.isnan(v))
            ev = pd.DataFrame([(d, d.weekday(), v * 1e4 - COST_BPS) for d, v in e if not np.isnan(v)], columns=["date", "wd", "net"])
            pl = pd.DataFrame([(d, d.weekday(), wret(s, w, d) * 1e4 - COST_BPS) for d in plac_days], columns=["date", "wd", "net"]).dropna()
            pwd = pl.groupby("wd")["net"].mean().to_dict()
            r = cell_stats(ev, pl["net"].values, pwd, 2021)
            r["lost"] = lostn
            r["pl_n"] = len(pl)
            ev["pc"] = ev["date"].map(dict(zip(dates["date"], dates["press_conf"])))
            pre = ev[ev["date"].map(lambda d: d.year) < 2019]
            r["pc_yes"] = (pre[pre.pc == 1]["net"].mean(), int((pre.pc == 1).sum()))
            r["pc_no"] = (pre[pre.pc == 0]["net"].mean(), int((pre.pc == 0).sum()))
            r["byyear"] = ev.groupby(ev["date"].map(lambda d: d.year))["net"].agg(["mean", "count"]).to_dict("index")
            res[(s, w)] = r
            for _, q in ev.iterrows():
                rows.append(dict(instrument=s, window=w, date=q["date"], net_bps=q["net"], gross_bps=q["net"] + COST_BPS))
            log(s, w, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k not in ("byyear",)})
    # W4 daily
    import yfinance as yf
    y = yf.download("SPY", start="1993-01-01", end="2026-10-03", auto_adjust=True, progress=False)
    cl_ = y["Close"].squeeze()
    cl_.index = [i.date() for i in cl_.index]
    ret = (cl_ / cl_.shift(1) - 1) * 1e4 - COST_BPS
    ret = ret.dropna()
    ev4 = pd.DataFrame([(d, d.weekday(), ret[d]) for d in ev_dates if d in ret.index], columns=["date", "wd", "net"])
    lost4 = [d for d in ev_dates if d not in ret.index]
    pl4 = pd.DataFrame([(d, d.weekday(), v) for d, v in ret.items() if d not in all_ev and d >= date(1994, 2, 1)], columns=["date", "wd", "net"])
    pwd4 = pl4.groupby("wd")["net"].mean().to_dict()
    r4 = cell_stats(ev4, pl4["net"].values, pwd4, 2012)
    r4["lost"] = len(lost4)
    r4["byyear"] = {}
    for nm, lo, hi in (("1994-2011", 1994, 2011), ("2012-2026", 2012, 2026)):
        e = ev4[ev4.date.map(lambda d: lo <= d.year <= hi)]
        p = pl4[pl4.date.map(lambda d: lo <= d.year <= hi)]
        pw = p.groupby("wd")["net"].mean().to_dict()
        dwd = e["net"].values - e["wd"].map(pw).values
        r4[nm] = dict(n=len(e), mean=e.net.mean(), t=tstat(e.net), pos=(e.net > 0).mean(), plc=p.net.mean(), diff=e.net.mean() - p.net.mean(),
                      diff_t=welch(e.net.values, p.net.values), diff_wd=dwd.mean(), diff_wd_t=tstat(dwd), sd=e.net.std(ddof=1), mde=2.8 * e.net.std(ddof=1) / math.sqrt(len(e)))
    res[("SPY", "W4")] = r4
    log("W4", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r4.items()})
    for _, q in ev4.iterrows():
        rows.append(dict(instrument="SPY", window="W4", date=q["date"], net_bps=q["net"], gross_bps=q["net"] + COST_BPS))
    # alpaca vs yfinance daily on event days
    ad = piv["SPY"]["dc"]
    cmp = []
    for d in ev_intraday:
        p = prev[d]
        if d in ad.index and p in ad.index and d in ret.index:
            cmp.append(((ad[d] / ad[p] - 1) * 1e4 - COST_BPS) - ret[d])
    log("alpaca vs yfinance event-day close-to-close diff bps: n", len(cmp), "max abs", np.max(np.abs(cmp)), "mean abs", np.mean(np.abs(cmp)))
    pd.DataFrame(rows).to_csv(f"{HERE}/1702_events.csv", index=False)
    # SPY W4 by half N-years math
    meta = dict(cov=cov.to_dict(), gate=gate, lost=lost, prior_close=(len(diffs), float(np.mean(diffs)), float(np.mean(np.abs(diffs))), float(np.max(np.abs(diffs)))),
                ayf=(len(cmp), float(np.max(np.abs(cmp))), float(np.mean(np.abs(cmp)))), lost4=[str(d) for d in lost4],
                excluded=[(str(r.date), r.excluded_reason) for r in dates.itertuples() if r.scheduled == 0],
                n_events=len(ev_dates), nfuture=int(dates["future"].sum()))
    write_md(res, meta)


def f(v, d=1):
    """Format helper."""
    return "nan" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{d}f}"


def write_md(res, meta):
    """RESULT_1702.md from computed numbers (caveats appended by hand after reading)."""
    L = ["# RESULT 1,702 -- pre-FOMC drift (PREREG_1702.md, FROZEN; 10 cells; net of 2 bps round trip; bps)", ""]
    cov = meta["cov"]
    L.append("Coverage (scheduled meetings/yr, Fed pages): " + " ".join(f"{y}:{n}" for y, n in cov.items()))
    L.append(f"Years != 8: {[y for y, n in cov.items() if n != 8]}; events analysed (<= {TODAY}): {meta['n_events']}; future-dated rows in fomc_dates.csv: {meta['nfuture']}")
    L.append("Excluded (listed in fomc_dates.csv): " + "; ".join(f"{d} {w}" for d, w in meta["excluded"] if d >= '2008-01-01' or True)[:900])
    g = meta["gate"]
    L.append(f"Bars completeness (missing/sessions): " + ", ".join(f"{k} {a}/{b}" for k, (a, b) in g.items()) + f"; LOST chunks {meta['lost']}")
    pc = meta["prior_close"]
    L.append(f"Prior close = official daily close (Alpaca daily, adj=ALL); vs last 15:59 minute close: n {pc[0]}, mean {f(pc[1],2)} bps, mean|.| {f(pc[2],2)}, max|.| {f(pc[3],2)}. "
             f"Alpaca vs yfinance event-day close-to-close: n {meta['ayf'][0]}, max|diff| {f(meta['ayf'][1],2)} bps.")
    L += ["", "| cell | n | mean | t | %pos | H1 | H2 | ex-top5 | ex-worst5 | worst | plc | diff (t) | diff same-wd (t) | MDE | PASS |", "|" + "---|" * 15]
    for k, r in res.items():
        h = ("16-20", "21-26") if k[1] != "W4" else ("94-11", "12-26")
        L.append(f"| {k[0]} {k[1]} | {r['n']} | {f(r['mean'])} | {f(r['t'],2)} | {f(100*r['pos'],0)} | {f(r['h1'])} ({h[0]}) | {f(r['h2'])} ({h[1]}) | {f(r['ex_top5'])} | {f(r['ex_worst5'])} | {f(r['worst'],0)} | {f(r['plc'])} | {f(r['diff'])} ({f(r['diff_t'],2)}) | {f(r['diff_wd'])} ({f(r['diff_wd_t'],2)}) | {f(r['mde'])} | {'PASS' if r['pass'] else 'fail'} |")
    r4 = res[("SPY", "W4")]
    L += ["", "W4 SPY close-to-close by era (net bps): "]
    for nm in ("1994-2011", "2012-2026"):
        e = r4[nm]
        L.append(f"- {nm}: n {e['n']}, mean {f(e['mean'])}, t {f(e['t'],2)}, %pos {f(100*e['pos'],0)}, placebo {f(e['plc'])}, diff {f(e['diff'])} (t {f(e['diff_t'],2)}), same-weekday diff {f(e['diff_wd'])} (t {f(e['diff_wd_t'],2)}), MDE {f(e['mde'])}")
    for w in ("W1", "W2"):
        r = res[("SPY", w)]
        L += ["", f"SPY {w} by year (mean bps, n): " + "; ".join(f"{y} {f(v['mean'],0)} ({int(v['count'])})" for y, v in r["byyear"].items())]
    L += ["", "Press conference vs not, events < 2019 (mean bps, n):"]
    for k, r in res.items():
        if k[1] != "W4":
            L.append(f"- {k[0]} {k[1]}: PC {f(r['pc_yes'][0])} (n {r['pc_yes'][1]}), no-PC {f(r['pc_no'][0])} (n {r['pc_no'][1]})")
    spy_pass = res[("SPY", "W1")]["pass"] or res[("SPY", "W2")]["pass"]
    e2 = r4["2012-2026"]
    w4ok = e2["mean"] > 0 and e2["t"] >= 1.5
    npass = sum(r["pass"] for r in res.values())
    L += ["", f"Cells passing: {npass} of 10. SPY passes W1 or W2: {spy_pass}. W4 2012-26 mean>0 and t>=1.5: {w4ok}.",
          f"VERDICT: {'SLEEVE RECOMMENDED (independent rebuild still required before any paper flag)' if spy_pass and w4ok else 'NOT RECOMMENDED'}"]
    best = max((r for k, r in res.items() if k[0] == "SPY" and k[1] in ("W1", "W2")), key=lambda r: r["mean"])
    if best["mean"] > 0:
        L.append(f"Best SPY point estimate {f(best['mean'])} bps, sd {f(best['sd'])}: years to resolve at 2.8 SE with 8 events/yr = {f((2.8*best['sd']/best['mean'])**2/8,0)}")
    open(f"{HERE}/RESULT_1702.md", "w").write("\n".join(L) + "\n")
    log("wrote RESULT_1702.md")


if __name__ == "__main__":
    main()
