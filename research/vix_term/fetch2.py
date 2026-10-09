"""PREREG_2 fetch+rebuild: CBOE per-contract VX monthly settlements -> SPVXSP-style constant-30d index.
Writes data/vx/VX_<expiry>.csv (atomic), index_rebuild.csv, tracking.csv. Verbose; every fallback logs WARNING."""
import logging, os, sys, urllib.request, urllib.error, time
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(ROOT, 'data'); VX = os.path.join(DATA, 'vx')
URL = 'https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/VX/VX_{d}.csv'
log = logging.getLogger('vix_term.fetch2')
FEE = {'VXX': 0.0089, 'SVXY': 0.0095}
MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def std_expiry(y, m):
    """Standard VX monthly expiry for contract month (y, m): Wednesday 30 days before the 3rd Friday of month m+1."""
    y2, m2 = (y + (m == 12), m % 12 + 1)
    d = date(y2, m2, 1); fr = [d + timedelta(i) for i in range(31) if (d + timedelta(i)).month == m2 and (d + timedelta(i)).weekday() == 4]
    return fr[2] - timedelta(30)


def get(url):
    """HTTP GET text; returns None on 403/404, retries 3x on other errors (WARNING)."""
    for k in range(3):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 research'}), timeout=60).read().decode()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404): return None
            log.warning("HTTP %s %s retry %d", e.code, url, k)
        except Exception as e:                                  # noqa: BLE001
            log.warning("fetch error %s %s retry %d", e, url, k)
        time.sleep(2)
    return None


def fetch_contract(ym):
    """Fetch the monthly contract (y,m): try the standard expiry then up to 3 days earlier (holiday shift); validate the
    'Futures' label month. Returns (ym, expiry date or None, n_rows)."""
    y, m = ym; e0 = std_expiry(y, m)
    for back in range(4):
        e = e0 - timedelta(back); p = os.path.join(VX, f'VX_{e}.csv')
        txt = open(p).read() if os.path.exists(p) else get(URL.format(d=e))
        if not txt or not txt.startswith('Trade Date'): continue
        lab = txt.split('\n')[1].split(',')[1] if len(txt.split('\n')) > 1 else ''
        if f'({MON[m-1]} {y})' not in lab:
            log.warning("label mismatch %s expiry %s: %r", ym, e, lab); continue
        if not os.path.exists(p):
            open(p + '.tmp', 'w').write(txt); os.replace(p + '.tmp', p)
        return ym, e, txt.count('\n')
    log.error("NO FILE for contract %s (std expiry %s)", ym, e0)
    return ym, None, 0


ARCH = 'https://cdn.cboe.com/resources/futures/archive/volume-and-price/CFE_{c}{yy}_VX.csv'
CODE = 'FGHJKMNQUVXZ'


def fetch_archive(ym):
    """2011-2012 contracts live only in the CBOE archive (CFE_<code><yy>_VX.csv, MM/DD/YYYY). Normalise to the new format,
    expiry = last row with Settle > 0 (validated within 0-3 days before the standard expiry). Returns (ym, expiry, n_rows)."""
    y, m = ym; txt = get(ARCH.format(c=CODE[m - 1], yy=str(y)[2:]))
    if not txt or not txt.startswith('Trade Date'):
        log.error("NO ARCHIVE FILE for %s", ym); return ym, None, 0
    d = pd.read_csv(__import__('io').StringIO(txt)); d['Trade Date'] = pd.to_datetime(d['Trade Date'], format='%m/%d/%Y')
    live = d[d['Settle'] > 0]
    if live.empty: log.error("archive %s has no settles", ym); return ym, None, 0
    e = live['Trade Date'].max().date(); e0 = std_expiry(y, m)
    if not (0 <= (e0 - e).days <= 3): log.warning("archive %s last settle %s vs std expiry %s", ym, e, e0)
    d = d[d['Trade Date'] <= pd.Timestamp(e)]; d['Trade Date'] = d['Trade Date'].dt.strftime('%Y-%m-%d')
    p = os.path.join(VX, f'VX_{e}.csv'); d.to_csv(p + '.tmp', index=False); os.replace(p + '.tmp', p)
    return ym, e, len(d)


def fetch_both(ym):
    """<=2012: archive only. >=2013: new-site file, then (if the archive has the contract) union the archive's earlier rows
    (new-site files for 2013 contracts start 2013-01) -- new-site rows win where Settle > 0."""
    if ym[0] <= 2012: return fetch_archive(ym)
    r = fetch_contract(ym)
    if r[1] is None or ym[0] > 2014: return r
    y, m = ym; txt = get(ARCH.format(c=CODE[m - 1], yy=str(y)[2:]))
    if not txt or not txt.startswith('Trade Date'):
        log.warning("no archive for %s (new-site history may be short)", ym); return r
    a = pd.read_csv(__import__('io').StringIO(txt)); a['Trade Date'] = pd.to_datetime(a['Trade Date'], format='%m/%d/%Y')
    p = os.path.join(VX, f'VX_{r[1]}.csv'); n = pd.read_csv(p, parse_dates=['Trade Date'])
    keep = n[n['Settle'] > 0]; fill = a[(a['Settle'] > 0) & ~a['Trade Date'].isin(keep['Trade Date']) & (a['Trade Date'] <= pd.Timestamp(r[1]))]
    if len(fill):
        u = pd.concat([n, fill[n.columns]]).sort_values('Trade Date').drop_duplicates('Trade Date', keep='last')
        # rows in n with Settle 0 are superseded by archive rows with Settle>0 (concat order: fill last wins)
        u['Trade Date'] = u['Trade Date'].dt.strftime('%Y-%m-%d'); u.to_csv(p + '.tmp', index=False); os.replace(p + '.tmp', p)
        log.info("merged %s: +%d archive rows with Settle>0", ym, len(fill))
    return r


def load_contracts():
    """Dict expiry -> settle Series (Settle, fallback Close where Settle == 0, counted)."""
    out, nfb, ntot = {}, {}, 0
    for f in sorted(os.listdir(VX)):
        if not f.endswith('.csv'): continue
        e = pd.Timestamp(f[3:-4]); d = pd.read_csv(os.path.join(VX, f), parse_dates=['Trade Date']).set_index('Trade Date')
        s = d['Settle'].astype(float).copy(); z = (s <= 0)
        s[z] = d.loc[z, 'Close'].astype(float); s[s <= 0] = np.nan
        nfb[e.year] = nfb.get(e.year, 0) + int(z.sum()); ntot += len(d); out[e] = s.dropna()
    log.warning("settle==0 -> close fallback rows by contract-expiry year: %s (of %d rows)", {k: v for k, v in nfb.items() if v}, ntot)
    return out


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    os.makedirs(VX, exist_ok=True)
    yms = [(y, m) for y in range(2010, 2028) for m in range(1, 13) if (2010, 11) < (y, m) <= (2026, 12)]
    t0 = time.time()
    with ThreadPoolExecutor(4) as ex: res = list(ex.map(fetch_both, yms))
    got = [r for r in res if r[1]]; lost = [r[0] for r in res if not r[1] and r[0] <= (2026, 12)]
    log.info("contracts fetched %d / %d in %.0fs; no file for: %s", len(got), len(yms), time.time() - t0, lost)


if __name__ == '__main__':
    main()
