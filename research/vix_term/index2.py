"""PREREG_2 index rebuild + tracking validation. Reads data/vx/*.csv (fetch2.py), data/cboe.csv, Alpaca daily bars.
Writes index_rebuild.csv (date, front, second, weight, level, ret), tracking.csv, prints the validation. Verbose."""
import logging, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch2, run
log = logging.getLogger('vix_term.index2')
ROOT, DATA = fetch2.ROOT, fetch2.DATA
LEV_CHANGE = pd.Timestamp('2018-02-27')


def nyse_cal():
    """NYSE-day calendar 2010-12-01..2026-10-08: CBOE VIX rows, 2016+ restricted to SPY bars (as run.load)."""
    c = pd.read_csv(f'{DATA}/cboe.csv', index_col=0, parse_dates=True).loc['2010-12-01':'2026-10-08']
    spy = pd.read_csv(f'{DATA}/SPY_daily.csv', index_col=0, parse_dates=True).index
    return c[(c.index < spy.min()) | c.index.isin(spy)]


def build_index(contracts, cal):
    """Constant-30d SPVXSP-style excess-return index. Weights set at close t-1 (w_front = business days t..E1 / business days
    E0+1..E1) apply to the t-1 -> t return of the then-front/second contracts. NaN return days are carried flat (0) and
    counted (completeness)."""
    exps = sorted(contracts); idx = cal.index
    hol = np.array([d for d in pd.bdate_range(idx.min(), idx.max()) if d not in set(idx)], dtype='datetime64[D]')
    bd = lambda a, b: int(np.busday_count(np.datetime64((a + pd.Timedelta(days=1)).date()), np.datetime64((b + pd.Timedelta(days=1)).date()), holidays=hol))
    rows, prev, lvl = [], None, 100.0
    for t in idx:
        if prev is None:
            rows.append((t, pd.NaT, pd.NaT, np.nan, np.nan, lvl)); prev = t; continue
        nxt = [e for e in exps if e > prev]
        e1, e2 = nxt[0], nxt[1]; e0 = exps[exps.index(e1) - 1]
        w1 = bd(prev, e1) / bd(e0, e1)
        try:
            r = w1 * contracts[e1][t] / contracts[e1][prev] + (1 - w1) * contracts[e2][t] / contracts[e2][prev] - 1
        except KeyError:
            log.warning("missing settle %s (front %s second %s)", t.date(), e1.date(), e2.date()); r = np.nan
        if r == r: lvl *= 1 + r
        rows.append((t, e1, e2, w1, r, lvl)); prev = t
    return pd.DataFrame(rows, columns=['date', 'front', 'second', 'weight', 'ret', 'level']).set_index('date')


def regress(y, x):
    """OLS y = a + b x -> dict(beta, alpha_bps, r2, n, te_bps) ; te = mean(y - x) in bps (mean daily tracking error)."""
    ok = y.notna() & x.notna(); y, x = y[ok], x[ok]
    b, a = np.polyfit(x, y, 1); r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
    return dict(beta=b, alpha_bps=a * 1e4, r2=r2, n=len(y), te_mean_bps=float((y - x).mean() * 1e4), te_sd_bps=float((y - x).std() * 1e4))


def synth_legs(ret, svxy_alt=False):
    """Daily-reset synthetic ETP returns: VXX = ret - fee/252 ; SVXY = -0.5 ret - fee/252 (svxy_alt: -1.0x through 2018-02-27)."""
    lev = pd.Series(0.5, index=ret.index)
    if svxy_alt: lev[ret.index <= LEV_CHANGE] = 1.0
    return ret - fetch2.FEE['VXX'] / 252, -lev * ret - fetch2.FEE['SVXY'] / 252


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    cal = nyse_cal(); contracts = fetch2.load_contracts()
    log.info("contracts %d, calendar %d days %s..%s", len(contracts), len(cal), cal.index.min().date(), cal.index.max().date())
    ix = build_index(contracts, cal); ix.to_csv(f'{ROOT}/index_rebuild.csv')
    r = ix['ret']; sub = ix.loc['2011-01-03':]
    log.info("index level 2011-01-03 %.2f ... 2026-10-08 %.2f ; NaN-return days in window: %d of %d", sub['level'].iloc[0], sub['level'].iloc[-1], int(sub['ret'].isna().sum()), len(sub))
    print("index returns 2018-02-02..06:", ix.loc['2018-02-02':'2018-02-07', ['front', 'second', 'weight', 'ret', 'level']].to_string())
    out = []
    for name, sym, lo in (('VXX', 'VXX', '2018-01-19'), ('SVXY', 'SVXY', '2016-01-05')):
        b = pd.read_csv(f'{DATA}/{sym}_daily.csv', index_col=0, parse_dates=True)['close']
        etp = b.pct_change()
        vxx_s, svxy_s = synth_legs(r, svxy_alt=True)                  # tracking vs the era-correct leg (-1x then -0.5x)
        leg = vxx_s if sym == 'VXX' else svxy_s
        ok = etp.index.intersection(leg.index); d = pd.DataFrame({'etp': etp.reindex(ok), 'leg': leg.reindex(ok)}).loc[lo:]
        # the pct_change gap over non-consecutive ETP bars is excluded by requiring both bars on adjacent calendar days
        res = regress(d.etp, d.leg); out.append({'leg': sym, 'window': f'{lo}..2026-10-08', 'leverage': 'era (-1x then -0.5x)' if sym == 'SVXY' else '+1x', **res})
        if sym == 'SVXY':
            for lab, a_, b_ in (('-1x era', '2016-01-05', '2018-02-27'), ('-0.5x era', '2018-02-28', '2026-10-08')):
                dd = d.loc[a_:b_]; out.append({'leg': sym, 'window': f'{a_}..{b_}', 'leverage': lab, **regress(dd.etp, dd.leg)})
            d2 = d.loc['2018-02-28':]; _, s05 = synth_legs(r)
            out.append({'leg': sym, 'window': 'full, forced -0.5x', 'leverage': '-0.5x', **regress(d.etp, s05.reindex(d.index))})
    tr = pd.DataFrame(out); tr.to_csv(f'{ROOT}/tracking.csv', index=False)
    pd.set_option('display.width', 220); print(tr.round(4).to_string())
    gate = {'n_days': len(sub), 'index_ret_missing': int(sub['ret'].isna().sum()) , 'VIX_missing': int(sub['VIX'].isna().sum()) if 'VIX' in sub else None}
    c = cal.loc['2011-01-03':]; gate.update(vix_missing=int(c['VIX'].isna().sum()), vix3m_missing=int(c['VIX3M'].isna().sum()))
    print("GATE", gate)
    big = d.assign(err=d.etp - d.leg)['err'].abs().sort_values(ascending=False).head(8) if False else None


if __name__ == '__main__':
    main()
