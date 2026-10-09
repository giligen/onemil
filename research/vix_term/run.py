"""PREREG_1 runner: VIX term-structure sleeve (SVXY/VXX gated on VIX3M/VIX). Reads data/ (see fetch.py), writes
trades.csv, weekly.csv, results.csv, console summary. Verbose. Usage: python run.py [--r-dollars 50]
"""
import argparse, logging, math, os, random, sys
from datetime import date
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data')
sys.path.insert(0, '/home/ec2-user/onemil/scripts')
import cadence_bar as cb                                   # noqa: E402  (scorer reused, thresholds = defaults)
log = logging.getLogger('vix_term')

SLICE = 5000.0
CAL_LO, CAL_HI = pd.Timestamp('2011-01-03'), pd.Timestamp('2026-10-08')
HALVES = {'A': (pd.Timestamp('2011-01-03'), pd.Timestamp('2017-12-31')), 'B': (pd.Timestamp('2018-01-01'), CAL_HI)}
LEV_CHANGE = pd.Timestamp('2018-02-27')                    # SVXY -1x through this close, -0.5x after
EXPENSE = {'SVXY': 0.0095, 'VXX': 0.0089}                  # pro rata per held day /252 (PREREG, charged literally)
VARIANTS = {                                               # name: (hi, lo, short_only, hysteresis_days)
    'base_1.05/0.95': (1.05, 0.95, False, 1), 'thr_1.03/0.97': (1.03, 0.97, False, 1),
    'thr_1.10/0.90': (1.10, 0.90, False, 1), 'short_only_1.05': (1.05, 0.95, True, 1),
    'hyst5_1.05/0.95': (1.05, 0.95, False, 5)}


def load():
    """Load cboe + ETP bars; the NYSE-day calendar proxy is the CBOE VIX/VIX3M common index in [CAL_LO, CAL_HI]."""
    cb_ = pd.read_csv(f'{DATA}/cboe.csv', index_col=0, parse_dates=True)
    cal = cb_.loc[CAL_LO:CAL_HI]
    spy = pd.read_csv(f'{DATA}/SPY_daily.csv', index_col=0, parse_dates=True).index
    keep = (cal.index < spy.min()) | cal.index.isin(spy)               # 2016+: NYSE days = SPY bars (drops CBOE-only holiday rows)
    log.info("calendar: dropped %d CBOE-only non-NYSE rows from %s", int((~keep).sum()), spy.min().date())
    cal = cal[keep]
    px = {s: pd.read_csv(f'{DATA}/{s}_daily.csv', index_col=0, parse_dates=True) for s in ['SVXY', 'VXX']}
    return cal, px


def completeness(cal, px):
    """Gate: every calendar day must have VIX, VIX3M and the ETP close (>= 99 % else VOID). Returns dict + LOST ranges."""
    out = {'n_days': len(cal), 'vix_missing': int(cal['VIX'].isna().sum()), 'vix3m_missing': int(cal['VIX3M'].isna().sum())}
    for s, b in px.items():
        miss = cal.index[~cal.index.isin(b.index)]
        out[s + '_cov'] = 1 - len(miss) / len(cal)
        out[s + '_lost_n'] = len(miss)
        out[s + '_first'] = b.index.min().date()
        out[s + '_lost_post_first'] = [d.date() for d in miss if d >= b.index.min()]
    return out


def signal_state(r, hi, lo, short_only, hyst):
    """Desired position per close: +1 SVXY, -1 VXX, 0 cash. Hysteresis: state flips only after ``hyst`` consecutive
    closes of the same new raw state (applies to entries AND exits)."""
    raw = pd.Series(0, index=r.index)
    raw[r >= hi] = 1
    if not short_only:
        raw[r <= lo] = -1
    raw[r.isna()] = 0
    if hyst <= 1:
        return raw
    out, cur, cand, cnt = [], 0, 0, 0
    for v in raw.values:
        if v == cur:
            cand, cnt = v, 0
        else:
            cnt = cnt + 1 if v == cand else 1
            cand = v
            if cnt >= hyst:
                cur, cnt = v, 0
        out.append(cur)
    return pd.Series(out, index=r.index)


def build_trades(pos, px, cal, exec_mode):
    """Walk the position series into trades and per-day step returns. Position decided at close t is held t -> t+1.
    c2c: enter at close of first signal day, exit at the close of the first day the signal differs. next_open: enter at
    the next day's open, exit at the next day's open after the exit signal. A leg whose ETP has no bar is skipped
    (WARNING counted). Returns (trades DataFrame, steps DataFrame[date, sym, ret_raw, ret_scaled, is_entry, is_exit])."""
    dates = list(cal.index)
    trades, steps, skipped = [], [], 0
    i, n = 0, len(dates)
    while i < n:
        p = pos.iloc[i]
        if p == 0:
            i += 1; continue
        j = i
        while j + 1 < n and pos.iloc[j + 1] == p:
            j += 1
        sym = 'SVXY' if p == 1 else 'VXX'
        b = px[sym]
        k = j + 1 if j + 1 < n else j                           # last held decision day -> exit signal day (or sample end)
        pts = []                                                # (date, price)
        try:
            if exec_mode == 'c2c':
                path = dates[i:k + 1]
                pts = [(d, b.loc[d, 'close']) for d in path]
            else:
                if i + 1 >= n:
                    i = j + 1; continue
                e = dates[i + 1]
                pts = [(e, b.loc[e, 'open'])]
                pts += [(d, b.loc[d, 'close']) for d in dates[i + 1:k + 1]]
                if j + 2 < n:
                    pts.append((dates[j + 2], b.loc[dates[j + 2], 'open']))
        except KeyError:
            skipped += 1; i = j + 1; continue
        if len(pts) < 2 or any(pd.isna(pp) for _, pp in pts):
            skipped += 1; i = j + 1; continue
        for m in range(1, len(pts)):
            d, pr = pts[m]
            ret = pr / pts[m - 1][1] - 1
            lev = 0.5 if (sym == 'SVXY' and d <= LEV_CHANGE) else 1.0
            steps.append((d, sym, ret, ret * lev, m == 1, m == len(pts) - 1, len(trades)))
        trades.append({'entry_signal': dates[i].date(), 'entry_date': pts[0][0].date(), 'exit_date': pts[-1][0].date(),
                       'sym': sym, 'entry_px': pts[0][1], 'exit_px': pts[-1][1]})
        i = j + 1
    if skipped:
        log.warning("%s: %d trade legs skipped (no ETP bar) -- flat on those days", exec_mode, skipped)
    st = pd.DataFrame(steps, columns=['date', 'sym', 'ret_raw', 'ret_scaled', 'is_entry', 'is_exit', 'trade_id'])
    return pd.DataFrame(trades), st


def daily_pnl(steps, cal, scaled, cost_mult, hs):
    """Daily $ P&L on the $5K slice: slice*ret - half-spread on the entry step and the exit step (measured per symbol x
    cost_mult) - expense ratio pro rata per held day. Also returns per-trade pnl Series."""
    ret = steps['ret_scaled'] if scaled else steps['ret_raw']
    hsv = steps['sym'].map(hs) * 1e-4 * cost_mult * SLICE
    pnl = SLICE * ret - steps['is_entry'] * hsv - steps['is_exit'] * hsv - steps['sym'].map(EXPENSE) / 252 * SLICE
    s = pnl.groupby(steps['date']).sum().reindex(cal.index, fill_value=0.0)
    return s, pnl.groupby(steps['trade_id']).sum()


def weekly_from_daily(d, lo, hi):
    """Monday-indexed weekly sums spanning every week in [lo, hi] (zero weeks kept)."""
    w = d.groupby(d.index.map(lambda x: cb.week_monday(x.date()))).sum()
    idx, wk = [], cb.week_monday(lo.date())
    while wk <= cb.week_monday(hi.date()):
        idx.append(wk); wk += pd.Timedelta(days=7).to_pytimedelta()
    return w.reindex(idx, fill_value=0.0)


def tstat(x):
    x = np.asarray(x, float)
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))) if len(x) > 2 and x.std(ddof=1) > 0 else float('nan')


def score_half(dpnl, tpnl, lo, hi, r_dollars):
    """Per-half stats + cadence scorecard (C1-C5, C7; C6 not audited) from the daily and per-trade P&L."""
    d = dpnl.loc[lo:hi]
    wk = weekly_from_daily(d, d.index.min(), d.index.max())
    weekly = [(w, v / r_dollars) for w, v in wk.items()]
    wr = [r for _, r in weekly]
    nz = [{'date': x.date(), 'r': v / r_dollars} for x, v in d.items() if v != 0.0]
    cycles, _ = cb.compute_cycles(weekly, 5.0)
    rng = random.Random(0)
    c = {1: cb.score_c1(cycles, 3.0, 6.0), 2: cb.score_c2(cycles, -4.0, 0.75), 3: cb.score_c3(weekly, -2.0, -4.0, 8.0, 6),
         4: cb.score_c4(weekly, nz, 0.55, 0.10, rng=rng), 5: cb.score_c5(len(tpnl), len(weekly), 3.0),
         7: cb.score_c7(cycles, wr, 10, 6.0, 5.0, rng=rng)}
    k = max(1, math.ceil(0.05 * len(wk)))
    ex_top = float(wk.sort_values().iloc[:-k].sum())
    sd = float(wk.std(ddof=1)) if len(wk) > 2 else float('nan')
    return {'weeks': len(wk), 'total': float(d.sum()), 'wk_mean': float(wk.mean()), 'wk_sd': sd,
            't_iid_trades': tstat(tpnl.values) if len(tpnl) else float('nan'), 't_week_cl': tstat(wk.values),
            'ex_top5pct_wk_total': ex_top, 'worst_day': float(d.min()), 'worst_day_date': d.idxmin().date(),
            'mde_wk_t2.5': 2.5 * sd / math.sqrt(len(wk)), 'n_trades': len(tpnl),
            'c_pass': {k_: v['pass'] for k_, v in c.items()}, 'c': c}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--r-dollars', type=float, default=50.0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    cal, px = load()
    gate = completeness(cal, px)
    log.info("completeness: %s", {k: v for k, v in gate.items() if 'lost_post' not in k})
    log.info("lost post-first-bar days: SVXY %s VXX %s", gate['SVXY_lost_post_first'][:20], gate['VXX_lost_post_first'][:20])
    q = pd.read_csv(f'{DATA}/quotes_1558.csv')
    hs = q.groupby('sym').half_spread_bps.mean().to_dict()
    log.info("measured half-spread bps mean %s p90 %s", hs, q.groupby('sym').half_spread_bps.quantile(.9).to_dict())
    ratio = cal['VIX3M'] / cal['VIX']
    rows, all_tr, all_wk = [], [], []
    runs = dict(VARIANTS); runs['PLACEBO_lag20_1.05/0.95'] = VARIANTS['base_1.05/0.95']
    for vn, (hi, lo, so, hy) in runs.items():
        r_ = ratio.shift(20) if vn.startswith('PLACEBO') else ratio
        pos = signal_state(r_, hi, lo, so, hy)
        for em in ('c2c', 'next_open'):
            tr, st = build_trades(pos, px, cal, em)
            log.info("%s %s: %d trades, %d held-day steps", vn, em, len(tr), len(st))
            for scaled in (True, False):
                for cm in (1.0, 2.0):
                    dp, tp = daily_pnl(st, cal, scaled, cm, hs)
                    if cm == 1.0:
                        t2 = tr.copy(); t2['pnl'] = tp.reindex(range(len(tr))).values
                        t2['variant'], t2['exec'], t2['lev_scaled'] = vn, em, scaled
                        all_tr.append(t2)
                        w = weekly_from_daily(dp, CAL_LO, CAL_HI).rename('pnl').reset_index()
                        w.columns = ['week', 'pnl']; w['variant'], w['exec'], w['lev_scaled'] = vn, em, scaled
                        all_wk.append(w)
                    for hn, (lo_, hi_) in HALVES.items():
                        d0 = dp.loc[lo_:hi_]
                        first = max(d0.index.min(), px['SVXY'].index.min())     # data actually available
                        sub = tr[(pd.to_datetime(tr.exit_date) >= lo_) & (pd.to_datetime(tr.exit_date) <= hi_)]
                        s = score_half(dp, tp.reindex(sub.index).dropna(), first, hi_, a.r_dollars)
                        rows.append({'variant': vn, 'exec': em, 'lev_scaled': scaled, 'cost_mult': cm, 'half': hn,
                                     'from': first.date(), **{k: v for k, v in s.items() if k not in ('c', 'c_pass')},
                                     'C_pass': ''.join(str(int(s['c_pass'][i])) for i in (1, 2, 3, 4, 5, 7)),
                                     'c1_med': s['c'][1]['median'], 'c3_mdd': s['c'][3]['mdd'], 'c3_p10': s['c'][3]['p10'],
                                     'c4_green': s['c'][4]['green'], 'c4_null': s['c'][4]['null'], 'cycles': s['c'][7]['cycles']})
    res = pd.DataFrame(rows)
    res.to_csv(f'{ROOT}/results.csv', index=False)
    pd.concat(all_tr).to_csv(f'{ROOT}/trades.csv', index=False)
    pd.concat(all_wk).to_csv(f'{ROOT}/weekly.csv', index=False)
    # tail lines: 2018-02 episode, base rule
    for em in ('c2c', 'next_open'):
        pos = signal_state(ratio, 1.05, 0.95, False, 1)
        tr, st = build_trades(pos, px, cal, em)
        for sc in (False, True):
            dp, _ = daily_pnl(st, cal, sc, 1.0, hs)
            print(f"TAIL {em} {'scaled' if sc else 'raw -1x'}: 2018-02-05 pnl {dp.loc['2018-02-05']!r}  2018-02-06 pnl {dp.loc['2018-02-06']!r}")
    print("ratio 2018-02-02", ratio.loc['2018-02-02'], "2018-02-05", ratio.loc['2018-02-05'], "2018-02-06", ratio.loc['2018-02-06'])
    b = px['SVXY']; print("SVXY close 2018-02-02/05/06, open 02-06:", b.loc['2018-02-02', 'close'], b.loc['2018-02-05', 'close'], b.loc['2018-02-06', 'close'], b.loc['2018-02-06', 'open'])
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    m = res[(res.cost_mult == 1.0) & (res.exec == 'c2c') & (res.lev_scaled)]
    print(m[['variant', 'half', 'from', 'weeks', 'n_trades', 'total', 'wk_mean', 't_iid_trades', 't_week_cl', 'ex_top5pct_wk_total', 'worst_day', 'worst_day_date', 'mde_wk_t2.5', 'C_pass']].to_string())
    print(gate['n_days'], {k: v for k, v in gate.items() if k.endswith('_cov') or k.endswith('_first') or k.endswith('lost_n')})


if __name__ == '__main__':
    main()
