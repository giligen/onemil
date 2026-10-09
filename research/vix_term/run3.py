"""PREREG_3 driver. Gate FIRST: R2 of overlapping 5-day log returns, synthetic leg vs real ETP >= 0.95 both legs, and
|annualised return difference| <= 3 %/yr both legs; else VOID and stop. If PASS: PREREG_1 rule + variants on the synthetic
legs 2011-2026 (c2c only: a synthetic series has no open) and on the real ETP legs side by side. Verbose."""
import json, logging, sys
import numpy as np, pandas as pd
import index2, run
log = logging.getLogger('vix_term.run3')
ROOT, DATA = index2.ROOT, index2.DATA


def horizon_r2(real_close, leg_ret, lo, h):
    """R2 + annualised difference of overlapping h-day log returns, synthetic cumulative leg vs real ETP close (NYSE-day index)."""
    idx = real_close.index.intersection(leg_ret.index); idx = idx[idx >= pd.Timestamp(lo)]
    lr = np.log(real_close.reindex(idx)).diff(); ls = np.log1p(leg_ret.reindex(idx)).fillna(0)
    lr.iloc[0] = 0.0; lr = lr.fillna(0)
    cr, cs = lr.cumsum(), ls.cumsum()
    yr, ys = (cr - cr.shift(h)).dropna(), (cs - cs.shift(h)).dropna()
    x, y = ys.values, yr.values
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2); beta = float(np.polyfit(x, y, 1)[0])
    years = (idx[-1] - idx[0]).days / 365.25
    diff = float((cs.iloc[-1] - cr.iloc[-1]) / years)               # synthetic - real, log %/yr (mean)
    return {'h': h, 'r2': r2, 'beta': beta, 'n': len(x), 'ann_diff_pct': diff * 100,
            'tot_log_real': float(cr.iloc[-1]), 'tot_log_syn': float(cs.iloc[-1]), 'years': years}


def synth_px(ix_ret):
    """Synthetic price frames (open=close: no intraday synthetic) from the era-correct legs."""
    vxx, svxy = index2.synth_legs(ix_ret, svxy_alt=True)
    mk = lambda r: pd.DataFrame({'close': 100 * (1 + r.fillna(0)).cumprod()}).assign(open=lambda d: d.close)
    return {'VXX': mk(vxx), 'SVXY': mk(svxy)}, vxx, svxy


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    ix = pd.read_csv(f'{ROOT}/index_rebuild.csv', index_col=0, parse_dates=True)
    spx, vxx_l, svxy_l = synth_px(ix['ret'])
    rows = []
    for sym, leg, lo in (('VXX', vxx_l, '2018-01-18'), ('SVXY', svxy_l, '2016-01-04')):
        b = pd.read_csv(f'{DATA}/{sym}_daily.csv', index_col=0, parse_dates=True)['close']
        for h in (5, 20):
            r = horizon_r2(b, leg, lo, h); r['leg'] = sym; rows.append(r)
            log.info("%s h=%d R2 %.4f beta %.3f n %d annual diff (syn-real) %.2f %%/yr", sym, h, r['r2'], r['beta'], r['n'], r['ann_diff_pct'])
    tr = pd.DataFrame(rows)[['leg', 'h', 'r2', 'beta', 'n', 'ann_diff_pct', 'tot_log_real', 'tot_log_syn', 'years']]
    tr.to_csv(f'{ROOT}/tracking3.csv', index=False); print(tr.round(4).to_string())
    g5 = tr[tr.h == 5].set_index('leg')
    ok = bool((g5.r2 >= 0.95).all() and (g5.ann_diff_pct.abs() <= 3).all())
    log.info("GATE (5d R2>=0.95 both, |diff|<=3 %%/yr both): %s", 'PASS' if ok else 'VOID')
    json.dump({'pass': ok, 'tracking': tr.to_dict('records')}, open(f'{ROOT}/gate3.json', 'w'), indent=1, default=float)
    if not ok:
        log.error("GATE FAILED -> VOID; strategy NOT run (PREREG_3)"); sys.exit(3)
    # ---- strategy (only on PASS): synthetic 2011-2026 and real ETP side by side, c2c, scaled, measured cost
    cal, px_real = run.load(); q = pd.read_csv(f'{DATA}/quotes_1558.csv'); hs = q.groupby('sym').half_spread_bps.mean().to_dict()
    ratio = cal['VIX3M'] / cal['VIX']; out, alltr, allwk = [], [], []
    runs = dict(run.VARIANTS); runs['PLACEBO_lag20_1.05/0.95'] = run.VARIANTS['base_1.05/0.95']
    for src, px in (('synthetic', spx), ('real', px_real)):
        for vn, (hi, lo, so, hy) in runs.items():
            pos = run.signal_state(ratio.shift(20) if vn.startswith('PLACEBO') else ratio, hi, lo, so, hy)
            t, st = run.build_trades(pos, px, cal, 'c2c')
            for cm in (1.0, 2.0):
                dp, tp = run.daily_pnl(st, cal, True, cm, hs)
                if cm == 1.0:
                    t2 = t.copy(); t2['pnl'] = tp.reindex(range(len(t))).values; t2['variant'], t2['src'] = vn, src; alltr.append(t2)
                    w = run.weekly_from_daily(dp, run.CAL_LO, run.CAL_HI).rename('pnl').reset_index(); w.columns = ['week', 'pnl']
                    w['variant'], w['src'] = vn, src; allwk.append(w)
                for hn, (lo_, hi_) in run.HALVES.items():
                    first = max(dp.loc[lo_:hi_].index.min(), px['SVXY'].index.min() if src == 'real' else lo_)
                    sub = t[(pd.to_datetime(t.exit_date) >= lo_) & (pd.to_datetime(t.exit_date) <= hi_)]
                    s = run.score_half(dp, tp.reindex(sub.index).dropna(), first, hi_, 50.0)
                    out.append({'src': src, 'variant': vn, 'cost_mult': cm, 'half': hn, **{k: v for k, v in s.items() if k not in ('c', 'c_pass')},
                                'C_pass': ''.join(str(int(s['c_pass'][i])) for i in (1, 2, 3, 4, 5, 7))})
                    log.info("%s %s x%.0f %s total %.0f", src, vn, cm, hn, s['total'])
            if vn.startswith('base'):
                for sc in (src,):
                    dp, _ = run.daily_pnl(st, cal, True, 1.0, hs)
                    print(f"TAIL {src} c2c scaled: 2018-02-05 {dp.loc['2018-02-05']!r} 2018-02-06 {dp.loc['2018-02-06']!r}")
    res = pd.DataFrame(out); res.to_csv(f'{ROOT}/results3.csv', index=False)
    pd.concat(alltr).to_csv(f'{ROOT}/trades3.csv', index=False); pd.concat(allwk).to_csv(f'{ROOT}/weekly3.csv', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    print(res[res.cost_mult == 1.0][['src', 'variant', 'half', 'weeks', 'n_trades', 'total', 'wk_mean', 't_iid_trades', 't_week_cl', 'ex_top5pct_wk_total', 'worst_day', 'mde_wk_t2.5', 'C_pass']].to_string())


if __name__ == '__main__':
    main()
