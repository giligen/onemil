"""PREREG_2 driver: runs the tracking + completeness GATE on the rebuilt index; backtest runs ONLY if the gate passes
(R2 >= 0.95 on both legs, completeness >= 99 %). Otherwise writes gate.json, prints VOID and stops (PREREG_2 rule).
Diagnostics (clearly labelled, NOT a pass path): R2 excluding the 2018-02-05/06 pair, per-year R2, day-loss arithmetic."""
import json, logging, sys
import numpy as np, pandas as pd
import fetch2, index2
log = logging.getLogger('vix_term.run2')


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    ix = pd.read_csv(f'{index2.ROOT}/index_rebuild.csv', index_col=0, parse_dates=True)
    cal = index2.nyse_cal().loc['2011-01-03':]; sub = ix.loc['2011-01-03':]
    gate = {'days': len(cal), 'VIX_cov': 1 - cal['VIX'].isna().mean(), 'VIX3M_cov': 1 - cal['VIX3M'].isna().mean(),
            'index_cov': 1 - sub['ret'].isna().mean(), 'index_first': str(sub.index.min().date()), 'index_last': str(sub.index.max().date())}
    vxx_s, svxy_s = index2.synth_legs(ix['ret'], svxy_alt=True); _, svxy_05 = index2.synth_legs(ix['ret'])
    tr, diag = {}, {}
    for sym, leg, lo in (('VXX', vxx_s, '2018-01-19'), ('SVXY', svxy_s, '2016-01-05'), ('SVXY_forced_-0.5x', svxy_05, '2016-01-05')):
        p = sym.split('_')[0]; b = pd.read_csv(f'{index2.DATA}/{p}_daily.csv', index_col=0, parse_dates=True)['close']
        d = pd.DataFrame({'e': b.pct_change(), 'l': leg}).dropna().loc[lo:]
        tr[sym] = index2.regress(d.e, d.l)
        d2 = d.drop(pd.to_datetime(['2018-02-05', '2018-02-06']), errors='ignore')
        diag[sym] = {'r2_ex_2018-02-05/06': index2.regress(d2.e, d2.l)['r2'], 'r2_ex_2018': index2.regress(d.loc[d.index.year != 2018].e, d.loc[d.index.year != 2018].l)['r2'],
                     'r2_2019_2026': index2.regress(d.loc['2019':].e, d.loc['2019':].l)['r2']}
        pair = d.loc['2018-02-05':'2018-02-06']
        if len(pair): diag[sym]['pair_2d_compound_etp_vs_leg'] = [float((1 + pair.e).prod() - 1), float((1 + pair.l).prod() - 1)]
    gate['tracking'] = tr; gate['diagnostics_not_a_pass_path'] = diag
    ok = all(tr[k]['r2'] >= 0.95 for k in ('VXX', 'SVXY')) and min(gate['VIX_cov'], gate['VIX3M_cov'], gate['index_cov']) >= 0.99
    gate['verdict'] = 'PASS (backtest may run)' if ok else 'VOID (R2 < 0.95 on a leg) -- PREREG_2: stop'
    # day-loss arithmetic on $5K of the SYNTHETIC leg (not a strategy result: gate failed)
    r5, r6 = ix.loc['2018-02-05', 'ret'], ix.loc['2018-02-06', 'ret']
    gate['synthetic_day_pnl_5k'] = {'idx_ret_0205': float(r5), 'idx_ret_0206': float(r6),
                                    '-0.5x_0205': -0.5 * r5 * 5000, '-0.5x_0206': -0.5 * r6 * 5000, '-1x_0205': -r5 * 5000, '-1x_0206': -r6 * 5000}
    b = pd.read_csv(f'{index2.DATA}/SVXY_daily.csv', index_col=0, parse_dates=True)['close']
    gate['real_SVXY_ret_0205_0206'] = [float(b.loc['2018-02-05'] / b.loc['2018-02-02'] - 1), float(b.loc['2018-02-06'] / b.loc['2018-02-05'] - 1)]
    json.dump(gate, open(f'{index2.ROOT}/gate.json', 'w'), indent=1, default=float)
    print(json.dumps(gate, indent=1, default=float))
    if not ok:
        log.error("GATE FAILED -> VOID; no backtest run (PREREG_2). See REPORT2.md")
        sys.exit(3)


if __name__ == '__main__':
    main()
