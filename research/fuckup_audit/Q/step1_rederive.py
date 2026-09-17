#!/usr/bin/env python3
"""Step 1 — re-derive RESULTS.md rows 1/1b (SPY + QQQ, paper conventions) with the independent
re-implementation in zsim.py, and diff the trade sets against the original script's simulator.

Writes Q/step1_rederive.md and Q/step1_trades_diff.txt.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/fuckup_audit/Q')
sys.path.insert(0, '/home/ec2-user/onemil/research/lit_review_2026')
import zsim as Z
import test_zarattini_spy as ORIG   # the original implementation, for a trade-by-trade diff

OUT = '/home/ec2-user/onemil/research/fuckup_audit/Q/step1_rederive.md'

HDR = ('| sim | symbol | period | costs | days | traded | bps/day dyn | t | hit % | trades/day '
       '| ann % 1x | SR 1x | MDD % 1x | ann % dyn | SR dyn | MDD % dyn |')
SEP = '|---' * 16 + '|'


def row(sim, sym, label, cost, df):
    m1 = Z.metrics(df, 'r1x'); md = Z.metrics(df, 'rdyn')
    f = Z.fmt
    return (f'| {sim} | {sym} | {label} | {cost} | {m1["days"]} | {m1["traded"]} | {f(md["bps"])} | '
            f'{f(md["t"], 2)} | {f(md["hit"], 0)} | {f(m1["trades_day"], 2)} | {f(m1["ann"])} | '
            f'{f(m1["sharpe"], 2)} | {f(m1["mdd"])} | {f(md["ann"])} | {f(md["sharpe"], 2)} | {f(md["mdd"])} |')


def main():
    md = ['# Step 1 — re-derivation of RESULTS.md rows 1 (SPY) and 1b (QQQ)', '',
          'Paper conventions exactly: bands from the 14-prior-day time-of-day sigma, semi-hourly checks from',
          '10:00, VWAP/band trailing stop, fill at the CHECK BAR CLOSE, flat at the 15:59 close, no slippage',
          'beyond the stated cost model. "mine" = Q/zsim.py (re-implementation), "orig" =',
          'research/lit_review_2026/test_zarattini_spy.py.', '', HDR, SEP]
    diffs = []
    for sym in ('SPY', 'QQQ'):
        data = Z.load_symbol(sym)
        UB, LB = Z.bands(data, 1.0)
        tr_mine, _ = Z.simulate(data, UB, LB, fill='close', eod='moc', slip_bp=0.0)

        # ORIG.load_symbol reads all 2.1M bars (incl. extended hours) in one fetch -> MemoryError
        # under the 1 GB ulimit. The loader is verified separately (Q/loader_check.py, checksum
        # equality of C/O/VW/dopen/prevclose/sig14/sigma); here we reuse it and diff the LOGIC
        # (bands + simulate), which is what the audit is about.
        odata = data
        oUB, oLB = ORIG.bands(odata, 'open_prevclose', 1.0)
        tr_orig = ORIG.simulate(odata, oUB, oLB, ORIG.CHECKS_SEMI, 'vwap_band', 'close')

        a = pd.DataFrame([(t['d'], t['side'], t['e'], t['x']) for t in tr_mine],
                         columns=['d', 'side', 'e', 'x'])
        b = pd.DataFrame(tr_orig, columns=['d', 'side', 'e', 'x'])
        same = len(a) == len(b) and np.allclose(a.values.astype(float), b.values.astype(float), atol=1e-9)
        maxdiff = (np.abs(a.values.astype(float) - b.values.astype(float)).max()
                   if len(a) == len(b) else float('nan'))
        diffs.append(f'{sym}: mine {len(a)} trades, orig {len(b)} trades, identical={same}, '
                     f'max|diff|={maxdiff:.3g}')
        print(diffs[-1], flush=True)

        for cost in ('paper', 'bp', 'gross'):
            dfm = Z.daily_returns(data, tr_mine, cost)
            dfo = ORIG.daily_returns(odata, tr_orig, cost)
            is_m, oos_m = Z.split(dfm); is_o, oos_o = Z.split(dfo)
            md.append(row('mine', sym, 'IS 2016-2023', cost, is_m))
            md.append(row('orig', sym, 'IS 2016-2023', cost, is_o))
            md.append(row('mine', sym, 'OOS 2024->', cost, oos_m))
            md.append(row('orig', sym, 'OOS 2024->', cost, oos_o))
        del data, odata
    md += ['', '## Trade-set diff', ''] + [f'- {d}' for d in diffs]
    with open(OUT, 'w') as f:
        f.write('\n'.join(md) + '\n')
    print('wrote', OUT)


if __name__ == '__main__':
    main()
