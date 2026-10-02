"""RECON_1700tu_corrected.py: the rebuild's engine (REBUILD_1700tu.py, unchanged, imported) re-run with the reconciled conventions:
history gate = close 273 rows back must exist (274 bars, as the first build and the live sleeve), the LAST rebalance included,
window 2017-02-06..2026-09-28 on Monday-open marks, and a switchable cost rule: 'A' = first build (rate from the TRADE-day bar,
spread proxy clipped at 20 bp -> <=15 bp, missing -> 15 bp), 'B' = rebuild (signal-day proxy, unclipped, cap 20 bp, missing 5 bp).
Also 'B273' = the rebuild exactly (273-row gate) as a reproduction check. Outputs RECON_1700tu_corrected.csv.
Run: bash scripts/research_run.sh -m 2500M python3 research/momentum_weekly/RECON_1700tu_corrected.py > RECON_1700tu_corrected.log"""
import sys, inspect
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path('/home/ec2-user/onemil/research/momentum_weekly'); sys.path.insert(0, str(HERE))
import REBUILD_1700tu as RB
P = lambda *a: print(*a, flush=True)


def features_gate(gate_rows):
    """Copy of RB.features with the history gate changed to `gate_rows` closes back (272 -> 273 means 274 bars)."""
    s = inspect.getsource(RB.features).replace('pos >= 272', f'pos >= {gate_rows}').replace('p + 1 < 273', f'p < {gate_rows}')
    assert f'pos >= {gate_rows}' in s; ns = dict(RB.__dict__); exec(s, ns); return ns['features']


def trade_rate(data, s, reb, i, spmap, rule):
    """Per-name cost rate under the A or B convention."""
    if rule == 'B':
        return min(0.0005 + 0.5 * spmap.get((i, s), 0.0), 0.0020)
    d, o, h, l, c, v = data[s]; p = np.searchsorted(d, np.datetime64(reb))
    sp = min(max((h[p] - l[p]) / c[p], 0.0) * 0.1, 0.002) if p < len(d) and d[p] == np.datetime64(reb) and c[p] > 0 else 0.002
    return min(0.0005 + 0.5 * sp, 0.0020)


def sim(data, cal, rebs, picks, hf, spmap, rule):
    """RB.simulate with the last rebalance included and the cost rule switchable. Returns Monday-open equity series and daily close marks."""
    cash, hold, wk, daily = RB.CAP0, {}, [], []
    for i, reb in enumerate(rebs):
        nxt = rebs[i + 1] if i + 1 < len(rebs) else None; tg = picks[i]
        E = cash + sum(hold.values())
        if tg is None:
            wk.append((reb, E)); hold = {}; continue
        w = 1.0 / (2 * RB.TOPN if hf[i] else RB.TOPN); cost = 0.0
        for s in set(hold) | set(tg):
            delta = abs((w * E if s in tg else 0.0) - hold.get(s, 0.0)); cost += delta * trade_rate(data, s, reb, i, spmap, rule)
        wk.append((reb, E - cost))
        if nxt is None: break
        cash = E - cost - w * E * len(tg); qty = {s: (w * E) / RB.px(data, s, reb, True) for s in tg}
        for dd in cal[(cal >= reb) & (cal < nxt)]:
            daily.append((dd, cash + sum(qty[s] * RB.px(data, s, dd, False) for s in qty)))
        hold = {s: qty[s] * RB.px(data, s, nxt, True) for s in qty}
    return pd.Series(dict(wk)), pd.Series(dict(daily))


def main():
    data = RB.load(); cal = pd.DatetimeIndex(data['SPY'][0]); rebs = RB.rebal_dates(cal)
    ts = np.array([cal[cal.searchsorted(r) - 1] for r in rebs], dtype='datetime64[ns]'); pct = RB.gate_series(); half = []
    for i in range(len(rebs)):
        p = pct.loc[:pd.Timestamp(ts[i])].iloc[-1] if pd.Timestamp(ts[i]) >= pct.index[0] else np.nan; half.append(bool(p < 0.20) if np.isfinite(p) else False)
    rows = []
    for gate_rows, cfgs in ((273, [('B273', 'B')]), (272, [('G274_costB', 'B'), ('G274_costA', 'A')])):
        feats = features_gate(gate_rows)(data, cal, ts, None); spmap, plain, guarded = {}, [], []
        for i, rr in enumerate(feats):
            if len(rr) < RB.TOPN: plain.append(None); guarded.append(None); continue
            for s, sg, b, sp in rr: spmap[(i, s)] = sp
            plain.append([x[0] for x in sorted(rr, key=lambda x: -x[1])[:RB.TOPN]]); guarded.append([x[0] for x in sorted([x for x in rr if not x[2]], key=lambda x: -x[1])[:RB.TOPN]])
        P(f'features gate {gate_rows}: weeks with picks {sum(g is not None for g in guarded)}')
        for name, rule in cfgs:
            for book, pk, hf in (('plain', plain, [False] * len(rebs)), ('guarded', guarded, [False] * len(rebs)), ('gated', guarded, half)):
                wk, dl = sim(data, cal, rebs, pk, hf, spmap, rule); wk = wk[wk.index >= pd.Timestamp('2017-02-06')]; dl = dl[dl.index >= pd.Timestamp('2017-02-06')]
                yrs = (wk.index[-1] - wk.index[0]).days / 365.25; cagr = (wk.iloc[-1] / wk.iloc[0]) ** (1 / yrs) - 1
                ddm = (wk / wk.cummax() - 1).min(); ddd = (dl / dl.cummax() - 1).min()
                rows.append(dict(config=name, book=book, cagr=cagr, dd_monday=ddm, dd_daily_close=ddd, end=wk.iloc[-1], last=str(wk.index[-1].date())))
                P(rows[-1])
    pd.DataFrame(rows).to_csv(HERE / 'RECON_1700tu_corrected.csv', index=False); P('DONE')


main()
