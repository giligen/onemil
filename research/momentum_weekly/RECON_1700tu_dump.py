"""RECON_1700tu_dump.py MODE (A|B): dump Monday-open weekly equity, holdings (weights, entry/exit price) and cost per name
for plain / guarded / gated books. A = first build engine (exec of 1700u_gate_guarded.py up to the GREF gate, unchanged);
B = REBUILD_1700tu.py functions (imported, unchanged). Outputs RECON_1700tu_{MODE}_{book}_{weeks,hold}.csv.
Run: bash scripts/research_run.sh -m 2500M python3 research/momentum_weekly/RECON_1700tu_dump.py A > log"""
import sys, logging
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path('/home/ec2-user/onemil/research/momentum_weekly')
MODE = sys.argv[1]
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def dump(book, wk_rows, hold_rows):
    pd.DataFrame(wk_rows).to_csv(HERE / f'RECON_1700tu_{MODE}_{book}_weeks.csv', index=False)
    pd.DataFrame(hold_rows).to_csv(HERE / f'RECON_1700tu_{MODE}_{book}_hold.csv', index=False)
    print('wrote', book, len(wk_rows), len(hold_rows), flush=True)


def mode_a():
    """First build: instrumented copy of its simulate() (same arithmetic), run on its own panel/ranks."""
    src = (HERE / '1700u_gate_guarded.py').read_text().splitlines()
    stop = next(i for i, ln in enumerate(src) if 'GREF repro gate' in ln)
    ns = {'__name__': 'a_dump_bt', '__file__': str(HERE / '1700u_gate_guarded.py')}
    exec(compile('\n'.join(src[:stop]), '1700u[:gate]', 'exec'), ns)
    O, RATE, cand, didx, prior, DAYMAP, tdays = (ns[k] for k in ('O', 'RATE', 'cand', 'didx', 'prior', 'DAYMAP', 'tdays'))
    START_EQ, T0, T1, cidx, rebal_dates = ns['START_EQ'], ns['T0'], ns['T1'], ns['cidx'], ns['rebal_dates']
    # gate (first build's definition: VIX/VIX3M percentile, trailing 252, <20% at prior day)
    def rd(f):
        x = pd.read_csv(HERE / f); x['DATE'] = pd.to_datetime(x['DATE'], format='%m/%d/%Y'); return x.set_index('DATE')['CLOSE'].astype(float)
    s = (rd('REBUILD_1700tu_VIX.csv') / rd('REBUILD_1700tu_VIX3M.csv')).dropna()
    pr = s.rolling(252, min_periods=126).apply(lambda a: (a[:-1] < a[-1]).mean() + 0.5 * (a[:-1] == a[-1]).mean() if len(a) > 1 else np.nan, raw=True)
    pctd = pr.reindex(tdays).ffill()
    sc = {didx[d]: 0.5 for d in rebal_dates if np.nan_to_num(pctd.loc[prior[d]], nan=1.0) < 0.20}
    print('A gate half weeks', len(sc), flush=True)

    def sim(rank_map, scale=None, n=20):
        ns_ = len(cand); sh = np.zeros(ns_); cash = START_EQ
        rk = {didx[x]: [cidx[s_] for s_ in rank_map[prior[x]]] for x in DAYMAP[0] if prior[x] in rank_map and didx[x] <= T1}
        tis = sorted(rk)
        wk, hd = [], []
        for j, ti in enumerate(tis):
            o = O[ti]; pre = cash + sh @ o; eq = pre; sf = 1.0 if scale is None else scale.get(ti, 1.0)
            top = [i for i in rk[ti][:n] if o[i] > 0]
            tgt = np.zeros(ns_); tgt[top] = eq * sf / n
            delta = tgt - sh * o; cs = tr = 0.0; cost_by = {}
            held_before = set(np.flatnonzero(sh > 0))
            for i in np.where(delta < -1e-9)[0]:
                v = -delta[i]; c = RATE[ti, i] * v; cash += v - c; sh[i] -= v / o[i]; tr += v; cs += c; cost_by[i] = c
                if sh[i] <= 1e-12: sh[i] = 0
            for i in np.where(delta > 1e-9)[0]:
                v = delta[i]; c = RATE[ti, i] * v; cash -= v + c; sh[i] += v / o[i]; tr += v; cs += c; cost_by[i] = c
            post = cash + sh @ o
            nxt = tis[j + 1] if j + 1 < len(tis) else None
            wk.append(dict(date=str(tdays[ti].date()), eq_pre=pre, eq_post=post, cost=cs, traded=tr))
            for i in np.flatnonzero(sh > 0):
                hd.append(dict(date=str(tdays[ti].date()), sym=cand[i], w=sh[i] * o[i] / post, p_in=o[i], p_out=(O[nxt, i] if nxt else np.nan),
                               cost=cost_by.get(i, 0.0), rate=RATE[ti, i], kept=int(i in held_before), shares=sh[i]))
            for i in held_before - set(np.flatnonzero(sh > 0)):
                hd.append(dict(date=str(tdays[ti].date()), sym=cand[i], w=0.0, p_in=o[i], p_out=np.nan, cost=cost_by.get(i, 0.0), rate=RATE[ti, i], kept=-1, shares=0.0))
        return wk, hd
    for book, rm, scl in [('plain', ns['ranked_syms'], None), ('guarded', ns['ranked_g'], None), ('gated', ns['ranked_g'], sc)]:
        dump(book, *sim(rm, scl))


def mode_b():
    """Rebuild: instrumented copy of REBUILD_1700tu.simulate (same arithmetic)."""
    sys.path.insert(0, str(HERE))
    import REBUILD_1700tu as RB
    data = RB.load(); cal = pd.DatetimeIndex(data['SPY'][0]); rebs = RB.rebal_dates(cal)
    ts = np.array([cal[cal.searchsorted(r) - 1] for r in rebs], dtype='datetime64[ns]')
    rows = RB.features(data, cal, ts, None); RB.spmap = spmap = {}
    plain, guarded = [], []
    for i, rr in enumerate(rows):
        if len(rr) < RB.TOPN: plain.append(None); guarded.append(None); continue
        for s, sg, b, sp in rr: spmap[(i, s)] = sp
        plain.append([x[0] for x in sorted(rr, key=lambda x: -x[1])[:RB.TOPN]])
        guarded.append([x[0] for x in sorted([x for x in rr if not x[2]], key=lambda x: -x[1])[:RB.TOPN]])
    pct = RB.gate_series(); half = []
    for i in range(len(rebs)):
        p = pct.loc[:pd.Timestamp(ts[i])].iloc[-1] if pd.Timestamp(ts[i]) >= pct.index[0] else np.nan
        half.append(bool(p < 0.20) if np.isfinite(p) else False)

    def sim(picks, hf):
        cash, hold = RB.CAP0, {}; wk, hd = [], []
        for i, reb in enumerate(rebs[:-1]):
            nxt = rebs[i + 1]; tg = picks[i]
            E = cash + sum(hold.values())
            if tg is None:
                wk.append(dict(date=str(reb.date()), eq_pre=E, eq_post=E, cost=0.0, traded=0.0)); continue
            w = 1.0 / (2 * RB.TOPN if hf[i] else RB.TOPN); cost = tr = 0.0; cb = {}
            for s in set(hold) | set(tg):
                delta = abs((w * E if s in tg else 0.0) - hold.get(s, 0.0))
                r = min(0.0005 + 0.5 * spmap.get((i, s), 0.0), 0.0020); cb[s] = (delta * r, r); cost += delta * r; tr += delta
            wk.append(dict(date=str(reb.date()), eq_pre=E, eq_post=E - cost, cost=cost, traded=tr))
            cash = E - cost - w * E * len(tg); new = {}
            for s in tg:
                p0 = RB.px(data, s, reb, True); q = (w * E) / p0; p1 = RB.px(data, s, nxt, True); new[s] = q * p1
                hd.append(dict(date=str(reb.date()), sym=s, w=w * E / (E - cost), p_in=p0, p_out=p1, cost=cb[s][0], rate=cb[s][1], kept=int(s in hold), shares=q))
            for s in set(hold) - set(tg):
                hd.append(dict(date=str(reb.date()), sym=s, w=0.0, p_in=RB.px(data, s, reb, True), p_out=np.nan, cost=cb[s][0], rate=cb[s][1], kept=-1, shares=0.0))
            hold = new
        return wk, hd
    for book, pk, hf in [('plain', plain, [False] * len(rebs)), ('guarded', guarded, [False] * len(rebs)), ('gated', guarded, half)]:
        dump(book, *sim(pk, hf))


(mode_a if MODE == 'A' else mode_b)()
print('DONE', flush=True)
