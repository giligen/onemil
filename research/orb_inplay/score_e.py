#!/usr/bin/env python3
"""orb_inplay Cell E — liquid sub-universe (price>=$20, ADV20>=5M), SAME exit/cost as base
book (score.py's target/stop spec: stop=0.10*ATR14, target=10R, time exit 15:55). See
PREREG.md ## Cell E.
"""
import os, sys, json
import numpy as np, pandas as pd
D = '/home/ec2-user/onemil/research/orb_inplay'
EQ = 66000.0; COMM = 0.0035
SPLITS = {'TRAIN': ('2025-01-01', '2025-12-31'), 'TRAIN_H1': ('2025-01-01', '2025-06-30'),
          'TRAIN_H2': ('2025-07-01', '2025-12-31'), 'VAL': ('2026-01-01', '2026-05-31')}


def build():
    pk = pd.read_parquet(f'{D}/picks_e.parquet')
    pk = pk[pk.bar_date >= '2025-01-01']
    bars = pd.read_parquet(f'{D}/daybars_e.parquet')
    bars = bars.sort_values(['bar_date', 'symbol', 'm'])
    bg = {k: v for k, v in bars.groupby(['bar_date', 'symbol'], sort=False)}
    rows = []
    for r in pk.itertuples():
        rec = dict(day=r.bar_date, symbol=r.symbol, rvol=r.rvol, rk=int(r.rk), side=r.side,
                   prev_close=r.prev_close, atr14=r.atr14, adv20=r.adv20)
        b = bg.get((r.bar_date, r.symbol))
        if r.side == 'doji': rec.update(status='doji'); rows.append(rec); continue
        if b is None or not len(b) or not (b.m == 575).any():
            rec.update(status='no_bars'); rows.append(rec); continue
        if not np.isfinite(r.atr14) or r.atr14 <= 0:
            rec.update(status='no_atr'); rows.append(rec); continue
        e = float(b[b.m == 575].open.iloc[0])
        if r.side == 'short' and e <= r.prev_close * 0.90:
            rec.update(status='regsho', entry=e); rows.append(rec); continue
        R = 0.10 * r.atr14
        if R <= 0 or R >= e: rec.update(status='bad_R', entry=e); rows.append(rec); continue
        sgn = 1 if r.side == 'long' else -1
        stop = e - sgn * R; tgt = e + sgn * 10 * R
        px, exm, reason = None, None, None
        for bb in b.itertuples():
            hi, lo, op = bb.high, bb.low, bb.open
            if sgn == 1:
                if op <= stop: px, reason = op, 'stop'
                elif op >= tgt: px, reason = op, 'target'
                elif lo <= stop: px, reason = stop, 'stop'
                elif hi >= tgt: px, reason = tgt, 'target'
            else:
                if op >= stop: px, reason = op, 'stop'
                elif op <= tgt: px, reason = op, 'target'
                elif hi >= stop: px, reason = stop, 'stop'
                elif lo <= tgt: px, reason = tgt, 'target'
            if px is not None: exm = bb.m; break
        if px is None:
            last = b.iloc[-1]; px, exm, reason = float(last.close), int(last.m), 'time'
        rec.update(status='ok', entry=e, R=R, stop=stop, tgt=tgt, exit=px, exit_m=int(exm),
                   reason=reason, gross_r=sgn * (px - e) / R)
        rows.append(rec)
    return pd.DataFrame(rows)


def costs(t, hs_tab):
    def hs(px, m):
        pb = pd.cut([px], [0, 10, 20, 50, 1e9], labels=['a', 'b', 'c', 'd'])[0]
        cb = 'entry' if m == 575 else ('mid' if m < 900 else 'late')
        v = hs_tab.get((pb, cb), hs_tab.get(('glob', 'glob')))
        return v * px / 100.0
    t['cost_ps'] = [hs(r.entry, 575) + hs(r.exit, r.exit_m) + 2 * COMM for r in t.itertuples()]
    t['net_r'] = t.gross_r - t.cost_ps / t.R
    return t


def size(t, lev):
    out = []
    for day, g in t.groupby('day'):
        g = g.sort_values('rk'); cap = lev * EQ; used = 0.0; sh = []
        for r in g.itertuples():
            s = int(np.floor(0.01 * EQ / r.R))
            n = s * r.entry
            if used + n > cap:
                s = int(max(0, np.floor((cap - used) / r.entry)))
                n = s * r.entry
            used += n; sh.append(s)
        g = g.assign(shares=sh); out.append(g)
    t = pd.concat(out)
    t['pnl'] = t.shares * (t.net_r * t.R)
    return t


def mdd(daily):
    c = daily.cumsum(); return float((c - c.cummax()).min())


def stats(t, label, f):
    n = len(t)
    if n < 5: print(f'{label}: n={n}', file=f); return {}
    r = t.net_r.values; mu = r.mean(); sd = r.std(ddof=1)
    se_iid = sd / np.sqrt(n)
    dg = t.groupby('day').net_r.agg(['sum', 'size'])
    nd = len(dg); se_cl = (dg['sum'] - mu * dg['size']).std(ddof=1) * np.sqrt(nd) / n if nd > 1 else np.nan
    t_iid = mu / se_iid; t_cl = mu / se_cl if se_cl and se_cl > 0 else np.nan
    mde = 2.802 * sd / np.sqrt(n)
    w = t[t.net_r > 0]; l = t[t.net_r <= 0]
    posp = t.pnl[t.pnl > 0].sum()
    top5 = t.nlargest(5, 'pnl').pnl.sum() / posp if posp > 0 else np.nan
    gr = t.gross_r.mean(); gse = t.gross_r.std(ddof=1) / np.sqrt(n)
    k1 = max(1, int(np.ceil(0.01 * n))); k5 = max(1, int(np.ceil(0.05 * n)))
    ex1 = t.drop(t.nlargest(k1, 'net_r').index).net_r.mean()
    ex5 = t.drop(t.nlargest(k5, 'net_r').index).net_r.mean()
    wk = pd.to_datetime(t.day).dt.to_period('W')
    weeks = wk.nunique()
    wpl = t.groupby(wk).pnl.sum(); gw = (wpl > 0).mean()
    rng = np.random.default_rng(7); pv = t.pnl.values; wkc = wk.values; nulls = []
    for _ in range(2000):
        s = rng.choice([-1, 1], size=n)
        nulls.append((pd.Series(pv * s).groupby(wkc).sum() > 0).mean())
    nulls = np.array(nulls)
    d = dict(gross_r=gr, gross_t=gr / gse, n=n, days=nd, fills_day=n / nd, fills_wk=n / weeks,
              net_r=mu, se_iid=se_iid, se_cl=se_cl, t_iid=t_iid, t_cl=t_cl, mde=mde,
              wr=len(w) / n, avg_w=w.net_r.mean() if len(w) else np.nan,
              avg_l=l.net_r.mean() if len(l) else np.nan, top5_share=top5, ex1=ex1, ex5=ex5,
              green_wk=gw, null_green=float(nulls.mean()), p_green=float((nulls >= gw).mean()),
              pnl=float(t.pnl.sum()), mdd=mdd(t.groupby('day').pnl.sum()))
    print(f'\n### {label}', file=f)
    print(f'  GROSS R/trade={gr:+.4f} (iid t {gr/gse:+.2f})  |  avg cost {t.cost_ps.div(t.R).mean():.3f}R/trade', file=f)
    print(f'  n={n} days={nd} weeks={weeks} fills/day={d["fills_day"]:.1f} fills/wk={d["fills_wk"]:.1f}  '
          f'net R/trade={mu:+.4f} (iid SE {se_iid:.4f}, t {t_iid:+.2f} | clustered SE {se_cl:.4f}, t {t_cl:+.2f}) MDE={mde:.3f}R', file=f)
    print(f'  WR={d["wr"]:.1%} avgW={d["avg_w"]:+.2f}R avgL={d["avg_l"]:+.2f}R  '
          f'ex-top1%={ex1:+.4f}R ex-top5%={ex5:+.4f}R top5 P&L share={top5:.1%}', file=f)
    print(f'  green weeks {gw:.1%} vs null {nulls.mean():.1%} (p={d["p_green"]:.3f})  '
          f'P&L ${d["pnl"]:,.0f}  MDD ${d["mdd"]:,.0f}', file=f)
    return d


def main():
    t = build(); t.to_csv(f'{D}/trades_raw_E.csv', index=False)
    hs = json.load(open(f'{D}/hs_table.json'))
    hs_tab = {tuple(k.split('|')): v for k, v in hs['table'].items()}
    ok = t[t.status == 'ok'].copy()
    ok = costs(ok, hs_tab)
    res = {}
    with open(f'{D}/score_e.out', 'w') as f:
        print('=== STATUS MIX ===', file=f); print(t.status.value_counts().to_dict(), file=f)
        for lev in (1, 2):
            s = size(ok.copy(), lev)
            for sp, (a, b) in SPLITS.items():
                x = s[(s.day >= a) & (s.day <= b)]
                for side in ('combined', 'long', 'short'):
                    y = x if side == 'combined' else x[x.side == side]
                    y = y[y.shares > 0]
                    res[f'{lev}x|{sp}|{side}'] = stats(y, f'{lev}x {sp} {side}', f)
            posday = s[s.shares > 0].groupby('day').size()
            capped_n = int((s.shares == 0).sum())
            print(f'\n{lev}x admitted positions/day: mean={posday.mean():.2f} max={posday.max()}  '
                  f'cap-zeroed rows={capped_n}/{len(s)}', file=f)
            s.to_csv(f'{D}/book_E_{lev}x.csv', index=False)
        s1 = size(ok.copy(), 1)
        cad = s1[s1.shares > 0][['day', 'net_r', 'symbol']].rename(columns={'day': 'date', 'net_r': 'pnl_R'})
        cad.to_csv(f'{D}/trades_E_cadence.csv', index=False)
    json.dump(res, open(f'{D}/results_E.json', 'w'), indent=1, default=float)
    print(open(f'{D}/score_e.out').read())


if __name__ == '__main__':
    main()
