#!/usr/bin/env python3
"""orb_inplay Cell D — range-stop + static-lock exit (see PREREG.md ## Cell D).

Same picks/direction/entry/cost model as the base book (score.py); different
exit: stop = opposite side of the 09:30-09:35 range, static lock at 1.75R ->
0.5R, no target, time exit 15:55 (m=955).
"""
import json
import numpy as np, pandas as pd

D = '/home/ec2-user/onemil/research/orb_inplay'
EQ = 66000.0
COMM = 0.0035
LOCK_ARM_R = 1.75
LOCK_STOP_R = 0.5
MIN_RANGE_PCT = 0.005
SPLITS = {'TRAIN': ('2025-01-01', '2025-12-31'), 'TRAIN_H1': ('2025-01-01', '2025-06-30'),
          'TRAIN_H2': ('2025-07-01', '2025-12-31'), 'VAL': ('2026-01-01', '2026-05-31')}


def walk(bars, sgn, entry, stop, R):
    """Return (exit_px, exit_m, reason) under the range-stop + static-lock spec."""
    trigger = entry + sgn * LOCK_ARM_R * R
    locked_stop = entry + sgn * LOCK_STOP_R * R
    locked = False
    for bb in bars.itertuples():
        hi, lo, op = bb.high, bb.low, bb.open
        active = locked_stop if locked else stop
        # gap-through against the currently active stop
        if sgn == 1 and op <= active:
            return op, bb.m, ('lock' if locked else 'stop')
        if sgn == -1 and op >= active:
            return op, bb.m, ('lock' if locked else 'stop')
        if not locked:
            triggered = (hi >= trigger) if sgn == 1 else (lo <= trigger)
            if triggered:
                locked = True
                active = locked_stop
                hit_now = (lo <= active) if sgn == 1 else (hi >= active)
                if hit_now:
                    return active, bb.m, 'lock'
                continue
        hit = (lo <= active) if sgn == 1 else (hi >= active)
        if hit:
            return active, bb.m, ('lock' if locked else 'stop')
    last = bars.iloc[-1]
    return float(last.close), int(last.m), 'time'


def build():
    pk = pd.read_parquet(f'{D}/picks.parquet')
    pk = pk[(pk.bar_date >= '2025-01-01') & (pk.bar_date < '2026-06-01')]
    bars = pd.read_parquet(f'{D}/daybars.parquet').sort_values(['bar_date', 'symbol', 'm'])
    bg = {k: v for k, v in bars.groupby(['bar_date', 'symbol'], sort=False)}
    o5 = pd.read_parquet(f'{D}/open5.parquet')
    o5 = o5[(o5.m >= 570) & (o5.m <= 574)]
    rg = o5.groupby(['bar_date', 'symbol']).agg(rhi=('high', 'max'), rlo=('low', 'min'))
    rows = []
    for r in pk.itertuples():
        rec = dict(day=r.bar_date, symbol=r.symbol, rvol=r.rvol, rk=int(r.rk), side=r.side,
                   prev_close=r.prev_close, atr14=r.atr14, adv20=r.adv20)
        if r.side == 'doji':
            rec.update(status='doji'); rows.append(rec); continue
        b = bg.get((r.bar_date, r.symbol))
        if b is None or not len(b) or not (b.m == 575).any():
            rec.update(status='no_bars'); rows.append(rec); continue
        key = (r.bar_date, r.symbol)
        if key not in rg.index:
            rec.update(status='no_range'); rows.append(rec); continue
        rhi, rlo = float(rg.loc[key, 'rhi']), float(rg.loc[key, 'rlo'])
        e = float(b[b.m == 575].open.iloc[0])
        if r.side == 'short' and e <= r.prev_close * 0.90:
            rec.update(status='regsho', entry=e); rows.append(rec); continue
        if (rhi - rlo) / e < MIN_RANGE_PCT:
            rec.update(status='range_too_small', entry=e, range_pct=(rhi - rlo) / e); rows.append(rec); continue
        sgn = 1 if r.side == 'long' else -1
        stop = rlo if sgn == 1 else rhi
        R = abs(e - stop)
        if R <= 0:
            rec.update(status='bad_R', entry=e); rows.append(rec); continue
        px, exm, reason = walk(b, sgn, e, stop, R)
        rec.update(status='ok', entry=e, R=R, stop=stop, exit=px, exit_m=int(exm),
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
    if n < 5:
        print(f'{label}: n={n}', file=f); return {}
    r = t.net_r.values; mu = r.mean(); sd = r.std(ddof=1)
    se_iid = sd / np.sqrt(n)
    dg = t.groupby('day').net_r.agg(['sum', 'size'])
    nd = len(dg)
    se_cl = (dg['sum'] - mu * dg['size']).std(ddof=1) * np.sqrt(nd) / n if nd > 1 else np.nan
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
    d = dict(gross_r=gr, gross_t=gr / gse, n=n, days=nd, fills_day=n / nd, fills_wk=n / weeks,
              net_r=mu, se_iid=se_iid, se_cl=se_cl, t_iid=t_iid, t_cl=t_cl, mde=mde,
              wr=len(w) / n, avg_w=w.net_r.mean() if len(w) else np.nan,
              avg_l=l.net_r.mean() if len(l) else np.nan, top5_share=top5, ex1=ex1, ex5=ex5,
              pnl=float(t.pnl.sum()), mdd=mdd(t.groupby('day').pnl.sum()))
    print(f'\n### {label}', file=f)
    print(f'  GROSS R/trade={gr:+.4f} (iid t {gr/gse:+.2f})  |  avg cost {t.cost_ps.div(t.R).mean():.3f}R/trade', file=f)
    print(f'  n={n} days={nd} weeks={weeks} fills/day={d["fills_day"]:.1f} fills/wk={d["fills_wk"]:.1f}  '
          f'net R/trade={mu:+.4f} (iid SE {se_iid:.4f}, t {t_iid:+.2f} | clustered SE {se_cl:.4f}, t {t_cl:+.2f}) MDE={mde:.3f}R', file=f)
    print(f'  WR={d["wr"]:.1%} avgW={d["avg_w"]:+.2f}R avgL={d["avg_l"]:+.2f}R  '
          f'ex-top1%={ex1:+.4f}R ex-top5%={ex5:+.4f}R top5 P&L share={top5:.1%}', file=f)
    print(f'  P&L ${d["pnl"]:,.0f}  MDD ${d["mdd"]:,.0f}', file=f)
    return d


def main():
    t = build(); t.to_csv(f'{D}/trades_raw_D.csv', index=False)
    hs = json.load(open(f'{D}/hs_table.json'))
    hs_tab = {tuple(k.split('|')): v for k, v in hs['table'].items()}
    ok = t[t.status == 'ok'].copy()
    ok = costs(ok, hs_tab)
    s1 = size(ok.copy(), 1)
    res = {}
    with open(f'{D}/score_d.out', 'w') as f:
        print('=== STATUS MIX ===', file=f); print(t.status.value_counts().to_dict(), file=f)
        skip_n = int((t.status == 'range_too_small').sum())
        ok_n = int((t.status == 'ok').sum())
        elig = skip_n + ok_n
        print(f'range_too_small: {skip_n}/{elig} ({skip_n/elig:.1%} of ok+skipped)', file=f)
        for sp, (a, b) in SPLITS.items():
            x = s1[(s1.day >= a) & (s1.day <= b)]
            for side in ('combined', 'long', 'short'):
                y = x if side == 'combined' else x[x.side == side]
                y = y[y.shares > 0]
                res[f'{sp}|{side}'] = stats(y, f'1x {sp} {side}', f)
        # positions/day (1x cap binding check) -- admitted (shares>0) only
        posday = s1[s1.shares > 0].groupby('day').size()
        capped_n = int((s1.shares == 0).sum())
        print(f'\nadmitted positions/day (1x cap): mean={posday.mean():.2f} max={posday.max()}  '
              f'cap-zeroed rows={capped_n}/{len(s1)}', file=f)
        s1.to_csv(f'{D}/book_D_1x.csv', index=False)
        # cadence_bar input
        cad = s1[s1.shares > 0][['day', 'net_r', 'symbol']].rename(columns={'day': 'date', 'net_r': 'pnl_R'})
        cad.to_csv(f'{D}/trades_D_cadence.csv', index=False)
    json.dump(res, open(f'{D}/results_D.json', 'w'), indent=1, default=float)
    print(open(f'{D}/score_d.out').read())


if __name__ == '__main__':
    main()
