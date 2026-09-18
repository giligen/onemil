"""S1 step 5 — everything the two G1/G2/TEST survivors have to answer before anyone writes code.

Survivors (both SHORTS, both the +5 minute horizon):
  A  up-halt   x fade         = short the first bar after a limit-UP resume
  B  down-halt x continuation = short the first bar after a limit-DOWN resume
Under the 0.6% no-chase cap, A fills only when the resume printed ABOVE ref*1.006 and B only when it
printed BELOW ref*0.994 — so both are the same trade: fade the RESUME GAP, whichever way it points.
"""
import numpy as np
import pandas as pd

import score_cells as S

HERE = '/home/ec2-user/onemil/research/fuckup_audit/O_halt'
SPLITS = S.SPLITS


def survivor_rows(tr):
    a = S.score(tr[tr['side'] == 'up'], 'fade', 'h5')
    a['cell'] = 'A up-fade'
    b = S.score(tr[tr['side'] == 'down'], 'continuation', 'h5')
    b['cell'] = 'B down-cont'
    keys = tr.set_index(['day', 'symbol'])
    out = []
    for src, sc in (('up', a), ('down', b)):
        sc = sc[sc['ok']].copy()
        out.append(sc)
    return pd.concat(out, ignore_index=True), a, b


def main():
    tr = pd.read_parquet(f'{HERE}/trades.parquet')
    for c in ('halt_ts', 'resume_ts', 'entry_t'):
        tr[c] = pd.to_datetime(tr[c], utc=True).dt.tz_convert('America/New_York')
    ssr = pd.read_parquet(f'{HERE}/ssr_state.parquet')
    ssr['resume_ts'] = pd.to_datetime(ssr['resume_ts'], utc=True).dt.tz_convert('America/New_York')
    tr = tr.merge(ssr[['day', 'symbol', 'resume_ts', 'ssr']].drop_duplicates(['day', 'symbol', 'resume_ts']),
                  on=['day', 'symbol', 'resume_ts'], how='left')
    tr['ssr'] = tr['ssr'].fillna('unknown')

    cells = {'A up-fade  (short a limit-UP resume)': ('up', 'fade'),
             'B down-cont(short a limit-DOWN resume)': ('down', 'continuation')}

    print('=== 1. gap / obtainability, whole sample ===')
    for nm, (side, rule) in cells.items():
        ss = tr[tr['side'] == side]
        sc = S.score(ss, rule, 'h5')
        g = ss['gap_pct'].values * 100
        print(f'{nm}: events {len(ss)}  filled {int(sc["ok"].sum())} ({sc["ok"].mean():.1%})   '
              f'resume gap %% median {np.median(g):+.2f}  filled-median {np.median(g[sc["ok"].values]):+.2f}  '
              f'rejected-median {np.median(g[~sc["ok"].values]):+.2f}')

    print('\n=== 2. SSR state at the resume (Reg SHO rule 201) on the FILLED short trades ===')
    for nm, (side, rule) in cells.items():
        ss = tr[tr['side'] == side].reset_index(drop=True)
        sc = S.score(ss, rule, 'h5')
        f = ss[sc['ok'].values]
        vc = f['ssr'].value_counts(normalize=True)
        print(f'{nm}: ' + '  '.join(f'{k}={v:.1%}' for k, v in vc.items()))
        for split, a, b in SPLITS:
            k = f[(f['day'] >= a) & (f['day'] <= b)]
            if len(k):
                print(f'    {split}: SSR-restricted {(k["ssr"]=="Y").mean():.1%} of {len(k)}')

    print('\n=== 3. the same cells with SSR="Y" removed (the trades a long-only-plus-shorts stack could actually place) ===')
    for nm, (side, rule) in cells.items():
        ss = tr[tr['side'] == side].reset_index(drop=True)
        sc = S.score(ss, rule, 'h5')
        keep = sc['ok'].values & (ss['ssr'] != 'Y').values
        for split, a, b in SPLITS:
            sel = keep & (ss['day'] >= a).values & (ss['day'] <= b).values
            x = sc.loc[sel, 'net_R'].values
            st = S.stats(x)
            print(f'  {nm[:12]:12s} {split:5s} n={st["n"]:4d} mean {st["mean"]:+.4f} t {st["t"]:+.2f} mde {st["mde"]:.3f}')

    print('\n=== 4. breakeven quoted spread (what the cost table would have to be to kill the edge) ===')
    for nm, (side, rule) in cells.items():
        ss = tr[tr['side'] == side].reset_index(drop=True)
        sc = S.score(ss, rule, 'h5')
        k = sc[sc['ok']]
        gross = k['gross_R'].mean()
        # net = gross - (0.25+0.875)*0.5*spread_pct/R_PCT  ->  spread_pct at net=0
        be = gross * S.R_PCT / (1.125 * 0.5)
        mod = ss.loc[sc['ok'].values, 'entry_t'].dt.hour * 60 + ss.loc[sc['ok'].values, 'entry_t'].dt.minute
        used = np.mean([S.SPREAD_BPS[S.band(p)][S.hourbucket(m)] for p, m in
                        zip(ss.loc[sc['ok'].values, 'fill'], mod)]) / 100.0
        print(f'{nm}: gross {gross:+.4f} R  -> breakeven quoted spread {be:.2f}%  '
              f'(the cost table charges {used:.2f}% for this sample)')

    print('\n=== 5. price band / liquidity of the filled shorts ===')
    for nm, (side, rule) in cells.items():
        ss = tr[tr['side'] == side].reset_index(drop=True)
        sc = S.score(ss, rule, 'h5')
        f = ss[sc['ok'].values]
        bands = pd.Series([S.band(p) for p in f['fill']]).value_counts(normalize=True)
        print(f'{nm}: ' + '  '.join(f'{k}={v:.0%}' for k, v in bands.items()) +
              f'   median ADV20 {f["adv20"].median():,.0f}  median prev close ${f["prev_close"].median():.2f}')

    print('\n=== 6. the union book (both shorts together), per split and per month ===')
    rows = []
    for side, rule in (('up', 'fade'), ('down', 'continuation')):
        ss = tr[tr['side'] == side].reset_index(drop=True)
        sc = S.score(ss, rule, 'h5')
        k = sc[sc['ok']].copy()
        k['ssr'] = ss.loc[sc['ok'].values, 'ssr'].values
        rows.append(k)
    u = pd.concat(rows, ignore_index=True)
    for split, a, b in SPLITS:
        s = u[(u['day'] >= a) & (u['day'] <= b)]
        st = S.stats(s['net_R'].values)
        wg, nw = S.weeks_green(s)
        x = np.sort(s['net_R'].values)
        print(f'  {split:5s} n={st["n"]:4d}  {st["n"]/nw:5.1f}/wk  mean {st["mean"]:+.4f}R  t {st["t"]:+.2f}  '
              f'mde {st["mde"]:.3f}  weeks green {wg:.1%}  ex-top5% {S.stats(x[:int(len(x)*0.95)])["mean"]:+.4f}  '
              f'cap+3R {np.minimum(x,3.0).mean():+.4f}  median {np.median(x):+.4f}  win% {(x>0).mean():.1%}')
    u['month'] = u['day'].str[:7]
    mt = u.groupby('month')['net_R'].agg(['count', 'mean', 'sum']).round(3)
    print('\n  per-month (union book):')
    print(mt.to_string())
    u.to_csv(f'{HERE}/survivor_trades.csv', index=False)


if __name__ == '__main__':
    main()
