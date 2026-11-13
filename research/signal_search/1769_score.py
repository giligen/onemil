"""Cell 1,769 scorer (PREREG_1769, frozen 2026-11-13 21:10 UTC): short the >=10 % gapper at the 09:35 bar close (E1),
stop = (max high 09:30-09:35) * (1 + 50 bp) filled at max(stop, that bar's open) when a later bar's high >= stop, cover at
the 10:30 close.  'Price at HH:MM' = close of the bar that ENDS at HH:MM (stamp HH:MM-1), as in 1768.  Minute bars cannot
time intrabar: a stop and the cover bar resolve to the stop.  Cost = 2 bp/leg + measured entry+exit half-spread from the
1748_cost.csv sample by price band (median by band; sample = 400 extreme-loser name-days 2021-23, quoted at the next-day open
and close -- NOT 09:35/10:30, NOT per-trade NBBO: stated caveat) + borrow 50 bp (100 bp sensitivity).
Sensitivity (reported, not searched): 1768 stop convention (trigger at H, fill max(H,open)*1.005); cover 10:00.
TRAIN table is written before VAL is read.  Usage: cd /home/ec2-user/onemil && ulimit -v 2500000 && nice -n 19 timeout 600 python3 research/signal_search/1769_score.py
"""
import os, importlib.util
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, '1769_cache')
OPEN_M, ENTRY_STAMP, SLIP, FEE, BORROW = 570, 574, 50.0, 2.0, 50.0
COVER = {'1030': 629, '1000': 599}
BANDS = [0, 10, 20, 50, 1e9]


def log(m):
    print(m, flush=True)


def cl_t(x, d):
    """(mean, day-clustered t, clustered SE) of x grouped by day labels d."""
    n = len(x); mu = x.mean(); e = pd.Series(x - mu).groupby(np.asarray(d)).sum()
    se = np.sqrt((e ** 2).sum()) / n
    return mu, (mu / se if se > 0 else np.nan), se


def ex_top5(x):
    """Mean after dropping the top 5 % of trades by net (1754 convention)."""
    x = np.sort(np.asarray(x)); k = int(np.ceil(0.05 * len(x)))
    return x[:len(x) - k].mean() if len(x) > k else np.nan


def cost_bands():
    """Median entry / exit half-spread (bp) by price band from 1748_cost.csv joined to 1748_events (c_t price); bands <10,10-20,20-50,50+."""
    c = pd.read_csv(os.path.join(HERE, '1748_cost.csv'))
    e = pd.read_parquet(os.path.join(HERE, '1748_events.parquet'), columns=['symbol', 'date', 'c_t'])
    m = c.merge(e, on=['symbol', 'date'], how='left')
    log(f'cost sample {len(c)}, price joined {m.c_t.notna().sum()}')
    m = m[m.c_t.notna()]
    m['band'] = pd.cut(m.c_t, BANDS, labels=False, right=False)
    tab = m.groupby('band').agg(n=('c_t', 'size'), entry=('entry_hs_bp', 'median'), exit=('exit_hs_bp', 'median'))
    log('half-spread band table (bp):\n' + tab.round(1).to_string())
    tab = tab.reindex(range(len(BANDS) - 1)).ffill().bfill()
    return tab, (lambda p: tab.iloc[int(np.clip(np.searchsorted(BANDS, p, side='right') - 1, 0, len(tab) - 1))])


def trade_rows(day, row, g, band):
    """Per symbol-day: dict with bucket flag + list of trade variants (cover x stop convention)."""
    m = g.m.to_numpy(); o, h, c = g.o.to_numpy(), g.h.to_numpy(), g.c.to_numpy()
    ent = np.flatnonzero((m >= OPEN_M) & (m <= ENTRY_STAMP))
    if not len(ent):
        return 'no_open_bars', []
    H = float(h[ent].max()); entry = float(c[ent[-1]]); b = band(float(o[ent[0]]))
    hs = float(b['entry'] + b['exit'])
    flag = 'ok' if m[ent[0]] == OPEN_M else 'first_bar_late'
    out = []
    for cn, cm in COVER.items():
        win = np.flatnonzero((m > ENTRY_STAMP) & (m <= cm))
        for conv in ('prereg', 'c1768'):
            trig = H * (1 + SLIP / 1e4) if conv == 'prereg' else H
            hit = win[h[win] >= trig] if len(win) else win
            if len(hit):
                j = hit[0]; px = max(trig, float(o[j])) * (1 + (SLIP / 1e4 if conv == 'c1768' else 0)); st = 1
            else:
                k = np.flatnonzero(m <= cm)
                if not len(k) or m[k[-1]] <= ENTRY_STAMP:
                    continue            # no bar after entry -> no cover price
                px = float(c[k[-1]]); st = 0
            gross = (entry - px) / entry * 1e4
            out.append(dict(day=day, symbol=row.symbol, cover=cn, conv=conv, gross=gross, net=gross - 4 * FEE / 2 - hs - BORROW,
                            stop=st, sd=((H * (1 + SLIP / 1e4)) / entry - 1) * 1e4, hs_bp=hs, flag=flag, price=entry, gap=row.gap))
    return flag, out


def build(P, days, band):
    """Walk the cached day files; return (trade frame, bucket counts, fetched-any count)."""
    rows, buckets, anybars = [], {'no_bars_at_all': 0, 'no_open_bars': 0, 'first_bar_late': 0, 'ok': 0}, 0
    for k, d in enumerate(days):
        b = pd.read_parquet(os.path.join(CACHE, f'{d}.parquet'))
        et = pd.to_datetime(b.t, utc=True).dt.tz_convert('America/New_York')
        b = b.assign(m=et.dt.hour * 60 + et.dt.minute).sort_values(['symbol', 'm'])
        grp = {s: g for s, g in b.groupby('symbol', sort=False)}
        for row in P[P.date == d].itertuples():
            g = grp.get(row.symbol)
            if g is None:
                buckets['no_bars_at_all'] += 1; continue
            anybars += 1
            flag, out = trade_rows(d, row, g, band)
            buckets[flag] += 1; rows += out
        if k % 300 == 0:
            log(f'  day {k + 1}/{len(days)} {d}, trades {len(rows)}')
    return pd.DataFrame(rows), buckets, anybars


def stats(s, nweeks, wk_all, rng):
    """PREREG numbers for one trade frame (single cover x convention)."""
    x, d = s.net.to_numpy(), s.day.to_numpy()
    mu, tt, se = cl_t(x, d)
    dm = s.groupby('day').net.mean().sort_values().to_numpy()
    k5, k1 = max(1, int(np.floor(len(dm) * .05))), max(1, int(np.floor(len(dm) * .01)))
    wk = s.groupby('wk').net.mean().reindex(sorted(wk_all), fill_value=0.0)
    cnt = s.groupby('wk').size(); wm = s.groupby('wk').net.mean()
    green = float((wm > 0).mean())
    z = x - x.mean(); sims = []                       # count-matched null: week counts kept, trades drawn from the demeaned pool
    for _ in range(300):
        sims.append(np.mean([rng.choice(z, size=n).mean() > 0 for n in cnt.to_numpy()]))
    return dict(n=len(x), n_wk=len(x) / nweeks, mean_net=mu, day_t=tt, hit=float((x > 0).mean()), ex_top5=ex_top5(x),
                worst1_day=float(dm[:k1].mean()), worst5_day=float(dm[:k5].mean()), max_loss=float(x.min()), stop_hit=float(s.stop.mean()),
                wk_p10=float(np.percentile(wk, 10)), green_wk=green, green_null=float(np.mean(sims)), net_borrow100=mu - BORROW,
                MDE80=2.8 * se, gross=float(s.gross.mean()), hs_bp=float(s.hs_bp.mean()),
                **{f'y{y}': float(s[s.day.str[:4] == str(y)].net.mean()) if (s.day.str[:4] == str(y)).any() else np.nan for y in range(2016, 2024)})


def main():
    """TRAIN table first, then VAL once, then the kill checks on the primary (E1, cover 10:30, PREREG stop)."""
    E = pd.read_parquet(os.path.join(HERE, '1769_events.parquet'))
    Dy = pd.read_parquet(os.path.join(HERE, '1769_days.parquet'))
    log('events gap>=10 % per year: ' + str(E.groupby('year').agg(all=('gap', 'size'), raw5=('raw_ok', 'sum')).to_dict('index')))
    P = E[E.raw_ok].copy()
    days = sorted(P.date.unique()); missing = [d for d in days if not os.path.exists(os.path.join(CACHE, f'{d}.parquet'))]
    assert not missing, f'{len(missing)} days not fetched: rerun 1769_fetch.py'
    tab, band = cost_bands()
    band2 = lambda p: band(p)
    wkl = lambda dd: pd.to_datetime(pd.Series(list(dd))).dt.strftime('%G-%V')
    Dy['wk'] = wkl(Dy.date).values
    rng = np.random.default_rng(1769)
    out, ev_out = [], []
    for split, (a, bnd) in (('TRAIN', ('2016-01-01', '2020-12-31')), ('VAL', ('2021-01-01', '2023-12-31'))):
        if split == 'VAL':
            pd.DataFrame(out).to_csv(os.path.join(HERE, '1769_summary.csv'), index=False); log('TRAIN table written; reading VAL once')
        Ps = P[(P.date >= a) & (P.date <= bnd)]
        t, bk, anyb = build(Ps, sorted(Ps.date.unique()), band2)
        t['wk'] = pd.to_datetime(t.day).dt.strftime('%G-%V')
        cov = anyb / len(Ps)
        log(f'{split}: events {len(Ps)}, fetched-any-bars coverage {cov:.1%}; buckets {bk}')
        ev_out.append(dict(split=split, events=len(Ps), coverage_any=cov, **bk))
        parts = [('H1_16_18', t[t.day < '2019-01-01'], '2016-01-01', '2018-12-31'), ('H2_19_20', t[t.day >= '2019-01-01'], '2019-01-01', '2020-12-31')] \
            if split == 'TRAIN' else []
        parts.append((split, t, a, bnd))
        for name, ts, a2, b2 in parts:
            dd = Dy[(Dy.date >= a2) & (Dy.date <= b2)]
            for cn in COVER:
                for conv in ('prereg', 'c1768'):
                    s = ts[(ts.cover == cn) & (ts.conv == conv)]
                    if len(s) >= 5:
                        out.append(dict(sample=name, cover=cn, conv=conv, **stats(s, len(dd) / 5.0, set(dd.wk), rng)))
        t.to_parquet(os.path.join(HERE, f'1769_trades_{split}.parquet'))
    S = pd.DataFrame(out); S.to_csv(os.path.join(HERE, '1769_summary.csv'), index=False)
    pd.DataFrame(ev_out).to_csv(os.path.join(HERE, '1769_coverage.csv'), index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
    log(S[(S.cover == '1030') & (S.conv == 'prereg')].round(2).T.to_string())
    log(S[~((S.cover == '1030') & (S.conv == 'prereg'))][['sample', 'cover', 'conv', 'n', 'mean_net', 'day_t', 'stop_hit', 'net_borrow100']].round(1).to_string())
    q = lambda smp: S[(S['sample'] == smp) & (S.cover == '1030') & (S.conv == 'prereg')].iloc[0]
    k, h1, h2 = q('VAL'), q('H1_16_18'), q('H2_19_20')
    kills = {'VAL net<=50bp': k.mean_net <= 50, 'day-t<2': k.day_t < 2, 'TRAIN halves opposite': np.sign(h1.mean_net) != np.sign(h2.mean_net),
             'ex-top5<=0': k.ex_top5 <= 0, 'worst5 day<-800bp': k.worst5_day < -800, '2022<0': k.y2022 < 0, '<3 names/wk': k.n_wk < 3,
             'net@borrow100<=0': k.net_borrow100 <= 0}
    log(f'KILL CHECKS {kills} => {"KILL" if any(kills.values()) else "SURVIVE"}')


if __name__ == '__main__':
    main()
