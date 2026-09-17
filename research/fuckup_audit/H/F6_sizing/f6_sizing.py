#!/usr/bin/env python3
"""Stage H/F6_sizing — how much money the F6 red-to-green book can carry.

Population/cost/book contract: exactly B/score5.py's `next` fill, reproduced in f6_load.py.
Liquidity: f6_bars.py (fill-bar and trailing-5-minute dollar volume, builder's bar-source precedence).

Outputs: REPORT tables to stdout (captured into REPORT.md) and the per-trade CSV trades_f6_sizing.csv.
"""
import os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
from trading.hod_break import run_book

D = 'research/fuckup_audit/H/F6_sizing'
EXIT_RATIO = {'stop': 0.875, 'lock': 0.875, 'eod': 0.412, 'target': 0.875, 'none': 0.875}
ENTRY_MULT = 0.25
RISKS = [100, 250, 400, 700, 1000, 2000]
EQUITY, DTBP = 66_000.0, 264_000.0
EXITS = {'hold': ('next_rr_hold', 'next_why_hold', 'next_exit_m_hold'),
         '2R':   ('next_rr_2r',   'next_why_2r',   'next_exit_m_2r')}
SPLITS = ['TRAIN', 'VAL', 'TEST']
OUT = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    OUT.append(s); print(s, flush=True)


def net_r(d, rr, why, mult=1.0):
    half = 0.5 * (d.spread_cc_bps * mult / 100.0) / d.next_r_pct.clip(lower=0.05)
    return d[rr] - ENTRY_MULT * half - half * d[why].map(EXIT_RATIO).fillna(0.875)


def book(d, exit_name, mult=1.0):
    rr, why, xm = EXITS[exit_name]
    x = d.copy()
    x['net'] = net_r(x, rr, why, mult)
    x['xm'] = x[xm]
    x = x[x.net.notna() & x.xm.notna()]
    rows = [(r.day, int(r.next_entry_m), int(r.xm), r.symbol, float(r.net), int(r.Index)) for r in x.itertuples()]
    t = pd.DataFrame(run_book(rows, 12, 4), columns=['day', 'em', 'xm', 'symbol', 'net', 'idx'])
    return t


def wk_stats(t, weeks):
    w = t.groupby('wk').net.sum().reindex(weeks).fillna(0.0)
    sd = t.net.std(ddof=1)
    return dict(n=len(t), meanR=t.net.mean(), t=t.net.mean() / (sd / np.sqrt(len(t))) if sd > 0 else 0.0,
                wkR=float(w.mean()), wkSE=float(w.std(ddof=1) / np.sqrt(len(w))), green=float((w > 0).mean()),
                worst=float(w.min()), nweeks=len(w))


def concurrent_notional(t, notional):
    """Gross $ exposure at each entry: this trade plus every booked trade still open at its entry minute."""
    res = []
    for day, gg in t.groupby('day'):
        gg = gg.sort_values(['em', 'symbol'])
        for r in gg.itertuples():
            open_now = gg[(gg.em <= r.em) & (gg.xm >= r.em)]
            res.append(notional.loc[open_now.idx].sum())
    return np.array(res)


def main():
    d = pd.read_csv(f'{D}/pop_f6_liq.csv', dtype={'symbol': str, 'day': str},
                    keep_default_na=False, na_values=[''])
    d['split'] = np.where(d.day < '2026-01-01', 'TRAIN', np.where(d.day < '2026-06-01', 'VAL', 'TEST'))
    d['wk'] = pd.to_datetime(d.day).dt.to_period('W-FRI').astype(str)
    d['risk_per_share'] = d.next_entry * d.next_r_pct / 100.0
    weeks = {s: sorted(d.loc[d.split == s, 'wk'].unique()) for s in SPLITS}

    say('## 0. Reproduction of the C numbers (must match C/score5_results.csv before anything else)')
    say('')
    say('| exit | split | n | mean net R | t | ref (C) |')
    say('|---|---|---:|---:|---:|---|')
    ref = {('hold', 'TRAIN'): '+0.0510 / t 1.34', ('hold', 'VAL'): '+0.1629 / t 2.46',
           ('2R', 'TRAIN'): '+0.0273 / t 1.05', ('2R', 'VAL'): '+0.0702 / t 1.72'}
    books = {}
    for e in EXITS:
        for s in SPLITS:
            t = book(d[d.split == s], e)
            t['wk'] = d.loc[t.idx, 'wk'].values
            books[(e, s)] = t
            st = wk_stats(t, weeks[s])
            say(f"| {e} | {s} | {st['n']} | {st['meanR']:+.4f} | {st['t']:+.2f} | "
                f"{ref.get((e, s), '(TEST - not in the C table)')} |")
    say('')

    # ---- the pooled booked set (hold exit is the headline book; the fill bar is exit-independent)
    pooled = pd.concat([books[('hold', s)].assign(split=s) for s in SPLITS], ignore_index=True)
    b = d.loc[pooled.idx].copy().reset_index(drop=True)
    b['split'] = pooled.split.values
    b['net'] = pooled.net.values
    b2 = d.loc[pd.concat([books[('2R', s)] for s in SPLITS], ignore_index=True).idx].copy()
    n = len(b)
    say(f'## 1. The booked set measured (hold exit, all three splits pooled): n = {n}')
    say(f"(TRAIN {int((b.split=='TRAIN').sum())} / VAL {int((b.split=='VAL').sum())} / "
        f"TEST {int((b.split=='TEST').sum())}; the 2R book is n = {len(b2)}. "
        f'Liquidity is a property of the FILL BAR, identical under either exit.)')
    say('')
    say('| quantity | p10 | median | mean | p90 |')
    say('|---|---:|---:|---:|---:|')
    for lbl, col, nd in [('entry price $', b.next_entry, 2), ('R per share $', b.risk_per_share, 2),
                         ('R as % of price', b.next_r_pct, 2), ('fill-bar $ volume', b.fill_bar_dollar_vol, 0),
                         ('5-min $ volume (bars t-4..t)', b.five_min_dollar_vol, 0),
                         ('ADV20 (shares)', b.adv20, 0), ('spread_cc_bps', b.spread_cc_bps, 1)]:
        say(f'| {lbl} | {col.quantile(.10):,.{nd}f} | {col.quantile(.50):,.{nd}f} | {col.mean():,.{nd}f} | '
            f'{col.quantile(.90):,.{nd}f} |')
    say('')
    say(f'bars with a print in all 5 of the 5 minutes: {(b.n_bars_in_5 == 5).mean():.1%}; '
        f'4 or fewer: {(b.n_bars_in_5 < 5).mean():.1%}. Obtainability: the next-open fill lies inside its own bar '
        f'on {b.fill_in_bar.mean():.2%} of booked trades.')
    say('')

    # ---- participation
    for R in RISKS:
        b[f'sh_{R}'] = R / b.risk_per_share
        b[f'part_{R}'] = b[f'sh_{R}'] * b.next_entry / b.five_min_dollar_vol
        b[f'partfill_{R}'] = b[f'sh_{R}'] * b.next_entry / b.fill_bar_dollar_vol
        b[f'not_{R}'] = b[f'sh_{R}'] * b.next_entry

    say('## 2. Participation = shares x entry / 5-minute $ volume (booked trades, n = %d)' % n)
    say('')
    say('| risk $/trade | median shares | p<=1% | p<=2% | p<=5% | median participation | p90 participation | '
        'median part. of the FILL BAR alone |')
    say('|---:|---:|---:|---:|---:|---:|---:|---:|')
    for R in RISKS:
        p = b[f'part_{R}']
        say(f'| {R} | {b[f"sh_{R}"].median():,.0f} | {(p <= .01).mean():.1%} | {(p <= .02).mean():.1%} | '
            f'{(p <= .05).mean():.1%} | {p.median():.2%} | {p.quantile(.90):.2%} | '
            f'{b[f"partfill_{R}"].median():.2%} |')
    say('')

    # ---- notional vs DTBP
    say('### 2b. The implied capacity of each trade: the risk $ at which participation hits exactly 1%')
    say('')
    say('cap_risk$ = 1% x (5-min $ volume) / entry x (R per share). It is the largest risk-per-trade that trade '
        'could have carried at 1% of the 5-minute tape.')
    say('')
    say('| set | n | p10 | p25 | median | p75 | p90 | share of trades carrying >= $400 |')
    say('|---|---:|---:|---:|---:|---:|---:|---:|')
    b['cap_risk'] = 0.01 * b.five_min_dollar_vol / b.next_entry * b.risk_per_share
    for lbl, g in [('all booked', b)] + [(s_, b[b.split == s_]) for s_ in SPLITS] + \
                  [('winners (net R > 0)', b[b.net > 0]), ('losers (net R <= 0)', b[b.net <= 0])]:
        say(f'| {lbl} | {len(g)} | {g.cap_risk.quantile(.10):,.0f} | {g.cap_risk.quantile(.25):,.0f} | '
            f'{g.cap_risk.median():,.0f} | {g.cap_risk.quantile(.75):,.0f} | {g.cap_risk.quantile(.90):,.0f} | '
            f'{(g.cap_risk >= 400).mean():.1%} |')
    say('')

    say('## 3. Notional per trade and 4-concurrent gross exposure vs day-trading buying power')
    say('')
    say(f'Assumption: account equity ~ ${EQUITY:,.0f}, pattern-day-trader margin 4x -> '
        f'DTBP ~ ${DTBP:,.0f}. "4-concurrent" is the REAL gross exposure measured at every booked entry '
        f'(this trade plus every booked trade still open at that minute, run_book 12/4), not 4 x the median.')
    say('')
    say('| risk $/trade | notional median | notional p90 | notional max | concurrent median | concurrent p90 | '
        'concurrent max | % of entries over DTBP |')
    say('|---:|---:|---:|---:|---:|---:|---:|---:|')
    conc_store = {}
    for R in RISKS:
        notional = pd.Series(b[f'not_{R}'].values, index=b.index)
        cn = concurrent_notional(pooled.assign(idx=b.index), notional)
        conc_store[R] = cn
        say(f'| {R} | {b[f"not_{R}"].median():,.0f} | {b[f"not_{R}"].quantile(.90):,.0f} | '
            f'{b[f"not_{R}"].max():,.0f} | {np.median(cn):,.0f} | {np.percentile(cn, 90):,.0f} | '
            f'{cn.max():,.0f} | {(cn > DTBP).mean():.1%} |')
    say('')

    # ---- weekly $ , as-is and liquidity-capped
    say('## 4. Expected weekly $ = weekly R x risk, as-is and liquidity-capped at 1% participation')
    say('')
    say('The capped book drops every POPULATION row whose participation at that risk level exceeds 1% and '
        're-runs run_book(12,4) on the survivors, so a freed slot refills.')
    say('')
    rows = []
    for e in EXITS:
        for R in RISKS:
            line = {'exit': e, 'risk': R}
            for s in ('TRAIN', 'VAL'):
                st = wk_stats(books[(e, s)].assign(wk=d.loc[books[(e, s)].idx, 'wk'].values), weeks[s])
                line[f'{s}_wkR'] = st['wkR']; line[f'{s}_tpw'] = st['n'] / st['nweeks']
                line[f'{s}_usd'] = st['wkR'] * R
            # capped: (a) the as-measured window t-4..t, (b) the CAUSAL window t-4..t-1 scaled x5/4 (the fill
            # bar's own volume is not known when the order is sent, so a live cap can only use prior bars)
            dd = d.copy()
            dd['part'] = (R / dd.risk_per_share) * dd.next_entry / dd.five_min_dollar_vol
            prior4 = (dd.five_min_dollar_vol - dd.fill_bar_dollar_vol).clip(lower=1.0) * 1.25
            dd['part_causal'] = (R / dd.risk_per_share) * dd.next_entry / prior4
            for tag, mask in (('cap', dd.part <= 0.01), ('capc', dd.part_causal <= 0.01)):
                surv = dd[mask]
                if tag == 'cap':
                    line['kept_pop'] = len(surv) / len(dd)
                else:
                    line['kept_pop_causal'] = len(surv) / len(dd)
                for s in ('TRAIN', 'VAL'):
                    tc = book(surv[surv.split == s], e)
                    if len(tc) < 20:
                        line[f'{s}_{tag}_wkR'] = np.nan; line[f'{s}_{tag}_usd'] = np.nan
                        line[f'{s}_{tag}_tpw'] = np.nan; line[f'{s}_{tag}_meanR'] = np.nan
                        continue
                    tc['wk'] = dd.loc[tc.idx, 'wk'].values
                    stc = wk_stats(tc, weeks[s])
                    line[f'{s}_{tag}_wkR'] = stc['wkR']; line[f'{s}_{tag}_tpw'] = stc['n'] / stc['nweeks']
                    line[f'{s}_{tag}_usd'] = stc['wkR'] * R
                    line[f'{s}_{tag}_meanR'] = stc['meanR']
            rows.append(line)
    W = pd.DataFrame(rows)
    say('| exit | risk $ | TRAIN wkR | TRAIN $/wk | VAL wkR | VAL $/wk | pop kept @1% | TRAIN capped wkR | '
        'TRAIN capped $/wk | VAL capped wkR | VAL capped $/wk | capped trades/wk TRAIN / VAL |')
    say('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|')
    for r in W.itertuples():
        say(f'| {r.exit} | {r.risk} | {r.TRAIN_wkR:.2f} | {r.TRAIN_usd:,.0f} | {r.VAL_wkR:.2f} | '
            f'{r.VAL_usd:,.0f} | {r.kept_pop:.1%} | {r.TRAIN_cap_wkR:.2f} | {r.TRAIN_cap_usd:,.0f} | '
            f'{r.VAL_cap_wkR:.2f} | {r.VAL_cap_usd:,.0f} | {r.TRAIN_cap_tpw:.1f} / {r.VAL_cap_tpw:.1f} |')
    say('')
    say('The same cap made LIVE-COMPUTABLE (the fill bar\'s own volume is unknown when the order is sent, so the '
        'rule uses the four bars t-4..t-1 scaled x1.25):')
    say('')
    say('| exit | risk $ | pop kept @1% causal | TRAIN causal wkR | TRAIN causal $/wk | VAL causal wkR | '
        'VAL causal $/wk | causal trades/wk TRAIN / VAL |')
    say('|---|---:|---:|---:|---:|---:|---:|---|')
    for r in W.itertuples():
        say(f'| {r.exit} | {r.risk} | {r.kept_pop_causal:.1%} | {r.TRAIN_capc_wkR:.2f} | '
            f'{r.TRAIN_capc_usd:,.0f} | {r.VAL_capc_wkR:.2f} | {r.VAL_capc_usd:,.0f} | '
            f'{r.TRAIN_capc_tpw:.1f} / {r.VAL_capc_tpw:.1f} |')
    say('')
    W.to_csv(f'{D}/weekly_dollars.csv', index=False)

    # ---- spread sensitivity
    say('## 5. Spread-cost sensitivity (the untraded names may quote wider than the cost curve)')
    say('')
    say('run_book ignores the payload, so the booked SET is unchanged; only the charge moves.')
    say('')
    say('| exit | split | mean net R x1.0 | x1.5 | x2.0 | weekly R x1.0 | x1.5 | x2.0 | mean spread charge (R) |')
    say('|---|---|---:|---:|---:|---:|---:|---:|---:|')
    for e in EXITS:
        for s in SPLITS:
            cells = []
            for m in (1.0, 1.5, 2.0):
                t = book(d[d.split == s], e, m)
                t['wk'] = d.loc[t.idx, 'wk'].values
                cells.append(wk_stats(t, weeks[s]))
            chg = cells[0]['meanR'] - cells[1]['meanR']
            say(f'| {e} | {s} | {cells[0]["meanR"]:+.4f} | {cells[1]["meanR"]:+.4f} | {cells[2]["meanR"]:+.4f} | '
                f'{cells[0]["wkR"]:+.2f} | {cells[1]["wkR"]:+.2f} | {cells[2]["wkR"]:+.2f} | '
                f'{2*chg:.4f} |')
    say('')

    # ---- where the capacity lives
    say('## 6. Where the capacity lives — participation by price band and by ADV20 band ($400 risk)')
    say('')
    pb = pd.cut(b.next_entry, [5, 10, 20, 50, 100, 1e9], right=False,
                labels=['$5-10', '$10-20', '$20-50', '$50-100', '$100+'])
    ab = pd.cut(b.adv20, [0, 5e5, 2e6, 1e7, 5e7, 1e12], right=False,
                labels=['<500K', '500K-2M', '2M-10M', '10M-50M', '50M+'])
    for lbl, band in [('price band', pb), ('ADV20 band', ab)]:
        say(f'### by {lbl}')
        say('')
        say('| band | n | share of book | median 5-min $vol | median part. @$400 | p<=1% @$400 | '
            'p<=1% @$1000 | p<=1% @$2000 | mean net R | max risk $ at 1% for the MEDIAN trade |')
        say('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
        for k, g in b.groupby(band, observed=True):
            cap = (0.01 * g.five_min_dollar_vol / g.next_entry * g.risk_per_share).median()
            say(f'| {k} | {len(g)} | {len(g)/n:.1%} | {g.five_min_dollar_vol.median():,.0f} | '
                f'{g.part_400.median():.2%} | {(g.part_400 <= .01).mean():.0%} | '
                f'{(g.part_1000 <= .01).mean():.0%} | {(g.part_2000 <= .01).mean():.0%} | '
                f'{g.net.mean():+.3f} | {cap:,.0f} |')
        say('')
        say(f'(n missing {lbl}: {int(band.isna().sum())})')
        say('')

    # ---- per-trade CSV
    cols = (['day', 'symbol', 'split', 'wk', 'sig_m', 'next_entry_m', 'next_entry', 'stop', 'next_r_pct',
             'risk_per_share', 'spread_cc_bps', 'adv20', 'rv_adv', 'fill_bar_vol', 'fill_bar_dollar_vol',
             'five_min_dollar_vol', 'n_bars_in_5', 'fill_in_bar', 'net']
            + [f'{p}_{R}' for R in RISKS for p in ('sh', 'not', 'part')])
    b[cols].to_csv(f'{D}/trades_f6_sizing.csv', index=False)
    say(f'per-trade CSV: `{D}/trades_f6_sizing.csv` ({len(b)} booked hold-exit trades, all columns above)')
    open(f'{D}/tables.md', 'w').write('\n'.join(OUT) + '\n')


if __name__ == '__main__':
    main()
