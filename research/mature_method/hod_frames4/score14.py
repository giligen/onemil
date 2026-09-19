#!/usr/bin/env python3
"""hod_frames4 / F14 — THE DAY AS THE UNIT.

Cells exactly as declared in PREREG.md §2 (commit 92db20a, before any cell was scored).
The observation is the trading DAY: y_d = the shipped 12/4 B2 book's P&L on day d at $100 risk,
no-trade days entered as 0 and kept in the denominator.  TEST sealed.  Read-only.  One process.
"""
import sqlite3, sys
import numpy as np, pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/mature_method/hod_frames4')
from common4 import ROOT, D4, S, SPLITS, book_ranked   # noqa: E402

PR = f'{ROOT}/research/mature_method/hod_preopen_regime'


def day_table(s):
    """The day universe (every session the break population covers) + the day fields."""
    days = pd.DataFrame({'day': sorted(s.day.unique())})
    days['split'] = S.split_of(days.day.values)
    days['wk'] = pd.to_datetime(days.day).dt.to_period('W-FRI').astype(str)
    df = pd.read_csv(f'{PR}/day_fields.csv', dtype={'day': str}, keep_default_na=False,
                     na_values=[''])
    days = days.merge(df[['day', 'spy_r5_pct', 'qqq_r5_pct', 'spy_gap_pct']], on='day', how='left')
    ix = pd.read_csv(f'{PR}/idx_1min.csv', dtype={'symbol': str, 'day': str},
                     keep_default_na=False, na_values=[''])
    sp5 = ix[(ix.symbol == 'SPY') & (ix.m_et >= 570) & (ix.m_et <= 574)].groupby('day').agg(
        hi=('h', 'max'), lo=('l', 'min'), op=('o', 'first'))
    sp5['rng5_pct'] = (sp5.hi - sp5.lo) / sp5.op * 100
    con = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=120)
    sd = pd.read_sql("select bar_date as day, open, high, low from daily_bars where symbol='SPY' "
                     "order by bar_date", con)
    con.close()
    sd['rngp'] = (sd.high - sd.low) / sd.open.replace(0, np.nan) * 100
    sd['atr20'] = sd.rngp.rolling(20, min_periods=15).mean().shift(1)
    sp5 = sp5.join(sd.set_index('day')[['atr20']])
    sp5['spy_rng5_atr'] = sp5.rng5_pct / sp5.atr20.replace(0, np.nan)
    days = days.merge(sp5[['rng5_pct', 'spy_rng5_atr']].reset_index(), on='day', how='left')
    br = s[s.entry_m <= 576].groupby('day').size().rename('breadth_0935')
    days = days.merge(br.reset_index(), on='day', how='left')
    days['breadth_0935'] = days.breadth_0935.fillna(0)
    days['h'] = np.where(days.split == 'VAL', 'VAL',
                         np.where(days.day < '2025-07-01', 'H1', 'H2'))
    return days


def day_row(name, days, mask, bk, out):
    """Day-level statistics of the gate `mask` over the day universe."""
    rows = []
    for sp in SPLITS:
        d = days[(days.split == sp) & mask.reindex(days.index).fillna(False)]
        y = d.y.values.astype(float)
        n = len(y)
        mu = y.mean() if n else np.nan
        se = y.std(ddof=1) / np.sqrt(n) if n > 2 else np.nan
        b = bk[bk.day.isin(set(d.day))]
        w = S.week_stats(b, sp)
        sub = {}
        for hh in (('H1', 'H2') if sp == 'TRAIN' else ('VAL',)):
            dd = d[d.h == hh]
            sub[hh] = (dd.y.mean() if len(dd) else np.nan, len(dd))
        rows.append(dict(cell=name, split=sp, n_days=n, day_mean=mu, day_t=mu / se if se else np.nan,
                         green_days=float((y > 0).mean() * 100) if n else np.nan,
                         total=w['total'], green=w['green'], per_wk=w['per_wk'], n=w['n'],
                         gross=w['gross'], net=w['net'], mdd=w['mdd'], worst=w['worst'],
                         h1=sub.get('H1', (np.nan, 0))[0], h2=sub.get('H2', (np.nan, 0))[0],
                         val=sub.get('VAL', (np.nan, 0))[0],
                         mde_day=2.80 * se if se == se else np.nan))
        print(f'| {name:<34s} | {sp:5s} | {n:4d} | {mu:+8.1f} | {rows[-1]["day_t"]:+5.2f} | '
              f'{rows[-1]["green_days"]:5.1f} | {w["n"]:5d} | {w["per_wk"]:5.1f} | {w["gross"]:+.3f}'
              f' | {w["green"]:5.1f} | {w["total"]:+8.0f} | {w["mdd"]:+8.0f} |', flush=True)
    t = rows[0]; v = rows[1]
    print(f'    day-mean halves  H1 {t["h1"]:+8.1f} | H2 {t["h2"]:+8.1f} | VAL {v["val"]:+8.1f} '
          f'-> same-signed POSITIVE: '
          f'{all(x == x and x > 0 for x in (t["h1"], t["h2"], v["val"]))}', flush=True)
    out.extend(rows)
    return rows


def main():
    s = pd.read_csv(f'{D4}/sig4.csv', dtype={'symbol': str, 'day': str, 'wk': str, 'split': str,
                                             'why': str}, keep_default_na=False, na_values=[''])
    bk = book_ranked(s, 12, 4)
    days = day_table(s)
    pnl = bk.groupby('day').pnl.sum()
    days['y'] = days.day.map(pnl).fillna(0.0)
    print(f'== hod_frames4 / F14 — the DAY as the unit ==')
    print(f'   day universe {len(days)} sessions: TRAIN {int((days.split=="TRAIN").sum())} '
          f'(H1 {int((days.h=="H1").sum())} / H2 {int((days.h=="H2").sum())}), VAL '
          f'{int((days.split=="VAL").sum())};  book rows {len(bk)}; '
          f'no-trade days {int((days.y==0).sum())}')
    print(f'   field coverage: spy_r5 {days.spy_r5_pct.notna().mean():.1%} | qqq_r5 '
          f'{days.qqq_r5_pct.notna().mean():.1%} | spy_rng5_atr {days.spy_rng5_atr.notna().mean():.1%}'
          f' | breadth_0935 100.0%')
    print('   IWM and the VIX open gap are NOT in this repo index tape (SPY/QQQ only) -> '
          'declared NOT SCORED.\n', flush=True)

    print('| cell                               | split | days | day $   |  t    | grn d | n     '
          '| /wk   | grossR | grn % | total $  | MDD $    |')
    print('|' + '|'.join(['-' * 6] * 12) + '|')
    out = []
    T = days[days.split == 'TRAIN']
    day_row('F14-d0 base (every day)', days, pd.Series(True, index=days.index), bk, out)
    day_row('F14-d1 spy_r5>0 [D2]', days, days.spy_r5_pct > 0, bk, out)
    q = [float(T.spy_r5_pct.quantile(x)) for x in (1 / 3, 2 / 3)]
    print(f'   [TRAIN terciles spy_r5_pct: {q[0]:+.4f} / {q[1]:+.4f}]')
    day_row('F14-d2a spy_r5 T1 (most down)', days, days.spy_r5_pct <= q[0], bk, out)
    day_row('F14-d2b spy_r5 T2', days, (days.spy_r5_pct > q[0]) & (days.spy_r5_pct <= q[1]), bk, out)
    day_row('F14-d2c spy_r5 T3 (most up)', days, days.spy_r5_pct > q[1], bk, out)
    day_row('F14-d3 qqq_r5>0', days, days.qqq_r5_pct > 0, bk, out)
    day_row('F14-d4 spy_r5>0 AND qqq_r5>0', days, (days.spy_r5_pct > 0) & (days.qqq_r5_pct > 0),
            bk, out)
    r = [float(T.spy_rng5_atr.quantile(x)) for x in (1 / 3, 2 / 3)]
    print(f'   [TRAIN terciles spy_rng5_atr: {r[0]:.4f} / {r[1]:.4f}]')
    day_row('F14-d5a spy_rng5_atr T1 (quiet)', days, days.spy_rng5_atr <= r[0], bk, out)
    day_row('F14-d5b spy_rng5_atr T2', days, (days.spy_rng5_atr > r[0]) & (days.spy_rng5_atr <= r[1]),
            bk, out)
    day_row('F14-d5c spy_rng5_atr T3 (wild)', days, days.spy_rng5_atr > r[1], bk, out)
    bq = [float(T.breadth_0935.quantile(x)) for x in (1 / 3, 2 / 3)]
    print(f'   [TRAIN terciles breadth_0935: {bq[0]:.0f} / {bq[1]:.0f}]')
    day_row('F14-d6a breadth T1 (narrow)', days, days.breadth_0935 <= bq[0], bk, out)
    day_row('F14-d6b breadth T2', days, (days.breadth_0935 > bq[0]) & (days.breadth_0935 <= bq[1]),
            bk, out)
    day_row('F14-d6c breadth T3 (broad)', days, days.breadth_0935 > bq[1], bk, out)

    # ---- F14-d7: the H2-2025 question ------------------------------------------------------
    print('\n\n== F14-d7 — THE H2-2025 QUESTION: is there ANY 09:35 state under which the flat '
          'half is positive? ==')
    states = {
        'spy_r5>0': days.spy_r5_pct > 0,
        'spy_r5 T3 (most up)': days.spy_r5_pct > q[1],
        'qqq_r5>0': days.qqq_r5_pct > 0,
        'spy_r5>0 AND qqq_r5>0': (days.spy_r5_pct > 0) & (days.qqq_r5_pct > 0),
        'spy_rng5_atr T3 (wild)': days.spy_rng5_atr > r[1],
        'breadth T3 (broad)': days.breadth_0935 > bq[1],
    }
    print('   SEARCH over 6 declared states, H2-2025 only; multiplicity = 6, stated before the read.')
    print('| state | H2 days | H2 day $ | H2 t | H2 green d % | H2 total $ | H1 day $ | VAL day $ |')
    print('|---|---|---|---|---|---|---|---|')
    h2best, h2bv = None, -1e18
    for nm, m in states.items():
        d2 = days[(days.h == 'H2') & m.reindex(days.index).fillna(False)]
        d1 = days[(days.h == 'H1') & m.reindex(days.index).fillna(False)]
        dv = days[(days.h == 'VAL') & m.reindex(days.index).fillna(False)]
        y = d2.y.values
        se = y.std(ddof=1) / np.sqrt(len(y)) if len(y) > 2 else np.nan
        print(f'| {nm} | {len(y)} | {y.mean():+.1f} | {y.mean()/se if se else np.nan:+.2f} | '
              f'{(y>0).mean()*100:.1f} | {y.sum():+.0f} | {d1.y.mean():+.1f} | {dv.y.mean():+.1f} |')
        if y.sum() > h2bv:
            h2best, h2bv = nm, y.sum()
    print(f'\n   best H2-2025 state = {h2best} ({h2bv:+.0f} over the half).')
    dv = days[(days.h == 'VAL') & states[h2best].reindex(days.index).fillna(False)]
    print(f'   its VAL read: {len(dv)} days, day mean {dv.y.mean():+.1f}, total {dv.y.sum():+.0f} '
          f'-> the frame answers YES only if this is non-negative.')

    # ---- F14-d8 / d9 -----------------------------------------------------------------------
    print('\n\n== F14-d8 the day gate x F13 s best causal ranking; F14-d9 the ORACLE day set ==')
    print('| cell                               | split | days | day $   |  t    | grn d | n     '
          '| /wk   | grossR | grn % | total $  | MDD $    |')
    print('|' + '|'.join(['-' * 6] * 12) + '|')
    s2 = s.copy()
    s2['dist_own'] = s2.dist_open_pct / s2.med_rng.replace(0, np.nan)
    s2['spr_r'] = s2.sp_pct / s2.r_pct.clip(lower=0.05)
    tr = s2[s2.split == 'TRAIN']
    comp = sum(sgn * ((s2[f] - float(tr[f].mean())) / float(tr[f].std(ddof=1))).fillna(0.0)
               for f, sgn in (('rv_profile', 1), ('dollar_frac', 1), ('dist_own', 1), ('spr_r', -1)))
    bk_r5 = book_ranked(s2, 12, 4, score=comp, descending=True)
    pnl_r5 = bk_r5.groupby('day').pnl.sum()
    days_r5 = days.copy(); days_r5['y'] = days_r5.day.map(pnl_r5).fillna(0.0)
    day_row('F14-d8 spy_r5>0 x rank composite', days_r5, days_r5.spy_r5_pct > 0, bk_r5, out)
    day_row('F14-d9 ORACLE day set (y>0) [bound]', days, days.y > 0, bk, out)

    pd.DataFrame(out).to_csv(f'{D4}/cells14.csv', index=False)

    # ---- nulls on the day gates ------------------------------------------------------------
    print('\n\n== count-matched permutation null on green WEEKS (2,000 draws) ==')
    print('| cell | split | observed green % | null mean | [p5, p95] | outside? |')
    print('|---|---|---|---|---|---|')
    nulls = []
    gates = {'F14-d0 base (every day)': pd.Series(True, index=days.index),
             'F14-d1 spy_r5>0 [D2]': days.spy_r5_pct > 0,
             'F14-d2c spy_r5 T3 (most up)': days.spy_r5_pct > q[1],
             'F14-d3 qqq_r5>0': days.qqq_r5_pct > 0,
             'F14-d4 spy_r5>0 AND qqq_r5>0': (days.spy_r5_pct > 0) & (days.qqq_r5_pct > 0),
             'F14-d5c spy_rng5_atr T3 (wild)': days.spy_rng5_atr > r[1],
             'F14-d6c breadth T3 (broad)': days.breadth_0935 > bq[1]}
    for nm, m in gates.items():
        dd = set(days[m.reindex(days.index).fillna(False)].day)
        b = bk[bk.day.isin(dd)]
        for sp in SPLITS:
            obs, mu_, p5, p95 = S.null_band(b, sp)
            o = ('ABOVE' if obs == obs and obs > p95 else
                 ('below' if obs == obs and obs < p5 else 'inside'))
            print(f'| {nm} | {sp} | {obs:.1f} | {mu_:.1f} | [{p5:.1f}, {p95:.1f}] | {o} |')
            nulls.append(dict(cell=nm, split=sp, obs=obs, mu=mu_, p5=p5, p95=p95, outside=o))
    pd.DataFrame(nulls).to_csv(f'{D4}/nulls14.csv', index=False)

    cf = pd.DataFrame(out)
    print('\n== F14 PRE-COMMITTED SELECTOR (day-level mean positive in H1, H2 and VAL; day t>=2 on '
          'TRAIN; >=50% green weeks both splits) ==')
    sel = []
    for nm in cf.cell.unique():
        t_ = cf[(cf.cell == nm) & (cf.split == 'TRAIN')].iloc[0]
        v_ = cf[(cf.cell == nm) & (cf.split == 'VAL')].iloc[0]
        ok = (t_.h1 > 0 and t_.h2 > 0 and v_.val > 0 and t_.day_t >= 2.0 and
              t_.green >= 50 and v_.green >= 50 and min(t_.per_wk, v_.per_wk) >= 10)
        if ok:
            sel.append(nm)
    print(f'  -> {len(sel)}: {sel if sel else "NONE"}')
    print('\n== day-level MDE (80% power, $ per day) ==')
    for nm in cf.cell.unique():
        row = ' '.join(f'{r.split} n {int(r.n_days)} MDE ${r.mde_day:.0f}'
                       for _, r in cf[cf.cell == nm].iterrows())
        print(f'  {nm:36s} {row}')
    print('\nDONE', flush=True)


if __name__ == '__main__':
    main()
