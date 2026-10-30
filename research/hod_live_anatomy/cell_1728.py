#!/usr/bin/env python3
"""Cell 1,728 (EXPLORATORY, not a PREREG'd test): why do the live HOD paper fills (+0.075 R gross, n 231,
fills_1721.csv) beat the BT population of the same rule (5,506 fills, 1663_features.csv, the >=1.5 % stop floor)?

Steps (all read-only):
  1. put both books on ONE R basis (entry at the level, R = level - stop, exits as recorded) and decompose the
     quoted "+0.2 R gap" into cost-model, fill-basis and gross parts;
  2. characterise both populations on every shared causal dimension (price, R %, time, weekday, gap, extension
     above the open, prior-day return, ADV $, ATR %, fills/day, rank in day) with quantiles, and the BT's R by bin;
  3. apply the live engine's gates the BT lacks (4 concurrent / 12 per day, first-come by fill time) to the BT;
  4. score explicit candidate rules: TRAIN-H2 / VAL kept n, net R, gross R, day-clustered t, ex-top-5 %,
     alternating-week halves, and the share / R of the live fills each rule keeps.
Daily context comes from data/cache.db daily_bars opened read-only (mode=ro, immutable=1) for BOTH books.
Outputs: research/hod_live_anatomy/cell_1728_out.txt, cell_1728_bt.csv, cell_1728_live.csv."""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
CACHE_DB = os.path.join(ROOT, 'data', 'cache.db')
OUT_TXT = os.path.join(HERE, 'cell_1728_out.txt')
_out = open(OUT_TXT, 'w')


def log(*a):
    """Print to stdout (flushed) and to the cell's output file."""
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    _out.write(s + '\n')
    _out.flush()


def load_bt() -> pd.DataFrame:
    """The 5,506-fill BT population (1663) joined 1:1 on (date, symbol) to causal_arming_causal.csv status==fill
    for level / fill / exit minute / n_cross; adds the level-basis gross R."""
    a = pd.read_csv(os.path.join(ROOT, 'research/hod_entry/1663_features.csv'))
    c = pd.read_csv(os.path.join(ROOT, 'research/hod_entry/causal_arming_causal.csv'), low_memory=False)
    c = c[c.status == 'fill'].rename(columns={'day': 'date'})
    b = a.merge(c[['date', 'symbol', 'level', 'fill', 'exit_m', 'n_cross', 'raw_R', 'cost_R', 'net_R']]
                .rename(columns={'net_R': 'net_R_oldcost'}), on=['date', 'symbol'], how='left', validate='1:1')
    log(f'[bt] {len(b)} fills, join misses {b.level.isna().sum()}')
    b['price'] = b.level
    b['r_pct'] = (b.level - b.stop) / b.level * 100
    b['gross_lvl'] = (b.exit_price - b.level) / (b.level - b.stop)
    b['entry_slip_bps'] = (b.fill / b.level - 1) * 1e4
    b['fill_m'] = b.fill_min.astype(float)
    b['split'] = b.half
    return b


def load_live() -> pd.DataFrame:
    """The 231 R-valid live paper fills from cell 1,721 (gross_R at the intended level / stop / target)."""
    L = pd.read_csv(os.path.join(HERE, 'fills_1721.csv'), low_memory=False)
    L = L[L.valid_r.astype(str) == 'True'].copy()
    L['date'] = pd.to_datetime(L.trade_date).dt.strftime('%Y-%m-%d')
    L['price'] = L.level
    L['r_pct'] = (L.level - L.consol_low) / L.level * 100
    L['fill_m'] = L.fill_min.astype(float)
    L['net_R'] = L.R
    L['split'] = 'LIVE'
    L['n_cross'] = np.nan          # prior crossings that day: not recoverable for live fills (no bars after 2026-09)
    log(f'[live] {len(L)} fills, gross_R non-null {L.gross_R.notna().sum()}')
    return L


def daily_context(keys: pd.DataFrame) -> pd.DataFrame:
    """Causal daily context per (symbol, date) from cache.db daily_bars (read-only): today's open, prev close,
    prior-day return, 20-day mean dollar volume and 14-day ATR % (both ending the PRIOR day). Today's high/low/close
    are returned only as look-ahead descriptors (prefixed la_)."""
    syms = sorted(keys.symbol.unique())
    con = sqlite3.connect(f'file:{CACHE_DB}?mode=ro&immutable=1', uri=True, timeout=30)
    parts = []
    for i in range(0, len(syms), 400):
        ch = syms[i:i + 400]
        q = ','.join('?' * len(ch))
        parts.append(pd.read_sql(f"select symbol, bar_date, open, high, low, close, volume from daily_bars "
                                 f"where bar_date >= '2024-11-01' and symbol in ({q})", con, params=ch))
        log(f'[daily] symbols {i + len(ch)}/{len(syms)}')
    con.close()
    d = pd.concat(parts).sort_values(['symbol', 'bar_date'])
    d['bar_date'] = d.bar_date.astype(str).str[:10]
    g = d.groupby('symbol')
    d['prev_close'] = g.close.shift(1)
    d['prev2_close'] = g.close.shift(2)
    d['dv'] = d.close * d.volume
    d['adv20_usd'] = g.dv.transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())
    tr = np.maximum(d.high - d.low, np.maximum((d.high - d.prev_close).abs(), (d.low - d.prev_close).abs()))
    d['tr_pct'] = tr / d.prev_close * 100
    d['atr_pct'] = d.assign(t=d.tr_pct).groupby('symbol').t.transform(lambda s: s.shift(1).rolling(14, min_periods=7).mean())
    d['gap_pct'] = (d.open / d.prev_close - 1) * 100
    d['prev_ret'] = (d.prev_close / d.prev2_close - 1) * 100
    d['la_range_pct'] = (d.high - d.low) / d.open * 100
    d['la_close'] = d.close
    d['la_high'] = d.high
    return d.rename(columns={'bar_date': 'date', 'open': 'day_open'})[
        ['symbol', 'date', 'day_open', 'prev_close', 'gap_pct', 'prev_ret', 'adv20_usd', 'atr_pct',
         'la_range_pct', 'la_close', 'la_high']]


def add_features(x: pd.DataFrame, ctx: pd.DataFrame) -> pd.DataFrame:
    """Shared causal features: extension of the level above the open, weekday, fills/day, rank in day, alt-week half."""
    x = x.drop(columns=[c for c in ctx.columns if c in x.columns and c not in ('symbol', 'date')])
    x = x.merge(ctx, on=['symbol', 'date'], how='left')
    x['ext_open'] = (x.level / x.day_open - 1) * 100
    x['dow'] = pd.to_datetime(x.date).dt.dayofweek
    x = x.sort_values(['date', 'fill_m'])
    x['rank_day'] = x.groupby('date').cumcount() + 1
    x['fills_day'] = x.groupby('date').symbol.transform('count')
    x['wkn'] = pd.to_datetime(x.date).dt.isocalendar().week.astype(int)
    x['halfAB'] = np.where(x.wkn % 2 == 0, 'A', 'B')
    x['la_close_vs_level'] = (x.la_close / x.level - 1) * 100
    x['la_high_vs_level'] = (x.la_high / x.level - 1) * 100
    return x


def apply_caps(b: pd.DataFrame, max_conc=4, max_day=12) -> pd.Series:
    """Live slot rule on the BT: first-come by fill minute, at most `max_conc` open positions (exit minute from the
    causal walk) and `max_day` fills per day. Returns a boolean keep mask aligned to b.index."""
    keep = pd.Series(False, index=b.index)
    for _, g in b.sort_values('fill_m').groupby('date'):
        opened = []
        n = 0
        for i, r in g.iterrows():
            opened = [e for e in opened if e > r.fill_m]
            if n < max_day and len(opened) < max_conc:
                keep[i] = True
                n += 1
                opened.append(r.exit_m if pd.notna(r.exit_m) else 955)
    return keep


def dclust_t(x: pd.DataFrame, col: str) -> float:
    """Day-clustered t of the mean of `col`."""
    v = x[col].dropna()
    if len(v) < 3:
        return np.nan
    dev = (x.loc[v.index, col] - v.mean()).groupby(x.loc[v.index, 'date']).sum()
    se = np.sqrt((dev ** 2).sum()) / len(v)
    return v.mean() / se if se > 0 else np.nan


def ex_top5(v: pd.Series) -> float:
    """Mean after dropping the top 5 % of rows by value."""
    v = v.dropna().sort_values()
    return v.iloc[:int(np.floor(len(v) * 0.95))].mean() if len(v) else np.nan


def score(b: pd.DataFrame, L: pd.DataFrame, mb, ml, label: str) -> dict:
    """Kept-set statistics of one rule on the BT splits (net, gross, t, ex-top-5 %, halves) and on the live fills."""
    row = {'rule': label}
    for s in ('TRAIN-H2', 'VAL'):
        k = b[mb & (b.split == s)]
        row[f'{s} n'] = len(k)
        row[f'{s} kept%'] = round(100 * len(k) / max(1, (b.split == s).sum()), 1)
        row[f'{s} net'] = round(k.net_R.mean(), 3)
        row[f'{s} gross_lvl'] = round(k.gross_lvl.mean(), 3)
        row[f'{s} t'] = round(dclust_t(k, 'net_R'), 2)
        row[f'{s} ex5'] = round(ex_top5(k.net_R), 3)
        row[f'{s} A/B'] = f"{k[k.halfAB == 'A'].net_R.mean():+.3f}/{k[k.halfAB == 'B'].net_R.mean():+.3f}"
    kl = L[ml]
    row['live n'] = len(kl)
    row['live kept%'] = round(100 * len(kl) / len(L), 1)
    row['live gross'] = round(kl.gross_R.mean(), 3)
    row['live net'] = round(kl.net_R.mean(), 3)
    row['live drop gross'] = round(L[~ml].gross_R.mean(), 3)
    return row


def dim_table(b: pd.DataFrame, L: pd.DataFrame, col: str, bins) -> None:
    """Per-bin table of one dimension: BT n / gross_lvl / net by split, live share and live gross."""
    bb = pd.cut(b[col], bins, include_lowest=True)
    lb = pd.cut(L[col], bins, include_lowest=True)
    rows = []
    for iv in bb.cat.categories:
        m = bb == iv
        ml = lb == iv
        r = {'bin': str(iv), 'BT%': round(100 * m.mean(), 1), 'live%': round(100 * ml.mean(), 1)}
        for s in ('TRAIN-H2', 'VAL'):
            k = b[m & (b.split == s)]
            r[f'{s[:2]} n'] = len(k)
            r[f'{s[:2]} gross'] = round(k.gross_lvl.mean(), 3)
            r[f'{s[:2]} net'] = round(k.net_R.mean(), 3)
        r['live n'] = int(ml.sum())
        r['live gross'] = round(L[ml].gross_R.mean(), 3)
        rows.append(r)
    t = pd.DataFrame(rows)
    # reweighting: BT gross_lvl re-weighted to the live bin mix (explained share of the gross gap)
    w = t['live%'] / 100
    bt_bin = b.groupby(bb, observed=False).gross_lvl.mean().values
    rw = np.nansum(w.values * bt_bin) / np.nansum(w.values[~np.isnan(bt_bin)])
    log(f'\n### {col}  (BT pooled gross_lvl {b.gross_lvl.mean():+.3f} -> reweighted to live mix {rw:+.3f}; '
        f'missing BT {b[col].isna().mean():.1%} live {L[col].isna().mean():.1%})')
    log(t.to_markdown(index=False))


def main():
    """Run steps 1-4 and write the outputs."""
    b = load_bt()
    L = load_live()
    keys = pd.concat([b[['symbol', 'date']], L[['symbol', 'date']]]).drop_duplicates()
    ctx = daily_context(keys)
    b = add_features(b, ctx)
    L = add_features(L, ctx)
    log(f'[ctx] gap coverage BT {b.gap_pct.notna().mean():.1%} live {L.gap_pct.notna().mean():.1%}; '
        f'adv coverage BT {b.adv20_usd.notna().mean():.1%} live {L.adv20_usd.notna().mean():.1%}')

    log('\n## Step 1: one R basis')
    for s in ('TRAIN-H2', 'VAL'):
        k = b[b.split == s]
        log(f'BT {s}: n {len(k)} | old-cost net (1438 causal cost_R) {k.net_R_oldcost.mean():+.3f} (cost_R {k.cost_R.mean():.3f}) '
            f'| fill-basis raw {k.raw_R.mean():+.3f} | 1663 gross {k.gross_R.mean():+.3f} | measured-cost net {k.net_R.mean():+.3f} '
            f'| LEVEL-basis gross {k.gross_lvl.mean():+.3f} (t {dclust_t(k, "gross_lvl"):.2f}) | entry slip {k.entry_slip_bps.mean():.1f} bps')
    lg = L.gross_R.dropna()
    log(f'LIVE: n {len(L)} | gross_R {lg.mean():+.3f} sd {lg.std():.2f} 95% CI [{lg.mean() - 1.96 * lg.std() / np.sqrt(len(lg)):+.3f}, '
        f'{lg.mean() + 1.96 * lg.std() / np.sqrt(len(lg)):+.3f}] day-t {dclust_t(L, "gross_R"):.2f} | net {L.net_R.mean():+.3f} '
        f'| entry slip {L.entry_slip_bps.mean():.1f} bps | exit mix {L.exit_kind.value_counts(normalize=True).round(2).to_dict()}')
    log(f'BT exit mix {b.exit_type.value_counts(normalize=True).round(2).to_dict()}')
    for s in ('TRAIN-H2', 'VAL'):
        k = b[b.split == s]
        log(f'BT {s} exit-type gross_lvl: ' + str(k.groupby('exit_type').gross_lvl.mean().round(3).to_dict()))
    log('LIVE exit-kind gross_R: ' + str(L.groupby('exit_kind').gross_R.mean().round(3).to_dict()))

    log('\n## Step 2: quantiles live vs BT')
    cols = ['price', 'r_pct', 'fill_m', 'dow', 'gap_pct', 'ext_open', 'prev_ret', 'adv20_usd', 'atr_pct',
            'fills_day', 'rank_day', 'entry_slip_bps', 'la_range_pct', 'la_high_vs_level']
    qs = [0.1, 0.25, 0.5, 0.75, 0.9]
    rows = []
    for c in cols:
        for nm, x in (('BT', b), ('LIVE', L)):
            v = x[c].dropna()
            rows.append([c, nm, len(v)] + [round(v.quantile(q), 2) for q in qs] + [round(v.mean(), 2)])
    log(pd.DataFrame(rows, columns=['dim', 'pop', 'n'] + [f'q{int(q * 100)}' for q in qs] + ['mean']).to_markdown(index=False))

    dim_table(b, L, 'price', [0, 25, 35, 50, 80, 1e6])
    dim_table(b, L, 'r_pct', [1.5, 1.75, 2.0, 2.5, 3.0, 100])
    dim_table(b, L, 'fill_m', [0, 600, 630, 660, 720, 960])
    dim_table(b, L, 'dow', [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5])
    dim_table(b, L, 'gap_pct', [-100, 0, 2, 5, 10, 1e6])
    dim_table(b, L, 'ext_open', [0, 6, 7.5, 10, 15, 1e6])
    dim_table(b, L, 'prev_ret', [-100, -3, 0, 3, 10, 1e6])
    dim_table(b, L, 'adv20_usd', [0, 2e7, 5e7, 1.5e8, 5e8, 1e13])
    dim_table(b, L, 'atr_pct', [0, 3, 4.5, 6, 9, 1e6])
    dim_table(b, L, 'fills_day', [0, 5, 10, 15, 25, 1e6])
    dim_table(b, L, 'rank_day', [0, 1, 3, 6, 12, 1e6])
    dim_table(b, L, 'n_cross', [0, 1, 2, 4, 8, 1e6])
    dim_table(b, L, 'entry_slip_bps', [-1e3, 3, 6, 10, 15.01])

    log('\n## Step 3: live gates applied to the BT')
    b['cap_keep'] = apply_caps(b)
    L['cap_keep'] = True
    res = [score(b, L, pd.Series(True, index=b.index), pd.Series(True, index=L.index), 'ALL')]
    res.append(score(b, L, b.cap_keep, L.cap_keep, 'caps 4 concurrent / 12 per day (first-come)'))
    for k in (3, 6):
        res.append(score(b, L, b.rank_day <= k, L.rank_day <= k, f'rank_day <= {k} (uncapped BT order)'))
    res.append(score(b, L, b.prev_close >= 15, L.prev_close >= 15, 'prev close >= 15 (stream universe)'))
    log(pd.DataFrame(res).to_markdown(index=False))

    log('\n## Step 4: candidate rules (exploratory)')
    cands = [
        ('gap_pct >= 2', lambda x: x.gap_pct >= 2),
        ('gap_pct < 2', lambda x: x.gap_pct < 2),
        ('ext_open < 10', lambda x: x.ext_open < 10),
        ('ext_open >= 10', lambda x: x.ext_open >= 10),
        ('fill before 10:30', lambda x: x.fill_m < 630),
        ('fill before 11:00', lambda x: x.fill_m < 660),
        ('r_pct >= 2', lambda x: x.r_pct >= 2),
        ('r_pct >= 2.5', lambda x: x.r_pct >= 2.5),
        ('price >= 35', lambda x: x.price >= 35),
        ('adv20 >= $50M', lambda x: x.adv20_usd >= 5e7),
        ('adv20 >= $150M', lambda x: x.adv20_usd >= 1.5e8),
        ('atr_pct >= 4.5', lambda x: x.atr_pct >= 4.5),
        ('prev_ret >= 0', lambda x: x.prev_ret >= 0),
        ('fills_day <= 15', lambda x: x.fills_day <= 15),
        ('entry slip <= 6 bps (fill-model, BT only meaningful)', lambda x: x.entry_slip_bps <= 6),
        ('Tue-Thu', lambda x: x.dow.between(1, 3)),
        ('caps AND fill before 11:00', None),
    ]
    res = []
    for lab, f in cands:
        if f is None:
            mb, ml = b.cap_keep & (b.fill_m < 660), L.fill_m < 660
        else:
            mb, ml = f(b).fillna(False), f(L).fillna(False)
        res.append(score(b, L, mb, ml, lab))
    log(pd.DataFrame(res).to_markdown(index=False))

    b.to_csv(os.path.join(HERE, 'cell_1728_bt.csv'), index=False)
    L[['id', 'date', 'symbol', 'price', 'r_pct', 'fill_m', 'gap_pct', 'ext_open', 'prev_ret', 'adv20_usd', 'atr_pct',
       'rank_day', 'fills_day', 'gross_R', 'net_R', 'exit_kind', 'halfAB']].to_csv(os.path.join(HERE, 'cell_1728_live.csv'), index=False)
    log('[done]')


if __name__ == '__main__':
    sys.exit(main())
