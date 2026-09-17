"""D5 stage 2 — qualitative classes, day/week tables, entry-fact split, sequence.

Reads trades_facts.csv / days_facts.csv, writes trades_classed.csv and the
markdown fragments that go into REPORT.md.  Pure pandas, no tape access.
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = []
CELLS = {}


def log(*a):
    print(*a); sys.stdout.flush()


def w(s=''):
    OUT.append(s)


def cells(name, n):
    CELLS[name] = CELLS.get(name, 0) + n


# ------------------------------------------------------------------ classes
# Declared precedence: facts knowable AT THE FILL first (e, d), then the tape
# pathologies (c, b, a), then the stop-shape / path classes (f, g), then other.
THIN_DV5 = 200_000.0        # 5-min $ volume floor
LATE_PCT = 8.0              # entry more than this % above the 09:30 open
DUMP_PCT = -0.5             # SPY or IWM in the hour after entry
HALT_GAP = 5                # consecutive missing 1-min bars between fill+exit
HALT_RANGE = 5.0            # any 1-min bar range, % of its low
BLEED_MFE = 0.3
BLEED_MIN = 30

CLASS_NAMES = {
    'e_late': '(e) late/extended  entry >8% above the 09:30 open',
    'd_thin': '(d) spread/thin    5-min $ volume < $200K',
    'c_news': '(c) news/halt      bar gap >=5 min or a 1-min range >5%',
    'b_dump': '(b) market dump    SPY or IWM -0.5% in the hour after entry',
    'a_levelfail': '(a) level failure  never traded above the level after the fill',
    'f_wick': '(f) wick stop      stop bar closed above the stop, back over entry <30m',
    'g_bleed': '(g) slow bleed     MFE <0.3R then >30 min to the stop',
    'h_other': '(h) other',
}
ORDER = ['e_late', 'd_thin', 'c_news', 'b_dump', 'a_levelfail', 'f_wick',
         'g_bleed', 'h_other']


def flag(d):
    d['f_e_late'] = (d.open_to_entry_pct > LATE_PCT)
    d['f_d_thin'] = (d.dollar_vol_5m < THIN_DV5)
    d['f_c_news'] = (d.biggest_bar_gap >= HALT_GAP) | (d.max_bar_range_pct > HALT_RANGE)
    d['f_b_dump'] = (d.spy_ret_1h <= DUMP_PCT) | (d.iwm_ret_1h <= DUMP_PCT)
    d['f_a_levelfail'] = (d.level_fail == 1)
    d['f_f_wick'] = (d.exit_type == 'stop') & (d.wick == 1) & (d.recovered_30m == 1)
    d['f_g_bleed'] = (d.mfe_R < BLEED_MFE) & (d.hold_min > BLEED_MIN)
    cl = np.full(len(d), 'h_other', dtype=object)
    for k in reversed(ORDER[:-1]):
        cl = np.where(d['f_' + k].values, k, cl)
    d['cls'] = cl
    return d


def fmt(x, n=2):
    return '' if x != x else f'{x:.{n}f}'


def money(x):
    return f'${x:,.0f}'


# -------------------------------------------------------------- day tables
def day_lines(d, days, dd, title):
    w(f'### {title}')
    w()
    dmap = dd.set_index('day')
    for day in days:
        r = dmap.loc[day]
        g = d[d.day == day].sort_values('entry_min')
        w(f'**{day}** ({r.split})  book **{r.book_R:+.2f} R** · {int(r.n)} trades '
          f'({int(r.wins)} green) · candidates {int(r.candidates)} · '
          f'SPY o->c {fmt(r.spy_oc)}% / o->10:00 {fmt(r.spy_o10)}% · '
          f'IWM o->c {fmt(r.iwm_oc)}% / o->10:00 {fmt(r.iwm_o10)}%')
        for t in g.itertuples():
            hh, mm = divmod(t.entry_min, 60)
            what = (f'stopped {t.exit_min - t.entry_min}m later after MFE '
                    f'{t.mfe_R:+.2f}R' if t.exit_type == 'stop'
                    else f'held {t.exit_min - t.entry_min}m to the close, MFE '
                         f'{t.mfe_R:+.2f}R')
            w(f'  - `{t.symbol:<6}` {hh:02d}:{mm:02d} '
              f'net **{t.net_R:+.2f}R** · entry {t.entry:.2f} '
              f'({t.entry_vs_prev_close_pct:+.1f}% vs prior close, gap '
              f'{t.gap_pct:+.1f}%, PDR {t.prev_range_pct:.0f}%, stop '
              f'{t.stop_dist_pct:.1f}% away, 5m$ {money(t.dollar_vol_5m)}) · '
              f'{what} · **{t.cls[2:]}**')
        w()


# --------------------------------------------------------------------- main
def main():
    d = pd.read_csv(os.path.join(HERE, 'trades_facts.csv'), keep_default_na=False)
    dd = pd.read_csv(os.path.join(HERE, 'days_facts.csv'), keep_default_na=False)
    d = flag(d)
    d['loser'] = d.net_R < 0
    d.to_csv(os.path.join(HERE, 'trades_classed.csv'), index=False)
    log(f'classed {len(d)} trades')

    d['hour'] = d.entry_min // 60
    d['seq_b'] = np.where(d.seq >= 4, '4+', d.seq.astype(str))
    d['price_band'] = pd.cut(d.entry, [0, 10, 20, 50, 1e9],
                             labels=['$5-10', '$10-20', '$20-50', '>$50'])

    # ---------------------------------------------------------------- header
    w('# D5 — red-to-green (F6-PDR, FIRST-BREAK rule): the losers, by hand')
    w()
    w('Book: `research/fuckup_audit/H/F6_rebuild/trades_hold_ai.csv` — implementation B '
      '(the independent rebuild), HOLD exit, booked 12/day at 4 concurrent. '
      f'2,011 rows minus 2 `Z?ZZT` test-ticker rows = **{len(d)} trades**, '
      '417 trading days, 2025-01-02 .. 2026-09-04.')
    w()
    w('| split | n | total net R | R/trade | WR | losers | loser R | winner R |')
    w('|---|---|---|---|---|---|---|---|')
    for s in ('TRAIN', 'VAL', 'TEST'):
        g = d[d.split == s]
        L = g[g.loser]; W = g[~g.loser]
        w(f'| {s} | {len(g)} | {g.net_R.sum():+.1f} | {g.net_R.mean():+.4f} | '
          f'{(~g.loser).mean()*100:.1f}% | {len(L)} | {L.net_R.sum():+.1f} | '
          f'{W.net_R.sum():+.1f} |')
    g = d
    w(f'| ALL | {len(g)} | {g.net_R.sum():+.1f} | {g.net_R.mean():+.4f} | '
      f'{(~g.loser).mean()*100:.1f}% | {int(g.loser.sum())} | '
      f'{g[g.loser].net_R.sum():+.1f} | {g[~g.loser].net_R.sum():+.1f} |')
    w()
    w('Reference (`H/F6_reconcile/REPORT.md`): this is the rule the two studies ran, '
      'NOT the rule `trading/red_to_green.py` implements. The shipped engine keeps '
      'scanning past a floor-failing bar and its book is -0.027 / -0.012 / -0.102 R. '
      'Everything below is the anatomy of the FIRST-BREAK book.')
    w()

    # ------------------------------------------------------------ 1. days
    w('---')
    w()
    w('## 1. Losing days as tape')
    w()
    dd2 = dd.sort_values('book_R')
    worst = list(dd2.head(25).day)
    best = list(dd2.tail(25).sort_values if False else dd2.tail(25).day)[::-1]
    cells('day tape lines (25 worst + 25 best days)', 50)
    day_lines(d, worst, dd, '1a. The 25 worst days')
    day_lines(d, best, dd, '1b. The 25 best days')

    w('### 1c. Class counts across ALL losers')
    w()
    w('Precedence (declared before the run, entry-knowable first): '
      'e -> d -> c -> b -> a -> f -> g -> h. A trade gets exactly one class; the '
      'raw flag counts (a trade can carry several) are in the second table.')
    w()
    L = d[d.loser]
    w('| class | TRAIN n | TRAIN R | VAL n | VAL R | TEST n | TEST R | ALL n | ALL R | % of loss |')
    w('|---|---|---|---|---|---|---|---|---|---|')
    tot_loss = L.net_R.sum()
    for k in ORDER:
        row = [CLASS_NAMES[k]]
        for s in ('TRAIN', 'VAL', 'TEST'):
            gg = L[(L.split == s) & (L.cls == k)]
            row += [str(len(gg)), f'{gg.net_R.sum():+.1f}']
        gg = L[L.cls == k]
        row += [str(len(gg)), f'{gg.net_R.sum():+.1f}',
                f'{gg.net_R.sum()/tot_loss*100:.1f}%']
        w('| ' + ' | '.join(row) + ' |')
        cells('loser class x split', 3)
    w(f'| **all losers** | {len(L[L.split=="TRAIN"])} | '
      f'{L[L.split=="TRAIN"].net_R.sum():+.1f} | {len(L[L.split=="VAL"])} | '
      f'{L[L.split=="VAL"].net_R.sum():+.1f} | {len(L[L.split=="TEST"])} | '
      f'{L[L.split=="TEST"].net_R.sum():+.1f} | {len(L)} | {tot_loss:+.1f} | 100% |')
    w()
    w('Raw flags, NOT exclusive (share of all losers carrying the flag, and the '
      'same flag among winners for contrast):')
    w()
    w('| flag | losers n | losers % | loser R | winners n | winners % | winner R |')
    w('|---|---|---|---|---|---|---|')
    W = d[~d.loser]
    for k in ORDER[:-1]:
        c = 'f_' + k
        w(f'| {CLASS_NAMES[k]} | {int(L[c].sum())} | {L[c].mean()*100:.1f}% | '
          f'{L[L[c]].net_R.sum():+.1f} | {int(W[c].sum())} | {W[c].mean()*100:.1f}% | '
          f'{W[W[c]].net_R.sum():+.1f} |')
        cells('raw flag x (winner|loser)', 2)
    w()

    # ------------------------------------------------- 2. entry facts
    w('---')
    w()
    w('## 2. Winners at entry vs losers at entry')
    w()
    FACTS = [('entry_min', 'entry minute (ET min of day)', 0),
             ('gap_pct', 'gap % (09:30 open vs prior close)', 2),
             ('prev_range_pct', 'prior-day range %', 2),
             ('entry_vs_prev_close_pct', 'entry vs prior close %', 2),
             ('open_to_entry_pct', 'open -> entry %', 2),
             ('stop_dist_pct', 'stop distance % of entry', 2),
             ('dollar_vol_5m', '5-min $ volume', 0),
             ('range_so_far_pct', 'range-so-far % at the signal', 2),
             ('iwm_open_to_entry', 'IWM open -> entry %', 3),
             ('spy_open_to_entry', 'SPY open -> entry %', 3),
             ('seq', 'trades already taken that day (this trade''s ordinal)', 1),
             ('entry', 'entry price $', 2)]
    w('Median [Q1, Q3] per split. `gap` = the relative gap between the winner and '
      'loser medians, `(w-l)/|l|`.')
    w()
    for s in ('TRAIN', 'VAL', 'TEST'):
        g = d[d.split == s]
        gw, gl = g[~g.loser], g[g.loser]
        w(f'**{s}** (n {len(gw)} winners / {len(gl)} losers)')
        w()
        w('| fact | winners med [Q1,Q3] | losers med [Q1,Q3] | gap |')
        w('|---|---|---|---|')
        for c, lab, nd in FACTS:
            a, b = gw[c].astype(float), gl[c].astype(float)
            ma, mb = a.median(), b.median()
            rel = (ma - mb) / abs(mb) if abs(mb) > 1e-9 else float('nan')
            w(f'| {lab} | {ma:,.{nd}f} [{a.quantile(.25):,.{nd}f}, '
              f'{a.quantile(.75):,.{nd}f}] | {mb:,.{nd}f} '
              f'[{b.quantile(.25):,.{nd}f}, {b.quantile(.75):,.{nd}f}] | '
              f'{rel*100:+.0f}% |' if rel == rel else
              f'| {lab} | {ma:,.{nd}f} [{a.quantile(.25):,.{nd}f}, '
              f'{a.quantile(.75):,.{nd}f}] | {mb:,.{nd}f} '
              f'[{b.quantile(.25):,.{nd}f}, {b.quantile(.75):,.{nd}f}] | n/a |')
            cells('entry-fact x split x (winner|loser)', 2)
        w()
    # agreement test
    w('**Which facts separate by >=20% in the SAME direction in TRAIN and VAL** '
      '(the declared bar):')
    w()
    w('| fact | TRAIN gap | VAL gap | same sign & both >=20%? | TEST gap |')
    w('|---|---|---|---|---|')
    agree = []
    for c, lab, nd in FACTS:
        gaps = {}
        for s in ('TRAIN', 'VAL', 'TEST'):
            g = d[d.split == s]
            ma = g[~g.loser][c].astype(float).median()
            mb = g[g.loser][c].astype(float).median()
            gaps[s] = (ma - mb) / abs(mb) if abs(mb) > 1e-9 else float('nan')
        ok = (abs(gaps['TRAIN']) >= .20 and abs(gaps['VAL']) >= .20 and
              np.sign(gaps['TRAIN']) == np.sign(gaps['VAL']))
        if ok:
            agree.append(lab)
        w(f'| {lab} | {gaps["TRAIN"]*100:+.0f}% | {gaps["VAL"]*100:+.0f}% | '
          f'{"**YES**" if ok else "no"} | {gaps["TEST"]*100:+.0f}% |')
    w()
    w('Passing: ' + (', '.join(agree) if agree else '**none**') + '.')
    w()
    # categorical
    w('**Day of week and price band** (net R per trade):')
    w()
    for col, lab in (('dow', 'day of week'), ('price_band', 'price band')):
        w(f'| {lab} | TRAIN n | TRAIN R/tr | VAL n | VAL R/tr | TEST n | TEST R/tr |')
        w('|---|---|---|---|---|---|---|')
        order = (['Mon', 'Tue', 'Wed', 'Thu', 'Fri'] if col == 'dow'
                 else ['$5-10', '$10-20', '$20-50', '>$50'])
        for v in order:
            row = [str(v)]
            for s in ('TRAIN', 'VAL', 'TEST'):
                g = d[(d.split == s) & (d[col].astype(str) == v)]
                row += [str(len(g)), f'{g.net_R.mean():+.3f}' if len(g) else '-']
                cells('categorical entry cell', 1)
            w('| ' + ' | '.join(row) + ' |')
        w()

    # ------------------------------------------------------------ 3. weeks
    w('---')
    w()
    w('## 3. The 10 worst weeks')
    w()
    d['week'] = pd.to_datetime(d.day).dt.to_period('W').astype(str)
    dd['week'] = pd.to_datetime(dd.day).dt.to_period('W').astype(str)
    wk = d.groupby('week').agg(R=('net_R', 'sum'), n=('net_R', 'size')).reset_index()
    dw = dd.groupby('week').agg(spy=('spy_oc', 'sum'), iwm=('iwm_oc', 'sum'),
                                vol=('spy_oc', 'std'), days=('day', 'size')).reset_index()
    wk = wk.merge(dw, on='week')
    wk = wk.sort_values('R')
    w('SPY/IWM weekly return = sum of the daily open->close moves in the week '
      '(the book is flat overnight, so the intraday sum is the relevant market). '
      '"realised vol" = stdev of the SPY daily open->close in that week.')
    w()
    w('| week | split | book R | n | SPY wk % | IWM wk % | SPY daily sd | '
      'worst loser class (share of the week\'s loss) |')
    w('|---|---|---|---|---|---|---|---|')
    for r in wk.head(10).itertuples():
        g = d[d.week == r.week]
        gl = g[g.loser]
        by = gl.groupby('cls').net_R.sum().sort_values()
        top = by.index[0] if len(by) else '-'
        shr = by.iloc[0] / gl.net_R.sum() * 100 if len(by) else 0
        sp = g.split.iloc[0]
        w(f'| {r.week[:10]} | {sp} | **{r.R:+.2f}** | {r.n} | {r.spy:+.2f} | '
          f'{r.iwm:+.2f} | {r.vol:.2f} | {top[2:]} ({shr:.0f}% of '
          f'{gl.net_R.sum():+.1f}R) |')
        cells('worst-week row', 1)
    w()
    wkall = wk.copy()
    w(f'Weeks total {len(wkall)} · green {int((wkall.R>0).sum())} '
      f'({(wkall.R>0).mean()*100:.0f}%) · worst {wkall.R.min():+.2f}R · '
      f'best {wkall.R.max():+.2f}R.')
    w()
    w('Correlation of weekly book R with the weekly SPY intraday sum: '
      f'**{wkall.R.corr(wkall.spy):+.3f}**; with IWM: '
      f'**{wkall.R.corr(wkall.iwm):+.3f}**; with SPY daily sd: '
      f'**{wkall.R.corr(wkall.vol):+.3f}** (n={len(wkall)} weeks).')
    w()

    # --------------------------------------------------------- 4. sequence
    w('---')
    w()
    w('## 4. Sequence within the day')
    w()
    w('| ordinal | TRAIN n | TRAIN R/tr | VAL n | VAL R/tr | TEST n | TEST R/tr | ALL n | ALL R/tr |')
    w('|---|---|---|---|---|---|---|---|---|')
    for v in ['1', '2', '3', '4+']:
        row = [v]
        for s in ('TRAIN', 'VAL', 'TEST'):
            g = d[(d.split == s) & (d.seq_b == v)]
            row += [str(len(g)), f'{g.net_R.mean():+.3f}' if len(g) else '-']
            cells('sequence cell', 1)
        g = d[d.seq_b == v]
        row += [str(len(g)), f'{g.net_R.mean():+.3f}']
        w('| ' + ' | '.join(row) + ' |')
    w()
    w('| entry hour (ET) | TRAIN n | TRAIN R/tr | VAL n | VAL R/tr | TEST n | TEST R/tr | ALL n | ALL R/tr |')
    w('|---|---|---|---|---|---|---|---|---|')
    for h in sorted(d.hour.unique()):
        row = [f'{h:02d}:xx']
        for s in ('TRAIN', 'VAL', 'TEST'):
            g = d[(d.split == s) & (d.hour == h)]
            row += [str(len(g)), f'{g.net_R.mean():+.3f}' if len(g) else '-']
            cells('hour cell', 1)
        g = d[d.hour == h]
        row += [str(len(g)), f'{g.net_R.mean():+.3f}']
        w('| ' + ' | '.join(row) + ' |')
    w()
    w('### The two declared rule cells')
    w()
    w('Both are SUBSET filters on the already-booked trades: dropping a trade does '
      'NOT free its slot for a candidate the book passed over, so these are a lower '
      'bound on what a re-booked run would show.')
    w()
    w('| cell | TRAIN n | TRAIN R/tr | TRAIN total | VAL n | VAL R/tr | VAL total | '
      'TEST n | TEST R/tr | TEST total |')
    w('|---|---|---|---|---|---|---|---|---|---|')
    for lab, m in (('as booked', d.index == d.index),
                   ('first two entries only (seq <= 2)', d.seq <= 2),
                   ('no entries after 11:00 (entry_min < 660)', d.entry_min < 660)):
        row = [lab]
        for s in ('TRAIN', 'VAL', 'TEST'):
            g = d[(d.split == s) & m]
            row += [str(len(g)), f'{g.net_R.mean():+.4f}', f'{g.net_R.sum():+.1f}']
            cells('declared sequence rule cell', 1)
        w('| ' + ' | '.join(row) + ' |')
    w()

    # --------------------------------------------------------- 5. avoidance
    w('---')
    w()
    w('## 5. Avoidance arithmetic')
    w()
    MONTHS = {'TRAIN': 12.0, 'VAL': 5.0, 'TEST': 3.13}
    w('Only two of the eight classes are decidable AT THE FILL: **(e) late/extended** '
      '(the entry is >8% above the 09:30 open — known from the fill price and the '
      'first bar) and **(d) spread/thin** (the 5-min $ volume before the fill). '
      'The others need the future tape. Below: drop each entry-knowable class from '
      'the book (again a subset, not a re-book), and both together. '
      '$/month at $300 risk = total net R x $300 / months in the split '
      '(TRAIN 12, VAL 5, TEST 3.13).')
    w()
    w('| rule | TRAIN n | TRAIN R/tr | TRAIN $/mo | VAL n | VAL R/tr | VAL $/mo | '
      'TEST n | TEST R/tr | TEST $/mo |')
    w('|---|---|---|---|---|---|---|---|---|---|')
    rules = [('as booked', pd.Series(True, index=d.index)),
             ('drop (e) late/extended', ~d.f_e_late),
             ('drop (d) spread/thin', ~d.f_d_thin),
             ('drop both (e) and (d)', ~(d.f_e_late | d.f_d_thin))]
    for lab, m in rules:
        row = [lab]
        for s in ('TRAIN', 'VAL', 'TEST'):
            g = d[(d.split == s) & m]
            row += [str(len(g)), f'{g.net_R.mean():+.4f}',
                    money(g.net_R.sum() * 300 / MONTHS[s])]
            cells('avoidance rule cell', 1)
        w('| ' + ' | '.join(row) + ' |')
    w()
    w('Share of TOTAL loser R carried by each entry-knowable flag:')
    w()
    for k in ('e_late', 'd_thin'):
        c = 'f_' + k
        for s in ('TRAIN', 'VAL', 'TEST'):
            gl = L[L.split == s]
            w(f'- {k} · {s}: {int(gl[c].sum())}/{len(gl)} losers, '
              f'{gl[gl[c]].net_R.sum():+.1f}R of {gl.net_R.sum():+.1f}R '
              f'({gl[gl[c]].net_R.sum()/gl.net_R.sum()*100:.1f}% of the loss); '
              f'those same trades among winners: {int(W[(W.split==s)][c].sum())} '
              f'for {W[(W.split==s)&W[c]].net_R.sum():+.1f}R')
            cells('flag loss-share cell', 1)
    w()

    txt = '\n'.join(OUT)
    open(os.path.join(HERE, 'body.md'), 'w').write(txt)
    tot = sum(CELLS.values())
    log('CELLS:')
    for k, v in CELLS.items():
        log(f'  {k}: {v}')
    log(f'TOTAL {tot}')
    open(os.path.join(HERE, 'cells.txt'), 'w').write(
        '\n'.join(f'{k}: {v}' for k, v in CELLS.items()) + f'\nTOTAL: {tot}\n')


if __name__ == '__main__':
    main()
