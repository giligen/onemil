#!/usr/bin/env python3
"""hod_losers PART 2 — the 20 declared cells of PREREG.md. Nothing here was run before the
PREREG was committed (06e2327). TEST is sealed: no TEST number unless FREEZE.md exists AND --test.
"""
import os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT); sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_break')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')
import score as S            # noqa: E402
import score2 as S2          # noqa: E402
from trading.hod_break import run_book   # noqa: E402

D = f'{ROOT}/research/mature_method/hod_losers'
OUT = open(f'{D}/part2.txt', 'w')
RD = dict(dtype={'symbol': str, 'day': str}, keep_default_na=False, na_values=[''])
ROWS = []


def p(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); OUT.write(s + '\n'); OUT.flush()


def recost(x):
    """Recompute net/netb after an exit variant changed rr / why. R, price and spread are
    untouched by an exit rule, so only the exit-side ratio and rr move."""
    x = x.copy()
    ratio = x.why.map(S.RATIO).fillna(0.875)
    half = 0.5 * x.sp_pct / x.r_pct.clip(lower=0.05)
    x['net'] = x.rr - half - half * ratio
    hb = 0.5 * x.sp_band / x.r_pct.clip(lower=0.05)
    x['netb'] = x.rr - hb - hb * ratio
    return x


def book(s, kill_k=None):
    if not len(s):
        return s.assign(pnl=pd.Series(dtype=float))
    rows = [(r.day, int(r.entry_m), int(r.exit_m), r.symbol, r.Index, r.why) for r in s.itertuples()]
    if kill_k is None:
        take = [t[4] for t in run_book(rows, 12, 4)]
    else:
        take = run_book_kill(rows, 12, 4, kill_k)
    b = s.loc[take].copy()
    b['pnl'] = b.net * S.RISK
    return b


def run_book_kill(rows, max_per_day, max_concurrent, K):
    """run_book + the in-day kill switch: once K consecutive CLOSED trades of the session have
    exited at a stop, no further entry that day. Causal — only exits with exit_m < entry_m of the
    candidate are visible."""
    taken = []
    by_day = {}
    for r in rows:
        by_day.setdefault(r[0], []).append(r)
    for day in sorted(by_day):
        open_exits, n_day, closed = [], 0, []
        for r in sorted(by_day[day], key=lambda x: (int(x[1]), str(x[3]))):
            entry_m, exit_m = int(r[1]), int(r[2])
            open_exits = [e for e in open_exits if e >= entry_m]
            run = 0
            for em, wy in sorted(closed):
                if em >= entry_m:
                    break
                run = run + 1 if wy in ('stop', 'ruleD', 'timestop', 'shape', 'bestop') else 0
            if run >= K:
                continue
            if n_day >= max_per_day or len(open_exits) >= max_concurrent:
                continue
            taken.append(r[4]); open_exits.append(exit_m); n_day += 1
            closed.append((exit_m, r[5]))
    return taken


def emit(name, b, base, note=''):
    line = {'cell': name, 'base': base, 'note': note}
    for sp in S.SPLITS:
        w = S.week_stats(b, sp)
        line[sp] = w
        ROWS.append(dict(cell=name, base=base, split=sp, **w))
    a, v = line.get('TRAIN'), line.get('VAL')
    p(f'{name:26s} {base:3s} | TRAIN n{a["n"]:5d} {a["per_wk"]:5.1f}/wk g{a["gross"]:+.3f} '
      f'n{a["net"]:+.3f} t{a["t"]:+5.2f} grn{a["green"]:5.1f}% rs{a["redstreak"]:2d} '
      f'${a["total"]:+8,.0f} w${a["worst"]:+7,.0f} | VAL n{v["n"]:5d} {v["per_wk"]:5.1f}/wk '
      f'g{v["gross"]:+.3f} n{v["net"]:+.3f} t{v["t"]:+5.2f} grn{v["green"]:5.1f}% rs{v["redstreak"]:2d} '
      f'${v["total"]:+8,.0f} w${v["worst"]:+7,.0f}  {note}')


def main():
    pop = S2.load_pop(); S.build_impute(pop)
    ex = pd.read_csv(f'{D}/exits.csv', **RD).drop_duplicates(['day', 'symbol', 'entry_m'])
    pa = pd.read_csv(f'{D}/path.csv', **RD).drop_duplicates(['day', 'symbol', 'entry_m'])
    pa = pa[['day', 'symbol', 'entry_m', 'touch_n', 'consol_bars', 'e_low', 'e_high', 'e_close']]
    SG = {}
    for nm in ('B0', 'B2'):
        sg = S2.sig_set(pop, **S2.BASES[nm])
        sg = sg[sg.split.isin(S.SPLITS)]
        sg = sg.merge(ex, on=['day', 'symbol', 'entry_m'], how='left')
        sg = sg.merge(pa, on=['day', 'symbol', 'entry_m'], how='left')
        SG[nm] = sg.reset_index(drop=True)

    # ------------------------------------------------------------ parity gate
    p('=' * 170)
    p('PARITY GATE — the exit walk\'s `base` variant vs pop.csv, and the B0/B2 books vs the two '
      'prior reports')
    p('=' * 170)
    for nm in ('B0', 'B2'):
        sg = SG[nm]; ok = sg.rr_base.notna()
        p(f'  {nm}: rows with a walk {ok.sum()}/{len(sg)}  max|d rr| '
          f'{np.abs(sg.rr[ok] - sg.rr_base[ok]).max():.2e}  why match '
          f'{(sg.why[ok] == sg.why_base[ok]).mean():.6f}  exit-minute match '
          f'{(sg.exit_m[ok] == sg.exit_m_base[ok]).mean():.6f}')
    p('')
    p(f'{"cell":26s} {"bse":3s} | TRAIN ... | VAL ...')
    for nm in ('B0', 'B2'):
        emit(f'{nm} shipped (reference)', book(SG[nm]), nm,
             'reference: hod_break/hod_filter_stack')

    def variant(nm, v):
        s = SG[nm]
        s = s[s.rr_base.notna()].copy()
        s['rr'] = s[f'rr_{v}']; s['why'] = s[f'why_{v}']; s['exit_m'] = s[f'exit_m_{v}']
        return recost(s)

    p('\n' + '=' * 170)
    p('A. POST-FILL EXITS (P1-P9)')
    p('=' * 170)
    emit('P1  T10/0.25', book(variant('B0', 't10')), 'B0')
    emit('P2  T10/0.25', book(variant('B2', 't10')), 'B2')
    emit('P3  T5/0.25', book(variant('B0', 't5')), 'B0')
    emit('P4  RuleD 0.75->-0.5R', book(variant('B0', 'ruleD')), 'B0')
    emit('P5  RuleD 0.75->-0.5R', book(variant('B2', 'ruleD')), 'B2')
    emit('P6  fill-bar shape exit', book(variant('B0', 'shape')), 'B0')
    emit('P7  fill-bar shape exit', book(variant('B2', 'shape')), 'B2')
    emit('P8  BE at +0.5R', book(variant('B0', 'be')), 'B0')
    emit('P9  T10 + BE', book(variant('B0', 't10be')), 'B0')

    p('\n' + '=' * 170)
    p('B. ENTRY / LEVEL GATES (P10-P15)')
    p('=' * 170)
    def gate(nm, mask, v='base'):
        s = variant(nm, v)
        return s[mask(s)]
    emit('P10 RuleM veto (range>=.5)', book(gate('B0', lambda s: s.range_pos >= 0.5)), 'B0',
         'FALSIFICATION cell')
    emit('P11 touch_n < 5', book(gate('B0', lambda s: s.touch_n < 5)), 'B0')
    emit('P12 touch_n < 5', book(gate('B2', lambda s: s.touch_n < 5)), 'B2')
    emit('P13 consol_bars < 8', book(gate('B0', lambda s: s.consol_bars < 8)), 'B0')
    emit('P14 level >= 20d high', book(gate('B0', lambda s: s.dist_20d_high_pct >= 0)), 'B0')
    emit('P15 first break only', book(gate('B0', lambda s: s.n_break == 0)), 'B0')
    emit('P15c re-breaks only (compl)', book(gate('B0', lambda s: s.n_break > 0)), 'B0',
         'complement, reported not claimed')

    p('\n' + '=' * 170)
    p('C. IN-DAY KILL SWITCH (P16-P17)')
    p('=' * 170)
    emit('P16 kill after 3 stops', book(variant('B0', 'base'), kill_k=3), 'B0')
    emit('P17 kill after 2 stops', book(variant('B0', 'base'), kill_k=2), 'B0')

    # ------------------------------------------------------------ the selector
    p('\n' + '=' * 170)
    p('D. THE SELECTOR (PREREG §3) and the stack')
    p('=' * 170)
    df = pd.DataFrame(ROWS)
    cand = df[df.cell.str.startswith('P') & ~df.cell.str.startswith('P15c')]
    pv = cand.pivot_table(index=['cell', 'base'], columns='split',
                          values=['green', 'total', 'per_wk'])
    pv.columns = [f'{a}_{b}' for a, b in pv.columns]
    ok = pv[(pv['per_wk_TRAIN'] >= 10) & (pv['per_wk_VAL'] >= 10)].copy()
    ok['mg'] = ok[['green_TRAIN', 'green_VAL']].min(axis=1)
    ok['mt'] = ok[['total_TRAIN', 'total_VAL']].min(axis=1)
    ok = ok.sort_values(['mg', 'mt'], ascending=False)
    p('  eligible cells (>=10/wk both splits), ranked by min(green% TRAIN, green% VAL):')
    for (c, b), r in ok.iterrows():
        p(f'    {c:26s} {b} min-green {r["mg"]:5.1f}%  min-$ {r["mt"]:+8,.0f}  '
          f'(TRAIN {r["green_TRAIN"]:.1f}% ${r["total_TRAIN"]:+,.0f} | '
          f'VAL {r["green_VAL"]:.1f}% ${r["total_VAL"]:+,.0f})')
    best = ok.index[0]
    p(f'\n  SELECTED for the stack: {best[0]} on {best[1]}')

    # P18/P19 -- the selected rule, on both bases, stacked with the shipped book
    sel = best[0]
    VMAP = {'P1': ('v', 't10'), 'P2': ('v', 't10'), 'P3': ('v', 't5'), 'P4': ('v', 'ruleD'),
            'P5': ('v', 'ruleD'), 'P6': ('v', 'shape'), 'P7': ('v', 'shape'), 'P8': ('v', 'be'),
            'P9': ('v', 't10be'), 'P16': ('k', 3), 'P17': ('k', 2)}
    GMAP = {'P10': lambda s: s.range_pos >= 0.5, 'P11': lambda s: s.touch_n < 5,
            'P12': lambda s: s.touch_n < 5, 'P13': lambda s: s.consol_bars < 8,
            'P14': lambda s: s.dist_20d_high_pct >= 0, 'P15': lambda s: s.n_break == 0}
    key = sel.split()[0]
    for nm, tag in (('B0', 'P18'), ('B2', 'P19')):
        if key in VMAP:
            kind, val = VMAP[key]
            b = book(variant(nm, val), kill_k=None) if kind == 'v' else \
                book(variant(nm, 'base'), kill_k=val)
        else:
            b = book(gate(nm, GMAP[key]))
        emit(f'{tag} {sel} stack', b, nm, 'the selected rule on this base')

    # P20 -- count-matched permutation null on the two stack cells
    p('\n' + '=' * 170)
    p('P20. COUNT-MATCHED PERMUTATION NULL (2,000 draws, per-week pick count fixed)')
    p('=' * 170)
    nb = []
    for nm, tag in (('B0', 'P18'), ('B2', 'P19')):
        if key in VMAP:
            kind, val = VMAP[key]
            b = book(variant(nm, val)) if kind == 'v' else book(variant(nm, 'base'), kill_k=val)
        else:
            b = book(gate(nm, GMAP[key]))
        for sp in S.SPLITS:
            o, mu, lo, hi = S.null_band(b, sp)
            outside = 'ABOVE' if o > hi else ('below' if o < lo else 'inside')
            p(f'  {tag} {nm} {sp:5s}  observed green {o:5.1f}%  null mean {mu:5.1f}% '
              f'[{lo:.1f}, {hi:.1f}]  -> {outside}')
            nb.append(dict(cell=tag, base=nm, split=sp, obs=o, mu=mu, p5=lo, p95=hi))
    for nm in ('B0', 'B2'):
        b = book(SG[nm])
        for sp in S.SPLITS:
            o, mu, lo, hi = S.null_band(b, sp)
            p(f'  {nm} shipped   {sp:5s}  observed green {o:5.1f}%  null mean {mu:5.1f}% '
              f'[{lo:.1f}, {hi:.1f}]')

    pd.DataFrame(ROWS).to_csv(f'{D}/cells.csv', index=False)
    pd.DataFrame(nb).to_csv(f'{D}/nulls.csv', index=False)

    # ------------------------------------------------------------ the bars
    p('\n' + '=' * 170)
    p('BOTH BARS')
    p('=' * 170)
    g1 = df[(df.split == 'TRAIN') & df.cell.str.startswith('P')]
    passed = g1[(g1.net > 0) & (g1.t >= 2.0) & (g1.per_wk >= 10) & (g1.gross >= 0.25)]
    p(f'  CLAIM BAR G1 (TRAIN net R>0, t>=2.0, >=10/wk, TRAIN gross >= +0.25R): '
      f'{len(passed)} of {len(g1)} cell-rows pass')
    if len(passed):
        p(passed[['cell', 'base', 'n', 'per_wk', 'gross', 'net', 't', 'green']].to_string(index=False))
    else:
        p(f'  best TRAIN net R among cells {g1.net.max():+.4f} (cell '
          f'{g1.loc[g1.net.idxmax(), "cell"]} / {g1.loc[g1.net.idxmax(), "base"]}), '
          f'best TRAIN t {g1.t.max():+.2f}, best TRAIN gross {g1.gross.max():+.4f}')
    df = pd.DataFrame(ROWS)
    le = df[df.cell.str.startswith('P')].pivot_table(index=['cell', 'base'], columns='split',
                                                     values=['green', 'total', 'net', 'per_wk'])
    le.columns = [f'{a}_{b}' for a, b in le.columns]
    lex = le[(le['total_TRAIN'] > 0) & (le['total_VAL'] > 0) &
             (le['per_wk_TRAIN'] >= 10) & (le['per_wk_VAL'] >= 10)]
    p(f'  LIVE-EXPLORATION BAR (positive $ on BOTH splits at >=10/wk): {len(lex)} cells')
    if len(lex):
        p(lex.to_string())

    # ------------------------------------------------------------ ex-tail
    p('\n  ex-tail diagnostics on the selected stack (reported, never a rejection reason):')
    for nm, tag in (('B0', 'P18'), ('B2', 'P19')):
        if key in VMAP:
            kind, val = VMAP[key]
            b = book(variant(nm, val)) if kind == 'v' else book(variant(nm, 'base'), kill_k=val)
        else:
            b = book(gate(nm, GMAP[key]))
        for sp in S.SPLITS:
            d = b[b.split == sp]
            if len(d) < 30:
                continue
            p(f'    {tag} {nm} {sp}: net {d.net.mean():+.3f} -> ex-top1% '
              f'{d.net[d.net <= d.net.quantile(0.99)].mean():+.3f} -> ex-top5% '
              f'{d.net[d.net <= d.net.quantile(0.95)].mean():+.3f}')
    OUT.close()


if __name__ == '__main__':
    main()
