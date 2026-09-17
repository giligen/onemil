#!/usr/bin/env python3
"""Stage I Part A — the 8 declared cells on TRAIN and VAL (TEST untouched here).

Also: the S0 parity check against H/F6/f6_pdr_book.md, the marginal contribution of each added family, and the
frequency-vs-displacement decomposition. Writes I/a_trainval.md + I/a_cells.csv.
"""
import numpy as np, pandas as pd
from icore import D, STACKS, load, weeks_of, dedupe, book, stats

OUT = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    OUT.append(s); print(s, flush=True)


c = load()
W = weeks_of(c)
say('# Stage I Part A — the stacked intraday book, TRAIN and VAL (TEST not read here)')
say('')
say(f'population rows {len(c):,} | ' + ' '.join(f'{k} {v}' for k, v in c.tag.value_counts().items()))
say('')

# ---------- 0. parity: S0 must reproduce H/F6/f6_pdr_book.md ----------
say('## 0. Parity — S0 (F6 alone) vs H/F6/f6_pdr_book.md `exit hold | PDR>=8`')
say('')
REF = {('hold', 'TRAIN'): (1122, 0.078, 1.81), ('hold', 'VAL'): (519, 0.251, 3.28),
       ('2r', 'TRAIN'): (1217, 0.058, 2.00), ('2r', 'VAL'): (593, 0.110, 2.56)}
say('| exit | split | n | ref n | meanR | ref meanR | t | ref t | match |')
say('|---|---|---:|---:|---:|---:|---:|---:|---|')
s0 = dedupe(c, STACKS['S0'])
for tag in ('hold', '2r'):
    for sp in ('TRAIN', 'VAL'):
        st = stats(book(s0[s0.split == sp], tag), sp, W)
        rn, rm, rt = REF[(tag, sp)]
        ok = (st['n'] == rn) and abs(st['meanR'] - rm) < 5e-4 and abs(st['t'] - rt) < 0.02
        say(f"| {tag} | {sp} | {st['n']} | {rn} | {st['meanR']:+.3f} | {rm:+.3f} | {st['t']:.2f} | {rt:.2f} | "
            f"{'EXACT' if ok else 'MISMATCH'} |")
say('')

# ---------- 1. the 8 cells ----------
cells = []
books = {}
say('## 1. The 8 declared cells')
say('')
say('| stack | families | exit | split | n | tr/wk | mean net R | gross | t | WR% | wkR | weeks green | '
    'worst wk | MDD | months green | ex-top5% | cap+3R | family mix |')
say('|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|')
for sname, fams in STACKS.items():
    dd = dedupe(c, fams)
    for tag in ('hold', '2r'):
        for sp in ('TRAIN', 'VAL'):
            t = book(dd[dd.split == sp], tag)
            books[(sname, tag, sp)] = t
            st = stats(t, sp, W)
            st.update(stack=sname, exit=tag, fams='+'.join(fams))
            cells.append(st)
            say(f"| {sname} | {'+'.join(fams)} | {tag} | {sp} | {st['n']} | {st['tpw']} | {st['meanR']:+.3f} | "
                f"{st['gross']:+.3f} | {st['t']:.2f} | {st['WR']} | {st['wkR']:+.2f} | {st['green']:.2f} | "
                f"{st['worst']:.1f} | {st['mdd']:.1f} | {st['moG']} | {st['ex5']:+.3f} | {st['cap3']:+.3f} | "
                f"{st['mix']} |")
say('')
pd.DataFrame(cells).to_csv(f'{D}/a_cells.csv', index=False)

# ---------- 2. freeze rule ----------
say('## 2. The freeze (declared rule: highest VAL weekly R)')
say('')
vv = [x for x in cells if x['split'] == 'VAL']
vv.sort(key=lambda x: -x['wkR'])
say('| rank | stack | exit | VAL wkR | VAL meanR | VAL t | VAL weeks green |')
say('|---:|---|---|---:|---:|---:|---:|')
for i, x in enumerate(vv, 1):
    say(f"| {i} | {x['stack']} | {x['exit']} | {x['wkR']:+.2f} | {x['meanR']:+.3f} | {x['t']:.2f} | {x['green']:.2f} |")
best = vv[0]
say('')
say(f"**FROZEN: stack {best['stack']} ({best['fams']}), exit `{best['exit']}`** — VAL weekly R {best['wkR']:+.2f}. "
    f"TEST is read once for this cell and for S0 in `i_test.py`; nothing below changed the choice.")
say('')
open(f'{D}/FREEZE.md', 'w').write(
    f"# Stage I — frozen stack (written before TEST was read)\n\n"
    f"stack = {best['stack']} ({best['fams']})\nexit = {best['exit']}\n"
    f"rule = highest VAL weekly R over the 8 declared cells (PREREG.md Part A)\n"
    f"VAL: n {best['n']}, {best['tpw']} tr/wk, mean net R {best['meanR']:+.3f}, t {best['t']:.2f}, "
    f"weekly R {best['wkR']:+.2f}, weeks green {best['green']:.2f}\n\n"
    f"TEST will be read once, for this cell and for S0 {best['exit']}.\n")

# ---------- 3. marginal contribution ----------
say('## 3. Marginal contribution of each added family (booked trades, both splits, both exits)')
say('')
say('| exit | split | step | n booked | delta n | mean net R | total net R | delta total R | trades the NEW family '
    'contributed | their mean net R | their total R |')
say('|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|')
order = ['S0', 'S1', 'S2', 'S3']
for tag in ('hold', '2r'):
    for sp in ('TRAIN', 'VAL'):
        prev = None
        for sname in order:
            t = books[(sname, tag, sp)]
            new_fam = STACKS[sname][-1] if sname != 'S0' else 'F6'
            sub = t[t.tag == new_fam]
            dn = len(t) - (len(prev) if prev is not None else 0)
            dtot = t.net.sum() - (prev.net.sum() if prev is not None else 0.0)
            say(f"| {tag} | {sp} | {sname} (+{new_fam}) | {len(t)} | {dn:+d} | {t.net.mean():+.3f} | "
                f"{t.net.sum():+.1f} | {dtot:+.1f} | {len(sub)} | "
                f"{sub.net.mean():+.3f} | {sub.net.sum():+.1f} |" if len(sub) else
                f"| {tag} | {sp} | {sname} (+{new_fam}) | {len(t)} | {dn:+d} | {t.net.mean():+.3f} | "
                f"{t.net.sum():+.1f} | {dtot:+.1f} | 0 | - | - |")
            prev = t
say('')

# ---------- 4. frequency vs displacement ----------
say('## 4. Where the stack\'s extra trades come from: added frequency vs displacement of an F6 trade')
say('')
say('A booked trade of the stack is classified against the S0 (F6-alone) book of the same exit/split:')
say('- **kept**: the same (day, symbol) is booked in S0 too;')
say('- **added**: not in S0 and S0 booked FEWER than 4 trades open at that minute and < 12 that day '
    '(the slot was genuinely free);')
say('- **displaced-in**: not in S0 and S0 was at its 4-concurrent or 12/day limit at that minute '
    '(this trade took a slot an F6 trade would have had).')
say('And the mirror: S0 trades the stack DROPPED.')
say('')
say('| exit | split | stack | kept | added (free slot) | displaced-in | S0 trades dropped | net R of added | '
    'net R of displaced-in | net R of dropped S0 trades |')
say('|---|---|---|---:|---:|---:|---:|---:|---:|---:|')


for tag in ('hold', '2r'):
    for sp in ('TRAIN', 'VAL'):
        t0 = books[('S0', tag, sp)]
        k0 = set(zip(t0.day, t0.symbol))
        byday = {d_: (g.em.values, g.xm.values) for d_, g in t0.groupby('day')}
        for sname in ('S1', 'S2', 'S3'):
            t = books[(sname, tag, sp)]
            ks = set(zip(t.day, t.symbol))
            kept = t[[(d_, s_) in k0 for d_, s_ in zip(t.day, t.symbol)]]
            new = t[[(d_, s_) not in k0 for d_, s_ in zip(t.day, t.symbol)]]
            free, disp = [], []
            for r in new.itertuples():
                em0, xm0 = byday.get(r.day, (np.array([]), np.array([])))
                op = int(((em0 <= r.em) & (xm0 >= r.em)).sum())
                tk = int((em0 <= r.em).sum())
                (free if (op < 4 and tk < 12) else disp).append(r.net)
            drop = t0[[(d_, s_) not in ks for d_, s_ in zip(t0.day, t0.symbol)]]
            say(f'| {tag} | {sp} | {sname} | {len(kept)} | {len(free)} | {len(disp)} | {len(drop)} | '
                f'{np.sum(free):+.1f} | {np.sum(disp):+.1f} | {drop.net.sum():+.1f} |')
say('')

open(f'{D}/a_trainval.md', 'w').write('\n'.join(OUT) + '\n')
print('wrote', f'{D}/a_trainval.md', flush=True)
