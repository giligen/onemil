"""Stage M — summarise the 18 exit-shape x slot-count cells (+ sizing rows).

Reads the books written by run_grid.sh; writes the tables the report is built
from.  Nothing here fits anything: it is arithmetic on the shipped selector's
output.
"""
from __future__ import annotations
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, '/home/ec2-user/onemil')
os.chdir('/home/ec2-user/onemil')
from trading.orb_csv import read_orb_csv  # noqa: E402

D = 'research/fuckup_audit/M'
SHAPES = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5']
NS = [3, 8, 12]
pd.set_option('display.width', 260)


def split_of(d: str) -> str:
    s = str(d)[:7]
    if s < '2026-01':
        return 'TRAIN 2025'
    if s <= '2026-05':
        return 'VAL 2026-01..05'
    return 'TEST 2026-06+'


def mdd(daily: pd.Series) -> float:
    cum = daily.cumsum()
    return float((cum - cum.cummax()).min())


def load(tag):
    p = f'{D}/book_{tag}.csv'
    if not os.path.exists(p):
        return None
    b = read_orb_csv(p)
    b['date'] = pd.to_datetime(b['date'])
    b['day'] = b['date'].dt.strftime('%Y-%m-%d')
    b['month'] = b['day'].str[:7]
    b['week'] = b['date'].dt.strftime('%G-W%V')
    b['split'] = b['day'].apply(split_of)
    b['fill'] = (b['entered'].astype(float) != 0)
    # dollar risk the sizer actually took (per-position cap binds at stage size)
    b['R$'] = (b['_rp_position'].astype(float)
               * b['range_size_pct'].astype(float).clip(lower=1.0) / 100.0)
    b['R'] = b['_sized_pnl'].astype(float) / b['R$']
    return b


def summarise(b, label):
    daily = b.groupby('day')['_sized_pnl'].sum()
    monthly = b.groupby('month')['_sized_pnl'].sum()
    weekly = b.groupby('week')['_sized_pnl'].sum()
    f = b[b['fill']]
    pnl = float(b['_sized_pnl'].sum())
    # tails
    if len(f):
        k = max(1, int(np.ceil(0.05 * len(f))))
        top = f['_sized_pnl'].nlargest(k)
        ex_top5 = pnl - float(top.sum())
        capped = float(np.minimum(f['_sized_pnl'].astype(float),
                                  3.0 * f['R$'].astype(float)).sum())
    else:
        ex_top5 = pnl
        capped = pnl
    return {
        'cell': label, 'picks': len(b), 'fills': int(b['fill'].sum()),
        'fill_pct': 100 * b['fill'].mean() if len(b) else 0.0,
        'pnl': pnl,
        'wr_fills_pct': 100 * float((f['_sized_pnl'] > 0).mean()) if len(f) else 0.0,
        'mean_$_fill': float(f['_sized_pnl'].mean()) if len(f) else 0.0,
        'mean_R_fill': float(f['R'].mean()) if len(f) else 0.0,
        'mean_R_pick': float(b['R'].mean()) if len(b) else 0.0,
        'n_weeks': int(weekly.size),
        '$_per_week': pnl / max(weekly.size, 1),
        'weeks_green_pct': 100 * float((weekly > 0).mean()) if weekly.size else 0.0,
        'mdd': mdd(daily),
        'worst_month': float(monthly.min()) if monthly.size else 0.0,
        'red_months': int((monthly < 0).sum()),
        'n_months': int(monthly.size),
        '$_per_month': pnl / max(monthly.size, 1),
        'pnl_ex_top5pct': ex_top5,
        'pnl_cap3R': capped,
        # power: the smallest per-fill mean R this cell could have called
        # significant at t = 2 (2 x the standard error of mean R on fills)
        'mde_R_t2': (2.0 * float(f['R'].std(ddof=1)) / np.sqrt(len(f))
                     if len(f) > 1 else float('nan')),
        't_R': (float(f['R'].mean()) / (float(f['R'].std(ddof=1)) / np.sqrt(len(f)))
                if len(f) > 1 and f['R'].std(ddof=1) > 0 else float('nan')),
    }


def main():
    books = {}
    for sh in SHAPES:
        for n in NS:
            t = f'{sh}_n{n}'
            b = load(t)
            if b is not None:
                books[t] = b
    for extra in ('X0_n8_3xlit', 'X0_n8_3xprop'):
        b = load(extra)
        if b is not None:
            books[extra] = b
    for sh in SHAPES:
        for extra in (f'{sh}_n8_3xlit', f'{sh}_n8_3xprop'):
            if extra in books:
                continue
            b = load(extra)
            if b is not None:
                books[extra] = b
    print(f"loaded {len(books)} books: {sorted(books)}")

    rows = []
    for t, b in books.items():
        rows.append(summarise(b, t))
        for sp in ('TRAIN 2025', 'VAL 2026-01..05', 'TEST 2026-06+'):
            sb = b[b['split'] == sp]
            if len(sb):
                rows.append(summarise(sb, f'{t} | {sp}'))
    tab = pd.DataFrame(rows)
    tab.to_csv(f'{D}/summary.csv', index=False)
    print("\n=== ALL CELLS — whole window and per split ===")
    print(tab.to_string(index=False, float_format=lambda x: f'{x:,.1f}'))

    # ---- the gate, evaluated at N=8 ----
    print("\n=== GATE (N=8): total AND mdd must improve on TRAIN and VAL ===")
    base = {sp: summarise(books['X0_n8'][books['X0_n8']['split'] == sp], 'x')
            for sp in ('TRAIN 2025', 'VAL 2026-01..05', 'TEST 2026-06+')}
    for sh in SHAPES[1:]:
        t = f'{sh}_n8'
        if t not in books:
            continue
        b = books[t]
        verdicts = []
        for sp in ('TRAIN 2025', 'VAL 2026-01..05'):
            s = summarise(b[b['split'] == sp], 'x')
            dp = s['pnl'] - base[sp]['pnl']
            dm = s['mdd'] - base[sp]['mdd']
            ok = (dp > 50) and (dm > 25)
            verdicts.append(ok)
            print(f"  {sh} {sp:16s} dP&L {dp:+9,.0f}  dMDD {dm:+9,.0f}  "
                  f"{'PASS' if ok else 'fail'}")
        print(f"  {sh} -> {'PASSES THE GATE' if all(verdicts) else 'does not pass'}")

    # ---- per-month, per shape, at N=8 ----
    mrows = {}
    for sh in SHAPES:
        t = f'{sh}_n8'
        if t in books:
            mrows[sh] = books[t].groupby('month')['_sized_pnl'].sum()
            mrows[sh + '_n'] = books[t].groupby('month').size()
    mt = pd.DataFrame(mrows).fillna(0)
    mt.to_csv(f'{D}/monthly_n8.csv')
    print("\n=== per month at N=8 (P&L and picks per shape) ===")
    print(mt.to_string(float_format=lambda x: f'{x:,.0f}'))

    # ---- exit mix ----
    print("\n=== exit-reason mix (N=8, picks) ===")
    mix = {}
    for sh in SHAPES:
        t = f'{sh}_n8'
        if t in books:
            mix[sh] = books[t]['exit_reason'].value_counts()
    mx = pd.DataFrame(mix).fillna(0).astype(int)
    mx.to_csv(f'{D}/exit_mix_n8.csv')
    print(mx.to_string())

    print("\n=== exit-reason P&L (N=8) ===")
    pmix = {}
    for sh in SHAPES:
        t = f'{sh}_n8'
        if t in books:
            pmix[sh] = books[t].groupby('exit_reason')['_sized_pnl'].sum()
    pm = pd.DataFrame(pmix).fillna(0)
    pm.to_csv(f'{D}/exit_pnl_n8.csv')
    print(pm.to_string(float_format=lambda x: f'{x:,.0f}'))

    # ---- how many picks actually changed vs X0 (book level, N=8) ----
    print("\n=== picks whose exit differs from X0 (N=8 book, same picks by construction) ===")
    b0 = books['X0_n8'].set_index(['symbol', 'day'])
    for sh in SHAPES[1:]:
        t = f'{sh}_n8'
        if t not in books:
            continue
        b = books[t].set_index(['symbol', 'day'])
        common = b0.index.intersection(b.index)
        diff = (b0.loc[common, 'exit_reason'].astype(str).values
                != b.loc[common, 'exit_reason'].astype(str).values)
        dpnl = (b.loc[common, '_sized_pnl'].values
                - b0.loc[common, '_sized_pnl'].values)
        print(f"  {sh}: {int(diff.sum())}/{len(common)} picks changed exit "
              f"({100*diff.mean():.1f}%), net dP&L {dpnl.sum():+,.0f}, "
              f"of the changed: won {int((dpnl[diff] > 0).sum())} "
              f"lost {int((dpnl[diff] < 0).sum())}")

    # ---- participation (N=8, the live book) ----
    print("\n=== participation of the 5-min opening-range dollar volume (N=8) ===")
    for sh in SHAPES:
        t = f'{sh}_n8'
        if t not in books:
            continue
        b = books[t]
        rd = (b['range_total_volume'].astype(float)
              * b['entry_price'].astype(float))
        for k, lbl in ((1, 'stage $3,333'), (3, '3x $10,000')):
            p = 100 * b['_rp_position'].astype(float) * k / rd
            print(f"  {sh} {lbl:14s} median {p.median():.2f}%  p75 {p.quantile(.75):.2f}%  "
                  f"p90 {p.quantile(.9):.2f}%  >5%: {int((p > 5).sum())}/{len(b)}")
        break_after = os.environ.get('M_PART_ALL')
        if not break_after:
            break

    print(f"\nWrote {D}/summary.csv, {D}/monthly_n8.csv, {D}/exit_mix_n8.csv, "
          f"{D}/exit_pnl_n8.csv")


if __name__ == '__main__':
    main()
