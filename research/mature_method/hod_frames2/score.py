#!/usr/bin/env python3
"""hod_frames2 — the scorer.  Cells exactly as declared in PREREG.md (commit 1f26c7a, before any
cell was scored).  TEST is sealed: no TEST number unless FREEZE.md exists AND --test is passed.
Read-only on every DB.  One process."""
import os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT); sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_break')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')
import score as S            # noqa: E402
import score2 as S2          # noqa: E402
from research.scripts.pit_listings import is_test_ticker   # noqa: E402

D = f'{ROOT}/research/mature_method/hod_frames2'
RD = dict(dtype={'symbol': str, 'day': str, 'why_n': str}, keep_default_na=False, na_values=[''])
WITH_TEST = '--test' in sys.argv and os.path.exists(f'{D}/FREEZE.md')
S.SPLITS = S2.SPLITS = SPLITS = ('TRAIN', 'VAL', 'TEST') if WITH_TEST else ('TRAIN', 'VAL')
HDR = ('| cell                               | split | n     | /wk   | grossR | cost   | netR   |'
       '  t    | tc    | green | rs | worst $  | total $   | MDD $    | ex5    | imp |')
SEP = '|' + '|'.join(['-' * 6] * 16) + '|'
CELLS, BOOKS = [], {}


def clustered_t(d, col='net'):
    if len(d) < 3:
        return np.nan
    x = d[col].values.astype(float); mu = x.mean()
    g = pd.Series(x - mu).groupby(d.day.values).sum().values
    se = np.sqrt((g ** 2).sum()) / len(x)
    return float(mu / se) if se > 0 else np.nan


def halves(b):
    tr = b[b.split == 'TRAIN']
    parts = (tr[tr.day < '2025-07-01'], tr[tr.day >= '2025-07-01'], b[b.split == 'VAL'])
    g = [float(x.rr.mean()) if len(x) else np.nan for x in parts]
    return g, [len(x) for x in parts], all(v == v and v > 0 for v in g)


def show(name, b, note=''):
    BOOKS[name] = b
    out = [HDR, SEP] if not CELLS else []
    for sp in SPLITS:
        w = S.week_stats(b, sp); d = b[b.split == sp]
        cost = float((d.rr - d.net).mean()) if len(d) else np.nan
        out.append(f'| {name:<34s} | {sp:5s} | {w["n"]:5d} | {w["per_wk"]:5.1f} | {w["gross"]:+.3f} | '
                   f'{cost:+.3f} | {w["net"]:+.3f} | {w["t"]:+5.2f} | {clustered_t(d):+5.2f} | '
                   f'{w["green"]:5.1f} | {w["redstreak"]:2d} | {w["worst"]:8.0f} | {w["total"]:9.0f} | '
                   f'{w["mdd"]:8.0f} | {w["ex5"]:+.3f} | {w["imp"]:3.0f} |')
    print('\n'.join(out), flush=True)
    g, n, ok = halves(b)
    for sp in SPLITS:
        w = S.week_stats(b, sp); d = b[b.split == sp]
        w.update(cell=name, split=sp, tc=clustered_t(d),
                 cost=float((d.rr - d.net).mean()) if len(d) else np.nan,
                 corr_after=float(np.corrcoef(d.rr, d.rng_after)[0, 1]) if len(d) > 5 and
                 'rng_after' in d and d.rng_after.notna().all() else np.nan,
                 h1=g[0], h2=g[1], vgross=g[2], half_ok=ok, note=note)
        CELLS.append(w)
    print(f'    halves H1 {g[0]:+.3f} (n {n[0]}) | H2 {g[1]:+.3f} (n {n[1]}) | VAL {g[2]:+.3f} '
          f'(n {n[2]}) -> same-signed POSITIVE in all three: {ok}', flush=True)
    return b


def sigset(d, min_price=20.0, r_min=1.0, max_bps=100.0, max_frac_r=0.15, last_m=840):
    """The shipped pre-book cascade, identical to score2.sig_set but on an arbitrary break set."""
    d = d[(d.entry_m <= last_m + 1) & d.r_pct_n.notna() & (d.fill_capped == 1) &
          (d.r_pct_n >= r_min) & (d.next_open >= min_price)]
    x = S.attach_cost(d, 'n')
    x = x[(x.sp_pct * 100) <= max_bps]
    x = x[(x.sp_pct / x.r_pct.clip(lower=0.05)) <= max_frac_r]
    return x[x.obtainable.astype(bool)]


def daily_ctx():
    """ADV$ and the 20-prior-session median daily range %, from universe.csv — strictly prior."""
    u = pd.read_csv(f'{ROOT}/research/bf_zero/universe.csv',
                    usecols=['symbol', 'bar_date', 'open', 'high', 'low', 'close', 'volume'],
                    dtype={'symbol': str}, keep_default_na=False, na_values=[''])
    u = u.rename(columns={'bar_date': 'day'})
    for k in ('open', 'high', 'low', 'close', 'volume'):
        u[k] = pd.to_numeric(u[k], errors='coerce')
    u = u.sort_values(['symbol', 'day'], kind='mergesort')
    u['dv'] = u.close * u.volume
    u['rngp'] = (u.high - u.low) / u.open.replace(0, np.nan) * 100
    g = u.groupby('symbol', sort=False)
    u['adv_dollar'] = g.dv.transform(lambda s: s.rolling(20, min_periods=15).mean().shift(1))
    u['med_rng'] = g.rngp.transform(lambda s: s.rolling(20, min_periods=15).median().shift(1))
    return u[['day', 'symbol', 'adv_dollar', 'med_rng']]


def downstream(pre, fields, label):
    """PREREG §2 — the mandatory already-there / arrived-after gate.  Returns the excluded set."""
    print(f'\n== DOWNSTREAM GATE ({label}) — PREREG §2, applied BEFORE any cell is scored ==')
    print('| field | rung | split | n top | P(EOD>=10%) top | rest | dP | d mean rng_sig | '
          'd mean rng_after | gross top | gross rest | EXCLUDED |')
    print('|' + '|'.join(['---'] * 12) + '|')
    excl = set()
    for f, thr in fields:
        for sp in SPLITS:
            d = pre[(pre.split == sp) & pre[f].notna()]
            if len(d) < 50:
                continue
            top = d[d[f] >= thr]; rest = d[d[f] < thr]
            if len(top) < 10 or len(rest) < 10:
                print(f'| {f} | >={thr} | {sp} | {len(top)} | — | — | — | — | — | — | — | (too few) |')
                continue
            dp = float((top.rng_day >= 10).mean() - (rest.rng_day >= 10).mean()) * 100
            ds = float(top.rng_sig.mean() - rest.rng_sig.mean())
            da = float(top.rng_after.mean() - rest.rng_after.mean())
            ex = (da <= 0) and (ds > 0)
            if sp == 'TRAIN' and ex:
                excl.add(f)
            print(f'| {f} | >={thr} | {sp} | {len(top)} | {(top.rng_day>=10).mean():.1%} | '
                  f'{(rest.rng_day>=10).mean():.1%} | {dp:+.1f}pp | {ds:+.2f}pp | {da:+.2f}pp | '
                  f'{top.rr.mean():+.3f} | {rest.rr.mean():+.3f} | {"YES" if ex else "no"} |')
        # §2.3's own split inside the top rung
        for sp in SPLITS:
            d = pre[(pre.split == sp) & pre[f].notna()]
            top = d[d[f] >= thr]
            if len(top) < 30:
                continue
            med = top.rng_sig.median()
            a = top[top.rng_sig >= med]; b = top[top.rng_sig < med]
            print(f'    §2.3 split inside `{f} >= {thr}` {sp}: already-wide (rng_sig >= median '
                  f'{med:.1f}%) n {len(a)} gross {a.rr.mean():+.3f} WR {(a.rr>0).mean():.1%} | '
                  f'arrived-after n {len(b)} gross {b.rr.mean():+.3f} WR {(b.rr>0).mean():.1%}')
    return excl


def main():
    print('== hod_frames2 — F5 retest / F6 absorption / F9 signal-minute cohort fields ==')
    print(f'   splits scored: {SPLITS}   (TEST sealed: {not WITH_TEST})\n', flush=True)

    pop = S2.load_pop(); S.build_impute(pop)
    sig = S2.sig_set(pop, **S2.BASES['B2'])
    b2 = S.apply_book(sig, 12, 4)
    print('== STEP 0 — REPRODUCTION GATE (ref B2: 1,622/30.6/-0.039/-0.107/32.1%/-17,346 | '
          '706/30.7/+0.083/+0.013/43.5%/+893) ==')
    b2['rng_after'] = np.nan
    show('R1/R2 B2 shipped (pop.csv)', b2)

    # ---------------- this pass's own break set -------------------------------------------
    br = pd.read_csv(f'{D}/breaks2.csv', **RD)
    n0 = len(br)
    br = br[~br.day.isin(S.EARLY_CLOSE)]
    br = br[~br.symbol.map(lambda s: is_test_ticker(str(s)))]
    br['split'] = S.split_of(br.day.values)
    br['wk'] = pd.to_datetime(br.day).dt.to_period('W-FRI').astype(str)
    nb = pd.read_csv(f'{ROOT}/research/bf_zero/causal_filter/nbbo.csv',
                     dtype={'symbol': str, 'day': str}, keep_default_na=False,
                     na_values=['']).drop_duplicates(['day', 'symbol', 'entry_m'])
    br = br.merge(nb[['day', 'symbol', 'entry_m', 'spread_mean', 'ask_dec', 'bid_dec']],
                  on=['day', 'symbol', 'entry_m'], how='left')
    dc = daily_ctx()
    br = br.merge(dc, on=['day', 'symbol'], how='left')
    br = br.sort_values(['day', 'symbol', 'entry_m'], kind='mergesort').reset_index(drop=True)
    br['rng_after'] = br.rng_day - br.rng_sig
    br['dollar_frac'] = br.cum_dollar / br.adv_dollar.replace(0, np.nan) * 100
    br['add30_ratio'] = np.where(br.break_m >= 630,
                                 (br.rng_sig - br.rng_30) / br.rng_first30.replace(0, np.nan), np.nan)
    br['rng_own'] = br.rng_sig / br.med_rng.replace(0, np.nan)
    print(f'\nbreak rows {n0} -> {len(br)} after membership | symbol-days '
          f'{br.drop_duplicates(["day","symbol"]).shape[0]} | days {br.day.nunique()} | '
          f'n_prior>=1 {int((br.n_prior>=1).sum())}')

    # ---------------- R3: the independent rebuild ------------------------------------------
    mine = br[br.n_prior == 0][['day', 'symbol', 'entry_m', 'stop_n', 'rr_n']]
    ref = pop[pop.first_n0 == 1][['day', 'symbol', 'entry_m', 'stop_n', 'rr_n']]
    ref = ref[(ref.day >= '2025-01-02') & (ref.day <= '2026-05-31')]
    j = mine.merge(ref, on=['day', 'symbol'], suffixes=('', '_ref'))
    same = (j.entry_m == j.entry_m_ref)
    dstop = np.abs(j.stop_n[same] - j.stop_n_ref[same]).max()
    drr = np.abs(j.rr_n[same] - j.rr_n_ref[same]).max()
    print(f'== R3 INDEPENDENT REBUILD: {len(j)} shared symbol-days of {len(ref)} reference | '
          f'same entry minute {same.mean():.4%} | max |d stop| {dstop:.2e} | max |d rr| {drr:.2e} ==')

    PRE = sigset(br)
    PRE0 = PRE[PRE.n_prior == 0]
    print(f'pre-book: all qualifying breaks {len(PRE)} | first-break only {len(PRE0)} '
          f'(reference B2 pre-book {len(sig)})')
    show('R3b B2 rebuilt (this pass)', S.apply_book(PRE0, 12, 4))

    # ---------------- availability audit ----------------------------------------------------
    print('\n== AVAILABILITY AUDIT (B2 pre-book, this pass) — coverage, missingness win vs loss ==')
    for c in ('shelf_share', 'shelf_bars', 'hod_age_bars', 'dollar_frac', 'exp5_n', 'add30_ratio',
              'rng_own', 'rng_sig', 'rng_day', 'prev_stopped'):
        v = pd.to_numeric(PRE0[c], errors='coerce'); win = PRE0.rr > 0
        mw, ml = float(v[win].isna().mean()), float(v[~win].isna().mean())
        flag = '  <-- DROP (outcome-dependent missingness)' if abs(mw - ml) > 0.05 else ''
        print(f'  {c:<14s} cov {v.notna().mean():6.1%}  miss win {mw:6.1%} loss {ml:6.1%}{flag}')

    # ================= F5 — the RETEST book ==============================================
    print('\n\n================ F5 — THE RETEST BOOK (7 cells) ================')
    print(f'qualifying breaks per symbol-day (pre-book): '
          f'{PRE.groupby(["day","symbol"]).size().value_counts().head(6).to_dict()}')
    for sp in SPLITS:
        d = PRE[PRE.split == sp]
        print(f'  {sp}: first {int((d.n_prior==0).sum())} | 2nd {int((d.n_prior==1).sum())} | '
              f'3rd+ {int((d.n_prior>=2).sum())} | of 2nd: prev_stopped '
              f'{int(((d.n_prior==1)&(d.prev_stopped==1)).sum())} back5 '
              f'{int(((d.n_prior==1)&(d.prev_back5==1)).sum())} back15 '
              f'{int(((d.n_prior==1)&(d.prev_back15==1)).sum())}')
    F5 = {
        'F5-0 first break only (=B2)': PRE.n_prior == 0,
        'F5-a 2nd, prev STOPPED': (PRE.n_prior == 1) & (PRE.prev_stopped == 1),
        'F5-b 2nd, prev back<5 bars': (PRE.n_prior == 1) & (PRE.prev_back5 == 1),
        'F5-c 2nd, prev back<15 bars': (PRE.n_prior == 1) & (PRE.prev_back15 == 1),
        'F5-d 2nd, stopped OR back15': (PRE.n_prior == 1) & ((PRE.prev_stopped == 1) |
                                                             (PRE.prev_back15 == 1)),
        'F5-e ANY re-break [ctrl]': PRE.n_prior >= 1,
        'F5-f 3rd+ break [ctrl]': PRE.n_prior >= 2,
    }
    for nm, mk in F5.items():
        show(nm, S.apply_book(PRE[mk], 12, 4))
    print('\n-- F5 downstream check: corr(gross R, range added after the signal) --')
    for nm in F5:
        b = BOOKS[nm]
        row = ' '.join(f'{sp} {np.corrcoef(b[b.split==sp].rr, b[b.split==sp].rng_after)[0,1]:+.3f}'
                       for sp in SPLITS if len(b[b.split == sp]) > 5)
        print(f'  {nm:<34s} {row}')

    # ================= the downstream gate, then F6 and F9 ================================
    F9F = [('dollar_frac', 50.0), ('exp5_n', 6.0), ('add30_ratio', 2.0), ('rng_own', 1.5)]
    excl = downstream(PRE0, F9F + [('shelf_share', 20.0)], 'F9 candidates + F6 shelf')
    print(f'\nEXCLUDED by the pre-committed rule (membership lift is ONLY the already-there '
          f'channel): {sorted(excl) if excl else "none"}')

    print('\n\n================ F6 — ABSORPTION AT THE LEVEL (9 cells) ================')
    for sp in SPLITS:
        d = PRE0[PRE0.split == sp]
        print(f'  {sp} shelf_share pct: p10 {d.shelf_share.quantile(.1):.1f} p50 '
              f'{d.shelf_share.median():.1f} p90 {d.shelf_share.quantile(.9):.1f} | shelf_bars p50 '
              f'{d.shelf_bars.median():.0f} p90 {d.shelf_bars.quantile(.9):.0f} | hod_age p50 '
              f'{d.hod_age_bars.median():.0f}')
    F6 = {}
    for x in (2, 5, 10, 20):
        F6[f'F6 shelf>={x}% of ADV'] = PRE0.shelf_share >= x
    F6['F6-e shelf>=5% AND age>=20'] = (PRE0.shelf_share >= 5) & (PRE0.hod_age_bars >= 20)
    F6['F6-f age>=20 alone [arm]'] = PRE0.hod_age_bars >= 20
    for x in (5, 10, 20):
        F6[f'F6 shelf_bars>={x}'] = PRE0.shelf_bars >= x
    for nm, mk in F6.items():
        show(nm, S.apply_book(PRE0[mk.fillna(False)], 12, 4))

    print('\n\n================ F9 — SIGNAL-MINUTE COHORT FIELDS (10 cells) ================')
    F9 = {}
    for x in (10, 25, 50):
        F9[f'F9-a dollar_frac>={x}%'] = ('dollar_frac', PRE0.dollar_frac >= x)
    for x in (3, 6):
        F9[f'F9-b exp5_n>={x}'] = ('exp5_n', PRE0.exp5_n >= x)
    for x in (1.0, 2.0):
        F9[f'F9-c add30_ratio>={x}'] = ('add30_ratio', PRE0.add30_ratio >= x)
    for x in (0.5, 1.0, 1.5):
        F9[f'F9-d rng_own>={x}'] = ('rng_own', PRE0.rng_own >= x)
    for nm, (f, mk) in F9.items():
        if f in excl:
            print(f'| {nm:<34s} | EXCLUDED by the PREREG §2 downstream gate — not scored |')
            continue
        show(nm, S.apply_book(PRE0[mk.fillna(False)], 12, 4))

    # ================= nulls + bars ======================================================
    print('\n\n== count-matched permutation null on green weeks (2,000 draws) ==')
    print('| cell | split | observed green % | null mean | [p5, p95] | outside? |')
    print('|---|---|---|---|---|---|')
    nulls = []
    for nm, bb in BOOKS.items():
        for sp in SPLITS:
            obs, mu_, p5, p95 = S.null_band(bb, sp)
            o = 'ABOVE' if obs == obs and obs > p95 else ('below' if obs == obs and obs < p5 else 'inside')
            print(f'| {nm} | {sp} | {obs:.1f} | {mu_:.1f} | [{p5:.1f}, {p95:.1f}] | {o} |')
            nulls.append(dict(cell=nm, split=sp, obs=obs, mu=mu_, p5=p5, p95=p95, outside=o))
    pd.DataFrame(nulls).to_csv(f'{D}/nulls.csv', index=False)
    cf = pd.DataFrame(CELLS); cf.to_csv(f'{D}/cells.csv', index=False)

    print('\n== BOTH BARS ==')
    g1 = cf[(cf.split == 'TRAIN') & (cf.net > 0) & (cf.t >= 2.0) & (cf.tc >= 2.0) & (cf.per_wk >= 10)]
    print(f'G1 (TRAIN net>0, iid t>=2, CLUSTERED t>=2, >=10/wk): {len(g1)} of '
          f'{cf[cf.split=="TRAIN"].cell.nunique()} cells -> {list(g1.cell)}')
    ship = []
    for nm in cf.cell.unique():
        t_ = cf[(cf.cell == nm) & (cf.split == 'TRAIN')].iloc[0]
        v_ = cf[(cf.cell == nm) & (cf.split == 'VAL')].iloc[0]
        if (t_.total > 0 and v_.total > 0 and t_.green >= 50 and v_.green >= 50 and
                min(t_.per_wk, v_.per_wk) >= 10 and t_.tc >= 2.0 and t_.half_ok):
            ship.append(nm)
    print(f'LIVE-EXPLORATION BAR ($>0 both, green>=50% both, >=10/wk both, clustered t>=2 TRAIN, '
          f'halves same-signed): {len(ship)} -> {ship}')
    print('\nMDE (80% power, per trade, net) per cell:')
    for nm, bb in BOOKS.items():
        row = ' '.join(f'{sp} {2.80*bb[bb.split==sp].net.std(ddof=1)/np.sqrt(max(len(bb[bb.split==sp]),1)):.3f}'
                       for sp in SPLITS if len(bb[bb.split == sp]) > 2)
        print(f'  {nm:36s} {row}')
    print('\nDONE', flush=True)


if __name__ == '__main__':
    main()
