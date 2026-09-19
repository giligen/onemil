#!/usr/bin/env python3
"""hod_frames4 / F15 — WHO IS BREAKING OUT: the instrument, not the bar.

Cells exactly as declared in PREREG.md §3 (commit 92db20a, before any cell was scored).
Every field is point-in-time; every field goes through the availability audit FIRST.
TEST sealed.  Read-only.  One process.
"""
import sys
import numpy as np, pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/mature_method/hod_frames4')
from common4 import ROOT, D4, S, SPLITS, clustered_t, halves, book_ranked   # noqa: E402
from trading.orb_asset_class import classify_asset, load_class_map, underlying_anchor  # noqa: E402
from research.scripts.pit_listings import PitListings   # noqa: E402

HDR = ('| cell                                  | split | n     | /wk   | grossR | cost   | netR   |'
       '  t    | tc    | green | worst $  | total $   | MDD $    | >=10% |')
SEP = '|' + '|'.join(['-' * 6] * 14) + '|'
CELLS, BOOKS = [], {}


def show(name, b, note='', first=[True]):
    BOOKS[name] = b
    if first[0]:
        print(HDR); print(SEP); first[0] = False
    for sp in SPLITS:
        w = S.week_stats(b, sp); d = b[b.split == sp]
        cost = float((d.rr - d.net).mean()) if len(d) else np.nan
        big = float((d.day_range_pct >= 10).mean() * 100) if len(d) else np.nan
        print(f'| {name:<37s} | {sp:5s} | {w["n"]:5d} | {w["per_wk"]:5.1f} | {w["gross"]:+.3f} | '
              f'{cost:+.3f} | {w["net"]:+.3f} | {w["t"]:+5.2f} | {clustered_t(d):+5.2f} | '
              f'{w["green"]:5.1f} | {w["worst"]:8.0f} | {w["total"]:9.0f} | {w["mdd"]:8.0f} | '
              f'{big:5.0f} |', flush=True)
    g, n, ok = halves(b)
    for sp in SPLITS:
        w = S.week_stats(b, sp); d = b[b.split == sp]
        w.update(cell=name, split=sp, tc=clustered_t(d), note=note,
                 cost=float((d.rr - d.net).mean()) if len(d) else np.nan,
                 mde=2.80 * float(d.net.std(ddof=1) / np.sqrt(len(d))) if len(d) > 2 else np.nan,
                 h1=g[0], h2=g[1], vgross=g[2], half_ok=ok)
        CELLS.append(w)
    # the already-there / arrived-after diagnostic (reported, never an exclusion)
    for sp in SPLITS:
        d = b[b.split == sp]
        if len(d) < 20:
            continue
        aw = d[d.rng_sig >= 10]; aa = d[d.rng_sig < 10]
        print(f'      [2.3 diag {sp}] already-wide n {len(aw)} gross {aw.rr.mean():+.3f} | '
              f'arrived-after n {len(aa)} gross {aa.rr.mean():+.3f}', flush=True)
    print(f'    halves H1 {g[0]:+.3f} (n {n[0]}) | H2 {g[1]:+.3f} (n {n[1]}) | VAL {g[2]:+.3f} '
          f'(n {n[2]}) -> same-signed POSITIVE in all three: {ok}', flush=True)
    return b


def attach(s):
    """Every instrument field, point-in-time.  Returns (s, coverage dict)."""
    cov = {}
    # ---- short interest (FINRA, keyed on usable_from = settlement + 13d) --------------------
    si = pd.read_csv(f'{D4}/short_interest.csv', dtype={'symbolCode': str},
                     keep_default_na=False, na_values=[''])
    si['usable_from'] = (pd.to_datetime(si.settlementDate) +
                         pd.Timedelta(days=13)).dt.strftime('%Y-%m-%d')
    si = si.rename(columns={'symbolCode': 'symbol'})
    for c in ('currentShortPositionQuantity', 'daysToCoverQuantity'):
        si[c] = pd.to_numeric(si[c], errors='coerce')
    si = si[['symbol', 'usable_from', 'currentShortPositionQuantity', 'daysToCoverQuantity']]
    si['d'] = pd.to_datetime(si.usable_from)
    si = si.dropna(subset=['d']).sort_values('d', kind='mergesort')
    s = s.copy()
    left = s[['symbol', 'day']].copy()
    left['d'] = pd.to_datetime(left.day); left['_ix'] = s.index
    m = pd.merge_asof(left.sort_values('d', kind='mergesort'), si, on='d', by='symbol',
                      direction='backward').set_index('_ix').reindex(s.index)
    s['si_qty'] = m.currentShortPositionQuantity
    s['days_to_cover'] = m.daysToCoverQuantity
    s['si_report_age'] = (pd.to_datetime(s.day) - pd.to_datetime(m.usable_from)).dt.days
    s['si_ratio'] = s.si_qty / s.adv20.replace(0, np.nan)
    cov['days_to_cover'] = float(s.days_to_cover.notna().mean())
    cov['si_ratio'] = float(s.si_ratio.notna().mean())

    # ---- shares outstanding (EDGAR, keyed on the FILING date) ------------------------------
    ck = pd.read_parquet(f'{ROOT}/research/multiday/data/cik_map.parquet')
    sf = pd.read_parquet(f'{ROOT}/research/multiday/data/shares_facts.parquet')
    sf = sf[sf.tag.str.contains('EntityCommonStockSharesOutstanding', na=False)]
    sf['d'] = pd.to_datetime(sf.filed)
    sf = sf.merge(ck, on='cik', how='inner')[['symbol', 'd', 'val']]
    sf = sf.dropna(subset=['d']).sort_values('d', kind='mergesort')
    m2 = pd.merge_asof(left.sort_values('d', kind='mergesort'), sf, on='d', by='symbol',
                       direction='backward').set_index('_ix').reindex(s.index)
    s['shares_out'] = m2.val
    s['float_turn'] = s.cumv / s.shares_out.replace(0, np.nan)
    cov['shares_out'] = float(s.shares_out.notna().mean())

    # ---- asset class + underlying anchor ----------------------------------------------------
    ass = pd.read_csv(f'{ROOT}/data/research/alpaca_assets_all_20260905.csv',
                      dtype=str, keep_default_na=False, na_values=[''])
    nm = dict(zip(ass.symbol, ass.name))
    cmap = load_class_map()
    uniq = sorted(s.symbol.unique())
    cls = {u: (cmap.get(u) or classify_asset(u, nm.get(u))) for u in uniq}
    anc = {u: underlying_anchor(u, nm.get(u), cmap) for u in uniq}
    s['asset_class'] = s.symbol.map(cls)
    s['anchor'] = s.symbol.map(anc)
    cov['asset_class'] = float((s.asset_class != 'unknown').mean())
    s['anchor_cohort'] = s.groupby(['day', 'anchor']).symbol.transform('size')
    s.loc[s.anchor.isna(), 'anchor_cohort'] = np.nan

    # ---- listing venue (Databento PIT definitions) ------------------------------------------
    pl = PitListings()
    key = s[['symbol', 'day']].drop_duplicates()
    ven = {}
    for r in key.itertuples():
        try:
            ven[(r.symbol, r.day)] = pl.listing_exchange(r.symbol, r.day)
        except Exception:
            ven[(r.symbol, r.day)] = None
    s['venue'] = [ven.get((a, b)) for a, b in zip(s.symbol, s.day)]
    cov['venue'] = float(s.venue.notna().mean())

    # ---- premarket news ---------------------------------------------------------------------
    nw = pd.read_csv(f'{ROOT}/data/research/orb_news_catalyst_nightly.csv',
                     dtype={'symbol': str, 'day': str}, keep_default_na=False, na_values=[''])
    nw['n_articles'] = pd.to_numeric(nw.n_articles, errors='coerce')
    nw = nw.drop_duplicates(['symbol', 'day'])
    s = s.merge(nw[['symbol', 'day', 'n_articles', 'latest']], on=['symbol', 'day'], how='left')
    s['news_covered'] = s.n_articles.notna()
    s['has_news'] = (s.n_articles.fillna(0) >= 1)
    s['news_same_morning'] = (pd.to_datetime(s.latest, errors='coerce', utc=True).dt.strftime(
        '%Y-%m-%d') == s.day)
    cov['news_file'] = float(s.news_covered.mean())
    return s, cov


def main():
    s = pd.read_csv(f'{D4}/sig4.csv', dtype={'symbol': str, 'day': str, 'wk': str, 'split': str,
                                             'why': str}, keep_default_na=False, na_values=[''])
    print(f'== hod_frames4 / F15 — WHO is breaking out ==  pre-book signals {len(s)}\n', flush=True)
    s, cov = attach(s)
    base = book_ranked(s, 12, 4)

    print('== AVAILABILITY AUDIT (pre-book set) — coverage + missingness on winners vs losers ==')
    print('| field | coverage | miss on winners | miss on losers | gap | verdict |')
    print('|---|---|---|---|---|---|')
    drop = set()
    for c in ('days_to_cover', 'si_ratio', 'shares_out', 'float_turn', 'venue', 'n_articles'):
        v = s[c] if c in s else pd.Series(np.nan, index=s.index)
        win = s.rr > 0
        mw, ml = float(v[win].isna().mean()), float(v[~win].isna().mean())
        bad = abs(mw - ml) > 0.05
        if bad:
            drop.add(c)
        print(f'| `{c}` | {v.notna().mean():.1%} | {mw:.1%} | {ml:.1%} | {abs(mw-ml)*100:.1f} pp | '
              f'{"**DROP (outcome-dependent missingness)**" if bad else "ok"} |')
    print(f'| `asset_class` identified | {cov["asset_class"]:.1%} | — | — | — | ok |')
    print(f'\n   SI report age at the signal: median {s.si_report_age.median():.0f} days, '
          f'p95 {s.si_report_age.quantile(.95):.0f} (dissemination lag is IN the key: '
          f'usable_from = settlement + 13 calendar days)')
    void = []
    if cov['shares_out'] < 0.50:
        void += ['F15-b1', 'F15-b2']
        print(f'   shares_out coverage {cov["shares_out"]:.1%} < 50 % -> **F15-b1 and F15-b2 VOID**, '
              'as the PREREG declared.')
    if cov['news_file'] < 0.50:
        print(f'   news-file coverage {cov["news_file"]:.1%} — the ORB nightly file is keyed to the '
              'ORB candidate set, NOT this universe.  F15-e1/e2 are reported as DIAGNOSTICS with '
              'the covered subset named, not as clean cells.')

    print('\n\n================ F15 — THE CELLS ================', flush=True)
    show('F15-base (B2, every name)', base)
    T = s[s.split == 'TRAIN']
    dtc50 = float(T.days_to_cover.median()); dtc75 = float(T.days_to_cover.quantile(.75))
    sir75 = float(T.si_ratio.quantile(.75))
    print(f'   [TRAIN cuts] daysToCover median {dtc50:.2f} p75 {dtc75:.2f} | si_ratio p75 '
          f'{sir75:.3f}', flush=True)
    show(f'F15-a1 daysToCover >= {dtc50:.2f} [med]', book_ranked(s[s.days_to_cover >= dtc50], 12, 4))
    show(f'F15-a2 daysToCover >= {dtc75:.2f} [p75]', book_ranked(s[s.days_to_cover >= dtc75], 12, 4))
    show(f'F15-a3 si/adv20 >= {sir75:.3f} [p75]', book_ranked(s[s.si_ratio >= sir75], 12, 4))
    if 'F15-b1' in void:
        print(f'| F15-b1 float turnover >= p75            | VOID — shares_out coverage '
              f'{cov["shares_out"]:.1%} < 50 % (PREREG §3) |')
        print(f'| F15-b2 shares_out <= p25 [low float]    | VOID — same |')
    else:
        ft75 = float(T.float_turn.quantile(.75)); so25 = float(T.shares_out.quantile(.25))
        show(f'F15-b1 float turn >= {ft75:.3f} [p75]', book_ranked(s[s.float_turn >= ft75], 12, 4))
        show(f'F15-b2 shares_out <= {so25:.3g} [p25]', book_ranked(s[s.shares_out <= so25], 12, 4))
    show('F15-c1 common stock only', book_ranked(s[s.asset_class == 'stock'], 12, 4))
    show('F15-c2 leveraged wrapper only', book_ranked(s[s.asset_class == 'wrapper'], 12, 4))
    show('F15-c3 anchor cohort >= 2', book_ranked(s[s.anchor_cohort >= 2], 12, 4))
    vc = s.venue.value_counts()
    print(f'   [venues] {dict(vc.head(8))}', flush=True)
    nasd = s.venue.astype(str).str.upper().str.contains('NAS|XNAS', na=False)
    nyse = s.venue.astype(str).str.upper().str.contains('NYS|XNYS|ARC|ARCX|AMEX|XASE|BATS', na=False)
    show('F15-d1 venue NASDAQ', book_ranked(s[nasd], 12, 4))
    show('F15-d2 venue NYSE/ARCA/AMEX', book_ranked(s[nyse], 12, 4))
    show('F15-e1 premarket news present', book_ranked(s[s.has_news], 12, 4))
    show('F15-e2 news same morning', book_ranked(s[s.has_news & s.news_same_morning], 12, 4))
    show('[diag] news-file COVERED subset', book_ranked(s[s.news_covered], 12, 4))
    show('[diag] covered AND no news', book_ranked(s[s.news_covered & ~s.has_news], 12, 4))

    # ---- nulls, selector, bars -------------------------------------------------------------
    print('\n\n== count-matched permutation null on green weeks (2,000 draws) ==')
    print('| cell | split | observed green % | null mean | [p5, p95] | outside? |')
    print('|---|---|---|---|---|---|')
    nulls = []
    for nm2, bb in BOOKS.items():
        for sp in SPLITS:
            obs, mu_, p5, p95 = S.null_band(bb, sp)
            o = ('ABOVE' if obs == obs and obs > p95 else
                 ('below' if obs == obs and obs < p5 else 'inside'))
            print(f'| {nm2} | {sp} | {obs:.1f} | {mu_:.1f} | [{p5:.1f}, {p95:.1f}] | {o} |')
            nulls.append(dict(cell=nm2, split=sp, obs=obs, mu=mu_, p5=p5, p95=p95, outside=o))
    pd.DataFrame(nulls).to_csv(f'{D4}/nulls15.csv', index=False)
    cf = pd.DataFrame(CELLS); cf.to_csv(f'{D4}/cells15.csv', index=False)

    print('\n== F15 PRE-COMMITTED SELECTOR (era-consistency at >=10 trades/week) ==')
    sel = []
    for nm2 in cf.cell.unique():
        t_ = cf[(cf.cell == nm2) & (cf.split == 'TRAIN')].iloc[0]
        v_ = cf[(cf.cell == nm2) & (cf.split == 'VAL')].iloc[0]
        if t_.half_ok and min(t_.per_wk, v_.per_wk) >= 10:
            sel.append(nm2)
        print(f'  {nm2:40s} H1 {t_.h1:+.3f} H2 {t_.h2:+.3f} VAL {v_.vgross:+.3f} | /wk '
              f'{t_.per_wk:.1f}/{v_.per_wk:.1f} | era-consistent {t_.half_ok}')
    print(f'  -> {len(sel)}: {sel if sel else "NONE"}')

    print('\n== BOTH BARS ==')
    g1 = cf[(cf.split == 'TRAIN') & (cf.net > 0) & (cf.t >= 2.0) & (cf.tc >= 2.0) & (cf.per_wk >= 10)]
    print(f'G1 (TRAIN net>0, iid t>=2, CLUSTERED t>=2, >=10/wk): {len(g1)} -> {list(g1.cell)}')
    ship = []
    for nm2 in cf.cell.unique():
        t_ = cf[(cf.cell == nm2) & (cf.split == 'TRAIN')].iloc[0]
        v_ = cf[(cf.cell == nm2) & (cf.split == 'VAL')].iloc[0]
        if (t_.total > 0 and v_.total > 0 and t_.green >= 50 and v_.green >= 50 and
                min(t_.per_wk, v_.per_wk) >= 10 and t_.tc >= 2.0 and t_.half_ok):
            ship.append(nm2)
    print(f'LIVE-EXPLORATION BAR: {len(ship)} -> {ship}')
    print('\nMDE (80% power, per trade, net):')
    for nm2, bb in BOOKS.items():
        row = ' '.join(f'{sp} {2.80*bb[bb.split==sp].net.std(ddof=1)/np.sqrt(max(len(bb[bb.split==sp]),1)):.3f}'
                       for sp in SPLITS if len(bb[bb.split == sp]) > 2)
        print(f'  {nm2:40s} {row}')
    s.to_csv(f'{D4}/sig4_inst.csv', index=False)
    print('\nDONE', flush=True)


if __name__ == '__main__':
    main()
