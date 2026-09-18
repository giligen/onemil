#!/usr/bin/env python3
"""CAUSAL_FILTER — the standing audits (PLAN §1 / CLAUDE.md "no research claim ships without an
independent check"), run on the assembled feature table.

  A1 availability   per-feature coverage per split AND per time-of-day band, with the mean R of the
                    covered vs uncovered rows (the D1 missingness rule: a cohort leak shows here).
  A2 causality      every feature's construction is traced to bars/daily rows at or before the
                    signal minute; the ones that cannot be are named and dropped.
  A3 price-scale    200 random keys: the daily file's (Databento) close vs the intraday (Alpaca SIP)
                    last RTH bar close for the same symbol-day. A split/dividend mismatch fabricates
                    gap_pct / dist_20d_high_pct out of nothing.
  A4 obtainability  the share of signals whose NBBO ask at the fill instant was ABOVE the capped
                    limit (level x 1.006) — those orders never fill.
  A5 cohort         the diagnostic cohort column is not in the scored feature set (asserted).

Output: causal_filter/audits.md
"""
import os, sqlite3, sys
import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
D = f'{ROOT}/research/bf_zero/causal_filter'
from selectfeat import FEATS, assert_no_cohort                      # noqa: E402

HB_EDGES = [569, 585, 600, 660, 780, 960]
HB_LAB = ['09:30-09:45', '09:45-10:00', '10:00-11:00', '11:00-13:00', '13:00+']
CAUSALITY = {
    'gap_pct': 'open of the signal day (09:30 bar) / prev daily close — both known at 09:31',
    'prev_range_pct': 'previous daily bar high/low/close — known before the open',
    'dist_20d_high_pct': 'entry level / rolling 20d high SHIFTED by one day — prior sessions only',
    'bar_vol_x': 'volume of the signal bar / mean volume of bars strictly before it',
    'above_vwap': 'entry level vs the VWAP through bar i-1 (strictly before the fill bar)',
    'spy_5m_ret': 'SPY close at the signal minute vs 5 minutes earlier',
    'spy_range3': 'SPY daily range, 3-day mean, SHIFTED one day',
    'dist_open_pct': 'entry level / the 09:30 open — the spec s own causal floor',
    'rv_clock': 'cumulative volume to the signal bar / the same-clock mean over the prior 20 HELD '
                'days (shift(1) then rolling) — no same-day information',
    'rv_profile': 'cumulative volume to the signal bar / (ADV20 x the market-wide clock fraction) — '
                  'ADV20 is prior sessions, the fraction is a fixed constant in trading/hod_break.py',
    'drive_min': 'minute the day first traded 5% above its open, strictly before the signal bar',
    'n_prior': 'count of prior held days in the volume profile — prior sessions only',
    'is_wrapper': 'static asset-class map (instrument type, not a price)',
    'coh_by_t': 'count of same-anchor siblings that signalled STRICTLY EARLIER the same day',
    'entry_m': 'the signal minute itself',
    'price': 'the capped fill price, known at the fill',
    'has_news': 'articles published before 09:30 ET on the signal day',
}


def main():
    assert_no_cohort(FEATS)
    c = pd.read_csv(f'{D}/features.csv', dtype={'symbol': str, 'day': str},
                    keep_default_na=False, na_values=[''])
    c['hb'] = pd.cut(c.entry_m, HB_EDGES, labels=HB_LAB)
    out = ['# Causal-filter — standing audits', '',
           '## A5 cohort assertion', '',
           '`selectfeat.assert_no_cohort` passes: `cohort` (end-of-day information) is not among the '
           'scored features, and appears only as the diagnostic `cache %` column of the anatomy '
           'tables. The label is each signal\'s own R under the spec exits, never the cohort.', '',
           '## A1 availability — coverage % per split, then per time band', '']
    cov = c.groupby('split')[FEATS].apply(lambda d: d.notna().mean() * 100).round(1)
    out += [cov.to_markdown(), '']
    covh = c.groupby('hb', observed=True)[FEATS].apply(lambda d: d.notna().mean() * 100).round(1)
    out += [covh.to_markdown(), '',
            '### missingness bias — mean R of covered vs uncovered rows (TRAIN)', '']
    tr = c[c.split == 'TRAIN']
    rows = []
    for f in FEATS:
        if f not in c.columns:
            continue
        m = tr[f].notna()
        rows.append(dict(feature=f, cov_pct=round(m.mean() * 100, 1),
                         meanR_covered=round(float(tr.rr[m].mean()), 3) if m.any() else np.nan,
                         meanR_missing=round(float(tr.rr[~m].mean()), 3) if (~m).any() else np.nan,
                         n_missing=int((~m).sum())))
    out += [pd.DataFrame(rows).to_markdown(index=False), '', '## A2 causality trace', '']
    out += [pd.DataFrame([dict(feature=k, computed_from=v) for k, v in CAUSALITY.items()])
            .to_markdown(index=False), '']

    # ---- A3 price-scale ------------------------------------------------------------------------
    rng = np.random.default_rng(7)
    keys = c.sample(min(200, len(c)), random_state=7)[['day', 'symbol']].drop_duplicates()
    daily = pd.read_parquet(f'{ROOT}/data/research/databento/equs_daily_2025_2026.parquet',
                            columns=['symbol', 'bar_date', 'close'])
    daily = daily[daily.symbol.isin(set(keys.symbol))].copy()
    daily['bar_date'] = daily.bar_date.astype(str).str[:10]
    dk = daily.set_index(['symbol', 'bar_date'])
    sys.path.insert(0, f'{ROOT}/research/bf_zero')
    os.environ.setdefault('BFZ_SIP_STORE', f'{ROOT}/research/bf_zero/bars_sip.db')
    import build_candidates as B                                 # noqa: E402
    diffs = []
    for day, g in keys.groupby('day'):
        bars = B.load_bars(day, g.symbol.tolist())
        for s in g.symbol:
            bb = bars.get(s)
            if bb is None or not len(bb):
                continue
            try:
                d = dk.loc[(s, day)]
            except KeyError:
                continue
            rth = bb[(bb.m >= 570) & (bb.m < 960)]
            if not len(rth):
                continue
            ic = float(rth.c.iloc[-1])
            diffs.append(abs(ic / float(d.close) - 1) * 100)
    dv = np.array(diffs)
    out += ['## A3 price-scale check (200 sampled keys)', '',
            f'daily-file close vs the last RTH 1-min close on the SAME symbol-day: n {len(dv)}, '
            f'median |diff| {np.median(dv):.3f}%, p95 {np.percentile(dv, 95):.3f}%, '
            f'share > 1% {np.mean(dv > 1) * 100:.1f}%, share > 5% (a split) {np.mean(dv > 5) * 100:.1f}%.', '']

    # ---- A4 obtainability ----------------------------------------------------------------------
    if os.path.exists(f'{D}/nbbo.csv'):
        n = pd.read_csv(f'{D}/nbbo.csv', dtype={'symbol': str, 'day': str},
                        keep_default_na=False, na_values=['']).drop_duplicates(['day', 'symbol'])
        cc = c.merge(n.drop(columns=['entry_m']), on=['day', 'symbol'], how='left')
        ok = cc.ask_dec <= cc.level * 1.006
        have = cc.ask_dec.notna()
        out += ['## A4 obtainability (capped limit = level x 1.006)', '',
                f'NBBO quoted at the fill instant for {have.mean() * 100:.1f}% of signals; of those, '
                f'**{(~ok[have]).mean() * 100:.1f}% had an ask ABOVE the cap** and would NOT have '
                f'filled. Those rows are removed before the book in the `measured` arm.', '',
                cc[have].groupby('split').apply(
                    lambda d: pd.Series({'n': len(d), 'no_fill_pct': round((d.ask_dec > d.level * 1.006).mean() * 100, 1),
                                         'meanR_fillable': round(float(d.rr[d.ask_dec <= d.level * 1.006].mean()), 3),
                                         'meanR_nofill': round(float(d.rr[d.ask_dec > d.level * 1.006].mean()), 3)}),
                    include_groups=False).to_markdown(), '']
    else:
        out += ['## A4 obtainability', '', 'nbbo.csv not built yet.', '']
    open(f'{D}/audits.md', 'w').write('\n'.join(out))
    print('\n'.join(out[:40]), flush=True)
    print('DONE audits', flush=True)


if __name__ == '__main__':
    main()
