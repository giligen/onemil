"""Point-in-time ORB parameters (cell 1,733, spec docs/orb_bt_pit_params_spec_20261111.md).

Unit: params_for_day boundaries. Integration: the REAL pipeline scoring functions over two refit weeks.
Parity: live composites logged for 2026-11-03 (research/book_review_20261111/selection_1733.csv) equal the
PIT-scored BT composites on every shared name.
"""
import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import study_orb_pipeline_static_lock as P
from study_orb_filter import FILTER_FEATURES, composite_score
from trading import orb_pit_params as M

ROOT = Path(__file__).resolve().parent.parent
FEATS = [f for f, _ in FILTER_FEATURES]


def _params(shift):
    return {f: {'mean': 1.0 + shift, 'std': 2.0 + shift / 10, 'sign': -1 if i % 2 else 1}
            for i, f in enumerate(FEATS)}


def _cut(shift):
    return [0.1 + shift, 0.2 + shift, 0.3 + shift, 0.4 + shift]


def _write_hist(path, entries):
    with open(path, 'w') as fh:
        for ts, shift in entries:
            fh.write(json.dumps({'ts': ts, 'params': _params(shift), 'cutoffs': _cut(shift)}) + '\n')


@pytest.fixture
def hist(tmp_path):
    p = tmp_path / 'hist.jsonl'
    _write_hist(p, [('20261004_200003', 0.5), ('20261011_200002', 1.5)])   # two Sunday 20:00 UTC refits
    M._WARNED.clear()
    return p


CUR = {'params': _params(9.0), 'cutoffs': _cut(0.09)}


# ------------------------------------------------------------------ unit
def test_day_before_first_refit_is_loud_current_fallback(hist, caplog):
    with caplog.at_level(logging.WARNING, logger='trading.orb_pit_params'):
        pp = M.params_for_day(date(2026, 10, 2), hist, CUR)
        M.params_for_day(date(2026, 10, 1), hist, CUR)
    assert pp.source == 'current_fallback' and pp.refit_ts is None and pp.params == CUR['params']
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warns) == 1 and 'precedes the first recorded refit' in warns[0].getMessage()


def test_day_before_first_refit_without_current_raises(hist):
    with pytest.raises(ValueError):
        M.params_for_day(date(2026, 10, 2), hist)


def test_monday_after_sunday_refit_uses_that_refit(hist):
    pp = M.params_for_day(date(2026, 10, 5), hist, CUR)          # Monday after the 10/4 Sunday fit
    assert (pp.source, pp.refit_ts, pp.params) == ('pit', '20261004_200003', _params(0.5))
    assert pp.cutoffs == _cut(0.5)


def test_sunday_itself_does_not_see_its_own_later_refit(hist):
    # refit at 20:00 UTC Sunday is after 09:30 ET that Sunday -> previous set (none -> fallback)
    assert M.params_for_day(date(2026, 10, 4), hist, CUR).source == 'current_fallback'
    assert M.params_for_day(date(2026, 10, 11), hist, CUR).refit_ts == '20261004_200003'


def test_two_refits_later_applies_only_after_its_timestamp(hist):
    assert M.params_for_day(date(2026, 10, 9), hist, CUR).refit_ts == '20261004_200003'
    assert M.params_for_day(date(2026, 10, 12), hist, CUR).refit_ts == '20261011_200002'
    assert M.params_for_day(date(2026, 11, 20), hist, CUR).refit_ts == '20261011_200002'


def test_same_day_refit_before_open_applies(tmp_path):
    p = tmp_path / 'h.jsonl'
    _write_hist(p, [('20261012_120000', 0.5)])   # 12:00 UTC = 08:00 ET, before the 09:30 ET open
    assert M.params_for_day(date(2026, 10, 12), p, CUR).source == 'pit'
    _write_hist(p, [('20261012_140000', 0.5)])   # 14:00 UTC = 10:00 ET, after the open
    assert M.params_for_day(date(2026, 10, 12), p, CUR).source == 'current_fallback'


def test_malformed_line_raises(tmp_path):
    p = tmp_path / 'bad.jsonl'
    _write_hist(p, [('20261004_200003', 0.5)])
    with open(p, 'a') as fh:
        fh.write('{"ts": "20261011_200002", "params": {}\n')
    with pytest.raises(ValueError, match='line 2 malformed'):
        M.params_for_day(date(2026, 10, 12), p, CUR)


def test_missing_history_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        M.params_for_day(date(2026, 10, 12), tmp_path / 'nope.jsonl', CUR)


def test_params_mode_flag_and_default():
    assert P.params_mode([]) == 'pit'
    assert P.params_mode(['--params', 'current']) == 'current'
    with pytest.raises(SystemExit):
        P.params_mode(['--params', 'bogus'])


# ----------------------------------------------------------- integration
def _frame():
    rng = np.random.default_rng(7)
    days = pd.to_datetime(['2026-10-02', '2026-10-05', '2026-10-06', '2026-10-12', '2026-10-13'])
    rows = []
    for d in days:
        for i in range(12):
            r = {'symbol': f'S{i}', 'date': d}
            r.update({f: rng.normal(1, 3) for f in FEATS})
            rows.append(r)
    return pd.DataFrame(rows)


def test_pipeline_scoring_two_refit_weeks(hist, caplog):
    df = _frame()
    with caplog.at_level(logging.WARNING, logger=P.logger.name):
        comp, day_pp = P.score_days(df, CUR['params'], CUR['cutoffs'], 'pit', hist)
    assert any('precede the first recorded refit' in r.getMessage() for r in caplog.records)
    expect = {'2026-10-05': 0.5, '2026-10-06': 0.5, '2026-10-12': 1.5, '2026-10-13': 1.5}
    for day, shift in expect.items():
        m = df['date'] == day
        alone = composite_score(df[m], _params(shift))          # that week's set, scored alone
        assert np.allclose(comp[m].values, alone.values, atol=1e-12)
        assert (df.loc[m, 'params_source'] == 'pit').all()
        q = P.assign_quintile_by_day(df[m].assign(c=comp[m]), 'c', day_pp)
        assert (q == P.assign_quintile(comp[m], _cut(shift))).all()
    m0 = df['date'] == '2026-10-02'
    assert (df.loc[m0, 'params_source'] == 'current_fallback').all()
    assert df.loc[~m0, 'params_refit_ts'].isin(['20261004_200003', '20261011_200002']).all()


def test_params_current_path_differs_for_earlier_week(hist):
    df = _frame()
    pit, _ = P.score_days(df.copy(), _params(1.5), _cut(0.15), 'pit', hist)
    d2 = df.copy()
    cur, _ = P.score_days(d2, _params(1.5), _cut(0.15), 'current', hist)
    wk1 = df['date'] <= '2026-10-06'
    wk2 = df['date'] >= '2026-10-12'
    assert not np.allclose(pit[wk1 & (df['date'] > '2026-10-02')], cur[wk1 & (df['date'] > '2026-10-02')])
    assert np.allclose(pit[wk2], cur[wk2])            # later week: the current set IS that week's set
    assert (d2['params_source'] == 'current').all() and (d2['params_refit_ts'] == '').all()


# ---------------------------------------------------------------- parity
SEL = ROOT / 'research/book_review_20261111/selection_1733.csv'
FEATURES_CSV = ROOT / 'analysis_results/orb_features_20261110_2203.csv'
REAL_HIST = ROOT / 'logs/orb_refit_history.jsonl'


@pytest.mark.skipif(not (SEL.exists() and FEATURES_CSV.exists() and REAL_HIST.exists()),
                    reason='live-parity artefacts not on this node')
def test_pit_composite_equals_live_composite_2026_11_03():
    from trading.orb_csv import read_orb_csv
    day = '2026-11-03'
    sel = pd.read_csv(SEL, keep_default_na=False, na_values=[''])
    sel = sel[sel['date'] == day]
    live = sel.dropna(subset=['live_comp']).set_index('symbol')
    assert len(live) >= 5
    raw = read_orb_csv(str(FEATURES_CSV))
    raw['date'] = pd.to_datetime(raw['date'])
    df = raw.dropna(subset=FEATS)
    df = df[df['date'] == day].copy()
    M._WARNED.clear()
    comp, day_pp = P.score_days(df, _params(0), _cut(0), 'pit', REAL_HIST)   # `current` is never used here
    assert (df['params_source'] == 'pit').all()
    df['c'] = comp
    bt = df.set_index('symbol')['c']
    shared = live.index.intersection(bt.index)
    assert len(shared) == len(live), f"live names missing from BT features: {set(live.index) - set(bt.index)}"
    # live logs 4 decimals -> |diff| <= 5e-5 from rounding alone
    assert (live.loc[shared, 'live_comp'].astype(float) - bt.loc[shared]).abs().max() < 1e-4
    # quintile agreement 100 %
    q = P.assign_quintile_by_day(df, 'c', day_pp).set_axis(df['symbol'])
    assert (q.loc[shared].values == live.loc[shared, 'live_q'].values).all()
    # ranking: live's final top-8 are the PIT top ranks (Q4,Q5,Q3,Q2 then composite), in live's order
    order = {'Q4': 0, 'Q5': 1, 'Q3': 2, 'Q2': 3}
    k = df.assign(q=q.values)
    k = k[(k['c'] >= 0.012081536791) & (k['q'] != 'Q1')]
    k = k.assign(qr=k['q'].map(order)).sort_values(['qr', 'c'], ascending=[True, False])
    pit_rank = {s: i + 1 for i, s in enumerate(k['symbol'])}
    final = sel[sel['live_final_pos'].notna()].sort_values('live_final_pos')
    assert len(final) == 8
    ranks = [pit_rank[s] for s in final['symbol']]
    assert ranks == sorted(ranks) and max(ranks) <= 10      # family dedup may skip <= 2 names, never reorders
