"""Unit tests for research/hod_ofi/pipeline.py -- one test per SPEC_FIX.md defect (1-6).
Synthetic data only; no network, no raw parquet reads.

Run: pytest research/hod_ofi/test_pipeline.py
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import pipeline as pl  # noqa: E402


# ---------------------------------------------------------- defect 1: CKS ask-side sign
def test_ofi_updates_hand_computed_book():
    """6-row book: row0 baseline (e forced 0 -- no previous record), then ask lifted,
    ask size added, ask improved (ask moved down), bid improved, bid size cut.
    e = [b>b' ? q_b : b==b' ? q_b-q_b' : -q_b'] + [a>a' ? +q_a' : a==a' ? q_a'-q_a : -q_a]."""
    mbp = pd.DataFrame({
        'bid_px_00': [10.00, 10.00, 10.00, 10.00, 10.05, 10.05],
        'ask_px_00': [10.10, 10.20, 10.20, 10.15, 10.15, 10.15],
        'bid_sz_00': [100, 100, 100, 100, 120, 90],
        'ask_sz_00': [200, 150, 250, 180, 180, 180],
    })
    e = pl.ofi_updates(mbp)
    # row1 ask lifted: +q_a'=200 | row2 ask size added (a==a'): q_a'-q_a=150-250=-100
    # row3 ask improved (a<a'): -q_a=-180 | row4 bid improved: q_b=120, ask unchanged: q_a'-q_a=0
    # row5 bid size cut (b==b'): q_b-q_b'=90-120=-30, ask unchanged: 0
    expected = np.array([0.0, 200.0, -100.0, -180.0, 120.0, -30.0])
    np.testing.assert_allclose(e, expected)


# ---------------------------------------------------------- defect 2: coverage_frac
def test_coverage_one_quote_before_window_full_coverage():
    """Single two-sided quote at t=-10s (before the window, fetched earlier), no updates
    inside the window -> every window second's prevailing quote is it -> 1.0."""
    mbp = pd.DataFrame({'sec': [990.0], 'bid_px_00': [10.0], 'ask_px_00': [10.05]})
    assert pl._coverage_frac(mbp, feat_start=1000.0, end=1300.0) == pytest.approx(1.0)


def test_coverage_late_first_quote_half_coverage():
    """First record arrives 151s into the window: instant checked for window-second
    offset o is o+1, so offsets 0-149 (150s) have no prior record (unquoted) and
    offsets 150-299 (150s) see it -> exactly 0.5."""
    mbp = pd.DataFrame({'sec': [1151.0], 'bid_px_00': [10.0], 'ask_px_00': [10.05]})
    assert pl._coverage_frac(mbp, feat_start=1000.0, end=1300.0) == pytest.approx(0.5)


def test_coverage_no_prior_record_is_unquoted():
    mbp = pd.DataFrame({'sec': pd.Series(dtype=float), 'bid_px_00': pd.Series(dtype=float),
                         'ask_px_00': pd.Series(dtype=float)})
    assert pl._coverage_frac(mbp, feat_start=1000.0, end=1300.0) == 0.0


# ---------------------------------------------------------- defect 3: Lee-Ready causality
def test_lee_ready_excludes_same_timestamp_quote():
    """A quote change at the SAME sec as a trade must not classify that trade --
    merge_asof(allow_exact_matches=False) forces the strictly-prior quote."""
    mbp = pd.DataFrame({
        'sec': [800.0, 805.0],
        'bid_px_00': [9.90, 10.05], 'ask_px_00': [10.00, 10.15],
        'bid_sz_00': [100, 100], 'ask_sz_00': [100, 100],
    })
    trades = pd.DataFrame({'sec': [805.0], 'price': [10.00], 'size': [100.0]})
    # kind='placebo' -> end = entry_m*60 deterministically, independent of trades/entry_px
    feat = pl.compute_window_features(mbp, trades, kind='placebo', entry_m=17, entry_px=np.nan)
    # prior quote mid=9.95 -> price 10.00 is a BUY (+1); the same-tick quote (mid 10.10) would flip it to -1
    assert feat['TSI_5'] == pytest.approx(1.0)


# ---------------------------------------------------------- defect 4: causal fallback end
def test_fallback_end_is_minute_start_not_plus_60():
    mbp = pd.DataFrame(columns=['sec', 'bid_px_00', 'ask_px_00', 'bid_sz_00', 'ask_sz_00'])
    trades = pd.DataFrame(columns=['sec', 'price', 'size'])  # no confirming print >= entry_px
    feat = pl.compute_window_features(mbp, trades, kind='signal', entry_m=20, entry_px=100.0)
    assert feat['locate_fallback'] is True
    assert feat['end_sec'] == pytest.approx(20 * 60)  # S, not S+60


# ---------------------------------------------------------- defect 7: the decision instant is the entry minute START
def test_signal_window_ends_at_minute_start_even_when_a_print_confirms():
    """A print through the entry at S+5 must not move the window end past S (next-open entries decide at S)."""
    mbp = pd.DataFrame({'sec': [1190.0, 1203.0], 'bid_px_00': [99.9, 100.0], 'ask_px_00': [100.0, 100.1],
                        'bid_sz_00': [100.0, 100.0], 'ask_sz_00': [100.0, 100.0]})
    trades = pd.DataFrame({'sec': [1195.0, 1205.0], 'price': [99.95, 100.05], 'size': [100.0, 100.0]})
    feat = pl.compute_window_features(mbp, trades, kind='signal', entry_m=20, entry_px=100.0)
    assert feat['end_sec'] == pytest.approx(20 * 60)
    assert feat['locate_fallback'] is False
    # the S+3 quote update and the S+5 trade are after the decision: OFI sees one record only, TSI one trade only
    assert np.isnan(feat['OFI_5'])
    assert feat["TSI_5"] == pytest.approx(0.0)  # the 1195 trade prints AT the 1190 mid (99.95) -> sign 0


# ---------------------------------------------------------- defect 5: sig_entry_m join key
def test_build_windows_sig_entry_m_disambiguates_multi_signal_name_day():
    """Same (day, symbol) fires twice at different entry_m -- each placebo must carry
    the entry_m of the ONE signal it was drawn for, not be joinable by (day, symbol) alone."""
    pop = pd.DataFrame({
        'day': ['2025-01-10', '2025-01-10'], 'symbol': ['AAA', 'AAA'],
        'entry_m': [650, 700], 'entry': [10.0, 12.0],
        'split': ['TRAIN', 'TRAIN'], 'half': ['H1', 'H1'],
    })
    w = pl.build_windows(pop)
    assert len(w) == 4
    sig = w[w.kind == 'signal'].sort_values('entry_m')
    plc = w[w.kind == 'placebo'].sort_values('sig_entry_m')
    assert list(sig['sig_entry_m']) == [650, 700]   # a signal's sig_entry_m is itself
    assert list(plc['sig_entry_m']) == [650, 700]   # each placebo ties back to its OWN parent


# ---------------------------------------------------------- defect 6: score fixes
def test_decile_table_drops_nan_and_is_monotone():
    """TSI and placebo Spearman were NaN because NaN feature/net_R rows reached qcut."""
    n = 100
    feature = np.arange(n, dtype=float)
    net_r = feature * 0.01
    df = pd.DataFrame({'f': feature, 'net_R': net_r})
    df.loc[0:4, 'f'] = np.nan      # would have made qcut raise / propagate NaN pre-fix
    df.loc[5:9, 'net_R'] = np.nan
    tbl, rho = pl._decile_table(df, 'f')
    assert tbl['n'].sum() == n - 10   # the 10 NaN rows were dropped, not silently counted
    assert rho == pytest.approx(1.0)  # perfectly monotone feature vs net_R


def test_keep_lift_ols_matches_group_mean_difference():
    """lift == OLS keep-dummy coefficient == keep_mean - drop_mean; cluster t >> iid noise
    is well-separated so cov_type='cluster' should still report a large, finite t."""
    rng = np.random.default_rng(0)
    rows = [dict(day=f'd{i}', _keep=keep, net_R=base + rng.normal(0, 0.01))
            for i in range(10) for keep, base in [(True, 1.0), (False, 0.0)] for _ in range(5)]
    df = pd.DataFrame(rows)
    out = pl._keep_lift_ols(df)
    assert out['lift'] == pytest.approx(1.0, abs=0.05)
    assert out['t_lift'] > 5
    assert np.isfinite(out['t_iid'])


def test_corrections_section_is_verbatim_from_spec():
    text = pl._corrections_section()
    assert 'CKS ask side has the wrong sign' in text
    assert 'Coverage counts seconds with an UPDATE' in text
