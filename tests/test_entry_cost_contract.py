"""The research cost contract's ENTRY leg pays a FULL half-spread.

A buy order that elects at a level lifts the offer, so the obtainable entry price is the
NBBO ask -- 1.00 x half-spread above the mid, not 0.25 x. Measured twice:

  * research/mature_method/red_to_green/REPORT.md §3 -- 59.3 bps spread, the next-open fill
    printed at the ask.
  * research/mature_method/entry_cost_audit/REPORT.md -- 103 bull-flag entries, Alpaca SIP
    NBBO at the fill print.

These tests pin the corrected coefficient and guard against a new 0.25-hardcoding site
appearing in research code. Every file in LEGACY_SITES was scored under the old contract and
is SUPERSEDED, not correct -- the list may only shrink.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_acore_entry_coefficient_is_a_full_half_spread():
    acore = _load(ROOT / 'research' / 'fuckup_audit' / 'A' / 'acore.py', 'acore_under_test')
    assert acore.ENTRY_COEF_C == 1.00
    assert acore.LEGACY_ENTRY_COEF_C == 0.25


def test_p_cost_band_arm_entry_coefficient_is_a_full_half_spread():
    src = (ROOT / 'research' / 'fuckup_audit' / 'P_cost' / 'rescore.py').read_text()
    assert re.search(r'^ENTRY_COEF = 1\.00$', src, re.M)
    assert re.search(r'^LEGACY_ENTRY_COEF = 0\.25', src, re.M)


def test_full_half_spread_costs_four_times_the_legacy_entry_leg():
    """The correction is arithmetic: the entry leg quadruples, the exit leg is untouched."""
    acore = _load(ROOT / 'research' / 'fuckup_audit' / 'A' / 'acore.py', 'acore_under_test2')
    half, exit_ratio, gross = 0.08, 0.875, 0.20
    legacy = gross - acore.LEGACY_ENTRY_COEF_C * half - half * exit_ratio
    corrected = gross - acore.ENTRY_COEF_C * half - half * exit_ratio
    assert legacy - corrected == pytest.approx(0.75 * half)
    assert corrected < legacy


# Research scripts that hardcode the legacy 0.25 entry leg. Every one of them is SUPERSEDED
# by the measurement above; they are listed, not fixed, because re-scoring them is a study,
# not an edit. A NEW site must not appear -- this list may only shrink.
LEGACY_SITES = {
    'research/fuckup_audit/A/a4_decomp.py',
    'research/fuckup_audit/B/build_candidates4.py',
    'research/fuckup_audit/D1/d1_features.py',
    'research/fuckup_audit/H/F14_F8_F11/h_nofloor_score.py',
    'research/fuckup_audit/H/F6_rebuild/book.py',
    'research/fuckup_audit/H/F6_reconcile/r8_engine_book.py',
    'research/fuckup_audit/O_halt/PASSIVE/score.py',
    # the study that MEASURED the error -- 0.25 survives there only as the declared
    # legacy cell `M2meas`, scored side by side with the corrected `M3entry`.
    'research/mature_method/red_to_green/score.py',
    'research/fuckup_audit/D/d0_features.py',
    'research/fuckup_audit/G/build_candidates_short.py',
    'research/fuckup_audit/G/score_short.py',
    'research/fuckup_audit/H/F14_F8_F11/hcore.py',
    'research/fuckup_audit/H/F5/h5core.py',
    'research/fuckup_audit/H/F6_sizing/f6_load.py',
    'research/fuckup_audit/J/score_u3.py',
    'research/fuckup_audit/L/l1_pop.py',
    'research/fuckup_audit/L/l3_score.py',
    'research/fuckup_audit/O_halt/score_cells.py',
    'research/fuckup_audit/O_halt/robustness.py',
}

PATTERN = re.compile(r'0\.25\s*\*\s*half')


def test_no_new_legacy_entry_leg_site():
    found = set()
    for p in (ROOT / 'research').rglob('*.py'):
        try:
            if PATTERN.search(p.read_text(errors='ignore')):
                found.add(str(p.relative_to(ROOT)))
        except OSError:                      # pragma: no cover - unreadable file
            continue
    new = found - LEGACY_SITES
    assert not new, f'new 0.25 x half-spread entry leg introduced: {sorted(new)}'
