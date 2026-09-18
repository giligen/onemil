"""Tests for trading/orb_anchor_dedup.py — the pre-registered anchor rule.

STATUS: research infrastructure only. The rule FAILED its pre-committed ship
rule (research/orb_anchor_dedup/REPORT.md §4) and is NOT wired into
trading/orb_engine.py. These tests pin the helper's semantics and — the part
that matters for production — prove the backtest pipeline's DEFAULT path is
unchanged (the hook is env-gated OFF).
"""
import inspect
from pathlib import Path

from trading.orb_anchor_dedup import AnchorDedup, reject_mask


class TestAnchorDedupHelper:
    def test_first_by_rank_survives(self):
        # Rank order in, reject flags out: the first sighting is kept.
        assert reject_mask(['CIFR', 'CIFR']) == [False, True]

    def test_distinct_anchors_never_collide(self):
        assert reject_mask(['CIFR', 'MSTR', 'CIFR', 'MSTR']) == \
            [False, False, True, True]

    def test_unknown_anchor_fails_open(self):
        # None/'' never dedups — same convention as every underlying_anchor
        # consumer (an unresolved wrapper must not silently block a pick).
        assert reject_mask([None, None, '', '']) == [False] * 4

    def test_state_is_per_instance(self):
        a, b = AnchorDedup(), AnchorDedup()
        assert a.reject('CIFR') is False
        assert a.reject('CIFR') is True
        assert b.reject('CIFR') is False     # a fresh day starts empty

    def test_seen_exposes_taken_anchors(self):
        s = AnchorDedup()
        s.reject('CIFR'); s.reject(None)
        assert s.seen == {'CIFR'}

    def test_cifg_cifu_case(self):
        """The 2026-09-18 incident: both 2X CIFR wrappers, one anchor."""
        assert reject_mask(['CIFR', 'CIFR']) == [False, True]


class TestPipelineHookIsOffByDefault:
    """Production safety: the study hook must not change the default book."""

    SRC = Path(__file__).parent.parent / 'study_orb_pipeline_static_lock.py'

    def test_env_gate_defaults_to_off(self):
        src = self.SRC.read_text()
        assert "os.environ.get('ORB_ANCHOR_DEDUP', '0')" in src, \
            "anchor dedup must default OFF in the pipeline"
        assert 'from trading.orb_anchor_dedup import' in src

    def test_no_refill_anywhere_in_the_hook(self):
        src = self.SRC.read_text()
        start = src.index('Anchor dedup (research/orb_anchor_dedup')
        block = src[start:start + 2600]
        # The slot stays empty — no candidate is promoted into it.
        assert 'top_syms.extend' not in block
        assert 'kept_today.append' not in block

    def test_engine_does_not_import_the_rule(self):
        """It did not ship — the live engine must be untouched by it."""
        engine_src = (Path(__file__).parent.parent /
                      'trading' / 'orb_engine.py').read_text()
        assert 'orb_anchor_dedup' not in engine_src


class TestHelperIsSelfContained:
    def test_helper_has_no_pipeline_or_engine_imports(self):
        """Pure rule: no pandas, no engine — so both sides could share it."""
        import ast
        mod = __import__('trading.orb_anchor_dedup', fromlist=['x'])
        tree = ast.parse(inspect.getsource(mod))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.add(node.module or '')
        assert names == {'__future__', 'typing'}, names
