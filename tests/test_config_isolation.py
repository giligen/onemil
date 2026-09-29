"""Guard: no test module may read a live-config numeric value into an
assertion without overriding it (CLAUDE.md: tests must never depend on live
config values).

2026-09-29 incident: the live (gitignored, node-local) orb.yaml gained
`entry.preplace_submit_delay_s: 5.0` (cell 1,655) and silently broke three
scheduler-interval assertions in tests/test_orb_preplace_at_close.py, whose
`_base_cfg()` loaded that file raw and never pinned the key. Fixed by pinning
`preplace_submit_delay_s` explicitly in `_base_cfg()`. This file is the
regression guard for that fixture, at minimum -- it must keep failing loudly,
independent of the scheduler tests themselves, if the pin is ever removed or
silently stops working.
"""
import inspect

import yaml

import tests.test_orb_preplace_at_close as preplace_tests


class TestPreplaceFixtureOverridesLiveConfig:
    """The preplace test fixtures must pin entry.preplace_submit_delay_s
    themselves, never inherit it from whatever the live orb.yaml carries."""

    def test_base_cfg_pins_submit_delay_to_zero(self):
        cfg = preplace_tests._base_cfg()
        assert cfg['entry']['preplace_submit_delay_s'] == 0.0

    def test_base_cfg_source_sets_the_key_explicitly(self):
        """Source-level guard: fails immediately (no engine, no timers) if a
        future edit deletes the explicit pin from _base_cfg, instead of
        waiting to be caught indirectly by a scheduler-interval assertion."""
        src = inspect.getsource(preplace_tests._base_cfg)
        assert "cfg['entry']['preplace_submit_delay_s']" in src

    def test_engine_with_delay_override_wins_over_base_cfg_pin(self):
        eng = preplace_tests._engine_with_delay(5.0)
        assert eng.preplace_submit_delay_s == 5.0

    def test_fixture_value_independent_of_live_orb_yaml_value(self, monkeypatch):
        """Prove isolation, not coincidence: even if the live orb.yaml's
        delay were some OTHER nonzero number, the pinned fixture must still
        hand the engine 0.0 -- simulated here by poisoning the yaml.safe_load
        call inside the preplace test module (the same one `_base_cfg` uses
        to read the live file) rather than trusting today's actual value."""
        real_safe_load = yaml.safe_load

        def _poisoned_load(stream):
            cfg = real_safe_load(stream)
            if isinstance(cfg, dict) and 'entry' in cfg:
                cfg['entry']['preplace_submit_delay_s'] = 999.0
            return cfg

        monkeypatch.setattr(preplace_tests.yaml, 'safe_load', _poisoned_load)
        cfg = preplace_tests._base_cfg()
        assert cfg['entry']['preplace_submit_delay_s'] == 0.0
