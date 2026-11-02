"""2026-11-02: Config.hod_break_cfg silently dropped squeeze_* / fc_leg_poll_s (third occurrence of the 9/25 defect).

The 10/31 squeeze unit tests built the engine from a RAW dict and never went through Config, so the whitelist gap
shipped. These tests close the loop: (a) the keys flow Config -> engine, (b) a dropped YAML key is an ERROR,
(c) a source-level parity test fails CI when the engine reads a key the whitelist lacks, (d) the boot gates line
proves the flags, (e) an engine built FROM Config honours squeeze_enabled.
"""
import logging
import re
from pathlib import Path

import pytest
import yaml

from config import Config
from trading import hod_break_engine as hbe
from trading.hod_break_engine import HodBreakEngine

ROOT = Path(__file__).resolve().parent.parent
ENGINE_SRC = (ROOT / 'trading' / 'hod_break_engine.py').read_text()


def _config_from(tmp_path, monkeypatch, block, name='hod_break'):
    """A real Config built from a tmp YAML (never the live config.yaml) holding one book block."""
    env = tmp_path / 'e.env'; env.touch()
    monkeypatch.setenv('ALPACA_API_KEY', 'k'); monkeypatch.setenv('ALPACA_API_SECRET', 's')
    y = tmp_path / 'c.yaml'; y.write_text(yaml.safe_dump({name: block}))
    return Config(env_path=str(env), yaml_path=str(y))


def _init_source():
    """Source text of HodBreakEngine.__init__ (up to the next method)."""
    start = ENGINE_SRC.index('    def __init__(self, alpaca_client')
    end = ENGINE_SRC.index('\n    def ', start + 10)
    return ENGINE_SRC[start:end]


ENGINE_KEYS = set(re.findall(r"\bcfg\.get\('([a-z_0-9]+)'", _init_source()))
# 'book' is the R2G dispatch key set by Config.red_to_green_cfg, not a hod_break YAML knob; 'params' is a nested whitelist.
HOD_EXEMPT = {'book'}


class TestWhitelistKeys:
    def test_squeeze_and_poll_keys_flow_with_types(self, tmp_path, monkeypatch):
        c = _config_from(tmp_path, monkeypatch, {'squeeze_enabled': True, 'squeeze_minute': 830, 'squeeze_max_r': 0.7,
                                                  'fc_leg_poll_s': 2})
        out = c.hod_break_cfg
        assert out['squeeze_enabled'] is True and out['squeeze_minute'] == 830 and isinstance(out['squeeze_minute'], int)
        assert out['squeeze_max_r'] == 0.7 and out['fc_leg_poll_s'] == 2.0 and isinstance(out['fc_leg_poll_s'], float)

    def test_defaults_match_engine(self, tmp_path, monkeypatch):
        out = _config_from(tmp_path, monkeypatch, {}).hod_break_cfg
        assert out['squeeze_enabled'] is False and out['squeeze_minute'] == 840 and out['squeeze_max_r'] == 0.5
        assert out['fc_leg_poll_s'] == HodBreakEngine.FC_LEG_POLL_S
        # compare to the SOURCE literals: an autouse conftest fixture redirects the module constants to tmp_path
        for const, key in (('DEFAULT_LIVE_PARITY_LEDGER_PATH', 'live_parity_ledger_path'),
                           ('DEFAULT_LIVE_ORDERS_STATE_PATH', 'live_orders_state_path')):
            literal = re.search(rf"^{const}\s*=\s*'([^']+)'", ENGINE_SRC, re.M).group(1)
            assert out[key] == literal

    def test_r2g_twin_carries_engine_mechanics(self, tmp_path, monkeypatch):
        out = _config_from(tmp_path, monkeypatch, {'squeeze_enabled': True}, name='red_to_green').red_to_green_cfg
        assert out['squeeze_enabled'] is True and out['book'] == 'red_to_green'


class TestLoudDrop:
    def test_unknown_key_logs_one_error(self, tmp_path, monkeypatch, caplog):
        c = _config_from(tmp_path, monkeypatch, {'enabled': True, 'zzz_unknown': 1})
        with caplog.at_level(logging.ERROR, logger='config'):
            c.hod_break_cfg
        errs = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errs) == 1
        assert "hod_break" in errs[0].getMessage() and "'zzz_unknown'" in errs[0].getMessage() and 'ignored' in errs[0].getMessage()

    def test_clean_block_no_error(self, tmp_path, monkeypatch, caplog):
        c = _config_from(tmp_path, monkeypatch, {'enabled': True, 'consol_bars': 5, 'squeeze_enabled': True,
                                                  'failure_short': {'enabled': False}})
        with caplog.at_level(logging.ERROR, logger='config'):
            c.hod_break_cfg
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]

    def test_r2g_unknown_key_logs_error(self, tmp_path, monkeypatch, caplog):
        c = _config_from(tmp_path, monkeypatch, {'enabled': False, 'zzz_unknown': 1}, name='red_to_green')
        with caplog.at_level(logging.ERROR, logger='config'):
            c.red_to_green_cfg
        assert any("red_to_green" in r.getMessage() and "'zzz_unknown'" in r.getMessage() for r in caplog.records)


class TestParity:
    def test_engine_reads_only_whitelisted_keys(self, tmp_path, monkeypatch):
        """Every cfg.get('<key>') literal in HodBreakEngine.__init__ is a key of Config.hod_break_cfg."""
        assert len(ENGINE_KEYS) > 30, 'regex found too few cfg.get literals - the parity test went blind'
        c = _config_from(tmp_path, monkeypatch, {})
        missing = sorted(k for k in ENGINE_KEYS - HOD_EXEMPT if k not in c.hod_break_cfg)
        assert not missing, f"engine reads cfg keys absent from Config.hod_break_cfg (silently dropped from YAML): {missing}"

    def test_fc_leg_poll_default_matches_engine(self):
        assert Config._FC_LEG_POLL_S_DEFAULT == HodBreakEngine.FC_LEG_POLL_S

    @pytest.mark.parametrize('name,prop', [('hod_break', 'hod_break_cfg'), ('red_to_green', 'red_to_green_cfg')])
    def test_live_config_yaml_keys_all_whitelisted(self, name, prop, caplog):
        """Read-only: every key in the live config.yaml block is read by the engine's Config whitelist (no ERROR)."""
        live = ROOT / 'config.yaml'
        if not live.exists():
            pytest.skip('config.yaml absent (gitignored)')
        raw = (yaml.safe_load(live.read_text()) or {}).get(name) or {}
        c = Config.__new__(Config); c._yaml = {name: raw}
        with caplog.at_level(logging.ERROR, logger='config'):
            out = getattr(c, prop)
        read = set(out) | set(out['params'])
        assert [k for k in raw if k not in read] == []
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


class TestGatesLineAndEngineFromConfig:
    def _engine(self, tmp_path, monkeypatch, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, **block):
        out = _config_from(tmp_path, monkeypatch, block).hod_break_cfg
        out['dry_ledger_path'] = str(tmp_path / 'dry.csv')
        out['live_parity_ledger_path'] = str(tmp_path / 'par.csv')
        out['live_orders_state_path'] = str(tmp_path / 'st.json')
        out['stream_list_dir'] = str(tmp_path)
        return HodBreakEngine(hod_live_alpaca, hod_live_db, hod_live_sm, cfg=out, order_stream=hod_live_stream)

    def test_engine_from_config_honours_squeeze_enabled(self, tmp_path, monkeypatch, hod_live_alpaca, hod_live_db,
                                                       hod_live_sm, hod_live_stream):
        e = self._engine(tmp_path, monkeypatch, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream,
                         squeeze_enabled=True, squeeze_minute=845, squeeze_max_r=0.4)
        assert e.squeeze_enabled is True and e.squeeze_minute == 845 and e.squeeze_max_r == 0.4

    def test_gates_line_prints_squeeze_and_entry_mode(self, tmp_path, monkeypatch, caplog, hod_live_alpaca, hod_live_db,
                                                     hod_live_sm, hod_live_stream):
        with caplog.at_level(logging.INFO, logger='trading.hod_break_engine'):
            self._engine(tmp_path, monkeypatch, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream,
                         squeeze_enabled=True, entry_mode='resting_stop_limit')
        line = next(r.getMessage() for r in caplog.records if 'engine gates' in r.getMessage())
        assert 'squeeze=on@840/<0.5R' in line and 'entry_mode=resting_stop_limit' in line

    def test_gates_line_squeeze_off(self, tmp_path, monkeypatch, caplog, hod_live_alpaca, hod_live_db, hod_live_sm,
                                    hod_live_stream):
        with caplog.at_level(logging.INFO, logger='trading.hod_break_engine'):
            self._engine(tmp_path, monkeypatch, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream)
        line = next(r.getMessage() for r in caplog.records if 'engine gates' in r.getMessage())
        assert 'squeeze=off@840/<0.5R' in line
