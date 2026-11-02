"""Pass 4: Config loads .env without dotenv's per-variable environment copies unless the file uses ``$``."""
import os

import pytest

from config import Config


def test_plain_env_file_is_applied_and_existing_variables_win(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("ONEMIL_T_NEW=fresh\nONEMIL_T_KEEP=from_file\n")
    monkeypatch.delenv("ONEMIL_T_NEW", raising=False)
    monkeypatch.setenv("ONEMIL_T_KEEP", "from_process")
    Config._load_env_file(env)
    assert os.environ["ONEMIL_T_NEW"] == "fresh" and os.environ["ONEMIL_T_KEEP"] == "from_process"
    monkeypatch.delenv("ONEMIL_T_NEW")


def test_file_with_dollar_still_interpolates(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("ONEMIL_T_BASE=/srv\nONEMIL_T_PATH=${ONEMIL_T_BASE}/data\n")
    for k in ("ONEMIL_T_BASE", "ONEMIL_T_PATH"):
        monkeypatch.delenv(k, raising=False)
    Config._load_env_file(env)
    assert os.environ["ONEMIL_T_PATH"] == "/srv/data"
    for k in ("ONEMIL_T_BASE", "ONEMIL_T_PATH"):
        monkeypatch.delenv(k)


def test_plain_file_never_calls_the_interpolating_path(tmp_path, monkeypatch):
    seen = {}
    import config as cfg
    monkeypatch.setattr(cfg, "load_dotenv", lambda p, interpolate=True: seen.update(i=interpolate))
    (tmp_path / "a.env").write_text("A=1\n")
    Config._load_env_file(tmp_path / "a.env")
    assert seen["i"] is False
    (tmp_path / "b.env").write_text("A=${B}\n")
    Config._load_env_file(tmp_path / "b.env")
    assert seen["i"] is True
