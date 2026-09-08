"""2026-09-08 nightly crash: a 'NA' ticker became NaN and refresh_class_map's sorted() raised TypeError."""
import importlib.util, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('orb_pm_news_nightly_append', ROOT / 'research' / 'scripts' / 'orb_pm_news_nightly_append.py')
m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)


def test_refresh_class_map_ignores_nan_symbols(monkeypatch, tmp_path, capsys):
    import trading.orb_asset_class as oac
    cm = tmp_path / 'map.csv'; cm.write_text('symbol,name,asset_class\nAAPL,Apple,stock\n')
    monkeypatch.setattr(oac, 'DEFAULT_CLASS_MAP', str(cm))
    # every symbol known except a NaN and an empty string → no API call, no crash
    m.refresh_class_map({'AAPL', float('nan'), ''}, dry=True)
    assert 'coverage complete' in capsys.readouterr().out


def test_features_read_keeps_NA_ticker(tmp_path):
    import pandas as pd
    f = tmp_path / 'orb_features_x.csv'; f.write_text('symbol,date\nNA,2026-09-08\nAAPL,2026-09-08\n')
    df = pd.read_csv(f, usecols=['symbol', 'date'], keep_default_na=False, na_values=[''])
    assert df.symbol.tolist() == ['NA', 'AAPL']
