import importlib.util, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('cache_append_dedupe', ROOT / 'scripts' / 'cache_append_dedupe.py')
m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)


def test_drops_rows_already_in_production_keeps_new():
    prod = pd.DataFrame({'symbol': ['PATX', 'FCUV'], 'date': ['2026-09-04'] * 2, 'entry_time_et': ['10:01:00', '10:04:00'], 'pnl': ['268.12', '-2200.11']})
    tmp = pd.DataFrame({'symbol': ['PATX', 'SMU'], 'date': ['2026-09-04', '2026-09-08'], 'entry_time_et': ['10:01:00', '09:50:00'], 'pnl': ['-368.53', '4249.83']})
    out = m.dedupe_against(prod, tmp)
    assert out.symbol.tolist() == ['SMU']            # the re-walked PATX never overwrites production


def test_empty_tmp():
    prod = pd.DataFrame({'symbol': ['A'], 'date': ['2026-09-04'], 'entry_time_et': ['10:00:00']})
    assert m.dedupe_against(prod, prod.iloc[0:0]).empty


def test_nightly_script_calls_dedupe_before_append():
    s = (ROOT / 'scripts' / 'nightly_bt_update.sh').read_text()
    assert s.index('cache_append_dedupe.py') < s.index('tail -n +2 "$TMP_CACHE" >> "${CACHE_PATH}.tmp"')
