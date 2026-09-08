"""ORB weekly selection refit job: fit = harness fit, in-place yaml rewrite, refusals."""
import importlib.util, os, sys
from pathlib import Path
import numpy as np, pandas as pd, pytest, yaml

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('orb_weekly_refit', ROOT / 'scripts' / 'orb_weekly_refit.py')
m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)


def _frame(n=800, seed=0):
    import study_orb_pipeline_static_lock as P
    rng = np.random.default_rng(seed)
    d = {'symbol': [f'S{i}' for i in range(n)], 'date': pd.date_range('2026-01-05', periods=n, freq='6h').strftime('%Y-%m-%d')}
    for f, _ in P.FILTER_FEATURES:
        d[f] = rng.normal(10, 3, n)
    return pd.DataFrame(d)


def test_fit_matches_harness_fit():
    import study_orb_pipeline_static_lock as P
    from study_orb_sizing import fit_quintile_cutoffs
    df = _frame()
    params, cutoffs, meta = m.fit_window(df, pd.Timestamp('2026-08-01').date(), weeks=26)
    d = df.copy(); d['date'] = pd.to_datetime(d['date'])
    lo = pd.Timestamp('2026-08-01') - pd.Timedelta(weeks=26)
    train = d[(d['date'] >= lo) & (d['date'] < pd.Timestamp('2026-08-01'))]
    exp = P.fit_z_params(train, P.FILTER_FEATURES)
    assert params == exp
    thresh = meta['threshold']; comp = P.composite_score(train, exp)
    assert cutoffs == [float(x) for x in fit_quintile_cutoffs(comp[comp >= thresh])]


def test_refuses_small_window():
    with pytest.raises(SystemExit):
        m.fit_window(_frame(n=100), pd.Timestamp('2026-08-01').date())


def test_rewrite_preserves_everything_but_values():
    text = (ROOT / 'orb.yaml').read_text()
    old = yaml.safe_load(text)
    params = {f: {'mean': 1.5, 'std': 2.5, 'sign': old['filter']['features'][f]['sign']} for f in old['filter']['features']}
    cutoffs = [0.1, 0.2, 0.3, 0.4]
    new_text = m.rewrite_yaml(text, params, cutoffs)
    new = yaml.safe_load(new_text)
    for f in params:
        assert new['filter']['features'][f]['mean'] == 1.5 and new['filter']['features'][f]['std'] == 2.5
        assert new['filter']['features'][f]['sign'] == old['filter']['features'][f]['sign']
    assert new['quintile_cutoffs'] == cutoffs
    assert new['adaptive_mults'] == old['adaptive_mults']
    # everything else identical
    for k in old:
        if k not in ('filter', 'quintile_cutoffs'):
            assert new[k] == old[k], k
    for k in old['filter']:
        if k != 'features':
            assert new['filter'][k] == old['filter'][k], k
    # comments survive (a known comment line in the features block)
    assert '# lower gap = better' in new_text
    assert new_text.count('\n') == text.count('\n')


def test_dry_run_writes_nothing(tmp_path, monkeypatch):
    src = (ROOT / 'orb.yaml').read_text()
    y = tmp_path / 'orb.yaml'; y.write_text(src)
    monkeypatch.setattr(m, 'ORB_YAML', y)
    monkeypatch.setattr(m, 'HISTORY', tmp_path / 'hist.jsonl')
    f = tmp_path / 'orb_features_x.csv'; _frame().to_csv(f, index=False)
    monkeypatch.setattr(sys, 'argv', ['x', '--dry-run', '--as-of', '2026-08-01', '--features', str(f)])
    assert m.main() == 0
    assert y.read_text() == src and not (tmp_path / 'hist.jsonl').exists()


def test_write_backs_up_and_logs(tmp_path, monkeypatch):
    src = (ROOT / 'orb.yaml').read_text()
    y = tmp_path / 'orb.yaml'; y.write_text(src)
    monkeypatch.setattr(m, 'ORB_YAML', y)
    monkeypatch.setattr(m, 'HISTORY', tmp_path / 'hist.jsonl')
    f = tmp_path / 'orb_features_x.csv'; _frame().to_csv(f, index=False)
    monkeypatch.setattr(sys, 'argv', ['x', '--as-of', '2026-08-01', '--features', str(f)])
    assert m.main() == 0
    assert y.read_text() != src and list(tmp_path.glob('orb.yaml.bak.refit_*'))
    rec = [l for l in (tmp_path / 'hist.jsonl').read_text().splitlines() if l]
    assert len(rec) == 1 and 'cutoffs' in rec[0]
    assert yaml.safe_load(y.read_text())['adaptive_mults'] == yaml.safe_load(src)['adaptive_mults']
