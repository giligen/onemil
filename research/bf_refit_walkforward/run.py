#!/usr/bin/env python3
"""BF walk-forward SELECTION refit — owner 2026-09-08 "run it".

Question: BF's remaining fitted selection layer under P1 is the two-tier
composite (z-params of conviction_mult, qf_vwap_dist_pct,
qf_fill_vwap_dist_pct, entry_minute; thresholds frozen). It was fit on 2025
and decayed in 2026. Does a weekly refit of those z-params on a trailing
window beat the frozen fit, walk-forward, under the LIVE P1 config?

Method: every Monday from 2025-03-03, fit mean/std per feature on the cache
rows (raw detections = every candidate Stage-2 sees) dated strictly before
the week and within the window; write a temp config = live config.yaml with
those z-params; run the real Stage-2 (batch_backtest.py) on that week only
against the P1 exit cache (resim +2R partial, regen-7 entries) at the $2K /
$50K normalization; concatenate. Frozen = the same loop with the yaml
literals — must reproduce the full-period P1 Stage-2 run.

Nothing else is refit: conviction threshold, composite threshold, MACD tier
multipliers (sizing) and every P1 rule stay as shipped.
"""
import os
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(ROOT); sys.path.insert(0, ROOT)
from trading.orb_csv import read_orb_csv  # noqa: E402

CACHE = 'research/bf_consistency/resim_PPU_r2.0_f0.5.csv'
OUT = 'research/bf_refit_walkforward'
FEATS = ['conviction_mult', 'qf_vwap_dist_pct', 'qf_fill_vwap_dist_pct', 'entry_minute']
LIVE_CFG = yaml.safe_load(open('config.yaml'))
FROZEN = LIVE_CFG['trading']['bull_flag']['two_tier_filter']['composite_features']
MIN_ROWS = 40


def entry_minute(s):
    t = pd.to_datetime(s.astype(str), format='%H:%M:%S', errors='coerce')
    return t.dt.hour * 60 + t.dt.minute


def fit(rows: pd.DataFrame):
    if len(rows) < MIN_ROWS:
        return None
    out = {}
    for f in FEATS:
        col = pd.to_numeric(rows[f], errors='coerce').dropna()
        if len(col) < MIN_ROWS // 2 or float(col.std(ddof=0)) <= 1e-9:
            return None
        out[f] = {'mean': round(float(col.mean()), 3), 'std': round(float(col.std(ddof=0)), 3),
                  'sign': int(FROZEN[f]['sign'])}
    return out


def stage2(params, w0, w1, tag):
    cfg = yaml.safe_load(open('config.yaml'))
    cfg['trading']['bull_flag']['two_tier_filter']['composite_features'] = params
    cfg['trading']['daily_loss_limit'] = -10000.0            # -5u at $2K (ramp-consistent)
    fd, path = tempfile.mkstemp(suffix='.yaml', prefix=f'bfwf_{tag}_'); os.close(fd)
    yaml.safe_dump(cfg, open(path, 'w'))
    env = dict(os.environ, BT_CACHE_PATH_OVERRIDE=CACHE)
    r = subprocess.run([sys.executable, 'batch_backtest.py', '--config', path, '--start', w0, '--end', w1,
                        '--capital', '50000', '--risk', '2000', '--max-shares', '10000'],
                       capture_output=True, text=True, env=env)
    os.unlink(path)
    if r.returncode != 0:
        raise SystemExit(f"Stage-2 failed {w0}: {r.stderr[-500:]}")
    try:
        b = read_orb_csv('backtest_results_march_2026.csv')
    except Exception:
        return pd.DataFrame()
    return b


def run(mode, cache, weeks):
    parts = []
    for w0 in weeks:
        w1 = w0 + pd.Timedelta(days=6)
        if mode == 'frozen':
            prm = FROZEN
        else:
            lo = pd.Timestamp('2000-01-01') if mode == 'expanding' else w0 - pd.Timedelta(weeks=int(mode[:-1]))
            prm = fit(cache[(cache['date'] >= lo) & (cache['date'] < w0)])
            if prm is None:
                continue
        b = stage2(prm, w0.strftime('%Y-%m-%d'), w1.strftime('%Y-%m-%d'), mode)
        if len(b):
            parts.append(b)
    b = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=['date', 'pnl'])
    b['date'] = pd.to_datetime(b['date'].astype(str).str[:10])
    months = pd.period_range('2025-03', '2026-09', freq='M')
    m = b.groupby(b['date'].dt.to_period('M')).pnl.sum().reindex(months).fillna(0)
    c = m.cumsum()
    era = lambda a, z: round(b[(b.date >= a) & (b.date <= z)].pnl.sum())
    return dict(mode=mode, trades=len(b), total=round(m.sum()), mdd=round((c - c.cummax()).min()),
                red=int((m < 0).sum()), worst=round(m.min()), green=int((m > 0).sum()),
                e25=era('2025-01-01', '2025-12-31'), e2026=era('2026-01-01', '2026-12-31')), b


def main():
    cache = read_orb_csv(CACHE); cache['date'] = pd.to_datetime(cache['date'].astype(str).str[:10])
    cache['entry_minute'] = entry_minute(cache['entry_time_et'])
    freq = os.environ.get('REFIT_WEEK_FREQ', 'W-MON')
    weeks = pd.date_range('2025-03-03', '2026-09-04', freq=freq)
    modes = sys.argv[1:] or ['frozen', '13w', '20w', '26w', '39w', 'expanding']
    out = []
    for mode in modes:
        s, b = run(mode, cache, weeks)
        b.to_csv(f'{OUT}/{mode}{"" if freq == "W-MON" else "_" + freq}_book.csv', index=False)
        print(s, flush=True); out.append(s)
    T = pd.DataFrame(out); T.to_csv(f'{OUT}/summary{"" if freq == "W-MON" else "_" + freq}.csv', index=False)
    print('\n' + T.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
