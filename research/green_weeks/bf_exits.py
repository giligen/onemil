#!/usr/bin/env python3
"""green_weeks — bull-flag exit cells, selection frozen at F7 (PREREG §3/§4).

Every cell is the SHIPPED `batch_backtest.py`, driven through scratch copies of
config.yaml.  Production config, caches, orders, services and crons are never
written.

Two-step per cell, exactly BT_STATUS §6's recipe:
  1. `--resim-exits` rebuilds the regen-7 cache's EXIT columns with that cell's
     exit knobs (`--resim-rich` recovers the true pattern fields);
  2. Stage-2 runs on the re-simulated cache with the F7 gate set at the LIVE
     rails (`max_positions 3`, `max_trades_per_day 5`, daily rail -5u).

**Every BF cell including E0 goes through the resim** (PREREG §5): the resim is
not exit-faithful (BT_STATUS §2a: -$10.3K / -7.4% vs regen-7's own exits), so the
comparison is resim-vs-resim and the drift is common-mode.  `E0_CACHE` is the
same F7 selection on regen-7's OWN exits, reported as a level anchor only.

Usage: python3 research/green_weeks/bf_exits.py [cell ...]
"""
import os
import subprocess
import sys

import yaml

ROOT = '/home/ec2-user/onemil'
D = f'{ROOT}/research/green_weeks'
RUNS = f'{D}/bf_runs'
SCRATCH = ('/tmp/claude-1000/-home-ec2-user-onemil/'
           '257c3e2d-cf38-45d5-94e7-4877f8170f44/scratchpad/green_weeks')
CACHE = f'{ROOT}/data/bull_flag_cache_causal_full_20260905.csv'
RICH = f'{ROOT}/backtest_results/backtest_full_2025_01_to_2026_08.csv'
START, END = '2025-01-01', '2026-08-31'

# cell -> exit knobs.  None = regen-7's own exits (no resim).
#   pp = (r_multiple, fraction)  |  be = trailing_stop.breakeven_at_r
CELLS = {
    'E0_CACHE': dict(resim=False, note="F7 on regen-7's OWN exits (level anchor)"),
    'E0':       dict(pp=(2.0, 0.5), note='shipped: 50% @ +2R, breakeven, trail'),
    'E1a':      dict(pp=(0.5, 0.999), note='whole position @ +0.5R'),
    'E1b':      dict(pp=(0.75, 0.999), note='whole position @ +0.75R'),
    'E1c':      dict(pp=(1.0, 0.999), note='whole position @ +1.0R'),
    'E1d':      dict(pp=(1.5, 0.999), note='whole position @ +1.5R'),
    'E2a':      dict(pp=(0.5, 0.5), note='50% @ +0.5R, breakeven, trail'),
    'E2b':      dict(pp=(1.0, 0.5), note='50% @ +1.0R, breakeven, trail (= E3)'),
    'E2c':      dict(pp=(1.5, 0.5), note='50% @ +1.5R, breakeven, trail'),
    'E5':       dict(pp=None, be=0.75, note='breakeven stop at +0.75R, no partial'),
}


def log(*a):
    print(*a, flush=True)


def build_cfg(cell, spec):
    """P1 config + the F7 gate relaxations + this cell's exit knobs."""
    with open(f'{ROOT}/config.yaml') as f:
        c = yaml.safe_load(f)
    t = c['trading']
    t['daily_loss_limit'] = -10000.0            # -5u at the $2K normalization
    # ---- F7 selection (bf_frequency §11), at the LIVE rails
    cs = t['conviction_scoring']
    cs['enabled'] = False
    cs['min_threshold'] = 0.0
    c['scanner']['min_daily_volume'] = 0
    t['bull_flag']['max_entry_price'] = 20.0
    t['bull_flag']['min_pole_gain_pct'] = 5.0
    t['bull_flag']['two_tier_filter']['drop_extras_macd_below'] = 1.25
    t['max_positions'] = 3
    t['max_trades_per_day'] = 5
    # ---- this cell's exit
    pp = t.setdefault('profit_partial', {})
    if spec.get('pp'):
        r, frac = spec['pp']
        pp.update(enabled=True, r_multiple=float(r), fraction=float(frac),
                  move_to_breakeven=True, fill='close', shadow=False,
                  runner_target_r=0.0)
    else:
        pp['enabled'] = False
    if 'be' in spec:
        t.setdefault('trailing_stop', {})['breakeven_at_r'] = float(spec['be'])
    p = f'{SCRATCH}/cfg_{cell}.yaml'
    with open(p, 'w') as f:
        yaml.safe_dump(c, f)
    return p


def run(cell, spec):
    cfgp = build_cfg(cell, spec)
    env = dict(os.environ)
    env['_ONEMIL_CFG'] = cfgp
    src = CACHE
    if spec.get('resim', True):
        src = f'{SCRATCH}/resim_{cell}.csv'
        env['BT_CACHE_PATH_OVERRIDE'] = CACHE
        p = subprocess.run(
            ['python3', 'batch_backtest.py', '--start', START, '--end', END,
             '--resim-exits', src, '--resim-rich', RICH],
            cwd=ROOT, env=env, capture_output=True, text=True)
        if p.returncode != 0:
            log(f'  !! {cell} RESIM FAILED rc={p.returncode}')
            log(p.stdout[-2500:], p.stderr[-2500:])
            return None
        for ln in (p.stdout + p.stderr).splitlines():
            if 'RESIM EXITS DONE' in ln:
                log('   ', ln.split('] ', 1)[-1])
    env['BT_CACHE_PATH_OVERRIDE'] = src
    out = f'{RUNS}/{cell}.csv'
    p = subprocess.run(
        ['python3', 'batch_backtest.py', '--start', START, '--end', END,
         '--capital', '50000', '--risk', '2000', '--max-shares', '10000',
         '--output', out],
        cwd=ROOT, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        log(f'  !! {cell} STAGE2 FAILED rc={p.returncode}')
        log(p.stdout[-2500:], p.stderr[-2500:])
        return None
    chain = [ln.split('] ', 1)[-1] for ln in (p.stdout + p.stderr).splitlines()
             if any(s in ln for s in ('filter (', 'Filter:', 'gate:', 'Two-tier:',
                                      'Universe filter', 'Price filter',
                                      'Pole-gain', 'Conviction', 'Profit partial'))]
    with open(f'{RUNS}/{cell}.log', 'w') as f:
        f.write('\n'.join(chain))
    return out


def main():
    os.makedirs(RUNS, exist_ok=True)
    os.makedirs(SCRATCH, exist_ok=True)
    want = set(sys.argv[1:])
    for cell, spec in CELLS.items():
        if want and cell not in want:
            continue
        log(f'-- {cell:9s} {spec["note"]}')
        run(cell, spec)
    log('BF CELLS DONE')


if __name__ == '__main__':
    main()
