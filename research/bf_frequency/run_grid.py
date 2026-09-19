#!/usr/bin/env python3
"""BF frequency frontier — cell runner.

Runs the SHIPPED batch_backtest.py Stage-2 once per declared cell, varying only
config knobs through a scratch copy of config.yaml (_ONEMIL_CFG). Production
config, caches, orders, services and crons are never written.

Every cell uses regen-7's OWN exits (no --resim-exits), so the profit partial is
off in all of them and the baseline is BT_STATUS run A (56 tr / $139,113.67).

Usage:  python3 research/bf_frequency/run_grid.py [cell_id ...]
"""
import os
import sys
import copy
import json
import subprocess

import yaml

ROOT = '/home/ec2-user/onemil'
OUT = f'{ROOT}/research/bf_frequency'
RUNS = f'{OUT}/runs'
SCRATCH = ('/tmp/claude-1000/-home-ec2-user-onemil/'
           '257c3e2d-cf38-45d5-94e7-4877f8170f44/scratchpad/bf_freq')
CACHE = f'{ROOT}/data/bull_flag_cache_causal_full_20260905.csv'
START, END = '2025-01-01', '2026-08-31'

os.makedirs(RUNS, exist_ok=True)
os.makedirs(SCRATCH, exist_ok=True)


def log(*a):
    print(*a)
    sys.stdout.flush()


# --------------------------------------------------------------- knob setters
def set_conv(c, v):
    cs = c['trading']['conviction_scoring']
    if v is None:                      # "off"
        cs['enabled'] = False
        cs['min_threshold'] = 0.0
    else:
        cs['enabled'] = True
        cs['min_threshold'] = float(v)


def set_vol(c, v):
    c['scanner']['min_daily_volume'] = int(v or 0)


def set_price(c, v):
    c['trading']['bull_flag']['max_entry_price'] = float(v)


def set_pole(c, v):
    c['trading']['bull_flag']['min_pole_gain_pct'] = float(v)


def set_macd_gate(c, on):
    tt = c['trading']['bull_flag']['two_tier_filter']
    tt['drop_extras_macd_below'] = 1.25 if on else 0.0


def set_conc(c, v):
    c['trading']['max_positions'] = int(v)


def set_mtd(c, v):
    c['trading']['max_trades_per_day'] = int(v)


# --------------------------------------------------------------- the grid
# Each cell: (id, description, dict of knob overrides, full_market flag)
# P1 defaults: conv 1.8, vol 200000, price 20, pole 5.0, macd gate on,
#              max_positions 3, max_trades_per_day 5.
P1 = dict(conv=1.8, vol=200000, price=20.0, pole=5.0, macd=True, conc=3, mtd=5)

CELLS = []


def cell(cid, desc, block, full_market=False, **kw):
    k = dict(P1)
    k.update(kw)
    CELLS.append(dict(id=cid, desc=desc, block=block, knobs=k,
                      full_market=full_market))


# --- baseline ---------------------------------------------------------------
cell('P1', 'shipped P1 (baseline = BT_STATUS run A)', 'baseline')

# --- A2 ladders (one knob at a time, everything else P1) --------------------
cell('CONV_1.5', 'conviction >= 1.5', 'ladder_conviction', conv=1.5)
cell('CONV_1.2', 'conviction >= 1.2', 'ladder_conviction', conv=1.2)
cell('CONV_1.0', 'conviction >= 1.0', 'ladder_conviction', conv=1.0)
cell('CONV_OFF', 'conviction gate off', 'ladder_conviction', conv=None)

cell('VOL_100K', 'ADV20 >= 100K', 'ladder_volume', vol=100000)
cell('VOL_50K', 'ADV20 >= 50K', 'ladder_volume', vol=50000)
cell('VOL_OFF', 'ADV20 gate off', 'ladder_volume', vol=0)

cell('PX_25', 'entry <= $25', 'ladder_price', price=25.0)
cell('PX_30', 'entry <= $30 (= off, cache band)', 'ladder_price', price=30.0)

cell('POLE_4', 'pole >= 4%', 'ladder_pole', pole=4.0)
cell('POLE_3', 'pole >= 3% (= off, cache floor)', 'ladder_pole', pole=3.0)

cell('MACD_OFF', 'two-tier MACD surgical-drop leg off', 'ladder_macd', macd=False)

# --- A3 book constraints on the P1 gate set ---------------------------------
cell('BK_3x8', 'max_conc 3 / max_trades_day 8', 'book', conc=3, mtd=8)
cell('BK_5x5', 'max_conc 5 / max_trades_day 5', 'book', conc=5, mtd=5)
cell('BK_5x8', 'max_conc 5 / max_trades_day 8', 'book', conc=5, mtd=8)
cell('BK_8x5', 'max_conc 8 / max_trades_day 5', 'book', conc=8, mtd=5)
cell('BK_8x8', 'max_conc 8 / max_trades_day 8', 'book', conc=8, mtd=8)

# --- A4 frontier points (provenance written in REPORT; declared here) -------
# Slots are raised to 8x8 on every frontier point above P1 so the gate change,
# not a slot rail, is what moves frequency (BK_* measures the rail in isolation).
cell('F1', 'P1 + vol 100K [vol ladder rung 2]', 'frontier',
     vol=100000, conc=8, mtd=8)
cell('F2', 'P1 + vol 100K + conv 1.5 [vol r2 + conv r2]', 'frontier',
     vol=100000, conv=1.5, conc=8, mtd=8)
cell('F3', 'P1 + vol 50K + conv 1.5 + price 25 [vol r3 + conv r2 + px r2]',
     'frontier', vol=50000, conv=1.5, price=25.0, conc=8, mtd=8)
cell('F4', 'P1 + vol off + conv 1.5 + price 30 [vol r4 + conv r2 + px r3]',
     'frontier', vol=0, conv=1.5, price=30.0, conc=8, mtd=8)
cell('F5', 'P1 + vol off + conv 1.2 + price 30 + pole 4 '
     '[vol r4 + conv r3 + px r3 + pole r2]', 'frontier',
     vol=0, conv=1.2, price=30.0, pole=4.0, conc=8, mtd=8)
cell('F6', 'all gates off except VWAP + two-tier [every ladder at its last rung]',
     'frontier', vol=0, conv=None, price=30.0, pole=3.0, conc=8, mtd=8)

# --- A4 (cont.) frontier points CHOSEN BY THE LADDERS, per PREREG §A4 -------
# The two ladders that added frequency without costing total R on either split
# were volume and conviction; price/pole/MACD were near-inert on frequency.
cell('F7', 'P1 + vol off + conv off [vol ladder r4 + conv ladder r5]',
     'frontier', vol=0, conv=None, conc=8, mtd=8)
cell('F8', 'P1 + vol off + conv off + price 30 [vol r4 + conv r5 + px r3]',
     'frontier', vol=0, conv=None, price=30.0, conc=8, mtd=8)

# --- A3b book rails AT FRONTIER DENSITY (added after A3 came back inert at
#     P1 density; the live config runs max_positions 3 / max_trades_day 5, so
#     every candidate must also be scored at the LIVE rails, not only at 8x8).
for _f, _k in (('F3_L', dict(vol=50000, conv=1.5, price=25.0)),
               ('F4_L', dict(vol=0, conv=1.5, price=30.0)),
               ('F7_L', dict(vol=0, conv=None)),
               ('F8_L', dict(vol=0, conv=None, price=30.0)),
               ('F6_L', dict(vol=0, conv=None, price=30.0, pole=3.0))):
    cell(_f, f'{_f[:-2]} at the LIVE rails (max_conc 3 / max_trades_day 5)',
         'live_rails', conc=3, mtd=5, **_k)

# --- bend attribution (DIAGNOSTIC ONLY — not eligible for the §5 rule) ------
cell('F4b', 'F4 + pole 4 (which F4→F5 leg bends the curve?)', 'bend_diag',
     vol=0, conv=1.5, price=30.0, pole=4.0, conc=8, mtd=8)
cell('F4c', 'F4 + conv 1.2 (which F4→F5 leg bends the curve?)', 'bend_diag',
     vol=0, conv=1.2, price=30.0, conc=8, mtd=8)

# --- A5 PIT bracket: the same frontier points with --full-market ------------
for _f in ('P1', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F7_L'):
    _src = next(c for c in CELLS if c['id'] == _f)
    CELLS.append(dict(id=f'{_f}_FM', desc=f'{_src["desc"]} + --full-market (no universe snapshot)',
                      block='pit_bracket', knobs=dict(_src['knobs']), full_market=True))


# --------------------------------------------------------------- run one cell
def run(c):
    with open(f'{ROOT}/config.yaml') as f:
        cfg = yaml.safe_load(f)
    # The live ramp's proportional rail at the $2K normalization: -5u.
    cfg['trading']['daily_loss_limit'] = -10000.0
    k = c['knobs']
    set_conv(cfg, k['conv'])
    set_vol(cfg, k['vol'])
    set_price(cfg, k['price'])
    set_pole(cfg, k['pole'])
    set_macd_gate(cfg, k['macd'])
    set_conc(cfg, k['conc'])
    set_mtd(cfg, k['mtd'])

    cfgp = f'{SCRATCH}/cfg_{c["id"]}.yaml'
    with open(cfgp, 'w') as f:
        yaml.safe_dump(cfg, f)

    outp = f'{RUNS}/{c["id"]}.csv'
    cmd = ['python3', 'batch_backtest.py', '--start', START, '--end', END,
           '--capital', '50000', '--risk', '2000', '--max-shares', '10000',
           '--output', outp]
    if c['full_market']:
        cmd.append('--full-market')
    env = dict(os.environ)
    env['_ONEMIL_CFG'] = cfgp
    env['BT_CACHE_PATH_OVERRIDE'] = CACHE
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        log(f'  !! {c["id"]} FAILED rc={p.returncode}')
        log(p.stdout[-3000:], p.stderr[-3000:])
        return None
    # capture the filter-chain log lines for the cascade table + BP bind rate
    chain = [ln.split('] ', 1)[-1] for ln in p.stdout.splitlines() + p.stderr.splitlines()
             if any(s in ln for s in ('filter (', 'Filter:', 'gate:', 'Two-tier:',
                                      'BP ceiling', 'Risk tiers', 'Universe filter',
                                      'universe filter', 'Volume filter', 'Price filter',
                                      'Pole-gain filter', 'Conviction filter',
                                      'Pole-bars filter', 'Intraday-change filter'))]
    with open(f'{RUNS}/{c["id"]}.log', 'w') as f:
        f.write('\n'.join(chain))
    return outp


def main():
    want = set(sys.argv[1:])
    meta = []
    for c in CELLS:
        if want and c['id'] not in want:
            continue
        log(f'-- {c["id"]:10s} {c["desc"]}')
        r = run(c)
        meta.append(dict(id=c['id'], desc=c['desc'], block=c['block'],
                         knobs={kk: vv for kk, vv in c['knobs'].items()},
                         full_market=c['full_market'], csv=r))
    with open(f'{RUNS}/meta.json', 'w') as f:
        json.dump(meta, f, indent=1)
    log(f'\n{len(meta)} cells run.')


if __name__ == '__main__':
    main()
