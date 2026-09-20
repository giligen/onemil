#!/usr/bin/env python3
"""frames10 / F33 — THE HISTORICAL REPLAY of the pooled ramp statistic.

Replays `trading/ramp_pool` over the streams that actually exist today:

  * ORB's live stage trades (`trades.db`, strategy='orb', from the stage start),
  * BF's live stage trades (strategy='bull_flag', from its stage start),
  * the HOD-break DRY run's simulated book (the EXECUTABLE would-be book printed by
    `scripts/hod_break_eod_check.py` for each dry session, parsed from `replay_dry.log`).

and prints what the pooled reading WOULD have said beside each book's own BT-band reading.

Read-only: it opens `trades.db` in ro mode and reads a log file. Nothing is written except the
dry pool CSV inside `frames10/`.
"""
import ast
import os
import re
import sys

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from pathlib import Path                                              # noqa: E402

from trading import ramp_bt_band as bb                                # noqa: E402
from trading import ramp_pool as rp                                   # noqa: E402
from trading import ramp_stage                                        # noqa: E402

D10 = Path(f'{ROOT}/research/mature_method/frames10')
LOG = D10 / 'replay_dry.log'
POOL = D10 / 'dry_pool_replay.csv'
RX = re.compile(r'DRY-RUN EXECUTABLE book .*?\|\s*(\[.*\])\s*$')


def parse_dry():
    """(day, symbol, R) per simulated dry trade, from the EOD check's own executable book."""
    if POOL.exists():
        POOL.unlink()
    day, n = None, 0
    for ln in open(LOG):
        if ln.startswith('=== '):
            day = ln.strip()[4:]
            continue
        m = RX.search(ln.strip())
        if m and day:
            rows = ast.literal_eval(m.group(1))
            n += rp.append_dry_trades(day, [(s, float(r)) for s, r in rows], POOL)
    print(f'  dry stream: {n} simulated trades parsed from {LOG.name}', flush=True)
    return n


def book_line(book, since, trades, ref_label, bt_r):
    live_r = [t.r for t in trades]
    mean = sum(live_r) / len(live_r) if live_r else None
    band = bb.bootstrap_band(bt_r, len(live_r)) if live_r else None
    st = bb.classify(mean, band)
    print(f'  {book.upper():4s} since {since}: n={len(live_r)} mean R '
          f'{"n/a" if mean is None else f"{mean:+.3f}"} vs {band.fmt() if band else "no band"} '
          f'-> {st}   [{ref_label}]', flush=True)
    return st


def main():
    print('== frames10 / F33 — the pooled-statistic historical replay ==', flush=True)
    parse_dry()

    # The CURRENT stage table opens both live books on 2026-09-21 (tomorrow), so the replay also
    # reads each book's PREVIOUS stage — the only window with live trades in it.
    cur_orb, _ = ramp_stage.resolve('orb')
    cur_bf, _ = ramp_stage.resolve('bf')
    prev_orb, prev_bf = '2026-08-17', '2026-09-07'
    print(f'  stage table: orb current {cur_orb}, previous {prev_orb}; '
          f'bf current {cur_bf}, previous {prev_bf}', flush=True)

    import yaml
    cfg = yaml.safe_load(open(f'{ROOT}/config.yaml'))
    bf_base = float(cfg['trading']['risk_per_trade'])
    oc = yaml.safe_load(open(f'{ROOT}/orb.yaml')) if os.path.exists(f'{ROOT}/orb.yaml') else {}
    veto = bool((oc or {}).get('filter', {}).get('catalyst_veto', {}).get('enabled'))

    sds, rr = rp.reference_sds_and_r([rp.BOOK_ORB, rp.BOOK_BF, rp.BOOK_HOD_DRY],
                                     orb_catalyst_veto=veto)
    print('\n  frozen BT references (SD and band from ONE distribution):', flush=True)
    pass9 = {'orb': 1.694, 'bf': 1.939, 'hod_dry': 1.260}
    for b in (rp.BOOK_ORB, rp.BOOK_BF, rp.BOOK_HOD_DRY):
        sd = sds.get(b, float('nan'))
        flag = 'MATCHES pass 9' if abs(sd - pass9[b]) < 0.01 else 'SUPERSEDES pass 9'
        print(f'    {b:8s} n_ref {len(rr.get(b, [])):5d}  SD {sd:.3f}  '
              f'(pass-9 walker SD {pass9[b]:.3f} -> {flag})', flush=True)

    for label, o_since, b_since in (('CURRENT stage', cur_orb, cur_bf),
                                    ('PREVIOUS stage', prev_orb, prev_bf)):
        print(f'\n== {label} ==', flush=True)
        t_orb = rp.load_live_trades(rp.BOOK_ORB, o_since)
        t_bf = rp.load_live_trades(rp.BOOK_BF, b_since, risk_base=bf_base)
        t_dry = rp.load_dry_trades(POOL, since=rp.HOD_DRY_SINCE)
        s_orb = book_line('orb', o_since, t_orb, 'orb_gates2/book_G3_meas', rr.get(rp.BOOK_ORB, []))
        s_bf = book_line('bf', b_since, t_bf, 'bf_frequency/runs/P1', rr.get(rp.BOOK_BF, []))
        print(f'  HOD-dry since {rp.HOD_DRY_SINCE}: n={len(t_dry)} mean R '
              f'{sum(t.r for t in t_dry) / len(t_dry):+.3f}' if t_dry else
              f'  HOD-dry since {rp.HOD_DRY_SINCE}: n=0', flush=True)

        for tag, pool in (('live books only', t_orb + t_bf),
                          ('live + HOD-dry', t_orb + t_bf + t_dry)):
            stat = rp.pooled_z(pool, sds)
            band = rp.pooled_band(rr, sds, stat.per_book_n) if stat.n else None
            st = rp.classify_pooled(stat, band)
            print(f'  [{tag:16s}] ' + rp.pooled_line(stat, band, st).strip(), flush=True)
            if stat.n:
                w = (band.p90 - band.p5) if band else float('nan')
                print(f'      band width p5..p90 = {w:.3f} z-units on n={stat.n}', flush=True)
            # the counterfactual the frame exists to print
            for book, bst in (('ORB', s_orb), ('BF', s_bf)):
                v, why = rp.apply_pooled_gate('ADVANCE', st)
                print(f'      counterfactual: if {book}\'s own gate had said ADVANCE '
                      f'(its own band read {bst}), the pooled gate would say {v}'
                      + (f' — {why}' if why else ''), flush=True)
    print('\nREPLAY DONE', flush=True)


if __name__ == '__main__':
    main()
