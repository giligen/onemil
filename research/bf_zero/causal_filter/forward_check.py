#!/usr/bin/env python3
"""CAUSAL_FILTER step 6 — the FORWARD check: the dry-run journal since 2026-09-14.

Every `[HOD DRY] WOULD BUY` line (captured once to causal_filter/hod_lines.txt) is scored to spec with
`scripts/hod_break_eod_check.py`'s own machinery — the same regex, the same `bars_for`, the same
`trading.hod_break.simulate` — then the book rule `run_book(rows, 12, 4)`, then the surviving filter.
n is tiny by construction; it is reported as-is, never annualised.

Output: causal_filter/forward.csv + a printed summary.
"""
import os, re, sys
import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/scripts')
from dotenv import load_dotenv                                  # noqa: E402
load_dotenv(f'{ROOT}/.env')
from hod_break_eod_check import rx_dry, bars_for                # noqa: E402
from trading.hod_break import HodBreakParams, simulate, run_book  # noqa: E402
from data_sources.alpaca_client import AlpacaClient                  # noqa: E402
from config import Config                                       # noqa: E402

D = f'{ROOT}/research/bf_zero/causal_filter'
SRC = f'{D}/hod_lines.txt'
RX_DAY = re.compile(r'(\d{4}-\d{2}-\d{2}) \d{2}:\d{2}:\d{2}')


def main():
    lines = [ln for ln in open(SRC, errors='replace') if 'WOULD BUY' in ln]
    rx = rx_dry()
    sigs = []
    for ln in lines:
        d = RX_DAY.search(ln)
        m = rx.search(ln)
        if d and m:
            sigs.append(dict(day=d.group(1), symbol=m.group(1), level=float(m.group(2)),
                             stop=float(m.group(4)), rv=float(m.group(10)),
                             spread_bps=float(m.group(11)), dist_open=float(m.group(9))))
    S = pd.DataFrame(sigs).drop_duplicates(['day', 'symbol'])
    print(f'WOULD BUY lines {len(lines)} | unique (day, symbol) {len(S)} | days {sorted(S.day.unique())}', flush=True)
    if not len(S):
        return
    cfg = Config()
    P = HodBreakParams(**(cfg.hod_break_cfg.get('params') or {}))
    alp = AlpacaClient(cfg.alpaca_api_key, cfg.alpaca_api_secret)
    rows = []
    for day, sub in S.groupby('day'):
        syms = sorted(sub.symbol.unique())
        bars = bars_for(alp, syms, day)
        # ADV20 exactly as the engine reads it: the 20 daily bars strictly before the session
        dr = alp.get_daily_bars_range(syms, (pd.Timestamp(day) - pd.Timedelta(days=45)).date(),
                                      (pd.Timestamp(day) - pd.Timedelta(days=1)).date())
        adv = {}
        for s in syms:
            dd = (dr or {}).get(s) or []
            adv[s] = float(pd.DataFrame(dd).volume.tail(20).mean()) if len(dd) else np.nan
        for sym in syms:
            b = bars.get(sym)
            if b is None:
                print(f'  {day} {sym}: no bars', flush=True)
                continue
            o, h, l, c, v, m = b
            a = adv.get(sym, np.nan)
            if not (a == a):
                print(f'  {day} {sym}: no ADV20', flush=True)
                continue
            t = simulate(o, h, l, c, v, m, a, P)
            if t is None:
                print(f'  {day} {sym}: spec had no trade', flush=True)
                continue
            i = int(t.entry_idx)
            vm = float(np.mean(v[:i])) if i > 0 else np.nan
            rows.append(dict(day=day, symbol=sym, entry_m=int(m[t.entry_idx]), exit_m=int(m[t.exit_idx]),
                             entry=t.entry, stop=t.stop, rr=t.rr, why=t.reason, price=t.entry,
                             r_pct=t.r_per_share / t.entry * 100,
                             bar_vol_x=float(v[i] / vm) if vm and vm > 0 else np.nan,
                             spread_bps=float(sub[sub.symbol == sym].spread_bps.iloc[0]),
                             adv20=a))
    F = pd.DataFrame(rows)
    if not len(F):
        print('no spec trades reconstructed', flush=True)
        return
    F.to_csv(f'{D}/forward.csv', index=False)
    bk = pd.DataFrame(run_book([(r.day, r.entry_m, r.exit_m, r.symbol, r.rr, r.why, r.r_pct, r.spread_bps)
                                for r in F.itertuples()], P.max_per_day, P.max_concurrent),
                      columns=['day', 'em', 'xm', 'symbol', 'rr', 'why', 'r_pct', 'spread_bps'])
    RATIO = {'stop': 0.875, 'eod': 0.412, 'target': 0.0}
    half = 0.5 * (bk.spread_bps / 100.0) / bk.r_pct.clip(lower=0.05)
    bk['net'] = bk.rr - half - half * bk.why.map(RATIO).fillna(0.875)
    bk.to_csv(f'{D}/forward_book.csv', index=False)
    print(f'\nspec signals {len(F)} | booked {len(bk)} | gross R {bk.rr.mean():+.3f} | '
          f'net R {bk.net.mean():+.3f} | total net {bk.net.sum():+.2f}R over {bk.day.nunique()} sessions', flush=True)
    print(bk.to_string(index=False), flush=True)
    # Diagnostic: no cell cleared G1, so there is no shipped filter to apply. The best of the 12
    # (veto the worst TRAIN tercile of bar_vol_x) is applied anyway, with its TRAIN edges.
    tr = pd.read_csv(f'{D}/features.csv', dtype={'symbol': str, 'day': str},
                     keep_default_na=False, na_values=[''])
    tr = tr[tr.split == 'TRAIN'].bar_vol_x.astype(float)
    lo = float(np.nanquantile(tr, 1 / 3))
    keep = F[F.bar_vol_x > lo]
    bk2 = pd.DataFrame(run_book([(r.day, r.entry_m, r.exit_m, r.symbol, r.rr, r.why, r.r_pct, r.spread_bps)
                                 for r in keep.itertuples()], P.max_per_day, P.max_concurrent),
                       columns=['day', 'em', 'xm', 'symbol', 'rr', 'why', 'r_pct', 'spread_bps'])
    if len(bk2):
        h2 = 0.5 * (bk2.spread_bps / 100.0) / bk2.r_pct.clip(lower=0.05)
        bk2['net'] = bk2.rr - h2 - h2 * bk2.why.map(RATIO).fillna(0.875)
        print(f'\nbest-of-12 cell (veto bar_vol_x <= {lo:.2f}, TRAIN edge): booked {len(bk2)} | '
              f'gross {bk2.rr.mean():+.3f} | net {bk2.net.mean():+.3f} | total {bk2.net.sum():+.2f}R', flush=True)


if __name__ == '__main__':
    main()
