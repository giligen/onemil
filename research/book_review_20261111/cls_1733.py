"""Cell 1,733 part 2 -- classify every live/BT divergence and re-sim the paper book under selection parity.

Reads selection_1733.csv (sel_1733.py), the nightly book, the candidate sim and stops_1732.csv. Read-only; prints
tables that RESULT_1733.md quotes. The point-in-time (PIT) BT book = PIT top-8 (refit params in force that week)
minus the shared post-ranking vetoes (pdr <= 11.0, g1 rv20 < 7.106 & pdr >= 9.226, range <= 2.221), sim-entered.
"""
import math
import os
import sys

import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil')
os.chdir('/home/ec2-user/onemil')
from trading.orb_csv import read_orb_csv  # noqa: E402
from trading.orb_g1_veto import g1_reject  # noqa: E402
from trading.orb_pdr_veto import pdr_veto_applies  # noqa: E402
from trading.orb_range_size_veto import range_size_veto_applies  # noqa: E402

D0, D1 = '2026-09-29', '2026-11-09'
R = 'research/book_review_20261111/'
# Manual class per divergence, each backed by the journal lines quoted in RESULT_1733.md
CLASS = {
    ('2026-09-29', 'AXTL'): 'U+T', ('2026-09-30', 'ASTX'): 'U', ('2026-10-06', 'APLX'): 'P',
    ('2026-10-08', 'SMCZ'): 'P', ('2026-10-12', 'ASTY'): 'U', ('2026-10-16', 'CORD'): 'R',
    ('2026-10-23', 'SK'): 'U', ('2026-11-02', 'SKUU'): 'V', ('2026-11-02', 'SKHX'): 'U',
    ('2026-11-03', 'SNXX'): 'R', ('2026-11-03', 'AAOX'): 'R',
    ('2026-10-05', 'DFDV'): 'U', ('2026-10-13', 'AAOZ'): 'U', ('2026-10-23', 'SAIQ'): 'U',
    ('2026-11-02', 'AEYE'): 'U', ('2026-11-03', 'RAM'): 'R', ('2026-11-03', 'SKHL'): 'R',
    ('2026-11-09', 'BEG'): 'T', ('2026-11-09', 'CRDU'): 'T',
}


def veto(r):
    """Shared post-ranking vetoes, BT inputs."""
    if pdr_veto_applies(None if pd.isna(r.prev_day_range_pct) else float(r.prev_day_range_pct), 11.0):
        return 'pdr'
    if g1_reject(r.return_volatility_20d, r.prev_day_range_pct, 7.106, 9.226, short_history_veto=False) is not None:
        return 'g1'
    if range_size_veto_applies(r.range_size_pct, 2.221):
        return 'rs'
    return None


def day_stats(s, days):
    """Day-clustered mean/SE/t of a per-day P&L series over all trading days."""
    v = s.reindex(days, fill_value=0.0)
    n = len(v)
    sd = v.std(ddof=1)
    return f'sum {v.sum():+.0f}  mean/day {v.mean():+.1f}  SE {sd / math.sqrt(n):.1f}  t {v.mean() / (sd / math.sqrt(n)):+.2f}  days {n}'


def main():
    sel = pd.read_csv(R + 'selection_1733.csv')
    ft = read_orb_csv('analysis_results/orb_features_20261110_2203.csv')
    ft['date'] = pd.to_datetime(ft['date']).dt.strftime('%Y-%m-%d')
    pit = sel[sel.bt_topk_pit == True][['date', 'symbol']].merge(  # noqa: E712
        ft[['date', 'symbol', 'prev_day_range_pct', 'return_volatility_20d', 'range_size_pct', 'entered',
            'exit_reason', 'pnl']], on=['date', 'symbol'], how='left')
    pit['veto'] = pit.apply(veto, axis=1)
    pit_book = pit[pit.veto.isna() & (pit.entered == 1) & (pit.date <= D1)]
    book = pd.read_csv('analysis_results/orb_bplus_book.csv')
    book = book[(book.date >= D0) & (book.date <= D1) & (book.entered == 1)]
    fills = pd.read_csv(R + 'stops_1732.csv')
    live = fills[['date', 'symbol', 'live_pnl', 'live_exit_reason']]
    days = sorted(set(ft[(ft.date >= D0) & (ft.date <= D1)].date))
    L = set(zip(live.date, live.symbol))
    B = set(zip(book.date, book.symbol))
    P = set(zip(pit_book.date, pit_book.symbol))
    print(f'live fills {len(L)}  BT book entered {len(B)}  PIT-BT entered {len(P)}  trading days {len(days)}')
    print(f'live∩BT {len(L & B)}  live-only {len(L - B)}  BT-only {len(B - L)}')
    print(f'live∩PIT {len(L & P)}  live-only vs PIT {sorted(L - P)}  PIT-only {sorted(P - L)}')
    rows = []
    bpnl = dict(zip(zip(book.date, book.symbol), book._sized_pnl))
    lp = dict(zip(zip(live.date, live.symbol), live.live_pnl))
    lx = dict(zip(zip(live.date, live.symbol), live.live_exit_reason))
    bx = dict(zip(zip(book.date, book.symbol), book.exit_reason))
    for k in sorted((L - B) | (B - L)):
        side = 'live-only' if k in L else 'BT-only'
        pnl = lp.get(k) if k in L else bpnl.get(k)
        ex = lx.get(k) if k in L else bx.get(k)
        rows.append(dict(date=k[0], symbol=k[1], side=side, cls=CLASS.get(k, '?'), pnl=round(pnl, 1), exit=ex,
                         stop=ex in ('stop_loss', 'stop'), in_pit_book=k in P))
    d = pd.DataFrame(rows)
    print(d.to_string(index=False))
    print(d.groupby(['side', 'cls']).agg(n=('pnl', 'size'), pnl=('pnl', 'sum'), stops=('stop', 'sum')).to_string())
    print('PIT veto check, PIT top-8 vetoed by BT inputs:')
    print(pit[pit.veto.notna()].groupby('veto').size().to_string())
    # Re-sim. A: live traded exactly the published BT names (matched -> live P&L, BT-only -> BT row).
    # B: exactly the PIT-BT names (matched -> live, PIT-only -> BT row if published else candidate-sim exit).
    def per_day(keys, prefer_live):
        out = {}
        for k in keys:
            v = lp[k] if (prefer_live and k in L) else bpnl.get(k, lp.get(k, float('nan')))
            out[k[0]] = out.get(k[0], 0.0) + (0.0 if pd.isna(v) else v)
        return pd.Series(out, dtype=float)
    print('Live actual            ', day_stats(live.groupby('date').live_pnl.sum(), days),
          f"stops {(live.live_exit_reason == 'stop_loss').sum()}/{len(live)}")
    print('A live∩BT names (pub.) ', day_stats(per_day(B, True), days))
    print('B live∩PIT names       ', day_stats(per_day(P, True), days),
          f"(PIT-only rows without a BT/live P&L: {[k for k in P - L if k not in bpnl]})")
    print('BT published realized  ', day_stats(book.groupby('date')._sized_pnl.sum(), days),
          f"stops {(book.exit_reason == 'stop').sum()}/{len(book)}")
    a_st = sum(1 for k in B if (lx.get(k) == 'stop_loss' if k in L else bx.get(k) == 'stop'))
    p_st = sum(1 for k in P if (lx.get(k) == 'stop_loss' if k in L else bx.get(k) == 'stop'))
    print(f'stop share A {a_st}/{len(B)}  B {p_st}/{len(P)}')
    print('PIT-only names and their candidate-sim exits:')
    print(pit_book[pit_book.apply(lambda r: (r.date, r.symbol) not in L, axis=1)][
        ['date', 'symbol', 'exit_reason', 'pnl']].to_string(index=False))


if __name__ == '__main__':
    main()
