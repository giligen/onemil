"""PREREG 1,738 Stage 0 scoring: coverage of contract + entry/exit prints, by read / price bucket / B0 outcome. Cache-only."""
import importlib.util, os, sys
from datetime import datetime
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sp = importlib.util.spec_from_file_location('f1738', os.path.join(HERE, '1738_fetch.py'))
F = importlib.util.module_from_spec(sp); sp.loader.exec_module(F)


def bar_minute(t):
    """Bar start (UTC ISO) -> ET minute of day."""
    dt = datetime.fromisoformat(t.replace('Z', '+00:00')).astimezone(F.ET)
    return dt.hour * 60 + dt.minute


def prints(bars, entry_m, exit_m):
    """Entry print (first bar in [entry, entry+5]) and exit print (last bar in [exit-15, exit]; fallback first in (exit, exit+15])."""
    bs = sorted(((bar_minute(b['t']), b['vw']) for b in bars))
    e = next(((m, v) for m, v in bs if entry_m <= m <= entry_m + 5), None)
    x = [(m, v) for m, v in bs if exit_m - 15 <= m <= exit_m]
    fb = False
    if x: x = x[-1]
    else:
        x = next(((m, v) for m, v in bs if exit_m < m <= exit_m + 15), None); fb = x is not None
    return e, x, fb


def main(n):
    """Score the sample from cache (fetch must have run) and write 1738_stage0.csv."""
    df = F.sample(n); out = []
    for _, r in df.iterrows():
        c = F.pick_contract(r.symbol, r.day, r.entry)
        bars = F.fetch_bars(c['symbol'], r.day, r.entry_m, r.exit_m) if c else []
        e, x, fb = prints(bars, r.entry_m, r.exit_m) if c else (None, None, False)
        out.append(dict(day=r.day, symbol=r.symbol, split=r.split, entry=r.entry, entry_m=r.entry_m, exit_m=r.exit_m, why=r.why,
                        net_R=r.net_R, has_contract=c is not None, contract=c and c['symbol'], expiry=c and c['expiration_date'],
                        strike=c and float(c['strike_price']), n_bars=len(bars),
                        entry_print=e[1] if e else None, entry_print_m=e[0] if e else None,
                        exit_print=x[1] if x else None, exit_print_m=x[0] if x else None, exit_fallback=fb))
    d = pd.DataFrame(out)
    d['has_entry'] = d.entry_print.notna(); d['has_exit'] = d.exit_print.notna()
    d['covered'] = d.has_contract & d.has_entry & d.has_exit
    d['bucket'] = pd.cut(d.entry, [0, 20, 50, 100, 1e9], labels=['<20', '20-50', '50-100', '100+'], right=False)
    d['outcome'] = (d.net_R > 0).map({True: 'winner', False: 'loser'})
    d.to_csv(os.path.join(HERE, '1738_stage0.csv'), index=False)
    pd.set_option('display.width', 200)
    cols = ['has_contract', 'has_entry', 'has_exit', 'covered']
    print('OVERALL', d[cols].mean().round(3).to_dict(), 'n', len(d))
    for k in ('split', 'bucket', 'outcome', 'why'):
        print(f'\nby {k}'); print(d.groupby(k, observed=True)[cols].mean().round(3).assign(n=d.groupby(k, observed=True).size()))
    w, l = d[d.outcome == 'winner'].covered.mean(), d[d.outcome == 'loser'].covered.mean()
    print(f'\nwinner cov {w:.3f} loser cov {l:.3f} gap {abs(w-l)*100:.1f} pp; exit fallback share {d.exit_fallback.mean():.3f}')
    print('requests: ', F.STATS)
    print('median n_bars among contracts', d[d.has_contract].n_bars.median(), 'contracts', d.has_contract.sum())


if __name__ == '__main__':
    main(200)
