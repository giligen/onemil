"""Refuter-1: staleness of the as-of daily marks used for Mgmt-A exits, and 2025-04 MTM of open 1574 legs."""
import os, pandas as pd, sqlite3
HERE = os.path.dirname(os.path.abspath(__file__)); P = os.path.dirname(HERE)
cyc = pd.read_csv(os.path.join(P, 'cell_1567_cycles.csv'))
od = pd.read_parquet(os.path.join(P, 'opt_cache/option_daily.parquet'))
con = sqlite3.connect(os.path.join(P, 'opt_cache/state.db'))
ct = pd.read_sql_query('select symbol, expiry, strike from contracts', con)
key = ct.set_index(['expiry', 'strike'])['symbol'].to_dict()
have = set(zip(od.symbol, od.day))
a = cyc[(cyc.mgmt == 'A') & (cyc.exit_reason.isin(['profit_50', 'dte21']))].copy()
def stale(r):
    s = key.get((r.expiry, float(r.short_strike))); l = key.get((r.expiry, float(r.long_strike)))
    return int((s, r.exit_date) not in have), int((l, r.exit_date) not in have)
a[['s_stale', 'l_stale']] = a.apply(lambda r: pd.Series(stale(r)), axis=1)
a['any_stale'] = (a.s_stale | a.l_stale)
print('Mgmt-A same-close exits:', len(a), ' share with a leg marked from an OLDER (forward-filled) close:', round(a.any_stale.mean(), 3))
print(a.groupby('exit_reason').any_stale.mean())
print('mean pnl stale vs fresh:', a.groupby('any_stale').pnl_usd.mean().to_dict())
st = cyc[cyc.exit_reason.str.startswith('stop')]
print('stop exits all cells:', len(st))
for exp, k in [('2025-04-25', 515.0), ('2025-04-25', 505.0), ('2025-04-30', 525.0), ('2025-04-30', 515.0), ('2025-04-04', 560.0), ('2025-04-04', 550.0), ('2025-04-17', 550.0), ('2025-04-17', 540.0)]:
    s = key.get((exp, k)); d = od[(od.symbol == s) & (od.day >= '2025-04-03') & (od.day <= '2025-04-09')]
    print(exp, k, s, d[['day', 'o', 'c', 'v']].values.tolist())
