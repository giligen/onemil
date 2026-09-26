"""Compare the independent rebuild (rebuild_1479_1480.csv) against the target implementation's
per-fill outputs (cell_1479_fills.csv, cell_1480_fills.csv) on (day, symbol, fill_min).

Verdict rule (per cell): REPRODUCED if >=99% of paired rows agree within 0.01 R AND the VAL means
agree within 0.02 R; PARTIAL if >=95% / 0.05; else NOT_REPRODUCED. Overall verdict = the weaker of
the two cells.
"""
import pandas as pd

HERE = '/home/ec2-user/onemil/research/hod_entry'


def verdict(share, val_gap):
    if share >= 0.99 and val_gap <= 0.02:
        return 'REPRODUCED'
    if share >= 0.95 and val_gap <= 0.05:
        return 'PARTIAL'
    return 'NOT_REPRODUCED'


mine = pd.read_csv(f'{HERE}/rebuild_1479_1480.csv')
theirs_1479 = pd.read_csv(f'{HERE}/cell_1479_fills.csv')
theirs_1480 = pd.read_csv(f'{HERE}/cell_1480_fills.csv')

KEY = ['day', 'symbol', 'fill_min']

print(f'mine total rows: {len(mine)}')
print(f'theirs 1479 rows: {len(theirs_1479)}  theirs 1480 (flip) rows: {len(theirs_1480)}')

# ---------------------------------------------------------------------------------------- cell 1479
m1479 = mine[KEY + ['split', 'c1479_net_R']].rename(columns={'c1479_net_R': 'mine_net_R'})
t1479 = theirs_1479[KEY + ['c1479_net_R']].rename(columns={'c1479_net_R': 'theirs_net_R'})
j1479 = m1479.merge(t1479, on=KEY, how='inner')
print(f'\n[1479] joined rows: {len(j1479)} (mine {len(m1479)}, theirs {len(t1479)})')
j1479['diff'] = (j1479.mine_net_R - j1479.theirs_net_R).abs()
share_1479 = (j1479['diff'] <= 0.01).mean()
val_mine_1479 = j1479[j1479.split == 'VAL'].mine_net_R.mean()
val_theirs_1479 = j1479[j1479.split == 'VAL'].theirs_net_R.mean()
gap_1479 = abs(val_mine_1479 - val_theirs_1479)
print(f'[1479] share within 0.01R: {share_1479:.4f}  max diff: {j1479["diff"].max():.4f}  '
      f'median diff: {j1479["diff"].median():.5f}')
print(f'[1479] VAL mean mine={val_mine_1479:+.4f}  theirs={val_theirs_1479:+.4f}  gap={gap_1479:.4f}')
v1479 = verdict(share_1479, gap_1479)
print(f'[1479] VERDICT: {v1479}')
worst_1479 = j1479.sort_values('diff', ascending=False).head(5)
print('[1479] worst mismatches:\n', worst_1479[KEY + ['mine_net_R', 'theirs_net_R', 'diff']].to_string(index=False))

# ---------------------------------------------------------------------------------------- cell 1480
m1480 = mine[KEY + ['split', 'long_delta_R', 'short_net_R', 'shortable', 'ssr']].dropna(subset=['short_net_R'])
m1480 = m1480.rename(columns={'long_delta_R': 'mine_long_delta_R', 'short_net_R': 'mine_short_net_R'})
t1480 = theirs_1480[KEY + ['split', 'long_delta_R', 'short_net_R', 'shortable', 'ssr']].rename(
    columns={'long_delta_R': 'theirs_long_delta_R', 'short_net_R': 'theirs_short_net_R'})
j1480 = m1480.merge(t1480[KEY + ['theirs_long_delta_R', 'theirs_short_net_R']], on=KEY, how='inner')
print(f'\n[1480] mine flip rows: {len(m1480)}  theirs flip rows: {len(t1480)}  '
      f'joined (intersection): {len(j1480)}')

j1480['diff_short'] = (j1480.mine_short_net_R - j1480.theirs_short_net_R).abs()
j1480['diff_long'] = (j1480.mine_long_delta_R - j1480.theirs_long_delta_R).abs()
# a row "agrees within 0.01R" only if BOTH the short leg and the long-side delta agree
j1480['ok'] = (j1480.diff_short <= 0.01) & (j1480.diff_long <= 0.01)
share_1480 = j1480['ok'].mean()
val_mine_short = j1480[j1480.split == 'VAL'].mine_short_net_R.mean()
val_theirs_short = j1480[j1480.split == 'VAL'].theirs_short_net_R.mean()
val_mine_long = j1480[j1480.split == 'VAL'].mine_long_delta_R.mean()
val_theirs_long = j1480[j1480.split == 'VAL'].theirs_long_delta_R.mean()
gap_short = abs(val_mine_short - val_theirs_short)
gap_long = abs(val_mine_long - val_theirs_long)
gap_1480 = max(gap_short, gap_long)
print(f'[1480] share within 0.01R (both legs): {share_1480:.4f}  '
      f'max diff_short: {j1480["diff_short"].max():.4f}  max diff_long: {j1480["diff_long"].max():.4f}')
print(f'[1480] VAL short: mine={val_mine_short:+.4f} theirs={val_theirs_short:+.4f} gap={gap_short:.4f}')
print(f'[1480] VAL long_delta: mine={val_mine_long:+.4f} theirs={val_theirs_long:+.4f} gap={gap_long:.4f}')
v1480 = verdict(share_1480, gap_1480)
print(f'[1480] VERDICT: {v1480}')
worst_1480 = j1480.sort_values('diff_short', ascending=False).head(5)
print('[1480] worst short mismatches:\n',
      worst_1480[KEY + ['mine_short_net_R', 'theirs_short_net_R', 'diff_short']].to_string(index=False))

order = {'REPRODUCED': 0, 'PARTIAL': 1, 'NOT_REPRODUCED': 2}
overall = max([v1479, v1480], key=lambda v: order[v])
print(f'\n=== OVERALL VERDICT: {overall} (1479={v1479}, 1480={v1480}) ===')
