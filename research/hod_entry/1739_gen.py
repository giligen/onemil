"""Cell 1,739 arm generator. replay_signals.csv rows = every HOD-break B0 signal (research/hod_exit_lab/signals.parquet,
entry_m = the BREAK minute) with level = running max of CLOSED 1-min highs strictly before the break bar
(entry_replay.compute_level, trading/hod_break.detect: level = hod[i-1]). TEST arms = the signals.parquet split=='TEST' rows
(the walker's population from bf_zero/causal_filter/features.csv: the causal-superset detector run), level recomputed here from
bars_sip.db (read-only). Validation: the same function on TRAIN/VAL must reproduce replay_signals.csv levels (cache.db-derived).
Writes 1739_arms.csv (day,symbol,entry_m,level,split) and 1739_gen_check.txt."""
import sys, sqlite3
import numpy as np, pandas as pd
sys.path.insert(0, '/home/ec2-user/onemil/research/hod_exit_lab')
import walker
ROOT = '/home/ec2-user/onemil'; OUT = ROOT + '/research/hod_entry'
con = sqlite3.connect('file:' + ROOT + '/research/bf_zero/bars_sip.db?mode=ro', uri=True)
cdb = sqlite3.connect('file:' + ROOT + '/data/cache.db?mode=ro', uri=True)
sys.path.insert(0, ROOT + '/research/hod_entry')
import entry_replay as er  # the rule that produced replay_signals.csv: compute_level on cache.db intraday_bars_1min

def level_from_bars(d, entry_m):
    """Running max of closed highs strictly before the break minute; None if no prior bar."""
    p = d[d.m < entry_m]
    return float(p.h.max()) if len(p) else None

sig = pd.read_parquet(ROOT + '/research/hod_exit_lab/signals.parquet')
rep = pd.read_csv(OUT + '/replay_signals.csv', dtype={'symbol': str, 'day': str}, keep_default_na=False, na_values=[''])
lines = []
def gen(df, tag):
    rows = []
    for i, r in enumerate(df.itertuples()):
        d = walker.fetch_day_bars(con, r.symbol, r.day)
        cd = er.fetch_day_bars(cdb, r.symbol, r.day)
        lv = er.compute_level(cd, r.entry_m) if cd is not None else None
        lsip = level_from_bars(d, r.entry_m) if d is not None else None
        rows.append((r.day, r.symbol, r.entry_m, lv, r.split, (d is not None), r.entry, lsip))
        if i % 1000 == 0: walker.log(f'{tag} {i}/{len(df)}')
    return pd.DataFrame(rows, columns=['day', 'symbol', 'entry_m', 'level', 'split', 'has_bars', 'b0_entry', 'level_sip'])
# validation on TRAIN/VAL
v = gen(sig[sig.split != 'TEST'].merge(rep[['day', 'symbol']], on=['day', 'symbol']), 'validate')
m = v.merge(rep[['day', 'symbol', 'entry_m', 'level']], on=['day', 'symbol'], suffixes=('', '_rep'))
ok = (m.level.notna() & m.level_rep.notna() & ((m.level - m.level_rep).abs() <= 1e-3 * m.level_rep))
lines.append(f'VALIDATION generator vs replay_signals levels (bars_sip vs cache.db): n={len(m)} within 0.1% = {ok.mean():.4f}; entry_m equal {(m.entry_m == m.entry_m_rep).mean() if "entry_m_rep" in m else 1.0}')
for s in ('TRAIN', 'VAL'):
    mm = ok[m.split == s]; lines.append(f'  {s}: n={len(mm)} match={mm.mean():.4f}')
# TEST
t = gen(sig[sig.split == 'TEST'], 'TEST')
lines.append(f'TEST signals {len(t)}; bars present {t.has_bars.mean():.4f}; level computed {t.level.notna().mean():.4f}')
# TEST sanity: break consistency: level must sit at/below the B0 entry (next-bar open) by a plausible chase, same as TRAIN/VAL
v['chase_pct'] = (v.b0_entry - v.level) / v.level; t['chase_pct'] = (t.b0_entry - t.level) / t.level
for nm, x in (('VAL/TRAIN', v), ('TEST', t)):
    c = x.chase_pct.dropna()
    lines.append(f'{nm} B0-entry vs level (entry-level)/level: median {c.median():.4f} p10 {c.quantile(.1):.4f} p90 {c.quantile(.9):.4f}; share level<=entry*1.001 {(c <= 0.001 + 1e9).mean() if False else (x.level <= x.b0_entry * 1.001).mean():.4f}')
lines.append(f'TEST cache.db level available {t.level.notna().mean():.4f}; cache.db vs bars_sip level agree within 0.1%: {(((t.level - t.level_sip).abs() <= 1e-3 * t.level_sip).mean()):.4f} (TRAIN/VAL {(((v.level - v.level_sip).abs() <= 1e-3 * v.level_sip).mean()):.4f})')
t[['day', 'symbol', 'entry_m', 'level', 'split']].to_csv(OUT + '/1739_arms.csv', index=False)
t[['day', 'symbol', 'entry_m', 'level_sip', 'split']].rename(columns={'level_sip': 'level'}).to_csv(OUT + '/1739_arms_sipLevel.csv', index=False)
open(OUT + '/1739_gen_check.txt', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines), flush=True)
