"""Stress v2: touch-stop slip charged NET of the exit leg B0 already charges (half-spread + 2bp ~ cost_R/2, a lower
bound on the exit leg because the entry leg is selected tight) -> no double-charged slip (CLAUDE.md rule 4).
Also writes the slotted live-config series for scripts/cadence_bar.py."""
import numpy as np, pandas as pd
import judge_stress as J

f = J.load()
exit_leg = f.cost_R / 2.0


def hc(g, slip_bps, size_R):
    """Incremental touch-stop slip over the modeled exit leg (floored at 0) plus the size haircut."""
    inc = np.clip(slip_bps * g.bps_to_R - exit_leg.loc[g.index], 0, None)
    return g.net_R - np.where(g.touch_stop, inc, 0.0) - size_R


print('mean modeled exit leg on touch-stops, bps of price:',
      round(float((exit_leg[f.touch_stop] / f.bps_to_R[f.touch_stop]).mean()), 1), flush=True)
rows = []
test = f[f.split == 'TEST']
best = test.groupby('wk').net_R.sum().idxmax()
cohorts = [('TEST', test), ('TEST ex best wk', test[test.wk != best]), ('TRAIN-H2', f[f.split == 'TRAIN']),
           ('VAL', f[f.split == 'VAL']), ('POOLED', f)]
specs = [('plausible-net 22.9bps', 0.488 * 47, 0.024), ('severe-net 79bps', 79, 0.048)]
for label, g in cohorts:
    for slots in (False, True):
        for name, slip, size in specs:
            for lat in (False, True):
                rng = np.random.default_rng(42)
                draws = []
                for _ in range(J.N_DRAWS if lat else 1):
                    h = g[rng.random(len(g)) >= J.LAT_DROP] if lat else g
                    if slots:
                        s = h.assign(entry_m=h.fill_min)
                        h = s[J.simulate_slots(s)]
                    r = hc(h, slip, size)
                    t, nd = J.day_clustered_t(r, h.day)
                    draws.append((r.mean(), t, len(r)))
                d = np.array(draws)
                rows.append(dict(cohort=label, slots='12/4' if slots else 'none', scen=name + (' +lat32%' if lat else ''),
                                 n=int(np.median(d[:, 2])), mean_R=round(float(np.median(d[:, 0])), 3),
                                 day_t=round(float(np.median(d[:, 1])), 2), p_t_ge_2=round(float((d[:, 1] >= 2).mean()), 2)))
    print('done', label, flush=True)
res = pd.DataFrame(rows)
print(res.to_string(index=False))
# slotted live-config series for the cadence scorer: base and plausible-net
s = f.assign(entry_m=f.fill_min)
k = s[J.simulate_slots(s)]
k.assign(date=k.day, pnl_R=k.net_R)[['date', 'symbol', 'pnl_R']].to_csv('slot_base.csv', index=False)
k.assign(date=k.day, pnl_R=hc(k, 0.488 * 47, 0.024))[['date', 'symbol', 'pnl_R']].to_csv('slot_plaus.csv', index=False)
print('wrote slot_base.csv / slot_plaus.csv', len(k))
